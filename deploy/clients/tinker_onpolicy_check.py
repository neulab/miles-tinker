"""Sampler-vs-trainer consistency probe for a live Tinker gateway.

Trains a few steps so the adapter (or the full weights) differ from the base,
publishes a sampler version, samples completions from it, then scores exactly
those tokens with a forward-only pass on the training side at the same
version. If weight publication is faithful, the per-token log-prob difference
is numerical noise (bf16 sampling vs. training kernels: |mean delta| around
1e-3, KL below ~1e-3). A systematic gap points at a weight-sync or adapter
scaling mismatch and would silently bias importance-sampling RL.

Args:
  --endpoint-file / --base-url: gateway address (or $TINKER_BASE_URL).
  --base-model: model name the gateway advertises.
  --lora-rank: adapter rank (ignored by a full fine-tuning gateway).
  --train-steps: optimizer steps before the probe (default 3).
  --num-prompts / --max-tokens: probe sample count and length.

Example:
  /home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/tinker_onpolicy_check.py \\
    --endpoint-file $P/name-resolve/gw.endpoint
"""

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("TINKER_API_KEY", "tml-onpolicy-check")
os.environ.setdefault("HF_HOME", "/groups/gai51740/aci18912it/rasp/cache/huggingface")

import numpy as np  # noqa: E402
import tinker  # noqa: E402
from tinker import types  # noqa: E402

PROMPTS = [
    "Write two sentences about the ocean.",
    "Explain what a prime number is.",
    "List three fruits and one fact about each.",
    "Describe a morning routine in a few sentences.",
    "What is the difference between weather and climate?",
    "Give a short recipe for scrambled eggs.",
    "Why is the sky blue? Answer briefly.",
    "Summarize the plot of a fairy tale you know.",
]


def resolve_base_url(args) -> str:
    if args.base_url:
        return args.base_url
    if args.endpoint_file:
        return Path(args.endpoint_file).read_text().strip()
    return os.environ["TINKER_BASE_URL"]


def chat_tokens(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return [int(t) for t in tokenizer.encode(rendered, add_special_tokens=False)]


def training_datums(tokenizer, seq_len: int = 512, count: int = 32, seed: int = 0) -> list[types.Datum]:
    """RL-shaped datums (random tokens, +-1 advantages) so a few optimizer steps move the weights."""
    rng = random.Random(seed)
    datums = []
    for _ in range(count):
        tokens = [rng.randrange(1, 150_000) for _ in range(seq_len + 1)]
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "logprobs": [-rng.random() * 2 for _ in range(seq_len)],
                    "advantages": [rng.choice((-1.0, 1.0)) / seq_len for _ in range(seq_len)],
                },
            )
        )
    return datums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-file")
    parser.add_argument("--base-url")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    service = tinker.ServiceClient(base_url=resolve_base_url(args))
    training = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank)
    tokenizer = training.get_tokenizer()

    datums = training_datums(tokenizer)
    adam = types.AdamParams(learning_rate=args.learning_rate)
    for step in range(args.train_steps):
        fb = training.forward_backward(datums, "importance_sampling")
        optim = training.optim_step(adam)
        fb.result()
        print(f"train step {step}: grad_norm {optim.result().metrics.get('grad_norm', float('nan')):.4g}")

    sampler = training.save_weights_and_get_sampling_client()
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, stop=[tokenizer.eos_token_id])

    probe_datums, sampled_logprobs = [], []
    for prompt in PROMPTS[: args.num_prompts]:
        prompt_ids = chat_tokens(tokenizer, prompt)
        seq = sampler.sample(types.ModelInput.from_ints(prompt_ids), num_samples=1, sampling_params=params).result().sequences[0]
        tokens = prompt_ids + list(seq.tokens)
        probe_datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={"target_tokens": tokens[1:], "weights": [0.0] * (len(prompt_ids) - 1) + [1.0] * len(seq.tokens)},
            )
        )
        sampled_logprobs.append(np.asarray(seq.logprobs, dtype=np.float64))

    scored = training.forward(probe_datums, "cross_entropy").result()
    deltas, ratios = [], []
    for datum, sampled, output in zip(probe_datums, sampled_logprobs, scored.loss_fn_outputs):
        trainer_lp = output["logprobs"].to_numpy().astype(np.float64)
        weights = datum.loss_fn_inputs["weights"].to_numpy()
        completion_lp = trainer_lp[weights > 0]
        assert len(completion_lp) == len(sampled), (len(completion_lp), len(sampled))
        delta = completion_lp - sampled
        deltas.append(delta)
        ratios.append(np.exp(delta))
        print(
            f"  {len(sampled):4d} tokens | mean delta {delta.mean():+.4f} | mean |delta| {np.abs(delta).mean():.4f} "
            f"| max |delta| {np.abs(delta).max():.3f} | mean ratio {np.exp(delta).mean():.4f}"
        )
    all_delta = np.concatenate(deltas)
    kl_est = float(np.mean(np.exp(all_delta) - 1 - all_delta))  # E_sampler[ratio - 1 - log ratio] (k3 estimator of KL(sampler || trainer))
    print(
        f"overall: {len(all_delta)} tokens | mean delta {all_delta.mean():+.5f} | mean |delta| {np.abs(all_delta).mean():.5f} "
        f"| kl(sampler||trainer) ~ {kl_est:.5f} | p95 |delta| {np.percentile(np.abs(all_delta), 95):.4f}"
    )
    verdict = "PASS" if np.abs(all_delta).mean() < 0.02 and kl_est < 0.005 else "FAIL"
    print(f"{verdict}: sampler and trainer agree at the same version" if verdict == "PASS" else f"{verdict}: systematic sampler/trainer gap")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
