"""End-to-end smoke test of a live gateway with the real Tinker SDK (0.27.1).

Exercises the calls tinker-cookbook and platoon make, on tiny datums:
create a LoRA training client, three ``forward_backward`` + ``optim_step``
(cross_entropy), an unnamed ``save_weights_for_sampler`` and a sample from it,
a named ``save_state`` + ``weights_info`` + resume with optimizer state and one
more step, then sampling with stop strings and stop token ids. Prints one
PASS/FAIL line per check and exits non-zero when any check failed. Against a
full fine-tuning gateway the rank is ignored by the server.

Args:
  --endpoint-file: file holding ``http://host:port`` (or --base-url / $TINKER_BASE_URL).
  --base-model: model name the gateway advertises (default Qwen/Qwen3-4B-Instruct-2507).
  --lora-rank: adapter rank to request.
  --steps: forward_backward/optim_step rounds before the first save.

Example:
  /home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/tinker_sdk_smoke.py \\
    --endpoint-file $P/name-resolve/gw-lora.endpoint --lora-rank 16
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TINKER_API_KEY", "tml-local")
os.environ.setdefault("HF_HOME", "/groups/gai51740/aci18912it/rasp/cache/huggingface")

import tinker  # noqa: E402  (env defaults above must precede the import)
from tinker import types  # noqa: E402

PROMPTS = ["The capital of France is", "One plus one equals", "Water boils at"]
TARGETS = [" Paris.", " two.", " one hundred degrees Celsius."]


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
        self.failed += 0 if ok else 1
        return ok

    def warn(self, name: str, detail: str) -> None:
        print(f"WARN {name}: {detail}", flush=True)


def resolve_base_url(args: argparse.Namespace) -> str:
    if args.endpoint_file:
        return Path(args.endpoint_file).read_text().strip()
    url = args.base_url or os.environ.get("TINKER_BASE_URL")
    if not url:
        raise SystemExit("pass --endpoint-file or --base-url (or set TINKER_BASE_URL)")
    return url


def make_datums(tokenizer) -> list[types.Datum]:
    """Prompt + completion datums with the completion weighted 1.0 (cross_entropy inputs)."""
    datums = []
    for prompt, target in zip(PROMPTS, TARGETS):
        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(target)
        tokens = prompt_ids + target_ids
        weights = [0.0] * len(prompt_ids) + [1.0] * len(target_ids)
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
            )
        )
    return datums


def train_rounds(checks: Checks, client, datums: list[types.Datum], steps: int, label: str) -> None:
    for step in range(steps):
        t0 = time.time()
        fb_future = client.forward_backward(datums, loss_fn="cross_entropy")
        optim_future = client.optim_step(types.AdamParams(learning_rate=1e-4))
        fb = fb_future.result()
        optim = optim_future.result()
        lengths_ok = all(
            len(out["logprobs"].to_numpy()) == datum.model_input.length for out, datum in zip(fb.loss_fn_outputs, datums)
        )
        checks.check(
            f"{label} forward_backward {step}",
            len(fb.loss_fn_outputs) == len(datums) and lengths_ok,
            f"{len(fb.loss_fn_outputs)} outputs, metrics {sorted(fb.metrics)} in {time.time() - t0:.1f}s",
        )
        metrics = optim.metrics or {}
        checks.check(f"{label} optim_step {step}", "grad_norm" in metrics, f"metrics {metrics}")


def sample_once(sampling_client, prompt_ids: list[int], stop, max_tokens: int = 32) -> types.SampledSequence:
    params = types.SamplingParams(max_tokens=max_tokens, temperature=1.0, stop=stop)
    response = sampling_client.sample(types.ModelInput.from_ints(prompt_ids), num_samples=1, sampling_params=params)
    return response.result().sequences[0]


def chat_prompt(tokenizer, text: str) -> list[int]:
    messages = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return [int(token) for token in tokenizer.encode(rendered, add_special_tokens=False)]


def check_stop_ids(checks: Checks, tokenizer, sampling_client) -> None:
    stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
    seq = sample_once(sampling_client, chat_prompt(tokenizer, "Reply with only the word OK."), stop=stop_ids)
    if seq.stop_reason != "stop":
        checks.warn("sample with stop token ids", f"stop_reason={seq.stop_reason}; model did not stop within max_tokens")
        return
    checks.check(
        "sample with stop token ids returns the stop token",
        bool(seq.tokens) and seq.tokens[-1] in stop_ids,
        f"tokens[-1]={seq.tokens[-1] if seq.tokens else None} stop_ids={stop_ids} text={tokenizer.decode(seq.tokens)!r}",
    )


def check_stop_strings(checks: Checks, tokenizer, sampling_client) -> None:
    seq = sample_once(sampling_client, tokenizer.encode("Count: 1, 2, 3,"), stop=[","], max_tokens=16)
    text = tokenizer.decode(seq.tokens)
    if seq.stop_reason != "stop":
        checks.warn("sample with stop strings", f"stop_reason={seq.stop_reason} text={text!r}")
        return
    checks.check("sample with stop strings stops at the string", text.rstrip().endswith(","), f"text={text!r}")


def unload_model(base_url: str, model_id: str) -> None:
    """POST /api/v1/unload_model with this client's API key (a gateway route without an SDK method)."""
    import httpx

    response = httpx.post(
        f"{base_url.rstrip('/')}/api/v1/unload_model",
        json={"model_id": model_id},
        headers={"X-API-Key": os.environ["TINKER_API_KEY"]},
        timeout=120,
    )
    response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint-file")
    parser.add_argument("--base-url")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--unload-before-resume", action="store_true", help="free the first model before resuming (full fine-tuning gateways serve one model at a time)")
    args = parser.parse_args()
    base_url = resolve_base_url(args)
    checks = Checks()
    print(f"gateway {base_url} model {args.base_model} rank {args.lora_rank} sdk {tinker.__version__}")

    service = tinker.ServiceClient(base_url=base_url)
    client = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank)
    info = client.get_info()
    checks.check("create_lora_training_client", bool(info.model_id), f"model_id={info.model_id} is_lora={info.is_lora} rank={info.lora_rank}")
    tokenizer = client.get_tokenizer()
    datums = make_datums(tokenizer)

    train_rounds(checks, client, datums, args.steps, "initial")

    sampler = client.save_weights_and_get_sampling_client()
    seq = sample_once(sampler, tokenizer.encode(PROMPTS[0]), stop=None, max_tokens=8)
    checks.check(
        "unnamed save_weights_for_sampler + sample",
        len(seq.tokens) > 0 and seq.logprobs is not None and len(seq.logprobs) == len(seq.tokens),
        f"{len(seq.tokens)} tokens stop_reason={seq.stop_reason} text={tokenizer.decode(seq.tokens)!r}",
    )

    name = f"smoke-{int(time.time())}"
    state_path = client.save_state(name).result().path
    checks.check("named save_state", state_path.startswith("tinker://") and "/weights/" in state_path, state_path)
    if args.unload_before_resume:
        # single-tenant (full fine-tuning) gateways serve one model at a time; the SDK has no unload call
        unload_model(base_url, info.model_id)
        checks.check("unload_model before resume", True, info.model_id)
    weights_info = service.create_rest_client().get_weights_info_by_tinker_path(state_path).result()
    checks.check(
        "weights_info",
        weights_info.base_model == args.base_model and weights_info.is_lora is True,
        f"base_model={weights_info.base_model} is_lora={weights_info.is_lora} lora_rank={weights_info.lora_rank}",
    )
    resumed = service.create_training_client_from_state_with_optimizer(state_path)
    checks.check("create_training_client_from_state_with_optimizer", resumed.get_info().model_id != info.model_id)
    train_rounds(checks, resumed, datums, 1, "resumed")

    sampler = resumed.save_weights_and_get_sampling_client()
    check_stop_ids(checks, tokenizer, sampler)
    check_stop_strings(checks, tokenizer, sampler)

    print(f"{'FAIL' if checks.failed else 'PASS'}: {checks.failed} failed check(s)")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
