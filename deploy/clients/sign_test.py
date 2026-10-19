"""Same-batch sign test: does an update move its own batch the way its advantages say?

Run against a live full fine-tuning gateway (run.py with TINKER_CLIENT_SCRIPT set to
this file), with the training run's own server settings. It samples real
completions from the starting weights, gives half of each prompt's completions
advantage +1 and half -1 (normalised as the train loop does: per-trajectory token
mean, then over trajectories), and repeats on that one fixed batch:

    forward (score) -> forward_backward (CISPO, same clip) -> optim_step (same Adam)

After every step it re-scores the batch. A correct trainer raises the mean
logprob of every +1 completion and lowers every -1 completion, and the gap grows
step after step. Held-out completions (never trained) show the drift an update
causes on data it did not see. A final phase runs plain supervised steps
(cross_entropy on the +1 completions only), which must raise their logprob fast.

Prompts include two long ones (~16k and ~32k tokens of context) so the long-sequence
path (dynamic batching, chunked logprobs, FP32 head) is covered too.

Writes sign_test.json to --out and prints a PASS/FAIL summary.
"""

import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

os.environ.setdefault("TINKER_API_KEY", "tml-local")

import tinker  # noqa: E402
from tinker import types  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TASK = (
    "You are working in a Linux terminal. Create a Python script /home/user/stats.py that reads "
    "/home/user/data.csv (columns: id,value), drops rows whose value is not a number, and prints the "
    "mean and median of the remaining values rounded to 3 decimals, one per line."
)
SHORT_PROMPTS = [
    TASK,
    "Write a bash one-liner that counts the lines containing ERROR in every *.log file under /var/log, "
    "printing file name and count.",
    "Explain what this C code does and fix the bug:\n\nint sum(int *a, int n) { int s; for (int i = 0; i <= n; i++) s += a[i]; return s; }",
    "In a terminal, a Makefile fails with 'stdio.h: No such file or directory'. List the commands you would run to fix it.",
]
FILLER = "The quick brown fox jumps over the lazy dog while the log rotates every hour. " * 20


def chat_prompt(tokenizer, user: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text, add_special_tokens=False)


def long_prompt(tokenizer, target_tokens: int) -> list[int]:
    block_tokens = len(tokenizer.encode(FILLER, add_special_tokens=False))
    context = FILLER * max(1, target_tokens // block_tokens)
    return chat_prompt(tokenizer, f"Here is a log file:\n\n{context}\n\nSummarise it in two sentences, then {TASK}")


def make_datum(prompt: list[int], completion: list[int], logprobs: list[float], advantage: float, scale: float):
    tokens = prompt + completion
    n_prompt = len(prompt)
    adv = [0.0] * (n_prompt - 1) + [advantage * scale] * len(completion)
    lps = [0.0] * (n_prompt - 1) + list(logprobs)
    weights = [0.0] * (n_prompt - 1) + [scale] * len(completion)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "logprobs": lps, "advantages": adv, "weights": weights},
    )


def completion_mean_logprob(output, n_prompt: int, n_completion: int) -> float:
    lp = output["logprobs"].to_numpy()
    return float(lp[n_prompt - 1 : n_prompt - 1 + n_completion].mean())


def score(client, datums, meta) -> list[float]:
    ce = [
        types.Datum(model_input=d.model_input, loss_fn_inputs={"target_tokens": d.loss_fn_inputs["target_tokens"], "weights": [0.0] * d.model_input.length})
        for d in datums
    ]
    out = client.forward(ce, loss_fn="cross_entropy").result()
    return [completion_mean_logprob(o, m["n_prompt"], m["n_completion"]) for o, m in zip(out.loss_fn_outputs, meta)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=float(os.environ.get("SIGN_TEST_LR", "1e-6")))
    p.add_argument("--sft-steps", type=int, default=4)
    a = p.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base_model)
    service = tinker.ServiceClient(base_url=a.base_url)
    client = service.create_lora_training_client(base_model=a.base_model, rank=32)
    sampler = client.save_weights_and_get_sampling_client()
    params = types.SamplingParams(max_tokens=a.max_tokens, temperature=1.0)

    prompts = [chat_prompt(tok, u) for u in SHORT_PROMPTS] + [long_prompt(tok, 16000), long_prompt(tok, 32000)]
    held_out = [chat_prompt(tok, "Write a Python function that reverses a linked list, with a short explanation.")]
    train_d, train_m, held_d, held_m = [], [], [], []
    t0 = time.time()
    for pi, prompt in enumerate(prompts + held_out):
        res = sampler.sample(types.ModelInput.from_ints(prompt), num_samples=a.samples, sampling_params=params).result()
        is_held = pi >= len(prompts)
        seqs = [s for s in res.sequences if len(s.tokens) >= 8]
        n_traj = len(seqs)
        for si, seq in enumerate(seqs):
            advantage = 0.0 if is_held else (1.0 if si % 2 == 0 else -1.0)
            scale = 1.0 / (n_traj * len(seq.tokens))
            d = make_datum(prompt, list(seq.tokens), list(seq.logprobs), advantage, scale)
            m = {"prompt": pi, "adv": advantage, "n_prompt": len(prompt), "n_completion": len(seq.tokens)}
            (held_d if is_held else train_d).append(d)
            (held_m if is_held else train_m).append(m)
    print(f"sampled {len(train_d)} train + {len(held_d)} held-out completions in {time.time() - t0:.0f}s; "
          f"prompt lengths {[len(x) for x in prompts]}", flush=True)

    adam = types.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip_norm=1.0)
    base_train, base_held = score(client, train_d, train_m), score(client, held_d, held_m)
    history = []
    for step in range(1, a.steps + 1):
        fb = client.forward_backward(train_d, loss_fn="cispo", loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}).result()
        opt = client.optim_step(adam).result()
        now_train, now_held = score(client, train_d, train_m), score(client, held_d, held_m)
        deltas = [n - b for n, b in zip(now_train, base_train)]
        pos = [d for d, m in zip(deltas, train_m) if m["adv"] > 0]
        neg = [d for d, m in zip(deltas, train_m) if m["adv"] < 0]
        right = sum(1 for d, m in zip(deltas, train_m) if d * m["adv"] > 0)
        long_ok = [d * m["adv"] > 0 for d, m in zip(deltas, train_m) if m["n_prompt"] > 10000]
        held = st.mean(n - b for n, b in zip(now_held, base_held))
        row = {
            "step": step, "mean_delta_pos": st.mean(pos), "mean_delta_neg": st.mean(neg),
            "sign_correct": right / len(deltas), "long_sign_correct": sum(long_ok) / max(1, len(long_ok)),
            "held_out_delta": held, "grad_norm": (opt.metrics or {}).get("grad_norm"), "fb_metrics": fb.metrics,
        }
        history.append(row)
        print(f"step {step}: +adv mean dlogp {row['mean_delta_pos']:+.5f}  -adv {row['mean_delta_neg']:+.5f}  "
              f"sign correct {row['sign_correct']:.0%} (long prompts {row['long_sign_correct']:.0%})  "
              f"held-out {held:+.5f}  grad_norm {row['grad_norm']}", flush=True)

    sft = []
    pos_d = [types.Datum(model_input=d.model_input, loss_fn_inputs={"target_tokens": d.loss_fn_inputs["target_tokens"], "weights": d.loss_fn_inputs["weights"]})
             for d, m in zip(train_d, train_m) if m["adv"] > 0]
    pos_m = [m for m in train_m if m["adv"] > 0]
    sft_base = score(client, pos_d, pos_m)
    for step in range(1, a.sft_steps + 1):
        client.forward_backward(pos_d, loss_fn="cross_entropy").result()
        client.optim_step(types.AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip_norm=1.0)).result()
        now = score(client, pos_d, pos_m)
        gain = st.mean(n - b for n, b in zip(now, sft_base))
        sft.append({"step": step, "mean_gain": gain, "all_up": all(n > b for n, b in zip(now, sft_base))})
        print(f"sft step {step}: mean dlogp {gain:+.4f}  all raised {sft[-1]['all_up']}", flush=True)

    last = history[-1]
    widening = all(h2["mean_delta_pos"] - h2["mean_delta_neg"] >= h1["mean_delta_pos"] - h1["mean_delta_neg"] - 1e-4 for h1, h2 in zip(history, history[1:]))
    checks = {
        "+adv completions rise, -adv fall (final step)": last["mean_delta_pos"] > 0 > last["mean_delta_neg"],
        "per-completion sign correct >= 90% (final step)": last["sign_correct"] >= 0.9,
        "long-context completions follow their advantage": last["long_sign_correct"] >= 0.9,
        "gap widens step after step": widening,
        "supervised steps raise every target completion": sft[-1]["all_up"] if sft else False,
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name, flush=True)
    (out_dir / "sign_test.json").write_text(json.dumps({"lr": a.lr, "history": history, "sft": sft, "checks": checks}, indent=1, default=str))


if __name__ == "__main__":
    main()
