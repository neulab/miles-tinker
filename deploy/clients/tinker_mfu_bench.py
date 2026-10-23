"""Training-MFU benchmark for a live Tinker gateway (real SDK, synthetic data).

Drives ``forward_backward`` + ``optim_step`` rounds with RL-shaped datums
(random tokens, uniform weights, importance-sampling advantages) whose sizes
mimic an agentic RL step, and reports the client-side step time next to the
gateway's own ``perf.jsonl`` record (``actor_train_mfu``, tokens/s). Use it to
tune ``--max-tokens-per-gpu`` / packing before the acceptance runs.

Args:
  --endpoint-file / --base-url: gateway address (or $TINKER_BASE_URL).
  --base-model: model name the gateway advertises.
  --lora-rank: adapter rank (ignored by a full fine-tuning gateway).
  --num-datums: datums per step (default 128).
  --seq-len: tokens per datum (default 8192).
  --steps: measured steps after one warm-up step (default 3).
  --loss-fn: importance_sampling (default) or cross_entropy.
  --perf-jsonl: the gateway's <checkpoint_root>/perf.jsonl to print alongside.

Example:
  /home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/tinker_mfu_bench.py \\
    --endpoint-file $P/name-resolve/gw.endpoint --num-datums 128 --seq-len 8192 \\
    --perf-jsonl $P/checkpoints/gw/perf.jsonl
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import os
import random
import time
from pathlib import Path

import hostguard  # noqa: E402,F401  (refuses login nodes; MILES_ALLOW_LOGIN_NODE=1 overrides)

os.environ.setdefault("TINKER_API_KEY", "tml-mfu-bench")
os.environ.setdefault("HF_HOME", "/groups/gai51740/aci18912it/rasp/cache/huggingface")

import tinker  # noqa: E402
from tinker import types  # noqa: E402

VOCAB_HINT = 150_000  # below Qwen3's 151936-token vocabulary


def build_datums(num_datums: int, seq_len: int, loss_fn: str, seed: int = 0) -> list[types.Datum]:
    rng = random.Random(seed)
    datums = []
    for _ in range(num_datums):
        tokens = [rng.randrange(1, VOCAB_HINT) for _ in range(seq_len + 1)]
        inputs: dict = {"target_tokens": tokens[1:]}
        if loss_fn == "cross_entropy":
            inputs["weights"] = [1.0] * seq_len
        else:
            inputs["logprobs"] = [-rng.random() * 2 for _ in range(seq_len)]
            inputs["advantages"] = [rng.choice((-1.0, 1.0)) / seq_len for _ in range(seq_len)]
        datums.append(types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs))
    return datums


def perf_records_since(path: Path | None, since: float) -> list[dict]:
    if path is None or not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("time", 0) >= since:
            records.append(record)
    return records


def resolve_base_url(args) -> str:
    if args.base_url:
        return args.base_url
    if args.endpoint_file:
        return Path(args.endpoint_file).read_text().strip()
    return os.environ["TINKER_BASE_URL"]


def unload_model(base_url: str, model_id: str) -> None:
    """POST /api/v1/unload_model with this client's API key (a gateway route without an SDK method)."""
    import httpx

    httpx.post(
        f"{base_url.rstrip('/')}/api/v1/unload_model",
        json={"model_id": model_id},
        headers={"X-API-Key": os.environ["TINKER_API_KEY"]},
        timeout=120,
    ).raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-file")
    parser.add_argument("--base-url")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-datums", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--loss-fn", default="importance_sampling", choices=["importance_sampling", "cross_entropy"])
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--perf-jsonl")
    parser.add_argument("--keep-model", action="store_true", help="leave the model loaded at exit")
    args = parser.parse_args()
    base_url = resolve_base_url(args)

    service = tinker.ServiceClient(base_url=base_url)
    training = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank)
    model_id = training.get_info().model_id
    datums = build_datums(args.num_datums, args.seq_len, args.loss_fn)
    tokens_per_step = args.num_datums * args.seq_len
    adam = types.AdamParams(learning_rate=args.learning_rate)
    perf_path = Path(args.perf_jsonl) if args.perf_jsonl else None
    print(
        f"bench: {args.num_datums} datums x {args.seq_len} tokens = {tokens_per_step} tokens/step, loss {args.loss_fn}"
    )

    for step in range(args.steps + 1):
        started = time.time()
        fb = training.forward_backward(datums, args.loss_fn)
        optim = training.optim_step(adam)
        fb_result = fb.result()
        optim_result = optim.result()
        elapsed = time.time() - started
        label = "warmup" if step == 0 else f"step {step}"
        client_tok_s = tokens_per_step / elapsed
        line = f"{label}: {elapsed:.2f}s client wall ({client_tok_s:,.0f} tok/s incl. transfer)"
        metrics = optim_result.metrics or {}
        if "actor_train_mfu" in metrics:
            line += (
                f" | server actor_train_time {metrics['actor_train_time']:.2f}s, "
                f"mfu {100 * metrics['actor_train_mfu']:.1f}%, {metrics['actor_train_tok_per_s']:,.0f} tok/s, "
                f"grad_norm {metrics.get('grad_norm', float('nan')):.3g}"
            )
        print(line)
        print(f"  fwdbwd metrics: { {k: round(v, 4) for k, v in (fb_result.metrics or {}).items()} }")
        for record in perf_records_since(perf_path, started):
            print(f"  perf.jsonl: {json.dumps(record)}")
    if not args.keep_model:
        unload_model(base_url, model_id)


if __name__ == "__main__":
    main()
