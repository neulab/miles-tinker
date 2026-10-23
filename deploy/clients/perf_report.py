"""Per-step training MFU and step-time report for a gateway run.

Joins the gateway's ``<checkpoint_root>/perf.jsonl`` (one record per optimizer
step per slot: tokens, datums, ``actor_train_time`` = forward + backward +
optimizer step on the training GPUs, ``actor_train_mfu`` against the H200 bf16
peak) with, optionally, a platoon client log (its ``time/elapsed_seconds`` and
``task_reward/avg`` stats blocks). The wall gap between consecutive optimizer
steps of one slot is the end-to-end step time as the client experienced it
(rollouts + transfer + training + sampler push).

Args:
  --perf-jsonl: the gateway's perf.jsonl.
  --slot: restrict to one slot (default: the slot with the most steps).
  --client-log: platoon run log to join task_reward/avg and elapsed seconds.
  --since: only records after this unix time (default: all).
  --every: print every N-th step in the table (summary uses all).

Example:
  python deploy/clients/perf_report.py --perf-jsonl $P/checkpoints/full-ns/perf.jsonl \\
    --client-log $P/runs/number-search-full.log
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

PEAK_TFLOPS_H200 = 989.0


def load_perf(path: Path, since: float) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("time", 0) >= since and record.get("num_tokens", 0) > 0:
            records.append(record)
    return records


def client_steps(log_path: Path | None) -> dict[int, dict]:
    """platoon stats blocks: '... | Step N ...' headers followed by 'key: value' lines."""
    if log_path is None or not log_path.exists():
        return {}
    steps: dict[int, dict] = {}
    current = None
    for line in log_path.read_text(errors="replace").splitlines():
        header = re.search(r"stats_logger \| Step (\d+)\s*$", line)
        if header:
            current = int(header.group(1))
            steps.setdefault(current, {})
            continue
        match = re.search(r"stats_logger \|\s+([\w/.@-]+): ([-\d.]+)$", line)
        if match and current is not None:
            key, value = match.groups()
            if key in (
                "task_reward/avg",
                "elapsed_seconds",
                "grad_norm",
                "kl_sample_train_v1",
                "optim/grad_norm",
                "optim/kl_sample_train_v1",
            ):
                try:
                    steps[current][key] = float(value)
                except ValueError:
                    pass
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-jsonl", required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--client-log")
    parser.add_argument("--since", type=float, default=0.0)
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args()

    records = load_perf(Path(args.perf_jsonl), args.since)
    if not records:
        raise SystemExit("no perf records")
    slot = args.slot if args.slot is not None else Counter(r["slot"] for r in records).most_common(1)[0][0]
    rows = [r for r in records if r["slot"] == slot]
    client = client_steps(Path(args.client_log) if args.client_log else None)

    print(f"slot {slot}: {len(rows)} optimizer steps (records after {args.since:.0f})")
    print(
        f"{'step':>4} {'tokens':>8} {'datums':>6} {'train_s':>7} {'mfu%':>5} {'tok/s':>8} {'gap_s':>7} {'grad_norm':>9} {'reward':>7} {'elapsed_s':>9}"
    )
    gaps = []
    for i, r in enumerate(rows):
        gap = r["time"] - rows[i - 1]["time"] if i else float("nan")
        if i:
            gaps.append(gap)
        c = client.get(i, {})
        if i % args.every == 0 or i == len(rows) - 1:
            print(
                f"{i:>4} {int(r['num_tokens']):>8} {int(r['num_datums']):>6} {r['actor_train_time']:>7.2f} "
                f"{100 * r['actor_train_mfu']:>5.1f} {r['actor_train_tok_per_s']:>8.0f} {gap:>7.1f} "
                f"{r.get('grad_norm', float('nan')):>9.3g} {c.get('task_reward/avg', float('nan')):>7.3f} "
                f"{c.get('elapsed_seconds', float('nan')):>9.0f}"
            )
    train = [r["actor_train_time"] for r in rows]
    mfu = [r["actor_train_mfu"] for r in rows]
    tokens = [r["num_tokens"] for r in rows]
    print("\nsummary")
    print(f"  tokens/step        mean {statistics.mean(tokens):,.0f}  (min {min(tokens):,} max {max(tokens):,})")
    print(f"  train time/step    mean {statistics.mean(train):.2f} s  median {statistics.median(train):.2f} s")
    print(
        f"  training MFU       mean {100 * statistics.mean(mfu):.1f}%  median {100 * statistics.median(mfu):.1f}%  (peak {PEAK_TFLOPS_H200:.0f} TFLOPS/GPU x 8)"
    )
    if gaps:
        print(
            f"  wall time/step     mean {statistics.mean(gaps):.1f} s  median {statistics.median(gaps):.1f} s  (gap between optimizer steps: rollouts + training + push)"
        )
        print(f"  training share     {100 * sum(train[1:]) / sum(gaps):.1f}% of wall time")
    if client:
        rewards = [c["task_reward/avg"] for c in client.values() if "task_reward/avg" in c]
        if rewards:
            print(f"  task_reward/avg    first {rewards[0]:.3f}  last {rewards[-1]:.3f}  max {max(rewards):.3f}")


if __name__ == "__main__":
    main()
