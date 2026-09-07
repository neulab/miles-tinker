#!/usr/bin/env python3
"""Prune the recovery checkpoints of a platoon/cookbook Tinker run (stdlib only).

The client appends one record per save to ``<log_path>/<experiment>/<trial>/checkpoints.jsonl``
(``{"name", "batch", "state_path": "tinker://<model_id>/weights/<name>",
"sampler_path": "tinker://<model_id>/sampler_weights/<name>", ...}``); the gateway stores each
path under ``<checkpoint_root>/<model_id>/<kind>/<name>/``. Every (re)launch creates a new
``model_id``, so the directories must be pruned across all ``model_id``s.

The client writes checkpoints on an AReaL-style two-tier cadence, and only the disposable tier
is prunable:

* **recovery** checkpoints - name ``recovery-<batch>``, written by ``checkpoint.recovery_every``
  (typically every step) and by every (re)launch, each a full ~490 GB weights+optimizer save
  that only exists so a crash loses at most one step. These are the ONLY prunable directories;
* **permanent** checkpoints - every other name (``<batch>`` from ``checkpoint.every``,
  ``<batch>_interrupted``, ``final``) - and any other directory that is not ``recovery-*``.

The rules, in order:

1. a ``recovery-*`` directory of one of the newest ``--keep`` recovery records of a
   ``checkpoints.jsonl`` is kept (both its state and sampler directory);
2. the last record WITH a ``state_path`` is the resume point (cookbook
   ``get_last_checkpoint(required_key="state_path")``) and its state directory is kept whatever
   its name;
3. every directory whose name does not start with ``recovery-`` is kept forever;
4. a directory modified within the last ``--grace-min`` minutes is kept (a save in progress, or
   a recovery save whose record has not been appended yet);
5. every other ``recovery-*`` directory under ``<checkpoint_root>/*/weights/`` and
   ``*/sampler_weights/`` is removed. A record whose directory does not exist is tolerated: in
   the gateway's full mode ``sampler_path`` has no directory of its own.

Args:
  --log-path: the client's ``log_path`` (searched recursively for ``checkpoints.jsonl``).
  --checkpoint-root: the gateway's ``--checkpoint-root`` for this run.
  --keep: recovery records to keep per checkpoints.jsonl (default 2).
  --grace-min: minutes since the last modification below which a directory is untouched (30).
  --dry-run: print the plan without deleting.

Example:
  python3 deploy/abci/clients/prune_tinker_checkpoints.py --log-path $RUNS/$RUN/logs \\
    --checkpoint-root $P/checkpoints/$RUN --keep 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

CHECKPOINT_KINDS = ("weights", "sampler_weights")
PATH_KEYS = ("state_path", "sampler_path")
# platoon.train.tinker.rl.RECOVERY_CHECKPOINT_PREFIX: the disposable tier.
RECOVERY_PREFIX = "recovery-"


@dataclass(frozen=True)
class PrunePlan:
    keep: frozenset[Path]
    delete: tuple[Path, ...]
    reasons: dict[Path, str] = field(default_factory=dict)


def tinker_path_to_dir(checkpoint_root: Path, tinker_path: str) -> Path | None:
    """``tinker://<model_id>/<kind>/<name>`` -> ``<checkpoint_root>/<model_id>/<kind>/<name>``."""
    if not tinker_path.startswith("tinker://"):
        return None
    parts = PurePosixPath(tinker_path[len("tinker://") :]).parts
    if len(parts) != 3 or parts[1] not in CHECKPOINT_KINDS:
        return None
    return checkpoint_root / parts[0] / parts[1] / parts[2]


def read_records(checkpoints_jsonl: Path) -> list[dict]:
    records = []
    for line in checkpoints_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a partially written trailing line
    return records


def record_dirs(record: dict, checkpoint_root: Path) -> list[Path]:
    dirs = []
    for key in PATH_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            directory = tinker_path_to_dir(checkpoint_root, value)
            if directory is not None:
                dirs.append(directory)
    return dirs


def list_checkpoint_dirs(checkpoint_root: Path) -> dict[Path, float]:
    """Every ``<root>/<model_id>/<kind>/<name>`` directory with its mtime."""
    found = {}
    if not checkpoint_root.is_dir():
        return found
    for model_dir in checkpoint_root.iterdir():
        if not model_dir.is_dir():
            continue
        for kind in CHECKPOINT_KINDS:
            kind_dir = model_dir / kind
            if not kind_dir.is_dir():
                continue
            for name_dir in kind_dir.iterdir():
                if name_dir.is_dir():
                    found[name_dir] = name_dir.stat().st_mtime
    return found


def is_recovery_name(name: str) -> bool:
    """A ``recovery-<batch>`` checkpoint: the disposable tier of the client's cadence."""
    return name.startswith(RECOVERY_PREFIX)


def plan_prune(
    record_lists: list[list[dict]],
    existing: dict[Path, float],
    checkpoint_root: Path,
    *,
    keep: int,
    now: float,
    grace_s: float = 1800.0,
) -> PrunePlan:
    """Pure decision: which existing directories to keep and which to delete.

    ``record_lists`` holds the records of every ``checkpoints.jsonl`` in append order;
    ``existing`` maps every candidate directory to its mtime. Only ``recovery-*``
    directories are ever deleted; see the module docstring for the full rule order.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    keep_dirs: set[Path] = set()
    reasons: dict[Path, str] = {}
    for records in record_lists:
        recovery = [r for r in records if is_recovery_name(str(r.get("name", "")))]
        for record in recovery[-keep:]:
            for directory in record_dirs(record, checkpoint_root):
                keep_dirs.add(directory)
                reasons.setdefault(directory, f"newest {keep} recovery record(s)")
        # the resume point is the last record WITH a state_path (cookbook get_last_checkpoint(required_key="state_path"))
        for record in reversed(records):
            last_state = record.get("state_path")
            if isinstance(last_state, str):
                directory = tinker_path_to_dir(checkpoint_root, last_state)
                if directory is not None:
                    keep_dirs.add(directory)
                    reasons[directory] = "last record's state_path"
                break
    for directory in existing:  # permanent names (NNNNNN, *_interrupted, final) are kept forever
        if not is_recovery_name(directory.name):
            keep_dirs.add(directory)
            reasons.setdefault(directory, "permanent checkpoint")
    for directory, mtime in existing.items():
        if now - mtime < grace_s and directory not in keep_dirs:
            keep_dirs.add(directory)
            reasons[directory] = f"modified {int(now - mtime)} s ago (< {int(grace_s)} s grace)"
    delete = tuple(sorted(d for d in existing if d not in keep_dirs))
    for directory in delete:
        reasons[directory] = f"recovery checkpoint older than the newest {keep}"
    return PrunePlan(keep=frozenset(keep_dirs), delete=delete, reasons=reasons)


def apply_plan(plan: PrunePlan, *, dry_run: bool) -> int:
    for directory in plan.delete:
        print(f"{'would delete' if dry_run else 'delete'} {directory}: {plan.reasons.get(directory, '')}", flush=True)
        if not dry_run:
            shutil.rmtree(directory, ignore_errors=True)
            parent = directory.parent  # drop empty <kind> / <model_id> dirs of finished launches
            for ancestor in (parent, parent.parent):
                try:
                    ancestor.rmdir()
                except OSError:
                    break
    return len(plan.delete)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--grace-min", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    jsonl_files = sorted(args.log_path.rglob("checkpoints.jsonl")) if args.log_path.is_dir() else []
    record_lists = [read_records(path) for path in jsonl_files]
    if not any(record_lists):
        print(f"no checkpoint records under {args.log_path}; nothing pruned", flush=True)
        return 0
    existing = list_checkpoint_dirs(args.checkpoint_root)
    plan = plan_prune(record_lists, existing, args.checkpoint_root, keep=args.keep, now=time.time(), grace_s=args.grace_min * 60)
    kept = sorted(d for d in existing if d in plan.keep)
    print(f"{len(jsonl_files)} checkpoints.jsonl, {sum(map(len, record_lists))} records, {len(existing)} directories: keep {len(kept)}, delete {len(plan.delete)}", flush=True)
    for directory in kept:
        print(f"keep {directory}: {plan.reasons.get(directory, '')}", flush=True)
    apply_plan(plan, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
