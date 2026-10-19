"""Task lookup over a staged terminal-task pool.

``prepare_litecoder.py`` and ``prepare_tmax.py`` write the pool: ``selected.json`` lists
the staged tasks, ``tasks/<task_id>/`` holds each task's files and ``sifs/``
holds the images. Only staged tasks are returned, so the sandbox never asks for
an image that is not on disk.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from platoon.envs.base import Task

DATA_DIR_ENV = "TERMINAL_RL_DATA_DIR"
# Held-out tasks. The default suits a smoke pool; raise it for a full 602-task run.
EVAL_SIZE = int(os.getenv("TERMINAL_RL_EVAL_SIZE", "2"))
# tmax's VANILLUX_CALL_LIMIT: the per-episode bash-call budget. Note that
# GroupRolloutWorkflow overwrites Task.max_steps with rollout_config.max_steps
# when that is set, so the yaml wins over this default during training.
DEFAULT_MAX_STEPS = int(os.getenv("VANILLUX_CALL_LIMIT", "100"))

Split = Literal["train", "val"]

# How the held-out tasks are chosen. "hashed" (default) drops the excluded tasks
# first, then holds out the EVAL_SIZE tasks with the smallest sha256 of their id:
# a fixed pseudo-random sample spread over the whole pool, which stays the same
# set for a given size and grows by adding tasks when the size grows. "tail"
# restores the old split -- the last EVAL_SIZE task ids, excluded tasks kept in
# the eval set -- for comparisons against runs that used it. The tail is the
# highest-numbered ids, not a sample, and held one unsolvable excluded task.
EVAL_SPLIT = os.getenv("TERMINAL_RL_EVAL_SPLIT", "hashed")

# TERMINAL_RL_TASK_LIST: a file of task ids (one per line, "#" comments) that
# replaces both splits -- training and eval run on exactly these tasks. For the
# fixed-task learning test: if reward on tasks it sees again and again does not
# climb, the pipeline is broken; if it does, the problem is signal and
# generalisation. Unset (default) keeps the normal held-out split.
TASK_LIST_ENV = "TERMINAL_RL_TASK_LIST"
# TERMINAL_RL_EVAL_TASK_LIST: with a task list, a separate file for the eval split
# (e.g. a held-out set of the same difficulty); unset, eval runs on the task list.
EVAL_TASK_LIST_ENV = "TERMINAL_RL_EVAL_TASK_LIST"


def listed_task_ids(env: str = TASK_LIST_ENV) -> list[str] | None:
    path = os.environ.get(env)
    if not path:
        return None
    ids = (line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines())
    return [i for i in ids if i]


# Tasks whose environment cannot be set up in the training sandbox (missing
# headers in the image, overlay rename limits, ...); see excluded_tasks.txt.
# Point the variable at another file, or set it to "" to use every staged task.
EXCLUDED_TASKS_ENV = "TERMINAL_RL_EXCLUDED_TASKS"
DEFAULT_EXCLUDED_TASKS = Path(__file__).with_name("excluded_tasks.txt")


@functools.cache
def excluded_task_ids() -> frozenset[str]:
    path = os.environ.get(EXCLUDED_TASKS_ENV, str(DEFAULT_EXCLUDED_TASKS))
    if not path or not Path(path).is_file():
        return frozenset()
    ids = (line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines())
    return frozenset(i for i in ids if i)


def data_dir() -> Path:
    directory = os.environ.get(DATA_DIR_ENV)
    if not directory:
        raise RuntimeError(f"Set {DATA_DIR_ENV} to the directory prepare_litecoder.py wrote.")
    return Path(directory).expanduser().resolve()


@functools.cache
def _staged() -> dict[str, dict]:
    """Map task id to its staged entry, keyed in a stable order."""
    root = data_dir()
    entries = json.loads((root / "selected.json").read_text())
    staged: dict[str, dict] = {}
    for entry in sorted(entries, key=lambda item: item["task_id"]):
        task_dir = root / "tasks" / entry["task_id"]
        if not task_dir.is_dir():
            continue
        staged[entry["task_id"]] = {**entry, "task_dir": task_dir}
    if not staged:
        raise RuntimeError(f"No staged tasks under {root}; run <science-agent>/experiments/qwen38_27b_tinker_amd/pool/prepare_litecoder.py.")
    return staged


def get_task_ids(split: Split = "train", eval_size: int | None = None) -> list[str]:
    """Split the staged tasks deterministically into train and held-out eval."""
    eval_size = EVAL_SIZE if eval_size is None else eval_size
    task_ids = list(_staged())
    listed = listed_task_ids()
    eval_listed = listed_task_ids(EVAL_TASK_LIST_ENV)
    if listed is None and eval_listed is not None:
        # An eval list on its own (e.g. Terminal-Bench 2.0 staged as an extra pool,
        # <science-agent>/experiments/qwen38_27b_tinker_amd/pool/build_tb2_pool.py) replaces the hashed held-out split: eval
        # runs exactly those tasks and training uses every other kept task.
        if split == "val":
            missing = [t for t in eval_listed if t not in set(task_ids)]
            if missing:
                raise RuntimeError(f"{EVAL_TASK_LIST_ENV} names {len(missing)} tasks that are not staged, e.g. {missing[:3]}")
            return eval_listed
        held = set(eval_listed)
        return [t for t in task_ids if t not in held and t not in excluded_task_ids() and not _staged()[t].get("eval_only")]
    if listed is not None and split == "val" and listed_task_ids(EVAL_TASK_LIST_ENV) is not None:
        listed = listed_task_ids(EVAL_TASK_LIST_ENV)
    if listed is not None:
        staged = set(task_ids)
        missing = [t for t in listed if t not in staged]
        if missing:
            raise RuntimeError(f"{TASK_LIST_ENV} names {len(missing)} tasks that are not staged, e.g. {missing[:3]}")
        return listed
    if EVAL_SPLIT == "hashed":
        kept = [t for t in task_ids if t not in excluded_task_ids() and not _staged()[t].get("eval_only")]
        if eval_size <= 0 or eval_size >= len(kept):
            return kept if split == "train" else []
        by_hash = sorted(kept, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
        held_out = set(by_hash[:eval_size])
        return [t for t in kept if (t in held_out) == (split == "val")]
    if EVAL_SPLIT != "tail":
        raise ValueError(f"TERMINAL_RL_EVAL_SPLIT must be 'hashed' or 'tail', not {EVAL_SPLIT!r}")
    if eval_size <= 0 or eval_size >= len(task_ids):
        return [t for t in task_ids if t not in excluded_task_ids()] if split == "train" else task_ids
    if split == "train":
        return [t for t in task_ids[:-eval_size] if t not in excluded_task_ids()]
    return task_ids[-eval_size:]


@functools.cache
def get_task(id: str) -> Task:
    entry = _staged().get(id)
    if entry is None:
        raise KeyError(f"Task {id!r} is not staged; stage it with prepare_litecoder.py --task-id {id}")
    task_dir: Path = entry["task_dir"]
    instruction_file = task_dir / "instruction.md"
    instruction = instruction_file.read_text(encoding="utf-8").strip() if instruction_file.is_file() else ""
    return Task(
        goal=instruction or f"Complete the task in {id}.",
        id=id,
        max_steps=DEFAULT_MAX_STEPS,
        misc={
            "task_dir": str(task_dir),
            "image": entry["image"],
            "sif_name": entry.get("sif_name"),
            # Per-task verifier timeout and test working directory (TB2 ships both).
            "test_timeout": entry.get("test_timeout"),
            "workdir": entry.get("workdir"),
            "instruction": instruction,
            # Harbor separate-verifier tasks (prepare_harbor.py): files uploaded at
            # reset, the agent's starting directory, and the verifier image with its
            # own uploads and the artifact paths carried over from the agent.
            "harbor": entry.get("harbor", False),
            "inputs": entry.get("inputs") or [],
            "agent_workdir": entry.get("agent_workdir"),
            "verifier": entry.get("verifier"),
        },
    )
