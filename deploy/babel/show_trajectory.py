"""Print saved rollout trajectories from a training run.

Usage:
    python deploy/babel/show_trajectory.py RUN_DIR              # list them
    python deploy/babel/show_trajectory.py RUN_DIR --index 2    # print one
    python deploy/babel/show_trajectory.py RUN_DIR --task mysql --full

RUN_DIR is the directory the launcher reported as "training outputs".
"""

import argparse
import json
from pathlib import Path

RULE = "=" * 78


def _load(path: Path) -> dict:
    steps, finished, task = [], None, None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "trajectory_step_added":
            steps.append(event["step"])
        elif event["type"] == "trajectory_finished":
            finished = event
        elif event["type"] == "trajectory_task_set":
            task = event.get("task")
    name = path.name[len("events_") :].rsplit("_", 1)[0]
    return {
        "path": path,
        "name": name.replace("litecoder_terminal_rl__", ""),
        "steps": steps,
        "reward": (finished or {}).get("reward"),
        "finish_message": (finished or {}).get("finish_message"),
        "error_message": (finished or {}).get("error_message"),
        "goal": (task or {}).get("goal") or "",
    }


def _clip(text: str, limit: int | None) -> str:
    text = (text or "").strip()
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n    ...[{len(text) - limit} more chars; use --full]"


def _print(trajectory: dict, limit: int | None, show_goal: bool) -> None:
    print(f"\n{RULE}\nTASK    {trajectory['name']}")
    print(f"REWARD  {trajectory['reward']}   TURNS  {len(trajectory['steps'])}")
    if trajectory["finish_message"]:
        print(f"FINISH  {trajectory['finish_message']}")
    if trajectory["error_message"]:
        print(f"ERROR   {trajectory['error_message'].splitlines()[0]}")
    if show_goal:
        print(f"\n--- TASK TEXT ---\n{_clip(trajectory['goal'], limit)}")
    for index, step in enumerate(trajectory["steps"]):
        usage = step.get("misc", {}).get("action_misc", {}).get("usage") or {}
        tokens = usage.get("completion_tokens")
        print(f"\n{'-' * 78}\nTURN {index}" + (f"   ({tokens} completion tokens)" if tokens else ""))
        if step.get("thought"):
            print(f"\n  THOUGHT\n{_clip(step['thought'], limit)}")
        print(f"\n  COMMAND\n{_clip(step.get('code') or '', limit)}")
        body = step.get("output") or step.get("error") or "(no output)"
        print(f"\n  OBSERVATION\n{_clip(body, limit)}")
        if step.get("reward"):
            print(f"\n  REWARD {step['reward']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--index", type=int, help="Print this trajectory from the listing.")
    parser.add_argument("--task", help="Print every trajectory whose task name contains this.")
    parser.add_argument("--full", action="store_true", help="Do not clip long text.")
    parser.add_argument("--chars", type=int, default=1500, help="Clip length without --full.")
    parser.add_argument("--goal", action="store_true", help="Also print the task instructions.")
    args = parser.parse_args()

    files = sorted(args.run_dir.glob("**/events_*.jsonl"))
    if not files:
        raise SystemExit(f"No trajectories under {args.run_dir}")
    trajectories = [_load(path) for path in files]
    limit = None if args.full else args.chars

    if args.index is None and not args.task:
        print(f"{len(trajectories)} trajectories in {args.run_dir}\n")
        for index, trajectory in enumerate(trajectories):
            print(f"  [{index}] {trajectory['name']:<40} turns={len(trajectory['steps']):<4} reward={trajectory['reward']}")
        print("\nPrint one with --index N, or all of a task with --task NAME.")
        return

    chosen = [trajectories[args.index]] if args.index is not None else [t for t in trajectories if args.task in t["name"]]
    if not chosen:
        raise SystemExit(f"No trajectory matching {args.task!r}")
    for trajectory in chosen:
        _print(trajectory, limit, args.goal)


if __name__ == "__main__":
    main()
