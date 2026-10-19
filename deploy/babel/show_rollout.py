"""Render Platoon trajectory event logs as readable transcripts.

A rollout's events_*.jsonl holds every recorded action with its thought,
arguments and observation, plus the verifier's reward. This turns one into
something a person can read, which is the only way to judge whether an agent
stopped because it was done, because it was stuck, or because the harness
stopped recording.

Usage:
    python show_rollout.py <run-dir> --out <dir> [--reward 1.0] [--limit 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def text_of(parts) -> str:
    if isinstance(parts, str):
        return parts
    return "".join(p.get("text", "") for p in (parts or []) if isinstance(p, dict))


def load(path: str) -> dict:
    steps, reward, task_id, goal, error, finish = [], None, None, None, None, None
    for line in open(path, errors="replace"):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        kind = ev.get("type")
        if kind == "trajectory_task_set":
            task = ev.get("task") or {}
            task_id, goal = task.get("id"), task.get("goal")
        elif kind == "trajectory_step_added":
            steps.append(ev.get("step") or {})
        elif kind == "trajectory_finished":
            traj = ev.get("trajectory") or {}
            error, finish = traj.get("error_message"), traj.get("finish_message")
            if traj.get("reward") is not None:
                reward = traj["reward"]
    # The top-level reward is only assigned when the episode loop recognises the
    # conversation as finished, which it usually does not; the verifier's own
    # output is recorded per step and is the reliable source.
    if reward is None:
        for step in steps:
            verifier = ((step.get("misc") or {}).get("reward_misc") or {}).get("verifier") or ""
            found = re.search(r"Reward:\s*([0-9.]+)", verifier)
            if found:
                reward = float(found.group(1))
    return {"steps": steps, "reward": reward, "task_id": task_id, "goal": goal,
            "error": error, "finish": finish, "path": path}


def render(traj: dict) -> str:
    out = ["=" * 78, f"TRAJECTORY  {traj['task_id']}", f"reward: {traj['reward']}", "=" * 78, ""]
    if traj["goal"]:
        out += ["--- TASK ---", traj["goal"].strip()[:1500], ""]
    for i, step in enumerate(traj["steps"], 1):
        actions = step.get("action_events") or {}
        events = actions.get("action_events") if isinstance(actions, dict) else actions
        for action in events or []:
            body = action.get("action") or {}
            args = {k: v for k, v in body.items() if k != "kind" and v not in (None, [], "")}
            out += [
                "-" * 78,
                f"STEP {i}: {action.get('tool_name')}",
                "--- thought ---",
                text_of(action.get("thought")).strip() or "(none)",
                "--- arguments ---",
                json.dumps(args, indent=2)[:2500],
            ]
        for obs in step.get("observation_events") or []:
            if obs.get("kind") == "ConversationStateUpdateEvent":
                continue
            body = obs.get("observation") or {}
            out += ["--- observation ---", ANSI.sub("", text_of(body.get("content")))[:1800].strip()]
        if step.get("reward"):
            out.append(f"--- step reward: {step['reward']} ---")
        out.append("")
    out += ["=" * 78,
            f"steps recorded : {len(traj['steps'])}",
            f"final reward   : {traj['reward']}",
            f"finish message : {traj['finish'] or '(none)'}",
            f"error          : {(traj['error'] or '(none)').strip()[:300]}"]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reward", type=float, help="only trajectories with this reward")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--min-steps", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    written = 0
    for path in sorted(glob.glob(os.path.join(args.run_dir, "**", "events_*.jsonl"), recursive=True)):
        traj = load(path)
        if not traj["steps"] or len(traj["steps"]) < args.min_steps:
            continue
        if args.reward is not None and traj["reward"] != args.reward:
            continue
        name = f"{traj['task_id']}_r{traj['reward']}_{len(traj['steps'])}steps.txt"
        with open(os.path.join(args.out, name), "w") as handle:
            handle.write(render(traj))
        print(f"  wrote {name}")
        written += 1
        if written >= args.limit:
            break
    if not written:
        print("no matching trajectories")


if __name__ == "__main__":
    main()
