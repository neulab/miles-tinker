"""Adaptive training-task selection (platoon.terminal_rl.adaptive_tasks).

Under PLATOON_TASK_SELECTION=adaptive:

* the training task loader takes each batch from adaptive_tasks.draw instead of
  ``train_dataloader.get_batch(i_batch)`` (active sampling's pacing is kept: the
  loader still hands out batch_size tasks at a time and waits on consumption);
* GroupRolloutWorkflow.arun_episode reports every training group's root task
  rewards to adaptive_tasks.observe as soon as its rollouts are back, before the
  step-limit drop and the all-equal filter.

Unset, both hooks are skipped. Order-sensitive: after train_batch_start.py (the
loader) and before drop_step_limit_rollouts.py is irrelevant -- it anchors on the
``valid_results`` line, which no other patch rewrites.

Usage: python adaptive_task_selection.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "adaptive_tasks"
LOOP = "platoon/train/tinker/rl.py"
LOOP_ANCHOR = "            batch = shared_state.train_dataloader.get_batch(i_batch)\n"
LOOP_REPLACEMENT = (
    "            # PLATOON_TASK_SELECTION=adaptive: draw by learnability (platoon.terminal_rl.adaptive_tasks).\n"
    "            from platoon.terminal_rl import adaptive_tasks\n"
    "\n"
    "            if adaptive_tasks.enabled():\n"
    "                batch = adaptive_tasks.draw(self, shared_state.train_dataloader, self.config.train.batch_size)\n"
    "            else:\n"
    "                batch = shared_state.train_dataloader.get_batch(i_batch)\n"
)
WORKFLOW = "platoon/train/tinker/workflows/group_rollout_workflow.py"
WORKFLOW_ANCHOR = "        valid_results = [result for result in results if result is not None]\n"
WORKFLOW_REPLACEMENT = WORKFLOW_ANCHOR + (
    "        if self.stats_scope == \"train\":\n"
    "            # Adaptive task selection learns from every scored group, filtered or not.\n"
    "            from platoon.terminal_rl import adaptive_tasks\n"
    "\n"
    "            adaptive_tasks.observe(data[\"task_id\"], [r.task_reward for r in valid_results], self.tracker)\n"
)


def _patch(path: Path, anchor: str, replacement: str, what: str) -> None:
    source = path.read_text()
    if MARKER in source:
        print(f"adaptive task selection already present in {path}")
        return
    if source.count(anchor) != 1:
        raise SystemExit(f"Could not find {what} in {path}; this patch needs review.")
    path.write_text(source.replace(anchor, replacement, 1))
    print(f"PLATOON_TASK_SELECTION=adaptive available in {path}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    for rel, anchor, replacement, what in (
        (LOOP, LOOP_ANCHOR, LOOP_REPLACEMENT, "the task loader's get_batch call"),
        (WORKFLOW, WORKFLOW_ANCHOR, WORKFLOW_REPLACEMENT, "arun_episode's valid_results"),
    ):
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"No {rel} under {root}")
        _patch(path, anchor, replacement, what)


if __name__ == "__main__":
    main()
