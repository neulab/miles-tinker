"""Leave unsolved step-limit rollouts out of the training group (TERMINAL_RL_STEP_LIMIT_PENALTY=drop).

Scoring an unsolved run into the step limit -1 (155841) or 0 (155931) gave the
same early trough: either way it sits in the group mean and is pushed down
whenever a sibling solved. The sign test showed that push-down spreads to text
the update never saw, and step-limit rollouts are ~30% of the rollouts and the
longest ones. Under "drop" they get no gradient at all.

The rollout's root rewards dict carries ``step_limit_unsolved`` (1.0/0.0, from
the last step's reward_misc: step_limit > 0 and task_reward < 1). After the
group's stats are tracked (so the logged reward, solve and step-limit rates still
count every rollout), the train workflow removes those rollouts before the
baseline, the all-equal check and the datums. Kept:

* solved runs that then used up the steps (reward 1, as before);
* a rollout that is malformed as well (``malformed_final``): its -1 (or, under
  targeted credit, its break-span penalty) is a signal worth keeping.

A group with fewer than 2 rollouts left is dropped (returns None), and active
sampling then draws another task. The eval workflow is untouched.

Logged: "[step-limit-drop]" per group that lost rollouts and the per-group
counter train/step_limit_dropped.

Usage: python drop_step_limit_rollouts.py <platoon-checkout>
(run after targeted_malformed_credit.py, whose root-rewards edit it extends)
"""

import sys
from pathlib import Path

MARKER = "step_limit_unsolved"
DATA = "platoon/utils/tinker_data_processing.py"
WORKFLOW = "platoon/train/tinker/workflows/group_rollout_workflow.py"

DATA_EDITS = (
    (
        "                root_rewards_dict[\"malformed_final\"] = float(float(_last_misc[\"tool_json_error_reward\"]) == 0.0)\n",
        "                root_rewards_dict[\"malformed_final\"] = float(float(_last_misc[\"tool_json_error_reward\"]) == 0.0)\n"
        "                # TERMINAL_RL_STEP_LIMIT_PENALTY=drop: the group workflow leaves these out.\n"
        "                root_rewards_dict[\"step_limit_unsolved\"] = float(\n"
        "                    float(_last_misc.get(\"step_limit\", 0) or 0) > 0\n"
        "                    and float(_last_misc.get(\"task_reward\", 0) or 0) < 1.0\n"
        "                )\n",
    ),
)

HELPER = '''
# TERMINAL_RL_STEP_LIMIT_PENALTY=drop: unsolved step-limit rollouts get no gradient
# (see deploy/babel/patches/drop_step_limit_rollouts.py).
_DROP_STEP_LIMIT = os.environ.get("TERMINAL_RL_STEP_LIMIT_PENALTY", "") == "drop"


def _drops_step_limit(result) -> bool:
    """An unsolved run into the step limit whose ending is not malformed."""
    rewards = result.root_rewards_dict or {}
    return bool(rewards.get("step_limit_unsolved")) and not rewards.get("malformed_final")
'''

WORKFLOW_EDITS = (
    ("\nclass GroupRolloutWorkflow", HELPER + "\n\nclass GroupRolloutWorkflow"),
    (
        "        # === Now compute advantages and filter ===\n",
        "        # === Now compute advantages and filter ===\n"
        "\n"
        "        if _DROP_STEP_LIMIT and self.stats_scope == \"train\":\n"
        "            kept = [result for result in valid_results if not _drops_step_limit(result)]\n"
        "            dropped = len(valid_results) - len(kept)\n"
        "            self.tracker.scalar(step_limit_dropped=float(dropped))\n"
        "            if dropped:\n"
        "                logger.info(\n"
        "                    f\"[step-limit-drop] {data['task_id']}: dropped {dropped} of {len(valid_results)} rollouts\"\n"
        "                    + (\"; group skipped\" if len(kept) < 2 else \"\")\n"
        "                )\n"
        "                if len(kept) < 2:\n"
        "                    return None\n"
        "                valid_results = kept\n"
        "                all_data = [datum for result in kept for datum in result.datums]\n"
        "                task_rewards = [result.task_reward for result in kept]\n"
        "                all_trajectory_stats = [stats for result in kept for stats in result.trajectory_stats]\n"
        "\n",
    ),
)


def apply(path: Path, edits) -> str:
    source = path.read_text()
    for anchor, _ in edits:
        if source.count(anchor) != 1:
            raise SystemExit(f"Could not find a unique {anchor.strip()[:80]!r} in {path}; this patch needs review.")
    for anchor, replacement in edits:
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    data, workflow = Path(sys.argv[1]) / DATA, Path(sys.argv[1]) / WORKFLOW
    for path in (data, workflow):
        if not path.is_file():
            raise SystemExit(f"No {path.name} at {path}")
    if MARKER in workflow.read_text():
        print(f"step-limit drop already applied in {workflow}")
        return
    new_data, new_workflow = apply(data, DATA_EDITS), apply(workflow, WORKFLOW_EDITS)
    data.write_text(new_data)
    workflow.write_text(new_workflow)
    print(f"TERMINAL_RL_STEP_LIMIT_PENALTY=drop available in {workflow}")


if __name__ == "__main__":
    main()
