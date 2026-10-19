"""Leave rollouts whose training sequence is too long out of the training group (TERMINAL_RL_MAX_DATUM_TOKENS).

The learner's peak memory grows with the longest single sequence it trains on:
with --fp32-lm-head the logits of one sequence take L x vocab/TP x 4 bytes per
GPU (12.2 GiB at ~52k tokens, 155785) and their gradient as much again, on top
of the model and optimizer states. 155785 (~52k) and 156491 (step 58) died of
CUDA OOM in forward_backward this way, while 59.6k and 68.1k sequences passed in
other steps -- the limit depends on what else is resident, so it is a cliff
somewhere around 50-70k. The server's --tinker-max-tokens-per-datum rejects an
over-long datum with an error, which would end the run just the same; this
drops the rollout on the client instead.

Measured on 14.5k training rollouts (work/context_lengths.py, longest call per
rollout): p99 40-44k tokens; > 44k is 0.14-0.76% of rollouts, > 48k 0.02-0.25%.

After the group's stats are tracked (logged reward / solve rate and the adaptive
sampler still count every rollout), the train workflow removes rollouts with a
datum over the limit before the baseline, the all-equal check and the datums. A
group with fewer than 2 rollouts left is dropped (returns None), and active
sampling then draws another task. The eval workflow is untouched (it does not
train).

Logged: "[long-datum-drop]" per group that lost rollouts and the per-group
counter train/long_datum_dropped.

Usage: python drop_long_datums.py <platoon-checkout>
(run after drop_step_limit_rollouts.py; both insert after the same anchor)
"""

import sys
from pathlib import Path

MARKER = "TERMINAL_RL_MAX_DATUM_TOKENS"
WORKFLOW = "platoon/train/tinker/workflows/group_rollout_workflow.py"

HELPER = '''
# TERMINAL_RL_MAX_DATUM_TOKENS: rollouts with a longer training sequence get no
# gradient, so one long trajectory cannot OOM the learner (see
# deploy/babel/patches/drop_long_datums.py). 0 = off.
_MAX_DATUM_TOKENS = int(os.environ.get("TERMINAL_RL_MAX_DATUM_TOKENS", "0") or 0)


def _longest_datum(result) -> int:
    """Tokens in the rollout's longest datum (model input plus the last target)."""
    return max((datum.model_input.length + 1 for datum in result.datums), default=0)
'''

WORKFLOW_EDITS = (
    ("\nclass GroupRolloutWorkflow", HELPER + "\n\nclass GroupRolloutWorkflow"),
    (
        "        # === Now compute advantages and filter ===\n",
        "        # === Now compute advantages and filter ===\n"
        "\n"
        "        if _MAX_DATUM_TOKENS > 0 and self.stats_scope == \"train\":\n"
        "            kept = [result for result in valid_results if _longest_datum(result) <= _MAX_DATUM_TOKENS]\n"
        "            dropped = len(valid_results) - len(kept)\n"
        "            self.tracker.scalar(long_datum_dropped=float(dropped))\n"
        "            if dropped:\n"
        "                longest = max(_longest_datum(result) for result in valid_results)\n"
        "                logger.info(\n"
        "                    f\"[long-datum-drop] {data['task_id']}: dropped {dropped} of {len(valid_results)} rollouts \"\n"
        "                    f\"over {_MAX_DATUM_TOKENS} tokens (longest {longest})\"\n"
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    workflow = Path(sys.argv[1]) / WORKFLOW
    if not workflow.is_file():
        raise SystemExit(f"No {workflow.name} at {workflow}")
    source = workflow.read_text()
    if MARKER in source:
        print(f"long-datum drop already applied in {workflow}")
        return
    for anchor, _ in WORKFLOW_EDITS:
        if source.count(anchor) != 1:
            raise SystemExit(f"Could not find a unique {anchor.strip()[:80]!r} in {workflow}; this patch needs review.")
    for anchor, replacement in WORKFLOW_EDITS:
        source = source.replace(anchor, replacement, 1)
    workflow.write_text(source)
    print(f"TERMINAL_RL_MAX_DATUM_TOKENS available in {workflow}")


if __name__ == "__main__":
    main()
