"""Penalise a malformed final response on the tokens where it broke (TERMINAL_RL_MALFORMED_CREDIT=targeted).

By default a malformed final response scores its whole trajectory -1, and GRPO
spreads that over every action token: in job 155806 a malformed rollout carried a
median ~18.5k generated tokens, so the token where the call broke got about
1/18,500 of the signal while the normal work before it was pushed down. Long file
writes kept breaking more often as training went on (8k+ char writes: 0/66 at v0
-> 5/18 by v6 in 155717), and the mean-KL anchor cannot see a per-token derail
rate of ~1e-4.

Under "targeted" the environment scores the trajectory on the task instead
(openhands_env.MALFORMED_CREDIT), and this patch, after group centering, puts a
fixed advantage of -TERMINAL_RL_MALFORMED_PENALTY (default 1.0) on the break span
of the final response, found by platoon.terminal_rl.malformed_break.penalty_span
(the stop token of a call left open; the opener of a call started inside the
reasoning; up to 64 tokens from fake tool output or the onset of a loop). The span
is weighted like a whole trajectory: its tokens get
-penalty * trajectory_tokens / span_tokens, which the loss normalisation then
scales by 1 / (n_trajectories * trajectory_tokens), so the span as a whole counts
as one trajectory with advantage -penalty.

Two smaller edits make that reach the loss:

* the rollout's root rewards dict carries ``malformed_final`` (1.0/0.0, from the
  last step's tool_json_error_reward), logged as train/root_malformed_final;
* a group whose rewards are all equal is still kept when one of its rollouts has
  a malformed ending: its advantages are all zero except the break spans.

Logged: "[malformed-credit]" per penalised rollout (kind, span, task) and the
counter train/malformed_tail_penalized.

Usage: python targeted_malformed_credit.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "TERMINAL_RL_MALFORMED_CREDIT"
DATA = "platoon/utils/tinker_data_processing.py"
WORKFLOW = "platoon/train/tinker/workflows/group_rollout_workflow.py"

DATA_EDITS = (
    (
        "            root_rewards_dict = rewards_dict\n",
        "            root_rewards_dict = rewards_dict\n"
        "            # TERMINAL_RL_MALFORMED_CREDIT: flag a malformed final response for the\n"
        "            # group workflow's targeted penalty (and the root_malformed_final metric).\n"
        "            _last_misc = ((trajectory.get(\"steps\") or [{}])[-1].get(\"misc\") or {}).get(\"reward_misc\") or {}\n"
        "            if \"tool_json_error_reward\" in _last_misc:\n"
        "                root_rewards_dict = dict(rewards_dict)\n"
        "                root_rewards_dict[\"malformed_final\"] = float(float(_last_misc[\"tool_json_error_reward\"]) == 0.0)\n",
    ),
)

HELPER = '''
# TERMINAL_RL_MALFORMED_CREDIT=targeted: penalise a malformed final response on
# the tokens where it broke instead of its whole trajectory (see
# deploy/babel/patches/targeted_malformed_credit.py).
_MALFORMED_CREDIT = os.environ.get("TERMINAL_RL_MALFORMED_CREDIT", "trajectory")
_MALFORMED_PENALTY = float(os.environ.get("TERMINAL_RL_MALFORMED_PENALTY", "1.0"))


def _final_response_range(mask: torch.Tensor) -> tuple[int, int] | None:
    """[start, end) of the last contiguous run of trained tokens (the final response)."""
    trained = (mask > 0).nonzero().flatten()
    if len(trained) == 0:
        return None
    end = int(trained[-1]) + 1
    start = end - 1
    while start > 0 and mask[start - 1] > 0:
        start -= 1
    return start, end


def _apply_malformed_penalty(result, tokenizer) -> tuple[str, int, int] | None:
    """Put the targeted penalty on a malformed rollout's break span; (kind, a, b) or None."""
    from platoon.terminal_rl.malformed_break import penalty_span

    if not result.datums or not (result.root_rewards_dict or {}).get("malformed_final"):
        return None
    datum = result.datums[-1]
    mask = datum.loss_fn_inputs["mask"].to_torch()
    span = _final_response_range(mask)
    if span is None:
        return None
    start, end = span
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()[start:end].tolist()
    pieces = [tokenizer.decode([token]) for token in targets]
    kind, a, b = penalty_span(pieces)
    if b <= a:
        return None
    if "traj_tokens" in datum.loss_fn_inputs:
        traj_tokens = float(datum.loss_fn_inputs["traj_tokens"].to_torch().reshape(-1)[0])
    else:
        traj_tokens = float(sum(float(d.loss_fn_inputs["mask"].to_torch().sum()) for d in result.datums))
    advantages = datum.loss_fn_inputs["advantages"].to_torch().clone().float()
    advantages[start + a : start + b] = -_MALFORMED_PENALTY * traj_tokens / (b - a)
    datum.loss_fn_inputs["advantages"] = TensorData.from_torch(advantages)
    return kind, a, b
'''

WORKFLOW_EDITS = (
    ("\nclass GroupRolloutWorkflow", HELPER + "\n\nclass GroupRolloutWorkflow"),
    (
        "        if len(all_trajectory_rewards) > 1 and max(all_trajectory_rewards) == min(all_trajectory_rewards):\n",
        "        has_malformed = _MALFORMED_CREDIT == \"targeted\" and any(\n"
        "            (result.root_rewards_dict or {}).get(\"malformed_final\") for result in valid_results\n"
        "        )\n"
        "        if (\n"
        "            len(all_trajectory_rewards) > 1\n"
        "            and max(all_trajectory_rewards) == min(all_trajectory_rewards)\n"
        "            and not has_malformed\n"
        "        ):\n",
    ),
    (
        '                datum.loss_fn_inputs["advantages"] = TensorData.from_torch(new_advantages)\n'
        "\n"
        "        return all_data\n",
        '                datum.loss_fn_inputs["advantages"] = TensorData.from_torch(new_advantages)\n'
        "\n"
        "        if has_malformed:\n"
        "            tokenizer = getattr(getattr(self.model_info.llm, \"renderer\", None), \"tokenizer\", None)\n"
        "            penalized = 0\n"
        "            for result in valid_results:\n"
        "                try:\n"
        "                    applied = _apply_malformed_penalty(result, tokenizer) if tokenizer is not None else None\n"
        "                except Exception as error:  # a diagnostic shape problem must not drop the group\n"
        "                    logger.warning(f\"[malformed-credit] {data['task_id']}: could not place the penalty ({error})\")\n"
        "                    applied = None\n"
        "                if applied:\n"
        "                    penalized += 1\n"
        "                    kind, a, b = applied\n"
        "                    logger.info(f\"[malformed-credit] {data['task_id']}: {kind}, final-response tokens {a}..{b}\")\n"
        "            self.tracker.scalar(malformed_tail_penalized=float(penalized))\n"
        "\n"
        "        return all_data\n",
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
        print(f"targeted malformed credit already applied in {workflow}")
        return
    new_data, new_workflow = apply(data, DATA_EDITS), apply(workflow, WORKFLOW_EDITS)
    data.write_text(new_data)
    workflow.write_text(new_workflow)
    print(f"TERMINAL_RL_MALFORMED_CREDIT=targeted available in {workflow}")


if __name__ == "__main__":
    main()
