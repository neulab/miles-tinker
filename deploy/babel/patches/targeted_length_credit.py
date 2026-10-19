"""Put the output-length penalty on the long response's own tokens (TERMINAL_RL_LENGTH_PENALTY_CREDIT=targeted).

By default (openhands_env.py, "trajectory") a trajectory whose longest completion
has n > N tokens loses alpha * ln(n / N) from its reward, and GRPO spreads that
over every action token of a 30-50 turn rollout. In 157001 that did not stop call
stacking: turns with 2-9 short calls were never penalised (0 of 156 such
rollouts), only the tail that reached 16k, and the trajectory-level push on the
offending turn was tiny.

Under "targeted" the environment leaves the trajectory reward alone, and this
patch, after group centering, gives every response longer than N tokens a fixed
advantage on its overflow tokens -- the ones after the N-th, where it should have
stopped (for a stacked turn, the extra calls):

    -alpha * ln(n / N) * trajectory_tokens / overflow_tokens

weighted like targeted malformed credit, so after the per-trajectory loss
normalisation the overflow span counts as one trajectory with advantage
-alpha * ln(n / N). Responses are the contiguous runs of trained tokens in a
datum (tool output and template tokens are masked out between them).

Two more edits make the penalty reach the loss:
* a train group whose rewards are all equal is still kept when one of its
  rollouts has a long response (its advantages are zero except the spans);
* under TERMINAL_RL_STEP_LIMIT_PENALTY=drop, an unsolved step-limit rollout with a
  long response is kept as penalty-only: left out of the baseline, advantages zero
  except its spans. Otherwise "drop" would hide exactly the runs that stacked.

Logged: "[length-credit]" per group with penalised responses and the counter
train/length_penalised_responses. The eval workflow is untouched.

Usage: python targeted_length_credit.py <platoon-checkout>
(run after drop_long_datums.py; it edits text that targeted_malformed_credit.py
and drop_step_limit_rollouts.py put in place)
"""

import sys
from pathlib import Path

MARKER = "TERMINAL_RL_LENGTH_PENALTY_CREDIT"
WORKFLOW = "platoon/train/tinker/workflows/group_rollout_workflow.py"

HELPER = '''
# TERMINAL_RL_LENGTH_PENALTY_CREDIT=targeted: the length penalty goes on each long
# response's overflow tokens instead of the trajectory reward (see
# deploy/babel/patches/targeted_length_credit.py).
_LENGTH_CREDIT = os.environ.get("TERMINAL_RL_LENGTH_PENALTY_CREDIT", "trajectory")
_LENGTH_N = int(os.environ.get("TERMINAL_RL_LENGTH_PENALTY_N", "0") or 0)
_LENGTH_ALPHA = float(os.environ.get("TERMINAL_RL_LENGTH_PENALTY_ALPHA", "0.3") or 0.3)


def _length_targeted() -> bool:
    return _LENGTH_CREDIT == "targeted" and _LENGTH_N > 0


def _response_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """[start, end) of each contiguous run of trained tokens (one per response)."""
    runs, start = [], None
    for i, on in enumerate((mask > 0).tolist()):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _long_responses(result) -> int:
    """How many responses of the rollout are longer than the length budget."""
    if not _length_targeted():
        return 0
    return sum(
        end - start > _LENGTH_N
        for datum in result.datums
        for start, end in _response_runs(datum.loss_fn_inputs["mask"].to_torch())
    )


def _apply_length_penalty(result, zero_rest: bool = False) -> tuple[int, int]:
    """Penalise the overflow tokens of every long response; (responses, longest).

    zero_rest: set every other trained token's advantage to 0 first (a penalty-only
    rollout that took no part in the baseline).
    """
    if "traj_tokens" in result.datums[0].loss_fn_inputs:
        traj_tokens = float(result.datums[0].loss_fn_inputs["traj_tokens"].to_torch().reshape(-1)[0])
    else:
        traj_tokens = float(sum(float(d.loss_fn_inputs["mask"].to_torch().sum()) for d in result.datums))
    count, longest = 0, 0
    for datum in result.datums:
        mask = datum.loss_fn_inputs["mask"].to_torch()
        advantages = datum.loss_fn_inputs["advantages"].to_torch().clone().float()
        if zero_rest:
            advantages = torch.where(mask > 0, torch.zeros_like(advantages), advantages)
        for start, end in _response_runs(mask):
            n = end - start
            if n <= _LENGTH_N:
                continue
            penalty = _LENGTH_ALPHA * math.log(n / _LENGTH_N)
            advantages[start + _LENGTH_N : end] = -penalty * traj_tokens / (n - _LENGTH_N)
            count += 1
            longest = max(longest, n)
        datum.loss_fn_inputs["advantages"] = TensorData.from_torch(advantages)
    return count, longest
'''

WORKFLOW_EDITS = (
    ("\nclass GroupRolloutWorkflow", HELPER + "\n\nclass GroupRolloutWorkflow"),
    # Remember the drop-mode rollouts that carry a long response, before the drop.
    (
        "        if _DROP_STEP_LIMIT and self.stats_scope == \"train\":\n",
        "        penalty_only = []\n"
        "        if _DROP_STEP_LIMIT and self.stats_scope == \"train\" and _length_targeted():\n"
        "            penalty_only = [r for r in valid_results if _drops_step_limit(r) and _long_responses(r)]\n"
        "\n"
        "        if _DROP_STEP_LIMIT and self.stats_scope == \"train\":\n",
    ),
    # Keep all-equal groups that have a long response.
    (
        "            and not has_malformed\n"
        "        ):\n",
        "            and not has_malformed\n"
        "            and not has_long\n"
        "        ):\n",
    ),
    (
        "        has_malformed = _MALFORMED_CREDIT == \"targeted\" and any(\n",
        "        has_long = self.stats_scope == \"train\" and any(_long_responses(r) for r in valid_results + penalty_only)\n"
        "        has_malformed = _MALFORMED_CREDIT == \"targeted\" and any(\n",
    ),
    # After centering (and the malformed spans), place the length spans.
    (
        "            self.tracker.scalar(malformed_tail_penalized=float(penalized))\n"
        "\n"
        "        return all_data\n",
        "            self.tracker.scalar(malformed_tail_penalized=float(penalized))\n"
        "\n"
        "        if has_long:\n"
        "            responses, longest, kept_only = 0, 0, 0\n"
        "            for result in valid_results:\n"
        "                count, top = _apply_length_penalty(result)\n"
        "                responses, longest = responses + count, max(longest, top)\n"
        "            for result in penalty_only:\n"
        "                count, top = _apply_length_penalty(result, zero_rest=True)\n"
        "                responses, longest = responses + count, max(longest, top)\n"
        "                all_data.extend(result.datums)\n"
        "                kept_only += 1\n"
        "            self.tracker.scalar(length_penalised_responses=float(responses))\n"
        "            if responses:\n"
        "                logger.info(\n"
        "                    f\"[length-credit] {data['task_id']}: {responses} responses over {_LENGTH_N} tokens \"\n"
        "                    f\"(longest {longest}); {kept_only} penalty-only step-limit rollouts kept\"\n"
        "                )\n"
        "\n"
        "        return all_data\n",
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
        print(f"targeted length credit already applied in {workflow}")
        return
    for anchor, _ in WORKFLOW_EDITS:
        if source.count(anchor) != 1:
            raise SystemExit(f"Could not find a unique {anchor.strip()[:80]!r} in {workflow}; this patch needs review.")
    for anchor, replacement in WORKFLOW_EDITS:
        source = source.replace(anchor, replacement, 1)
    if "\nimport math\n" not in source:
        source = source.replace("\nimport os\n", "\nimport math\nimport os\n", 1)
    workflow.write_text(source)
    print(f"TERMINAL_RL_LENGTH_PENALTY_CREDIT=targeted available in {workflow}")


if __name__ == "__main__":
    main()
