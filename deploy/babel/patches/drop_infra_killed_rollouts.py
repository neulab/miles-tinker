"""Keep infrastructure-killed rollouts out of the GRPO group.

``run_episode`` bounds both halves of the loop with the same ``step_timeout``::

    action = await asyncio.wait_for(agent.act(obs),   timeout=timeout)
    obs    = await asyncio.wait_for(env.step(action), timeout=timeout)

When ``get_actions_for_last_obs`` goes permanently empty, ``act`` polls at 0.2s
until that deadline fires, the episode is truncated mid-task, and the verifier
scores whatever the sandbox happens to hold -- in practice the ``tool_json``
shaping term alone. Measured over batch 0 of job 10617059: 108 of 237 finished
rollouts died this way, scoring 0.133 against 0.536 for clean ones, and only 4
of the 108 had actually solved their task.

Those rollouts are not dropped today. ``advantage = reward - group_mean`` hands
each of them roughly -0.2, applied to tokens that were ordinary agent behaviour
right up to the hang, so the policy is pushed away from conduct that merely
preceded an infrastructure fault. The push has a direction: the timeout rate
climbs with episode length (31/44/51/57% across length quartiles), and reward by
length falls 0.368/0.463/0.320/0.260 over all rollouts but only
0.491/0.616/0.584/0.455 over clean ones. The extra decline is the fault, not
task difficulty, and it is a gradient toward finishing early.

Dropping beats zeroing the advantage because the group mean is computed from
``task_rewards`` (group_rollout_workflow.py:214). Leaving the rollouts in with a
zero advantage still drags that baseline down, which inflates every clean
rollout's advantage and -- worse -- gives a genuine failure the same advantage as
an infrastructure one. Returning None instead removes them before the baseline,
before the ``max == min`` degeneracy check, and before the datums are collected,
all of which already handle a None rollout.

Only asyncio truncation counts as infrastructure. ``Agent got stuck`` and
``Exhausted budget when running episode`` are real policy failures and are left
in the group, where they belong.

Cost: on batch 0 this drops the usable-group count from 22/26 to 19/26 and about
45% of the tokens, and the reward the tracker reports will be clean-rollout-only
and so read higher than before. Set PLATOON_KEEP_INFRA_KILLED=1 to disable.

Usage: python drop_infra_killed_rollouts.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_infra_killed"
TARGET = "platoon/train/tinker/workflows/group_rollout_workflow.py"

ANCHOR = """                if not results.get("trajectories"):
                    logger.warning(f"No trajectories found for task {task_id} and rollout {rollout_number}")
                    return None
"""

REPLACEMENT = '''                if not results.get("trajectories"):
                    logger.warning(f"No trajectories found for task {task_id} and rollout {rollout_number}")
                    return None

                if _infra_killed(results):
                    logger.warning(
                        f"[{task_id}] rollout {rollout_number} truncated by step_timeout; "
                        "dropping it from the group so it does not bias the baseline"
                    )
                    return None
'''

HELPER_ANCHOR = """class GroupRolloutWorkflow"""

HELPER = '''def _infra_killed(trajectory_collection: dict) -> bool:
    """True when the root episode was truncated by the loop's asyncio deadline.

    ``run_episode`` records the traceback in ``error_message``. A TimeoutError or
    a CancelledError there means the episode was cut mid-task by ``step_timeout``
    rather than ended by the agent, so its reward measures the fault and not the
    policy. "Agent got stuck" and "Exhausted budget" are genuine policy failures
    and deliberately do not match.
    """
    if os.environ.get("PLATOON_KEEP_INFRA_KILLED") == "1":
        return False
    trajectories = trajectory_collection.get("trajectories") or {}
    if not trajectories:
        return False
    # The root is the first trajectory, as _compute_trajectory_depths assumes.
    root = next(iter(trajectories.values()))
    message = root.get("error_message") if isinstance(root, dict) else getattr(root, "error_message", None)
    if not message:
        return False
    return "TimeoutError" in message or "CancelledError" in message


class GroupRolloutWorkflow'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No group_rollout_workflow at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"infra-killed rollouts already dropped in {target}")
        return
    for name, anchor in (("helper", HELPER_ANCHOR), ("rollout", ANCHOR)):
        if source.count(anchor) != 1:
            raise SystemExit(
                f"Could not find a unique {name} anchor in {target}; "
                "Platoon changed upstream and this patch needs review."
            )
    if "\nimport os\n" not in source:
        raise SystemExit(f"{target} does not import os; this patch needs review.")
    source = source.replace(HELPER_ANCHOR, HELPER, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    target.write_text(source)
    print(f"dropped infra-killed rollouts from the group in {target}")


if __name__ == "__main__":
    main()
