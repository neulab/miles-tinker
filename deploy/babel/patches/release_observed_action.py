"""Release an observation once it has arrived, not once the agent acts again.

``OpenHandsEnv.step`` records the action it just took and then polls until
``get_obs_for_last_action`` hands back that action's observation::

    self._state.last_step_action_id = action.action_events[-1].id
    obs_events = get_obs_for_last_action(self._state)
    while not obs_events and not is_finished(self._state):
        await asyncio.sleep(0.2)

The extractor walks the conversation backwards from the newest event, stopping
at ``last_step_action_id``, and collects the non-action events it finds. It then
discards all of them unless it saw an action *after* the one being waited on::

    if not at_least_one_future_action_seen and not is_finished(...):
        new_obs.clear()

That is a completeness check using the wrong evidence. Nothing else can
complete an action's observations, so "the agent has acted again" stands in for
"this observation set is final" -- but it can only become true once the model
produces its next action. Run against the real event shapes, the current code
returns ``[]`` for ``[ObservationEvent(action_id=A_k), ConversationStateUpdateEvent]``
and only releases them once an ``ActionEvent`` follows.

So ``env.step`` does not wait for the tool; it waits for the *model's next
response*. Measured on job 10615209: env.step runs 13.5s median and 48.1s p90
while ``act`` runs 0.1s median, i.e. the waiting has migrated to the wrong half
of the loop. When a response lands in the tail past ``step_timeout`` the step
dies -- 63.9% of rollouts in one batch, concentrated in slow tasks (a Rust
cargo task lost all 8 of its group, quick tasks lost 1 of 8). Tuning
``step_timeout`` cannot help, because the quantity it bounds is model latency:
120s truncated 29% of rollouts and 300s truncated 32%.

The evidence the guard actually wants is the observation itself. Across 8,760
actions in this workload every one produced exactly one observation, and it
carries ``action_id`` referencing its action, so an observation naming
``last_step_action_id`` is the complete result rather than the first of several.
Releasing on that keeps the guard's intent and drops its unmeetable precondition.

Usage: python release_observed_action.py <platoon-checkout>
"""

import sys
from pathlib import Path

# get_actions_for_last_obs's docstring mentions observations too, so the marker
# has to be the inserted code rather than any prose.
MARKER = "observed_the_awaited_action"
TARGET = "platoon/utils/openhands_utils.py"

ANCHOR = """    # If not at least one future action seen and if this obs is not the final one, empty the list.
    last_event_seen = new_obs[-1].id if new_obs else None
    if not at_least_one_future_action_seen and not is_finished(observation, last_event_seen=last_event_seen):
        new_obs.clear()
"""

REPLACEMENT = """    # If not at least one future action seen and if this obs is not the final one, empty the list.
    last_event_seen = new_obs[-1].id if new_obs else None
    # ... unless the observation being waited on has already arrived. A future
    # action was only ever a proxy for "this observation set is complete", and it
    # cannot become true until the model responds again, which turns env.step
    # into a wait on model latency rather than on the tool.
    observed_the_awaited_action = any(
        getattr(event, "action_id", None) == observation.last_step_action_id for event in new_obs
    )
    if (
        not at_least_one_future_action_seen
        and not observed_the_awaited_action
        and not is_finished(observation, last_event_seen=last_event_seen)
    ):
        new_obs.clear()
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No openhands_utils at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"observation release already current in {target}")
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"Could not find get_obs_for_last_action's completeness guard in {target}; "
            "Platoon changed upstream and this patch needs review."
        )
    target.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"released observations on arrival in {target}")


if __name__ == "__main__":
    main()
