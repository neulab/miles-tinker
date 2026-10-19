"""Stop status bookkeeping from emptying the episode loop's action list.

``get_actions_for_last_obs`` walks the conversation backwards from the newest
event, breaking at ``last_step_observation_id`` -- the observation the loop last
consumed -- and clears its candidate list on every event that is not an action::

    for event in reversed(events):
        if event.id == observation.last_step_observation_id:
            break
        if not is_action(event):
            new_actions.clear()
            at_least_one_future_obs_seen = True

Because the walk is backwards, the *last* event it touches is the oldest one in
the window: whatever the agent-server emitted immediately after the consumed
observation. The SDK trails each observation with a
``ConversationStateUpdateEvent`` -- status and token-usage bookkeeping, the same
events that made ``is_finished`` unsatisfiable before
``openhands_finish_detection`` dropped its catch-up clause. When one of those
lands in that position the scan's final act is ``clear()``, and the function
returns an empty list.

It then returns empty on every later poll too, because the window only grows and
its oldest event never changes. ``OpenHandsAgent.act`` polls at 0.2s until
``run_episode``'s ``step_timeout`` truncates the episode, while the agent keeps
working: the act() stall probe measured 40-50 complete, fully observed steps
sitting unconsumed, with ``ActionEvent`` and ``ObservationEvent`` counts equal
and one ``ConversationStateUpdateEvent`` per pair. Reproducing the function
against that exact composition returns ``[]`` when the post-stop event is a
state update and the action when it is not -- independent of backlog size, so
the type of that single event decides the whole episode. That is the 43%
clean / 57% timed-out split, and about 70% of rollout wall clock.

Skipping those events restores the intent: they are neither an action nor an
observation of one, so they should not clear the candidates nor satisfy
``at_least_one_future_obs_seen``. Condensation is deliberately left alone -- it
rewrites history rather than reporting status, so an episode loop that has not
seen it really is out of date.

Usage: python state_updates_are_transparent.py <platoon-checkout>
"""

import sys
from pathlib import Path

# openhands_finish_detection.py inserts a comment mentioning
# ConversationStateUpdateEvents, so the marker has to be the code itself.
MARKER = "isinstance(event, ConversationStateUpdateEvent)"
TARGET = "platoon/utils/openhands_utils.py"

IMPORT_ANCHOR = (
    "from openhands.sdk.event import ActionEvent, AgentErrorEvent, Event, EventID, MessageEvent\n"
)
IMPORT = (
    "from openhands.sdk.event import (\n"
    "    ActionEvent,\n"
    "    AgentErrorEvent,\n"
    "    ConversationStateUpdateEvent,\n"
    "    Event,\n"
    "    EventID,\n"
    "    MessageEvent,\n"
    ")\n"
)

ANCHOR = """    for event in reversed(events):
        if event.id == observation.last_step_observation_id:
            break
        if not is_action(event):
"""

REPLACEMENT = """    for event in reversed(events):
        if event.id == observation.last_step_observation_id:
            break
        # Status and token-usage bookkeeping, not an observation of anything.
        # The walk is backwards, so the oldest event in the window is the last
        # one seen; a state update there would clear the candidates and empty
        # the return for the rest of the episode.
        if isinstance(event, ConversationStateUpdateEvent):
            continue
        if not is_action(event):
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No openhands_utils at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"state updates already transparent in {target}")
        return
    for name, anchor in (("import", IMPORT_ANCHOR), ("scan", ANCHOR)):
        if source.count(anchor) != 1:
            raise SystemExit(
                f"Could not find a unique {name} anchor in {target}; "
                "Platoon changed upstream and this patch needs review."
            )
    source = source.replace(IMPORT_ANCHOR, IMPORT, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    target.write_text(source)
    print(f"made ConversationStateUpdateEvent transparent in {target}")


if __name__ == "__main__":
    main()
