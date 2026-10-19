"""Stop Platoon's episode loop from hanging on a finished openhands conversation.

``is_finished`` requires two things: that openhands reports a terminal status,
*and* that the episode loop has "caught up" -- that the conversation's last
event is one the loop has already consumed::

    last_event_id = conversation_state.events[-1].id
    platoon_episode_caught_up = last_event_id in (
        observation.last_step_action_id,
        observation.last_step_observation_id,
        last_event_seen,
    )
    return oh_conversation_finished and platoon_episode_caught_up

The second clause is unsatisfiable in practice. A captured event stream from a
finished conversation ends::

    ... ActionEvent, ObservationEvent,
        ConversationStateUpdateEvent, ConversationStateUpdateEvent
    execution_status = FINISHED

Those trailing ConversationStateUpdateEvents are status and token-usage
bookkeeping that the loop never consumes, and even ignoring them the loop has
not consumed the final ObservationEvent either -- it stops extracting once the
agent stops acting. So ``is_finished`` stays False, ``OpenHandsAgent.act``
polls until its step timeout, and every rollout burns ~900s after the agent
has already stopped: about 95% of a training step.

A terminal status means openhands has nothing further to emit, so there is
nothing left to catch up with; the guard protects against an ordering that
cannot occur. Dropping it also improves the recorded trajectory, because
``get_actions_for_last_obs`` keeps an action that has no following observation
when ``is_finished`` is true -- which is exactly the agent's final action.

Usage: python openhands_finish_detection.py <platoon-checkout>
"""

import re
import sys
from pathlib import Path

REPLACEMENT = '''    # A terminal status means openhands has nothing further to emit, so there is
    # nothing for the episode loop to catch up with. Requiring catch-up here
    # deadlocked every rollout: the loop never consumes the conversation's final
    # observation or the ConversationStateUpdateEvents that follow it, so the
    # condition could not become true and act() polled until its step timeout.
    return oh_conversation_finished
'''

# Everything from the last_event_id assignment through the return, in either the
# pristine form or the earlier partial fix.
REGION = re.compile(
    r"(?:    # ConversationStateUpdateEvent carries.*?\n(?:    #.*\n)*)?"
    r"(?:    _events = conversation_state\.events\n.*?)?"
    r"    last_event_id = .*?\n"
    r"    platoon_episode_caught_up = .*?\n(?:.*?\n)*?"
    r"    return oh_conversation_finished and platoon_episode_caught_up\n",
    re.S,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / "platoon/utils/openhands_utils.py"
    if not target.is_file():
        raise SystemExit(f"No openhands_utils at {target}")
    source = target.read_text()
    if "nothing for the episode loop to catch up with" in source:
        print(f"finish detection already patched in {target}")
        return
    patched, count = REGION.subn(REPLACEMENT, source, count=1)
    if count != 1:
        raise SystemExit(
            f"Could not find the is_finished catch-up block in {target}; "
            "Platoon changed upstream and this patch needs review."
        )
    target.write_text(patched)
    print(f"patched finish detection in {target}")


if __name__ == "__main__":
    main()
