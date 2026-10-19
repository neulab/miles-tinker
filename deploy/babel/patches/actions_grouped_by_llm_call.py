"""Stop the action scan depending on what event sits at the window's oldest edge.

``get_actions_for_last_obs`` walks the conversation backwards from the newest
event to ``last_step_observation_id`` and clears its candidate list on every
event that is not an action::

    for event in reversed(events):
        if event.id == observation.last_step_observation_id:
            break
        if isinstance(event, ConversationStateUpdateEvent):
            continue
        if not is_action(event):
            new_actions.clear()

Because the walk is backwards, the *last* event it touches is the oldest one in
the window, so a non-action there makes ``clear()`` the scan's final act and the
function returns empty. The window's newest end grows as the agent works, but
its oldest end only moves when ``env.step`` runs, and ``env.step`` only runs
after ``act`` returns. So the empty return is self-sustaining: ``act`` polls at
0.2s until ``run_episode``'s ``step_timeout`` truncates the episode, while the
sandbox agent keeps completing steps nobody collects.

Measured in job 10622896: 199 of 378 rollouts (52.6%) died this way. Taking the
timestamp of the last observation ``env.step`` consumed -- the moment the
boundary froze -- and asking whether the in-sandbox condenser fired in the 310s
after it separates the two populations cleanly:

    hung in act()   112/144 (77.8%) had a Condensation land after the boundary
    clean            2/98   ( 2.0%)

The two clean exceptions are the race resolving the other way, with ``env.step``
consuming the Condensation before the boundary froze. The remaining 22% of hangs
had no condenser firing and are a different event in the same position: a
``SystemPromptEvent`` and the user's task ``MessageEvent``, left in the window
when ``env.reset`` pins the boundary to the first event the conversation emits
before openhands has emitted them.

Exempting event types one at a time does not hold. ``ConversationStateUpdateEvent``
was exempted in September and ``Condensation`` took its place; exempting that
would leave the reset-race variant, and the SDK is free to emit new kinds there.
So this takes the fix the file's own TODO names: group the window's actions by
``llm_response_id`` and return the oldest group that is fully observed. The
decision then rests on which actions exist and whether they have been observed,
and no non-action can empty the result by position.

The guards are preserved exactly: an unobserved action in the chosen group still
returns empty, ``at_least_one_future_obs_seen`` still requires something other
than bookkeeping in the window, a terminal conversation and a preceding
``AgentErrorEvent`` still bypass both, and ``require_same_llm_call_id`` still
raises on a mixed group -- which grouping now makes unreachable rather than
merely unlikely.

Usage: python actions_grouped_by_llm_call.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_oldest_group"
TARGET = "platoon/utils/openhands_utils.py"

ANCHOR = '''    events = observation.conversation_state.events
    new_actions: list[Event] = list()
    new_actions_candidates: list[Event] = list()
    seen_action_ids: set[EventID] = set()
    at_least_one_future_obs_seen = False
    at_least_one_future_error_event_seen = False
    for event in reversed(events):
        if event.id == observation.last_step_observation_id:
            break
        # Status and token-usage bookkeeping, not an observation of anything.
        # The walk is backwards, so the oldest event in the window is the last
        # one seen; a state update there would clear the candidates and empty
        # the return for the rest of the episode.
        if isinstance(event, ConversationStateUpdateEvent):
            continue
        if not is_action(event):
            new_actions.clear()
            at_least_one_future_obs_seen = True
            if hasattr(event, "action_id"):
                seen_action_ids.add(event.action_id)
            if isinstance(event, AgentErrorEvent):
                at_least_one_future_error_event_seen = True
            continue
        else:
            new_actions.append(event)
            new_actions_candidates.append(event)

    last_event_seen = new_actions[-1].id if new_actions else None
    if not is_finished(observation, last_event_seen=last_event_seen) and not at_least_one_future_error_event_seen:
        for action in new_actions:
            if isinstance(action, ActionEvent) and action.id not in seen_action_ids:
                new_actions.clear()
                break

        if not at_least_one_future_obs_seen:
            new_actions.clear()

    if require_same_llm_call_id and new_actions:
        llm_call_id = new_actions[0].llm_response_id
        if any(action.llm_response_id != llm_call_id for action in new_actions):
            raise ValueError(
                "Detected at least two actions in a step with differing llm_response_id. "
                "This is unexpected and can lead to undefined behavior."
            )

    return list(reversed(new_actions))
'''

REPLACEMENT = '''    events = observation.conversation_state.events

    # Collect the window oldest-first instead of deciding during a backwards
    # walk. The old scan cleared its candidates on every non-action, so the
    # oldest event in the window decided the result and a Condensation or a
    # SystemPromptEvent there emptied it for the rest of the episode.
    window: list[Event] = list()
    for event in reversed(events):
        if event.id == observation.last_step_observation_id:
            break
        window.append(event)
    window.reverse()

    seen_action_ids: set[EventID] = {
        event.action_id for event in window if hasattr(event, "action_id")
    }
    at_least_one_future_error_event_seen = any(isinstance(e, AgentErrorEvent) for e in window)
    # Bookkeeping is neither an action nor an observation of one, so it cannot
    # stand in for "something responded to these actions".
    at_least_one_future_obs_seen = any(
        not is_action(e) and not isinstance(e, ConversationStateUpdateEvent) for e in window
    )

    # One step is one LLM call, so the oldest unconsumed call is the next step
    # to hand back. MessageEvents from the agent carry no llm_response_id; they
    # stand alone, which _group_key reflects by falling back to the event id.
    new_actions = _oldest_group(window)

    last_event_seen = new_actions[-1].id if new_actions else None
    if not is_finished(observation, last_event_seen=last_event_seen) and not at_least_one_future_error_event_seen:
        # Keep the leading actions that have been observed rather than dropping
        # the whole group. A model turn can emit several tool calls under one
        # llm_response_id (2.1% of steps in job 10622896), and waiting for all
        # of them would hold back actions the backwards walk handed over.
        observed: list[Event] = []
        for action in new_actions:
            if isinstance(action, ActionEvent) and action.id not in seen_action_ids:
                break
            observed.append(action)
        new_actions = observed

        if not at_least_one_future_obs_seen:
            new_actions = []

    if require_same_llm_call_id and new_actions:
        llm_call_id = new_actions[0].llm_response_id
        if any(action.llm_response_id != llm_call_id for action in new_actions):
            raise ValueError(
                "Detected at least two actions in a step with differing llm_response_id. "
                "This is unexpected and can lead to undefined behavior."
            )

    return new_actions
'''

HELPER_ANCHOR = "# TODO: Simplify by looking at changes in llm_response_id."

HELPER = '''def _group_key(event: Event) -> object:
    """Which LLM call produced this action.

    An agent MessageEvent has no llm_response_id, so it keys on its own id and
    forms a group of one rather than merging with whatever precedes it.
    """
    return getattr(event, "llm_response_id", None) or event.id


def _oldest_group(window: list[Event]) -> list[Event]:
    """The oldest run of actions in the window that share one LLM call.

    Returned oldest-first, matching what the episode loop consumed before. The
    window is in chronological order, so the first action's group is the one the
    loop has not taken yet; later groups are returned on later calls as env.step
    advances the boundary past each one.

    Observation is checked by the caller, not here, because the caller skips that
    check entirely for a finished conversation or one that already emitted an
    AgentErrorEvent -- and a group filtered here would be filtered even in those
    cases, which would hold back actions the previous scan returned.
    """
    actions = [event for event in window if is_action(event)]
    if not actions:
        return []
    key = _group_key(actions[0])
    group: list[Event] = []
    for action in actions:
        if _group_key(action) != key:
            break
        group.append(action)
    return group


# TODO: Simplify by looking at changes in llm_response_id.'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No openhands_utils at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"actions already grouped by llm call in {target}")
        return
    for name, anchor in (("helper", HELPER_ANCHOR), ("scan", ANCHOR)):
        if source.count(anchor) != 1:
            raise SystemExit(
                f"Could not find a unique {name} anchor in {target}; apply this "
                "after state_updates_are_transparent.py, or review it against upstream."
            )
    source = source.replace(HELPER_ANCHOR, HELPER, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    target.write_text(source)
    print(f"grouped actions by llm_response_id in {target}")


if __name__ == "__main__":
    main()
