#!/usr/bin/env python3
"""Reproduce the act() hang without a GPU, a sandbox, or the OpenHands SDK.

``OpenHandsAgent.act`` polls ``get_actions_for_last_obs`` every 0.2s and returns
only when it yields something. When it yields an empty list *permanently*, the
episode loop's ``asyncio.wait_for(..., step_timeout)`` truncates the rollout --
in job 10617579 that was 36.6% of all rollouts.

The hang is pure list logic over the conversation's event window, so this script
imports the real function out of the Platoon checkout and feeds it constructed
events. The only things stubbed are the SDK classes the function calls
``isinstance`` against, so the decision logic under test is unmodified.

    python act_stall.py [--checkout PATH]

Exit status is 1 when a scenario hangs, so it works as a regression test.
"""

import argparse
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from enum import Enum


def install_sdk_stub() -> None:
    """Minimal stand-ins for the SDK symbols openhands_utils imports."""

    class ConversationExecutionStatus(Enum):
        IDLE = "idle"
        RUNNING = "running"
        FINISHED = "finished"
        STUCK = "stuck"
        ERROR = "error"

    class Event:
        def __init__(self, id):
            self.id = id

        def __repr__(self):
            return f"{type(self).__name__}({self.id})"

    class ActionEvent(Event):
        def __init__(self, id, llm_response_id="call-0"):
            super().__init__(id)
            self.llm_response_id = llm_response_id

    class ObservationEvent(Event):
        def __init__(self, id, action_id):
            super().__init__(id)
            self.action_id = action_id

    class AgentErrorEvent(Event):
        def __init__(self, id, action_id=None):
            super().__init__(id)
            if action_id is not None:
                self.action_id = action_id

    class ConversationStateUpdateEvent(Event):
        pass

    class MessageEvent(Event):
        def __init__(self, id, source="agent"):
            super().__init__(id)
            self.source = source

    class SystemPromptEvent(Event):
        pass

    class Condensation(Event):
        pass

    for name, mod in [
        ("openhands", types.ModuleType("openhands")),
        ("openhands.sdk", types.ModuleType("openhands.sdk")),
        ("openhands.sdk.conversation", types.ModuleType("openhands.sdk.conversation")),
        ("openhands.sdk.conversation.state", types.ModuleType("openhands.sdk.conversation.state")),
        ("openhands.sdk.event", types.ModuleType("openhands.sdk.event")),
        ("platoon", types.ModuleType("platoon")),
        ("platoon.openhands", types.ModuleType("platoon.openhands")),
        ("platoon.openhands.types", types.ModuleType("platoon.openhands.types")),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules["openhands.sdk.conversation.state"].ConversationExecutionStatus = ConversationExecutionStatus
    ev = sys.modules["openhands.sdk.event"]
    for cls in (Event, ActionEvent, ObservationEvent, AgentErrorEvent,
                ConversationStateUpdateEvent, MessageEvent, SystemPromptEvent, Condensation):
        setattr(ev, cls.__name__, cls)
    ev.EventID = str
    sys.modules["platoon.openhands.types"].OpenHandsObservation = object

    globals().update({c.__name__: c for c in (
        ConversationExecutionStatus, Event, ActionEvent, ObservationEvent,
        AgentErrorEvent, ConversationStateUpdateEvent, MessageEvent,
        SystemPromptEvent, Condensation)})


@dataclass
class FakeConversationState:
    events: list = field(default_factory=list)
    execution_status: object = None


@dataclass
class FakeObservation:
    conversation_state: FakeConversationState
    last_step_observation_id: str | None = None
    last_step_action_id: str | None = None


def load_real_utils(checkout):
    path = f"{checkout}/platoon/utils/openhands_utils.py"
    spec = importlib.util.spec_from_file_location("_real_openhands_utils", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def turn(n, trailing_state_update=True):
    """One complete agent turn: action, its observation, and SDK bookkeeping."""
    out = [ActionEvent(f"A{n}", llm_response_id=f"call-{n}"), ObservationEvent(f"O{n}", action_id=f"A{n}")]
    if trailing_state_update:
        out.append(ConversationStateUpdateEvent(f"S{n}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", default=os.path.expandvars("/data/user_data/$USER/miles-babel/platoon"))
    args = ap.parse_args()

    install_sdk_stub()
    utils = load_real_utils(args.checkout)
    get_actions = utils.get_actions_for_last_obs
    IDLE = ConversationExecutionStatus.IDLE

    print(f"driving the real get_actions_for_last_obs from {args.checkout}\n")

    scenarios = []

    # 1. healthy: env.step consumed O0, the agent has produced turn 1.
    ev = turn(0) + turn(1)
    scenarios.append(("healthy: one complete turn pending", ev, "S0"))

    # 2. the live bug. The condenser emits a Condensation event into the stream.
    #    When it lands immediately after the observation env.step consumed, the
    #    backward walk reaches it LAST, and because it is neither an action nor a
    #    ConversationStateUpdateEvent it falls into the `not is_action` branch and
    #    clears every candidate. state_updates_are_transparent.py deliberately
    #    exempted Condensation -- that exemption is this hang.
    ev = turn(0) + [Condensation("C0")] + turn(1)
    scenarios.append(("STALL: Condensation at the oldest edge", ev, "S0"))

    # 3. the window only grows and its oldest event never changes, so act()
    #    never recovers -- it polls until step_timeout truncates the episode.
    ev = turn(0) + [Condensation("C0")]
    for i in range(1, 56):
        ev += turn(i)
    scenarios.append(("STALL persists after 55 more turns", ev, "S0"))

    # 4. the pre-patch failure, fixed by state_updates_are_transparent.py.
    ev = turn(0) + [ConversationStateUpdateEvent("S0b")] + turn(1)
    scenarios.append(("fixed: trailing ConversationStateUpdateEvent", ev, "S0"))

    width = max(len(s[0]) for s in scenarios)
    failed = 0
    for label, events, stop in scenarios:
        obs = FakeObservation(FakeConversationState(events, IDLE), last_step_observation_id=stop)
        got = get_actions(obs, require_same_llm_call_id=True)
        ids = [e.id for e in got]
        ok = bool(got)
        if not ok:
            failed += 1
        import collections
        cut = max(i for i, e in enumerate(events) if e.id == stop)
        hist = collections.Counter(type(e).__name__ for e in events[cut + 1:])
        kinds = ",".join(f"{k}={v}" for k, v in hist.most_common())
        print(f"  {label:<{width}}  window={len(events) - cut - 1:<4} -> {ids if ids else '[]  act() HANGS'}")
        print(f"  {'':<{width}}  {kinds}")

    print()
    if failed:
        print(f"{failed} scenario(s) return [] forever; act() polls until step_timeout truncates the episode.")
    else:
        print("no scenario hangs.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
