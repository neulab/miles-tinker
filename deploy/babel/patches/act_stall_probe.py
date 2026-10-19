"""Record what ``OpenHandsAgent.act`` is waiting on when a step is about to expire.

About 55-60% of rollouts stop producing actions partway through an episode and
are ended by ``run_episode``'s ``step_timeout`` rather than by finishing. The
stall is uncorrelated with the agent's last action, the episode's length and
concurrent sampler load; the in-sandbox server stays responsive throughout, and
its log cannot distinguish a working agent from an idle one because it records
only the client's 1 Hz status poll.

``act`` polls ``get_actions_for_last_obs`` until it returns something. That
function clears its candidate list when an action has no following observation::

    for action in new_actions:
        if isinstance(action, ActionEvent) and action.id not in seen_action_ids:
            new_actions.clear(); break
    if not at_least_one_future_obs_seen:
        new_actions.clear()

So an empty return has two very different meanings: the agent produced nothing
at all, or it produced an action whose observation has not arrived. Those point to
opposite causes -- a sampling request that never returned, versus a tool still
executing -- and the trajectory records neither, because a step is only written
once both halves exist.

This logs the distinction once per stalled step, at a soft deadline short of the
real one. It changes no behaviour: the loop still polls, and the step still
expires exactly when it would have. Deliberately not a retry -- the previous
"keep the loop moving" intervention (condensation_is_progress) doubled the cost
of a stalled rollout, from 506s to 1026s, and is reverted.

Usage: python act_stall_probe.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_probe_stall"
TARGET = "plugins/openhands/platoon/openhands/agent.py"

ANCHOR = """    async def act(self, obs: OpenHandsObservation) -> OpenHandsAction:
        step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
        while not step_actions and not is_finished(obs):
            await asyncio.sleep(0.2)
            step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
"""

REPLACEMENT = '''    @staticmethod
    def _probe_stall(obs: OpenHandsObservation, waited: float) -> None:
        """Say which half of the step is missing, once, when it is nearly over.

        Mirrors get_actions_for_last_obs: walk back only to
        ``last_step_observation_id`` and use the same ``is_action`` predicate and
        observed-id bookkeeping. Reports the window's event-type histogram
        because the counts alone are ambiguous -- ``is_action`` admits agent
        MessageEvents as well as ActionEvents, and only the histogram says which
        the window is actually full of.
        """
        try:
            events = list(obs.conversation_state.events)
            stop = obs.last_step_observation_id
        except Exception:  # a lost conversation must not wedge the loop
            return
        try:
            traj_id = current_trajectory.get().id
        except Exception:
            traj_id = "?"
        # If the stop id is absent the scan never breaks, walks the whole
        # history, and the final clear() at the oldest event returns empty for
        # good -- report it, because that alone would explain a permanent stall.
        stop_present = any(getattr(event, "id", None) == stop for event in events)
        candidates = []
        seen_action_ids = set()
        window = []
        for event in reversed(events):
            if event.id == stop:
                break
            window.append(event)
            if is_action(event):
                candidates.append(event)
            elif hasattr(event, "action_id"):
                seen_action_ids.add(event.action_id)
        unobserved = [
            event
            for event in candidates
            if type(event).__name__ == "ActionEvent" and event.id not in seen_action_ids
        ]
        histogram = Counter(type(event).__name__ for event in window)
        kinds = ",".join(f"{name}={count}" for name, count in histogram.most_common())
        if not candidates:
            why = "no action produced at all (sampling request outstanding)"
        elif unobserved:
            why = f"{len(unobserved)} action(s) awaiting an observation (tool still running)"
        elif len(candidates) == len(window):
            why = "window is all actions, nothing observed them"
        else:
            why = "actions present but discarded for another reason"
        logger.warning(
            "[stall] traj=%s after %.0fs: events=%d stop_present=%s window=%d "
            "candidates=%d unobserved=%d kinds=%s -- %s",
            traj_id,
            waited,
            len(events),
            "yes" if stop_present else "NO",
            len(window),
            len(candidates),
            len(unobserved),
            kinds,
            why,
        )

    async def act(self, obs: OpenHandsObservation) -> OpenHandsAction:
        deadline = time.monotonic() + STALL_PROBE_AFTER_S
        probed = False
        step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
        started = time.monotonic()
        while not step_actions and not is_finished(obs):
            await asyncio.sleep(0.2)
            step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
            if not step_actions and not probed and time.monotonic() >= deadline:
                probed = True
                self._probe_stall(obs, time.monotonic() - started)
'''

BASE_ANCHOR = "from platoon.envs.base import Task\n"
BASE_IMPORT = (
    "from platoon.envs.base import Task\n"
    "from platoon.episode.context import current_trajectory\n"
)
UTILS_ANCHOR = (
    "from platoon.utils.openhands_utils import get_actions_for_last_obs, is_finished\n"
)
UTILS_IMPORT = (
    "from platoon.utils.openhands_utils import get_actions_for_last_obs, is_action, is_finished\n"
)

IMPORTS_ANCHOR = """import asyncio
from copy import deepcopy
"""
IMPORTS = '''import asyncio
import logging
import os
import time
from collections import Counter
from copy import deepcopy
'''

CONST_ANCHOR = """class OpenHandsAgent:
"""
CONST = '''logger = logging.getLogger(__name__)

# Probe this far into a stalled step. Short of run_episode's step_timeout (300s
# in training) so the line is written before the step is cancelled.
STALL_PROBE_AFTER_S = float(os.getenv("TERMINAL_RL_STALL_PROBE_AFTER_S", "240"))


class OpenHandsAgent:
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No openhands agent adapter at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"stall probe already current in {target}")
        return
    for name, anchor in (
        ("act", ANCHOR),
        ("imports", IMPORTS_ANCHOR),
        ("class", CONST_ANCHOR),
        ("utils import", UTILS_ANCHOR),
        ("base import", BASE_ANCHOR),
    ):
        if source.count(anchor) != 1:
            raise SystemExit(
                f"Could not find a unique {name} anchor in {target}; "
                "Platoon changed upstream and this patch needs review."
            )
    source = source.replace(BASE_ANCHOR, BASE_IMPORT, 1)
    source = source.replace(UTILS_ANCHOR, UTILS_IMPORT, 1)
    source = source.replace(IMPORTS_ANCHOR, IMPORTS, 1)
    source = source.replace(CONST_ANCHOR, CONST, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    target.write_text(source)
    print(f"added the act() stall probe to {target}")


if __name__ == "__main__":
    main()
