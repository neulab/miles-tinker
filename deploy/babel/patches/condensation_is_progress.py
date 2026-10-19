"""Stop the episode loop mistaking history condensation for a hung agent.

DISABLED -- kept for the analysis below, but removed from the tmax-15k-openhands
recipe because measurement contradicted it. Across run 260928-232958-550, none
of the 26 rollouts that ran to the rollout timeout carried a single client-side
``Condensation`` event (0/26), and only 1 of 52 trajectories carried one at all,
although 30 of 64 agent-server logs show the condenser running. So the event
this patch watches for almost never reaches the client, and never in the
rollouts it was meant to rescue. What it did do was convert "stop at the 300s
step timeout" into "grind to the 700s rollout timeout": step_timeout fell 42 ->
2 while cancelled rose 3 -> 26, the cost of a stalled rollout went 506s -> 1026s
and the batch grew 867 -> 1411 steps, which is what pushed forward_backward past
its hardcoded 900s limit and killed jobs 10595807 and 10597116 at step 1.
Re-enable only with a measurement showing the Condensation event actually
reaching the client in the rollouts that stall.

``OpenHandsAgent.act`` waits for either a new action or ``is_finished``. When
the conversation outgrows its token budget, ``Agent.step`` hands the condenser
the history, emits a ``Condensation`` event and returns *without an action*:

    if isinstance(_messages_or_condensation, Condensation):
        on_event(_messages_or_condensation)
        return

The condenser's own summarising call is large-context and slow, and the retry
step that follows is another full call. Neither produces an action, so ``act``
polls at 0.2s intervals until ``run_episode``'s ``step_timeout`` fires and the
episode is truncated mid-work.

This was 1% of rollouts before prior-turn reasoning was retained in history and
28% after -- history now grows far faster -- and it took ``step_timeout`` from
1.5% of trajectories to 50.7%. The agent was working the whole time; a traced
rollout answered 321 polls during its 326-second "silence" and logged the
condenser at the end of it.

Returning an empty action on a condensation lets the loop tick: the step clock
restarts, and the step carries no ``completion_id`` so the datum builder skips
it. ``last_step_action_id`` is only advanced for a non-empty action, so the
following ``env.step`` still finds the condensation event as its observation and
does not block either.

Usage: python condensation_is_progress.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_condensation_count"
TARGET = "plugins/openhands/platoon/openhands/agent.py"

ANCHOR = """    async def act(self, obs: OpenHandsObservation) -> OpenHandsAction:
        step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
        while not step_actions and not is_finished(obs):
            await asyncio.sleep(0.2)
            step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
"""

REPLACEMENT = '''    @staticmethod
    def _condensation_count(obs: OpenHandsObservation) -> int:
        """How many condensations the conversation has emitted so far."""
        try:
            return sum(
                1
                for event in obs.conversation_state.events
                if type(event).__name__ == "Condensation"
            )
        except Exception:  # a lost conversation must not wedge the loop
            return 0

    async def act(self, obs: OpenHandsObservation) -> OpenHandsAction:
        step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
        condensations = self._condensation_count(obs)
        while not step_actions and not is_finished(obs):
            await asyncio.sleep(0.2)
            step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
            if not step_actions and self._condensation_count(obs) > condensations:
                # History is being rewritten, which produces no action. Hand the
                # loop an empty step so its timeout restarts instead of expiring
                # on an agent that is still working.
                break
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    agent = Path(sys.argv[1]) / TARGET
    if not agent.is_file():
        raise SystemExit(f"No openhands agent adapter at {agent}")
    source = agent.read_text()
    if MARKER in source:
        print(f"condensation handling already current in {agent}")
        return
    if ANCHOR not in source:
        raise SystemExit(
            f"Could not find OpenHandsAgent.act in {agent}; "
            "Platoon changed upstream and this patch needs review."
        )
    agent.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"treated condensation as progress in {agent}")


if __name__ == "__main__":
    main()
