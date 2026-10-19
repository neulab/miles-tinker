"""Per-rollout phase timing, to find where a rollout's wall clock actually goes.

Measurement in job 10590936 put the median rollout at 244s, of which only 33s
lay between its first and last observation -- the agent's visible work. The
other 169s was unattributed, and every component we could measure directly was
far too fast to explain it: the verifier's pytest ran in 0.03s, `apptainer
exec` in 0.07s, `instance stop` in 0.13s, and deleting the overlay in 0.026s.
The sampler GPUs were 22% utilised with `#queue-req: 0` throughout, so the time
was not spent queueing for inference either.

That leaves waiting somewhere in the episode loop, and the loop polls in three
places: openhands' RemoteConversation polls the in-sandbox server for status,
Platoon's agent.act polls for new events, and env.step polls again for the
observations belonging to the last action. This module times those calls so the
next run answers the question with a number instead of a hypothesis.

State is per rollout, held in ContextVars. Rollouts are asyncio tasks in one
process -- the workflow runs them through the in-process path, not the
subprocess pool -- so module-level dicts would pool all 64 concurrent rollouts
into one set of numbers. A task inherits a copy of the context at creation, so
a ``set`` here stays inside the rollout that made it.
"""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger("platoon.terminal_rl.phase_timing")

_totals: ContextVar[dict[str, float]] = ContextVar("phase_totals")
_counts: ContextVar[dict[str, int]] = ContextVar("phase_counts")


def reset() -> None:
    _totals.set({})
    _counts.set({})


def add(name: str, seconds: float) -> None:
    try:
        totals, counts = _totals.get(), _counts.get()
    except LookupError:  # a caller that never reset; keep diagnostics harmless
        return
    totals[name] = totals.get(name, 0.0) + seconds
    counts[name] = counts.get(name, 0) + 1


@contextmanager
def phase(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        add(name, time.perf_counter() - start)


def instrument(obj, method: str, name: str) -> None:
    """Wrap an async method so its total time and call count are recorded."""
    original = getattr(obj, method)

    async def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return await original(*args, **kwargs)
        finally:
            add(name, time.perf_counter() - start)

    setattr(obj, method, timed)


def report(task_id: str, total: float) -> None:
    """Emit one greppable line per rollout: PHASES <task> total=.. name=secs/calls.

    ``sandbox``, ``server_start``, ``episode`` and ``teardown`` partition the
    rollout and sum to ``accounted``; whatever ``total`` has beyond that is
    outside the instrumented path. ``act`` and ``step`` partition ``episode``,
    and ``score_within_step`` is a *subset* of ``step`` -- env.step calls
    evaluate() -- so adding it to the others double-counts.
    """
    try:
        totals, counts = _totals.get(), _counts.get()
    except LookupError:
        return
    parts = " ".join(
        f"{k}={totals[k]:.1f}/{counts[k]}" for k in sorted(totals, key=lambda k: -totals[k])
    )
    accounted = sum(totals.get(k, 0.0) for k in ("sandbox", "server_start", "episode", "teardown"))
    logger.info(
        "PHASES %s total=%.1f accounted=%.1f %s", task_id, total, accounted, parts
    )
