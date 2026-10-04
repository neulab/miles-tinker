"""Promise store: the ledger behind submit-then-poll.

The HTTP side reads, the execution side writes; this is the only interface
between the two worlds. Promises live in memory: a gateway restart answers
410, which the SDK treats as a retryable "promise expired" for sampling and
as fatal for training (the stream dedup makes a resubmit safe either way).
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from miles.tinker.core.types import OwnershipError

PENDING = "pending"
DONE = "done"
FAILED = "failed"

logger = logging.getLogger(__name__)

_FINISHED_TTL_S = 3600.0  # idle retention after completion / the command window drains
# A delivered sampling result only serves a retry of a lost response, so it leaves much sooner:
# at hundreds of samples per second an hour of results is GBs of heap, and every full
# GC pass and expiry sweep walks all of it (1 s and 0.2 s per million results).
_DELIVERED_TTL_S = 60.0
_SWEEP_INTERVAL_S = 60.0  # full expiry pass; the polled promise itself is always checked


@dataclass
class ResultWindow:
    """Shared by one model's commands until its queue drains; never reopened."""

    closed_at: float | None = None

    def close(self) -> None:
        if self.closed_at is None:
            self.closed_at = time.monotonic()


@dataclass
class Promise:
    request_id: str
    model_id: str
    tenant: str
    session_id: str = ""
    sampling: bool = False
    window: ResultWindow | None = None
    state: str = PENDING
    result: dict | None = None
    error: str | None = None
    error_category: str | None = None
    finished_at: float | None = None
    delivered_at: float | None = None  # latest result response, not an acknowledgement
    created_at: float = field(default_factory=time.monotonic)
    settled: asyncio.Event = field(default_factory=asyncio.Event)  # long-poll wake-up


class PromiseStore:
    def __init__(self) -> None:
        self._promises: dict[str, Promise] = {}
        self._last_sweep = time.monotonic()

    def create(
        self,
        model_id: str,
        tenant: str,
        *,
        session_id: str = "",
        sampling: bool = False,
        window: ResultWindow | None = None,
    ) -> Promise:
        promise = Promise(
            request_id=f"req-{uuid.uuid4().hex}",
            model_id=model_id,
            tenant=tenant,
            session_id=session_id,
            sampling=sampling,
            window=window,
        )
        self._promises[promise.request_id] = promise
        return promise

    def resolve(self, request_id: str, result: dict) -> None:
        promise = self._promises.get(request_id)
        if promise is None or promise.state != PENDING:
            return  # released work can finish after its failure result has expired
        promise.state = DONE
        promise.result = result
        promise.finished_at = time.monotonic()
        promise.settled.set()

    def fail(self, request_id: str, error: str, category: str) -> None:
        promise = self._promises.get(request_id)
        if promise is None or promise.state != PENDING:
            return
        promise.state = FAILED
        promise.error = error
        promise.error_category = category
        promise.finished_at = time.monotonic()
        promise.settled.set()

    def delivered(self, promise: Promise) -> None:
        """Refresh the bounded retry window; an HTTP response is not a client ACK."""
        promise.delivered_at = time.monotonic()

    def get(self, request_id: str, tenant: str) -> Promise | None:
        """None means unknown/expired: the HTTP layer answers 410."""
        self.sweep()
        promise = self._promises.get(request_id)
        if promise is None:
            return None
        if _expired(promise, time.monotonic()):
            self._expire(request_id)
            return None
        if promise.tenant != tenant:
            raise OwnershipError(f"request {request_id} does not belong to this tenant")
        return promise

    def sweep(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep < _SWEEP_INTERVAL_S:
            return  # every poll would otherwise walk every promise of the last hour
        self._last_sweep = now
        for rid in [rid for rid, p in self._promises.items() if _expired(p, now)]:
            self._expire(rid)

    def _expire(self, request_id: str) -> None:
        promise = self._promises.pop(request_id)
        # Training cannot safely repeat a lost gradient result. Keep enough diagnostics
        # to distinguish its retention expiry from a session lease or server restart.
        if not promise.sampling:
            logger.info(
                "promise expired: request=%s model=%s state=%s finished_at=%s delivered_at=%s window_closed_at=%s",
                request_id,
                promise.model_id,
                promise.state,
                promise.finished_at,
                promise.delivered_at,
                promise.window.closed_at if promise.window else None,
            )


def _expired(promise: Promise, now: float) -> bool:
    if promise.finished_at is None:
        return False
    if promise.window is not None and promise.window.closed_at is None:
        return False
    retained_at = max(promise.finished_at, promise.window.closed_at if promise.window else promise.finished_at)
    if promise.delivered_at is not None:
        retained_at = max(retained_at, promise.delivered_at)
        if promise.sampling:
            return now - promise.delivered_at > _DELIVERED_TTL_S
    return now - retained_at > _FINISHED_TTL_S
