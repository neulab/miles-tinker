"""In-memory future results. Request deduplication lasts only for the current process."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from miles.tinker.core.types import OwnershipError

PENDING = "pending"
DONE = "done"
FAILED = "failed"

_FINISHED_TTL_S = 3600.0  # an undelivered result waits this long for its poller
# A delivered result only serves a retry of a lost response, so it leaves much sooner:
# at hundreds of samples per second an hour of results is GBs of heap, and every full
# GC pass and expiry sweep walks all of it.
_DELIVERED_TTL_S = 60.0
_SWEEP_INTERVAL_S = 60.0  # full expiry pass; the polled future itself is always checked


@dataclass
class RequestFuture:
    request_id: str
    model_id: str
    tenant: str
    state: str = PENDING
    result: dict | None = None
    error: str | None = None
    error_category: str | None = None
    finished_at: float | None = None
    delivered_at: float | None = None  # first time the settled result left through the HTTP layer
    # long-poll wakeup: set when the future settles
    settled: asyncio.Event = field(default_factory=asyncio.Event)


class RequestFutureStore:
    def __init__(self) -> None:
        self._futures: dict[str, RequestFuture] = {}
        self._last_sweep = time.monotonic()

    def create(self, model_id: str, tenant: str) -> RequestFuture:
        future = RequestFuture(request_id=f"req-{uuid.uuid4().hex}", model_id=model_id, tenant=tenant)
        self._futures[future.request_id] = future
        return future

    def resolve(self, request_id: str, result: dict) -> None:
        future = self._futures[request_id]
        if future.state != PENDING:  # e.g. already failed by lease expiry
            return
        future.state = DONE
        future.result = result
        future.finished_at = time.monotonic()
        future.settled.set()

    def fail(self, request_id: str, error: str, category: str) -> None:
        future = self._futures[request_id]
        if future.state != PENDING:
            return
        future.state = FAILED
        future.error = error
        future.error_category = category
        future.finished_at = time.monotonic()
        future.settled.set()

    def delivered(self, future: RequestFuture) -> None:
        """The settled result reached its poller; the future now expires on the short TTL."""
        if future.state != PENDING and future.delivered_at is None:
            future.delivered_at = time.monotonic()

    def get(self, request_id: str, tenant: str) -> RequestFuture | None:
        """Return None for unknown or expired futures; enforce tenant ownership otherwise."""
        now = time.monotonic()
        self._sweep(now)
        future = self._futures.get(request_id)
        if future is None:
            return None
        if _expired(future, now):
            del self._futures[request_id]
            return None
        if future.tenant != tenant:
            raise OwnershipError(f"request {request_id} does not belong to this tenant")
        return future

    def request_id_for_retry(self, request_id: str, model_id: str, tenant: str) -> str:
        if self.get(request_id, tenant) is not None:
            return request_id
        replacement = self.create(model_id, tenant)
        self.fail(replacement.request_id, "result expired after retention", "user")
        return replacement.request_id

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < _SWEEP_INTERVAL_S:
            return  # every poll would otherwise walk every future of the last hour
        self._last_sweep = now
        for request_id in [request_id for request_id, future in self._futures.items() if _expired(future, now)]:
            del self._futures[request_id]


def _expired(future: RequestFuture, now: float) -> bool:
    if future.delivered_at is not None and now - future.delivered_at > _DELIVERED_TTL_S:
        return True
    return future.finished_at is not None and now - future.finished_at > _FINISHED_TTL_S
