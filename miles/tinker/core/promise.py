"""Promise store: the ledger behind submit-then-poll.

The HTTP side reads, the execution side writes; this is the only interface
between the two worlds. Promises live in memory: a gateway restart answers
410, which the SDK treats as a retryable "promise expired" for sampling and
as fatal for training (the stream dedup makes a resubmit safe either way).
"""

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
# GC pass and expiry sweep walks all of it (1 s and 0.2 s per million results).
_DELIVERED_TTL_S = 300.0
_SWEEP_INTERVAL_S = 60.0  # full expiry pass; the polled promise itself is always checked


@dataclass
class Promise:
    request_id: str
    model_id: str
    tenant: str
    state: str = PENDING
    result: dict | None = None
    error: str | None = None
    error_category: str | None = None
    finished_at: float | None = None
    delivered_at: float | None = None  # first time the result left through the HTTP layer
    created_at: float = field(default_factory=time.monotonic)
    settled: asyncio.Event = field(default_factory=asyncio.Event)  # long-poll wake-up


class PromiseStore:
    def __init__(self) -> None:
        self._promises: dict[str, Promise] = {}
        self._last_sweep = time.monotonic()

    def create(self, model_id: str, tenant: str) -> Promise:
        promise = Promise(request_id=f"req-{uuid.uuid4().hex}", model_id=model_id, tenant=tenant)
        self._promises[promise.request_id] = promise
        return promise

    def resolve(self, request_id: str, result: dict) -> None:
        promise = self._promises[request_id]
        promise.state = DONE
        promise.result = result
        promise.finished_at = time.monotonic()
        promise.settled.set()

    def fail(self, request_id: str, error: str, category: str) -> None:
        promise = self._promises[request_id]
        promise.state = FAILED
        promise.error = error
        promise.error_category = category
        promise.finished_at = time.monotonic()
        promise.settled.set()

    def delivered(self, promise: Promise) -> None:
        """The settled result reached its poller; the promise now expires on the short TTL."""
        if promise.delivered_at is None:
            promise.delivered_at = time.monotonic()

    def get(self, request_id: str, tenant: str) -> Promise | None:
        """None means unknown/expired: the HTTP layer answers 410."""
        self._sweep()
        promise = self._promises.get(request_id)
        if promise is None:
            return None
        if _expired(promise, time.monotonic()):
            del self._promises[request_id]
            return None
        if promise.tenant != tenant:
            raise OwnershipError(f"request {request_id} does not belong to this tenant")
        return promise

    def _sweep(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep < _SWEEP_INTERVAL_S:
            return  # every poll would otherwise walk every promise of the last hour
        self._last_sweep = now
        for rid in [rid for rid, p in self._promises.items() if _expired(p, now)]:
            del self._promises[rid]


def _expired(promise: Promise, now: float) -> bool:
    if promise.delivered_at is not None and now - promise.delivered_at > _DELIVERED_TTL_S:
        return True
    return promise.finished_at is not None and now - promise.finished_at > _FINISHED_TTL_S
