"""The promise store is the submit-then-poll ledger: tenant-scoped reads,
410 semantics for the expired."""

import pytest

from miles.tinker.core import promise as promise_module
from miles.tinker.core.promise import DONE, FAILED, PENDING, PromiseStore
from miles.tinker.core.types import OwnershipError


def test_resolve_and_fail_settle_the_state():
    store = PromiseStore()
    done = store.create("model", "tenant")
    failed = store.create("model", "tenant")
    assert done.state == PENDING

    store.resolve(done.request_id, {"kind": "optim_step"})
    store.fail(failed.request_id, "boom", "user")

    assert store.get(done.request_id, "tenant").state == DONE
    fetched = store.get(failed.request_id, "tenant")
    assert (fetched.state, fetched.error, fetched.error_category) == (FAILED, "boom", "user")


def test_cross_tenant_get_raises_ownership():
    store = PromiseStore()
    promise = store.create("model", "tenant-a")
    with pytest.raises(OwnershipError):
        store.get(promise.request_id, "tenant-b")


def test_an_expired_promise_returns_none(monkeypatch):
    store = PromiseStore()
    promise = store.create("model", "tenant")
    store.resolve(promise.request_id, {"kind": "optim_step"})

    monkeypatch.setattr(promise_module, "_FINISHED_TTL_S", -1.0)
    assert store.get(promise.request_id, "tenant") is None, "finished past TTL must read as unknown (410)"


def test_the_polled_promise_expires_inline_and_the_rest_on_the_sweep_interval(monkeypatch):
    store = PromiseStore()
    polled, other = store.create("model", "tenant"), store.create("model", "tenant")
    for promise in (polled, other):
        store.resolve(promise.request_id, {"kind": "optim_step"})

    monkeypatch.setattr(promise_module, "_FINISHED_TTL_S", -1.0)
    assert store.get(polled.request_id, "tenant") is None
    assert other.request_id in store._promises, "a poll must not walk every promise"
    store._last_sweep -= promise_module._SWEEP_INTERVAL_S
    assert store.get("req-nope", "tenant") is None
    assert other.request_id not in store._promises, "the periodic pass drops the rest"


def test_a_delivered_result_expires_on_the_short_ttl(monkeypatch):
    store = PromiseStore()
    delivered, undelivered = store.create("model", "tenant"), store.create("model", "tenant")
    for promise in (delivered, undelivered):
        store.resolve(promise.request_id, {"kind": "sample"})
    store.delivered(delivered)
    assert store.get(delivered.request_id, "tenant") is not None, "a retry of a lost response still finds it"

    monkeypatch.setattr(promise_module, "_DELIVERED_TTL_S", -1.0)
    assert store.get(delivered.request_id, "tenant") is None
    assert store.get(undelivered.request_id, "tenant") is not None, "an unpolled result keeps the long TTL"


def test_unknown_request_returns_none():
    assert PromiseStore().get("req-nope", "tenant") is None
