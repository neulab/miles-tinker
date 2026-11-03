"""The promise store is the submit-then-poll ledger: tenant-scoped reads,
410 semantics for the expired."""

from types import SimpleNamespace

import pytest

from miles.tinker.core import promise as promise_module
from miles.tinker.core.promise import DONE, FAILED, PENDING, PromiseStore, ResultWindow
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
    delivered = store.create("model", "tenant", sampling=True)
    undelivered = store.create("model", "tenant", sampling=True)
    for promise in (delivered, undelivered):
        store.resolve(promise.request_id, {"kind": "sample"})
    store.delivered(delivered)
    assert store.get(delivered.request_id, "tenant") is not None, "a retry of a lost response still finds it"

    monkeypatch.setattr(promise_module, "_DELIVERED_TTL_S", -1.0)
    assert store.get(delivered.request_id, "tenant") is None
    assert store.get(undelivered.request_id, "tenant") is not None, "an unpolled result keeps the long TTL"


def test_unknown_request_returns_none():
    assert PromiseStore().get("req-nope", "tenant") is None


def test_training_results_survive_a_long_window_then_expire_without_a_poller(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(promise_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    store = PromiseStore()
    window = ResultWindow()
    delivered = store.create("model", "tenant", window=window)
    failed = store.create("model", "tenant", window=window)
    store.resolve(delivered.request_id, {"kind": "forward_backward"})
    store.fail(failed.request_id, "backend failed", "server")
    store.delivered(delivered)

    now[0] += 7200  # later chunks still running; old delivered and undelivered TTLs both elapsed
    assert store.get(delivered.request_id, "tenant") is delivered
    assert store.get(failed.request_id, "tenant") is failed
    window.close()
    now[0] += 120
    store.delivered(delivered)  # a lost-response retry extends only this result's idle grace
    now[0] += 3500
    assert store.get(failed.request_id, "tenant") is None
    assert store.get(delivered.request_id, "tenant") is delivered
    now[0] += 101
    store.sweep()  # no incoming poll is required to reclaim an abandoned result
    assert not store._promises


def test_control_result_retry_is_not_limited_to_sixty_seconds(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(promise_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    store = PromiseStore()
    promise = store.create("model", "tenant")
    store.resolve(promise.request_id, {"kind": "optim_step"})
    store.delivered(promise)
    now[0] += 120
    assert store.get(promise.request_id, "tenant") is promise


def test_late_backend_completion_cannot_revive_a_released_promise(monkeypatch):
    store = PromiseStore()
    promise = store.create("model", "tenant")
    store.fail(promise.request_id, "model unloaded", "user")
    store.resolve(promise.request_id, {"kind": "forward_backward"})
    assert promise.state == FAILED and promise.error == "model unloaded"
    monkeypatch.setattr(promise_module, "_FINISHED_TTL_S", -1.0)
    assert store.get(promise.request_id, "tenant") is None
    store.resolve(promise.request_id, {"kind": "forward_backward"})
    store.fail(promise.request_id, "late backend error", "server")
    assert store.get(promise.request_id, "tenant") is None
