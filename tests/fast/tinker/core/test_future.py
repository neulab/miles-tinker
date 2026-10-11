"""RequestFuture reads enforce tenant ownership and expire finished results."""

import pytest

from miles.tinker.core import future as future_module
from miles.tinker.core.future import DONE, FAILED, PENDING, RequestFutureStore
from miles.tinker.core.types import OwnershipError


def test_resolve_and_fail_settle_the_state():
    store = RequestFutureStore()
    done = store.create("model", "tenant")
    failed = store.create("model", "tenant")
    assert done.state == PENDING

    store.resolve(done.request_id, {"op": "optim_step"})
    store.fail(failed.request_id, "boom", "user")

    assert store.get(done.request_id, "tenant").state == DONE
    fetched = store.get(failed.request_id, "tenant")
    assert (fetched.state, fetched.error, fetched.error_category) == (FAILED, "boom", "user")


def test_cross_tenant_get_raises_ownership():
    store = RequestFutureStore()
    future = store.create("model", "tenant-a")
    with pytest.raises(OwnershipError):
        store.get(future.request_id, "tenant-b")


def test_an_expired_future_returns_none(monkeypatch):
    store = RequestFutureStore()
    future = store.create("model", "tenant")
    store.resolve(future.request_id, {"op": "optim_step"})

    monkeypatch.setattr(future_module, "_FINISHED_TTL_S", -1.0)
    assert store.get(future.request_id, "tenant") is None, "finished past TTL must read as unknown (410)"


def test_terminal_states_do_not_flip():
    store = RequestFutureStore()
    failed = store.create("model", "tenant")
    store.fail(failed.request_id, "lease expired", "user")
    store.resolve(failed.request_id, {"op": "forward_backward"})
    fetched = store.get(failed.request_id, "tenant")
    assert (fetched.state, fetched.result) == (FAILED, None), "a unit that raced with lease expiry must not revive it"

    done = store.create("model", "tenant")
    store.resolve(done.request_id, {"op": "optim_step"})
    store.fail(done.request_id, "late failure", "server")
    assert store.get(done.request_id, "tenant").state == DONE


def test_a_delivered_result_expires_on_the_short_ttl(monkeypatch):
    store = RequestFutureStore()
    undelivered = store.create("model", "tenant")
    delivered = store.create("model", "tenant")
    store.resolve(undelivered.request_id, {"op": "optim_step"})
    store.resolve(delivered.request_id, {"op": "optim_step"})
    store.delivered(delivered)

    monkeypatch.setattr(future_module, "_DELIVERED_TTL_S", -1.0)
    assert store.get(delivered.request_id, "tenant") is None, "a delivered result only serves a lost-response retry"
    assert store.get(undelivered.request_id, "tenant") is not None, "an unread result waits for its poller"


def test_a_pending_future_is_never_marked_delivered():
    store = RequestFutureStore()
    future = store.create("model", "tenant")
    store.delivered(future)
    assert future.delivered_at is None


def test_the_full_sweep_runs_at_most_once_per_interval(monkeypatch):
    store = RequestFutureStore()
    stale = store.create("model", "tenant")
    store.resolve(stale.request_id, {"op": "optim_step"})
    store.delivered(stale)
    monkeypatch.setattr(future_module, "_DELIVERED_TTL_S", -1.0)

    other = store.create("model", "tenant")
    store.get(other.request_id, "tenant")
    assert stale.request_id in store._futures, "a lookup of another future must not walk the whole store"

    monkeypatch.setattr(future_module, "_SWEEP_INTERVAL_S", -1.0)
    store.get(other.request_id, "tenant")
    assert stale.request_id not in store._futures
