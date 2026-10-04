"""Server-owned work outlives a stalled SDK heartbeat; idle state is still reclaimed."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from tests.fast.tinker.harness import create_model_payload, created_model, fb_payload, row

from miles.tinker.core import promise as promise_module
from miles.tinker.core import service as service_module
from miles.tinker.core.promise import DONE, FAILED, PENDING


@pytest.fixture
def clock(monkeypatch, service):
    now = [100.0]
    fake = SimpleNamespace(monotonic=lambda: now[0], time=time.time)
    monkeypatch.setattr(promise_module, "time", fake)
    monkeypatch.setattr(service_module, "time", fake)
    service.promises._last_sweep = now[0]
    service.config.coalesce_wait_s = 0
    return now


async def test_running_and_queued_chunks_survive_without_heartbeat_then_release(service, clock):
    service.planner.unit_token_budget = 4
    service.config.coalesce_wait_s = 0
    owner = service.create_session("tenant", {})
    model = await created_model(service, session_id=owner)
    second_started, release = asyncio.Event(), asyncio.Event()
    run = service.backend.forward_backward

    async def delayed(unit_id, *args):
        if unit_id == 2:
            second_started.set()
            await release.wait()
        return await run(unit_id, *args)

    service.backend.forward_backward = delayed
    first = service.submit("tenant", "forward_backward", fb_payload(model, 1, [row()]))
    second = service.submit("tenant", "forward_backward", fb_payload(model, 2, [row()]))
    await asyncio.wait_for(second_started.wait(), 2)
    early = service.promises.get(first, "tenant")
    assert early.state == DONE and early.window.closed_at is None
    service.promises.delivered(early)
    clock[0] += 7200
    await service._sweep_once()
    assert model in service.models and owner in service.sessions
    assert service.promises.get(first, "tenant") is early
    assert service.promises.get(second, "tenant").state == PENDING
    release.set()
    await asyncio.wait_for(service.promises.get(second, "tenant").settled.wait(), 2)
    assert early.window.closed_at == clock[0]
    assert model not in service._result_windows
    clock[0] += service.config.lease_timeout_s + 1
    await service._sweep_once()
    assert model not in service.models and owner not in service.sessions
    assert service.promises.get(first, "tenant") is early
    clock[0] += 3600
    service._sweep_backend()
    assert first not in service.promises._promises and second not in service.promises._promises


async def test_windows_do_not_cross_models_or_reopen_for_later_work(service, clock):
    owner = service.create_session("tenant", {})
    model = await created_model(service, session_id=owner)
    other = await created_model(service, session_id=owner)
    first = service.submit("tenant", "forward_backward", fb_payload(model, 1, [row()]))
    promise = service.promises.get(first, "tenant")
    await asyncio.wait_for(promise.settled.wait(), 2)
    old_window = promise.window
    assert old_window.closed_at is not None
    gate = asyncio.Event()
    original = service.backend.forward_backward

    async def blocked(*args):
        await gate.wait()
        return await original(*args)

    service.backend.forward_backward = blocked
    later = service.submit("tenant", "forward_backward", fb_payload(model, 2, [row()]))
    independent = service.submit("tenant", "forward_backward", fb_payload(other, 1, [row()]))
    assert service.promises.get(later, "tenant").window is not old_window
    assert service.promises.get(later, "tenant").window is not service.promises.get(independent, "tenant").window
    clock[0] += 3601
    assert service.promises.get(first, "tenant") is None
    gate.set()
    await asyncio.wait_for(service.promises.get(independent, "tenant").settled.wait(), 2)


@pytest.mark.parametrize("failure", [False, True])
async def test_failed_work_also_closes_window(service, clock, failure):
    model = await created_model(service)
    if failure:
        service.backend.fail_next = RuntimeError("backend unavailable")
    rid = service.submit("tenant", "forward_backward", fb_payload(model, 1, [row()]))
    promise = service.promises.get(rid, "tenant")
    await asyncio.wait_for(promise.settled.wait(), 2)
    assert promise.state == (FAILED if failure else DONE)
    assert promise.window.closed_at == clock[0]
    assert not service._result_windows


async def test_queued_and_running_samples_keep_only_their_owner_then_cancel_drains(service, clock):
    service._sample_slots = asyncio.Semaphore(1)
    owner = service.create_session("tenant", {})
    stale = service.create_session("tenant", {})
    stale_model = await created_model(service, session_id=stale)
    sid = service.create_sampling_session("tenant", {"session_id": owner, "base_model": "base"})
    gate = asyncio.Event()

    async def blocked(*args):
        await gate.wait()
        return {"sequences": []}

    service.backend.sample = blocked
    requests = [service.submit_sample("tenant", {"sampling_session_id": sid, "num_samples": 1})[0] for _ in range(2)]
    await asyncio.sleep(0)
    clock[0] += 7200
    await service._sweep_once()
    assert owner in service.sessions and sid in service.sampling_sessions
    assert stale not in service.sessions and stale_model not in service.models
    for rid in requests:
        service.cancel("tenant", rid)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not service._sample_tasks
    assert all(service.promises.get(rid, "tenant").state == FAILED for rid in requests)
    clock[0] += service.config.lease_timeout_s + 1
    await service._sweep_once()
    assert owner not in service.sessions and sid not in service.sampling_sessions


async def test_lease_sweep_rechecks_owners_after_awaiting_another_unload(service, clock):
    one = service.create_session("tenant", {})
    two = service.create_session("tenant", {})
    first = await created_model(service, session_id=one)
    second = await created_model(service, session_id=two)
    started, release = asyncio.Event(), asyncio.Event()
    unload = service.backend.unload_slot

    async def delayed(slot):
        started.set()
        await release.wait()
        return await unload(slot)

    service.backend.unload_slot = delayed
    clock[0] += service.config.lease_timeout_s + 1
    sweep = asyncio.create_task(service._sweep_once())
    await asyncio.wait_for(started.wait(), 2)
    rid = service.submit("tenant", "forward_backward", fb_payload(second, 1, [row()]))
    other_sweep = asyncio.create_task(service._sweep_once())
    release.set()
    await asyncio.wait_for(asyncio.gather(sweep, other_sweep), 2)
    assert first not in service.models and one not in service.sessions
    assert second in service.models and two in service.sessions
    await asyncio.wait_for(service.promises.get(rid, "tenant").settled.wait(), 2)


async def test_session_end_before_initialization_never_loads_the_released_slot(service):
    owner = service.create_session("tenant", {})
    async with service._backend_lock:
        rid, model = service.create_model("tenant", create_model_payload(service, session_id=owner))
        ending = asyncio.create_task(service.end_session(owner))
        await asyncio.sleep(0)
        assert model not in service.models
    await asyncio.wait_for(ending, 2)
    promise = service.promises.get(rid, "tenant")
    await asyncio.wait_for(promise.settled.wait(), 2)
    assert promise.state == FAILED
    assert not service.backend.named("load_slot")
    assert len(service.free_slots) == service.config.n_slots


async def test_initialization_failure_after_session_end_settles_without_double_release(service, clock):
    started, release = asyncio.Event(), asyncio.Event()

    async def fail_load(*args):
        started.set()
        await release.wait()
        raise RuntimeError("load failed")

    service.backend.load_slot = fail_load
    owner = service.create_session("tenant", {})
    rid, model = service.create_model("tenant", create_model_payload(service, session_id=owner))
    await asyncio.wait_for(started.wait(), 2)
    clock[0] += 7200
    await service._sweep_once()
    assert owner in service.sessions and model in service.models  # active control work owns its lease
    ending = asyncio.create_task(service.end_session(owner))
    await asyncio.sleep(0)
    assert model not in service.models
    release.set()
    await asyncio.wait_for(ending, 2)
    promise = service.promises.get(rid, "tenant")
    assert promise.state == FAILED and promise.error == "load failed"
    assert len(service.backend.named("unload_slot")) == 1
    assert len(service.free_slots) == service.config.n_slots


async def test_completed_control_task_keeps_lease_until_its_done_callback(service, clock):
    owner = service.create_session("tenant", {})
    _, model = service.create_model("tenant", create_model_payload(service, session_id=owner))
    await asyncio.sleep(0)  # initialization finished; its done callback is queued for the next turn
    assert any(task.done() for task in service._background_owners)
    clock[0] += service.config.lease_timeout_s + 1
    await service._sweep_once()
    assert model in service.models and owner in service.sessions
    await asyncio.sleep(0)
    assert not service._background_owners
    clock[0] += service.config.lease_timeout_s + 1
    await service._sweep_once()
    assert model not in service.models and owner not in service.sessions
