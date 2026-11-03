"""Long training windows retain delayed HTTP results without replaying gradients."""

import asyncio
import time
from types import SimpleNamespace

from miles.tinker.core import promise as promise_module
from miles.tinker.core import service as service_module
from miles.tinker.core.promise import DONE
from tests.fast.tinker.server.test_app import _create_model, _fb_body, _headers, _poll
from tests.fast.tinker.server.test_app import client as client  # noqa: F401  pytest fixture


async def test_delayed_training_result_retry_and_session_expiry_over_http(client, monkeypatch):
    service = client.service
    clock = SimpleNamespace(now=time.monotonic())
    lifecycle_time = SimpleNamespace(monotonic=lambda: clock.now, time=time.time)
    # Advance retention/lease time only: HTTP and event-loop deadlines stay real.
    monkeypatch.setattr(promise_module, "time", lifecycle_time)
    monkeypatch.setattr(service_module, "time", lifecycle_time)
    service.config.coalesce_wait_s = service.config.coalesce_max_wait_s = 0
    service.planner.unit_token_budget = 4  # One three-token datum per backend unit.

    async def session(tenant):
        response = await client.post("/api/v1/create_session", json={}, headers=_headers(tenant))
        assert response.status_code == 200
        return response.json()["session_id"]

    owner, other = "tml-window-owner", "tml-unrelated-owner"
    active_session, idle_session = await session(owner), await session(other)
    active_model = await _create_model(client, owner, session_id=active_session, model_seq_id=0)
    idle_model = await _create_model(client, other, session_id=idle_session, model_seq_id=0)
    idle_slot = service.models[idle_model].slot

    later_started, finish_later = asyncio.Event(), asyncio.Event()
    original_forward_backward = service.backend.forward_backward
    calls = 0

    async def forward_backward(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            later_started.set()
            await finish_later.wait()
        return await original_forward_backward(*args, **kwargs)

    monkeypatch.setattr(service.backend, "forward_backward", forward_backward)
    # Post both chunks before the first dispatch, so the first result belongs
    # to the same continuous model queue as the deliberately slow second one.
    body = _fb_body(active_model, 1)
    first = await client.post("/api/v1/forward_backward", json=body, headers=_headers(owner))
    request_id = first.json()["request_id"]
    later = await client.post(
        "/api/v1/forward_backward", json=_fb_body(active_model, 2), headers=_headers(owner)
    )
    await asyncio.wait_for(later_started.wait(), timeout=5)
    assert service.promises.get(request_id, owner).state == DONE

    clock.now += 2 * 3600  # Beyond the former one-hour completed-result TTL and ten-minute lease.
    sweep = asyncio.create_task(service._sweep_once())
    try:
        # Releasing the unrelated idle model may wait for the backend lock,
        # but the pending owner's logical model/session must remain live.
        await asyncio.sleep(0.01)
        assert active_model in service.models
        assert active_session in service.sessions
        denied = await client.post(
            "/api/v1/retrieve_future", json={"request_id": request_id}, headers=_headers(other)
        )
        assert denied.status_code == 403
        result = await _poll(client, request_id, owner)
        assert result["metrics"]["num_tokens:sum"] == 3
        retried = await client.post("/api/v1/forward_backward", json=body, headers=_headers(owner))
        assert retried.json()["request_id"] == request_id
        assert (await _poll(client, request_id, owner)) == result
        assert calls == 2, "Retrying a completed chunk must not accumulate gradients again"
        # The real SDK sends heartbeats while it drains the remaining results.
        heartbeat = await client.post(
            "/api/v1/session_heartbeat", json={"session_id": active_session}, headers=_headers(owner)
        )
        assert heartbeat.status_code == 200
    finally:
        finish_later.set()
        await asyncio.wait_for(sweep, timeout=5)

    assert (await _poll(client, later.json()["request_id"], owner))["metrics"]["num_tokens:sum"] == 3
    assert active_model in service.models
    assert idle_model not in service.models and idle_session not in service.sessions
    assert service.backend.named("unload_slot") == [{"slot": idle_slot}]
    assert len(service.backend.named("forward_backward")) == 2

    # Once work and its bounded retention grace end, normal expiry remains.
    # Keep the model alive to prove even an expired request's sequence ID
    # cannot accidentally execute the same gradients a second time.
    clock.now += 2 * 3600
    await client.post(
        "/api/v1/session_heartbeat", json={"session_id": active_session}, headers=_headers(owner)
    )
    expired = await client.post(
        "/api/v1/retrieve_future", json={"request_id": request_id}, headers=_headers(owner)
    )
    assert expired.status_code == 410
    retried = await client.post("/api/v1/forward_backward", json=body, headers=_headers(owner))
    assert retried.json()["request_id"] == request_id
    await asyncio.sleep(0)
    assert calls == 2
    clock.now += service.config.lease_timeout_s + 1
    await service._sweep_once()
    assert active_model not in service.models
