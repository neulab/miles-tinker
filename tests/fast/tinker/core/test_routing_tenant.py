"""The service's share of R3: sample payloads and unit rows are tagged with
their tenant, a handle-less forward_backward is warned about once per request
when the backend replays routing, and the lease sweep drives the backend sweep."""

import asyncio
import logging

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, created_model, fb_payload, make_service, row

from miles.tinker.core.promise import DONE


def _sample_payload(**overrides) -> dict:
    return {
        "num_samples": 1,
        "prompt_tokens": [1, 2],
        "sampling_params": {"max_tokens": 4},
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
        **overrides,
    }


@pytest.fixture
async def service():
    svc = make_service()
    run_task = asyncio.create_task(svc.run())
    yield svc
    for task in (run_task, getattr(svc, "_sweep_task", None)):
        if task is not None:
            task.cancel()


async def test_sample_payloads_carry_the_tenant(service):
    request_id, _ = service.submit_sample("acme", _sample_payload())
    assert (await await_settled(service, "acme", request_id)).state == DONE
    assert service.backend.named("sample")[0]["payload"]["tenant"] == "acme"


async def test_unit_rows_carry_the_tenant_without_touching_the_request_rows(service):
    model_id = await created_model(service, tenant="acme")
    payload = fb_payload(model_id, 1, [row(3)])
    request_id = service.submit("acme", "forward_backward", payload)
    service.submit("acme", "optim_step", {"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)})
    assert (await await_settled(service, "acme", request_id)).state == DONE
    (slot, unit_row) = service.backend.named("forward_backward")[0]["slot_rows"][0]
    assert unit_row["tenant"] == "acme" and unit_row["tokens"] == payload["rows"][0]["tokens"]
    assert "tenant" not in payload["rows"][0], "a shallow copy: the decoded request row is untouched"


async def test_a_handle_less_request_warns_once_only_when_routing_replays(service, caplog):
    model_id = await created_model(service)
    with caplog.at_level(logging.WARNING, logger="miles.tinker.core.service"):
        service.submit("tenant", "forward_backward", fb_payload(model_id, 0, [row(3), row(2)]))
        assert not [r for r in caplog.records if "routing_handle" in r.getMessage()]
        service.backend.routing_replay = True
        service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row(3), row(2)]))
        service.submit(
            "tenant", "forward_backward", fb_payload(model_id, 2, [{**row(3), "routing_handle": 5}, row(2)])
        )
        service.submit("tenant", "forward_only", fb_payload(model_id, 3, [row(3)]))
    warnings = [r.getMessage() for r in caplog.records if "routing_handle" in r.getMessage()]
    assert warnings == ["R3 enabled but no datum carries routing_handle (2 datums in this request)"]


async def test_the_lease_sweep_calls_the_backend_sweep_and_survives_its_failure(service, monkeypatch, caplog):
    calls = []

    def sweep():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(service.backend, "sweep", sweep)
    with caplog.at_level(logging.WARNING, logger="miles.tinker.core.service"):
        service._sweep_backend()
        service._sweep_backend()
    assert len(calls) == 2
    assert any("backend sweep failed: RuntimeError: boom" in r.getMessage() for r in caplog.records)


def test_the_default_backend_sweep_is_a_no_op():
    backend = make_service().backend
    assert backend.routing_replay is False and backend.sweep() is None
