"""HTTP conversations over the real FastAPI app: the 0.27.1-style JSON flow,
the 0.16.1 body dialect (exclude_unset bodies), the long-poll, and the error
mapping the SDK's retry loop depends on (400/403/404/408/410)."""

import asyncio
import json
import os
import subprocess
import sys
import time

import httpx
import pytest
from tests.fast.tinker.harness import ADAM, make_service

from miles.tinker.server.app import build_app

TINKER016_SITE = os.environ.get(
    "TINKER016_SITE",
    "/tmp/claude-18912/-home-aci18912it/f39085d7-e0f4-47a7-9042-fab61946e8de/scratchpad/notes/tinker016",
)


@pytest.fixture
async def client():
    service = make_service(retrieve_poll_timeout_s=0.2)
    run_task = asyncio.create_task(service.run())
    transport = httpx.ASGITransport(app=build_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as http:
        http.service = service
        yield http
    for task in (run_task, getattr(service, "_sweep_task", None)):
        if task is not None:
            task.cancel()


def _headers(tenant: str = "tml-tenant-a") -> dict:
    return {"X-API-Key": tenant}


async def _poll(client, request_id: str, tenant: str = "tml-tenant-a") -> dict:
    for _ in range(50):
        response = await client.post(
            "/api/v1/retrieve_future", json={"request_id": request_id}, headers=_headers(tenant)
        )
        if response.status_code != 408:
            assert response.status_code == 200, response.text
            return response.json()
    raise AssertionError(f"{request_id} never settled")


def _fb_body(model_id: str, seq_id: int) -> dict:
    return {
        "model_id": model_id,
        "seq_id": seq_id,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "loss_fn_inputs": {"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
                }
            ],
            "loss_fn": "cross_entropy",
        },
    }


async def _create_model(client, tenant: str = "tml-tenant-a", **extra) -> str:
    body = {"base_model": "base", "lora_config": {"rank": 8}, **extra}
    created = (await client.post("/api/v1/create_model", json=body, headers=_headers(tenant))).json()
    assert (await _poll(client, created["request_id"], tenant))["type"] == "create_model"
    return created["model_id"]


async def test_the_training_conversation(client):
    session = await client.post("/api/v1/create_session", json={"sdk_version": "0.27.1"}, headers=_headers())
    session_id = session.json()["session_id"]
    assert session_id.startswith("session-")
    model_id = await _create_model(client, session_id=session_id, model_seq_id=0, user_metadata={"renderer_name": "r"})

    info = (await client.post("/api/v1/get_info", json={"model_id": model_id}, headers=_headers())).json()
    assert (info["model_data"]["model_name"], info["model_data"]["tokenizer_id"]) == ("base", "base")
    assert (info["is_lora"], info["lora_rank"], info["model_name"]) == (True, 8, "base")

    fb = (await client.post("/api/v1/forward_backward", json=_fb_body(model_id, 1), headers=_headers())).json()
    fb_result = await _poll(client, fb["request_id"])
    assert [len(o["logprobs"]["data"]) for o in fb_result["loss_fn_outputs"]] == [3]
    assert fb_result["metrics"] == {"loss:sum": 1.0, "num_tokens:sum": 3.0, "num_datums:sum": 1.0}

    optim = (
        await client.post(
            "/api/v1/optim_step",
            json={"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)},
            headers=_headers(),
        )
    ).json()
    assert "grad_norm" in (await _poll(client, optim["request_id"]))["metrics"]

    named = (
        await client.post(
            "/api/v1/save_weights_for_sampler",
            json={"model_id": model_id, "seq_id": 3, "path": "000001"},
            headers=_headers(),
        )
    ).json()
    sampler_path = (await _poll(client, named["request_id"]))["path"]
    assert sampler_path == f"tinker://{model_id}/sampler_weights/000001"

    unnamed = (
        await client.post(
            "/api/v1/save_weights_for_sampler",
            json={"model_id": model_id, "seq_id": 4, "sampling_session_seq_id": 0},
            headers=_headers(),
        )
    ).json()
    unnamed_result = await _poll(client, unnamed["request_id"])
    assert unnamed_result["path"] is None and unnamed_result["sampling_session_id"].startswith("sampling-")

    sampler = (
        await client.get(f"/api/v1/samplers/{unnamed_result['sampling_session_id']}", headers=_headers())
    ).json()
    assert sampler["model_path"] == f"tinker://{model_id}/sampler_weights/2"

    created = (
        await client.post(
            "/api/v1/create_sampling_session",
            json={"session_id": session_id, "sampling_session_seq_id": 1, "model_path": sampler_path},
            headers=_headers(),
        )
    ).json()
    sample = (
        await client.post(
            "/api/v1/asample",
            json={
                "sampling_session_id": created["sampling_session_id"],
                "seq_id": 0,
                "num_samples": 2,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
                "sampling_params": {"max_tokens": 4},
            },
            headers=_headers(),
        )
    ).json()
    assert len(sample["sample_sequence_ids"]) == 2, "asample must answer an UntypedAPIFuture"
    assert len((await _poll(client, sample["request_id"]))["sequences"]) == 2

    saved = (
        await client.post(
            "/api/v1/save_weights", json={"model_id": model_id, "seq_id": 5, "path": "ckpt"}, headers=_headers()
        )
    ).json()
    state_path = (await _poll(client, saved["request_id"]))["path"]
    weights_info = (
        await client.post("/api/v1/weights_info", json={"tinker_path": state_path}, headers=_headers())
    ).json()
    assert (weights_info["base_model"], weights_info["is_lora"], weights_info["lora_rank"]) == ("base", True, 8)

    run = (await client.get(f"/api/v1/training_runs/{model_id}?access_scope=owned", headers=_headers())).json()
    assert (run["training_run_id"], run["model_owner"], run["user_metadata"]) == (
        model_id,
        "tml-tenant-a",
        {"renderer_name": "r"},
    )
    assert run["last_request_time"].endswith("+00:00")

    deleted = await client.delete(f"/api/v1/training_runs/{model_id}/checkpoints/weights/ckpt", headers=_headers())
    assert deleted.status_code == 204
    assert (
        await client.post("/api/v1/weights_info", json={"tinker_path": state_path}, headers=_headers())
    ).status_code == 404

    unloaded = (await client.post("/api/v1/unload_model", json={"model_id": model_id}, headers=_headers())).json()
    assert (await _poll(client, unloaded["request_id"]))["type"] == "unload_model"
    assert (await client.get(f"/api/v1/training_runs/{model_id}", headers=_headers())).status_code == 404


# -------- 0.16.1 dialect: exclude_unset bodies --------

SDK016_BODIES = {
    "forward_backward": {
        "forward_backward_input": {
            "data": [
                {
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2, 3, 4], "dtype": "int64", "shape": [3]},
                        "weights": {"data": [1.0, 1.0, 1.0], "dtype": "float32", "shape": [3]},
                    },
                    "model_input": {"chunks": [{"tokens": [1, 2, 3]}]},
                }
            ],
            "loss_fn": "cross_entropy",
        },
        "model_id": "MODEL",
        "seq_id": 1,
    },
    "forward": {
        "forward_input": {
            "data": [
                {
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2, 3, 4], "dtype": "int64", "shape": [3]},
                        "weights": {"data": [1.0, 1.0, 1.0], "dtype": "float32", "shape": [3]},
                    },
                    "model_input": {"chunks": [{"tokens": [1, 2, 3]}]},
                }
            ],
            "loss_fn": "cross_entropy",
        },
        "model_id": "MODEL",
        "seq_id": 2,
    },
    "optim_step": {"adam_params": {"learning_rate": 0.0001}, "model_id": "MODEL", "seq_id": 3},
    "save_weights_for_sampler": {"model_id": "MODEL", "sampling_session_seq_id": 0, "seq_id": 4},
    "asample": {
        "num_samples": 1,
        "prompt": {"chunks": [{"tokens": [5, 6]}]},
        "sampling_params": {"max_tokens": 4, "seed": 42, "stop": ["</python>"]},
        "sampling_session_id": "SESSION",
        "seq_id": 0,
        "prompt_logprobs": True,
        "topk_prompt_logprobs": 2,
    },
}

_DUMP_016 = """
import json, sys
from tinker import types
d = types.Datum(model_input=types.ModelInput.from_ints([1, 2, 3]), loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]})
fb = types.ForwardBackwardRequest(model_id="MODEL", seq_id=1, forward_backward_input=types.ForwardBackwardInput(data=[d], loss_fn="cross_entropy"))
fw = types.ForwardRequest(model_id="MODEL", seq_id=2, forward_input=types.ForwardBackwardInput(data=[d], loss_fn="cross_entropy"))
op = types.OptimStepRequest(model_id="MODEL", seq_id=3, adam_params=types.AdamParams(learning_rate=1e-4))
sw = types.SaveWeightsForSamplerRequest(model_id="MODEL", seq_id=4, sampling_session_seq_id=0)
sa = types.SampleRequest(sampling_session_id="SESSION", seq_id=0, num_samples=1, prompt=types.ModelInput.from_ints([5, 6]),
                         sampling_params=types.SamplingParams(max_tokens=4, seed=42, stop=["</python>"]), prompt_logprobs=True, topk_prompt_logprobs=2)
print(json.dumps({k: v.model_dump(exclude_unset=True) for k, v in
                  [("forward_backward", fb), ("forward", fw), ("optim_step", op), ("save_weights_for_sampler", sw), ("asample", sa)]}))
"""


def test_the_016_fixtures_match_the_real_sdk_dumps():
    """Cross-check the hand-written bodies against the 0.16.1 package when it is available."""
    if not os.path.isdir(os.path.join(TINKER016_SITE, "tinker")):
        pytest.skip("tinker 0.16.1 site-packages not available")
    env = {**os.environ, "PYTHONPATH": TINKER016_SITE, "PYTHONDONTWRITEBYTECODE": "1"}
    dumped = subprocess.run([sys.executable, "-c", _DUMP_016], env=env, capture_output=True, text=True, check=True)
    assert json.loads(dumped.stdout) == SDK016_BODIES


def _body_016(kind: str, model_id: str, sampling_session_id: str | None = None) -> dict:
    body = json.loads(json.dumps(SDK016_BODIES[kind]))
    if "model_id" in body:
        body["model_id"] = model_id
    if "sampling_session_id" in body:
        body["sampling_session_id"] = sampling_session_id
    return body


async def test_the_016_json_flow(client):
    """platoon's dialect: chunks without type, sparse adam params, POST /forward,
    unnamed sampler saves, JSON sample responses with null-padded prompt logprobs."""
    model_id = await _create_model(client)

    async def fake_sample(payload, lora_name):
        return {
            "sequences": [{"tokens": [7], "logprobs": [-0.5], "stop_reason": "stop"}],
            "prompt_logprobs": [float("nan"), -0.25],
            "topk_prompt_logprobs": {"token_ids": [[0, 0], [5, 6]], "logprobs": [[float("nan")] * 2, [-0.1, -0.2]]},
        }

    client.service.backend.sample = fake_sample

    fb = (
        await client.post("/api/v1/forward_backward", json=_body_016("forward_backward", model_id), headers=_headers())
    ).json()
    fb_result = await _poll(client, fb["request_id"])
    assert set(fb_result) >= {"loss_fn_output_type", "loss_fn_outputs", "metrics"}

    fw = (await client.post("/api/v1/forward", json=_body_016("forward", model_id), headers=_headers())).json()
    assert (await _poll(client, fw["request_id"]))["loss_fn_outputs"][0]["loss:sum"]["data"] == [0.0]
    assert client.service.backend.named("forward_only"), "POST /forward is forward-only"

    optim = (
        await client.post("/api/v1/optim_step", json=_body_016("optim_step", model_id), headers=_headers())
    ).json()
    assert "grad_norm" in (await _poll(client, optim["request_id"]))["metrics"]
    assert client.service.backend.named("optim_step")[0]["adam_params_by_slot"][0]["beta2"] == 0.95

    unnamed = (
        await client.post(
            "/api/v1/save_weights_for_sampler",
            json=_body_016("save_weights_for_sampler", model_id),
            headers=_headers(),
        )
    ).json()
    result = await _poll(client, unnamed["request_id"])
    assert result["path"] is None and result["sampling_session_id"]

    sample = (
        await client.post(
            "/api/v1/asample", json=_body_016("asample", model_id, result["sampling_session_id"]), headers=_headers()
        )
    ).json()
    body = await _poll(client, sample["request_id"])
    assert body["sequences"] == [{"tokens": [7], "logprobs": [-0.5], "stop_reason": "stop"}]
    assert body["prompt_logprobs"] == [None, -0.25]
    assert body["topk_prompt_logprobs"] == [None, [[5, -0.1], [6, -0.2]]]


# -------- long-poll --------


async def test_retrieve_future_long_polls_then_answers_408(client):
    gate = asyncio.Event()

    async def slow_sample(payload, lora_name):
        await gate.wait()
        return {"sequences": []}

    client.service.backend.sample = slow_sample
    sid = (
        await client.post("/api/v1/create_sampling_session", json={"base_model": "base"}, headers=_headers())
    ).json()
    sample = (
        await client.post(
            "/api/v1/asample",
            json={
                "sampling_session_id": sid["sampling_session_id"],
                "seq_id": 0,
                "prompt": {"chunks": [{"tokens": [1]}]},
                "sampling_params": {"max_tokens": 1},
            },
            headers=_headers(),
        )
    ).json()

    started = time.monotonic()
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": sample["request_id"]}, headers=_headers()
    )
    held = time.monotonic() - started
    assert response.status_code == 408
    assert response.json() == {"type": "try_again", "request_id": sample["request_id"], "queue_state": "active"}
    assert held >= client.service.config.retrieve_poll_timeout_s * 0.9, "the request must be held for the cap"

    async def release_soon():
        await asyncio.sleep(0.03)
        gate.set()

    asyncio.create_task(release_soon())
    started = time.monotonic()
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": sample["request_id"]}, headers=_headers()
    )
    assert response.status_code == 200 and response.json()["type"] == "sample"
    assert (
        time.monotonic() - started < client.service.config.retrieve_poll_timeout_s
    ), "a settling promise wakes the poll"


# -------- error mapping --------


async def test_bad_input_answers_400(client):
    body = _fb_body("model-missing", 1)
    body["forward_backward_input"]["data"][0]["loss_fn_inputs"]["target_tokens"] = [9, 9, 9]
    response = await client.post("/api/v1/forward_backward", json=body, headers=_headers())
    assert response.status_code == 400


async def test_a_foreign_promise_answers_403(client):
    created = (
        await client.post(
            "/api/v1/create_model", json={"base_model": "base", "lora_config": {"rank": 8}}, headers=_headers("tml-a")
        )
    ).json()
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": created["request_id"]}, headers=_headers("tml-b")
    )
    assert response.status_code == 403


async def test_the_bearer_header_is_the_fallback_tenant(client):
    created = (
        await client.post(
            "/api/v1/create_model",
            json={"base_model": "base", "lora_config": {"rank": 8}},
            headers={"Authorization": "Bearer tml-bearer"},
        )
    ).json()
    assert client.service.models[created["model_id"]].tenant == "tml-bearer"
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": created["request_id"]}, headers=_headers("tml-bearer")
    )
    assert response.status_code == 200


async def test_an_unknown_promise_answers_410(client):
    response = await client.post("/api/v1/retrieve_future", json={"request_id": "req-gone"}, headers=_headers())
    assert response.status_code == 410


async def test_unknown_samplers_answer_400_and_404(client):
    response = await client.post(
        "/api/v1/asample",
        json={
            "sampling_session_id": "sampling-nope",
            "prompt": {"chunks": [{"tokens": [1]}]},
            "sampling_params": {"max_tokens": 1},
        },
        headers=_headers(),
    )
    assert response.status_code == 400
    assert (await client.get("/api/v1/samplers/sampling-nope", headers=_headers())).status_code == 404


async def test_a_failed_promise_reports_the_category(client):
    client.service.backend.fail_next = RuntimeError("boom")
    created = (
        await client.post(
            "/api/v1/create_model", json={"base_model": "base", "lora_config": {"rank": 8}}, headers=_headers()
        )
    ).json()
    body = await _poll(client, created["request_id"])
    assert (body["category"], "boom" in body["error"]) == ("server", True)


async def test_capabilities_health_and_telemetry_shapes(client):
    capabilities = (await client.get("/api/v1/get_server_capabilities")).json()
    assert capabilities["supported_models"][0]["model_name"] == "base"
    assert (await client.post("/api/v1/telemetry", json={})).json() == {"status": "accepted"}
    assert (await client.get("/health")).json() == (await client.get("/api/v1/healthz")).json() == {"status": "ok"}


async def test_a_session_end_event_frees_the_slots_at_once(client):
    session = await client.post("/api/v1/create_session", json={"sdk_version": "0.27.1"}, headers=_headers())
    session_id = session.json()["session_id"]
    model_id = await _create_model(client, session_id=session_id, model_seq_id=0)
    free_before = len(client.service.free_slots)

    batch = {
        "session_id": session_id,
        "sdk_version": "0.27.1",
        "platform": "linux",
        "process_uuid": "p",
        "events": [{"event": "SESSION_END", "session_id": session_id, "timestamp": 0}],
    }
    assert (await client.post("/api/v1/telemetry", json=batch)).json() == {"status": "accepted"}

    assert model_id not in client.service.models
    assert len(client.service.free_slots) == free_before + 1
    # telemetry without a session end (or malformed) still just accepts
    assert (await client.post("/api/v1/telemetry", content=b"not json")).json() == {"status": "accepted"}
