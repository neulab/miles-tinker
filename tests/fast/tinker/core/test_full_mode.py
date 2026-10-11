"""train_mode="full": one model, LoRA configs ignored, sampler saves push the
base weights, every sampler path samples the latest weights, and checkpoints
record their mode."""

import asyncio
from contextlib import suppress

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, created_model, datum, fb_payload, make_service

from miles.tinker.core.future import DONE, FAILED
from miles.tinker.core.types import UserInputError


@pytest.fixture
async def full_service(tmp_path):
    service = make_service(tmp_path, train_mode="full", n_slots=1)
    run_task = asyncio.create_task(service.run())
    try:
        yield service
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


def _save_payload(model_id: str, seq_id: int, name: str | None = None) -> dict:
    return {"model_id": model_id, "seq_id": seq_id, "name": name, "overwrite": False}


def test_full_mode_serves_exactly_one_model(tmp_path):
    with pytest.raises(ValueError, match="n_slots"):
        make_service(tmp_path, train_mode="full", n_slots=2)


async def test_create_model_ignores_the_lora_config(full_service):
    model_id = await created_model(full_service)  # the harness asks for rank 8
    record = full_service.models[model_id]
    assert (record.lora_rank, record.lora_alpha) == (0, 0.0)
    assert full_service.backend.named("load_slot")[0]["ckpt_path"] is None, "a new model starts from the base"

    with pytest.raises(UserInputError, match="no free adapter slots"):
        full_service.create_model("tenant", {"base_model": "base", "session_id": record.session_id, "model_seq_id": 2})


async def test_sampler_saves_sample_the_latest_weights(full_service):
    model_id = await created_model(full_service)
    first = full_service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    second = full_service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2})
    first_result = (await await_settled(full_service, "tenant", first)).result
    second_result = (await await_settled(full_service, "tenant", second)).result
    assert first_result["path"] == f"tinker://{model_id}/sampler_weights/1"
    assert second_result["path"] == f"tinker://{model_id}/sampler_weights/2"

    for result in (first_result, second_result):
        request_id, _ = full_service.submit_sample(
            "tenant",
            {
                "sampling_session_id": result["sampling_session_id"],
                "seq_id": 0,
                "num_samples": 1,
                "prompt_tokens": [1],
                "sampling_params": {"max_tokens": 2},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
        )
        assert (await await_settled(full_service, "tenant", request_id)).state == DONE
    samples = full_service.backend.named("sample")
    assert [(call["lora_name"], call["lora_path"]) for call in samples] == [(None, None), (None, None)]

    with pytest.raises(UserInputError, match="unknown checkpoint"):
        full_service.submit_sample(
            "tenant",
            {
                "model_path": f"tinker://{model_id}/sampler_weights/7",
                "num_samples": 1,
                "prompt_tokens": [1],
                "sampling_params": {"max_tokens": 2},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
        )


async def test_a_full_checkpoint_resumes_and_reports_lora_for_the_sdk(full_service):
    model_id = await created_model(full_service)
    save = full_service.submit("tenant", "save_state", _save_payload(model_id, 1, "ckpt"))
    path = (await await_settled(full_service, "tenant", save)).result["path"]

    info = full_service.weights_info("tenant", path)
    assert (info["is_lora"], info["lora_rank"]) == (True, 0), "the SDK asserts is_lora when it resumes"

    load = full_service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    assert (await await_settled(full_service, "tenant", load)).state == DONE


async def test_a_lora_checkpoint_cannot_load_into_a_full_model(full_service, tmp_path):
    lora_gateway = make_service(tmp_path)
    lora_task = asyncio.create_task(lora_gateway.run())
    try:
        lora_model = await created_model(lora_gateway)
        save = lora_gateway.submit("tenant", "save_state", _save_payload(lora_model, 1, "ckpt"))
        path = (await await_settled(lora_gateway, "tenant", save)).result["path"]
    finally:
        lora_task.cancel()
        with suppress(asyncio.CancelledError):
            await lora_task

    model_id = await created_model(full_service)
    load = full_service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 1, "path": path, "optimizer": True}
    )
    future = await await_settled(full_service, "tenant", load)
    assert future.state == FAILED and "train_mode" in future.error


async def test_optim_step_in_full_mode(full_service):
    model_id = await created_model(full_service)
    full_service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    optim = full_service.submit("tenant", "optim_step", {"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)})
    assert (await await_settled(full_service, "tenant", optim)).result["metrics"] == {"grad_norm": 0.5}
