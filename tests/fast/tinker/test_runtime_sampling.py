"""The sampling half of runtime.py: neutral sampling payloads -> sglang
/generate requests with Tinker semantics (stop handling, seed offsets,
context clamping, abort as a failure)."""

from argparse import Namespace

import aiohttp
import pytest
from aiohttp import web

from miles.tinker.core.types import UserInputError
from miles.tinker.runtime import (
    GENERATE_MAX_RETRIES,
    MilesBackend,
    _clamp_max_new_tokens,
    _per_sample_requests,
    _to_sequence,
)


def _request(params: dict, args: Namespace | None = None, **payload_extra) -> dict:
    payload = {
        "prompt_tokens": [1, 2],
        "sampling_params": params,
        "num_samples": 1,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
        **payload_extra,
    }
    return MilesBackend(_args(args), None, "http://router")._generate_request(payload, lora_name="m@1")


def _args(overrides: Namespace | None) -> Namespace:
    """The topology the backend derives its DP size from, plus any test overrides."""
    base = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tinker_train_mode="lora",
        tinker_checkpoint_root=None,
    )
    return Namespace(**{**base, **(vars(overrides) if overrides is not None else {})})


def test_stop_tokens_stay_in_the_output():
    assert _request({"max_tokens": 4})["sampling_params"]["no_stop_trim"] is True


def test_an_empty_stop_list_disables_eos():
    params = _request({"max_tokens": 4, "stop": []})["sampling_params"]
    assert params["ignore_eos"] is True and "stop" not in params


def test_a_none_stop_keeps_the_engine_default():
    params = _request({"max_tokens": 4, "stop": None})["sampling_params"]
    assert "ignore_eos" not in params and "stop" not in params


def test_max_new_tokens_is_clamped_only_with_an_explicit_context():
    assert _request({"max_tokens": 100})["sampling_params"]["max_new_tokens"] == 100
    clamped = _request({"max_tokens": 100}, args=Namespace(rollout_max_context_len=10))
    assert clamped["sampling_params"]["max_new_tokens"] == 8
    with pytest.raises(UserInputError, match="no room"):
        _clamp_max_new_tokens(4, prompt_len=10, context_len=10)


def test_seeds_are_offset_per_sample():
    request = {"input_ids": [1], "sampling_params": {"sampling_seed": 42, "max_new_tokens": 1}}
    seeds = [r["sampling_params"]["sampling_seed"] for r in _per_sample_requests(request, 3)]
    assert seeds == [42, 43, 44]
    assert _per_sample_requests(request, 1)[0]["sampling_params"]["sampling_seed"] == 42
    assert (
        "sampling_seed" not in _per_sample_requests({"input_ids": [1], "sampling_params": {}}, 2)[1]["sampling_params"]
    )


def test_an_aborted_generation_is_a_failure_not_a_stop():
    response = {"meta_info": {"output_token_logprobs": [(-0.1, 11)], "finish_reason": {"type": "abort"}}}
    with pytest.raises(RuntimeError, match="aborted"):
        _to_sequence(response)


async def test_sample_posts_once_per_sample(monkeypatch):
    posted = []

    async def fake_post(session, url, payload, max_retries=GENERATE_MAX_RETRIES):
        posted.append((url, payload, max_retries))
        return {"meta_info": {"output_token_logprobs": [(-0.1, 5)], "finish_reason": {"type": "stop"}}}

    monkeypatch.setattr("miles.tinker.runtime.post_generate", fake_post)
    backend = MilesBackend(_args(None), None, "http://router")
    payload = {
        "prompt_tokens": [1],
        "sampling_params": {"max_tokens": 2, "seed": 7},
        "num_samples": 2,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }
    result = await backend.sample(payload, lora_name=None)
    await backend._generate_client.close()
    assert len(result["sequences"]) == 2
    assert [entry[0] for entry in posted] == ["http://router/generate"] * 2
    assert [entry[1]["sampling_params"]["sampling_seed"] for entry in posted] == [7, 8]


@pytest.fixture
async def flaky_router():
    """A /generate that answers 503 to its first ``failures`` calls, then a one-token sequence."""
    state = {"failures": 1, "calls": 0}

    async def generate(request):
        state["calls"] += 1
        if state["calls"] <= state["failures"]:
            return web.Response(status=503, text="engine busy")
        return web.json_response(
            {"meta_info": {"output_token_logprobs": [(-0.1, 5)], "finish_reason": {"type": "stop"}}}
        )

    app = web.Application()
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield state, f"http://127.0.0.1:{port}"
    await runner.cleanup()


async def test_generate_posts_retry_a_bounded_number_of_times(monkeypatch, flaky_router):
    monkeypatch.setattr("miles.tinker.runtime.GENERATE_RETRY_PAUSE_S", 0.0)
    state, router_url = flaky_router
    backend = MilesBackend(_args(None), None, router_url)
    payload = {
        "prompt_tokens": [1],
        "sampling_params": {"max_tokens": 2},
        "num_samples": 1,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }
    result = await backend.sample(payload, lora_name=None)
    assert result["sequences"][0]["tokens"] == [5] and state["calls"] == 2, "one 503, one retry"

    state["failures"], state["calls"] = 10, 0
    with pytest.raises(aiohttp.ClientResponseError):
        await backend.sample(payload, lora_name=None)
    assert state["calls"] == GENERATE_MAX_RETRIES, "a persistent failure gives up after the bounded retries"
    await backend._generate_client.close()
