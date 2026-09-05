"""Load shape of the sampling path: hundreds of concurrent asample + long-poll
conversations through the real app, service and MilesBackend.sample against a
stub router. The single event loop must not starve: every request sees a
bounded gateway overhead on top of the stub's engine latency."""

import asyncio
import statistics
import time
from argparse import Namespace

import httpx
import pytest
from aiohttp import web
from tests.fast.tinker.harness import make_config

from miles.tinker.core.service import TinkerService
from miles.tinker.runtime import MilesBackend
from miles.tinker.server.app import build_app

ENGINE_DELAY_S = 0.05
CONCURRENCY = 256
HEADERS = {"X-API-Key": "tml-tenant-a"}


@pytest.fixture
async def router_url():
    """/generate answers after ENGINE_DELAY_S with a 12-token sequence."""

    async def generate(request):
        body = await request.json()
        n = body["sampling_params"]["max_new_tokens"]
        await asyncio.sleep(ENGINE_DELAY_S)
        return web.json_response(
            {"meta_info": {"output_token_logprobs": [(-0.1, 5)] * n, "finish_reason": {"type": "length"}}}
        )

    app = web.Application()
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    await runner.cleanup()


class _SamplingBackend(MilesBackend):
    async def warmup(self):
        pass


def _backend(router_url: str) -> MilesBackend:
    args = Namespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tinker_train_mode="lora",
        tinker_checkpoint_root=None,
        rollout_max_context_len=None,
    )
    return _SamplingBackend(args, None, router_url)


async def _sample(http: httpx.AsyncClient, sampling_session_id: str, seq_id: int) -> float:
    started = time.monotonic()
    body = {
        "sampling_session_id": sampling_session_id,
        "seq_id": seq_id,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": list(range(142))}]},
        "num_samples": 1,
        "sampling_params": {"max_tokens": 12},
    }
    request_id = (await http.post("/api/v1/asample", json=body, headers=HEADERS)).json()["request_id"]
    while True:
        response = await http.post("/api/v1/retrieve_future", json={"request_id": request_id}, headers=HEADERS)
        if response.status_code != 408:
            break
    assert response.status_code == 200, response.text
    assert len(response.json()["sequences"][0]["tokens"]) == 12
    return time.monotonic() - started


async def test_hundreds_of_concurrent_samples_see_bounded_gateway_overhead(router_url):
    service = TinkerService(_backend(router_url), make_config(retrieve_poll_timeout_s=5.0))
    run_task = asyncio.create_task(service.run())
    transport = httpx.ASGITransport(app=build_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as http:
        created = await http.post("/api/v1/create_sampling_session", json={"base_model": "base"}, headers=HEADERS)
        sampling_session_id = created.json()["sampling_session_id"]
        started = time.monotonic()
        latencies = await asyncio.gather(*[_sample(http, sampling_session_id, seq) for seq in range(CONCURRENCY)])
        wall = time.monotonic() - started
    await service.backend._generate_client.close()
    for task in (run_task, getattr(service, "_sweep_task", None)):
        if task is not None:
            task.cancel()

    overheads = [latency - ENGINE_DELAY_S for latency in latencies]
    # measured ~0.05 s median / ~0.15 s max on a loaded login node; the httpx-pooled
    # sample path read ~2 s / 16 s here before it was replaced
    assert statistics.median(overheads) < 0.5, f"median gateway overhead {statistics.median(overheads):.2f}s"
    assert max(overheads) < 2.0, f"slowest request waited {max(overheads):.2f}s on the loop"
    assert wall < 3.0, f"{CONCURRENCY} samples took {wall:.2f}s"
