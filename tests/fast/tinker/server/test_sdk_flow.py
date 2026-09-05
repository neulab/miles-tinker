"""The unmodified tinker SDK (0.27.1, the cookbook's) drives the gateway over a
real uvicorn socket with the FakeBackend behind it: bootstrap, proto
forward_backward, optim_step, named and unnamed sampler saves, sampling,
save_state -> weights_info -> resume, training_runs, unload."""

import asyncio
import os
import socket
import threading
import time

import httpx
import pytest
from tests.fast.tinker.harness import make_service

tinker = pytest.importorskip("tinker")
uvicorn = pytest.importorskip("uvicorn")

from miles.tinker.server.app import build_app  # noqa: E402
from tinker import types  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class _Gateway:
    """The app + service loop in a daemon thread; the sync SDK runs in the test thread."""

    def __init__(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.service = None
        self._server = None
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        async def main():
            self.service = make_service(base_model=BASE_MODEL, n_slots=2, retrieve_poll_timeout_s=1.0)
            run_task = asyncio.create_task(self.service.run())
            self._server = uvicorn.Server(
                uvicorn.Config(build_app(self.service), host="127.0.0.1", port=self.port, log_level="warning")
            )
            await self._server.serve()
            run_task.cancel()

        asyncio.run(main())

    def start(self) -> "_Gateway":
        self._thread.start()
        for _ in range(200):
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/api/v1/healthz").status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.05)
        raise RuntimeError("gateway did not come up")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture(scope="module")
def gateway():
    gw = _Gateway().start()
    yield gw
    gw.stop()


@pytest.fixture(scope="module")
def sdk(gateway):
    os.environ.setdefault("TINKER_API_KEY", "tml-fast-suite")
    return tinker.ServiceClient(base_url=gateway.url)


def _datum(n: int) -> types.Datum:
    tokens = [10, 20, 30, 40, 50, 60]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:n]),
        loss_fn_inputs={
            "target_tokens": tokens[1 : n + 1],
            "weights": [1.0] * n,
            "logprobs": [-0.5] * n,
            "advantages": [0.1] * n,
        },
    )


def test_the_sdk_training_loop(gateway, sdk):
    assert sdk.get_server_capabilities().supported_models[0].model_name == BASE_MODEL
    training = sdk.create_lora_training_client(base_model=BASE_MODEL, rank=8, user_metadata={"renderer_name": "qwen3"})
    info = training.get_info()
    assert (info.is_lora, info.lora_rank, info.model_data.model_name) == (True, 8, BASE_MODEL)

    output = training.forward_backward([_datum(5), _datum(4), _datum(3)], loss_fn="importance_sampling").result()
    assert [len(record["logprobs"].data) for record in output.loss_fn_outputs] == [5, 4, 3]
    assert output.metrics["num_tokens:sum"] == 12.0 and output.metrics["num_datums:sum"] == 3.0

    forward = training.forward([_datum(2)], loss_fn="cross_entropy").result()
    assert len(forward.loss_fn_outputs) == 1

    optim = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    assert optim.metrics["grad_norm"] > 0

    named = training.save_weights_for_sampler(name="000001").result()
    assert named.path.endswith("/sampler_weights/000001")
    sampler = sdk.create_sampling_client(model_path=named.path)
    response = sampler.sample(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=2,
        sampling_params=types.SamplingParams(max_tokens=4, temperature=1.0, stop=["\n"]),
    ).result()
    assert [list(sequence.tokens) for sequence in response.sequences] == [[1, 2], [1, 2]]
    assert all(sequence.sequence_id for sequence in response.sequences), "ids come from sample_sequence_ids"

    ephemeral = training.save_weights_and_get_sampling_client()
    assert (
        len(
            ephemeral.sample(types.ModelInput.from_ints([1]), 1, types.SamplingParams(max_tokens=2)).result().sequences
        )
        == 1
    )
    assert ephemeral.get_base_model() == BASE_MODEL

    base = sdk.create_sampling_client(base_model=BASE_MODEL)
    base.sample(types.ModelInput.from_ints([1]), 1, types.SamplingParams(max_tokens=2)).result()

    state = training.save_state(name="ckpt").result()
    resumed = sdk.create_training_client_from_state_with_optimizer(state.path)
    assert resumed.get_info().lora_rank == 8
    assert gateway.service.backend.named("load_slot")[-1]["load_optimizer"] is True
    assert gateway.service.backend.named("load_slot")[-1]["ckpt_path"].endswith("/weights/ckpt")

    rest = sdk.create_rest_client()
    run = rest.get_training_run_by_tinker_path(state.path).result()
    assert (run.is_lora, run.lora_rank, run.user_metadata) == (True, 8, {"renderer_name": "qwen3"})
    rest.delete_checkpoint_from_tinker_path(state.path).result()

    lora_names = [push["lora_name"] for push in gateway.service.backend.named("push_slot")]
    assert len(lora_names) == 2 and all("@" in name for name in lora_names)
    assert not any("warning" in call for call, _ in gateway.service.backend.calls)

    # unload_model has no 0.27.1 SDK wrapper: the REST route frees the slot behind a future
    headers = {"X-API-Key": os.environ["TINKER_API_KEY"]}
    unload = httpx.post(f"{gateway.url}/api/v1/unload_model", json={"model_id": training.model_id}, headers=headers)
    body = unload.json()
    for _ in range(20):
        polled = httpx.post(
            f"{gateway.url}/api/v1/retrieve_future", json={"request_id": body["request_id"]}, headers=headers
        )
        if polled.status_code != 408:
            break
    assert polled.json() == {"type": "unload_model", "model_id": training.model_id}
    assert training.model_id not in gateway.service.models
    assert {u["lora_name"] for u in gateway.service.backend.named("unload_adapter")} == set(lora_names)
