"""The unmodified tinker SDK drives the gateway over a real uvicorn socket with
the FakeBackend behind it: bootstrap, proto forward_backward with the request
metrics, optim_step, named and unnamed sampler saves, sampling, and
save_state -> weights_info -> resume."""

import asyncio
import os
import socket
import tempfile
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
            self.service = make_service(
                tempfile.mkdtemp(prefix="tinker-sdk-flow-"), base_model=BASE_MODEL, n_slots=2, trains_unembed=True
            )
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


def _datum(n: int, loss_fn: str = "importance_sampling") -> types.Datum:
    """Exactly the inputs the loss reads: the gateway rejects any other loss_fn_inputs key."""
    tokens = [10, 20, 30, 40, 50, 60]
    if loss_fn == "cross_entropy":
        inputs = {"weights": [1.0] * n}
    else:
        inputs = {"logprobs": [-0.5] * n, "advantages": [0.1] * n}
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:n]),
        loss_fn_inputs={"target_tokens": tokens[1 : n + 1], **inputs},
    )


def test_the_sdk_training_loop(gateway, sdk):
    assert sdk.get_server_capabilities().supported_models[0].model_name == BASE_MODEL
    training = sdk.create_lora_training_client(base_model=BASE_MODEL, rank=8, user_metadata={"renderer_name": "qwen3"})
    info = training.get_info()
    assert (info.is_lora, info.lora_rank, info.model_data.model_name) == (True, 8, BASE_MODEL)

    output = training.forward_backward([_datum(5), _datum(4), _datum(3)], loss_fn="importance_sampling").result()
    assert [len(record["logprobs"].data) for record in output.loss_fn_outputs] == [5, 4, 3]
    assert output.metrics["num_tokens:sum"] == 12.0 and output.metrics["num_datums:sum"] == 3.0

    forward = training.forward([_datum(2, "cross_entropy")], loss_fn="cross_entropy").result()
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

    assert gateway.service.backend.named("export_slot"), "sampler saves export the adapter for the engines"
