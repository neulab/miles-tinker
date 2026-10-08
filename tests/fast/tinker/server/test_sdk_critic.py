"""Tinker 0.16.1 exercises SkyRL's critic creation contract over HTTP.

Set TINKER016_PYTHON to an interpreter installed from
examples/tinker/requirements-ppo-client.txt. The server keeps its newer SDK
for protobuf schemas; the isolated client imports no Miles or SkyRL modules.
"""

import os
import subprocess

import pytest
from tests.fast.tinker.harness import FakeBackend, make_config
from tests.fast.tinker.server.test_sdk_flow import BASE_MODEL, _Gateway

from miles.tinker.core.service import TinkerService


class CriticBackend(FakeBackend):
    async def forward_only(self, unit_id, slot_rows, loss_fn, loss_fn_config):
        outputs = await super().forward_only(unit_id, slot_rows, loss_fn, loss_fn_config)
        return self._values(slot_rows, outputs)

    async def forward_backward(self, unit_id, slot_rows, loss_fn, loss_fn_config):
        outputs = await super().forward_backward(unit_id, slot_rows, loss_fn, loss_fn_config)
        return self._values(slot_rows, outputs)

    @staticmethod
    def _values(slot_rows, outputs):
        if slot_rows[0][0] == 1:
            for output in outputs:
                output["values"] = output.pop("logprobs")
        return outputs


CLIENT_SETUP = """
import sys
import tinker

assert "miles" not in sys.modules
assert tinker.__version__ == "0.16.1"

server_url = sys.argv[1]
base_model = sys.argv[2]
mode = sys.argv[3]
"""

CLIENT_CREATION = """
import asyncio
import tinker
from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType

service = tinker.ServiceClient(base_url=server_url)
policy = service.create_lora_training_client(base_model, rank=0)
holder = service.holder
model_seq_id = holder.get_training_client_id()

async def create_critic_model():
    request = types.CreateModelRequest(
        session_id=holder.get_session_id(),
        model_seq_id=model_seq_id,
        base_model=base_model,
        lora_config=types.LoraConfig(
            rank=0, seed=0, train_mlp=True, train_attn=True, train_unembed=True,
        ),
    )
    with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
        future = await client.models.create(request=request, extra_body={"model_role": "critic"})
    while True:
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            result = await client.futures.retrieve(
                request=types.FutureRetrieveRequest(request_id=future.request_id)
            )
        if getattr(result, "type", None) == "try_again":
            await asyncio.sleep(0.2)
            continue
        if isinstance(result, types.RequestFailedResponse):
            raise RuntimeError(result.error)
        if not isinstance(result, types.CreateModelResponse):
            raise TypeError(f"Unexpected create response: {type(result)!r}")
        return result.model_id

model_id = holder.run_coroutine_threadsafe(create_critic_model()).result()
critic = tinker.TrainingClient(holder, model_seq_id=model_seq_id, model_id=model_id)
"""

CLIENT_SCRIPT = """
assert type(policy) is tinker.TrainingClient
assert type(critic) is tinker.TrainingClient

def datum(backward=False, policy=False):
    inputs = {"target_tokens": [20, 30, 40, 50], "weights": [0.0, 0.0, 0.0, 0.5]}
    if backward:
        inputs.update(
            {"logprobs": [0.0, 0.0, -1.0, -1.0], "advantages": [0.0, 0.0, 0.0, 1.0]} if policy
            else {"values": [0.0, 0.0, 0.1, 0.2], "returns": [0.0, 0.0, 1.0, 2.0]}
        )
    return types.Datum(
        model_input=types.ModelInput.from_ints([10, 20, 30, 40]),
        loss_fn_inputs={
            k: types.TensorData(data=v, dtype="int64" if k == "target_tokens" else "float32", shape=[4])
            for k, v in inputs.items()
        },
    )

def check_values(result):
    assert list(result.loss_fn_outputs[0]["values"].data) == [0.0, 0.0, 0.0, 0.0]
    assert "logprobs" not in result.loss_fn_outputs[0]

if mode == "sync":
    check_values(critic.forward([datum()], "ppo").result())
    # The SDK splits at 1024 datums. Exercise real chunking without
    # replacing SDK internals, and let the server split those requests again.
    result = critic.forward_backward([datum(True)] * 1025, "ppo", {"value_clip": 0.3}).result()
    check_values(result)
    assert len(result.loss_fn_outputs) == 1025
    assert result.metrics["loss:sum"] == 1025
    critic.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    path = critic.save_state("critic").result().path
    critic.load_state_with_optimizer(path).result()
    check_values(critic.forward([datum()], "ppo").result())
    result = policy.forward_backward([datum(True, True)], "ppo").result()
    assert "logprobs" in result.loss_fn_outputs[0] and "values" not in result.loss_fn_outputs[0]
    policy.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    sampler = policy.save_weights_and_get_sampling_client()
    sample = sampler.sample(types.ModelInput.from_ints([10]), 1, types.SamplingParams(max_tokens=2)).result()
    assert len(sample.sequences) == 1
else:
    async def run():
        forward = await critic.forward_async([datum()], "ppo")
        check_values(await forward.result_async())
        backward = await critic.forward_backward_async([datum(True)], "ppo", {"value_clip": 0.3})
        step = await critic.optim_step_async(types.AdamParams(learning_rate=1e-4))
        check_values(await backward.result_async())
        await step.result_async()
        saved = await critic.save_state_async("critic")
        path = (await saved.result_async()).path
        loaded = await critic.load_state_with_optimizer_async(path)
        await loaded.result_async()
        # Both roles use independent SDK-owned command sequences.
        forward = await policy.forward_async([datum()], "cross_entropy")
        output = (await forward.result_async()).loss_fn_outputs[0]
        assert "logprobs" in output and "values" not in output
    asyncio.run(run())
assert not any(name.split(".")[0] in {"miles", "skyrl", "ppo_sdk"} for name in sys.modules)
print("SDK critic contract passed", tinker.__version__, mode)
"""


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_sdk_critic_contract(tmp_path, mode):
    sdk_python = os.environ.get("TINKER016_PYTHON")
    if sdk_python is None:
        pytest.skip("set TINKER016_PYTHON to exercise the pinned PPO client")
    script = "\n".join((CLIENT_SETUP, CLIENT_CREATION, CLIENT_SCRIPT))
    backend = CriticBackend(full=True)
    config = make_config(
        base_model=BASE_MODEL,
        train_mode="full",
        enable_critic=True,
        checkpoint_root=str(tmp_path),
        coalesce_wait_s=0,
        retrieve_poll_timeout_s=0.1,
        unit_token_budget=1000,
    )
    gateway = _Gateway(lambda: TinkerService(backend, config)).start()
    try:
        result = subprocess.run(
            [sdk_python, "-I", "-c", script, gateway.url, BASE_MODEL, mode],
            cwd=tmp_path,
            env={**os.environ, "TINKER_API_KEY": "tml-sdk-critic-test"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        calls = backend.named("forward_backward")
        critic = next(call for call in calls if call["slot_rows"][0][0] == 1)
        assert critic["loss_fn"] == "ppo"
        assert critic["loss_fn_config"] == {"value_clip": 0.3}
        row = critic["slot_rows"][0][1]
        assert row["tokens"] == [10, 20, 30, 40, 50]
        assert row["target_len"] == 4
        assert row["values"] == pytest.approx([0.0, 0.0, 0.1, 0.2])
        assert row["returns"] == pytest.approx([0.0, 0.0, 1.0, 2.0])
        assert "old_values" not in row and "loss_mask" not in row
        assert row["weights"] == [0.0, 0.0, 0.0, 0.5]
        assert all(
            row["weights"] == [0.0, 0.0, 0.0, 0.5] for call in calls for slot, row in call["slot_rows"] if slot == 1
        )
        assert backend.named("load_slot")[-1]["load_optimizer"] is True
        assert all(len({slot for slot, _ in c["slot_rows"]}) == 1 for c in calls)
    finally:
        gateway.stop()
