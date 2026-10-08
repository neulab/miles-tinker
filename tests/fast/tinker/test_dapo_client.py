"""CPU checks for the standalone PPO client's math, SDK datums, and phase ordering."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tinker")
pytest.importorskip("pylatexenc")

CLIENT = Path(__file__).resolve().parents[3] / "deploy/clients/dapo_ppo.py"
spec = importlib.util.spec_from_file_location("dapo_ppo_client", CLIENT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def _args(tmp_path, **overrides):
    args = client._parse_args(
        [
            "--base-url",
            "http://unused",
            "--model-path",
            "unused",
            "--data",
            str(tmp_path / "data.jsonl"),
            "--output",
            str(tmp_path),
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_reference_defaults_and_gae_returns_are_unchanged(tmp_path):
    args = _args(tmp_path)
    assert (args.prompts_per_rollout, args.samples_per_prompt, args.batch_size) == (8, 8, 32)
    assert (args.actor_lr, args.critic_lr, args.temperature, args.critic_warmup) == (1e-6, 1e-5, 0.8, 1)
    advantages, returns = client.gae([0.2, 0.3], 1.0)
    np.testing.assert_allclose(advantages, [0.8, 0.7])
    np.testing.assert_allclose(returns, [1.0, 1.0])
    raw_returns = returns.copy()
    normalized = client.whiten([advantages, np.array([-0.1, 0.4, 0.9], dtype=np.float32)])
    flat = np.concatenate(normalized)
    assert flat.mean() == pytest.approx(0, abs=1e-6)
    assert flat.var(ddof=1) == pytest.approx(1, abs=1e-6)
    np.testing.assert_array_equal(returns, raw_returns)
    np.testing.assert_array_equal(client.whiten([np.ones(3, dtype=np.float32)])[0], [0, 0, 0])


def test_sdk_datums_align_mask_and_normalize_the_complete_optimizer_batch():
    trajectory = dict(
        prompt=[10, 20],
        tokens=[30, 40],
        values=[0.2, 0.3],
        returns=[1.0, 1.0],
        logprobs=[-1.0, -2.0],
        advantages=[0.8, 0.7],
    )
    critic = client.datum(trajectory, "critic", 32)
    policy = client.datum(trajectory, "policy", 32)
    assert critic.model_input.to_ints() == [10, 20, 30]
    assert critic.loss_fn_inputs["target_tokens"].data == [20, 30, 40]
    assert critic.loss_fn_inputs["values"].data == pytest.approx([0, 0.2, 0.3])
    assert critic.loss_fn_inputs["weights"].data == [0, 1 / 64, 1 / 64]
    assert set(critic.loss_fn_inputs) == {"target_tokens", "values", "returns", "weights"}
    assert policy.loss_fn_inputs["advantages"].data == pytest.approx([0, 0.8 / 64, 0.7 / 64])
    assert policy.loss_fn_inputs["logprobs"].data == [0, -1, -2]


@pytest.mark.parametrize("async_rollouts", [False, True])
async def test_training_awaits_optimizer_and_sampling_before_role_switch_or_publication(
    tmp_path, monkeypatch, async_rollouts
):
    args = _args(
        tmp_path,
        rollouts=3,
        critic_warmup=1,
        async_rollouts=async_rollouts,
        prompts_per_rollout=1,
        samples_per_prompt=2,
        batch_size=2,
        reference_dp=1,
    )
    events = []
    sampling = set()

    class Model:
        def __init__(self, role):
            self.role = role

        async def save_weights_and_get_sampling_client_async(self):
            assert not sampling, "publication must wait for all lookahead sampling"
            events.append("publish")
            return object()

        async def forward_backward_async(self, data, loss_fn, config):
            events.append(self.role + ":backward_submit")

            async def result_async():
                events.append(self.role + ":backward_done")
                return SimpleNamespace(metrics={"loss:sum": 0.5})

            return SimpleNamespace(result_async=result_async)

        async def optim_step_async(self, params):
            assert events[-1] == self.role + ":backward_done"

            async def result_async():
                await asyncio.sleep(0)
                events.append(self.role + ":step_done")
                return SimpleNamespace(metrics={"skipped": 0})

            return SimpleNamespace(result_async=result_async)

    async def collect(sampler, tokenizer, records, rollout, args):
        sampling.add(rollout)
        await asyncio.sleep(0.01)
        sampling.remove(rollout)
        events.append(f"sample:{rollout}")
        return [
            dict(prompt=[10], tokens=[20], logprobs=[-0.5], reward=reward, stop_reason="stop") for reward in (0, 1)
        ]

    async def evaluate(model, trajectories, role, chunk_size=32):
        # An async old rollout must never be rescored with the updated policy.
        assert role == "critic" or not async_rollouts
        for trajectory in trajectories:
            trajectory["values" if role == "critic" else "logprobs"] = [0.1 if role == "critic" else -0.4]
        await asyncio.sleep(0)

    async def save(*args):
        events.append("save")

    monkeypatch.setattr(client, "collect_rollout", collect)
    monkeypatch.setattr(client, "evaluate", evaluate)
    monkeypatch.setattr(client, "save_pair", save)
    await client.train(args, Model("policy"), Model("critic"), None, [], "hash")
    training = [event for event in events if event.endswith((":backward_done", ":step_done"))]
    critic = ["critic:backward_done", "critic:step_done"]
    actor = ["policy:backward_done", "policy:step_done"]
    assert training == critic + critic + actor + critic + actor
    assert events[-1] == "save"


async def test_failed_second_checkpoint_does_not_replace_resume_manifest(tmp_path, monkeypatch):
    args = _args(tmp_path)
    manifest = tmp_path / "latest.json"
    manifest.write_text('{"previous": true}')

    class Model:
        def __init__(self, fail):
            self.fail = fail

        async def save_state_async(self, name):
            async def result_async():
                if self.fail:
                    raise RuntimeError("checkpoint failed")
                return SimpleNamespace(path="tinker://critic/weights/saved")

            return SimpleNamespace(result_async=result_async)

    async def evaluate(*args):
        pass

    monkeypatch.setattr(client, "evaluate", evaluate)
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        await client.save_pair(Model(True), Model(False), args, 0, dict(prompt=[10], tokens=[20]), "hash", None)
    assert manifest.read_text() == '{"previous": true}'
