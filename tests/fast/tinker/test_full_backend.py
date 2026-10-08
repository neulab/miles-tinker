"""Independent model routing must preserve sampling/training overlap."""

import asyncio

import pytest
from tests.fast.tinker.harness import ADAM, FakeBackend, row

from miles.tinker.core.types import UserInputError
from miles.tinker.full_backend import FullModelBackend


@pytest.mark.parametrize("slot", [0, 1])
async def test_model_lifecycle_and_forward_use_the_selected_trainers_local_slot(slot):
    policy, critic = FakeBackend(full=True), FakeBackend(full=True)
    backend = FullModelBackend(policy, critic)
    selected, other = (policy, critic) if slot == 0 else (critic, policy)
    datum = row()

    await backend.load_slot(slot, 0, 0.0)
    await backend.forward_only(7, [(slot, datum)], "ppo", {})
    await backend.save_slot(slot, "checkpoint")
    await backend.load_slot(slot, 0, 0.0, "checkpoint", False)
    await backend.discard_grads(slot)
    await backend.unload_slot(slot)

    assert other.calls == []
    assert selected.named("load_slot") == [
        dict(slot=0, rank=0, alpha=0.0, ckpt_path=None, load_optimizer=True),
        dict(slot=0, rank=0, alpha=0.0, ckpt_path="checkpoint", load_optimizer=False),
    ]
    assert selected.named("forward_only") == [
        dict(unit_id=7, slot_rows=[(0, datum)], loss_fn="ppo", loss_fn_config={})
    ]
    assert selected.named("save_slot") == [dict(slot=0, path="checkpoint")]
    assert selected.named("discard_grads") == selected.named("unload_slot") == [dict(slot=0)]


async def test_policy_training_and_publication_remain_on_the_policy():
    policy, critic = FakeBackend(full=True), FakeBackend(full=True)
    backend = FullModelBackend(policy, critic)
    await backend.warmup()
    await backend.forward_backward(1, [(0, row())], "importance_sampling", {})
    metrics = await backend.optim_step({0: ADAM})
    assert metrics[0]["lr"] == ADAM["learning_rate"]
    assert await backend.push_slot(0, "policy@1", 0, 0.0) == 1
    assert policy.named("forward_backward")[0]["loss_fn"] == "importance_sampling"
    assert critic.calls == []
    with pytest.raises(UserInputError, match="only the policy"):
        await backend.push_slot(1, "critic@1", 0, 0.0)


async def test_critic_training_and_optimizer_calls_use_independent_trainers():
    policy, critic = FakeBackend(full=True), FakeBackend(full=True)
    backend = FullModelBackend(policy, critic)
    datum = row()
    await backend.forward_backward(1, [(1, datum)], "ppo", {"value_clip": 0.3})
    assert policy.calls == []
    assert critic.named("forward_backward") == [
        dict(unit_id=1, slot_rows=[(0, datum)], loss_fn="ppo", loss_fn_config={"value_clip": 0.3})
    ]
    critic_adam = {**ADAM, "learning_rate": 0.01}
    results = await backend.optim_step({0: ADAM, 1: critic_adam})
    assert results[0]["lr"] == ADAM["learning_rate"]
    assert results[1]["lr"] == critic_adam["learning_rate"]
    assert policy.named("optim_step") == [dict(adam_params_by_slot={0: ADAM})]
    assert critic.named("optim_step") == [dict(adam_params_by_slot={0: critic_adam})]


async def test_mixed_role_work_units_are_rejected():
    policy, critic = FakeBackend(full=True), FakeBackend(full=True)
    backend = FullModelBackend(policy, critic)
    with pytest.raises(UserInputError, match="exactly one model role"):
        await backend.forward_only(1, [(0, row()), (1, row())], "ppo", {})
    assert policy.calls == critic.calls == []


async def test_training_does_not_drain_sampling(monkeypatch):
    policy, critic = FakeBackend(full=True), FakeBackend(full=True)
    backend = FullModelBackend(policy, critic)
    sampling_started, finish_sampling = asyncio.Event(), asyncio.Event()

    async def sample(payload, lora_name):
        sampling_started.set()
        await finish_sampling.wait()
        return {"sequences": []}

    monkeypatch.setattr(policy, "sample", sample)
    sampling = asyncio.create_task(backend.sample({}, None))
    try:
        await asyncio.wait_for(sampling_started.wait(), timeout=1)
        await asyncio.wait_for(backend.forward_only(1, [(1, row())], "ppo", {}), timeout=1)
        await asyncio.wait_for(backend.forward_backward(2, [(0, row())], "ppo", {}), timeout=1)
        await asyncio.wait_for(backend.forward_backward(3, [(1, row())], "ppo", {}), timeout=1)
        await asyncio.wait_for(backend.optim_step({0: ADAM, 1: ADAM}), timeout=1)
        assert not sampling.done()
    finally:
        finish_sampling.set()
        await sampling
