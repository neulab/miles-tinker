"""Shared trainer GPUs may switch only after gradients are consumed or discarded."""

import asyncio

import pytest
from tests.fast.tinker.harness import ADAM, FakeBackend, row

from miles.tinker.core.types import UserInputError
from miles.tinker.full_backend import FullModelBackend


class ResidentBackend(FakeBackend):
    def __init__(self, slot, resident, events):
        super().__init__(full=True)
        self.slot, self.resident, self.events = slot, resident, events

    def _record(self, name, **kwargs):
        if name not in {"sample", "onload"}:
            assert self.resident == {self.slot}, f"{name} on sleeping trainer {self.slot}"
        self.events.append((self.slot, name))
        super()._record(name, **kwargs)

    async def onload(self):
        assert not self.resident, "offload the previous trainer before loading the next"
        self._record("onload")
        self.resident.add(self.slot)

    async def offload(self):
        self._record("offload")
        self.resident.remove(self.slot)


@pytest.fixture
def rig():
    resident, events = set(), []
    policy, critic = (ResidentBackend(slot, resident, events) for slot in (0, 1))
    return FullModelBackend(policy, critic, offload=True), policy, critic, resident, events


async def test_shared_gpu_lifecycle_and_publication(rig):
    backend, policy, critic, resident, events = rig
    await backend.warmup()
    await backend.load_slot(1, 0, 0.0)
    await backend.forward_only(1, [(1, row())], "ppo", {})
    await backend.forward_backward(2, [(1, row())], "ppo", {})
    await backend.optim_step({1: ADAM})
    await backend.save_slot(1, "critic-checkpoint")
    await backend.load_slot(1, 0, 0.0, "critic-checkpoint", True)
    await backend.forward_backward(3, [(0, row())], "ppo", {})
    await backend.optim_step({0: ADAM})
    await backend.push_slot(0, "policy@1", 0, 0.0)
    assert events == [
        (0, "onload"),
        (0, "warmup"),
        (0, "offload"),
        (1, "onload"),
        (1, "load_slot"),
        (1, "forward_only"),
        (1, "forward_backward"),
        (1, "optim_step"),
        (1, "save_slot"),
        (1, "load_slot"),
        (1, "offload"),
        (0, "onload"),
        (0, "forward_backward"),
        (0, "optim_step"),
        (0, "push_slot"),
    ]
    assert resident == {0}
    assert critic.named("load_slot")[-1]["load_optimizer"] is True


async def test_switch_rejects_pending_gradients_without_changing_residency(rig):
    backend, policy, critic, resident, events = rig
    await backend.forward_backward(1, [(1, row())], "ppo", {})
    before = list(events)
    with pytest.raises(UserInputError, match="complete optim_step for slot 1"):
        await backend.forward_backward(2, [(0, row())], "ppo", {})
    # The failed policy request never ran; discard must not evict the critic.
    await backend.discard_grads(0)
    assert events == before and resident == {1}
    await backend.optim_step({1: ADAM})
    await backend.forward_backward(3, [(0, row())], "ppo", {})
    assert resident == {0}


@pytest.mark.parametrize("slot", [0, 1])
async def test_checkpoint_restore_activates_the_saved_role_and_allows_next_update(rig, slot):
    backend, policy, critic, resident, events = rig
    other = 1 - slot
    await backend.forward_backward(1, [(slot, row())], "ppo", {})
    await backend.optim_step({slot: ADAM})
    await backend.save_slot(slot, "saved")
    await backend.forward_only(2, [(other, row())], "ppo", {})
    assert resident == {other}

    await backend.load_slot(slot, 0, 0.0, "saved", True)
    assert events[-3:] == [(other, "offload"), (slot, "onload"), (slot, "load_slot")]
    await backend.forward_backward(3, [(slot, row())], "ppo", {})
    await backend.optim_step({slot: ADAM})
    await backend.forward_only(4, [(other, row())], "ppo", {})
    assert resident == {other}, "restored model's completed update must not leave a stale pending marker"


async def test_failed_backward_keeps_trainer_resident_until_discard(rig):
    backend, policy, critic, resident, events = rig
    await backend.load_slot(1, 0, 0.0)
    critic.fail_next = RuntimeError("backward failed")
    with pytest.raises(RuntimeError, match="backward failed"):
        await backend.forward_backward(1, [(1, row())], "ppo", {})
    with pytest.raises(UserInputError, match="complete optim_step"):
        await backend.forward_only(2, [(0, row())], "ppo", {})
    await backend.discard_grads(1)
    await backend.forward_only(3, [(0, row())], "ppo", {})
    assert (1, "discard_grads") in events and resident == {0}


async def test_merged_optimizer_barrier_consumes_resident_gradients_first(rig):
    backend, policy, critic, resident, events = rig
    await backend.forward_backward(1, [(1, row())], "ppo", {})
    result = await backend.optim_step({0: ADAM, 1: ADAM})
    assert set(result) == {0, 1}
    assert events.index((1, "optim_step")) < events.index((1, "offload")) < events.index((0, "optim_step"))


async def test_unload_discards_active_gradients_without_waking_sleeping_model(rig):
    backend, policy, critic, resident, events = rig
    await backend.forward_backward(1, [(1, row())], "ppo", {})
    before = list(events)
    await backend.unload_slot(0)
    assert events == before
    await backend.unload_slot(1)
    assert events[-2:] == [(1, "unload_slot"), (1, "offload")]
    assert not resident
    await backend.load_slot(0, 0, 0.0)
    assert resident == {0}


async def test_failed_offload_does_not_activate_the_other_trainer(rig):
    backend, policy, critic, resident, events = rig
    await backend.load_slot(1, 0, 0.0)
    critic.fail_next = RuntimeError("offload failed")
    with pytest.raises(RuntimeError, match="offload failed"):
        await backend.forward_only(1, [(0, row())], "ppo", {})
    assert resident == {1} and not policy.calls
    await backend.forward_only(2, [(0, row())], "ppo", {})
    assert resident == {0}


async def test_failed_onload_does_not_mark_trainer_as_resident(rig):
    backend, policy, critic, resident, events = rig
    critic.fail_next = RuntimeError("onload failed")
    with pytest.raises(RuntimeError, match="onload failed"):
        await backend.forward_only(1, [(1, row())], "ppo", {})
    assert not resident
    await backend.forward_only(2, [(1, row())], "ppo", {})
    assert events == [(1, "onload"), (1, "onload"), (1, "forward_only")]


async def test_sampling_overlaps_training_and_residency_transitions(rig, monkeypatch):
    backend, policy, critic, resident, events = rig
    sampling_started, finish_sampling = asyncio.Event(), asyncio.Event()

    async def sample(payload, lora_name):
        sampling_started.set()
        await finish_sampling.wait()
        return {"sequences": []}

    monkeypatch.setattr(policy, "sample", sample)
    sampling = asyncio.create_task(backend.sample({}, None))
    try:
        await asyncio.wait_for(sampling_started.wait(), timeout=1)
        for slot in (1, 0, 1):
            await asyncio.wait_for(backend.forward_backward(slot, [(slot, row())], "ppo", {}), timeout=1)
            await asyncio.wait_for(backend.optim_step({slot: ADAM}), timeout=1)
        assert not sampling.done()
    finally:
        finish_sampling.set()
        await sampling

    # Sampling admission also stays open during an in-flight GPU transition.
    unloading, finish_unload = asyncio.Event(), asyncio.Event()
    real_offload = critic.offload

    async def offload():
        unloading.set()
        await finish_unload.wait()
        await real_offload()

    monkeypatch.setattr(critic, "offload", offload)
    transition = asyncio.create_task(backend.forward_only(4, [(0, row())], "ppo", {}))
    try:
        await asyncio.wait_for(unloading.wait(), timeout=1)
        await asyncio.wait_for(backend.sample({}, None), timeout=1)
        assert not transition.done()
    finally:
        finish_unload.set()
        await transition
