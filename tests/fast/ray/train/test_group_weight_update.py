"""A weight push or adapter unload hands the inference controller back however it ends: the lock released, engine
health checks running again as before the update, and no engine marked weights-ready after a failure. A full-weight
push that hit a connection reset killed the trainer cell and kept the controller's lock: its engine-cell tick then
waited on that lock for an hour, with engine health checks paused all along.

Real Ray cells (the dummy trainer actors of this directory) and the real InferenceController lock. The controller
module imports sglang, which is why these tests do not live in test_group.py."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles
from tests.fast.ray.train.test_group import _cell, _make_alive_controller

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.train.group import TrainerController
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker

pytestmark = pytest.mark.asyncio


def _make_inference_controller() -> InferenceController:
    """The real controller and lock with no engine servers: a weight push sees an empty updatable engine set."""
    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(debug_train_only=False, colocate=False)
    controller.servers = {}
    controller.context_lock = ContextLock("InferenceController")
    controller._health_checker_activeness = ActivenessTracker(active=True)
    controller._health_active_before_update = False
    return controller


def _caused_by_the_injected_failure(error: BaseException | None) -> bool:
    while error is not None:
        if "Injected failure" in str(error):
            return True
        error = error.__cause__
    return False


async def _push_weights(group: TrainerController, command: str) -> object:
    if command == "push_slot":
        return await group.push_slot(slot=0, lora_name="model@1", rank=8, alpha=16.0)
    if command == "unload_adapter":
        return await group.unload_adapter(lora_name="model@1")
    return await group.update_weights(rollout_id=0)


class TestAWeightPushHandsTheInferenceControllerBack:
    @pytest.mark.parametrize("command", ["push_slot", "unload_adapter", "update_weights"])
    async def test_a_failed_push_releases_the_lock_and_resumes_the_health_checks_it_paused(self, command):
        inference = _make_inference_controller()
        group = await _make_alive_controller(num_cells=1, inference_controller=inference)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote([command]))

        # the cell's own error, or the controller's "all cells failed" with it as the cause (update_weights)
        with pytest.raises(Exception) as raised:  # noqa: B017
            await _push_weights(group, command)

        assert _caused_by_the_injected_failure(raised.value)
        assert not inference.context_lock.locked
        assert inference._health_checker_activeness.get().active
        await asyncio.wait_for(inference._tick_cells(), timeout=10)
        assert not group.has_alive_cell(), "the failed command killed the only cell"

    @pytest.mark.parametrize("command", ["push_slot", "unload_adapter", "update_weights"])
    async def test_a_push_that_succeeds_still_ends_the_update(self, command):
        inference = _make_inference_controller()
        group = await _make_alive_controller(num_cells=1, inference_controller=inference)

        await _push_weights(group, command)

        assert not inference.context_lock.locked
        assert inference._health_checker_activeness.get().active
        assert group.has_alive_cell()

    async def test_a_failed_push_marks_no_engine_weights_ready(self, monkeypatch):
        """An engine left mid-update holds part of the new weights: serving it would sample from a mix."""
        inference = _make_inference_controller()
        group = await _make_alive_controller(num_cells=1, inference_controller=inference)
        ended: list[dict] = []

        async def _end_update_weights(*, snapshot_cell_id_to_hashes):
            ended.append(snapshot_cell_id_to_hashes)

        monkeypatch.setattr(inference, "end_update_weights", _end_update_weights)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["push_slot"]))

        with pytest.raises(Exception, match="Injected failure"):
            await _push_weights(group, "push_slot")

        assert ended == []
        assert not inference.context_lock.locked


class TestACancelledPushHandsTheInferenceControllerBack:
    """Cancellation is a BaseException: cell.execute neither catches it nor kills the cell, so only the update frame
    hands the controller back. The real lock checks its holder through a context variable that start detached and
    the abort must reattach in the same task."""

    @pytest.mark.parametrize("command", ["push_slot", "unload_adapter", "update_weights"])
    async def test_a_push_cancelled_mid_command_releases_the_lock_and_the_next_push_runs(self, command, monkeypatch):
        inference = _make_inference_controller()
        group = await _make_alive_controller(num_cells=1, inference_controller=inference)
        cell = _cell(group, 0)
        real_execute = cell.execute
        started = asyncio.Event()

        async def _hang(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(cell, "execute", _hang)
        push = asyncio.create_task(_push_weights(group, command))
        await asyncio.wait_for(started.wait(), timeout=10)
        assert inference.context_lock.locked and not inference._health_checker_activeness.get().active
        push.cancel()
        with pytest.raises(asyncio.CancelledError):
            await push

        assert not inference.context_lock.locked
        assert inference._health_checker_activeness.get().active
        assert group.has_alive_cell(), "a cancellation kills no cell"
        await asyncio.wait_for(inference._tick_cells(), timeout=10)
        monkeypatch.setattr(cell, "execute", real_execute)
        await asyncio.wait_for(_push_weights(group, command), timeout=60)
        assert not inference.context_lock.locked

    async def test_a_push_cancelled_while_the_engines_start_releases_the_lock(self):
        """start_update_weights waits for engine cells with the lock released between polls; a cancellation there
        re-takes the lock, puts the health checks back and releases it, and no update frame was ever opened."""
        inference = _make_inference_controller()
        starting_engine = SimpleNamespace(
            is_pending_weights_or_serving=False, is_uninitialized=False, tick=AsyncMock()
        )
        inference.servers = {"model": SimpleNamespace(server_cells={"engine-0": starting_engine}, update_weights=True)}
        group = await _make_alive_controller(num_cells=1, inference_controller=inference)

        push = asyncio.create_task(_push_weights(group, "push_slot"))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if not inference._health_checker_activeness.get().active and not inference.context_lock.locked:
                break  # paused and polling with the lock released
        else:
            pytest.fail("the push never started waiting for the engine")
        push.cancel()
        with pytest.raises(asyncio.CancelledError):
            await push

        assert not inference.context_lock.locked
        assert inference._health_checker_activeness.get().active
        await asyncio.wait_for(inference._tick_cells(), timeout=10)
