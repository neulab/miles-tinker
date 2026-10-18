"""Slot commands: a LoRA checkpoint save/load fails on every rank together with no collective
pending, so it must not recycle the trainer cell; slot init, the training steps and full
fine-tuning checkpoints keep cell.execute's kill-on-failure default. A weight push or adapter
unload, failed or not, hands the inference controller's update frame back."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.ray.train.group import TrainerController


def _controller(train_mode: str) -> tuple[TrainerController, AsyncMock]:
    controller = TrainerController.__new__(TrainerController)
    controller.args = SimpleNamespace(tinker_train_mode=train_mode)
    execute = AsyncMock(return_value=["per-actor"])
    controller._cells_by_id = {"cell-0": SimpleNamespace(cell_index=0, execute=execute)}
    return controller, execute


def _kill_flags(execute: AsyncMock) -> list[bool]:
    return [call.kwargs["kill_on_failure"] for call in execute.await_args_list]


async def test_a_lora_checkpoint_load_and_save_do_not_kill_the_cell():
    controller, execute = _controller("lora")
    await controller.load_slot(0, 8, 16.0, ckpt_path="/ckpt", load_optimizer=True)
    await controller.save_slot(0, "/ckpt")
    assert _kill_flags(execute) == [False, False]
    assert execute.await_args_list[0].args == ("load_slot",)
    assert execute.await_args_list[0].kwargs["ckpt_path"] == "/ckpt"


async def test_slot_init_and_training_steps_keep_the_default():
    controller, execute = _controller("lora")
    await controller.load_slot(0, 8, 16.0)
    assert await controller.forward_backward(1, "ref") == ["per-actor"]
    await controller.optim_step({0: {}})
    assert _kill_flags(execute) == [True, True, True]


async def test_full_fine_tuning_checkpoints_keep_the_default():
    controller, execute = _controller("full")
    await controller.load_slot(0, 0, 0.0, ckpt_path="/ckpt")
    await controller.save_slot(0, "/ckpt")
    assert _kill_flags(execute) == [True, True]


class _RecordingInferenceController:
    """The inference controller's update frame as the trainer controller drives it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start_update_weights(self):
        self.calls.append("start")
        return SimpleNamespace(snapshot_cell_id_to_hashes={"engine-0": "hash-0"})

    async def end_update_weights(self, *, snapshot_cell_id_to_hashes):
        self.calls.append("end")

    async def abort_update_weights(self):
        self.calls.append("abort")


def _pushing_controller(execute) -> tuple[TrainerController, _RecordingInferenceController]:
    controller = TrainerController.__new__(TrainerController)
    controller.args = SimpleNamespace(tinker_train_mode="full")
    controller._cells_by_id = {"cell-0": SimpleNamespace(cell_index=0, execute=execute, is_alive=True)}
    controller._inference_controller = _RecordingInferenceController()
    return controller, controller._inference_controller


async def test_a_push_that_succeeds_ends_the_engine_update():
    controller, inference = _pushing_controller(AsyncMock(return_value=[7, 7]))
    assert await controller.push_slot(0, "model@1", 0, 0.0) == 7
    await controller.unload_adapter("model@1")
    assert inference.calls == ["start", "end", "start", "end"]


async def test_a_push_or_unload_that_fails_aborts_the_engine_update():
    """A failed push must not run end_update_weights (it marks the engines weights-ready) and must still hand
    back the controller's lock: without that, the controller's tick waited on the lock for the rest of the run."""
    controller, inference = _pushing_controller(AsyncMock(side_effect=RuntimeError("ReadError on engine 1")))
    with pytest.raises(RuntimeError, match="ReadError on engine 1"):
        await controller.push_slot(0, "model@1", 0, 0.0)
    with pytest.raises(RuntimeError, match="ReadError on engine 1"):
        await controller.unload_adapter("model@1")
    assert inference.calls == ["start", "abort", "start", "abort"]


async def test_a_cancelled_push_aborts_the_engine_update():
    started = asyncio.Event()

    async def _execute(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    controller, inference = _pushing_controller(_execute)
    push = asyncio.create_task(controller.push_slot(0, "model@1", 0, 0.0))
    await started.wait()
    push.cancel()
    with pytest.raises(asyncio.CancelledError):
        await push
    assert inference.calls == ["start", "abort"]

