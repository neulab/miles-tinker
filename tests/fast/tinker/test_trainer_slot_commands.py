"""Slot commands: a LoRA checkpoint save/load fails on every rank together with no collective
pending, so it must not recycle the trainer cell; slot init, the training steps and full
fine-tuning checkpoints keep cell.execute's kill-on-failure default."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

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
