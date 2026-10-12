import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from miles.ray import train_actor
from miles.ray.train.group import TrainerController


def test_worker_flushes_its_own_tracking_process(monkeypatch):
    calls = []
    monkeypatch.setattr(train_actor, "finish_tracking", lambda: calls.append("flushed"))
    train_actor.TrainRayActor.__new__(train_actor.TrainRayActor).finish_tracking()
    assert calls == ["flushed"]


def test_shutdown_stops_watching_then_flushes_live_writers():
    events = []

    async def stop_watching():
        events.append("watcher stopped")

    async def flush(method, *, kill_on_failure):
        assert events == ["watcher stopped"]
        assert method == "finish_tracking"
        assert kill_on_failure is False
        events.append("writer flushed")

    live = SimpleNamespace(cell_index=0, is_alive=True, execute=AsyncMock(side_effect=flush))
    dead = SimpleNamespace(cell_index=1, is_alive=False, execute=AsyncMock())
    controller = TrainerController.__new__(TrainerController)
    controller._watcher_disposer = stop_watching
    controller._cells_by_id = {"live": live, "dead": dead}

    asyncio.run(controller.dispose())

    assert events == ["watcher stopped", "writer flushed"]
    dead.execute.assert_not_called()


def test_logging_failure_does_not_fail_completed_training(caplog):
    cell = SimpleNamespace(
        cell_index=0,
        is_alive=True,
        execute=AsyncMock(side_effect=RuntimeError("logging unavailable")),
    )
    controller = TrainerController.__new__(TrainerController)
    controller._watcher_disposer = None
    controller._cells_by_id = {"live": cell}

    asyncio.run(controller.dispose())

    assert "logging unavailable" in caplog.text
    cell.execute.assert_awaited_once_with("finish_tracking", kill_on_failure=False)
