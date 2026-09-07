from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.replay_data as replay_data_mod
from miles.backends.training_utils.replay_data import register_replay_list_sequential, rollout_replay_stage


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


def test_register_replay_list_sequential_falls_back_to_enum_when_no_stream_idx():
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    replays = [_Replay(), _Replay(), _Replay()]

    register_replay_list_sequential(replays, replay_data)

    for replay_idx, replay in enumerate(replays):
        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay_idx])


def test_register_replay_list_sequential_uses_stream_idx_when_set():
    # PP case: 2 local modules at global streams 1 and 3 out of 4.
    replay_data = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    replays = [_Replay(stream_idx=1), _Replay(stream_idx=3)]

    register_replay_list_sequential(replays, replay_data)

    torch.testing.assert_close(replays[0].recorded[0], replay_data[:, 1])
    torch.testing.assert_close(replays[1].recorded[0], replay_data[:, 3])


def test_register_replay_list_sequential_rejects_out_of_range_stream_idx():
    replay_data = torch.zeros(5, 4, 2)
    replays = [_Replay(stream_idx=4)]

    with pytest.raises(AssertionError, match="out of range"):
        register_replay_list_sequential(replays, replay_data)


# ------------------------------- rollout_replay_stage -------------------------------


class _Manager:
    """The BaseReplayManager surface rollout_replay_stage touches."""

    name = "routing"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    replay_indices_are_token_positions = False

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.stage = "fallthrough"
        self.replays = []
        self.register_replay_list_func = object()
        self.cleared = 0

    def clear_all(self):
        self.cleared += 1


def _stage_rig(monkeypatch, *, enabled: bool, flag: bool):
    manager = _Manager(enabled)
    monkeypatch.setattr(replay_data_mod, "all_replay_managers", [manager])
    fills = []
    monkeypatch.setattr(replay_data_mod, "fill_replay_data", lambda **kwargs: fills.append(kwargs))
    args = SimpleNamespace(use_rollout_routing_replay=flag)
    return manager, fills, args


@pytest.mark.parametrize("stage", ["replay_forward", "replay_backward"])
def test_rollout_replay_stage_fills_holds_the_stage_and_clears_on_exit(monkeypatch, stage):
    manager, fills, args = _stage_rig(monkeypatch, enabled=True, flag=True)
    rollout_data = {"rollout_routed_experts": ["rows"]}

    with rollout_replay_stage(args, ["model"], ["it"], [1], rollout_data, stage=stage):
        assert manager.stage == stage
        assert manager.cleared == 0

    assert [f["data_key"] for f in fills] == ["rollout_routed_experts"]
    assert fills[0]["replay_list"] is manager.replays
    assert fills[0]["register_replay_list_func"] is manager.register_replay_list_func
    assert fills[0]["if_sp_region"] is True
    assert (manager.stage, manager.cleared) == ("fallthrough", 1)


def test_rollout_replay_stage_clears_when_the_body_raises(monkeypatch):
    manager, _, args = _stage_rig(monkeypatch, enabled=True, flag=True)

    with pytest.raises(RuntimeError):
        with rollout_replay_stage(args, ["model"], ["it"], [1], {}, stage="replay_forward"):
            raise RuntimeError("cuda oom")

    assert (manager.stage, manager.cleared) == ("fallthrough", 1)


@pytest.mark.parametrize("enabled, flag", [(True, False), (False, True)])
def test_rollout_replay_stage_leaves_inactive_managers_alone(monkeypatch, enabled, flag):
    manager, fills, args = _stage_rig(monkeypatch, enabled=enabled, flag=flag)

    with rollout_replay_stage(args, ["model"], ["it"], [1], {}, stage="replay_forward"):
        assert manager.stage == "fallthrough"

    assert fills == [] and manager.cleared == 0
