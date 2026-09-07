from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.cp_utils as cp_utils_mod
import miles.backends.training_utils.replay_data as replay_data_mod
from miles.backends.training_utils.replay_data import (
    fill_replay_data,
    register_replay_list_sequential,
    rollout_replay_stage,
)
from miles.utils.replay_base import ReplayError


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


# ------------------------------- fill guards (job 2245536) -------------------------------


class _Queue:
    """A registered router's replay queue: fill_replay_data counts the rows it holds."""

    def __init__(self):
        self.top_indices_list = []

    def record(self, rows):
        self.top_indices_list.append(rows)


def _fill_rig(monkeypatch):
    state = SimpleNamespace(tp=SimpleNamespace(rank=0, size=1), cp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(replay_data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)
    args = SimpleNamespace(qkv_format="thd", data_pad_size_multiplier=1, allgather_cp=False, sequence_parallel=False)
    rows = [torch.zeros(3, 2, 2, dtype=torch.int32)]
    tokens = [torch.zeros(4, dtype=torch.int64)]
    batch = {"rollout_routed_experts": rows, "tokens": tokens, "max_seq_lens": [4]}
    return args, batch


class _Iterator:
    def __init__(self, batch):
        self._batch = batch

    def reset(self):
        pass

    def get_next(self, keys):
        return {key: self._batch[key] for key in keys}


def _fill(monkeypatch, *, num_microbatches, replay_list, register):
    args, batch = _fill_rig(monkeypatch)
    fill_replay_data(
        args=args,
        models=["model"],
        data_iterator=[_Iterator(batch)],
        num_microbatches=num_microbatches,
        rollout_data=dict(batch),
        data_key="rollout_routed_experts",
        replay_list=replay_list,
        register_replay_list_func=register,
        if_sp_region=True,
    )


def test_a_fill_with_no_microbatches_fails_typed_instead_of_leaving_empty_queues(monkeypatch):
    # forward_backward_no_pipelining always runs the last microbatch, so num_microbatches=0
    # still forwards once: the routers would pop from queues the fill never touched
    with pytest.raises(ReplayError, match="nothing to fill"):
        _fill(monkeypatch, num_microbatches=[0], replay_list=[_Queue()], register=lambda *a, **k: 0)


def test_a_fill_against_an_unregistered_manager_fails_typed(monkeypatch):
    with pytest.raises(ReplayError, match="registered replays=0"):
        _fill(monkeypatch, num_microbatches=[1], replay_list=[], register=lambda *a, **k: 0)


def test_a_fill_that_leaves_a_registered_router_empty_fails_typed(monkeypatch):
    """A model built twice in one process leaves manager.replays twice as long as the live
    routers, and the moe mapping fills only the stale first half. Before, the second half
    stayed empty and the run died with an opaque IndexError inside Megatron's router."""
    stale, live = [_Queue(), _Queue()], [_Queue(), _Queue()]

    def register_first_half(replay_list, replay_data, *, models, **_kwargs):
        for replay in replay_list[: len(replay_list) // 2]:
            replay.record(replay_data)
        return len(replay_list) // 2

    with pytest.raises(ReplayError, match=r"left 2 of 4 replays off the expected 1 rows"):
        _fill(monkeypatch, num_microbatches=[1], replay_list=stale + live, register=register_first_half)


def test_a_complete_fill_records_one_row_per_registered_router(monkeypatch):
    queues = [_Queue(), _Queue()]

    def register_all(replay_list, replay_data, *, models, **_kwargs):
        for replay in replay_list:
            replay.record(replay_data)
        return len(replay_list)

    _fill(monkeypatch, num_microbatches=[2], replay_list=queues, register=register_all)

    assert [len(q.top_indices_list) for q in queues] == [2, 2]
