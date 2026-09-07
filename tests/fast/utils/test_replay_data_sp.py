"""fill_replay_data under sequence parallelism: the per-microbatch replay tensor
is padded to a multiple of tp_size * data_pad_size_multiplier and each TP rank
registers its own SP-region slice. Megatron forces sequence_parallel off at
TP1 (the slice is a no-op there); this layout is live only at TP>1."""

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.cp_utils as cp_utils_mod
import miles.backends.training_utils.replay_data as replay_data_mod
from miles.backends.training_utils.replay_data import fill_replay_data

DATA_KEY = "rollout_routed_experts"
NUM_LAYERS, TOPK = 2, 3


class _FakeIterator:
    """One microbatch; get_next serves the keys fill_replay_data asks for."""

    def __init__(self, batch: dict):
        self._batch = batch
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_next(self, keys):
        return {key: self._batch[key] for key in keys}


class _FakeReplay:
    """A registered module's queue: fill_replay_data only counts the rows it appends."""

    def __init__(self):
        self.top_indices_list = []

    def record(self, rows):
        self.top_indices_list.append(rows)


def _rows(num_tokens: int, base: int) -> torch.Tensor:
    # row i of a sample = the routing used when token i was the input: one row per token but the last
    return (torch.arange((num_tokens - 1) * NUM_LAYERS * TOPK, dtype=torch.int32) + base).reshape(
        num_tokens - 1, NUM_LAYERS, TOPK
    )


def _fill(monkeypatch, *, tp_rank: int, tp_size: int, sequence_parallel: bool):
    state = SimpleNamespace(tp=SimpleNamespace(rank=tp_rank, size=tp_size), cp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(replay_data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)
    args = SimpleNamespace(
        qkv_format="thd", data_pad_size_multiplier=1, allgather_cp=False, sequence_parallel=sequence_parallel
    )
    tokens = [torch.zeros(4, dtype=torch.int64), torch.zeros(3, dtype=torch.int64)]
    samples = [_rows(4, 100), _rows(3, 200)]
    rollout_data = {DATA_KEY: samples, "tokens": tokens}
    iterator = _FakeIterator({DATA_KEY: samples, "tokens": tokens, "max_seq_lens": [4]})
    registered = []
    replay_list = [_FakeReplay()]

    def register(replay_list, replay_data, *, models, **_kwargs):
        registered.append(replay_data)
        for replay in replay_list:
            replay.record(replay_data)
        return len(replay_list)

    fill_replay_data(
        args=args,
        models=["model"],
        data_iterator=[iterator],
        num_microbatches=[1],
        rollout_data=rollout_data,
        data_key=DATA_KEY,
        replay_list=replay_list,
        register_replay_list_func=register,
        if_sp_region=True,
    )
    assert DATA_KEY not in rollout_data, "the replay tensors are consumed by the fill"
    assert iterator.resets == 2, "the iterator is rewound before and after the walk"
    assert len(registered) == 1
    return registered[0]


def _padded_global_rows() -> torch.Tensor:
    # sample rows + one -1 row per sample (the loss-masked final token), concatenated, then -1 padded
    # to a multiple of tp_size * data_pad_size_multiplier = 2: 3 + 1 + 2 + 1 = 7 -> 8 rows
    pad = torch.full((1, NUM_LAYERS, TOPK), -1, dtype=torch.int32)
    return torch.cat([_rows(4, 100), pad, _rows(3, 200), pad, pad], dim=0)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_sp_region_slice_yields_the_ranks_half(monkeypatch, tp_rank):
    local = _fill(monkeypatch, tp_rank=tp_rank, tp_size=2, sequence_parallel=True)

    expected = _padded_global_rows().chunk(2, dim=0)[tp_rank]
    assert local.shape == (4, NUM_LAYERS, TOPK)
    torch.testing.assert_close(local, expected)


def test_tp1_without_sequence_parallel_registers_the_whole_padded_tensor(monkeypatch):
    local = _fill(monkeypatch, tp_rank=0, tp_size=1, sequence_parallel=False)

    torch.testing.assert_close(local, _padded_global_rows()[:7], msg="pad multiple is 1: no trailing pad row")


def test_tp2_without_sequence_parallel_keeps_all_rows_on_every_rank(monkeypatch):
    for tp_rank in (0, 1):
        local = _fill(monkeypatch, tp_rank=tp_rank, tp_size=2, sequence_parallel=False)
        torch.testing.assert_close(local, _padded_global_rows())
