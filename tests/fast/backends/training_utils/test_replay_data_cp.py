"""R3 under context parallelism: fill_replay_data's routing rows line up with get_batch's local tokens.

Token ids encode (sample, position); routing row i of sample s encodes the same (s, i) (row i is the
routing used when token i was the input). For every CP rank x TP rank, the rows a router would pop
(after the CP slice and the sequence-parallel split) must carry the identity of the local token at
that position, or -1 where the local token is padding or a sample's final token. Covers CP 1-8,
TP/SP 1-4 and three pad multipliers on gateway-shaped datums (one-token prompt, 2-token rows); the
negative control slices the routing contiguously (--allgather-cp) and must misalign at CP>1.
Ported from the CP plan's step-5 scratch check (cp_replay_align.py and its negative control).
"""

import importlib
import itertools
import random
import sys
import types
from types import SimpleNamespace

import pytest
import torch

DATA_KEY = "rollout_routed_experts"
NUM_STREAMS, TOPK = 3, 2  # MoE layers, routed experts per token
ENC = 100_000
LAYOUTS = [(1, False), (2, True), (4, True)]  # (TP, sequence parallel): Megatron turns SP off at TP1


@pytest.fixture(scope="module")
def mods():
    """get_batch's module needs only two helpers of miles.utils.data, which imports sglang (through the chat
    templates); without sglang (CPU venvs) those two are stubbed for this module's tests only."""
    saved = dict(sys.modules)
    try:
        try:
            importlib.import_module("miles.utils.data")
        except ImportError:
            stub = types.ModuleType("miles.utils.data")
            stub.get_minimum_num_micro_batch_size = lambda *args, **kwargs: 1
            stub.process_rollout_data = lambda *args, **kwargs: None
            sys.modules["miles.utils.data"] = stub
            sys.modules.pop("miles.backends.training_utils.data", None)
        yield SimpleNamespace(
            cp_utils=importlib.import_module("miles.backends.training_utils.cp_utils"),
            data=importlib.import_module("miles.backends.training_utils.data"),
            replay=importlib.import_module("miles.backends.training_utils.replay_data"),
        )
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def _stub_cuda(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)


def _token_id(sample: int, position: int) -> int:
    return sample * ENC + position + 1  # never 0, get_batch's pad token


class _Iterator:
    def __init__(self, batch: dict):
        self._batch = batch
        self.rollout_data = {}

    def get_next(self, keys):
        return {key: self._batch.get(key) for key in keys}

    def reset(self):
        pass


class _Queue:
    def __init__(self):
        self.top_indices_list = []

    def record(self, rows):
        self.top_indices_list.append(rows.clone())


def _register(replay_list, replay_data, **_kwargs):
    for stream, replay in enumerate(replay_list):
        replay.record(replay_data[:, stream])
    return len(replay_list)


def _batch(lengths: list[int]) -> dict:
    tokens = [torch.tensor([_token_id(s, i) for i in range(n)], dtype=torch.long) for s, n in enumerate(lengths)]
    routes = []
    for s, n in enumerate(lengths):
        ids = torch.tensor([_token_id(s, i) for i in range(n - 1)], dtype=torch.int64)
        # unique per (token, stream, slot)
        routes.append(ids[:, None, None] * 10 + torch.arange(NUM_STREAMS * TOPK).reshape(1, NUM_STREAMS, TOPK))
    response_lengths = [max(1, n - 1) for n in lengths]  # Tinker datums: the first token is the prompt
    return {
        "tokens": tokens,
        "total_lengths": list(lengths),
        "response_lengths": response_lengths,
        "loss_masks": [torch.ones(r, dtype=torch.int) for r in response_lengths],
        "max_seq_lens": None,
        "adapter_slots": None,
        DATA_KEY: routes,
    }


def _expected_rows(local_tokens: torch.Tensor, lengths: list[int], stream: int) -> torch.Tensor:
    sample, position = (local_tokens - 1) // ENC, (local_tokens - 1) % ENC
    last_position = torch.tensor(lengths)[sample.clamp(0, len(lengths) - 1)] - 1
    has_route = (local_tokens != 0) & (position < last_position)
    ids = local_tokens[:, None] * 10 + stream * TOPK + torch.arange(TOPK)[None, :]
    return torch.where(has_route[:, None], ids, torch.full_like(ids, -1))


def _misaligned_ranks(
    mods, monkeypatch, lengths, cp_size, tp_size, sequence_parallel, pad_multiplier, allgather_fill=False
):
    """(cp_rank, tp_rank) pairs whose replay rows do not match their local tokens."""
    misaligned = []
    for cp_rank, tp_rank in itertools.product(range(cp_size), range(tp_size)):
        state = SimpleNamespace(
            cp=SimpleNamespace(rank=cp_rank, size=cp_size, group=None),
            tp=SimpleNamespace(rank=tp_rank, size=tp_size, group=None),
        )
        for module in (mods.cp_utils, mods.data, mods.replay):
            monkeypatch.setattr(module, "get_parallel_state", lambda state=state: state)
        batch = _batch(lengths)
        got = mods.data.get_batch(
            _Iterator(batch),
            ["tokens", "total_lengths", "response_lengths", "loss_masks", "max_seq_lens"],
            pad_multiplier,
            "thd",
            allgather_cp=False,
        )
        local_tokens = got["tokens"].reshape(-1)
        if sequence_parallel and tp_size > 1:  # Megatron scatters the sequence over TP ranks contiguously
            width = local_tokens.numel() // tp_size
            local_tokens = local_tokens[tp_rank * width : (tp_rank + 1) * width]
        args = SimpleNamespace(
            qkv_format="thd",
            allgather_cp=allgather_fill,
            sequence_parallel=sequence_parallel,
            data_pad_size_multiplier=pad_multiplier,
        )
        queues = [_Queue() for _ in range(NUM_STREAMS)]
        mods.replay.fill_replay_data(
            args=args,
            models=None,
            data_iterator=[_Iterator(batch)],
            num_microbatches=[1],
            rollout_data={DATA_KEY: list(batch[DATA_KEY])},
            data_key=DATA_KEY,
            replay_list=queues,
            register_replay_list_func=_register,
            if_sp_region=True,
        )
        for stream, queue in enumerate(queues):
            rows = queue.top_indices_list[0].to(torch.int64)
            if rows.shape[0] != local_tokens.numel() or not torch.equal(
                rows, _expected_rows(local_tokens, lengths, stream)
            ):
                misaligned.append((cp_rank, tp_rank))
                break
    return misaligned


def _cases(seed: int, count: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.choice([2, 3, 5, 17, 64, 129, 1000, 4097]) for _ in range(rng.randint(1, 5))] for _ in range(count)]


@pytest.mark.parametrize("pad_multiplier", [1, 8, 128])
@pytest.mark.parametrize("tp_size, sequence_parallel", LAYOUTS)
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_replay_rows_follow_the_zigzag_tokens(mods, monkeypatch, cp_size, tp_size, sequence_parallel, pad_multiplier):
    for lengths in _cases(seed=cp_size * 100 + tp_size * 10 + pad_multiplier, count=2) + [[2], [2, 3, 4097]]:
        misaligned = _misaligned_ranks(mods, monkeypatch, lengths, cp_size, tp_size, sequence_parallel, pad_multiplier)
        assert not misaligned, f"routing rows off their tokens on (cp, tp) ranks {misaligned}: lengths={lengths}"


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_a_contiguous_routing_slice_misaligns(mods, monkeypatch, cp_size):
    """Negative control: a fill that slices the packed row contiguously (--allgather-cp's layout) against
    zigzag tokens must be caught, so the alignment check above can fail."""
    lengths = [17, 129, 1000]
    misaligned = _misaligned_ranks(mods, monkeypatch, lengths, cp_size, 1, False, 8, allgather_fill=True)
    assert misaligned, "the zigzag token layout and a contiguous routing slice cannot agree at CP>1"
    assert not _misaligned_ranks(mods, monkeypatch, lengths, 1, 1, False, 8, allgather_fill=True), "same at CP=1"
