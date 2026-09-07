from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import torch

from miles.utils.replay_base import BaseReplayManager


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)

    def pop_forward(self):
        return self.top_indices.pop(0)

    def pop_backward(self):
        return self.pop_forward()


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices.to(torch.int32)


def _make_replay_manager(top_indices):
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_get_topk_fn_routes_all_invalid_rows_through_the_live_router():
    # an all-(-1) row (padding, or a datum without cached rollout routing) takes
    # the natural top-k of the live router, never a synthetic arange fill
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_replay_manager(torch.tensor([[-1, -1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), torch.tensor([[4, 3, 2]], dtype=torch.int32))


def test_get_topk_fn_mixed_batch_keeps_replayed_rows_and_routes_uncovered_rows_naturally():
    scores = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 4.0, 1.0, 3.0, 2.0],
            [1.0, 0.0, 4.0, 2.0, 3.0],
        ]
    )
    replayed = torch.tensor([[1, 0, 3], [-1, -1, -1], [2, -1, -1], [-1, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)
    out = topk_fn(scores, 3)

    expected = torch.tensor([[1, 0, 3], [0, 1, 2], [2, -1, -1], [2, 4, 3]], dtype=torch.int32)
    torch.testing.assert_close(out, expected)
    assert out.dtype == replayed.dtype


def test_get_topk_fn_fallback_accepts_megatron_tuple_topk_and_returns_probs():
    # Megatron's _compute_topk returns (values, indices) with int64 indices; the
    # fallback takes the indices and keeps the replayed dtype for the gather
    def megatron_topk(scores, topk):
        return torch.topk(scores, topk, dim=1)

    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0, 0.0]])
    replayed = torch.tensor([[-1, -1, -1], [4, 3, 1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed)

    probs, top_indices = manager.get_topk_fn(megatron_topk, return_probs=True)(scores, 3)

    torch.testing.assert_close(top_indices, torch.tensor([[4, 3, 2], [4, 3, 1]], dtype=torch.int32))
    torch.testing.assert_close(probs, torch.tensor([[4.0, 3.0, 2.0], [0.0, 1.0, 3.0]]))


def test_get_topk_fn_preserves_partial_padding():
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)
