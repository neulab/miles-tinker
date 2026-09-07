from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.replay_base import BaseReplayManager, ReplayError


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


# ------------------------------- registration bookkeeping -------------------------------


class _Router(torch.nn.Module):
    """The surface register_to_module touches (a Megatron TopKRouter registers itself in __init__)."""

    def forward(self, x):
        return x


def _enabled_manager():
    manager = BaseReplayManager()
    manager.enabled = True
    return manager


def test_register_to_module_is_idempotent_per_module_instance():
    # a router that registers twice would push every later router one stream along,
    # and register_replay_list_* would then fill the stale half of manager.replays
    manager = _enabled_manager()
    router = _Router()

    manager.register_to_module(router, "routing_replay")
    manager.register_to_module(router, "routing_replay")

    assert len(manager.replays) == 1
    assert router.routing_replay is manager.replays[0]


def test_reset_registrations_drops_a_previous_model_builds_routers():
    # setup_model_and_optimizer calls this before the build: the managers are process-wide
    # singletons, so without it a second build leaves the live routers at streams 40..79
    manager = _enabled_manager()
    first = [_Router() for _ in range(3)]
    for router in first:
        manager.register_to_module(router, "routing_replay")

    manager.reset_registrations()
    second = [_Router() for _ in range(3)]
    for router in second:
        manager.register_to_module(router, "routing_replay")

    assert len(manager.replays) == 3
    assert [router.routing_replay.index for router in second] == [0, 1, 2]
    assert manager.get_current() is None


def test_an_exhausted_replay_raises_a_typed_error_naming_the_manager_and_its_queues():
    manager = _enabled_manager()
    for _ in range(3):
        manager.register_to_module(_Router(), "routing_replay")
    # record() pins host memory (CUDA-only); the queue itself is a plain list
    manager.replays[0].top_indices_list.append(torch.zeros(2, 2, dtype=torch.int32))

    with pytest.raises(ReplayError) as excinfo:
        manager.replays[2].pop_forward()

    message = str(excinfo.value)
    assert "pop_forward out of range" in message
    assert "replays=3" in message and "non_empty=1" in message and "rows_per_replay=[0, 1]" in message


class _Block(torch.nn.Module):
    """A decoder block whose layers register a replay each, as Megatron's TopKRouter does."""

    def __init__(self, manager, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Router() for _ in range(num_layers))
        for layer in self.layers:
            manager.register_to_module(layer, "routing_replay")


class _RebuildingModel(torch.nn.Module):
    """megatron-bridge's Qwen3VLGPTModel: GPTModel.__init__ builds self.decoder, then the
    subclass replaces it with its own block built from the same spec (text_model.py:118)."""

    def __init__(self, manager, num_layers: int):
        super().__init__()
        self.decoder = _Block(manager, num_layers)  # GPTModel.__init__
        self.decoder = _Block(manager, num_layers)  # "# rebuild the transformer block"


def test_prune_to_model_keeps_the_live_routers_of_a_rebuilt_block():
    # job 2245536: the discarded block's replays took streams 0..39, the live routers took
    # 40..79, and register_replay_list_moe filled the discarded half
    manager = _enabled_manager()
    model = _RebuildingModel(manager, num_layers=4)
    assert len(manager.replays) == 8, "both blocks registered"

    kept, dropped = manager.prune_to_model([model])

    assert (kept, dropped) == (4, 4)
    assert manager.replays == [layer.routing_replay for layer in model.decoder.layers]
    assert [replay.index for replay in manager.replays] == [0, 1, 2, 3]


def test_prune_to_model_is_a_no_op_for_a_model_that_builds_its_block_once():
    manager = _enabled_manager()
    model = _Block(manager, num_layers=3)
    before = list(manager.replays)

    assert manager.prune_to_model([model]) == (3, 0)
    assert manager.replays == before
