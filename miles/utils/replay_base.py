import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# The stages in which the router pops recorded rows instead of routing live.
REPLAY_STAGES = ("replay_forward", "replay_backward")


class ReplayError(RuntimeError):
    """R3 replay bookkeeping is inconsistent.

    Always a bug in the wiring (registration, fill, or stage handling), never a
    data condition: an uncovered datum is expressed as all -1 rows, not as a
    missing queue. Raised with the counts that identify which of the three is
    broken instead of surfacing as an opaque IndexError deep inside the router.
    """


class Replay:
    def __init__(self, stream_idx: int | None = None, manager=None, index: int = -1):
        self.stream_idx = stream_idx
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []
        # identity for diagnostics: which manager registered this replay, and its
        # position in manager.replays (= the stream index register_replay_list_* maps to)
        self.manager = manager
        self.index = index

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def _exhausted(self, op: str, cursor: int) -> ReplayError:
        shapes = [t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list]
        manager = self.manager
        context = ""
        if manager is not None:
            lengths = manager.queue_lengths()
            context = (
                f"; manager {manager.name}: enabled={manager.enabled} stage={manager.stage} "
                f"replays={len(manager.replays)} non_empty={sum(1 for n in lengths if n)} "
                f"rows_per_replay={sorted(set(lengths))}"
            )
        return ReplayError(
            f"rank {_get_rank()}: {op} out of range on {getattr(manager, 'name', '?')} replay "
            f"{self.index}: {op.split('_')[1]}_index={cursor}, "
            f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}{context}. "
            "The queues this router reads were never filled: check the 'R3 fill' and "
            "'R3 registration' INFO lines for this rank."
        )

    def pop_forward(self) -> torch.Tensor:
        if self.forward_index >= len(self.top_indices_list):
            raise self._exhausted("pop_forward", self.forward_index)
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self) -> torch.Tensor:
        if self.backward_index >= len(self.top_indices_list):
            raise self._exhausted("pop_backward", self.backward_index)
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0


class BaseReplayManager:
    name: str = ""
    filename: str = ""
    replay_check_min_overlap_ratio = 0.0  # 0.0 = mismatch only on zero overlap
    # True when replayed indices are token/KV positions (indexer): rebase the
    # per-sample 0-based rollout indices onto the packed training sequence.
    replay_indices_are_token_positions = False

    def __init__(self):
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"
        self.register_replay_list_func = None

    def create_replay(self, stream_idx: int | None = None) -> Replay:
        replay = Replay(stream_idx=stream_idx, manager=self, index=len(self.replays))
        self.replays.append(replay)
        return replay

    def reset_registrations(self) -> None:
        """Forget every registered replay.

        ``register_to_module`` appends to ``self.replays`` and the managers are
        process-wide singletons, so a model built twice in one process would
        leave the live routers at stream indices past the ones
        ``register_replay_list_*`` fills -- they would pop from empty queues.
        Called once per model build, before the routers construct themselves.
        """
        self.replays = []
        self.current = None

    def prune_to_model(self, models) -> tuple[int, int]:
        """Keep only the replays a module of ``models`` still holds, in module order.

        A router registers itself in its own ``__init__``, so a wrapper that builds a
        decoder and then throws it away registers twice as many replays as the model
        has live routers -- and the live ones end up in the second half, past the
        streams ``register_replay_list_*`` fills. megatron-bridge's ``Qwen3VLGPTModel``
        does exactly that: ``GPTModel.__init__`` builds ``self.decoder =
        TransformerBlock(...)`` and the subclass then replaces it with a
        ``Qwen3VLTransformerBlock`` built from the same spec. Called once per model
        build; returns ``(kept, dropped)``.
        """
        live: list[Replay] = []
        seen: set[int] = set()
        for model in models:
            for module in model.modules():
                for value in vars(module).values():
                    if isinstance(value, Replay) and value.manager is self and id(value) not in seen:
                        seen.add(id(value))
                        live.append(value)
        dropped = len(self.replays) - len(live)
        self.replays = live
        for index, replay in enumerate(live):
            replay.index = index
        self.current = None
        return len(live), dropped

    def queue_lengths(self) -> list[int]:
        """Rows currently queued per registered replay (diagnostics)."""
        return [len(replay.top_indices_list) for replay in self.replays]

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def clear_all(self):
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    def get_topk_fn(self, old_topk_fn, return_probs):
        manager = self

        def _get_replay_result(top_indices, scores, topk, *args, **kwargs):
            assert (
                top_indices.shape[0] == scores.shape[0]
            ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

            assert (
                top_indices.shape[1] == topk
            ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

            if self.enable_check_replay_result:
                self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

            # rows without any recorded routing (padding, or a datum the rollout side could not
            # cover) take the live router's top-k; rows with some valid picks keep their -1 padding
            all_invalid = (top_indices == -1).all(dim=-1)
            if all_invalid.any():
                natural = old_topk_fn(scores, topk, *args, **kwargs)
                if isinstance(natural, tuple):
                    _, natural = natural
                top_indices = torch.where(all_invalid.unsqueeze(-1), natural.to(top_indices.dtype), top_indices)

            if return_probs:
                return scores.gather(1, top_indices), top_indices
            else:
                return top_indices

        def new_topk_fn(scores, topk, *args, **kwargs):
            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                replay.record(top_indices)
                return result

            elif stage == "replay_forward":
                return _get_replay_result(replay.pop_forward(), scores, topk, *args, **kwargs)

            elif stage == "replay_backward":
                return _get_replay_result(replay.pop_backward(), scores, topk, *args, **kwargs)

            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

        return new_topk_fn

    def register_to_module(self, module, attr_name: str, stream_idx: int | None = None):
        if not self.enabled:
            return
        existing = getattr(module, attr_name, None)
        if isinstance(existing, Replay) and existing.manager is self:
            # idempotent per module instance: a second registration on the same router
            # would grow self.replays past the number of live routers and shift every
            # stream mapping by one (the fill then feeds stale replays).
            return
        replay = self.create_replay(stream_idx=stream_idx)
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Per token, measure the overlap between the training engine's recomputed
        topk and the replayed topk (ignoring -1 padding). A token is mismatched
        when overlap < replay_check_min_overlap_ratio of its valid picks (ratio 0.0 =
        only zero overlap). Raise when the mismatched fraction exceeds
        replay_check_max_mismatch_fraction.
        """
        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.view(-1, top_indices.shape[-1])
        valid_orig = orig_flat != -1
        valid_replay = replay_flat != -1

        # token-wise set overlap via a membership mask, avoiding the O(n*topk^2)
        # all-pairs tensor; -1 padding is routed to a sentinel column
        n_kv = int(torch.maximum(orig_flat.max(), replay_flat.max()).clamp_min(0)) + 1
        membership = orig_flat.new_zeros((orig_flat.shape[0], n_kv + 1), dtype=torch.bool)
        membership.scatter_(1, torch.where(valid_orig, orig_flat, n_kv).long(), True)
        replay_idx = torch.where(valid_replay, replay_flat, n_kv).long()
        hit = membership.gather(1, replay_idx) & valid_replay  # [n_tokens, topk]

        overlap = hit.sum(dim=1)
        valid_count = valid_replay.sum(dim=1)
        is_padding = valid_count == 0
        if self.replay_check_min_overlap_ratio == 0.0:
            required = torch.ones_like(overlap)  # legacy: mismatch only on zero overlap
        else:
            required = (valid_count * self.replay_check_min_overlap_ratio).ceil().to(overlap.dtype)
        is_mismatch = (overlap < required) & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_max_mismatch_fraction))
        mismatch_threshold = threshold * orig_flat.shape[0]
        logger.info(
            f"Replay check (rank {_get_rank()}, stage {self.stage}): "
            f"mismatch {mismatch_count}/{orig_flat.shape[0]} tokens, threshold {mismatch_threshold:.0f}"
        )
        if mismatch_count == 0:
            return

        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        for idx in mismatch_indices[:10]:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} overlap {overlap[i].item()}/{valid_count[i].item()}, topk={topk}\n" + "\n".join(lines)
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    replay_check_min_overlap_ratio = 0.8
    replay_indices_are_token_positions = True


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
