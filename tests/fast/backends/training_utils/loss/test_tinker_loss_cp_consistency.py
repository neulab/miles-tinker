"""Context-parallel consistency of the Tinker losses and the per-datum recorder.

Ported from radixark/miles 3d8b5284 (#3320, a test file of the same name) onto the
fork's paths (``loss_hub/tinker_losses.py``, ``cp_utils.py``, the start/drain
per-datum recorder) and extended for full fine-tuning through the gateway:

* the gateway's datum layout: tokens = model_input + [last target], so a one-token
  prompt and ``response_length = len(model_input)`` (``encoding.build_row``);
* the runtime's 2-token ``PAD_ROW`` padding rows (all-zero loss mask);
* tiny datums of 3 .. 2*CP+1 tokens, where a rank holds one or no response token;
* a microbatch made only of pad rows (one rank scores the single target, the others
  hold none and must still return a loss connected to the logits);
* the fused fp32 lm head (the loss receives the head's hidden states);
* ``--recompute-loss-function`` (outputs recorded once; the full-length client
  vectors are sliced per call, never in place).

Each case runs one batch at cp=1 and at cp>1 (zigzag: rank r holds chunks r and
2*cp-1-r of every sequence) and checks the per-token logprobs, the summed loss, the
input gradients and the per-datum outputs against cp=1. Logprobs go through the
production scorer; ``cpu_cross_entropy`` stands in for Megatron's fused kernel.
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.cp_utils import (
    all_gather_with_cp,
    gather_cp_response_values,
    slice_log_prob_with_cp,
    slice_with_cp,
)
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.loss_hub import fused_lm_head, tinker_losses
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

from . import cpu_cross_entropy
from .loss_test_utils import make_args

VOCAB_SIZE = 64
HIDDEN_SIZE = 16
LOSS_FNS = sorted(tinker_losses.TINKER_LOSS_FUNCTIONS)

# (total_length, response_length), upstream's cases; (10, 6) leaves rank 0 of cp2 without response logits
RAGGED_SHAPES = [(12, 8), (7, 5), (10, 6)]
# a lone one-token response: rank 0 holds no response token of the whole microbatch
EMPTY_RANK_SHAPES = [(9, 1)]
# upstream zeroes a few positions of the first two masks so the per-token mask is sliced too
PARTIAL_MASK_ZEROS = {0: [2], 1: [0]}


def _gateway_shapes(cp_size: int) -> list[tuple[int, int]]:
    """Gateway datums: a one-token prompt, ordinary lengths plus every tiny one of 2 .. 2*cp+1 tokens."""
    model_input_lengths = [11, 6, 17, *range(1, 2 * cp_size + 1)]
    return [(n + 1, n) for n in model_input_lengths]


def _set_parallel_state(cp_rank: int, cp_size: int, tp_group) -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=cp_rank, size=cp_size, group=dist.group.WORLD if cp_size > 1 else None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=cp,
            tp=GroupInfo(rank=0, size=1, group=tp_group),
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )


def _make_datums(
    shapes: list[tuple[int, int]], *, num_pad_rows: int = 0, partial_masks: bool = False, width: int = VOCAB_SIZE
) -> list[dict]:
    g = torch.Generator().manual_seed(7)
    datums = []
    for i, (total_length, response_length) in enumerate(shapes):
        loss_mask = torch.ones(response_length, dtype=torch.int32)
        if partial_masks:
            loss_mask[[z for z in PARTIAL_MASK_ZEROS.get(i, []) if z < response_length]] = 0
        datums.append(
            dict(
                tokens=torch.randint(0, VOCAB_SIZE, (total_length,), generator=g),
                loss_mask=loss_mask,
                loss_weights=(torch.rand(response_length, generator=g) + 0.5).tolist(),
                advantages=torch.randn(response_length, generator=g).tolist(),
                rollout_log_probs=torch.randn(response_length, generator=g) - 2.0,
                inputs=torch.randn(total_length, width, generator=g),
            )
        )
    for _ in range(num_pad_rows):  # runtime.PAD_ROW: tokens [0, 0], one zero-weight target, all-zero mask
        datums.append(
            dict(
                tokens=torch.zeros(2, dtype=torch.long),
                loss_mask=torch.zeros(1, dtype=torch.int32),
                loss_weights=[0.0],
                advantages=[0.0],
                rollout_log_probs=torch.zeros(1),
                inputs=torch.randn(2, width, generator=g),
            )
        )
    return datums


def _batch(datums: list[dict], rollout_log_probs: list, loss_fn: str) -> dict:
    return {
        "loss_fn": loss_fn,
        "unconcat_tokens": [d["tokens"] for d in datums],
        "total_lengths": [d["tokens"].numel() for d in datums],
        "response_lengths": [d["loss_mask"].numel() for d in datums],
        "loss_masks": [d["loss_mask"] for d in datums],
        "loss_weights": [list(d["loss_weights"]) for d in datums],
        "advantages": [list(d["advantages"]) for d in datums],
        "rollout_log_probs": rollout_log_probs,
        "sample_indices": list(range(len(datums))),
        "dynamic_global_batch_size": len(datums),
    }


def _score(args, batch: dict, inputs: torch.Tensor, head_weight: torch.Tensor | None) -> dict:
    """One microbatch through loss_function with the recorder on, then backward."""
    inputs = inputs.detach().clone().requires_grad_(True)
    weight = None if head_weight is None else torch.nn.Parameter(head_weight.detach().clone())
    state = None
    if weight is not None:
        state = fused_lm_head.HeadState(
            output_layer=None, original_forward=None, tp_group=None, active_weight=weight, chunk_size=3
        )
    with fused_lm_head.activate(state):
        tinker_losses.start_per_datum_outputs()
        try:
            loss, _, _ = loss_function(args, batch, 1, inputs)
            loss.backward()
        finally:
            per_datum = tinker_losses.drain_per_datum_outputs()
        with torch.no_grad():
            log_probs = [lp.detach() for lp in tinker_losses._target_logprobs(args, batch, inputs)]
    return dict(
        loss=loss.detach(),
        log_probs=log_probs,
        per_datum=per_datum,
        input_grad=inputs.grad.squeeze(0),
        # a rank holding no response token never touches the head weight: no gradient there
        weight_grad=None if weight is None else (weight.grad if weight.grad is not None else torch.zeros_like(weight)),
    )


def _check_against_cp1(
    rank: int,
    world_size: int,
    port: int,
    *,
    shapes: list[tuple[int, int]],
    loss_fns: list[str],
    num_pad_rows: int = 0,
    partial_masks: bool = False,
    fused_head: bool = False,
    recompute: bool = False,
) -> None:
    torch.set_num_threads(1)  # world_size processes on one host: avoid oversubscribing intra-op threads
    cpu_cross_entropy.install()
    init_gloo(rank, world_size, port=port)
    # the cross-entropy stand-in needs a real one-rank TP group per rank; new_group is collective
    tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
    args = make_args(
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=4,
        allgather_cp=False,
        multi_lora=False,  # full fine-tuning: loss_fn on the batch selects the Tinker loss
        calculate_per_token_loss=False,
        recompute_loss_function=recompute,
        use_dynamic_global_batch_size=True,
    )
    width = HIDDEN_SIZE if fused_head else VOCAB_SIZE
    head_weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=torch.Generator().manual_seed(3)) * 0.3
    datums = _make_datums(shapes, num_pad_rows=num_pad_rows, partial_masks=partial_masks, width=width)
    total_lengths = [d["tokens"].numel() for d in datums]
    response_lengths = [d["loss_mask"].numel() for d in datums]
    inputs_full = torch.cat([d["inputs"] for d in datums]).unsqueeze(0)

    for loss_fn in loss_fns:
        _set_parallel_state(0, 1, tp_group)
        base_batch = _batch(datums, [d["rollout_log_probs"] for d in datums], loss_fn)
        base = _score(args, base_batch, inputs_full, head_weight if fused_head else None)

        _set_parallel_state(rank, world_size, tp_group)
        inputs_local = torch.cat([slice_with_cp(d["inputs"], 0.0, "thd") for d in datums]).unsqueeze(0)
        local_rollout_log_probs = [  # pre-sliced, as get_rollout_data does
            torch.as_tensor(slice_log_prob_with_cp(d["rollout_log_probs"], total, response))
            for d, total, response in zip(datums, total_lengths, response_lengths, strict=True)
        ]
        cp_batch = _batch(datums, local_rollout_log_probs, loss_fn)
        cp = _score(args, cp_batch, inputs_local, head_weight if fused_head else None)
        context = f"{loss_fn} cp={world_size} rank={rank}"
        # the unfused scorer is row-wise, so reassembled logprobs are bitwise equal; the fused head's
        # chunked GEMM rounds differently with the number of rows per chunk (CPU mm blocking)
        exact = dict(rtol=1e-5, atol=1e-6) if fused_head else dict(rtol=0, atol=0)

        # the client vectors are sliced per call, never in place: a recompute sees the full batch again
        for key in ("loss_weights", "advantages", "loss_masks"):
            assert [len(v) for v in cp_batch[key]] == response_lengths, context

        # (a) per-token logprobs: the scorer is row-wise, so the reassembled vectors are exact
        for i, (total, response) in enumerate(zip(total_lengths, response_lengths, strict=True)):
            full = all_gather_with_cp(cp["log_probs"][i], total, response)
            torch.testing.assert_close(full, base["log_probs"][i], **exact, msg=context)

        # (b) each rank's loss is its local shard sum; the CP group sums to the cp=1 loss
        cp_loss = cp["loss"].clone()
        dist.all_reduce(cp_loss)
        torch.testing.assert_close(cp_loss, base["loss"], rtol=1e-6, atol=1e-6, msg=context)

        # (c) input gradients land on this rank's zigzag shard of the cp=1 gradient
        offset, expected = 0, []
        for total in total_lengths:
            expected.append(slice_with_cp(base["input_grad"][offset : offset + total], 0.0, "thd"))
            offset += total
        torch.testing.assert_close(cp["input_grad"], torch.cat(expected), rtol=1e-5, atol=1e-6, msg=context)
        if fused_head:  # the weight gradient is summed over CP ranks by the DP*CP reduction
            weight_grad = cp["weight_grad"].clone()
            dist.all_reduce(weight_grad)
            torch.testing.assert_close(weight_grad, base["weight_grad"], rtol=1e-5, atol=1e-6, msg=context)

        # (d) per-datum outputs: recorded once, full-length, identical on every rank and equal to cp=1
        assert [o["sample_index"] for o in cp["per_datum"]] == list(range(len(datums))), context
        assert [o["sample_index"] for o in base["per_datum"]] == list(range(len(datums))), context
        for cp_output, base_output, response in zip(cp["per_datum"], base["per_datum"], response_lengths, strict=True):
            assert cp_output["logprobs"].shape == (response,) and cp_output["logprobs"].device.type == "cpu"
            torch.testing.assert_close(cp_output["logprobs"], base_output["logprobs"], **exact, msg=context)
            torch.testing.assert_close(cp_output["loss"], base_output["loss"], rtol=1e-6, atol=1e-6, msg=context)

        # (e) pad rows: no loss, no gradient on any rank
        for index in range(len(datums) - num_pad_rows, len(datums)):
            assert float(cp["per_datum"][index]["loss"]) == 0.0, context
        if num_pad_rows:
            pad_rows = sum(total_lengths[len(datums) - num_pad_rows :])
            assert torch.count_nonzero(base["input_grad"][-pad_rows:]) == 0, context
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "shapes, partial_masks",
    [(RAGGED_SHAPES, True), (EMPTY_RANK_SHAPES, False)],
    ids=["ragged", "empty_rank"],
)
def test_upstream_shapes_cp2_match_cp1(shapes, partial_masks):
    run_multiprocess(
        partial(_check_against_cp1, shapes=shapes, loss_fns=LOSS_FNS, partial_masks=partial_masks), world_size=2
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_gateway_datums_with_tiny_rows_and_pad_rows_match_cp1(world_size):
    run_multiprocess(
        partial(_check_against_cp1, shapes=_gateway_shapes(world_size), loss_fns=LOSS_FNS, num_pad_rows=2),
        world_size=world_size,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_pad_row_only_microbatch(world_size):
    """Fewer real rows than DP ranks: a rank's whole microbatch is pad rows; ranks without its target stay connected."""
    run_multiprocess(partial(_check_against_cp1, shapes=[], loss_fns=LOSS_FNS, num_pad_rows=2), world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_fused_head_matches_cp1(world_size):
    run_multiprocess(
        partial(
            _check_against_cp1,
            shapes=_gateway_shapes(world_size) + EMPTY_RANK_SHAPES,
            loss_fns=["cross_entropy", "importance_sampling"],
            num_pad_rows=1,
            fused_head=True,
        ),
        world_size=world_size,
    )


def test_fused_head_pad_row_only_microbatch():
    run_multiprocess(
        partial(_check_against_cp1, shapes=[], loss_fns=["cross_entropy"], num_pad_rows=1, fused_head=True),
        world_size=2,
    )


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling"])
def test_recompute_loss_function_cp4(loss_fn):
    run_multiprocess(
        partial(
            _check_against_cp1,
            shapes=RAGGED_SHAPES + _gateway_shapes(4),
            loss_fns=[loss_fn],
            num_pad_rows=1,
            partial_masks=True,
            recompute=True,
        ),
        world_size=4,
    )


# ---------------------------------------------------------------------------
# gather_cp_response_values: the batched gather the recorder and Tinker forward share
# ---------------------------------------------------------------------------

GATHER_SHAPES = [(12, 8), (3, 2), (2, 1), (9, 1), (17, 16), (5, 4)]


def _check_gather(rank: int, world_size: int, port: int, *, dtype: torch.dtype) -> None:
    torch.set_num_threads(1)
    init_gloo(rank, world_size, port=port)
    _set_parallel_state(rank, world_size, None)
    g = torch.Generator().manual_seed(11)
    full = [torch.randn(response, generator=g).to(dtype) for _, response in GATHER_SHAPES]
    partials = torch.randn(world_size, len(GATHER_SHAPES), generator=g)  # every rank draws the same table
    local = [
        torch.as_tensor(slice_log_prob_with_cp(vector, total, response))
        for vector, (total, response) in zip(full, GATHER_SHAPES, strict=True)
    ]
    totals = [total for total, _ in GATHER_SHAPES]
    responses = [response for _, response in GATHER_SHAPES]
    vectors, summed = gather_cp_response_values(local, totals, responses, scalars=list(partials[rank]))
    for vector, expected in zip(vectors, full, strict=True):
        assert vector.dtype == dtype and vector.device.type == "cpu" and not vector.requires_grad
        torch.testing.assert_close(vector, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(summed), partials.sum(dim=0), rtol=1e-6, atol=1e-6)
    # no scalars, and an empty batch
    vectors, summed = gather_cp_response_values(local, totals, responses)
    assert summed == [] and all(torch.equal(v, e) for v, e in zip(vectors, full, strict=True))
    assert gather_cp_response_values([], [], []) == ([], [])
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_gather_cp_response_values_rebuilds_every_vector(world_size, dtype):
    run_multiprocess(partial(_check_gather, dtype=dtype), world_size=world_size)


def test_gather_cp_response_values_at_cp1_is_a_cpu_copy():
    _set_parallel_state(0, 1, None)
    values = [torch.tensor([1.0, 2.0], requires_grad=True), torch.tensor([3.0], dtype=torch.bfloat16)]
    vectors, summed = gather_cp_response_values(values, [3, 2], [2, 1], scalars=[torch.tensor(4.0)])
    assert [v.dtype for v in vectors] == [torch.float32, torch.bfloat16]
    assert all(not v.requires_grad for v in vectors)
    assert torch.equal(vectors[0], torch.tensor([1.0, 2.0])) and float(summed[0]) == 4.0
