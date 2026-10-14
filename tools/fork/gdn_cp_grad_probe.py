"""GatedDeltaNet context-parallel gradient probe (docs/fork/full-finetuning.md, "Context parallelism").

One Megatron ``GatedDeltaNet`` layer (Qwen3.5-4B linear-attention sizes by default) runs on a
packed THD row laid out as miles' ``get_batch`` lays it out: every datum zigzag-sharded over the
CP ranks, each rank's share padded to a multiple of 128, ``cu_seqlens`` of the full row. It runs
once without CP (the reference, on every rank) and once per ``linear_cp_mode``, with the same
weights and inputs, for each CP size. ``--layers attention`` probes a Qwen3.5-4B full-attention
layer (TE attention, qk layernorm, output gate, no rotary) the same way, once per
``--attention-modes`` (``cp_comm_type``).

The loss weights only the part after the first contiguous CP split of the datum that straddles
it (``chunkwise`` mode hands the recurrent state and the conv halo to the next rank there), so
the input gradients before the split, and part of every parameter gradient, reach the earlier
rank only through the cross-rank backward. Half the value heads get slow decays, so the state
carries across the split. For each layout the probe prints, against the reference, the cosine
and norm ratio of every parameter gradient, of the input gradients before and after the split,
and of the outputs. A correct layout matches at the bf16 level; a missing or wrong cross-rank
term shows as a low cosine or a norm ratio away from 1, first on the pre-split input gradients.

Run on one node inside the miles image (Megatron-LM and fla from the image):

  torchrun --nproc_per_node 8 tools/fork/gdn_cp_grad_probe.py --cp 2 4 8 --modes chunkwise headwise
  torchrun --nproc_per_node 8 tools/fork/gdn_cp_grad_probe.py --cp 2 4 --layers gdn attention

Exit status 1 when a layout falls below ``--min-cosine`` or outside ``--max-norm-error``.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import build_module

PAD_SIZE = 128  # TP1 * miles' --data-pad-size-multiplier


def _config(args, layer: str, cp: int, mode: str | None) -> TransformerConfig:
    common = dict(
        num_layers=1,
        hidden_size=args.hidden,
        num_attention_heads=16,
        num_query_groups=4,
        attention_dropout=0.0,  # as the gateway trains (--attention-dropout 0.0 --hidden-dropout 0.0)
        hidden_dropout=0.0,
        cp_partition_mode="zigzag",
        context_parallel_size=cp,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        transformer_impl="transformer_engine",
    )
    if layer == "attention":
        return TransformerConfig(
            **common,
            kv_channels=256,
            qk_layernorm=True,
            add_bias_linear=False,  # Qwen3.5: --disable-bias-linear
            attention_output_gate=not args.no_attention_gate,
            attention_softmax_in_fp32=True,
            attention_backend=AttnBackend.flash,
            cp_comm_type=mode or "p2p",
        )
    return TransformerConfig(
        **common,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=args.key_heads,
        linear_num_value_heads=args.value_heads,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        linear_cp_mode=mode or "chunkwise",
    )


def _build_layer(args, layer: str, cp: int, mode: str | None):
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(args.seed)
    config = _config(args, layer, cp, mode)
    if layer == "attention":
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True).submodules.self_attention
    else:
        spec = get_experimental_attention_variant_module_spec(config)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return build_module(spec, config=config, layer_number=1, pg_collection=pg_collection)


def _global_cu(lengths: list[int], cp: int) -> list[int]:
    """cu_seqlens of the full row get_batch builds at this CP size: each datum padded to 2*cp,
    each rank's share padded to PAD_SIZE (one trailing pad segment)."""
    padded = [-(-n // (2 * cp)) * 2 * cp for n in lengths]
    local = sum(padded) // cp
    pad = (PAD_SIZE - local % PAD_SIZE) % PAD_SIZE
    cu = [0]
    for length in padded + ([pad * cp] if pad else []):
        cu.append(cu[-1] + length)
    return cu


def _local_positions(cu: list[int], cp: int, rank: int) -> torch.Tensor:
    """The global row position of every local row: chunks rank and 2*cp-1-rank of each segment."""
    if cp == 1:
        return torch.arange(cu[-1])
    positions = []
    for start, end in zip(cu[:-1], cu[1:], strict=True):
        chunk = (end - start) // (2 * cp)
        for index in (rank, 2 * cp - 1 - rank):
            positions.append(torch.arange(start + index * chunk, start + (index + 1) * chunk))
    return torch.cat(positions)


def _forward_backward(args, layer_type: str, cp: int, mode, state: dict, row, probe, loss_mask, cu: list[int]):
    """One layout's forward/backward. Returns (parameter gradients summed over the CP group, the
    global positions of this rank's rows, their input gradients, their outputs)."""
    layer = _build_layer(args, layer_type, cp, mode).cuda()
    layer.load_state_dict(state)
    positions = _local_positions(cu, cp, parallel_state.get_context_parallel_rank()).cuda()
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
    max_seqlen = int((cu_t[1:] - cu_t[:-1]).max())
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu_t, cu_seqlens_kv=cu_t, max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen
    )
    local_input = row[positions].detach().clone().requires_grad_(True)
    output, bias = layer(local_input, attention_mask=None, packed_seq_params=packed)
    if bias is not None:
        output = output + bias
    loss = ((output.float().squeeze(1) @ probe) * loss_mask[positions]).sum()
    loss.backward()
    grads = {name: p.grad.detach().float().clone() for name, p in layer.named_parameters() if p.grad is not None}
    if cp > 1:
        for grad in grads.values():
            dist.all_reduce(grad, group=parallel_state.get_context_parallel_group())
    return grads, positions, local_input.grad.detach().float().squeeze(1), output.detach().float().squeeze(1)


def _rows_stats(cand: torch.Tensor, ref: torch.Tensor, cp: int) -> dict:
    """Cosine and norm ratio over rows spread across the CP group (sums reduced over its ranks)."""
    sums = torch.stack([(cand * ref).sum(), (cand * cand).sum(), (ref * ref).sum()]).double()
    if cp > 1:
        dist.all_reduce(sums, group=parallel_state.get_context_parallel_group())
    dot, cand_sq, ref_sq = sums.tolist()
    if ref_sq == 0.0:
        return {"cosine": 1.0 if cand_sq == 0.0 else 0.0, "norm_ratio": 1.0 if cand_sq == 0.0 else math.inf}
    return {"cosine": dot / math.sqrt(max(cand_sq * ref_sq, 1e-300)), "norm_ratio": math.sqrt(cand_sq / ref_sq)}


def _tensor_stats(cand: torch.Tensor, ref: torch.Tensor) -> dict:
    cand, ref = cand.double().flatten(), ref.double().flatten()
    if ref.norm() == 0:
        return {"cosine": 1.0 if cand.norm() == 0 else 0.0, "norm_ratio": 1.0 if cand.norm() == 0 else math.inf}
    return {"cosine": float(F.cosine_similarity(cand, ref, dim=0)), "norm_ratio": float(cand.norm() / ref.norm())}


def _initial_state(args, layer: str) -> dict:
    """Weights every layout loads. GatedDeltaNet: half the value heads remember ~1000 tokens
    (A ~ 1e-3), so the recurrent state matters across the split; the rest keep their
    initialization (fast decays). Attention: the q/k layernorm gains scale the logits'
    standard deviation to ``--attention-sharpness`` (1 at initialization spreads each query over
    thousands of keys, unlike a trained model, and turns bf16 rounding of the near-cancelling
    partial results into large relative errors)."""
    torch.manual_seed(args.seed)
    built = _build_layer(args, layer, 1, None)
    state = {key: value.detach().cpu().clone() for key, value in built.state_dict().items()}
    if "A_log" in state:
        a_log = state["A_log"].clone()
        a_log[: a_log.numel() // 2] = math.log(1e-3)
        state["A_log"] = a_log
    gain = math.sqrt(args.attention_sharpness)  # q and k each: logits scale with the product
    for key in ("q_layernorm.weight", "k_layernorm.weight"):
        if key in state:  # zero-centered gamma: the weight stores gain - 1
            state[key] = state[key] + (gain - 1.0)
    return state


def _probe_cp(args, layer: str, modes: list[str], cp: int, state: dict, probe: torch.Tensor) -> list[dict]:
    cu = _global_cu(args.lengths, cp)
    total, split = cu[-1], cu[-1] // cp
    datum = next(i for i in range(len(args.lengths)) if cu[i] < split < cu[i] + args.lengths[i])
    start, end = cu[datum], cu[datum] + args.lengths[datum]
    generator = torch.Generator().manual_seed(args.seed + cp)
    row = (torch.randn(total, 1, args.hidden, generator=generator) * 0.5).to(torch.bfloat16).cuda()
    loss_mask = torch.zeros(total, device="cuda")
    loss_mask[split:end] = 1.0  # the straddling datum after the split only
    region = torch.zeros(total, dtype=torch.long, device="cuda")
    region[start:split], region[split:end] = 1, 2
    ref_grads, _, ref_input_grad, ref_out = _forward_backward(args, layer, 1, None, state, row, probe, loss_mask, cu)
    records = []
    for mode in modes:
        grads, positions, input_grad, output = _forward_backward(
            args, layer, cp, mode, state, row, probe, loss_mask, cu
        )
        local_region, ref_in = region[positions], ref_input_grad[positions]
        records.append(
            {
                "layer": layer,
                "cp": cp,
                "mode": mode,
                "straddling_datum": {"index": datum, "tokens": args.lengths[datum], "split_row": split - start},
                "input_grad_before_split": _rows_stats(input_grad[local_region == 1], ref_in[local_region == 1], cp),
                "input_grad_after_split": _rows_stats(input_grad[local_region == 2], ref_in[local_region == 2], cp),
                "output": _rows_stats(output, ref_out[positions], cp),
                "params": {name: _tensor_stats(grads[name], ref_grads[name]) for name in sorted(ref_grads)},
                "output_rows_off": _rows_off(output, ref_out[positions], positions, cu, cp),
            }
        )
    return records


def _rows_off(output: torch.Tensor, ref: torch.Tensor, positions: torch.Tensor, cu: list[int], cp: int) -> dict:
    """Where the outputs disagree: rows whose cosine against the reference is below 0.99, counted per
    (segment, zigzag chunk) over the whole CP group, with the first few global positions."""
    off = positions[F.cosine_similarity(output, ref, dim=1) < 0.99].tolist()
    gathered = [off]
    if cp > 1:
        gathered = [None] * cp
        dist.all_gather_object(gathered, off, group=parallel_state.get_context_parallel_group())
    rows = sorted(row for part in gathered for row in part)
    by_chunk: dict[str, int] = {}
    for row in rows:
        segment = max(i for i in range(len(cu) - 1) if cu[i] <= row)
        chunk = (row - cu[segment]) // ((cu[segment + 1] - cu[segment]) // (2 * cp))
        key = f"segment{segment}/chunk{chunk}"
        by_chunk[key] = by_chunk.get(key, 0) + 1
    return {"count": len(rows), "of": cu[-1], "by_segment_chunk": by_chunk, "first": rows[:16]}


def _report(results: list[dict], args) -> list[str]:
    failures = []
    for record in results:
        label = f"{record['layer']} cp{record['cp']} {record['mode']}"
        datum = record["straddling_datum"]
        print(
            f"== {label}: datum {datum['index']} ({datum['tokens']} tokens), CP split at its row {datum['split_row']}"
        )
        rows = [(key, record[key]) for key in ("input_grad_before_split", "input_grad_after_split", "output")]
        rows += [(f"grad {name}", stats) for name, stats in record["params"].items()]
        for name, stats in rows:
            bad = stats["cosine"] < args.min_cosine or abs(stats["norm_ratio"] - 1.0) > args.max_norm_error
            mark = "  <--" if bad else ""
            print(f"  {name:48s} cosine {stats['cosine']:.6f}  norm ratio {stats['norm_ratio']:.5f}{mark}")
            if bad:
                failures.append(f"{label} {name}: cosine {stats['cosine']:.6f}, norm ratio {stats['norm_ratio']:.5f}")
        off = record["output_rows_off"]
        print(
            f"  output rows below cosine 0.99: {off['count']} of {off['of']} {off['by_segment_chunk']} first {off['first']}"
        )
    print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    sys.stdout.flush()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cp", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--layers", nargs="+", default=["gdn"], choices=["gdn", "attention"])
    parser.add_argument("--modes", nargs="+", default=["chunkwise", "headwise"], choices=["chunkwise", "headwise"])
    parser.add_argument("--attention-modes", nargs="+", default=["p2p"], choices=["p2p", "a2a", "all_gather"])
    parser.add_argument("--attention-sharpness", type=float, default=1.0, help="std of the attention logits")
    parser.add_argument("--no-attention-gate", action="store_true", help="drop Qwen3.5's attention output gate")
    # the datums (tokens) of one packed row: the 8000-token one straddles the first CP split
    parser.add_argument("--lengths", type=int, nargs="+", default=[1000, 8000, 3000])
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-norm-error", type=float, default=0.01)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    assert all(world % cp == 0 for cp in args.cp), f"every --cp must divide the {world} ranks"

    probe = torch.randn(args.hidden, generator=torch.Generator().manual_seed(args.seed + 100)).cuda()
    results = []
    for layer in args.layers:
        state = _initial_state(args, layer)
        modes = args.attention_modes if layer == "attention" else args.modes
        results += [record for cp in args.cp for record in _probe_cp(args, layer, modes, cp, state, probe)]

    failures = _report(results, args) if dist.get_rank() == 0 else []
    if dist.get_rank() == 0 and args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump({"args": vars(args), "results": results, "failures": failures}, handle, indent=1)
    status = torch.tensor([1 if failures else 0], device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MAX)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    return int(status.item())


if __name__ == "__main__":
    raise SystemExit(main())
