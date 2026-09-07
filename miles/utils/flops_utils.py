"""Forward-FLOPs model behind the ``perf/actor_train_*`` and Tinker ``actor_train_*`` metrics.

Per layer: QKV projections, causal softmax attention (or the Gated DeltaNet
linear-attention terms for ``linear_attention`` layers), the output projection
and a SwiGLU FFN (MoE layers count the active experts + shared expert), plus the
lm head; ``docs/fork/mfu.md`` spells out the formulas. The optional attributes
``flops_layer_types`` / ``flops_linear_attention`` (filled from the HF config by
``ensure_hf_layer_pattern``), ``attention_output_gate`` and
``enable_mtp_training`` / ``mtp_num_layers`` refine the model; without them the
dense / MoE formulas are unchanged.
"""

import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
LINEAR_ATTENTION_DIMS = ("num_k_heads", "num_v_heads", "k_dim", "v_dim", "conv_kernel")
_HF_LINEAR_ATTENTION_DIMS = {
    "num_k_heads": "linear_num_key_heads",
    "num_v_heads": "linear_num_value_heads",
    "k_dim": "linear_key_head_dim",
    "v_dim": "linear_value_head_dim",
    "conv_kernel": "linear_conv_kernel_dim",
}


def calculate_embedding_flops(seqlen, hidden_size):
    return 2 * seqlen * hidden_size


def calculate_lm_head_flops(seqlen, hidden_size, vocab_size):
    return 2 * seqlen * hidden_size * vocab_size


def calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups):
    if args.q_lora_rank is None:
        q_flops = 2 * seqlen * hidden_size * num_attention_heads * args.kv_channels
        if getattr(args, "attention_output_gate", False):
            # Qwen3.5/3.6 ``attn_output_gate``: the q projection also emits a
            # per-head gate of the same width (gate * attention output).
            q_flops *= 2
    else:
        q_flops = (
            2
            * seqlen
            * args.q_lora_rank
            * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim))
        )
    if args.kv_lora_rank is None:
        kv_flops = 2 * 2 * seqlen * hidden_size * num_query_groups * args.kv_channels
    else:
        kv_flops = (
            2
            * seqlen
            * (
                args.kv_lora_rank
                * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim))
                + args.hidden_size * args.qk_pos_emb_head_dim
            )
        )

    return q_flops + kv_flops


def calculate_attention_flops(args, seqlen, num_attention_heads):
    # QK^T with causal
    if args.qk_pos_emb_head_dim:
        flops = 2 * num_attention_heads * seqlen * seqlen * (args.qk_head_dim + args.qk_pos_emb_head_dim) / 2
    else:
        flops = 2 * num_attention_heads * seqlen * seqlen * args.kv_channels / 2
    # A*V
    if args.v_head_dim:
        flops += num_attention_heads * seqlen * seqlen * args.v_head_dim
    else:
        flops += num_attention_heads * seqlen * seqlen * args.kv_channels
    return flops


def calculate_output_flops(seqlen, hidden_size):
    return 2 * seqlen * hidden_size * hidden_size


def calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size):
    return 2 * seqlen * hidden_size * ffn_hidden_size * 3


def calculate_linear_attention_flops(seqlen, hidden_size, linear):
    """Gated DeltaNet (Qwen3.5/3.6 ``linear_attention`` layers): projections,
    short causal conv, recurrent state update + readout, output projection.
    Linear in ``seqlen``; ``linear`` carries ``LINEAR_ATTENTION_DIMS``."""
    num_k_heads, num_v_heads = linear["num_k_heads"], linear["num_v_heads"]
    k_dim, v_dim, conv_kernel = linear["k_dim"], linear["v_dim"], linear["conv_kernel"]
    key_width = num_k_heads * k_dim
    value_width = num_v_heads * v_dim
    qk_proj = 2 * seqlen * hidden_size * (2 * key_width)  # in_proj q, k
    vz_proj = 2 * seqlen * hidden_size * (2 * value_width)  # in_proj v, z (gate)
    ab_proj = 2 * seqlen * hidden_size * 2 * num_v_heads  # in_proj a (decay), b (beta)
    conv = 2 * seqlen * (2 * key_width + value_width) * conv_kernel
    recurrence = 4 * seqlen * num_v_heads * k_dim * v_dim  # state update + readout
    out_proj = 2 * seqlen * value_width * hidden_size
    return qk_proj + vz_proj + ab_proj + conv + recurrence + out_proj


def calculate_layer_flops(
    args,
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    ffn_hidden_size,
    attention_kind=FULL_ATTENTION,
):
    if attention_kind == LINEAR_ATTENTION:
        attention = calculate_linear_attention_flops(seqlen, hidden_size, args.flops_linear_attention)
    elif attention_kind == FULL_ATTENTION:
        attention = (
            calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups)
            + calculate_attention_flops(args, seqlen, num_attention_heads)
            + calculate_output_flops(seqlen, hidden_size)
        )
    else:
        raise ValueError(
            f"unknown attention kind {attention_kind!r}; expected {FULL_ATTENTION!r} or {LINEAR_ATTENTION!r}"
        )
    return attention + calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size)


def _ffn_widths(args):
    """``(dense_ffn, moe_ffn)``; ``moe_ffn`` counts the active experts + shared expert (None for dense)."""
    dense_ffn = args.ffn_hidden_size
    if args.num_experts is None:
        return dense_ffn, None
    shared_expert_ffn = getattr(args, "moe_shared_expert_intermediate_size", None)
    if shared_expert_ffn is None:
        shared_expert_ffn = 0
    return dense_ffn, args.moe_ffn_hidden_size * args.moe_router_topk + shared_expert_ffn


def _layer_is_moe(args, layer_index):
    if args.num_experts is None:
        return False
    if not hasattr(args, "moe_layer_freq"):
        return True
    if isinstance(args.moe_layer_freq, list):
        return args.moe_layer_freq[layer_index] > 0
    return layer_index % args.moe_layer_freq == 0


def _bucketed_layer_counts(args):
    """``(num_dense_layers, num_moe_layers)`` of the layer_types-agnostic path."""
    if args.num_experts is None:
        return args.num_layers, 0
    if hasattr(args, "moe_layer_freq"):
        if isinstance(args.moe_layer_freq, list):
            num_dense_layers = sum(1 for freq in args.moe_layer_freq if freq == 0)
            num_moe_layers = sum(1 for freq in args.moe_layer_freq if freq > 0)
        else:
            num_dense_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq != 0)
            num_moe_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq == 0)
        return num_dense_layers, num_moe_layers
    return 0, args.num_layers


def _layer_types(args):
    """The per-layer attention kinds, or None when the model has no linear-attention layers declared."""
    layer_types = getattr(args, "flops_layer_types", None)
    if layer_types is None:
        return None
    if len(layer_types) != args.num_layers:
        raise ValueError(f"flops_layer_types has {len(layer_types)} entries for num_layers={args.num_layers}")
    if LINEAR_ATTENTION in layer_types and not getattr(args, "flops_linear_attention", None):
        raise ValueError("flops_layer_types declares linear_attention layers but flops_linear_attention is unset")
    return list(layer_types)


def _mtp_num_layers(args):
    if not getattr(args, "enable_mtp_training", False):
        return 0
    return int(getattr(args, "mtp_num_layers", None) or 0)


def calculate_mtp_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size, vocab_size):
    """One MTP depth: a full-attention transformer layer, ``eh_proj`` (2H -> H),
    the (shared) embedding and one more lm-head pass on the shifted tokens."""
    layer = calculate_layer_flops(
        args, seqlen, hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size, FULL_ATTENTION
    )
    eh_proj = 2 * seqlen * (2 * hidden_size) * hidden_size
    return (
        layer
        + eh_proj
        + calculate_embedding_flops(seqlen, hidden_size)
        + calculate_lm_head_flops(seqlen, hidden_size, vocab_size)
    )


def calculate_fwd_flops(
    seqlens,
    args,
):
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    vocab_size = args.vocab_size

    total_flops = 0

    dense_ffn, moe_ffn = _ffn_widths(args)
    layer_types = _layer_types(args)
    if layer_types is None:
        num_dense_layers, num_moe_layers = _bucketed_layer_counts(args)
    mtp_num_layers = _mtp_num_layers(args)

    def layer(seqlen, ffn_hidden_size, attention_kind=FULL_ATTENTION):
        return calculate_layer_flops(
            args, seqlen, hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size, attention_kind
        )

    for seqlen in seqlens:
        if layer_types is not None:
            for i, attention_kind in enumerate(layer_types):
                total_flops += layer(seqlen, moe_ffn if _layer_is_moe(args, i) else dense_ffn, attention_kind)
        else:
            if num_dense_layers > 0:
                total_flops += layer(seqlen, dense_ffn) * num_dense_layers

            if num_moe_layers > 0:
                total_flops += layer(seqlen, moe_ffn) * num_moe_layers

        total_flops += calculate_lm_head_flops(seqlen, hidden_size, vocab_size)

        if mtp_num_layers:
            total_flops += mtp_num_layers * calculate_mtp_flops(
                args,
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                moe_ffn if args.num_experts is not None else dense_ffn,
                vocab_size,
            )

    return total_flops


def fwd_tflops_per_gpu(seqlens, args, world_size):
    return calculate_fwd_flops(seqlens, args) / world_size / 1e12


def _first(config, *names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _moe_layer_pattern(config, num_layers, num_experts):
    if num_experts is None:
        return None
    first_dense = getattr(config, "first_k_dense_replace", None)
    if first_dense is not None:
        return [0 if i < first_dense else 1 for i in range(num_layers)]
    mlp_only = set(getattr(config, "mlp_only_layers", None) or ())
    step = getattr(config, "decoder_sparse_step", 1) or 1
    return [0 if i in mlp_only or (i + 1) % step != 0 else 1 for i in range(num_layers)]


def flops_args_from_hf_config(config):
    getter = getattr(config, "get_text_config", None)
    config = (getter() if callable(getter) else getattr(config, "text_config", None)) or config

    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_layers = _first(config, "num_hidden_layers", "num_layers")
    assert num_layers is not None, f"no layer count on {type(config).__name__}; cannot size the FLOPs model"
    num_experts = _first(config, "n_routed_experts", "num_experts", "num_local_experts")

    dense_ffn = getattr(config, "intermediate_size", None)

    moe_ffn = getattr(config, "moe_intermediate_size", None)
    if num_experts is not None and moe_ffn is None:
        moe_ffn = dense_ffn

    shared_ffn = getattr(config, "shared_expert_intermediate_size", None)
    if shared_ffn is None and getattr(config, "n_shared_experts", None):
        shared_ffn = config.n_shared_experts * moe_ffn

    moe_layer_freq = _moe_layer_pattern(config, num_layers, num_experts)
    needs_dense = moe_layer_freq is None or any(f == 0 for f in moe_layer_freq)
    needs_moe = moe_layer_freq is not None and any(f > 0 for f in moe_layer_freq)
    if (needs_dense and dense_ffn is None) or (needs_moe and moe_ffn is None):
        raise ValueError(
            f"{type(config).__name__} does not expose the FFN widths the FLOPs model needs "
            f"(dense={dense_ffn}, moe={moe_ffn}); it likely nests them under a sub-config"
        )

    layer_types, linear_attention = layer_pattern_from_hf_config(config)
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=_first(config, "num_key_value_heads", default=num_attention_heads),
        vocab_size=config.vocab_size,
        num_layers=num_layers,
        ffn_hidden_size=dense_ffn,
        kv_channels=_first(config, "head_dim", default=hidden_size // num_attention_heads),
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn,
        moe_router_topk=_first(config, "num_experts_per_tok", "moe_topk", default=1),
        moe_shared_expert_intermediate_size=shared_ffn,
        moe_layer_freq=moe_layer_freq,
        q_lora_rank=getattr(config, "q_lora_rank", None),
        kv_lora_rank=getattr(config, "kv_lora_rank", None),
        qk_head_dim=getattr(config, "qk_nope_head_dim", None) or 0,
        qk_pos_emb_head_dim=getattr(config, "qk_rope_head_dim", None) or 0,
        v_head_dim=getattr(config, "v_head_dim", None) or 0,
        attention_output_gate=bool(getattr(config, "attn_output_gate", False)),
        flops_layer_types=layer_types,
        flops_linear_attention=linear_attention,
    )


def _text_config(config):
    getter = getattr(config, "get_text_config", None)
    return (getter() if callable(getter) else getattr(config, "text_config", None)) or config


def layer_pattern_from_hf_config(config):
    """``(layer_types, linear_attention_dims)`` for the FLOPs model from an HF (text) config.

    ``layer_types`` comes from the config or, for Qwen3.5-style configs without
    it, from ``full_attention_interval`` (every ``interval``-th layer is full
    attention, like ``miles_plugins/models/qwen3_5.py``). Kinds other than
    ``linear_attention`` count as full attention. Returns ``(None, None)`` for
    dense / plain-MoE configs (no linear-attention layers), which keeps the
    layer_types-agnostic formula.
    """
    config = _text_config(config)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        interval = getattr(config, "full_attention_interval", None)
        if not interval:
            return None, None
        num_layers = _first(config, "num_hidden_layers", "num_layers")
        layer_types = [FULL_ATTENTION if (i + 1) % interval == 0 else LINEAR_ATTENTION for i in range(num_layers)]
    layer_types = [LINEAR_ATTENTION if kind == LINEAR_ATTENTION else FULL_ATTENTION for kind in layer_types]
    if LINEAR_ATTENTION not in layer_types:
        return None, None
    linear = {name: getattr(config, hf_name, None) for name, hf_name in _HF_LINEAR_ATTENTION_DIMS.items()}
    missing = [_HF_LINEAR_ATTENTION_DIMS[name] for name, value in linear.items() if value is None]
    if missing:
        raise ValueError(f"{type(config).__name__} declares linear_attention layers but lacks {missing}")
    return layer_types, linear


def _load_hf_config(hf_checkpoint):
    # Deferred: transformers is a heavy import and only the perf paths that read
    # the HF config need it (dp_schedule imports this module on the rollout side).
    from miles.utils.hf_config import load_hf_config

    return load_hf_config(hf_checkpoint, trust_remote_code=True)


def ensure_hf_layer_pattern(args) -> bool:
    """Fill ``args.flops_layer_types`` / ``args.flops_linear_attention`` once from the HF config.

    No-op when the attributes are already set, when ``args.hf_checkpoint`` is
    unset, or after a first attempt (success or failure) - the result is cached
    on ``args``. Never raises: a config that cannot be read or understood logs a
    warning and leaves the layer_types-agnostic formula in place. Returns True
    when linear-attention layers are declared on ``args`` afterwards.
    """
    if getattr(args, "flops_layer_types", None) is not None:
        return True
    if getattr(args, "flops_hf_layer_pattern_checked", False):
        return False
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        return False
    args.flops_hf_layer_pattern_checked = True
    try:
        layer_types, linear = layer_pattern_from_hf_config(_load_hf_config(hf_checkpoint))
        num_layers = getattr(args, "num_layers", None)
        if layer_types is not None and num_layers is not None and len(layer_types) != num_layers:
            raise ValueError(f"HF config has {len(layer_types)} layers but args.num_layers is {num_layers}")
    except Exception as exc:  # noqa: BLE001 - perf logging must never take the trainer down
        logger.warning(
            "MFU: could not read the layer pattern from %s (%s); attention FLOPs use the "
            "full-attention formula on every layer",
            hf_checkpoint,
            exc,
        )
        return False
    args.flops_layer_types = layer_types
    args.flops_linear_attention = linear
    if layer_types is not None:
        logger.info(
            "MFU: %d linear-attention + %d full-attention layers from %s",
            layer_types.count(LINEAR_ATTENTION),
            layer_types.count(FULL_ATTENTION),
            hf_checkpoint,
        )
    return layer_types is not None
