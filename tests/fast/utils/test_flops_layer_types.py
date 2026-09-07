"""layer_types-aware FLOPs model: dense/MoE goldens unchanged, GDN + output gate + MTP for Qwen3.6."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from miles.utils import flops_utils
from miles.utils.flops_utils import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    calculate_embedding_flops,
    calculate_fwd_flops,
    calculate_layer_flops,
    calculate_linear_attention_flops,
    calculate_lm_head_flops,
    calculate_mlp_flops,
    ensure_hf_layer_pattern,
    flops_args_from_hf_config,
    layer_pattern_from_hf_config,
)

S64K = 65536
QWEN36_LINEAR = dict(num_k_heads=16, num_v_heads=32, k_dim=128, v_dim=128, conv_kernel=4)
QWEN36_LAYER_TYPES = [FULL_ATTENTION if (i + 1) % 4 == 0 else LINEAR_ATTENTION for i in range(40)]
# megatron/training/arguments.py defaults, present on every trainer Namespace whether or not the model is MLA
MEGATRON_MLA_DEFAULTS = dict(
    multi_latent_attention=False,
    q_lora_rank=None,
    kv_lora_rank=32,
    qk_head_dim=128,
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
)
ZEROED_MLA_FIELDS = dict(q_lora_rank=None, kv_lora_rank=None, qk_head_dim=0, qk_pos_emb_head_dim=0, v_head_dim=0)
DEEPSEEK_MLA = dict(
    multi_latent_attention=True,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_head_dim=128,
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
)


def dense_args(**overrides):
    base = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=4,
        vocab_size=152064,
        num_layers=8,
        ffn_hidden_size=8192,
        kv_channels=128,
        num_experts=None,
        moe_ffn_hidden_size=None,
        moe_router_topk=1,
        moe_shared_expert_intermediate_size=None,
        **MEGATRON_MLA_DEFAULTS,
    )
    return SimpleNamespace(**(base | overrides))


def qwen36_args(**overrides):
    """scripts/models/qwen3.6-35B-A3B.py as Megatron args (+ the optional FLOPs attributes)."""
    base = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        vocab_size=248320,
        num_layers=40,
        ffn_hidden_size=512,
        kv_channels=256,
        num_experts=256,
        moe_ffn_hidden_size=512,
        moe_router_topk=8,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        **MEGATRON_MLA_DEFAULTS,
        attention_output_gate=True,
        flops_layer_types=None,
        flops_linear_attention=None,
        enable_mtp_training=False,
        mtp_num_layers=None,
    )
    return SimpleNamespace(**(base | overrides))


def hand_full_attention_layer(args, seqlen, ffn):
    """The pre-existing per-layer formula, written out (q width doubled by the output gate)."""
    h, heads, groups, kv = args.hidden_size, args.num_attention_heads, args.num_query_groups, args.kv_channels
    q = 2 * seqlen * h * heads * kv * (2 if args.attention_output_gate else 1)
    k_v = 2 * 2 * seqlen * h * groups * kv
    scores = 2 * heads * seqlen * seqlen * kv / 2
    av = heads * seqlen * seqlen * kv
    out = 2 * seqlen * h * h
    mlp = 2 * seqlen * h * ffn * 3
    return q + k_v + scores + av + out + mlp


def hand_linear_attention_layer(args, seqlen, ffn):
    """Section 5.2 of the design, written out for Qwen3.6's GDN dims."""
    h = args.hidden_size
    kh, kd, vh, vd, conv = 16, 128, 32, 128, 4
    return (
        2 * seqlen * h * (2 * kh * kd)
        + 2 * seqlen * h * (2 * vh * vd)
        + 2 * seqlen * h * 2 * vh
        + 2 * seqlen * (2 * kh * kd + vh * vd) * conv
        + 4 * seqlen * vh * kd * vd
        + 2 * seqlen * vh * vd * h
        + 2 * seqlen * h * ffn * 3
    )


# --- (a) dense golden unchanged -------------------------------------------------------------


def test_dense_golden_is_unchanged():
    args = dense_args()
    seqlens = [1024, 3072]
    expected = sum(
        8 * (2 * s * 2048 * 16 * 128 + 4 * s * 2048 * 4 * 128 + 16 * s * s * 128 + 16 * s * s * 128 + 2 * s * 2048**2)
        + 8 * 2 * s * 2048 * 8192 * 3
        + 2 * s * 2048 * 152064
        for s in seqlens
    )
    assert calculate_fwd_flops(seqlens, args) == expected


def test_dense_args_without_the_new_attributes_still_work():
    args = dense_args()
    assert not hasattr(args, "flops_layer_types")
    assert calculate_fwd_flops([2048], args) == calculate_fwd_flops(
        [2048], dense_args(flops_layer_types=None, attention_output_gate=False, enable_mtp_training=False)
    )


def test_output_gate_only_changes_the_q_projection():
    gated, plain = dense_args(attention_output_gate=True), dense_args()
    assert calculate_fwd_flops([1024], gated) - calculate_fwd_flops([1024], plain) == 8 * 2 * 1024 * 2048 * 16 * 128


# --- (b) MoE with flops_layer_types=None unchanged --------------------------------------------


def test_moe_without_layer_types_is_the_two_bucket_formula():
    args = qwen36_args(attention_output_gate=False)
    moe_ffn = 512 * 8 + 512
    expected = sum(40 * hand_full_attention_layer(args, s, moe_ffn) + 2 * s * 2048 * 248320 for s in [4096, S64K])
    assert calculate_fwd_flops([4096, S64K], args) == expected


def test_moe_int_layer_freq_and_missing_freq_keep_their_bucket_rule():
    every_second = dense_args(num_experts=64, moe_ffn_hidden_size=512, moe_router_topk=8, moe_layer_freq=2)
    all_moe = dense_args(num_experts=64, moe_ffn_hidden_size=512, moe_router_topk=8)
    plain = dense_args(attention_output_gate=False)
    dense_layer = hand_full_attention_layer(plain, 1024, 8192)
    moe_layer = hand_full_attention_layer(plain, 1024, 512 * 8)
    head = 2 * 1024 * 2048 * 152064
    assert dense_layer != moe_layer
    assert calculate_fwd_flops([1024], every_second) == 4 * dense_layer + 4 * moe_layer + head
    assert calculate_fwd_flops([1024], all_moe) == 8 * moe_layer + head


def test_per_layer_path_matches_the_bucket_path_when_every_layer_is_full_attention():
    bucket = qwen36_args()
    per_layer = qwen36_args(flops_layer_types=[FULL_ATTENTION] * 40)
    assert calculate_fwd_flops([4096, S64K], per_layer) == calculate_fwd_flops([4096, S64K], bucket)


# --- (c) Qwen3.6 at 64k: hand formula and old/new ratio ---------------------------------------


def test_qwen36_linear_attention_layer_matches_the_hand_formula():
    args = qwen36_args(flops_linear_attention=QWEN36_LINEAR)
    got = calculate_layer_flops(args, S64K, 2048, 16, 2, 4608, LINEAR_ATTENTION)
    assert got == hand_linear_attention_layer(args, S64K, 4608)
    assert calculate_linear_attention_flops(S64K, 2048, QWEN36_LINEAR) / S64K == pytest.approx(69.53e6, rel=1e-3)


def test_qwen36_64k_total_matches_the_hand_formula():
    args = qwen36_args(flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR)
    moe_ffn = 512 * 8 + 512
    expected = (
        30 * hand_linear_attention_layer(args, S64K, moe_ffn)
        + 10 * hand_full_attention_layer(args, S64K, moe_ffn)
        + 2 * S64K * 2048 * 248320
    )
    assert calculate_fwd_flops([S64K], args) == expected
    assert expected / S64K == pytest.approx(11.20e9, rel=1e-3)  # docs/fork/mfu.md


def test_megatron_mla_defaults_do_not_leak_into_a_non_mla_model():
    """The trainer passes its raw Megatron Namespace (kv_lora_rank=32, qk_pos_emb_head_dim=64,
    v_head_dim=128 by default); only kv_channels=256 prices Qwen3.6's 10 full-attention layers."""
    real = qwen36_args(flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR)
    zeroed = qwen36_args(
        flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR, **ZEROED_MLA_FIELDS
    )
    del zeroed.multi_latent_attention
    assert real.kv_lora_rank == 32 and real.qk_pos_emb_head_dim == 64 and real.v_head_dim == 128
    for s in (2048, 16384, S64K):
        assert calculate_fwd_flops([s], real) == calculate_fwd_flops([s], zeroed)
    assert calculate_fwd_flops([S64K], real) / S64K == pytest.approx(11.20e9, rel=1e-3)


def test_mla_formulas_stay_live_for_deepseek_style_args():
    plain = dense_args(attention_output_gate=False)
    mla = dense_args(attention_output_gate=False, **DEEPSEEK_MLA)
    s = 4096
    h, heads, groups = 2048, 16, 4
    q = 2 * s * 1536 * (h + heads * (128 + 64))
    kv = 2 * s * (512 * (h + heads * (128 + 128)) + h * 64)
    scores = 2 * heads * s * s * (128 + 64) / 2
    av = heads * s * s * 128
    plain_attention = (
        2 * s * h * heads * 128 + 2 * 2 * s * h * groups * 128 + 2 * heads * s * s * 128 / 2 + heads * s * s * 128
    )
    assert calculate_fwd_flops([s], mla) - calculate_fwd_flops([s], plain) == 8 * (
        q + kv + scores + av - plain_attention
    )
    # the switch alone decides: the same lora/head dims with multi_latent_attention=False price as GQA
    assert calculate_fwd_flops(
        [s], dense_args(attention_output_gate=False, **{**DEEPSEEK_MLA, "multi_latent_attention": False})
    ) == calculate_fwd_flops([s], plain)
    # namespaces without the switch (HF adapter output, hand-built args) fall back to kv_lora_rank being set
    legacy = dense_args(
        attention_output_gate=False, **{k: v for k, v in DEEPSEEK_MLA.items() if k != "multi_latent_attention"}
    )
    del legacy.multi_latent_attention
    assert calculate_fwd_flops([s], legacy) == calculate_fwd_flops([s], mla)


def test_qwen36_64k_old_over_new_ratio_is_in_band():
    old = calculate_fwd_flops([S64K], qwen36_args())
    new = calculate_fwd_flops(
        [S64K], qwen36_args(flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR)
    )
    assert 2.2 <= old / new <= 2.6


def test_per_layer_path_honours_the_moe_layer_freq_list():
    mixed = qwen36_args(
        moe_layer_freq=[0] * 4 + [1] * 36, flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR
    )
    all_moe = qwen36_args(flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR)
    # layers 0-2 linear, 3 full: each dense layer trades the 4608-wide FFN for the 512-wide one
    delta = 4 * (calculate_mlp_flops(S64K, 2048, 4608) - calculate_mlp_flops(S64K, 2048, 512))
    assert calculate_fwd_flops([S64K], all_moe) - calculate_fwd_flops([S64K], mixed) == delta


def test_layer_types_length_and_missing_linear_dims_are_rejected():
    with pytest.raises(ValueError, match="entries for num_layers"):
        calculate_fwd_flops([16], qwen36_args(flops_layer_types=[FULL_ATTENTION] * 39))
    with pytest.raises(ValueError, match="flops_linear_attention is unset"):
        calculate_fwd_flops([16], qwen36_args(flops_layer_types=QWEN36_LAYER_TYPES))
    with pytest.raises(ValueError, match="unknown attention kind"):
        calculate_layer_flops(qwen36_args(), 16, 2048, 16, 2, 4608, "sliding_attention")


# --- (d) MTP term -----------------------------------------------------------------------------


def test_mtp_term_is_one_layer_plus_eh_proj_embedding_and_lm_head():
    base = qwen36_args(flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR)
    with_mtp = qwen36_args(
        flops_layer_types=QWEN36_LAYER_TYPES,
        flops_linear_attention=QWEN36_LINEAR,
        enable_mtp_training=True,
        mtp_num_layers=1,
    )
    s = S64K
    expected = (
        hand_full_attention_layer(base, s, 512 * 8 + 512)
        + 2 * s * (2 * 2048) * 2048
        + calculate_embedding_flops(s, 2048)
        + calculate_lm_head_flops(s, 2048, 248320)
    )
    assert calculate_fwd_flops([s], with_mtp) - calculate_fwd_flops([s], base) == expected


def test_mtp_needs_both_the_switch_and_a_depth():
    base = calculate_fwd_flops([1024], qwen36_args())
    assert calculate_fwd_flops([1024], qwen36_args(enable_mtp_training=True, mtp_num_layers=0)) == base
    assert calculate_fwd_flops([1024], qwen36_args(enable_mtp_training=False, mtp_num_layers=1)) == base
    assert calculate_fwd_flops([1024], qwen36_args(enable_mtp_training=True, mtp_num_layers=2)) > base


# --- (e) ensure_hf_layer_pattern / layer_pattern_from_hf_config -------------------------------


def qwen36_text_config(**overrides):
    base = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        vocab_size=248320,
        num_hidden_layers=40,
        num_experts=256,
        moe_intermediate_size=512,
        num_experts_per_tok=8,
        shared_expert_intermediate_size=512,
        attn_output_gate=True,
        full_attention_interval=4,
        layer_types=list(QWEN36_LAYER_TYPES),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
    )
    return SimpleNamespace(**(base | overrides))


def test_pattern_read_from_layer_types():
    layer_types, linear = layer_pattern_from_hf_config(qwen36_text_config())
    assert layer_types == QWEN36_LAYER_TYPES
    assert linear == QWEN36_LINEAR


def test_pattern_derived_from_full_attention_interval_when_layer_types_is_absent():
    cfg = qwen36_text_config()
    del cfg.layer_types
    layer_types, linear = layer_pattern_from_hf_config(cfg)
    assert layer_types == QWEN36_LAYER_TYPES and linear == QWEN36_LINEAR


def test_pattern_is_none_for_dense_and_plain_moe_configs():
    dense = SimpleNamespace(hidden_size=64, num_attention_heads=4, num_hidden_layers=2)
    assert layer_pattern_from_hf_config(dense) == (None, None)
    sliding = SimpleNamespace(num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"])
    assert layer_pattern_from_hf_config(sliding) == (None, None)


def test_pattern_follows_the_vlm_text_config():
    vlm = SimpleNamespace(text_config=qwen36_text_config(), num_hidden_layers=1)
    assert layer_pattern_from_hf_config(vlm)[0] == QWEN36_LAYER_TYPES


def test_pattern_rejects_linear_layers_without_their_dims():
    cfg = qwen36_text_config()
    del cfg.linear_num_value_heads
    with pytest.raises(ValueError, match="linear_num_value_heads"):
        layer_pattern_from_hf_config(cfg)


def test_ensure_fills_args_once_from_the_checkpoint(monkeypatch):
    loads = []

    def fake_load(path):
        loads.append(path)
        return SimpleNamespace(text_config=qwen36_text_config())

    monkeypatch.setattr(flops_utils, "_load_hf_config", fake_load)
    args = SimpleNamespace(hf_checkpoint="/snap/qwen3.6", num_layers=40)
    assert ensure_hf_layer_pattern(args) is True
    assert args.flops_layer_types == QWEN36_LAYER_TYPES
    assert args.flops_linear_attention == QWEN36_LINEAR
    assert ensure_hf_layer_pattern(args) is True
    assert loads == ["/snap/qwen3.6"]


def test_ensure_leaves_dense_configs_on_the_old_formula(monkeypatch):
    monkeypatch.setattr(flops_utils, "_load_hf_config", lambda path: SimpleNamespace(num_hidden_layers=8))
    args = SimpleNamespace(hf_checkpoint="/snap/dense", num_layers=8)
    assert ensure_hf_layer_pattern(args) is False
    assert args.flops_layer_types is None and args.flops_linear_attention is None
    assert calculate_fwd_flops([1024], dense_args(**vars(args))) == calculate_fwd_flops([1024], dense_args())


def test_ensure_is_a_noop_without_a_checkpoint_or_when_already_set(monkeypatch):
    monkeypatch.setattr(flops_utils, "_load_hf_config", lambda path: pytest.fail("must not load"))
    assert ensure_hf_layer_pattern(SimpleNamespace(num_layers=8)) is False
    assert ensure_hf_layer_pattern(SimpleNamespace(hf_checkpoint=None, num_layers=8)) is False
    preset = SimpleNamespace(hf_checkpoint="/snap/x", flops_layer_types=QWEN36_LAYER_TYPES)
    assert ensure_hf_layer_pattern(preset) is True


def test_ensure_warns_and_keeps_the_old_formula_on_failure(monkeypatch, caplog):
    def boom(path):
        raise OSError("no config.json")

    monkeypatch.setattr(flops_utils, "_load_hf_config", boom)
    args = SimpleNamespace(hf_checkpoint="/snap/missing", num_layers=40)
    with caplog.at_level(logging.WARNING, logger="miles.utils.flops_utils"):
        assert ensure_hf_layer_pattern(args) is False
    assert "no config.json" in caplog.text
    assert getattr(args, "flops_layer_types", None) is None
    # the failure is cached: no second attempt
    monkeypatch.setattr(flops_utils, "_load_hf_config", lambda path: pytest.fail("must not retry"))
    assert ensure_hf_layer_pattern(args) is False


def test_ensure_rejects_a_layer_count_mismatch(monkeypatch, caplog):
    monkeypatch.setattr(flops_utils, "_load_hf_config", lambda path: qwen36_text_config())
    args = SimpleNamespace(hf_checkpoint="/snap/qwen3.6", num_layers=39)
    with caplog.at_level(logging.WARNING, logger="miles.utils.flops_utils"):
        assert ensure_hf_layer_pattern(args) is False
    assert "args.num_layers is 39" in caplog.text


def test_hf_config_adapter_carries_the_pattern_and_the_gate():
    adapted = flops_args_from_hf_config(SimpleNamespace(text_config=qwen36_text_config()))
    assert adapted.flops_layer_types == QWEN36_LAYER_TYPES
    assert adapted.flops_linear_attention == QWEN36_LINEAR
    assert adapted.attention_output_gate is True
    assert adapted.multi_latent_attention is False
    deepseek = flops_args_from_hf_config(
        SimpleNamespace(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=8,
            intermediate_size=16,
            kv_lora_rank=512,
        )
    )
    assert deepseek.multi_latent_attention is True
    megatron = qwen36_args(flops_layer_types=QWEN36_LAYER_TYPES, flops_linear_attention=QWEN36_LINEAR)
    assert calculate_fwd_flops([4096, S64K], adapted) == calculate_fwd_flops([4096, S64K], megatron)
