"""Serve-entry validation of the Tinker gateway modes."""

from argparse import Namespace

import pytest

from miles.utils import tinker_args as arguments


def _serve_args(**overrides) -> Namespace:
    defaults = dict(
        tinker_train_mode="full",
        multi_lora_n_adapters=0,
        lora_rank=0,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        offload_train=False,
        use_dynamic_batch_size=True,
        calculate_per_token_loss=False,
        context_parallel_size=1,
        rollout_temperature=1.0,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        check_for_nan_in_loss_and_grad=True,
        use_precision_aware_optimizer=False,
    )
    return Namespace(**{**defaults, **overrides})


def test_full_mode_accepts_the_recipe_and_disables_the_ddp_nan_check():
    args = _serve_args()
    arguments.validate_tinker_serve_args(args)
    assert args.check_for_nan_in_loss_and_grad is False, "the executors veto non-finite steps instead"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(multi_lora_n_adapters=2),
        dict(lora_rank=8),
        dict(overlap_grad_reduce=True),
        dict(overlap_param_gather=True),
        dict(offload_train=True),
        dict(use_dynamic_batch_size=False),
        dict(use_precision_aware_optimizer=True),
        dict(context_parallel_size=2, qkv_format="bshd"),
        dict(context_parallel_size=2, allgather_cp=True),
        dict(rollout_temperature=0.7),
        dict(calculate_per_token_loss=True),
    ],
)
def test_full_mode_rejects_incompatible_flags(overrides):
    with pytest.raises(AssertionError):
        arguments.validate_tinker_serve_args(_serve_args(**overrides))


def test_lora_mode_requires_adapter_slots():
    with pytest.raises(AssertionError, match="multi-lora-n-adapters"):
        arguments.validate_tinker_serve_args(_serve_args(tinker_train_mode="lora"))
    arguments.validate_tinker_serve_args(_serve_args(tinker_train_mode="lora", multi_lora_n_adapters=2, lora_rank=8))


def test_lora_mode_warns_about_tp_ep_marked_grad_double_sum(caplog):
    args = _serve_args(tinker_train_mode="lora", multi_lora_n_adapters=2, lora_rank=8, tensor_model_parallel_size=2)
    with caplog.at_level("WARNING", logger="miles.utils.tinker_args"):
        arguments.validate_tinker_serve_args(args)
    assert any("TP*EP" in message for message in caplog.messages)


def test_full_mode_overrides_in_place_pausing(caplog):
    from miles.utils.tinker_args import validate_tinker_serve_args

    args = _serve_args(
        tinker_train_mode="full", multi_lora_n_adapters=0, lora_rank=0, pause_generation_mode="in_place"
    )
    validate_tinker_serve_args(args)
    assert args.pause_generation_mode == "retract"

    args = _serve_args(tinker_train_mode="full", multi_lora_n_adapters=0, lora_rank=0, pause_generation_mode="abort")
    validate_tinker_serve_args(args)
    assert args.pause_generation_mode == "abort"


# -------- R3 (rollout routing replay) through the gateway --------


def _r3_args(**overrides) -> Namespace:
    defaults = dict(use_rollout_routing_replay=True, num_experts=256, sglang_speculative_algorithm=None)
    return _serve_args(**{**defaults, **overrides})


def test_r3_accepts_an_moe_model_without_speculative_decoding():
    arguments.validate_tinker_serve_args(_r3_args())


def test_r3_requires_num_experts():
    with pytest.raises(AssertionError, match="num-experts"):
        arguments.validate_tinker_serve_args(_r3_args(num_experts=None))
    with pytest.raises(AssertionError, match="num-experts"):
        arguments.validate_tinker_serve_args(_r3_args(num_experts=0))


def test_r3_rejects_speculative_decoding():
    with pytest.raises(AssertionError, match="speculative"):
        arguments.validate_tinker_serve_args(_r3_args(sglang_speculative_algorithm="EAGLE"))


def test_r3_warns_about_mtp_layers(caplog):
    with caplog.at_level("WARNING", logger="miles.utils.tinker_args"):
        arguments.validate_tinker_serve_args(_r3_args(mtp_num_layers=1))
    assert any("mtp-num-layers" in message for message in caplog.messages)
    caplog.clear()
    with caplog.at_level("WARNING", logger="miles.utils.tinker_args"):
        arguments.validate_tinker_serve_args(_r3_args(mtp_num_layers=0))
    assert not caplog.messages


def test_r3_rejects_the_fused_router():
    """The fused top-k returns before the replay hook: the rollout routing would be ignored."""
    with pytest.raises(AssertionError, match="moe-router-fusion"):
        arguments.validate_tinker_serve_args(_r3_args(moe_router_fusion=True))
    arguments.validate_tinker_serve_args(_r3_args(moe_router_fusion=False))


def test_r3_off_ignores_the_moe_requirements():
    arguments.validate_tinker_serve_args(_serve_args(use_rollout_routing_replay=False, num_experts=None))


# -------- fused chunked lm head --------


def _fused_args(**overrides) -> Namespace:
    defaults = dict(
        fused_lm_head_logprobs=True,
        loss_type="policy_loss",
        true_on_policy_mode=False,
        enable_mtp_training=False,
        mtp_num_layers=0,
        untie_embeddings_and_output_weights=True,
    )
    return _serve_args(**{**defaults, **overrides})


def test_fused_head_accepts_the_recipe():
    arguments.validate_tinker_serve_args(_fused_args())
    arguments.validate_tinker_serve_args(_fused_args(mtp_num_layers=None))


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(loss_type="sft_loss"), "policy_loss"),
        (dict(true_on_policy_mode=True), "true-on-policy"),
        (dict(enable_mtp_training=True, mtp_num_layers=1), "MTP"),
        (dict(mtp_num_layers=1), "MTP"),
        (dict(untie_embeddings_and_output_weights=False), "tied"),
    ],
)
def test_fused_head_rejects_incompatible_flags(overrides, match):
    with pytest.raises(AssertionError, match=match):
        arguments.validate_tinker_serve_args(_fused_args(**overrides))


def test_fused_head_rejects_per_token_loss_before_the_head_check():
    with pytest.raises(AssertionError):
        arguments.validate_tinker_serve_args(_fused_args(calculate_per_token_loss=True))


def test_fused_head_off_skips_its_checks():
    arguments.validate_tinker_serve_args(_fused_args(fused_lm_head_logprobs=False, mtp_num_layers=1))


# -------- per-datum token cap against the per-rank budget (CP-aware, from radixark/miles #3320) --------


def test_datum_cap_at_cp1_warns_only_above_the_budget(caplog):
    with caplog.at_level("WARNING", logger="miles.utils.tinker_args"):
        arguments.validate_tinker_serve_args(
            _serve_args(tensor_model_parallel_size=4, max_tokens_per_gpu=131073, tinker_max_tokens_per_datum=131073)
        )
    assert not caplog.messages, "the v25 128k rows (budget == datum cap) have always run"
    with caplog.at_level("WARNING", logger="miles.utils.tinker_args"):
        arguments.validate_tinker_serve_args(_serve_args(max_tokens_per_gpu=16384, tinker_max_tokens_per_datum=32768))
    assert any("oversized microbatch" in message for message in caplog.messages)


@pytest.mark.parametrize(
    "tp, cp, budget, longest",
    [(4, 2, 66048, 132096), (4, 4, 33280, 133120), (1, 8, 16512, 132096), (1, 2, 4224, 8448)],
)
def test_datum_cap_at_cp_holds_the_zigzag_shard(tp, cp, budget, longest):
    """Each rank keeps two zigzag chunks padded to TP*128: the longest datum is cp * floor(budget/pad) * pad."""
    common = dict(context_parallel_size=cp, tensor_model_parallel_size=tp, data_pad_size_multiplier=128)
    arguments._validate_datum_token_cap(
        _serve_args(**common, max_tokens_per_gpu=budget, tinker_max_tokens_per_datum=longest)
    )
    with pytest.raises(AssertionError, match=f"needs --max-tokens-per-gpu >= {budget + tp * 128}"):
        arguments._validate_datum_token_cap(
            _serve_args(**common, max_tokens_per_gpu=budget, tinker_max_tokens_per_datum=longest + 1)
        )


def test_datum_cap_rejects_a_cp1_budget_left_in_place():
    args = _serve_args(
        context_parallel_size=2,
        tensor_model_parallel_size=4,
        max_tokens_per_gpu=65537,
        tinker_max_tokens_per_datum=131073,
    )
    with pytest.raises(AssertionError, match="131072 tokens .* needs --max-tokens-per-gpu >= 66048"):
        arguments._validate_datum_token_cap(args)


def test_datum_cap_is_skipped_without_both_limits():
    arguments._validate_datum_token_cap(_serve_args(context_parallel_size=2, max_tokens_per_gpu=None))
    arguments._validate_datum_token_cap(_serve_args(context_parallel_size=2, tinker_max_tokens_per_datum=None))


# -------- context parallelism (full fine-tuning only) --------


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_full_mode_accepts_zigzag_context_parallelism(cp):
    arguments.validate_tinker_serve_args(_serve_args(context_parallel_size=cp, qkv_format="thd", allgather_cp=False))
    # the per-rank budget of one 131073-token row at TP4 (docs/fork/full-finetuning.md)
    budget = {2: 66048, 4: 33280, 8: 16896}[cp]
    arguments.validate_tinker_serve_args(
        _serve_args(
            context_parallel_size=cp,
            tensor_model_parallel_size=4,
            data_pad_size_multiplier=128,
            max_tokens_per_gpu=budget,
            tinker_max_tokens_per_datum=131073,
        )
    )


def test_cp_keeps_the_datum_cap_per_rank():
    with pytest.raises(AssertionError, match="needs --max-tokens-per-gpu >= 66048"):
        arguments.validate_tinker_serve_args(
            _serve_args(
                context_parallel_size=2,
                tensor_model_parallel_size=4,
                max_tokens_per_gpu=65537,
                tinker_max_tokens_per_datum=131073,
            )
        )


def test_lora_mode_stays_at_cp1():
    with pytest.raises(AssertionError, match="LoRA gateway requires --context-parallel-size 1"):
        arguments.validate_tinker_serve_args(
            _serve_args(tinker_train_mode="lora", multi_lora_n_adapters=2, lora_rank=8, context_parallel_size=2)
        )


def test_cp_with_r3_and_the_fused_head_is_allowed_but_not_with_a_fused_router():
    arguments.validate_tinker_serve_args(_r3_args(context_parallel_size=2))
    arguments.validate_tinker_serve_args(_fused_args(context_parallel_size=2))
    with pytest.raises(AssertionError, match="moe-router-fusion"):
        arguments.validate_tinker_serve_args(_r3_args(context_parallel_size=2, moe_router_fusion=True))
