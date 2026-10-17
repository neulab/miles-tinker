"""Serve-entry (Tinker gateway) argument validation.

Kept stdlib-only so the fast tests exercise it without the sglang/megatron
imports that ``miles.utils.arguments`` pulls in.
"""

import logging

logger = logging.getLogger(__name__)

# An SGLang engine refuses (HTTP 400) a prompt of ``context_length - SGLANG_PROMPT_MARGIN`` tokens or more:
# TpModelWorker.get_worker_info sets max_req_len = context_length - 1 and max_req_input_len = max_req_len - 5,
# and validate_input_length (sglang/srt/managers/utils.py) refuses len(input_ids) >= max_req_input_len. The
# gateway admits prompts up to --rollout-max-context-len - 1 tokens (_clamp_max_new_tokens in
# miles/tinker/runtime.py), so the engines need this many tokens of context over it. The same rule holds in
# the miles-latest-cu13 and dev-202610102218 images; it assumes no speculative decoding (EAGLE reserves its
# draft tokens on top) and a KV pool of at least one context (SGLang takes the smaller of the two).
SGLANG_PROMPT_MARGIN = 6


def engine_context_length(rollout_max_context_len: int) -> int:
    """The SGLang ``--context-length`` that takes every request the gateway admits under
    ``--rollout-max-context-len``: prompts up to ``rollout_max_context_len - 1`` tokens, and completions the
    gateway clamps to ``rollout_max_context_len`` (SGLang caps max_new_tokens at
    ``context_length - 2 - prompt_len``, which this context also covers)."""
    return rollout_max_context_len + SGLANG_PROMPT_MARGIN


def validate_tinker_serve_args(args) -> None:
    """Constraints of the Tinker gateway executors (entry == "serve")."""
    if args.tinker_train_mode == "full":
        assert (
            args.multi_lora_n_adapters == 0
        ), "--tinker-train-mode full trains the base weights; drop --multi-lora-n-adapters"
        assert not args.lora_rank or args.lora_rank <= 0, "--tinker-train-mode full does not take --lora-rank"
        assert not args.overlap_grad_reduce and not args.overlap_param_gather, (
            "--tinker-train-mode full accumulates gradients across forward_backward calls by deferring "
            "finalize_model_grads to optim_step; --overlap-grad-reduce / --overlap-param-gather are not supported"
        )
        assert not args.offload_train, (
            "--tinker-train-mode full keeps accumulated gradients in GPU buffers between calls; "
            "--offload-train (implied by --colocate) is not supported"
        )
        assert (
            args.use_dynamic_batch_size
        ), "--tinker-train-mode full requires --use-dynamic-batch-size --max-tokens-per-gpu"
        assert not getattr(args, "use_precision_aware_optimizer", False), (
            "--tinker-train-mode full resets Adam moments in place on load_state/reset_to_base; "
            "--use-precision-aware-optimizer (scaled master params) is not supported"
        )
        if getattr(args, "pause_generation_mode", "retract") == "in_place":
            # a full-weight push invalidates every cached KV entry; in_place keeps the prefix cache
            logger.warning(
                "--tinker-train-mode full pushes the base weights on every sampler save; overriding "
                "--pause-generation-mode in_place with retract so engines drop their prefix cache"
            )
            args.pause_generation_mode = "retract"
    else:
        assert args.multi_lora_n_adapters > 0, "--tinker-train-mode lora requires --multi-lora-n-adapters > 0"
        if args.tensor_model_parallel_size * args.expert_model_parallel_size > 1:
            logger.warning(
                "Tinker LoRA gateway with TP*EP > 1: adapter grads marked for TP/EP sum-reduction are re-summed on "
                "every forward_backward call, so a multi-call optim window double-counts them; prefer TP=1, EP=1."
            )
    assert not args.calculate_per_token_loss, "Tinker losses are exact sums; drop --calculate-per-token-loss"
    if args.context_parallel_size > 1:
        _validate_context_parallel(args)
    _validate_datum_token_cap(args)
    _validate_engine_context(args)
    assert (
        args.rollout_temperature == 1.0
    ), "the Tinker gateway scores logprobs at --rollout-temperature; Tinker semantics are untempered (keep 1.0)"
    if getattr(args, "check_for_nan_in_loss_and_grad", False):
        # Megatron's DDP nan check raises inside the grad sync and kills the actor; the executors veto the step instead.
        logger.info("Tinker gateway: setting check_for_nan_in_loss_and_grad=False (non-finite steps are vetoed)")
        args.check_for_nan_in_loss_and_grad = False
    if getattr(args, "use_rollout_routing_replay", False):
        _validate_routing_replay(args)
    if getattr(args, "fused_lm_head_logprobs", False):
        _validate_fused_lm_head(args)


def _validate_context_parallel(args) -> None:
    """CP>1 (docs/fork/full-finetuning.md): the Tinker losses, the forward-only and per-datum
    gathers, the bridge's pre-sharded input detection and the R3 fill all assume miles' zigzag
    THD layout (get_batch's slice_with_cp), and only the full fine-tuning executor is wired."""
    assert args.tinker_train_mode == "full", (
        "the Tinker LoRA gateway requires --context-parallel-size 1: its adapter token spans and slot "
        "executor are untested under CP; use --tinker-train-mode full"
    )
    assert getattr(args, "qkv_format", "thd") == "thd", (
        f"--context-parallel-size {args.context_parallel_size} through the Tinker gateway needs --qkv-format thd "
        "(the per-datum zigzag layout the losses and gathers slice)"
    )
    assert not getattr(args, "allgather_cp", False), (
        "--allgather-cp splits the packed row contiguously; the Tinker losses and gathers expect the zigzag "
        "layout of --context-parallel-size without --allgather-cp"
    )
    if getattr(args, "use_rollout_routing_replay", False) and not getattr(args, "tinker_r3_check", False):
        # the fill slices routing rows exactly like the tokens (tests/fast/backends/training_utils/
        # test_replay_data_cp.py), but no GPU run has checked R3 under CP yet
        logger.warning(
            f"--use-rollout-routing-replay at --context-parallel-size {args.context_parallel_size} has CPU tests "
            "only: run this layout once with --tinker-r3-check (replayed vs rollout routing) before relying on it"
        )


def _validate_datum_token_cap(args) -> None:
    """A datum trains as one sequence, so one microbatch of ``--max-tokens-per-gpu`` must hold
    this rank's share of the longest one.

    Under CP each rank keeps two equal zigzag chunks, padded to a multiple of
    TP*``--data-pad-size-multiplier`` (get_batch), so the longest datum is
    ``CP * floor(max_tokens_per_gpu / pad) * pad`` (the cap of radixark/miles #3320); the
    budget is per rank and easy to leave at its CP=1 value, so a longer cap is an error. At
    CP=1 rows up to the budget have always run (their padding block on top), so only a cap
    above the budget itself is reported, as a warning.
    """
    max_tokens_per_gpu = getattr(args, "max_tokens_per_gpu", None)
    max_tokens_per_datum = getattr(args, "tinker_max_tokens_per_datum", None)
    if max_tokens_per_gpu is None or max_tokens_per_datum is None:
        return
    cp = args.context_parallel_size
    if cp == 1:
        if max_tokens_per_datum > max_tokens_per_gpu:
            logger.warning(
                f"--tinker-max-tokens-per-datum {max_tokens_per_datum} exceeds --max-tokens-per-gpu "
                f"{max_tokens_per_gpu}: such a datum trains alone in an oversized microbatch"
            )
        return
    pad = args.tensor_model_parallel_size * getattr(args, "data_pad_size_multiplier", 128)
    limit = max_datum_tokens(max_tokens_per_gpu, cp, pad)
    assert max_tokens_per_datum <= limit, (
        f"--tinker-max-tokens-per-datum {max_tokens_per_datum} exceeds the {limit} tokens one microbatch of "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} holds per rank at --context-parallel-size {cp} (two zigzag "
        f"chunks padded to a multiple of {pad}); a full-length datum needs --max-tokens-per-gpu >= "
        f"{cp_shard_tokens(max_tokens_per_datum, cp, pad)}"
    )


def _validate_engine_context(args) -> None:
    """The engines must take every prompt the gateway admits: one they refuse fails the sample as a server
    error (a platoon eval on 2026-10-17: engines at context 32769 refused a 32763-token prompt that the
    gateway had admitted at --rollout-max-context-len 32768)."""
    gateway_context = getattr(args, "rollout_max_context_len", None)
    engine_context = getattr(args, "sglang_context_length", None)
    if not gateway_context or engine_context is None:
        return  # no gateway clamp, or the engines use the model's own context length
    needed = engine_context_length(gateway_context)
    assert engine_context >= needed, (
        f"--sglang-context-length {engine_context} refuses prompts that --rollout-max-context-len "
        f"{gateway_context} admits: SGLang refuses a prompt of context_length - {SGLANG_PROMPT_MARGIN} tokens or "
        f"more, and the gateway admits up to {gateway_context - 1}; pass --sglang-context-length >= {needed}"
    )


def cp_shard_tokens(row_tokens: int, cp: int, pad: int) -> int:
    """Tokens one CP rank holds for a ``row_tokens`` row: two equal zigzag chunks, padded to a multiple of ``pad``."""
    shard = 2 * -(-row_tokens // (2 * cp))
    return -(-shard // pad) * pad


def max_datum_tokens(max_tokens_per_gpu: int, cp: int, pad: int) -> int:
    """The longest row whose CP shard fits one microbatch of ``max_tokens_per_gpu`` tokens per rank."""
    return cp * (max_tokens_per_gpu // pad * pad)


def _validate_routing_replay(args) -> None:
    """R3 through the gateway: the routing cache stores one expert id per (token, layer, slot)."""
    assert getattr(args, "num_experts", None), "--use-rollout-routing-replay needs an MoE model (--num-experts)"
    assert not getattr(args, "moe_router_fusion", False), (
        "--use-rollout-routing-replay replays the rollout routing through the unfused MoE top-k; with "
        "--moe-router-fusion the fused kernel returns before the replay hook and the routing is ignored"
    )
    assert not getattr(args, "sglang_speculative_algorithm", None), (
        "--use-rollout-routing-replay through the Tinker gateway does not support speculative decoding "
        "(the engines' routed-experts capture is target-only and its buffer ignores verify tokens); "
        "drop --sglang-speculative-algorithm"
    )
    if getattr(args, "mtp_num_layers", None):
        logger.warning(
            "--use-rollout-routing-replay with --mtp-num-layers %s: the MTP block is excluded from the replay on "
            "both sides; pass --mtp-num-layers 0 unless MTP training is intended",
            args.mtp_num_layers,
        )


def _validate_fused_lm_head(args) -> None:
    """The fused chunked head only yields target-token logprobs of the policy loss."""
    assert not getattr(args, "use_rollout_entropy", False), (
        "--fused-lm-head-logprobs cannot compute entropy; drop --use-rollout-entropy"
    )
    assert (
        getattr(args, "loss_type", "policy_loss") == "policy_loss"
    ), f"--fused-lm-head-logprobs serves policy_loss only, got --loss-type {args.loss_type!r}"
    assert not getattr(
        args, "calculate_per_token_loss", False
    ), "--fused-lm-head-logprobs is incompatible with --calculate-per-token-loss"
    assert not getattr(
        args, "true_on_policy_mode", False
    ), "--fused-lm-head-logprobs computes fp32 logits per chunk; --true-on-policy-mode is not supported"
    assert not getattr(args, "enable_mtp_training", False) and not getattr(args, "mtp_num_layers", None), (
        "--fused-lm-head-logprobs cannot serve the MTP block's logits; drop --enable-mtp-training and pass "
        "--mtp-num-layers 0"
    )
    assert getattr(
        args, "untie_embeddings_and_output_weights", True
    ), "--fused-lm-head-logprobs takes the output layer's own weight; tied embeddings are not supported"
