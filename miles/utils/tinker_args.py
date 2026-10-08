"""Serve-entry (Tinker gateway) argument validation.

Kept stdlib-only so the fast tests exercise it without the sglang/megatron
imports that ``miles.utils.arguments`` pulls in.
"""

import logging

logger = logging.getLogger(__name__)


def validate_tinker_serve_args(args) -> None:
    """Constraints of the Tinker gateway executors (entry == "serve")."""
    if getattr(args, "tinker_enable_critic", False):
        assert args.tinker_train_mode == "full", "--tinker-enable-critic requires --tinker-train-mode full"
        assert args.train_backend == "megatron", "--tinker-enable-critic requires the Megatron backend"
        assert not getattr(args, "colocate", False), "Tinker PPO currently requires dedicated inference GPUs"
    if args.tinker_train_mode == "full":
        assert (
            args.multi_lora_n_adapters == 0
        ), "--tinker-train-mode full trains the base weights; drop --multi-lora-n-adapters"
        assert not args.lora_rank or args.lora_rank <= 0, "--tinker-train-mode full does not take --lora-rank"
        assert not args.overlap_grad_reduce and not args.overlap_param_gather, (
            "--tinker-train-mode full accumulates gradients across forward_backward calls by deferring "
            "finalize_model_grads to optim_step; --overlap-grad-reduce / --overlap-param-gather are not supported"
        )
        if args.offload_train:
            assert getattr(
                args, "tinker_enable_critic", False
            ), "--tinker-train-mode full supports --offload-train only with --tinker-enable-critic"
            # Native training may discard parameter buffers after a step.
            # Tinker must restore weights for forward/save/load as well.
            args.disable_param_buffers_cpu_backup = False
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
    assert (
        args.context_parallel_size == 1
    ), "the Tinker gateway returns per-datum logprobs; context parallelism is not supported"
    if getattr(args, "tinker_enable_critic", False):
        # Native PPO scores the actor at the rollout temperature. The PPO client
        # must sample at this same temperature; ordinary SDK serving stays at 1.
        assert 0 < args.rollout_temperature < float("inf"), "PPO requires a positive finite --rollout-temperature"
    else:
        assert (
            args.rollout_temperature == 1.0
        ), "the Tinker gateway scores logprobs at --rollout-temperature; Tinker semantics are untempered (keep 1.0)"
    if getattr(args, "check_for_nan_in_loss_and_grad", False):
        # Megatron's DDP nan check raises inside the grad sync and kills the actor; the executors veto the step instead.
        logger.info("Tinker gateway: setting check_for_nan_in_loss_and_grad=False (non-finite steps are vetoed)")
        args.check_for_nan_in_loss_and_grad = False
