import logging

from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets

logger = logging.getLogger(__name__)


def add_tinker_arguments(parser):
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "base-model",
        help="Model name advertised by the gateway (default: --hf-checkpoint)",
    )
    add_argument(
        "checkpoint-root",
        help="Directory for tinker:// checkpoints (default: <save>/tinker)",
    )
    add_argument(
        "train-mode",
        choices=["lora", "full"],
        default="lora",
        help="'lora' serves multi-LoRA slots; 'full' fine-tunes the base weights for a single tenant",
    )
    add_argument(
        "lease-timeout-s",
        type=float,
        default=600.0,
        help="Seconds without a heartbeat before a session's models and slots are released",
    )
    add_argument(
        "max-concurrent-samples",
        type=int,
        default=512,
        help="Maximum in-flight /generate posts across all sampling requests",
    )
    add_argument(
        "coalesce-wait-s",
        type=float,
        default=0.3,
        help="Idle seconds a forward_backward batch waits for more chunks of a request still arriving (0 disables)",
    )
    add_argument(
        "coalesce-max-wait-s",
        type=float,
        default=2.0,
        help="Cap on the total coalesce wait for one batch, however many chunks keep arriving",
    )
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    if args.tinker_train_mode == "full":
        _configure_full_fine_tuning(args)
        return
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups


def _configure_full_fine_tuning(args) -> None:
    """Constraints of the full fine-tuning trainer (full/model.py)."""
    assert (
        not args.multi_lora_n_adapters
    ), "--tinker-train-mode full trains the base weights; drop --multi-lora-n-adapters"
    assert not args.lora_rank, "--tinker-train-mode full does not take --lora-rank"
    assert not args.overlap_grad_reduce and not args.overlap_param_gather, (
        "--tinker-train-mode full accumulates gradients across forward_backward calls by deferring "
        "finalize_model_grads to optim_step; --overlap-grad-reduce / --overlap-param-gather are not supported"
    )
    assert not args.colocate and not args.offload_train, (
        "--tinker-train-mode full keeps accumulated gradients in GPU buffers between calls; "
        "--offload-train (implied by --colocate) is not supported"
    )
    assert (
        args.use_dynamic_batch_size
    ), "--tinker-train-mode full requires --use-dynamic-batch-size --max-tokens-per-gpu"
    assert not getattr(args, "use_precision_aware_optimizer", False), (
        "--tinker-train-mode full resets Adam moments in place on load_state and reset_to_base; "
        "--use-precision-aware-optimizer (scaled master params) is not supported"
    )
    assert not args.calculate_per_token_loss, "Tinker losses are exact sums; drop --calculate-per-token-loss"
    if args.pause_generation_mode == "in_place":
        # a full-weight push invalidates every cached KV entry; in_place keeps the prefix cache
        logger.warning(
            "--tinker-train-mode full pushes the base weights on every sampler save; overriding "
            "--pause-generation-mode in_place with retract so engines drop their prefix cache"
        )
        args.pause_generation_mode = "retract"
    if getattr(args, "check_for_nan_in_loss_and_grad", False):
        # Megatron's DDP nan check raises inside the grad sync and kills the trainer; optim_step vetoes the step instead
        logger.info("--tinker-train-mode full: disabling check_for_nan_in_loss_and_grad (non-finite steps are vetoed)")
        args.check_for_nan_in_loss_and_grad = False
    # gateway metadata only: full fine-tuning trains every weight
    args.tinker_lora_groups = list(LORA_TARGET_GROUPS)
