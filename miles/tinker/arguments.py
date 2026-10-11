from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


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
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
