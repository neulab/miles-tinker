"""Serve-entry configuration of the Tinker gateway modes."""

from argparse import Namespace

import pytest

from miles.tinker.arguments import configure_tinker_args


def _full_args(**overrides) -> Namespace:
    defaults = dict(
        train_backend="megatron",
        tinker_train_mode="full",
        multi_lora_n_adapters=0,
        lora_rank=0,
        exclude_modules=None,
        target_modules=None,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        colocate=False,
        offload_train=False,
        use_dynamic_batch_size=True,
        calculate_per_token_loss=False,
        pause_generation_mode="retract",
        check_for_nan_in_loss_and_grad=True,
        use_precision_aware_optimizer=False,
    )
    return Namespace(**{**defaults, **overrides})


def test_full_mode_accepts_the_recipe_and_disables_the_ddp_nan_check():
    args = _full_args()
    configure_tinker_args(args)
    assert args.check_for_nan_in_loss_and_grad is False, "optim_step vetoes non-finite steps instead"
    assert args.target_modules is None, "full fine-tuning must not configure LoRA targets"
    assert args.tinker_lora_groups == ["attn", "mlp", "unembed"]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(multi_lora_n_adapters=2),
        dict(lora_rank=8),
        dict(overlap_grad_reduce=True),
        dict(overlap_param_gather=True),
        dict(colocate=True),
        dict(offload_train=True),
        dict(use_dynamic_batch_size=False),
        dict(use_precision_aware_optimizer=True),
        dict(calculate_per_token_loss=True),
    ],
    ids=lambda overrides: next(iter(overrides)),
)
def test_full_mode_rejects_settings_that_break_accumulation(overrides):
    with pytest.raises(AssertionError):
        configure_tinker_args(_full_args(**overrides))


def test_full_mode_forces_retract_pausing():
    args = _full_args(pause_generation_mode="in_place")
    configure_tinker_args(args)
    assert args.pause_generation_mode == "retract", "a full-weight push must drop the prefix cache"


def test_lora_mode_selects_the_target_groups():
    args = _full_args(tinker_train_mode="lora", multi_lora_n_adapters=2, target_modules="attn,mlp")
    configure_tinker_args(args)
    assert args.tinker_lora_groups == args.target_modules == ["attn", "mlp"]
