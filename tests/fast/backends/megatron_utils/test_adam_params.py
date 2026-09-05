"""AdamParams land in the Megatron param groups the way its scheduler would
(wd_mult / lr_mult honored, tensor lr updated in place); non-finite grads veto."""

import pytest
import torch

from miles.backends.megatron_utils.adam_params import apply_adam_params, set_clip_grad, should_veto_step

ADAM = {
    "learning_rate": 2e-5,
    "beta1": 0.8,
    "beta2": 0.99,
    "eps": 1e-6,
    "weight_decay": 0.1,
    "grad_clip_norm": 0.0,
}


def test_apply_adam_params_writes_groups_with_megatron_multipliers():
    weight = torch.nn.Parameter(torch.zeros(2))
    bias = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([{"params": [weight], "wd_mult": 1.0}, {"params": [bias], "wd_mult": 0.0}])

    apply_adam_params(optimizer.param_groups, ADAM)

    for group in optimizer.param_groups:
        assert group["lr"] == 2e-5 and group["betas"] == (0.8, 0.99) and group["eps"] == 1e-6
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0, "1-D params keep Megatron's zero weight decay"


def test_apply_adam_params_fills_a_tensor_lr_in_place():
    lr = torch.tensor(1.0, dtype=torch.float64)
    group = {"params": [], "lr": lr, "lr_mult": 0.5}
    apply_adam_params([group], ADAM)
    assert group["lr"] is lr and lr.item() == pytest.approx(1e-5)


def test_set_clip_grad_reaches_chained_children():
    child = type("Child", (), {})()
    child.config = type("Config", (), {"clip_grad": 1.0})()
    optimizer = type("Chained", (), {})()
    optimizer.config = type("Config", (), {"clip_grad": 1.0})()
    optimizer.chained_optimizers = [child]
    set_clip_grad(optimizer, 0)
    assert optimizer.config.clip_grad == 0.0 and child.config.clip_grad == 0.0


def test_should_veto_step_on_inf_or_nan():
    assert should_veto_step(False, 1.0) is False
    assert should_veto_step(True, 1.0) is True
    assert should_veto_step(False, float("nan")) is True
    assert should_veto_step(False, float("inf")) is True
