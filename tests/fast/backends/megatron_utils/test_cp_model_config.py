"""Model-config side of context parallelism (miles/backends/megatron_utils/cp_model_config.py).

The Qwen3-VL / Qwen3.5-VL megatron-bridge providers force calculate_per_token_loss on
in finalize() at CP>1 and assert it while building Qwen3VLModel; Megatron's DDP and
schedule read that config while miles' loss scaling reads args, so left alone every
gradient is num_microbatches * |DP*CP| too large. A fake provider with the same
finalize/assert behaviour drives the real get_model_provider_func; fake DDP chunks
drive the post-build invariant check.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_model_provider_bridge_mtp import (  # noqa: F401  (fixture)
    _args,
    _install_bridge,
    _Provider,
    provider_module,
)

from miles.backends.megatron_utils import cp_model_config

# ---------------------------------------------------------------------------
# A VL provider/model with the bridge's per-token-loss behaviour
# ---------------------------------------------------------------------------


class _Router(torch.nn.Module):
    """Megatron's TopKRouter caches the flag at construction (router.py: self.calculate_per_token_loss)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.calculate_per_token_loss = config.calculate_per_token_loss


class _Language(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mtp_process = False
        self.router = _Router(config)


class _Vision(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(calculate_per_token_loss=config.calculate_per_token_loss)  # a separate copy


class _FakeQwenVLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # language_transformer_config = self (the provider)
        self.language_model = _Language(config)
        self.vision_model = _Vision(config)
        if config.context_parallel_size > 1:  # Qwen3VLModel.__init__
            assert config.calculate_per_token_loss, "Qwen3-VL model only supports CP with calculate_per_token_loss"


class _VLProvider(_Provider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.provide_calls = 0

    def finalize(self):  # qwen35_vl_provider.finalize at CP>1
        if (self.context_parallel_size or 1) > 1:
            self.calculate_per_token_loss = True
        self.finalized = True

    def provide(self, pre_process=True, post_process=True, vp_stage=None):
        self.provide_calls += 1
        return _FakeQwenVLModel(self)


def _all_flags(model) -> list[bool]:
    flags = []
    for module in model.modules():
        if hasattr(module.config, "calculate_per_token_loss"):
            flags.append(module.config.calculate_per_token_loss)
        if "calculate_per_token_loss" in vars(module):
            flags.append(module.calculate_per_token_loss)
    return flags


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_bridge_vl_model_ends_with_the_args_flag(provider_module, cp_size):  # noqa: F811  (the imported fixture)
    provider = _VLProvider()
    _install_bridge(provider_module, provider)
    provide = provider_module.get_model_provider_func(_args(mtp_num_layers=0, context_parallel_size=cp_size))
    assert provider.calculate_per_token_loss is (cp_size > 1), "the bridge finalize() forced the flag at CP>1"

    # two chunks (VPP): the second build's own assert must still see the finalized flag
    models = [provide(vp_stage=0), provide(vp_stage=1)]
    assert provider.provide_calls == 2
    for model in models:
        flags = _all_flags(model)
        assert flags and not any(flags), "config, vision copy and the router's cached copy all reset"
    assert provider.calculate_per_token_loss is False, "the provider is the model config: reset as well"


def test_restore_per_token_loss_counts_changes_and_honours_args_true():
    config = SimpleNamespace(calculate_per_token_loss=True, context_parallel_size=2)
    model = _FakeQwenVLModel(config)
    assert cp_model_config.restore_per_token_loss(model, SimpleNamespace(calculate_per_token_loss=False)) == 3
    assert not any(_all_flags(model))
    assert cp_model_config.restore_per_token_loss(model, SimpleNamespace(calculate_per_token_loss=False)) == 0
    assert cp_model_config.restore_per_token_loss(model, SimpleNamespace(calculate_per_token_loss=True)) == 3
    assert all(_all_flags(model))


def test_restore_per_token_loss_accepts_a_plain_object_model():
    model = SimpleNamespace(config=SimpleNamespace(calculate_per_token_loss=True))
    assert cp_model_config.restore_per_token_loss(model, SimpleNamespace(calculate_per_token_loss=False)) == 1
    assert model.config.calculate_per_token_loss is False


# ---------------------------------------------------------------------------
# CP modes onto the provider
# ---------------------------------------------------------------------------


def _cp_args(cp=2, **kw):
    return SimpleNamespace(context_parallel_size=cp, **kw)


def test_cp_modes_copy_onto_the_provider():
    provider = SimpleNamespace(cp_comm_type=None, linear_cp_mode="chunkwise")
    cp_model_config.apply_context_parallel_modes(provider, _cp_args(cp_comm_type=["p2p"], linear_cp_mode="headwise"))
    assert provider.cp_comm_type == "p2p" and provider.linear_cp_mode == "headwise"

    per_layer = SimpleNamespace(cp_comm_type=None, linear_cp_mode="chunkwise")
    cp_model_config.apply_context_parallel_modes(per_layer, _cp_args(cp_comm_type=["p2p", "a2a"]))
    assert per_layer.cp_comm_type == ["p2p", "a2a"] and per_layer.linear_cp_mode == "chunkwise"


def test_megatrons_default_comm_type_keeps_a_providers_own_choice():
    """--cp-comm-type defaults to ["p2p"]: it must not override e.g. the Mistral provider's a2a for sliding windows."""
    provider = SimpleNamespace(cp_comm_type="a2a", linear_cp_mode="chunkwise")
    cp_model_config.apply_context_parallel_modes(provider, _cp_args(cp_comm_type=["p2p"]))
    assert provider.cp_comm_type == "a2a"
    cp_model_config.apply_context_parallel_modes(provider, _cp_args(cp_comm_type=["all_gather"]))
    assert provider.cp_comm_type == "all_gather", "a non-default --cp-comm-type is an explicit request"


def test_cp_modes_leave_the_provider_alone_when_unset_and_reject_unknown_fields():
    provider = SimpleNamespace(cp_comm_type="a2a", linear_cp_mode="chunkwise")
    cp_model_config.apply_context_parallel_modes(provider, _cp_args())
    assert provider.cp_comm_type == "a2a" and provider.linear_cp_mode == "chunkwise"
    # a provider without linear-attention CP modes is fine as long as none is asked for
    cp_model_config.apply_context_parallel_modes(SimpleNamespace(), _cp_args(cp_comm_type=["p2p"]))
    with pytest.raises(ValueError, match="has no linear_cp_mode"):
        cp_model_config.apply_context_parallel_modes(SimpleNamespace(), _cp_args(linear_cp_mode="headwise"))


def test_cp_modes_do_nothing_without_context_parallelism():
    provider = SimpleNamespace(cp_comm_type="a2a", linear_cp_mode="chunkwise")
    cp_model_config.apply_context_parallel_modes(
        provider, _cp_args(cp=1, cp_comm_type=["all_gather"], linear_cp_mode="headwise")
    )
    assert provider.cp_comm_type == "a2a" and provider.linear_cp_mode == "chunkwise"
    # an older Megatron/bridge without linear_cp_mode builds at CP=1 whatever the flag says
    cp_model_config.apply_context_parallel_modes(SimpleNamespace(), _cp_args(cp=1, linear_cp_mode="headwise"))


def test_bridge_runtime_config_carries_the_cp_modes(provider_module):  # noqa: F811  (the imported fixture)
    provider = _Provider(cp_comm_type=None, linear_cp_mode="chunkwise")
    provider_module._apply_bridge_runtime_config(
        provider, _args(mtp_num_layers=0, context_parallel_size=2, cp_comm_type=["p2p"], linear_cp_mode="headwise")
    )
    assert (provider.context_parallel_size, provider.cp_comm_type, provider.linear_cp_mode) == (2, "p2p", "headwise")
    unset = _Provider(cp_comm_type=None, linear_cp_mode="chunkwise")
    provider_module._apply_bridge_runtime_config(unset, _args(mtp_num_layers=0, context_parallel_size=2))
    assert unset.linear_cp_mode == "chunkwise", "no --linear-cp-mode: the provider keeps its own"


# ---------------------------------------------------------------------------
# MRoPE patch check
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_vl_class(monkeypatch):
    """A class living in a module named like megatron-bridge's Qwen3VLModel module."""
    name = "fake.bridge.models.qwen_vl.modelling_qwen3_vl.model"
    module = types.ModuleType(name)
    cls = type("Qwen3VLModel", (), {"__module__": name})
    module.Qwen3VLModel = cls
    monkeypatch.setitem(sys.modules, name, module)
    return module, cls


def test_mrope_patch_is_required_at_cp_above_1(bridge_vl_class):
    from miles_plugins.models.qwen3_vl import _PATCHED

    module, cls = bridge_vl_class
    model = cls()
    cp_model_config.check_qwen_vl_mrope_patch(model, SimpleNamespace(context_parallel_size=1))
    with pytest.raises(RuntimeError, match="packed MRoPE patch"):
        cp_model_config.check_qwen_vl_mrope_patch(model, SimpleNamespace(context_parallel_size=2))
    setattr(module, _PATCHED, True)
    with pytest.raises(RuntimeError, match="packed MRoPE patch"):  # get_rope_index patched, forward not
        cp_model_config.check_qwen_vl_mrope_patch(model, SimpleNamespace(context_parallel_size=2))
    setattr(cls, _PATCHED, True)
    cp_model_config.check_qwen_vl_mrope_patch(model, SimpleNamespace(context_parallel_size=2))
    # a subclass of the bridge model is checked through its bridge base
    cp_model_config.check_qwen_vl_mrope_patch(type("Sub", (cls,), {})(), SimpleNamespace(context_parallel_size=4))


def test_mrope_check_ignores_other_models():
    cp_model_config.check_qwen_vl_mrope_patch(torch.nn.Linear(2, 2), SimpleNamespace(context_parallel_size=8))


# ---------------------------------------------------------------------------
# Post-DDP invariants
# ---------------------------------------------------------------------------


@pytest.fixture
def loss_scale_group(monkeypatch):
    """miles' parallel state as loss.py sees it: the loss is multiplied by intra_dp_cp.size."""
    state = SimpleNamespace(intra_dp_cp=SimpleNamespace(size=4))
    monkeypatch.setattr(cp_model_config, "get_parallel_state", lambda: state)
    return state


def _ddp_chunk(*, per_token=False, dp_cp=4, fusion=False):
    config = SimpleNamespace(calculate_per_token_loss=per_token, moe_router_fusion=fusion)
    return SimpleNamespace(config=config, dp_cp_group=SimpleNamespace(size=lambda: dp_cp))


def _run_args(**kw):
    return SimpleNamespace(**{"calculate_per_token_loss": False, "use_rollout_routing_replay": False, **kw})


def test_invariants_accept_the_fixed_model(loss_scale_group):
    cp_model_config.check_loss_scaling_invariants([_ddp_chunk(), _ddp_chunk()], _run_args())
    loss_scale_group.intra_dp_cp.size = 8  # DP2 x CP4
    cp_model_config.check_loss_scaling_invariants([_ddp_chunk(dp_cp=8)], _run_args())


def test_invariants_catch_the_bridge_bug(loss_scale_group):
    with pytest.raises(AssertionError, match="disagrees with --calculate-per-token-loss"):
        cp_model_config.check_loss_scaling_invariants([_ddp_chunk(per_token=True)], _run_args())


def test_invariants_catch_a_reduction_group_the_loss_scale_does_not_cancel(loss_scale_group):
    """DDP pre-scales by 1/|its DP*CP group| (asserted by its own constructor); loss.py multiplies by
    intra_dp_cp: the two must be the same size or the sum over DP*CP is off by their ratio."""
    with pytest.raises(AssertionError, match="group of 2 ranks .* intra_dp_cp=4"):
        cp_model_config.check_loss_scaling_invariants([_ddp_chunk(dp_cp=2)], _run_args())


def test_invariants_skip_the_group_check_with_per_token_loss(loss_scale_group):
    cp_model_config.check_loss_scaling_invariants(
        [_ddp_chunk(per_token=True, dp_cp=2)], _run_args(calculate_per_token_loss=True)
    )


def test_invariants_find_the_config_through_wrappers_and_skip_non_ddp_chunks(loss_scale_group):
    inner = SimpleNamespace(config=SimpleNamespace(calculate_per_token_loss=False))
    cp_model_config.check_loss_scaling_invariants([SimpleNamespace(module=SimpleNamespace(module=inner))], _run_args())


def test_r3_requires_the_unfused_router(loss_scale_group):
    with pytest.raises(AssertionError, match="unfused MoE router"):
        cp_model_config.check_loss_scaling_invariants(
            [_ddp_chunk(fusion=True)], _run_args(use_rollout_routing_replay=True)
        )
    cp_model_config.check_loss_scaling_invariants([_ddp_chunk(fusion=True)], _run_args())
    cp_model_config.check_loss_scaling_invariants([_ddp_chunk()], _run_args(use_rollout_routing_replay=True))


def test_the_per_step_flag_check():
    """full/executor.forward_backward re-checks the config the schedule reads on every call."""
    cp_model_config.check_per_token_loss_flag(SimpleNamespace(calculate_per_token_loss=False), _run_args())
    with pytest.raises(AssertionError, match="config calculate_per_token_loss=True disagrees"):
        cp_model_config.check_per_token_loss_flag(SimpleNamespace(calculate_per_token_loss=True), _run_args())
