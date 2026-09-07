"""Bridge mode honours ``--mtp-num-layers``: megatron.bridge derives ``mtp_num_layers``
from the HF config (``mtp_num_hidden_layers`` -> 1 for Qwen3.5/3.6), so the runtime-config
copy must override it before ``finalize()`` and the built model must not carry an MTP block
when the flag is 0 (the fused lm head rejects ``mtp_process`` at install; without the fused
head the MTP objective would train on a second [T, V] logits tensor)."""

import importlib
import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _stub(name: str, attrs: dict | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    for attr, value in (attrs or {}).items():
        setattr(module, attr, value)
    sys.modules[name] = module
    return module


@pytest.fixture(scope="module")
def provider_module():
    saved = dict(sys.modules)
    try:
        _stub("megatron", is_package=True)
        core = _stub("megatron.core", is_package=True)
        core.tensor_parallel = SimpleNamespace(gather_from_sequence_parallel_region=MagicMock())
        _stub("megatron.core.models", is_package=True)
        _stub("megatron.core.models.gpt", {"GPTModel": type("GPTModel", (), {})}, is_package=True)
        _stub(
            "megatron.core.models.gpt.gpt_layer_specs",
            {
                "get_gpt_decoder_block_spec": MagicMock(),
                "get_gpt_layer_local_spec": MagicMock(),
                "get_gpt_layer_with_transformer_engine_spec": MagicMock(),
                "get_gpt_mtp_block_spec": MagicMock(),
            },
        )
        _stub("megatron.core.optimizer", is_package=True)
        _stub("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": type("DO", (), {})})
        _stub("megatron.core.optimizer.optimizer", {"ChainedOptimizer": type("CO", (), {})})
        _stub("megatron.core.transformer", is_package=True)
        _stub("megatron.core.transformer.spec_utils", {"import_module": MagicMock()})
        _stub("megatron.core.transformer.transformer_config", {"TransformerConfig": type("TransformerConfig", (), {})})
        _stub("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub("megatron.training", is_package=True)
        _stub("megatron.training.arguments", {"core_transformer_config_from_args": MagicMock()})
        _stub("megatron.bridge", {"AutoBridge": None})
        for name in [m for m in sys.modules if m.startswith("miles.backends.megatron_utils.model_provider")]:
            del sys.modules[name]
        yield importlib.import_module("miles.backends.megatron_utils.model_provider")
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


class _Args(Namespace):
    """Megatron-style args: every runtime-config field the copy reads, defaulting to None."""

    def __getattr__(self, name):  # only reached for attributes not set explicitly
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def _args(**overrides) -> _Args:
    base = dict(
        megatron_to_hf_mode="bridge",
        hf_checkpoint="/snap/qwen3.6",
        custom_model_provider_path=None,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        context_parallel_size=1,
        calculate_per_token_loss=False,
        variable_seq_lengths=False,
        attention_softmax_in_fp32=True,
        gradient_accumulation_fusion=False,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=None,
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe=None,
        attention_backend="flash",
        moe_token_dispatcher_type="alltoall",
        mtp_num_layers=0,
        enable_mtp_training=False,
        enable_witness=False,
    )
    return _Args(**{**base, **overrides})


class _Provider(SimpleNamespace):
    """What ``AutoBridge.to_megatron_provider`` returns for the Qwen3.6 snapshot: the HF
    config's mtp_num_hidden_layers=1 already applied by the bridge's CONFIG_MAPPING."""

    def __init__(self, **kw):
        super().__init__(mtp_num_layers=1, mtp_loss_scaling_factor=0.1, mtp_detach_heads=False, finalized=False, **kw)

    def finalize(self):
        self.finalized = True

    def provide(self, pre_process=True, post_process=True, vp_stage=None):
        # mirrors GPTModel: mtp_process <=> an MTP block spec was built (PP=1)
        inner = SimpleNamespace(mtp_process=bool(self.mtp_num_layers), config=self, forward=lambda *a, **k: "logits")
        return SimpleNamespace(language_model=inner, config=self, forward=inner.forward)


def _install_bridge(provider_module, provider):
    sys.modules["megatron.bridge"].AutoBridge = SimpleNamespace(
        from_hf_pretrained=lambda path, trust_remote_code: SimpleNamespace(
            to_megatron_provider=lambda load_weights: provider
        )
    )


def test_bridge_runtime_config_overrides_the_hf_derived_mtp_depth(provider_module):
    provider = _Provider()
    provider_module._apply_bridge_runtime_config(provider, _args(mtp_num_layers=0))
    assert provider.mtp_num_layers is None, "build_mtp_spec checks truthiness; None builds no MTP block"
    assert provider.mtp_loss_scaling_factor == 0.1 and provider.mtp_detach_heads is False

    kept = _Provider()
    provider_module._apply_bridge_runtime_config(kept, _args(mtp_num_layers=1, enable_mtp_training=True))
    assert kept.mtp_num_layers == 1 and kept.mtp_detach_heads is True, "raw-path detach rule mirrored"

    absent = _Provider()
    provider_module._apply_bridge_runtime_config(absent, _args(mtp_num_layers=None))
    assert absent.mtp_num_layers is None


def test_bridge_provider_builds_no_mtp_block_with_mtp_num_layers_0(provider_module):
    provider = _Provider()
    _install_bridge(provider_module, provider)
    provide = provider_module.get_model_provider_func(_args(mtp_num_layers=0))
    assert provider.finalized and provider.mtp_num_layers is None
    model = provide(pre_process=True, post_process=True)
    assert model.language_model.mtp_process is False
    assert model.forward() == "logits"


def test_bridge_provider_keeps_the_mtp_block_when_asked_for(provider_module):
    provider = _Provider()
    _install_bridge(provider_module, provider)
    provide = provider_module.get_model_provider_func(_args(mtp_num_layers=1))
    assert provider.mtp_num_layers == 1
    assert provide().language_model.mtp_process is True


def test_a_provider_that_ignores_the_override_fails_before_any_weight_load(provider_module):
    class _Stubborn(_Provider):
        def finalize(self):  # a future bridge re-deriving the depth from the HF config
            self.mtp_num_layers = 1
            self.finalized = True

    _install_bridge(provider_module, _Stubborn())
    with pytest.raises(ValueError, match="mtp_num_layers=1 disagrees with --mtp-num-layers None"):
        provider_module.get_model_provider_func(_args(mtp_num_layers=0))


def test_a_model_with_an_unwanted_mtp_block_is_rejected_at_provide(provider_module):
    class _LeakyProvider(_Provider):
        def provide(self, pre_process=True, post_process=True, vp_stage=None):
            inner = SimpleNamespace(mtp_process=True, config=self, forward=lambda *a, **k: None)
            return SimpleNamespace(language_model=inner, config=self, forward=inner.forward)

    _install_bridge(provider_module, _LeakyProvider())
    provide = provider_module.get_model_provider_func(_args(mtp_num_layers=0))
    with pytest.raises(ValueError, match="mtp_process is True but --mtp-num-layers is 0"):
        provide()
