"""Bridge mode for megatron-bridge families built by its ModelBuilder (Muse Glimmer).

Such a family sets ``USE_MODEL_CONFIG_FOR_CONVERSION``; its ``to_megatron_provider()`` is a deprecated
generic GPTModelProvider (for Glimmer an empty one whose ``finalize()`` divides by zero). So miles takes
``get_model_config()``, copies the runtime config onto it as onto a provider, takes ``transformer_impl``
from args (the bridge leaves Glimmer at its unfused ``local`` attention), finalizes it once, and builds
each pipeline stage with the config's builder. A fake bridge and builder drive the real
``get_model_provider_func`` (megatron itself is stubbed by the shared fixture).
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_model_provider_bridge_mtp import (  # noqa: F401  (fixture)
    _args,
    _install_bridge,
    _Provider,
    provider_module,
)


class _ModelConfig(SimpleNamespace):
    """What ``AutoBridge.get_model_config()`` gives for Glimmer: the HF-derived layer pattern, megatron-bridge's
    ``local`` transformer_impl and the vision freeze fields, plus a count of ``finalize()`` calls."""

    def __init__(self, **kw):
        defaults = dict(
            transformer_impl="local",
            num_layers=4,
            window_size=(2047, 0),
            window_attn_skip_freq=[True, True, True, False],
            no_rope_freq=[False, False, False, True],
            attention_output_gate=True,
            output_multiplier=0.196,
            final_logit_softcapping=20.0,
            mtp_num_layers=None,
            freeze_vision_model=False,
            freeze_vision_projection=False,
            share_embeddings_and_output_weights=False,
            finalized=0,
        )
        super().__init__(**{**defaults, **kw})

    def finalize(self):
        self.finalized += 1

    def get_builder_cls(self):
        return _Builder


class _Stage(torch.nn.Module):
    """A built stage: the model returns (logits, loss_mask) when given a loss mask, as MuseGlimmerModel does."""

    def __init__(self, config, pre_process, post_process):
        super().__init__()
        self.config = config
        self.pre_process, self.post_process = pre_process, post_process
        self.mtp_process = False

    def forward(self, input_ids=None, loss_mask=None, **kwargs):
        return ("logits", loss_mask) if loss_mask is not None else "logits"


class _Builder:
    def __init__(self, model_config):
        self.model_config = model_config
        self.calls = []
        _Builder.last = self

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        self.calls.append((pg_collection, pre_process, post_process, vp_stage))
        return _Stage(self.model_config, pre_process, post_process)


def _install_builder_bridge(model_config):
    bridge = SimpleNamespace(
        _model_bridge=SimpleNamespace(USE_MODEL_CONFIG_FOR_CONVERSION=True),
        get_model_config=lambda: model_config,
        to_megatron_provider=lambda load_weights: pytest.fail("a builder-backed family took the provider path"),
    )
    sys.modules["megatron.bridge"].AutoBridge = SimpleNamespace(
        from_hf_pretrained=lambda path, trust_remote_code: bridge
    )


def _glimmer_args(**overrides):
    return _args(
        hf_checkpoint="/snap/muse-glimmer-30b",
        tensor_model_parallel_size=4,
        expert_model_parallel_size=1,
        transformer_impl="transformer_engine",
        freeze_vision_model=True,
        **overrides,
    )


def test_builder_family_builds_each_stage_with_its_builder(provider_module):  # noqa: F811  (the imported fixture)
    model_config = _ModelConfig()
    _install_builder_bridge(model_config)

    provide = provider_module.get_model_provider_func(_glimmer_args())

    # the runtime config reached the nested config once, before any stage is built
    assert model_config.finalized == 1
    assert model_config.tensor_model_parallel_size == 4 and model_config.sequence_parallel is True
    assert model_config.recompute_granularity == "full" and model_config.attention_backend == "flash"
    assert model_config.transformer_impl == "transformer_engine", "the bridge's local attention is O(S^2), no THD"
    assert model_config.freeze_vision_model and model_config.freeze_vision_projection
    assert model_config.mtp_num_layers is None and model_config.calculate_per_token_loss is False

    stage = provide(pre_process=True, post_process=True, vp_stage=None, pg_collection="pgs")
    assert _Builder.last.calls == [("pgs", True, True, None)]
    assert isinstance(stage, _Stage) and stage.config is model_config
    assert stage(input_ids=None, loss_mask="mask") == "logits", "(logits, loss_mask) is cut to the logits"
    assert stage(input_ids=None) == "logits"
    assert model_config.finalized == 1, "finalize() runs once for all stages"


def test_builder_stage_falls_back_to_the_mpu_groups(provider_module, monkeypatch):  # noqa: F811  (the fixture)
    groups = SimpleNamespace(use_mpu_process_groups=lambda: "mpu-groups")
    monkeypatch.setitem(
        sys.modules, "megatron.core.process_groups_config", SimpleNamespace(ProcessGroupCollection=groups)
    )
    _install_builder_bridge(_ModelConfig())
    provider_module.get_model_provider_func(_glimmer_args())(pre_process=True, post_process=False)
    assert _Builder.last.calls == [("mpu-groups", True, False, None)]


def test_builder_backed_family_refuses_context_parallelism(provider_module):  # noqa: F811  (the imported fixture)
    model_config = _ModelConfig()
    _install_builder_bridge(model_config)
    with pytest.raises(ValueError, match="sliced twice"):
        provider_module.get_model_provider_func(_glimmer_args(context_parallel_size=2))
    assert model_config.finalized == 0


def test_builder_backed_family_checks_the_mtp_depth(provider_module):  # noqa: F811  (the imported fixture)
    class _StubbornConfig(_ModelConfig):
        def finalize(self):
            super().finalize()
            self.mtp_num_layers = 1  # a config whose finalize() brings back an MTP block

    _install_builder_bridge(_StubbornConfig())
    with pytest.raises(ValueError, match="disagrees with --mtp-num-layers"):
        provider_module.get_model_provider_func(_glimmer_args(mtp_num_layers=0))


def test_a_critic_unties_the_output_layer_before_finalize(provider_module):  # noqa: F811  (the imported fixture)
    model_config = _ModelConfig(share_embeddings_and_output_weights=True)
    _install_builder_bridge(model_config)
    provider_module.get_model_provider_func(_glimmer_args(), role="critic")
    assert model_config.share_embeddings_and_output_weights is False and model_config.finalized == 1


def test_provider_backed_families_keep_the_provider_path(provider_module):  # noqa: F811  (the imported fixture)
    provider = _Provider()
    _install_bridge(provider_module, provider)  # a bridge without the builder flag
    provider_module.get_model_provider_func(_args(mtp_num_layers=0))
    assert provider.finalized
