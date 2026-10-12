"""Exercise head construction and bridge wiring on CPU, without Megatron imports."""

import argparse
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal

import pytest
import torch


@pytest.fixture
def provider_module(monkeypatch):
    # Load the real definitions while avoiding unrelated Megatron/CUDA imports.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/model_provider.py"
    definitions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"LinearForLastLayer", "get_model_provider_func"}
    ]
    namespace = dict(
        torch=torch,
        argparse=argparse,
        Literal=Literal,
        TransformerConfig=SimpleNamespace,
        GPTModel=torch.nn.Module,
        _apply_bridge_runtime_config=lambda provider, args: None,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)

    class FakeProvider:
        def finalize(self):
            pass

        def provide(self, **kwargs):
            model = torch.nn.Linear(4, 4)
            model.config = SimpleNamespace(hidden_size=2560, sequence_parallel=False)
            model.output_layer = torch.nn.Linear(4, 4)
            return model

    bridge = ModuleType("megatron.bridge")
    bridge.AutoBridge = SimpleNamespace(
        from_hf_pretrained=lambda *args, **kwargs: SimpleNamespace(
            to_megatron_provider=lambda **kwargs: FakeProvider()
        )
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("role,post_process", [("actor", True), ("critic", False), ("critic", True)])
def test_bridge_initializes_only_the_final_critic_head(provider_module, role, post_process):
    args = argparse.Namespace(megatron_to_hf_mode="bridge", hf_checkpoint="base")
    model = provider_module.get_model_provider_func(args, role=role)(post_process=post_process)
    head = model.output_layer
    if role != "critic" or not post_process:
        assert not isinstance(head, provider_module.LinearForLastLayer)
        assert head.out_features == 4
        return

    assert head.weight.shape == (1, model.config.hidden_size)
    assert torch.count_nonzero(head.weight) > 0
    assert torch.count_nonzero(head.bias) == 0
    expected_std = 1 / (model.config.hidden_size + 1)
    assert head.weight.std().item() == pytest.approx(expected_std, rel=0.1)
    inputs = torch.ones(3, model.config.hidden_size)
    values, _ = head(inputs)
    assert torch.isfinite(values).all()
    initial_weight = head.weight.detach().clone()

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    (values - 1).square().mean().backward()
    optimizer.step()
    assert not torch.equal(head.weight, initial_weight)
    assert torch.count_nonzero(head.bias) == head.bias.numel()


@pytest.mark.parametrize("hidden_size", [2560, 4096])
def test_default_head_initialization_scales_with_hidden_size(provider_module, hidden_size):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        head = provider_module.LinearForLastLayer(hidden_size, 1, config=SimpleNamespace(sequence_parallel=False))
    expected_std = 1 / (hidden_size + 1)
    assert head.weight.std().item() == pytest.approx(expected_std, rel=0.1)
    assert abs(head.weight.mean().item()) < expected_std * 0.1
    assert torch.count_nonzero(head.bias) == 0


def test_explicit_zero_initialization_is_still_supported(provider_module):
    head = provider_module.LinearForLastLayer(4, 1, config=SimpleNamespace(sequence_parallel=False), init_std=0.0)
    assert torch.count_nonzero(head.weight) == torch.count_nonzero(head.bias) == 0
