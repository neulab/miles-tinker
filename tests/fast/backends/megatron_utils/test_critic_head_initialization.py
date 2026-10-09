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
            model.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
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
def test_bridge_zero_initializes_only_the_final_critic_head(provider_module, role, post_process):
    args = argparse.Namespace(megatron_to_hf_mode="bridge", hf_checkpoint="base")
    model = provider_module.get_model_provider_func(args, role=role)(post_process=post_process)
    head = model.output_layer
    if role != "critic" or not post_process:
        assert not isinstance(head, provider_module.LinearForLastLayer)
        assert head.out_features == 4
        return

    assert torch.count_nonzero(head.weight) == torch.count_nonzero(head.bias) == 0
    inputs = torch.ones(3, 4)
    values, _ = head(inputs)
    assert torch.count_nonzero(values) == 0

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    (values - 1).square().mean().backward()
    optimizer.step()
    assert torch.count_nonzero(head.weight) == head.weight.numel()
    assert torch.count_nonzero(head.bias) == head.bias.numel()


def test_default_head_initialization_remains_random(provider_module):
    head = provider_module.LinearForLastLayer(4, 1, config=SimpleNamespace(sequence_parallel=False))
    assert torch.count_nonzero(head.weight) > 0
    assert torch.count_nonzero(head.bias) == 0
