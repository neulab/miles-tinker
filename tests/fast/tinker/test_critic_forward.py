"""Exercise the worker methods with a CPU schedule and native value alignment.

Only the selected methods are loaded from the actor: importing its full module
requires Megatron/CUDA. This checks worker orchestration, not GPU execution.
"""

import ast
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.training_utils.loss_hub import logit_processors
from miles.utils.ray_utils import Box

ACTOR_PATH = Path(__file__).resolve().parents[3] / "miles/backends/megatron_utils/actor.py"


def _worker_method(name, namespace):
    tree = ast.parse(ACTOR_PATH.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    method = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(ACTOR_PATH), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("last_stage", [True, False])
def test_value_forward_preserves_pre_action_alignment_without_backward(monkeypatch, last_stage):
    args = SimpleNamespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False, rollout_temperature=1.0)
    parallel = SimpleNamespace(cp=SimpleNamespace(size=1), is_pp_last_stage=last_stage)
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel)
    rows = [torch.tensor([10, 20, 30, 40]), torch.tensor([50, 60])]
    data = {"sample_indices": [4, 2]}
    model = Mock()
    schedule_calls = []

    def forward_only(callback, args, model_arg, iterator, microbatches, *, rollout_id):
        schedule_calls.append((model_arg, iterator, microbatches, rollout_id))
        # A deterministic causal value model: each position reports its prefix sum.
        logits = torch.cat([tokens.cumsum(0) for tokens in rows]).float().reshape(1, -1, 1)
        return callback(logits, args=args, unconcat_tokens=rows, total_lengths=[4, 2], response_lengths=[3, 1])

    namespace = dict(
        Box=Box,
        ExitStack=ExitStack,
        get_rollout_data=lambda args, ref: (data, nullcontext()),
        get_data_iterator=lambda args, model, data: (["iterator"], [1]),
        forward_only=forward_only,
        get_values=logit_processors.get_values,
        get_parallel_state=lambda: parallel,
        ray=SimpleNamespace(put=lambda value: value),
    )
    worker = SimpleNamespace(args=args, role="critic", model=model, _heartbeat=Mock(), _tinker_full_mode=lambda: True)
    result = _worker_method("forward_only_values", namespace)(worker, 7, Box("data"))
    assert schedule_calls == [(model, ["iterator"], [1], 7)]
    assert model.mock_calls == [], "the worker must not perform a backward or optimizer operation"
    if last_stage:
        assert result.inner["sample_indices"] == [4, 2]
        assert [value.tolist() for value in result.inner["values"]] == [[10.0, 30.0, 60.0], [50.0]]
        assert all(value.device.type == "cpu" and not value.requires_grad for value in result.inner["values"])
    else:
        assert result is None


def test_new_critic_handle_restores_initial_value_head_but_checkpoint_load_keeps_saved_head():
    head = torch.nn.Linear(2, 1)
    initial = {key: value.detach().clone() for key, value in head.state_dict().items()}
    optimizer, executor = Mock(), Mock()
    worker = SimpleNamespace(
        args=SimpleNamespace(),
        role="critic",
        model=[head],
        optimizer=optimizer,
        _tinker_full_mode=lambda: True,
        _critic_initial_head=[(head, initial)],
    )
    load_slot = _worker_method("load_slot", {"full_executor": executor})
    with torch.no_grad():
        head.weight.add_(10)
        head.bias.add_(10)
    load_slot(worker, 0, 0, 0.0)
    executor.reset_to_base.assert_called_once_with(worker.args, worker.model, optimizer)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, initial[key])
    optimizer.reload_model_params.assert_called_once()

    saved_head = {key: value + 3 for key, value in initial.items()}
    executor.load_state.side_effect = lambda *args: head.load_state_dict(saved_head)
    load_slot(worker, 0, 0, 0.0, "saved-critic", True)
    executor.load_state.assert_called_once_with(worker.args, worker.model, optimizer, "saved-critic", True)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, saved_head[key])
    assert optimizer.reload_model_params.call_count == 1

    # A later client gets the startup head, not the previous client's resumed head.
    def reload_model_params():
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, initial[key])

    optimizer.reload_model_params.side_effect = reload_model_params
    load_slot(worker, 0, 0, 0.0)
    assert optimizer.reload_model_params.call_count == 2
