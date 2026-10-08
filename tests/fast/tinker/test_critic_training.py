"""CPU integration of critic routing, real value loss, accumulation, and Adam.

The distributed worker schedule is replaced by a small trainable model; the
runtime, role router, value alignment, and PPO loss are the production code.
"""

import copy
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.tinker.harness import ADAM, FakeBackend

from miles.backends.training_utils.loss_hub import logit_processors
from miles.backends.training_utils.loss_hub.tinker_losses import (
    drain_per_datum_outputs,
    ppo_loss_function,
    start_per_datum_outputs,
)
from miles.tinker.full_backend import FullModelBackend
from miles.tinker.runtime import MilesBackend, _build_train_data


def test_critic_role_and_inputs_reach_the_microbatch(monkeypatch):
    # The data module imports dataset/tokenizer helpers requiring SGLang; none
    # are used by get_batch. Load it separately with only those helpers stubbed.
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.data",
        SimpleNamespace(get_minimum_num_micro_batch_size=None, process_rollout_data=None),
    )
    path = Path(__file__).resolve().parents[3] / "miles/backends/training_utils/data.py"
    spec = importlib.util.spec_from_file_location("miles.backends.training_utils._critic_test_data", path)
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    parallel = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1))
    monkeypatch.setattr(data_module, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(data_module, "slice_with_cp", lambda tensor, *args: tensor)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    data = dict(
        tokens=[torch.tensor([10, 20, 30])],
        loss_masks=[torch.tensor([1, 1])],
        total_lengths=[3],
        response_lengths=[2],
        values=[[0.0, 0.1]],
        returns=[[1.0, 2.0]],
        loss_weights=[[0.0, 0.5]],
        sample_indices=[7],
        model_role="critic",
        loss_fn="ppo",
        loss_fn_config={"value_clip": 0.3},
    )
    iterator = data_module.DataIterator(data, micro_batch_size=1)
    keys = [
        "tokens",
        "loss_masks",
        "total_lengths",
        "response_lengths",
        "values",
        "returns",
        "loss_weights",
        "sample_indices",
    ]
    batch = data_module.get_batch(iterator, keys, pad_multiplier=1, qkv_format="thd")
    for key in ("model_role", "loss_fn", "loss_fn_config", "values", "returns", "loss_weights", "sample_indices"):
        assert batch[key] == data[key]


def _loss(args, model, data):
    batch = {
        **data,
        "unconcat_tokens": [torch.tensor(tokens) for tokens in data["tokens"]],
        "loss_masks": [torch.tensor(mask) for mask in data["loss_masks"]],
    }
    inputs = torch.cat(batch["unconcat_tokens"]).float().view(-1, 1)
    logits = model(inputs).view(1, -1, 1)
    return ppo_loss_function(args, batch, logits, None)[0]


async def test_critic_accumulation_updates_backbone_head_and_adam_state(monkeypatch):
    args = Namespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tinker_train_mode="full",
        qkv_format="thd",
        true_on_policy_mode=False,
        allgather_cp=False,
        rollout_temperature=1.0,
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    model = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.1)
    reference = copy.deepcopy(model)
    before = [param.detach().clone() for param in model.parameters()]
    adam = {**ADAM, "learning_rate": 0.01, "weight_decay": 0.0}
    optim_kwargs = dict(lr=adam["learning_rate"], betas=(adam["beta1"], adam["beta2"]), eps=adam["eps"])
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    reference_optimizer = torch.optim.Adam(reference.parameters(), **optim_kwargs)

    async def optim_step(*, adam_params_by_slot):
        assert adam_params_by_slot == {0: adam}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return [{0: {"grad_norm": 1.0, "skipped": 0}}]

    async def run_unit(method, unit_id, data):
        assert method == "forward_backward"
        start_per_datum_outputs()
        try:
            _loss(args, model, data).backward()
        finally:
            outputs = drain_per_datum_outputs()
        return [{"per_datum": outputs}]

    critic = MilesBackend(args, SimpleNamespace(optim_step=optim_step), "http://unused", role="critic")
    monkeypatch.setattr(critic, "_run_unit", run_unit)
    policy = FakeBackend(full=True)
    backend = FullModelBackend(policy, critic)
    rows = [
        {"tokens": [1, 2, 3], "target_len": 2, "values": [0.0, 0.0], "returns": [2.0, 2.0], "weights": [0.0, 0.5]},
        {"tokens": [3, 4], "target_len": 1, "values": [0.0], "returns": [2.0], "weights": [1.0]},
    ]
    loss_config = {"value_clip": 10.0}
    outputs = []
    for unit_id, row in enumerate(rows):
        outputs.extend(await backend.forward_backward(unit_id, [(1, row)], "ppo", loss_config))
    assert len(outputs) == 2 and all("values" in output and "logprobs" not in output for output in outputs)
    for param, original in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(param, original)  # Backward accumulates; it does not step.
        assert param.grad is not None and param.grad.abs().sum() > 0
    assert not optimizer.state

    combined = _build_train_data([(0, row) for row in rows], with_adapter_slots=False)
    combined.update(model_role="critic", loss_fn="ppo", loss_fn_config=loss_config)
    reference_loss = _loss(args, reference, combined)
    assert sum(output["loss"] for output in outputs) == pytest.approx(reference_loss.item())
    reference_loss.backward()
    reference_optimizer.step()
    metrics = await backend.optim_step({1: adam})
    assert metrics[1]["num_units"] == metrics[1]["num_datums"] == 2
    for param, expected, original in zip(model.parameters(), reference.parameters(), before, strict=True):
        torch.testing.assert_close(param, expected)
        assert not torch.equal(param, original)
        state = optimizer.state[param]
        assert state["step"].item() == 1 and state["exp_avg"].abs().sum() > 0 and state["exp_avg_sq"].sum() > 0
        assert param.grad is None
    assert policy.calls == [], "critic updates must not reach the policy"
