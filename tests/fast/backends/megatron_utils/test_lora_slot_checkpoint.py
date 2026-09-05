"""Slot checkpoints are slot- and DP-layout-independent: optimizer state travels by
parameter name, so a save from slot 0 under one LayerWise ownership loads into any
slot under any other. A load validates before it mutates, and every phase raises the
same SlotCheckpointError on every rank. Megatron and the bridge are stubbed."""

import importlib
import logging
import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CHECKPOINT = "miles.backends.megatron_utils.lora.checkpoint"
SHAPES = {
    "l0.adapter.linear_in.weight": (4, 2),
    "l0.adapter.linear_out.weight": (2, 4),
    "l1.adapter.linear_in.weight": (4, 2),
    "l1.adapter.linear_out.weight": (2, 4),
}
NAMES = list(SHAPES)


class FakeChunk:
    """Per-slot adapter params; expose_adapter_slot selects which slot named_parameters shows."""

    def __init__(self, slots: dict[int, dict[str, torch.nn.Parameter]]) -> None:
        self.slots = slots
        self.exposed: int | None = None

    def named_parameters(self):
        assert self.exposed is not None, "adapter params are only visible under expose_adapter_slot"
        return list(self.slots[self.exposed].items())


@contextmanager
def expose_adapter_slot(model, slot):
    for chunk in model:
        chunk.exposed = slot
    try:
        yield
    finally:
        for chunk in model:
            chunk.exposed = None


def load_adapter(model, slot, state_dict):
    loaded = 0
    with expose_adapter_slot(model, slot):
        for chunk in model:
            for name, param in chunk.named_parameters():
                param.data.copy_(state_dict[name].to(param.dtype))
                loaded += 1
    return loaded


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def ckpt(monkeypatch):
    """The checkpoint module imported against stubs, as rank 0 of a one-rank job."""
    stubs = {
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core"),
        "megatron.core.distributed": _module("megatron.core.distributed", DistributedDataParallel=object),
        "megatron.core.optimizer": _module("megatron.core.optimizer", MegatronOptimizer=object),
        "megatron.bridge.peft.multi_lora_layers": _module(
            "megatron.bridge.peft.multi_lora_layers",
            expose_adapter_slot=expose_adapter_slot,
            load_adapter=load_adapter,
        ),
        "miles.backends.megatron_utils.lora.optimizer": _module(
            "miles.backends.megatron_utils.lora.optimizer",
            _slot_children=lambda optimizer, slot: optimizer.children[slot],
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, CHECKPOINT, raising=False)
    module = importlib.import_module(CHECKPOINT)
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0), ep=SimpleNamespace(rank=0, size=1)
    )
    monkeypatch.setattr(module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(module, "adapter_shard_topology", lambda: (True, ((0, 0, 0),)))
    become(module, monkeypatch, rank=0, world=1)
    yield module
    sys.modules.pop(CHECKPOINT, None)


def become(module, monkeypatch, *, rank: int, world: int) -> None:
    """Act as one rank of a ``world``-rank job whose ranks all report this rank's phase outcome."""
    monkeypatch.setattr(module, "_rank", lambda: rank)
    monkeypatch.setattr(module, "_world_size", lambda: world)
    monkeypatch.setattr(module, "_all_gather", lambda value: [value] * world)


def make_model(n_slots: int = 2, shapes: dict[str, tuple] = SHAPES) -> list[FakeChunk]:
    return [
        FakeChunk(
            {
                slot: {name: torch.nn.Parameter(torch.zeros(shape)) for name, shape in shapes.items()}
                for slot in range(n_slots)
            }
        )
    ]


def seed(name: str) -> float:
    return float(NAMES.index(name) + 1)


def fill_slot(model: list[FakeChunk], slot: int) -> None:
    for name, param in model[0].slots[slot].items():
        param.data.fill_(seed(name))


def make_optimizer(model, slot: int, owned: list[str], *, step: int = 0, stepped: bool = False):
    """LayerWise-shaped fake: one Float16 child whose fp32 masters are this rank's owned subset of
    ``slot``; ``stepped`` gives every master distinctive Adam moments."""
    model_params = [model[0].slots[slot][name] for name in owned]
    masters = [param.detach().clone().float() for param in model_params]
    state: dict = defaultdict(dict)
    if stepped:
        for name, master in zip(owned, masters, strict=True):
            state[master] = {
                "exp_avg": torch.full_like(master, seed(name)),
                "exp_avg_sq": torch.full_like(master, -seed(name)),
            }
    inner = SimpleNamespace(param_groups=[{"params": masters, "step": step}], state=state)
    child = SimpleNamespace(
        optimizer=inner,
        float16_groups=[model_params],
        fp32_from_float16_groups=[masters],
        fp32_from_fp32_groups=[[]],
    )
    return SimpleNamespace(
        children={slot: [child]},
        child=child,
        masters=dict(zip(owned, masters, strict=True)),
        reload_model_params=lambda: None,
    )


def saved_checkpoint(ckpt, tmp_path, *, owned: list[str] = NAMES, world: int = 1, stepped: bool = True) -> str:
    """Slot 0 with distinctive weights saved by rank 0 of ``world`` owning ``owned``."""
    model = make_model()
    fill_slot(model, 0)
    path = str(tmp_path / "ckpt")
    with pytest.MonkeyPatch.context() as mp:
        become(ckpt, mp, rank=0, world=world)
        ckpt.save_slot(model, make_optimizer(model, 0, owned, step=7, stepped=stepped), 0, path)
    return path


# -------- pure helpers --------


def test_main_params_pair_masters_with_model_params(ckpt):
    a, b = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16)), torch.nn.Parameter(
        torch.zeros(3, dtype=torch.bfloat16)
    )
    a32, b32 = a.detach().float(), b.detach().float()
    c = torch.nn.Parameter(torch.zeros(1))
    child = SimpleNamespace(
        float16_groups=[[a], [b]], fp32_from_float16_groups=[[a32], [b32]], fp32_from_fp32_groups=[[], [c]]
    )

    pairs = ckpt._main_params_by_model_param(child)

    assert [(id(model), id(main)) for model, main in pairs] == [(id(a), id(a32)), (id(b), id(b32)), (id(c), id(c))]


def test_merge_unions_the_replicas_and_takes_the_lockstep_step(ckpt):
    shards = [
        {"replicas": 2, "group_steps": [7, 0], "params": {"a": {"x": 1}}},
        {"replicas": 2, "group_steps": [7], "params": {"b": {"x": 2}}},
    ]
    assert ckpt._merge_optimizer_shards(shards) == {"params": {"a": {"x": 1}, "b": {"x": 2}}, "step": 7}
    with pytest.raises(ValueError, match="incomplete"):
        ckpt._merge_optimizer_shards(shards[:1])


def test_owned_state_is_keyed_by_name_and_a_never_stepped_slot_is_empty(ckpt):
    model = make_model()
    optimizer = make_optimizer(model, 0, NAMES[:2], step=3, stepped=False)
    names = {id(param): name for name, param in model[0].slots[0].items()}

    params, group_steps = ckpt._owned_optimizer_state(optimizer, 0, names)

    assert params == {NAMES[0]: {}, NAMES[1]: {}} and group_steps == [3]


# -------- save / load --------


def test_save_publishes_one_weight_shard_and_one_optimizer_file_per_rank(ckpt, tmp_path):
    path = Path(saved_checkpoint(ckpt, tmp_path))
    assert sorted(entry.name for entry in path.iterdir()) == ["adapter_megatron_tp0_pp0.pt", "optim_tp0_pp0_rank0.pt"]
    assert not (tmp_path / "_tmp_ckpt").exists()
    shard = torch.load(path / "optim_tp0_pp0_rank0.pt", weights_only=True)
    assert (shard["world_size"], shard["shard"], shard["replicas"], shard["group_steps"]) == (1, "tp0_pp0", 1, [7])
    assert set(shard["params"]) == set(NAMES)


def test_state_follows_names_into_another_slot_and_another_ownership(ckpt, tmp_path, monkeypatch):
    model = make_model()
    fill_slot(model, 0)
    path = str(tmp_path / "ckpt")
    tag = ckpt._shard_tag()
    # save-time world: two DP replicas with ping-pong ownership of slot 0
    become(ckpt, monkeypatch, rank=0, world=2)
    ckpt.save_slot(model, make_optimizer(model, 0, NAMES[0::2], step=7, stepped=True), 0, path)
    become(ckpt, monkeypatch, rank=1, world=2)
    names = {id(param): name for name, param in ckpt._exposed_adapter_params(model, 0).items()}
    replica = ckpt._optimizer_shard(make_optimizer(model, 0, NAMES[1::2], step=7, stepped=True), 0, names, tag, 2)
    torch.save(replica, Path(path) / ckpt._optim_shard_name(tag))

    # load-time world: three replicas, a fresh slot 1 owning yet another subset here
    become(ckpt, monkeypatch, rank=2, world=3)
    target = make_optimizer(model, 1, [NAMES[0], NAMES[3]])
    ckpt.load_slot(model, target, 1, path, load_optimizer=True)

    for name in NAMES:
        assert torch.equal(model[0].slots[1][name], model[0].slots[0][name])
    for name, master in target.masters.items():
        state = target.child.optimizer.state[master]
        assert torch.equal(state["exp_avg"], torch.full_like(master, seed(name)))
        assert torch.equal(state["exp_avg_sq"], torch.full_like(master, -seed(name)))
    assert target.child.optimizer.param_groups[0]["step"] == 7


def test_tensor_step_clocks_are_restored_in_place(ckpt, tmp_path):
    path = saved_checkpoint(ckpt, tmp_path)
    model = make_model()
    target = make_optimizer(model, 1, NAMES)
    step = torch.tensor(0)
    target.child.optimizer.param_groups[0]["step"] = step

    ckpt.load_slot(model, target, 1, path, load_optimizer=True)

    assert int(step) == 7


def test_unsaved_state_stays_fresh_with_a_warning(ckpt, tmp_path, caplog):
    path = saved_checkpoint(ckpt, tmp_path, owned=NAMES[:1])
    model = make_model()
    target = make_optimizer(model, 1, NAMES)

    with caplog.at_level(logging.WARNING):
        ckpt.load_slot(model, target, 1, path, load_optimizer=True)

    assert "3 owned param(s) had no saved optimizer state" in caplog.text
    state = target.child.optimizer.state
    assert torch.equal(state[target.masters[NAMES[0]]]["exp_avg"], torch.full(SHAPES[NAMES[0]], seed(NAMES[0])))
    assert all(state[target.masters[name]] == {} for name in NAMES[1:])


def test_a_never_stepped_checkpoint_loads_silently(ckpt, tmp_path, caplog):
    path = saved_checkpoint(ckpt, tmp_path, stepped=False)
    model = make_model()
    target = make_optimizer(model, 1, NAMES)
    with caplog.at_level(logging.WARNING):
        ckpt.load_slot(model, target, 1, path, load_optimizer=True)
    assert caplog.text == ""
    assert all(target.child.optimizer.state[master] == {} for master in target.masters.values())


def test_a_weights_only_load_keeps_the_fresh_optimizer_state(ckpt, tmp_path):
    path = saved_checkpoint(ckpt, tmp_path)
    for optim_file in Path(path).glob("optim_*.pt"):
        optim_file.unlink()
    model = make_model()
    target = make_optimizer(model, 1, NAMES, step=0)

    ckpt.load_slot(model, target, 1, path, load_optimizer=False)

    assert torch.equal(model[0].slots[1][NAMES[0]], torch.full(SHAPES[NAMES[0]], seed(NAMES[0])))
    assert target.child.optimizer.param_groups[0]["step"] == 0
    with pytest.raises(ckpt.SlotCheckpointError, match="no optimizer state"):
        ckpt.load_slot(model, target, 1, path, load_optimizer=True)


# -------- validation before mutation, consistent failure --------


def _untouched(model, target) -> bool:
    params = model[0].slots[1].values()
    return all(float(param.detach().abs().sum()) == 0.0 for param in params) and not target.child.optimizer.state


def test_a_missing_checkpoint_raises_before_touching_the_slot(ckpt, tmp_path):
    model = make_model()
    target = make_optimizer(model, 1, NAMES)
    with pytest.raises(ckpt.SlotCheckpointError, match="no adapter shard"):
        ckpt.load_slot(model, target, 1, str(tmp_path / "nope"), load_optimizer=True)
    assert _untouched(model, target)


def test_a_checkpoint_of_another_shape_is_rejected_before_mutation(ckpt, tmp_path):
    path = saved_checkpoint(ckpt, tmp_path)
    model = make_model(shapes={name: (rows * 2, cols) for name, (rows, cols) in SHAPES.items()})
    target = make_optimizer(model, 1, NAMES)
    with pytest.raises(ckpt.SlotCheckpointError, match="shape"):
        ckpt.load_slot(model, target, 1, path, load_optimizer=True)
    assert _untouched(model, target)


def test_a_checkpoint_of_other_modules_is_rejected(ckpt, tmp_path):
    path = saved_checkpoint(ckpt, tmp_path)
    model = make_model(shapes={**SHAPES, "l2.adapter.linear_in.weight": (4, 2)})
    with pytest.raises(ckpt.SlotCheckpointError, match="1 missing"):
        ckpt.load_slot(model, make_optimizer(model, 1, NAMES), 1, path, load_optimizer=False)


def test_an_incomplete_replica_set_is_rejected(ckpt, tmp_path):
    path = saved_checkpoint(ckpt, tmp_path, world=2)  # rank 1's file never arrived
    model = make_model()
    target = make_optimizer(model, 1, NAMES)
    with pytest.raises(ckpt.SlotCheckpointError, match="incomplete"):
        ckpt.load_slot(model, target, 1, path, load_optimizer=True)
    assert _untouched(model, target)


def test_an_apply_failure_is_a_slot_checkpoint_error(ckpt, tmp_path):
    path = saved_checkpoint(ckpt, tmp_path)
    model = make_model()
    target = make_optimizer(model, 1, NAMES)

    def boom():
        raise RuntimeError("boom")

    target.reload_model_params = boom
    with pytest.raises(ckpt.SlotCheckpointError, match="RuntimeError: boom"):
        ckpt.load_slot(model, target, 1, path, load_optimizer=True)


def test_a_peer_failure_raises_here_too_and_drops_the_partial_save(ckpt, tmp_path, monkeypatch):
    good = saved_checkpoint(ckpt, tmp_path)
    monkeypatch.setattr(ckpt, "_all_gather", lambda value: [value, "rank 1: OSError: disk full"])
    model = make_model()
    with pytest.raises(ckpt.SlotCheckpointError, match="1/2 rank.*disk full"):
        ckpt.save_slot(model, make_optimizer(model, 0, NAMES), 0, str(tmp_path / "other"))
    assert not (tmp_path / "_tmp_other").exists() and not (tmp_path / "other").exists()
    target = make_optimizer(model, 1, NAMES)
    with pytest.raises(ckpt.SlotCheckpointError, match="disk full"):
        ckpt.load_slot(model, target, 1, good, load_optimizer=True)
    assert _untouched(model, target)
