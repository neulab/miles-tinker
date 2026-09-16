"""The full fine-tuning executor's optim window: finalize exactly once per
window, AdamParams applied per step, non-finite grads vetoed, windows
discarded on failure, rollout routing replay (R3) filled and cleared around
the pass. Megatron and the heavy miles modules are stubbed; the test drives
the executor's own state machine."""

import importlib
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

# imported before the rig installs its module stubs so the real module binds its real dependencies
import miles.backends.training_utils.replay_data as replay_data_mod
from miles.utils.replay_base import Replay, indexer_replay_manager, routing_replay_manager

EXECUTOR = "miles.backends.megatron_utils.full.executor"
ADAM = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip_norm": 1.0,
}


class FakeOptimizer:
    """Records the Megatron optimizer API calls the executor makes."""

    def __init__(self, grad_norm: float = 0.5, found_inf: bool = False) -> None:
        self.config = SimpleNamespace(clip_grad=1.0)
        self.inner = SimpleNamespace(
            param_groups=[{"params": [], "lr": 0.0, "wd_mult": 0.0, "step": 7}],
            state={"p": {"exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2), "step": torch.tensor(7)}},
        )
        self.chained_optimizers = [SimpleNamespace(optimizer=self.inner, config=self.config)]
        self.grad_norm = grad_norm
        self.found_inf = found_inf
        self.calls: list[str] = []

    @property
    def param_groups(self):
        return self.inner.param_groups

    def prepare_grads(self):
        self.calls.append("prepare_grads")
        return self.found_inf

    def get_grad_norm(self):
        self.calls.append("get_grad_norm")
        return self.grad_norm

    def step(self):
        self.calls.append("step")
        return True, self.grad_norm, None

    def zero_grad(self):
        self.calls.append("zero_grad")

    def reload_model_params(self):
        self.calls.append("reload_model_params")


class FakeModelChunk:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def train(self):
        self.calls.append("train")

    def zero_grad_buffer(self):
        self.calls.append("zero_grad_buffer")


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def rig(monkeypatch):
    """Import the executor against stubs and expose the calls it makes."""
    log: list[str] = []
    config = SimpleNamespace(finalize_model_grads_func=None)

    def run_forward_backward_pass(args, dumper, data_iterator, model, num_microbatches, num_rollouts):
        log.append("fwd_bwd")
        # the schedule ends by calling the (now deferred) finalize hook
        config.finalize_model_grads_func([model], None)
        return [{"loss": torch.tensor(1.0)}]

    def finalize(model, num_tokens, **kwargs):
        log.append("finalize")

    def zero_grads(model, optimizer, disable_optimizer):
        log.append("zero_grads")

    model_stub = _module(
        "miles.backends.megatron_utils.model",
        _has_loadable_ckpt=lambda path: bool(path),
        _zero_grads=zero_grads,
        finalize_model_grads_with_empty_cache=finalize,
        run_forward_backward_pass=run_forward_backward_pass,
        save=lambda iteration, model, optimizer, sched: log.append(f"save:{iteration}"),
        setup_train_iteration_config=lambda args, model, optimizer, disable_optimizer: config,
    )
    stubs = {
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core"),
        "megatron.core.distributed": _module("megatron.core.distributed", DistributedDataParallel=object),
        "megatron.core.optimizer": _module("megatron.core.optimizer", MegatronOptimizer=object),
        "miles.backends.megatron_utils.model": model_stub,
        "miles.backends.megatron_utils.checkpoint": _module(
            "miles.backends.megatron_utils.checkpoint",
            load_checkpoint=lambda *a, **k: log.append("load_checkpoint"),
        ),
        "miles.backends.megatron_utils.lora.optimizer": _module(
            "miles.backends.megatron_utils.lora.optimizer",
            reset_grad_metadata_keep_grads=lambda model: log.append("reset_grad_metadata"),
        ),
        "miles.backends.training_utils.data": _module(
            "miles.backends.training_utils.data",
            get_data_iterator=lambda args, model, rollout_data: ([SimpleNamespace(reset=lambda: None)], [1]),
        ),
        "miles.backends.training_utils.log_utils": _module("miles.backends.training_utils.log_utils", aggregate_train_losses=lambda losses, n: {"loss": 1.0}),
        "miles.backends.training_utils.loss_hub.tinker_losses": _module(
            "miles.backends.training_utils.loss_hub.tinker_losses",
            start_per_datum_outputs=lambda: None,
            drain_per_datum_outputs=lambda: [{"sample_index": 0, "loss": torch.tensor(1.0), "logprobs": torch.tensor([-1.0])}],
        ),
        "miles.backends.training_utils.parallel": _module(
            "miles.backends.training_utils.parallel",
            get_parallel_state=lambda: SimpleNamespace(is_pp_last_stage=True),
        ),
        "miles.utils.dumper_utils": _module(
            "miles.utils.dumper_utils",
            DumperMegatronUtil=lambda *a, **k: SimpleNamespace(finalize=lambda model: None),
            DumperPhase=SimpleNamespace(FWD_BWD="fwd_bwd"),
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in (EXECUTOR, "miles.backends.megatron_utils.full"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    executor = importlib.import_module(EXECUTOR)
    executor._state.update(pending_finalize=False, dirty=False, num_updates=0)
    from miles.backends.megatron_utils import tinker_perf

    tinker_perf.reset_window()
    monkeypatch.setattr(tinker_perf, "pop_optim_window_metrics", lambda args: {"actor_train_time": 0.5})
    from miles.utils.timer import Timer

    Timer().seq_lens = [3, 3]
    args = Namespace(overlap_grad_reduce=False, overlap_param_gather=False, load="/base", mfu_peak_tflops=None)
    rig = SimpleNamespace(executor=executor, log=log, config=config, args=args, model=[FakeModelChunk()])
    yield rig
    for name in (EXECUTOR, "miles.backends.megatron_utils.full"):
        sys.modules.pop(name, None)


def test_units_defer_finalize_and_optim_step_runs_it_once(rig):
    optimizer = FakeOptimizer()
    for unit_id in (1, 2):
        out = rig.executor.forward_backward(rig.args, unit_id, rig.model, optimizer, {})
        assert out["metrics"] == {"loss": 1.0}
        assert out["per_datum"][0]["sample_index"] == 0
        assert out["per_datum"][0]["loss"].item() == 1.0
    assert rig.log == ["reset_grad_metadata", "fwd_bwd", "reset_grad_metadata", "fwd_bwd"], "no collective per unit"
    assert rig.executor._state["pending_finalize"] is True

    metrics = rig.executor.optim_step(rig.args, rig.model, optimizer, ADAM)

    assert rig.log[4:] == ["finalize", "zero_grads"], "one finalize for the whole window, grads cleared after"
    assert optimizer.calls == ["prepare_grads", "get_grad_norm", "step"]
    assert metrics == {
        "grad_norm": 0.5,
        "skipped": 0,
        "lr": 1e-4,
        "num_updates": 1,
        "actor_train_time": 0.5,
    }
    assert rig.executor._state["pending_finalize"] is False


def test_optim_step_applies_adam_params_and_clip(rig):
    optimizer = FakeOptimizer()
    rig.executor.optim_step(rig.args, rig.model, optimizer, {**ADAM, "grad_clip_norm": 0.0})
    group = optimizer.param_groups[0]
    assert group["lr"] == 1e-4 and group["betas"] == (0.9, 0.95) and group["eps"] == 1e-8
    assert group["weight_decay"] == 0.0, "wd_mult 0 (norm/bias group) masks the client's weight decay"
    assert optimizer.config.clip_grad == 0.0, "Tinker's 0.0 means no clipping"


def test_non_finite_grad_norm_vetoes_the_step(rig):
    optimizer = FakeOptimizer(grad_norm=float("nan"))
    rig.executor.forward_backward(rig.args, 1, rig.model, optimizer, {})
    metrics = rig.executor.optim_step(rig.args, rig.model, optimizer, ADAM)
    assert "step" not in optimizer.calls
    assert metrics["skipped"] == 1 and metrics["num_updates"] == 0
    assert metrics["grad_norm"] != metrics["grad_norm"]
    assert rig.log[-2:] == ["finalize", "zero_grads"], "vetoed grads are discarded, not kept for the next window"


def test_failed_unit_poisons_the_window(rig):
    optimizer = FakeOptimizer()

    def boom(*args, **kwargs):
        raise RuntimeError("cuda oom")

    sys.modules["miles.backends.megatron_utils.model"].run_forward_backward_pass = boom
    rig.executor.run_forward_backward_pass = boom
    with pytest.raises(RuntimeError):
        rig.executor.forward_backward(rig.args, 1, rig.model, optimizer, {})
    assert rig.executor._state["dirty"] is True

    metrics = rig.executor.optim_step(rig.args, rig.model, optimizer, ADAM)
    assert metrics["skipped"] == 1 and optimizer.calls == []
    assert rig.log[-1] == "zero_grads" and "finalize" not in rig.log
    assert rig.executor._state["dirty"] is False


def test_nonfinite_reporting_vetoes_the_whole_window_and_next_window_recovers(rig, monkeypatch):
    optimizer = FakeOptimizer()
    bad = [{"sample_index": 0, "loss": torch.tensor(float("nan")), "logprobs": torch.tensor([-1.0])}]
    monkeypatch.setattr(rig.executor, "drain_per_datum_outputs", lambda: bad)
    out = rig.executor.forward_backward(rig.args, 1, rig.model, optimizer, {})
    assert out["per_datum"][0]["loss"].item() == 0.0
    assert out["per_datum"][0]["nonfinite_loss"] == 1
    metrics = rig.executor.optim_step(rig.args, rig.model, optimizer, ADAM)
    assert metrics["skipped"] == 1 and metrics["nonfinite_output"] == 1
    assert metrics["num_updates"] == 0 and optimizer.calls == []
    assert "finalize" not in rig.log
    assert not rig.executor._state["nonfinite_output"]
    monkeypatch.setattr(rig.executor, "drain_per_datum_outputs", lambda: [])
    rig.executor.forward_backward(rig.args, 2, rig.model, optimizer, {})
    recovered = rig.executor.optim_step(rig.args, rig.model, optimizer, ADAM)
    assert recovered["skipped"] == 0 and recovered["num_updates"] == 1
    assert optimizer.calls.count("step") == 1


def test_nonfinite_on_a_remote_rank_vetoes_local_optimizer_without_finalize(rig, monkeypatch):
    def remote_failure(flag, op, group):
        assert group == "all-trainer-ranks" and op == rig.executor.dist.ReduceOp.MAX
        assert flag.item() == 0
        flag.fill_(1)

    with monkeypatch.context() as distributed:
        distributed.setattr(rig.executor.dist, "is_initialized", lambda: True)
        distributed.setattr(rig.executor, "get_gloo_group", lambda: "all-trainer-ranks")
        distributed.setattr(rig.executor.dist, "all_reduce", remote_failure)
        rig.executor._check_per_datum_outputs(1, [], {})
    optimizer = FakeOptimizer()
    metrics = rig.executor.optim_step(rig.args, rig.model, optimizer, ADAM)
    assert metrics["skipped"] == 1 and metrics["nonfinite_output"] == 1
    assert optimizer.calls == [] and "finalize" not in rig.log


def test_discard_grads_clears_pending_finalize(rig):
    optimizer = FakeOptimizer()
    rig.executor.forward_backward(rig.args, 1, rig.model, optimizer, {})
    rig.executor.discard_grads(rig.model, optimizer)
    assert rig.executor._state["pending_finalize"] is False
    rig.executor.optim_step(rig.args, rig.model, optimizer, ADAM)
    assert "finalize" not in rig.log, "nothing accumulated: no collective to run"


def test_reset_optimizer_state_zeroes_moments_and_clocks(rig):
    optimizer = FakeOptimizer()
    rig.executor.reset_optimizer_state(optimizer)
    state = optimizer.inner.state["p"]
    assert state["exp_avg"].sum() == 0 and state["exp_avg_sq"].sum() == 0 and state["step"].item() == 0
    assert optimizer.inner.param_groups[0]["step"] == 0


def test_save_state_uses_a_temporary_synchronous_save_dir(rig):
    rig.args.save = "/native"
    rig.args.async_save = True
    rig.executor.save_state(rig.args, rig.model, FakeOptimizer(), "/ckpt/weights/step-1")
    assert rig.log == [f"save:{rig.executor.SAVE_ITERATION}"]
    assert (rig.args.save, rig.args.async_save) == ("/native", True), "args restored after the save"


def test_load_state_without_optimizer_gives_a_fresh_optimizer(rig):
    rig.args.load, rig.args.finetune, rig.args.no_load_optim, rig.args.no_load_rng, rig.args.ckpt_step = (
        "/base",
        True,
        True,
        True,
        None,
    )
    optimizer = FakeOptimizer()
    rig.executor.load_state(rig.args, rig.model, optimizer, "/ckpt/weights/step-1", load_optimizer=False)
    assert rig.log == ["load_checkpoint", "zero_grads"]
    assert optimizer.calls == ["reload_model_params"], "masters refreshed from the loaded weights"
    assert optimizer.inner.state["p"]["step"].item() == 0, "weights-only load means a fresh optimizer"
    assert rig.args.load == "/base", "args restored after the load"


def test_reset_to_base_reloads_the_startup_weights(rig):
    rig.args.finetune, rig.args.no_load_optim, rig.args.no_load_rng = False, False, False
    optimizer = FakeOptimizer()
    rig.executor._state["num_updates"] = 3
    rig.executor.reset_to_base(rig.args, rig.model, optimizer)
    assert rig.log == ["load_checkpoint", "zero_grads"]
    assert optimizer.calls == ["reload_model_params"]
    assert optimizer.inner.state["p"]["step"].item() == 0
    assert rig.executor._state["num_updates"] == 0
    assert (rig.args.finetune, rig.args.no_load_optim, rig.args.no_load_rng) == (False, False, False)


# ------------------------------- rollout routing replay (R3) -------------------------------


def _replay_rig(rig, monkeypatch, *, flag: bool, enabled: bool):
    """Point the routing manager at one queued replay and record the fills the executor asks for."""
    rig.args.use_rollout_routing_replay = flag
    replay = Replay()
    replay.top_indices_list = [torch.zeros(2, 3, dtype=torch.int32)]
    monkeypatch.setattr(routing_replay_manager, "enabled", enabled)
    monkeypatch.setattr(routing_replay_manager, "stage", "fallthrough")
    monkeypatch.setattr(routing_replay_manager, "replays", [replay])
    monkeypatch.setattr(routing_replay_manager, "register_replay_list_func", _register_moe)
    monkeypatch.setattr(indexer_replay_manager, "enabled", False)
    fills: list[dict] = []

    def fill_replay_data(**kwargs):
        fills.append({**kwargs, "stage": routing_replay_manager.stage})
        del kwargs["rollout_data"][kwargs["data_key"]]

    monkeypatch.setattr(replay_data_mod, "fill_replay_data", fill_replay_data)
    stages: list[str] = []
    run = rig.executor.run_forward_backward_pass

    def spy(*args, **kwargs):
        stages.append(routing_replay_manager.stage)
        return run(*args, **kwargs)

    rig.executor.run_forward_backward_pass = spy
    return SimpleNamespace(fills=fills, stages=stages, replay=replay)


def _register_moe(replay_list, replay_data, *, models, **_kwargs):
    raise AssertionError("the stubbed fill never registers")


def test_r3_fills_the_routing_replay_and_clears_it_after_the_pass(rig, monkeypatch):
    r3 = _replay_rig(rig, monkeypatch, flag=True, enabled=True)
    rollout_data = {"rollout_routed_experts": ["rows"], "tokens": ["t"]}

    rig.executor.forward_backward(rig.args, 1, rig.model, FakeOptimizer(), rollout_data)

    assert len(r3.fills) == 1
    fill = r3.fills[0]
    assert fill["data_key"] == "rollout_routed_experts"
    assert fill["replay_list"] is routing_replay_manager.replays
    assert fill["register_replay_list_func"] is _register_moe
    assert fill["if_sp_region"] is True and fill["indices_are_token_positions"] is False
    assert fill["args"] is rig.args and fill["models"] is rig.model and fill["num_microbatches"] == [1]
    assert fill["stage"] == "fallthrough", "filled before the stage is raised"
    assert r3.stages == ["replay_backward"], "the pass runs with the backward recompute replaying"
    assert routing_replay_manager.stage == "fallthrough", "back to the live router after the unit"
    assert r3.replay.top_indices_list == [] and r3.replay.backward_index == 0, "queues cleared"
    assert "rollout_routed_experts" not in rollout_data


def test_r3_clears_the_replay_when_the_pass_fails(rig, monkeypatch):
    r3 = _replay_rig(rig, monkeypatch, flag=True, enabled=True)

    def boom(*args, **kwargs):
        raise RuntimeError("cuda oom")

    rig.executor.run_forward_backward_pass = boom
    with pytest.raises(RuntimeError):
        rig.executor.forward_backward(rig.args, 1, rig.model, FakeOptimizer(), {"rollout_routed_experts": ["rows"]})

    assert len(r3.fills) == 1
    assert routing_replay_manager.stage == "fallthrough"
    assert r3.replay.top_indices_list == []
    assert rig.executor._state["dirty"] is True


@pytest.mark.parametrize("flag, enabled", [(False, True), (True, False), (False, False)])
def test_r3_is_not_filled_unless_the_flag_and_the_manager_are_on(rig, monkeypatch, flag, enabled):
    r3 = _replay_rig(rig, monkeypatch, flag=flag, enabled=enabled)
    rollout_data = {"rollout_routed_experts": ["rows"]}

    rig.executor.forward_backward(rig.args, 1, rig.model, FakeOptimizer(), rollout_data)

    assert r3.fills == []
    assert r3.stages == ["fallthrough"]
    assert r3.replay.top_indices_list != [], "an inactive manager is left alone"
    assert "rollout_routed_experts" in rollout_data


def test_the_r3_fill_and_the_schedule_get_their_own_wall_clock_stages(rig, monkeypatch):
    """actor_train_time covers only the forward/backward; the R3 fill (~29 GB of int32
    routing per step) and the micro-batch schedule are charged separately so the gap
    between two optimizer steps can be accounted for (docs/fork/mfu.md)."""
    from miles.backends.megatron_utils import tinker_perf
    from miles.utils.timer import Timer

    tinker_perf.reset_window()
    _replay_rig(rig, monkeypatch, flag=True, enabled=True)
    ticks = iter([0.0, 1.0, 10.0, 14.0])
    monkeypatch.setattr(tinker_perf, "time", lambda: next(ticks))
    monkeypatch.setattr(
        rig.executor,
        "get_data_iterator",
        lambda args, model, rollout_data: (
            [SimpleNamespace(reset=lambda: None, micro_batch_indices=None, micro_batch_size=2)],
            [1],
        ),
    )
    monkeypatch.setattr(Timer(), "dp_seq_lens", [[5, 7], [9]], raising=False)

    rig.executor.forward_backward(
        rig.args, 1, rig.model, FakeOptimizer(), {"rollout_routed_experts": ["rows"], "total_lengths": [5, 7]}
    )

    shapes, stages = tinker_perf.pop_diagnostics()
    assert stages[tinker_perf.STAGE_DATA_ITER] == pytest.approx(1.0)
    assert stages[tinker_perf.STAGE_REPLAY_FILL] == pytest.approx(4.0)
    assert shapes == [{"dp_seq_lens": [[5, 7], [9]], "mb_tokens": [12]}]


def test_the_optimizer_step_is_charged_inside_the_train_timer(rig, monkeypatch):
    from miles.backends.megatron_utils import tinker_perf

    tinker_perf.reset_window()
    ticks = iter([0.0, 3.0])
    monkeypatch.setattr(tinker_perf, "time", lambda: next(ticks))

    rig.executor.optim_step(rig.args, rig.model, FakeOptimizer(), ADAM)

    _, stages = tinker_perf.pop_diagnostics()
    assert stages == {tinker_perf.STAGE_OPTIM: pytest.approx(3.0)}
