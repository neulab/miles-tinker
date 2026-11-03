"""runtime.py is the miles translator: rows -> RolloutBatch keys, neutral
sampling payloads -> sglang /generate requests, engine responses -> sequences."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.tinker.core.types import UserInputError
from miles.tinker.runtime import (
    PAD_ROW,
    MilesBackend,
    _build_train_data,
    _prompt_logprobs,
    _to_sequence,
    _topk_prompt_logprobs,
)


def _row(tokens: list[int], **extra) -> dict:
    return {"tokens": tokens, "target_len": len(tokens) - 1, **extra}


def _args(**overrides) -> Namespace:
    defaults = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tinker_train_mode="lora",
        tinker_checkpoint_root=None,
    )
    return Namespace(**{**defaults, **overrides})


class FakeTrainer:
    """Records TrainerController calls; answers per-worker lists like the controller does."""

    def __init__(self, num_workers: int = 2) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.num_workers = num_workers
        self.optim_metrics: dict[int, dict] = {}
        self.version = 7

    def __getattr__(self, name):
        async def call(**kwargs):
            self.calls.append((name, kwargs))
            if name == "optim_step":
                return [dict(self.optim_metrics)] * self.num_workers
            if name == "push_slot":
                return self.version
            if name == "update_weights":
                self.version += 1
                return self.version
            return None

        return call


class FakeInferenceController:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start_update_weights(self):
        self.calls.append("start")
        return SimpleNamespace(snapshot_cell_id_to_hashes={"cell": "hash"})

    async def end_update_weights(self, snapshot_cell_id_to_hashes):
        self.calls.append(f"end:{snapshot_cell_id_to_hashes}")


def _backend(trainer=None, inference_controller=None, **arg_overrides) -> MilesBackend:
    return MilesBackend(
        _args(**arg_overrides), trainer=trainer, router_url="http://router", inference_controller=inference_controller
    )


class TestBuildTrainData:
    def test_rows_become_rollout_batch_keys(self):
        train_data = _build_train_data([(3, _row([1, 2, 3])), (5, _row([4, 5]))])
        assert train_data["tokens"] == [[1, 2, 3], [4, 5]]
        assert train_data["response_lengths"] == [2, 1]
        assert train_data["total_lengths"] == [3, 2]
        assert train_data["loss_masks"] == [[1, 1], [1]]
        assert train_data["adapter_slots"] == [3, 5]
        assert train_data["sample_indices"] == [0, 1]
        assert train_data["dynamic_global_batch_size"] == 2

    def test_optional_row_keys_map_to_batch_names(self):
        train_data = _build_train_data([(0, _row([1, 2], weights=[1.0], advantages=[2.0], sampling_logprobs=[-0.5]))])
        assert train_data["loss_weights"] == [[1.0]]
        assert train_data["advantages"] == [[2.0]]
        assert train_data["rollout_log_probs"] == [[-0.5]]

    def test_units_are_padded_to_a_multiple_of_dp_size_with_no_op_rows(self):
        rows = [(3, _row([1, 2, 3], weights=[1.0, 1.0])), (5, _row([4, 5], weights=[1.0]))]
        train_data = _build_train_data(rows, dp_size=4)
        assert train_data["dynamic_global_batch_size"] == 4, "the trainer schedules len(rows) // dp_size per rank"
        assert train_data["tokens"] == [[1, 2, 3], [4, 5], PAD_ROW["tokens"], PAD_ROW["tokens"]]
        assert train_data["response_lengths"] == [2, 1, 1, 1]
        assert train_data["loss_weights"] == [[1.0, 1.0], [1.0], [0.0], [0.0]]
        assert train_data["loss_masks"] == [[1, 1], [1], [0], [0]], "pad rows are masked out under every loss"
        assert train_data["adapter_slots"] == [3, 5, 3, 3], "pad rows ride on the first real row's slot"
        assert train_data["sample_indices"] == [0, 1, 2, 3]

    def test_a_full_multiple_is_not_padded(self):
        train_data = _build_train_data([(0, _row([1, 2])), (0, _row([3, 4]))], dp_size=2)
        assert train_data["dynamic_global_batch_size"] == 2

    def test_full_fine_tuning_omits_adapter_slots(self):
        train_data = _build_train_data([(0, _row([1, 2]))], dp_size=2, with_adapter_slots=False)
        assert "adapter_slots" not in train_data
        assert train_data["dynamic_global_batch_size"] == 2


def test_dp_size_derives_from_the_actor_topology():
    assert _backend().dp_size == 2
    assert _backend(actor_num_gpus_per_node=8, tensor_model_parallel_size=1).dp_size == 8


async def test_forward_backward_merges_worker_replicas_and_strips_padding():
    backend = _backend(actor_num_gpus_per_node=8, tensor_model_parallel_size=2)  # dp 4
    per_datum = [
        {"sample_index": 1, "loss": 2.0, "logprobs": torch.tensor([-0.2])},
        {"sample_index": 0, "loss": 1.0, "logprobs": torch.tensor([-0.1])},
        {"sample_index": 2, "loss": 0.0, "logprobs": torch.tensor([0.0])},  # pad row
        {"sample_index": 3, "loss": 0.0, "logprobs": torch.tensor([0.0])},  # pad row
    ]
    seen = []

    async def fake_run_unit(method, unit_id, train_data):
        seen.append(train_data)
        return [{"per_datum": per_datum}, {"per_datum": per_datum}]  # two ranks report the same datums

    backend._run_unit = fake_run_unit
    outputs = await backend.forward_backward(1, [(0, _row([1, 2])), (0, _row([3, 4]))], "cross_entropy", {})
    assert outputs == [
        {"loss": 1.0, "logprobs": [pytest.approx(-0.1)]},
        {"loss": 2.0, "logprobs": [pytest.approx(-0.2)]},
    ]
    assert seen[0]["dynamic_global_batch_size"] == 4 and seen[0]["loss_fn"] == "cross_entropy"
    assert "adapter_slots" in seen[0]


async def test_full_mode_units_carry_no_adapter_slots():
    backend = _backend(tinker_train_mode="full", actor_num_gpus_per_node=2, tensor_model_parallel_size=1)
    seen = []

    async def fake_run_unit(method, unit_id, train_data):
        seen.append(train_data)
        return [{"per_datum": [{"sample_index": 0, "loss": 1.0, "logprobs": torch.tensor([-0.1])}]}]

    backend._run_unit = fake_run_unit
    await backend.forward_backward(1, [(0, _row([1, 2]))], "cross_entropy", {})
    assert "adapter_slots" not in seen[0] and seen[0]["dynamic_global_batch_size"] == 2


def _shape_of(lengths: list[int]) -> dict:
    """The datum_seq_len_* block a window of these unpadded rows reports."""
    ordered = sorted(lengths)
    return {
        "datum_seq_len_count": float(len(ordered)),
        "datum_seq_len_mean": pytest.approx(sum(ordered) / len(ordered)),
        "datum_seq_len_p50": float(ordered[(len(ordered) - 1) // 2]),
        "datum_seq_len_p90": float(ordered[-1]),
        "datum_seq_len_max": float(ordered[-1]),
    }


class TestOptimStep:
    async def _train_one_unit(self, backend, slot_rows):
        async def fake_run_unit(method, unit_id, train_data):
            return [
                {"per_datum": [{"sample_index": i, "loss": 0.0, "logprobs": torch.tensor([0.0])} for i in range(8)]}
            ]

        backend._run_unit = fake_run_unit
        await backend.forward_backward(1, slot_rows, "cross_entropy", {})

    async def test_returns_first_worker_metrics_with_exact_window_counts(self, tmp_path):
        trainer = FakeTrainer()
        trainer.optim_metrics = {
            0: {"grad_norm": 0.5, "skipped": 0, "actor_train_mfu": 0.2, "num_tokens": 99, "num_datums": 9}
        }
        backend = _backend(trainer, tinker_checkpoint_root=str(tmp_path))
        await self._train_one_unit(backend, [(0, _row([1, 2, 3])), (0, _row([4, 5]))])
        await self._train_one_unit(backend, [(0, _row([6, 7]))])

        metrics = await backend.optim_step({0: {"learning_rate": 1e-4}})

        assert trainer.calls == [("optim_step", {"adam_params_by_slot": {0: {"learning_rate": 1e-4}}})]
        assert metrics == {
            0: {
                "grad_norm": 0.5,
                "skipped": 0,
                "actor_train_mfu": 0.2,
                "num_tokens": 7,
                "num_datums": 3,
                "num_units": 2,
                # rows 3, 2, 2: the shape metrics see the unpadded lengths
                "datum_seq_len_count": 3.0,
                "datum_seq_len_mean": pytest.approx(7 / 3),
                "datum_seq_len_p50": 2.0,
                "datum_seq_len_p90": 3.0,
                "datum_seq_len_max": 3.0,
            }
        }, "padded actor counts are replaced by the driver's exact per-slot counts"
        record = json.loads((tmp_path / "perf.jsonl").read_text().splitlines()[0])
        assert record["slot"] == 0 and record["actor_train_mfu"] == 0.2 and record["num_units"] == 2

    async def test_window_counts_are_per_slot_and_reset(self):
        trainer = FakeTrainer()
        trainer.optim_metrics = {3: {"grad_norm": 1.0}, 5: {"grad_norm": 2.0}}
        backend = _backend(trainer)
        await self._train_one_unit(backend, [(3, _row([1, 2, 3])), (5, _row([4, 5]))])

        metrics = await backend.optim_step({3: {}, 5: {}})
        assert metrics[3] == {
            "grad_norm": 1.0,
            "num_tokens": 3,
            "num_datums": 1,
            "num_units": 1,
            **_shape_of([3]),
        }
        assert metrics[5] == {
            "grad_norm": 2.0,
            "num_tokens": 2,
            "num_datums": 1,
            "num_units": 1,
            **_shape_of([2]),
        }

        trainer.optim_metrics = {3: {"grad_norm": 0.0}}
        assert (await backend.optim_step({3: {}}))[3] == {
            "grad_norm": 0.0,
            "num_tokens": 0,
            "num_datums": 0,
            "num_units": 0,
        }

    async def test_padding_is_measured_against_the_trainers_padded_window(self):
        """Pad rows are a third of the row SLOTS and ~1e-5 of the tokens; the metric
        exists so padding is never again mistaken for the throughput problem."""
        trainer = FakeTrainer()
        # the trainer forwarded 4 rows / 9 tokens; the client asked for 3 rows / 7 tokens
        trainer.optim_metrics = {0: {"grad_norm": 0.0, "padded_num_datums": 4, "padded_num_tokens": 9}}
        backend = _backend(trainer)
        await self._train_one_unit(backend, [(0, _row([1, 2, 3])), (0, _row([4, 5]))])
        await self._train_one_unit(backend, [(0, _row([6, 7]))])

        metrics = (await backend.optim_step({0: {}}))[0]

        assert metrics["real_rows"] == 3
        assert metrics["pad_rows"] == 1
        assert metrics["pad_tokens"] == 2
        assert metrics["pad_row_frac"] == pytest.approx(0.25), "a quarter of the row slots"
        assert metrics["pad_token_frac"] == pytest.approx(2 / 9), "but only 2 of 9 tokens"

    async def test_padding_is_absent_when_it_cannot_be_attributed_to_one_slot(self):
        """The trainer's window is process-wide: with two slots stepped together its
        padded counts cover both, so the per-slot subtraction would be nonsense."""
        trainer = FakeTrainer()
        padded = {"padded_num_datums": 4, "padded_num_tokens": 9}
        trainer.optim_metrics = {3: {**padded}, 5: {**padded}}
        backend = _backend(trainer)
        await self._train_one_unit(backend, [(3, _row([1, 2, 3])), (5, _row([4, 5]))])

        metrics = await backend.optim_step({3: {}, 5: {}})
        assert not any(key.startswith("pad_") for key in metrics[3])

    async def test_padding_is_absent_when_the_trainer_reports_no_padded_counts(self):
        trainer = FakeTrainer()
        trainer.optim_metrics = {0: {"grad_norm": 0.0}}
        backend = _backend(trainer)
        await self._train_one_unit(backend, [(0, _row([1, 2, 3]))])

        metrics = (await backend.optim_step({0: {}}))[0]
        assert not any(key.startswith("pad_") or key == "real_rows" for key in metrics)

    async def test_padding_is_absent_when_the_two_windows_disagree(self):
        """A window the trainer discarded (fewer padded rows than the gateway counted)
        reports nothing rather than a negative pad count."""
        trainer = FakeTrainer()
        trainer.optim_metrics = {0: {"padded_num_datums": 1, "padded_num_tokens": 2}}
        backend = _backend(trainer)
        await self._train_one_unit(backend, [(0, _row([1, 2, 3])), (0, _row([4, 5]))])

        metrics = (await backend.optim_step({0: {}}))[0]
        assert not any(key.startswith("pad_") for key in metrics)

    async def test_discard_grads_drops_the_window_and_reaches_the_trainer(self):
        trainer = FakeTrainer()
        trainer.optim_metrics = {0: {"grad_norm": 0.0}}
        backend = _backend(trainer)
        await self._train_one_unit(backend, [(0, _row([1, 2, 3]))])
        await backend.discard_grads(0)
        assert trainer.calls == [("discard_grads", {"slot": 0})]
        metrics = (await backend.optim_step({0: {}}))[0]
        assert metrics["num_datums"] == 0
        assert not any(key.startswith("datum_seq_len") for key in metrics), "the shape window went with it"


class TestControlCommands:
    async def test_warmup_lora_runs_an_empty_update_weights_round(self):
        controller = FakeInferenceController()
        trainer = FakeTrainer()
        await _backend(trainer, controller).warmup()
        assert controller.calls == ["start", "end:{'cell': 'hash'}"]
        assert trainer.calls == []

    async def test_warmup_full_syncs_the_base_weights(self):
        controller = FakeInferenceController()
        trainer = FakeTrainer()
        await _backend(trainer, controller, tinker_train_mode="full").warmup()
        assert trainer.calls == [("update_weights", {})]
        assert controller.calls == []

    async def test_push_slot_returns_the_trainer_version(self):
        trainer = FakeTrainer()
        assert await _backend(trainer, tinker_train_mode="full").push_slot(0, "m@1", 0, 0.0) == 7
        assert trainer.calls == [("push_slot", {"slot": 0, "lora_name": "m@1", "rank": 0, "alpha": 0.0})]

    async def test_unload_adapter_and_load_slot_pass_through(self):
        trainer = FakeTrainer()
        backend = _backend(trainer)
        await backend.unload_adapter("m@1")
        await backend.load_slot(0, 8, 16.0, ckpt_path=None, load_optimizer=False)
        assert trainer.calls == [
            ("unload_adapter", {"lora_name": "m@1"}),
            ("load_slot", {"slot": 0, "rank": 8, "alpha": 16.0, "ckpt_path": None, "load_optimizer": False}),
        ]


class TestHealthProblem:
    """healthz answers 503 with this reason once the trainer cannot run a command any more."""

    def test_a_trainer_with_an_alive_cell_is_healthy(self):
        assert _backend(SimpleNamespace(has_alive_cell=lambda: True)).health_problem() is None

    def test_a_trainer_whose_cells_all_failed_is_reported_dead(self):
        problem = _backend(SimpleNamespace(has_alive_cell=lambda: False)).health_problem()
        assert problem is not None and "trainer is dead" in problem


class TestGenerateRequest:
    def _request(self, params: dict, **payload_extra) -> dict:
        payload = {
            "prompt_tokens": [1, 2],
            "sampling_params": params,
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
            **payload_extra,
        }
        return _backend()._generate_request(payload, lora_name="m@1")

    def test_max_tokens_is_required(self):
        with pytest.raises(UserInputError, match="max_tokens"):
            self._request({})

    def test_params_map_to_sglang_names(self):
        request = self._request({"max_tokens": 8, "temperature": 0.0, "seed": 7})
        assert request["sampling_params"]["max_new_tokens"] == 8
        assert request["sampling_params"]["sampling_seed"] == 7
        assert request["lora_path"] == "m@1"
        assert request["return_logprob"] is True

    def test_stop_splits_token_ids_from_strings(self):
        by_ids = self._request({"max_tokens": 1, "stop": [7, 8]})
        by_text = self._request({"max_tokens": 1, "stop": ["\n"]})
        assert by_ids["sampling_params"]["stop_token_ids"] == [7, 8]
        assert by_text["sampling_params"]["stop"] == ["\n"]

    def test_topk_requests_prompt_logprobs(self):
        request = self._request({"max_tokens": 1}, topk_prompt_logprobs=3)
        assert request["logprob_start_len"] == 0
        assert request["top_logprobs_num"] == 3


class TestEngineResponseParsing:
    def test_to_sequence_reads_tokens_and_stop_reason(self):
        response = {
            "meta_info": {
                "output_token_logprobs": [(-0.1, 11), (-0.2, 12)],
                "finish_reason": {"type": "length"},
            }
        }
        sequence = _to_sequence(response)
        assert (sequence["tokens"], sequence["logprobs"], sequence["stop_reason"]) == (
            [11, 12],
            [-0.1, -0.2],
            "length",
        )

    def test_prompt_logprobs_pad_the_unscored_first_token(self):
        response = {"meta_info": {"input_token_logprobs": [(None, 1), (-0.5, 2)]}}
        first, second = _prompt_logprobs(response)
        assert first != first, "the unscored position must be NaN"
        assert second == -0.5

    def test_topk_pads_ragged_positions(self):
        response = {"meta_info": {"input_top_logprobs": [None, [(-0.1, 5)]]}}
        topk = _topk_prompt_logprobs(response, k=2)
        assert topk["token_ids"] == [[0, 0], [5, 0]]
        assert topk["logprobs"][1][0] == -0.1
        assert topk["logprobs"][0][0] != topk["logprobs"][0][0]
