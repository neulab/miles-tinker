"""Runtime translation preserves datum order, sampling parameters, and token logprobs."""

import pytest
import torch

from miles.tinker.core.types import UserInputError
from miles.tinker.runtime import (
    MilesBackend,
    _build_train_data,
    _pad_to_dp_multiple,
    _prompt_logprobs,
    _slot_failure,
    _to_sequence,
    _topk_prompt_logprobs,
)
from miles.tinker.server.proto_codec import encode_sample_response
from tinker.proto.response_conv import deserialize_sample_response


def _datum(tokens: list[int], **extra) -> dict:
    return {"tokens": tokens, "target_len": len(tokens) - 1, "target_tokens": tokens[1:], **extra}


class TestBuildTrainData:
    def test_datums_become_rollout_batch_keys(self):
        train_data = _build_train_data([(3, _datum([1, 2, 3])), (5, _datum([4, 5]))])
        assert train_data["tokens"] == [[1, 2, 3], [4, 5]]
        assert train_data["target_tokens"] == [[2, 3], [5]]
        assert train_data["response_lengths"] == [2, 1]
        assert train_data["total_lengths"] == [3, 2]
        assert train_data["loss_masks"] == [[1, 1], [1]]
        assert train_data["adapter_slots"] == [3, 5]
        assert train_data["sample_indices"] == [0, 1]
        assert train_data["dynamic_global_batch_size"] == 2

    def test_optional_datum_keys_map_to_batch_names(self):
        train_data = _build_train_data(
            [(0, _datum([1, 2], weights=[1.0], advantages=[2.0], sampling_logprobs=[-0.5]))]
        )
        assert train_data["loss_weights"] == [[1.0]]
        assert train_data["advantages"] == [[2.0]]
        assert train_data["rollout_log_probs"] == [[-0.5]]


async def test_forward_backward_merges_worker_replicas():
    backend = MilesBackend(trainer=None, router_url="http://router")
    per_datum = [
        {"sample_index": 1, "loss": 2.0, "logprobs": torch.tensor([-0.2])},
        {"sample_index": 0, "loss": 1.0, "logprobs": torch.tensor([-0.1])},
    ]

    async def fake_call_trainer(method, batch_id, train_data):
        return [{"per_datum": per_datum}, {"per_datum": per_datum}]  # two ranks report the same datums

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2])), (0, _datum([3, 4]))], "cross_entropy", {})
    assert outputs == [
        {"loss": 1.0, "logprobs": [pytest.approx(-0.1)]},
        {"loss": 2.0, "logprobs": [pytest.approx(-0.2)]},
    ]


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
        return MilesBackend(None, "http://router")._generate_request(payload, lora_name="m@1")

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

    def test_an_empty_stop_list_disables_eos(self):
        request = self._request({"max_tokens": 1, "stop": []})
        assert request["sampling_params"]["ignore_eos"] is True
        assert "stop" not in request["sampling_params"] and "stop_token_ids" not in request["sampling_params"]
        default = self._request({"max_tokens": 1})
        assert "ignore_eos" not in default["sampling_params"], "stop=None keeps the default EOS behavior"

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
        result = {"sequences": [], "topk_prompt_logprobs": topk}
        decoded = deserialize_sample_response(encode_sample_response(result)).topk_prompt_logprobs
        assert decoded[0] is None
        assert len(decoded[1]) == 1
        token_id, logprob = decoded[1][0]
        assert token_id == 5
        assert logprob == pytest.approx(-0.1)


async def test_forward_only_runs_the_requested_loss():
    backend = MilesBackend(trainer=None, router_url="http://router")
    captured = {}

    async def fake_call_trainer(method, batch_id, train_data):
        captured["method"] = method
        captured["loss_fn"] = train_data["loss_fn"]
        return [{"per_datum": [{"sample_index": 0, "loss": 3.0, "logprobs": torch.tensor([-0.3])}]}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_only(1, [(0, _datum([1, 2]))], "importance_sampling", {})
    assert captured == {"method": "forward_only", "loss_fn": "importance_sampling"}
    assert outputs == [{"loss": 3.0, "logprobs": [pytest.approx(-0.3)]}]


def test_a_pinned_seed_still_gets_one_model_queue_per_sample():
    from miles.tinker.runtime import _with_sample_seed

    request = {"sampling_params": {"sampling_seed": 7, "temperature": 0.0}}
    assert [_with_sample_seed(request, i)["sampling_params"]["sampling_seed"] for i in range(3)] == [7, 8, 9]
    assert request["sampling_params"]["sampling_seed"] == 7
    assert "sampling_seed" not in _with_sample_seed({"sampling_params": {}}, 2)["sampling_params"]


class TestPadToDpMultiple:
    def test_padding_replicates_the_last_datum_with_a_padding_marker(self):
        slot_datums = [(0, _datum([1, 2, 3], weights=[1.0, 1.0], advantages=[2.0, 2.0]))]
        padded = _pad_to_dp_multiple(slot_datums, 4)
        assert len(padded) == 4
        for slot, filler in padded[1:]:
            assert slot == 0 and filler["tokens"] == [1, 2, 3] and filler["padding"]
        assert "padding" not in slot_datums[0][1], "the original datum must not be mutated"


async def test_forward_backward_pads_the_batch_and_drops_padding_outputs():
    backend = MilesBackend(trainer=None, router_url="http://router", dp_size=2)
    seen = {}

    async def fake_call_trainer(method, batch_id, train_data):
        seen.update(train_data)
        per_datum = [
            {"sample_index": index, "loss": float(index), "logprobs": torch.tensor([-0.1])}
            for index in range(len(train_data["tokens"]))
        ]
        return [{"per_datum": per_datum}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2], weights=[1.0]))], "cross_entropy", {})
    assert seen["dynamic_global_batch_size"] == 2, "a singleton batch must be padded to the DP size"
    assert seen["loss_masks"] == [[1], [0]], "the zero mask removes padding from every loss term"
    assert len(outputs) == 1, "padding outputs are dropped"


def test_an_actor_error_verdict_survives_runtime_translation():
    failure = {"error": "bad shard"}
    assert _slot_failure([None, failure]) is failure


def test_an_aborted_sample_fails_instead_of_passing_as_a_stop():
    """A truncated sequence fed to RL as a completed sample corrupts training data silently."""
    response = {"meta_info": {"output_token_logprobs": [(-0.1, 11)], "finish_reason": {"type": "abort"}}}
    assert "abort" in _to_sequence(response)["error"]


class TestContextClamp:
    def _max_new_tokens(self, max_tokens: int, context_len: int | None, prompt_len: int = 2) -> int:
        payload = {
            "prompt_tokens": list(range(prompt_len)),
            "sampling_params": {"max_tokens": max_tokens},
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        }
        backend = MilesBackend(None, "http://router", context_len=context_len)
        return backend._generate_request(payload, lora_name=None)["sampling_params"]["max_new_tokens"]

    def test_generation_is_clamped_to_the_room_left_in_the_context(self):
        assert self._max_new_tokens(100, context_len=10, prompt_len=4) == 6

    def test_a_request_that_fits_is_unchanged(self):
        assert self._max_new_tokens(5, context_len=10, prompt_len=4) == 5
        assert self._max_new_tokens(5, context_len=None) == 5

    def test_a_prompt_that_fills_the_context_is_a_user_error(self):
        with pytest.raises(UserInputError, match="no room"):
            self._max_new_tokens(5, context_len=4, prompt_len=4)


async def test_in_flight_generate_posts_are_capped(monkeypatch):
    import asyncio

    import miles.tinker.runtime as runtime

    in_flight = peak = 0

    async def fake_post(url, body):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {"meta_info": {"output_token_logprobs": [[-0.1, 5, None]], "finish_reason": {"type": "stop"}}}

    monkeypatch.setattr(runtime, "post", fake_post)
    backend = MilesBackend(None, "http://router", max_concurrent_samples=3)
    payload = {
        "prompt_tokens": [1, 2],
        "sampling_params": {"max_tokens": 4},
        "num_samples": 8,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }
    results = await asyncio.gather(*[backend.sample(payload, lora_name=None) for _ in range(2)])
    assert all(len(result["sequences"]) == 8 for result in results)
    assert peak == 3, "posts from every sampling request share one cap"


async def test_optim_step_logs_each_slot_window_to_perf_jsonl(tmp_path):
    import json
    from types import SimpleNamespace

    perf_log = tmp_path / "ckpt" / "perf.jsonl"
    window_perf = {"actor_train_mfu": 0.2, "num_tokens": 999, "num_datums": 9, "num_units": 9}

    async def optim_step(adam_params_by_slot):
        return [{0: {"grad_norm": 1.0, **window_perf}, 1: {"error": "boom"}}]

    backend = MilesBackend(SimpleNamespace(optim_step=optim_step), "http://router", perf_log_path=str(perf_log))

    async def fake_call_trainer(method, batch_id, train_data):
        return [{"per_datum": [{"sample_index": i, "loss": 0.0, "logprobs": torch.tensor([0.0])} for i in range(3)]}]

    backend._call_trainer = fake_call_trainer
    await backend.forward_backward(1, [(0, _datum([1, 2, 3])), (0, _datum([4, 5])), (1, _datum([6, 7]))], "ce", {})
    await backend.forward_only(2, [(0, _datum([1, 2]))], "ce", {})  # scoring passes are not training work

    metrics = await backend.optim_step({0: {}, 1: {}})
    assert metrics[0] == {
        "grad_norm": 1.0,
        "actor_train_mfu": 0.2,
        "num_tokens": 5,
        "num_datums": 2,
        "num_units": 1,
    }, "the gateway's exact per-slot counts replace the trainer's padded ones"
    assert metrics[1] == {"error": "boom"}
    records = [json.loads(line) for line in perf_log.read_text().splitlines()]
    assert [(record["slot"], record["num_tokens"]) for record in records] == [(0, 5)]

    metrics = await backend.optim_step({0: {}})
    assert (metrics[0]["num_tokens"], metrics[0]["num_units"]) == (0, 0), "each optim step starts a new window"


def test_full_fine_tuning_batches_carry_no_adapter_slots():
    train_data = _build_train_data([(0, _datum([1, 2, 3]))], with_adapter_slots=False)
    assert "adapter_slots" not in train_data


async def test_a_full_mode_sampler_save_pushes_the_base_weights(tmp_path):
    import json
    from types import SimpleNamespace

    calls = []

    class Controller:
        async def start_update_weights(self):
            calls.append("start")
            return SimpleNamespace(snapshot_cell_id_to_hashes={"cell": "hash"})

        async def end_update_weights(self, snapshot_cell_id_to_hashes):
            calls.append(("end", snapshot_cell_id_to_hashes))

        async def abort_update_weights(self):
            calls.append("abort")

    async def update_weights(info):
        calls.append("update_weights")
        return 7

    backend = MilesBackend(
        SimpleNamespace(update_weights=update_weights),
        "http://router",
        train_mode="full",
        inference_controller=Controller(),
    )
    path = tmp_path / "sampler_weights" / "1"
    assert await backend.export_slot(0, 0, 0.0, str(path), metadata={"train_mode": "full"}) is None
    assert calls == ["start", "update_weights", ("end", {"cell": "hash"})]
    assert json.loads((path / "META.json").read_text()) == {"train_mode": "full"}


async def test_a_failed_full_mode_push_aborts_the_weight_update(tmp_path):
    from types import SimpleNamespace

    calls = []

    class Controller:
        async def start_update_weights(self):
            return SimpleNamespace(snapshot_cell_id_to_hashes={})

        async def end_update_weights(self, snapshot_cell_id_to_hashes):
            calls.append("end")

        async def abort_update_weights(self):
            calls.append("abort")

    async def update_weights(info):
        raise RuntimeError("nccl")

    backend = MilesBackend(
        SimpleNamespace(update_weights=update_weights),
        "http://router",
        train_mode="full",
        inference_controller=Controller(),
    )
    with pytest.raises(RuntimeError, match="nccl"):
        await backend.export_slot(0, 0, 0.0, str(tmp_path / "s"), metadata={})
    assert calls == ["abort"]
    assert not (tmp_path / "s").exists(), "no sampler path for weights the engines never received"
