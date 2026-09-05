"""The JSON half of the wire: decode enforces next-token supervision and the
AdamParams schema; render produces the SDK response shapes."""

import math

import pytest

from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import (
    ADAM_PARAM_DEFAULTS,
    build_row,
    decode_command,
    decode_forward_request,
    decode_sample_request,
    materialize_adam_params,
    render_result,
    tensor_data_to_list,
)


def _datum(tokens: list[int], **extra_inputs) -> dict:
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
        "loss_fn_inputs": {"target_tokens": tokens[1:], **extra_inputs},
    }


class TestDecodeForwardBackward:
    def _decode(self, datum, forward_only=False):
        payload = {
            "model_id": "model",
            "seq_id": 1,
            "forward_only": forward_only,
            "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"},
        }
        return decode_command("forward_backward", payload)

    def test_a_shifted_datum_becomes_one_row(self):
        kind, decoded = self._decode(_datum([1, 2, 3, 4], weights=[1.0, 1.0, 1.0]))
        assert kind == "forward_backward"
        assert decoded["rows"] == [{"tokens": [1, 2, 3, 4], "target_len": 3, "weights": [1.0, 1.0, 1.0]}]

    def test_forward_only_reroutes_the_kind(self):
        kind, _ = self._decode(_datum([1, 2, 3]), forward_only=True)
        assert kind == "forward_only"

    def test_a_chunk_without_type_is_encoded_text(self):
        datum = _datum([1, 2, 3])
        datum["model_input"]["chunks"][0].pop("type")  # 0.16.1 exclude_unset
        _, decoded = self._decode(datum)
        assert decoded["rows"][0]["tokens"] == [1, 2, 3]

    def test_an_image_chunk_is_rejected(self):
        datum = _datum([1, 2, 3])
        datum["model_input"]["chunks"][0]["type"] = "image"
        with pytest.raises(UserInputError, match="chunk type"):
            self._decode(datum)

    def test_a_length_mismatch_is_rejected(self):
        datum = _datum([1, 2, 3])
        datum["loss_fn_inputs"]["target_tokens"] = [2]
        with pytest.raises(UserInputError, match="length"):
            self._decode(datum)

    def test_unshifted_targets_are_rejected(self):
        datum = _datum([1, 2, 3])
        datum["loss_fn_inputs"]["target_tokens"] = [9, 9]
        with pytest.raises(UserInputError, match="shifted"):
            self._decode(datum)


def test_decode_forward_request_maps_the_016_key():
    kind, decoded = decode_forward_request(
        {"model_id": "m", "seq_id": 2, "forward_input": {"data": [_datum([1, 2])], "loss_fn": "cross_entropy"}}
    )
    assert (kind, decoded["seq_id"], len(decoded["rows"])) == ("forward_only", 2, 1)


def test_build_row_maps_the_wire_input_names():
    row = build_row([1, 2], {"target_tokens": [2, 3], "logprobs": [0.5, 0.5], "advantages": [1.0, -1.0]}, 0)
    assert row["sampling_logprobs"] == [0.5, 0.5]
    assert row["advantages"] == [1.0, -1.0]


def test_save_weights_for_sampler_keeps_name_and_session_seq():
    _, named = decode_command("save_weights_for_sampler", {"model_id": "m", "seq_id": 1, "path": "000010"})
    _, unnamed = decode_command(
        "save_weights_for_sampler", {"model_id": "m", "seq_id": 2, "sampling_session_seq_id": 3}
    )
    assert (named["name"], named["sampling_session_seq_id"]) == ("000010", None)
    assert (unnamed["name"], unnamed["sampling_session_seq_id"]) == (None, 3)


class TestAdamParams:
    def test_defaults_fill_missing_keys(self):
        materialized = materialize_adam_params({"learning_rate": 3e-4})
        assert materialized["learning_rate"] == 3e-4
        assert materialized["eps"] == ADAM_PARAM_DEFAULTS["eps"]
        assert set(materialized) == set(ADAM_PARAM_DEFAULTS)

    def test_unknown_keys_are_ignored_with_a_warning(self, caplog):
        materialized = materialize_adam_params({"momentum": 0.9, "learning_rate": 1e-3})
        assert "momentum" not in materialized and materialized["learning_rate"] == 1e-3
        assert "momentum" in caplog.text

    def test_explicit_nulls_take_the_default(self):
        assert materialize_adam_params({"weight_decay": None})["weight_decay"] == 0.0


class TestTensorData:
    def test_a_plain_list_passes_through(self):
        assert tensor_data_to_list([1, 2]) == [1, 2]

    def test_dense_tensor_data(self):
        assert tensor_data_to_list({"data": [1.0, 2.0], "shape": [2]}) == [1.0, 2.0]

    def test_csr_expands_to_dense(self):
        sparse = {"shape": [4], "sparse_crow_indices": [0, 2], "sparse_col_indices": [1, 3], "data": [5.0, 7.0]}
        assert tensor_data_to_list(sparse) == [0, 5.0, 0, 7.0]


def test_load_state_carries_the_optimizer_flag():
    _, decoded = decode_command(
        "load_state", {"model_id": "m", "seq_id": 1, "path": "tinker://m/weights/x", "optimizer": False}
    )
    assert decoded["optimizer"] is False


def test_decode_sample_request_defaults():
    decoded = decode_sample_request({"prompt": {"chunks": [{"tokens": [1, 2]}]}})
    assert decoded["num_samples"] == 1
    assert decoded["prompt_tokens"] == [1, 2]
    assert decoded["topk_prompt_logprobs"] == 0
    assert decoded["seq_id"] is None


class TestRenderResult:
    def test_forward_backward_renders_per_datum_records(self):
        rendered = render_result({"kind": "forward_backward", "outputs": [{"loss": 2.0, "logprobs": [0.1, 0.2]}]})
        assert rendered["loss_fn_output_type"] == "ArrayRecord"
        assert rendered["metrics"] == {"loss:sum": 2.0, "num_tokens:sum": 2.0, "num_datums:sum": 1.0}
        assert rendered["loss_fn_outputs"][0]["logprobs"] == {"dtype": "float32", "shape": [2], "data": [0.1, 0.2]}

    def test_optim_step_carries_metrics(self):
        rendered = render_result({"kind": "optim_step", "metrics": {"grad_norm": 1.5}})
        assert rendered == {"type": "optim_step", "metrics": {"grad_norm": 1.5}}

    def test_save_state_renders_the_tinker_path(self):
        rendered = render_result({"kind": "save_state", "path": "tinker://m/weights/x"})
        assert rendered == {"type": "save_weights", "path": "tinker://m/weights/x"}

    def test_sampler_saves_render_path_or_session(self):
        named = render_result(
            {"kind": "save_weights_for_sampler", "path": "tinker://m/sampler_weights/a", "sampling_session_id": None}
        )
        unnamed = render_result(
            {"kind": "save_weights_for_sampler", "path": "tinker://m/sampler_weights/1", "sampling_session_id": "s"}
        )
        assert named == {
            "type": "save_weights_for_sampler",
            "path": "tinker://m/sampler_weights/a",
            "sampling_session_id": None,
        }
        assert unnamed == {"type": "save_weights_for_sampler", "path": None, "sampling_session_id": "s"}

    def test_sample_renders_nan_as_null_and_topk_as_pairs(self):
        nan = float("nan")
        rendered = render_result(
            {
                "kind": "sample",
                "sequences": [{"tokens": [1], "logprobs": [-1.0], "stop_reason": "stop"}],
                "prompt_logprobs": [nan, -0.5],
                "topk_prompt_logprobs": {"token_ids": [[0, 0], [3, 4]], "logprobs": [[nan, nan], [-0.1, -0.2]]},
            }
        )
        assert rendered["prompt_logprobs"] == [None, -0.5]
        assert rendered["topk_prompt_logprobs"] == [None, [[3, -0.1], [4, -0.2]]]
        assert not any(isinstance(v, float) and math.isnan(v) for v in rendered["prompt_logprobs"] if v is not None)
