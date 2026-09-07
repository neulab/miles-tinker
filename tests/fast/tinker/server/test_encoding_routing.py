"""R3 datum handle on the wire: loss_fn_inputs["routing_handle"] (int64 [1])
becomes row["routing_handle"]: int on both wire halves."""

import pytest

from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import INPUT_ROW_KEYS, build_row, decode_command


def _datum(tokens: list[int], **extra_inputs) -> dict:
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
        "loss_fn_inputs": {"target_tokens": tokens[1:], **extra_inputs},
    }


def _decode(datum: dict) -> dict:
    payload = {"model_id": "m", "seq_id": 1, "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"}}
    return decode_command("forward_backward", payload)[1]["rows"][0]


def test_the_handle_is_decoded_to_an_int_from_json():
    handle = 2**52 + 3
    row = _decode(_datum([1, 2, 3], routing_handle={"dtype": "int64", "shape": [1], "data": [handle]}))
    assert row["routing_handle"] == handle and isinstance(row["routing_handle"], int)
    assert INPUT_ROW_KEYS["routing_handle"] == "routing_handle"


def test_a_missing_handle_leaves_the_row_untouched():
    assert "routing_handle" not in _decode(_datum([1, 2, 3], weights=[1.0, 1.0]))


@pytest.mark.parametrize("values", [[], [1, 2]])
def test_a_handle_of_the_wrong_length_is_rejected(values):
    with pytest.raises(UserInputError, match="routing_handle"):
        build_row([1, 2], {"target_tokens": [2, 3], "routing_handle": values}, 0)


def test_the_handle_survives_the_proto_half():
    tinker = pytest.importorskip("tinker")
    from tinker import types
    from tinker.proto.request_conv import forward_backward_request_to_proto

    from miles.tinker.server.proto_codec import decode_forward_backward_request

    del tinker
    handle = 987654321012
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([3, 1, 4, 1]),
        loss_fn_inputs={
            "target_tokens": [1, 4, 1, 5],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "routing_handle": types.TensorData(data=[handle], dtype="int64", shape=[1]),
        },
    )
    request = types.ForwardBackwardRequest(
        model_id="model-x",
        seq_id=7,
        forward_backward_input=types.ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
    )
    _, decoded = decode_forward_backward_request(forward_backward_request_to_proto(request).SerializeToString())
    assert decoded["rows"][0]["routing_handle"] == handle
    assert decoded["rows"][0]["weights"] == [1.0, 1.0, 1.0, 1.0]
