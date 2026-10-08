"""The binary half round-trips against the real SDK converters: requests the
SDK encodes must decode to the same rows as the JSON half, and our responses
must parse with the SDK's own deserializers."""

import pytest

tinker = pytest.importorskip("tinker")

from miles.tinker.server.encoding import decode_command, render_result  # noqa: E402
from miles.tinker.server.proto_codec import (  # noqa: E402
    decode_forward_backward_request,
    encode_forward_backward_output,
    encode_sample_response,
    maybe_decompress,
)
from tinker import types  # noqa: E402
from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results  # noqa: E402
from tinker.proto.request_conv import forward_backward_request_to_proto  # noqa: E402
from tinker.proto.response_conv import deserialize_forward_backward_output, deserialize_sample_response  # noqa: E402

TOKENS = [3, 1, 4, 1, 5]


def _sdk_request(loss_fn_inputs: dict) -> bytes:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(TOKENS[:-1]),
        loss_fn_inputs={"target_tokens": TOKENS[1:], **loss_fn_inputs},
    )
    request = types.ForwardBackwardRequest(
        model_id="model-x",
        seq_id=7,
        forward_backward_input=types.ForwardBackwardInput(
            data=[datum], loss_fn="cross_entropy", loss_fn_config={"beta": 0.5}
        ),
    )
    return forward_backward_request_to_proto(request).SerializeToString()


def test_an_sdk_proto_request_decodes_to_internal_rows():
    kind, decoded = decode_forward_backward_request(_sdk_request({"weights": [1.0, 0.0, 1.0, 1.0]}))

    assert (kind, decoded["model_id"], decoded["seq_id"]) == ("forward_backward", "model-x", 7)
    assert decoded["loss_fn_config"] == {"beta": 0.5}
    assert decoded["rows"] == [{"tokens": TOKENS, "target_len": 4, "weights": [1.0, 0.0, 1.0, 1.0]}]


def test_both_wire_halves_produce_the_same_rows():
    proto_rows = decode_forward_backward_request(_sdk_request({"weights": [1.0, 1.0, 1.0, 1.0]}))[1]["rows"]
    _, json_decoded = decode_command(
        "forward_backward",
        {
            "model_id": "model-x",
            "seq_id": 7,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": TOKENS[:-1]}]},
                        "loss_fn_inputs": {"target_tokens": TOKENS[1:], "weights": [1.0, 1.0, 1.0, 1.0]},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        },
    )
    assert proto_rows == json_decoded["rows"]


def test_a_sparse_tensor_decodes_dense():
    sparse = types.TensorData(
        dtype="float32", shape=[4], data=[2.0], sparse_crow_indices=[0, 1], sparse_col_indices=[2]
    )
    _, decoded = decode_forward_backward_request(_sdk_request({"weights": sparse}))
    assert decoded["rows"][0]["weights"] == [0.0, 0.0, 2.0, 0.0]


def test_a_zstd_body_decompresses():
    zstandard = pytest.importorskip("zstandard")
    body = _sdk_request({"weights": [1.0, 1.0, 1.0, 1.0]})
    compressed = zstandard.ZstdCompressor().compress(body)

    assert maybe_decompress(compressed, "zstd") == body
    assert maybe_decompress(body, None) == body


def test_the_sdk_parses_our_forward_backward_output():
    encoded = encode_forward_backward_output(
        {
            "kind": "forward_backward",
            "outputs": [{"loss": 2.5, "logprobs": [-0.1, -0.2]}, {"loss": 1.0, "logprobs": [-0.3]}],
        }
    )
    parsed = deserialize_forward_backward_output(encoded)

    assert parsed.metrics["loss:sum"] == pytest.approx(3.5)
    assert [list(record["loss:sum"].data) for record in parsed.loss_fn_outputs] == [[2.5], [1.0]]
    assert list(parsed.loss_fn_outputs[0]["logprobs"].data) == pytest.approx([-0.1, -0.2])


def test_the_sdk_parses_our_sample_response():
    encoded = encode_sample_response(
        {
            "kind": "sample",
            "sequences": [{"sequence_id": "s", "tokens": [5, 6], "logprobs": [-0.5, -0.6], "stop_reason": "length"}],
            "prompt_logprobs": [-1.0, -2.0],
            "topk_prompt_logprobs": {"token_ids": [[1, 2]], "logprobs": [[-0.1, -0.2]]},
        }
    )
    parsed = deserialize_sample_response(encoded)

    sequence = parsed.sequences[0]
    assert (list(sequence.tokens), sequence.stop_reason) == ([5, 6], "length")
    assert list(sequence.logprobs) == pytest.approx([-0.5, -0.6])
    assert list(parsed.prompt_logprobs) == pytest.approx([-1.0, -2.0])


def test_string_loss_fn_config_comes_from_the_v2_map():
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(TOKENS[:-1]), loss_fn_inputs={"target_tokens": TOKENS[1:]}
    )
    request = types.ForwardBackwardRequest(
        model_id="model-x",
        seq_id=1,
        forward_backward_input=types.ForwardBackwardInput(
            data=[datum], loss_fn="cispo", loss_fn_config={"mode": "ratio", "clip_low_threshold": 0.8}
        ),
    )
    _, decoded = decode_forward_backward_request(forward_backward_request_to_proto(request).SerializeToString())
    assert decoded["loss_fn_config"] == {"mode": "ratio", "clip_low_threshold": 0.8}


def test_our_forward_backward_metrics_carry_token_and_datum_counts():
    encoded = encode_forward_backward_output(
        {
            "kind": "forward_backward",
            "outputs": [{"loss": 1.0, "logprobs": [-0.1, -0.2]}, {"loss": 1.0, "logprobs": [-0.3]}],
        }
    )
    parsed = deserialize_forward_backward_output(encoded)
    assert parsed.metrics == {"loss:sum": 2.0, "num_tokens:sum": 3.0, "num_datums:sum": 2.0}


def test_sdk_chunk_reduction_preserves_value_outputs_and_counts():
    chunks = []
    for _ in range(2):
        result = {
            "kind": "forward_only",
            "outputs": [{"loss": 0.0, "values": [0.25, -0.5]}],
        }
        parsed = deserialize_forward_backward_output(encode_forward_backward_output(result))
        assert parsed.metrics == render_result(result)["metrics"]
        chunks.append(parsed)
    combined = combine_fwd_bwd_output_results(chunks)
    assert combined.metrics == {"loss:sum": 0.0, "num_tokens:sum": 4.0, "num_datums:sum": 2.0}
    assert [row["values"].data for row in combined.loss_fn_outputs] == [[0.25, -0.5], [0.25, -0.5]]
