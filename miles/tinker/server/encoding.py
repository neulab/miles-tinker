"""SDK wire <-> gateway internal language, JSON half (proto_codec.py is the
binary half; both produce and consume the same internal shapes).

Decode errors are protocol violations and answer HTTP 400; size admission
happens later in core and fails the promise instead.

Sequence mapping: a datum's model_input tokens x[0..T-1] and target_tokens
t[0..T-1] must satisfy t[i] == x[i+1] for i < T-1 (standard next-token
supervision). The encoded row is x + [t[-1]] with T supervised positions, so
the trainer scores exactly logprob(t[i] | x[0..i]).

Two SDK dialects arrive here: 0.27.1 serialises every field, 0.16.1 omits
unset defaults (chunk ``type``, adam fields) and sends explicit nulls.
"""

import logging
import math

from miles.tinker.core.types import UserInputError

logger = logging.getLogger(__name__)

# materialized at the boundary so core and the executor can require every key
ADAM_PARAM_DEFAULTS = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-12,
    "weight_decay": 0.0,
    "grad_clip_norm": 0.0,
}

# wire loss_fn_inputs key -> internal row key
INPUT_ROW_KEYS = {
    "weights": "weights",
    "advantages": "advantages",
    "logprobs": "sampling_logprobs",
    "routing_handle": "routing_handle",  # R3: int64 [1] = routing_id of the sample call the datum came from
}


def decode_command(kind: str, payload: dict) -> tuple[str, dict]:
    """One JSON command body -> (kind, internal payload)."""
    decoded = {"model_id": payload["model_id"], "seq_id": payload["seq_id"]}
    if kind == "forward_backward":
        fb_input = payload["forward_backward_input"]
        datums = [
            (model_input_tokens(datum["model_input"]), _decode_inputs(datum["loss_fn_inputs"]))
            for datum in fb_input["data"]
        ]
        decoded |= {
            "rows": [build_row(tokens, inputs, i) for i, (tokens, inputs) in enumerate(datums)],
            "loss_fn": fb_input["loss_fn"],
            "loss_fn_config": fb_input.get("loss_fn_config") or {},
        }
        return ("forward_only" if payload.get("forward_only") else kind), decoded
    if kind == "optim_step":
        return kind, decoded | {"adam_params": materialize_adam_params(payload["adam_params"])}
    if kind == "save_state":
        return kind, decoded | {"name": payload.get("path")}
    if kind == "load_state":
        return kind, decoded | {"path": payload["path"], "optimizer": payload["optimizer"]}
    if kind == "save_weights_for_sampler":
        return kind, decoded | {
            "name": payload.get("path"),
            "sampling_session_seq_id": payload.get("sampling_session_seq_id"),
        }
    raise UserInputError(f"unknown command kind {kind!r}")


def decode_forward_request(payload: dict) -> tuple[str, dict]:
    """0.16.1's POST /forward: the same body under ``forward_input``."""
    body = {**payload, "forward_backward_input": payload["forward_input"], "forward_only": True}
    return decode_command("forward_backward", body)


def materialize_adam_params(raw: dict) -> dict:
    unknown = set(raw) - set(ADAM_PARAM_DEFAULTS)
    if unknown:
        logger.warning(f"ignoring unknown adam_params keys: {sorted(unknown)}")
    known = {key: value for key, value in raw.items() if key in ADAM_PARAM_DEFAULTS and value is not None}
    return {**ADAM_PARAM_DEFAULTS, **known}


def model_input_tokens(model_input: dict) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input["chunks"]:
        chunk_type = chunk.get("type", "encoded_text")  # 0.16.1 omits the default discriminator
        if chunk_type != "encoded_text":
            raise UserInputError(f"unsupported model_input chunk type: {chunk_type}")
        tokens.extend(chunk["tokens"])
    return tokens


def build_row(input_tokens: list[int], inputs: dict[str, list], index: int) -> dict:
    """One decoded datum (token list + loss_fn_inputs lists) -> internal row."""
    targets = [int(t) for t in inputs["target_tokens"]]
    if len(targets) != len(input_tokens):
        raise UserInputError(
            f"datum {index}: target_tokens length {len(targets)} != model_input length {len(input_tokens)}"
        )
    if targets[:-1] != input_tokens[1:]:
        raise UserInputError(
            f"datum {index}: target_tokens must be model_input shifted by one (next-token supervision)"
        )
    row = {"tokens": input_tokens + targets[-1:], "target_len": len(targets)}
    for wire_key, row_key in INPUT_ROW_KEYS.items():
        if wire_key not in inputs:
            continue
        if wire_key == "routing_handle":
            row[row_key] = _routing_handle(inputs[wire_key], index)
        else:
            row[row_key] = [float(value) for value in inputs[wire_key]]
    return row


def _routing_handle(values: list, index: int) -> int:
    if len(values) != 1:
        raise UserInputError(f"datum {index}: routing_handle must hold exactly one value, got {len(values)}")
    return int(values[0])


def _decode_inputs(loss_fn_inputs: dict) -> dict[str, list]:
    return {name: tensor_data_to_list(value) for name, value in loss_fn_inputs.items()}


def tensor_data_to_list(tensor_data) -> list:
    if isinstance(tensor_data, list):
        return tensor_data
    if not isinstance(tensor_data, dict):
        raise UserInputError(f"expected TensorData, got {type(tensor_data).__name__}")
    if tensor_data.get("sparse_crow_indices") is not None:
        return _dense_from_csr(tensor_data)
    data = tensor_data.get("data")
    if data is None:
        raise UserInputError("TensorData without data")
    return list(data)


def _dense_from_csr(tensor_data: dict) -> list:
    (length,) = tensor_data["shape"]
    assert len(tensor_data["sparse_crow_indices"]) == 2, "1-D CSR expected"
    dense = [0] * length
    for col, value in zip(tensor_data["sparse_col_indices"], tensor_data["data"], strict=True):
        dense[col] = value
    return dense


def decode_sample_request(payload: dict) -> dict:
    return {
        "model_path": payload.get("model_path"),
        "sampling_session_id": payload.get("sampling_session_id"),
        "seq_id": payload.get("seq_id"),
        "num_samples": payload.get("num_samples", 1),
        "prompt_tokens": model_input_tokens(payload["prompt"]),
        "sampling_params": payload.get("sampling_params") or {},
        "prompt_logprobs": bool(payload.get("prompt_logprobs")),
        "topk_prompt_logprobs": payload.get("topk_prompt_logprobs", 0) or 0,
    }


# -------- result rendering (JSON; proto_codec renders the binary forms) --------


def forward_backward_metrics(outputs: list[dict]) -> dict[str, float]:
    """Request-level metrics in the SDK's ``name:reduction`` form."""
    return {
        "loss:sum": float(sum(output["loss"] for output in outputs)),
        "num_tokens:sum": float(sum(len(output["logprobs"]) for output in outputs)),
        "num_datums:sum": float(len(outputs)),
    }


def render_result(result: dict) -> dict:
    kind = result["kind"]
    if kind in ("forward_backward", "forward_only"):
        outputs = result["outputs"]
        return {
            "type": "forward_backward",
            "loss_fn_output_type": "ArrayRecord",
            "loss_fn_outputs": [
                {"loss:sum": _tensor_json([output["loss"]]), "logprobs": _tensor_json(output["logprobs"])}
                for output in outputs
            ],
            "metrics": forward_backward_metrics(outputs),
        }
    if kind == "sample":
        return _render_sample(result)
    if kind == "create_model":
        return {"type": "create_model", "model_id": result["model_id"]}
    if kind == "unload_model":
        return {"type": "unload_model", "model_id": result["model_id"]}
    if kind == "save_state":
        return {"type": "save_weights", "path": result["path"]}
    if kind == "save_weights_for_sampler":
        sampling_session_id = result.get("sampling_session_id")
        # unnamed saves answer only the minted session: 0.16.1 asserts path is None there
        path = None if sampling_session_id is not None else result["path"]
        return {"type": "save_weights_for_sampler", "path": path, "sampling_session_id": sampling_session_id}
    if kind == "load_state":
        return {"type": "load_weights"}
    if kind == "optim_step":
        return {"type": "optim_step", "metrics": result["metrics"]}
    raise AssertionError(f"unrenderable result kind {kind!r}")


def _render_sample(result: dict) -> dict:
    """JSON SampleResponse (0.16.1 shape): NaN becomes null, top-k rows are
    lists of [token, logprob] pairs with null for unscored positions."""
    rendered = {"type": "sample", "sequences": result["sequences"]}
    if result.get("prompt_logprobs") is not None:
        rendered["prompt_logprobs"] = [_nan_to_none(value) for value in result["prompt_logprobs"]]
    if result.get("topk_prompt_logprobs") is not None:
        topk = result["topk_prompt_logprobs"]
        rendered["topk_prompt_logprobs"] = [
            _topk_pairs(tokens, logprobs) for tokens, logprobs in zip(topk["token_ids"], topk["logprobs"], strict=True)
        ]
    return rendered


def _topk_pairs(tokens: list[int], logprobs: list[float]) -> list | None:
    pairs = [[token, logprob] for token, logprob in zip(tokens, logprobs, strict=True) if not math.isnan(logprob)]
    return pairs or None


def _nan_to_none(value: float) -> float | None:
    return None if math.isnan(value) else value


def _tensor_json(values: list[float]) -> dict:
    return {"dtype": "float32", "shape": [len(values)], "data": values}
