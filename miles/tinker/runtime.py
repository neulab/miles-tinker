"""The miles-bound half of the gateway.

Implements ExecutorBackend over TrainerController (training commands) and the
sglang router (sampling). This is where the gateway's internal language turns
into miles vocabulary: rows become RolloutBatch keys, neutral sampling params
become /generate payloads. Together with serve_tinker.py this is the only
place allowed to import miles.ray / miles.backends.
"""

import asyncio
import json
import logging
import time
import uuid
from argparse import Namespace
from pathlib import Path

import aiohttp

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.service import ExecutorBackend
from miles.tinker.core.types import UserInputError
from miles.utils import object_store
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp

logger = logging.getLogger(__name__)

# internal row key -> trainer batch key
ROW_TO_BATCH_KEYS = {"weights": "loss_weights", "advantages": "advantages", "sampling_logprobs": "rollout_log_probs"}

# Units are padded to a multiple of dp_size with rows that contribute nothing:
# weight/advantage 0 and an all-zero loss mask (every Tinker loss drops such a
# datum, tinker_losses._finish); their outputs are stripped before leaving the backend.
PAD_TOKEN_ID = 0
PAD_ROW = {
    "tokens": [PAD_TOKEN_ID, PAD_TOKEN_ID],
    "target_len": 1,
    "weights": [0.0],
    "advantages": [0.0],
    "sampling_logprobs": [0.0],
    "pad": True,
}
PERF_LOG_NAME = "perf.jsonl"
# /generate retries mirror miles.utils.http_utils.post: a bounded count with a pause between attempts
GENERATE_MAX_RETRIES = 3
GENERATE_RETRY_PAUSE_S = 1.0


class MilesBackend(ExecutorBackend):
    def __init__(self, args: Namespace, trainer, router_url: str, inference_controller=None) -> None:
        self.args = args
        self.trainer = trainer
        self.router_url = router_url
        self.inference_controller = inference_controller
        self.train_mode = getattr(args, "tinker_train_mode", "lora")
        self.dp_size = _actor_dp_size(args)
        checkpoint_root = getattr(args, "tinker_checkpoint_root", None)
        self.perf_log_path = Path(checkpoint_root) / PERF_LOG_NAME if checkpoint_root else None
        # exact (unpadded) per-slot counts of the open optim window
        self._window_counts: dict[int, dict[str, int]] = {}
        # bound lazily to the serving loop; see _generate_session
        self._generate_client: aiohttp.ClientSession | None = None

    async def warmup(self) -> None:
        """Make the engines router-visible: a full base sync for full fine-tuning,
        an empty update-weights round for LoRA (adapters are pushed per version)."""
        if self.train_mode == "full":
            await self.trainer.update_weights()
            if getattr(self.args, "check_weight_update_equal", False):
                # same post-push audit train.py runs: engines compare their weights against the checkpoint
                mismatches = await self.inference_controller.check_weights(
                    action="compare",
                    allow_quant_error=self.args.check_weight_update_allow_quant_error,
                    selector=self.args.check_weight_update_selector,
                    skip_list=self.args.check_weight_update_skip_list,
                )
                logger.info(f"tinker warmup: check_weights(compare) -> {mismatches}")
            return
        info = await self.inference_controller.start_update_weights()
        await self.inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        # full fine-tuning: the actor ignores rank/alpha and maps ckpt_path None to reset_to_base
        await self.trainer.load_slot(
            slot=slot, rank=rank, alpha=alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer
        )

    async def unload_slot(self, slot: int) -> None:
        self._window_counts.pop(slot, None)
        await self.trainer.unload_slot(slot)

    async def forward_backward(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        train_data = self._build_train_data(slot_rows)
        train_data["loss_fn"] = loss_fn
        train_data["loss_fn_config"] = loss_fn_config
        worker_results = await self._run_unit("forward_backward", unit_id, train_data)
        self._note_window(slot_rows)
        by_index: dict[int, dict] = {}
        for result in worker_results:
            for item in result["per_datum"]:
                index = int(item["sample_index"])
                if index not in by_index:
                    by_index[index] = {"loss": float(item["loss"]), "logprobs": item["logprobs"].tolist()}
        # padded rows sit past len(slot_rows) and are dropped here
        return [by_index[index] for index in range(len(slot_rows))]

    async def forward_only(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        worker_results = await self._run_unit("forward_only_logprobs", unit_id, self._build_train_data(slot_rows))
        by_index: dict[int, dict] = {}
        for box in worker_results:
            if box is None:
                continue
            value = _box_get(box)
            for index, logprobs in zip(value["sample_indices"], value["logprobs"], strict=True):
                if index not in by_index:
                    by_index[index] = {"loss": 0.0, "logprobs": logprobs.tolist()}
        return [by_index[index] for index in range(len(slot_rows))]

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        worker_results = await self.trainer.optim_step(adam_params_by_slot=adam_params_by_slot)
        metrics_by_slot = {}
        for slot, metrics in worker_results[0].items():
            counts = self._window_counts.pop(slot, _empty_window())
            metrics_by_slot[slot] = {**metrics, **counts}
            self._log_perf(slot, metrics_by_slot[slot])
        return metrics_by_slot

    async def discard_grads(self, slot: int) -> None:
        self._window_counts.pop(slot, None)
        await self.trainer.discard_grads(slot=slot)

    async def save_slot(self, slot: int, path: str) -> None:
        await self.trainer.save_slot(slot=slot, path=path)

    async def push_slot(self, slot: int, lora_name: str, rank: int, alpha: float) -> int | None:
        return await self.trainer.push_slot(slot=slot, lora_name=lora_name, rank=rank, alpha=alpha)

    async def unload_adapter(self, lora_name: str) -> None:
        await self.trainer.unload_adapter(lora_name=lora_name)

    def _build_train_data(self, slot_rows: list) -> dict:
        return _build_train_data(slot_rows, dp_size=self.dp_size, with_adapter_slots=self.train_mode != "full")

    def _note_window(self, slot_rows: list) -> None:
        touched = set()
        for slot, row in slot_rows:
            counts = self._window_counts.setdefault(slot, _empty_window())
            counts["num_tokens"] += len(row["tokens"])
            counts["num_datums"] += 1
            if slot not in touched:
                counts["num_units"] += 1
                touched.add(slot)

    def _log_perf(self, slot: int, metrics: dict) -> None:
        record = {"time": time.time(), "slot": slot, **metrics}
        logger.info(f"tinker perf: {json.dumps(record)}")
        if self.perf_log_path is not None:
            _append_json_line(self.perf_log_path, record)

    async def _run_unit(self, method: str, unit_id: int, train_data: dict) -> list:
        store = object_store.get_instance()
        data_ref = store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        try:
            return await getattr(self.trainer, method)(unit_id=unit_id, data_ref=data_ref)
        finally:
            store.remove(data_ref)

    # -------- sampling --------

    async def sample(self, payload: dict, lora_name: str | None) -> dict:
        request = self._generate_request(payload, lora_name)
        session = self._generate_session()
        responses = await asyncio.gather(
            *[
                post_generate(session, f"{self.router_url}/generate", per_sample)
                for per_sample in _per_sample_requests(request, payload["num_samples"])
            ]
        )
        result = {"sequences": [_to_sequence(response) for response in responses]}
        if payload["prompt_logprobs"]:
            result["prompt_logprobs"] = _prompt_logprobs(responses[0])
        if payload["topk_prompt_logprobs"]:
            result["topk_prompt_logprobs"] = _topk_prompt_logprobs(responses[0], payload["topk_prompt_logprobs"])
        return result

    def _generate_session(self) -> aiohttp.ClientSession:
        """The /generate client. aiohttp rather than the shared httpx client of
        miles.utils.http_utils: httpcore's pool rescans (and poll()s the socket of)
        every pooled connection on each request event, which costs the gateway's
        event loop 4-12 ms per sample and collapses throughput past ~100 in flight.
        In-flight posts are bounded by the service's sample semaphore."""
        if self._generate_client is None or self._generate_client.closed:
            self._generate_client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None)
            )
        return self._generate_client

    def _generate_request(self, payload: dict, lora_name: str | None) -> dict:
        params = payload["sampling_params"]
        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            raise UserInputError("sampling_params.max_tokens is required")
        prompt_tokens = payload["prompt_tokens"]
        sampling_params = {
            "max_new_tokens": _clamp_max_new_tokens(
                max_tokens, len(prompt_tokens), getattr(self.args, "rollout_max_context_len", None)
            ),
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", -1),
            "no_stop_trim": True,  # the matched stop token stays in the returned tokens (Tinker semantics)
        }
        if params.get("seed") is not None:
            sampling_params["sampling_seed"] = params["seed"]
        stop = params.get("stop")
        if stop == []:
            sampling_params["ignore_eos"] = True  # Tinker: an empty stop list disables every stop, EOS included
        elif stop is not None:
            if isinstance(stop, list) and stop and isinstance(stop[0], int):
                sampling_params["stop_token_ids"] = stop
            else:
                sampling_params["stop"] = stop
        request = {"input_ids": prompt_tokens, "sampling_params": sampling_params, "return_logprob": True}
        if payload["prompt_logprobs"] or payload["topk_prompt_logprobs"]:
            request["logprob_start_len"] = 0
        if payload["topk_prompt_logprobs"]:
            request["top_logprobs_num"] = payload["topk_prompt_logprobs"]
        if lora_name is not None:
            request["lora_path"] = lora_name
        return request


async def post_generate(
    session: aiohttp.ClientSession, url: str, payload: dict, max_retries: int = GENERATE_MAX_RETRIES
) -> dict:
    """POST one /generate body; transport and HTTP errors are retried like http_utils.post."""
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            logger.info(f"Error: {error!r}, retrying... (attempt {attempt}/{max_retries}, url={url})")
            if attempt >= max_retries:
                raise
            await asyncio.sleep(GENERATE_RETRY_PAUSE_S)
    raise AssertionError("unreachable")


def _clamp_max_new_tokens(max_tokens: int, prompt_len: int, context_len: int | None) -> int:
    """Fit the generation into an explicit engine context; without one the
    engine's own limit applies."""
    if not context_len:
        return max_tokens
    room = context_len - prompt_len
    if room <= 0:
        raise UserInputError(f"prompt of {prompt_len} tokens leaves no room in a {context_len}-token context")
    return min(max_tokens, room)


def _per_sample_requests(request: dict, num_samples: int) -> list[dict]:
    """One /generate body per sample; a client seed is offset per sample so
    deterministic engines do not return num_samples identical sequences."""
    requests = []
    for index in range(num_samples):
        sampling_params = dict(request["sampling_params"])
        if "sampling_seed" in sampling_params and num_samples > 1:
            sampling_params["sampling_seed"] = sampling_params["sampling_seed"] + index
        requests.append({**request, "sampling_params": sampling_params})
    return requests


def _actor_dp_size(args: Namespace) -> int:
    """Trainer data-parallel size: every unit is split round-robin across it."""
    actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    return max(1, actor_gpus // compute_megatron_world_size_except_dp(args))


def _pad_slot_rows(slot_rows: list, dp_size: int) -> list:
    """Append no-op rows so every DP rank gets the same number of rows (the
    trainer schedules ``len(rows) // dp_size`` per rank and asserts one step)."""
    assert slot_rows, "a unit carries at least one row"
    pad_count = -len(slot_rows) % dp_size
    pad_slot = slot_rows[0][0]
    return [*slot_rows, *([(pad_slot, PAD_ROW)] * pad_count)]


def _build_train_data(slot_rows: list, *, dp_size: int = 1, with_adapter_slots: bool = True) -> dict:
    padded = _pad_slot_rows(slot_rows, dp_size)
    rows = [row for _, row in padded]
    train_data = {
        "tokens": [row["tokens"] for row in rows],
        "loss_masks": [[0 if row.get("pad") else 1] * row["target_len"] for row in rows],
        "response_lengths": [row["target_len"] for row in rows],
        "total_lengths": [len(row["tokens"]) for row in rows],
        "sample_indices": list(range(len(rows))),
        "dynamic_global_batch_size": len(rows),
    }
    if with_adapter_slots:
        train_data["adapter_slots"] = [slot for slot, _ in padded]
    # optional keys follow the first (real) row; pad rows carry every key
    for row_key, batch_key in ROW_TO_BATCH_KEYS.items():
        if row_key in rows[0]:
            train_data[batch_key] = [row[row_key] for row in rows]
    return train_data


def _empty_window() -> dict[str, int]:
    return {"num_tokens": 0, "num_datums": 0, "num_units": 0}


def _append_json_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _box_get(box):
    import ray

    return ray.get(box.inner)


def _prompt_logprobs(response: dict) -> list[float]:
    entries = response["meta_info"]["input_token_logprobs"]
    return [float("nan") if entry[0] is None else float(entry[0]) for entry in entries]


def _topk_prompt_logprobs(response: dict, k: int) -> dict:
    token_ids, logprobs = [], []
    for position in response["meta_info"]["input_top_logprobs"]:
        candidates = position or []
        row_tokens = [entry[1] for entry in candidates][:k]
        row_logprobs = [float(entry[0]) for entry in candidates][:k]
        token_ids.append(row_tokens + [0] * (k - len(row_tokens)))
        logprobs.append(row_logprobs + [float("nan")] * (k - len(row_logprobs)))
    return {"token_ids": token_ids, "logprobs": logprobs}


def _to_sequence(response: dict) -> dict:
    output_token_logprobs = response["meta_info"]["output_token_logprobs"]
    finish = response["meta_info"]["finish_reason"]["type"]
    if finish == "abort":
        # an aborted generation is a server failure, not a clean stop
        raise RuntimeError("generation aborted by the engine")
    return {
        "sequence_id": f"seq-{uuid.uuid4().hex}",
        "tokens": [entry[1] for entry in output_token_logprobs],
        "logprobs": [entry[0] for entry in output_token_logprobs],
        "stop_reason": "length" if finish == "length" else "stop",
    }
