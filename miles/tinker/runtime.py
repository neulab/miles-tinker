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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
import pybase64

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.service import ExecutorBackend
from miles.tinker.core.types import UserInputError
from miles.tinker.routing_cache import (
    DEFAULT_MAX_BYTES,
    DEFAULT_PREFIX_STRIDE,
    DEFAULT_TTL_S,
    UNCOVERED,
    RoutingCache,
    RoutingCaptureError,
    RoutingEntry,
)
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
# R3 (rollout routing replay) policy when a datum has no cached routing
ROUTING_POLICY_NATURAL = "natural"  # rows of -1: the trainer routes through the live router, metrics count it
ROUTING_POLICY_STRICT = "strict"  # the unit fails typed (category user)
PREFIX_HIT_LOG_LIMIT = 8  # INFO lines for the first prefix-parent hits of a process (the smoke greps one)
ROUTING_WINDOW_KEYS = ("routing_covered_datums", "routing_uncovered_datums", "routing_uncovered_rows")


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
        # R3: routing captured at sampling time, materialized into the units (docs/fork/r3.md)
        self.routing: RoutingCache | None = _make_routing_cache(args)
        self.routing_replay = self.routing is not None
        self.routing_policy = getattr(args, "tinker_routing_replay_policy", ROUTING_POLICY_NATURAL)
        self._prefix_hit_logs_left = PREFIX_HIT_LOG_LIMIT
        # JSON parse / base64 decode of /generate bodies, materialization and object_store.put run off the loop
        self._routing_executor = (
            ThreadPoolExecutor(max_workers=4, thread_name_prefix="r3") if self.routing is not None else None
        )

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
        train_data, routing_counts = await self._prepare_train_data(slot_rows)
        train_data["loss_fn"] = loss_fn
        train_data["loss_fn_config"] = loss_fn_config
        worker_results = await self._run_unit("forward_backward", unit_id, train_data)
        self._note_window(slot_rows, routing_counts)
        by_index: dict[int, dict] = {}
        for result in worker_results:
            for item in result["per_datum"]:
                index = int(item["sample_index"])
                if index not in by_index:
                    by_index[index] = {"loss": float(item["loss"]), "logprobs": item["logprobs"].tolist()}
        # padded rows sit past len(slot_rows) and are dropped here
        return [by_index[index] for index in range(len(slot_rows))]

    async def forward_only(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        train_data, _ = await self._prepare_train_data(slot_rows)
        worker_results = await self._run_unit("forward_only_logprobs", unit_id, train_data)
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
        routing_metrics = self._routing_metrics()
        for slot, metrics in worker_results[0].items():
            counts = self._window_counts.pop(slot, _empty_window(self.routing_replay))
            # trainer metrics win over the gateway defaults (fused_lm_head); exact window counts win over both
            metrics_by_slot[slot] = {**routing_metrics, **metrics, **counts}
            self._log_perf(slot, metrics_by_slot[slot])
        return metrics_by_slot

    def sweep(self) -> None:
        """Periodic housekeeping from the service's lease sweep: routing cache TTL / byte cap."""
        if self.routing is not None:
            stats = self.routing.sweep()
            logger.debug(f"tinker R3 cache: {stats}")

    async def discard_grads(self, slot: int) -> None:
        self._window_counts.pop(slot, None)
        await self.trainer.discard_grads(slot=slot)

    async def save_slot(self, slot: int, path: str) -> None:
        await self.trainer.save_slot(slot=slot, path=path)

    async def push_slot(self, slot: int, lora_name: str, rank: int, alpha: float) -> int | None:
        return await self.trainer.push_slot(slot=slot, lora_name=lora_name, rank=rank, alpha=alpha)

    async def unload_adapter(self, lora_name: str) -> None:
        await self.trainer.unload_adapter(lora_name=lora_name)

    def _build_train_data(self, slot_rows: list, routing=None) -> dict:
        return _build_train_data(
            slot_rows, dp_size=self.dp_size, with_adapter_slots=self.train_mode != "full", routing=routing
        )

    async def _prepare_train_data(self, slot_rows: list) -> tuple[dict, dict | None]:
        """The unit's train_data; with R3 the routing rows are materialized in
        the executor and the per-datum coverage counts come back with it."""
        if self.routing is None:
            return self._build_train_data(slot_rows), None
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        train_data, covered = await loop.run_in_executor(self._routing_executor, self._materialize_unit, slot_rows)
        return train_data, {"covered": covered, "materialize_s": time.perf_counter() - started}

    def _materialize_unit(self, slot_rows: list) -> tuple[dict, list[bool]]:
        """Executor-side: rollout_routed_experts for every row (pad rows and
        uncovered datums are all -1) and a per-datum coverage flag. strict
        policy: uncovered datums fail the unit."""
        routing = self.routing
        shape = (routing.num_layers, routing.topk)
        covered: list[bool] = []

        def rows_for(row: dict) -> np.ndarray:
            num_rows = len(row["tokens"]) - 1
            if row.get("pad"):
                return np.full((num_rows, *shape), UNCOVERED, dtype=np.int32)
            rows = routing.materialize(row.get("tenant", ""), row.get("routing_handle"), row["tokens"])
            covered.append(rows is not None)
            if rows is None:
                return np.full((num_rows, *shape), UNCOVERED, dtype=np.int32)
            return rows

        train_data = self._build_train_data(slot_rows, routing=rows_for)
        uncovered = [i for i, ok in enumerate(covered) if not ok]
        if uncovered and self.routing_policy == ROUTING_POLICY_STRICT:
            raise UserInputError(
                "; ".join(f"datum {i}: R3 is enabled but no routing is cached for it" for i in uncovered)
            )
        return train_data, covered

    def _routing_metrics(self) -> dict:
        if self.routing is None:
            return {}
        stats = self.routing.stats()
        return {
            "routing_capture_rejected": stats["capture_rejected"],
            "routing_cache_entries": stats["entries"],
            "routing_cache_bytes": stats["bytes"],
            "routing_cache_pinned": stats["pinned"],
            "routing_cache_evictions": stats["evictions"],
            "routing_prefix_hits": stats["prefix_hits"],
            "fused_lm_head": int(bool(getattr(self.args, "fused_lm_head_logprobs", False))),
        }

    def _note_window(self, slot_rows: list, routing_note: dict | None = None) -> None:
        touched = set()
        for index, (slot, row) in enumerate(slot_rows):
            counts = self._window_counts.setdefault(slot, _empty_window(self.routing_replay))
            counts["num_tokens"] += len(row["tokens"])
            counts["num_datums"] += 1
            if slot not in touched:
                counts["num_units"] += 1
                touched.add(slot)
                if routing_note is not None:
                    counts["routing_materialize_s"] += routing_note["materialize_s"]
            if routing_note is not None:
                if routing_note["covered"][index]:
                    counts["routing_covered_datums"] += 1
                else:
                    counts["routing_uncovered_datums"] += 1
                    counts["routing_uncovered_rows"] += len(row["tokens"]) - 1

    def _log_perf(self, slot: int, metrics: dict) -> None:
        record = {"time": time.time(), "slot": slot, **metrics}
        logger.info(f"tinker perf: {json.dumps(record)}")
        if self.perf_log_path is not None:
            _append_json_line(self.perf_log_path, record)

    async def _run_unit(self, method: str, unit_id: int, train_data: dict) -> list:
        store = object_store.get_instance()
        if self._routing_executor is not None:  # ~GBs per unit with R3: serialize off the loop
            data_ref = await asyncio.get_running_loop().run_in_executor(
                self._routing_executor, lambda: store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
            )
        else:
            data_ref = store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        try:
            return await getattr(self.trainer, method)(unit_id=unit_id, data_ref=data_ref)
        finally:
            store.remove(data_ref)

    # -------- sampling --------

    async def sample(self, payload: dict, lora_name: str | None) -> dict:
        request = self._generate_request(payload, lora_name)
        capture = self._begin_capture(payload, request)
        try:
            session = self._generate_session()
            bodies = await asyncio.gather(
                *[
                    post_generate(session, f"{self.router_url}/generate", per_sample)
                    for per_sample in _per_sample_requests(request, payload["num_samples"])
                ]
            )
            if capture is None:
                responses = [_parse_body(body) for body in bodies]
                sequences = [_to_sequence(response) for response in responses]
            else:
                responses, sequences = await self._capture_sequences(capture, bodies)
        finally:
            if capture is not None:
                capture.release(self.routing)
        result = {"sequences": sequences}
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

    def _begin_capture(self, payload: dict, request: dict) -> "_RoutingCapture | None":
        """R3: ask the engine for the routing and, when a cached sequence shares
        a prefix with the prompt, only for the rows past it. The parent stays
        pinned until every sample is recorded (or the request fails)."""
        if self.routing is None:
            return None
        tenant = payload.get("tenant", "")
        prompt_tokens = payload["prompt_tokens"]
        request["return_routed_experts"] = True
        found = self.routing.find_prefix_parent(tenant, prompt_tokens)
        capture = _RoutingCapture(tenant=tenant, prompt_tokens=prompt_tokens)
        if found is not None:
            capture.parent, capture.start_len = found
            capture.pins = 1
            for _ in range(payload["num_samples"] - 1):  # one pin per record() to come
                self.routing.pin(capture.parent)
                capture.pins += 1
            request["routed_experts_start_len"] = capture.start_len
            if self._prefix_hit_logs_left > 0:  # the first few hits at INFO; the count lives in routing_prefix_hits
                self._prefix_hit_logs_left -= 1
                logger.info(
                    f"tinker R3: prefix parent hit routing_id={capture.parent.routing_id} "
                    f"routed_experts_start_len={capture.start_len} prompt_len={len(prompt_tokens)}"
                )
        return capture

    async def _capture_sequences(self, capture: "_RoutingCapture", bodies: list) -> tuple[list[dict], list[dict]]:
        """Decode the routed responses off the loop, then record each sequence's
        routing and stamp its routing_id."""
        loop = asyncio.get_running_loop()
        routing = self.routing
        try:
            decoded = await asyncio.gather(
                *[
                    loop.run_in_executor(
                        self._routing_executor,
                        _decode_routed_response,
                        body,
                        len(capture.prompt_tokens),
                        capture.start_len,
                        routing.num_layers,
                        routing.topk,
                        routing.num_experts,
                    )
                    for body in bodies
                ]
            )
        except RoutingCaptureError as error:
            routing.note_capture_rejected()
            raise RuntimeError(f"routing capture rejected: {error}") from error
        responses, sequences = [], []
        for response, delta_rows in decoded:
            sequence = _to_sequence(response)  # abort -> RuntimeError before any record
            try:
                sequence["routing_id"] = routing.record(
                    capture.tenant,
                    capture.prompt_tokens,
                    sequence["tokens"],
                    delta_rows,
                    capture.parent,
                    capture.start_len,
                )
            except RoutingCaptureError as error:
                raise RuntimeError(f"routing capture rejected: {error}") from error
            if capture.parent is not None:
                capture.pins -= 1  # record() consumed one pin
            responses.append(response)
            sequences.append(sequence)
        return responses, sequences

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


@dataclass
class _RoutingCapture:
    """Request-local R3 state of one sample() call."""

    tenant: str
    prompt_tokens: list[int]
    parent: RoutingEntry | None = None
    start_len: int = 0  # rows [0, start_len) come from the parent
    pins: int = 0  # pins on the parent not yet consumed by record()

    def release(self, routing: RoutingCache) -> None:
        for _ in range(self.pins):
            routing.unpin(self.parent)
        self.pins = 0


async def post_generate(
    session: aiohttp.ClientSession, url: str, payload: dict, max_retries: int = GENERATE_MAX_RETRIES
) -> bytes:
    """POST one /generate body and return the raw response bytes (parsed by the
    caller, off the loop for routed responses); transport and HTTP errors are
    retried like http_utils.post."""
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.read()
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


def _build_train_data(slot_rows: list, *, dp_size: int = 1, with_adapter_slots: bool = True, routing=None) -> dict:
    """The trainer batch of one unit. ``routing(row) -> np.int32 [len(tokens)-1, L, K]``
    attaches ``rollout_routed_experts`` for every row (pad rows included)."""
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
    if routing is not None:
        train_data["rollout_routed_experts"] = [routing(row) for row in rows]
    return train_data


def _empty_window(routing: bool = False) -> dict[str, int | float]:
    window: dict[str, int | float] = {"num_tokens": 0, "num_datums": 0, "num_units": 0}
    if routing:
        window.update(dict.fromkeys(ROUTING_WINDOW_KEYS, 0))
        window["routing_materialize_s"] = 0.0
    return window


def _make_routing_cache(args: Namespace) -> RoutingCache | None:
    if not getattr(args, "use_rollout_routing_replay", False):
        return None
    num_experts = getattr(args, "num_experts", None)
    num_layers = getattr(args, "num_layers", None)
    topk = getattr(args, "moe_router_topk", None)
    if not (num_experts and num_layers and topk):
        raise ValueError("--use-rollout-routing-replay needs --num-experts, --num-layers and --moe-router-topk")
    if num_experts > 256:
        raise ValueError(f"R3 routing cache stores expert ids as uint8: num_experts={num_experts} > 256")
    cache = RoutingCache(
        num_layers=num_layers,
        topk=topk,
        num_experts=num_experts,
        max_bytes=getattr(args, "tinker_routing_cache_max_bytes", DEFAULT_MAX_BYTES),
        ttl_s=getattr(args, "tinker_routing_cache_ttl_s", DEFAULT_TTL_S),
        prefix_stride=getattr(args, "tinker_routing_prefix_stride", DEFAULT_PREFIX_STRIDE),
    )
    logger.info(
        f"tinker R3: routing cache on, routing shape (L={num_layers}, K={topk}), num_experts={num_experts}, "
        f"policy={getattr(args, 'tinker_routing_replay_policy', ROUTING_POLICY_NATURAL)}, "
        f"max_bytes={cache.max_bytes}, ttl_s={cache.ttl_s}, prefix_stride={cache.prefix_stride}"
    )
    return cache


def _parse_body(body) -> dict:
    return body if isinstance(body, dict) else json.loads(body)


def _decode_routed_response(
    body, prompt_len: int, start_len: int, num_layers: int, topk: int, num_experts: int
) -> tuple[dict, np.ndarray | None]:
    """Executor-side: the /generate response dict and its routing rows
    [prompt_len + output_len - 1 - start_len, L, K] as uint8. Mirrors
    miles/rollout/sglang_rollout.py (stop-edge trim), plus the range check."""
    response = _parse_body(body)
    meta = response["meta_info"]
    if meta["finish_reason"]["type"] == "abort":
        return response, None  # _to_sequence turns the abort into the failure
    encoded = meta.get("routed_experts")
    if not encoded:
        raise RoutingCaptureError(
            "response carries no routed_experts (engines need enable_return_routed_experts: --use-rollout-routing-replay)"
        )
    raw = np.frombuffer(pybase64.b64decode(encoded.encode("ascii")), dtype=np.int32)
    ntok = prompt_len + len(meta["output_token_logprobs"]) - 1 - start_len
    per_row = num_layers * topk
    if raw.size == (ntok + 1) * per_row:
        # stop-edge: the engine also forwarded the final token; its row feeds no training position
        raw = raw[: ntok * per_row]
    if ntok < 0 or raw.size != ntok * per_row:
        raise RoutingCaptureError(
            f"routed_experts buffer {raw.size} != ntok({ntok}) x layers({num_layers}) x topk({topk}); "
            f"prompt={prompt_len} output={len(meta['output_token_logprobs'])} start_len={start_len}"
        )
    rows = raw.reshape(ntok, num_layers, topk)
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= num_experts):
        raise RoutingCaptureError(
            f"expert ids outside [0, {num_experts}): min {int(rows.min())} max {int(rows.max())}"
        )
    return response, rows.astype(np.uint8)


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
