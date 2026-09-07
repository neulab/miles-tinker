"""The gateway service: sessions, models, streams, promises, and the single
dispatch loop.

Speaks only the internal language: server/ hands it decoded commands and
renders its results; runtime.py turns units into trainer batches. All backend
calls go through one loop / one lock: the trainer is an SPMD domain and must
see a single totally ordered unit stream.
"""

import asyncio
import json
import logging
import math
import os
import shutil
import time
import uuid

from miles.tinker.core.planner import BarrierUnit, Planner, WorkUnit
from miles.tinker.core.promise import PENDING, Promise, PromiseStore
from miles.tinker.core.stream import ModelStream
from miles.tinker.core.types import Command, GatewayConfig, ModelRecord, OwnershipError, UserInputError

logger = logging.getLogger(__name__)

SIDECAR_NAME = "tinker_meta.json"


class ExecutorBackend:
    """What runtime.py implements. Speaks rows and plain lists; core stays
    torch-free and miles-free."""

    # R3 (rollout routing replay): sequences carry routing_id, datums may carry routing_handle
    routing_replay: bool = False

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        """FFT: rank/alpha are ignored; ckpt_path None resets slot 0 to the
        pristine base weights with a fresh optimizer."""
        raise NotImplementedError

    async def unload_slot(self, slot: int) -> None:
        raise NotImplementedError

    async def forward_backward(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        """slot_rows: slot-sorted [(slot, row)]. Returns one
        {"loss": float, "logprobs": [float]} per row, in order."""
        raise NotImplementedError

    async def forward_only(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        raise NotImplementedError

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        """-> per slot a plain metrics dict, always carrying "grad_norm"."""
        raise NotImplementedError

    async def discard_grads(self, slot: int) -> None:
        """Zero the gradients a slot accumulated (poisoned window)."""
        raise NotImplementedError

    async def save_slot(self, slot: int, path: str) -> None:
        raise NotImplementedError

    async def push_slot(self, slot: int, lora_name: str, rank: int, alpha: float) -> int | None:
        """LoRA: register/push the adapter under lora_name, return None.
        FFT: sync the base weights, return the new engine weight version."""
        raise NotImplementedError

    async def unload_adapter(self, lora_name: str) -> None:
        raise NotImplementedError

    async def warmup(self) -> None:
        """Make the engines router-visible before the first client arrives."""
        raise NotImplementedError

    async def sample(self, payload: dict, lora_name: str | None) -> dict:
        """-> {"sequences": [{"tokens", "logprobs", "stop_reason", "routing_id"?}],
        "prompt_logprobs"?, "topk_prompt_logprobs"?}. lora_name None: base
        weights (LoRA) or the latest pushed weights (FFT). payload["tenant"]
        names the caller (R3 routing is cached per tenant)."""
        raise NotImplementedError

    def sweep(self) -> None:
        """Periodic housekeeping from the lease sweep (R3 cache TTL / byte cap); no-op by default."""
        return None


class TinkerService:
    def __init__(self, backend: ExecutorBackend, config: GatewayConfig) -> None:
        if not config.is_lora and config.n_slots != 1:
            raise ValueError(f"train_mode='full' serves exactly one slot, got n_slots={config.n_slots}")
        self.backend = backend
        self.config = config
        self.promises = PromiseStore()
        self.planner = Planner(config.unit_token_budget)
        self.models: dict[str, ModelRecord] = {}
        self.sessions: dict[str, dict] = {}
        self.sampling_sessions: dict[str, dict] = {}
        self.free_slots = set(range(config.n_slots))
        self._wake = asyncio.Event()
        self._backend_lock = asyncio.Lock()
        self._sample_slots = asyncio.Semaphore(config.max_concurrent_samples)
        self._sample_tasks: dict[str, tuple] = {}  # request_id -> (task, tenant, sampling_session_id, lora_name, dedup_key)
        self._sample_by_seq: dict[tuple, tuple[str, list[str]]] = {}  # (sampling_session_id, seq_id) -> reply
        self._model_by_seq: dict[tuple, tuple[str, str]] = {}  # (session_id, model_seq_id) -> reply
        self._background_tasks: set = set()
        self._last_submit_at: dict[str, float] = {}  # model_id -> monotonic time of the last window submit
        self._arrival_counter = 0
        self._unit_counter = 0

    # -------- control plane --------

    def create_session(self, tenant: str, payload: dict) -> str:
        session_id = f"session-{uuid.uuid4().hex}"
        self.sessions[session_id] = {"tenant": tenant, "last_heartbeat": time.monotonic(), "payload": payload}
        return session_id

    def heartbeat(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            session["last_heartbeat"] = time.monotonic()

    def create_model(self, tenant: str, payload: dict) -> tuple[str, str]:
        """Two-phase like every command: allocate now, initialize the slot behind the promise."""
        dedup_key = (payload.get("session_id"), payload.get("model_seq_id"))
        if None not in dedup_key and dedup_key in self._model_by_seq:
            return self._model_by_seq[dedup_key]  # the SDK retried; the first allocation stands
        base_model = payload["base_model"]
        if base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        is_lora, rank, alpha = _resolve_lora_config(self.config, payload.get("lora_config"))
        if not self.free_slots:
            raise UserInputError(_no_slot_message(self.config))
        slot = min(self.free_slots)
        self.free_slots.remove(slot)

        model_id = f"model-{uuid.uuid4().hex[:12]}"
        session_id = payload.get("session_id") or ""
        record = ModelRecord(
            model_id=model_id,
            tenant=tenant,
            slot=slot,
            base_model=base_model,
            lora_rank=rank,
            lora_alpha=alpha,
            session_id=session_id,
            is_lora=is_lora,
            sdk_version=str(self.sessions.get(session_id, {}).get("payload", {}).get("sdk_version") or ""),
            user_metadata=payload.get("user_metadata") or {},
        )
        self.models[model_id] = record
        self.planner.add_stream(ModelStream(model_id, tenant, slot))
        promise = self.promises.create(model_id, tenant)
        if None not in dedup_key:
            self._model_by_seq[dedup_key] = (promise.request_id, model_id)
        self._spawn(self._run_create_model(promise.request_id, record))
        return promise.request_id, model_id

    async def _run_create_model(self, request_id: str, record: ModelRecord) -> None:
        try:
            async with self._backend_lock:
                await self.backend.load_slot(record.slot, record.lora_rank, record.lora_alpha)
        except Exception as error:  # noqa: BLE001
            self._forget_model(record)
            self.free_slots.add(record.slot)
            self.promises.fail(request_id, str(error), "server")
            return
        self.promises.resolve(request_id, {"kind": "create_model", "model_id": record.model_id})

    def get_model(self, tenant: str, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            raise UserInputError(f"unknown model {model_id!r}")
        if record.tenant != tenant:
            raise OwnershipError(f"model {model_id} does not belong to this tenant")
        return record

    def unload_model(self, tenant: str, model_id: str) -> str:
        """Free the slot behind a promise; queued commands fail typed."""
        record = self.get_model(tenant, model_id)
        promise = self.promises.create(model_id, tenant)

        async def _unload() -> None:
            try:
                await self._release_model(record, "model unloaded")
            except Exception as error:  # noqa: BLE001
                logger.exception(f"unload of {model_id} failed")
                self.promises.fail(promise.request_id, f"{type(error).__name__}: {error}", "server")
                return
            self.promises.resolve(promise.request_id, {"kind": "unload_model", "model_id": model_id})

        self._spawn(_unload())
        return promise.request_id

    def training_run(self, tenant: str, model_id: str) -> dict:
        record = self.get_model(tenant, model_id)
        return {
            "training_run_id": record.model_id,
            "base_model": record.base_model,
            "model_owner": record.tenant,
            "is_lora": record.is_lora,
            "lora_rank": record.lora_rank if record.is_lora else None,
            "user_metadata": record.user_metadata,
            "created_at": record.created_at,
            "last_request_at": record.last_request_at,
            "corrupted": False,
        }

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # -------- command plane --------

    def submit(self, tenant: str, kind: str, payload: dict) -> str:
        """payload is server-decoded; content errors here are admission
        rejections and fail the promise (the SDK sees RequestFailedError)."""
        model_id = payload["model_id"]
        record = self.get_model(tenant, model_id)
        record.last_request_at = time.time()
        seq_id = payload["seq_id"]
        stream = self.planner.stream(model_id)

        # idempotency: the SDK resends the same seq_id after timeouts/410;
        # re-executing forward_backward would double-accumulate gradients
        if seq_id in stream.request_id_by_seq:
            return stream.request_id_by_seq[seq_id]

        promise = self.promises.create(model_id, tenant)
        stream.request_id_by_seq[seq_id] = promise.request_id
        try:
            self._admit(kind, payload)
        except UserInputError as error:
            self.promises.fail(promise.request_id, str(error), "user")
            # the rejected command still consumes its seq position, or the
            # stream would wait for it forever
            payload = {**payload, "rows": []}

        self._arrival_counter += 1
        stream.submit(
            Command(
                model_id=model_id,
                seq_id=seq_id,
                kind=kind,
                payload=payload,
                request_id=promise.request_id,
                arrival=self._arrival_counter,
            )
        )
        if kind in ("forward_backward", "forward_only"):
            self._last_submit_at[model_id] = time.monotonic()
        self._wake.set()
        return promise.request_id

    def _admit(self, kind: str, payload: dict) -> None:
        if kind not in ("forward_backward", "forward_only"):
            return
        rows = payload["rows"]
        if not rows:
            raise UserInputError("forward_backward with no data")
        if len(rows) > self.config.max_datums_per_request:
            raise UserInputError(
                f"{len(rows)} datums exceeds max_datums_per_request={self.config.max_datums_per_request}"
            )
        total_tokens = 0
        for index, row in enumerate(rows):
            if len(row["tokens"]) > self.config.max_tokens_per_datum:
                raise UserInputError(
                    f"datum {index}: {len(row['tokens'])} tokens exceeds {self.config.max_tokens_per_datum}"
                )
            total_tokens += len(row["tokens"])
        if total_tokens > self.config.max_tokens_per_request:
            raise UserInputError(
                f"{total_tokens} tokens exceeds max_tokens_per_request={self.config.max_tokens_per_request}"
            )
        if kind == "forward_backward" and self.backend.routing_replay and not any("routing_handle" in r for r in rows):
            logger.warning(f"R3 enabled but no datum carries routing_handle ({len(rows)} datums in this request)")

    def retrieve(self, tenant: str, request_id: str) -> Promise | None:
        """None -> the HTTP layer answers 410."""
        return self.promises.get(request_id, tenant)

    # -------- sampling plane (promise-based but never queues) --------

    def create_sampling_session(self, tenant: str, payload: dict) -> str:
        base_model = payload.get("base_model")
        model_path = payload.get("model_path")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        if model_path:
            self._resolve_sampler(tenant, model_path)  # unknown paths are rejected up front
        return self._mint_sampling_session(tenant, payload.get("session_id") or "", model_path)

    def _mint_sampling_session(self, tenant: str, session_id: str, model_path: str | None) -> str:
        sampling_session_id = f"sampling-{uuid.uuid4().hex}"
        self.sampling_sessions[sampling_session_id] = {
            "tenant": tenant,
            "session_id": session_id,
            "base_model": self.config.base_model,
            "model_path": model_path,
        }
        return sampling_session_id

    def get_sampler(self, tenant: str, sampling_session_id: str) -> dict:
        session = self.sampling_sessions.get(sampling_session_id)
        if session is None:
            raise UserInputError(f"unknown sampling session {sampling_session_id!r}")
        if session["tenant"] != tenant:
            raise OwnershipError("sampling session does not belong to this tenant")
        return {
            "sampler_id": sampling_session_id,
            "base_model": session["base_model"],
            "model_path": session["model_path"],
        }

    def submit_sample(self, tenant: str, payload: dict) -> tuple[str, list[str]]:
        sampling_session_id = payload.get("sampling_session_id")
        dedup_key = (sampling_session_id, payload.get("seq_id"))
        if None not in dedup_key and dedup_key in self._sample_by_seq:
            return self._sample_by_seq[dedup_key]  # a 429/timeout retry of the same seq
        model_path = payload.get("model_path")
        if sampling_session_id:
            session = self.get_sampler(tenant, sampling_session_id)
            model_path = model_path or session["model_path"]
        lora_name = self._resolve_sampler(tenant, model_path) if model_path else None
        promise = self.promises.create(model_path or "base", tenant)
        sequence_ids = [f"seq-{uuid.uuid4().hex}" for _ in range(payload.get("num_samples", 1))]
        payload = {**payload, "tenant": tenant}  # R3 caches routing per tenant
        task = asyncio.create_task(self._run_sample(promise.request_id, payload, lora_name))
        self._sample_tasks[promise.request_id] = (task, tenant, sampling_session_id, lora_name, dedup_key)
        task.add_done_callback(lambda _t, rid=promise.request_id: self._sample_done(rid))
        reply = (promise.request_id, sequence_ids)
        if None not in dedup_key:
            self._sample_by_seq[dedup_key] = reply
        return reply

    def _sample_done(self, request_id: str) -> None:
        entry = self._sample_tasks.pop(request_id, None)
        if entry is None:
            return
        if self._sample_by_seq.get(entry[4], (None,))[0] == request_id:
            del self._sample_by_seq[entry[4]]  # a later resubmit of this seq samples afresh (harmless)
        promise = self.promises.get(request_id, entry[1])
        if promise is not None and promise.state == PENDING:
            self.promises.fail(request_id, "cancelled", "user")  # cancelled before the task ever ran

    async def _run_sample(self, request_id: str, payload: dict, lora_name: str | None) -> None:
        try:
            async with self._sample_slots:
                result = await self.backend.sample(payload, lora_name)
            self.promises.resolve(request_id, {"kind": "sample", **result})
        except asyncio.CancelledError:
            self.promises.fail(request_id, "cancelled", "user")
        except UserInputError as error:
            self.promises.fail(request_id, str(error), "user")
        except Exception as error:  # noqa: BLE001
            logger.exception("sample failed")
            self.promises.fail(request_id, f"{type(error).__name__}: {error}", "server")

    def cancel(self, tenant: str, request_id: str) -> None:
        """Cancel an in-flight sampling promise; training commands have no
        cancel in the protocol and are ignored."""
        if self.promises.get(request_id, tenant) is None:
            return
        entry = self._sample_tasks.get(request_id)
        if entry is not None:
            entry[0].cancel()

    def _resolve_sampler(self, tenant: str, model_path: str) -> str | None:
        """Sampler path -> adapter name; None means the engines' own weights
        (FFT serves only the latest pushed version, see docs)."""
        model_id, kind, name = _parse_tinker_path(model_path)
        record = self.get_model(tenant, model_id)
        if kind != "sampler_weights":
            raise UserInputError(f"cannot sample from {model_path!r}: not a sampler_weights path")
        version = record.sampler_names.get(name)
        if version is None:
            raise UserInputError(f"unknown sampler version {name!r} for {model_id}")
        if not record.is_lora:
            return None
        if version not in record.loaded_versions:
            raise UserInputError(
                f"sampler version {name!r} of {model_id} was retired "
                f"(the gateway keeps the last {self.config.sampler_versions_to_keep} versions)"
            )
        return f"{model_id}@{version}"

    def _versions_in_flight(self, model_id: str) -> set[int]:
        prefix = f"{model_id}@"
        return {
            int(lora_name.removeprefix(prefix))
            for _task, _tenant, _sid, lora_name, _key in self._sample_tasks.values()
            if lora_name is not None and lora_name.startswith(prefix)
        }

    # -------- leases --------

    async def end_session(self, session_id: str) -> None:
        """The client announced its exit (SESSION_END telemetry): reclaim the
        session's models and slots now instead of after the lease timeout."""
        session = self.sessions.get(session_id)
        if session is None:
            return
        session["last_heartbeat"] = float("-inf")
        await self._sweep_once()

    async def sweep_leases(self) -> None:
        """Reclaim from stale sessions: cancel sampling, unload models, free
        slots. Training state dies with the lease; only checkpoints survive."""
        while True:
            await asyncio.sleep(30)
            await self._sweep_once()
            self._sweep_backend()

    def _sweep_backend(self) -> None:
        try:
            self.backend.sweep()
        except Exception as error:  # noqa: BLE001  housekeeping never stops the lease sweep
            logger.warning(f"backend sweep failed: {type(error).__name__}: {error}")

    async def _sweep_once(self) -> None:
        now = time.monotonic()
        stale_sessions = {
            session_id
            for session_id, session in self.sessions.items()
            if now - session["last_heartbeat"] >= self.config.lease_timeout_s
        }
        fresh_tenants = {s["tenant"] for sid, s in self.sessions.items() if sid not in stale_sessions}

        def is_stale(session_id: str, tenant: str) -> bool:
            if session_id in self.sessions:
                return session_id in stale_sessions
            # no session of its own: falls back to the tenant's freshest session
            return bool(self.sessions) and tenant not in fresh_tenants

        for sampling_session_id, session in list(self.sampling_sessions.items()):
            if is_stale(session["session_id"], session["tenant"]):
                del self.sampling_sessions[sampling_session_id]
        for request_id, (task, tenant, sampling_session_id, _lora, _key) in list(self._sample_tasks.items()):
            orphaned = sampling_session_id is not None and sampling_session_id not in self.sampling_sessions
            if orphaned or is_stale("", tenant):
                logger.warning(f"lease expired for sample {request_id}; cancelling")
                task.cancel()
        for record in list(self.models.values()):
            if is_stale(record.session_id, record.tenant):
                logger.warning(f"lease expired for {record.model_id}; freeing slot {record.slot}")
                await self._release_model(record, "lease expired")
        for session_id in stale_sessions:
            del self.sessions[session_id]

    async def _release_model(self, record: ModelRecord, reason: str) -> None:
        """Forget the model, fail what it still owes, and give the slot and
        every resident adapter version back."""
        if record.model_id not in self.models:
            return
        stream = self.planner.stream(record.model_id)
        self._forget_model(record)
        for request_id in stream.request_id_by_seq.values():
            promise = self.promises.get(request_id, record.tenant)
            if promise is not None and promise.state == PENDING:
                self.promises.fail(request_id, reason, "user")
        for sampling_session_id, session in list(self.sampling_sessions.items()):
            if _path_model_id(session["model_path"]) == record.model_id:
                del self.sampling_sessions[sampling_session_id]
        try:
            async with self._backend_lock:
                for version in sorted(record.loaded_versions):
                    await self._unload_version(record, version)
                await self.backend.unload_slot(record.slot)
        finally:
            # The record is already forgotten, so the slot must come back even when the
            # trainer is unreachable (a dead actor group after a failed unit): leaking it
            # would leave a single-tenant gateway answering "serves one model at a time"
            # to every later client for the rest of the run.
            self.free_slots.add(record.slot)

    def _forget_model(self, record: ModelRecord) -> None:
        self.models.pop(record.model_id, None)
        self.planner.remove_stream(record.model_id)
        self._last_submit_at.pop(record.model_id, None)
        for key, (_request_id, model_id) in list(self._model_by_seq.items()):
            if model_id == record.model_id:
                del self._model_by_seq[key]

    async def _unload_version(self, record: ModelRecord, version: int) -> bool:
        try:
            await self.backend.unload_adapter(f"{record.model_id}@{version}")
        except Exception as error:  # noqa: BLE001  stays resident; retried at the next push
            logger.warning(f"unload_adapter {record.model_id}@{version} failed: {error}")
            return False
        record.loaded_versions.discard(version)
        return True

    # -------- dispatch loop --------

    async def run(self) -> None:
        await self._warmup()
        self._sweep_task = asyncio.create_task(self.sweep_leases())
        while True:
            unit = self.planner.plan_unit()
            if unit is None:
                await self._wait_wake(None)
                continue
            if isinstance(unit, WorkUnit):
                hold = self._coalesce_hold(unit)
                if hold > 0:
                    await self._wait_wake(hold)
                    continue
                self.planner.commit(unit)
            async with self._backend_lock:
                if isinstance(unit, WorkUnit):
                    await self._run_work(unit)
                else:
                    await self._run_barrier(unit)

    async def _warmup(self) -> None:
        try:
            async with self._backend_lock:
                await self.backend.warmup()
        except Exception as error:  # noqa: BLE001  serve anyway; the first push makes engines visible
            logger.warning(f"backend warmup failed: {type(error).__name__}: {error}")

    async def _wait_wake(self, timeout: float | None) -> None:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
        except TimeoutError:
            pass
        self._wake.clear()

    def _coalesce_hold(self, unit: WorkUnit) -> float:
        """Seconds to keep an under-filled unit open for more chunks of the
        same window; 0 runs it now. Barriers never come through here."""
        if unit.num_tokens >= self.config.unit_token_budget:
            return 0.0
        growing = {ref.stream for ref in unit.rows if ref.stream.window_may_grow()}
        if not growing:
            return 0.0
        now = time.monotonic()
        last_submit = max(self._last_submit_at.get(stream.model_id, 0.0) for stream in growing)
        first_ready = min(ref.request.command.submitted_at for ref in unit.rows)
        remaining = min(
            self.config.coalesce_wait_s - (now - last_submit),
            self.config.coalesce_max_wait_s - (now - first_ready),
        )
        return max(0.0, remaining)

    async def _run_work(self, unit: WorkUnit) -> None:
        # slot-contiguous order; outputs come back aligned to it
        refs = sorted(unit.rows, key=lambda ref: ref.stream.slot)
        # rows are tagged with their tenant (R3 routing is cached per tenant); shallow copies
        slot_rows = [(ref.stream.slot, {**ref.row, "tenant": ref.stream.tenant}) for ref in refs]
        self._unit_counter += 1
        run = self.backend.forward_backward if unit.kind == "forward_backward" else self.backend.forward_only
        poison = unit.kind == "forward_backward"
        try:
            outputs = await run(self._unit_counter, slot_rows, unit.loss_fn, unit.loss_fn_config)
        except UserInputError as error:
            self._fail_riders(refs, str(error), "user", poison=poison)
            return
        except Exception as error:  # noqa: BLE001  infra failure: fail the riders, keep serving
            logger.exception(f"{unit.kind} unit {self._unit_counter} failed")
            self._fail_riders(refs, f"{type(error).__name__}: {error}", "server", poison=poison)
            return

        if len(outputs) != len(refs):  # a backend contract breach fails the riders, never the loop
            self._fail_riders(
                refs, f"unit returned {len(outputs)} outputs for {len(refs)} rows", "server", poison=poison
            )
            return
        for ref, output in zip(refs, outputs, strict=True):
            request = ref.request
            request.outputs[ref.local_index] = output
            request.remaining -= 1
            if request.remaining == 0:
                self.promises.resolve(
                    request.command.request_id, {"kind": request.command.kind, "outputs": request.outputs}
                )
                ref.stream.finish(request)

    def _fail_riders(self, refs, error: str, category: str, poison: bool) -> None:
        seen: set[int] = set()
        for ref in refs:
            if poison:
                ref.stream.dirty = error
            if id(ref.request) in seen:
                continue
            seen.add(id(ref.request))
            self.promises.fail(ref.request.command.request_id, error, category)
            ref.stream.finish(ref.request)

    async def _run_barrier(self, unit: BarrierUnit) -> None:
        if unit.kind == "optim_step":
            unit = await self._discard_dirty(unit)
            if not unit.entries:
                return
        try:
            results = await self._execute_barrier(unit)
        except (UserInputError, OwnershipError) as error:
            self._fail_barrier(unit, str(error), "user")
            return
        except Exception as error:  # noqa: BLE001
            logger.exception(f"{unit.kind} barrier failed")
            self._fail_barrier(unit, f"{type(error).__name__}: {error}", "server")
            return
        for (stream, pending), result in zip(unit.entries, results, strict=True):
            self.promises.resolve(pending.command.request_id, result)
            stream.finish(pending)

    async def _discard_dirty(self, unit: BarrierUnit) -> BarrierUnit:
        """A poisoned window must not step: drop its gradients and fail the
        optim_step typed; the clean entries proceed."""
        clean = []
        for stream, pending in unit.entries:
            if stream.dirty is None:
                clean.append((stream, pending))
                continue
            try:
                await self.backend.discard_grads(stream.slot)
            except Exception as error:  # noqa: BLE001
                logger.exception(f"discard_grads for slot {stream.slot} failed: {error}")
            self.promises.fail(
                pending.command.request_id,
                f"a forward_backward in this gradient window failed ({stream.dirty}); gradients discarded",
                "user",
            )
            stream.dirty = None
            stream.finish(pending)
        return BarrierUnit(kind=unit.kind, entries=clean)

    def _fail_barrier(self, unit: BarrierUnit, error: str, category: str) -> None:
        for stream, pending in unit.entries:
            self.promises.fail(pending.command.request_id, error, category)
            stream.finish(pending)

    async def _execute_barrier(self, unit: BarrierUnit) -> list[dict]:
        if unit.kind == "optim_step":
            metrics_by_slot = await self.backend.optim_step(
                {stream.slot: pending.command.payload["adam_params"] for stream, pending in unit.entries}
            )
            return [
                {"kind": "optim_step", "metrics": _plain_metrics(metrics_by_slot[stream.slot])}
                for stream, _ in unit.entries
            ]

        ((stream, pending),) = unit.entries
        record = self.models[stream.model_id]
        payload = pending.command.payload
        if unit.kind == "save_state":
            name = payload["name"] or f"checkpoint-{pending.command.seq_id:06d}"
            path = self._checkpoint_dir(record.model_id, "weights", name)
            await self.backend.save_slot(record.slot, path)
            write_sidecar(path, sidecar_meta(record, self.config, "weights", name))
            return [{"kind": "save_state", "path": f"tinker://{record.model_id}/weights/{name}"}]
        if unit.kind == "load_state":
            # a raise here fails this promise only (_run_barrier); the backend leaves the slot freshly
            # initialized, so the model stays usable and its window keeps whatever dirt it had
            await self._load_state(record, payload)
            stream.dirty = None
            return [{"kind": "load_state"}]
        if unit.kind == "save_weights_for_sampler":
            return [await self._publish_sampler(record, payload)]
        raise UserInputError(f"unknown barrier kind {unit.kind!r}")

    async def _load_state(self, record: ModelRecord, payload: dict) -> None:
        path, meta = self._resolve_checkpoint(record, payload["path"])
        if record.is_lora and (record.lora_rank, record.lora_alpha) != (
            int(meta["lora_rank"]),
            float(meta["lora_alpha"]),
        ):
            logger.info(f"{record.model_id}: taking rank/alpha {meta['lora_rank']}/{meta['lora_alpha']} from {path}")
            record.lora_rank, record.lora_alpha = int(meta["lora_rank"]), float(meta["lora_alpha"])
        await self.backend.load_slot(
            record.slot,
            record.lora_rank,
            record.lora_alpha,
            ckpt_path=path,
            load_optimizer=payload["optimizer"],
        )

    def _resolve_checkpoint(self, record: ModelRecord, tinker_path: str) -> tuple[str, dict]:
        path, meta = self._checkpoint_meta(record.tenant, tinker_path)
        if bool(meta.get("is_lora", True)) != record.is_lora:
            raise UserInputError(f"checkpoint {tinker_path} was trained in mode {meta.get('train_mode')!r}")
        return path, meta

    def _checkpoint_meta(self, tenant: str, tinker_path: str) -> tuple[str, dict]:
        """Checkpoint dir + its sidecar, ownership checked; the sidecar makes
        resume work after a gateway restart, a live record fills in for
        checkpoints saved before sidecars existed."""
        model_id, kind, name = _parse_tinker_path(tinker_path)
        path = self._checkpoint_dir(model_id, kind, name)
        meta = read_sidecar(path)
        if meta is None:
            record = self.models.get(model_id)
            if record is None or not os.path.isdir(path):
                raise UserInputError(f"unknown checkpoint {tinker_path!r}")
            meta = sidecar_meta(record, self.config, kind, name)
        if meta.get("tenant") != tenant:
            raise OwnershipError(f"checkpoint {tinker_path} does not belong to this tenant")
        return path, meta

    async def _publish_sampler(self, record: ModelRecord, payload: dict) -> dict:
        requested_name = payload.get("name") or None
        version_hint = record.sampler_version + 1
        if record.is_lora:
            version = version_hint
            name = requested_name or str(version)
            await self.backend.save_slot(record.slot, self._checkpoint_dir(record.model_id, "sampler_weights", name))
            await self.backend.push_slot(
                record.slot, f"{record.model_id}@{version}", record.lora_rank, record.lora_alpha
            )
            record.loaded_versions.add(version)
        else:
            # FFT: the push syncs the base weights; the engine weight version is the sampler version
            pushed = await self.backend.push_slot(
                record.slot, f"{record.model_id}@{version_hint}", record.lora_rank, record.lora_alpha
            )
            if pushed is None:
                logger.warning("push_slot returned no weight version in full mode; counting locally")
            version = int(pushed) if pushed is not None else version_hint
            name = requested_name or str(version)
        record.sampler_version = version
        record.sampler_names[name] = version
        write_sidecar(
            self._checkpoint_dir(record.model_id, "sampler_weights", name),
            sidecar_meta(record, self.config, "sampler_weights", name),
        )
        if record.is_lora:
            await self._retire_sampler_versions(record)
        tinker_path = f"tinker://{record.model_id}/sampler_weights/{name}"
        result = {"kind": "save_weights_for_sampler", "path": tinker_path, "sampling_session_id": None}
        if requested_name is None:
            result["sampling_session_id"] = self._mint_sampling_session(record.tenant, record.session_id, tinker_path)
        return result

    async def _retire_sampler_versions(self, record: ModelRecord) -> None:
        """Keep the newest K adapter versions resident; older ones leave the
        engines and the disk unless a sample is still running on them."""
        cutoff = record.sampler_version - max(1, self.config.sampler_versions_to_keep) + 1
        busy = self._versions_in_flight(record.model_id)
        for version in sorted(record.loaded_versions):
            if version >= cutoff or version in busy:
                continue
            if await self._unload_version(record, version):
                for name, named_version in record.sampler_names.items():
                    if named_version == version:
                        shutil.rmtree(
                            self._checkpoint_dir(record.model_id, "sampler_weights", name), ignore_errors=True
                        )

    # -------- checkpoint metadata --------

    def weights_info(self, tenant: str, tinker_path: str) -> dict:
        _path, meta = self._checkpoint_meta(tenant, tinker_path)
        return {
            "base_model": meta["base_model"],
            # both SDKs assert is_lora on resume; an FFT checkpoint reports rank 0 (documented)
            "is_lora": True,
            "lora_rank": int(meta.get("lora_rank") or 0),
            "train_unembed": True,
            "train_mlp": True,
            "train_attn": True,
            "train_mode": meta.get("train_mode", "lora"),
        }

    def delete_checkpoint(self, tenant: str, model_id: str, checkpoint_id: str) -> None:
        """Tolerant: a missing checkpoint is already deleted."""
        kind, _, name = checkpoint_id.partition("/")
        if kind not in ("weights", "sampler_weights") or not name:
            raise UserInputError(f"malformed checkpoint id {checkpoint_id!r}")
        path = self._checkpoint_dir(model_id, kind, name)
        meta = read_sidecar(path)
        owner = meta.get("tenant") if meta is not None else getattr(self.models.get(model_id), "tenant", tenant)
        if owner != tenant:
            raise OwnershipError(f"checkpoint {checkpoint_id} does not belong to this tenant")
        shutil.rmtree(path, ignore_errors=True)

    def _checkpoint_dir(self, model_id: str, kind: str, name: str) -> str:
        for part in (model_id, kind, name):
            _check_path_component(part)
        return f"{self.config.checkpoint_root}/{model_id}/{kind}/{name}"


# -------- pure helpers --------


def _resolve_lora_config(config: GatewayConfig, lora_config: dict | None) -> tuple[bool, int, float]:
    """(is_lora, rank, alpha) for a create_model request in this gateway's mode."""
    if not config.is_lora:
        if lora_config:
            logger.info(f"train_mode=full: ignoring lora_config {lora_config}")
        return False, 0, 0.0
    rank = int((lora_config or {}).get("rank") or 0)
    if rank <= 0:
        raise UserInputError("this gateway serves LoRA; lora_config.rank must be > 0")
    alpha = config.lora_alpha if config.lora_alpha is not None else float(2 * rank)
    return True, rank, alpha


def _no_slot_message(config: GatewayConfig) -> str:
    if not config.is_lora:
        return "this full fine-tuning gateway serves one model at a time"
    return f"no free adapter slots (capacity {config.n_slots})"


def _plain_metrics(metrics) -> dict:
    """Wire-safe optim metrics: the SDK types them as dict[str, float] and JSON has
    no NaN, so a vetoed step's non-finite grad norm becomes 0.0 (its ``skipped``
    flag carries the meaning)."""
    if not isinstance(metrics, dict):
        metrics = {"grad_norm": metrics}
    return {
        key: (_finite_float(value) if isinstance(value, (int, float)) else value) for key, value in metrics.items()
    }


def _finite_float(value) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def sidecar_meta(record: ModelRecord, config: GatewayConfig, kind: str, name: str) -> dict:
    return {
        "model_id": record.model_id,
        "tenant": record.tenant,
        "session_id": record.session_id,
        "base_model": record.base_model,
        "is_lora": record.is_lora,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "slot": record.slot,
        "kind": kind,
        "name": name,
        "sampler_version": record.sampler_version,
        "created_at": time.time(),
        "train_mode": config.train_mode,
    }


def write_sidecar(path: str, meta: dict) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, SIDECAR_NAME), "w") as handle:
        json.dump(meta, handle, indent=2)


def read_sidecar(path: str) -> dict | None:
    try:
        with open(os.path.join(path, SIDECAR_NAME)) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _check_path_component(part: str) -> None:
    """Client-chosen names and ids become directory names under checkpoint_root."""
    if not part or part in (".", "..") or "/" in part or "\\" in part:
        raise UserInputError(f"invalid checkpoint path component {part!r}")


def _path_model_id(tinker_path: str | None) -> str | None:
    if not tinker_path or not tinker_path.startswith("tinker://"):
        return None
    return tinker_path.removeprefix("tinker://").split("/")[0]


def _parse_tinker_path(path: str) -> tuple[str, str, str]:
    if not path.startswith("tinker://"):
        raise UserInputError(f"not a tinker path: {path!r}")
    parts = path.removeprefix("tinker://").split("/")
    if len(parts) != 3 or parts[1] not in ("weights", "sampler_weights"):
        raise UserInputError(f"malformed tinker path: {path!r}")
    return parts[0], parts[1], parts[2]
