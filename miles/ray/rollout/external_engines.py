"""Attached external SGLang engines (``--rollout-external``).

The engines are launched and restarted by someone else (a deployment, an engine pool). miles
only observes them: this provider reports one cell per ``--rollout-external-engine-addrs``
entry, alive while the engine answers ``/health``, and the rollout cells then run their normal
pending-weights -> serving lifecycle against the given address. Nothing here launches, stops or
kills an engine; a dead engine's cell is removed (deregistered from the router) and a recovered
one is re-added as a new generation, which makes it take a full weight sync before it serves.

Kept apart from the Ray worker path so the external path stays one module plus a few branches.
"""

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import httpx

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.ray.specs.inference import compute_engine_pool_id
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.lora import lora_rollout_enabled
from miles.utils.misc import cancel_and_await_task
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.rollout_external_args import ExternalEngineAddr, parse_external_engine_addr
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0
PROBE_TIMEOUT_SECONDS = 5.0
# consecutive failed probes before a serving engine counts as down (one blip must not cost a resync)
DOWN_AFTER_FAILED_PROBES = 3
# the sglang default of an engine no weight sync ever versioned
_UNSET_WEIGHT_VERSIONS = (None, "default")


class ExternalEngineProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        addrs: list[ExternalEngineAddr],
        cell_meta: dict[str, Any],
        api_key: str | None,
        poll_interval_seconds: float,
        down_after_failed_probes: int,
    ):
        self._addrs = addrs
        self._cell_meta = cell_meta
        self._api_key = api_key
        self._poll_interval_seconds = poll_interval_seconds
        self._down_after_failed_probes = down_after_failed_probes
        self._pool_id = compute_engine_pool_id(model_idx=0, group_index=0)

    @classmethod
    def create(cls, args) -> "ExternalEngineProvider":
        return cls(
            addrs=[parse_external_engine_addr(addr) for addr in args.rollout_external_engine_addrs],
            cell_meta=compute_external_engine_cell_meta(args),
            api_key=args.sglang_api_key,
            poll_interval_seconds=POLL_INTERVAL_SECONDS,
            down_after_failed_probes=DOWN_AFTER_FAILED_PROBES,
        )

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        addr = self._addrs[self._cell_index_of_worker(worker_name)]
        return {"primary": HostAndPort(host=addr.host, port=addr.port)}

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # no miles-side worker exists for an external engine
        return [[] for _ in cell_ids]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        liveness = [_EngineLiveness() for _ in self._addrs]
        # the initial sync must complete before the watch is considered established
        await self._poll_once(reconcile, liveness=liveness)
        task = asyncio.create_task(self._watch_loop(reconcile, liveness=liveness))
        return partial(cancel_and_await_task, task)

    async def _watch_loop(self, reconcile: ReconcileFn, *, liveness: list["_EngineLiveness"]) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_seconds)
            try:
                await self._poll_once(reconcile, liveness=liveness)
            except Exception:
                logger.exception("External engine poll failed; retrying")

    async def _poll_once(self, reconcile: ReconcileFn, *, liveness: list["_EngineLiveness"]) -> None:
        healthy = await asyncio.gather(*[self._probe(addr) for addr in self._addrs])
        for index, (engine, ok) in enumerate(zip(liveness, healthy, strict=True)):
            alive = engine.observe(ok=ok, down_after_failed_probes=self._down_after_failed_probes)
            if alive == engine.alive:
                continue
            cell_id = compute_cell_id(pool_id=self._pool_id, cell_index=index)
            generation = engine.generation + 1 if alive else engine.generation
            if alive:
                logger.info(f"External engine {self._url(index)} is up (cell {cell_id}, generation {generation})")
            else:
                logger.error(
                    f"External engine {self._url(index)} is down (cell {cell_id}); deregistering it, "
                    f"restarting it is up to whoever launched it"
                )
            try:
                await reconcile(cell_id, self._cell_info(index, generation=generation) if alive else None)
            except Exception:
                logger.exception(f"Reconciling external cell {cell_id} failed; retrying next poll")
                continue
            engine.alive, engine.generation = alive, generation

    async def _probe(self, addr: ExternalEngineAddr) -> bool:
        try:
            response = await GeneralHttpClientProvider.client().get(
                f"{build_server_url(host=addr.host, port=addr.port)}/health",
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else None,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
            return response.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def _cell_info(self, index: int, *, generation: int) -> CellInfo:
        num_gpus_per_engine = self._cell_meta["num_gpus_per_engine"]
        return CellInfo(
            cell_id=compute_cell_id(pool_id=self._pool_id, cell_index=index),
            pool_id=self._pool_id,
            alive=True,
            worker_names=[compute_worker_name(pool_id=self._pool_id, cell_index=index)],
            # a new generation is a new hash: a recovered engine reconnects and resyncs like a restarted cell
            workers_hash=f"external:{self._url(index)}:gen{generation}",
            meta={
                **self._cell_meta,
                "gpu_offset": index * num_gpus_per_engine,
                "external_server_url": self._url(index),
            },
        )

    def _url(self, index: int) -> str:
        addr = self._addrs[index]
        return build_server_url(host=addr.host, port=addr.port)

    def _cell_index_of_worker(self, worker_name: str) -> int:
        for index in range(len(self._addrs)):
            if worker_name == compute_worker_name(pool_id=self._pool_id, cell_index=index):
                return index
        raise KeyError(f"{worker_name} is not an external engine worker of {self._pool_id}")


@dataclass
class _EngineLiveness:
    """What the provider last reported for one engine; mutated only by its poll loop."""

    alive: bool = False
    failed_probes: int = 0
    # bumped on every up transition so each comeback is a new workers_hash
    generation: int = 0

    def observe(self, *, ok: bool, down_after_failed_probes: int) -> bool:
        """Whether the engine should now count as alive."""
        self.failed_probes = 0 if ok else self.failed_probes + 1
        return ok or (self.alive and self.failed_probes < down_after_failed_probes)


def compute_external_engine_cell_meta(args) -> dict[str, Any]:
    """The cell meta the inference engine spec would carry, minus the per-cell fields."""
    [model_cfg] = resolve_sglang_config(args).models
    [server_group_config] = model_cfg.server_groups
    return dict(
        model_id=model_cfg.name,
        worker_type=server_group_config.worker_type,
        num_gpus_per_engine=server_group_config.num_gpus_per_engine,
        sglang_api_key=args.sglang_api_key,
        needs_offload=False,
        update_weights=model_cfg.update_weights,
    )


# ------------------------------ server args check ------------------------------


def check_external_engine_server_info(
    args, *, server_info: dict, weight_version: str | None, num_gpus_per_engine: int, server_url: str
) -> None:
    """Check the launch flags miles would have chosen against what the engine reports in /server_info.

    Raises on a mismatch that breaks rollout or weight sync; warns on ones that are only
    suspicious (a host may mount the checkpoint at another path)."""
    errors = [
        f"{key}={server_info.get(key)!r}, expected {expected!r} ({why})"
        for key, expected, why in _compute_required_server_args(args, num_gpus_per_engine=num_gpus_per_engine)
        if key in server_info and server_info[key] != expected
    ]
    if errors:
        raise ValueError(f"External engine {server_url} was launched with incompatible flags: " + "; ".join(errors))

    if (model_path := server_info.get("model_path")) != args.hf_checkpoint:
        logger.warning(
            f"External engine {server_url} serves model_path={model_path!r}, not --hf-checkpoint "
            f"{args.hf_checkpoint!r}; it must be the same checkpoint (disk-delta patches its bytes)"
        )
    if weight_version not in _UNSET_WEIGHT_VERSIONS:
        logger.warning(
            f"External engine {server_url} reports weight_version={weight_version!r}: it may serve another "
            f"policy than --hf-checkpoint until the first published weight update"
        )


def _compute_required_server_args(args, *, num_gpus_per_engine: int) -> list[tuple[str, Any, str]]:
    required = [
        ("tp_size", num_gpus_per_engine, "the broadcast NCCL rank layout assumes one rank per engine GPU"),
        ("pp_size", args.sglang_pp_size, "--sglang-pp-size"),
        ("dp_size", args.sglang_dp_size, "--sglang-dp-size"),
    ]
    if lora_rollout_enabled(args) or is_multi_lora_enabled(args):
        required.append(("enable_lora", True, "LoRA weight sync loads adapters"))
    if args.use_rollout_routing_replay:
        required.append(("enable_return_routed_experts", True, "--use-rollout-routing-replay"))
    if args.use_rollout_indexer_replay:
        required.append(("enable_return_indexer_topk", True, "--use-rollout-indexer-replay"))
    if args.fp16:
        required.append(("dtype", "float16", "--fp16"))
    return required
