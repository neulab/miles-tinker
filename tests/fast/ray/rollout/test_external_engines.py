"""Attached external SGLang engines (``--rollout-external``): fake engines are plain HTTP servers
miles did not launch, so every lifecycle step here runs against real sockets."""

from __future__ import annotations

import asyncio
import logging
import time
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import external_engines as external_engines_module
from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout.external_engines import ExternalEngineProvider, check_external_engine_server_info
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.specs.inference import specs_inference_engine
from miles.utils.rollout_external_args import (
    ExternalEngineAddr,
    parse_external_engine_addr,
    validate_rollout_external_args,
)
from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort

_SERVER_INFO = dict(
    tp_size=1,
    pp_size=1,
    dp_size=1,
    model_path="/fake/model",
    weight_version="default",
    enable_lora=False,
)


def _external_args(engines: list[MockSGLangHttpServer], **overrides: Any) -> Namespace:
    return make_args(
        rollout_external=True,
        rollout_external_engine_addrs=[f"{engine.host}:{engine.port}" for engine in engines],
        rollout_num_gpus=len(engines),
        rollout_num_gpus_per_engine=1,
        use_miles_router=True,  # legacy router api: add/remove are one POST each, easy to assert on
        **overrides,
    )


async def _until(predicate: Callable[[], bool], *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.02)


# ------------------------------ args ------------------------------


def _validation_args(**overrides: Any) -> Namespace:
    return Namespace(
        **{
            "rollout_external": True,
            "rollout_external_engine_addrs": ["10.0.0.1:30000", "10.0.0.2:30000"],
            "rollout_num_gpus": None,
            "rollout_num_gpus_per_engine": 2,
            "colocate": False,
            "eval_num_gpus": 0,
            "update_weight_transfer_mode": "disk-delta",
            **overrides,
        }
    )


class TestArgs:
    def test_the_rollout_gpu_count_is_derived_from_the_engine_list(self):
        """Concurrency limits and metrics divide by rollout_num_gpus, so it must count the attached engines."""
        args = _validation_args()

        validate_rollout_external_args(args)

        assert args.rollout_num_gpus == 4

    def test_a_contradicting_rollout_gpu_count_is_refused(self):
        """Silently overriding it would hide a wrong engine list."""
        with pytest.raises(AssertionError, match="--rollout-num-gpus 8"):
            validate_rollout_external_args(_validation_args(rollout_num_gpus=8))

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(rollout_external_engine_addrs=None), "requires --rollout-external-engine-addrs"),
            (dict(rollout_external_engine_addrs=["10.0.0.1:30000"] * 2), "duplicates"),
            (dict(rollout_external_engine_addrs=["http://10.0.0.1:30000"]), "host:port"),
            (dict(colocate=True), "--colocate"),
            (dict(eval_num_gpus=2), "eval"),
            (dict(update_weight_transfer_mode="p2p"), "p2p"),
        ],
    )
    def test_unsupported_setups_are_refused(self, overrides, match):
        """Each of these would launch or address engines miles does not own."""
        with pytest.raises(AssertionError, match=match):
            validate_rollout_external_args(_validation_args(**overrides))

    def test_addrs_without_the_flag_are_refused(self):
        """The addresses alone used to be parsed and then ignored."""
        with pytest.raises(AssertionError, match="requires --rollout-external"):
            validate_rollout_external_args(_validation_args(rollout_external=False))

    def test_a_managed_rollout_is_untouched(self):
        args = _validation_args(rollout_external=False, rollout_external_engine_addrs=None, rollout_num_gpus=8)

        validate_rollout_external_args(args)

        assert args.rollout_num_gpus == 8

    def test_ipv6_hosts_are_unbracketed(self):
        assert parse_external_engine_addr("[fd00::1]:30000") == ExternalEngineAddr(host="fd00::1", port=30000)

    def test_external_engines_get_no_launch_spec(self):
        """miles must never launch an engine it was told to attach to."""
        assert specs_inference_engine(make_args(rollout_external=True, rollout_external_engine_addrs=["h:1"])) == []


# ------------------------------ server args check ------------------------------


class TestServerInfoCheck:
    def _check(self, args: Namespace, *, weight_version: str | None = "default", **server_info: Any) -> None:
        check_external_engine_server_info(
            args,
            server_info={**_SERVER_INFO, **server_info},
            weight_version=weight_version,
            num_gpus_per_engine=1,
            server_url="http://e:1",
        )

    def test_matching_flags_pass(self):
        self._check(make_args())

    def test_a_tp_size_that_disagrees_with_the_engine_gpu_count_is_refused(self):
        """The broadcast NCCL group assigns one rank per engine GPU; a wrong count hangs the first update."""
        with pytest.raises(ValueError, match="tp_size=2, expected 1"):
            self._check(make_args(), tp_size=2)

    def test_lora_rollout_requires_lora_on_the_engine(self):
        with pytest.raises(ValueError, match="enable_lora"):
            self._check(make_args(lora_rank=8))

    def test_routing_replay_requires_the_engine_to_return_routed_experts(self):
        with pytest.raises(ValueError, match="enable_return_routed_experts"):
            self._check(make_args(use_rollout_routing_replay=True), enable_return_routed_experts=False)

    def test_fields_an_older_engine_does_not_report_are_skipped(self):
        check_external_engine_server_info(
            make_args(), server_info={}, weight_version=None, num_gpus_per_engine=1, server_url="http://e:1"
        )

    def test_another_checkpoint_or_a_used_engine_only_warns(self, caplog):
        """A host may mount the checkpoint elsewhere; an engine reused from another run is legal but suspicious."""
        with caplog.at_level(logging.WARNING, logger=external_engines_module.__name__):
            self._check(make_args(), model_path="/elsewhere", weight_version="7")

        assert "/elsewhere" in caplog.text
        assert "weight_version='7'" in caplog.text


# ------------------------------ provider ------------------------------


class _ScriptedProvider(ExternalEngineProvider):
    """Probes answer from ``self.healthy`` instead of the network."""

    def __init__(self, num_engines: int) -> None:
        super().__init__(
            addrs=[ExternalEngineAddr(host=f"10.0.0.{i}", port=30000) for i in range(num_engines)],
            cell_meta=dict(
                model_id="default",
                worker_type="regular",
                num_gpus_per_engine=2,
                sglang_api_key=None,
                needs_offload=False,
                update_weights=True,
            ),
            api_key=None,
            poll_interval_seconds=3600,
            down_after_failed_probes=2,
        )
        self.liveness = [external_engines_module._EngineLiveness() for _ in range(num_engines)]
        self.healthy = [True] * num_engines

    async def _probe(self, addr: ExternalEngineAddr) -> bool:
        return self.healthy[self._addrs.index(addr)]

    async def poll(self, reconcile) -> None:
        await self._poll_once(reconcile, liveness=self.liveness)


class _RecordingReconcile:
    def __init__(self) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []
        self.fail = False

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        if self.fail:
            raise RuntimeError("injected reconcile failure")
        self.calls.append((cell_id, info))


@pytest.fixture
def scripted():
    provider = _ScriptedProvider(2)
    return provider, _RecordingReconcile()


class TestProvider:
    async def test_healthy_engines_are_offered_as_cells_with_their_address(self, scripted):
        provider, reconcile = scripted

        await provider.poll(reconcile)

        assert [cell_id for cell_id, _ in reconcile.calls] == ["inference-engine-0-0-0", "inference-engine-0-0-1"]
        info = reconcile.calls[1][1]
        assert info.meta["external_server_url"] == "http://10.0.0.1:30000"
        assert info.meta["gpu_offset"] == 2
        assert info.worker_names == ["inference-engine-0-0-1-0"]
        assert await provider.get_addrs("inference-engine-0-0-1-0") == {
            "primary": HostAndPort(host="10.0.0.1", port=30000)
        }

    async def test_an_engine_that_is_not_up_yet_is_not_offered(self, scripted):
        provider, reconcile = scripted
        provider.healthy[1] = False

        await provider.poll(reconcile)

        assert [cell_id for cell_id, _ in reconcile.calls] == ["inference-engine-0-0-0"]

    async def test_a_single_failed_probe_does_not_take_a_serving_engine_down(self, scripted):
        """A blip must not cost the engine a full resync."""
        provider, reconcile = scripted
        await provider.poll(reconcile)
        reconcile.calls.clear()
        provider.healthy[0] = False

        await provider.poll(reconcile)

        assert reconcile.calls == []

    async def test_a_dead_engine_is_removed_and_its_comeback_is_a_new_generation(self, scripted):
        """The new workers_hash is what makes the trainer reconnect and fully resync a restarted engine."""
        provider, reconcile = scripted
        await provider.poll(reconcile)
        first_hash = reconcile.calls[0][1].workers_hash
        provider.healthy[0] = False
        await provider.poll(reconcile)
        await provider.poll(reconcile)

        assert reconcile.calls[-1] == ("inference-engine-0-0-0", None)

        provider.healthy[0] = True
        await provider.poll(reconcile)

        cell_id, info = reconcile.calls[-1]
        assert cell_id == "inference-engine-0-0-0"
        assert info.workers_hash != first_hash

    async def test_a_failed_reconcile_is_retried_on_the_next_poll(self, scripted):
        provider, reconcile = scripted
        reconcile.fail = True
        await provider.poll(reconcile)
        reconcile.fail = False

        await provider.poll(reconcile)

        assert len(reconcile.calls) == 2


# ------------------------------ lifecycle against fake engines ------------------------------


@pytest.fixture
def fake_engines():
    engines = [MockSGLangHttpServer(response_payload=_SERVER_INFO) for _ in range(2)]
    router = MockSGLangHttpServer(response_payload={})
    yield engines, router
    for server in [*engines, router]:
        server.close()


@pytest.fixture
def fast_external_loops(monkeypatch):
    monkeypatch.setattr(external_engines_module, "POLL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(external_engines_module, "DOWN_AFTER_FAILED_PROBES", 2)
    monkeypatch.setattr(inference_controller_module, "TICK_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_INITIAL_DELAY_SECONDS", 0.05)
    monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_MAX_DELAY_SECONDS", 0.05)


def _patch_rollout_servers(
    monkeypatch, *, router: MockSGLangHttpServer, expected_num_cells: int | None = None
) -> None:
    async def _create_rollout_servers(args, *, context_lock, global_health_checker_activeness):
        return {
            "default": RolloutServer(
                server_cells={},
                args=args,
                context_lock=context_lock,
                router_ip=router.host,
                router_port=router.port,
                model_name="default",
                global_health_checker_activeness=global_health_checker_activeness,
                expected_num_cells=(
                    len(args.rollout_external_engine_addrs) if expected_num_cells is None else expected_num_cells
                ),
            )
        }

    async def _no_session_server(args) -> None:
        return None

    monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _create_rollout_servers)
    monkeypatch.setattr(inference_controller_module, "wait_session_server_ready", _no_session_server)


def _router_posts(router: MockSGLangHttpServer, path: str) -> list[str]:
    return [request.path for request in router.requests if request.path == path]


async def _sync_weights(controller: InferenceController) -> list[str]:
    engines = await controller.start_update_weights()
    await controller.end_update_weights(engines.snapshot_cell_id_to_hashes)
    return [client.server_url for client in engines.rollout_engines]


@pytest.mark.usefixtures("fast_external_loops")
class TestLifecycleAgainstFakeEngines:
    async def test_attach_sync_route_survive_a_restart_and_leave_the_engines_running(
        self, fake_engines, monkeypatch
    ):
        engines, router = fake_engines
        urls = [engine.url for engine in engines]
        _patch_rollout_servers(monkeypatch, router=router)
        controller = InferenceController(_external_args(engines))

        # attach: every engine is checked and waits for its weights, unrouted
        await controller.init()
        cells = controller.servers["default"].server_cells
        assert all(cell.is_pending_weights for cell in cells.values())
        assert router.requests == []
        for engine in engines:
            assert {"/health", "/health_generate", "/server_info", "/model_info"} <= set(engine.paths)

        # the weight sync addresses exactly the attached engines, then they are routed
        assert await _sync_weights(controller) == urls
        assert sorted(r.path for r in router.requests) == ["/add_worker", "/add_worker"]
        assert all(cell.is_serving for cell in cells.values())

        # a dead engine is deregistered, and the rest keep syncing without waiting for it
        dead_port = engines[1].port
        engines[1].close()
        await _until(lambda: len(cells) == 1)
        assert len(_router_posts(router, "/remove_worker")) == 1
        assert await _sync_weights(controller) == urls[:1]

        # its restart (by the pool, not miles) rejoins unrouted until the next sync reaches it
        engines[1] = MockSGLangHttpServer(response_payload=_SERVER_INFO, port=dead_port)
        await _until(lambda: len(cells) == 2 and all(c.is_pending_weights_or_serving for c in cells.values()))
        restarted = cells["inference-engine-0-0-1"]
        assert restarted.is_pending_weights
        assert await _sync_weights(controller) == urls
        assert restarted.is_serving
        assert len(_router_posts(router, "/add_worker")) == 3

        # shutdown deregisters and never asks an engine to do anything
        await controller.dispose()
        assert len(_router_posts(router, "/remove_worker")) == 3
        for engine in engines:
            assert {request.method for request in engine.requests} == {"GET"}

    async def test_an_engine_launched_with_incompatible_flags_never_becomes_ready(
        self, fake_engines, monkeypatch, caplog
    ):
        """Attaching it would hang or corrupt the first weight sync, so it stays out and says why."""
        engines, router = fake_engines
        engines[1].close()
        engines[1] = MockSGLangHttpServer(response_payload={**_SERVER_INFO, "tp_size": 4})
        _patch_rollout_servers(monkeypatch, router=router, expected_num_cells=0)
        controller = InferenceController(_external_args(engines))

        with caplog.at_level(logging.ERROR):
            await controller.init()
            await _until(lambda: "tp_size=4, expected 1" in caplog.text)

        cells = controller.servers["default"].server_cells
        assert cells["inference-engine-0-0-0"].is_pending_weights
        assert cells["inference-engine-0-0-1"].is_initializing
        await controller.dispose()

