"""R3 through the gateway backend: /generate asks for the routing (and only the
rows past a cached prefix), the base64 int32 payload lands in the routing cache
under a per-sequence routing_id, and units carry rollout_routed_experts."""

import json
from argparse import Namespace

import numpy as np
import pybase64
import pytest
from aiohttp import web

from miles.tinker.core.types import UserInputError
from miles.tinker.runtime import PAD_ROW, MilesBackend, _build_train_data, _empty_window

L, K, E = 3, 2, 16


def _args(**overrides) -> Namespace:
    base = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tinker_train_mode="full",
        tinker_checkpoint_root=None,
        use_rollout_routing_replay=True,
        num_layers=L,
        moe_router_topk=K,
        num_experts=E,
        tinker_routing_prefix_stride=4,
    )
    return Namespace(**{**base, **overrides})


def _payload(prompt: list[int], num_samples: int = 1, **extra) -> dict:
    return {
        "prompt_tokens": prompt,
        "sampling_params": {"max_tokens": 8},
        "num_samples": num_samples,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
        "tenant": "t",
        **extra,
    }


def _routing_for(tokens: list[int], start_len: int, extra_rows: int = 0) -> np.ndarray:
    """Deterministic routing rows [start_len, len(tokens)-1) (+ optional stop-edge rows)."""
    rows = np.arange(start_len, len(tokens) - 1 + extra_rows, dtype=np.int32)
    return (rows[:, None, None] + np.arange(L)[None, :, None] + np.arange(K)[None, None, :]) % E


@pytest.fixture
async def router():
    """A /generate that answers a fixed 3-token completion with routed_experts,
    honouring routed_experts_start_len. ``state`` controls the payload."""
    state = {"bodies": [], "generated": [50, 51, 52], "extra_rows": 0, "omit_routing": False, "bad_expert": False}

    async def generate(request):
        body = await request.json()
        state["bodies"].append(body)
        tokens = body["input_ids"] + state["generated"]
        start = body.get("routed_experts_start_len", 0)
        meta = {
            "output_token_logprobs": [(-0.1, t) for t in state["generated"]],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": len(body["input_ids"]),
        }
        if body.get("return_routed_experts") and not state["omit_routing"]:
            rows = _routing_for(tokens, start, state["extra_rows"]).astype(np.int32)
            if state["bad_expert"]:
                rows[0, 0, 0] = E
            meta["routed_experts"] = pybase64.b64encode(rows.tobytes()).decode("ascii")
        return web.json_response({"text": "", "meta_info": meta})

    app = web.Application()
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield state, f"http://127.0.0.1:{port}"
    await runner.cleanup()


@pytest.fixture
async def backend(router):
    state, url = router
    backend = MilesBackend(_args(), None, url)
    yield backend
    if backend._generate_client is not None:
        await backend._generate_client.close()
    backend._routing_executor.shutdown(wait=True)


async def test_the_generate_body_asks_for_routing_and_the_prefix_delta_on_the_second_turn(router, backend):
    state, _ = router
    prompt1 = list(range(10))
    result1 = await backend.sample(_payload(prompt1), lora_name=None)
    (seq1,) = result1["sequences"]
    assert isinstance(seq1["routing_id"], int) and 0 < seq1["routing_id"] < 2**53
    assert seq1["tokens"] == [50, 51, 52] and seq1["stop_reason"] == "stop"
    assert state["bodies"][0]["return_routed_experts"] is True
    assert "routed_experts_start_len" not in state["bodies"][0]

    prompt2 = prompt1 + [50, 51, 52] + [60, 61]
    result2 = await backend.sample(_payload(prompt2), lora_name=None)
    (seq2,) = result2["sequences"]
    assert state["bodies"][1]["routed_experts_start_len"] == len(prompt1) + 3 - 1, "P = parent.num_rows"
    assert seq2["routing_id"] != seq1["routing_id"]
    assert backend.routing.stats()["prefix_hits"] == 1 and backend._routing_metrics()["routing_prefix_hits"] == 1

    full2 = prompt2 + [50, 51, 52]
    rows = backend.routing.materialize("t", seq2["routing_id"], full2)
    assert rows.dtype == np.int32 and rows.shape == (len(full2) - 1, L, K)
    np.testing.assert_array_equal(rows, _routing_for(full2, 0)), "parent rows ++ delta rows = the full routing"
    assert backend.routing.stats()["pinned"] == 0


async def test_prefix_parent_hits_are_logged_a_few_times_then_only_counted(router, backend, caplog):
    """The smoke greps head.log for the INFO line; the 15-node job must not get one line per sample."""
    import logging

    from miles.tinker import runtime as runtime_module

    backend._prefix_hit_logs_left = 2
    prompt = list(range(10))
    with caplog.at_level(logging.INFO, logger="miles.tinker.runtime"):
        (seq,) = (await backend.sample(_payload(prompt), lora_name=None))["sequences"]
        assert "prefix parent hit" not in caplog.text, "the first turn has no parent"
        for turn in range(3):
            prompt = prompt + [50, 51, 52] + [60 + turn]
            await backend.sample(_payload(prompt), lora_name=None)
    lines = [r.getMessage() for r in caplog.records if "prefix parent hit" in r.getMessage()]
    assert len(lines) == 2, lines
    assert f"routing_id={seq['routing_id']} routed_experts_start_len={len(range(10)) + 3 - 1}" in lines[0]
    assert backend.routing.stats()["prefix_hits"] == 3
    assert runtime_module.PREFIX_HIT_LOG_LIMIT >= 1


async def test_a_stop_edge_row_is_trimmed(router, backend):
    state, _ = router
    state["extra_rows"] = 1  # the engine also forwarded the final token
    (seq,) = (await backend.sample(_payload([1, 2, 3]), lora_name=None))["sequences"]
    rows = backend.routing.materialize("t", seq["routing_id"], [1, 2, 3, 50, 51, 52])
    assert rows.shape == (5, L, K)


async def test_a_response_without_routing_fails_the_sample_as_a_server_error(router, backend):
    state, _ = router
    state["omit_routing"] = True
    with pytest.raises(RuntimeError, match="routed_experts"):
        await backend.sample(_payload([1, 2, 3]), lora_name=None)
    assert backend.routing.stats() == {
        "entries": 0,
        "bytes": 0,
        "evictions": 0,
        "releases": 0,
        "pinned": 0,
        "capture_rejected": 1,
        "prefix_hits": 0,
    }


async def test_an_out_of_range_expert_is_rejected_and_the_parent_unpinned(router, backend):
    state, _ = router
    (seq,) = (await backend.sample(_payload([1, 2, 3, 4]), lora_name=None))["sequences"]
    state["bad_expert"] = True
    with pytest.raises(RuntimeError, match="expert ids"):
        await backend.sample(_payload([1, 2, 3, 4, 50, 51, 52, 9]), lora_name=None)
    parent = backend.routing.get(seq["routing_id"])
    assert parent.pending_count == 0 and parent.child_count == 0
    assert backend.routing.stats()["capture_rejected"] == 1


async def test_every_sample_of_a_request_gets_its_own_routing_id(router, backend):
    result = await backend.sample(_payload([1, 2, 3, 4], num_samples=3), lora_name=None)
    ids = [seq["routing_id"] for seq in result["sequences"]]
    assert len(set(ids)) == 3
    # a second request extending the prompt pins the parent once per sample and record() consumes each pin
    result2 = await backend.sample(_payload([1, 2, 3, 4, 50, 51, 52, 7], num_samples=2), lora_name=None)
    assert len({seq["routing_id"] for seq in result2["sequences"]}) == 2
    assert backend.routing.stats()["pinned"] == 0


def test_build_train_data_attaches_routed_experts_for_every_row_including_pads():
    calls = []

    def routing(row):
        calls.append(row.get("pad", False))
        return np.full((len(row["tokens"]) - 1, L, K), -1 if row.get("pad") else 3, dtype=np.int32)

    rows = [(0, {"tokens": [1, 2, 3], "target_len": 2}), (0, {"tokens": [4, 5], "target_len": 1})]
    train_data = _build_train_data(rows, dp_size=4, with_adapter_slots=False, routing=routing)
    routed = train_data["rollout_routed_experts"]
    assert len(routed) == 4 and calls == [False, False, True, True]
    assert routed[0].shape == (2, L, K) and routed[0].dtype == np.int32
    assert routed[2].shape == (len(PAD_ROW["tokens"]) - 1, L, K) and (routed[2] == -1).all()
    assert "rollout_routed_experts" not in _build_train_data(rows, dp_size=1)


class TestMaterialization:
    def _row(self, tokens: list[int], **extra) -> dict:
        return {"tokens": tokens, "target_len": len(tokens) - 1, "tenant": "t", **extra}

    async def _sample(self, backend, prompt: list[int]) -> tuple[int, list[int]]:
        (seq,) = (await backend.sample(_payload(prompt), lora_name=None))["sequences"]
        return seq["routing_id"], prompt + seq["tokens"]

    async def test_natural_policy_fills_uncovered_datums_with_minus_one_and_counts(self, backend):
        handle, tokens = await self._sample(backend, [1, 2, 3, 4])
        seen = []

        async def fake_run_unit(method, unit_id, train_data):
            seen.append(train_data)
            return [{"per_datum": [{"sample_index": i, "loss": 0.0, "logprobs": np.zeros(1)} for i in range(2)]}]

        backend._run_unit = fake_run_unit
        rows = [(0, self._row(tokens, routing_handle=handle)), (0, self._row([7, 8, 9]))]
        await backend.forward_backward(1, rows, "cross_entropy", {})
        routed = seen[0]["rollout_routed_experts"]
        np.testing.assert_array_equal(routed[0], _routing_for(tokens, 0))
        assert routed[1].shape == (2, L, K) and (routed[1] == -1).all()
        assert backend.routing.get(handle) is None, "a trained leaf is released"

        counts = backend._window_counts[0]
        assert counts["routing_covered_datums"] == 1 and counts["routing_uncovered_datums"] == 1
        assert counts["routing_uncovered_rows"] == 2 and counts["routing_materialize_s"] > 0
        assert counts["num_datums"] == 2 and counts["num_units"] == 1

    async def test_strict_policy_fails_the_unit_typed(self, router):
        _, url = router
        backend = MilesBackend(_args(tinker_routing_replay_policy="strict"), None, url)
        try:
            handle, tokens = await self._sample(backend, [1, 2, 3, 4])
            backend._run_unit = None  # never reached
            rows = [(0, self._row(tokens, routing_handle=handle)), (0, self._row([7, 8, 9]))]
            with pytest.raises(UserInputError, match="datum 1: R3 is enabled but no routing is cached"):
                await backend.forward_backward(1, rows, "cross_entropy", {})
        finally:
            await backend._generate_client.close()
            backend._routing_executor.shutdown(wait=True)

    async def test_optim_step_metrics_carry_the_routing_keys(self, backend, tmp_path):
        class Trainer:
            async def optim_step(self, adam_params_by_slot):
                return [{0: {"grad_norm": 0.5, "skipped": 0}}]

        backend.trainer = Trainer()
        backend.perf_log_path = tmp_path / "perf.jsonl"
        handle, tokens = await self._sample(backend, [1, 2, 3, 4])

        async def fake_run_unit(method, unit_id, train_data):
            return [{"per_datum": [{"sample_index": 0, "loss": 0.0, "logprobs": np.zeros(1)}]}]

        backend._run_unit = fake_run_unit
        await backend.forward_backward(1, [(0, self._row(tokens, routing_handle=handle))], "cross_entropy", {})
        metrics = (await backend.optim_step({0: {"learning_rate": 1e-4}}))[0]
        expected_keys = {
            "grad_norm",
            "skipped",
            "num_tokens",
            "num_datums",
            "num_units",
            "routing_covered_datums",
            "routing_uncovered_datums",
            "routing_uncovered_rows",
            "routing_materialize_s",
            "routing_capture_rejected",
            "routing_cache_entries",
            "routing_cache_bytes",
            "routing_cache_pinned",
            "routing_cache_evictions",
            "routing_prefix_hits",
            "fused_lm_head",
            "datum_seq_len_count",
            "datum_seq_len_mean",
            "datum_seq_len_p50",
            "datum_seq_len_p90",
            "datum_seq_len_max",
        }
        assert set(metrics) == expected_keys
        assert metrics["routing_covered_datums"] == 1 and metrics["routing_cache_entries"] == 0
        assert metrics["fused_lm_head"] == 0
        record = json.loads(backend.perf_log_path.read_text().splitlines()[0])
        assert record["routing_covered_datums"] == 1 and record["routing_cache_bytes"] == 0
        assert (await backend.optim_step({0: {}}))[0]["routing_covered_datums"] == 0, "windows reset"


def test_routing_is_off_without_the_flag_and_windows_stay_plain():
    backend = MilesBackend(_args(use_rollout_routing_replay=False), None, "http://router")
    assert backend.routing is None and backend.routing_replay is False and backend._routing_executor is None
    assert _empty_window() == {"num_tokens": 0, "num_datums": 0, "num_units": 0}
    assert backend._routing_metrics() == {}


def test_routing_needs_the_moe_shape_and_uint8_experts():
    with pytest.raises(ValueError, match="num-experts"):
        MilesBackend(_args(num_experts=None), None, "http://router")
    with pytest.raises(ValueError, match="256"):
        MilesBackend(_args(num_experts=512), None, "http://router")
