"""The dispatch loop holds an under-filled work unit briefly so chunks of the
same window arriving in a burst pack into one trainer call; barriers and
full units never wait."""

import asyncio
import time

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, command, created_model, fb_payload, make_service, row

from miles.tinker.core.planner import Planner, WorkUnit
from miles.tinker.core.stream import ModelStream


@pytest.fixture
async def coalescing():
    svc = make_service(coalesce_wait_s=0.05, coalesce_max_wait_s=0.5, unit_token_budget=1_000)
    run_task = asyncio.create_task(svc.run())
    yield svc
    for task in (run_task, getattr(svc, "_sweep_task", None)):
        if task is not None:
            task.cancel()


async def test_chunks_arriving_in_a_burst_share_one_unit(coalescing):
    model_id = await created_model(coalescing)
    first = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    await asyncio.sleep(0.01)
    second = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 2, [row()]))

    for request_id in (first, second):
        await await_settled(coalescing, "tenant", request_id)
    units = coalescing.backend.named("forward_backward")
    assert [len(unit["slot_rows"]) for unit in units] == [2]


async def test_a_full_unit_runs_immediately(coalescing):
    coalescing.config.unit_token_budget = 4  # one row already fills it
    model_id = await created_model(coalescing)
    started = time.monotonic()
    request_id = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    await await_settled(coalescing, "tenant", request_id)
    assert time.monotonic() - started < coalescing.config.coalesce_wait_s


async def test_a_queued_barrier_stops_the_wait(coalescing):
    model_id = await created_model(coalescing)
    started = time.monotonic()
    fb = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    optim = coalescing.submit("tenant", "optim_step", {"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)})
    for request_id in (fb, optim):
        await await_settled(coalescing, "tenant", request_id)
    assert time.monotonic() - started < coalescing.config.coalesce_wait_s, "nothing more can join a closed window"


async def test_the_wait_is_bounded_by_the_max(coalescing):
    coalescing.config.coalesce_wait_s = 0.05
    coalescing.config.coalesce_max_wait_s = 0.1
    model_id = await created_model(coalescing)
    first = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    for seq_id in range(2, 8):  # keep the window busy every 30 ms, longer than the max wait
        await asyncio.sleep(0.03)
        coalescing.submit("tenant", "forward_backward", fb_payload(model_id, seq_id, [row()]))
    await await_settled(coalescing, "tenant", first)
    assert len(coalescing.backend.named("forward_backward")) >= 1
    assert len(coalescing.backend.named("forward_backward")[0]["slot_rows"]) < 7


def test_plan_unit_is_pure_and_commit_issues_rows():
    planner = Planner(1_000)
    stream = ModelStream("model", "tenant", slot=0)
    planner.add_stream(stream)
    stream.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [row(), row()]), arrival=1))

    planned = planner.plan_unit()
    assert isinstance(planned, WorkUnit) and planned.num_tokens == 8
    assert planner.plan_unit().rows == planned.rows, "planning twice sees the same rows"
    planner.commit(planned)
    assert planner.plan_unit() is None
    assert stream.window_may_grow()
