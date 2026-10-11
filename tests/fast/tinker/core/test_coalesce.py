"""The dispatch loop holds an under-filled batch briefly so chunks of one SDK
request arriving in a burst pack into one trainer call; barriers and full
batches never wait."""

import asyncio
import time
from contextlib import suppress

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, command, created_model, datum, fb_payload, make_service

from miles.tinker.core.model_queue import ModelRequestQueue
from miles.tinker.core.scheduler import BatchUnit, RequestScheduler


@pytest.fixture
async def coalescing(tmp_path):
    service = make_service(tmp_path, coalesce_wait_s=0.05, coalesce_max_wait_s=0.5, batch_token_budget=1_000)
    run_task = asyncio.create_task(service.run())
    try:
        yield service
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def test_chunks_arriving_in_a_burst_share_one_batch(coalescing):
    model_id = await created_model(coalescing)
    first = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    await asyncio.sleep(0.01)
    second = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))

    for request_id in (first, second):
        await await_settled(coalescing, "tenant", request_id)
    batches = coalescing.backend.named("forward_backward")
    assert [len(batch["slot_datums"]) for batch in batches] == [2]


async def test_a_full_batch_runs_immediately(coalescing):
    coalescing.config.batch_token_budget = 4  # one datum already fills it
    model_id = await created_model(coalescing)
    started = time.monotonic()
    request_id = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    await await_settled(coalescing, "tenant", request_id)
    assert time.monotonic() - started < coalescing.config.coalesce_wait_s


async def test_a_queued_barrier_stops_the_wait(coalescing):
    model_id = await created_model(coalescing)
    started = time.monotonic()
    fb = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    optim = coalescing.submit("tenant", "optim_step", {"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)})
    for request_id in (fb, optim):
        await await_settled(coalescing, "tenant", request_id)
    assert time.monotonic() - started < coalescing.config.coalesce_wait_s, "nothing more can join a closed run"


async def test_the_wait_is_bounded_by_the_max(coalescing):
    coalescing.config.coalesce_max_wait_s = 0.1
    model_id = await created_model(coalescing)
    first = coalescing.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    for seq_id in range(2, 8):  # keep the run busy every 30 ms, longer than the max wait
        await asyncio.sleep(0.03)
        coalescing.submit("tenant", "forward_backward", fb_payload(model_id, seq_id, [datum()]))
    await await_settled(coalescing, "tenant", first)
    batches = coalescing.backend.named("forward_backward")
    assert batches and len(batches[0]["slot_datums"]) < 7


async def test_a_zero_wait_runs_every_batch_at_once(tmp_path):
    service = make_service(tmp_path)  # the config default disables coalescing
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        started = time.monotonic()
        request_id = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
        await await_settled(service, "tenant", request_id)
        assert time.monotonic() - started < 0.05
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


def test_a_preview_issues_nothing_and_the_commit_issues_the_same_datums():
    scheduler = RequestScheduler(1_000)
    model_queue = ModelRequestQueue("model", "tenant", slot=0)
    scheduler.add_model_queue(model_queue)
    model_queue.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum(), datum()]), arrival=1))

    preview = scheduler.schedule_next(commit=False)
    assert isinstance(preview, BatchUnit) and len(preview.datums) == 2
    assert scheduler.schedule_next(commit=False).datums == preview.datums, "previewing twice sees the same datums"
    committed = scheduler.schedule_next()
    assert committed.datums == preview.datums
    assert scheduler.schedule_next() is None
    assert model_queue.batch_run_may_grow()

    model_queue.submit(command("model", 2, "optim_step", {"model_id": "model", "seq_id": 2}, arrival=2))
    assert not model_queue.batch_run_may_grow(), "a queued barrier closes the run"
