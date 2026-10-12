import asyncio
from types import SimpleNamespace

from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.utils.types import Sample


def buffer(policy="block"):
    unused = []
    args = SimpleNamespace(rollout_batch_size=1, async_data_buffer_capacity_factor=1, async_data_buffer_overflow_policy=policy, dynamic_sampling_filter_path=None, max_weight_staleness=None)
    return DefaultDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=unused.append)), unused


def entry(index):
    group = [Sample(group_index=index, status=Sample.Status.COMPLETED, reward=1)]
    return DataBufferInput(prompt_group=group, group=group)


def test_drop_oldest_never_waits_on_a_full_buffer_and_keeps_complete_groups():
    async def run():
        data, unused = buffer("drop_oldest")
        old, new = entry(0), entry(1)
        await data.put(old)
        await asyncio.wait_for(data.put(new), 0.2)
        assert unused == [old.prompt_group]
        assert (await data.get()).group == new.group
        assert data.get_metrics()["rollout/fully_async/overflow_groups_dropped"] == 1
        assert data.get_metrics()["rollout/fully_async/overflow_groups_dropped"] == 0

    asyncio.run(run())


def test_default_overflow_policy_preserves_backpressure():
    async def run():
        data, unused = buffer()
        old, new = entry(0), entry(1)
        await data.put(old)
        pending = asyncio.create_task(data.put(new))
        await asyncio.sleep(0)
        assert not pending.done()
        assert (await data.get()).group == old.group
        await asyncio.wait_for(pending, 0.2)
        assert (await data.get()).group == new.group
        assert unused == []

    asyncio.run(run())
