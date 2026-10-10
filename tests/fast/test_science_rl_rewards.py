"""Scientific failures are rewards; runtime failures must never become negatives."""

import asyncio
from types import SimpleNamespace

import pytest

from examples.science_rl.agent_function import reward_func
from miles.rollout.filter_hub.dynamic_sampling_filters import check_no_aborted
from miles.utils.types import Sample


@pytest.mark.parametrize("reward", [0, 1])
def test_completed_trial_preserves_binary_reward(reward):
    sample = Sample(metadata={"science_status": "completed", "science_reward": reward})
    assert asyncio.run(reward_func(None, sample)) == reward
    assert sample.status != Sample.Status.ABORTED
    assert check_no_aborted(None, [sample]).keep


@pytest.mark.parametrize("status", ["error", "timeout", "oom", None])
def test_infrastructure_failure_discards_entire_group(status):
    failure = Sample(metadata={"science_status": status, "science_reward": 0})
    success = Sample(metadata={"science_status": "completed", "science_reward": 1})
    asyncio.run(reward_func(None, failure))
    assert failure.status == Sample.Status.ABORTED
    assert not check_no_aborted(None, [success, failure]).keep


def test_missing_verifier_reward_is_aborted():
    sample = Sample(metadata={"science_status": "completed"})
    asyncio.run(reward_func(SimpleNamespace(), sample))
    assert sample.status == Sample.Status.ABORTED
