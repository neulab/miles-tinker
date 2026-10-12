from types import SimpleNamespace

from miles.ray.rollout.metrics import _compute_perf_metrics_from_samples


def test_async_queue_drain_is_not_generation_throughput():
    args = SimpleNamespace(fully_async=True, rollout_num_gpus=2)
    samples = [
        SimpleNamespace(response_length=8000, effective_response_length=4000, non_generation_time=60),
        SimpleNamespace(response_length=4000, effective_response_length=2000, non_generation_time=20),
    ]

    metrics = _compute_perf_metrics_from_samples(args, samples, rollout_time=0.01)

    assert metrics["async_batch_collection_time"] == 0.01
    assert metrics["rollout_time"] == 0.01
    assert metrics["non_generation_time/mean"] == 40
    assert not any("tokens_per" in key for key in metrics)


def test_synchronous_generation_throughput_keeps_its_elapsed_time():
    args = SimpleNamespace(fully_async=False, rollout_num_gpus=2)
    samples = [
        SimpleNamespace(response_length=200, effective_response_length=100, non_generation_time=0),
        SimpleNamespace(response_length=100, effective_response_length=50, non_generation_time=0),
    ]

    metrics = _compute_perf_metrics_from_samples(args, samples, rollout_time=10)

    assert metrics["tokens_per_gpu_per_sec"] == 15
    assert metrics["effective_tokens_per_gpu_per_sec"] == 7.5
    assert metrics["longest_sample_tokens_per_sec"] == 20
    assert metrics["longest_effective_sample_tokens_per_sec"] == 10
    assert "async_batch_collection_time" not in metrics
