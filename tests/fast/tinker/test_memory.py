"""The gateway allocation default must bound VMAs as full R3 captures grow."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miles.tinker.memory import configure_numpy_allocation, process_memory_metrics


def test_explicit_numpy_allocation_policy_is_preserved(monkeypatch):
    monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", "1")
    configure_numpy_allocation()
    assert os.environ["NUMPY_MADVISE_HUGEPAGE"] == "1"


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="Linux VMA regression")
def test_many_retained_arrays_do_not_grow_one_mapping_per_capture():
    # Fresh interpreter is essential: NumPy reads the setting on first import.
    # Sparse allocations reserve 21 GiB of address space but touch <100 MiB.
    code = """
import json
from miles.tinker.memory import configure_numpy_allocation, process_memory_metrics
configure_numpy_allocation()
import numpy as np
before = process_memory_metrics()['gateway_memory_maps']
arrays = [np.empty((16384 + i % 1000, 40, 8), dtype=np.uint8) for i in range(4096)]
after = process_memory_metrics()['gateway_memory_maps']
print(json.dumps({'growth': after - before}))
"""
    env = {key: value for key, value in os.environ.items() if key != "NUMPY_MADVISE_HUGEPAGE"}
    env["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["growth"] < 100


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="Linux memory telemetry")
def test_process_memory_metrics_measure_current_process():
    metrics = process_memory_metrics()
    assert 0 < metrics["gateway_memory_maps"] < metrics["gateway_memory_map_limit"]
    assert 0 < metrics["gateway_rss_bytes"] <= metrics["gateway_virtual_bytes"]
