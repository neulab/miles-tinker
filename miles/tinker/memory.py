"""Gateway host allocation defaults and Linux memory-pressure telemetry.

Keep this module NumPy-free: configuration must precede imports of NumPy,
including indirect imports through Ray and Torch.
"""

import os
from pathlib import Path


def configure_numpy_allocation() -> None:
    """Avoid one MADV_HUGEPAGE region per retained routing array.

    Tens of thousands of full R3 captures can exhaust vm.max_map_count while
    host RAM is plentiful. NumPy reads this option only at import time. An
    explicit operator setting still takes precedence.
    """
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")


def process_memory_metrics() -> dict[str, int]:
    """Small Linux-only snapshot; unavailable proc files are omitted."""
    metrics = {}
    try:
        with Path("/proc/self/maps").open() as stream:
            metrics["gateway_memory_maps"] = sum(1 for _ in stream)
        metrics["gateway_memory_map_limit"] = int(Path("/proc/sys/vm/max_map_count").read_text())
        for line in Path("/proc/self/status").read_text().splitlines():
            field, _, value = line.partition(":")
            if field in ("VmRSS", "VmSize"):
                key = "gateway_rss_bytes" if field == "VmRSS" else "gateway_virtual_bytes"
                metrics[key] = int(value.split()[0]) * 1024
    except OSError:
        pass  # Other platforms, or a container without proc mounted.
    return metrics
