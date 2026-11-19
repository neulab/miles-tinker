from argparse import Namespace
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.training_utils.weight_update.protocols.delta import (
    UpdateWeightFromDiskDelta,
    _checkpoint_float_dtype,
)

_DELTA_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


class _RejectingApiClient:
    def __init__(self, calls: list[tuple[str, dict]], failing_method: str) -> None:
        self._calls = calls
        self._failing_method = failing_method

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            if name == self._failing_method:
                return {"success": False, "error_message": "engine rejected the weights"}
            return {"success": True}

        return method


class TestPostWriteHookConstruction:
    def test_configured_post_write_hook_is_loaded_from_function_registry(self, tmp_path: Path) -> None:
        """A configured post-write path becomes the hook, resolved through the shared function registry."""
        hook = object()
        args = Namespace(
            update_weight_disk_dir=str(tmp_path / "delta"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3",
            custom_update_weight_post_write_path="miles_plugins.example:upload_delta",
        )

        with patch("miles.utils.function_registry.load_function", return_value=hook) as load_function:
            protocol = UpdateWeightFromDiskDelta(args)

        load_function.assert_called_once_with("miles_plugins.example:upload_delta")
        assert protocol._post_write_hook is hook


class TestReloadEnginesFailureTransitions:
    @staticmethod
    def _make_protocol(calls: list[tuple[str, dict]], failing_method: str) -> UpdateWeightFromDiskDelta:
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(
            update_weight_local_checkpoint_dir="/local/ckpt",
            update_weight_disk_dir="/shared/delta",
            pause_generation_mode="retract",
            check_weight_update_equal=False,
        )
        protocol.rollout_engines = [_RejectingApiClient(calls, failing_method)]
        protocol._post_write_hook = None
        protocol._version_dir = "/shared/delta/v7"
        return protocol

    @pytest.mark.parametrize(
        ("failing_method", "expected_calls"),
        [
            ("pull_weights", ["pull_weights"]),
            (
                "update_weights_from_disk",
                ["pull_weights", "pause_generation", "flush_cache", "update_weights_from_disk"],
            ),
        ],
    )
    def test_reload_engine_failure_stops_before_the_next_lifecycle_phase(
        self, failing_method: str, expected_calls: list[str]
    ) -> None:
        """A rejected pull never pauses the engine, and a rejected disk reload never resumes it."""
        calls: list[tuple[str, dict]] = []
        protocol = self._make_protocol(calls, failing_method)

        with (
            patch(f"{_DELTA_MODULE}.dist") as dist_mock,
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
        ):
            dist_mock.get_rank.return_value = 0
            with pytest.raises(RuntimeError, match="engine rejected the weights"):
                protocol._reload_engines(7)

        assert [name for name, _kwargs in calls] == expected_calls


_NORM = "model.layers.0.linear_attn.norm.weight"
_PROJ = "model.layers.0.mlp.down_proj.weight"


class TestCheckpointDtypeCast:
    """Qwen3.5 stores some tensors (e.g. linear_attn.norm.weight) in F32 while Megatron emits BF16."""

    @pytest.mark.parametrize(
        ("emitted", "stored", "expected"),
        [
            (torch.bfloat16, "F32", torch.float32),
            (torch.float32, "F16", torch.float16),
            (torch.bfloat16, "BF16", None),
            (torch.bfloat16, None, None),
            (torch.bfloat16, "I64", None),
            (torch.int64, "F32", None),
        ],
    )
    def test_only_float_mismatches_cast_to_the_stored_dtype(self, emitted, stored, expected) -> None:
        assert _checkpoint_float_dtype(emitted, stored) is expected

    @staticmethod
    def _make_protocol(tmp_path: Path) -> UpdateWeightFromDiskDelta:
        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        save_file(
            {
                _NORM: torch.tensor([1.0, 0.5, -2.0, 0.25], dtype=torch.float32),
                _PROJ: torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            },
            str(hf_dir / "model.safetensors"),
        )
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(
            hf_checkpoint=str(hf_dir),
            update_weight_local_checkpoint_dir="/local/ckpt",
            update_weight_disk_dir=str(tmp_path / "delta"),
            check_weight_update_equal=False,
        )
        protocol.delta_dir = str(tmp_path / "delta")
        protocol.rollout_engines = []
        protocol.is_sender = True
        protocol._post_write_hook = None
        protocol._snapshot = {}
        return protocol

    def test_baseline_casts_bf16_emitted_tensors_to_the_f32_checkpoint_dtype(self, tmp_path: Path) -> None:
        """The raw-byte baseline must match the engines' checkpoint files, so the F32 tensor is upcast."""
        protocol = self._make_protocol(tmp_path)
        bucket = [
            (_NORM, torch.tensor([1.0, 0.5, -2.0, 0.25], dtype=torch.bfloat16)),
            (_PROJ, torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)),
        ]

        with (
            patch(f"{_DELTA_MODULE}.dist") as dist_mock,
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
        ):
            dist_mock.get_rank.return_value = 0
            dist_mock.get_world_size.return_value = 1
            protocol._capture_baseline(lambda materialize: [bucket])

        assert protocol._cast == {_NORM: torch.float32}
        assert protocol._snapshot[_NORM].nbytes == 4 * 4
        assert protocol._snapshot[_PROJ].nbytes == 6 * 2

    def test_send_bucket_diffs_the_cast_bytes(self, tmp_path: Path) -> None:
        """Later syncs emit the same BF16 tensor; its bytes are diffed in the checkpoint's F32 layout."""
        protocol = self._make_protocol(tmp_path)
        protocol._cast = {_NORM: torch.float32}
        protocol._use_pinned = False
        protocol._max_bytes = 0
        protocol._pool = MagicMock()
        protocol._inflight = deque()
        protocol.total_bytes = 0
        emitted = torch.tensor([1.0, 0.5, -2.0, 0.25], dtype=torch.bfloat16)

        protocol.send_bucket([(_NORM, emitted)])

        _fn, name, payload, nbytes, pinned = protocol._pool.submit.call_args.args
        assert (name, nbytes, pinned) == (_NORM, 16, False)
        assert payload.tobytes() == emitted.to(torch.float32).numpy().tobytes()
