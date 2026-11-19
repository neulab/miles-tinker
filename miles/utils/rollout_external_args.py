"""Argument validation for attached external SGLang engines (``--rollout-external``).

Kept stdlib-only so the fast tests exercise it without the sglang/megatron
imports that ``miles.utils.arguments`` pulls in.
"""

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

_SUPPORTED_WEIGHT_TRANSFER_MODES = ("broadcast", "disk-delta")


class ExternalEngineAddr(NamedTuple):
    host: str
    port: int


def parse_external_engine_addr(addr: str) -> ExternalEngineAddr:
    """``host:port`` or ``[v6-host]:port``; a scheme or path is rejected rather than silently kept."""
    host, sep, port = addr.rpartition(":")
    assert sep and host and port.isdigit() and "/" not in addr, (
        f"--rollout-external-engine-addrs entry {addr!r} must be host:port (no scheme, no path)"
    )
    return ExternalEngineAddr(host=host.strip("[]"), port=int(port))


def validate_rollout_external_args(args) -> None:
    """Constraints of attaching engines miles did not launch, and the engine count they imply."""
    if not args.rollout_external:
        assert not args.rollout_external_engine_addrs, "--rollout-external-engine-addrs requires --rollout-external"
        return

    addrs = args.rollout_external_engine_addrs or []
    assert addrs, "--rollout-external requires --rollout-external-engine-addrs host:port [host:port ...]"
    parsed = [parse_external_engine_addr(addr) for addr in addrs]
    assert len(set(parsed)) == len(parsed), f"--rollout-external-engine-addrs has duplicates: {addrs}"

    assert not args.colocate, "--rollout-external engines run on their own GPUs; --colocate is not supported"
    assert args.eval_num_gpus == 0, (
        "--rollout-external attaches rollout engines only; a dedicated eval fleet (--eval-num-gpus) "
        "would still be launched by miles"
    )
    assert args.update_weight_transfer_mode in _SUPPORTED_WEIGHT_TRANSFER_MODES, (
        f"--rollout-external supports --update-weight-transfer-mode {'/'.join(_SUPPORTED_WEIGHT_TRANSFER_MODES)}, "
        f"not {args.update_weight_transfer_mode!r}"
    )

    num_gpus = len(addrs) * args.rollout_num_gpus_per_engine
    assert args.rollout_num_gpus in (None, num_gpus), (
        f"--rollout-num-gpus {args.rollout_num_gpus} != {len(addrs)} external engines * "
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine}; drop --rollout-num-gpus "
        f"to derive it from the engine list"
    )
    args.rollout_num_gpus = num_gpus
    logger.info(f"--rollout-external: attaching {len(addrs)} engines ({num_gpus=}): {addrs}")
