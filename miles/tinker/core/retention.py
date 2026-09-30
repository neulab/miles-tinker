"""Background retention of completed checkpoints in one resume lineage."""

import fcntl
import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def checkpoint_lock(path: str, *, deleting: bool = False):
    """Protect reads/writes across gateways sharing an NFS checkpoint root."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep locks outside checkpoint directories so deleting a directory cannot
    # replace a lock inode while another process still holds it.
    with (path.parent / f".{path.name}.lock").open("a") as lock:
        mode = fcntl.LOCK_EX if deleting else fcntl.LOCK_SH
        fcntl.flock(lock, mode | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prune_checkpoints(root: str, lineage: str, legacy_models: tuple[str, ...], keep: int) -> None:
    """Keep K states, preferring the newest resume and its copy of a state.

    Only sidecars published after successful saves are candidates. A failed
    prune is logged and retried on the next save; it never fails training.
    """
    try:
        candidates = []
        for sidecar in Path(root).glob("*/weights/*/tinker_meta.json"):
            if sidecar.parent.is_symlink() or sidecar.parent.parent.is_symlink() or sidecar.parents[2].is_symlink():
                continue
            try:
                meta = json.loads(sidecar.read_text())
                same_lineage = meta.get("checkpoint_lineage", meta.get("model_id")) == lineage
                legacy = "checkpoint_lineage" not in meta and meta.get("model_id") in legacy_models
                if same_lineage or legacy:
                    candidates.append((sidecar.parent, meta))
            except (OSError, ValueError):
                continue  # incomplete or concurrently removed metadata
        candidates.sort(key=lambda item: (item[1].get("resume_started_at", 0), item[1]["created_at"]), reverse=True)
        kept_states = set()
        for path, meta in candidates:
            state = meta.get("checkpoint_state", str(path))
            if state not in kept_states and len(kept_states) < keep:
                kept_states.add(state)
                continue
            try:
                with checkpoint_lock(str(path), deleting=True):
                    # A concurrent save may have replaced this named checkpoint.
                    if json.loads((path / "tinker_meta.json").read_text()) == meta:
                        shutil.rmtree(path)
            except (BlockingIOError, FileNotFoundError):
                continue  # a save/load has pinned it; try again on the next save
    except Exception:  # noqa: BLE001  retention must never fail a save or training
        logger.exception("Checkpoint retention failed for lineage %s", lineage)
