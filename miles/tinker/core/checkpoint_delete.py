"""Deleting a checkpoint directory (``DELETE /api/v1/training_runs/{id}/checkpoints/{kind}/{name}``) on disk.

The directory is reached from the checkpoint root one component at a time, never through a symbolic link, so a
delete cannot leave the root. It leaves the namespace with one rename into ``<root>/.tinker-trash/``: the same
filesystem, so the rename is atomic and instant even for a 352 GB full fine-tuning save, and no later load or save
can see a half-deleted checkpoint. One daemon thread then frees the files, so neither the request nor the event
loop waits for the filesystem. A gateway that exits while freeing leaves its entry in the trash; the next gateway
on the same root frees it on its first delete.
"""

import errno
import json
import logging
import os
import queue
import shutil
import stat
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from miles.tinker.core.stream import ModelStream
from miles.tinker.core.types import Command, NotFoundError, PermissionDeniedError, UserInputError

logger = logging.getLogger(__name__)

TRASH_DIR = ".tinker-trash"

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_OPEN_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _CLOEXEC


@dataclass(frozen=True)
class CheckpointDir:
    """``<root>/<model_id>/<kind>/<name>``, opened without following a symbolic link. The descriptors are valid
    inside the ``open_checkpoint_dir`` block only."""

    root: str  # the checkpoint root, resolved
    root_fd: int
    parent_fd: int  # <root>/<model_id>/<kind>
    parent: str  # "<model_id>/<kind>"
    name: str
    label: str  # tinker://<model_id>/<kind>/<name>, for messages
    sidecar: dict | None


def parse_checkpoint_id(model_id: str, checkpoint_id: str) -> tuple[str, str]:
    """The SDK's checkpoint_id, ``weights/<name>`` or ``sampler_weights/<name>`` -> (kind, name)."""
    kind, _, name = checkpoint_id.partition("/")
    if kind not in ("weights", "sampler_weights") or not name:
        raise UserInputError(
            f"malformed checkpoint id {checkpoint_id!r}: expected weights/<name> or sampler_weights/<name>"
        )
    if "\x00" in model_id + checkpoint_id:
        raise UserInputError(f"invalid checkpoint path component in {model_id!r}/{checkpoint_id!r}")
    return kind, name


def checkpoint_user(
    streams: Iterable[ModelStream], running: Mapping[str, str], path: str, *, model_id: str, kind: str, name: str
) -> str | None:
    """What writes or reads the checkpoint now or next, or None: the running barrier (``running``: checkpoint
    dir -> activity) or a queued command that names it, a load_state of any run or a save of its own run. An
    unnamed sampler save gets its name only when it runs, so no client can name it while it is queued."""
    if path in running:
        return running[path]
    tinker_path = f"tinker://{model_id}/{kind}/{name}"
    for stream in streams:
        for command in [pending.command for pending in stream.queue] + list(stream.arrivals.values()):
            if command.kind == "load_state" and command.payload.get("path") == tinker_path:
                return "a queued load_state will read it"
            if command.model_id == model_id and _saved_checkpoint(command) == (kind, name):
                return f"a queued {command.kind} will write it"
    return None


def _saved_checkpoint(command: Command) -> tuple[str, str] | None:
    """(kind, name) a queued save writes, as service._execute_barrier names it."""
    if command.kind == "save_state":
        return "weights", command.payload.get("name") or f"checkpoint-{command.seq_id:06d}"
    if command.kind == "save_weights_for_sampler" and command.payload.get("name"):
        return "sampler_weights", command.payload["name"]
    return None


@contextmanager
def open_checkpoint_dir(root: str, model_id: str, kind: str, name: str, sidecar_name: str) -> Iterator[CheckpointDir]:
    """Raises NotFoundError when a component is missing or not a directory and PermissionDeniedError when one is
    a symbolic link. The root itself may be a link: it is configuration, resolved once."""
    label = f"tinker://{model_id}/{kind}/{name}"
    if model_id == TRASH_DIR:  # what is being freed is nobody's checkpoint any more
        raise NotFoundError(f"checkpoint {label} not found")
    resolved_root = os.path.realpath(root)
    fds: list[int] = []
    try:
        fds.append(_open_root(resolved_root, label))
        for component in (model_id, kind, name):
            fds.append(_open_component(component, fds[-1], label))
        yield CheckpointDir(
            root=resolved_root,
            root_fd=fds[0],
            parent_fd=fds[2],
            parent=f"{model_id}/{kind}",
            name=name,
            label=label,
            sidecar=_read_sidecar(sidecar_name, fds[3], label),
        )
    finally:
        for fd in reversed(fds):
            os.close(fd)


def move_to_trash(checkpoint: CheckpointDir) -> str:
    """Rename the checkpoint out of its run's namespace; returns where it went, for the reaper."""
    entry = uuid.uuid4().hex
    trash_fd = _open_trash(checkpoint.root_fd)
    if trash_fd is not None:
        try:
            os.rename(checkpoint.name, entry, src_dir_fd=checkpoint.parent_fd, dst_dir_fd=trash_fd)
            return os.path.join(checkpoint.root, TRASH_DIR, entry)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
        finally:
            os.close(trash_fd)
    # a mount point below the root (or no usable trash): hide it beside itself, which never crosses a filesystem
    hidden = f"{TRASH_DIR}-{entry}"
    os.rename(checkpoint.name, hidden, src_dir_fd=checkpoint.parent_fd, dst_dir_fd=checkpoint.parent_fd)
    return os.path.join(checkpoint.root, checkpoint.parent, hidden)


class CheckpointReaper:
    """Frees moved-away checkpoint directories on one daemon thread, in delete order, and logs what each freed.
    Daemon: a gateway that is told to stop does not wait for the filesystem."""

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._adopted_roots: set[str] = set()

    def free(self, path: str, label: str) -> None:
        self._ensure_started()
        self._jobs.put((path, label))

    def adopt_leftovers(self, root: str) -> int:
        """Once per root and gateway: queue what an earlier gateway left in the trash. Call it before moving
        anything there, so that only leftovers are listed."""
        if root in self._adopted_roots:
            return 0
        self._adopted_roots.add(root)
        trash = os.path.join(root, TRASH_DIR)
        try:
            if not stat.S_ISDIR(os.lstat(trash).st_mode):
                return 0
            entries = sorted(os.listdir(trash))
        except OSError as error:  # never fails the delete that asked
            if not isinstance(error, FileNotFoundError):
                logger.warning(f"checkpoint delete: cannot list {trash} for leftovers: {error}")
            return 0
        for entry in entries:
            self.free(os.path.join(trash, entry), f"{TRASH_DIR}/{entry}, left by an earlier gateway")
        return len(entries)

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until everything queued so far is freed."""
        done = threading.Event()
        self._ensure_started()
        self._jobs.put(done)
        return done.wait(timeout)

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="tinker-checkpoint-reaper", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if isinstance(job, threading.Event):
                job.set()
                continue
            path, label = job
            try:
                free_tree(path, label)
            except Exception:  # noqa: BLE001  one tree that cannot be freed never stops the reaper
                logger.exception(f"checkpoint delete: freeing {label} at {path} failed")


def free_tree(path: str, label: str) -> None:
    """Remove ``path`` (never following a link inside it) and log what it held, or what could not be removed."""
    started = time.monotonic()
    files, size = _tree_size(path)
    errors: list[str] = []

    def on_error(_function, failed_path, error) -> None:
        error = error[1] if isinstance(error, tuple) else error  # onerror (Python < 3.12) passes exc_info
        if not isinstance(error, FileNotFoundError):  # another gateway on the same root freed it first
            errors.append(f"{failed_path}: {error}")

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=on_error)
    else:
        shutil.rmtree(path, onerror=on_error)
    if errors:
        logger.warning(
            f"checkpoint delete: {len(errors)} path(s) of {label} could not be freed from {path}, first: {errors[0]}"
        )
        return
    seconds = time.monotonic() - started
    logger.info(f"checkpoint delete: freed {label}: {files} files, {size / 1e9:.1f} GB in {seconds:.1f} s")


def _open_root(root: str, label: str) -> int:
    try:
        return os.open(root, os.O_RDONLY | os.O_DIRECTORY | _CLOEXEC)
    except (FileNotFoundError, NotADirectoryError):
        raise NotFoundError(f"checkpoint {label} not found") from None


def _open_component(component: str, parent_fd: int, label: str) -> int:
    try:
        mode = os.stat(component, dir_fd=parent_fd, follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise NotFoundError(f"checkpoint {label} not found") from None
    if stat.S_ISLNK(mode):
        raise PermissionDeniedError(_symlink_message(component, label))
    if not stat.S_ISDIR(mode):
        raise NotFoundError(f"checkpoint {label} not found")
    try:
        return os.open(component, _OPEN_DIR_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        raise NotFoundError(f"checkpoint {label} not found") from None
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):  # replaced by a link since the stat
            raise PermissionDeniedError(_symlink_message(component, label)) from None
        raise


def _symlink_message(component: str, label: str) -> str:
    return (
        f"checkpoint {label}: {component!r} is a symbolic link; "
        "the gateway only deletes directories inside its checkpoint root"
    )


def _read_sidecar(sidecar_name: str, checkpoint_fd: int, label: str) -> dict | None:
    try:
        fd = os.open(sidecar_name, os.O_RDONLY | os.O_NOFOLLOW | _CLOEXEC, dir_fd=checkpoint_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        logger.warning(f"checkpoint delete: unreadable sidecar of {label}: {error}")
        return None
    with os.fdopen(fd) as handle:
        try:
            meta = json.load(handle)
        except ValueError as error:
            logger.warning(f"checkpoint delete: unreadable sidecar of {label}: {error}")
            return None
    return meta if isinstance(meta, dict) else None


def _open_trash(root_fd: int) -> int | None:
    try:
        os.mkdir(TRASH_DIR, dir_fd=root_fd)
    except FileExistsError:
        pass
    except OSError as error:
        logger.warning(f"checkpoint delete: cannot create {TRASH_DIR} in the checkpoint root: {error}")
        return None
    try:
        return os.open(TRASH_DIR, _OPEN_DIR_FLAGS, dir_fd=root_fd)
    except OSError as error:  # a file or a link where the trash should be
        logger.warning(f"checkpoint delete: {TRASH_DIR} in the checkpoint root is unusable: {error}")
        return None


def _tree_size(path: str) -> tuple[int, int]:
    files = size = 0
    for directory, _subdirs, names in os.walk(path):
        for name in names:
            try:
                size += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                continue
            files += 1
    return files, size
