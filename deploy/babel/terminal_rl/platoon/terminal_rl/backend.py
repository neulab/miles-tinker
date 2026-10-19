"""Apptainer sandbox backend.

Vendored from tmax's open-instruct fork so the sandbox posture matches the one
the LiteCoder task images were built and smoke-tested against:
https://github.com/hamishivi/tmax/blob/master/training/open-instruct/open_instruct/environments/backends.py

Changes from the original: the Docker backend and its factory are dropped, the
logger is local, and images resolve to a local SIF pool (Babel compute nodes
cannot pull from a registry).
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("platoon.terminal_rl.backend")

SIF_DIR_ENV = "SWERL_APPTAINER_SIF_DIR"

# Sandbox resource guards. A sandbox's processes (agent server, every command)
# share the node's RAM with no limit: apptainer --memory/--pids-limit refuse to
# run under --fakeroot ("rootless cgroups is not usable in fakeroot mode") and
# the Slurm job cgroup is not delegated, so we cannot cap them with cgroups. In
# 156937 the node ran out of RAM with a load average of ~2.6 million and the
# kernel killed the learner, which Ray marks oom_score_adj 1000, not a sandbox.
#
# TERMINAL_RL_SANDBOX_OOM_SCORE_ADJ (default 1000; 0 = leave as is): every
#   sandbox process starts at this oom_score_adj, so the kernel weighs a runaway
#   sandbox against the learner by size instead of always killing the learner.
# TERMINAL_RL_SANDBOX_MEM_LIMIT_GB / TERMINAL_RL_SANDBOX_PIDS_LIMIT (0 = off): a
#   watchdog thread sums the resident memory and counts the processes of each
#   sandbox (every process in its mount namespace) every
#   TERMINAL_RL_SANDBOX_WATCH_S seconds and SIGKILLs a sandbox over either
#   limit. Its agent server dies with it, so the rollout is cancelled and dropped
#   from the group as infrastructure-killed (openhands_rollout._cancel_if_server_dies).
SANDBOX_OOM_SCORE_ADJ = int(os.getenv("TERMINAL_RL_SANDBOX_OOM_SCORE_ADJ", "1000") or 0)
SANDBOX_MEM_LIMIT_GB = float(os.getenv("TERMINAL_RL_SANDBOX_MEM_LIMIT_GB", "0") or 0)
SANDBOX_PIDS_LIMIT = int(os.getenv("TERMINAL_RL_SANDBOX_PIDS_LIMIT", "0") or 0)
SANDBOX_WATCH_S = float(os.getenv("TERMINAL_RL_SANDBOX_WATCH_S", "5") or 5)


def sandbox_command(argv: list[str]) -> list[str]:
    """argv run with the sandbox oom_score_adj (inherited by everything it starts)."""
    if SANDBOX_OOM_SCORE_ADJ <= 0:
        return list(argv)
    script = f'echo {SANDBOX_OOM_SCORE_ADJ} > /proc/self/oom_score_adj 2>/dev/null; exec "$@"'
    return ["sh", "-c", script, "sh", *argv]


class SandboxWatchdog:
    """Kills sandboxes whose processes exceed the memory or process-count limit."""

    def __init__(self, mem_limit_gb: float, pids_limit: int, interval: float):
        self.mem_limit_kb = int(mem_limit_gb * 1024 * 1024)
        self.pids_limit = pids_limit
        self.interval = interval
        self._watched: dict[str, str] = {}  # instance name -> mount namespace
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._own_ns = _mount_ns("self")
        self._page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
        self.killed: list[tuple[str, int, int]] = []  # (name, rss kB, processes)

    @property
    def enabled(self) -> bool:
        return self.mem_limit_kb > 0 or self.pids_limit > 0

    def register(self, name: str, namespace: str | None) -> None:
        if not self.enabled or not namespace or namespace == self._own_ns:
            return
        with self._lock:
            self._watched[name] = namespace
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="sandbox-watchdog", daemon=True)
                self._thread.start()

    def unregister(self, name: str) -> None:
        with self._lock:
            self._watched.pop(name, None)

    def usage(self) -> dict[str, tuple[int, int, list[int]]]:
        """namespace -> (resident kB, processes, pids) for the watched namespaces."""
        with self._lock:
            wanted = set(self._watched.values())
        found: dict[str, list] = collections.defaultdict(lambda: [0, 0, []])
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            namespace = _mount_ns(entry)
            if namespace not in wanted:
                continue
            try:
                resident = int(open(f"/proc/{entry}/statm").read().split()[1])
            except (OSError, ValueError, IndexError):
                continue
            row = found[namespace]
            row[0] += resident * self._page_kb
            row[1] += 1
            row[2].append(int(entry))
        return {ns: (row[0], row[1], row[2]) for ns, row in found.items()}

    def check(self) -> list[str]:
        """One pass: kill every sandbox over a limit; returns their names."""
        usage = self.usage()
        with self._lock:
            watched = dict(self._watched)
        over = []
        for name, namespace in watched.items():
            rss_kb, processes, pids = usage.get(namespace, (0, 0, []))
            if (self.mem_limit_kb and rss_kb > self.mem_limit_kb) or (self.pids_limit and processes > self.pids_limit):
                logger.warning(
                    "[sandbox-watchdog] %s: %.1f GB resident, %d processes (limits %.0f GB, %d); killing it",
                    name, rss_kb / 1024 / 1024, processes, self.mem_limit_kb / 1024 / 1024, self.pids_limit,
                )
                for pid in pids:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGKILL)
                self.killed.append((name, rss_kb, processes))
                self.unregister(name)
                over.append(name)
        return over

    def _loop(self) -> None:
        while True:
            time.sleep(self.interval)
            try:
                self.check()
            except Exception as error:  # the guard must never take the client down
                logger.warning("[sandbox-watchdog] check failed: %s", error)


def _mount_ns(pid: str) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/ns/mnt")
    except OSError:
        return None


WATCHDOG = SandboxWatchdog(SANDBOX_MEM_LIMIT_GB, SANDBOX_PIDS_LIMIT, SANDBOX_WATCH_S)

# Apptainer instances do not auto-reap on parent death (unlike Docker --rm), so
# without this the tmpfs overlay of an abruptly killed rollout lingers.
_LIVE_INSTANCES: set[str] = set()


def _cleanup_all() -> None:
    for name in list(_LIVE_INSTANCES):
        with contextlib.suppress(Exception):
            subprocess.run(["apptainer", "instance", "stop", name], capture_output=True, timeout=10)
    _LIVE_INSTANCES.clear()


def _install_cleanup() -> None:
    """Reap instances on exit and on the signals that skip atexit.

    atexit does not run when Slurm cancels a job or a supervisor kills the
    client, which would strand every live instance and its tmpfs overlay.
    Previous handlers are chained so this does not swallow another one.
    """
    atexit.register(_cleanup_all)

    def handle(signum, frame):
        _cleanup_all()
        previous = _PREVIOUS_HANDLERS.get(signum)
        if callable(previous):
            previous(signum, frame)
        elif previous is signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        # Only the main thread may install handlers; workers inherit them.
        with contextlib.suppress(ValueError, OSError):
            _PREVIOUS_HANDLERS[signum] = signal.signal(signum, handle)


_PREVIOUS_HANDLERS: dict[int, object] = {}
_install_cleanup()


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


class SandboxStartError(RuntimeError):
    """The sandbox could not be started; the rollout cannot score this task."""


def sif_name_for_image(image: str) -> str:
    """Map a pinned Docker reference to its file name in the SIF pool.

    The pool publishes ``repo/name@sha256:digest`` as
    ``repo__name__sha256__digest.sif``; see the pool's README on tmax-compatible
    file names.
    """
    return image.replace("/", "__").replace("@sha256:", "__sha256__") + ".sif"


def resolve_image(
    image: str, sif_dir: str | os.PathLike[str] | None = None, sif_name: str | None = None
) -> str:
    """Return the local SIF path for a pinned image reference."""
    if image.endswith(".sif") or "://" in image:
        return image
    pool = Path(sif_dir or os.environ.get(SIF_DIR_ENV, "")).expanduser()
    if not str(pool):
        raise SandboxStartError(f"Set {SIF_DIR_ENV} to the staged image pool to resolve {image!r}.")
    path = pool / (sif_name or sif_name_for_image(image))
    if not path.is_file():
        raise SandboxStartError(f"No staged image for {image!r}; expected {path}. Stage it with prepare_litecoder.py.")
    return str(path)


# Images built from Harbor Dockerfiles (prepare_harbor.py) keep simulators and
# checkpoints under /opt, which Babel's apptainer.conf ("bind path = /opt")
# shadows with the host's. Drop the configured bind paths and keep /etc/hosts,
# which Docker would have written, so localhost still resolves.
HARBOR_START_FLAGS: tuple[str, ...] = ("--no-mount", "bind-paths", "--bind", "/etc/hosts")


class ApptainerBackend:
    """Runs commands in an Apptainer instance started from a local SIF.

    Per-rollout container state lives in an overlay: on disk when
    TERMINAL_RL_OVERLAY_DIR is set, otherwise in tmpfs (``--writable-tmpfs``),
    and ``--fakeroot`` so commands inside see uid 0 regardless of the host uid.
    All file I/O goes through ``apptainer exec`` with stdin/stdout piping -- no
    bind mounts -- so the container filesystem stays isolated from the host.
    """

    _MAX_OUTPUT_BYTES = 1_000_000
    # Where per-instance overlays live when they are kept off RAM. A sandbox's
    # writable layer runs to several GB (apt caches, cargo builds, task data),
    # and --writable-tmpfs puts that in shared memory, which counts against the
    # job's cgroup: 64 concurrent rollouts reached the 256 GB Slurm limit and
    # the kernel killed a Megatron actor rather than a sandbox, because Ray
    # marks its actors with oom_score_adj 1000. Pointing the overlay at local
    # disk takes the sandboxes out of the memory budget entirely.
    _OVERLAY_DIR_ENV = "TERMINAL_RL_OVERLAY_DIR"
    _DEFAULT_START_FLAGS: tuple[str, ...] = (
        "--fakeroot",
        "--containall",
        "--no-home",
        "--cleanenv",
        # Babel sets "mount hostfs = yes", so the host's autofs /home, /data and
        # /scratch are mounted over the image's. That shadows /home with the real
        # home tree, where the sandbox cannot mkdir: every tmax task works in
        # /home/user, so setup.sh failed with "Permission denied" for all of
        # them. It also kept host files reachable from agent commands.
        "--no-mount",
        "hostfs",
    )

    def __init__(
        self,
        image: str,
        timeout: int = 1800,
        pwd: str = "/workspace",
        cache_dir: str | None = None,
        tmp_dir: str | None = None,
        extra_start_flags: tuple[str, ...] = (),
        apptainer_binary: str = "apptainer",
    ):
        self._image = image
        self._timeout = timeout
        self._pwd = pwd
        self._cache_dir = cache_dir
        self._tmp_dir = tmp_dir
        self._overlay_root = os.environ.get(self._OVERLAY_DIR_ENV) or None
        self._overlay: str | None = None
        # Without an overlay directory the writable layer stays in tmpfs, which
        # is the historical behaviour and fine at low concurrency.
        writable = () if self._overlay_root else ("--writable-tmpfs",)
        self._start_flags = tuple(self._DEFAULT_START_FLAGS) + writable + tuple(extra_start_flags)
        self._apptainer = apptainer_binary
        self._name: str | None = None

    @property
    def image(self) -> str:
        return self._image

    def _exec_env(self, pwd: str | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env["APPTAINER_PWD"] = pwd or self._pwd
        if self._cache_dir:
            env["APPTAINER_CACHEDIR"] = self._cache_dir
        if self._tmp_dir:
            env["APPTAINER_TMPDIR"] = self._tmp_dir
        return env

    def _ensure_started(self) -> None:
        if self._name is None:
            raise RuntimeError("Instance not started. Call start() first.")

    def start(self) -> None:
        if shutil.which(self._apptainer) is None:
            raise SandboxStartError(f"Apptainer binary {self._apptainer!r} not found on PATH.")
        if self._name is not None:
            self.close()
        name = f"litecoder-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        flags = list(self._start_flags)
        if self._overlay_root:
            self._overlay = os.path.join(self._overlay_root, name)
            os.makedirs(os.path.join(self._overlay, "upper"), exist_ok=True)
            os.makedirs(os.path.join(self._overlay, "work"), exist_ok=True)
            flags += ["--overlay", self._overlay]
        command = sandbox_command([self._apptainer, "instance", "start", *flags, self._image, name])
        proc = subprocess.run(command, capture_output=True, env=self._exec_env())
        if proc.returncode != 0:
            raise SandboxStartError(
                f"apptainer instance start failed (image={self._image}, exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        self._name = name
        _LIVE_INSTANCES.add(name)
        if WATCHDOG.enabled:
            WATCHDOG.register(name, self._instance_namespace(name))
        # APPTAINER_PWD applies to every exec and apptainer refuses to start one
        # whose working directory is missing. Task images ship /app, not
        # /workspace, so create it before any other command runs.
        created = self._exec(["sh", "-c", f"mkdir -p {shlex.quote(self._pwd)}"], pwd="/")
        if created.returncode != 0:
            self.close()
            raise SandboxStartError(
                f"could not create {self._pwd} in {self._image}: {created.stderr.decode('utf-8', 'replace').strip()}"
            )
        logger.debug("Apptainer instance started: %s (%s)", name, self._image)

    def _instance_namespace(self, name: str) -> str | None:
        """Mount namespace of the instance's init process (shared by every exec)."""
        try:
            listed = subprocess.run(
                [self._apptainer, "instance", "list", "--json", name],
                capture_output=True, text=True, timeout=60, env=self._exec_env(),
            )
            for instance in json.loads(listed.stdout or "{}").get("instances", []):
                if instance.get("instance") == name:
                    return _mount_ns(str(instance["pid"]))
        except Exception as error:
            logger.warning("[sandbox-watchdog] could not find the namespace of %s: %s", name, error)
        return None

    def close(self) -> None:
        if self._name is None:
            return
        name, self._name = self._name, None
        WATCHDOG.unregister(name)
        with contextlib.suppress(Exception):
            subprocess.run([self._apptainer, "instance", "stop", name], capture_output=True, timeout=30)
        # The overlay outlives the instance, so a rollout that does not clean up
        # leaks several GB of disk per episode.
        if self._overlay:
            with contextlib.suppress(Exception):
                shutil.rmtree(self._overlay, ignore_errors=True)
            self._overlay = None
        _LIVE_INSTANCES.discard(name)
        logger.debug("Apptainer instance stopped: %s", name)

    def _exec(self, argv: list[str], stdin: bytes | None = None, pwd: str | None = None) -> subprocess.CompletedProcess:
        self._ensure_started()
        # --cleanenv per exec, not just per instance: without it Apptainer
        # forwards the launcher's environment into every command, so the client's
        # UV_PROJECT_ENVIRONMENT, VIRTUAL_ENV and API keys reach the agent's shell
        # and break task verifiers that build their own uv project.
        command = sandbox_command([self._apptainer, "exec", "--cleanenv", f"instance://{self._name}", *argv])
        return subprocess.run(command, input=stdin, capture_output=True, env=self._exec_env(pwd))

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        effective_timeout = self._timeout if timeout is None else timeout
        wrapped = (
            f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(effective_timeout))} "
            f"bash -c {shlex.quote(command)}"
        )
        proc = self._exec(["bash", "-c", wrapped])
        stdout = proc.stdout[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = proc.stderr[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if proc.returncode == 124:
            stderr = f"Command timed out after {effective_timeout}s.\n" + stderr
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)

    def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        directory = os.path.dirname(path) or "/"
        proc = self._exec(
            ["sh", "-c", f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}"], stdin=content
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"write_file failed for {path!r} (exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )

    def get_archive(self, paths: list[str]) -> bytes:
        """Tar the given absolute paths (those that exist), relative to /."""
        listing = " ".join(shlex.quote(p.lstrip("/")) for p in paths if p.strip("/"))
        script = f'cd / && for p in {listing}; do [ -e "$p" ] && printf "%s\\0" "$p"; done | tar --null -cf - -T -'
        proc = self._exec(["sh", "-c", script], pwd="/")
        if proc.returncode != 0:
            raise RuntimeError(
                f"get_archive failed for {paths!r} (exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        return proc.stdout

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        proc = self._exec(["tar", "-xf", "-", "-C", root], stdin=tar_bytes)
        if proc.returncode != 0:
            raise RuntimeError(
                f"put_archive failed at root={root!r} (exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
