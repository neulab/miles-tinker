"""A Platoon CodeExecutor that runs bash in a per-rollout Apptainer sandbox.

The sandbox lifecycle, the persistent-shell wrapper, the seed/test upload split
and the reward file follow tmax's ``SWERLVanilluxSandboxEnv``. Tests are
uploaded only at submission time, so the agent cannot read them while working.

Every backend call is blocking, so each one is handed to a worker thread:
Platoon runs many rollouts concurrently on one event loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import os
import shlex
import tarfile
from pathlib import Path

from platoon.envs.base import Task
from platoon.envs.codeact import CodeActStep
from platoon.episode.context import finish_message

from platoon.terminal_rl.backend import HARBOR_START_FLAGS, ApptainerBackend, SandboxStartError, resolve_image
from platoon.terminal_rl.prompts import (
    SUBMIT_MARKER,
    format_error_message,
    render_instance,
    truncate_observation,
)

logger = logging.getLogger("platoon.terminal_rl.executor")

# Reward shape. "binary" keeps each task's own 1/0 from /logs/verifier/reward.txt.
# "passrate" instead scores the fraction of the task's unit tests that pass.
#
# The tasks all use the same template -- `pytest /tmp/test_final_state.py -x` and
# an if/else writing 1 or 0 -- so under binary scoring a rollout that satisfies 2
# of 3 tests is indistinguishable from one that did nothing. That matters here
# because GRPO drops any group whose members all score alike, and 54% of groups
# in job 10635972 were all-0.2 with nobody solving: no variance, no gradient.
# Grading by tests passed gives those groups something to differentiate.
#
# -x has to go as well, or pytest stops at the first failure and the counts are
# truncated rather than wrong-but-complete.
REWARD_MODE = os.getenv("TERMINAL_RL_REWARD_MODE", "binary")
_PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|error|errors)")


_WRAPPER_PATH = "/tmp/.litecoder_bash_wrapper.sh"
_WRAPPER_PATH_QUOTED = shlex.quote(_WRAPPER_PATH)
_CWD_PATH = "/tmp/.litecoder_cwd"
_ENV_PATH = "/tmp/.litecoder_env"
# Variables owned by the container runtime, not by the agent's shell. FAKEROOTKEY
# in particular is minted per `apptainer exec`: restoring a previous exec's key
# makes libfakeroot block forever on a faked daemon that no longer exists, which
# hangs every command after the agent's first `export`. tmax's wrapper persists
# the environment unfiltered because it runs Docker, where this does not arise.
_RUNTIME_VARIABLES = (
    "FAKEROOT[A-Z_]*",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "APPTAINER_[A-Z_]*",
    "SINGULARITY_[A-Z_]*",
    "PWD",
    "OLDPWD",
    "SHLVL",
    "_",
)
_RUNTIME_FILTER = f"^declare -x ({'|'.join(_RUNTIME_VARIABLES)})(=|$)"

# Persist cwd and exported variables between calls, so the agent sees one shell.
_WRAPPER = f"""#!/bin/bash
set -a
source {shlex.quote(_ENV_PATH)} 2>/dev/null || true
set +a
_cwd="$(cat {shlex.quote(_CWD_PATH)} 2>/dev/null || echo /app)"
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p | grep -vE {shlex.quote(_RUNTIME_FILTER)} > {shlex.quote(_ENV_PATH)}
pwd > {shlex.quote(_CWD_PATH)}
exit $_exit_code
"""


def _tar_directory(host_dir: Path, container_dir: str) -> bytes:
    """Pack a host directory for extraction at the container root."""
    prefix = container_dir.lstrip("/")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        root_info = tarfile.TarInfo(name=prefix)
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        tar.addfile(root_info)
        for root, directories, files in os.walk(host_dir):
            relative_root = os.path.relpath(root, host_dir)
            for directory in directories:
                name = directory if relative_root == "." else os.path.join(relative_root, directory)
                info = tarfile.TarInfo(name=f"{prefix}/{name}")
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            for file_name in files:
                name = file_name if relative_root == "." else os.path.join(relative_root, file_name)
                tar.add(os.path.join(root, file_name), arcname=f"{prefix}/{name}")
    return stream.getvalue()


def _tar_file(host_file: Path, container_path: str) -> bytes:
    """Pack one host file for extraction at the container root, keeping its mode."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        tar.add(host_file, arcname=container_path.lstrip("/"))
    return stream.getvalue()


class ApptainerBashExecutor:
    """Runs one task's rollout inside its own Apptainer instance."""

    def __init__(
        self,
        task: Task,
        sif_dir: str | None = None,
        command_timeout: int = 120,
        test_timeout: int = 600,
        setup_timeout: int = 900,
        cache_dir: str | None = None,
        tmp_dir: str | None = None,
    ):
        self._task = task
        self._sif_dir = sif_dir
        self._command_timeout = command_timeout
        self._test_timeout = test_timeout
        self._setup_timeout = setup_timeout
        self._cache_dir = cache_dir
        self._tmp_dir = tmp_dir
        self._backend: ApptainerBackend | None = None
        self._task_dir = Path(task.misc["task_dir"])
        self._instruction: str = task.misc.get("instruction", "")
        self._tests_dir: Path | None = None
        self._pending_reward: float | None = None
        self._submitted = False

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def submitted(self) -> bool:
        return self._submitted

    @property
    def backend(self) -> ApptainerBackend:
        """The running sandbox, for harnesses that drive it themselves."""
        if self._backend is None:
            raise RuntimeError("Executor not reset. Call reset() first.")
        return self._backend

    def take_reward(self) -> float:
        """Pop the verifier's reward; 0.0 until the agent submits."""
        reward, self._pending_reward = self._pending_reward, None
        return 0.0 if reward is None else reward

    async def describe_action_space(self) -> str:
        return render_instance(self._instruction or f"[Task: {self._task.id}] Sandbox ready.")

    async def reset(self) -> "ApptainerBashExecutor":
        await self.close()
        # Pools name their SIFs in the manifest; prefer that over deriving a name
        # from the image reference, which differs between digest and tag pools.
        image = resolve_image(self._task.misc["image"], self._sif_dir, self._task.misc.get("sif_name"))
        backend = ApptainerBackend(
            image=image,
            timeout=self._command_timeout,
            cache_dir=self._cache_dir,
            tmp_dir=self._tmp_dir,
            extra_start_flags=HARBOR_START_FLAGS if self._task.misc.get("harbor") else (),
        )
        await asyncio.to_thread(backend.start)
        self._backend = backend
        self._pending_reward = None
        self._submitted = False
        await self._load_task_data()
        await self._prepare_shell()
        return self

    async def _configure_apt(self) -> None:
        """Make apt work under --fakeroot, as it does under Docker's real root.

        apt drops privileges to the `_apt` user to download, which cannot write
        /var/cache/apt/archives/partial through the fakeroot overlay: every
        install fails with exit 100. Task verifiers apt-get their own tooling
        before running the tests, so without this no task can score above 0.
        Redirecting apt's cache and log to /tmp restores the intended behaviour
        without touching any task's own scripts.
        """
        assert self._backend is not None
        await asyncio.to_thread(
            self._backend.run_command,
            "mkdir -p /tmp/apt-archives/partial /tmp/apt-log && "
            'printf \'APT::Sandbox::User "root";\\nDir::Cache::archives "/tmp/apt-archives/";\\n'
            'Dir::Log "/tmp/apt-log/";\\n\' > /etc/apt/apt.conf.d/99-litecoder-fakeroot',
        )

    async def _load_task_data(self) -> None:
        assert self._backend is not None
        await asyncio.to_thread(self._backend.run_command, "mkdir -p /workspace /output /logs/verifier")
        await self._configure_apt()
        seeds = self._task_dir / "environment" / "seeds"
        if seeds.is_dir():
            archive = await asyncio.to_thread(_tar_directory, seeds, "/workspace")
            await asyncio.to_thread(self._backend.put_archive, "/", archive)
        await self._upload_inputs(self._backend, self._task.misc.get("inputs") or [])
        await self._run_setup()
        tests = self._task_dir / "tests"
        self._tests_dir = tests if tests.is_dir() else None

    async def _run_setup(self) -> None:
        """Run the task's setup.sh, when it ships one.

        tmax's swerl_sandbox skips setup.sh, on the grounds that task images are
        "prebuilt with dependencies and setup already applied at image build
        time". For allenai/tmax-15k-open-instruct that holds only for some
        images: others carry none of the files their setup.sh creates, nor the
        pip packages their tests import, and would score 0 for every rollout.
        Running it is therefore the safe default, and a non-zero exit is logged
        rather than raised, because on an already-prepared image setup.sh fails
        harmlessly (duplicate inserts, "user already exists") while the
        environment is correct. LiteCoder pools ship no setup.sh and are
        unaffected.
        """
        assert self._backend is not None
        setup = self._task_dir / "setup.sh"
        if not setup.is_file():
            return
        await asyncio.to_thread(self._backend.write_file, "/tmp/.terminal_rl_setup.sh", setup.read_text())
        result = await asyncio.to_thread(
            self._backend.run_command, "bash /tmp/.terminal_rl_setup.sh", self._setup_timeout
        )
        if result.exit_code != 0:
            logger.warning(
                "[%s] setup.sh exited %s; the task may be unsolvable: %s",
                self._task.id,
                result.exit_code,
                (result.stderr or result.stdout).strip()[-300:],
            )

    async def _prepare_shell(self) -> None:
        assert self._backend is not None
        await asyncio.to_thread(
            self._backend.run_command,
            "mkdir -p /workspace /root && cd /workspace && "
            '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
            f"printf '%s\\n' /app > {shlex.quote(_CWD_PATH)} && : > {shlex.quote(_ENV_PATH)}",
        )
        await asyncio.to_thread(self._backend.write_file, _WRAPPER_PATH, _WRAPPER)
        await asyncio.to_thread(self._backend.run_command, f"chmod +x {_WRAPPER_PATH_QUOTED}")

    async def run(self, code: str) -> CodeActStep:
        if self._backend is None:
            raise RuntimeError("Executor not reset. Call reset() first.")
        command = (code or "").strip()
        if not command:
            return CodeActStep(code=code, error=format_error_message("No <bash> block was provided."))
        if self._submitted:
            return CodeActStep(code=code, error="The task was already submitted; no further commands are accepted.")

        result = await asyncio.to_thread(
            self._backend.run_command, f"bash {_WRAPPER_PATH_QUOTED} {shlex.quote(command)}"
        )
        output = result.stdout or ""
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr

        if SUBMIT_MARKER in output:
            return await self._submit(code)

        observation = truncate_observation(output) if output else "(no output)"
        return CodeActStep(code=code, output=f"{observation}\n\n(exit_code={result.exit_code})")

    async def score(self) -> tuple[float, str]:
        """Upload the tests, run them and read the verifier's reward.

        Deferred until now so the agent cannot read the tests while working.
        Returns the reward and the verifier's output, so any harness -- the
        CodeAct submit marker or an openhands conversation that has finished --
        can score a task the same way.
        """
        assert self._backend is not None
        self._submitted = True
        verifier = self._task.misc.get("verifier")
        if verifier:
            return await self._score_in_verifier(verifier)
        if self._tests_dir is None:
            return 0.0, f"No test data for task {self._task.id}."

        archive = await asyncio.to_thread(_tar_directory, self._tests_dir, "/tests")
        await asyncio.to_thread(self._backend.put_archive, "/", archive)
        await asyncio.to_thread(self._backend.run_command, "chmod +x /tests/test.sh 2>/dev/null || true")
        check = await asyncio.to_thread(self._backend.run_command, "test -f /tests/test.sh && echo EXISTS")
        if check.stdout.strip() != "EXISTS":
            return 0.0, f"No test.sh in the test data for task {self._task.id}."

        if REWARD_MODE == "passrate":
            # Drop -x so every test runs; the task's own if/else still writes
            # reward.txt, which is kept as a fallback when the counts are absent.
            await asyncio.to_thread(
                self._backend.run_command, "sed -i 's/ -x / /g; s/ -x$/ /g' /tests/test.sh"
            )
        # A task that names its working directory (TB2: the image's WORKDIR) runs its
        # tests from there, as its own harness does; others keep the sandbox default.
        workdir = (self._task.misc or {}).get("workdir")
        command = f"cd {shlex.quote(workdir)} && bash /tests/test.sh" if workdir else "bash /tests/test.sh"
        result = await asyncio.to_thread(self._backend.run_command, command, self._test_timeout)
        reward = await self._read_reward(self._backend)
        if REWARD_MODE == "passrate":
            graded = self._pass_rate(result.stdout or "")
            if graded is not None:
                reward = graded
        stdout = truncate_observation(result.stdout) if result.stdout else ""
        stderr = truncate_observation(result.stderr) if result.stderr else ""
        observation = (
            f"Test execution complete.\n"
            f"Exit code: {result.exit_code}\n"
            f"[STDOUT]\n{stdout}\n"
            f"[STDERR]\n{stderr}\n"
            f"Reward: {reward}"
        )
        return reward, observation

    async def _submit(self, code: str) -> CodeActStep:
        """Score the task in response to the agent's submit marker."""
        reward, observation = await self.score()
        self._pending_reward = reward
        finish_message.set(f"Submitted task {self._task.id}; reward {reward}.")
        if self._tests_dir is None or observation.startswith("No test"):
            return CodeActStep(code=code, error=observation)
        return CodeActStep(code=code, output=observation)

    def _pass_rate(self, stdout: str) -> float | None:
        """Fraction of unit tests that passed, from pytest's summary line.

        Returns None when no counts can be found, so the caller keeps the task's
        own binary reward rather than silently scoring a broken verifier as 0.
        """
        counts = {"passed": 0, "failed": 0, "error": 0, "errors": 0}
        for number, kind in _PYTEST_COUNT.findall(stdout):
            counts[kind] = max(counts[kind], int(number))
        passed = counts["passed"]
        total = passed + counts["failed"] + max(counts["error"], counts["errors"])
        if total == 0:
            return None
        rate = passed / total
        logger.info("[%s] pass rate %d/%d = %.3f", self._task.id, passed, total, rate)
        return max(0.0, min(1.0, rate))

    async def _upload_inputs(self, backend: ApptainerBackend, inputs: list[dict]) -> None:
        """Upload a task's run-time files (prepare_harbor.py's peeled COPYs) to their destinations."""
        for item in inputs:
            source = self._task_dir / item["path"]
            if item["kind"] == "dir":
                archive = await asyncio.to_thread(_tar_directory, source, item["dest"])
            else:
                archive = await asyncio.to_thread(_tar_file, source, item["dest"])
            await asyncio.to_thread(backend.put_archive, "/", archive)

    async def _score_in_verifier(self, verifier: dict) -> tuple[float, str]:
        """Harbor's separate verifier: grade the agent's artifacts in the tests image.

        The declared artifact paths are copied out of the agent's sandbox into a
        fresh sandbox started from the verifier image, which alone holds the
        grader and its key; the agent's sandbox never sees them.
        """
        assert self._backend is not None
        artifacts = await asyncio.to_thread(self._backend.get_archive, verifier.get("artifacts") or [])
        workdir = verifier.get("workdir") or "/"
        grader = ApptainerBackend(
            image=resolve_image("harbor-verifier", self._sif_dir, verifier["sif_name"]),
            timeout=self._test_timeout,
            pwd=workdir,
            cache_dir=self._cache_dir,
            tmp_dir=self._tmp_dir,
            extra_start_flags=HARBOR_START_FLAGS,
        )
        try:
            await asyncio.to_thread(grader.start)
            await self._upload_inputs(grader, verifier.get("inputs") or [])
            if artifacts:
                await asyncio.to_thread(grader.put_archive, "/", artifacts)
            await asyncio.to_thread(grader.run_command, "mkdir -p /logs/verifier")
            result = await asyncio.to_thread(
                grader.run_command, f"cd {shlex.quote(workdir)} && bash /tests/test.sh", self._test_timeout
            )
            reward = await self._read_reward(grader)
        finally:
            await asyncio.to_thread(grader.close)
        stdout = truncate_observation(result.stdout) if result.stdout else ""
        stderr = truncate_observation(result.stderr) if result.stderr else ""
        observation = (
            f"Test execution complete.\nExit code: {result.exit_code}\n"
            f"[STDOUT]\n{stdout}\n[STDERR]\n{stderr}\nReward: {reward}"
        )
        return reward, observation

    async def _read_reward(self, backend: ApptainerBackend) -> float:
        """Read /logs/verifier/reward.txt, clamped to [0, 1]; 0.0 if absent."""
        result = await asyncio.to_thread(backend.run_command, "cat /logs/verifier/reward.txt")
        if result.exit_code != 0 or not result.stdout.strip():
            return 0.0
        try:
            return max(0.0, min(1.0, float(result.stdout.strip())))
        except ValueError:
            logger.warning("[%s] unparseable reward: %r", self._task.id, result.stdout[:100])
            return 0.0

    async def close(self) -> None:
        if self._backend is not None:
            backend, self._backend = self._backend, None
            await asyncio.to_thread(backend.close)


__all__ = ["ApptainerBashExecutor", "SandboxStartError"]
