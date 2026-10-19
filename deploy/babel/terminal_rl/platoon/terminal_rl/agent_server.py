"""Run openhands-agent-server inside a task's Apptainer sandbox.

openhands' TerminalTool resolves ``conv_state.workspace.working_dir`` and then
spawns a shell locally, so an agent hosted on the compute node would run the
task's commands outside the sandbox. Hosting the server in the container is the
only arrangement where the agent's action space is the task's container, which
is what the terminal-bench evaluation does too.

Apptainer shares the host's network namespace, so a server bound inside the
sandbox is reachable on 127.0.0.1 from the client process -- and, in the other
direction, the agent can reach the Tinker proxy bridge the same way. Nothing is
published beyond loopback.

The image must carry the runtime at /openhands-venv; see
<science-agent>/experiments/qwen38_27b_tinker_amd/pool/convert_openhands_images.py.
"""

import asyncio
import shlex
import logging
import os
import socket
import subprocess
from contextlib import suppress

from platoon.terminal_rl.backend import sandbox_command

logger = logging.getLogger("platoon.terminal_rl.agent_server")

VENV_PYTHON = "/openhands-venv/bin/python"
# Long enough for a cold import of the SDK on a loaded node; the server is
# otherwise quick, and a rollout that waits here is not sampling.
STARTUP_TIMEOUT = float(os.getenv("TERMINAL_RL_AGENT_SERVER_TIMEOUT", "300"))
# After /health answers, how long to watch our own process for a late bind failure
# (the answer may have come from another sandbox's server on the same port).
BIND_GRACE = float(os.getenv("TERMINAL_RL_AGENT_SERVER_BIND_GRACE", "2"))
# Host variables the in-container server needs. The SDK refuses to run an LLM
# whose context window is under MIN_CONTEXT_WINDOW_TOKENS (16384) unless this
# override is set, and the condenser budget doubles as that window.
FORWARDED_ENV = ("ALLOW_SHORT_CONTEXT_WINDOWS",)

# The model name to register as a thinking model inside the sandbox. Empty
# disables the edit and openhands keeps dropping reasoning between turns.
RETAIN_REASONING_MODEL = os.getenv("TERMINAL_RL_RETAIN_REASONING_MODEL", "")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class PortInUse(RuntimeError):
    """The chosen port was claimed by someone else before the server bound it."""


class SandboxAgentServer:
    """An openhands-agent-server process living in one rollout's sandbox."""

    def __init__(self, backend, port: int | None = None, log_path: str | None = None):
        self._backend = backend
        self.port = port or free_port()
        self._log_path = log_path
        self._process: subprocess.Popen | None = None
        self._log = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _spawn(self) -> subprocess.Popen:
        name = self._backend._name
        if not name:
            raise RuntimeError("the sandbox is not started; call executor.reset() first")
        # APPTAINER_PWD applies to every exec and apptainer refuses one whose
        # working directory is missing, so anchor at / rather than inheriting
        # the client's cwd, which does not exist inside the container.
        environment = dict(os.environ, APPTAINER_PWD="/")
        # `apptainer exec --cleanenv` drops the host environment, so anything
        # the server needs has to be forwarded with an APPTAINERENV_ prefix.
        for variable in FORWARDED_ENV:
            if os.environ.get(variable):
                environment[f"APPTAINERENV_{variable}"] = os.environ[variable]
        if self._log_path:
            # The rollout's events directory is created by the trajectory sink,
            # which only runs once the conversation starts -- after this.
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._log = open(self._log_path, "wb") if self._log_path else subprocess.DEVNULL
        return subprocess.Popen(
            sandbox_command([
                "apptainer", "exec", "--cleanenv", f"instance://{name}",
                VENV_PYTHON, "-m", "openhands.agent_server",
                "--host", "0.0.0.0", "--port", str(self.port),
            ]),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=environment,
        )

    async def start(self, attempts: int = 4) -> "SandboxAgentServer":
        """Start the server, retrying on a lost port race.

        free_port() finds a port by binding and releasing it, so between that
        release and the server's own bind another rollout can claim the same
        number -- with dozens of sandboxes starting at once that happens often
        enough to lose a few percent of episodes to "address already in use".
        Retrying with a fresh port costs a second and recovers the rollout.
        """
        for attempt in range(attempts):
            try:
                return await self._start_once()
            except PortInUse:
                await self.close()
                self.port = free_port()
                logger.warning(
                    "agent-server port was taken; retrying on %d (attempt %d/%d)",
                    self.port, attempt + 2, attempts,
                )
        raise RuntimeError(f"agent-server could not claim a free port after {attempts} attempts")

    # openhands only echoes reasoning back to the model for names it recognises
    # as thinking models (SEND_REASONING_CONTENT_MODELS), and the check is a bare
    # substring match that ignores the LLM's capability_overrides. The agent runs
    # inside the sandbox, so a host-side monkeypatch cannot reach it and the only
    # alternative would be to misname the model after a Kimi checkpoint. Adding
    # our own name to that list is the honest version of the same thing: one exec
    # of about 0.07s against the sandbox's own copy, which the overlay discards.
    _FEATURES = "/openhands-venv/lib/python3.12/site-packages/openhands/sdk/llm/utils/model_features.py"

    def _allow_reasoning_resend(self, model_name: str) -> None:
        """Let the in-sandbox openhands send reasoning_content back to the model."""
        script = (
            f"grep -q '\"{model_name}\"' {self._FEATURES} || "
            f"sed -i 's|^SEND_REASONING_CONTENT_MODELS: list\\[str\\] = \\[|"
            f"&\\n    \"{model_name}\",|' {self._FEATURES}"
        )
        try:
            self._backend.run_command(f"bash -lc {shlex.quote(script)}")
        except Exception as error:  # reasoning retention is an optimisation
            logger.warning("could not enable reasoning resend in the sandbox: %s", error)

    async def _start_once(self) -> "SandboxAgentServer":
        import httpx

        if RETAIN_REASONING_MODEL:
            await asyncio.to_thread(self._allow_reasoning_resend, RETAIN_REASONING_MODEL)
        self._process = await asyncio.to_thread(self._spawn)
        deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT
        async with httpx.AsyncClient(timeout=5) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._process.poll() is not None:
                    if self._log_path and self._bind_failed():
                        raise PortInUse(self.port)
                    raise RuntimeError(
                        f"agent-server exited {self._process.returncode}"
                        + (f"; see {self._log_path}" if self._log_path else "")
                    )
                healthy = False
                with suppress(Exception):
                    response = await client.get(f"{self.url}/health")
                    healthy = response.status_code == 200
                if healthy:
                    # The port is shared with every other sandbox on the node (Apptainer
                    # uses the host's network), so a 200 here can come from *another*
                    # rollout's server that won the port: ours then fails its bind and
                    # exits a moment later (job 10825533: "address already in use",
                    # exit 3, reported as "agent-server died" 3 s into the episode).
                    # Confirm our own process survived and did not log a bind failure.
                    await asyncio.sleep(BIND_GRACE)
                    if self._process.poll() is not None or (self._log_path and self._bind_failed()):
                        if self._log_path and self._bind_failed():
                            raise PortInUse(self.port)
                        raise RuntimeError(
                            f"agent-server exited {self._process.returncode} after answering /health"
                            + (f"; see {self._log_path}" if self._log_path else "")
                        )
                    logger.info("agent-server ready on port %d", self.port)
                    return self
                await asyncio.sleep(1)
        await self.close()
        raise RuntimeError(f"agent-server did not become healthy within {STARTUP_TIMEOUT:.0f}s on {self.url}")

    def is_alive(self) -> bool:
        """Whether the in-sandbox server process is still running."""
        return self._process is not None and self._process.poll() is None

    def _bind_failed(self) -> bool:
        """Whether the server died because its port was already taken."""
        try:
            with open(self._log_path, "rb") as log:
                return b"address already in use" in log.read().lower()
        except Exception:
            return False

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            with suppress(Exception):
                await asyncio.to_thread(process.wait, 15)
            if process.poll() is None:
                process.kill()
        if self._log not in (None, subprocess.DEVNULL):
            with suppress(Exception):
                self._log.close()
        self._log = None
