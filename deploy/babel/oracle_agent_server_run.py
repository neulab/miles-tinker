"""Replay a task's ground-truth solution as a real openhands conversation.

The point is fidelity, not scoring: every step here is a genuine openhands
tool call, chosen by an *oracle LLM* -- a local OpenAI-compatible server that
ignores the prompt and returns a scripted tool call for turn N. Everything
downstream of the LLM is the real thing:

  * the agent runs inside the task's Apptainer sandbox, under
    ``openhands-agent-server`` from the converted image;
  * TerminalTool, FileEditorTool and TaskTrackerTool execute in the container,
    so the trajectory shows the same action space the terminal-bench
    evaluation offers;
  * the reward comes from the task's own verifier through
    ``ApptainerBashExecutor.score``, the same call the RL env makes.

That makes the saved trajectory a ceiling: what a perfect policy's rollout
looks like end to end, and proof that the converted image can carry one.

Usage:
    python oracle_agent_server_run.py \
        --task task_009170_beee8951 \
        --steps <science-agent>/experiments/qwen38_27b_tinker_amd/pool/oracle_steps/task_009170_beee8951.json \
        --sif /data/user_data/$USER/oh-convert/converted.sif \
        --task-dir /data/user_data/$USER/tmax15k/tasks \
        --out /data/user_data/$USER/trajectories
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# SIGUSR1 dumps every thread's stack, so a hung sandbox or agent call can be
# diagnosed without restarting the run.
faulthandler.register(signal.SIGUSR1, all_threads=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("oracle")

# The oracle answers as a model litellm already knows, so no registry lookups
# fail and tool calling is taken to be native; the base_url sends it here.
ORACLE_MODEL = "openai/gpt-4o"

# Logical step names in the steps file -> the tool's function name as the agent
# advertises it. The server names tools in snake_case ("terminal"), while the
# Tool spec uses the class name ("TerminalTool"), so match on substrings rather
# than pinning either spelling.
TOOL_ALIASES = {
    "terminal": ("terminal", "execute_bash", "bash"),
    "file_editor": ("file_editor", "str_replace_editor", "edit"),
    "task_tracker": ("task_tracker", "tracker"),
    "finish": ("finish",),
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OracleServer:
    """An OpenAI-compatible endpoint that replays a scripted list of tool calls.

    The turn index is the number of assistant messages already in the request,
    so the server keeps no state and a retried request replays the same step.
    """

    def __init__(self, steps: list[dict], port: int):
        self.steps = steps
        self.port = port
        self.served: list[dict] = []
        oracle = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # keep the driver's output readable
                pass

            def _send(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send({"object": "list", "data": [{"id": ORACLE_MODEL, "object": "model"}]})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
                try:
                    self._send(oracle.respond(request))
                except Exception as error:  # surface it, do not hang the agent
                    logger.exception("oracle failed")
                    self._send({"error": {"message": str(error), "type": "oracle_error"}}, 500)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        logger.info("oracle LLM listening on http://127.0.0.1:%d/v1", self.port)

    def stop(self) -> None:
        # shutdown() waits for serve_forever to acknowledge, so calling it on a
        # server that never started hangs the process instead of reporting
        # whatever went wrong before start().
        if self._thread.is_alive():
            self._server.shutdown()
        self._server.server_close()

    def _resolve_tool(self, logical: str, request: dict) -> str:
        names = [
            tool.get("function", {}).get("name", "")
            for tool in request.get("tools") or []
            if isinstance(tool, dict)
        ]
        for candidate in TOOL_ALIASES.get(logical, (logical,)):
            for name in names:
                if candidate in name.lower():
                    return name
        raise RuntimeError(f"agent advertises no tool matching {logical!r}; it has {names}")

    def respond(self, request: dict) -> dict:
        messages = request.get("messages") or []
        if not request.get("tools"):
            # The agent-server also asks the LLM for a conversation title, with
            # no tools attached. Those calls must not consume a scripted step.
            return self._completion(request, "Ground-truth replay", tool_call=None, turn=-1)
        turn = sum(1 for message in messages if message.get("role") == "assistant")
        if turn >= len(self.steps):
            # The script ran out: stop cleanly rather than loop.
            step = {"thought": "Nothing left to do.", "tool": "finish", "args": {"message": "Done."}}
        else:
            step = self.steps[turn]
        name = self._resolve_tool(step["tool"], request)
        call_id = f"call_oracle_{turn:03d}"
        logger.info("turn %d -> %s(%s)", turn, name, json.dumps(step["args"])[:120])
        self.served.append({"turn": turn, "tool": name, "args": step["args"]})
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(step["args"])},
        }
        return self._completion(request, step.get("thought", ""), tool_call, turn)

    @staticmethod
    def _completion(request: dict, content: str, tool_call: dict | None, turn: int) -> dict:
        message = {"role": "assistant", "content": content}
        if tool_call is not None:
            message["tool_calls"] = [tool_call]
        return {
            "id": f"chatcmpl-oracle-{turn}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", ORACLE_MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_call else "stop",
                }
            ],
            # litellm insists on a usage block; the numbers are not used for
            # anything here, and a real run gets them from the Tinker proxy.
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def start_agent_server(instance: str, port: int, log_path: Path) -> subprocess.Popen:
    """Run openhands-agent-server inside the already-started sandbox."""
    env = dict(os.environ, APPTAINER_PWD="/")
    handle = log_path.open("wb")
    process = subprocess.Popen(
        [
            "apptainer", "exec", "--cleanenv", f"instance://{instance}",
            "/openhands-venv/bin/python", "-m", "openhands.agent_server",
            "--host", "0.0.0.0", "--port", str(port),
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    import urllib.request

    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"agent-server exited {process.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    logger.info("agent-server healthy on port %d", port)
                    return process
        except Exception:
            time.sleep(2)
    process.kill()
    raise RuntimeError(f"agent-server did not become healthy; see {log_path}")


# The terminal tool returns a real PTY's output, bracketed-paste markers and
# colour codes included; they are noise in a saved transcript.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def _text_of(parts) -> str:
    return "".join(part.get("text", "") for part in parts or [] if isinstance(part, dict))


def render_events(events: list[dict], instruction: str = "") -> tuple[str, list[dict]]:
    """Turn the conversation's event log into a readable transcript + tool calls."""
    lines: list[str] = []
    calls: list[dict] = []
    pending: dict[str, dict] = {}
    for event in events:
        kind = event.get("kind", "")
        if kind == "MessageEvent":
            message = event.get("llm_message") or {}
            text = _text_of(message.get("content")).strip()
            if text and text != instruction.strip():
                lines.append(f"\n[{message.get('role', '?')} message]\n{text}")
        elif kind == "ActionEvent":
            # tool_call carries the arguments the model actually emitted, as a
            # JSON string; action is the same thing parsed and validated, and
            # is the better record when the two can differ (defaults filled in).
            call = event.get("tool_call") or {}
            try:
                arguments = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": call.get("arguments")}
            action = {k: v for k, v in (event.get("action") or {}).items() if k != "kind"}
            record = {
                "call_id": event.get("tool_call_id") or call.get("id"),
                "tool": event.get("tool_name") or call.get("name"),
                "thought": _text_of(event.get("thought")).strip(),
                "arguments": arguments or action,
                "action": action,
                "observation": None,
            }
            calls.append(record)
            if record["call_id"]:
                pending[record["call_id"]] = record
            lines.append(
                f"\n{'=' * 72}\nTOOL CALL {len(calls)}: {record['tool']}\n"
                f"--- thought ---\n{record['thought'] or '(none)'}\n"
                f"--- arguments ---\n{json.dumps(record['arguments'], indent=2)}"
            )
        elif kind in ("ObservationEvent", "AgentErrorEvent", "UserRejectObservation"):
            observation = event.get("observation") or event.get("error") or {}
            if isinstance(observation, dict):
                text = _text_of(observation.get("content")) or json.dumps(
                    {k: v for k, v in observation.items() if k != "content"}
                )[:2000]
            else:
                text = str(observation)
            text = _ANSI.sub("", text).strip()
            record = pending.pop(event.get("tool_call_id") or "", None)
            if record is not None:
                record["observation"] = text
            lines.append(f"--- observation ---\n{text[:4000]}")
    return "\n".join(lines), calls


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", required=True, type=Path)
    parser.add_argument("--sif", required=True, help="converted SIF carrying /openhands-venv")
    parser.add_argument("--task-dir", required=True, type=Path, help="directory of extracted tasks")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--working-dir", default="/home/user")
    parser.add_argument("--max-iterations", type=int, default=40)
    arguments = parser.parse_args()

    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.sdk.workspace import RemoteWorkspace

    # The agent runs in the container, but its events stream back here and are
    # parsed against a discriminated union keyed on "kind". Importing the tool
    # packages registers TerminalAction, FileEditorObservation and the rest, so
    # without these three lines every ActionEvent fails validation on arrival
    # and conversation.state.events comes back empty.
    import openhands.tools.file_editor  # noqa: F401
    import openhands.tools.task_tracker  # noqa: F401
    import openhands.tools.terminal  # noqa: F401
    from platoon.envs.base import Task
    from platoon.terminal_rl.executor import ApptainerBashExecutor
    from pydantic import SecretStr

    task_dir = arguments.task_dir / arguments.task
    instruction = (task_dir / "instruction.md").read_text()
    steps = json.loads(arguments.steps.read_text())
    arguments.out.mkdir(parents=True, exist_ok=True)

    sif = Path(arguments.sif).resolve()
    task = Task(
        id=arguments.task,
        goal=instruction,
        misc={
            # resolve_image passes a ".sif" reference straight through, so give
            # it the absolute path rather than a name it would look up in a pool.
            "image": str(sif),
            "sif_name": sif.name,
            "task_dir": str(task_dir),
            "instruction": instruction,
        },
    )
    executor = ApptainerBashExecutor(
        task,
        sif_dir=str(sif.parent),
        command_timeout=600,
        test_timeout=900,
        setup_timeout=900,
        cache_dir=os.getenv("APPTAINER_CACHEDIR"),
        tmp_dir=os.getenv("APPTAINER_TMPDIR"),
    )

    oracle = OracleServer(steps, free_port())
    server = None
    reward, observation = 0.0, "(not scored)"
    try:
        logger.info("starting sandbox and running the task's setup.sh ...")
        await executor.reset()
        instance = executor.backend._name  # the rollout owns the lifecycle
        logger.info("sandbox instance: %s", instance)

        oracle.start()
        server_port = free_port()
        server = start_agent_server(instance, server_port, arguments.out / f"{arguments.task}_agent_server.log")

        agent = Agent(
            llm=LLM(
                model=ORACLE_MODEL,
                base_url=f"http://127.0.0.1:{oracle.port}/v1",
                api_key=SecretStr("oracle"),
                usage_id="oracle",
            ),
            # The registry keys are the snake_case names the agent-server
            # advertises at /api/tools, not the ToolDefinition class names.
            tools=[Tool(name=name) for name in ("terminal", "file_editor", "task_tracker")],
        )
        workspace = RemoteWorkspace(host=f"http://127.0.0.1:{server_port}", working_dir=arguments.working_dir)
        conversation = Conversation(agent=agent, workspace=workspace, max_iteration_per_run=arguments.max_iterations)
        conversation.send_message(instruction)
        await asyncio.to_thread(conversation.run)

        events = [event.model_dump(mode="json") for event in conversation.state.events]
        transcript, calls = render_events(events, instruction)
        reward, observation = await executor.score()

        stem = arguments.out / f"{arguments.task}_oracle_openhands"
        (stem.with_name(stem.name + "_events.json")).write_text(json.dumps(events, indent=2))
        (stem.with_name(stem.name + "_tool_calls.json")).write_text(json.dumps(calls, indent=2))
        header = (
            f"{'=' * 72}\nGROUND-TRUTH OPENHANDS TRAJECTORY\n"
            f"task:      {arguments.task}\nimage:     {sif}\n"
            f"tools:     TerminalTool, FileEditorTool, TaskTrackerTool\n"
            f"workspace: {arguments.working_dir} (inside the Apptainer sandbox)\n"
            f"{'=' * 72}\n\n--- INSTRUCTION ---\n{instruction.strip()}\n"
        )
        footer = (
            f"\n\n{'=' * 72}\nVERIFIER\n{observation}\n{'=' * 72}\n"
            f"tool calls: {len(calls)}\nreward:     {reward}\n"
        )
        (stem.with_suffix(".txt")).write_text(header + transcript + footer)
        logger.info("reward %.1f over %d tool calls; wrote %s.txt", reward, len(calls), stem)
        return 0 if reward > 0 else 1
    finally:
        if server is not None:
            server.terminate()
        oracle.stop()
        await executor.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
