#!/usr/bin/env python3
"""Replay a recorded rollout against a real sandbox to catch the act() hang live.

Why this exists: a stalled trajectory's JSONL records only the events the episode
loop *consumed*. The event that hangs it is by definition one it never consumed,
so it is nowhere on disk, and three attempts to infer it from the recordings have
failed. This harness re-runs the rollout for real and logs the window directly.

How it works. The in-sandbox agent reaches the model over an OpenAI-compatible
HTTP endpoint (openhands_rollout._build_agent passes bridge.base_url to LLM()).
So the model can be swapped for a replay server that returns the recorded tool
calls in order -- same actions, same sandbox, no GPU and no Tinker lease. The
episode loop, OpenHandsAgent.act, OpenHandsTerminalEnv.step and
get_actions_for_last_obs are all the real ones.

The payload is --trace: get_actions_for_last_obs is wrapped so that every poll
which returns [] logs the OLDEST events in the window and what
last_step_observation_id resolves to. That is the measurement the recordings
cannot provide.

    P=/data/user_data/$USER/miles-babel
    PYTHONPATH=$P/platoon:$P/platoon/plugins/openhands:$P/platoon/plugins/terminal-rl \
      $P/client-tmax-15k-openhands/bin/python replay_rollout.py \
      --events <events_task_*.jsonl> --trace

Run it on a node with apptainer and the task images. It starts one sandbox.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("replay")


# --------------------------------------------------------------------------
# 1. the recorded actions
# --------------------------------------------------------------------------
def load_actions(events_path: str):
    """Every ActionEvent the loop consumed, in order, as OpenAI tool calls.

    Each action also carries ``delay``: the wall-clock gap between the previous
    event in the stream and this action's own timestamp. That gap is the model's
    sampling latency in the original rollout (~14s median), and replaying without
    it lets the sandbox agent race so far ahead that act() always has a backlog
    and never sits on an empty window -- which is the state the hang occurs in.
    """
    from datetime import datetime

    actions, task_id, prev_ts = [], None, None
    for line in Path(events_path).read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "trajectory_task_set":
            task_id = (rec.get("task") or {}).get("id") or task_id
        if rec.get("type") != "trajectory_step_added":
            continue
        step = rec.get("step", rec)
        for ae in (step.get("action_events") or {}).get("action_events") or []:
            call = ae.get("tool_call") or {}
            ts = ae.get("timestamp")
            delay = 0.0
            if ts and prev_ts:
                try:
                    delay = max(0.0, (datetime.fromisoformat(ts) - prev_ts).total_seconds())
                except ValueError:
                    delay = 0.0
            actions.append({
                "name": call.get("name") or ae.get("tool_name"),
                "arguments": call.get("arguments") or "{}",
                "reasoning": ae.get("reasoning_content") or "",
                "delay": delay,
            })
            if ts:
                try:
                    prev_ts = datetime.fromisoformat(ts)
                except ValueError:
                    pass
        for oe in step.get("observation_events") or []:
            # Only the real tool observation marks when the agent could next
            # sample. The ConversationStateUpdateEvent that trails it is stamped
            # when the following LLM call returns, so measuring to it would
            # report a zero gap and erase the latency we are trying to replay.
            kind = (oe.get("observation") or {}).get("kind") or oe.get("kind")
            if kind == "ConversationStateUpdateEvent":
                continue
            ts = oe.get("timestamp")
            if ts:
                try:
                    prev_ts = datetime.fromisoformat(ts)
                except ValueError:
                    pass
    return task_id, actions


# 2. the replay "model"
# --------------------------------------------------------------------------
class ReplayServer:
    """Serves the recorded tool calls over /v1/chat/completions, in order.

    Once the script runs out of recorded actions it keeps answering, because the
    sandbox may legitimately need one more turn; an exhausted replay returns a
    harmless no-op so the conversation ends rather than erroring.
    """

    def __init__(self, actions, model="replay", speed=1.0):
        self.actions, self.model, self.n = actions, model, 0
        self.speed = speed
        self.condenser_calls = 0
        self.lock = threading.Lock()
        handler = self._handler()
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def _next(self):
        with self.lock:
            if self.n >= len(self.actions):
                return None
            a = self.actions[self.n]
            self.n += 1
            return a

    def _handler(server_self):
        outer = server_self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                # The condenser shares this endpoint. Its requests carry no tool
                # schema, so they are answered with a summary instead of consuming
                # a recorded action -- otherwise condensing shifts the whole
                # sequence and the replay stops matching the original.
                try:
                    payload = json.loads(body or b"{}")
                except json.JSONDecodeError:
                    payload = {}
                if not payload.get("tools"):
                    outer.condenser_calls += 1
                    logger.info("condenser call %d -> serving a summary (no action consumed)",
                                outer.condenser_calls)
                    self._reply({"role": "assistant", "content": "Summary of earlier steps."}, "stop")
                    return
                a = outer._next()
                idx = outer.n
                if a is not None and a["delay"] > 0 and outer.speed > 0:
                    time.sleep(min(a["delay"] / outer.speed, 120.0))
                if a is None:
                    msg = {"role": "assistant", "content": "Replay exhausted; stopping."}
                else:
                    logger.info("replay -> action %d/%d  %s  (waited %.1fs)",
                                idx, len(outer.actions), a["name"], a["delay"] / max(outer.speed, 1e-9))
                    msg = {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": a["reasoning"],
                        "tool_calls": [{
                            "id": f"replay-{uuid.uuid4()}",
                            "type": "function",
                            "function": {"name": a["name"], "arguments": a["arguments"]},
                        }],
                    }
                out = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": outer.model,
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason": "tool_calls" if a else "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _reply(self, message, finish_reason):
                out = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": outer.model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                raw = json.dumps({"data": [{"id": outer.model}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return H


# --------------------------------------------------------------------------
# 3. the instrumentation -- the reason this script exists
# --------------------------------------------------------------------------
def install_window_trace(every=25):
    """Log the OLDEST events in the window whenever act() gets an empty list."""
    from platoon.utils import openhands_utils as U

    real = U.get_actions_for_last_obs
    state = {"empty": 0}

    def traced(observation, require_same_llm_call_id=False):
        out = real(observation, require_same_llm_call_id=require_same_llm_call_id)
        if out:
            state["empty"] = 0
            return out
        state["empty"] += 1
        if state["empty"] % every:
            return out
        events = list(observation.conversation_state.events)
        stop = observation.last_step_observation_id
        cut = next((i for i in range(len(events) - 1, -1, -1) if events[i].id == stop), None)
        window = events[cut + 1:] if cut is not None else events
        stop_kind = type(events[cut]).__name__ if cut is not None else "NOT FOUND IN STREAM"
        logger.warning(
            "[window] empty_polls=%d  stop=%s(%s)  window=%d  kinds=%s",
            state["empty"], stop_kind, str(stop)[:8], len(window),
            ",".join(f"{k}={v}" for k, v in Counter(type(e).__name__ for e in window).most_common()),
        )
        logger.warning(
            "[window] OLDEST 4 events in the window (the backward walk ends here): %s",
            [f"{type(e).__name__}({str(e.id)[:8]})" for e in window[:4]],
        )
        logger.warning(
            "[window] NEWEST 3: %s",
            [f"{type(e).__name__}({str(e.id)[:8]})" for e in window[-3:]],
        )
        return out

    U.get_actions_for_last_obs = traced
    # agent.py imported the symbol directly, so rebind it there too.
    try:
        from platoon.openhands import agent as A
        A.get_actions_for_last_obs = traced
    except Exception:
        pass
    logger.info("window trace installed (logs every %d empty polls)", every)


# --------------------------------------------------------------------------
# 4. drive the real loop
# --------------------------------------------------------------------------
async def run(args):
    from platoon.episode.loop import run_episode
    from platoon.episode.trajectory import TrajectoryCollection, current_trajectory_collection
    from platoon.openhands.agent import OpenHandsAgent
    from platoon.terminal_rl.agent_server import SandboxAgentServer
    from platoon.terminal_rl.executor import ApptainerBashExecutor
    from platoon.terminal_rl.openhands_env import OpenHandsTerminalEnv
    from platoon.terminal_rl.openhands_rollout import _build_agent
    from platoon.terminal_rl.tasks import get_task

    task_id, actions = load_actions(args.events)
    task_id = args.task or task_id or Path(args.events).name.split("_")[1:3]
    if isinstance(task_id, list):
        task_id = "_".join(["task"] + task_id)
    logger.info("replaying %d recorded actions for %s", len(actions), task_id)
    if not actions:
        raise SystemExit("no ActionEvents in that events file")

    if args.trace:
        install_window_trace(args.trace_every)

    total_delay = sum(a["delay"] for a in actions)
    logger.info("recorded sampling latency: %.0fs total, %.1fs median; replaying at %gx",
                total_delay,
                sorted(a["delay"] for a in actions)[len(actions) // 2],
                args.speed)
    replay = ReplayServer(actions, speed=args.speed)
    logger.info("replay model serving at %s", replay.base_url)

    task = get_task(task_id)
    executor = ApptainerBashExecutor(
        task,
        sif_dir=args.sif_dir,
        cache_dir=os.getenv("APPTAINER_CACHEDIR"),
        tmp_dir=os.getenv("APPTAINER_TMPDIR"),
    )
    await executor.reset()
    server = await SandboxAgentServer(executor.backend, log_path=args.server_log).start()
    logger.info("agent-server up at %s", server.url)

    env = OpenHandsTerminalEnv(
        task, _build_agent(replay.base_url, "replay-key", args.max_output_tokens),
        executor, server.url, working_dir=os.getenv("TERMINAL_RL_WORKING_DIR", "/home/user"),
    )
    agent = OpenHandsAgent()

    collection = TrajectoryCollection()
    current_trajectory_collection.set(collection)

    started = time.perf_counter()
    try:
        await run_episode(agent, env, timeout=args.step_timeout)
    except BaseException as exc:                      # run_episode records rather than raises
        logger.warning("episode raised: %r", exc)
    elapsed = time.perf_counter() - started
    traj = next(iter(collection.trajectories.values()), None)
    steps = len(traj.steps) if traj else 0
    message = (traj.error_message or "") if traj else ""
    logger.info("episode ended after %.0fs, %d steps, replayed %d/%d actions",
                elapsed, steps, replay.n, len(actions))
    logger.info("error: %s", message.splitlines()[0] if message else "(none)")
    if "TimeoutError" in message or "CancelledError" in message:
        logger.warning("REPRODUCED: the loop hung and was truncated by step_timeout")
        return 0
    logger.info("did not reproduce the hang on this replay")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True, help="a recorded events_task_*.jsonl")
    p.add_argument("--task", help="task id override (default: inferred from the file)")
    p.add_argument("--sif-dir")
    p.add_argument("--server-log", default="/tmp/replay_agent_server.log")
    p.add_argument("--step-timeout", type=float, default=300.0)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--speed", type=float, default=1.0,
                   help="replay the recorded sampling latency divided by this "
                        "(1.0 = original pacing, 0 = as fast as possible)")
    p.add_argument("--trace", action="store_true", help="log the window on empty polls")
    p.add_argument("--trace-every", type=int, default=25, help="log every N empty polls (0.2s each)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
