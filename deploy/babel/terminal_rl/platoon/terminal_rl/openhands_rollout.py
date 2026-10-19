"""Run one terminal rollout with the openhands-sdk agent.

Mirrors rollout.run_rollout, but the agent is openhands' rather than Vanillux's
and it lives inside the task's sandbox. Three pieces have to line up for a
rollout to produce training data:

* the sandbox, started by ApptainerBashExecutor, which also runs setup.sh and
  owns the verifier;
* an openhands-agent-server inside it, so the agent's tools act on the task's
  container rather than on the compute node;
* a lease on the Tinker proxy bridge, so the sampling the in-container agent
  does over HTTP is recorded into *this* rollout's interactions and can be
  joined to its trajectory steps by completion id.

The tool set defaults to the evaluation's, not openhands' default preset:
terminal-bench registers TerminalTool, FileEditorTool and TaskTrackerTool (plus
openhands' built-in finish and think) and no BrowserToolSet, so offering more
here would train against an action space the evaluation never provides.
TERMINAL_RL_TOOLS can offer less, to training and in-training eval rollouts alike.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from logging import getLogger

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.openhands.agent import OpenHandsAgent
from platoon.visualization.event_sinks import JsonlFileSink

from platoon.terminal_rl.agent_server import SandboxAgentServer
from platoon.terminal_rl.executor import ApptainerBashExecutor
from platoon.terminal_rl.openhands_env import OpenHandsTerminalEnv
from platoon.terminal_rl import phase_timing, prompt_dump, render_check
from platoon.terminal_rl.proxy_bridge import get_bridge
from platoon.terminal_rl.sdk_patches import apply_sdk_patches
from platoon.terminal_rl.rollout import COMMAND_TIMEOUT, SETUP_TIMEOUT, TEST_TIMEOUT

logger = getLogger("platoon.terminal_rl.openhands_rollout")

apply_sdk_patches()

EVAL_TOOLS = ("terminal", "file_editor", "task_tracker", "finish", "think")
# openhands attaches its built-in tools on top of `tools` (Agent.include_default_tools,
# FinishTool and ThinkTool by default, as in the evaluation), so those two are
# named here by tool name and passed through that field instead.
BUILT_IN_TOOLS = {"finish": "FinishTool", "think": "ThinkTool"}
# TERMINAL_RL_TOOLS narrows the tool set of every rollout, training and eval, so
# eval measures the policy in the action space it trains in. Why one would: ThinkTool lets the policy reason in a tool call instead of its own <think>
# block, and in jobs 155637/155644 RL took that over -- think calls went from 4%
# of actions to 45-49% after ~16 updates, rollouts that leaned on it solved
# 13-38% against 50-75%, they burned the step budget, and opening a think call
# before </think> became the main malformed ending (the renderer files the call
# header as reasoning). task_tracker was 0.1-0.6% of actions.
TOOLS = tuple(name.strip() for name in os.getenv("TERMINAL_RL_TOOLS", ",".join(EVAL_TOOLS)).split(",") if name.strip())
_unknown = set(TOOLS) - set(EVAL_TOOLS)
if _unknown:
    raise ValueError(f"TERMINAL_RL_TOOLS names unknown tools {sorted(_unknown)}; choose from {EVAL_TOOLS}")
# What the in-container LiteLLM calls the model. It is cosmetic: the bridge
# replaces it with Platoon's provider string, and the tokens that reach the
# trainer are rendered host-side by the Tinker proxy's renderer.
CONTAINER_MODEL = os.getenv("TERMINAL_RL_CONTAINER_MODEL", "openai/gpt-4o")
# The evaluation runs LLM_MAX_INPUT_TOKENS=40000. Training cannot afford that:
# the condenser's budget is also the prompt length, and a step's datum is its
# prompt plus its response, so a 40k budget makes 40k-token datums -- and the
# learner materialises vocab_size x tokens x 4 bytes of logits for each. Keep
# the budget small enough that the backward pass fits, and raise it only with
# the learner's memory headroom.
MAX_INPUT_TOKENS = int(os.getenv("TERMINAL_RL_MAX_INPUT_TOKENS", "16384"))
# openhands refuses an LLM whose context window is under 16384 tokens. The
# condenser budget is that window, so a smaller budget needs the SDK's own
# override -- otherwise every conversation fails before it samples once.
if MAX_INPUT_TOKENS < 16384:
    os.environ.setdefault("ALLOW_SHORT_CONTEXT_WINDOWS", "true")
CONDENSER_MAX_SIZE = os.getenv("TERMINAL_RL_CONDENSER_MAX_SIZE")
# The in-container LLM's own request timeout. openhands defaults it to 300s and
# retries a timed-out request (LiteLLMTimeout is in LLM_RETRY_EXCEPTIONS), but
# the bridge keeps sampling the abandoned request, so each retry adds a second
# request to an already busy server -- and no call can outlast 300s however
# long STEP_TIMEOUT is. Follow the proxy's sample timeout instead, which the
# launcher keeps above the step timeout, so the step timeout always fires first.
LLM_TIMEOUT = int(float(os.getenv("TERMINAL_RL_LLM_TIMEOUT", os.getenv("PLATOON_SAMPLE_TIMEOUT", "600"))))
REASONING_EFFORT = os.getenv("TERMINAL_RL_REASONING_EFFORT", "")
WORKING_DIR = os.getenv("TERMINAL_RL_WORKING_DIR", "/home/user")


def _build_agent(base_url: str, api_key: str, max_output_tokens: int):
    """An openhands Agent whose LLM speaks to the Tinker bridge over HTTP.

    The condenser matters for more than matching evaluation: it rewrites the
    history once the conversation outgrows its budget, which is what keeps a
    long episode's prompts -- and therefore its training datums -- bounded.
    """
    from openhands.sdk import Agent
    from openhands.sdk.llm.llm import LLM
    from openhands.sdk.tool import Tool
    from pydantic import SecretStr

    # The agent runs in the container, but its events stream back here and are
    # parsed against unions discriminated on "kind". Importing the tool packages
    # registers TerminalTool/TerminalAction/TerminalObservation and friends;
    # without them every ActionEvent fails validation on arrival, the mirrored
    # conversation state stays empty, and the episode loop records no steps --
    # which reads exactly like a model that never acted.
    import openhands.tools.file_editor  # noqa: F401
    import openhands.tools.task_tracker  # noqa: F401
    import openhands.tools.terminal  # noqa: F401

    llm = LLM(
        model=CONTAINER_MODEL,
        base_url=base_url,
        api_key=SecretStr(api_key),
        usage_id="platoon-tinker",
        max_output_tokens=max_output_tokens,
        max_input_tokens=MAX_INPUT_TOKENS,
        timeout=LLM_TIMEOUT,
        # openhands sends reasoning_effort="high" by default. Training's bridge renders
        # the prompt itself (hf_chat_template, TERMINAL_RL_REASONING_EFFORT) and ignores
        # the field, but an OpenAI-compatible server hands it to the chat template, and
        # Qwen3.8's accepts only xhigh/medium/low -- every request fails with "Unexpected
        # reasoning effort high". Send the configured effort instead.
        **({"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}),
    )
    kwargs = {
        "llm": llm,
        "tools": [Tool(name=name) for name in TOOLS if name not in BUILT_IN_TOOLS],
        "include_default_tools": [BUILT_IN_TOOLS[name] for name in TOOLS if name in BUILT_IN_TOOLS],
    }
    try:
        from openhands.sdk.context.condenser import LLMSummarizingCondenser

        condenser_kwargs = {"llm": llm, "max_tokens": MAX_INPUT_TOKENS}
        if CONDENSER_MAX_SIZE:
            condenser_kwargs["max_size"] = int(CONDENSER_MAX_SIZE)
        kwargs["condenser"] = LLMSummarizingCondenser(**condenser_kwargs)
        logger.info("condenser enabled: max_tokens=%d", MAX_INPUT_TOKENS)
    except Exception as error:  # the evaluation also proceeds without one
        logger.warning("condenser unavailable (%s); history will not be condensed", error)
    return Agent(**kwargs)


async def _cancel_if_server_dies(server, rollout: asyncio.Task, task_id: str, env=None, interval: float = 5.0) -> None:
    """End the episode promptly if its agent-server process exits."""
    while not rollout.done():
        await asyncio.sleep(interval)
        if rollout.done() or getattr(env, "closing", False):
            # The episode finished, or is closing down, while we slept. openhands shuts its server
            # down once the conversation reaches a terminal state, so an exited
            # process here is the normal end of a healthy rollout, not a death.
            # Without this check the watchdog fired once per completed rollout
            # -- 29 firings against 29 finished trajectories in job 10590936 --
            # which cancelled nothing but made the log useless for spotting the
            # real thing.
            return
        if not server.is_alive():
            logger.error("[%s] agent-server died; ending the episode", task_id)
            rollout.cancel()
            return


# A task may carry its own verifier timeout (TB2's task.toml); it is capped so one
# slow verifier cannot hold a rollout for hours. Tasks without one use TEST_TIMEOUT.
TEST_TIMEOUT_MAX = int(os.getenv("TERMINAL_RL_TEST_TIMEOUT_MAX", "1800"))


def _test_timeout(task: Task) -> int:
    own = (task.misc or {}).get("test_timeout")
    return min(int(own), TEST_TIMEOUT_MAX) if own else TEST_TIMEOUT


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    executor = server = env = None
    phase_timing.reset()
    rollout_started = time.perf_counter()
    try:
        executor = ApptainerBashExecutor(
            task,
            command_timeout=COMMAND_TIMEOUT,
            test_timeout=_test_timeout(task),
            setup_timeout=SETUP_TIMEOUT,
            cache_dir=os.getenv("APPTAINER_CACHEDIR"),
            tmp_dir=os.getenv("APPTAINER_TMPDIR"),
        )
        with phase_timing.phase("sandbox"):
            await executor.reset()
        # A GRPO group runs the same task many times at once, so naming the log
        # after the task alone makes every group member truncate the same file
        # and no server's errors survive. The sandbox instance name is unique
        # per rollout.
        server_log = os.path.join(
            config.output_dir, "events", f"agent_server_{task.id}_{executor.backend._name}.log"
        )
        with phase_timing.phase("server_start"):
            server = await SandboxAgentServer(executor.backend, log_path=server_log).start()

        bridge = await get_bridge(config.model_name)
        # The lease binds the TinkerLLMProxySession the workflow entered around
        # this rollout, so sampling done in the container lands in the same dict
        # the trainer reads afterwards.
        # The bridge appends every completion's token count here; the reward's
        # output-length penalty reads the longest (openhands_env.py).
        response_tokens: list[int] = []
        with bridge.lease(label=f"{task.id}/{executor.backend._name}", response_tokens=response_tokens) as token:
            # Two different "agents": the openhands Agent runs the conversation
            # inside the sandbox, while Platoon's OpenHandsAgent is the
            # episode-loop adapter that reads actions back out of its state.
            env = OpenHandsTerminalEnv(
                task,
                _build_agent(bridge.base_url, token, config.inference_params.max_completion_tokens),
                executor,
                server.url,
                # Harbor tasks start where their image's WORKDIR is (/work, /app).
                working_dir=(task.misc or {}).get("agent_workdir") or WORKING_DIR,
                response_tokens=response_tokens,
            )
            agent = OpenHandsAgent()
            # The episode loop alternates agent.act (waits for openhands to
            # produce the next action) with env.step (waits for that action's
            # observations, then scores). Both poll, so timing them separately
            # says which one holds the rollout open.
            phase_timing.instrument(agent, "act", "act")
            phase_timing.instrument(env, "step", "step")
            phase_timing.instrument(env, "evaluate", "score_within_step")

            traj_collection = TrajectoryCollection()
            current_trajectory_collection.set(traj_collection)
            events_path = os.path.join(
                config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl"
            )
            traj_collection.register_event_handlers(
                JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
            )

            episode_started = time.perf_counter()
            rollout = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
            # If the in-sandbox server dies mid-episode the conversation can
            # never reach a terminal status, so the episode loop polls a dead
            # port until its step timeout -- one such rollout added twelve
            # minutes to a twenty-minute collection phase, because the phase
            # waits for its slowest member. Cancelling instead ends the episode
            # in seconds; run_episode treats cancellation like any other error,
            # recording it and still returning the trajectory, so the rollout is
            # scored on whatever the agent achieved rather than discarded.
            watchdog = asyncio.create_task(_cancel_if_server_dies(server, rollout, task.id, env))
            try:
                # Not asyncio.wait_for: run_episode catches the cancellation and
                # returns its trajectory, so wait_for never raised and the timeout
                # passed silently as a bare CancelledError -- 245 rollouts in job
                # 155424 with no log line. The trajectory carries that error, so
                # the workflow drops the rollout as infra-killed. The collection
                # phase waits for its slowest member, which is what the cap buys;
                # grep DISCARDED per step to see what it is costing.
                done, _ = await asyncio.wait({rollout}, timeout=config.timeout)
                if not done:
                    partial = current_trajectory_collection.get(None)
                    trajectories = list(partial.trajectories.values()) if partial and partial.trajectories else []
                    steps = len(trajectories[0].steps) if trajectories else 0
                    logger.error(
                        "Process %s: DISCARDED rollout for task %s after %ss (%d steps taken)",
                        os.getpid(),
                        task.id,
                        config.timeout,
                        steps,
                    )
                    rollout.cancel()
                    with suppress(asyncio.CancelledError):
                        await rollout
            finally:
                phase_timing.add("episode", time.perf_counter() - episode_started)
                # The interactions dict the lease bound is the session's, so it
                # now holds every sampling call this rollout made.
                if prompt_dump.DUMP_LIMIT:
                    from platoon.train.tinker.proxy import proxy_interactions

                    with suppress(Exception):
                        prompt_dump.dump(
                            proxy_interactions.get(),
                            os.path.join(config.output_dir, "events", f"prompts_{task.id}_{traj_collection.id}.txt"),
                        )
                watchdog.cancel()
                with suppress(asyncio.CancelledError):
                    await watchdog
                # After the watchdog is gone, so a cancellation landing here cannot
                # leak it. Comparing ~60 long prompts takes up to a second; keep it
                # off the event loop the other rollouts share.
                if render_check.ENABLED:
                    from platoon.train.tinker.proxy import proxy_interactions

                    with suppress(Exception):
                        await asyncio.to_thread(
                            render_check.check, proxy_interactions.get(), task.id, prompt_dump._tokenizer()
                        )

            collection = current_trajectory_collection.get()
            return collection.to_dict() if config.return_dict else collection
    except Exception as error:
        logger.error("Error running openhands rollout for task %s: %s", task.id, error)
        raise
    finally:
        # Order matters. The conversation's run thread polls the agent-server
        # until it is closed, so killing the server first leaves that thread
        # hammering a dead port and filling the log with "Connection refused".
        # env.close() scores the task (which needs the sandbox), closes the
        # conversation, and then closes the executor; the server dies with its
        # sandbox, so closing it afterwards is just belt and braces.
        with phase_timing.phase("teardown"):
            if env is not None:
                await env.close()
            elif executor is not None:
                await executor.close()
            if server is not None:
                await server.close()
        phase_timing.report(task.id, time.perf_counter() - rollout_started)
