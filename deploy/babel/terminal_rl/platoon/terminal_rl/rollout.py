"""Run one LiteCoder terminal rollout.

Mirrors the number-search plugin's rollout: build the agent and environment,
run an episode under the configured timeout and always tear the sandbox down.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from logging import getLogger

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from platoon.terminal_rl.agent import VanilluxAgent
from platoon.terminal_rl.env import LiteCoderTerminalEnv
from platoon.terminal_rl.executor import ApptainerBashExecutor

logger = getLogger("platoon.terminal_rl.rollout")

# Sandbox knobs are read from the environment, as they are in tmax, so that the
# launcher can set them once for every rollout worker.
COMMAND_TIMEOUT = int(os.getenv("TERMINAL_RL_COMMAND_TIMEOUT", "120"))
TEST_TIMEOUT = int(os.getenv("TERMINAL_RL_TEST_TIMEOUT", "600"))
# tmax tasks install their dependencies in setup.sh, so reset needs a long budget.
SETUP_TIMEOUT = int(os.getenv("TERMINAL_RL_SETUP_TIMEOUT", "900"))


def _executor(task: Task) -> ApptainerBashExecutor:
    return ApptainerBashExecutor(
        task,
        command_timeout=COMMAND_TIMEOUT,
        test_timeout=TEST_TIMEOUT,
        setup_timeout=SETUP_TIMEOUT,
        cache_dir=os.getenv("APPTAINER_CACHEDIR"),
        tmp_dir=os.getenv("APPTAINER_TMPDIR"),
    )


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        agent = VanilluxAgent(
            llm_client=LiteLLMClient(model=config.model_name),
            inference_params=config.inference_params,
        )
        env = LiteCoderTerminalEnv(task, code_executor=_executor(task))

        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl")
        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )

        if config.verbose:
            logger.info("Process %s: starting rollout for task %s", os.getpid(), task.id)

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        try:
            await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            logger.error("Process %s: rollout timed out for task %s", os.getpid(), task.id)
            rollout_task.cancel()
            with suppress(asyncio.CancelledError):
                await rollout_task
            raise

        collection = current_trajectory_collection.get()
        return collection.to_dict() if config.return_dict else collection
    except Exception as error:
        logger.error("Error running rollout for task %s: %s", task.id, error)
        raise
    finally:
        # The sandbox holds a tmpfs overlay; never leave one behind.
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()
