"""Run one LiteCoder episode and print it, without the trainer.

Two modes:

* ``--oracle`` replays the task's own reference solution. No model is needed, so
  it checks the environment, the episode loop and the reward path on their own.
* ``--base-url`` runs the real agent against any OpenAI-compatible endpoint --
  the Miles gateway of a live run, or a standalone vLLM server.

Usage:
    python -m platoon.terminal_rl.rollout_once --data-dir PATH --oracle
    python -m platoon.terminal_rl.rollout_once --data-dir PATH \\
        --base-url http://127.0.0.1:8008/v1 --model hosted_vllm/tmax-9b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class ReplayAgent:
    """Replays a recorded trajectory's commands, one per turn.

    Datasets such as TMaxxx/TMax-SFT-16.5K-Env ship successful rollouts whose
    assistant turns carry the bash commands that earned reward 1. Replaying them
    checks the environment against a known-good trajectory on pools that have no
    solution/ directory to drive --oracle.
    """

    def __init__(self, commands: list[str]):
        if not commands:
            raise SystemExit("The replay file contained no commands.")
        self._commands = commands
        self._index = 0

    async def act(self, obs):
        from platoon.envs.codeact import CodeActAction

        if self._index >= len(self._commands):
            command, thought = SUBMIT_COMMAND, "Trajectory exhausted; submitting."
        else:
            command = self._commands[self._index]
            thought = f"Replay step {self._index + 1} of {len(self._commands)}."
            self._index += 1
        return CodeActAction(action=command, parsed_code=command, parsed_thought=thought)

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass


class NoopAgent:
    """Submits immediately, touching nothing.

    A pool whose tasks ship no reference solution cannot be checked with
    --oracle. This still exercises the whole path -- image, setup.sh, deferred
    test upload, verifier, reward -- and should score 0 because the work was not
    done, not because the harness broke. Read the test output to tell those apart.
    """

    async def act(self, obs):
        from platoon.envs.codeact import CodeActAction

        return CodeActAction(
            action=SUBMIT_COMMAND, parsed_code=SUBMIT_COMMAND, parsed_thought="Submitting without doing the task."
        )

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass


class OracleAgent:
    """Replays the reference solution, then submits. Mirrors the agent contract."""

    def __init__(self, task):
        solution = Path(task.misc["task_dir"]) / "solution" / "solve.sh"
        if not solution.is_file():
            raise SystemExit(f"Task {task.id} ships no solution/solve.sh to replay.")
        script = solution.read_text()
        self._commands = [
            f"cat > /tmp/solve.sh <<'TERMINAL_RL_EOF'\n{script}\nTERMINAL_RL_EOF",
            "bash /tmp/solve.sh",
            SUBMIT_COMMAND,
        ]
        self._index = 0

    async def act(self, obs):
        from platoon.envs.codeact import CodeActAction

        if self._index >= len(self._commands):
            command, thought = SUBMIT_COMMAND, "Nothing left to do."
        else:
            command = self._commands[self._index]
            thought = f"Oracle step {self._index + 1} of {len(self._commands)}."
            self._index += 1
        return CodeActAction(action=command, parsed_code=command, parsed_thought=thought)

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Pool from prepare_litecoder.py.")
    parser.add_argument("--task-id", help="Defaults to the first staged task.")
    parser.add_argument("--oracle", action="store_true", help="Replay the reference solution instead of a model.")
    parser.add_argument("--noop", action="store_true", help="Submit immediately; checks the harness without a model.")
    parser.add_argument("--replay", type=Path, help="JSON list of bash commands from a known-good trajectory.")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint for the real agent.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-local"))
    parser.add_argument("--max-steps", type=int, default=30, help="Cap the episode; the pool default is 100.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-chars", type=int, default=1200, help="Per-turn output shown.")
    return parser


def _print_turn(index: int, step, limit: int) -> None:
    print(f"\n{'=' * 72}\nTURN {index}")
    if step.thought:
        print(f"--- thought ---\n{step.thought.strip()[:limit]}")
    print(f"--- command ---\n{(step.code or '').strip()[:limit]}")
    body = (step.output or step.error or "(no output)").strip()
    print(f"--- observation ---\n{body[:limit]}{' ...[cut]' if len(body) > limit else ''}")
    if step.reward:
        print(f"--- reward: {step.reward} ---")


async def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    os.environ["TERMINAL_RL_DATA_DIR"] = str(data_dir)
    os.environ.setdefault("SWERL_APPTAINER_SIF_DIR", str(data_dir / "sifs"))

    # Imported after the environment is set, because tasks.py reads it eagerly.
    from platoon.episode.context import current_trajectory_collection
    from platoon.episode.loop import run_episode
    from platoon.episode.trajectory import TrajectoryCollection

    from platoon.terminal_rl.agent import VanilluxAgent
    from platoon.terminal_rl.env import LiteCoderTerminalEnv
    from platoon.terminal_rl.executor import ApptainerBashExecutor
    from platoon.terminal_rl.tasks import get_task, get_task_ids

    task_id = args.task_id or get_task_ids("train")[0]
    task = get_task(task_id)
    task.max_steps = args.max_steps
    mode = "replay" if args.replay else ("noop" if args.noop else ("oracle" if args.oracle else args.model))
    print(f"task:  {task.id}\nimage: {task.misc['image']}\nmode:  {mode}")

    if args.replay:
        agent = ReplayAgent(json.loads(args.replay.read_text()))
    elif args.noop:
        agent = NoopAgent()
    elif args.oracle:
        agent = OracleAgent(task)
    else:
        if not args.base_url:
            raise SystemExit("Pass --base-url for a model run, or --oracle/--noop to check the harness.")
        from platoon.config_defs import InferenceParams
        from platoon.utils.llm_client import LiteLLMClient

        agent = VanilluxAgent(
            llm_client=LiteLLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key),
            inference_params=InferenceParams(temperature=1.0, max_completion_tokens=args.max_tokens),
        )

    env = LiteCoderTerminalEnv(task, code_executor=ApptainerBashExecutor(task))
    current_trajectory_collection.set(TrajectoryCollection())
    print("starting the sandbox; the first turn waits for the image to start...", flush=True)
    try:
        trajectory = await asyncio.create_task(run_episode(agent, env, timeout=1800))
    finally:
        await env.close()

    for index, step in enumerate(trajectory.steps, start=1):
        _print_turn(index, step, args.output_chars)
    print(f"\n{'=' * 72}")
    print(f"turns:   {len(trajectory.steps)}")
    print(f"reward:  {trajectory.reward}")
    print(f"finish:  {trajectory.finish_message}")
    if trajectory.error_message:
        print(f"error:   {trajectory.error_message.splitlines()[0]}")


if __name__ == "__main__":
    asyncio.run(main())
