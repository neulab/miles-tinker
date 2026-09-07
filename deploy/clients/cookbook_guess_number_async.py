"""Run tinker-cookbook's multi-turn guess_number recipe in async mode against a gateway.

The shipped recipe CLI (``tinker_cookbook.recipes.multiplayer_rl.guess_number.train``)
exposes neither ``base_url`` nor ``async_config``; this launcher builds the same
dataset and ``train.Config`` with ``AsyncConfig(max_steps_off_policy, groups_per_batch)``.
Acceptance run A of the fork (docs/fork/acceptance.md).

Args (chz ``key=value`` syntax):
  endpoint_file: file holding ``http://host:port`` (written by deploy/abci/tinker_gateway.pbs).
  base_url: gateway URL; used when endpoint_file is not given.
  log_path: run directory (metrics.jsonl, checkpoints.jsonl); deleted first per behavior_if_log_dir_exists.
  learning_rate: 3e-5 for a LoRA gateway; use 1e-6 against a full fine-tuning gateway.
  lora_rank: adapter rank requested from a LoRA gateway (ignored by a full gateway).

Example:
  TINKER_API_KEY=tml-local /home/aci18912it/miles-tinker/cookbook-venv/bin/python \\
    deploy/clients/cookbook_guess_number_async.py endpoint_file=$P/name-resolve/gw-lora.endpoint log_path=$P/runs/gn-async
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostguard  # noqa: E402,F401  (refuses login nodes; MILES_ALLOW_LOGIN_NODE=1 overrides)

import asyncio
import os
from pathlib import Path

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.multiplayer_rl.guess_number.env import GuessNumberDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class LauncherConfig:
    endpoint_file: str | None = None
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/guess-number-async"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str = "qwen3_instruct"
    batch_size: int = 32  # trajectory groups per step (= AsyncConfig.groups_per_batch)
    group_size: int = 8
    learning_rate: float = 3e-5
    max_tokens: int = 64
    max_steps_off_policy: int = 1
    eval_every: int = 5
    save_every: int = 10
    max_steps: int | None = 25
    lora_rank: int = 32
    wandb_project: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"


def resolve_base_url(endpoint_file: str | None, base_url: str | None) -> str:
    if endpoint_file:
        return Path(endpoint_file).read_text().strip()
    if base_url:
        return base_url
    if os.environ.get("TINKER_BASE_URL"):
        return os.environ["TINKER_BASE_URL"]
    raise SystemExit("pass endpoint_file=<file> or base_url=http://host:port (or set TINKER_BASE_URL)")


def build_config(c: LauncherConfig) -> train.Config:
    dataset_builder = GuessNumberDatasetBuilder(
        batch_size=c.batch_size,
        model_name=c.model_name,
        renderer_name=c.renderer_name,
        train_group_size=c.group_size,
    )
    return train.Config(
        model_name=c.model_name,
        recipe_name="recipe_multiplayer_guess_number",
        renderer_name=c.renderer_name,
        log_path=c.log_path,
        dataset_builder=dataset_builder,
        learning_rate=c.learning_rate,
        max_tokens=c.max_tokens,
        eval_every=c.eval_every,
        save_every=c.save_every,
        lora_rank=c.lora_rank,
        wandb_project=c.wandb_project,
        max_steps=c.max_steps,
        base_url=resolve_base_url(c.endpoint_file, c.base_url),
        async_config=train.AsyncConfig(max_steps_off_policy=c.max_steps_off_policy, groups_per_batch=c.batch_size),
    )


if __name__ == "__main__":
    os.environ.setdefault("TINKER_API_KEY", "tml-local")
    launcher_config = chz.entrypoint(LauncherConfig)
    config = build_config(launcher_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=launcher_config.behavior_if_log_dir_exists)
    asyncio.run(train.main(config))
