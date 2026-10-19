"""Terminal-RL training with the openhands-sdk agent, against a Tinker endpoint.

Usage:
    python -m platoon.terminal_rl.train_openhands --config terminal_tinker.yaml
"""

import asyncio
import logging
import sys
from pathlib import Path

from datasets import Dataset
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

from platoon.terminal_rl.openhands_rollout import run_rollout
from platoon.terminal_rl.tasks import get_task, get_task_ids

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("platoon").setLevel(logging.DEBUG)


async def main(args: list[str]):
    default_config = Path(__file__).parent / "terminal_tinker.yaml"
    config, _ = load_config(
        args=args,
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=str(default_config),
    )

    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in get_task_ids("train")])
    eval_dataset = Dataset.from_list([{"task_id": task_id} for task_id in get_task_ids("val")])

    trainer = PlatoonTinkerRLTrainer(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)
    async with trainer:
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=False,
        )
        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.eval.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="eval",
            filter_errors=False,
        )
        await trainer.train(train_workflow=train_workflow, eval_workflow=eval_workflow)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
