import json
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

# PPO (actor + critic) with the Megatron backend on a single node.
#
# Unlike GRPO, which derives its baseline from a group of samples per prompt, PPO trains a
# separate value model (the critic) and turns rewards into advantages with GAE. The critic is
# colocated on the actor's train GPUs and shares its parallelism, so a PPO run needs no extra
# GPUs over the GRPO equivalent -- it pays in memory instead, which is why --offload-train is
# forced on.
#
# python run_qwen3_4b_ppo.py


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: str = "qwen3-4B"
    # Physical GPUs available to Ray; actor and critic share the training pool.
    num_gpus_per_node: int = 4
    train_gpus: int | None = None  # Defaults to all GPUs.
    rollout_gpus: int = 0  # Zero keeps the original colocated inference layout.
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    train_script: str = "train.py"
    extra_args: str = ""

    def __post_init__(self):
        if self.train_gpus is None:
            self.train_gpus = self.num_gpus_per_node
        assert self.train_gpus > 0 and self.rollout_gpus >= 0
        assert self.train_gpus + self.rollout_gpus <= self.num_gpus_per_node


def prepare(args: ScriptArgs):
    # One-off Hyperion comparison: reuse prepared artifacts and its HF shuffle.
    # Keep the optional dataset dependency out of launcher imports.
    from datasets import Dataset

    source = Path(args.data_dir) / "dapo-math-17k/dapo-math-17k.jsonl"
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    indices = Dataset.from_dict({"index": list(range(len(rows)))})
    indices = list(indices.shuffle(seed=42)["index"])
    indices = indices[: len(indices) // 8 * 8]  # Hyperion drops incomplete prompt batches.
    if not indices:
        raise ValueError("Need at least eight prompts")
    output = Path(args.output_dir) / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dapo-hyperion-order.jsonl").open("w") as stream:
        for index in indices:
            stream.write(json.dumps(rows[index], ensure_ascii=False) + "\n")


def execute(args: ScriptArgs):
    # Fresh HF initialization for both models; no reference model or checkpoint saves.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}/ "
        f"--load {args.model_dir}/{args.model_name} "
        "--megatron-to-hf-mode bridge "
    )

    rollout_args = (
        f"--prompt-data {args.output_dir}/{args.run_id}/dapo-hyperion-order.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rm-type deepscaler "
        "--num-rollout 300 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 64 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    ppo_args = (
        # Selecting ppo is what creates the critic; everything else here is tuning.
        "--advantage-estimator ppo "
        "--critic-lr 1e-5 "  # the critic usually wants a larger lr than the actor
        "--num-critic-only-steps 1 "  # value-function warmup: actor frozen for this many steps
        "--normalize-advantages "
        "--calculate-per-token-loss "
        "--use-rollout-logprobs "
        # Reward-level KL (--kl-coef) is rejected with ppo: the critic trains before the actor and
        # never sees ref log probs, so its value targets would silently omit that penalty. Use
        # loss-level KL instead.
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        # Qwen3-4B fits on one GPU, so one engine per GPU beats sharding an engine across two.
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-max-running-requests 512 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.rollout_gpus:
        misc_args += f"--rollout-num-gpus {args.rollout_gpus} --offload-train --no-offload-rollout "
    else:
        misc_args += "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        train_script=args.train_script,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
