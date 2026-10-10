"""Train Qwen3.5-4B with OpenHands/Sail scientific-task GRPO on four GPUs.

Requires converted Qwen3.5-4B_torch_dist weights and the science agent server.

Args:
  --data-dir: Directory containing science-rl.jsonl (frozen HF task instructions).
  --model-dir: Directory containing HF and converted model checkpoints.
  --output-dir: Checkpoints and logs, separate from pretrained weights.
  --rollout-batch-size / --group-size: Prompts per batch and responses per prompt.
  --num-rollout: Number of optimizer updates (default: 100).

Example:
  python scripts/run_qwen35_4b_science.py --model-dir /models \\
      --data-dir /datasets --output-dir /runs/science --rollout-batch-size 32 --group-size 8
"""

import shlex
from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 4
    rollout_batch_size: int = 32
    group_size: int = 8
    num_rollout: int = 100
    max_tokens_per_gpu: int = 8192
    max_seq_len: int = 32768
    response_length: int = 4096
    learning_rate: float = 1e-6
    save_interval: int = 10
    extra_args: str = ""

    @property
    def global_batch_size(self):
        return self.rollout_batch_size * self.group_size


def _wandb_args(args):
    # The helper embeds its key in argv; remove it so command logs/config cannot leak it.
    # W&B authenticates using the host's scoped .netrc, mounted outside learner boxes.
    words = shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_id))
    if "--wandb-key" in words:
        index = words.index("--wandb-key")
        del words[index:index + 2]
    return shlex.join(words)


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {shlex.quote(args.model_dir + '/Qwen3.5-4B')} "
        f"--ref-load {shlex.quote(args.model_dir + '/Qwen3.5-4B_torch_dist')} "
        f"--load {shlex.quote(args.output_dir + '/checkpoints')} "
        f"--save {shlex.quote(args.output_dir + '/checkpoints')} --save-interval {args.save_interval} "
    )
    rollout_args = (
        f"--prompt-data {shlex.quote(args.data_dir + '/science-rl.jsonl')} "
        "--input-key prompt --metadata-key metadata --rollout-shuffle "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.group_size} --global-batch-size {args.global_batch_size} "
        f"--rollout-max-response-len {args.response_length} --max-seq-len {args.max_seq_len} "
        "--rollout-temperature 1 --rollout-top-p 1 --balance-data "
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path examples.science_rl.agent_function.run "
        "--custom-rm-path examples.science_rl.agent_function.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--use-session-server --session-server-workers 4 --tito-model qwen35 "
        "--session-message-matcher loose_tool_call "
    )
    perf_args = (
        "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )
    algorithm_args = (
        "--advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer_args = (
        f"--optimizer adam --lr {args.learning_rate} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
    )
    misc_args = (
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --colocate "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    U.execute_train(
        train_args=" ".join((ckpt_args, rollout_args, perf_args, algorithm_args,
                             optimizer_args, sglang_args, misc_args, _wandb_args(args), args.extra_args)),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-4B",
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
