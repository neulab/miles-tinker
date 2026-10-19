"""SFT Qwen3-4B-Instruct-2507 on solved openhands trajectories.

A sibling of scripts/run_qwen3_sft.py, reusing its machinery (ray startup,
megatron arg resolution, checkpoint conversion) with three differences:

* the model is an already instruction-tuned 2507 checkpoint rather than
  Qwen3-4B-Base, so it needs its own recipe entry -- the 2507 models use
  rope_theta 5e6 and a 262144 context, where the base qwen3-4B uses 1e6/40960;
* the dataset is our own trajectories rather than openhermes; and
* the GPU count is a parameter, because this runs on whatever `general` has
  free rather than a reserved 8-GPU node.

Memory, for choosing --num-gpus-per-node. Static cost of a 4B model is
7.5G weights + 7.5G grads + 44.7G Adam = 59.6G. With tp=1 the optimizer state
shards across data-parallel ranks, so per GPU it is 15G + 44.7/N:

    N=8 -> 20.6G    N=4 -> 26.2G    N=2 -> 37.4G    N=1 -> 59.6G

48GB cards (L40S, L40, A6000, 6000Ada -- what `general` mostly has) have about
44G usable, so N>=2 fits and N=1 does not. N=4 is the comfortable default;
N=2 works with --max-tokens-per-gpu lowered, and --optimizer-cpu-offload moves
the Adam state to host RAM if you need to go lower still.
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    num_gpus_per_node: int = 4
    model_dir: str = "/root/models"
    model_name: str = "Qwen3-4B-Instruct-2507"
    megatron_model_type: str = "qwen3-4B-Instruct-2507"
    prompt_data: str = "/root/datasets/oh_solved_finish_509.jsonl"
    megatron_path: str = "/root/Megatron-LM"
    num_epoch: int = 3
    batch_size: int = 16
    max_tokens_per_gpu: int = 9216
    tensor_model_parallel_size: int = 4
    seq_length: int = 32768
    lr: float = 1e-5
    optimizer_cpu_offload: bool = False
    wandb_project: str = "tmax-15k-sft"
    wandb_group: str = "oh-sft"
    save_hf: str = ""        # if set, export HF at every checkpoint save
    save_interval: int = 100
    extra_args: str = ""


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints "
        f"--save {args.output_dir}/checkpoints "
        f"--save-interval {args.save_interval} "
        # --save-hf exports HuggingFace weights right after each checkpoint save
        # (actor.py:756). tools/convert_to_hf.py cannot do this after the fact --
        # it imports update_weight_utils from megatron_utils, which no longer
        # exists there -- and tools/convert_torch_dist_to_hf.py expects a classic
        # Megatron dist-ckpt with common.pt rather than miles' bare .distcp shards.
        + (f"--save-hf {args.save_hf} " if args.save_hf else "")
    )

    # No --apply-chat-template: sft_rollout renders the messages itself, together
    # with the per-token loss mask, so only assistant spans contribute to the loss.
    sft_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.prompt_data} "
        "--input-key messages "
        "--metadata-key metadata "
        "--tool-key tools "
        "--rollout-shuffle "
        f"--num-epoch {args.num_epoch} "
        f"--rollout-batch-size {args.batch_size} "
        f"--global-batch-size {args.batch_size} "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        "--debug-train-only "
    )

    # tp=4, not the stock recipe's tp=1. These are multi-turn agent trajectories
    # with a median of ~20.7k tokens, and a sample cannot be split across the
    # token budget -- its logits are materialised whole at seq_len x vocab x 4
    # bytes, which is 11.7 GiB at tp=1 and OOMs a 48GB card (job 10644152).
    # Tensor parallelism shards the vocab dimension, taking that to ~2.9 GiB.
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--seq-length {args.seq_length} "
    )

    # lr 1e-5 matches the stock recipe, but note this starts from an already
    # instruction-tuned checkpoint on only 509 rows; if the model's existing
    # tool-calling degrades, lower it rather than training longer.
    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style cosine "
        "--min-lr 1e-6 "
        "--lr-warmup-fraction 0.1 "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.95 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += (
            "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
        )

    wandb_args = (
        "--use-wandb "
        f"--wandb-project {args.wandb_project} "
        f"--wandb-group {args.wandb_group} "
        "--disable-wandb-random-suffix "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    U.execute_train(
        train_args=f"{ckpt_args} {sft_args} {optimizer_args} {wandb_args} {perf_args} {misc_args} {args.extra_args}",
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
