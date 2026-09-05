"""Tinker gateway launcher for Qwen3-4B-Instruct-2507 on two 8-GPU nodes.

Serves ``serve_tinker.py`` disaggregated: one node trains (Megatron, TP1/PP1,
DP over 8 GPUs) and one node samples (8 SGLang engines, TP1). ``--mode lora``
shares the frozen base across ``n_adapters`` LoRA slots (one per Tinker
training client); ``--mode full`` fine-tunes the base weights for a single
tenant. The gateway advertises ``model_name`` so unmodified tinker-cookbook /
platoon clients can pass the HuggingFace id.

Requires: a Ray cluster already joined by both nodes (``MILES_SCRIPT_EXTERNAL_RAY=1``,
``MASTER_ADDR`` set), the HF snapshot under ``--hf-checkpoint``, and the
``tinker`` SDK importable by the driver (proto codec).

Args:
  --mode: ``lora`` (multi-slot adapters) or ``full`` (full fine-tuning, emits
    ``--tinker-train-mode full``).
  --hf-checkpoint: local HF snapshot of Qwen/Qwen3-4B-Instruct-2507 (default
    ``<model-dir>/Qwen3-4B-Instruct-2507``, bridge mode, no conversion).
  --tinker-port: HTTP port the gateway binds on 0.0.0.0.
  --checkpoint-root: where ``tinker://`` checkpoints are written.
  --n-adapters / --lora-rank / --lora-alpha / --target-modules: LoRA slot pool
    (``--mode lora``); the SGLang ``max_loras_per_batch`` follows ``--n-adapters``
    (``--sglang-max-loras-per-batch`` is shadowed by miles for multi-LoRA).
  --max-tokens-per-gpu: packed tokens per training microbatch (raise until OOM).
  --recompute-args: extra Megatron recompute flags, e.g.
    ``"--recompute-granularity full --recompute-method block --recompute-num-layers 8"``.
  --pause-generation-mode: how engines pause during weight pushes; ``in_place``
    keeps the prefix cache (default ``retract`` refills it on every push).
  --update-weight-buffer-size: bytes per weight-update batch (1 GiB).
  --sglang-mem-fraction-static / --sglang-context-length /
    --sglang-chunked-prefill-size / --sglang-max-running-requests: engine knobs.
  --extra-args: appended verbatim to the ``serve_tinker.py`` argv.

Example:
  MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=10.0.80.66 \\
    python examples/tinker/run_qwen3_4b_gateway.py serve --mode lora --tinker-port 20613
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()

    mode: str = "lora"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    checkpoint_root: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    # Disaggregated split across two nodes: node A trains, node B samples.
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int = 8
    rollout_num_gpus_per_engine: int = 1
    tp: int = 1

    # LoRA slot pool (mode=lora); per-client rank comes from the SDK, capped by lora_rank.
    lora_rank: int = 32
    lora_alpha: int = 32  # Tinker convention: alpha fixed at 32 regardless of rank
    n_adapters: int = 8
    target_modules: str = "linear_qkv,linear_proj,linear_fc1,linear_fc2"

    # Training performance knobs.
    max_tokens_per_gpu: int = 16384
    recompute_args: str = ""

    # Weight pushes: in_place keeps in-flight requests and the prefix cache across a push.
    pause_generation_mode: str = "in_place"
    update_weight_buffer_size: int = 1024**3

    # Sampling engines.
    sglang_mem_fraction_static: float = 0.85
    sglang_context_length: int = 32768
    sglang_chunked_prefill_size: int = 4096
    sglang_max_running_requests: int = 512

    tinker_port: int = 10613
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/Qwen3-4B-Instruct-2507"
        if self.checkpoint_root is None:
            self.checkpoint_root = f"{self.output_dir}/tinker/{self.run_id}"
        assert self.mode in ("lora", "full"), f"--mode must be lora or full, got {self.mode!r}"
        assert self.pause_generation_mode in ("abort", "retract", "in_place"), self.pause_generation_mode


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the HF checkpoint. Run once per shared filesystem."""
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {args.model_name} --local-dir {args.hf_checkpoint}")


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Serve the Tinker gateway (idles until clients connect)."""
    print(
        f"[run] tinker gateway mode={args.mode}: {args.actor_num_nodes}x{args.actor_num_gpus_per_node} train + "
        f"{args.rollout_num_gpus} rollout GPUs, port {args.tinker_port}, checkpoints {args.checkpoint_root}"
    )

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "

    if args.mode == "lora":
        mode_args = (
            f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
            f'--target-modules "{args.target_modules}" '
            "--no-gradient-accumulation-fusion "
            f"--multi-lora-n-adapters {args.n_adapters} "
            "--sglang-lora-backend triton "
        )
    else:
        mode_args = "--tinker-train-mode full "

    tinker_args = (
        f"--tinker-server-port {args.tinker_port} "
        f"--tinker-checkpoint-root {args.checkpoint_root} "
        f"--tinker-base-model {args.model_name} "
    )

    # initial config only; AdamParams come per optim_step request
    optimizer_args = "--optimizer adam --lr 1e-5 "

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"{args.recompute_args} "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-context-length {args.sglang_context_length} "
        f"--sglang-chunked-prefill-size {args.sglang_chunked_prefill_size} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
    )

    update_weight_args = (
        f"--pause-generation-mode {args.pause_generation_mode} "
        f"--update-weight-buffer-size {args.update_weight_buffer_size} "
    )

    topology_args = (
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} {mode_args} {tinker_args} {optimizer_args} "
        f"{perf_args} {sglang_args} {update_weight_args} {topology_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-4B-Instruct-2507",
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
