"""Tinker gateway launcher for Qwen3.6-35B-A3B (hybrid GDN/attention MoE, 1 MTP layer).

Serves ``serve_tinker.py`` disaggregated: ``actor_num_nodes`` x 8 GPUs train
(Megatron, TP1 / EP8 / ETP1, distributed optimizer, full uniform recompute) and
``rollout_num_gpus`` GPUs sample (SGLang, one 8-GPU TP8/EP8 engine per node).
Only ``--mode full`` is wired: the base weights are trained for a single tenant
(``docs/fork/full-finetuning.md``). The gateway advertises ``model_name`` so
unmodified tinker-cookbook / platoon clients can pass that id.

The weights load through megatron-bridge (``--megatron-to-hf-mode bridge``):
the HF snapshot is a ``Qwen3_5MoeForConditionalGeneration`` wrapper, which the
bridge maps to its Qwen3.5-VL MoE provider (language model + vision tower + the
MTP block read from ``mtp_num_hidden_layers``), so no torch_dist conversion is
needed. Model args come from ``scripts/models/qwen3.6-35B-A3B.py``.

Requires: a Ray cluster already joined by every node (``MILES_SCRIPT_EXTERNAL_RAY=1``,
``MASTER_ADDR`` set), the HF snapshot under ``--hf-checkpoint``, and the
``tinker`` SDK importable by the driver (proto codec).

Args:
  --mode: ``full`` (full fine-tuning, emits ``--tinker-train-mode full``); ``lora``
    is not supported by this recipe yet.
  --model-name: HuggingFace id the gateway advertises.
  --hf-checkpoint: local HF snapshot (default ``<model-dir>/<model basename>``).
  --tinker-port: HTTP port the gateway binds on 0.0.0.0.
  --checkpoint-root: where ``tinker://`` checkpoints are written.
  --actor-num-nodes / --actor-num-gpus-per-node / --rollout-num-gpus /
    --rollout-num-gpus-per-engine: the train / sample split.
  --tp / --ep / --etp: tensor, expert and expert-tensor parallel sizes (PP=CP=1).
  --max-tokens-per-gpu: packed tokens per training microbatch; keep it small
    when the expert DP is 1 (the expert optimizer state is not sharded).
  --recompute-args: Megatron recompute flags (default full uniform, 1 layer).
  --moe-token-dispatcher-type: Megatron MoE dispatcher (``alltoall``).
  --pause-generation-mode: ``retract`` (a base-weight push drops the prefix cache).
  --update-weight-buffer-size: bytes per weight-update batch (1 GiB).
  --sglang-ep-size / --sglang-mem-fraction-static / --sglang-context-length /
    --sglang-chunked-prefill-size / --sglang-max-running-requests: engine knobs.
  --extra-args: appended verbatim to the ``serve_tinker.py`` argv.

Example:
  MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=10.0.80.66 \\
    python examples/tinker/run_qwen3_6_35b_a3b_gateway.py serve --mode full \\
      --actor-num-nodes 1 --rollout-num-gpus 8 --max-tokens-per-gpu 8192 --tinker-port 20613
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MEGATRON_MODEL_TYPE = "qwen3.6-35B-A3B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()

    mode: str = "full"
    model_name: str = "apurvaga/Qwen3.6-35B-A3B-preserve-thinking"
    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    checkpoint_root: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    # Disaggregated split: the actor nodes train, the rollout GPUs sample (one 8-GPU engine per node).
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 4
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int = 88
    rollout_num_gpus_per_engine: int = 8

    # Megatron parallelism: TP1 (num_query_groups=2 caps TP at 2), experts spread over 8 ranks.
    tp: int = 1
    ep: int = 8
    etp: int = 1

    # Training performance knobs.
    max_tokens_per_gpu: int = 65537
    recompute_args: str = "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
    moe_token_dispatcher_type: str = "alltoall"

    # Weight pushes: a full-weight push must drop the engines' prefix cache, so retract.
    pause_generation_mode: str = "retract"
    update_weight_buffer_size: int = 1024**3

    # Sampling engines.
    sglang_ep_size: int = 8
    sglang_mem_fraction_static: float = 0.8
    sglang_context_length: int = 65537
    sglang_chunked_prefill_size: int = 16384
    sglang_max_running_requests: int = 256

    tinker_port: int = 10613
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name.rsplit('/', 1)[-1]}"
        if self.checkpoint_root is None:
            self.checkpoint_root = f"{self.output_dir}/tinker/{self.run_id}"
        assert self.mode == "full", (
            f"--mode must be full for this recipe, got {self.mode!r} "
            "(LoRA on Qwen3.6-35B-A3B needs the bshd/TP2 layout of scripts/run_qwen3_5_35b_a3b_lora.py)"
        )
        assert self.pause_generation_mode in ("abort", "retract", "in_place"), self.pause_generation_mode
        assert self.rollout_num_gpus % self.rollout_num_gpus_per_engine == 0, (
            f"--rollout-num-gpus {self.rollout_num_gpus} is not a multiple of "
            f"--rollout-num-gpus-per-engine {self.rollout_num_gpus_per_engine}"
        )


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
        f"[run] tinker gateway mode={args.mode} {args.model_name}: "
        f"{args.actor_num_nodes}x{args.actor_num_gpus_per_node} train (TP{args.tp}/EP{args.ep}/ETP{args.etp}) + "
        f"{args.rollout_num_gpus} rollout GPUs ({args.rollout_num_gpus_per_engine} per engine), "
        f"port {args.tinker_port}, checkpoints {args.checkpoint_root}"
    )

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "

    mode_args = "--tinker-train-mode full "

    tinker_args = (
        f"--tinker-server-port {args.tinker_port} "
        f"--tinker-checkpoint-root {args.checkpoint_root} "
        f"--tinker-base-model {args.model_name} "
    )

    # initial config only; AdamParams come per optim_step request
    optimizer_args = "--optimizer adam --lr 1e-6 --use-distributed-optimizer "

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {args.ep} --expert-tensor-parallel-size {args.etp} "
        f"{args.recompute_args} "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--moe-token-dispatcher-type {args.moe_token_dispatcher_type} "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-ep-size {args.sglang_ep_size} "
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
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
