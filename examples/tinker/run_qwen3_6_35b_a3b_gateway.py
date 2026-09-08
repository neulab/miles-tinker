"""Tinker gateway launcher for Qwen3.6-35B-A3B (hybrid GDN/attention MoE, MTP block unused).

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
needed. Model args come from ``scripts/models/qwen3.6-35B-A3B.py``; the
launcher overrides its ``--mtp-num-layers 1`` with ``0`` (argparse last wins)
and never passes ``--enable-mtp-training``: the engines run without EAGLE, the
fused lm head cannot serve the MTP block's logits, and one layer of activations
is saved. The bridge itself derives ``mtp_num_layers=1`` from the HF config, so
it is ``_apply_bridge_runtime_config`` (miles/backends/megatron_utils/
model_provider.py) that carries the 0 onto the provider and skips the MTP block;
the unused HF ``mtp.*`` tensors are never requested by the Megatron-param-driven
bridge load. The engines load the MTP weights from HF and never use them.

Numerics are matched across the two sides: the engines run with
``--sglang-enable-fp32-lm-head`` and the trainer scores the same fp32 logits
through the fused chunked head (``--fused-lm-head-logprobs``). Rollout routing
replay (R3, ``--use-rollout-routing-replay``) is on by default: the gateway
caches every sample's routed experts and replays them on the datums that
train (``--tinker-routing-*``); ``--routing-policy strict`` fails a datum whose
routing is not cached instead of routing it live.

The engines need no mamba flag: the image's ``ServerArgs`` resolves
``mamba_radix_cache_strategy=auto`` to ``extra_buffer`` for this architecture
(check the engine log for ``mamba_radix_cache_strategy=extra_buffer``).

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
    ``--tp 2`` is the memory fallback (``--sequence-parallel`` becomes effective,
    expert DP halves); a resumed run must keep the TP/EP/DP of its save.
  --max-tokens-per-gpu: packed tokens per training microbatch (one 64k row).
  --context-len: model_input tokens per datum; datums are ``context_len + 1``
    tokens, the engines get ``context_len + 1`` as context length and
    ``--rollout-max-context-len context_len``.
  --max-tokens-per-datum / --unit-token-budget / --coalesce-wait-s /
    --coalesce-max-wait-s / --max-concurrent-samples / --lease-timeout-s: the
    gateway limits (``--tinker-*`` flags).
  --routing-replay / --routing-policy / --routing-cache-max-bytes /
    --routing-cache-ttl-s / --routing-prefix-stride / --r3-check: R3 knobs.
  --fused-lm-head / --fused-lm-head-chunk-size: the fused fp32 head; with
    ``--no-fused-lm-head`` the trainer falls back to ``--log-probs-chunk-size``.
  --balance-data: Karmarkar-Karp token balancing across DP ranks per unit.
  --recompute-args: Megatron recompute flags (default full uniform, 1 layer).
  --moe-token-dispatcher-type: Megatron MoE dispatcher (``alltoall``).
  --pause-generation-mode: ``retract`` (a base-weight push drops the prefix cache).
  --update-weight-buffer-size: bytes per weight-update batch (2 GiB).
  --sglang-ep-size / --sglang-mem-fraction-static / --sglang-chunked-prefill-size /
    --sglang-max-running-requests / --sglang-max-total-tokens /
    --sglang-fp32-lm-head: engine knobs (``--sglang-max-total-tokens`` bounds
    the KV pool and the pinned routed-experts host cache when set).
  --check-weight-update-equal: compare the first weight push against the
    engines (smoke runs only).
  --extra-args: appended verbatim to the ``serve_tinker.py`` argv.

Example:
  MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=10.0.80.66 \\
    python examples/tinker/run_qwen3_6_35b_a3b_gateway.py serve --mode full \\
      --actor-num-nodes 4 --rollout-num-gpus 8 --tinker-port 20613 --check-weight-update-equal
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

    # Training performance knobs: one 64k row (65536 model_input + 1 target token) per DP rank per microbatch.
    max_tokens_per_gpu: int = 65537
    context_len: int = 65536
    balance_data: bool = True
    recompute_args: str = "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
    moe_token_dispatcher_type: str = "alltoall"

    # Gateway limits: 32 rows of 64k per unit, wait for the SDK's one-datum chunks before training.
    max_tokens_per_datum: int = 65537
    unit_token_budget: int = 2_097_184
    coalesce_wait_s: float = 5.0
    coalesce_max_wait_s: float = 120.0
    max_concurrent_samples: int = 1024
    lease_timeout_s: float = 600.0

    # R3: replay the engines' routing in training; the cache lives on the gateway host.
    routing_replay: bool = True
    routing_policy: str = "natural"
    routing_cache_max_bytes: int = 128 * 1024**3
    routing_cache_ttl_s: float = 21600.0
    routing_prefix_stride: int = 256
    r3_check: bool = False

    # fp32 lm head: fused chunked head in the trainer, fp32 logits on the engines.
    fused_lm_head: bool = True
    fused_lm_head_chunk_size: int = 1024
    log_probs_chunk_size: int = 2048  # only without the fused head
    sglang_fp32_lm_head: bool = True

    # Weight pushes: a full-weight push must drop the engines' prefix cache, so retract.
    pause_generation_mode: str = "retract"
    update_weight_buffer_size: int = 2 * 1024**3
    check_weight_update_equal: bool = False
    freeze_vision_model: bool = True  # text-only RL: never train (or weight-decay) the VL tower

    # Engine fault tolerance. A single SGLang replica dying takes the whole job with it
    # otherwise: the trainer's next push_slot to the dead engine raises, the client exits 1,
    # and restart_wrapper only retries on the watchdog's exit code 2. That is how both
    # 15-node runs died on 2026-09-08 after one engine hit a CUDA device-side assert
    # ("probability tensor contains either `inf`, `nan` or element < 0") - the other ten
    # replicas kept serving the identical weights for another 25 minutes.
    # --use-fault-tolerance health-checks each rollout cell with a real generation probe and
    # lets the mini controller suspend+resume a Fatal cell, respawning the engine instead.
    # A respawned cell gets a fresh workers_hash, so end_update_weights will not mark it
    # weights-ready and the next push waits for it: it cannot silently serve stale weights.
    use_fault_tolerance: bool = True
    # Consecutive failed probes before a cell is declared unhealthy. Miles defaults to 1;
    # 3 costs ~90 s of detection instead of ~30 s and avoids respawning the cache-affinity
    # hot replica just because it is briefly busy. Health checks are already paused around
    # weight pushes (inference_controller._health_monitoring_pause), so this only guards
    # against transient load.
    rollout_health_check_failure_threshold: int = 3

    # Sampling engines.
    sglang_ep_size: int = 8
    sglang_mem_fraction_static: float = 0.75
    sglang_chunked_prefill_size: int = 16384
    sglang_max_running_requests: int = 128
    sglang_max_total_tokens: int | None = None

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
        assert self.routing_policy in ("strict", "natural"), self.routing_policy
        assert self.rollout_num_gpus % self.rollout_num_gpus_per_engine == 0, (
            f"--rollout-num-gpus {self.rollout_num_gpus} is not a multiple of "
            f"--rollout-num-gpus-per-engine {self.rollout_num_gpus_per_engine}"
        )
        assert (
            self.max_tokens_per_gpu >= self.context_len + 1
        ), f"--max-tokens-per-gpu {self.max_tokens_per_gpu} cannot hold one {self.context_len + 1}-token row"

    @property
    def sglang_context_length(self) -> int:
        """One token of slack over the datum length: the gateway clamps generation to ``context_len``."""
        return self.context_len + 1


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
        f"port {args.tinker_port}, checkpoints {args.checkpoint_root}, "
        f"r3={'on' if args.routing_replay else 'off'}/{args.routing_policy}, "
        f"fused_lm_head={'on' if args.fused_lm_head else 'off'}"
    )

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
    if args.freeze_vision_model:
        ckpt_args += "--freeze-vision-model "

    mode_args = "--tinker-train-mode full "

    tinker_args = (
        f"--tinker-server-port {args.tinker_port} "
        f"--tinker-checkpoint-root {args.checkpoint_root} "
        f"--tinker-base-model {args.model_name} "
        f"--tinker-max-tokens-per-datum {args.max_tokens_per_datum} "
        f"--tinker-unit-token-budget {args.unit_token_budget} "
        f"--tinker-coalesce-wait-s {args.coalesce_wait_s} "
        f"--tinker-coalesce-max-wait-s {args.coalesce_max_wait_s} "
        f"--tinker-max-concurrent-samples {args.max_concurrent_samples} "
        f"--tinker-lease-timeout-s {args.lease_timeout_s:g} "
        f"--tinker-routing-replay-policy {args.routing_policy} "
        f"--tinker-routing-cache-max-bytes {args.routing_cache_max_bytes} "
        f"--tinker-routing-cache-ttl-s {args.routing_cache_ttl_s:g} "
        f"--tinker-routing-prefix-stride {args.routing_prefix_stride} "
    )
    if args.r3_check:
        tinker_args += "--tinker-r3-check "

    r3_args = "--use-rollout-routing-replay " if args.routing_replay else ""

    if args.fused_lm_head:
        head_args = f"--fused-lm-head-logprobs --fused-lm-head-chunk-size {args.fused_lm_head_chunk_size} "
    else:
        head_args = f"--log-probs-chunk-size {args.log_probs_chunk_size} "

    # initial config only; AdamParams come per optim_step request
    optimizer_args = "--optimizer adam --lr 3e-6 --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32 "

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {args.ep} --expert-tensor-parallel-size {args.etp} "
        "--sequence-parallel "  # a no-op at TP1 (Megatron disables it); effective under --tp 2
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"{'--balance-data ' if args.balance_data else ''}"
        f"{args.recompute_args} "
        f"--moe-token-dispatcher-type {args.moe_token_dispatcher_type} "
    )

    # last-wins over the model definition's --mtp-num-layers 1; no --enable-mtp-training.
    # The bridge provider only sees it through _apply_bridge_runtime_config (model_provider.py).
    mtp_args = "--mtp-num-layers 0 "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-ep-size {args.sglang_ep_size} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-context-length {args.sglang_context_length} "
        f"--rollout-max-context-len {args.context_len} "
        f"--sglang-chunked-prefill-size {args.sglang_chunked_prefill_size} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"{'--sglang-enable-fp32-lm-head ' if args.sglang_fp32_lm_head else ''}"
    )
    if args.sglang_max_total_tokens is not None:
        sglang_args += f"--sglang-max-total-tokens {args.sglang_max_total_tokens} "

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
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"{'--check-weight-update-equal ' if args.check_weight_update_equal else ''}"
    )

    # --ft-components defaults to "rollout" under --use-fault-tolerance, which is what we want:
    # heal sampling engines, leave the trainer cell alone.
    ft_args = (
        f"--use-fault-tolerance "
        f"--rollout-health-check-failure-threshold {args.rollout_health_check_failure_threshold} "
        if args.use_fault_tolerance
        else ""
    )

    train_args = (
        f"{ckpt_args} {mode_args} {tinker_args} {r3_args} {head_args} {optimizer_args} "
        f"{perf_args} {mtp_args} {sglang_args} {update_weight_args} {topology_args} {misc_args} {ft_args} {args.extra_args} "
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
