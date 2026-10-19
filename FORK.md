# miles-tinker fork

A fork of [radixark/miles](https://github.com/radixark/miles) that serves the
[Tinker](https://tinker-docs.thinkingmachines.ai) training/sampling API from
`serve_tinker.py`, so that an unmodified **tinker-cookbook** (SDK 0.27.1) and
**platoon** (SDK 0.16.1) train against a miles cluster. Targets: LoRA
(multi-slot, several clients on one base) and full fine-tuning (single tenant),
Qwen3-4B-Instruct-2507 on two 8xH200 ABCI nodes, training MFU above 12%; then
Qwen3.6-35B-A3B full fine-tuning at 64k context on 15 nodes with rollout routing
replay (R3) and a fused fp32 lm head (`docs/fork/r3.md`, `fused-lm-head.md`,
`deploy-pbs-15node.md`).

Branch: `miles-tinker`. Base: upstream PR stack "multi-LoRA v2" (#2839-#2849,
`upstream/yueming/multi-lora-example` @ `06716481e`), see
`tools/fork/upstream_state.json` for the recorded sync state.

## Repository layout (github.com/neulab/miles-tinker)

| Branch | Content | Rule |
|---|---|---|
| `miles-tinker` (default) | upstream base + every fork commit | all fork work lands here; PRs target it |
| `main` | untouched mirror of `radixark/miles` `main` | never commit to it; refreshed by `upstream_sync.py --push-mirror` (or `git push origin upstream/main:main`) so GitHub's compare view (`main...miles-tinker`) shows the fork delta |
| `upstream/multi-lora-v2` | the upstream PR-stack head the fork is currently based on | bookkeeping only; dropped once the stack merges upstream |

Remotes on a working clone: `origin` = neulab/miles-tinker, `upstream` = https://github.com/radixark/miles.git.

## Base branch strategy

1. **Now:** the fork is based on the open PR stack because the Tinker gateway
   (`miles/tinker/**`, `serve_tinker.py`, the slot executor) only exists there.
   `tools/fork/upstream_sync.py` merges `upstream/yueming/multi-lora-example`.
2. **Later:** once the stack head is an ancestor of `upstream/main`
   (`git merge-base --is-ancestor`), the same tool switches the base to
   `upstream/main` automatically and records `merged_into_main: true`. If the
   stack is squash-merged (its head never becomes an ancestor of `main`), set
   `merged_into_main: true` in the state file by hand; the flag is sticky.
3. Everything the fork adds is either a candidate for an upstream PR
   ("upstream-worthy? yes" below) or deliberately fork-only (deployment for one
   cluster, client launchers, this file).

## Fork deltas by file

Rationale ids: G* = `notes/v2_gateway_protocol_gaps.md`, E* =
`notes/v2_executor_full_ft_design.md`, S* = `notes/v2_sampling_versioning.md`
(design notes kept outside the repo; the interface contract is summarized in
`docs/fork/README.md`). The `tools/fork/upstream_sync.py` conflict report quotes
the rationale column of this table, so keep one row per file or directory
(directories end with `/`).

### Protocol face (`miles/tinker/core`, `miles/tinker/server`) - fixes G1-G22

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/tinker/core/types.py` | `GatewayConfig` gains train_mode, lease_timeout_s (600 s; SDK heartbeats every 10 s, G21), retrieve_poll_timeout_s, sampler_versions_to_keep, coalesce_wait_s, max_concurrent_samples; `ModelRecord.is_lora/sdk_version`, `lora_rank` int (0 = FFT); `NotFoundError` / `ConflictError` / `PermissionDeniedError` for the checkpoint delete (404 / 409 / 403) | yes |
| `miles/tinker/core/service.py` | `end_session` (SESSION_END -> immediate lease expiry); wire-safe optim metrics (non-finite -> 0.0); abstract `ExecutorBackend` (optim_step metrics dict, discard_grads, unload_adapter, warmup, push_slot version); unnamed sampler saves return `sampling_session_id` (G1); per-session leases and X-API-Key tenant (G4); `/weights_info` + sidecar-based resume (G5); idempotent create_model (G19); sampler-version GC via unload_adapter (G10); UserInputError instead of 500s (G13); FFT latest-only sampler resolution (G18); checkpoint delete: owner from the sidecar tenant (else the live run's), 404 alike for missing and foreign checkpoints, 409 while a save or `load_state` naming it is queued or running (`_checkpoint_in_use` around save_state, load_state and LoRA sampler saves), files freed off the event loop | yes |
| `miles/tinker/core/checkpoint_delete.py` | `DELETE .../checkpoints/{kind}/{name}` on disk: an `O_NOFOLLOW` walk from the resolved checkpoint root (a symbolic link on the way is refused, so a delete never leaves the root), one rename into `<root>/.tinker-trash/` (a hidden sibling across a mount point), a daemon reaper that frees and logs each deleted checkpoint and the leftovers of an earlier gateway; the in-use scan of running and queued saves and loads | yes |
| `miles/tinker/core/planner.py` | coalesce forward_backward chunks of one window (coalesce_wait_s / max_wait) so an SDK request split into chunks trains as one unit | yes |
| `miles/tinker/core/stream.py` | loss class keyed by `loss_fn` + `loss_fn_config` only (G11); optional inputs such as `routing_handle` never split a window into units | yes |
| `miles/tinker/core/promise.py` | per-promise event for the long-poll of `/retrieve_future` (G3); delivered promises expire after 60 s (memory + GC pauses at RL request rates) | yes |
| `miles/tinker/server/app.py` | X-API-Key tenant (G4); long-poll retrieve_future capped below the SDK's 45 s (G3); `/weights_info` (G5); `GET/DELETE training_runs` (G6), the DELETE answering as the official service's checkpoint routes: 204, or 400/403/404/409/500 with a `{"detail"}` body; richer `get_info` (G7); `GET /samplers/{id}` (G8); `POST /forward` for SDK 0.16.1 (G9); category `server` (G12); `/telemetry` releases a session's slots on the SDK's SESSION_END event; tenant headers read directly and results answered as `JSONResponse` (per-request overhead) | yes |
| `miles/tinker/server/encoding.py` | default chunk `type` for SDK 0.16.1 (G2); NaN-safe JSON and 0.16.1 topk shape (G14); unknown AdamParams keys warned, not rejected (G20); request-level `num_tokens:sum`/`num_datums:sum` metrics | yes |
| `miles/tinker/server/proto_codec.py` | protobuf face kept in step with the encoding changes above | yes |
| `miles/tinker/runtime.py` (sampling functions) | `no_stop_trim` so the stop token is returned (G15); per-sample seed offset (G16); `stop=[]` -> `ignore_eos` (G17); context-length clamp (S9); `/generate` posted through aiohttp (httpcore's pool rescans every connection per request: 81 -> 1326 req/s at 256 in flight in the fake-router harness) | yes |
| `tests/fast/tinker/` | fake-backend harness and tests for every gap above, both SDK body shapes | yes |
| `docs/examples/multi-lora.md` | protocol notes for the gateway | yes |

### Training side (`miles/backends`, `miles/ray/train`, `serve_tinker.py`) - FFT executor, MFU, padding, GC

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/backends/megatron_utils/full/` | full fine-tuning executor for slot 0 (E7): reuses `run_forward_backward_pass`, defers `finalize_model_grads` to optim_step, Megatron dist-ckpt save/load, `update_weights()` as the sampler push; R3: `run_forward_backward_pass` wrapped in `rollout_replay_stage(stage="replay_backward")` | yes |
| `miles/backends/megatron_utils/actor.py` | executor selection by `--tinker-train-mode`; FFT commands no longer assert `multi_lora`; `forward_only_logprobs` replays rollout routing (`rollout_replay_stage(stage="replay_forward")`); `--tinker-r3-check` (like `--ci-test`) turns on the replay managers' `enable_check_replay_result` | yes |
| `miles/backends/megatron_utils/model.py` | grad finalize hook split so LoRA and FFT executors share it | yes |
| `miles/backends/megatron_utils/checkpoint.py` | FFT checkpoint directory layout under `<checkpoint_root>/<model_id>/weights/<name>/` | yes |
| `miles/backends/megatron_utils/lora/executor.py` | `actor_train` timer + `log_perf_data` on the tinker path (MFU instrumentation, E5); `discard_grads` | yes |
| `miles/backends/megatron_utils/lora/optimizer.py` | non-finite step veto (`skipped` metric) and per-slot metrics dict returned from optim_step | yes |
| `miles/backends/training_utils/loss.py` | sum-reduction scaling keyed on the tinker train mode, not only `is_multi_lora_enabled` | yes |
| `miles/ray/rollout/train_data_conversion.py` | `advantages` shipped with every DP shard (the client's advantages never reached the trainer: every RL loss crashed) | yes |
| `miles/backends/training_utils/data.py` | units padded to a multiple of dp_size with dummy rows so small SDK chunks do not divide by zero (E6) | yes |
| `miles/backends/training_utils/loss_hub/tinker_losses.py` | loss fixes found in the review (E2) | yes |
| `miles/ray/train/group.py` | `warmup`, `unload_adapter` and FFT weight push plumbing on the trainer controller | yes |
| `miles/tinker/runtime.py` (MilesBackend) | dp padding of units and output stripping; FFT `train_data` without `adapter_slots`; optim_step metrics passthrough and `<checkpoint_root>/perf.jsonl` (per-slot MFU records) | yes |
| `serve_tinker.py` | uvloop when installed, no uvicorn access log, 60 s keep-alive; `--tinker-train-mode {lora,full}` and the GatewayConfig flags (`--tinker-lease-timeout-s`, `--tinker-sampler-versions-to-keep`, `--tinker-coalesce-wait-s`, `--tinker-max-concurrent-samples`, `--tinker-retrieve-poll-timeout-s`, `--tinker-unit-token-budget`); no `multi_lora` assert in full mode; `--tinker-max-tokens-per-datum`, `--tinker-max-tokens-per-request`, `--tinker-coalesce-max-wait-s`, `--tinker-routing-replay-policy`, `--tinker-routing-cache-max-bytes`, `--tinker-routing-cache-ttl-s`, `--tinker-routing-prefix-stride`, `--tinker-r3-check` wired into `GatewayConfig` (`max_tokens_per_datum`, `max_tokens_per_request`, `coalesce_max_wait_s`, `routing_replay_policy`) | yes |
| `miles/utils/arguments.py`, `miles/utils/tinker_args.py` | the `--tinker-*` flags above plus `--fused-lm-head-logprobs` / `--fused-lm-head-chunk-size`; serve-mode validation: full mode forces `retract` pausing, warns on TP*EP>1 for LoRA; R3 requires `--num-experts`, rejects speculative decoding, warns on `--mtp-num-layers`; the fused head requires `policy_loss`, no per-token loss, no true-on-policy, no `--use-rollout-entropy`, no MTP, untied embeddings; `--sglang-context-length` must be at least `--rollout-max-context-len` + 6 (SGLang refuses the last 6 tokens of its context to prompts) | yes |
| `miles/utils/train_metric_utils.py`, `miles/utils/timer.py` | per-unit MFU (`actor_train_mfu`, `actor_train_tok_per_s` from `Timer().seq_lens`); the FLOPs model itself is `miles/utils/flops_utils.py` below | yes |
| `tests/fast/tinker/test_runtime.py`, `tests/fast/tinker/test_full_executor.py`, `tests/fast/backends/` | padding, FFT executor and metrics tests | yes |

### Rollout routing replay (R3), fused fp32 lm head, MFU - Qwen3.6-35B-A3B at 64k context

Operator docs: `docs/fork/r3.md`, `docs/fork/r3-trainer.md`,
`docs/fork/fused-lm-head.md`, `docs/fork/mfu.md`, `docs/fork/flags-and-launcher.md`.

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/tinker/routing_cache.py` | R3 routing store for the gateway: per-tenant entries of sampled routing (uint8 [rows, L, K]) chained through prefix parents (stride-256 blake2b prefix index + exact LCP -> `routed_experts_start_len`), pins, consumed-leaf release, TTL + byte-cap LRU sweep, stats; materializes int32 `[T-1, L, K]` at unit build | yes |
| `miles/tinker/runtime.py` (R3) | `/generate` with `return_routed_experts` (+ `routed_experts_start_len`), off-loop JSON/base64 decode with the sglang_rollout stop-edge trim and shape/range checks, `routing_id` per sequence, `rollout_routed_experts` materialized in an executor with `object_store.put` (pad rows and uncovered datums -1; `--tinker-routing-replay-policy strict|natural`), `routing_*`/`fused_lm_head` optim metrics in perf.jsonl | yes |
| `miles/tinker/server/encoding.py` (R3) | `loss_fn_inputs["routing_handle"]` (int64 `[1]`) -> `row["routing_handle"]: int`, length-1 check | yes |
| `miles/tinker/core/service.py` (R3) | tenant on sample payloads and unit rows; one WARNING per handle-less `forward_backward` when the backend replays routing; `backend.sweep()` from the lease sweep | yes |
| `miles/backends/training_utils/replay_data.py` | `rollout_replay_stage` / `active_rollout_replay_managers`: one context manager that fills every active replay manager (`m.enabled` and `--use-rollout-<name>-replay`) from `rollout_data` before a pass, holds the stage (`replay_backward` training, `replay_forward` forward-only) and clears the queues on exit, success or failure (R3 on the Tinker path) | yes |
| `miles/utils/replay_base.py` | R3 rows without any recorded routing (padding, uncovered datums) take the live router's top-k (`old_topk_fn`) instead of `arange % num_experts`; partial rows unchanged | yes |
| `miles/backends/training_utils/loss_hub/fused_lm_head.py` | fused chunked lm head for the Tinker full-FT logprob path (`--fused-lm-head-logprobs`, port of platoon's areal `fused_linear_logprobs.py`): the Megatron `output_layer` returns TP/SP-mapped hidden states and target log-probs are computed per `--fused-lm-head-chunk-size` tokens with `torch.mm(..., out_dtype=float32)` numerics (= SGLang `--enable-fp32-lm-head`), recomputed in backward, fp32 dW added into DDP `main_grad`; never materializes `[T, V]` logits (65 GB at 64k x 248k) | yes |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `calculate_log_probs_and_entropy` dispatches to `fused_lm_head.target_logprobs` when the fused head is active (entropy / true-on-policy / sampling masks asserted off) | yes |
| `miles/backends/megatron_utils/full/executor.py`, `miles/backends/megatron_utils/actor.py` (`compute_log_prob`) | `fused_lm_head.maybe_activate(args, model)` around the forward/backward pass and the forward-only log-prob pass | yes |
| `miles/utils/flops_utils.py` | layer_types-aware FLOPs model: Gated DeltaNet `linear_attention` layers (no S^2 term), attention output gate (q width x2), MTP depths (layer + eh_proj + embedding + lm head); `ensure_hf_layer_pattern(args)` fills `flops_layer_types` / `flops_linear_attention` once from the HF config (never raises); dense/MoE arithmetic unchanged without the attributes. Qwen3.6-35B-A3B at 64k: 11.20 vs 26.60 GFLOPs/token (old formula over-credited 2.4x). See `docs/fork/mfu.md` | yes |
| `miles/backends/megatron_utils/tinker_perf.py`, `miles/utils/train_metric_utils.py` | call `ensure_hf_layer_pattern(args)` before the FLOPs model on the Tinker optim window and the native `perf/actor_train_*` site | yes |
| `scripts/run_qwen3_6_35b_a3b_mtp.py` | `--sglang-mamba-scheduler-strategy` -> `--sglang-mamba-radix-cache-strategy` (the image's canonical field; the old name is only a deprecated alias) | yes |
| `tests/fast/utils/test_replay_base.py`, `tests/fast/utils/test_replay_data_sp.py`, `tests/fast/backends/training_utils/test_replay_data.py` | natural-fallback and mixed-batch top-k cases; TP2 sequence-parallel slice of `fill_replay_data`; `rollout_replay_stage` fill/stage/clear cases | yes |
| `tests/fast/backends/training_utils/test_fused_lm_head.py` | CPU tests for the fused head: fp32 reference at 1e-6, gradients, chunk sizes 1/5/64, fake 2-shard TP with and without sequence parallelism, both weight-grad paths, install guards, wrapper, dispatch | yes |
| `tests/fast/utils/test_flops_layer_types.py` | dense/MoE goldens unchanged, GDN + 64k hand formulas, 2.2-2.6 ratio band, MTP term, `ensure_hf_layer_pattern` on fake configs | yes |

### Context parallelism for full fine-tuning - 128k rows on 80 GB GPUs

Operator docs: `docs/fork/full-finetuning.md` (section "Context parallelism").

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/backends/training_utils/loss_hub/tinker_losses.py` (CP) | ported from upstream #3320 (radixark/miles 3d8b5284) onto the fork's recorder: weights/advantages/masks sliced to the rank's zigzag shard per call, padding gate on the full mask, `_gather_per_datum_outputs` (one CP all-reduce per microbatch, full-length outputs on every rank, `output_key` for value heads), empty-rank loss `logits[..., :0].sum(dtype=fp32)`, outputs recorded once under `--recompute-loss-function` | yes (keep upstream's names: a later sync can take upstream's file) |
| `miles/backends/training_utils/cp_utils.py` (`gather_cp_response_values`) | batched CP gather of per-sample response vectors plus per-sample scalars: zero-filled full-length vectors summed over the CP group, one device-to-host copy; shared by the Tinker losses and `forward_only_logprobs` | yes |
| `miles/backends/megatron_utils/actor.py` (`forward_only_logprobs`, CP) | full-length logprobs at CP>1 through the same gather (the gateway kept the first CP rank's shard), returned once per DP rank (CP/TP rank 0) | yes |
| `miles/backends/megatron_utils/cp_model_config.py`, `miles/backends/megatron_utils/model_provider.py`, `miles/backends/megatron_utils/model.py`, `miles/backends/megatron_utils/full/executor.py` (flag check) | bridge Qwen3-VL/3.5-VL providers force `calculate_per_token_loss` at CP>1 (through the gateway's deferred finalize every gradient x num_microbatches x DP*CP; elsewhere the MoE aux/z-loss and MTP loss scales): reset on every config and the routers' cached copy after `provide()`, invariant check after `get_model` (config flag, DDP's DP*CP group the size of loss.py's `intra_dp_cp` factor, unfused router with R3), flag re-checked on every `forward_backward`; MRoPE patch required at CP>1; at CP>1 a non-default `--cp-comm-type` and a given `--linear-cp-mode` go onto the provider | yes |
| `miles/utils/arguments.py` (`--linear-cp-mode`), `miles/utils/tinker_args.py` (CP) | GatedDeltaNet CP mode flag (unset: the model config keeps its own, chunkwise); serve validation: CP>1 for full mode only, thd only, no `--allgather-cp`, per-rank datum cap (#3320), R3 rejects `--moe-router-fusion` and warns at CP>1 until a `--tinker-r3-check` run | yes |
| `miles_plugins/models/qwen3_vl.py` (`is_packed_mrope_patch_installed`, CP checks) | lets the model build check that the packed-MRoPE patch is installed; a forward on a CP-sharded row fails unless the patch built rank-local positions and the bridge asked for them | yes |
| `miles/backends/megatron_utils/tinker_perf.py` (`_peak_memory_metrics`) | `actor_peak_mem_{alloc,reserved}_gib_all_ranks`: the window's peak memory maximized over all trainer ranks (one gloo all-reduce); the gateway reports rank 0's metrics | yes |
| `examples/tinker/run_qwen3_6_35b_a3b_gateway.py` (CP, `4b`) | `--cp`, `--linear-cp-mode` (headwise by default at `--cp` > 1, with a check that TP*CP divides the recipe's GDN heads), per-rank `--max-tokens-per-gpu` default (zigzag shard of one row, padded with the `--data-pad-size-multiplier` of `--extra-args`), `--model-variant 4b` (Qwen3.5-4B parity recipe: standard head, TP1 engines) | maybe |
| `deploy/clients/tinker_cp_parity.py`, `deploy/abci/clients/cp_parity.sh` | CP parity probe (`run` dumps forward/forward_backward/optim results for a seeded datum set plus lone 4000/8000-token datums, with TP and pad multiplier; `compare` reports per-chunk, zigzag-boundary and GatedDeltaNet-split deltas, per-datum short-row maxima, loss deltas, grad_norm ratios with thresholds) and its CLIENT_SCRIPT | no |
| `tools/fork/gdn_cp_grad_probe.py` | one-node torchrun probe of a Megatron GatedDeltaNet layer on a get_batch-shaped packed row: chunkwise/headwise CP against a no-CP reference, per-parameter and pre-/post-split input-gradient cosine and norm ratio with the loss after the CP split | no |
| `tests/fast/backends/training_utils/loss/test_tinker_loss_cp_consistency.py`, `cpu_cross_entropy.py`, `tests/fast/backends/megatron_utils/test_cp_model_config.py`, `tests/fast/backends/training_utils/test_replay_data_cp.py`, `tests/fast/test_qwen3_vl_cp_mrope.py`, `tests/fast/fork/test_tinker_cp_parity.py`, `tests/fast/tinker/server/test_cp_parity_client.py` | gloo CP 1/2/4 loss and gather tests (the stand-in replaces Megatron's fused CE on CPU), fake-provider per-token-loss reset and invariant tests, R3 routing-row alignment under CP 1-8 / TP-SP 1-4 with a negative control, the MRoPE forward checks, parity client tests | yes (the loss test: upstream's file name) |

### Weight sync over HTTP keep-alive, failed pushes, dead trainers

A full-weight push died when the trainer posted a bucket to the engines 5 s after the previous one: SGLang's uvicorn
and httpx both drop an idle keep-alive connection after 5 s, and a request written on a connection the engine was
closing got a TCP reset (`httpx.ReadError`). The trainer cell was killed, the push leaked the inference controller's
lock and kept the engine health checks paused, and the gateway kept answering healthy. Operator notes:
`deploy/abci/README.md` section 8.

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/utils/http_utils.py` | `GeneralHttpClientProvider` drops an idle pooled connection after 2 s (`_KEEPALIVE_EXPIRY`, httpx's default is 5 s), well inside the engines' own idle limit | yes |
| `miles/ray/specs/inference.py` | every engine is launched with `SGLANG_TIMEOUT_KEEP_ALIVE=600` (SGLang's default is 5; a value in the driver's environment wins) | yes |
| `miles/backends/sglang_utils/sglang_api_client.py` | `pause_generation`, `continue_generation` and `update_weight_version` (without `abort_all_requests`) are sent once more after a connection reset (`ReadError`, `WriteError`, `RemoteProtocolError`); the weight buckets, `begin/end_weight_update` and the LoRA calls never are | yes |
| `miles/ray/rollout/inference_controller.py` | `abort_update_weights` releases the lock of a failed update and marks no engine weights-ready; `end_update_weights` and the abort put the engine health checks back the way the update found them (the Tinker gateway never calls `prepare_rollout`, so they had stayed paused from its first push on) | yes |
| `miles/ray/train/group.py` (weight-update frame) | `push_slot`, `unload_adapter` and `update_weights` run inside `_engine_weight_update`: end on success, abort on any exception (cancellation included); `has_alive_cell()` | yes |
| `miles/tinker/core/service.py`, `miles/tinker/runtime.py`, `miles/tinker/server/app.py` (health) | `ExecutorBackend.health_problem()`: `MilesBackend` reports a trainer with no alive cell, and `/api/v1/healthz` and `/health` answer 503 with the reason | yes |
| `miles/ray/placement_group.py` | `sort_bundles_head_last`: actor and engine slots by node IP and GPU id with the Ray head node (`node:__internal_head__`) last, so trainer ranks stay off the node of the driver (and of the client on ABCI) | yes |
| `deploy/abci/tinker_gateway.pbs` (roles, monitor) | trainer nodes exclude the head (it sorts last in `roles.tsv`; `launcher.log` prints a `node roles:` line); the wait for the gateway and the monitor end the job after `GATEWAY_DEAD_PROBES` (4) consecutive 503s from its healthz | no |
| `tools/fork/keepalive_race_repro.py` | CPU reproduction of the race: SGLang-like uvicorn servers (uvloop, h11, `timeout_keep_alive`) and miles' own engine client, per-engine scan or the trainer's bucket fan-out, optionally CPU-starved | no |
| `tests/fast/utils/test_http_utils.py`, `tests/fast/ray/specs/test_inference.py`, `tests/fast/backends/sglang_utils/test_sglang_api_client.py`, `tests/fast/ray/rollout/test_inference_controller.py`, `tests/fast/ray/train/test_group_weight_update.py`, `tests/fast/ray/train/dummy_actor.py`, `tests/fast/tinker/test_trainer_slot_commands.py`, `tests/fast/tinker/test_runtime.py`, `tests/fast/tinker/server/test_app.py`, `tests/fast/ray/test_placement_group.py`, `tests/fast/fork/test_deploy_dry_run.py`, `tests/fast/fork/test_frontier_deploy.py`, `tests/fast/fork/test_swe_deploy.py` (keep-alive, fail fast, head last) | idle expiry and reuse, engine keep-alive default and override, which calls a reset retries, the update frame on every exit path (real Ray cells and lock), healthz 503, head-last placement, roles and service URL order, the startup wait's and the monitor's probe count | yes (`deploy` tests: no) |

### Muse-Glimmer-30B - megatron-bridge's builder-backed models, the head's output transform, an image overlay

Operator docs: `deploy/abci/overlays/muse-glimmer/README.md`, `docs/fork/flags-and-launcher.md` (presets).

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/backends/megatron_utils/model_provider.py` (builder-backed bridge) | megatron-bridge families that build through its ModelBuilder (`USE_MODEL_CONFIG_FOR_CONVERSION`, Muse Glimmer the first) only keep a deprecated `to_megatron_provider()` (for Glimmer an empty GPT provider whose `finalize()` divides by zero): build them from `get_model_config()` with the runtime-config copy, `transformer_impl` from args (the bridge's default `local` attention is O(S^2) with no THD), one `finalize()`, the config's builder per stage, the same post-build checks; CP>1 refused (their models slice their own input) | yes |
| `miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py` (export shape) | newer megatron-bridge (8cd3466d) yields `(hf_name, weight)` from `export_hf_weights` unless asked `with_megatron_names`, then a tuple of source names; radixark's yields one name: ask where the export takes the flag and normalize to a tuple (unpadding and quantization key on the first name, units group by the tuple), else every push failed | yes |
| `miles/backends/megatron_utils/lora/utils.py` (`is_lora_model`) | any parameter name containing "adapter" counted as LoRA, so Muse Glimmer's `vision_adapter` sent `save` down the LoRA path: a full fine-tuning `save_state` wrote an empty adapter (88 KB, nothing to resume from) and `hf_export` an adapter-only export. LoRA is now a PEFT config, a `lora_*` name, or a parameter under a module named `adapter` or `adapters` (upstream decides at build time, `lora_attached`) | yes |
| `miles/backends/training_utils/loss_hub/fused_lm_head.py` (output transform) | the fused head bypassed the head's `forward`, so Glimmer's logit multiplier (x0.196) and softcap (20) were skipped: `install` reads the head class's transform (plain `ColumnParallelLinear` and GPTModel's `LinearCrossEntropyModule`, the head of every Qwen preset; megatron-bridge's `MuseGlimmerOutputLayerMixin`) and refuses any other, and each chunk applies it in fp32 in SGLang's order, with its derivative in backward | yes |
| `scripts/models/muse-glimmer-30B.py`, `tests/snapshots/model_args/muse-glimmer-30B.txt` | the text model's shape for `hf_validate_args` and the FLOPs model (the bridge builds the model); a vocab divisor of 1 keeps Megatron's padded vocabulary at 202048 | yes |
| `examples/tinker/run_qwen3_6_35b_a3b_gateway.py` (`glimmer-30b`) | recipe: TP4 (TP up to 8), TP2 engines, no routing replay, no CP; the recipe table now says which variants are dense, their TP sizes and whether they take CP | no |
| `deploy/abci/presets/muse-glimmer-30b-128k.env`, `deploy/abci/presets/muse-glimmer-30b-64k.env` | 5-node (3 train, 8 TP2 engines) 128K gateway with `--context-len 131066` (Glimmer's 131072 positions on the engines), and the 2-node 64K gateway that checks the first weight push | no |
| `deploy/abci/overlays/muse-glimmer/` | build script and docs of the overlay (transformers 5.15.0 with a qwen3_asr registration patch, radixark Megatron-Bridge 8cd3466d + NVIDIA PR #6256 vendored as a diff, Megatron-LM fd15ee20) passed as `EXTRA_SITE` and `ROOTFS` | no |
| `tests/fast/backends/megatron_utils/test_model_provider_builder.py`, `tests/fast/backends/megatron_utils/test_hf_weight_iterator_bridge.py`, `tests/fast/backends/training_utils/test_fused_lm_head.py` (transform), `tests/fast/backends/megatron_utils/test_lora_utils.py` (`is_lora_model`), `tests/fast/fork/test_qwen36_gateway_launcher.py` (Glimmer), `tests/fast/fork/test_deploy_dry_run.py` (Glimmer presets) | builder path with a fake bridge, both export shapes through the weight iterator, the transform against an fp32 autograd reference (also on fake TP2), head classes known and refused, LoRA parameters told from a model's own adapter modules, the recipe's argv and layouts, the presets rendered on 5 and 2 nodes with the overlay | no |

### Fork-only tooling, deployment and docs

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `README.md` | fork banner prepended above upstream's README (conflict policy: keep the banner, take upstream's body) | no |
| `FORK.md` | this file | no |
| `tools/fork/` | `upstream_sync.py` + `upstream_state.json`: base selection, merge/rebase, conflict report, fast tests, state record | no |
| `tests/fast/fork/` | tests for the sync tool, the gateway launchers, the PBS/service dry runs and the checkpoint pruner | no |
| `tools/fork/dump_sglang_server_args.py`, `tests/fast/fork/data/sglang_server_args_fields.txt` | `dataclasses.fields(ServerArgs)` snapshot of the serving image; the launcher test rejects any `--sglang-<name>` that is not a field (caught the stale `--sglang-mamba-scheduler-strategy`) | no |
| `tests/fast/fork/test_qwen36_gateway_launcher.py` | flag multiset of the 35B recipe + the `--sglang-*` allowlist; every gateway preset's options parse with its launcher, and the 128K preset's `serve_tinker.py` argv is fcs-q38's but for `--tinker-base-model` | no |
| `tests/fast/fork/test_deploy_dry_run.py`, `tests/fast/fork/test_prune_tinker_checkpoints.py` | DRY_RUN coverage of the PBS job, service helper, guard self-test, submit presets, gateway presets (read as bash sources them, rendered like their job variables, the 128K preset pinned to fcs-q38's strings), `CLIENT_CMD` (runs as written in `CLIENT_DIR`, early failures), broken pipes (each pipeline against a fake writer that outlasts a reader that stops early; no such reader left under errexit and pipefail); pruner rules | no |
| `.claude/skills/upstream-sync/` | Claude Code skill wrapping the sync tool and the conflict policy | no |
| `deploy/abci/` | ABCI PBS launchers for N nodes (2 for the 4B smoke, 5/15 for Qwen3.6-35B-A3B): trainers are the `TRAIN_NODES` lowest-IP nodes other than the head, engine nodes also host one Toolathlon service + memory guard when `SERVICES=1`; one container per node, `CLIENT_SCRIPT` or `CLIENT_CMD` runs the client on the head node and the job ends when it exits; `clients/*.sh` ready-made client scripts, `pull_image.pbs`, `smoke_container.pbs`, README runbook (sections 3-8) and `docs/fork/deploy-pbs-15node.md`. The PBS job and the service helper have a `DRY_RUN=1` mode that runs on a login node (`tests/fast/fork/test_deploy_dry_run.py`). Every pipeline the job runs under errexit and pipefail reads its input to the end: a reader that stops early (`head`, `grep -q`, awk's `exit`) can leave its writer to die of SIGPIPE, which ended a DRY_RUN at start-up (status 141 at the head's IPoIB address) and would end a job the same way | no |
| `deploy/abci/tinker_gateway.pbs` | N-node PBS job (trainers = TRAIN_NODES lowest IPoIB IPs other than the head's, as `placement_group.sort_bundles_head_last`, `--actor-num-nodes/--rollout-num-gpus` derived, Toolathlon services before Ray, release health quorum + monitor (head alive, gateway healthz not 503, services), C10 client exports, 4-phase teardown, `DRY_RUN=1` on a login node); `PRESET=` reads a gateway preset (parsed, never sourced; a non-empty job variable wins; `HF_MODEL` + `HF_REVISION` select the snapshot; the sampling timeout goes to the client); `CLIENT_CMD` is written verbatim to `$JOBDIR/client-cmd.sh` and run like a `CLIENT_SCRIPT`, in `CLIENT_DIR` (default `PBS_O_WORKDIR`), checked before any node starts. Without `PRESET` and `CLIENT_CMD` every launch renders byte-identically to before | no |
| `deploy/abci/presets/` | gateway presets for `PRESET=`: `qwen3.8-27b-128k` is the fcs-q38 gateway (rasp `launch/abci/submit.sh frontiercs-*`) advertising the HF id instead of the snapshot path | no |
| `deploy/abci/toolathlon_service.sh`, `deploy/abci/toolathlon.env`, `deploy/abci/openreward-node-memory-guard.py` | one Toolathlon service + node memory guard per engine node (trimmed from the platoon release helper; C12 non-secret env with the derived-SIF sha pins; guard is a verbatim release copy) | no |
| `deploy/abci/kill-leftovers.sh`, `deploy/abci/submit_toolathlon.sh` | teardown sweep run on every node (Toolathlon containers have no PID namespace); qsub presets smoke/recursive/nonrecursive with `--dry-run` | no |
| `deploy/abci/clients/platoon_toolathlon.sh`, `deploy/abci/clients/prune_tinker_checkpoints.py`, `deploy/abci/clients/smoke_q36.sh` | platoon Toolathlon client under `restart_wrapper` (lease 600 s / delay 700 s resume), checkpoint pruner driven by `checkpoints.jsonl`, the 5-node Qwen3.6 smoke (design 10.2) | no |
| `deploy/clients/` | client launchers (refuse login nodes via `hostguard`; run them through `CLIENT_SCRIPT`): cookbook guess_number async, math_rl arithmetic smoke, platoon textcraft synth, raw SDK smoke, `tinker_mfu_bench.py` (training-MFU benchmark with RL-shaped datums) | no |
| `deploy/clients/tinker_r3_probe.py` | SDK 0.16.1 R3 probe: `routing_id` on every sequence, `routing_handle` datum, `routing_covered/uncovered_datums` metrics | no |
| `examples/tinker/run_qwen3_4b_gateway.py` | `execute_train`-shaped launcher for the gateway (lora/full) on 8 + 8 GPUs | maybe (as `examples/tinker/`) once FFT lands upstream |
| `examples/tinker/run_qwen3_6_35b_a3b_gateway.py` | Qwen3.6-35B-A3B full-FT gateway recipe (bridge load, TP1/EP8/DP32, `--mtp-num-layers 0` without MTP training, R3 on, fused fp32 head + `--sglang-enable-fp32-lm-head`, 64k gateway limits, `--balance-data`, `--tp 2` fallback, no `--sglang-mamba-*` flag) | maybe (as `examples/tinker/`) once FFT lands upstream |
| `docs/fork/` | operator docs: `README.md` (overview), `full-finetuning.md`, `acceptance.md`, `r3.md`, `r3-trainer.md`, `fused-lm-head.md`, `mfu.md`, `flags-and-launcher.md`, `deploy-pbs-15node.md` | no |
| `docs/fork/r3.md` | R3 protocol, capture/cache/materialization semantics, metrics, run-checklist notes | no |
| `docs/fork/r3-trainer.md` | trainer-side R3: data contract, the one context manager around the pass, natural fallback, SP note | no |
| `docs/fork/fused-lm-head.md` | why the fused head exists, numerics parity with `--sglang-enable-fp32-lm-head`, memory, flags/requirements, smoke validation steps | no |
| `docs/fork/mfu.md` | MFU formulas, Qwen3.6 numbers at 64k, the 0.05-0.35 sanity band | no |
| `docs/fork/flags-and-launcher.md` | the `--tinker-*` / fused-head flags, their `args.<attr>` and `GatewayConfig` names, serve-time validation, the 35B launcher decisions, the gateway presets' options and their `serve_tinker.py` argv, allowlist regeneration | no |
| `docs/fork/deploy-pbs-15node.md` | topology, service/secret/teardown contracts (C10, C12) of the N-node job | no |

## Running the fast tests

No GPU is needed. The tinker tests must run with `--confcutdir` so the repo
root `tests/conftest.py` (which imports GPU-only fixtures) is skipped:

```bash
module load python/3.12/3.12.9   # ABCI login node
PY=/path/to/venv/bin/python      # pytest, pytest-asyncio, tinker>=0.27, fastapi, httpx, uvicorn, numpy, zstandard, torch (cpu), ray
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/fast/tinker -p no:cacheprovider --confcutdir=tests/fast/tinker -o addopts=""
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/fast/fork   -p no:cacheprovider --confcutdir=tests/fast/fork   -o addopts=""
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/fast/launch_scripts -p no:cacheprovider --confcutdir=tests/fast/launch_scripts -o addopts=""
# R3 / fused head / MFU (CPU). Run these by path: the whole tests/fast/utils and
# tests/fast/backends trees also hold files that import sglang, which the venv lacks.
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $PY -m pytest \
  tests/fast/utils/test_replay_base.py tests/fast/utils/test_replay_data_sp.py \
  tests/fast/utils/test_flops_layer_types.py tests/fast/utils/test_flops_from_hf_config.py \
  tests/fast/utils/test_train_metric_utils.py tests/fast/utils/test_dp_schedule.py \
  tests/fast/backends/training_utils/test_fused_lm_head.py tests/fast/backends/training_utils/test_replay_data.py \
  tests/fast/backends/megatron_utils/test_tinker_perf.py \
  -p no:cacheprovider --confcutdir=tests/fast -o addopts=""
```

Inside the serving container the same commands work with the image's python
(`deploy/abci/smoke_container.pbs` runs them).

`tests/fast/fork/test_qwen36_gateway_launcher.py` checks every `--sglang-*`
flag of the 35B launcher against the serving image's `ServerArgs` fields.
Regenerate the snapshot after an image change (`PYTHONNOUSERSITE=1` is required
on the login node because `~/.local` site-packages shadow the image's PIL):

```bash
SIF=/groups/gai51740/aci18912it/miles-tinker/images/miles-latest-cu13.sif
singularity exec --env PYTHONNOUSERSITE=1 $SIF python tools/fork/dump_sglang_server_args.py > tests/fast/fork/data/sglang_server_args_fields.txt
```

## Upstream sync procedure (summary)

```bash
git switch miles-tinker && git status --short          # clean tree on the fork branch
python tools/fork/upstream_sync.py --check             # 0 up to date, 1 sync needed, 2 conflicts expected
MILES_FAST_PYTHON=$PY python tools/fork/upstream_sync.py   # merge --no-ff, fast tests, state file
```

On conflicts the tool stops with a per-file report. Policy: prefer upstream for
files not in the tables above; re-apply the fork intent (the rationale column)
for listed files; never drop an upstream fix; then `git merge --continue` and
rerun the tool. Full procedure and checklist: `.claude/skills/upstream-sync/SKILL.md`.
