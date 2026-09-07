# Full fine-tuning through the Tinker gateway

`serve_tinker.py --tinker-train-mode full` (launchers: `examples/tinker/
run_qwen3_4b_gateway.py` and `run_qwen3_6_35b_a3b_gateway.py`, `serve --mode
full`) trains the **base weights** of the served model instead of LoRA
adapters. The Tinker SDKs only expose `create_lora_training_client`, so
this page documents where the full-fine-tuning (FFT) gateway deviates from
Tinker's LoRA semantics and why unmodified clients still work.

Code: `miles/backends/megatron_utils/full/executor.py` (P2),
`miles/tinker/core/service.py` (`_resolve_lora_config`, `_resolve_sampler`,
`weights_info`) and `miles/tinker/runtime.py` (P1/P2). Design notes:
`v2_executor_full_ft_design.md` section 4, `v2_sampling_versioning.md` section 5.

## Single tenant, `lora_config` ignored

- The deployment has exactly one trainable model: `GatewayConfig.n_slots == 1`
  (slot 0 = the base weights); the service refuses any other `n_slots`.
- `create_model` accepts **any** `lora_config` (cookbook and platoon always
  send `{"rank": N, ...}`); it is logged (`train_mode=full: ignoring lora_config
  ...`) and ignored, and the record is created with `is_lora=False,
  lora_rank=0`. A client-side `lora_rank` / `learning_rate` meant for LoRA is
  therefore a no-op for the rank and a real change for the learning rate: use
  an FFT learning rate (1e-6 in `deploy/clients/cookbook_guess_number_async.py`
  `learning_rate=1e-6`; the LoRA default is 3e-5).
- A second concurrent training client gets `UserInputError("this full
  fine-tuning gateway serves one model at a time")` until the first model's
  lease expires (`--tinker-lease-timeout-s` after the last heartbeat: flag
  default 3600 s, the 35B launcher passes 600 s) or its session ends. Run one
  client at a time.
- Every fresh `create_model` starts from the pristine base: `load_slot(slot=0,
  ckpt_path=None)` reloads the weights the server started from (the HF
  snapshot through the bridge) and zeroes the optimizer moments and step
  clocks in place (`reset_to_base`).

## Training semantics

- `forward_backward` calls accumulate gradients in the fp32 DDP buffers across
  calls (deferred `finalize_model_grads`: the DP reduce-scatter runs once, at
  `optim_step`). `overlap_grad_reduce` / `overlap_param_gather` must stay off
  (asserted at serve time), `--use-dynamic-batch-size --max-tokens-per-gpu`
  are required.
- `optim_step` applies the request's `AdamParams` (lr, betas, eps, weight decay,
  `grad_clip_norm`, 0 = no clip) and returns the metrics dict of the contract.
  A non-finite gradient norm vetoes the step (grads zeroed, `skipped: 1`).
- Units are padded to a multiple of `dp_size` with dummy rows (2 tokens,
  target_len 1, zero weights) whose outputs are stripped, so an SDK chunk of
  any size trains; `adapter_slots` is omitted from the train data.

## Latest-only sampler (the documented deviation)

Tinker guarantees that a sampler path pins a weight version. The FFT gateway
cannot: each SGLang engine holds one copy of the base weights and
`save_weights_for_sampler` is the ordinary full weight sync
(`TrainerController.update_weights()`), which returns the engines' global
`weight_version`. Therefore:

- `save_weights_for_sampler` (named or unnamed) pushes the current weights,
  stores the returned engine version as the model's `sampler_version` and
  returns `tinker://<model_id>/sampler_weights/<name>` as usual. Nothing is
  written to disk for sampler saves.
- `_resolve_sampler` validates the path (model owned by the tenant, version
  <= `sampler_version`) and returns `None`, i.e. **every sampling session of
  the model samples the newest pushed weights**, including sessions created
  from older sampler paths.
- Correctness for RL clients: the returned per-token `logprobs` are those of
  the weights that actually produced each token, so importance-sampling / PPO
  / CISPO ratios computed by the client remain valid; only the trajectory is
  "off" by at most the client's own pipelining (`max_steps_off_policy` in the
  cookbook, `max_staleness` in platoon). With `--pause-generation-mode
  in_place` a push freezes in-flight requests and resumes them on the new
  weights (mixed-version sequences; same behaviour as miles' native async loop).
- Base-model sampling sessions (`create_sampling_session(base_model=...)`,
  used by platoon's proxy at startup) resolve to the same engines, i.e. to
  the latest trained weights once the first push happened. Recipes that need a
  frozen reference sampler (KL against the base, `test_opponent=base_model`)
  are not supported on an FFT gateway.
- Cost: Qwen3-4B is ~8 GB bf16; one push to 8 TP=1 engines takes a few seconds
  (`--update-weight-buffer-size 1073741824`, broadcast). The cookbook and
  platoon push every step; this is a few percent of a step.

## `weights_info` and the `is_lora` lie

Both SDKs assert `is_lora` when resuming from a state path
(`create_training_client_from_state[_with_optimizer]`). The gateway's
`POST /api/v1/weights_info` therefore reports

```json
{"base_model": "...", "is_lora": true, "lora_rank": 0,
 "train_unembed": true, "train_mlp": true, "train_attn": true, "train_mode": "full"}
```

for an FFT checkpoint: `is_lora` is always `true` on this endpoint, `lora_rank`
is the real rank (0 for FFT) and `train_mode` tells the truth. `get_info` /
`GET training_runs/{id}` report the honest `is_lora: false, lora_rank: null`.
`load_state` refuses a checkpoint whose sidecar `train_mode` does not match the
gateway's mode (`UserInputError`).

## Checkpoints and resume

- `save_state(name)` -> `<checkpoint_root>/<model_id>/weights/<name>/`, a
  Megatron `torch_dist` checkpoint (model + optimizer, fixed iteration 1) plus
  the `tinker_meta.json` sidecar. Size for Qwen3-4B ~56 GB (8 GB bf16 weights
  + 16 GB fp32 master + 32 GB Adam), 30-60 s at 1-2 GB/s: prefer
  `save_every >= 5` (cookbook) / `--checkpoint.every 5` (platoon) over the
  per-step default of platoon's synth config.
- `load_state(path, optimizer=True)` restores weights and Adam state;
  `optimizer=False` restores weights and resets Adam (fresh optimizer). The
  parallel topology (TP/PP/DP) must match the save.
- Resume works across gateway restarts and sessions of the same tenant because
  `weights_info` and `load_state` read the sidecar; restart the gateway with the
  same `--checkpoint-root` (`RUN_NAME` in `deploy/abci/tinker_gateway.pbs`).
- Deleting a checkpoint (`DELETE .../checkpoints/...`) removes the directory;
  a missing checkpoint is treated as already deleted.

## Restart and resume on the ABCI Toolathlon jobs

On the ABCI Toolathlon jobs the client runs under
`platoon.train.tinker.restart_wrapper --max-restarts 20 --restart-delay 700`
with `TINKER_TELEMETRY=1`: a clean exit sends SESSION_END and frees the single
full-FT slot immediately; a killed trainer frees it when the 600 s lease
expires (the launcher passes `--tinker-lease-timeout-s 600`; the sweep runs
every 30 s), so 700 s guarantees that the next attempt re-acquires the slot and
`rl.py` resumes from `<log_path>/<experiment>/<trial>/checkpoints.jsonl` with
the same `TINKER_API_KEY` tenant and `RUN_NAME` (`--checkpoint-root`). Every
(re)launch writes a fresh ~490 GB `kind=both` checkpoint under a new
`model_id`; `deploy/abci/clients/prune_tinker_checkpoints.py` (run every 10 min
by the client script) keeps the newest two records' directories across all
`model_id`s, never the last `state_path` record's directory and never a
directory modified in the last 30 min. Job layout and teardown:
`docs/fork/deploy-pbs-15node.md`.

## Memory and performance (Qwen3-4B, 8 H200 trainers TP=1/DP=8)

- Static ~30 GB per rank (bf16 params, replicated fp32 grad buffer, distributed
  optimizer shards); activations ~3.5-4 MB/token without recompute. The
  launcher default `--max-tokens-per-gpu 16384` fits without recompute; 24576
  needs `--recompute-args "--recompute-granularity full --recompute-method
  block --recompute-num-layers 8"`; 32768 needs full recompute (MFU-hostile).
- The fp32 gradient buffer stays live between `forward_backward` calls (it is
  allocated permanently by DDP anyway).
- Per `optim_step`: reduce-scatter 16 GB + all-gather 8 GB over NVLink
  (~0.1-0.2 s) plus Adam on 0.5 B params per rank.
- `--tinker-train-mode full` rejects `--multi-lora-n-adapters` and
  `--lora-rank`; the launcher's `full` mode emits none of the LoRA flags.

## Prefix cache and weight pushes

A full-weight push invalidates every KV entry SGLang cached under the previous
weights, and full mode sends no `lora_path`, so radix-cache keys do not change
between versions. The serve validation therefore forces
`--pause-generation-mode retract` (miles' default) in full mode: engines drop
their cache during the push and re-prefill in-flight requests. `in_place`
pausing is only sound for LoRA adapter pushes, whose cache keys include the
adapter name. (A possible follow-up is a per-version `cache_salt` on
`/generate`, which would make `in_place` safe for full mode too.)

## Gateway limits and the 35B recipe

- The per-datum and per-request token caps are flags:
  `--tinker-max-tokens-per-datum` (default 32768; the 35B launcher passes 65537
  = 65536 `model_input` tokens + 1 target) and `--tinker-max-tokens-per-request`
  (4,000,000).
- `--tinker-coalesce-max-wait-s` (default 2.0 s) bounds how long an under-filled
  unit waits for more SDK chunks of the same window, counted from the window's
  first chunk; `--tinker-coalesce-wait-s` is the quiet time after the last
  chunk. The 35B recipe uses 5 s / 120 s with `--tinker-unit-token-budget
  2097184` so that 32 one-datum SDK chunks of 64k form one DP32 unit.
- `--tinker-lease-timeout-s` defaults to 3600 s; the 35B launcher passes 600 s
  so a restarted platoon client (restart delay 700 s) finds the single full-FT
  slot released.
- The Qwen3.6-35B-A3B recipe (`examples/tinker/run_qwen3_6_35b_a3b_gateway.py`)
  overrides the model definition's `--mtp-num-layers 1` with `0` and never
  passes `--enable-mtp-training`: engines run without EAGLE and load the MTP
  weights unused.
- The R3 knobs (`--tinker-routing-replay-policy`,
  `--tinker-routing-cache-max-bytes`, `--tinker-routing-cache-ttl-s`,
  `--tinker-routing-prefix-stride`, `--tinker-r3-check`), the fused head
  (`--fused-lm-head-logprobs`, `--fused-lm-head-chunk-size`) and the serve-time
  validation behind them (R3 needs an MoE model and no speculative decoding,
  warns on MTP layers; the fused head needs `--loss-type policy_loss`, no
  `--calculate-per-token-loss`, no `--true-on-policy-mode`, no
  `--use-rollout-entropy`, no MTP training/layers and untied embeddings) are
  described in
  `docs/fork/flags-and-launcher.md`.

## Rollout routing replay (R3)

With `--use-rollout-routing-replay` the gateway captures the expert routing
SGLang used for every sampled sequence and replays it in the trainer, so the
MoE routers train on the experts that produced the tokens.

- Protocol: SDK 0.16.1 JSON sample responses carry `routing_id` per sequence;
  a datum sends it back as `loss_fn_inputs["routing_handle"]` (int64 `[1]`,
  the id of the last sample call merged into the datum, whose prompt +
  generated tokens must equal the datum's tokens). Proto clients (cookbook,
  SDK 0.27.1) send no handle and train through the live router.
- Materialization: the gateway attaches `rollout_routed_experts` (int32
  `[T-1, L, K]`) to every row of a unit. Datums without cached routing get rows
  of -1 and train through the live router under
  `--tinker-routing-replay-policy natural` (default; counted in
  `optim/routing_uncovered_datums`) or fail typed under `strict`.
- Rows for radix-cached prefixes come from the KV slots' last prefill: after a
  retract push the prefix is re-prefilled under the new weights, so prompt rows
  may mix weight versions (inherent, same as AReaL). LCP parents from other
  trajectories are equivalent to SGLang's own radix reuse; a datum trained
  twice (platoon never does) misses the released leaf and falls back to natural
  routing.
- The cache is bounded by `--tinker-routing-cache-max-bytes` (128 GiB) and
  `--tinker-routing-cache-ttl-s` (6 h); a gateway restart empties it (one
  staleness window of natural routing, visible in the metrics). No EAGLE/MTP
  speculative decoding with R3; keep `--sglang-chunked-prefill-size 16384 >=
  --sglang-max-running-requests 128`.
- Trainer: one context manager
  (`miles/backends/training_utils/replay_data.py:rollout_replay_stage`) wraps
  each unit's pass. It loads `rollout_data["rollout_routed_experts"]` (one
  int32 `[len(tokens)-1, L, K]` tensor per row, DP pad rows included) into the
  MoE routers' replay queues with `fill_replay_data` (which consumes the key),
  holds the routing manager at `replay_backward` for `forward_backward` (the
  forward step forces `replay_forward` itself and restores `replay_backward`
  for the recompute) or `replay_forward` for `forward_only`, and on exit,
  success or failure, clears the queues and falls through to the live router.
  A manager is active only when it was enabled at actor init
  (`use_routing_replay`, derived from `--use-rollout-routing-replay`) and the
  flag is set; nothing changes without the flag. Rows that carry no routing at
  all (padding, or a datum the gateway could not cover under `natural`) take
  the live router's own top-k, not the old `arange % num_experts` fill; rows
  with some valid picks keep their -1 padding. The sequence-parallel slice in
  `fill_replay_data` is a no-op at TP1 (Megatron disables SP there) and is
  exercised only by the TP2 fallback (`tests/fast/utils/test_replay_data_sp.py`).

Full details: `docs/fork/r3.md` (gateway) and `docs/fork/r3-trainer.md`
(trainer).

## Log-probs at 64k context: the fused fp32 head

With `--fused-lm-head-logprobs` the trainer never materializes the `[T, V]`
logits. The Megatron output layer is wrapped so that, during `forward_backward`
and `forward` (`forward_only_logprobs`), it returns the TP/SP-mapped hidden
states; `calculate_log_probs_and_entropy` then projects the response rows in
chunks of `--fused-lm-head-chunk-size` tokens (default 1024) inside a custom
autograd function that recomputes the projection in backward. Each chunk's
logits are `torch.mm(h_bf16, W_bf16.T, out_dtype=float32)`, the same expression
SGLang uses under `--sglang-enable-fp32-lm-head`, so the trainer scores exactly
the distribution the sampler sampled from; the fp32 weight gradient is added
straight into the DDP `main_grad` buffer (accumulation across
`forward_backward` calls is unchanged). Peak transient memory at chunk 1024 is
~1 GB in forward and ~4.5 GB in backward instead of 2 x 32.5 GB of bf16 logits
+ gradient (plus a 65 GB fp32 copy when `--log-probs-chunk-size` is -1).
Requirements: `policy_loss`-style tinker losses without entropy, no
`--true-on-policy-mode`, no MTP (`--mtp-num-layers 0`), one lm head per
pipeline rank, no lm-head bias. Fallback without the fused head:
`--log-probs-chunk-size 2048` and TP2. Details and the validation procedure:
`docs/fork/fused-lm-head.md`.

## MFU

`actor_train_mfu` / `actor_train_tflops` use a layer_types-aware FLOPs model.
When `--hf-checkpoint` points at a Qwen3.5/3.6-style config, the trainer reads
`layer_types` (or `full_attention_interval`) and the `linear_*` Gated DeltaNet
dims once per process and charges quadratic attention only on the
full-attention layers; the attention output gate (`--attention-output-gate`)
doubles the q projection and `--enable-mtp-training --mtp-num-layers N` adds N
MTP depths. Dense and plain-MoE models keep the previous numbers exactly. For
Qwen3.6-35B-A3B at 64k context the model credits 11.20 GFLOPs/token (the old
formula gave 26.60, a 2.4x over-count), i.e. 3 x 11.20 G x 65536 = 2.20 PFLOP
per 64k row per rank, so an MFU of 0.15 at 989 TFLOPS needs <= 14.8 s per row.
Sanity band for the 64k smoke and the 15-node runs: 0.05-0.35; above 0.35 means
the layer pattern was not applied (look for the `MFU: could not read the layer
pattern` warning or the missing `MFU: 30 linear-attention + 10 full-attention
layers` info line); `actor_train_tok_per_s` and seconds per 64k row remain the
primary throughput numbers. Formulas and numbers: `docs/fork/mfu.md`.
