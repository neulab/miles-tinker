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
`model_id`, named `recovery-<start_batch>`; `checkpoint.recovery_every` writes
one `recovery-<batch>` per step on top of the permanent `<batch>` checkpoints
of `checkpoint.every`, so a crash loses at most one step.
`deploy/abci/clients/prune_tinker_checkpoints.py` (run every 10 min by the
client script) deletes `recovery-*` directories across all `model_id`s except
the newest two recovery records', the last `state_path` record's and any
directory modified in the last 30 min; permanent names (`<batch>`,
`<batch>_interrupted`, `final`) are kept forever. Job layout and teardown:
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

## Context parallelism (128k rows on 80 GB GPUs)

`--context-parallel-size N` (launcher: `run_qwen3_6_35b_a3b_gateway.py --cp N`)
splits every datum over N trainer ranks in Megatron's zigzag layout: the row is
padded to 2N equal chunks and rank r keeps chunks r and 2N-1-r, so every rank
does the same attention work. DP becomes GPUs / (TP * CP). Full fine-tuning
only: the LoRA gateway stays at CP=1.

What the gateway does under CP:

- **Losses** (ported from upstream #3320, radixark/miles 3d8b5284): each rank
  slices the datum's full-length weights, advantages and loss mask to its
  shard (`rollout_log_probs` arrive pre-sliced) and returns the sum over its
  shard; the DP*CP gradient reduction adds the shards up. A rank with no
  response token of a microbatch (short datums, pad-only microbatches) returns
  `logits[..., :0].sum()`, a zero that still runs the shared backward.
- **Per-datum outputs and `forward`**: one CP all-reduce per microbatch
  (`forward_backward`) or per unit (`forward`) rebuilds full-length logprobs and
  whole-datum losses, identical on every rank
  (`cp_utils.gather_cp_response_values`); `forward` returns them from CP/TP
  rank 0 of each DP rank only.
- **Bridge VL models** (Qwen3.5/3.6/3.8): their providers turn
  `calculate_per_token_loss` on at CP>1. Megatron's DDP, schedule and
  `finalize_model_grads` read that flag while miles' loss scaling reads args.
  Through the gateway, whose deferred finalize passes no token count, every
  gradient came out num_microbatches * DP*CP too large. Elsewhere Megatron's
  finalize divides by the summed loss normalizer, which cancels that for the
  main loss, but the MoE aux/z-loss and the MTP loss were scaled per token
  (about tokens/CP times their per-microbatch scale). The model build resets
  the flag for every entry point. After `get_model` a check stops the run
  unless the flag matches args and each DDP chunk reduces over a DP*CP group
  the size of loss.py's `intra_dp_cp` factor, and every `forward_backward`
  re-checks the flag.
- **MRoPE**: miles' packed-MRoPE patch (`miles_plugins/models/qwen3_vl.py`)
  all-gathers each row, rebuilds the per-datum positions and slices them like
  the tokens; the v25 image's bridge (megatron-bridge 0.5.0+582783a0) detects
  the pre-sharded input and uses it. A CP>1 model build fails if the patch is
  missing, and a forward on a CP-sharded row fails unless the patch built the
  positions and the bridge asked for them. Newer bridge builds (the 2026-10-10
  dev image) require explicit rank-local position ids for pre-sharded input and
  fail at the first forward: CP needs the v25 image for now.
- **GatedDeltaNet layers**: `--linear-cp-mode headwise` (the launcher's default
  at `--cp` > 1: an all-to-all gives each rank full sequences on 1/CP of the
  heads, so TP*CP must divide the 16 linear key heads) or `chunkwise` (fla CP
  kernels that pass the recurrent state between ranks; Megatron's default when
  serve_tinker.py gets no `--linear-cp-mode`). Both pass the parity and
  layer-gradient checks below. Headwise is the launcher's default because its
  numerics do not depend on how datums are packed and stay closest to CP=1;
  chunkwise covers layouts where TP*CP does not divide the heads. Check memory
  at the target row length for either. Attention layers follow
  `--cp-comm-type`; Megatron's default `p2p` does not override a provider's own
  choice.
- **R3**: routing rows are sliced exactly like the tokens (CPU-tested over CP
  1-8, TP/SP 1-4 and three pad multipliers); serve validation rejects a fused
  MoE router (it skips the replay hook) and warns at CP>1 until a run with
  `--tinker-r3-check` has checked the layout on GPU, which none has yet.

Knobs and limits:

| | |
|---|---|
| Launcher | `--cp N`, `--linear-cp-mode` (headwise at `--cp` > 1 unless chunkwise is given), `--max-tokens-per-gpu` (per rank; default: the shard of one `--context-len + 1` row, `2 * ceil(row / 2N)` rounded up to a multiple of TP * `--data-pad-size-multiplier`, 128 unless `--extra-args` sets it) |
| Headwise layouts | TP*CP must divide the linear key and value heads (16 and 32; 16 and 48 for the 27B): TP4/CP2, TP4/CP4, TP1/CP8 fit; TP4/CP8 needs chunkwise |
| Memory | `actor_peak_mem_{alloc,reserved}_gib_all_ranks` in the optim metrics and `perf.jsonl`: the window's peak over all trainer ranks (the plain keys are rank 0's) |
| Per-rank budget for 131073-token rows | TP4/CP2 66048, TP4/CP4 33280, TP4/CP8 16896, TP1/CP8 16512 |
| Serve validation | CP>1 only with `--tinker-train-mode full`, `--qkv-format thd`, no `--allgather-cp`; `--tinker-max-tokens-per-datum <= CP * floor(budget / (TP*128)) * TP*128` |
| Checkpoints | the distributed optimizer shards over DP*CP: resume with optimizer state at the same TP/EP/DP*CP layout (a CP1 save into a CP2 run is untested) |
| Images | v25 `miles-latest-cu13.sif` (see MRoPE above) |

ABCI example (two nodes, the Qwen3.5-4B parity recipe; `deploy/abci/README.md`):
`LAUNCHER_ARGS="--model-variant 4b --cp 2 --context-len 8192 --max-tokens-per-datum 8193"`.

Parity check: `deploy/clients/tinker_cp_parity.py run` (the CLIENT_SCRIPT
`deploy/abci/clients/cp_parity.sh`) against a CP=1 and a CP>1 gateway of the
same snapshot, plus a correct alternative CP=1 layout (TP2) as the noise
reference, then `tinker_cp_parity.py compare <cp1.json> <cpN.json> --noise
<tp2.json>`. It reports per-chunk and near-boundary logprob deltas, per-datum
loss deltas and grad_norm ratios. The per-token-loss bug above would show as a
grad_norm ratio of 2 or more. Use the default natural-text datums. On random
tokens the gradient norm moves 8% just from TP1 to TP2, which hides real
differences.

Give the CP=1 reference the candidate's row budget per microbatch,
`--max-tokens-per-gpu` = CP x the candidate's per-rank budget (8448 against
the CP2 recipe above, whose per-rank default is 4224; the CP=1 default is
8193), so that both runs pack the datum set alike. The packing moves the
numbers even at CP=1. Going from 8193 to 8448 puts each DP rank's share of
the datum set in one microbatch instead of two, so the short datums share a
microbatch with a long row, and the step-1 grad_norm moves by 0.31%. Forward
logprobs move by mean 9e-4 overall and by up to 0.29 on rows of 512 tokens
or fewer, while the rows of 1000 tokens and up and the datums sent alone
agree to 1e-6.

Boundaries depend on the layer type. Ring attention and the MRoPE positions
see each datum's zigzag chunk boundaries. Chunkwise GatedDeltaNet layers see a
contiguous split of the whole packed row into CP parts, where the recurrent
state and the conv halo pass between ranks, so its position depends on the
packing. The probe therefore also sends a 4000- and an 8000-token datum alone
(`forward`, and `forward_backward` in the first window with the loss on the
second half) and gates windows at their split rows,
`k * roundup(2 * ceil(L / 2CP), TP * 128)` (TP and the pad multiplier come
from the gateway's `head.log`). Rows of up to 512 tokens are also gated one at
a time on their largest delta before the first optimizer step, and are never
trimmed out of the KL, so an error confined to the short datums cannot hide
among the 8K-token rows. `tools/fork/gdn_cp_grad_probe.py` checks the
GatedDeltaNet backward directly: one layer on a packed row, each CP layout
against no CP, per-parameter and pre-split input gradients with the loss after
the split.

Validation (Qwen3.5-4B, v25 image, one H200 node training + one engine node,
`--context-len 8192`, natural-text datums of 2-8192 tokens plus the lone
4000- and 8000-token datums; reference CP1, noise reference TP2 at CP1; jobs
2633902-2634333 under `/groups/gai51740/aci18912it/platoon-next/jobs/cp-v2-4b-*`
(the GatedDeltaNet probe ran in 2634092, the attention probe in 2634717 under
`cp-v2-attnprobe-*`); the first round, 2632317-2633025 under
`cp-{par,txt}-4b-*`, had no lone datums):

| Check | Result |
|---|---|
| Bridge per-token-loss reset | fired on all 8 ranks at CP2, CP4 and CP8 in both GDN modes, never at CP1/TP2; the post-`get_model` check passed; the provider logged `linear_cp_mode=headwise` (or `chunkwise` when requested) |
| Weight push (`--check-weight-update-equal`) | all engines equal at CP1 and CP2 (first round) and at CP2 headwise |
| SDK smoke at CP2 headwise (fwd/bwd, optim, sampler push + sample, named `save_state`, resume with optimizer) | 0 failed checks (first round: also 0 at CP2 chunkwise) |
| Sampler vs trainer (on-policy probe) | KL 2.9e-4 CP1, 3.2e-4 TP2, 2.6e-4 CP2 headwise, 2.8e-4 CP4 headwise, 3.5e-4 CP8 headwise, 3.1e-4 CP2 chunkwise, 2.9e-4 CP4 chunkwise |
| `compare <cp1> <cand> --noise <tp2>` | PASS for CP2/CP4/CP8 headwise and CP2/CP4 chunkwise |
| Forward logprobs vs CP1: mean abs / trimmed KL | TP2 6.7e-3 / 2.7e-4; headwise CP2 3.5e-3 / 8.1e-5, CP4 4.8e-3 / 1.4e-4, CP8 5.6e-3 / 1.8e-4; chunkwise CP2 6.4e-3 / 2.2e-4, CP4 6.8e-3 / 2.3e-4 |
| Same forward, datums packed in another order | bit-stable at CP1 and with headwise (max 1e-6); chunkwise changes a row's numerics with where it straddles the contiguous split (mean 1.2e-3 to 1.7e-3, a few multi-nat flips on repeated text, as between TP1 and TP2) |
| GatedDeltaNet split windows (lone datums, chunkwise) | mean abs delta at the splits 3.6e-5 (CP2) and 4.3e-4 (CP4) against 8.6e-3 and 6.9e-3 elsewhere in those rows: no handoff error |
| GatedDeltaNet layer probe (`tools/fork/gdn_cp_grad_probe.py`, CP 2/4/8, loss after the CP split) | every parameter gradient and the pre-split input gradients match no CP at cosine >= 0.99996 in both modes; pre-split input-gradient norm ratio 0.9995-0.9996 chunkwise, 0.9999-1.0000 headwise |
| Attention layer probe (ring attention p2p, CP 2/4/8, logit std 1 and 8) | every gradient and the outputs match no CP at cosine >= 0.99994, norm ratios within 0.06% |
| Step-1 `cross_entropy` grad_norm vs CP1 (same weights) | TP2 1.000004; headwise CP2 1.0023, CP4 1.0016, CP8 1.0019; chunkwise CP2 1.0030, CP4 1.0020 (against a CP1 reference packed at 8193 tokens per microbatch; see the third round) |

Third round (head 14025130; jobs 2635003 = CP1/DP8 at `--max-tokens-per-gpu
8448` and 2635004 = CP2/DP4 headwise at 4224, run at the same time; both
under `cp-v3-4b-*`). The CP>1 runs of round two had packed 8448 to 9216
tokens per microbatch, while their CP1 reference used 8193.

| Check | Result |
|---|---|
| `compare <cp1> <cp2>` (no noise run) | PASS. Forward mean abs 3.3e-3, p99 0.059, trimmed KL 7.8e-5, short rows max 0.23; step-1 per-datum loss within 0.021 (sum 89.649 vs 89.644); logprobs after the three steps mean abs 7.8e-3, KL 4.7e-4 |
| Step-1 grad_norm at the same weights, CP2/CP1 | 0.9993 on the parity datums, 0.9996 on the on-policy probe's datums, 0.9995 on the SDK smoke's datums. Steps 2 and 3 of the parity probe: 0.932 and 0.980 (CP1 alone moved 8.4% at step 2 between the two packings) |
| Step-1 grad_norm offset of round two | Packing, not CP. Against the 8448 reference, round two's chunkwise CP2 dump is at 0.9999. The CP4 and CP8 runs (8704 and 9216 tokens per microbatch) are at 0.9984-0.9989: below this reference, where they had been above the 8193 one, and within the 0.31% that the packing moves CP1. TP2 at 8193 keeps the short datums apart from the long rows, as its CP1 reference does, hence its 1.000004. A rerun of the CP2 config reproduced the forward logprobs to 1e-6 and the step-1 grad_norm to 1.4e-4 |
| Boundary windows | the pooled window/interior ratio is 3.87 (gate 4). Per row, each window matches its own row's interior (for example 2.0e-2 vs 2.2e-2 at 127 tokens, 2.6e-5 vs 2.6e-3 at 8191), and nothing peaks at the boundary token. The pooled ratio puts the windows of the short rows, where deltas are larger, against the interior of the long rows |
| Sampler vs trainer, SDK smoke, weight push | KL 2.9e-4 (CP1) and 2.8e-4 (CP2); 0 failed SDK checks on both, including named `save_state` and resume with optimizer; `--check-weight-update-equal` at CP2: all 8 engines equal |
| Peak memory per rank (`perf.jsonl`, maximum over ranks) | 44.3 GiB allocated / 46.7 GiB reserved at CP1 with 8448 tokens; 39.9 / 41.9 GiB at CP2 with 4224 tokens per rank |

Thresholds of `compare` are calibrated so that correct layouts on natural
text pass with and without `--noise`: mean abs 0.03, p99 0.2, trimmed KL
1e-3, per short row 0.3, step-1 per-datum loss within 2% (+1e-3), step-1
grad_norm within 2%, later steps within 25% (the first Adam steps turn gradient
noise into sign flips), boundary windows up to 4x the interior. `--noise
<tp2.json>` widens them to twice the noise run's deviation (boundary windows:
1.5x its own window/interior ratio). With the noise run, a 1-token error at
each rank boundary fails from 0.05-0.15 nats, a 3-token one from 0.05, and a
single bad token in every tiny datum from 0.3-0.5. The lone datums' loss is
reported but not gated: it averages ~4000 tokens of repeated text, where a few
flips move it by percents under any layout (5% for TP2).

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
