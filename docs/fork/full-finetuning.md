# Full fine-tuning through the Tinker gateway

`serve_tinker.py --tinker-train-mode full` (launcher: `run_qwen3_4b_gateway.py
serve --mode full`) trains the **base weights** of the served model instead of
LoRA adapters. The Tinker SDKs only expose `create_lora_training_client`, so
this page documents where the full-fine-tuning (FFT) gateway deviates from
Tinker's LoRA semantics and why unmodified clients still work.

Code: `miles/backends/megatron_utils/full/` (`actor.py`, `model.py`),
`miles/tinker/core/service.py` (`create_model`, `submit_sample`, `weights_info`)
and `miles/tinker/runtime.py` (`export_slot`). Design notes:
`v2_executor_full_ft_design.md` section 4, `v2_sampling_versioning.md` section 5.

## Single tenant, `lora_config` ignored

- The deployment has exactly one trainable model: `GatewayConfig.n_slots == 1`
  (slot 0 = the base weights); the service refuses any other `n_slots`.
- `create_model` accepts **any** `lora_config` (the SDK always sends
  `{"rank": N, ...}`) and ignores it; the record is created with
  `lora_rank=0`. A client-side `lora_rank` / `learning_rate` meant for LoRA is
  therefore a no-op for the rank and a real change for the learning rate: use
  an FFT learning rate (1e-6 in `deploy/clients/cookbook_guess_number_async.py`
  `learning_rate=1e-6`; the LoRA default is 3e-5).
- A second concurrent training client gets `UserInputError("no free adapter
  slots (capacity 1)")` until the first model's
  lease expires (`--tinker-lease-timeout-s`, default 600 s after the last
  heartbeat) or its session ends. Run one client at a time.
- Every fresh `create_model` starts from the pristine base: `load_slot(slot=0,
  ckpt_path=None)` reloads the weights the server started from (the HF
  snapshot through the bridge) with fresh optimizer state (`reset_to_base`; a
  no-op while nothing has been trained or loaded yet). Resetting refuses
  `--chunked-optimizer-state-offload`, whose moments live outside the state it
  clears, so with offload only the first model of a server starts cleanly.

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
(`TrainerController.update_weights()` inside the inference controller's update
bracket). Therefore:

- `save_weights_for_sampler` (named or unnamed) pushes the current weights and
  returns `tinker://<model_id>/sampler_weights/<name>` as usual. Only the
  path's `META.json` (ownership and mode) is written to disk.
- Sampling resolves the path (it must exist and belong to the tenant) and then
  samples with no adapter name, i.e. **every sampling session of the model
  samples the newest pushed weights**, including sessions created from older
  sampler paths.
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
 "train_unembed": true, "train_mlp": true, "train_attn": true}
```

for an FFT checkpoint: `is_lora` is always `true` on this endpoint and
`lora_rank` is the real rank (0 for FFT). The checkpoint's `META.json` records
`train_mode`, and `load_state` refuses a checkpoint whose `train_mode` does not
match the gateway's mode (`UserInputError`).

## Checkpoints and resume

- `save_state(name)` -> `<checkpoint_root>/<model_id>/weights/<name>/`, a
  Megatron `torch_dist` checkpoint (model + optimizer, fixed iteration 1) plus
  the gateway's `META.json`. Size for Qwen3-4B ~56 GB (8 GB bf16 weights
  + 16 GB fp32 master + 32 GB Adam), 30-60 s at 1-2 GB/s: prefer
  `save_every >= 5` (cookbook) / `--checkpoint.every 5` (platoon) over the
  per-step default of platoon's synth config.
- `load_state(path, optimizer=True)` restores weights and Adam state;
  `optimizer=False` restores weights and resets Adam (fresh optimizer). The
  parallel topology (TP/PP/DP) must match the save.
- Resume works across gateway restarts and sessions of the same tenant because
  `weights_info` and `load_state` read `META.json`; restart the gateway with the
  same `--checkpoint-root` (`RUN_NAME` in `deploy/abci/tinker_gateway.pbs`).

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
