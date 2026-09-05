# Acceptance runs

Two unmodified clients must train against the gateway, in both `lora` and
`full` mode, on Qwen3-4B-Instruct-2507 with two 8xH200 ABCI nodes (8 train
GPUs + 8 SGLang engines), and the training MFU reported by the gateway must
exceed 12%. Everything below runs from the login node in the fork checkout;
`P=/groups/gai51740/aci18912it/miles-tinker`.

## 0. Bring up a gateway and smoke it

```bash
qsub -v MODE=lora,RUN_NAME=gw-lora deploy/abci/tinker_gateway.pbs      # or MODE=full,RUN_NAME=gw-full
E=$P/name-resolve/gw-lora.endpoint; until [ -f $E ]; do sleep 30; done; cat $E
tail -f $P/jobs/gw-lora/launcher.log                                    # wait for "gateway healthy"
/home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/tinker_sdk_smoke.py --endpoint-file $E --lora-rank 16
```

The SDK smoke prints one `PASS`/`FAIL` line per check (create client, 3x
forward_backward + optim_step with `grad_norm`, unnamed sampler save + sample,
named `save_state` + `weights_info` + resume with optimizer + one more step,
stop strings, stop token ids returned in `tokens`) and exits non-zero on any
failure. Fix the gateway before going on. Optional single-turn warm-up:

```bash
deploy/clients/cookbook_math_arithmetic_smoke.sh $E $P/runs/math-smoke      # math_rl env=arithmetic, 20 steps
```

## A. tinker-cookbook `multiplayer_rl/guess_number`, async (SDK 0.27.1)

Multi-turn (up to 10 assistant turns with user feedback), prefix-extension
datums with masked user turns, group-relative advantages, unnamed sampler
save every step, named checkpoints on `save_every`, eval against the live
sampler, `AsyncConfig(max_steps_off_policy=1, groups_per_batch=32)`.

```bash
/home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/cookbook_guess_number_async.py \
    endpoint_file=$E log_path=$P/runs/gn-async-lora                          # LoRA gateway: lr 3e-5, rank 32
/home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/cookbook_guess_number_async.py \
    endpoint_file=$P/name-resolve/gw-full.endpoint log_path=$P/runs/gn-async-full learning_rate=1e-6   # full gateway
```

Defaults: model `Qwen/Qwen3-4B-Instruct-2507`, renderer `qwen3_instruct`,
group_size 8, max_tokens 64, eval_every 5, save_every 10, max_steps 25,
`behavior_if_log_dir_exists=delete`, wandb off. Every field is a `key=value`
override (`--help` lists them).

Pass criteria (from `cookbook_async_agent.md` section 10.3):

- 25 steps without a crash; `<log_path>/checkpoints.jsonl` has records at
  steps 10, 20 and `final` with `state_path` and `sampler_path` of the form
  `tinker://<model_id>/(weights|sampler_weights)/<name>`.
- `<log_path>/metrics.jsonl`: `env/all/reward/total` rising,
  `test/env/all/reward/total >= 0.8` by step 20, `env/all/turns_per_episode`
  falling, `optim/kl_sample_train_v1` small (|v1| < ~0.05 at
  max_steps_off_policy=1), `sampling_client/step_min >= training_client/step - 1`.
- Resume: rerun with `behavior_if_log_dir_exists=resume` and confirm training
  continues from the last `state_path` (the gateway restores optimizer state
  from the checkpoint sidecar; on a full gateway the restored run reports
  `is_lora: true, lora_rank: 0` from `weights_info`, see `full-finetuning.md`).
- MFU: see section C.

## B. platoon textcraft synth (SDK 0.16.1, CISPO)

Self-contained (no dataset download, no external service), sequence-extension
prompts with string stops (`</python>`), `save_state` +
`save_weights_for_sampler` every step, seed 42 on every request (the gateway
offsets seeds per sample so groups do not collapse), `max_staleness 3`.

```bash
deploy/clients/run_platoon_textcraft.sh $E $P/runs/textcraft-smoke                       # 5 steps, batch 4, group 4
STEPS=200 BATCH_SIZE=16 GROUP_SIZE=8 WORKERS=16 ROLLOUT_MAX_STEPS=200 CONTEXT_LEN=8192 EVAL_EVERY=5 CKPT_EVERY=5 \
    deploy/clients/run_platoon_textcraft.sh $E $P/runs/textcraft-full                    # acceptance settings
```

The script sets `TINKER_API_KEY=tml-local`, `TINKER_BASE_URL=<endpoint>` (the
litellm proxy's `ServiceClient()` takes no base_url), `TINKER_TELEMETRY=0`,
wandb `offline`, passes `--config` explicitly (the recipe's default path does
not exist) and `--tinker_base_url`. Extra `--a.b.c value` overrides are
appended verbatim. Eval cannot be disabled; `EVAL_EVERY=100` skips it on short
runs. Against a full gateway keep `CKPT_EVERY=5` (a full checkpoint is ~56 GB).

Pass criteria: the configured number of steps completes; platoon's
`<log_path>/<experiment>/<trial>/` holds per-step metrics with rising
`reward`; `checkpoints` records resolve through `weights_info` (restart the
run with the restart wrapper and confirm it resumes from the last
`state_path`); no `RendererError` (at most one `<|im_end|>` per sample) and
no group discarded for identical trajectories.

## C. Reading MFU

The gateway measures the `actor_train` window (forward + backward + optimizer
step, exactly as miles' native `perf/actor_train_mfu`) per optimizer window
and reports it in three places:

1. **`OptimStepResponse.metrics`** (on the wire, per `optim_step`):
   `grad_norm`, `skipped`, `lr`, `num_tokens`, `num_datums`, `num_units`,
   `actor_train_time`, `actor_train_tflops`, `actor_train_mfu`,
   `actor_train_tok_per_s` (perf keys only when the window had trainable
   work). The cookbook merges `optim_result.metrics` unprefixed into the step
   record of `<log_path>/metrics.jsonl` (`actor_train_mfu` alongside
   `time/train_step`, `rl/train.py` `metrics.update(optim_result.metrics)`);
   platoon logs each key as `optim/<key>` on its train tracker
   (`platoon/train/tinker/rl.py`).
2. **`<checkpoint_root>/perf.jsonl`** on the gateway side
   (`MilesBackend._log_perf`, one JSON line per slot per `optim_step`, keys as
   above plus `slot`), e.g. `$P/checkpoints/gw-lora/perf.jsonl`. This is the
   file to quote: it is independent of what the client keeps.
3. `head.log` lines `tinker perf: {...}` (the same records) and, with
   `WANDB_MODE=offline`, the wandb run under `$P/jobs/<RUN_NAME>/wandb`.

```bash
python3 - <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "perf.jsonl")]
mfu = [r["actor_train_mfu"] for r in rows if "actor_train_mfu" in r]
print(f"windows={len(rows)} with_perf={len(mfu)} mean_mfu={statistics.fmean(mfu):.3f} "
      f"median={statistics.median(mfu):.3f} tokens/window={statistics.fmean(r['num_tokens'] for r in rows):.0f}")
PY
```

Report the mean and median `actor_train_mfu` over the steady-state windows
(skip the first two: warm-up, JIT) together with `num_tokens` and
`num_units` per window. Cross-check with the 6N formula from
`miles_qwen3_4b_recipe.md` section 9: `6 * 4.02e9 * actor_train_tok_per_s /
(8 * 989e12)` must be within ~30% of `actor_train_mfu` (miles' number includes
the attention term and credits 3x forward even under recompute).

What moves the number (`v2_executor_full_ft_design.md` section 3.3): unit size
(`--tinker-unit-token-budget`, `--tinker-coalesce-wait-s` so an SDK request
split into chunks trains as one unit; `num_units` per window should be 1-2),
`--max-tokens-per-gpu` (16384 default; raise until OOM), recompute (costs
~33% FLOPs the metric does not credit) and the `dp_size` padding overhead
(visible as `num_tokens` per window vs. the client's token count). Run A's
step is only ~77k tokens, so 12% needs coalesced single-unit windows; run B's
0.4-1 M tokens per step should reach 25-40%.

## D. Recording the result

For each of the four combinations (A/B x lora/full) keep: the job id and
`RUN_NAME`, the fork commit (`launcher.log` prints it), the client `log_path`,
`perf.jsonl`, the MFU summary above, and the pass-criteria checklist. Put
them under `$P/runs/<name>/ACCEPTANCE.md`; do not commit run artifacts to the
repo.
