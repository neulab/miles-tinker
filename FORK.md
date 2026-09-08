# miles-tinker fork

A fork of [radixark/miles](https://github.com/radixark/miles) that serves the
[Tinker](https://tinker-docs.thinkingmachines.ai) training/sampling API from
`serve_tinker.py`, so that an unmodified **tinker-cookbook** (SDK 0.27.1) and
**platoon** (SDK 0.16.1) train against a miles cluster. Targets: LoRA
(multi-slot, several clients on one base) and full fine-tuning (single tenant),
Qwen3-4B-Instruct-2507 on two 8xH200 ABCI nodes, training MFU above 12%.

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
| `miles/tinker/core/types.py` | `GatewayConfig` gains train_mode, lease_timeout_s (600 s; SDK heartbeats every 10 s, G21), retrieve_poll_timeout_s, sampler_versions_to_keep, coalesce_wait_s, max_concurrent_samples; `ModelRecord.is_lora/sdk_version`, `lora_rank` int (0 = FFT) | yes |
| `miles/tinker/core/service.py` | `end_session` (SESSION_END -> immediate lease expiry); wire-safe optim metrics (non-finite -> 0.0); abstract `ExecutorBackend` (optim_step metrics dict, discard_grads, unload_adapter, warmup, push_slot version); unnamed sampler saves return `sampling_session_id` (G1); per-session leases and X-API-Key tenant (G4); `/weights_info` + sidecar-based resume (G5); idempotent create_model (G19); sampler-version GC via unload_adapter (G10); UserInputError instead of 500s (G13); FFT latest-only sampler resolution (G18) | yes |
| `miles/tinker/core/planner.py` | coalesce forward_backward chunks of one window (coalesce_wait_s / max_wait) so an SDK request split into chunks trains as one unit | yes |
| `miles/tinker/core/stream.py` | loss class keyed by the set of optional inputs present (G11) | yes |
| `miles/tinker/core/promise.py` | per-promise event for the long-poll of `/retrieve_future` (G3); delivered promises expire after 60 s (memory + GC pauses at RL request rates) | yes |
| `miles/tinker/server/app.py` | X-API-Key tenant (G4); long-poll retrieve_future capped below the SDK's 45 s (G3); `/weights_info` (G5); `GET/DELETE training_runs` (G6); richer `get_info` (G7); `GET /samplers/{id}` (G8); `POST /forward` for SDK 0.16.1 (G9); category `server` (G12); `/telemetry` releases a session's slots on the SDK's SESSION_END event; tenant headers read directly and results answered as `JSONResponse` (per-request overhead) | yes |
| `miles/tinker/server/encoding.py` | default chunk `type` for SDK 0.16.1 (G2); NaN-safe JSON and 0.16.1 topk shape (G14); unknown AdamParams keys warned, not rejected (G20); request-level `num_tokens:sum`/`num_datums:sum` metrics | yes |
| `miles/tinker/server/proto_codec.py` | protobuf face kept in step with the encoding changes above | yes |
| `miles/tinker/runtime.py` (sampling functions) | `no_stop_trim` so the stop token is returned (G15); per-sample seed offset (G16); `stop=[]` -> `ignore_eos` (G17); context-length clamp (S9); `/generate` posted through aiohttp (httpcore's pool rescans every connection per request: 81 -> 1326 req/s at 256 in flight in the fake-router harness) | yes |
| `tests/fast/tinker/` | fake-backend harness and tests for every gap above, both SDK body shapes | yes |
| `docs/examples/multi-lora.md` | protocol notes for the gateway | yes |

### Training side (`miles/backends`, `miles/ray/train`, `serve_tinker.py`) - FFT executor, MFU, padding, GC

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/backends/megatron_utils/full/` | full fine-tuning executor for slot 0 (E7): reuses `run_forward_backward_pass`, defers `finalize_model_grads` to optim_step, Megatron dist-ckpt save/load, `update_weights()` as the sampler push | yes |
| `miles/backends/megatron_utils/actor.py` | executor selection by `--tinker-train-mode`; FFT commands no longer assert `multi_lora` | yes |
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
| `serve_tinker.py` | uvloop when installed, no uvicorn access log, 60 s keep-alive; `--tinker-train-mode {lora,full}` and the GatewayConfig flags (`--tinker-lease-timeout-s`, `--tinker-sampler-versions-to-keep`, `--tinker-coalesce-wait-s`, `--tinker-max-concurrent-samples`, `--tinker-retrieve-poll-timeout-s`, `--tinker-unit-token-budget`); no `multi_lora` assert in full mode | yes |
| `miles/utils/arguments.py`, `miles/utils/tinker_args.py` | the `--tinker-*` flags above and the serve-mode validation (full mode forces `retract` pausing, warns on TP*EP>1 for LoRA) | yes |
| `miles/utils/train_metric_utils.py`, `miles/utils/timer.py`, `miles/utils/flops_utils.py` | only if needed for per-unit MFU (`actor_train_mfu`, `actor_train_tok_per_s` from `Timer().seq_lens`) | yes |
| `tests/fast/tinker/test_runtime.py`, `tests/fast/tinker/test_full_executor.py`, `tests/fast/backends/` | padding, FFT executor and metrics tests | yes |

### Fork-only tooling, deployment and docs

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `README.md` | fork banner prepended above upstream's README (conflict policy: keep the banner, take upstream's body) | no |
| `FORK.md` | this file | no |
| `tools/fork/` | `upstream_sync.py` + `upstream_state.json`: base selection, merge/rebase, conflict report, fast tests, state record | no |
| `tests/fast/fork/` | tests for the sync tool | no |
| `.claude/skills/upstream-sync/` | Claude Code skill wrapping the sync tool and the conflict policy | no |
| `deploy/abci/` | ABCI PBS launchers: `tinker_gateway.pbs` (two nodes, one container per node, `CLIENT_SCRIPT` runs the client on the head node and the job ends when it exits), `clients/*.sh` ready-made client scripts, `pull_image.pbs`, `smoke_container.pbs`, README runbook | no |
| `deploy/babel/` | Babel setup and Apptainer runner for existing repository training commands | no |
| `deploy/clients/` | client launchers (refuse login nodes via `hostguard`; run them through `CLIENT_SCRIPT`): cookbook guess_number async, math_rl arithmetic smoke, platoon textcraft synth, raw SDK smoke, `tinker_mfu_bench.py` (training-MFU benchmark with RL-shaped datums) | no |
| `examples/tinker/run_qwen3_4b_gateway.py` | `execute_train`-shaped launcher for the gateway (lora/full) on 8 + 8 GPUs | maybe (as `examples/tinker/`) once FFT lands upstream |
| `docs/fork/` | fork overview, full fine-tuning semantics, acceptance procedure | no |

## Running the fast tests

No GPU is needed. The tinker tests must run with `--confcutdir` so the repo
root `tests/conftest.py` (which imports GPU-only fixtures) is skipped:

```bash
module load python/3.12/3.12.9   # ABCI login node
PY=/path/to/venv/bin/python      # pytest, pytest-asyncio, tinker>=0.27, fastapi, httpx, uvicorn, numpy, zstandard, torch (cpu), ray
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/fast/tinker -p no:cacheprovider --confcutdir=tests/fast/tinker -o addopts=""
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/fast/fork   -p no:cacheprovider --confcutdir=tests/fast/fork   -o addopts=""
```

Inside the serving container the same command works with the image's python
(`deploy/abci/smoke_container.pbs` runs it).

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
