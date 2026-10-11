# miles-tinker fork

A fork of [radixark/miles](https://github.com/radixark/miles) that serves the
[Tinker](https://tinker-docs.thinkingmachines.ai) training/sampling API from
`serve_tinker.py`, so that unmodified Tinker SDK clients (tinker-cookbook, and
platoon once it moves past SDK 0.16.1) train against a miles cluster. The
gateway speaks upstream's SDK pin (`tinker==0.26.2`). Targets: LoRA
(multi-slot, several clients on one base) and full fine-tuning (single tenant),
Qwen3-4B-Instruct-2507 on Babel and on the ABCI / GH200 clusters.

Branch: `miles-tinker`. Base: `upstream/main` (the multi-LoRA v2 stack
#2839-#2849 has merged upstream); see `tools/fork/upstream_state.json` for the
recorded sync state.

## Repository layout (github.com/neulab/miles-tinker)

| Branch | Content | Rule |
|---|---|---|
| `miles-tinker` (default) | upstream base + every fork commit | all fork work lands here; PRs target it |
| `main` | untouched mirror of `radixark/miles` `main` | never commit to it; refreshed by `upstream_sync.py --push-mirror` (or `git push origin upstream/main:main`) so GitHub's compare view (`main...miles-tinker`) shows the fork delta |
| `upstream/multi-lora-v2` | the upstream PR-stack head the fork used to be based on | bookkeeping only; can be dropped |

Remotes on a working clone: `origin` = neulab/miles-tinker, `upstream` = https://github.com/radixark/miles.git.

## Base branch strategy

1. The fork tracks `upstream/main`. `merged_into_main: true` is set in the
   state file (the stack landed without its head becoming an ancestor of
   `main`), so `tools/fork/upstream_sync.py` merges `upstream/main`.
2. The first sync onto `main` (2026-10-10, `05bd9cb2`) merged against the
   stack head as base, because upstream restructured the gateway after the
   stack landed: `scheduler`/`future`/`model_queue` replaced
   `planner`/`promise`/`stream`, multi-LoRA moved to a per-mode actor
   (`lora/actor.py` + `lora/model.py`), and sampler adapters are exported to
   disk and loaded by the engines on demand. The fork deltas below were
   re-implemented on that structure.
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

### Gateway (`miles/tinker`, `serve_tinker.py`)

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/tinker/arguments.py` | `--tinker-train-mode {lora,full}`, `--tinker-lease-timeout-s` (600 s; the SDK heartbeats every 10 s, G21), `--tinker-max-concurrent-samples`, `--tinker-coalesce-wait-s` / `--tinker-coalesce-max-wait-s`; full-mode serve validation in `configure_tinker_args` (no overlap/offload/colocate/precision-aware optimizer, dynamic batch size, `retract` pausing, DDP NaN check off) | yes |
| `miles/tinker/core/types.py` | `GatewayConfig.train_mode`, `coalesce_wait_s` (0 = off in-process; the serve flag turns it on), `coalesce_max_wait_s` | yes |
| `miles/tinker/core/service.py` | `end_session` (SESSION_END releases the session at once, shared with lease expiry); coalesce hold before a growing batch runs; wire-safe optim metrics (non-finite -> 0.0); full mode: one slot, LoRA configs ignored, sampling with no adapter name (latest weights, G18) | yes |
| `miles/tinker/core/scheduler.py`, `miles/tinker/core/model_queue.py` | `schedule_next(commit=False)` previews a batch without issuing it; `batch_run_may_grow`, `last_submit_at`, `enqueued_at` for the coalesce hold | yes |
| `miles/tinker/core/future.py` | delivered results expire after 60 s and the full expiry sweep runs once a minute instead of on every lookup (memory and GC pauses at RL request rates, G3) | yes |
| `miles/tinker/core/utils.py`, `miles/tinker/core/input_validation.py` | checkpoint metadata records `train_mode`; full checkpoints carry no adapter; a checkpoint loads only into a model of the same mode | yes |
| `miles/tinker/server/app.py` | `/telemetry` ends the session named by a SESSION_END event; delivered futures marked for the short TTL | yes |
| `miles/tinker/server/encoding.py`, `miles/tinker/server/proto_codec.py` | request-level `num_tokens:sum` / `num_datums:sum` with every forward_backward | yes |
| `miles/tinker/runtime.py` | `/generate` posts capped by one semaphore; `max_new_tokens` clamped to the engine context (S9); per-slot exact window counts and `<checkpoint_root>/perf.jsonl` MFU records; full mode: batches without `adapter_slots`, sampler saves push the base weights inside the inference controller's update bracket and write only the sampler path's `META.json` | yes |
| `serve_tinker.py` | the flags above into `GatewayConfig` / `MilesBackend`; context length for the clamp; full mode drops the multi-LoRA assert and serves one slot | yes |
| `tests/fast/tinker/` | tests for every delta above (coalesce, full mode, future expiry, runtime, serve args, SDK flow) | yes |

### Training side (`miles/backends`, `miles/ray`)

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `miles/backends/megatron_utils/full/` | full fine-tuning (E7): `actor.py` (`FullTinkerTrainRayActor`, the multi-LoRA actor's command surface) and `model.py` (deferred `finalize_model_grads` to optim_step, AdamParams per step, non-finite veto, Megatron dist-ckpt save/load with `META.json`, reset to base, forward-only scoring) | yes |
| `miles/backends/megatron_utils/adam_params.py` | per-step AdamParams, clip and veto helpers for the full model | yes |
| `miles/backends/megatron_utils/tinker_perf.py` | per optim window MFU bookkeeping (`actor_train_*`) from `Timer().seq_lens` | yes |
| `miles/backends/megatron_utils/lora/model.py`, `miles/backends/megatron_utils/lora/actor.py` | training passes and optimizer steps timed under `actor_train`; optim_step attaches the window's perf metrics (E5) | yes |
| `miles/backends/training_utils/loss/objective.py` | loss normalizer 1 for any batch with a Tinker `loss_fn` (full fine-tuning too), not only multi-LoRA | yes |
| `miles/ray/specs/train.py` | `_trainer_worker_class`: `--tinker-train-mode full` selects `FullTinkerTrainRayActor` | yes |
| `tests/fast/tinker/test_full_model.py`, `tests/fast/backends/megatron_utils/`, `tests/fast/ray/specs/test_train.py` | full model state machine, AdamParams, perf, actor selection | yes |

### Not carried over from the pre-sync fork

Deliberately left out at the 2026-10-10 sync: SDK 0.16.1 shims (`POST /forward`
G9, default chunk `type` G2, 0.16.1 topk shape G14); `training_runs` routes (G6);
aiohttp `/generate` (upstream's httpx client is kept until sampling load shows a
gateway bottleneck); discard-and-continue after a failed forward_backward
(upstream closes the model); live LoRA adapter push and sampler-version GC
(G10; upstream exports adapters to disk); reading `tinker_meta.json`
checkpoints of the old fork. Upstream already covers G1, G4, G5 (`META.json`),
G8, G11, G12, G13, G16, G17 and G19.

### Fork-only tooling, deployment and docs

| File | Rationale | Upstream-worthy? |
|---|---|---|
| `README.md` | fork banner prepended above upstream's README (conflict policy: keep the banner, take upstream's body) | no |
| `FORK.md` | this file | no |
| `tools/fork/` | `upstream_sync.py` + `upstream_state.json`: base selection, merge/rebase, conflict report, fast tests, state record | no |
| `tests/fast/fork/` | tests for the sync tool and the ABCI launcher | no |
| `.claude/skills/upstream-sync/` | Claude Code skill wrapping the sync tool and the conflict policy | no |
| `deploy/abci/` | ABCI PBS launchers: `tinker_gateway.pbs` (two nodes, one container per node, `CLIENT_SCRIPT` runs the client on the head node and the job ends when it exits), `clients/*.sh` ready-made client scripts, `pull_image.pbs`, `smoke_container.pbs`, README runbook; still pinned to the pre-sync image | no |
| `deploy/babel/` | Babel Tinker gateway (Slurm + Apptainer, one or more nodes) and a separable Platoon client | no |
| `deploy/clients/` | client launchers (refuse login nodes via `hostguard`; run them through `CLIENT_SCRIPT`): cookbook guess_number async, math_rl arithmetic smoke, platoon textcraft synth, raw SDK smoke, `tinker_mfu_bench.py` (training-MFU benchmark with RL-shaped datums) | no |
| `examples/tinker/run_qwen3_4b_gateway.py` | ABCI's launcher for the gateway (lora/full) on 8 + 8 GPUs; still calls the pre-sync `command_utils.execute_train` and needs porting to `create_backend().execute_train` (see `deploy/babel/container/gateway.py`) | maybe (as `examples/tinker/`) once FFT lands upstream |
| `docs/fork/` | fork overview, full fine-tuning semantics, acceptance procedure | no |

## Running the fast tests

No GPU is needed. The tinker tests must run with `--confcutdir` so the repo
root `tests/conftest.py` (which imports GPU-only fixtures) is skipped:

```bash
module load python/3.12/3.12.9   # ABCI login node
PY=/path/to/venv/bin/python      # pytest, pytest-asyncio, tinker==0.26.2, fastapi, httpx, uvicorn, numpy, zstandard, aiohttp, torch (cpu), ray
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
