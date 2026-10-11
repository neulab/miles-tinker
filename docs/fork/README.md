# miles-tinker fork: overview

This fork of [radixark/miles](https://github.com/radixark/miles) serves the
Tinker training/sampling API (`serve_tinker.py`) so that **unmodified** Tinker
SDK clients can train against a miles cluster. The gateway speaks upstream's
SDK pin (`tinker==0.26.2`); clients on SDK 0.16.1 (platoon today) are no longer
served. The authoritative list of what changed and why is `FORK.md` at the repo
root; this directory holds the operator-facing docs.

| Page | What it covers |
|---|---|
| `FORK.md` (repo root) | fork deltas by file, base-branch strategy, fast tests, upstream sync summary |
| `docs/fork/full-finetuning.md` | `--tinker-train-mode full`: latest-only sampler, `weights_info` compatibility, resume, memory |
| `docs/fork/acceptance.md` | the two pre-sync acceptance runs (cookbook guess_number async, platoon textcraft synth) and how MFU is read |
| `deploy/abci/README.md` | ABCI runbook: image pull, rootfs, gateway launch, endpoint file, clients, logs, stop |
| `deploy/babel/README.md` | Serving the Tinker gateway on Babel and running clients against it |
| `docs/examples/multi-lora.md` | protocol notes for the multi-LoRA gateway (upstream doc) |
| `.claude/skills/upstream-sync/SKILL.md` | the upstream-sync procedure and conflict policy |

## What the gateway is

```
client (tinker SDK) --HTTP--> serve_tinker.py (FastAPI, miles/tinker/server)
                                  |  TinkerService (miles/tinker/core): sessions, models, futures,
                                  |  scheduler (batches of datums), checkpoints + META.json metadata
                                  v
                              MilesBackend (miles/tinker/runtime.py)
                     /---------------------------------------------\
   TrainerController (Ray actors, Megatron)          sgl-router -> N SGLang engines
   LoRA: MultiLoRATrainRayActor, one slot per client LoRA: adapters exported to disk, loaded on demand
   full: FullTinkerTrainRayActor on the base weights full: base weights, latest push only
```

Two deployment modes are selected at launch (`deploy/babel/container/gateway.py --mode`):

- **lora** (`--multi-lora-n-adapters N`): N training clients share one frozen
  base; each `save_weights_for_sampler` exports an adapter directory, which a
  sample request names so the engines load (or reload) it on demand.
- **full** (`--tinker-train-mode full`): one client at a time trains the base
  weights; a sampler save pushes the base weights to the engines; every sampler
  path samples the latest weights. See `full-finetuning.md`.

## Interface summary

- `GatewayConfig` (`miles/tinker/core/types.py`): `train_mode`, `n_slots`,
  `lease_timeout_s`, `batch_token_budget` (262,144 tokens), `coalesce_wait_s` /
  `coalesce_max_wait_s` (wait for more chunks of a growing forward_backward
  batch). `serve_tinker.py` sets them from `--tinker-train-mode`,
  `--tinker-lease-timeout-s` (600 s), `--tinker-coalesce-wait-s` (0.3 s) and
  `--tinker-coalesce-max-wait-s` (2 s); `n_slots` comes from
  `--multi-lora-n-adapters` (1 in full mode). `--tinker-max-concurrent-samples`
  (512) caps in-flight `/generate` posts in `MilesBackend`.
- `MilesBackend` (`miles/tinker/runtime.py`): `load_slot`, `unload_slot`,
  `forward_backward`, `forward_only`, `optim_step` (per-slot metrics: `grad_norm`
  or `skipped`, plus `num_tokens`, `num_datums`, `num_units` and the window's
  `actor_train_time`, `actor_train_tflops`, `actor_train_mfu`,
  `actor_train_tok_per_s`, also appended to `<checkpoint_root>/perf.jsonl`),
  `save_slot`, `export_slot` (full mode: push the base weights), `sample`.
- Checkpoints: `<checkpoint_root>/<model_id>/<weights|sampler_weights>/<name>/`
  plus `META.json` (`tenant_digest, base_model, lora_rank, lora_alpha,
  train_attn, train_mlp, train_unembed, train_mode`). `/api/v1/weights_info` and
  `load_state` read it, so resume works after a gateway restart; a checkpoint
  loads only into a model of the same train mode.
- Metrics on the wire: `forward_backward` returns `loss:sum` and `logprobs`
  per datum plus `num_tokens:sum` / `num_datums:sum`; `optim_step` returns the
  backend metrics, non-finite values as 0.0.
- Tenant = the `X-API-Key` header (`TINKER_API_KEY`); a request without one is
  rejected. Sessions lease their models for `lease_timeout_s` after the last
  heartbeat, and a SESSION_END telemetry event releases them at once.
- Failure policy (upstream): a failed `forward_backward`, `optim_step` or
  `load_state` closes the model, including requests rejected for their content
  (each datum must carry exactly the `loss_fn_inputs` its loss reads).

## Where to start

1. Read `FORK.md`, then `deploy/abci/README.md` to bring up a gateway.
2. Run `deploy/clients/tinker_sdk_smoke.py` against it (PASS/FAIL per check).
3. Run the acceptance recipes in `docs/fork/acceptance.md`.
4. To pull upstream changes, use the `upstream-sync` skill or
   `python tools/fork/upstream_sync.py --check`.
