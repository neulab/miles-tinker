# miles-tinker fork: overview

This fork of [radixark/miles](https://github.com/radixark/miles) serves the
Tinker training/sampling API (`serve_tinker.py`) so that an **unmodified**
tinker-cookbook (SDK 0.27.1) and platoon (SDK 0.16.1) can train against a miles
cluster. The authoritative list of what changed and why is `FORK.md` at the
repo root; this directory holds the operator-facing docs.

| Page | What it covers |
|---|---|
| `FORK.md` (repo root) | fork deltas by file, base-branch strategy, fast tests, upstream sync summary |
| `docs/fork/full-finetuning.md` | `--tinker-train-mode full`: latest-only sampler, `weights_info` compatibility, resume, memory |
| `docs/fork/acceptance.md` | the two acceptance runs (cookbook guess_number async, platoon textcraft synth) and how MFU is read |
| `docs/fork/flags-and-launcher.md` | the `--tinker-*` and fused-head flags, serve-time validation, the Qwen3.6-35B-A3B launcher recipe and the `--sglang-*` allowlist |
| `docs/fork/r3.md` | rollout routing replay through the gateway: `routing_id` / `routing_handle` protocol, routing cache, materialization, metrics |
| `docs/fork/r3-trainer.md` | R3 on the trainer: replay context manager around each pass, natural fallback for uncovered rows, SP note |
| `docs/fork/fused-lm-head.md` | the fused chunked fp32 lm head (`--fused-lm-head-logprobs`): numerics parity with `--sglang-enable-fp32-lm-head`, memory, requirements, validation |
| `docs/fork/mfu.md` | the FLOPs model behind `actor_train_mfu` (GDN linear attention, output gate, MTP), Qwen3.6 numbers, sanity band |
| `docs/fork/deploy-pbs-15node.md` | the N-node ABCI job: topology and roles, Toolathlon services, secrets, teardown, client restart/resume, dry runs |
| `deploy/abci/README.md` | ABCI runbook: image pull, rootfs, gateway launch, endpoint file, clients, logs, stop |
| `docs/examples/multi-lora.md` | protocol notes for the multi-LoRA gateway (upstream doc, kept in step by P1) |
| `.claude/skills/upstream-sync/SKILL.md` | the upstream-sync procedure and conflict policy |

## What the gateway is

```
client (tinker SDK) --HTTP--> serve_tinker.py (FastAPI, miles/tinker/server)
                                  |  TinkerService (miles/tinker/core): sessions, models, promises,
                                  |  planner (units of rows), checkpoints + tinker_meta.json sidecars
                                  v
                              MilesBackend (miles/tinker/runtime.py)
                     /---------------------------------------------\
   TrainerController (Ray actors, Megatron)          sgl-router -> N SGLang engines
   LoRA: slot executor, one adapter per client       LoRA: versioned adapters "<model_id>@<v>"
   full: single-tenant executor on the base weights  full: base weights, latest push only
```

Two deployment modes are selected at launch (`examples/tinker/run_qwen3_4b_gateway.py --mode`;
`run_qwen3_6_35b_a3b_gateway.py` is the full-mode 35B MoE recipe with R3 and the fused head):

- **lora** (`--multi-lora-n-adapters N`): N training clients share one frozen
  base; each `save_weights_for_sampler` registers a versioned adapter in every
  engine; the gateway keeps the last `--tinker-sampler-versions-to-keep`
  versions loaded (older sampler paths are retired with a `UserInputError`).
- **full** (`--tinker-train-mode full`): one client at a time trains the base
  weights; a sampler push is a full weight sync; only the latest version is
  servable. See `full-finetuning.md`.

## Interface summary (from the design contract)

- `GatewayConfig` (`miles/tinker/core/types.py`): `train_mode`, `n_slots`,
  `lease_timeout_s` (600 s), `retrieve_poll_timeout_s` (20 s long-poll, below
  the SDK's 45 s), `sampler_versions_to_keep` (3), `coalesce_wait_s` /
  `coalesce_max_wait_s` (0.3 / 2.0 s: wait for more chunks of an open
  forward_backward window before training a unit), `max_concurrent_samples`
  (512), `unit_token_budget` (262,144 tokens), `max_tokens_per_datum` (32768) /
  `max_tokens_per_request` (4,000,000), `routing_replay_policy` (`natural`).
  `serve_tinker.py` sets them from `--tinker-train-mode`,
  `--tinker-lease-timeout-s`, `--tinker-sampler-versions-to-keep`,
  `--tinker-coalesce-wait-s`, `--tinker-coalesce-max-wait-s`,
  `--tinker-max-concurrent-samples`, `--tinker-retrieve-poll-timeout-s`,
  `--tinker-unit-token-budget`, `--tinker-max-tokens-per-datum`,
  `--tinker-max-tokens-per-request` and `--tinker-routing-replay-policy`
  (`n_slots` comes from `--multi-lora-n-adapters`, `base_model` /
  `checkpoint_root` from `--tinker-base-model` / `--tinker-checkpoint-root`).
  The routing-cache sizes (`--tinker-routing-cache-*`, `--tinker-r3-check`) and
  the fused head (`--fused-lm-head-*`) are read from `args` by the backend, see
  `flags-and-launcher.md`.
- `ExecutorBackend` (`miles/tinker/core/service.py`): `load_slot`,
  `unload_slot`, `forward_backward`, `forward_only`, `optim_step` (returns a
  per-slot metrics dict: `grad_norm`, `skipped`, `lr`, `num_tokens`,
  `num_datums`, `num_units`, `actor_train_time`, `actor_train_tflops`,
  `actor_train_mfu`, `actor_train_tok_per_s`; with R3 on also `routing_*`
  and `fused_lm_head`, see `r3.md`), `discard_grads`, `save_slot`,
  `push_slot` (LoRA: None; full: the new engine weight version),
  `unload_adapter`, `warmup`, `sample`.
- Checkpoints: `<checkpoint_root>/<model_id>/<weights|sampler_weights>/<name>/`
  plus `tinker_meta.json` (`model_id, tenant, session_id, base_model, is_lora,
  lora_rank, lora_alpha, kind, name, sampler_version, created_at, train_mode`).
  `/api/v1/weights_info` and `load_state` resolve from the sidecar, so resume
  works after a gateway restart and from another session of the same tenant.
- Metrics on the wire: `forward_backward` returns `loss:sum` and `logprobs`
  per datum plus `num_tokens:sum` / `num_datums:sum`; `optim_step` returns the
  backend metrics dict verbatim (plain keys), always with `grad_norm`.
- Tenant = the `X-API-Key` header (`TINKER_API_KEY`; SDK 0.16.1 needs a `tml-`
  prefix, any value works). Sessions lease their models for `lease_timeout_s`
  after the last heartbeat.

## Where to start

1. Read `FORK.md`, then `deploy/abci/README.md` to bring up a gateway.
2. Run `deploy/clients/tinker_sdk_smoke.py` against it (PASS/FAIL per check).
3. Run the acceptance recipes in `docs/fork/acceptance.md`.
4. To pull upstream changes, use the `upstream-sync` skill or
   `python tools/fork/upstream_sync.py --check`.
