# Gateway flags and the Qwen3.6 launcher

The existing `examples/tinker/run_qwen3_6_35b_a3b_gateway.py` also accepts
`serve --model-variant 27b` for Qwen3.6-27B. This selects the dense architecture,
TP4/EP1 training, TP8/EP1 inference, and no routing replay. Pass a custom
preserve-thinking checkpoint with `--hf-checkpoint`, and advertise its tokenizer
path or model identifier with `--model-name`. The default `35b-a3b` variant keeps
the existing MoE recipe. Node counts remain explicit launcher arguments.

The `--tinker-*` flags of `serve_tinker.py`, the two fused-head flags, the
serve-time validation they trigger, the launcher that emits them for the
35B MoE deployment (`examples/tinker/run_qwen3_6_35b_a3b_gateway.py`), and the
gateway presets that pin its options per model.
`FORK.md` lists the files; `full-finetuning.md` covers the training semantics.

## Flags (`miles/utils/arguments.py`, tinker group)

| Flag | `args.<attr>` | Default | `GatewayConfig` field / consumer |
|---|---|---|---|
| `--tinker-max-tokens-per-datum` | `tinker_max_tokens_per_datum` | 32768 | `max_tokens_per_datum` (`service._admit` rejects longer rows) |
| `--tinker-max-tokens-per-request` | `tinker_max_tokens_per_request` | 4000000 | `max_tokens_per_request` |
| `--tinker-coalesce-max-wait-s` | `tinker_coalesce_max_wait_s` | 2.0 | `coalesce_max_wait_s` (`service._coalesce_hold`: bound from the window's first chunk) |
| `--tinker-routing-replay-policy {strict,natural}` | `tinker_routing_replay_policy` | natural | `routing_replay_policy` (materialization: `strict` fails uncovered datums typed, `natural` routes them live + metrics) |
| `--tinker-routing-cache-max-bytes` | `tinker_routing_cache_max_bytes` | 137438953472 (128 GiB) | `RoutingCache(max_bytes=)` |
| `--tinker-routing-cache-ttl-s` | `tinker_routing_cache_ttl_s` | 21600.0 | `RoutingCache(ttl_s=)` |
| `--tinker-routing-prefix-stride` | `tinker_routing_prefix_stride` | 256 | `RoutingCache(prefix_stride=)` |
| `--tinker-r3-check` | `tinker_r3_check` | False | trainer: `routing_replay_manager.enable_check_replay_result = True` (threshold `MILES_TEST_R3_THRESHOLD`) |
| `--fused-lm-head-logprobs` | `fused_lm_head_logprobs` | False | `loss_hub/fused_lm_head.maybe_activate` |
| `--fused-lm-head-chunk-size` | `fused_lm_head_chunk_size` | 1024 | `fused_lm_head.chunk_size()` |

Existing flags the deployment relies on: `--use-rollout-routing-replay`
(engines get `enable_return_routed_experts=True`, the actor enables its
`routing_replay_manager`), `--sglang-enable-fp32-lm-head`,
`--rollout-max-context-len`, `--balance-data`, `--tinker-lease-timeout-s`
(flag default 3600 s; the launcher passes 600 s so a restarted client finds the
single full-FT slot free after `restart_wrapper --restart-delay 700`).

`serve_tinker.py` passes `max_tokens_per_datum`, `max_tokens_per_request`,
`coalesce_max_wait_s` and `routing_replay_policy` into `GatewayConfig` next to
the fields it already set. Other components read these with
`getattr(args, "<attr>", <default>)` / `getattr(config, "<field>", <default>)`
so that argparse namespaces built in tests need not carry them.

## Serve-time validation (`miles/utils/tinker_args.py`)

In addition to the existing full-mode / LoRA-mode checks:

- `--use-rollout-routing-replay` requires `--num-experts` (an MoE model),
  rejects `--sglang-speculative-algorithm` (the engines' routed-experts
  capture is target-only and its buffer ignores verify tokens) and warns when
  `--mtp-num-layers` is non-zero (the MTP block is outside the replay).
- `--fused-lm-head-logprobs` requires `--loss-type policy_loss`, no
  `--calculate-per-token-loss`, no `--true-on-policy-mode`, no
  `--use-rollout-entropy` (the head yields target log-probs only), no
  `--enable-mtp-training` and `--mtp-num-layers 0`/unset (the MTP block calls
  `output_layer` for its own logits), and untied embeddings.

## The 35B launcher

`examples/tinker/run_qwen3_6_35b_a3b_gateway.py serve` builds the
`serve_tinker.py` argv in blocks (ckpt, mode, tinker, r3, head, optimizer,
perf, mtp, sglang, update, topology, misc) for 4 training nodes (TP1 / EP8 /
ETP1 / DP32, distributed optimizer, fp32 grad accumulation, full uniform
recompute, `--balance-data`) and 11 TP8/EP8 engines at 64k context. Recipe
decisions pinned by `tests/fast/fork/test_qwen36_gateway_launcher.py`:

- `--mtp-num-layers 0` overrides the model definition's `1` (argparse last
  wins; model args precede `train_args` in `execute_train`) and
  `--enable-mtp-training` is never passed: no EAGLE on the engines, the fused
  head cannot serve MTP logits, one layer of activations saved.
- fp32 head on both sides: `--sglang-enable-fp32-lm-head` and
  `--fused-lm-head-logprobs --fused-lm-head-chunk-size 1024`
  (`--no-fused-lm-head` falls back to `--log-probs-chunk-size 2048`).
- R3 on by default (`--use-rollout-routing-replay`, policy `natural`, 128 GiB
  cache, 6 h TTL, stride 256); `--r3-check` adds `--tinker-r3-check`.
- Gateway limits sized for 32 rows of 65537 tokens per unit:
  `--tinker-max-tokens-per-datum 65537 --tinker-unit-token-budget 2097184
  --tinker-coalesce-wait-s 5.0 --tinker-coalesce-max-wait-s 120.0
  --tinker-max-concurrent-samples 1024 --tinker-lease-timeout-s 600`.
- `--context-len 65536` drives `--sglang-context-length 65537`,
  `--rollout-max-context-len 65536` and the `--max-tokens-per-gpu 65537`
  lower bound.
- No `--sglang-mamba-*` flag: the image resolves
  `mamba_radix_cache_strategy=auto` to `extra_buffer` for
  `Qwen3_5MoeForConditionalGeneration`; `--sglang-max-total-tokens` is emitted
  only when set (bounds the KV pool and the pinned routed-experts host cache).
- `--tp 2` is the memory fallback (`--sequence-parallel` becomes effective);
  a resumed run must keep the TP/EP/DP of its save.
- `--check-weight-update-equal` is opt-in (smoke runs).

## Gateway presets (`deploy/abci/presets/*.env`)

`deploy/abci/tinker_gateway.pbs` takes `PRESET=<name>` and reads the launcher, `LAUNCHER_ARGS`,
`EXTRA_ARGS`, the snapshot (`HF_MODEL` at `HF_REVISION` under `$HF_HOME`) and the client's
`TINKER_SAMPLE_TIMEOUT_SECONDS` from `deploy/abci/presets/<name>.env`; a non-empty job variable
replaces a preset value (runbook: `deploy/abci/README.md`, section 4).

`qwen3.8-27b-128k` is the fcs-q38 gateway, as rasp's `launch/abci/submit.sh frontiercs-*` submits it,
with one change: `--model-name` is the HF id `apurvaga/Qwen3.8-27B-preserve-thinking` where fcs-q38
passed the snapshot path, so the gateway advertises the id and a trainer's `model_name` names the id.
Through `run_qwen3_6_35b_a3b_gateway.py serve` its options give:

| Preset option | `serve_tinker.py` argv |
|---|---|
| `--model-variant 27b` | Megatron model `qwen3.6-27B`, EP1 training, TP8/EP1 engines (`--rollout-num-gpus-per-engine 8 --sglang-ep-size 1`) |
| `--model-name apurvaga/Qwen3.8-27B-preserve-thinking` | `--tinker-base-model apurvaga/Qwen3.8-27B-preserve-thinking`, the one word that differs from fcs-q38's argv |
| `--no-routing-replay` | no `--use-rollout-routing-replay` (the dense model has no expert routes) |
| `--tp 4` | `--tensor-model-parallel-size 4`: DP = 2 x `TRAIN_NODES`, DP10 at fcs-q38's `TRAIN_NODES=5` |
| `--context-len 131072` | `--rollout-max-context-len 131072`, and the engines' `--sglang-context-length`, which the launcher derives from it |
| `--max-tokens-per-gpu 131073` | `--max-tokens-per-gpu 131073`: one 128K row per microbatch |
| `--max-tokens-per-datum 131073 --unit-token-budget 1310730` | `--tinker-max-tokens-per-datum 131073 --tinker-unit-token-budget 1310730`: ten rows per unit |
| `--extra-env-vars NCCL_DEBUG=INFO` | `NCCL_DEBUG=INFO` in the Ray job's environment |
| `EXTRA_ARGS` | appended last, so they win: `--sglang-attention-backend fa3`, `--lr 1e-6` (the optimizer's initial lr; every `optim_step` sends its own Adam parameters), `--router-policy cache_aware --router-balance-abs-threshold 16 --router-balance-rel-threshold 1.5` |

`tests/fast/fork/test_qwen36_gateway_launcher.py` builds both argvs through the launcher's CLI and
requires that only `--tinker-base-model` differs, and requires every preset's options to parse with
its launcher (its `--sglang-*` flags included, against the allowlist below);
`tests/fast/fork/test_deploy_dry_run.py` pins the preset to fcs-q38's strings and checks that the job
renders a preset exactly as it renders the same job variables.

## The `--sglang-*` allowlist

miles registers `ServerArgs.add_cli_args` under the `--sglang-` prefix, so a
launcher may only emit flags that exist in the serving image.
`tools/fork/dump_sglang_server_args.py` prints `dataclasses.fields(ServerArgs)`
and its output is checked in as
`tests/fast/fork/data/sglang_server_args_fields.txt`; the launcher test maps
every emitted `--sglang-<name>` to `<name>` (dashes to underscores,
`--sglang-ep-size` to `ep_size`) and requires it in the file. Regenerate the
file inside the image whenever it changes:

```bash
module load singularity-ce/4.4.1
singularity exec --env PYTHONNOUSERSITE=1 $SIF \
    python tools/fork/dump_sglang_server_args.py > tests/fast/fork/data/sglang_server_args_fields.txt
```

The current image (sglang 0.5.19.dev56) still accepts
`--mamba-scheduler-strategy` as a deprecated alias of
`--mamba-radix-cache-strategy`, but it is not a field; the snapshot names
canonical fields only, and `scripts/run_qwen3_6_35b_a3b_mtp.py` now passes
`--sglang-mamba-radix-cache-strategy extra_buffer`.
