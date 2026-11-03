# Gateway flags and the Qwen3.6 launcher

The `--tinker-*` flags of `serve_tinker.py`, the two fused-head flags, the
serve-time validation they trigger, and the launcher that emits them for the
35B MoE and 27B dense deployments (`examples/tinker/run_qwen3_6_35b_a3b_gateway.py`).
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
  `--use-rollout-entropy` (the head yields target log-probs only), and untied
  embeddings. MTP training uses its own chunked auxiliary loss; see
  [the fused-head documentation](fused-lm-head.md#mtp-training).

## The 35B launcher

`examples/tinker/run_qwen3_6_35b_a3b_gateway.py serve` builds the
`serve_tinker.py` argv in blocks (ckpt, mode, tinker, r3, head, optimizer,
perf, mtp, sglang, update, topology, misc) for 4 training nodes (TP1 / EP8 /
ETP1 / DP32, distributed optimizer, fp32 grad accumulation, full uniform
recompute, `--balance-data`) and 11 TP8/EP8 engines at 64k context. Recipe
decisions pinned by `tests/fast/fork/test_qwen36_gateway_launcher.py`:

- By default, `--mtp-num-layers 0` overrides the model definition's `1`
  (argparse last wins; model args precede `train_args` in `execute_train`).
  MTP training and speculative decoding are both opt-in, preserving the
  existing recipe and its memory use.
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

## Dense 27B and MTP options

`--model-variant 27b` selects `scripts/models/qwen3.6-27B.py`, TP4/EP1
training, EP1 inference, and no routing replay. The default model is
`Qwen/Qwen3.6-27B`; use `--model-name` and `--hf-checkpoint` for a custom
preserve-thinking snapshot. Training TP can be 1, 2, or 4, matching the four
KV heads. Other allocation and context flags work as in the MoE recipe.

| Launcher option | Default | Effect |
|---|---|---|
| `--enable-mtp-training` | off | Load and train the MTP block alongside the policy. |
| `--mtp-num-layers` | 1 | Number of checkpoint MTP layers to load when training is enabled. |
| `--mtp-loss-scaling-factor` | 0.2 | Weight on the draft's token-mean auxiliary loss. |
| `--mtp-speculative-decoding` | off | Enable EAGLE with two draft steps, top-k 1, three verification tokens, and `extra_buffer` hybrid-state caching. |

The training and decoding switches are independent. A frozen-draft benchmark
uses only `--mtp-speculative-decoding`; a run that updates the draft uses both.
Speculative decoding rejects routing replay, so the 35B recipe must also pass
`--no-routing-replay`. The installed SGLang 0.5.19.dev56 uses its speculative
V2 worker without a separate environment switch. Older serving images may
need `--extra-env-vars SGLANG_ENABLE_SPEC_V2=1`; verify the image before use.

Example for a dense 27B experiment on an existing Ray cluster:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=10.0.80.66 \
  python examples/tinker/run_qwen3_6_35b_a3b_gateway.py serve \
    --model-variant 27b --actor-num-nodes 2 --rollout-num-gpus 48 \
    --enable-mtp-training --mtp-speculative-decoding
```

The default 35B arguments remain covered by the original exact-argument test.
Recordings for both models and frozen/trained MTP drafts live under
`tests/snapshots/launch_scripts/py/examples/tinker/run_qwen3_6_35b_a3b_gateway.py/`.

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
