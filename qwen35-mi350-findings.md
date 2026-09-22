# Qwen3.5-4B on MI350 with Miles-Tinker and Hyperion SWE-rebench

Analysis date: 2026-09-15.

## Scope and evidence

Read-only inspection of `/home1/adityabs/miles-tinker` at `bbe40deb9ae63b3f0de143492f0a8e993d793d23` and `babel-compute-node:~/hyperion_modal` at `a55addd904a80d92f2e9aa670b5183cce57b55ea`. No repository source files were edited. CPU probes used Babel's existing Tinker environment and the actual Qwen3.5 tokenizer. Scratch files and downloaded tokenizer assets were confined to `/tmp`.

GPU training was not tested. The documented MI350 host `k007-002` did not resolve from this session, so the installed image's implementation remains unverified.

## Summary

This is feasible in principle, but requires more than changing the model name:

1. Select and validate the Qwen3.5 training implementation on MI350.
2. Port the legacy plugin's Tinker transport to SWE-rebench.
3. Fix Qwen3.5 rendering, tool parsing, and reasoning round-tripping.

The intended training configuration should evaluate reasoning enabled with preserved rollout history and reuse of exact generated token IDs. Non-thinking mode provides a useful baseline. GPU correctness and long-context memory remain the largest unknowns.

## 1. Miles-Tinker model support

### Two distinct construction paths

The existing Qwen3.5 examples use Miles' local model spec and `mbridge` checkpoint conversion. The AMD Tinker launcher explicitly selects `--megatron-to-hf-mode bridge`, which uses `megatron.bridge.AutoBridge` and bypasses the local spec during model construction.

Relevant local sources:

- `tools/amd_tinker.py`
- `miles/backends/megatron_utils/model_provider.py`
- `scripts/models/qwen3.5-4B.py`
- `miles_plugins/models/qwen3_5.py`
- `miles_plugins/mbridge/qwen3_5.py`

Support in one construction path does not establish equivalent text-only behavior, kernels, precision, or weight synchronization in the other.

### Text-only training

The local Miles spec extracts the nested HF `text_config` and constructs the language model. It does not require an additional vision-freezing flag.

The inspected current upstream Bridge provider constructs a vision-language model. It offers vision-freezing options and a language-only construction method, but Miles' wrapper does not select those options. Text-only inputs alone do not prove that the vision tower is excluded from construction or optimizer handling.

Two routes to investigate:

- **Local Miles implementation:** override `--megatron-to-hf-mode raw`, convert the checkpoint with the existing conversion tool, and supply `--ref-load`. This also selects Miles' Qwen3.5 weight exporter. This is the first route worth investigating to reuse existing text-only support.
- **Bridge implementation:** verify the installed provider's text-only forward, vision parameter handling, GDN kernels, and weight synchronization. Provider wiring may be needed.

Neither route was GPU-validated here. The upstream Bridge source inspected is not evidence of the exact version embedded in the pinned AMD image.

Source: [Bridge provider](https://github.com/radixark/Megatron-Bridge/blob/40b930897717941cfe2bd9806f417e28bf1bfa65/src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py).

### Model selection and knobs

Use consistently for setup and serving:

```text
--model-id Qwen/Qwen3.5-4B
--model-type qwen3.5-4B
```

Use separate runtime/model and checkpoint locations. The existing model definition provides 32 layers, hidden size 2560, vocabulary 248320, attention-output gating, Q/K normalization, rotary base 10000000, and rotary fraction 0.25. Do not transplant Qwen3 architectural arguments manually.

| Setting | Initial guidance |
| --- | --- |
| Precision | BF16. |
| GDN backend | Local Miles defaults to `--linear-attention-backend fla`; FlashQLA requires NVIDIA hardware. This flag does not configure Bridge's separate implementation. |
| `A_log` | Local Miles explicitly preserves FP32. Verify equivalent behavior in Bridge if selected. |
| Context parallelism | Currently rejected by the Tinker gateway for CP > 1. See the follow-up CP analysis below when available. |
| Training TP | Begin from the existing TP1 deployment; investigate TP2 if needed for long-sequence memory. |
| Inference TP | One GPU per engine initially; existing dense Qwen3.5 recipes also choose TP1. |
| Memory | Keep full-layer recomputation; measure long-sequence memory before adopting a 65K packing budget. DP does not split one sequence. |
| MTP/speculation | Leave disabled for initial bring-up. |
| MoE knobs | Not applicable to dense Qwen3.5-4B. |

Qwen3.5 adds recurrent-state caching alongside attention caching. Current SGLang documentation recommends the `no_buffer` Mamba cache strategy on AMD; names and defaults are version-sensitive. Inspect the pinned image before adding overrides. `--sglang-language-only` is a potential inference memory optimization where supported, not a training-side solution.

Sources: [SGLang Qwen3.5](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5), [server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md).

### Existing coding examples

This checkout has Qwen3.5-4B math/fully-async launchers and Qwen3.5 support in generic agentic infrastructure. A dedicated Qwen3.5-4B coding-agent training launcher was not found; the Harbor coding launcher currently defaults to GLM. The useful reusable component is the Qwen3.5 TITO session implementation.

## 2. Hyperion SWE-rebench port

SWE-rebench currently has an AReaL entry point. Legacy already supplies the Tinker transport. Shared Platoon code supports a custom Miles endpoint, full training through `lora_rank: 0`, explicit optimizer parameters, and captured-interaction processing.

Required work:

1. Add a rebench Tinker entry point and YAML adapted from legacy's `train_tinker.py`.
2. Port `tinker_workflow.py` and `subprocess_worker.py`.
3. Port the inference-tunnel helpers for Modal access to the trainer's OpenAI-compatible proxy.
4. Forward `llm_extra_headers` in rebench's `prepare_llm()`. Legacy already does; rebench does not. The session header associates completions with rollout training interactions.
5. Add a `tinker` dependency extra and isolated environment. Both plugins install `platoon.issue_resolution`, so environment selection matters.

Request path:

```text
Modal OpenHands agent
  -> Hyperion OpenAI-compatible proxy
  -> Miles-Tinker service
  -> MI350 SGLang inference
```

The agent-to-proxy tunnel and access to the Miles service are distinct connections.

Preserve rebench's dataset, named images, `/testbed` workspace, base-commit patch extraction, terminal evaluation, and weighted composite reward. Legacy's easy SWE-Smith subset and binary reward should not be copied over. Preserve rebench's 40-turn and 1800-second limits initially, then tune them explicitly for Qwen3.5.

Rebench's optional length reward measures turns rather than generated tokens, so it does not directly penalize long reasoning within a turn.

## 3. Renderer and proxy findings

Hyperion pins tinker-cookbook at `0be951bd38eb83c4363c1d11caddf4388bf05262`. That version already provides `qwen3_5` and `qwen3_5_disable_thinking` renderers.

CPU probes reproduced:

| Probe | Observed result |
| --- | --- |
| Reasoning followed by tool response, default renderer | Reasoning removed. |
| `strip_thinking_from_history=False` | Reasoning retained, but prompt still differs from HF. |
| Reasoning and a tool call, without visible answer text | Two extra newlines before the tool call. |
| Consecutive tool results | Separate user blocks, whereas HF groups them. |
| XML argument containing four-space-indented `return 1` | Parsed as `return 1`; indentation lost. |
| XML argument containing textual `123` | Parsed as integer 123 without consulting the tool schema. |

The XML parsing issues affect execution correctness, especially code editing. Qwen3.5 uses XML function/parameter syntax; Qwen3's JSON exact-renderer fixes cannot simply be reused.

Current upstream cookbook fixes some grouping/separator behavior, but the XML parser still needs scrutiny. A full upgrade also changes response parsing from a boolean result to a termination object, requiring compatibility work in Hyperion.

Source: [current upstream renderer](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers/qwen3_5.py).

### Proxy changes are also necessary

In `platoon/train/tinker/proxy.py`:

- `_canonicalize_messages()` does not preserve incoming `reasoning_content`.
- `_normalize_message_content()` concatenates thinking into ordinary answer text.
- `_prepare_model_input()` calls the renderer without forwarding `chat_template_kwargs`.

Thus, the rollout's `enable_thinking=False` does not control thinking through this Tinker proxy; renderer selection currently controls it. Setting `strip_thinking_from_history=False` alone does not repair reasoning preservation.

The installed OpenHands SDK forwards `reasoning_content` only for selected model names; Qwen3.5 is absent. Preserving reasoning requires SDK support or server-side storage/restoration when the client replays history.

Probe source and results: `probe_renderers.py` and `probe-results.md` in this directory. The proxy probe mirrors the inspected normalization for a fixture; it does not start the HTTP proxy or run a full OpenHands episode.

## 4. Reasoning policy and datum merging

### HF behavior

The actual HF template preserves assistant reasoning throughout the tool chain after the latest real user query. A new real user message removes reasoning from earlier exchanges. CPU probes verified both behaviors. Tool responses alone do not require one datum per assistant turn. Framework-generated user reminders or other history edits can still break prefix extension.

Sources: [HF chat template](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja), [model guidance](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices).

| Policy | Implication |
| --- | --- |
| Disable thinking | Useful baseline; parser and serialization fixes remain necessary. |
| Native HF retention | A continuous tool chain can merge; new user messages may split it. |
| Preserve all rollout reasoning | Avoids deletion across inserted user messages, but increases context length and deliberately changes history policy. |

Recommended experiment: reasoning enabled with preserved rollout history, compared against a non-thinking baseline. Generation and training must use the same conditioning history. Adding reasoning only to training data would invalidate agreement with recorded sampling logprobs.

### Miles TITO

`miles/utils/chat_template_utils/tito_tokenizer.py` defines `Qwen35TITOTokenizer`, which selects `qwen3.5_fixed.jinja`, forces `preserve_thinking=True`, reuses generated token IDs, tokenizes only appended messages, and handles the end-of-message/newline boundary.

Hyperion's current merger in `platoon/utils/tinker_data_processing.py` instead tests whether the next rendered prompt starts with the previous prompt-plus-completion tokens; otherwise it creates another datum. TITO is not automatically used by the Tinker endpoint. Similar token preservation would need integration into Hyperion's proxy/session handling.

These mismatches are generally not tokenizer randomness. Encoding is deterministic; lossy decode/re-encode, chunk boundaries, whitespace normalization, tool serialization, and actual history edits cause divergence.

## 5. Paper findings

[FrogNano, sections 2, 4 and Appendix A.1](https://arxiv.org/pdf/2609.07925) uses reasoning and trains on reasoning, answer text, and tool calls while masking observations. It does not explicitly document retention or stripping of earlier reasoning between requests.

Relevant choices:

- Five-tool Leaf harness with natural termination; the reported base-model result improves from 8.3% to 37.2% versus R2E-Gym.
- Batch 32 × 8, temperature 1, top-p 1, and 8192 output tokens per turn.
- Context/turn limits grow from 65,536/75 to approximately 131K/150.
- DPPO with standardized group advantages; no reference KL or entropy bonus.
- Adam LR 1e-6, betas 0.9/0.98, decay 0.1, clipping 1.
- Two training GPUs with CP2 and six inference GPUs on B200.
- Success-gated logarithmic penalty on assistant-generated tokens rather than turns.

Reproducing this objective, normalization, optimizer, and reward is a separate experiment from preserving the current CISPO recipe.

## 6. Recommended validation order

1. Independently validate MI350 loading, packed forward/backward, logprob agreement, optimizer update, and inference weight refresh.
2. Port rebench's Tinker transport while preserving its dataset and evaluator.
3. Test indentation-sensitive edits, parallel calls, reasoning replay, truncation, and exact prefix equality.
4. Run a small non-thinking baseline, then reasoning with preserved history and a larger per-turn output budget.
5. Measure solve rate, termination/truncation, generated tokens, datums per rollout, actual training tokens, memory, and throughput.

## 7. What would it take to use context parallelism (CP)?

Here CP means splitting the training sequence across GPUs. This is useful when retaining reasoning makes individual training datums long. It does not fix prefix merging or reduce the number of datums, and it does not increase the inference server's context limit.

**Conclusion: CP2 is a reasonable next target, but the current Miles-Tinker backend cannot enable it through flags alone.** There is already a substantial CP foundation in Miles' local Qwen3.5 implementation. The remaining work includes Tinker input slicing, output reconstruction, loss/gradient validation, and verification in the actual MI350 image. This section concerns full fine-tuning; multi-LoRA CP would need separate validation.

### 7.1 Existing support and explicit blocker

The AMD launcher already accepts `--cp` and emits `--context-parallel-size`. However, `miles/utils/tinker_args.py` explicitly asserts `context_parallel_size == 1`, with the explanation that the gateway returns per-datum logprobs and CP is unsupported. Removing that assertion alone would expose both input-shape failures and incorrect API results.

Miles already provides:

- CP-aware packing, sequence offsets, and microbatch sizing in `miles/backends/training_utils/data.py`.
- Target-logprob slicing and `assemble_log_prob_from_cp` in `miles/backends/training_utils/cp_utils.py`. The latter reconstructs a full response from CPU tensors indexed by CP rank, accounts for shifted next-token targets and the two chunks assigned to each rank, and rejects missing ranks.
- Hybrid CP setup in `miles_plugins/models/cp_utils.py`, called by the Megatron actor. It configures the local `HuggingfaceAttention` wrappers containing GatedDeltaNet (GDN).
- Native FLA recurrent-state and convolution-state handling in `miles_plugins/models/qwen3_5.py`. `build_gdn_cp_context` requires an importable `fla.ops.cp`; its failure message specifies `flash-linear-attention >= 0.4.2`. Package version alone is insufficient: the installed kernels must work on MI350.

Qwen3.5-4B has 24 GDN layers and eight full-attention layers. With the local implementation and PP1, verify that hybrid CP setup reports all 24 GDN modules. A fallback that gathers the entire sequence and duplicates GDN computation would undermine the intended benefit.

### 7.2 Required Tinker changes

| Area | Current behavior | Necessary change |
| --- | --- | --- |
| Client tokenwise inputs | `get_rollout_data` CP-slices sampling logprobs, but leaves Tinker `advantages` and `loss_weights` as full-response arrays. Tinker losses multiply them by local logprobs. | Slice client weights and advantages with exactly the same target offsets. Scope this to the appropriate inputs so ordinary RL paths that compute their own advantages are not double-sliced. |
| Forward/backward outputs | `tinker_losses._record_per_datum` emits a local logprob vector and local scalar loss, without CP rank metadata. | Return sufficient CP rank/group and sequence metadata, or reconstruct within the CP group before emitting. Preserve empty local target slices. |
| Runtime assembly | `runtime.py:forward_backward` keeps the first result for a sample index and discards duplicates. | Reconstruct the complete vector in original target order and sum per-datum losses across CP exactly once. Deduplicate TP replicas; do not sum replicated TP results. |
| Forward-only API | The actor returns local logprobs; `runtime.py:forward_only` also keeps the first result per sample. | Apply the same reconstruction to forward-only calls, including calls used by client-side custom losses. |
| Exact gradient sums | Tinker defines gradients as the sum over datums, with client normalization encoded in weights/advantages. | Prove CP1 and CP2 preserve this contract across packing, DP, multiple requests, and optimizer steps before relaxing the validation guard. |

Relevant files: `miles/tinker/runtime.py`, `miles/backends/training_utils/data.py`, `miles/backends/training_utils/loss_hub/tinker_losses.py`, `miles/backends/training_utils/loss.py`, `miles/backends/megatron_utils/actor.py`, and `miles/backends/megatron_utils/full/executor.py`.

A practical implementation could attach rank metadata to detached per-datum outputs and use the existing CPU assembly helper in the runtime. The trainer-group interface already returns a list of actor results. This avoids gathering vocabulary-sized logits or adding collectives solely for API response construction. This proposal initially targets the default packed, zigzag CP layout; an all-gather CP mode needs its own layout audit.

**Loss scaling deserves its own correctness test.** `loss.py` currently forces the Tinker global batch divisor to one and scales by the microbatch count and `intra_dp_cp.size`. The full executor deliberately defers gradient finalization until `optim_step`, so several `forward_backward` requests accumulate into one exact-sum update. Preserve that design. Adding a CP multiplier or a parameter-gradient all-reduce without accounting for existing Megatron normalization could scale the update incorrectly or reduce already-reduced accumulated gradients again. Compare parameter gradients and the resulting optimizer update, not only forward logprobs.

### 7.3 The model-construction choice also matters for CP

The local raw/spec path has the FLA CP implementation described above. The AMD launcher's Bridge path bypasses that local spec, and the hybrid-CP scanner recognizes local wrappers rather than arbitrary Bridge-native GDN modules. Consequently, enabling CP on the launcher does not establish that Bridge uses this implementation.

There is an additional concrete compatibility concern: in the inspected Bridge revision, `Qwen35VLModelProvider.finalize()` forces `calculate_per_token_loss=True` for CP greater than one. Miles copies the CLI setting into the provider, while Tinker explicitly requires this setting to be false for its exact-sum protocol. The effective provider/model and schedule settings must agree. Resolve this in the chosen text-only model integration; simply enabling per-token loss is not a valid Tinker workaround. Source: [Bridge provider at the inspected revision](https://github.com/radixark/Megatron-Bridge/blob/40b930897717941cfe2bd9806f417e28bf1bfa65/src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py).

The local raw/spec path is therefore the clearest existing source-code starting point for CP experiments. It still requires the AMD loading, checkpoint conversion, precision, export, and inference-refresh validation described in section 1. The installed MI350 Bridge/FLA/Megatron versions were not inspected, so neither route is certified here.

### 7.4 Memory, communication, and GPU allocation

CP spreads sequence work across GPUs. It is not by itself parameter sharding, and the memory reduction will not necessarily be proportional to CP size.

In particular, `miles_plugins/models/hf_attention.py` converts the full-attention zigzag layout into contiguous packed shards for FLA GDN, then converts back. Those conversion helpers use full-hidden-state all-gathers, including corresponding backward transformations. GDN has native state passing, but these surrounding temporary allocations and communication remain. Profile them at the intended context length. `--allgather-cp` skips these particular conversions because it supplies a different layout; its full-attention, scoring, and reconstruction paths must be checked before treating it as an optimization.

At TP1/PP1, the training DP size is `training GPUs / CP`. Thus, on an eight-GPU machine:

| Training allocation | Training topology | GPUs left for inference |
| --- | --- | --- |
| 2 GPUs | CP2, DP1 | 6 |
| 4 GPUs | CP2, DP2 | 4 |
| 4 GPUs | CP4, DP1 | 4 |

These are candidate allocations, not validated MI350 performance recommendations. Start with CP2 on GPUs within one node, TP1/PP1, BF16, and the validated FLA implementation. CP4 is a later experiment after correctness and memory measurements. With a fixed number of training GPUs, increasing CP reduces independent DP replicas, so longer sequences fitting does not automatically mean better rollout throughput.

**Packing-budget detail:** `get_data_iterator` sizes microbatches using `max_tokens_per_gpu * cp_size`. With CP2, a setting of 32,768 therefore gives an approximately 65,536-token aggregate packing budget per CP group, subject to padding and the batching algorithm. Leaving a 65,536 setting unchanged while moving from CP1 to CP2 can instead allow approximately 131,072 packed tokens per group. That can spend the expected memory savings on a larger microbatch. This knob controls packing, and is not a hard per-sample context limit or an inference-context setting.

For a controlled experiment, keep the actual sequences and aggregate packed-token budget fixed while comparing CP1 and CP2. Then increase context and tune packing/recomputation from measured peak memory. Use a separate checkpoint directory: the current full executor documents optimizer resume as requiring the same parallel topology. Do not assume a CP1 optimizer checkpoint can resume under CP2.

### 7.5 Existing tests and the validation still needed

Existing source evidence:

- `tests/fast/backends/training_utils/test_cp_log_prob_assembly.py` covers reconstruction for CP2/4/8, uneven response lengths, padded layouts, and missing ranks. It does not exercise the Tinker integration.
- `tests/e2e/precision/test_qwen3_5_cp_correctness.py` is registered for the MI350 nightly suite as well as NVIDIA CI. It compares a small GDN module's forward outputs and input gradients against a non-CP reference. Registration is not evidence of a current passing run. It does not compare parameter gradients, optimizer updates, mixed packed documents, or the complete Tinker model.
- `tests/e2e/megatron/test_qwen3_5_35B_A3B_cp.py` contains full-model CP2/CP4 coverage for a different Qwen3.5 model, but its NVIDIA CI registration is currently disabled as flaky. It does not establish Qwen3.5-4B Tinker correctness on MI350.

Recommended acceptance sequence after implementing the gateway support:

1. Verify local GDN CP2 forward, input gradients, and parameter gradients on the actual MI350 image, including packed document boundaries and convolution/state resets.
2. Compare full Qwen3.5-4B CP1 versus CP2 on identical small packed batches: target logprobs, scalar loss, gradients, gradient norm, and one Adam update. Include attention and GDN paths, tied embeddings, FP32-sensitive parameters, and recomputation.
3. Exercise Tinker forward-only, weighted cross-entropy/custom loss, and the intended CISPO or PPO loss. Cover uneven lengths, zero weights, empty local target slices, dummy DP rows, several datums per request, and several requests per optimizer step. Reject missing CP results instead of returning partial answers.
4. Repeat with CP2/DP2 to detect scaling or sample-ownership mistakes, then validate checkpoint save/resume at the same topology and inference weight refresh.
5. Only then try 32K/64K/128K contexts as memory permits, measuring peak allocation, communication, time per update, and end-to-end SWE rollout throughput.

Hyperion should continue sending complete datums and receiving complete per-datum results; CP should remain a backend implementation detail. Its renderer/prefix fixes are still independently necessary. A sensible order is a working CP1 Qwen3.5 Tinker run, then CP2 correctness, then long-context SWE-rebench tuning.

**Scope estimate:** meaningful backend integration plus distributed correctness and performance validation, rather than a launcher-only change. Existing model and reconstruction helpers reduce the amount of new machinery needed, but a reliable timeline depends on the unresolved MI350 model-construction and kernel compatibility checks. No GPU tests or source-code changes were performed for this CP analysis.
