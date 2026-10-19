# Fused chunked fp32 lm head (`--fused-lm-head-logprobs`)

Code: `miles/backends/training_utils/loss_hub/fused_lm_head.py` (the head),
`miles/backends/training_utils/loss_hub/math_utils.py` (dispatch at the top of
`calculate_log_probs_and_entropy`), `miles/backends/megatron_utils/full/executor.py`
and `miles/backends/megatron_utils/actor.py` (activation around the
forward/backward and the forward-only pass). Tests:
`tests/fast/backends/training_utils/test_fused_lm_head.py` (CPU only). Port of
platoon's `platoon/train/areal/fused_linear_logprobs.py` with the numerics
below.

## Why

The Tinker losses (`policy_loss`, `cross_entropy`, `importance_sampling`,
`ppo`, `cispo`, `dro`) need one log-prob per response token. On the standard
path the model is called with `fp32_output=False`
(`megatron_utils/model.py`), the Megatron `output_layer` (a
`ColumnParallelLinear`, `gather_output=False`) emits bf16 logits `[T, 1, V/TP]`
and `calculate_log_probs_and_entropy` upcasts them to fp32 in chunks of
`--log-probs-chunk-size` (default -1 = the whole tensor). At 64k tokens and a
248,320-token vocabulary that is 32.5 GB of bf16 logits plus 32.5 GB for their
gradient per rank at TP1, plus a 65 GB fp32 copy when unchunked.

With `--fused-lm-head-logprobs` the `[T, V]` tensor is never materialized:

- the installed wrapper makes `output_layer.forward` return the TP/SP-mapped
  **hidden states** `[S, 1, H]` (and `None` for the bias), so the "logits" the
  loss receives are `[1, T, H]`;
- `_iter_response_chunks` slices the response rows `[R, H]` as before;
- `calculate_log_probs_and_entropy` dispatches to `fused_lm_head.target_logprobs`,
  a custom autograd function that projects `--fused-lm-head-chunk-size` tokens
  at a time (`[chunk, V/TP]` fp32 logits, log-sum-exp, target selection) and
  recomputes the projection in backward instead of saving it.

Peak transient memory at chunk 1024 and V = 248,320 (TP1): one fp32
`[1024, V]` chunk (~1 GB) in forward; in backward the chunk's probabilities,
the bf16 logit gradient and the fp32 weight-gradient accumulator
(`[V, H]` fp32, 2 GB) - about 4.5 GB in total.

## Numerics: parity with `--sglang-enable-fp32-lm-head`

SGLang's `enable_fp32_lm_head` computes
`torch.mm(hidden, W.T, out_dtype=torch.float32)` (bf16 GEMM, fp32 accumulate,
fp32 output) and runs the sampling softmax in fp32. The fused head uses the
same expression for every chunk, in the forward and in the backward recompute
(`_mm_fp32`); softmax statistics (max, sum-exp, target selection) are fp32. The
trainer therefore scores exactly the distribution the sampler sampled from,
whereas the standard path rounds the logits to bf16 first (platoon's original
port did the same through `F.linear` + `.float()`).

On CPU (unit tests) `aten::mm.dtype` has no kernel, so the head falls back to
`torch.mm(h.float(), W.float().t())`: the same arithmetic without the
tensor-core path, which is why the CPU tests compare against a pure fp32
reference at `atol=1e-6`.

Gradients: the logit gradient is cast to the weight dtype before the two
gradient GEMMs (like the standard bf16 head); dHidden is returned in the
hidden dtype; dW is accumulated in fp32 across chunks. When Megatron DDP
manages the weight (`weight.main_grad` exists and DDP tracks
`weight.grad_added_to_main_grad`) the fp32 dW is added into `main_grad`
directly and `grad_added_to_main_grad` is set (no bf16 rounding of the
gradient, `None` returned to autograd); otherwise dW is returned in the weight
dtype and DDP's backward post-hook folds it into `main_grad`.
`reset_grad_metadata_keep_grads` clears the flag per unit, so accumulation
across `forward_backward` calls works as on the standard path.

Tensor parallelism: the per-chunk max and the (sum-exp, selected) pair are
all-reduced over the TP group (one MAX and one stacked SUM collective per
chunk). Sequence parallelism (TP2 fallback only; Megatron disables SP at TP1)
is handled by the wrapper: it all-gathers the SP hidden states with
`gather_from_sequence_parallel_region` (backward reduce-scatters), otherwise
`copy_to_tensor_model_parallel_region` (backward all-reduces dHidden).

## Output transforms (Muse Glimmer)

The fused path replaces the head's `forward`, so whatever that forward does after the projection would be skipped
without an error. `install` therefore asks which class defines the `forward` the head runs
(`fused_lm_head.output_transform`):

- Megatron's `ColumnParallelLinear`, and GPTModel's `LinearCrossEntropyModule` (every GPT and Qwen head: its
  `forward` is `ColumnParallelLinear`'s unless asked for Megatron's fused cross-entropy loss, which the wrapper
  refuses while active): nothing after the projection.
- megatron-bridge's `MuseGlimmerOutputLayerMixin` (Muse Glimmer, put on the head with `extend_instance`):
  `logits * config.output_multiplier` (0.19611613513818404, 1/sqrt(26)), then
  `final_logit_softcapping * tanh(logits / final_logit_softcapping)` (cap 20).
- Any other class (a Gemma softcap mixin, a subclass that overrides `forward`): `install` refuses
  (`RuntimeError`), so a run either reproduces the head exactly or trains with `--no-fused-lm-head`.

Each chunk applies the transform to its fp32 projection in SGLang's order (`logit_scale`, then the softcap, both in
fp32 under `--sglang-enable-fp32-lm-head`) before the temperature and the log-softmax. The backward multiplies the
logit gradient by the transform's derivative, `multiplier * (1 - tanh^2)` (one extra fp32 `[chunk, V/TP]` tensor per
chunk), before it is cast for the two gradient GEMMs. Without the fix the fused head scored Glimmer's raw projection,
logits about 5 times too large and uncapped; megatron-bridge's own mixin runs the transform in bf16 (max error 0.12 in
a capped logit), the fused head in fp32 (2e-6 against SGLang's fp32 head).

## Flags and requirements

- `--fused-lm-head-logprobs` (store_true) turns the head on; the modules read
  it as `getattr(args, "fused_lm_head_logprobs", False)`.
- `--fused-lm-head-chunk-size N` (default 1024): tokens per projection chunk.
  `--log-probs-chunk-size` is not used while the fused head is active.
- Requirements checked at install time (`RuntimeError`): exactly one lm head
  per pipeline rank (VPP=1), no lm-head bias, `gather_output=False`, no
  `disable_grad_reduce`, no `defer_embedding_wgrad_compute`, no MuP, no MTP
  (`mtp_process`; run with `--mtp-num-layers 0`, the MTP block calls
  `output_layer` for its own logits), and a head class whose output transform
  it knows (above).
- Unsupported while active (`AssertionError`/`RuntimeError`): entropy
  (`use_rollout_entropy`), `--true-on-policy-mode`, sampling masks,
  `runtime_gather_output`, `output_cross_entropy_loss` (Megatron's fused CE).
  The Tinker losses use none of them; serve-time validation should reject the
  combinations (`validate_tinker_serve_args`).
- Activation is scoped: `fused_lm_head.maybe_activate(args, model)` wraps the
  full executor's `run_forward_backward_pass` and the actor's
  `compute_log_prob` (which `forward_only_logprobs` uses). Outside those
  contexts the wrapped `output_layer.forward` is the original Megatron
  forward (sampler pushes, checkpointing and the LoRA executor are untouched).
  The wrapper is installed lazily on the first activated call and is
  idempotent.

## Validation

Unit (CPU, `PYTHONPATH=$PWD $PY -m pytest tests/fast/backends/training_utils/test_fused_lm_head.py --confcutdir=tests/fast/backends/training_utils -p no:cacheprovider -o addopts=""`):
forward vs the fp32 reference at `atol=1e-6` for chunk sizes 1/5/64 and a
temperature variant; gradients vs a rounding-aware reference (fp32 autograd
gradient rounded to bf16 once, then fp32 GEMMs) and vs pure fp32 autograd at
bf16 tolerance; both weight-gradient paths (`main_grad` present / absent); a
fake 2-shard TP (two threads, barrier-based all-reduce) with and without
sequence parallelism; install guards; wrapper passthrough and errors;
`maybe_activate` on/off; dispatch through `calculate_log_probs_and_entropy`
and `get_log_probs_and_entropy` with `[1, T, H]` hidden states; the output
transform (Glimmer's 0.196 multiplier and cap 20 on inputs that reach the cap,
a cap that bites on most logits, a multiplier alone) against an fp32 autograd
reference in forward and backward, with a temperature and on the fake 2-shard
TP; the head classes it knows and the one it refuses.

Integration on the 5-node smoke (design 4.7 b/c, run after the launcher lands):

- (a) Train the same 3 datums (2k/16k/64k tokens, loss `cross_entropy`) with
  the fused head on and off: per-token logprob max abs diff < 2e-3 and
  `optim/grad_norm` within 1e-2 relative.
- (b) Sampler-vs-trainer: sample 3 sequences with
  `--sglang-enable-fp32-lm-head`, then score the identical token sequences
  through `forward` (`forward_only_logprobs` with the fused head). Record the
  median |delta| of per-token logprobs; it must be below the value obtained
  with `--log-probs-chunk-size 2048` (bf16 logits) on the same sequences.
- (c) Read `torch.cuda.max_memory_allocated` from the actor's `log_perf_data`
  for both configurations; the fused head should remove the `[T, V]` logits
  and their gradient from the peak.
- The `fused_lm_head` (0/1) key in the `optim_step` metrics
  (`optim/fused_lm_head`, `perf.jsonl`) records which configuration produced a
  step.

Fallback if the fused head is infeasible on the smoke:
`--log-probs-chunk-size 2048` with TP2.
