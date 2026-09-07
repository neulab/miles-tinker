# R3 on the trainer side (full executor, forward-only, natural fallback)

Rollout routing replay (R3, `--use-rollout-routing-replay`) replays the MoE
routing SGLang used while sampling in the Megatron training pass. Native
`train.py` already did this in `actor.train_actor`; this page covers the Tinker
gateway path, where a work unit is one `forward_backward` (or `forward_only`)
call on the full fine-tuning executor.

Code: `miles/backends/training_utils/replay_data.py` (`rollout_replay_stage`,
`active_rollout_replay_managers`), `miles/backends/megatron_utils/full/executor.py`
(`forward_backward`), `miles/backends/megatron_utils/actor.py`
(`forward_only_logprobs`), `miles/utils/replay_base.py` (`_get_replay_result`).
Tests: `tests/fast/tinker/test_full_executor.py` (R3 section),
`tests/fast/backends/training_utils/test_replay_data.py`,
`tests/fast/utils/test_replay_base.py`, `tests/fast/utils/test_replay_data_sp.py`.

## Data contract (C5)

`rollout_data["rollout_routed_experts"]` is a list with one int32 tensor per
row of shape `[len(tokens) - 1, L, K]` (L = `--num-layers` main-decoder layers,
MTP excluded; K = `--moe-router-topk`); row `i` is the routing used when token
`i` was the input. The gateway ships it for every row of the unit, DP pad rows
included (`[1, L, K]` of -1), and a datum without cached routing arrives as an
all -1 tensor. `fill_replay_data` asserts `rows == tokens - 1` per sample,
appends one -1 row per sample (the loss-masked final token), concatenates the
microbatch, pads to `tp_size * data_pad_size_multiplier`, slices the
sequence-parallel region and hands each MoE layer its `[:, layer]` column
(`register_replay_list_moe`).

## One context manager around the pass

`rollout_replay_stage(args, models, data_iterator, num_microbatches,
rollout_data, *, stage)` wraps exactly one pass:

- Active managers = `m.enabled` (set at actor init from `use_<name>_replay`)
  **and** `args.use_rollout_<name>_replay`; the routing manager is the only
  one the gateway feeds. Inactive managers are never touched.
- Enter: `fill_replay_data(...)` per active manager (this consumes and deletes
  `rollout_data[m.data_key]`), then `m.stage = stage`.
- `forward_backward` uses `stage="replay_backward"`: `forward_step`
  (`model.py`) switches a manager that is already in a replay stage to
  `replay_forward` during the forward and restores `replay_backward` for the
  recompute, so the router pops the same rows twice. A manager left at
  `fallthrough` (enabled but never filled) is *not* switched: forcing it would
  make every router pop from an empty queue.
  The fill runs outside the `actor_train` timer (MFU unaffected).
- `forward_only_logprobs` uses `stage="replay_forward"` (entered on the
  method's existing `ExitStack`).
- Exit, success or failure: `m.clear_all()` and `m.stage = "fallthrough"`, so a
  failed unit cannot leave stale rows for the next one (the executor still
  marks the window dirty).

Without the flag nothing changes; with the flag but no
`rollout_routed_experts` in the unit, `fill_replay_data` raises `ValueError`
(the gateway always attaches the key when R3 is on).

## Natural fallback for uncovered rows

`BaseReplayManager._get_replay_result` used to replace rows that were entirely
-1 (padding) with `arange % num_experts`, i.e. experts 0..K-1 regardless of the
scores. On the gateway path such rows are also the datums whose routing the
cache could not supply (`--tinker-routing-replay-policy natural`), so they now
take the live router's own top-k: `old_topk_fn(scores, topk)` (Megatron's
`_compute_topk` returns `(values, indices)`; the indices are cast to the
replayed dtype). Rows with some valid picks keep their -1 padding unchanged
(the gateway never produces them). Coverage is reported by the gateway
(`routing_uncovered_datums` / `routing_uncovered_rows` in the optim metrics).

## Sequence parallelism

Megatron forces `sequence_parallel=False` at TP1, so the SP slice inside
`fill_replay_data` is a no-op for the TP1 deployment and is live only in the
TP2 fallback; `tests/fast/utils/test_replay_data_sp.py` pins the TP2 layout
(each rank registers its half of the padded microbatch).

## Diagnostics and the typed failure (job 2245536)

The replay managers are process-wide singletons that three independent steps
have to agree on, and when they disagreed the run died with an
`IndexError: pop_forward out of range` inside Megatron's `compute_topk` with
no way to tell which step was wrong. Each step now states its numbers:

- Registration. A router registers itself in its own `__init__`
  (Megatron-LM `megatron/core/transformer/moe/router.py:284`), so a wrapper
  that builds a decoder and throws it away registers twice as many replays as
  the model has live routers -- and the live ones land in the second half,
  past the streams the fill reaches. megatron-bridge's `Qwen3VLGPTModel` does
  exactly that: `text_model.py:82` calls `GPTModel.__init__`, which builds
  `self.decoder = TransformerBlock(...)` (`gpt_model.py:232`), and
  `text_model.py:118` ("# rebuild the transformer block") replaces it with a
  `Qwen3VLTransformerBlock` from the same spec. That is what killed job
  2245536: 80 replays, the fill wrote 0..39, the 40 routers that ran read
  40..79.
  `setup_model_and_optimizer` therefore calls `reset_registrations()` before
  the build and `prune_to_model(model)` after it -- keeping only the replays a
  module of the built model still holds, in module order -- and logs one line
  per build:
  `R3 registration (actor): routing replay manager enabled=True, 80 routers
  registered by this model build, 40 live in the model, 40 dropped`.
  `register_to_module` is also idempotent per module instance.
- Fill. `fill_replay_data` logs
  `R3 fill rollout_routed_experts: microbatches=1 replays=40 streams=40
  rows_recorded=40 rows_per_replay=[1]` and raises `ReplayError` when there is
  nothing to fill (no microbatches -- `forward_backward_no_pipelining` runs the
  last microbatch even at `num_microbatches=0` -- or no registered replays), or
  when the register function left a registered router without its rows.
- Mapping. `register_replay_list_moe` raises `ReplayError` when the registered
  routers and the local MoE layers differ in number, or when a local layer id
  is past the streams the rollout tensor carries.
- Pop. An exhausted queue raises `ReplayError` naming the manager, the replay's
  stream index and the manager's `enabled`/`stage`/`replays`/`non_empty`/
  `rows_per_replay`, instead of a bare `IndexError`.

`ReplayError` is always a bug in the wiring, never a data condition: an
uncovered datum is expressed as all -1 rows (C5), not as a missing queue.
