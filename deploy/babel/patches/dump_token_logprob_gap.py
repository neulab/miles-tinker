"""Record per-token sample-vs-train logprob gaps, tagged with rollout staleness.

``kl_sample_train`` and ``importance_weight_*`` are scalars, so when they move
there is no way to ask which tokens moved them, or whether those tokens came
from fresh or stale rollouts. Everything needed is already in memory at
``compute_training_metrics`` and is discarded:

  sampling_logprobs   what sglang reported when it generated the token
  training_logprobs   what Megatron computed for that token at that position
  target_tokens       the token id (tinker_data_processing.py:174)
  advantages          the per-token advantage, UNSCALED (see below)
  checkpoint_version  the sampler version the rollout was generated at

``tinker_data_processing.py:168`` asserts the alignment this depends on::

    input_tokens_T.length == len(target_tokens_T) == len(sampled_logprobs_T)
                          == len(advantages_T) == len(mask_T)

so one action mask selects the same positions in all of them and attribution is
exact rather than inferred.

Why staleness is worth tagging. ``stale_rollouts`` cannot answer it: at
rl.py:601 that counter increments only when a rollout EXCEEDS max_staleness and
is dropped, so rollouts stale by exactly one step are trained on and never
counted. Recording ``checkpoint_version`` against the trainer's ``train_step``
gives the real per-rollout staleness, which separates a sglang-vs-Megatron
forward divergence (would appear on staleness=0) from a weight-lag effect (would
concentrate at staleness=1).

Why both a gap ranking and a gradient ranking. A large importance ratio on a
token with ~0 advantage moves nothing; a modest ratio on a high-advantage token
can dominate. Rows are therefore selected by |gap| AND by |clipped_iw * adv|,
so "largest divergence" and "largest gradient contribution" can be told apart
rather than conflated.

Note on ``adv``: these are the advantages BEFORE the ``1/sum(mask)`` microbatch
scaling applied at rl.py:639, which is applied to a separate filtered copy. The
recorded value is the raw per-token advantage; multiply by the batch's own
scale factor to recover what the optimizer saw.

Cost and safety. Off unless PLATOON_LOGPROB_DUMP_DIR is set. Keeps top-K by gap
plus top-K by gradient weight plus a uniform random sample (so the outliers have
a background distribution to be judged against), writes one JSONL per step, never
touches the returned metrics, and swallows its own exceptions -- a diagnostic
must not be able to kill a training run.

    PLATOON_LOGPROB_DUMP_DIR=<dir>     enable, write there
    PLATOON_LOGPROB_DUMP_TOPK=400      rows by |gap| per datum         (default 400)
    PLATOON_LOGPROB_DUMP_GRADK=400     rows by |clipped_iw * adv|      (default 400)
    PLATOON_LOGPROB_DUMP_SAMPLE=400    uniform rows per datum          (default 400)
    PLATOON_LOGPROB_DUMP_PREFIX_GAP=1.5  also write the datum's FULL input token
                                         sequence when its max |gap| reaches this,
                                         so the token can be replayed offline
                                         through both stacks. 0 disables.

Usage: python dump_token_logprob_gap.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_LOGPROB_DUMP_DIR"
TARGET = "platoon/train/tinker/rl.py"

# 1. give compute_training_metrics the trainer's step number
SIG_ANCHOR = """    forward_backward_result: tinker.ForwardBackwardOutput,
    loss_fn: str,
    loss_fn_config: dict[str, Any],
) -> dict[str, float]:
"""
SIG_REPLACEMENT = """    forward_backward_result: tinker.ForwardBackwardOutput,
    loss_fn: str,
    loss_fn_config: dict[str, Any],
    train_step: int | None = None,
) -> dict[str, float]:
"""

# 2. pass it from the call site, where shared_state is in scope
CALL_ANCHOR = """                            loss_fn=self.config.train.loss_fn,
                            loss_fn_config=self.config.train.loss_fn_config,
                        )
"""
CALL_REPLACEMENT = """                            loss_fn=self.config.train.loss_fn,
                            loss_fn_config=self.config.train.loss_fn_config,
                            train_step=shared_state.train_step,
                        )
"""

# 3. the dump itself, right after the importance weights exist
ANCHOR = """            importance_weights = torch.exp(log_ratio)
            all_importance_weights.append(importance_weights)
"""
REPLACEMENT = '''            importance_weights = torch.exp(log_ratio)
            all_importance_weights.append(importance_weights)

            if os.getenv("PLATOON_LOGPROB_DUMP_DIR"):
                try:
                    _lp_record_token_gap(
                        os.environ["PLATOON_LOGPROB_DUMP_DIR"], datum, action_mask,
                        sampling_logprobs_actions, training_logprobs_actions,
                        log_ratio, importance_weights, train_step, loss_fn, loss_fn_config,
                    )
                except Exception as _lp_err:  # a diagnostic must never kill a run
                    logger.warning(f"token logprob dump failed: {_lp_err}")
'''

HELPER_ANCHOR = "def compute_training_metrics("

HELPER = '''def _lp_record_token_gap(
    dump_dir, datum, action_mask, sample_lp, train_lp, log_ratio,
    importance_weights, train_step, loss_fn, loss_fn_config,
):
    """Write per-token sample-vs-train gaps for one datum, tagged with staleness.

    Selected by |gap| and separately by |clipped_iw * advantage|, plus a uniform
    sample, so a large importance ratio and a token that actually dominates the
    gradient can be told apart. Every row carries the rollout's
    checkpoint_version against the trainer's train_step.

    Written to be cheap enough to leave on for every step. Two things dominate a
    naive version and are avoided here:

      * Boolean-masking the full-length tensors costs ~30 ms each when torch has
        many threads configured (measured: 91 ms/datum at 128 threads vs 15 ms at
        8). The trainer runs with --cpus-per-task=128, so it would take the slow
        path. Only the kept rows are gathered instead -- a few hundred, not 19k.
      * Per-row float(tensor[i]) indexing from Python costs far more than the
        JSON itself. Values are converted once with .tolist().
    """
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np
    import torch as _torch

    topk = int(os.getenv("PLATOON_LOGPROB_DUMP_TOPK", "64"))
    gradk = int(os.getenv("PLATOON_LOGPROB_DUMP_GRADK", "64"))
    nsample = int(os.getenv("PLATOON_LOGPROB_DUMP_SAMPLE", "64"))

    fi = datum.loss_fn_inputs
    gap = log_ratio.detach().float().cpu()
    n = gap.numel()
    if n == 0:
        return
    iw = importance_weights.detach().float().cpu()

    # advantages are already action-masked by the caller for logprobs, but not
    # for these two, so index them through the action positions directly.
    pos = _torch.from_numpy(_np.flatnonzero(action_mask.detach().cpu().numpy()))
    adv_full = fi["advantages"].to_torch().detach().float().cpu() if "advantages" in fi else None
    adv = adv_full[pos] if adv_full is not None else _torch.zeros(n)

    lo = float(loss_fn_config.get("clip_low_threshold", 0.0)) if loss_fn in ("cispo", "ppo") else 0.0
    hi = float(loss_fn_config.get("clip_high_threshold", 5.0)) if loss_fn in ("cispo", "ppo") else float("inf")
    grad_w = _torch.clamp(iw, min=lo, max=hi) * adv

    keep = set()
    keep.update(gap.abs().topk(min(topk, n)).indices.tolist())
    keep.update(grad_w.abs().topk(min(gradk, n)).indices.tolist())
    if nsample > 0 and n > len(keep):
        keep.update(_torch.randperm(n)[: min(nsample, n)].tolist())
    sel = _torch.tensor(sorted(keep), dtype=_torch.long)

    # gather only what is written, then one .tolist() per column
    abs_pos = pos[sel].tolist()
    toks = fi["target_tokens"].to_torch().detach().cpu()[pos[sel]].tolist()
    s_l = sample_lp.detach().float().cpu()[sel].tolist()
    t_l = train_lp.detach().float().cpu()[sel].tolist()
    g_l = gap[sel].tolist()
    i_l = iw[sel].tolist()
    a_l = adv[sel].tolist()
    w_l = grad_w[sel].tolist()

    ckpt_ver = None
    if "checkpoint_version" in fi:
        try:
            ckpt_ver = int(fi["checkpoint_version"].to_torch().item())
        except Exception:
            ckpt_ver = None
    stale = (train_step - ckpt_ver) if (train_step is not None and ckpt_ver is not None) else None

    step_tag = train_step if train_step is not None else _LP_DUMP_STEP[0]
    path = _Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    dumps = _json.dumps

    # When a datum carries an extreme gap, also write its FULL input token
    # sequence. Without this an offline replay is impossible: the row gives the
    # token and its position, but scoring that position again needs every token
    # before it, and nothing else in the pipeline records the prompt ids (the
    # prompt dumps keep only the last 48 of ~7.7k).
    prefix_id = None
    thresh = float(os.getenv("PLATOON_LOGPROB_DUMP_PREFIX_GAP", "1.5"))
    max_gap = float(gap.abs().max())
    if thresh > 0 and max_gap >= thresh:
        try:
            toks_all = []
            for ch in datum.model_input.chunks:
                t_ = getattr(ch, "tokens", None)
                if t_ is not None:
                    toks_all.extend(list(t_))
            _LP_PREFIX_N[0] += 1
            prefix_id = f"s{step_tag:05d}_d{_LP_PREFIX_N[0]:04d}"
            with open(path / f"prefix_{prefix_id}.json", "w") as pf:
                _json.dump(
                    {
                        "prefix_id": prefix_id, "step": train_step,
                        "ckpt_ver": ckpt_ver, "stale": stale,
                        "max_abs_gap": round(max_gap, 6),
                        "n_action_tokens": n,
                        "n_input_tokens": len(toks_all),
                        "input_tokens": toks_all,
                    },
                    pf,
                )
        except Exception:
            prefix_id = None
    with open(path / f"logprob_gap_step{step_tag:05d}.jsonl", "a") as fh:
        fh.write(
            "".join(
                dumps(
                    {
                        "step": train_step, "ckpt_ver": ckpt_ver, "stale": stale,
                        "prefix_id": prefix_id,
                        "pos": abs_pos[k], "tok": toks[k],
                        "sample_lp": round(s_l[k], 6), "train_lp": round(t_l[k], 6),
                        "gap": round(g_l[k], 6), "iw": round(i_l[k], 6),
                        "adv": round(a_l[k], 8), "grad_w": round(w_l[k], 8),
                        "n_action_tokens": n,
                    }
                )
                + "\\n"
                for k in range(len(abs_pos))
            )
        )


_LP_DUMP_STEP = [0]
_LP_PREFIX_N = [0]


def compute_training_metrics('''

STEP_ANCHOR = "    kl_metrics = compute_kl_sample_train(datums, training_logprobs_list)\n"
STEP_REPLACEMENT = (
    '    if os.getenv("PLATOON_LOGPROB_DUMP_DIR"):\n'
    "        _LP_DUMP_STEP[0] += 1\n"
    "    kl_metrics = compute_kl_sample_train(datums, training_logprobs_list)\n"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No trainer at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"token logprob dump already present in {target}")
        return
    checks = (
        ("helper", HELPER_ANCHOR),
        ("dump site", ANCHOR),
        ("step counter", STEP_ANCHOR),
        ("signature", SIG_ANCHOR),
        ("call site", CALL_ANCHOR),
    )
    for name, anchor in checks:
        if source.count(anchor) != 1:
            raise SystemExit(
                f"Expected exactly one {name} anchor in {target} (found {source.count(anchor)}); "
                "Platoon changed upstream and this patch needs review."
            )
    source = source.replace(HELPER_ANCHOR, HELPER, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    source = source.replace(STEP_ANCHOR, STEP_REPLACEMENT, 1)
    source = source.replace(SIG_ANCHOR, SIG_REPLACEMENT, 1)
    source = source.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)
    target.write_text(source)
    print(f"added staleness-tagged per-token logprob gap dump to {target}")


if __name__ == "__main__":
    main()
