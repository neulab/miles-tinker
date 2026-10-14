"""Context-parallel parity probe for a full fine-tuning Tinker gateway.

``run`` sends a fixed, seeded set of datums to one gateway and dumps what comes
back as JSON: Tinker ``forward`` logprobs, ``forward_backward`` per-datum losses
and logprobs for cross_entropy, importance_sampling and cispo (one loss per
optimizer step), the grad_norm of each ``optim_step``, and the forward logprobs
after the steps. Run it against a CP=1 gateway and a CP>1 gateway serving the
same snapshot (fresh models, same order of calls), then ``compare`` the dumps.

``compare REF CAND --cp N [--noise NOISE]`` reports, for every recorded vector,
max/mean |dlogprob| and a KL estimate overall, per zigzag chunk of the
candidate's CP layout (rank r holds chunks r and 2N-1-r) and within +-8 tokens of
the boundaries, per-datum loss deltas and grad_norm ratios; it exits 1 when a
threshold fails. A boundary bug only moves tokens next to its boundary, which a
mean hides, and the boundaries differ by layer type:

* ring attention and MRoPE positions: the zigzag chunk boundaries of each datum;
* GatedDeltaNet layers in ``--linear-cp-mode chunkwise``: Megatron converts the
  zigzag layout to a contiguous split of the whole packed microbatch row (CP parts
  of total/CP tokens) and passes the recurrent state and the conv halo between
  ranks there, at positions that depend on the packing. A datum that trains alone
  in its microbatch puts them at a known place: multiples of
  ``roundup(2 * ceil(L / 2N), TP * pad_multiplier)`` for an L-token datum. The
  lone long datums (4000 and 8000 tokens, off the zigzag boundaries) gate windows
  there; headwise mode has no such split.

Short rows (up to 512 model_input tokens) are also gated one datum at a time
(max |dlogprob| before the first optimizer step) and never trimmed out of the KL,
so an error confined to the tiny datums cannot hide among the 8K-token rows.
``--noise`` names the dump of a correct alternative layout of the reference (for
example TP2 at CP=1): bf16 kernels differ between layouts, so the candidate may
deviate by up to ``--noise-factor`` times that run's deviation (or the absolute
thresholds, whichever is larger).

The datums: model_input lengths 1 (a 2-token, zero-weight row like the runtime's
PAD_ROW), 2 .. 2*max_cp (3 .. 2*max_cp+1 tokens: some CP rank holds one or no
target) and 31 .. 8191 (up to 8192 tokens), plus the lone 3999- and
7999-token datums. ``--tokens text`` (default) cuts
them from a fixed English passage repeated end to end, tokenized with the
model's tokenizer: long rows see the passage many times, so a model that attends
and positions correctly predicts the repeats with high confidence across chunk
boundaries. ``--tokens motif`` uses repeated random-token motifs instead (no
tokenizer; improbable targets make the gradient norm a fragile comparison).
Sampling logprobs are -1 (+-0.05), so importance ratios stay at or below e. More
probes per run: one short datum alone (fewer real rows than DP ranks: the runtime
pads the unit with PAD_ROWs, so some ranks train pad-only microbatches), each
lone long datum (forward, and forward_backward in the first window with its
cross_entropy weight on the second half only, so its gradient reaches the first
half through the GatedDeltaNet state handoff) and the forward of the datums in a
permuted order (different packing: the run's own noise floor).

Example (``run`` inside each gateway job, e.g. CLIENT_SCRIPT=deploy/abci/clients/cp_parity.sh; ``compare``
anywhere, it reads only the dumps):
  $PY deploy/clients/tinker_cp_parity.py run --base-model Qwen/Qwen3.5-4B --cp 2 --tp 1 --out $JOBDIR/parity-cp2.json
  python3 deploy/clients/tinker_cp_parity.py compare parity-cp1.json parity-cp2.json --cp 2 --noise parity-tp2.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

LOSSES = ("cross_entropy", "importance_sampling", "cispo")
LONG_MODEL_INPUT_LENGTHS = (31, 64, 127, 255, 512, 1000, 2047, 4095, 8191)
# lone datums of 4000 and 8000 tokens: not multiples of 2*CP*TP*128, so the GatedDeltaNet split of
# their microbatch row lies off their zigzag chunk boundaries
LONE_MODEL_INPUT_LENGTHS = (3999, 7999)
BOUNDARY_WINDOW = 8
SAMPLING_LOGPROB = -1.0
# rows up to this many model_input tokens never see the passage repeat: no multi-nat flips on ambiguous
# repeats, so they are gated one by one (max |d|) and kept whole in the KL
SHORT_ROW = 512
DEFAULT_PAD_MULTIPLIER = 128  # miles' --data-pad-size-multiplier

# --tokens text: an original passage, tokenized by the gateway model's tokenizer and repeated end to end
PASSAGE = (
    "The weather station on the north cliff was built in 1908, a square stone room with one window facing "
    "the sea and one facing the town. Every morning at six the keeper climbed the forty-two steps, wiped the "
    "salt from the glass, and wrote four numbers in a green ledger: the air temperature, the pressure, the "
    "speed of the wind, and the height of the swell at the harbour wall. In winter the numbers changed "
    "quickly. A pressure of 1012 at dawn could fall to 991 by noon, and the fishermen learned to read the "
    "ledger before they read the sky. In summer the numbers barely moved, and the keeper spent the afternoons "
    "repairing nets, counting gulls, and teaching the children of the town how to read a barometer. She told "
    "them that a falling needle is a question and a rising needle is an answer, and that the sea always "
    "replies to the wind a little later than you expect. When the station closed in 1987, the ledgers filled "
    "eleven shelves. A student from the university copied them by hand, one page each evening, for three "
    "years. Her table showed that the coldest morning had been the ninth of February 1929, at minus eleven "
    "degrees, and that the strongest wind, ninety-four knots, had come on a quiet October night with almost no "
    "warning at all. The station is a museum now. Visitors climb the same forty-two steps, look through the "
    "same two windows, and read the last line in the last ledger, written in the keeper's careful hand: "
    "pressure steady, wind from the west, swell low, all boats home."
)

# compare thresholds (CLI-overridable), calibrated on Qwen3.5-4B (bf16): a correct alternative layout
# (TP2 at CP=1) deviates from CP=1 by mean |d| 6.7e-3 (2.1e-2 on an 11-token row), p99 0.10 and KL 2.6e-4,
# the size of the sampler/trainer gap. --noise <that run> gates relative to it instead.
DEFAULT_THRESHOLDS = {
    "mean_abs": 0.03,  # mean |dlogprob| per recorded vector
    "kl": 1e-3,  # mean(exp(d) - 1 - d) without the top 0.1% |d| (a few flips on ambiguous repeats dominate the mean)
    "p99_abs": 0.2,  # 99th percentile |dlogprob| (the max is one token on an improbable target)
    "boundary_excess": 3.0,  # mean |d| within +-8 of a boundary <= this x interior mean + 1e-3
    # with --noise: or this x the noise run's own window/interior ratio at the same rows (it has no boundary
    # there; windows sit near the start of the short rows, which differ more under any layout change)
    "boundary_noise_margin": 1.5,
    "per_datum_max_abs": 0.3,  # max |d| of each short row before the first optimizer step (all layouts <= 0.24)
    "loss_rtol": 0.02,  # step 1 per-datum losses (same weights in both runs)
    "loss_atol": 1e-3,
    "grad_norm_rtol": 0.02,  # step 1 grad_norm (same weights): the per-token-loss bug is >= 2x
    # later steps: the first Adam steps move every weight by ~lr * sign(g), so gradient noise becomes
    # sign flips and the grad_norm wanders (a TP2 run moves 9% at step 2); a sanity bound only
    "post_update_grad_norm_rtol": 0.25,
}


# --------------------------------------------------------------------------- datums


def datum_lengths(max_cp: int, max_len: int) -> list[int]:
    """model_input lengths; a datum has n + 1 tokens."""
    tiny = list(range(1, 2 * max_cp + 1))
    return tiny + [n for n in LONG_MODEL_INPUT_LENGTHS if n + 1 <= max_len]


def build_rows(lengths: list[int], seed: int, stream: list[int] | None = None) -> list[dict]:
    """Seeded token rows and loss inputs, plain lists (JSON- and SDK-ready).

    With ``stream`` (the tokenized passage) each row is a window of the stream repeated end to
    end, starting at a seeded offset; without it, a repeated random motif."""
    rng = random.Random(seed)
    rows = []
    for n in lengths:
        if stream:
            offset = rng.randrange(len(stream))
            tokens = [stream[(offset + i) % len(stream)] for i in range(n + 1)]
        else:
            motif_len = max(4, min(256, (n + 1) // 4))
            motif = [rng.randrange(1000, 150_000) for _ in range(motif_len)]
            tokens = [motif[i % motif_len] for i in range(n + 1)]
        pad_like = n == 1
        sign = rng.choice((-1.0, 1.0))
        rows.append(
            {
                "tokens": tokens,
                "weights": [0.0 if pad_like else 1.0 / n] * n,
                "advantages": [0.0 if pad_like else sign / n] * n,
                "logprobs": [0.0 if pad_like else SAMPLING_LOGPROB + rng.gauss(0.0, 0.05) for _ in range(n)],
            }
        )
    return rows


def lone_rows(max_len: int, seed: int, stream: list[int] | None = None) -> tuple[list[int], list[dict]]:
    """The lone long datums (model_input lengths, rows) with their cross_entropy weight on the second
    half of each row only: the gradient then reaches the first half through the GatedDeltaNet state
    handed across the CP split of the row."""
    lengths = [n for n in LONE_MODEL_INPUT_LENGTHS if n + 1 <= max_len]
    rows = build_rows(lengths, seed, stream)
    for n, row in zip(lengths, rows, strict=True):
        half = n // 2
        row["weights"] = [0.0] * half + [1.0 / (n - half)] * (n - half)
    return lengths, rows


def to_datums(types, rows: list[dict], loss_fn: str) -> list:
    datums = []
    for row in rows:
        tokens = row["tokens"]
        inputs = {"target_tokens": tokens[1:]}
        if loss_fn == "cross_entropy":
            inputs["weights"] = row["weights"]
        else:
            inputs["logprobs"] = row["logprobs"]
            inputs["advantages"] = row["advantages"]
        datums.append(types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs))
    return datums


# --------------------------------------------------------------------------- run


def _outputs(result) -> dict:
    logprobs, losses = [], []
    for output in result.loss_fn_outputs:
        logprobs.append([float(x) for x in output["logprobs"].to_numpy()])
        loss = output.get("loss:sum")
        losses.append(float(loss.to_numpy().reshape(-1)[0]) if loss is not None else 0.0)
    return {"logprobs": logprobs, "losses": losses, "metrics": dict(result.metrics or {})}


def _unload(base_url: str, model_id: str) -> None:
    import httpx

    httpx.post(
        f"{base_url.rstrip('/')}/api/v1/unload_model",
        json={"model_id": model_id},
        headers={"X-API-Key": os.environ["TINKER_API_KEY"]},
        timeout=120,
    ).raise_for_status()


def run(args) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hostguard  # noqa: F401  (refuses login nodes)

    os.environ.setdefault("TINKER_API_KEY", "tml-cp-parity")
    import tinker
    from tinker import types

    base_url = args.base_url or (Path(args.endpoint_file).read_text().strip() if args.endpoint_file else None)
    base_url = base_url or os.environ["TINKER_BASE_URL"]
    service = tinker.ServiceClient(base_url=base_url)
    training = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank)
    model_id = training.get_info().model_id
    started = time.time()
    try:
        dump = _probe(training, types, args)
    finally:
        if not args.keep_model:
            _unload(base_url, model_id)
    dump["meta"].update(base_url=base_url, seconds=round(time.time() - started, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dump))
    print(f"cp parity run: wrote {args.out} in {dump['meta']['seconds']} s")
    return 0


def _merge(outputs: list[dict]) -> dict:
    """One output dict for datums sent one call each."""
    return {
        "logprobs": [vector for output in outputs for vector in output["logprobs"]],
        "losses": [loss for output in outputs for loss in output["losses"]],
        "metrics": {},
    }


def _probe(training, types, args) -> dict:
    """The fixed call sequence: forward (in order, permuted, a lone short datum, each lone long datum),
    one loss per optimizer step (the lone datums join the first window), forward after the steps."""
    stream = None
    if args.tokens == "text":
        stream = [int(t) for t in training.get_tokenizer().encode(PASSAGE, add_special_tokens=False)]
    lengths = datum_lengths(args.max_cp, args.max_len)
    rows = build_rows(lengths, args.seed, stream)
    single = [rows[lengths.index(min(lengths, key=lambda n: abs(n - 11)))]]
    lone_lengths, lone = lone_rows(args.max_len, args.seed + 2, stream)
    order = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(order)
    print(
        f"cp parity run: model={args.base_model} cp={args.cp} tp={args.tp} tokens={args.tokens} datums={len(rows)} "
        f"lengths={lengths} lone={lone_lengths}"
    )

    def forward(batch: list[dict]) -> dict:
        return _outputs(training.forward(to_datums(types, batch, "cross_entropy"), "cross_entropy").result())

    def forward_backward(batch: list[dict], loss_fn: str) -> dict:
        return _outputs(training.forward_backward(to_datums(types, batch, loss_fn), loss_fn).result())

    dump = {
        "meta": {
            "base_model": args.base_model,
            "cp": args.cp,
            "tp": args.tp,
            "pad_multiplier": args.pad_multiplier,
            "label": args.label,
            "seed": args.seed,
            "tokens": args.tokens,
            "stream_tokens": len(stream) if stream else 0,
            "learning_rate": args.learning_rate,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "lengths": lengths,
        "single_length": len(single[0]["tokens"]) - 1,
        "lone_lengths": lone_lengths,
        "forward_before": forward(rows),
    }
    permuted = forward([rows[i] for i in order])["logprobs"]
    unpermuted = [None] * len(rows)
    for position, index in enumerate(order):
        unpermuted[index] = permuted[position]
    dump["forward_before_permuted"] = {"logprobs": unpermuted, "losses": [0.0] * len(rows), "metrics": {}}
    dump["forward_single"] = forward(single)
    dump["forward_lone"] = _merge([forward([row]) for row in lone])

    adam = types.AdamParams(learning_rate=args.learning_rate)
    dump["steps"] = []
    for step, loss_fn in enumerate(LOSSES[: args.steps]):
        record = {"loss_fn": loss_fn, "forward_backward": forward_backward(rows, loss_fn)}
        if step == 0:  # the lone datums train in the same window: pad-only microbatches on some ranks
            record["forward_backward_single"] = forward_backward(single, loss_fn)
            record["forward_backward_lone"] = _merge([forward_backward([row], loss_fn) for row in lone])
        metrics = dict(training.optim_step(adam).result().metrics or {})
        record.update(optim_metrics=metrics, grad_norm=metrics.get("grad_norm"))
        dump["steps"].append(record)
        print(f"  step {step + 1} {loss_fn}: grad_norm {record['grad_norm']} skipped {metrics.get('skipped')}")
    dump["forward_after"] = forward(rows)
    return dump


# --------------------------------------------------------------------------- compare


def zigzag_layout(model_input_length: int, cp: int) -> tuple[np.ndarray, np.ndarray]:
    """Per target position: its zigzag chunk index and whether it lies within +-8 rows of a boundary.

    A datum has ``n + 1`` tokens and a one-token prompt, so target j is scored by logits row j;
    get_batch splits the row into 2*cp chunks of ceil((n + 1) / (2 * cp)) rows."""
    n = model_input_length
    chunk = -(-(n + 1) // (2 * cp))
    rows = np.arange(n)
    chunks = rows // chunk
    near = np.zeros(n, dtype=bool)
    for boundary in range(chunk, n, chunk):
        near |= _window(rows, boundary)
    return chunks, near


def gdn_split_rows(model_input_length: int, cp: int, tp: int, pad_multiplier: int) -> list[int]:
    """Logits rows where chunkwise GatedDeltaNet CP hands the recurrent state (and the conv halo) to
    the next rank, for a datum that trains alone in its microbatch.

    get_batch gives each rank ``2 * ceil(L / 2cp)`` tokens of an L-token datum, padded to a multiple
    of TP * pad_multiplier; Megatron splits the resulting ``cp * part`` global row into cp contiguous
    parts, so rank k starts at row ``k * part``."""
    tokens = model_input_length + 1
    pad = tp * pad_multiplier
    part = -(-(2 * -(-tokens // (2 * cp))) // pad) * pad
    return [k * part for k in range(1, cp) if k * part < model_input_length]


def _window(rows: np.ndarray, boundary: int) -> np.ndarray:
    return (rows >= boundary - BOUNDARY_WINDOW) & (rows < boundary + BOUNDARY_WINDOW)


def vector_report(
    ref: list[list[float]],
    cand: list[list[float]],
    lengths: list[int],
    cp: int,
    gdn_splits: list[list[int]] | None = None,
) -> dict:
    """Deltas of one recorded vector set: overall, per zigzag chunk, near zigzag boundaries, near the
    GatedDeltaNet splits ``gdn_splits`` (per datum; lone datums only) and per datum."""
    assert len(ref) == len(cand) == len(lengths), "dumps disagree on the datum count"
    gdn_splits = gdn_splits or [[] for _ in lengths]
    deltas, chunk_ids, nears, gdn_nears, interiors, long_rows = [], [], [], [], [], []
    for a, b, n, splits in zip(ref, cand, lengths, gdn_splits, strict=True):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        assert a.shape == b.shape == (n,), f"logprob length mismatch: {a.shape} {b.shape} for n={n}"
        chunks, near = zigzag_layout(n, cp)
        rows = np.arange(n)
        gdn_near = np.zeros(n, dtype=bool)
        for split in splits:
            gdn_near |= _window(rows, split)
        # boundary vs interior only where a chunk has an interior: short rows are all "near" a
        # boundary and differ more under any layout change, which would mask a boundary bug
        has_interior = -(-(n + 1) // (2 * cp)) > 2 * BOUNDARY_WINDOW
        deltas.append(b - a)
        chunk_ids.append(chunks)
        nears.append(near & has_interior)
        gdn_nears.append(gdn_near)
        interiors.append(~near & ~gdn_near & has_interior)
        long_rows.append(np.full(n, n > SHORT_ROW))
    d = np.concatenate(deltas)
    chunk_ids = np.concatenate(chunk_ids)
    near = np.concatenate(nears)
    gdn_near = np.concatenate(gdn_nears)
    interior = np.concatenate(interiors)
    long_tokens = np.concatenate(long_rows)
    abs_d = np.abs(d)
    k3 = np.exp(d) - 1.0 - d
    # a few flips on ambiguous repeats of the passage dominate the KL of long rows: drop their top 0.1%,
    # never a short row's token
    kept = np.ones(d.size, dtype=bool)
    if long_tokens.sum() >= 1000:
        kept = ~long_tokens | (abs_d <= np.quantile(abs_d[long_tokens], 0.999))
    report = {
        "tokens": int(d.size),
        "max_abs": float(abs_d.max()),
        "mean_abs": float(abs_d.mean()),
        "p99_abs": float(np.percentile(abs_d, 99)),
        "kl_all": float(k3.mean()),
        "kl": float(k3[kept].mean()),
        "nonfinite": int((~np.isfinite(d)).sum()),
        "near_boundary": _stats(abs_d[near]),
        "near_gdn_split": _stats(abs_d[gdn_near]),
        "interior": _stats(abs_d[interior]),
        "per_chunk": {
            int(k): {**_stats(abs_d[chunk_ids == k]), "rank": int(min(k, 2 * cp - 1 - k))}
            for k in np.unique(chunk_ids)
        },
    }
    report["per_datum_max_abs"] = [float(np.abs(delta).max()) if delta.size else 0.0 for delta in deltas]
    report["short_max_abs"] = max(
        (worst for worst, n in zip(report["per_datum_max_abs"], lengths, strict=True) if n <= SHORT_ROW), default=0.0
    )
    return report


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"tokens": 0, "max_abs": 0.0, "mean_abs": 0.0}
    return {"tokens": int(values.size), "max_abs": float(values.max()), "mean_abs": float(values.mean())}


def vector_failures(
    name: str,
    report: dict,
    limits: dict,
    thresholds: dict,
    *,
    lengths: list[int],
    pre_update: bool,
    noise: dict | None = None,
    factor: float = 2.0,
) -> list[str]:
    failures = []
    if report["nonfinite"]:
        failures.append(f"{name}: {report['nonfinite']} non-finite deltas")
    for key in ("mean_abs", "kl", "p99_abs"):
        if report[key] > limits[key]:
            failures.append(f"{name}: {key} {report[key]:.3g} > {limits[key]:.3g}")
    interior = report["interior"]
    for key, label in (("near_boundary", "chunk boundaries"), ("near_gdn_split", "the GatedDeltaNet CP splits")):
        window = report[key]
        # short datums lie entirely within the window of some boundary: no interior to compare against
        if not window["tokens"] or not interior["tokens"]:
            continue
        allowed = _window_ratio_limit(thresholds, noise, key) * interior["mean_abs"] + 1e-3
        if window["mean_abs"] > allowed:
            failures.append(
                f"{name}: mean |d| near {label} {window['mean_abs']:.3g} > {allowed:.3g} "
                f"(interior {interior['mean_abs']:.3g})"
            )
    if pre_update:
        failures.extend(_short_row_failures(name, report, thresholds, lengths, noise, factor))
    return failures


def _window_ratio_limit(thresholds: dict, noise: dict | None, key: str) -> float:
    """How many times the interior mean |d| a boundary window may reach: ``boundary_excess``, or
    ``boundary_noise_margin`` x the ratio a correct layout (the noise run, without that boundary)
    shows at the same rows."""
    ratio = thresholds["boundary_excess"]
    if noise is not None and noise[key]["tokens"] and noise["interior"]["mean_abs"] > 0:
        noise_ratio = noise[key]["mean_abs"] / noise["interior"]["mean_abs"]
        ratio = max(ratio, thresholds["boundary_noise_margin"] * noise_ratio)
    return ratio


def _short_row_failures(
    name: str, report: dict, thresholds: dict, lengths: list[int], noise: dict | None, factor: float
) -> list[str]:
    """Each short row on its own (same weights in both runs): its worst token may not exceed the floor
    or ``factor`` x the noise run's worst token of that datum."""
    failures = []
    for i, (n, worst) in enumerate(zip(lengths, report["per_datum_max_abs"], strict=True)):
        if n > SHORT_ROW:
            continue
        allowed = thresholds["per_datum_max_abs"]
        if noise is not None:
            allowed = max(allowed, factor * noise["per_datum_max_abs"][i])
        if worst > allowed:
            failures.append(f"{name}: datum {i} (n={n}) max |d| {worst:.3g} > {allowed:.3g}")
    return failures


def vector_limits(thresholds: dict, noise: dict | None, factor: float) -> dict:
    """Absolute thresholds, widened to ``factor`` x the deviation of a correct alternative layout."""
    if noise is None:
        return {key: thresholds[key] for key in ("mean_abs", "kl", "p99_abs")}
    return {key: max(thresholds[key], factor * noise[key]) for key in ("mean_abs", "kl", "p99_abs")}


def loss_failures(
    name: str, ref: list[float], cand: list[float], thresholds: dict, noise: list[float] | None, factor: float
) -> tuple[dict, list[str]]:
    a, b = np.asarray(ref, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    allowed = thresholds["loss_atol"] + thresholds["loss_rtol"] * np.abs(a)
    if noise is not None:
        allowed = np.maximum(allowed, factor * np.abs(np.asarray(noise, dtype=np.float64) - a))
    excess = np.abs(b - a) - allowed
    report = {
        "max_abs": float(np.abs(b - a).max()) if a.size else 0.0,
        "sum_ref": float(a.sum()),
        "sum_cand": float(b.sum()),
        "worst_datum": int(np.argmax(excess)) if a.size else -1,
    }
    failures = [f"{name}: per-datum loss off by {report['max_abs']:.3g}"] if a.size and excess.max() > 0 else []
    return report, failures


def vector_pairs(ref: dict, other: dict) -> list[tuple[str, list, list, list[int], bool, bool]]:
    """(name, reference logprobs, other logprobs, model_input lengths, lone, pre_update) of every
    recorded vector. ``lone``: one datum per call (GatedDeltaNet split windows apply); ``pre_update``:
    recorded before the first optimizer step, so both runs scored the same weights."""
    lengths, single, lone = ref["lengths"], [ref["single_length"]], ref.get("lone_lengths") or []
    pairs = [
        (
            "forward_before",
            ref["forward_before"]["logprobs"],
            other["forward_before"]["logprobs"],
            lengths,
            False,
            True,
        ),
        ("forward_single", ref["forward_single"]["logprobs"], other["forward_single"]["logprobs"], single, True, True),
    ]
    if lone:
        pairs.append(
            ("forward_lone", ref["forward_lone"]["logprobs"], other["forward_lone"]["logprobs"], lone, True, True)
        )
    for i, (a, b) in enumerate(zip(ref["steps"], other["steps"], strict=True)):
        assert a["loss_fn"] == b["loss_fn"], "the dumps ran different losses"
        name = f"step{i + 1}_{a['loss_fn']}"
        pre_update = i == 0
        pairs.append(
            (name, a["forward_backward"]["logprobs"], b["forward_backward"]["logprobs"], lengths, False, pre_update)
        )
        for key, suffix, probe_lengths in (
            ("forward_backward_single", "_single", single),
            ("forward_backward_lone", "_lone", lone),
        ):
            if key in a and probe_lengths:
                pairs.append((name + suffix, a[key]["logprobs"], b[key]["logprobs"], probe_lengths, True, pre_update))
    pairs.append(
        ("forward_after", ref["forward_after"]["logprobs"], other["forward_after"]["logprobs"], lengths, False, False)
    )
    return pairs


def _gdn_splits(meta: dict, lengths: list[int], lone: bool, cp: int) -> list[list[int]] | None:
    """Per datum the GatedDeltaNet split rows of the candidate's layout (lone datums only)."""
    if not lone or cp <= 1:
        return None
    tp = meta.get("tp") or 1
    pad_multiplier = meta.get("pad_multiplier") or DEFAULT_PAD_MULTIPLIER
    return [gdn_split_rows(n, cp, tp, pad_multiplier) for n in lengths]


def compare(args) -> int:
    ref, cand = json.loads(Path(args.ref).read_text()), json.loads(Path(args.cand).read_text())
    noise = json.loads(Path(args.noise).read_text()) if args.noise else None
    cp = args.cp or cand["meta"].get("cp") or 1
    thresholds = {key: getattr(args, key) for key in DEFAULT_THRESHOLDS}
    factor = args.noise_factor
    for other in (cand, noise):
        assert other is None or (
            other["lengths"] == ref["lengths"] and other.get("lone_lengths") == ref.get("lone_lengths")
        ), "the dumps were made with different datum sets"
    report = {"cp": cp, "thresholds": thresholds, "vectors": {}, "noise_vectors": {}, "losses": {}, "grad_norm": []}
    failures = []

    noise_by_name = {}
    if noise is not None:
        for name, a, b, lens, lone, _ in vector_pairs(ref, noise):
            splits = _gdn_splits(cand["meta"], lens, lone, cp)  # the candidate's split rows, measured on the noise run
            noise_by_name[name] = report["noise_vectors"][name] = vector_report(a, b, lens, cp, splits)
    for name, a, b, lens, lone, pre_update in vector_pairs(ref, cand):
        vector = report["vectors"][name] = vector_report(a, b, lens, cp, _gdn_splits(cand["meta"], lens, lone, cp))
        noise_vector = noise_by_name.get(name)
        limits = vector_limits(thresholds, noise_vector, factor)
        failures.extend(
            vector_failures(
                name,
                vector,
                limits,
                thresholds,
                lengths=lens,
                pre_update=pre_update,
                noise=noise_vector,
                factor=factor,
            )
        )

    for i, (a, b) in enumerate(zip(ref["steps"], cand["steps"], strict=True)):
        name = f"step{i + 1}_{a['loss_fn']}"
        noise_step = noise["steps"][i] if noise is not None else None
        loss_report, loss_fail = loss_failures(
            name,
            a["forward_backward"]["losses"],
            b["forward_backward"]["losses"],
            thresholds,
            noise_step["forward_backward"]["losses"] if noise_step else None,
            factor,
        )
        report["losses"][name] = loss_report
        if i == 0:  # later steps start from weights that already differ: reported, not gated
            failures.extend(loss_fail)
            for key, suffix in (("forward_backward_single", "_single"), ("forward_backward_lone", "_lone")):
                if key in a and key in b:
                    probe_report, probe_fail = loss_failures(
                        name + suffix,
                        a[key]["losses"],
                        b[key]["losses"],
                        thresholds,
                        noise_step[key]["losses"] if noise_step and key in noise_step else None,
                        factor,
                    )
                    report["losses"][name + suffix] = probe_report
                    failures.extend(probe_fail)
        ratio = b["grad_norm"] / a["grad_norm"] if a["grad_norm"] else math.nan
        noise_ratio = noise_step["grad_norm"] / a["grad_norm"] if noise_step and a["grad_norm"] else None
        allowed = thresholds["grad_norm_rtol"] if i == 0 else thresholds["post_update_grad_norm_rtol"]
        if noise_ratio is not None:
            allowed = max(allowed, factor * abs(noise_ratio - 1.0))
        report["grad_norm"].append(
            {"step": i + 1, "ref": a["grad_norm"], "cand": b["grad_norm"], "ratio": ratio, "noise_ratio": noise_ratio}
        )
        if not abs(ratio - 1.0) <= allowed:
            failures.append(
                f"step {i + 1}: grad_norm ratio {ratio:.4f} (ref {a['grad_norm']:.5g}, cand {b['grad_norm']:.5g}, "
                f"allowed +-{allowed:.3g})"
            )
    # each run's own noise floor: the same forward with the datums packed in another order
    for label, dump in (("ref", ref), ("cand", cand)):
        if "forward_before_permuted" in dump:
            floor = vector_report(
                dump["forward_before"]["logprobs"], dump["forward_before_permuted"]["logprobs"], ref["lengths"], cp
            )
            report[f"noise_floor_{label}"] = {
                key: floor[key] for key in ("max_abs", "mean_abs", "p99_abs", "kl", "kl_all")
            }

    print_report(report, ref["meta"], cand["meta"], noise["meta"] if noise else None)
    report["failures"] = failures
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))
    print("\nPASS: candidate matches the reference" if not failures else "\nFAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


def _label(meta: dict) -> str:
    return (
        f"{meta.get('label') or meta.get('base_url')} "
        f"(cp={meta.get('cp')}, tp={meta.get('tp', '?')}, tokens={meta.get('tokens', 'motif')})"
    )


def print_report(report: dict, ref_meta: dict, cand_meta: dict, noise_meta: dict | None = None) -> None:
    print(f"reference: {_label(ref_meta)}")
    print(f"candidate: {_label(cand_meta)}, layout cp={report['cp']}")
    if noise_meta:
        print(f"noise:     {_label(noise_meta)} (a correct alternative layout of the reference)")
    header = (
        f"{'vector':34s} {'tokens':>7s} {'max|d|':>9s} {'mean|d|':>9s} {'p99|d|':>9s} {'kl':>9s} {'kl(all)':>9s} "
        f"{'bnd mean':>9s} {'gdn mean':>9s} {'int mean':>9s} {'short max':>9s}"
    )
    print(header)
    for name, v in report["vectors"].items():
        rows = [("", v)] + ([("  noise", report["noise_vectors"][name])] if name in report["noise_vectors"] else [])
        for suffix, r in rows:
            gdn = f"{r['near_gdn_split']['mean_abs']:9.2e}" if r["near_gdn_split"]["tokens"] else f"{'-':>9s}"
            print(
                f"{name + suffix:34s} {r['tokens']:7d} {r['max_abs']:9.2e} {r['mean_abs']:9.2e} {r['p99_abs']:9.2e} "
                f"{r['kl']:9.2e} {r['kl_all']:9.2e} {r['near_boundary']['mean_abs']:9.2e} {gdn} "
                f"{r['interior']['mean_abs']:9.2e} {r['short_max_abs']:9.2e}"
            )
    print(
        "per zigzag chunk (forward_before): "
        + ", ".join(
            f"c{k}/r{c['rank']} max {c['max_abs']:.2e} mean {c['mean_abs']:.2e}"
            for k, c in report["vectors"]["forward_before"]["per_chunk"].items()
        )
    )
    for label in ("ref", "cand"):
        if f"noise_floor_{label}" in report:
            floor = report[f"noise_floor_{label}"]
            print(
                f"noise floor {label} (same run, permuted packing): max {floor['max_abs']:.2e} "
                f"mean {floor['mean_abs']:.2e} kl {floor['kl']:.2e} (all tokens {floor['kl_all']:.2e})"
            )
    for name, loss in report["losses"].items():
        print(
            f"loss {name}: max |dloss| {loss['max_abs']:.3e}, sum ref {loss['sum_ref']:.6g} cand {loss['sum_cand']:.6g}"
        )
    for g in report["grad_norm"]:
        noise = f", noise ratio {g['noise_ratio']:.5f}" if g["noise_ratio"] is not None else ""
        print(f"grad_norm step {g['step']}: ref {g['ref']:.6g} cand {g['cand']:.6g} ratio {g['ratio']:.5f}{noise}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    run_parser = sub.add_parser("run", help="send the parity datums to one gateway and dump the results")
    run_parser.add_argument("--endpoint-file")
    run_parser.add_argument("--base-url")
    run_parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    run_parser.add_argument("--lora-rank", type=int, default=32, help="ignored by a full fine-tuning gateway")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--cp", type=int, default=1, help="the gateway's --context-parallel-size (recorded)")
    run_parser.add_argument(
        "--tp", type=int, default=1, help="the gateway's --tensor-model-parallel-size (recorded: GDN split rows)"
    )
    run_parser.add_argument(
        "--pad-multiplier",
        type=int,
        default=DEFAULT_PAD_MULTIPLIER,
        help="the gateway's --data-pad-size-multiplier (recorded: GDN split rows)",
    )
    run_parser.add_argument("--label", default="")
    run_parser.add_argument("--max-cp", type=int, default=8, help="tiny datums of 3 .. 2*max_cp+1 tokens")
    run_parser.add_argument("--max-len", type=int, default=8192, help="longest datum in tokens")
    run_parser.add_argument("--steps", type=int, default=3, choices=[1, 2, 3])
    run_parser.add_argument("--learning-rate", type=float, default=1e-6)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--tokens", choices=["text", "motif"], default="text")
    run_parser.add_argument("--keep-model", action="store_true")
    compare_parser = sub.add_parser("compare", help="compare a candidate dump against a reference dump")
    compare_parser.add_argument("ref")
    compare_parser.add_argument("cand")
    compare_parser.add_argument("--cp", type=int, default=None, help="zigzag layout to report (default: cand's cp)")
    compare_parser.add_argument("--json-out")
    compare_parser.add_argument("--noise", help="dump of a correct alternative layout of the reference (e.g. TP2)")
    compare_parser.add_argument("--noise-factor", type=float, default=2.0)
    for key, value in DEFAULT_THRESHOLDS.items():
        compare_parser.add_argument(f"--{key.replace('_', '-')}", dest=key, type=float, default=value)
    args = parser.parse_args()
    return run(args) if args.mode == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
