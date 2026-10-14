"""deploy/clients/tinker_cp_parity.py: the parity datums and the compare report (no gateway needed)."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY = REPO_ROOT / "deploy" / "clients" / "tinker_cp_parity.py"


@pytest.fixture(scope="module")
def parity():
    spec = importlib.util.spec_from_file_location("tinker_cp_parity", PARITY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # no hostguard / tinker import until `run`
    return module


def test_datums_cover_tiny_pad_like_and_long_rows(parity):
    lengths = parity.datum_lengths(max_cp=4, max_len=8192)
    assert lengths[:8] == list(range(1, 9)), "3 .. 2*max_cp+1 tokens, plus the 2-token row"
    assert max(lengths) == 8191 and all(n + 1 <= 8192 for n in lengths)
    assert max(parity.datum_lengths(max_cp=8, max_len=1024)) == 1000
    rows = parity.build_rows(lengths, seed=0)
    assert rows == parity.build_rows(lengths, seed=0), "seeded: two gateways get identical datums"
    for n, row in zip(lengths, rows, strict=True):
        assert len(row["tokens"]) == n + 1
        assert len(row["weights"]) == len(row["advantages"]) == len(row["logprobs"]) == n
    pad_like = rows[0]
    assert len(pad_like["tokens"]) == 2 and not any(pad_like["weights"] + pad_like["advantages"])
    # importance ratios exp(lp - s) stay <= e for any lp <= 0
    assert all(-1.3 < lp < -0.7 for row in rows[1:] for lp in row["logprobs"])


def test_text_rows_are_windows_of_the_repeated_stream(parity):
    stream = list(range(100, 140))  # a 40-token "passage"
    lengths = parity.datum_lengths(max_cp=2, max_len=128)
    rows = parity.build_rows(lengths, seed=3, stream=stream)
    assert rows == parity.build_rows(lengths, seed=3, stream=stream)
    for n, row in zip(lengths, rows, strict=True):
        tokens = row["tokens"]
        assert len(tokens) == n + 1 and set(tokens) <= set(stream)
        start = stream.index(tokens[0])
        assert tokens == [stream[(start + i) % len(stream)] for i in range(n + 1)], "the passage, end to end"
    assert any(len(row["tokens"]) > len(stream) for row in rows), "long rows wrap around the passage"


def test_zigzag_layout_marks_chunks_and_boundaries(parity):
    chunks, near = parity.zigzag_layout(15, cp=2)  # 16 tokens -> 4 chunks of 4 rows
    assert chunks.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    chunks, near = parity.zigzag_layout(127, cp=2)  # 128 tokens -> chunks of 32
    assert chunks[31] == 0 and chunks[32] == 1 and chunks[-1] == 3
    assert near[24:40].all() and not near[:24].any() and not near[40:56].any()


LONE = [399, 799]  # small stand-ins for the 3999/7999-token lone datums: CP2/TP1 splits them at rows 256 and 512


def _dump(parity, *, cp=1, shift=None, grad_scale=1.0, loss_shift=0.0, lone_shift=None, after_shift=None):
    lengths = parity.datum_lengths(max_cp=2, max_len=1024)

    def vectors(offset, lens=lengths, shift=shift):
        out = []
        for n in lens:
            lp = -np.abs(np.linspace(0.1, 3.0, n)) - offset
            if shift is not None:
                lp = lp + shift(n)
            out.append(lp.tolist())
        return out

    def outputs(offset, loss=0.0, lens=lengths, shift=shift):
        logprobs = vectors(offset, lens, shift)
        return {"logprobs": logprobs, "losses": [loss + 0.01 * i for i in range(len(logprobs))], "metrics": {}}

    steps = [
        {
            "loss_fn": loss_fn,
            "forward_backward": outputs(0.01 * i, loss=1.0 + loss_shift),
            "optim_metrics": {},
            "grad_norm": (1.0 + i) * grad_scale,
        }
        for i, loss_fn in enumerate(parity.LOSSES)
    ]
    steps[0]["forward_backward_single"] = outputs(0.0, lens=[lengths[2]])
    steps[0]["forward_backward_lone"] = outputs(0.0, lens=LONE, shift=lone_shift)
    return {
        "meta": {"cp": cp, "tp": 1, "pad_multiplier": 128, "label": f"cp{cp}"},
        "lengths": lengths,
        "single_length": lengths[2],
        "lone_lengths": LONE,
        "forward_before": outputs(0.0),
        "forward_before_permuted": outputs(0.0),
        "forward_single": {"logprobs": [vectors(0.0)[2]], "losses": [0.0], "metrics": {}},
        "forward_lone": outputs(0.0, lens=LONE, shift=lone_shift),
        "steps": steps,
        "forward_after": outputs(0.05, shift=after_shift if after_shift is not None else shift),
    }


def _compare(parity, tmp_path, ref, cand, cp=2, noise=None):
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    noise_path = None
    if noise is not None:
        noise_path = tmp_path / "noise.json"
        noise_path.write_text(json.dumps(noise))
    args = Namespace(
        ref=str(tmp_path / "ref.json"),
        cand=str(tmp_path / "cand.json"),
        cp=cp,
        json_out=None,
        noise=str(noise_path) if noise_path else None,
        noise_factor=2.0,
    )
    for key, value in parity.DEFAULT_THRESHOLDS.items():
        setattr(args, key, value)
    return parity.compare(args)


def test_identical_dumps_pass(parity, tmp_path, capsys):
    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2)) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "per zigzag chunk" in out and "noise floor ref" in out


def test_small_uniform_noise_passes(parity, tmp_path):
    cand = _dump(parity, cp=2, shift=lambda n: np.full(n, 2e-3))
    assert _compare(parity, tmp_path, _dump(parity), cand) == 0


def test_a_boundary_error_fails_even_with_a_small_mean(parity, tmp_path, capsys):
    def at_boundaries(n):
        delta = np.zeros(n)
        _, near = parity.zigzag_layout(n, cp=2)
        delta[near & (np.arange(n) > n // 2)] = 0.05
        return delta

    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2, shift=at_boundaries)) == 1
    assert "near chunk boundaries" in capsys.readouterr().out


def test_a_doubled_gradient_fails(parity, tmp_path, capsys):
    """The bridge per-token-loss bug scales every gradient by num_microbatches * DP*CP (>= 2x)."""
    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2, grad_scale=2.0)) == 1
    assert "grad_norm ratio 2.0000" in capsys.readouterr().out


def test_a_noise_run_widens_the_limits_to_its_own_deviation(parity, tmp_path, capsys):
    """Deviations as large as a correct alternative layout's (TP2 at CP=1) pass; twice that fails."""
    ref, cand = _dump(parity), _dump(parity, cp=2, shift=lambda n: np.full(n, 0.03), grad_scale=1.1)
    assert _compare(parity, tmp_path, ref, cand) == 1, "over the absolute thresholds without a noise run"
    noise = _dump(parity, shift=lambda n: np.full(n, -0.025), grad_scale=0.92)
    assert _compare(parity, tmp_path, ref, cand, noise=noise) == 0
    assert "noise ratio 0.92000" in capsys.readouterr().out
    far = _dump(parity, cp=2, shift=lambda n: np.full(n, 0.06), grad_scale=1.3)
    assert _compare(parity, tmp_path, ref, far, noise=noise) == 1


def test_a_loss_offset_fails(parity, tmp_path, capsys):
    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2, loss_shift=0.5)) == 1
    assert "per-datum loss off by" in capsys.readouterr().out


def test_different_datum_sets_are_refused(parity, tmp_path):
    cand = _dump(parity, cp=2)
    cand["lengths"] = cand["lengths"][:-1]
    with pytest.raises(AssertionError, match="different datum sets"):
        _compare(parity, tmp_path, _dump(parity), cand)
    old = _dump(parity, cp=2)  # a dump from before the lone long datums
    del old["lone_lengths"]
    with pytest.raises(AssertionError, match="different datum sets"):
        _compare(parity, tmp_path, _dump(parity), old)


# -------- GatedDeltaNet chunkwise splits and short rows --------


@pytest.mark.parametrize(
    "n, cp, tp, pad, rows",
    [
        (7999, 2, 1, 128, [4096]),  # 8000 tokens: zigzag boundaries at 2000/4000/6000, the GDN split at 4096
        (3999, 2, 1, 128, [2048]),
        (7999, 4, 1, 128, [2048, 4096, 6144]),
        (7999, 2, 4, 128, [4096]),  # 4000 per rank, padded to 512
        (3999, 4, 4, 128, [1024, 2048, 3072]),
        (10, 2, 1, 128, []),  # a short lone datum sits inside rank 0's part: no handoff to check
    ],
)
def test_gdn_split_rows_follow_the_padded_contiguous_split(parity, n, cp, tp, pad, rows):
    assert parity.gdn_split_rows(n, cp, tp, pad) == rows


def test_lone_rows_put_their_weight_on_the_second_half(parity):
    lengths, rows = parity.lone_rows(max_len=8192, seed=2)
    assert lengths == [3999, 7999]
    for n, row in zip(lengths, rows, strict=True):
        assert not any(row["weights"][: n // 2]) and sum(row["weights"]) == pytest.approx(1.0)
    assert parity.lone_rows(max_len=4096, seed=2)[0] == [3999]


def _after_split(delta, split=None):
    """A handoff error: ``delta`` on the 3 rows after each CP2/TP1 GatedDeltaNet split of a lone datum."""

    def shift(n):
        out = np.zeros(n)
        for row in [split] if split is not None else [256 if n == 399 else 512]:
            out[row : row + 3] = delta
        return out

    return shift


def test_an_error_at_the_gdn_split_fails(parity, tmp_path, capsys):
    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2, lone_shift=_after_split(0.1))) == 1
    assert "near the GatedDeltaNet CP splits" in capsys.readouterr().out
    # the same error away from the split and from the zigzag boundaries (100-row and 200-row chunks)
    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2, lone_shift=_after_split(0.1, 150))) == 0


def _with_background(shift, level=2e-3):
    """``shift`` on top of a uniform ``level`` delta everywhere (an interior to compare the windows with)."""
    return lambda n: shift(n) + level


def test_a_noise_run_widens_the_gdn_windows(parity, tmp_path):
    """A flip on an ambiguous repeat can land in a split window under any layout; the noise run measures the
    window/interior ratio a correct layout shows there."""
    ref = _dump(parity)
    cand = _dump(parity, cp=2, lone_shift=_with_background(_after_split(0.1)))
    assert _compare(parity, tmp_path, ref, cand) == 1
    noise = _dump(parity, lone_shift=_with_background(_after_split(-0.08)))
    assert _compare(parity, tmp_path, ref, cand, noise=noise) == 0
    quiet_noise = _dump(parity, lone_shift=_with_background(lambda n: np.zeros(n)))
    assert _compare(parity, tmp_path, ref, cand, noise=quiet_noise) == 1


def test_a_noise_run_widens_the_zigzag_windows_by_its_own_ratio(parity, tmp_path, capsys):
    """Windows near the start of short rows differ more under any layout: a correct layout without boundaries
    (TP2) shows the window/interior ratio a candidate may reach (x1.5), not an absolute deviation."""

    def at_boundaries(delta):
        def shift(n):
            out = np.full(n, 2e-3)
            _, near = parity.zigzag_layout(n, cp=2)
            out[near] += delta
            return out

        return shift

    ref = _dump(parity)
    cand = _dump(parity, cp=2, shift=at_boundaries(0.008))  # window/interior ratio 5: over the default 4
    assert _compare(parity, tmp_path, ref, cand) == 1
    noise = _dump(parity, shift=at_boundaries(0.006))  # a correct layout at ratio 4
    assert _compare(parity, tmp_path, ref, cand, noise=noise) == 0
    worse = _dump(parity, cp=2, shift=at_boundaries(0.02))  # ratio 11: over 1.5 x the noise run's 4
    assert _compare(parity, tmp_path, ref, worse, noise=noise) == 1
    assert "near chunk boundaries" in capsys.readouterr().out


def test_short_rows_are_gated_one_by_one_before_the_first_step(parity, tmp_path, capsys):
    def one_token_of_tiny_rows(delta):
        """The last token of the 2- and 3-token rows (the lone short datum is the 4-token one)."""

        def shift(n):
            out = np.zeros(n)
            if n <= 2:
                out[-1] = delta
            return out

        return shift

    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2, shift=one_token_of_tiny_rows(0.2))) == 0
    assert _compare(parity, tmp_path, _dump(parity), _dump(parity, cp=2, shift=one_token_of_tiny_rows(0.35))) == 1
    assert "max |d| 0.35 > 0.3" in capsys.readouterr().out
    # the same error after the first optimizer step only: post-update vectors are not gated per datum
    cand = _dump(parity, cp=2, after_shift=one_token_of_tiny_rows(0.35))
    assert _compare(parity, tmp_path, _dump(parity), cand) == 0
    # a noise run with a 0.2 deviation on those rows allows twice that
    noise = _dump(parity, shift=one_token_of_tiny_rows(-0.2))
    cand = _dump(parity, cp=2, shift=one_token_of_tiny_rows(0.35))
    assert _compare(parity, tmp_path, _dump(parity), cand, noise=noise) == 0


def test_the_kl_trim_never_drops_a_short_row(parity):
    long_row = np.zeros(2000)
    ref = [long_row.tolist(), [0.0, 0.0]]
    cand = [long_row.tolist(), [0.0, -1.0]]  # one bad token of a short row: in the top 0.1% of all deltas
    report = parity.vector_report(ref, cand, [2000, 2], cp=2)
    assert report["kl"] > 0 and report["short_max_abs"] == pytest.approx(1.0)
    flipped = parity.vector_report([[0.0] * 2000], [[0.0] * 1999 + [-3.0]], [2000], cp=2)
    assert flipped["kl"] == 0.0 < flipped["kl_all"], "a long row's single flip is trimmed"
