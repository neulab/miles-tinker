"""CPU unit test for the Qwen3-VL CP+THD packed mRoPE reconstruction (issue #1296).

Under context parallelism each rank's THD row holds only its zigzag chunks of every packed
segment. `_reassemble_full_row` de-interleaves the all-gathered per-rank rows back to the
full natural-order row so per-segment MRoPE positions can be rebuilt and re-sliced. This
test checks that reconstruction is the exact inverse of `slice_with_cp` (the function miles
uses to shard the tokens), and that re-slicing with `_natural_to_zigzag_slice` round-trips.
"""

import pytest
import torch
import torch.nn.functional as F

from miles_plugins.models.qwen3_vl import _natural_to_zigzag_slice, _reassemble_full_row


def _slice_with_cp(tokens, cp_size, cp_rank, pad_value=0):
    """Reference copy of cp_utils.slice_with_cp's THD zigzag slicing (per sample)."""
    token_len = len(tokens)
    chunk = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk - token_len
    if pad:
        tokens = F.pad(tokens, (0, pad), value=pad_value)
    s1, e1 = chunk * cp_rank, chunk * (cp_rank + 1)
    s2, e2 = chunk * (2 * cp_size - cp_rank - 1), chunk * (2 * cp_size - cp_rank)
    return torch.cat([tokens[s1:e1], tokens[s2:e2]])


def _build_like_get_batch(sample_lens, cp_size, pad_size=8):
    """Mimic miles get_batch THD+CP packing: per-sample zigzag slice, concat, pad, cu*cp."""
    samples = []
    base = 1
    for L in sample_lens:
        samples.append(torch.arange(base, base + L))  # unique nonzero ids
        base += L
    per_rank = []
    for r in range(cp_size):
        row = torch.cat([_slice_with_cp(t, cp_size, r) for t in samples])
        per_rank.append(row)
    cu = [0]
    for t in samples:
        cu.append(cu[-1] + _slice_with_cp(t, cp_size, 0).size(0))
    final_pad = (pad_size - per_rank[0].size(0) % pad_size) % pad_size
    if final_pad:
        per_rank = [F.pad(row, (0, final_pad), value=0) for row in per_rank]
        cu.append(cu[-1] + final_pad)
    cu = [x * cp_size for x in cu]
    return samples, per_rank, cu


@pytest.mark.parametrize(
    "cp_size,sample_lens",
    [(2, [10, 7, 13]), (2, [16, 16]), (4, [20, 9, 30, 5]), (2, [3]), (4, [40, 17])],
)
def test_reassemble_is_inverse_of_slice_with_cp(cp_size, sample_lens):
    samples, per_rank, cu = _build_like_get_batch(sample_lens, cp_size)
    local_len = per_rank[0].size(0)
    assert cu[-1] == cp_size * local_len

    full = _reassemble_full_row(per_rank, cu, cp_size)
    assert full is not None and full.numel() == cu[-1]

    # Each real sample's tokens reappear (in order) at the start of its segment.
    for i, t in enumerate(samples):
        seg = full[cu[i] : cu[i + 1]]
        assert torch.equal(seg[: t.numel()], t)

    # Re-slicing the full row per segment recovers exactly each rank's local chunks.
    for r in range(cp_size):
        recon = []
        for i in range(len(cu) - 1):
            recon.append(_natural_to_zigzag_slice(full[cu[i] : cu[i + 1]], cp_size, r, dim=0))
        assert torch.equal(torch.cat(recon), per_rank[r])


def test_reassemble_bails_on_indivisible_segment():
    # A segment length not divisible by 2*cp -> None (caller falls back to dense path).
    cu = [0, 6]  # 6 not divisible by 2*cp=4
    gathered = [torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    assert _reassemble_full_row(gathered, cu, cp_size=2) is None


# ---------------------------------------------------------------------------
# The patched forward on a pre-sharded CP row: positions must be built and used
# ---------------------------------------------------------------------------

_BRIDGE_MODEL = "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"


class _Psp:
    qkv_format = "thd"

    def __init__(self, cu):
        self.cu_seqlens_q = torch.tensor(cu, dtype=torch.int32)


@pytest.fixture
def patched_bridge(monkeypatch):
    """A fake bridge model module, patched by miles: Qwen3VLModel.forward asks get_rope_index for positions
    (as megatron-bridge 0.5.0 does) unless the model skips it, and returns what it got."""
    import sys
    import types

    import miles_plugins.models.qwen3_vl as plugin

    module = types.ModuleType(_BRIDGE_MODEL)
    module.get_rope_index = lambda *args, **kwargs: ("dense", None)

    class Qwen3VLModel:
        _cp_local_vision_embed_indices = None
        config = types.SimpleNamespace(spatial_merge_size=2)
        image_token_id, video_token_id, vision_start_token_id = -1, -2, -3

        def __init__(self, asks_for_positions=True):
            self.asks_for_positions = asks_for_positions

        def forward(self, input_ids=None, packed_seq_params=None, **kwargs):
            if not self.asks_for_positions:
                return "no positions"
            positions, _ = module.get_rope_index(2, -1, -2, -3, input_ids)
            return positions

    module.Qwen3VLModel = Qwen3VLModel
    monkeypatch.setitem(sys.modules, _BRIDGE_MODEL, module)
    plugin._patch_model_forward_and_rope_index()
    return plugin, Qwen3VLModel


def _cp_rows(cp_size):
    """Rank 0's row of two packed samples, sliced like get_batch, and the full cu_seqlens."""
    samples, per_rank, cu = _build_like_get_batch([10, 7], cp_size)
    return per_rank, cu


def test_a_pre_sharded_row_gets_rank_local_positions(patched_bridge, monkeypatch):
    plugin, model_cls = patched_bridge
    per_rank, cu = _cp_rows(2)
    monkeypatch.setattr(plugin, "_cp_size_rank", lambda: (2, 0))
    monkeypatch.setattr(plugin, "_cp_allgather_unzigzag", lambda flat, cu, cp: _reassemble_full_row(per_rank, cu, cp))
    positions = model_cls().forward(input_ids=per_rank[0].unsqueeze(0), packed_seq_params=_Psp(cu))
    assert positions.shape == (3, 1, per_rank[0].numel())
    # each segment numbered from 0, then zigzag-sliced like the tokens
    expected = torch.cat(
        [_natural_to_zigzag_slice(torch.arange(e - s), 2, 0, dim=0) for s, e in zip(cu[:-1], cu[1:], strict=True)]
    )
    assert torch.equal(positions[0, 0], expected)


def test_a_pre_sharded_row_without_positions_fails(patched_bridge, monkeypatch):
    plugin, model_cls = patched_bridge
    per_rank, cu = _cp_rows(2)
    monkeypatch.setattr(plugin, "_cp_size_rank", lambda: (2, 0))
    monkeypatch.setattr(plugin, "_cp_allgather_unzigzag", lambda flat, cu, cp: None)
    with pytest.raises(RuntimeError, match="got no rank-local MRoPE positions"):
        model_cls().forward(input_ids=per_rank[0].unsqueeze(0), packed_seq_params=_Psp(cu))


def test_positions_the_bridge_never_asked_for_fail(patched_bridge, monkeypatch):
    """A bridge build that stops calling the module-level get_rope_index would silently use its own positions."""
    plugin, model_cls = patched_bridge
    per_rank, cu = _cp_rows(2)
    monkeypatch.setattr(plugin, "_cp_size_rank", lambda: (2, 0))
    monkeypatch.setattr(plugin, "_cp_allgather_unzigzag", lambda flat, cu, cp: _reassemble_full_row(per_rank, cu, cp))
    with pytest.raises(RuntimeError, match="never asked get_rope_index"):
        model_cls(asks_for_positions=False).forward(input_ids=per_rank[0].unsqueeze(0), packed_seq_params=_Psp(cu))


def test_full_rows_keep_the_unchanged_paths(patched_bridge, monkeypatch):
    plugin, model_cls = patched_bridge
    full = torch.arange(1, 17)
    monkeypatch.setattr(plugin, "_cp_size_rank", lambda: (2, 0))
    # the full packed row at CP2 (the bridge splits it itself): no check, whatever the model does
    model = model_cls(asks_for_positions=False)
    assert model.forward(input_ids=full.unsqueeze(0), packed_seq_params=_Psp([0, 16])) == "no positions"
    monkeypatch.setattr(plugin, "_cp_size_rank", lambda: (1, 0))
    positions = model_cls().forward(input_ids=full.unsqueeze(0), packed_seq_params=_Psp([0, 10, 16]))
    assert torch.equal(positions[0, 0], torch.cat([torch.arange(10), torch.arange(6)]))
