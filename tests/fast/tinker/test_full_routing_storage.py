"""Full capture must survive changed routing and lossless storage sharing."""

import numpy as np
import pytest

from miles.tinker.routing_cache import RoutingCache, RoutingCaptureError


def cache():
    return RoutingCache(num_layers=3, topk=2, num_experts=16, prefix_stride=4)


def test_changed_weights_do_not_reuse_token_matching_routing():
    store = cache()
    prompt = list(range(10))
    old = store.record_full("t", prompt, [10, 11], np.zeros((11, 3, 2), dtype=np.int32))
    current_prompt = prompt + [10, 11, 12]
    current_rows = np.ones((14, 3, 2), dtype=np.int32)
    current = store.record_full("t", current_prompt, [13, 14], current_rows)
    assert store.get(current).parent_id is None
    np.testing.assert_array_equal(store.materialize("t", current, current_prompt + [13, 14]), current_rows)
    np.testing.assert_array_equal(store.materialize("t", old, prompt + [10, 11]), 0)
    assert store.stats()["full_prefix_mismatches"] == 1
    assert store.stats()["verified_prefix_rows"] == 0 and store.stats()["pinned"] == 0


def test_a_verified_multiturn_chain_preserves_every_row_and_owns_only_the_suffix():
    store = cache()
    rng = np.random.default_rng(343)
    full_rows = rng.integers(0, 16, size=(45, 3, 2), dtype=np.int32)
    handles = []
    for length in [16, 32, 46]:
        prompt = list(range(length - 2))
        handle = store.record_full("t", prompt, [length - 2, length - 1], full_rows[: length - 1])
        entry = store.get(handle)
        if handles:
            assert entry.parent_id == handles[-1]
            assert len(entry.rows) == length - 1 - store.get(handles[-1]).num_rows
        assert entry.rows.flags.owndata and entry.rows.base is None
        handles.append(handle)
    assert store.stats()["verified_prefix_rows"] == 15 + 31
    assert store.stats()["prefix_hits"] == 0
    for handle, length in zip(reversed(handles), [46, 32, 16], strict=True):
        np.testing.assert_array_equal(store.materialize("t", handle, list(range(length))), full_rows[: length - 1])
    assert store.stats()["bytes"] == 0 and store.stats()["pinned"] == 0


def test_a_prefix_mismatch_inside_an_ancestor_cannot_be_hidden_by_a_matching_suffix():
    store = cache()
    rows = np.zeros((31, 3, 2), dtype=np.int32)
    store.record_full("t", list(range(14)), [14, 15], rows[:15])
    parent = store.record_full("t", list(range(22)), [22, 23], rows[:23])
    changed = rows.copy()
    changed[5, 1, 1] = 9
    handle = store.record_full("t", list(range(30)), [30, 31], changed)
    assert store.get(parent).parent_id is not None
    assert store.get(handle).parent_id is None
    np.testing.assert_array_equal(store.materialize("t", handle, list(range(32))), changed)


def test_stale_handle_cannot_resolve_to_a_new_policy_with_identical_tokens():
    store = cache()
    tokens = list(range(16))
    old = store.record_full("t", tokens[:-1], tokens[-1:], np.zeros((15, 3, 2), dtype=np.int32))
    store.materialize("t", old, tokens)
    new = store.record_full("t", tokens[:-1], tokens[-1:], np.ones((15, 3, 2), dtype=np.int32))
    assert store.materialize("t", old, tokens) is None
    np.testing.assert_array_equal(store.materialize("t", new, tokens), 1)


def test_invalid_full_capture_does_not_pin_a_parent():
    store = cache()
    store.record_full("t", list(range(14)), [14, 15], np.zeros((15, 3, 2), dtype=np.int32))
    with pytest.raises(RoutingCaptureError, match="shape"):
        store.record_full("t", list(range(30)), [30, 31], np.zeros((30, 3, 2), dtype=np.int32))
    assert store.stats()["pinned"] == 0 and store.stats()["entries"] == 1
