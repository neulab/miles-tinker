"""RoutingCache: the gateway-side store of sampled routing for R3 (docs/fork/r3.md).
Entries chain through prefix parents, rows are uint8 in and int32 out, and the
lifetime rules (pins, consumed leaves, TTL, byte cap) keep the cache bounded."""

import numpy as np
import pytest

from miles.tinker.routing_cache import RoutingCache, RoutingCaptureError, _cut_lengths, _lcp, _prefix_digests

L, K, E = 3, 2, 16


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _cache(**overrides) -> tuple[RoutingCache, Clock]:
    clock = Clock()
    params = dict(num_layers=L, topk=K, num_experts=E, max_bytes=1 << 40, ttl_s=3600.0, prefix_stride=4, clock=clock)
    return RoutingCache(**{**params, **overrides}), clock


def _rows(num_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, E, size=(num_rows, L, K), dtype=np.int32)


def _record(cache: RoutingCache, prompt: list[int], generated: list[int], seed: int, tenant: str = "t"):
    """Sample one turn the way runtime.py does: parent lookup, delta rows, record."""
    found = cache.find_prefix_parent(tenant, prompt)
    parent, start = found if found is not None else (None, 0)
    delta = _rows(len(prompt) + len(generated) - 1 - start, seed)
    routing_id = cache.record(tenant, prompt, generated, delta, parent, start)
    return routing_id, delta, start


def test_record_materialize_round_trip_is_int32_of_shape_s_minus_1():
    cache, _ = _cache()
    prompt, generated = [1, 2, 3], [4, 5]
    routing_id, delta, _ = _record(cache, prompt, generated, seed=0)
    assert 0 < routing_id < 2**53
    assert cache.get(routing_id).rows.dtype == np.uint8

    rows = cache.materialize("t", routing_id, prompt + generated)
    assert rows.dtype == np.int32 and rows.shape == (4, L, K)
    np.testing.assert_array_equal(rows, delta)


def test_an_exact_prefix_chain_of_three_turns_reuses_every_parent_row():
    cache, _ = _cache()
    t1 = list(range(10))
    id1, d1, s1 = _record(cache, t1, [10, 11], seed=1)
    assert s1 == 0
    t2 = t1 + [10, 11] + [20, 21, 22]  # turn 2 extends turn 1's full sequence
    id2, d2, s2 = _record(cache, t2, [30], seed=2)
    assert s2 == cache.get(id1).num_rows == 11, "P = parent.num_rows for an exact extension"
    assert d2.shape == (len(t2) + 1 - 1 - 11, L, K)
    t3 = t2 + [30] + [40]
    id3, d3, s3 = _record(cache, t3, [50, 51], seed=3)
    assert s3 == cache.get(id2).num_rows

    rows = cache.materialize("t", id3, t3 + [50, 51])
    expected = np.concatenate([d1, d2, d3])
    assert rows.shape == (len(t3) + 2 - 1, L, K)
    np.testing.assert_array_equal(rows, expected)


def test_a_retokenized_tail_gets_an_lcp_parent():
    cache, _ = _cache()
    t1 = list(range(20))
    id1, d1, _ = _record(cache, t1, [20, 21, 22, 23, 24], seed=1)
    full1 = t1 + [20, 21, 22, 23, 24]
    prompt2 = full1[:-5] + [90, 91, 92, 93, 94, 95, 96]  # diverges 5 tokens before the end of entry 1
    found = cache.find_prefix_parent("t", prompt2)
    assert found is not None
    parent, start = found
    assert parent.routing_id == id1 and start == len(full1) - 5, "P = LCP"
    delta = _rows(len(prompt2) + 1 - 1 - start, seed=2)
    id2 = cache.record("t", prompt2, [7], delta, parent, start)

    rows = cache.materialize("t", id2, prompt2 + [7])
    np.testing.assert_array_equal(rows[:start], d1[:start])
    np.testing.assert_array_equal(rows[start:], delta)


def test_a_handle_with_other_tokens_does_not_bind_to_another_capture():
    cache, _ = _cache()
    id_a, d_a, _ = _record(cache, [1, 2, 3, 4, 5], [6], seed=1)
    id_b, d_b, _ = _record(cache, [7, 8, 9], [10, 11], seed=2)
    assert cache.materialize("t", id_a, [7, 8, 9, 10, 11]) is None
    rows = cache.materialize("t", id_b, [7, 8, 9, 10, 11])
    np.testing.assert_array_equal(rows, d_b)
    rows = cache.materialize("t", None, [1, 2, 3, 4, 5, 6])  # no handle at all
    np.testing.assert_array_equal(rows, d_a)


def test_unknown_handles_and_tokens_materialize_to_none():
    cache, _ = _cache()
    _record(cache, [1, 2, 3], [4], seed=1)
    assert cache.materialize("t", 12345, [1, 2, 3, 4]) is None
    assert cache.materialize("t", None, [1, 2, 3, 4]) is not None, "plain clients retain token lookup"
    assert cache.materialize("t", 12345, [9, 9, 9]) is None
    assert cache.materialize("t", None, [1, 2, 3, 5]) is None


def test_the_range_check_rejects_out_of_range_experts_and_counts_them():
    cache, _ = _cache()
    bad = _rows(3, seed=1)
    bad[0, 0, 0] = E
    with pytest.raises(RoutingCaptureError, match="outside"):
        cache.record("t", [1, 2], [3, 4], bad)
    bad[0, 0, 0] = -1
    with pytest.raises(RoutingCaptureError):
        cache.record("t", [1, 2], [3, 4], bad)
    with pytest.raises(RoutingCaptureError, match="shape"):
        cache.record("t", [1, 2], [3, 4], _rows(2, seed=1))
    assert cache.stats()["capture_rejected"] == 3 and cache.stats()["entries"] == 0


def test_a_failed_record_leaves_the_parent_pin_for_the_callers_unpin():
    cache, _ = _cache()
    id1, _, _ = _record(cache, [1, 2, 3, 4], [5], seed=1)
    parent, start = cache.find_prefix_parent("t", [1, 2, 3, 4, 5, 6])
    assert parent.pending_count == 1
    with pytest.raises(RoutingCaptureError):
        cache.record("t", [1, 2, 3, 4, 5, 6], [7], _rows(1, seed=1), parent, start)
    assert parent.pending_count == 1
    cache.unpin(parent)
    assert parent.pending_count == 0


def test_byte_cap_evicts_unpinned_leaves_before_parents_and_never_a_pinned_parent():
    row_bytes = L * K  # uint8
    cache, clock = _cache(max_bytes=4_000)
    root_prompt = list(range(100))
    id_root, _, _ = _record(cache, root_prompt, [100, 101], seed=1)  # ~ 101 rows
    full_root = root_prompt + [100, 101]
    leaves = []
    for i in range(6):
        clock.now += 1
        leaf_id, _, _ = _record(cache, full_root + [200 + i] * 50, [1, 2], seed=10 + i)  # ~52 delta rows each
        leaves.append(leaf_id)
    stats = cache.stats()  # record() sweeps as soon as the cap is exceeded
    assert stats["bytes"] <= 4_000 and stats["evictions"] > 0
    assert cache.get(id_root) is not None, "a parent with live children is never evicted"
    assert cache.get(leaves[0]) is None and cache.get(leaves[-1]) is not None, "LRU order: oldest leaf first"

    # a pinned parent survives pressure between lookup and record
    parent, start = cache.find_prefix_parent("t", full_root + [7] * 300)
    assert parent.routing_id == id_root
    cache.max_bytes = 1
    cache.sweep()
    assert cache.get(id_root) is not None and parent.pending_count == 1
    cache.max_bytes = 1 << 40  # pressure gone: the record resolves against the surviving parent
    delta = _rows(len(full_root) + 300 + 1 - 1 - start, seed=99)
    child = cache.record("t", full_root + [7] * 300, [8], delta, parent, start)
    assert cache.get(child) is not None and parent.pending_count == 0 and parent.child_count >= 1
    assert cache.get(child).nbytes >= 300 * row_bytes


def test_a_consumed_leaf_is_released_and_a_consumed_parent_follows_its_last_child():
    cache, _ = _cache()
    id1, _, _ = _record(cache, [1, 2, 3, 4], [5], seed=1)
    id2, _, _ = _record(cache, [1, 2, 3, 4, 5, 6], [7], seed=2)
    assert cache.get(id1).child_count == 1
    assert cache.materialize("t", id2, [1, 2, 3, 4, 5, 6, 7]) is not None
    assert cache.get(id2) is None, "a trained datum is never extended again"
    assert cache.get(id1) is not None and cache.get(id1).child_count == 0
    assert cache.materialize("t", id1, [1, 2, 3, 4, 5]) is not None
    assert cache.get(id1) is None
    assert cache.stats() == {
        "entries": 0,
        "bytes": 0,
        "evictions": 0,
        "releases": 2,
        "pinned": 0,
        "capture_rejected": 0,
        "prefix_hits": 1,
        "full_captures": 0,
        "full_capture_rows": 0,
        "verified_prefix_rows": 0,
        "full_prefix_mismatches": 0,
    }


def test_a_consumed_parent_waits_for_its_children():
    cache, _ = _cache()
    id1, _, _ = _record(cache, [1, 2, 3, 4], [5], seed=1)
    id2, _, _ = _record(cache, [1, 2, 3, 4, 5, 6], [7], seed=2)
    assert cache.materialize("t", id1, [1, 2, 3, 4, 5]) is not None
    assert cache.get(id1) is not None, "still referenced by its child"
    assert cache.materialize("t", id2, [1, 2, 3, 4, 5, 6, 7]) is not None
    assert cache.get(id1) is None and cache.get(id2) is None


def test_ttl_sweep_expires_leaves_then_their_parents():
    cache, clock = _cache(ttl_s=100.0)
    id1, _, _ = _record(cache, [1, 2, 3, 4], [5], seed=1)
    clock.now += 50
    id2, _, _ = _record(cache, [1, 2, 3, 4, 5, 6], [7], seed=2)
    clock.now += 60  # id1 is past the TTL, id2 is not
    cache.sweep()
    assert cache.get(id1) is not None and cache.get(id2) is not None, "a parent outlives the TTL while a child lives"
    clock.now += 50
    cache.sweep()
    assert cache.get(id1) is None and cache.get(id2) is None
    assert cache.stats()["evictions"] == 2


def test_pinned_entries_are_never_ttl_evicted():
    cache, clock = _cache(ttl_s=10.0)
    id1, _, _ = _record(cache, [1, 2, 3, 4], [5], seed=1)
    parent, _ = cache.find_prefix_parent("t", [1, 2, 3, 4, 5, 6])
    clock.now += 100
    cache.sweep()
    assert cache.get(id1) is not None and cache.stats()["pinned"] == 1
    cache.unpin(parent)
    cache.sweep()
    assert cache.get(id1) is None


def test_tenants_are_isolated():
    cache, _ = _cache()
    id_a, d_a, _ = _record(cache, [1, 2, 3, 4], [5], seed=1, tenant="a")
    assert cache.find_prefix_parent("b", [1, 2, 3, 4, 5, 6]) is None
    assert cache.materialize("b", id_a, [1, 2, 3, 4, 5]) is None
    assert cache.materialize("b", None, [1, 2, 3, 4, 5]) is None
    np.testing.assert_array_equal(cache.materialize("a", id_a, [1, 2, 3, 4, 5]), d_a)


def test_the_stride_index_probes_one_key_per_cut_and_bounds_candidates():
    cache, _ = _cache(prefix_stride=4)
    system_prompt = list(range(40))
    ids = []
    for i in range(200):  # 200 entries sharing one system prompt
        rid, _, _ = _record(cache, system_prompt + [1000 + i, 1001 + i], [3], seed=i)
        ids.append(rid)
    # every cut key of the shared prefix holds at most 8 (the newest) candidates
    for cut_len in range(4, 41, 4):
        key = ("t", cut_len, _prefix_digests(np.asarray(system_prompt, dtype=np.int32), 4)[cut_len // 4 - 1][1])
        assert len(cache._by_prefix[key]) <= 8
    found = cache.find_prefix_parent("t", system_prompt + [5000, 5001, 5002, 5003])
    assert found is not None and found[1] == 40, "the shared prefix rows are reused from one of the newest entries"
    assert found[0].routing_id in ids[-8:]


def test_prefix_digests_cover_stride_multiples_and_the_full_length():
    tokens = np.arange(10, dtype=np.int32)
    cuts = [cut for cut, _ in _prefix_digests(tokens, 4)]
    assert cuts == [4, 8, 10]
    assert [cut for cut, _ in _prefix_digests(np.arange(8, dtype=np.int32), 4)] == [4, 8]
    full = dict(_prefix_digests(tokens, 4))
    assert full[8] == _prefix_digests(tokens[:8], 4)[-1][1], "digests are of tokens[:c], independent of the tail"
    assert _lcp(np.array([1, 2, 3]), np.array([1, 2, 4, 5])) == 2
    assert _lcp(np.array([1, 2]), np.array([1, 2, 4])) == 2


def test_routing_ids_are_unique_and_json_safe():
    cache, _ = _cache()
    ids = {_record(cache, [1, 2, i], [4], seed=i)[0] for i in range(50)}
    assert len(ids) == 50 and all(0 < i < 2**53 for i in ids)


def test_a_short_turn_finds_its_parent_under_the_production_stride():
    """The 09080157 smoke never logged a prefix-parent hit: with
    --tinker-routing-prefix-stride 256 a 72-token turn was indexed only at cut 72 and its
    93-token continuation probed only at cut 93, so the two never met. The sub-stride
    rungs of the cut ladder make every extension of a >= MIN_CUT-token turn findable."""
    cache, _ = _cache(prefix_stride=256)
    turn1_prompt, turn1_generated = list(range(24)), list(range(1000, 1048))  # S = 72
    rid, _, _ = _record(cache, turn1_prompt, turn1_generated, seed=1)

    turn2_prompt = turn1_prompt + turn1_generated + list(range(2000, 2021))  # 93 tokens
    found = cache.find_prefix_parent("t", turn2_prompt)

    assert found is not None, "a 72-token parent must be found for its 93-token extension"
    parent, reuse = found
    assert parent.routing_id == rid
    assert reuse == 71, "every row of the parent (S - 1) is causally valid for the extension"
    assert cache.stats()["prefix_hits"] == 1


def test_the_cut_ladder_doubles_below_the_stride_and_strides_above_it():
    assert _cut_lengths(72, 256) == [8, 16, 32, 64, 72]
    assert _cut_lengths(93, 256) == [8, 16, 32, 64, 93]
    assert _cut_lengths(600, 256) == [8, 16, 32, 64, 128, 256, 512, 600]
    assert _cut_lengths(512, 256) == [8, 16, 32, 64, 128, 256, 512]
    assert _cut_lengths(10, 4) == [4, 8, 10], "a stride under MIN_CUT keeps the plain stride ladder"
    assert _cut_lengths(5, 256) == [5], "shorter than MIN_CUT: only the exact cut"
