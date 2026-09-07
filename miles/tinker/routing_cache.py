"""Gateway-side store of the routing (expert top-k) SGLang used while sampling.

Rollout routing replay (R3) needs, for every trained datum, the experts the
engine routed each input token to. Tinker's SDKs cannot carry 64k x 40 x 8
integers per datum through the sample response, so the gateway keeps the
routing here and hands the client an opaque ``routing_id`` per sequence; the
datum comes back with ``loss_fn_inputs["routing_handle"]`` and the unit
builder materializes the rows (runtime.py). stdlib + numpy only.

Storage model
- An entry holds the tokens of one sampled sequence (prompt + generated, S)
  and the routing rows the engine returned for it: row i is the routing used
  when token i was the input (it produced token i+1), so a sequence has S-1
  rows. Rows are uint8 [rows, L, K] (num_experts <= 256).
- Multi-turn requests extend an earlier sequence: the new prompt shares a
  prefix with a cached entry (the parent). Rows [0, P) of the parent are
  causally valid for the new request (row i depends only on tokens[:i+1]),
  exactly like SGLang's own radix-cached prefix, so the engine is asked only
  for rows [P, S-1) (``routed_experts_start_len``) and the entry stores that
  delta with ``parent_id``/``parent_rows``. Materialization walks the chain.
- Parents are found through a prefix-hash index: every entry records the
  blake2b digest of ``tokens[:c]`` at ``c in {stride, 2*stride, ...} | {S}``;
  a prompt is hashed at the same cut points and the longest cut with live
  candidates is refined with an exact longest-common-prefix compare.
- Lifetime: a parent found for an in-flight request is pinned
  (``pending_count``) until ``record``/``unpin``; entries with children stay
  until the children are gone; a materialized (trained) leaf is released
  at once; ``sweep`` applies the TTL and the byte cap (LRU by ``last_used_at``
  over unpinned, childless entries). Pinned entries are never evicted.
"""

import hashlib
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_BYTES = 137_438_953_472  # 128 GiB
DEFAULT_TTL_S = 21_600.0
DEFAULT_PREFIX_STRIDE = 256
PREFIX_CANDIDATES = 8
ROUTING_ID_BITS = 53  # a JSON-safe integer for the SDK's JS-compatible clients
UNCOVERED = -1  # trainer: rows entirely -1 route through the live router


class RoutingCaptureError(ValueError):
    """The engine's routing payload does not fit the request (shape or expert range)."""


@dataclass
class RoutingEntry:
    routing_id: int
    tenant: str
    tokens: np.ndarray  # int32 [S]
    rows: np.ndarray  # uint8 [S-1-parent_rows, L, K]
    parent_id: int | None
    parent_rows: int
    created_at: float
    last_used_at: float
    child_count: int = 0
    pending_count: int = 0
    consumed: bool = False
    cut_keys: list[tuple] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return self.parent_rows + int(self.rows.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.rows.nbytes + self.tokens.nbytes)


class RoutingCache:
    def __init__(
        self,
        *,
        num_layers: int,
        topk: int,
        num_experts: int,
        max_bytes: int = DEFAULT_MAX_BYTES,
        ttl_s: float = DEFAULT_TTL_S,
        prefix_stride: int = DEFAULT_PREFIX_STRIDE,
        clock=time.monotonic,
    ) -> None:
        if not (0 < num_experts <= 256):
            raise ValueError(f"num_experts={num_experts} must be in [1, 256] for uint8 routing storage")
        if prefix_stride < 1:
            raise ValueError("prefix_stride must be positive")
        self.num_layers = int(num_layers)
        self.topk = int(topk)
        self.num_experts = int(num_experts)
        self.max_bytes = int(max_bytes)
        self.ttl_s = float(ttl_s)
        self.prefix_stride = int(prefix_stride)
        self._clock = clock
        self._lock = threading.RLock()  # record/lookup run on the loop, materialize in an executor
        self._entries: dict[int, RoutingEntry] = {}
        self._by_prefix: dict[tuple, deque] = {}
        self._bytes = 0
        self._evictions = 0
        self._releases = 0
        self._capture_rejected = 0
        self._prefix_hits = 0  # find_prefix_parent() lookups that reused a cached parent (P > 0)
        # unique per process and across restarts (a stale handle never resolves to a fresh entry)
        self._next_id = (secrets.randbits(ROUTING_ID_BITS - 9) << 8) + 1

    # -------- lookup --------

    def find_prefix_parent(self, tenant: str, prompt_tokens) -> tuple[RoutingEntry, int] | None:
        """The cached sequence sharing the longest prefix with ``prompt_tokens``
        and the number of its rows the new request may reuse (P). Pins the
        parent until record()/unpin()."""
        prompt = _as_tokens(prompt_tokens)
        if prompt.size == 0:
            return None
        cuts = _prefix_digests(prompt, self.prefix_stride)
        with self._lock:
            for cut_len, digest in reversed(cuts):
                candidates = self._live_candidates((tenant, cut_len, digest))
                if not candidates:
                    continue
                # longest common prefix wins; among equals the shortest entry (shallowest chain, so
                # leaves stay childless and releasable), then the newest (deque order)
                parent, lcp = max(
                    ((entry, _lcp(entry.tokens, prompt)) for entry in candidates),
                    key=lambda pair: (pair[1], -pair[0].num_rows),
                )
                reuse = min(lcp, parent.num_rows)
                if reuse <= 0:
                    return None
                self.pin(parent)
                self._prefix_hits += 1
                return parent, reuse
        return None

    def pin(self, entry: RoutingEntry) -> None:
        with self._lock:
            entry.pending_count += 1
            entry.last_used_at = self._clock()

    def unpin(self, entry: RoutingEntry) -> None:
        """Failure-path counterpart of the pin taken by find_prefix_parent."""
        with self._lock:
            entry.pending_count = max(0, entry.pending_count - 1)

    # -------- record --------

    def record(
        self,
        tenant: str,
        prompt_tokens,
        generated_tokens,
        delta_rows: np.ndarray,
        parent: RoutingEntry | None = None,
        parent_rows: int = 0,
    ) -> int:
        """Store one sampled sequence's routing; returns its routing_id.
        ``delta_rows`` covers rows [parent_rows, S-1); int32 or uint8."""
        tokens = np.concatenate([_as_tokens(prompt_tokens), _as_tokens(generated_tokens)])
        parent_rows = int(parent_rows) if parent is not None else 0
        rows = self._validated_rows(delta_rows, expected_rows=int(tokens.size) - 1 - parent_rows, parent=parent)
        now = self._clock()
        with self._lock:
            routing_id = self._mint_id()
            entry = RoutingEntry(
                routing_id=routing_id,
                tenant=tenant,
                tokens=tokens,
                rows=rows,
                parent_id=parent.routing_id if parent is not None else None,
                parent_rows=parent_rows,
                created_at=now,
                last_used_at=now,
            )
            if parent is not None:
                parent.child_count += 1
                parent.pending_count = max(0, parent.pending_count - 1)
                parent.last_used_at = now
            self._index(entry)
            self._entries[routing_id] = entry
            self._bytes += entry.nbytes
            if self._bytes > self.max_bytes:
                self._sweep_locked(now)
        return routing_id

    def note_capture_rejected(self) -> None:
        """A capture rejected before record() (decode-side validation)."""
        with self._lock:
            self._capture_rejected += 1

    # -------- materialize --------

    def materialize(self, tenant: str, handle: int | None, tokens) -> np.ndarray | None:
        """int32 [S-1, L, K] for a datum, or None when nothing cached matches.
        Lookup: the handle (with exact token equality), then the exact
        full-token hash. The matched leaf is consumed and released."""
        tokens = _as_tokens(tokens)
        with self._lock:
            entry = self._resolve(tenant, handle, tokens)
            if entry is None:
                return None
            chunks = self._chain_chunks(entry)
            if chunks is None:
                return None
            entry.consumed = True
            entry.last_used_at = self._clock()
            self._release_consumed(entry)
        # the concatenation runs outside the lock: chunks are never mutated after record()
        if not chunks:
            return np.empty((0, self.num_layers, self.topk), dtype=np.int32)
        return np.concatenate(chunks).astype(np.int32)

    # -------- eviction / stats --------

    def sweep(self) -> dict:
        with self._lock:
            self._sweep_locked(self._clock())
            return self.stats()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "evictions": self._evictions,
                "releases": self._releases,
                "pinned": sum(1 for e in self._entries.values() if e.pending_count > 0),
                "capture_rejected": self._capture_rejected,
                "prefix_hits": self._prefix_hits,
            }

    def get(self, routing_id: int) -> RoutingEntry | None:
        with self._lock:
            return self._entries.get(routing_id)

    def __len__(self) -> int:
        return len(self._entries)

    # -------- internals --------

    def _mint_id(self) -> int:
        routing_id = self._next_id
        self._next_id += 1
        assert 0 < routing_id < 2**ROUTING_ID_BITS
        return routing_id

    def _validated_rows(self, delta_rows, *, expected_rows: int, parent: RoutingEntry | None) -> np.ndarray:
        rows = np.asarray(delta_rows)
        expected = (expected_rows, self.num_layers, self.topk)
        try:
            if parent is not None and parent.routing_id not in self._entries:
                raise RoutingCaptureError(f"parent {parent.routing_id} is no longer cached")
            if expected_rows < 0 or rows.shape != expected:
                raise RoutingCaptureError(f"routing rows shape {tuple(rows.shape)} != expected {expected}")
            if rows.dtype != np.uint8:
                if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.num_experts):
                    raise RoutingCaptureError(
                        f"expert ids outside [0, {self.num_experts}): min {int(rows.min())} max {int(rows.max())}"
                    )
                rows = rows.astype(np.uint8)
            elif rows.size and self.num_experts < 256 and int(rows.max()) >= self.num_experts:
                raise RoutingCaptureError(f"expert id {int(rows.max())} >= num_experts {self.num_experts}")
        except RoutingCaptureError:
            # the parent's pin stays: the caller's failure path calls unpin()
            with self._lock:
                self._capture_rejected += 1
            raise
        return np.ascontiguousarray(rows)

    def _index(self, entry: RoutingEntry) -> None:
        for cut_len, digest in _prefix_digests(entry.tokens, self.prefix_stride):
            key = (entry.tenant, cut_len, digest)
            self._by_prefix.setdefault(key, deque(maxlen=PREFIX_CANDIDATES)).appendleft(entry.routing_id)
            entry.cut_keys.append(key)

    def _unindex(self, entry: RoutingEntry) -> None:
        for key in entry.cut_keys:
            candidates = self._by_prefix.get(key)
            if candidates is None:
                continue
            try:
                candidates.remove(entry.routing_id)
            except ValueError:
                pass  # pushed out by newer entries already
            if not candidates:
                del self._by_prefix[key]

    def _live_candidates(self, key: tuple) -> list[RoutingEntry]:
        ids = self._by_prefix.get(key)
        if not ids:
            return []
        return [self._entries[i] for i in ids if i in self._entries]

    def _resolve(self, tenant: str, handle: int | None, tokens: np.ndarray) -> RoutingEntry | None:
        if handle is not None:
            entry = self._entries.get(int(handle))
            if entry is not None and entry.tenant == tenant and np.array_equal(entry.tokens, tokens):
                return entry
        if tokens.size == 0:
            return None
        cut_len, digest = _prefix_digests(tokens, self.prefix_stride)[-1]
        for entry in self._live_candidates((tenant, cut_len, digest)):
            if entry.tokens.size == tokens.size and np.array_equal(entry.tokens, tokens):
                return entry
        return None

    def _chain_chunks(self, entry: RoutingEntry) -> list[np.ndarray] | None:
        """rows(entry) = rows(parent)[:parent_rows] ++ entry.rows, as chunks in order."""
        chunks: list[np.ndarray] = []
        need = entry.num_rows
        node: RoutingEntry | None = entry
        while need > 0:
            if node is None:
                return None  # broken chain (parent gone): treat as uncovered
            if need > node.parent_rows:
                chunks.append(node.rows[: need - node.parent_rows])
                need = node.parent_rows
            node = self._entries.get(node.parent_id) if node.parent_id is not None else None
        chunks.reverse()
        return chunks

    def _release_consumed(self, entry: RoutingEntry) -> None:
        """Drop a consumed entry that nothing references; ancestors follow when
        they are consumed and lose their last child."""
        node: RoutingEntry | None = entry
        while node is not None and node.consumed and node.child_count == 0 and node.pending_count == 0:
            parent = self._drop(node)
            self._releases += 1
            node = parent

    def _drop(self, entry: RoutingEntry) -> RoutingEntry | None:
        """Remove an entry; returns its parent (child_count decremented)."""
        self._entries.pop(entry.routing_id, None)
        self._unindex(entry)
        self._bytes -= entry.nbytes
        parent = self._entries.get(entry.parent_id) if entry.parent_id is not None else None
        if parent is not None:
            parent.child_count = max(0, parent.child_count - 1)
        return parent

    def _evictable(self, entry: RoutingEntry) -> bool:
        return entry.pending_count == 0 and entry.child_count == 0

    def _sweep_locked(self, now: float) -> None:
        # TTL: expired leaves go first; a parent follows once its last child is gone
        while True:
            expired = [e for e in self._entries.values() if self._evictable(e) and now - e.created_at >= self.ttl_s]
            if not expired:
                break
            for entry in expired:
                self._drop(entry)
                self._evictions += 1
        # byte cap: LRU over unpinned childless entries, cascading to freed parents
        while self._bytes > self.max_bytes:
            victims = sorted((e for e in self._entries.values() if self._evictable(e)), key=lambda e: e.last_used_at)
            if not victims:
                break
            for entry in victims:
                self._drop(entry)
                self._evictions += 1
                if self._bytes <= self.max_bytes:
                    break


def _as_tokens(tokens) -> np.ndarray:
    array = np.asarray(tokens, dtype=np.int32)
    return np.ascontiguousarray(array.reshape(-1))


def _prefix_digests(tokens: np.ndarray, stride: int) -> list[tuple[int, bytes]]:
    """(cut_len, digest) for cut lengths stride, 2*stride, ... and S, ascending,
    from one incremental blake2b pass over the int32 bytes."""
    size = int(tokens.size)
    if size == 0:
        return []
    raw = tokens.astype("<i4", copy=False).tobytes()
    cut_lens = list(range(stride, size + 1, stride))
    if not cut_lens or cut_lens[-1] != size:
        cut_lens.append(size)
    hasher = hashlib.blake2b(digest_size=16)
    digests = []
    fed = 0
    for cut_len in cut_lens:
        hasher.update(raw[fed * 4 : cut_len * 4])
        fed = cut_len
        digests.append((cut_len, hasher.copy().digest()))
    return digests


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    bound = min(a.size, b.size)
    if bound == 0:
        return 0
    mismatch = np.flatnonzero(a[:bound] != b[:bound])
    return int(mismatch[0]) if mismatch.size else bound
