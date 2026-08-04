"""Top-k parity: numpy tie-group-widened selection == the legacy Python sort.

``VectorIndex.search`` used to rank every row with a full Python
``sorted(range(N), key=lambda i: (-score, chunk_id))``. That is N lambda
invocations, each coercing a numpy scalar and building a tuple, and it
dominated ``search()`` end to end. It is replaced by
``np.argpartition`` + a **tie-group-widened** ``np.lexsort`` over the
candidate slice.

The whole risk of that swap lives at the k boundary. ``argpartition`` picks an
ARBITRARY subset of a group of equal scores straddling k, so tie-breaking only
within what it happened to select would silently change which member survives
the cut — while leaving the result *set* usually intact, which is exactly what
makes it invisible to a "top-k set is stable" assertion. So every case here is
built with deliberate exact ties across the boundary and asserts the full
``(chunk_id, score)`` SEQUENCE against the legacy path, which is retained
verbatim behind ``ED4ALL_RETRIEVAL_TOPK_LEGACY``.

Hermetic + CPU-only: fixture matrices are constructed in-process; no course
dir, no embedding client, no weights.
"""

from __future__ import annotations

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from LibV2.tools.libv2.retrieval import vector_index as vi
from LibV2.tools.libv2.retrieval.vector_index import (
    VectorIndex,
    VectorIndexManifest,
    resolve_topk_legacy,
)


# --------------------------------------------------------------------------
# Fixture construction — matrices whose SCORES we control exactly.
# --------------------------------------------------------------------------


def _index_from_scores(scores, chunk_ids=None) -> VectorIndex:
    """A 1-dim index whose ``matrix @ [1.0]`` reproduces ``scores`` exactly.

    Dim 1 keeps the dot product a pure copy, so a constructed tie is a
    bit-exact tie rather than "two floats that happen to be close".
    """
    arr = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    ids = list(chunk_ids) if chunk_ids is not None else [
        f"chunk_{i:04d}" for i in range(arr.shape[0])
    ]
    assert len(ids) == arr.shape[0]
    manifest = VectorIndexManifest(
        schema_version="1.0",
        embedding_provider="fake",
        embedding_kind="fake",
        embedding_model_id="fake-deterministic-v1",
        embedding_model_revision=None,
        embedding_dim=1,
        normalized=True,
        index_type="exact-numpy",
        chunker_version="v4",
        chunkset_kind="semantik",
        source_chunks_sha256="0" * 64,
        embeddings_sha256="0" * 64,
        id_map_sha256="0" * 64,
        chunks_count=arr.shape[0],
        text_field_policy="text+heading",
        batch_size=16,
        device="cpu",
    )
    return VectorIndex(manifest=manifest, chunk_ids=ids, matrix=arr)


_UNIT_QUERY = np.asarray([1.0], dtype=np.float32)


def _legacy_search(index: VectorIndex, top_k: int, monkeypatch):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_TOPK_LEGACY", "1")
    try:
        return index.search(_UNIT_QUERY, top_k=top_k)
    finally:
        monkeypatch.delenv("ED4ALL_RETRIEVAL_TOPK_LEGACY", raising=False)


# --------------------------------------------------------------------------
# The knob itself
# --------------------------------------------------------------------------


def test_topk_legacy_defaults_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_RETRIEVAL_TOPK_LEGACY", raising=False)
    assert resolve_topk_legacy() is False


@pytest.mark.parametrize("token", ["1", "true", "YES", "on"])
def test_topk_legacy_truthy_tokens(monkeypatch, token):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_TOPK_LEGACY", token)
    assert resolve_topk_legacy() is True


@pytest.mark.parametrize("token", ["0", "false", "", "garbage"])
def test_topk_legacy_non_truthy_tokens(monkeypatch, token):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_TOPK_LEGACY", token)
    assert resolve_topk_legacy() is False


# --------------------------------------------------------------------------
# Parity — the load-bearing property.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 5, 7, 8])
def test_boundary_tie_group_parity(monkeypatch, k):
    """A tie group straddling EVERY plausible k must break by chunk_id.

    Scores: four rows at 0.9, four at 0.5. With ids deliberately assigned in
    REVERSE lexical order inside each group, a selection that tie-breaks only
    within an arbitrary partition subset produces a different member at the
    cut than the full sort does.
    """
    scores = [0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5]
    ids = ["d_hi", "c_hi", "b_hi", "a_hi", "d_lo", "c_lo", "b_lo", "a_lo"]
    index = _index_from_scores(scores, ids)

    new = index.search(_UNIT_QUERY, top_k=k)
    legacy = _legacy_search(index, k, monkeypatch)
    assert new == legacy


def test_all_scores_identical_orders_by_chunk_id(monkeypatch):
    """The degenerate case: ONE tie group covering the whole matrix."""
    n = 64
    ids = [f"chunk_{n - i:04d}" for i in range(n)]  # reverse lexical order
    index = _index_from_scores([0.25] * n, ids)

    for k in (1, 3, 32, n - 1, n):
        new = index.search(_UNIT_QUERY, top_k=k)
        legacy = _legacy_search(index, k, monkeypatch)
        assert new == legacy, f"k={k}"
        assert [cid for cid, _ in new] == sorted(ids)[:k]


@pytest.mark.parametrize("k", [1, 5, 50, 999, 1000])
def test_random_matrix_parity_at_library_scale_k(monkeypatch, k):
    """Randomized parity with a HIGH tie density (scores quantized to 1/20th)
    so the boundary tie group is exercised at every k, including k == N and
    k > N."""
    rng = np.random.default_rng(20260730)
    n = 1000
    scores = (rng.integers(0, 20, size=n) / 20.0).astype(np.float32)
    ids = [f"chunk_{int(v):04d}" for v in rng.permutation(n)]
    index = _index_from_scores(scores, ids)

    new = index.search(_UNIT_QUERY, top_k=k)
    legacy = _legacy_search(index, k, monkeypatch)
    assert new == legacy


def test_duplicate_chunk_ids_preserve_row_order(monkeypatch):
    """Rows identical in BOTH keys keep ascending row order.

    ``sorted`` is stable and ``lexsort`` is stable, so a corrupt id_map with
    repeated ids must still rank identically under both paths.
    """
    index = _index_from_scores([0.5, 0.5, 0.5], ["same", "same", "same"])
    new = index.search(_UNIT_QUERY, top_k=3)
    legacy = _legacy_search(index, 3, monkeypatch)
    assert new == legacy


def test_negative_and_zero_scores_parity(monkeypatch):
    """Signed cosines, including the -0.0 / 0.0 pair, which compare EQUAL and
    must therefore fall through to the chunk_id tie-break in both paths."""
    scores = [0.0, -0.0, -0.5, 1.0, -1.0, 0.0]
    ids = ["f", "a", "e", "b", "d", "c"]
    index = _index_from_scores(scores, ids)
    for k in range(1, len(scores) + 1):
        new = index.search(_UNIT_QUERY, top_k=k)
        legacy = _legacy_search(index, k, monkeypatch)
        assert new == legacy, f"k={k}"


def test_empty_and_nonpositive_k_short_circuit():
    index = _index_from_scores([0.5, 0.4])
    assert index.search(_UNIT_QUERY, top_k=0) == []
    assert index.search(_UNIT_QUERY, top_k=-3) == []

    empty = _index_from_scores([])
    assert empty.search(_UNIT_QUERY, top_k=5) == []


def test_query_dim_mismatch_still_raises():
    """The fast path must not swallow the pre-existing dimension contract."""
    index = _index_from_scores([0.5, 0.4])
    with pytest.raises(ValueError, match="query dim"):
        index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)


# --------------------------------------------------------------------------
# Direct helper coverage — the widening step, isolated.
# --------------------------------------------------------------------------


def test_widened_candidate_set_covers_full_boundary_tie_group():
    """``_topk_ordered`` must consider EVERY row equal to the k-th score, not
    just the ones argpartition happened to place below the pivot."""
    scores = np.asarray([0.7] * 10, dtype=np.float32)
    ids = [f"id_{9 - i}" for i in range(10)]
    order = vi._topk_ordered(scores, ids, 3)
    assert [ids[i] for i in order] == ["id_0", "id_1", "id_2"]


def test_helper_parity_matches_legacy_helper():
    rng = np.random.default_rng(7)
    scores = (rng.integers(0, 5, size=200) / 5.0).astype(np.float32)
    ids = [f"c{i:03d}" for i in rng.permutation(200)]
    for k in (1, 4, 100, 200):
        assert vi._topk_ordered(scores, ids, k) == vi._topk_ordered_legacy(
            scores, ids, k
        )


def test_non_finite_scores_fail_loudly_rather_than_short_result():
    """A corrupt matrix breaks the total order the whole tie-break rests on.

    Enough NaNs to reach the k boundary make the widening threshold itself
    NaN, at which point ``scores >= threshold`` selects nothing. That must be
    a loud failure, never a quietly-short result set that reads to a caller
    as "the course only had one hit".
    """
    index = _index_from_scores([0.9, float("nan"), float("nan"), 0.2])
    with pytest.raises(ValueError, match="not totally ordered"):
        index.search(_UNIT_QUERY, top_k=3)
