"""W2 §5.4 — Pass D embedding-dedup invariants (hermetic fake embed)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lib.objectives.objective_dedup as od  # noqa: E402
from lib.objectives.objective_dedup import (  # noqa: E402
    _DEFAULT_TO_CLUSTER_THRESHOLD,
    _DEFAULT_TO_COS_PER_CLUSTER,
    cluster_by_cosine,
    cluster_to_target_k,
    dedup_candidates,
    resolve_to_cluster_k,
    resolve_to_cluster_threshold,
)
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402


def _cand(statement, chunk_ids, ent=0.9):
    return {
        "statement": statement,
        "source_chunk_ids": list(chunk_ids),
        "entailment_score": ent,
        "chapter_id": "ch1",
    }


def test_near_duplicates_collapse_and_union_chunks():
    """Two near-identical statements cluster to ONE canonical; chunks union."""
    grounded = [
        _cand("Define a prime number clearly.", ["c1"], ent=0.95),
        _cand("Define what a prime number is clearly.", ["c2"], ent=0.80),
        _cand("Compute the area of a triangle formula.", ["c3"], ent=0.90),
    ]
    embed = FakeEmbed()
    # Threshold tuned to the fake bag-of-words cosine: the two prime
    # statements share "define"/"prime"/"number"/"clearly" → high cosine; the
    # triangle statement is disjoint.
    result = dedup_candidates(grounded, embed=embed, threshold=0.5, allow_fake=True)
    assert result.available is True
    # 3 candidates → 2 canonical (the two prime defs merged).
    assert len(result.canonical) == 2
    assert result.near_dup_pairs >= 1
    assert result.max_pairwise_cosine > 0.0
    # The merged representative keeps BOTH supporting chunks.
    merged = [c for c in result.canonical if "prime" in c["statement"]]
    assert merged, "expected a merged prime-definition CO"
    assert set(merged[0]["source_chunk_ids"]) == {"c1", "c2"}
    # Best-grounded representative wins (entailment 0.95 statement).
    assert merged[0]["statement"] == "Define a prime number clearly."


def test_union_cap_and_relevance_prune():
    """Fix 1A — a 25-member cluster prunes its union to ≤K relevant chunks.

    All 25 statements share the divisibility keywords so they cluster to ONE
    canonical. Their unioned chunks are 24 off-topic (author/donor/footer) + 1
    on-topic ('divisibility'). The rep statement is about divisibility, so the
    relevance-prune keeps the on-topic chunk and drops the rest, capped at K.
    """
    rep_statement = "Apply the concept of divisibility to integer division."
    grounded = []
    # 24 near-dup statements, each citing one off-topic chunk.
    for i in range(24):
        grounded.append(
            _cand(
                "Apply the concept of divisibility carefully.",
                [f"c_off_{i:03d}"],
                ent=0.80,
            )
        )
    # 1 best-grounded member (highest entailment) citing the on-topic chunk.
    grounded.append(_cand(rep_statement, ["c_div"], ent=0.99))

    chunks_by_id = {f"c_off_{i:03d}": {"id": f"c_off_{i:03d}",
                                       "text": "List of authors and donors. Correction footer boilerplate."}
                    for i in range(24)}
    chunks_by_id["c_div"] = {
        "id": "c_div",
        "text": "Divisibility means one integer divides another integer evenly with no remainder.",
    }

    embed = FakeEmbed()
    result = dedup_candidates(
        grounded,
        embed=embed,
        threshold=0.5,
        allow_fake=True,
        chunks_by_id=chunks_by_id,
        max_chunks_per_objective=5,
        chunk_relevance_floor=0.30,
    )
    assert result.available is True
    assert len(result.canonical) == 1  # all 25 merged
    rep = result.canonical[0]
    kept = rep["source_chunk_ids"]
    # Pruned to ≤ K.
    assert len(kept) <= 5
    # The on-topic divisibility chunk survives.
    assert "c_div" in kept
    # The cap actually pruned (24 off-topic were unioned).
    assert result.pruned_chunk_total > 0
    assert result.max_chunks_per_objective == 5
    # source_refs kept consistent with source_chunk_ids.
    ref_ids = result.canonical[0]["source_refs"][0]["chunk_ids"]
    assert ref_ids == kept


def test_anti_fabrication_pruned_subset_of_union():
    """Fix 1A — the kept set is always a SUBSET of the original union."""
    grounded = [
        _cand("Define divisibility of integers.", ["c1", "c2"], ent=0.9),
        _cand("Define what divisibility of integers means.", ["c3", "c4"], ent=0.9),
    ]
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Divisibility of integers and remainders."},
        "c2": {"id": "c2", "text": "Author list and donor acknowledgements footer."},
        "c3": {"id": "c3", "text": "Divisibility rules for integers explained."},
        "c4": {"id": "c4", "text": "Page navigation and copyright correction notice."},
    }
    union = {"c1", "c2", "c3", "c4"}
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.5, allow_fake=True,
        chunks_by_id=chunks_by_id, max_chunks_per_objective=5,
        chunk_relevance_floor=0.30,
    )
    assert len(result.canonical) == 1
    kept = set(result.canonical[0]["source_chunk_ids"])
    assert kept.issubset(union)  # never added a non-cited chunk
    assert len(kept) >= 1


def test_keep_at_least_one_when_all_below_floor():
    """Fix 1A — keep-≥1 supersedes the floor (never zero provenance)."""
    grounded = [
        _cand("Define divisibility of integers.", ["c1"], ent=0.9),
        _cand("Define what divisibility means for integers.", ["c2"], ent=0.9),
    ]
    # Both cited chunks are totally off-topic → all below the floor.
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Author donor list footer boilerplate."},
        "c2": {"id": "c2", "text": "Copyright correction navigation menu."},
    }
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.5, allow_fake=True,
        chunks_by_id=chunks_by_id, max_chunks_per_objective=5,
        chunk_relevance_floor=0.99,  # so high nothing clears it
    )
    assert len(result.canonical) == 1
    kept = result.canonical[0]["source_chunk_ids"]
    assert len(kept) == 1  # exactly one survivor, never zero
    assert kept[0] in {"c1", "c2"}


def test_graceful_degrade_no_chunks_by_id_full_union():
    """Fix 1A — chunks_by_id None → legacy full union (no prune, no crash)."""
    grounded = [
        _cand("Define divisibility.", ["c1", "c2"], ent=0.9),
        _cand("Define what divisibility is.", ["c3"], ent=0.9),
    ]
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.5, allow_fake=True,
        chunks_by_id=None,  # graceful degrade
    )
    assert len(result.canonical) == 1
    kept = set(result.canonical[0]["source_chunk_ids"])
    assert kept == {"c1", "c2", "c3"}  # full union preserved
    assert result.pruned_chunk_total == 0


def test_prune_capture_emitted():
    """Fix 1A — an objective_chunk_prune decision fires when chunks are dropped."""
    events = []

    class _Cap:
        def log_decision(self, **kw):
            events.append(kw)

    grounded = [
        _cand("Apply divisibility to integers.", [f"c{i}" for i in range(10)], ent=0.9),
    ]
    chunks_by_id = {
        "c0": {"id": "c0", "text": "Divisibility of integers and remainders explained."},
    }
    for i in range(1, 10):
        chunks_by_id[f"c{i}"] = {"id": f"c{i}", "text": "Author donor footer boilerplate."}
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.5, allow_fake=True,
        chunks_by_id=chunks_by_id, max_chunks_per_objective=3,
        chunk_relevance_floor=0.30, capture=_Cap(),
    )
    assert result.pruned_chunk_total > 0
    assert any(e.get("decision_type") == "objective_chunk_prune" for e in events)
    ev = next(e for e in events if e["decision_type"] == "objective_chunk_prune")
    assert len(ev["rationale"]) >= 20
    alternative = ev["alternatives_considered"][0]
    assert alternative["option"] == "Keep the full citation union"
    assert isinstance(alternative["score"], float)
    assert len(alternative["reason_rejected"]) >= 20


# ===========================================================================
# WS1 — cluster_by_cosine extraction + resolve_to_cluster_threshold
# ===========================================================================


def test_cluster_by_cosine_single_link_groups_above_threshold():
    """WS1 — texts above the cosine threshold single-link cluster together."""
    embed = FakeEmbed()
    texts = [
        "Define a prime number clearly.",
        "Define what a prime number is clearly.",
        "Compute the area of a triangle formula.",
    ]
    vecs = embed.encode_batch(texts)
    clusters, max_cos, near_dup = cluster_by_cosine(vecs, 0.5)
    # The two prime statements cluster; the triangle is its own.
    assert clusters == [[0, 1], [2]]
    assert max_cos > 0.0
    assert near_dup >= 1


def test_cluster_by_cosine_returns_max_and_pair_count():
    """WS1 — max_pairwise_cosine + near_dup_pairs reflect the matrix."""
    embed = FakeEmbed()
    # Three identical texts → all pairs above threshold → 3 near-dup pairs.
    vecs = embed.encode_batch(["alpha beta gamma"] * 3)
    clusters, max_cos, near_dup = cluster_by_cosine(vecs, 0.5)
    assert clusters == [[0, 1, 2]]
    assert near_dup == 3  # C(3,2)
    assert max_cos > 0.99


def test_cluster_by_cosine_singletons_and_empty():
    """WS1 — n==0 → ([],0,0); n==1 → ([[0]],0,0); disjoint → singletons."""
    assert cluster_by_cosine([], 0.5) == ([], 0.0, 0)
    embed = FakeEmbed()
    one = embed.encode_batch(["lonely statement here"])
    assert cluster_by_cosine(one, 0.5) == ([[0]], 0.0, 0)
    disjoint = embed.encode_batch([
        "quadratic polynomial factoring",
        "photosynthesis chloroplast biology",
    ])
    clusters, _max, near = cluster_by_cosine(disjoint, 0.5)
    assert clusters == [[0], [1]]
    assert near == 0


def test_dedup_behavior_unchanged_after_extraction(monkeypatch):
    """WS1 — dedup still calls cluster_by_cosine at the 0.88 dedup threshold,
    NOT the lower TO-cluster threshold (extraction is behavior-preserving)."""
    seen = {}
    orig = od.cluster_by_cosine

    def _spy(vecs, threshold):
        seen["threshold"] = threshold
        return orig(vecs, threshold)

    monkeypatch.setattr(od, "cluster_by_cosine", _spy)
    grounded = [
        _cand("Define a prime number clearly.", ["c1"], ent=0.95),
        _cand("Define what a prime number is clearly.", ["c2"], ent=0.80),
    ]
    # No explicit threshold → resolve_dedup_threshold default 0.88.
    dedup_candidates(grounded, embed=FakeEmbed(), allow_fake=True)
    assert seen["threshold"] == 0.88
    assert seen["threshold"] != _DEFAULT_TO_CLUSTER_THRESHOLD


def test_resolve_to_cluster_threshold_env_and_default(monkeypatch):
    """WS1 — explicit arg → env → default; garbage/out-of-range → default."""
    monkeypatch.delenv("ED4ALL_TO_CLUSTER_THRESHOLD", raising=False)
    assert resolve_to_cluster_threshold() == _DEFAULT_TO_CLUSTER_THRESHOLD
    assert resolve_to_cluster_threshold(0.62) == 0.62
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_THRESHOLD", "0.40")
    assert resolve_to_cluster_threshold() == 0.40
    # Garbage / out of range → default.
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_THRESHOLD", "not-a-float")
    assert resolve_to_cluster_threshold() == _DEFAULT_TO_CLUSTER_THRESHOLD
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_THRESHOLD", "1.7")
    assert resolve_to_cluster_threshold() == _DEFAULT_TO_CLUSTER_THRESHOLD


# ===========================================================================
# WS1.1 — TARGET-K Ward agglomerative clustering + resolve_to_cluster_k
# ===========================================================================


def _blob_vecs(n_blobs: int = 5, per_blob: int = 6):
    """~30 vectors in ~5 gaussian blobs (distinct vocab per blob) via FakeEmbed.

    Each blob shares a unique base vocabulary; intra-blob members differ by a
    trailing index token so they cluster together but aren't byte-identical.
    """
    bases = [
        "algebra linear equation matrix vector",
        "geometry triangle area perimeter angle",
        "biology cell photosynthesis chloroplast membrane",
        "chemistry atom electron bond molecule reaction",
        "history empire revolution treaty monarchy republic",
    ][:n_blobs]
    texts = []
    for b, base in enumerate(bases):
        for j in range(per_blob):
            texts.append(f"{base} variant{b}{j}")
    return FakeEmbed().encode_batch(texts), len(texts)


def _assert_partition_k(clusters, n: int) -> None:
    seen = [i for c in clusters for i in c]
    assert sorted(seen) == list(range(n)), "every vector in exactly one cluster"
    assert len(seen) == len(set(seen)), "no vector duplicated across clusters"
    for c in clusters:
        assert c == sorted(c), "members ascending"
    mins = [min(c) for c in clusters]
    assert mins == sorted(mins), "clusters ordered by min member index"


def test_cluster_to_target_k_returns_k_balanced_clusters():
    """WS1.1 — ~30 vecs in ~5 blobs at K=5 → ~5 balanced, partitioned clusters."""
    vecs, n = _blob_vecs(n_blobs=5, per_blob=6)
    k = 5
    clusters = cluster_to_target_k(vecs, k)
    assert len(clusters) == k
    _assert_partition_k(clusters, n)
    # No single cluster dominates (>60%) when K matches the blob count.
    biggest = max(len(c) for c in clusters)
    assert biggest <= 0.60 * n, f"over-broad cluster ({biggest}/{n})"


def test_cluster_to_target_k_singletons_and_empty():
    """WS1.1 — guards: n==0 → []; n==1 → [[0]]; k<1 treated as 1."""
    assert cluster_to_target_k([], 5) == []
    one = FakeEmbed().encode_batch(["a single lonely statement"])
    assert cluster_to_target_k(one, 5) == [[0]]
    # k < 1 treated as 1 → a single all-inclusive cluster.
    vecs, n = _blob_vecs(n_blobs=3, per_blob=4)
    clusters = cluster_to_target_k(vecs, 0)
    assert len(clusters) == 1
    _assert_partition_k(clusters, n)


def test_resolve_to_cluster_k_auto_and_env(monkeypatch):
    """WS1.1 — auto from n; ED4ALL_TO_CLUSTER_K override; cos-per-cluster; garbage."""
    monkeypatch.delenv("ED4ALL_TO_CLUSTER_K", raising=False)
    monkeypatch.delenv("ED4ALL_TO_COS_PER_CLUSTER", raising=False)
    # Auto: 68 COs / 6 per cluster ≈ 12, clamped [3, 15].
    assert resolve_to_cluster_k(68) == 12
    # Auto clamps the floor (small n) and ceiling (huge n).
    assert resolve_to_cluster_k(4) == 3        # ceil(4/6)=1 → floor 3
    assert resolve_to_cluster_k(300) == 15     # ceil(300/6)=50 → cap 15
    # K can never exceed n.
    assert resolve_to_cluster_k(2) == 2        # min(floor 3, n=2) = 2
    # Explicit arg beats env + auto.
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "9")
    assert resolve_to_cluster_k(68, explicit=4) == 4
    # Fixed env K (clamped to n).
    assert resolve_to_cluster_k(68) == 9
    assert resolve_to_cluster_k(5) == 5        # min(9, n=5)
    # ED4ALL_TO_CLUSTER_K=0 → auto.
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "0")
    assert resolve_to_cluster_k(68) == 12
    # Garbage fixed K → auto.
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "not-an-int")
    assert resolve_to_cluster_k(68) == 12
    # cos-per-cluster divisor changes the auto count.
    monkeypatch.delenv("ED4ALL_TO_CLUSTER_K", raising=False)
    monkeypatch.setenv("ED4ALL_TO_COS_PER_CLUSTER", "12")
    assert resolve_to_cluster_k(60) == 5       # ceil(60/12)=5
    # Garbage cos-per-cluster → default 6.
    monkeypatch.setenv("ED4ALL_TO_COS_PER_CLUSTER", "junk")
    assert resolve_to_cluster_k(60) == 10      # ceil(60/6)=10
    assert _DEFAULT_TO_COS_PER_CLUSTER == 6


def test_cluster_to_target_k_sklearn_absent_falls_back(monkeypatch):
    """WS1.1 — sklearn ImportError → falls back to cluster_by_cosine, no crash."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sklearn.cluster" or name.startswith("sklearn"):
            raise ImportError("sklearn intentionally unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delenv("ED4ALL_TO_CLUSTER_THRESHOLD", raising=False)
    vecs, n = _blob_vecs(n_blobs=3, per_blob=4)
    # Must not raise; returns a valid partition from the single-link fallback.
    clusters = cluster_to_target_k(vecs, 5)
    _assert_partition_k(clusters, n)
    # The fallback used the single-link cosine path → matches cluster_by_cosine.
    expected, _max, _near = cluster_by_cosine(
        vecs, resolve_to_cluster_threshold()
    )
    assert clusters == expected


def test_embeddings_absent_passthrough():
    """build_embedding_client raising → pass-through (each its own cluster)."""
    grounded = [
        _cand("Statement one alpha.", ["c1"]),
        _cand("Statement two beta.", ["c2"]),
    ]

    def _raise(*a, **k):
        from lib.embedding.providers import EmbeddingBackendUnavailable
        raise EmbeddingBackendUnavailable("no extras in CI")

    orig = od.build_embedding_client
    od.build_embedding_client = _raise  # type: ignore[assignment]
    try:
        result = dedup_candidates(grounded)  # embed=None → builds → raises
    finally:
        od.build_embedding_client = orig  # type: ignore[assignment]
    assert result.available is False
    assert len(result.canonical) == 2  # no collapse
    assert result.clusters == [[0], [1]]


# --------------------------------------------------------------------- #
# I3 PRONG A — distinct-skill SPLIT
# --------------------------------------------------------------------- #


class _AngleEmbed:
    """Embed stub returning a FIXED vector per statement tag.

    Each candidate statement starts with a tag (``A:``/``B:``/``C:``); the
    stub maps the tag to a 3-D unit vector so the test pins cosines EXACTLY
    (cos(A,B)=0.90, cos(B,C)=0.90, cos(A,C)=0.60) regardless of the bag-of-words
    content — letting the statement TEXT carry the distinct skill signature
    independently of the geometry. ``resolved.kind`` is non-"fake" so the
    anti-poisoning gate passes (allow_fake not required).
    """

    class _R:
        kind = "fake-test"

    def __init__(self, tag_vecs):
        self._tag_vecs = tag_vecs
        self.resolved = self._R()

    def encode_batch(self, texts):
        out = []
        for t in texts:
            tag = str(t).split(":", 1)[0] + ":"
            out.append(list(self._tag_vecs.get(tag, [1.0, 0.0, 0.0])))
        return out


def _unit_from_cosines():
    """Build 3 unit vectors with cos(A,B)=0.90, cos(B,C)=0.90, cos(A,C)=0.60."""
    import math

    # A = e0. B at angle so A.B = 0.90.
    A = [1.0, 0.0, 0.0]
    bx = 0.90
    by = math.sqrt(1 - bx * bx)
    B = [bx, by, 0.0]
    # C: C.A = 0.60, C.B = 0.90. Solve cx=0.60; cx*bx + cy*by = 0.90.
    cx = 0.60
    cy = (0.90 - cx * bx) / by
    cz = math.sqrt(max(0.0, 1 - cx * cx - cy * cy))
    C = [cx, cy, cz]
    return A, B, C


def test_distinct_skill_split_separates_chained_skills():
    """cos(A,B)=cos(B,C)=0.90, cos(A,C)=0.60 + distinct sigs → split to ≥2 COs.

    Single-link at threshold 0.85 chains A-B-C into ONE cluster (A,C never
    directly similar). With the split ON, A (order-of-operations) and C
    (associative-property) keep their OWN representative statements instead of
    only the best-grounded one surviving.
    """
    A, B, C = _unit_from_cosines()
    embed = _AngleEmbed({"A:": A, "B:": B, "C:": C})
    grounded = [
        _cand("A: apply the order of operations to evaluate", ["c1"], ent=0.95),
        _cand("B: order of operations and simplify expression", ["c2"], ent=0.80),
        _cand("C: use the associative property to group terms", ["c3"], ent=0.90),
    ]
    # Split OFF (default) → single-link chains all 3 into ONE canonical CO.
    off = dedup_candidates(grounded, embed=embed, threshold=0.85)
    assert len(off.canonical) == 1
    assert off.clusters_split_for_distinct_skill == 0

    # Split ON → ≥2 canonical COs; order-of-operations + associative each kept.
    on = dedup_candidates(
        grounded, embed=embed, threshold=0.85, distinct_skill_split=True,
    )
    assert len(on.canonical) >= 2
    assert on.clusters_split_for_distinct_skill == 1
    assert on.distinct_skill_count >= 2
    statements = " || ".join(c["statement"].lower() for c in on.canonical)
    assert "order of operations" in statements
    assert "associative property" in statements


def test_distinct_skill_split_same_signature_still_collapses():
    """Same-signature restatements at cos≥0.95 still collapse to 1 even split-on.

    Two paraphrases of the SAME skill (order-of-operations) must NOT split — the
    0.88 same-signature collapse is preserved for exact restatements.
    """
    import math

    near = [1.0, 0.0, 0.0]
    other = [math.cos(0.1), math.sin(0.1), 0.0]  # cos ≈ 0.995
    embed = _AngleEmbed({"A:": near, "B:": other})
    grounded = [
        _cand("A: apply order of operations to evaluate expressions", ["c1"], ent=0.9),
        _cand("B: use order of operations to evaluate the expression", ["c2"], ent=0.8),
    ]
    on = dedup_candidates(
        grounded, embed=embed, threshold=0.85, distinct_skill_split=True,
    )
    assert len(on.canonical) == 1  # same skill → no split
    assert on.clusters_split_for_distinct_skill == 0
    # Chunks still union onto the survivor.
    assert set(on.canonical[0]["source_chunk_ids"]) == {"c1", "c2"}


def test_distinct_skill_split_off_byte_stable(monkeypatch):
    """Flag OFF (env unset) → identical canonical/cluster shape to pre-I3."""
    monkeypatch.delenv("ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT", raising=False)
    grounded = [
        _cand("Define a prime number clearly.", ["c1"], ent=0.95),
        _cand("Define what a prime number is clearly.", ["c2"], ent=0.80),
        _cand("Compute the area of a triangle formula.", ["c3"], ent=0.90),
    ]
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.5, allow_fake=True,
    )
    assert len(result.canonical) == 2
    assert result.clusters_split_for_distinct_skill == 0


# ---------------------------------------------------------------------------
# P1 (clustering rework) — post-cluster GUARD helpers
# ---------------------------------------------------------------------------
def _flat(clusters):
    out = []
    for c in clusters:
        out.extend(c)
    return out


def _assert_partition_invariant(new_clusters, original_clusters):
    """Flattened output indices are a permutation of the flattened input."""
    assert sorted(_flat(new_clusters)) == sorted(_flat(original_clusters))


# A coherent cohort of fraction-skill statements (high token overlap) + ONE
# semantic outlier (disjoint insurance vocabulary). FakeEmbed = bag-of-words
# cosine, so the outlier's centroid is far from the cohort's.
_COHORT_TEXTS = [
    "add fractions with like denominators",
    "subtract fractions with like denominators",
    "add fractions with unlike denominators common denominator",
    "multiply fractions numerator denominator together",
    "divide fractions reciprocal numerator denominator",
]
_OUTLIER_TEXT = "insurance deductible premium policy payment claim coverage"


def test_cluster_centroid_and_cohesion_helpers():
    vecs = FakeEmbed().encode_batch(_COHORT_TEXTS)
    assert od._cluster_centroid(vecs, []) == []
    cent = od._cluster_centroid(vecs, [0, 1])
    assert len(cent) == len(vecs[0])
    # Singleton / empty cohesion is 1.0 by convention.
    assert od._mean_intra_cluster_cosine(vecs, [0]) == 1.0
    assert od._mean_intra_cluster_cosine(vecs, []) == 1.0
    # A real multi-member cohesion is a finite cosine in [0, 1].
    coh = od._mean_intra_cluster_cosine(vecs, [0, 1, 2])
    assert 0.0 <= coh <= 1.0


def test_absorb_outlier_singleton_into_nearest_neighbor():
    """An outlier singleton with a clear home is absorbed; no CO is lost."""
    vecs = FakeEmbed().encode_batch(_COHORT_TEXTS + [_OUTLIER_TEXT])
    clusters = [[0, 1, 2, 3, 4], [5]]  # cohort + outlier singleton
    # Floor 0.0 → the outlier is folded into whichever neighbor it's closest to.
    new_clusters, n_absorbed = od.absorb_outlier_clusters(
        clusters, vecs, min_size=3, absorb_floor=0.0
    )
    assert n_absorbed == 1
    assert len(new_clusters) < len(clusters)
    _assert_partition_invariant(new_clusters, clusters)
    # Partition invariant in the explicit form the task asks for.
    assert set(_flat(new_clusters)) == set(range(6))


def test_outlier_with_no_home_remains_standing():
    """An outlier whose best neighbor is below the floor is NOT absorbed."""
    vecs = FakeEmbed().encode_batch(_COHORT_TEXTS + [_OUTLIER_TEXT])
    clusters = [[0, 1, 2, 3, 4], [5]]
    # A floor of 1.0 means "only absorb a near-identical neighbor" — the
    # disjoint outlier has no such home, so it stays standing.
    new_clusters, n_absorbed = od.absorb_outlier_clusters(
        clusters, vecs, min_size=3, absorb_floor=1.0
    )
    assert n_absorbed == 0
    assert new_clusters == [[0, 1, 2, 3, 4], [5]]
    _assert_partition_invariant(new_clusters, clusters)


def test_merge_undersize_runt_into_nearest_neighbor():
    """A 2-CO runt cluster merges into its nearest neighbor under min-size=3."""
    vecs = FakeEmbed().encode_batch(_COHORT_TEXTS)
    clusters = [[0, 1, 2], [3, 4]]  # a 3-cluster + a 2-CO runt
    new_clusters, n_merged = od.merge_undersize_clusters(
        clusters, vecs, min_size=3
    )
    assert n_merged == 1
    assert len(new_clusters) == 1  # runt folded in → one surviving cluster
    _assert_partition_invariant(new_clusters, clusters)
    assert set(_flat(new_clusters)) == set(range(5))


# Two NEAR-DUPLICATE clusters: both number-line themed with heavy token overlap
# across the two, plus a distinct area-of-shapes theme that must NOT merge in.
_NUMBERLINE_A = [
    "plot points on the number line",
    "locate integers on the number line",
]
_NUMBERLINE_B = [
    "graph numbers on the number line",
    "order numbers on the number line",
]
_AREA_THEME = [
    "compute the area of a triangle base height",
    "compute the area of a rectangle length width",
]


def test_merge_near_duplicate_clusters_collapse_to_one():
    """Two near-duplicate (centroid cosine >= floor) clusters collapse to one."""
    texts = _NUMBERLINE_A + _NUMBERLINE_B + _AREA_THEME
    vecs = FakeEmbed().encode_batch(texts)
    clusters = [[0, 1], [2, 3], [4, 5]]
    # Floor 0.5 merges the two number-line clusters (centroid cos ~0.84) but
    # leaves the distinct area theme separate.
    new_clusters, n_pairs = od.merge_near_duplicate_clusters(
        clusters, vecs, merge_floor=0.5
    )
    assert n_pairs >= 1
    assert len(new_clusters) == 2  # number-line collapsed; area survives
    assert [0, 1, 2, 3] in new_clusters
    assert [4, 5] in new_clusters
    _assert_partition_invariant(new_clusters, clusters)


def test_merge_near_duplicate_high_floor_is_noop():
    """A floor near 1.0 merges nothing (no centroids that similar)."""
    texts = _NUMBERLINE_A + _NUMBERLINE_B + _AREA_THEME
    vecs = FakeEmbed().encode_batch(texts)
    clusters = [[0, 1], [2, 3], [4, 5]]
    new_clusters, n_pairs = od.merge_near_duplicate_clusters(
        clusters, vecs, merge_floor=0.999
    )
    assert n_pairs == 0
    assert new_clusters == [[0, 1], [2, 3], [4, 5]]
    _assert_partition_invariant(new_clusters, clusters)


def test_apply_cluster_guards_default_is_noop(monkeypatch):
    """DEFAULT (no env, no kwargs) → input clusters unchanged + zeroed signals."""
    for var in (
        "ED4ALL_TO_CLUSTER_GUARDS",
        "ED4ALL_TO_MERGE_NEAR_DUP",
        "ED4ALL_TO_OUTLIER_MIN_SIZE",
        "ED4ALL_TO_OUTLIER_ABSORB_FLOOR",
        "ED4ALL_TO_MERGE_COSINE",
    ):
        monkeypatch.delenv(var, raising=False)
    vecs = FakeEmbed().encode_batch(_COHORT_TEXTS + [_OUTLIER_TEXT])
    clusters = [[0, 1, 2, 3, 4], [5]]
    new_clusters, signals = od.apply_cluster_guards(clusters, vecs)
    # Byte-identical (re-ordered to the contract, which it already satisfies).
    assert new_clusters == [[0, 1, 2, 3, 4], [5]]
    assert signals == {
        "outliers_absorbed": 0,
        "undersize_merged": 0,
        "near_dup_clusters_merged": 0,
        "clusters_before": 2,
        "clusters_after": 2,
    }


def test_apply_cluster_guards_consolidate_then_merge():
    """Orchestrator runs consolidate then near-dup merge with signals."""
    texts = _NUMBERLINE_A + _NUMBERLINE_B + _AREA_THEME + [_OUTLIER_TEXT]
    vecs = FakeEmbed().encode_batch(texts)
    # A number-line pair, a near-dup number-line pair, an area pair, and an
    # outlier singleton.
    clusters = [[0, 1], [2, 3], [4, 5], [6]]
    new_clusters, signals = od.apply_cluster_guards(
        clusters,
        vecs,
        enable_consolidate=True,
        enable_merge=True,
        min_size=2,
        merge_floor=0.5,
        absorb_floor=0.0,
    )
    _assert_partition_invariant(new_clusters, clusters)
    assert set(_flat(new_clusters)) == set(range(7))
    # The singleton outlier (size 1 < min_size 2) was absorbed.
    assert signals["outliers_absorbed"] == 1
    # The two number-line clusters merged on near-dup centroid cosine.
    assert signals["near_dup_clusters_merged"] >= 1
    assert signals["clusters_before"] == 4
    assert signals["clusters_after"] == len(new_clusters)


def test_apply_cluster_guards_env_gated(monkeypatch):
    """The master env gate alone (no kwargs) enables the consolidate pass."""
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_GUARDS", "true")
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_MIN_SIZE", "3")
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_ABSORB_FLOOR", "0.0")
    monkeypatch.delenv("ED4ALL_TO_MERGE_NEAR_DUP", raising=False)
    vecs = FakeEmbed().encode_batch(_COHORT_TEXTS + [_OUTLIER_TEXT])
    clusters = [[0, 1, 2, 3, 4], [5]]
    new_clusters, signals = od.apply_cluster_guards(clusters, vecs)
    assert signals["outliers_absorbed"] == 1
    assert signals["near_dup_clusters_merged"] == 0  # merge gate off
    _assert_partition_invariant(new_clusters, clusters)


def test_resolve_to_cluster_guards_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_CLUSTER_GUARDS", raising=False)
    assert od.resolve_to_cluster_guards() is False  # default OFF
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_GUARDS", "on")
    assert od.resolve_to_cluster_guards() is True
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_GUARDS", "garbage")
    assert od.resolve_to_cluster_guards() is False  # garbage → default
    assert od.resolve_to_cluster_guards(True) is True  # explicit arg wins


def test_resolve_to_merge_near_dup_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_MERGE_NEAR_DUP", raising=False)
    assert od.resolve_to_merge_near_dup() is False
    monkeypatch.setenv("ED4ALL_TO_MERGE_NEAR_DUP", "yes")
    assert od.resolve_to_merge_near_dup() is True
    monkeypatch.setenv("ED4ALL_TO_MERGE_NEAR_DUP", "nonsense")
    assert od.resolve_to_merge_near_dup() is False


def test_resolve_to_outlier_min_size_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_OUTLIER_MIN_SIZE", raising=False)
    assert od.resolve_to_outlier_min_size() == od._DEFAULT_TO_OUTLIER_MIN_SIZE
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_MIN_SIZE", "5")
    assert od.resolve_to_outlier_min_size() == 5
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_MIN_SIZE", "garbage")
    assert od.resolve_to_outlier_min_size() == od._DEFAULT_TO_OUTLIER_MIN_SIZE
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_MIN_SIZE", "-1")
    assert od.resolve_to_outlier_min_size() == od._DEFAULT_TO_OUTLIER_MIN_SIZE
    assert od.resolve_to_outlier_min_size(7) == 7


def test_resolve_to_outlier_absorb_floor_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_OUTLIER_ABSORB_FLOOR", raising=False)
    assert (
        od.resolve_to_outlier_absorb_floor()
        == od._DEFAULT_TO_OUTLIER_ABSORB_FLOOR
    )
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_ABSORB_FLOOR", "0.4")
    assert od.resolve_to_outlier_absorb_floor() == 0.4
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_ABSORB_FLOOR", "9.0")  # out of range
    assert (
        od.resolve_to_outlier_absorb_floor()
        == od._DEFAULT_TO_OUTLIER_ABSORB_FLOOR
    )
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_ABSORB_FLOOR", "garbage")
    assert (
        od.resolve_to_outlier_absorb_floor()
        == od._DEFAULT_TO_OUTLIER_ABSORB_FLOOR
    )


def test_resolve_to_merge_cosine_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_MERGE_COSINE", raising=False)
    assert od.resolve_to_merge_cosine() == od._DEFAULT_TO_MERGE_COSINE
    monkeypatch.setenv("ED4ALL_TO_MERGE_COSINE", "0.9")
    assert od.resolve_to_merge_cosine() == 0.9
    monkeypatch.setenv("ED4ALL_TO_MERGE_COSINE", "5.0")  # out of range
    assert od.resolve_to_merge_cosine() == od._DEFAULT_TO_MERGE_COSINE
    monkeypatch.setenv("ED4ALL_TO_MERGE_COSINE", "garbage")
    assert od.resolve_to_merge_cosine() == od._DEFAULT_TO_MERGE_COSINE


def test_guards_single_cluster_and_empty_inputs():
    """Degenerate partitions are returned unchanged (no crash)."""
    vecs = FakeEmbed().encode_batch(_COHORT_TEXTS)
    assert od.merge_near_duplicate_clusters([[0, 1, 2]], vecs) == ([[0, 1, 2]], 0)
    assert od.consolidate_small_clusters([[0]], vecs, min_size=3)[0] == [[0]]
    empty, sig = od.apply_cluster_guards([], [], enable_consolidate=True)
    assert empty == []
    assert sig["clusters_before"] == 0 and sig["clusters_after"] == 0


# ---------------------------------------------------------------------------
# dissolve_singletons — the unconditional anti-hallucinated-TO backstop.
#
# Topic-free synthetic vectors (no bag-of-words strings): 5 balanced 2-member
# clusters on orthogonal axes + ONE clear far-outlier singleton that leans
# slightly toward the third cluster so its nearest neighbor is deterministic.
# ---------------------------------------------------------------------------
_BALANCED_VECS = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 0  cluster A
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.05],  # 1  cluster A
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # 2  cluster B
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.05],  # 3  cluster B
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # 4  cluster C
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.05],  # 5  cluster C
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # 6  cluster D
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.05],  # 7  cluster D
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # 8  cluster E
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.05],  # 9  cluster E
    [0.0, 0.0, 0.2, 0.0, 0.0, 1.0],   # 10 far outlier, leans toward cluster C
]
_BALANCED_CLUSTERS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]


def test_dissolve_singletons_folds_outlier_into_nearest():
    """A far-outlier singleton is folded into its nearest cluster; none remain."""
    new_clusters, n_dissolved = od.dissolve_singletons(
        _BALANCED_CLUSTERS, _BALANCED_VECS
    )
    assert n_dissolved == 1
    # No size-1 cluster survives.
    assert all(len(c) > 1 for c in new_clusters)
    # The outlier (index 10) merged into cluster C ([4, 5]).
    assert [4, 5, 10] in new_clusters
    # Partition invariant: flattened output is a permutation of range(n).
    assert sorted(_flat(new_clusters)) == list(range(11))
    _assert_partition_invariant(new_clusters, _BALANCED_CLUSTERS)


def test_dissolve_singletons_tiny_course_protection():
    """<=min_clusters clusters with singletons are LEFT (never collapsed)."""
    # 3 clusters (two singletons); default min_clusters == 3 → any fold would
    # drop below the floor, so nothing is dissolved.
    tiny = [[0, 1], [2], [3]]
    new_clusters, n_dissolved = od.dissolve_singletons(tiny, _BALANCED_VECS)
    assert n_dissolved == 0
    assert new_clusters == [[0, 1], [2], [3]]
    _assert_partition_invariant(new_clusters, tiny)


def test_dissolve_singletons_determinism():
    """Same input → same partition (deterministic fold order)."""
    a, na = od.dissolve_singletons(_BALANCED_CLUSTERS, _BALANCED_VECS)
    b, nb = od.dissolve_singletons(_BALANCED_CLUSTERS, _BALANCED_VECS)
    assert a == b
    assert na == nb


def test_dissolve_singletons_partition_invariant_multi():
    """Multiple singletons all fold; partition invariant holds throughout."""
    # cluster A (0,1) + three singletons (2, 3, 4) → dissolve down to the floor.
    clusters = [[0, 1], [2], [3], [4], [8, 9]]
    new_clusters, n_dissolved = od.dissolve_singletons(
        clusters, _BALANCED_VECS, min_clusters=2
    )
    assert n_dissolved >= 1
    assert sorted(_flat(new_clusters)) == sorted(_flat(clusters))
    # Floor honored: never fewer than min_clusters surviving clusters.
    assert len(new_clusters) >= 2


def test_resolve_to_allow_singleton_to_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_ALLOW_SINGLETON_TO", raising=False)
    assert od.resolve_to_allow_singleton_to() is False  # default OFF → fix ON
    monkeypatch.setenv("ED4ALL_TO_ALLOW_SINGLETON_TO", "1")
    assert od.resolve_to_allow_singleton_to() is True
    monkeypatch.setenv("ED4ALL_TO_ALLOW_SINGLETON_TO", "on")
    assert od.resolve_to_allow_singleton_to() is True
    monkeypatch.setenv("ED4ALL_TO_ALLOW_SINGLETON_TO", "garbage")
    assert od.resolve_to_allow_singleton_to() is False  # garbage → default
    assert od.resolve_to_allow_singleton_to(True) is True  # explicit arg wins


def test_resolve_to_min_clusters_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_MIN_CLUSTERS", raising=False)
    assert od.resolve_to_min_clusters() == od._DEFAULT_TO_MIN_CLUSTERS
    monkeypatch.setenv("ED4ALL_TO_MIN_CLUSTERS", "5")
    assert od.resolve_to_min_clusters() == 5
    monkeypatch.setenv("ED4ALL_TO_MIN_CLUSTERS", "garbage")
    assert od.resolve_to_min_clusters() == od._DEFAULT_TO_MIN_CLUSTERS
    monkeypatch.setenv("ED4ALL_TO_MIN_CLUSTERS", "-1")
    assert od.resolve_to_min_clusters() == od._DEFAULT_TO_MIN_CLUSTERS
    assert od.resolve_to_min_clusters(7) == 7


def test_entailing_chunk_pinned_through_union_prune():
    """2026-07-04 pin — the rep's NLI-entailing chunk survives the cosine prune.

    A merged cluster's representative carries ``entailing_chunk_id`` (Pass-C
    stamp) pointing at a chunk whose TEXT is cosine-unrelated to the statement
    (a math-notation worked example). The Fix-1A relevance prune ranks by
    cosine and would drop it (below floor + out-cosined by topical chunks);
    the pin re-keeps it — entailment evidence outranks cosine relevance.
    """
    rep_statement = "Apply the concept of divisibility to integer division."
    rep = _cand(rep_statement, ["c_pin", "c_top_a"], ent=0.99)
    rep["entailing_chunk_id"] = "c_pin"
    grounded = [
        rep,
        _cand("Apply the concept of divisibility carefully.", ["c_top_b", "c_top_c"], ent=0.80),
        _cand("Apply the concept of divisibility precisely.", ["c_top_d", "c_top_e"], ent=0.70),
    ]
    chunks_by_id = {
        # The entailing chunk reads as math-notation soup — shares NO
        # vocabulary with the statement, so its fake-embed cosine is ~0.
        "c_pin": {"id": "c_pin", "text": "sqrt frac latex notation qquad cdot"},
    }
    for cid in ("c_top_a", "c_top_b", "c_top_c", "c_top_d", "c_top_e"):
        chunks_by_id[cid] = {
            "id": cid,
            "text": "divisibility of integers division concept applied",
        }
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.5, allow_fake=True,
        chunks_by_id=chunks_by_id, max_chunks_per_objective=3,
        chunk_relevance_floor=0.30,
    )
    assert result.available is True
    assert len(result.canonical) == 1
    merged = result.canonical[0]
    kept = merged["source_chunk_ids"]
    # Pin survives despite losing the cosine race.
    assert "c_pin" in kept
    # Cap widens by AT MOST one for the pin.
    assert len(kept) <= 3 + 1
    # Anti-fabrication: still a subset of the union.
    union = {"c_pin", "c_top_a", "c_top_b", "c_top_c", "c_top_d", "c_top_e"}
    assert set(kept).issubset(union)
    # source_refs mirror the kept set.
    assert merged["source_refs"][0]["chunk_ids"] == kept


def test_no_pin_field_behavior_unchanged():
    """Without ``entailing_chunk_id`` the prune output is the legacy shape."""
    grounded = [
        _cand("Define divisibility of integers.", ["c1", "c2"], ent=0.9),
        _cand("Define what divisibility of integers means.", ["c3"], ent=0.8),
    ]
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Divisibility of integers and remainders."},
        "c2": {"id": "c2", "text": "Author list and donor acknowledgements footer."},
        "c3": {"id": "c3", "text": "Divisibility rules for integers explained."},
    }
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.5, allow_fake=True,
        chunks_by_id=chunks_by_id, max_chunks_per_objective=5,
        chunk_relevance_floor=0.30,
    )
    assert len(result.canonical) == 1
    kept = set(result.canonical[0]["source_chunk_ids"])
    # Off-topic c2 pruned exactly as before; on-topic chunks kept.
    assert "c1" in kept and "c3" in kept
    assert "c2" not in kept


# --------------------------------------------------------------------------- #
# Defect E — cross-window lexical-dedup SECOND PASS (complete-linkage merge).
# --------------------------------------------------------------------------- #

from lib.objectives.objective_dedup import merge_clusters_lexical  # noqa: E402


def _lex_cand(statement, chunk_ids, ent=0.9, bloom="understand"):
    return {
        "statement": statement,
        "source_chunk_ids": list(chunk_ids),
        "entailment_score": ent,
        "bloom_level": bloom,
        "chapter_id": "ch1",
    }


def test_lexical_merge_pair_merges_at_floors():
    """A same-band restatement pair above BOTH floors merges to one cluster."""
    embed = FakeEmbed()
    grounded = [
        _lex_cand("Understand the associative property clearly.", ["c1"], ent=0.95),
        _lex_cand("Understand associative property fully.", ["c2"], ent=0.80),
        _lex_cand("Compute the triangle area formula.", ["c3"], bloom="apply"),
    ]
    vecs = embed.encode_batch([c["statement"] for c in grounded])
    clusters = [[0], [1], [2]]
    out, ops, absorbed, counts = merge_clusters_lexical(
        clusters, grounded, vecs, cosine_floor=0.6, jaccard_floor=0.6
    )
    assert [0, 1] in out          # the two associative-property COs merged
    assert [2] in out             # the triangle CO stayed separate
    assert ops == 1 and absorbed == 1
    # merged_counts is parallel to `out`; the merged cluster reports 2 inputs.
    merged_pos = out.index([0, 1])
    assert counts[merged_pos] == 2
    assert counts[out.index([2])] == 1


def test_lexical_merge_complete_linkage_blocks_chain():
    """A-B-C where the A-C pair fails a floor does NOT fully merge (complete-link).

    A and B share a signature and cosine; B and C share a signature and cosine;
    but A and C are distinct — complete linkage forbids folding all three into
    one cluster (that is exactly the single-link mega-chain this pass prevents).
    """
    embed = FakeEmbed()
    # A/B share "associative addition"; B/C share "distributive multiplication";
    # A/C share only the "property/understand" scaffolding (Jaccard 0.33 < 0.6).
    grounded = [
        _lex_cand("Understand associative addition property.", ["c1"], ent=0.95),
        _lex_cand("Understand associative addition distributive multiplication property.", ["c2"], ent=0.9),
        _lex_cand("Understand distributive multiplication property.", ["c3"], ent=0.85),
    ]
    vecs = embed.encode_batch([c["statement"] for c in grounded])
    out, ops, absorbed, _counts = merge_clusters_lexical(
        [[0], [1], [2]], grounded, vecs, cosine_floor=0.3, jaccard_floor=0.6
    )
    # NOT one 3-member cluster. At most a single pairwise merge (B with A or C),
    # never {0,1,2} together.
    assert [0, 1, 2] not in out
    assert all(len(c) <= 2 for c in out)
    # Partition invariant holds regardless of which pair (if any) merged.
    flat = sorted(i for c in out for i in c)
    assert flat == [0, 1, 2]


def test_lexical_merge_never_crosses_bloom_band():
    """Two clusters identical in signature + cosine but different Bloom bands
    never merge (a lower-order CO must not bleed into a higher-order cluster)."""
    embed = FakeEmbed()
    grounded = [
        _lex_cand("Understand associative property clearly.", ["c1"], bloom="understand"),
        _lex_cand("Apply associative property clearly.", ["c2"], bloom="apply"),
    ]
    vecs = embed.encode_batch([c["statement"] for c in grounded])
    out, ops, absorbed, _counts = merge_clusters_lexical(
        [[0], [1]], grounded, vecs, cosine_floor=0.3, jaccard_floor=0.3
    )
    assert ops == 0 and absorbed == 0
    assert sorted(out) == [[0], [1]]


def test_lexical_merge_partition_invariant_property():
    """Property: for a range of floors the flattened output is always a
    permutation of the input indices (never drops/adds/invents a CO)."""
    embed = FakeEmbed()
    grounded = [
        _lex_cand("Understand associative property clearly.", ["c1"]),
        _lex_cand("Understand associative property fully.", ["c2"]),
        _lex_cand("Understand commutative property here.", ["c3"]),
        _lex_cand("Compute triangle area formula.", ["c4"], bloom="apply"),
        _lex_cand("Compute triangle area precisely.", ["c5"], bloom="apply"),
    ]
    vecs = embed.encode_batch([c["statement"] for c in grounded])
    n = len(grounded)
    for cf in (0.2, 0.5, 0.8, 0.95):
        for jf in (0.2, 0.5, 0.8):
            out, ops, absorbed, counts = merge_clusters_lexical(
                [[i] for i in range(n)], grounded, vecs,
                cosine_floor=cf, jaccard_floor=jf,
            )
            flat = sorted(i for c in out for i in c)
            assert flat == list(range(n)), (cf, jf, out)
            assert absorbed == n - len(out)
            assert sum(counts) >= len(out)  # each output has >=1 input


def test_lexical_merge_flag_off_byte_identical(monkeypatch):
    """dedup_candidates with the lexical flag OFF is byte-identical to legacy."""
    monkeypatch.delenv("ED4ALL_OBJECTIVE_DEDUP_LEXICAL", raising=False)
    grounded = [
        _lex_cand("Understand the associative property clearly.", ["c1"], ent=0.95),
        _lex_cand("Understand associative property fully.", ["c2"], ent=0.80),
        _lex_cand("Compute the triangle area formula.", ["c3"], bloom="apply"),
    ]
    baseline = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.88, allow_fake=True
    ).to_dict()
    # Explicit False and env-off must both reproduce the baseline exactly.
    off_explicit = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.88, allow_fake=True,
        dedup_lexical=False,
    ).to_dict()
    assert off_explicit == baseline
    assert baseline["lexical_merge_pairs"] == 0
    assert baseline["lexical_clusters_merged"] == 0


def test_lexical_merge_runs_before_prong_a_split(monkeypatch):
    """PRONG-A composition: the lexical merge runs BEFORE the distinct-skill split.

    Spy on both passes to assert ordering (E must run first so a wrongly-chained
    cosine cluster can still be split afterward), and confirm both fire when both
    flags are on.
    """
    call_order = []
    real_merge = od.merge_clusters_lexical
    real_split = od.split_clusters_for_distinct_skills

    def _spy_merge(*a, **k):
        call_order.append("merge")
        return real_merge(*a, **k)

    def _spy_split(*a, **k):
        call_order.append("split")
        return real_split(*a, **k)

    monkeypatch.setattr(od, "merge_clusters_lexical", _spy_merge)
    monkeypatch.setattr(od, "split_clusters_for_distinct_skills", _spy_split)

    grounded = [
        _lex_cand("Understand associative property here.", ["c1"], ent=0.95),
        _lex_cand("Understand associative property now.", ["c2"], ent=0.90),
        _lex_cand("Understand commutative property today.", ["c3"], ent=0.85),
    ]
    dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.99, allow_fake=True,
        dedup_lexical=True, distinct_skill_split=True,
    )
    assert call_order == ["merge", "split"], call_order


def test_lexical_merge_fires_under_split_and_stamps(monkeypatch):
    """With a high cosine threshold the cosine pass leaves restatements as
    separate clusters; the lexical pass merges them + stamps the rep, even with
    the PRONG-A split also enabled."""
    monkeypatch.setenv("ED4ALL_OBJECTIVE_DEDUP_LEXICAL_COSINE", "0.5")
    monkeypatch.setenv("ED4ALL_OBJECTIVE_DEDUP_LEXICAL_JACCARD", "0.5")
    grounded = [
        _lex_cand("Understand associative property here.", ["c1"], ent=0.95),
        _lex_cand("Understand associative property now.", ["c2"], ent=0.90),
        _lex_cand("Compute the triangle area formula.", ["c3"], ent=0.85, bloom="apply"),
    ]
    # threshold 0.99 → the two associative restatements do NOT co-cluster in the
    # cosine pass; the lexical pass merges them.
    result = dedup_candidates(
        grounded, embed=FakeEmbed(), threshold=0.99, allow_fake=True,
        dedup_lexical=True, distinct_skill_split=True,
    )
    assert result.lexical_merge_pairs >= 1
    assert result.lexical_clusters_merged >= 1
    assert any(
        c.get("dedup_lexical_merged_count", 0) >= 2 for c in result.canonical
    )
