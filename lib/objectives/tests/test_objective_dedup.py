"""W2 §5.4 — Pass D embedding-dedup invariants (hermetic fake embed)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.objective_dedup import dedup_candidates  # noqa: E402
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


def test_embeddings_absent_passthrough():
    """build_embedding_client raising → pass-through (each its own cluster)."""
    import lib.objectives.objective_dedup as od

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
