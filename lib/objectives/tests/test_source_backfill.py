"""I3 PRONG B — source-richness backfill invariants (hermetic).

Verifies: (1) a dedup-discarded order-of-operations candidate is PROMOTED so it
names the skill + drops the uncovered-chunk count, (2) the anti-fabrication
contract (every backfilled CO's source_chunk_ids ⊆ the promoted pre-dedup
candidate's), (3) default-OFF is a no-op, (4) the content-bearing filter excludes
exercises / assessment_items / sub-floor stubs.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.source_backfill import (  # noqa: E402
    backfill_uncovered_chunks,
    is_content_bearing,
)


def _cand(statement, chunk_ids, ent=0.9, chapter="ch1"):
    return {
        "statement": statement,
        "source_chunk_ids": list(chunk_ids),
        "entailment_score": ent,
        "chapter_id": chapter,
    }


def _chunk(cid, ctype="explanation", words=60):
    return {
        "id": cid,
        "chunk_type": ctype,
        "word_count": words,
        "text": "teachable prose content " * (words // 3),
    }


def _chunks_by_id(*chunks):
    return {c["id"]: c for c in chunks}


def test_backfill_promotes_discarded_order_of_operations():
    """Dedup collapsed order-of-operations into 'simplify'; backfill re-surfaces it.

    Pre-dedup pool has BOTH a 'simplify expressions' candidate (kept rep) and an
    'order of operations' candidate (discarded — its chunk demoted to evidence).
    The canonical set names only 'simplify'; chunk c_oo is uncited. Backfill
    promotes the discarded order-of-operations candidate.
    """
    pre_dedup = [
        _cand("Simplify algebraic expressions using properties", ["c_simp", "c_oo"]),
        _cand("Apply the order of operations to evaluate expressions", ["c_oo"]),
    ]
    # Canonical (post-dedup): only the 'simplify' rep survives with the union,
    # but it does NOT NAME order-of-operations.
    canonical = [
        _cand("Simplify algebraic expressions using properties", ["c_simp"]),
    ]
    chunks = _chunks_by_id(_chunk("c_simp"), _chunk("c_oo"))

    result = backfill_uncovered_chunks(
        canonical=canonical,
        pre_dedup_candidates=pre_dedup,
        chunks_by_id=chunks,
        enabled=True,
    )
    assert result.available is True
    # c_oo was uncovered before; the promotion covers it.
    assert result.uncovered_before == 1
    assert result.uncovered_after == 0
    assert result.cos_promoted == 1
    # A backfilled CO now NAMES order-of-operations.
    statements = " || ".join(c["statement"].lower() for c in canonical)
    assert "order of operations" in statements


def test_backfill_promoted_source_refs_uses_real_chapter_not_self_ref():
    """FIX B coverage parity with FIX A: a promoted CO's ``source_refs`` entry
    carries the candidate's REAL ``chapter_id`` as ``ref`` (a resolvable chapter
    label), NOT a self-referential / fabricated id, and its chunk_ids subset the
    candidate's. Confirms the source-coverage backfill path is reachable and
    emits a gate-clean source_ref when enabled.
    """
    pre_dedup = [
        _cand("Simplify expressions broadly", ["c_simp"], chapter="ch1"),
        _cand("Define an ontology as a shared conceptualization",
              ["c_onto"], chapter="ch3"),
    ]
    canonical = [_cand("Simplify expressions broadly", ["c_simp"], chapter="ch1")]
    chunks = _chunks_by_id(_chunk("c_simp"), _chunk("c_onto"))

    result = backfill_uncovered_chunks(
        canonical=canonical,
        pre_dedup_candidates=pre_dedup,
        chunks_by_id=chunks,
        enabled=True,
    )
    assert result.cos_promoted == 1
    promoted = [c for c in canonical if c.get("backfilled")]
    assert len(promoted) == 1
    co = promoted[0]
    refs = co["source_refs"]
    assert refs == [{"ref": "ch3", "chunk_ids": ["c_onto"]}]
    # ref is the candidate's real chapter, never the CO's own id.
    assert refs[0]["ref"] == "ch3"
    assert set(co["source_chunk_ids"]) == {"c_onto"}


def test_backfill_anti_fabrication_subset():
    """Every backfilled CO's source_chunk_ids ⊆ the promoted candidate's."""
    cand_ids = ["c_oo", "c_extra"]
    pre_dedup = [
        _cand("Simplify expressions broadly", ["c_simp"]),
        _cand("Apply the order of operations rule", cand_ids),
    ]
    canonical = [_cand("Simplify expressions broadly", ["c_simp"])]
    chunks = _chunks_by_id(
        _chunk("c_simp"), _chunk("c_oo"), _chunk("c_extra"),
    )
    backfill_uncovered_chunks(
        canonical=canonical,
        pre_dedup_candidates=pre_dedup,
        chunks_by_id=chunks,
        enabled=True,
    )
    promoted = [c for c in canonical if c.get("backfilled")]
    assert promoted, "expected a backfilled CO"
    for co in promoted:
        assert set(co["source_chunk_ids"]).issubset(set(cand_ids))


def test_backfill_default_off_no_op():
    """enabled=None + env unset → no promotion (byte-stable)."""
    pre_dedup = [_cand("Apply order of operations", ["c_oo"])]
    canonical = [_cand("Simplify expressions", ["c_simp"])]
    chunks = _chunks_by_id(_chunk("c_simp"), _chunk("c_oo"))
    before = len(canonical)
    result = backfill_uncovered_chunks(
        canonical=canonical,
        pre_dedup_candidates=pre_dedup,
        chunks_by_id=chunks,
        enabled=None,  # env unset in test env → default OFF
    )
    assert result.available is False
    assert len(canonical) == before
    assert result.cos_promoted == 0


def test_backfill_skips_non_content_chunks():
    """An uncited EXERCISE chunk is NOT backfilled (no CO for non-instructional)."""
    pre_dedup = [
        _cand("Practice exercises set", ["c_ex"]),
    ]
    canonical = [_cand("Simplify expressions", ["c_simp"])]
    chunks = _chunks_by_id(
        _chunk("c_simp"),
        _chunk("c_ex", ctype="exercise", words=200),  # not content-bearing
    )
    result = backfill_uncovered_chunks(
        canonical=canonical,
        pre_dedup_candidates=pre_dedup,
        chunks_by_id=chunks,
        enabled=True,
    )
    # Only c_simp is content-bearing and it IS covered → nothing to backfill.
    assert result.content_bearing_chunks == 1
    assert result.cos_promoted == 0


def test_is_content_bearing_filter():
    assert is_content_bearing(_chunk("a", "explanation", 60)) is True
    assert is_content_bearing(_chunk("b", "exercise", 200)) is False
    assert is_content_bearing(_chunk("c", "assessment_item", 200)) is False
    assert is_content_bearing(_chunk("d", "explanation", 10)) is False  # stub
    # No chunk_type → word floor only.
    assert is_content_bearing({"id": "e", "word_count": 60, "text": "x"}) is True
