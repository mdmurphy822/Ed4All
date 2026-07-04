"""W2 §5.3 — Pass C grounding-filter invariants (hermetic fake NLI)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.objective_grounding import ground_candidates  # noqa: E402
from lib.objectives.tests._fakes import FakeNli  # noqa: E402


def _candidate(statement: str, chunk_ids, chapter_id="ch1") -> dict:
    return {
        "statement": statement,
        "source_chunk_ids": list(chunk_ids),
        "chapter_id": chapter_id,
    }


def test_grounded_vs_ungrounded_split_and_drop():
    """High-entailment candidate stays grounded; mismatched one is ungrounded."""
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Photosynthesis converts sunlight into glucose energy."},
        "c2": {"id": "c2", "text": "Mitochondria produce cellular respiration energy."},
    }
    candidates = [
        # Statement shares keywords with c1 → entailed.
        _candidate("Explain how photosynthesis converts sunlight into glucose.", ["c1"]),
        # Statement about an unrelated topic; cited chunk c2 shares nothing.
        _candidate("Describe the rules of medieval heraldry banners.", ["c2"]),
    ]
    nli = FakeNli()
    result = ground_candidates(candidates, chunks_by_id, nli=nli)
    assert result.available is True
    grounded_statements = {c["statement"] for c in result.grounded}
    assert "Explain how photosynthesis converts sunlight into glucose." in grounded_statements
    # The heraldry candidate has no keyword overlap with its cited chunk → dropped.
    assert any(
        "heraldry" in c["statement"] for c in result.ungrounded
    )
    assert result.dropped_count == len(result.ungrounded)


def test_per_chunk_prune_drops_non_entailing_cited_chunk():
    """Fix 1B — a grounded candidate keeps only the entailing cited chunk.

    The candidate cites two chunks; only c1 entails its statement. The
    per-chunk prune drops c2 (non-entailing) and keeps c1 (≥1 invariant).
    """
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Photosynthesis converts sunlight into glucose energy."},
        "c2": {"id": "c2", "text": "Medieval heraldry banners and coats of arms history."},
    }
    candidates = [
        _candidate(
            "Explain how photosynthesis converts sunlight into glucose.",
            ["c1", "c2"],
        ),
    ]
    nli = FakeNli()
    result = ground_candidates(candidates, chunks_by_id, nli=nli)
    assert len(result.grounded) == 1
    g = result.grounded[0]
    assert g["grounded"] is True
    kept = g["source_chunk_ids"]
    # Non-entailing c2 pruned; entailing c1 kept (≥1 invariant).
    assert kept == ["c1"]
    assert g.get("grounding_chunk_pruned") is True
    # source_refs resynced to match.
    assert g["source_refs"][0]["chunk_ids"] == ["c1"]


def test_per_chunk_prune_keeps_at_least_one():
    """Fix 1B — a single cited entailing chunk is never pruned away."""
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Photosynthesis converts sunlight into glucose energy."},
    }
    candidates = [
        _candidate("Explain how photosynthesis converts sunlight into glucose.", ["c1"]),
    ]
    result = ground_candidates(candidates, chunks_by_id, nli=FakeNli())
    assert len(result.grounded) == 1
    assert result.grounded[0]["source_chunk_ids"] == ["c1"]
    # No prune flag when nothing was dropped.
    assert "grounding_chunk_pruned" not in result.grounded[0]


def test_nli_absent_passthrough(monkeypatch):
    """NLI unavailable (extras absent) → pass-through, nothing dropped.

    Force the absent path deterministically by stubbing the NLI resolver to
    return None — env-portable (passes whether or not transformers/torch are
    installed in the running environment).
    """
    import lib.retrieval.groundedness as _g

    monkeypatch.setattr(_g, "_resolve_nli", lambda nli=None: None)
    chunks_by_id = {"c1": {"id": "c1", "text": "anything"}}
    candidates = [_candidate("a statement here", ["c1"])]

    # nli=None + resolver→None → score_groundedness available=False → Pass C
    # pass-through (no candidate dropped).
    result = ground_candidates(candidates, chunks_by_id, nli=None, require=False)
    assert result.available is False
    assert result.dropped_count == 0
    assert len(result.grounded) == 1
    assert result.grounded[0]["grounded"] is None
    assert result.grounded[0]["entailment_score"] is None


def test_reground_rescue_adopts_adjacent_chunk():
    """An ungrounded candidate is rescued by an entailing adjacent chapter chunk."""
    chunks_by_id = {
        # Cited chunk shares nothing with the statement.
        "c1": {"id": "c1", "text": "Unrelated boilerplate navigation footer text.", "chapter_id": "ch1"},
        # Adjacent (non-cited) chapter chunk DOES entail the statement.
        "c2": {"id": "c2", "text": "Quadratic equations factor into binomial roots cleanly.", "chapter_id": "ch1"},
    }
    candidates = [
        _candidate("Show how quadratic equations factor into binomial roots.", ["c1"]),
    ]
    nli = FakeNli()
    result = ground_candidates(candidates, chunks_by_id, nli=nli, reground=True)
    assert len(result.grounded) == 1
    rescued = result.grounded[0]
    assert rescued.get("reground") is True
    assert "c2" in rescued["source_chunk_ids"]
    assert result.reground_count == 1


# ---------------------------------------------------------------------------
# entailing_chunk_id stamp (2026-07-04 — the pin downstream citation
# reshapers honor so the entailment evidence survives dedup/reselect)
# ---------------------------------------------------------------------------
def test_cited_path_stamps_entailing_chunk_id():
    """cited_chunk_entailed → the argmax-entailing chunk id is stamped."""
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Photosynthesis converts sunlight into glucose energy."},
        "c2": {"id": "c2", "text": "Medieval heraldry banners and coats of arms history."},
    }
    candidates = [
        _candidate(
            "Explain how photosynthesis converts sunlight into glucose.",
            ["c1", "c2"],
        ),
    ]
    result = ground_candidates(candidates, chunks_by_id, nli=FakeNli())
    assert len(result.grounded) == 1
    g = result.grounded[0]
    assert g["grounding_reason"] == "cited_chunk_entailed"
    assert g["entailing_chunk_id"] == "c1"
    # The pin is always among the surviving citations.
    assert g["entailing_chunk_id"] in g["source_chunk_ids"]


def test_reground_path_stamps_entailing_chunk_id():
    """reground_adjacent_chunk → the ADOPTED chunk id is stamped."""
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Unrelated boilerplate navigation footer text.", "chapter_id": "ch1"},
        "c2": {"id": "c2", "text": "Quadratic equations factor into binomial roots cleanly.", "chapter_id": "ch1"},
    }
    candidates = [
        _candidate("Show how quadratic equations factor into binomial roots.", ["c1"]),
    ]
    result = ground_candidates(candidates, chunks_by_id, nli=FakeNli(), reground=True)
    assert len(result.grounded) == 1
    rescued = result.grounded[0]
    assert rescued["grounding_reason"] == "reground_adjacent_chunk"
    assert rescued["entailing_chunk_id"] == "c2"
    assert rescued["entailing_chunk_id"] in rescued["source_chunk_ids"]


def test_ungrounded_candidate_has_no_entailing_chunk_id():
    """A dropped candidate never carries the stamp (nothing entailed)."""
    chunks_by_id = {
        "c2": {"id": "c2", "text": "Mitochondria produce cellular respiration energy."},
    }
    candidates = [
        _candidate("Describe the rules of medieval heraldry banners.", ["c2"]),
    ]
    result = ground_candidates(candidates, chunks_by_id, nli=FakeNli(), reground=False)
    assert len(result.ungrounded) == 1
    assert "entailing_chunk_id" not in result.ungrounded[0]
