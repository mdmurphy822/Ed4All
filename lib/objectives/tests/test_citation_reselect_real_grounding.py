"""REAL-grounding invariants for the fabricated-citation re-selection fix.

The local stage-2 synthesizer (7B / nano-omni) sometimes cites FABRICATED
chunk ids — descriptive topic-labels ("Round Whole Numbers") that resolve
against nothing in the real chunkset — leaving objectives with ~0 real source
grounding. The owner ruled "0 real grounding is unacceptable". These tests prove
``reselect_citations`` (gated by ``ED4ALL_OBJECTIVE_CITATION_RESELECT``) rebuilds
each objective's candidate pool from the REAL window / chapter chunks the model
SAW — independent of what it cited — and re-cites the cosine-best real chunk, so
``ObjectiveSourceRefValidator`` passes with REAL grounding (not the sanitizer's
drop-to-empty floor).

Hermetic — ``FakeEmbed`` (token-hash unit vectors, no torch/numpy), no LLM,
no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.citation_reselect import reselect_citations  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402
from lib.validators.objective_source_refs import (  # noqa: E402
    ObjectiveSourceRefValidator,
)


# ---------------------------------------------------------------------------
# Real chunk universe: c_round matches the "round whole numbers" statement;
# c_order matches the "order of operations" statement; c_off is off-topic.
# ---------------------------------------------------------------------------
def _chunks() -> Dict[str, Dict[str, Any]]:
    return {
        "demo_chunk_00001": {
            "id": "demo_chunk_00001",
            "text": (
                "round whole numbers to a given place value rounding "
                "nearest ten hundred thousand"
            ),
            "chapter_id": "ch1",
        },
        "demo_chunk_00002": {
            "id": "demo_chunk_00002",
            "text": (
                "order of operations pemdas parentheses exponents "
                "multiplication division addition subtraction"
            ),
            "chapter_id": "ch1",
        },
        "demo_chunk_00003": {
            "id": "demo_chunk_00003",
            "text": "photosynthesis chloroplast light energy pigment leaf",
            "chapter_id": "ch1",
        },
    }


def _co(
    lo_id: str, statement: str, fabricated: List[str], chapter_id: str = "ch1"
) -> Dict[str, Any]:
    """A CO whose citations are FABRICATED topic-labels (resolve against
    nothing in the real chunk universe)."""
    return {
        "id": lo_id,
        "statement": statement,
        "chapter_id": chapter_id,
        "source_chunk_ids": list(fabricated),
        "source_refs": [{"ref": chapter_id, "chunk_ids": list(fabricated)}],
    }


_ALL_REAL = ["demo_chunk_00001", "demo_chunk_00002", "demo_chunk_00003"]


# ---------------------------------------------------------------------------
# (a) fabricated citations replaced with the cosine-best REAL chunk.
# ---------------------------------------------------------------------------
def test_fabricated_citations_replaced_with_real():
    chunks = _chunks()
    # The model invented "Round Whole Numbers" as the cited id.
    co = _co("CO-01", "Round whole numbers to a given place value.",
             ["Round Whole Numbers"])
    res = reselect_citations(
        [co], chunks, FakeEmbed(),
        # The real window pool the model SAW (stamped Pass-B provenance).
        window_chunk_ids_by_co={0: list(_ALL_REAL)},
        enabled=True,
    )
    assert res.available
    assert res.reselected_count == 1
    # The fabricated id is GONE; the cosine-best REAL chunk is cited.
    assert "Round Whole Numbers" not in co["source_chunk_ids"]
    assert co["source_chunk_ids"][0] == "demo_chunk_00001"
    assert all(cid in chunks for cid in co["source_chunk_ids"])
    # The fabricated id was dropped as an unresolvable pool miss.
    assert res.pool_misses == 1
    # Grounding is now > 0.
    assert res.citation_density_after >= 1
    # source_refs mirrored onto the kept real chunks.
    assert co["source_refs"][0]["chunk_ids"] == co["source_chunk_ids"]


def test_second_objective_grounds_its_own_topic():
    """Two fabricated COs each re-cite THEIR topic's real chunk (not the
    same one) — proves cosine, not a constant."""
    chunks = _chunks()
    co_round = _co("CO-01", "Round whole numbers to a given place value.",
                   ["Round Whole Numbers"])
    co_order = _co("CO-02", "Apply the order of operations to expressions.",
                   ["Order of Operations"])
    res = reselect_citations(
        [co_round, co_order], chunks, FakeEmbed(),
        window_chunk_ids_by_co={0: list(_ALL_REAL), 1: list(_ALL_REAL)},
        enabled=True,
    )
    assert res.available
    assert co_round["source_chunk_ids"][0] == "demo_chunk_00001"
    assert co_order["source_chunk_ids"][0] == "demo_chunk_00002"


# ---------------------------------------------------------------------------
# (b) zero-citation CO grounded from the provided real pool.
# ---------------------------------------------------------------------------
def test_zero_citation_grounded_from_provided_pool():
    chunks = _chunks()
    co = {
        "id": "CO-03",
        "statement": "Round whole numbers to a given place value.",
        "chapter_id": "ch1",
        "source_chunk_ids": [],  # sanitizer already dropped everything
        "source_refs": [],
    }
    res = reselect_citations(
        [co], chunks, FakeEmbed(),
        window_chunk_ids_by_co={0: list(_ALL_REAL)},
        enabled=True,
    )
    assert res.available
    assert res.skipped_no_citation == 0  # NOT skipped — real pool supplied
    assert co["source_chunk_ids"][0] == "demo_chunk_00001"


# ---------------------------------------------------------------------------
# (c) fabricated CO with NO provided pool stays skipped (never fabricates).
# ---------------------------------------------------------------------------
def test_fabricated_no_pool_is_skipped_not_invented():
    chunks = _chunks()
    co = _co("CO-04", "Round whole numbers.", ["Round Whole Numbers"])
    res = reselect_citations(
        [co], chunks, FakeEmbed(),
        window_chunk_ids_by_co={},   # no real pool for this CO
        chapter_chunks_by_co={},
        enabled=True,
    )
    # Pool resolves to nothing real -> the CO is not scored; its (fabricated)
    # citations are left untouched here (the sanitizer backstop drops them).
    assert res.reselected_count == 0
    assert co["source_chunk_ids"] == ["Round Whole Numbers"]


# ---------------------------------------------------------------------------
# (d) INTEGRATION — the gate flips ORPHANED (block) -> pass with real grounding.
# ---------------------------------------------------------------------------
def _write_chunkset(tmp_path: Path, chunks: Dict[str, Dict[str, Any]]) -> Path:
    d = tmp_path / "dart_chunks"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps({"chunkset_kind": "dart"}), encoding="utf-8"
    )
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks.values():
            fh.write(json.dumps({"id": c["id"], "text": c["text"]}) + "\n")
    return d / "manifest.json"


def test_source_ref_validator_flips_block_to_pass_after_reselect(tmp_path):
    chunks = _chunks()
    manifest = _write_chunkset(tmp_path, chunks)

    co = _co("CO-01", "Round whole numbers to a given place value.",
             ["Round Whole Numbers"])
    validator = ObjectiveSourceRefValidator()

    # BEFORE reselect: every cited id is fabricated -> the aggregate
    # split-brain net fires CRITICAL ORPHANED_CITATIONS -> gate BLOCKS.
    before = validator.validate({
        "objectives": [dict(co)],  # copy so reselect below sees the fabricated form
        "dart_chunks_manifest_path": str(manifest),
    })
    assert before.passed is False
    assert any(i.code == "ORPHANED_CITATIONS" for i in before.issues)

    # Re-select from the REAL window pool.
    res = reselect_citations(
        [co], chunks, FakeEmbed(),
        window_chunk_ids_by_co={0: list(_ALL_REAL)},
        enabled=True,
    )
    assert res.available and res.reselected_count == 1

    # AFTER reselect: citations are REAL ids resolving in the chunkset -> the
    # gate PASSES with real grounding (no ORPHANED_CITATIONS, no unresolved).
    after = validator.validate({
        "objectives": [co],
        "dart_chunks_manifest_path": str(manifest),
    })
    assert after.passed is True
    assert not any(i.code == "ORPHANED_CITATIONS" for i in after.issues)
    assert not any(
        i.code == "OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST" for i in after.issues
    )
    # The objective genuinely cites a real chunk (grounding > 0), not an
    # empty/dropped citation set.
    assert co["source_refs"][0]["chunk_ids"], "grounding must be > 0 (real cite)"
    assert co["source_refs"][0]["chunk_ids"][0] in chunks


# ---------------------------------------------------------------------------
# (e) gate is OFF by default -> no-op (enabled=None reads env, unset = off).
# ---------------------------------------------------------------------------
def test_disabled_by_default_is_noop(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_CITATION_RESELECT", raising=False)
    chunks = _chunks()
    co = _co("CO-01", "Round whole numbers.", ["Round Whole Numbers"])
    res = reselect_citations(
        [co], chunks, FakeEmbed(),
        window_chunk_ids_by_co={0: list(_ALL_REAL)},
    )
    assert res.available is False
    assert co["source_chunk_ids"] == ["Round Whole Numbers"]  # untouched
