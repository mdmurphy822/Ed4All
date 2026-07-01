"""W1b.5 — heuristic LO linker (existing-ids-only) unit tests."""
from __future__ import annotations

from lib.decision_capture import DecisionCapture
from lib.ontology.lo_heuristic_link import (
    heuristic_link_chunk,
    heuristic_link_chunks,
    resolve_lo_heuristic_enabled,
)

_OBJECTIVES = [
    {"id": "CO-01", "statement": "Explain how a vector index enables semantic retrieval of chunks."},
    {"id": "CO-02", "statement": "Describe SHACL node shapes and property constraints for validation."},
    {"id": "CO-03", "statement": "Understand OWL."},  # too few content tokens
]


def test_resolver_default_off():
    assert resolve_lo_heuristic_enabled({}) is False
    assert resolve_lo_heuristic_enabled({"ED4ALL_CHUNK_LO_HEURISTIC": "on"}) is True
    assert resolve_lo_heuristic_enabled({"ED4ALL_CHUNK_LO_HEURISTIC": "nope"}) is False


def test_heuristic_link_chunk_matches_by_content_overlap():
    chunk = (
        "A vector index stores embeddings so semantic retrieval can rank chunks "
        "by similarity. Building the index is the first step."
    )
    matched = heuristic_link_chunk(chunk, _OBJECTIVES)
    assert matched == ["CO-01"]


def test_heuristic_link_only_returns_existing_ids():
    chunk = "This passage mentions TO-99 and CO-42 which are NOT in the universe."
    # Even though the chunk names LO-shaped ids, the linker can only return ids
    # from the supplied objective set (anti-fabrication).
    matched = heuristic_link_chunk(chunk, _OBJECTIVES)
    assert all(m in {"CO-01", "CO-02"} for m in matched)


def test_generic_objective_skipped():
    chunk = "OWL is a web ontology language used across many owl examples here today."
    # CO-03 has < 3 content tokens, so it is never linked.
    assert "CO-03" not in heuristic_link_chunk(chunk, _OBJECTIVES)


def test_heuristic_link_chunks_only_when_empty_and_fires_capture():
    chunks = [
        {"id": "k1", "text": "A vector index enables semantic retrieval of chunks by similarity."},
        {"id": "k2", "text": "unrelated cooking recipe about onions and garlic simmering slowly."},
        {"id": "k3", "text": "SHACL node shapes validate graphs.", "learning_outcome_refs": ["CO-09"]},
    ]
    cap = DecisionCapture(course_code="TEST_101", phase="textbook-ingestor",
                          tool="trainforge", streaming=False)
    summary = heuristic_link_chunks(chunks, _OBJECTIVES, capture=cap)

    assert chunks[0]["learning_outcome_refs"] == ["CO-01"]
    assert not chunks[1].get("learning_outcome_refs")
    # Pre-populated chunk is untouched (only_when_empty).
    assert chunks[2]["learning_outcome_refs"] == ["CO-09"]
    assert summary["chunks_linked"] == 1
    # Regression: the scan emits exactly one decision-capture event.
    assert any(d.get("decision_type") == "content_selection" for d in cap.decisions)
