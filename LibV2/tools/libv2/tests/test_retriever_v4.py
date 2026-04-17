"""Worker J tests: v4 ChunkFilter fields, structured tokenizer, rationale,
retrieval_text awareness, back-compat guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from LibV2.tools.libv2.retriever import (
    ChunkFilter,
    LazyBM25,
    RetrievalResult,
    _matches_filter,
    _parse_week_num,
    retrieve_chunks,
    tokenize,
)


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------

def _make_course(tmp_path: Path, slug: str, chunks: List[dict]) -> Path:
    """Write a minimal course dir with chunks.jsonl + manifest.json."""
    course_dir = tmp_path / "courses" / slug
    (course_dir / "corpus").mkdir(parents=True)
    with open(course_dir / "corpus" / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    with open(course_dir / "manifest.json", "w") as f:
        json.dump({"classification": {"primary_domain": "test-domain"}}, f)
    # Minimal catalog so catalog-less paths can still resolve the slug.
    return course_dir


# ---------------------------------------------------------------------------
# v4 ChunkFilter fields
# ---------------------------------------------------------------------------

class TestV4ChunkFilterFields:
    def test_teaching_role_filter(self):
        chunk = {"teaching_role": "transfer", "source": {}}
        assert _matches_filter(chunk, ChunkFilter(teaching_role="transfer"))
        assert not _matches_filter(chunk, ChunkFilter(teaching_role="assess"))

    def test_content_type_label_filter(self):
        chunk = {"content_type_label": "explanation", "source": {}}
        assert _matches_filter(chunk, ChunkFilter(content_type_label="explanation"))
        assert not _matches_filter(chunk, ChunkFilter(content_type_label="example"))

    def test_module_id_filter(self):
        chunk = {"source": {"module_id": "week_03_content_01"}}
        assert _matches_filter(chunk, ChunkFilter(module_id="week_03_content_01"))
        assert not _matches_filter(chunk, ChunkFilter(module_id="week_01_content_01"))

    def test_week_num_filter(self):
        chunk = {"source": {"module_id": "week_07_application"}}
        assert _matches_filter(chunk, ChunkFilter(week_num=7))
        assert not _matches_filter(chunk, ChunkFilter(week_num=5))

    def test_parse_week_num(self):
        assert _parse_week_num("week_07_application") == 7
        assert _parse_week_num("week-12-overview") == 12
        assert _parse_week_num("overview") is None
        assert _parse_week_num(None) is None


# ---------------------------------------------------------------------------
# Structured tokenizer
# ---------------------------------------------------------------------------

class TestStructuredTokenizer:
    def test_preserves_hyphenated_slugs(self):
        tokens = tokenize("Use aria-labelledby and skip-link for focus-indicator")
        assert "aria-labelledby" in tokens
        assert "skip-link" in tokens
        assert "focus-indicator" in tokens
        # Hyphenated slugs must NOT also appear as bare tokens.
        assert "labelledby" not in tokens
        assert "link" not in tokens

    def test_preserves_wcag_sc_refs(self):
        tokens = tokenize("WCAG 2.2 SC 1.4.3 is the contrast criterion")
        # Query-side normalization happens in _canonicalize_query, not tokenize.
        # tokenize sees pre-normalized input, so ask it to handle the slug form:
        tokens_norm = tokenize("wcag-2.2 sc-1.4.3 is the contrast criterion")
        assert "wcag-2.2" in tokens_norm
        assert "sc-1.4.3" in tokens_norm

    def test_legacy_tokenization_still_available(self):
        """Setting structured_tokens=False reproduces pre-Worker-J behavior."""
        tokens = tokenize("aria-labelledby", structured_tokens=False)
        assert "aria" in tokens
        assert "labelledby" in tokens
        assert "aria-labelledby" not in tokens


# ---------------------------------------------------------------------------
# retrieval_text-aware indexing
# ---------------------------------------------------------------------------

class TestRetrievalTextIndexing:
    def test_indexes_retrieval_text_when_present(self):
        """BM25 should match against retrieval_text if the chunk carries one."""
        chunks = [
            {"text": "long body mentioning banana", "retrieval_text": "summary: unicorn"},
            {"text": "another chunk about banana"},
        ]
        idx = LazyBM25(chunks, use_retrieval_text=True)
        # Query "unicorn" should hit chunk 0 via retrieval_text, not chunk 1
        results = idx.search("unicorn", min_relevance=0.0)
        assert results
        assert results[0][0] is chunks[0]

    def test_falls_back_to_text_when_no_retrieval_text(self):
        chunks = [{"text": "chunk mentions gerbil"}]
        idx = LazyBM25(chunks, use_retrieval_text=True)
        results = idx.search("gerbil", min_relevance=0.0)
        assert results

    def test_use_retrieval_text_false_ignores_field(self):
        """Legacy callers passing use_retrieval_text=False index chunk.text only."""
        chunks = [
            {"text": "body about banana", "retrieval_text": "summary unicorn"},
        ]
        idx = LazyBM25(chunks, use_retrieval_text=False)
        # "unicorn" should NOT match when we're forced to use chunk.text
        results_unicorn = idx.search("unicorn", min_relevance=0.0)
        results_banana = idx.search("banana", min_relevance=0.0)
        assert not results_unicorn
        assert results_banana


# ---------------------------------------------------------------------------
# Rationale payload
# ---------------------------------------------------------------------------

class TestRationalePayload:
    def test_rationale_keys_present_when_enabled(self, tmp_path):
        chunks = [
            {
                "id": "c1", "text": "alpha bravo charlie",
                "chunk_type": "explanation", "difficulty": "foundational",
                "concept_tags": ["alpha", "bravo"],
                "learning_outcome_refs": ["co-01"],
                "source": {"module_id": "week_01_intro"},
            },
        ]
        _make_course(tmp_path, "fx", chunks)
        # Minimal concept graph so the boost path runs without error
        graph_dir = tmp_path / "courses" / "fx" / "graph"
        graph_dir.mkdir()
        (graph_dir / "concept_graph.json").write_text(json.dumps({
            "kind": "concept",
            "nodes": [{"id": "alpha", "label": "A", "frequency": 1}],
            "edges": [],
        }))
        # course.json with one outcome
        (tmp_path / "courses" / "fx" / "course.json").write_text(json.dumps({
            "learning_outcomes": [
                {"id": "co-01", "statement": "alpha bravo charlie concept"}
            ],
        }))
        results = retrieve_chunks(
            tmp_path, "alpha", course_slug="fx", limit=5,
            include_rationale=True, min_relevance=0.0,
        )
        assert results
        r = results[0].rationale
        assert r is not None
        for key in (
            "bm25_score", "ngram_score", "metadata_boost", "final_score",
            "matched_concept_tags", "matched_lo_refs", "matched_key_terms",
            "applied_filters", "boost_contributions",
        ):
            assert key in r, f"missing rationale key: {key}"
        # boost_contributions sub-shape
        for k in ("concept_graph_overlap", "lo_match", "prereq_coverage"):
            assert k in r["boost_contributions"]

    def test_rationale_absent_when_disabled_backcompat(self, tmp_path):
        """include_rationale=False must produce to_dict output with NO
        rationale key at all (byte-identical to pre-Worker-J)."""
        chunks = [
            {"id": "c1", "text": "alpha bravo", "chunk_type": "explanation",
             "concept_tags": [], "learning_outcome_refs": [],
             "source": {"module_id": "week_01_intro"}, "difficulty": "foundational",
             "tokens_estimate": 10},
        ]
        _make_course(tmp_path, "fx", chunks)
        results = retrieve_chunks(
            tmp_path, "alpha", course_slug="fx", limit=5,
            include_rationale=False, min_relevance=0.0,
        )
        assert results
        d = results[0].to_dict()
        assert "rationale" not in d, "rationale key leaked into back-compat output"
        # Confirm the keys match the pre-Worker-J public schema exactly.
        expected = {
            "chunk_id", "text", "score", "course_slug", "domain", "chunk_type",
            "difficulty", "concept_tags", "source", "tokens_estimate",
            "learning_outcome_refs", "bloom_level",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Worker B flow-metrics tests did the same for quality_report; this one does
# it for retrieval output — byte-identical dict shape when flag is off.
# ---------------------------------------------------------------------------

class TestBackCompatProductionCallerShape:
    def test_retrieval_result_dataclass_has_rationale_optional(self):
        rr = RetrievalResult(
            chunk_id="c", text="t", score=1.0, course_slug="s",
            domain="d", chunk_type="ct", difficulty=None,
            concept_tags=[], source={},
        )
        assert rr.rationale is None
        # Attribute-style access used by Trainforge/rag/libv2_bridge.py
        assert rr.chunk_id == "c"
        assert rr.text == "t"
        assert rr.score == 1.0
