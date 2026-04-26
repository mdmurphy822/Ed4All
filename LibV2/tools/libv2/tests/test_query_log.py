"""Tests for the query/answer log persisted alongside LibV2 corpora.

Pin the storage contract that ``libv2 ask`` and ``libv2 answer`` rely on:

  1. Per-course queries land in ``courses/<slug>/queries/``; cross-
     course queries land in ``catalog/queries/`` so there's always a
     home alongside the source corpus.
  2. ``mint_query_id`` produces stable IDs from the query text + UTC
     timestamp (deterministic when the timestamp is supplied).
  3. ``write_query_record`` creates an open record + index entry;
     ``attach_answer`` flips the same record to ``status='answered'``.
  4. The compact-chunk projection keeps records small while preserving
     the identifying handles a reader needs to sanity-check matches.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from LibV2.tools.libv2.query_log import (
    CompactChunk,
    attach_answer,
    compact_retrieval_result,
    find_answered_query,
    index_path,
    list_queries,
    load_record,
    mint_query_id,
    query_path,
    resolve_storage_dir,
    write_query_record,
)


# ---------------------------------------------------------------------------
# Storage directory resolution
# ---------------------------------------------------------------------------


class TestStorageDirResolution:
    def test_per_course_lands_under_courses_slug_queries(self, tmp_path: Path):
        d = resolve_storage_dir(tmp_path, "my-course")
        assert d == tmp_path / "courses" / "my-course" / "queries"

    def test_cross_course_lands_under_catalog_queries(self, tmp_path: Path):
        d = resolve_storage_dir(tmp_path, None)
        assert d == tmp_path / "catalog" / "queries"


# ---------------------------------------------------------------------------
# mint_query_id — stability + uniqueness
# ---------------------------------------------------------------------------


class TestMintQueryId:
    def test_same_text_same_timestamp_same_id(self):
        ts = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        a = mint_query_id("How does SHACL validate?", ts=ts)
        b = mint_query_id("How does SHACL validate?", ts=ts)
        assert a == b

    def test_different_text_different_id(self):
        ts = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        a = mint_query_id("Question A", ts=ts)
        b = mint_query_id("Question B", ts=ts)
        assert a != b

    def test_id_format_is_q_underscore_date_time_hash(self):
        ts = datetime(2026, 4, 26, 12, 30, 45, tzinfo=timezone.utc)
        qid = mint_query_id("hello", ts=ts)
        assert qid.startswith("q_20260426_123045_")
        # hash suffix is 8 hex chars
        suffix = qid.rsplit("_", 1)[-1]
        assert len(suffix) == 8
        int(suffix, 16)  # must parse as hex


# ---------------------------------------------------------------------------
# compact_retrieval_result — projection of RetrievalResult-shaped objects
# ---------------------------------------------------------------------------


def _fake_result(
    chunk_id="c1",
    text="body text " * 50,
    score=1.234,
    course_slug="rdf-shacl-551-2",
    heading="Node Shapes",
    module_id="module_03",
    tags=("node-shape", "shacl", "property-shape"),
):
    return SimpleNamespace(
        chunk_id=chunk_id,
        text=text,
        score=score,
        course_slug=course_slug,
        source={"section_heading": heading, "module_id": module_id},
        concept_tags=list(tags),
    )


class TestCompactRetrievalResult:
    def test_projects_identifying_fields(self):
        r = _fake_result()
        c = compact_retrieval_result(r, rank=1)
        assert isinstance(c, CompactChunk)
        d = c.to_dict()
        assert d["rank"] == 1
        assert d["chunk_id"] == "c1"
        assert d["score"] == pytest.approx(1.234)
        assert d["course_slug"] == "rdf-shacl-551-2"
        assert d["section_heading"] == "Node Shapes"
        assert d["module_id"] == "module_03"
        assert d["concept_tags"] == ["node-shape", "shacl", "property-shape"]

    def test_snippet_is_capped_with_ellipsis(self):
        r = _fake_result(text="x" * 1000)
        c = compact_retrieval_result(r, rank=2, snippet_chars=400)
        d = c.to_dict()
        assert d["snippet"].endswith("...")
        # 400 chars body + 3 ellipsis = 403
        assert len(d["snippet"]) <= 403

    def test_short_text_no_ellipsis(self):
        r = _fake_result(text="short body")
        c = compact_retrieval_result(r, rank=1)
        assert c.to_dict()["snippet"] == "short body"

    def test_concept_tags_capped_at_eight(self):
        r = _fake_result(tags=tuple(f"tag-{i}" for i in range(20)))
        c = compact_retrieval_result(r, rank=1)
        assert len(c.to_dict()["concept_tags"]) == 8

    def test_handles_missing_source(self):
        r = SimpleNamespace(
            chunk_id="c1", text="x", score=0.0, course_slug="cs",
            source=None, concept_tags=[],
        )
        c = compact_retrieval_result(r, rank=1)
        d = c.to_dict()
        assert d["section_heading"] == ""
        assert d["module_id"] == ""


# ---------------------------------------------------------------------------
# write_query_record + attach_answer round-trip
# ---------------------------------------------------------------------------


class TestWriteAndAttachAnswer:
    def test_per_course_record_lands_under_course(self, tmp_path: Path):
        path = write_query_record(
            repo_root=tmp_path,
            course_slug="rdf-shacl-551-2",
            query_text="How do property paths work?",
            method="bm25+intent",
            limit=10,
            retrieved=[{"rank": 1, "chunk_id": "c1", "score": 1.0,
                        "course_slug": "rdf-shacl-551-2",
                        "section_heading": "h", "module_id": "m",
                        "concept_tags": [], "snippet": "..."}],
        )
        assert path.parent == tmp_path / "courses" / "rdf-shacl-551-2" / "queries"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == "open"
        assert record["scope"] == "course"
        assert record["course_slug"] == "rdf-shacl-551-2"
        assert record["answer"] is None
        assert record["asked_by"] == "claude"

    def test_cross_course_record_lands_under_catalog(self, tmp_path: Path):
        path = write_query_record(
            repo_root=tmp_path,
            course_slug=None,
            query_text="x",
            method="hybrid",
            limit=5,
            retrieved=[],
        )
        assert path.parent == tmp_path / "catalog" / "queries"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["scope"] == "cross-course"
        assert record["course_slug"] is None

    def test_index_entry_created_on_write(self, tmp_path: Path):
        write_query_record(
            repo_root=tmp_path,
            course_slug="course-a",
            query_text="q1",
            method="bm25",
            limit=10,
            retrieved=[],
        )
        idx_file = index_path(resolve_storage_dir(tmp_path, "course-a"))
        assert idx_file.exists()
        data = json.loads(idx_file.read_text(encoding="utf-8"))
        assert data["scope"] == "course"
        assert data["course_slug"] == "course-a"
        assert len(data["queries"]) == 1
        assert data["queries"][0]["status"] == "open"

    def test_attach_answer_flips_status_to_answered(self, tmp_path: Path):
        path = write_query_record(
            repo_root=tmp_path,
            course_slug="course-a",
            query_text="q1",
            method="bm25",
            limit=10,
            retrieved=[],
        )
        qid = json.loads(path.read_text(encoding="utf-8"))["query_id"]
        attach_answer(tmp_path, "course-a", qid, "synthesized answer text")
        record = load_record(tmp_path, "course-a", qid)
        assert record["status"] == "answered"
        assert record["answer"] == "synthesized answer text"
        assert record["answered_at"] is not None
        assert record["answered_by"] == "claude"

    def test_attach_answer_updates_index(self, tmp_path: Path):
        path = write_query_record(
            repo_root=tmp_path,
            course_slug="course-a",
            query_text="q1",
            method="bm25",
            limit=10,
            retrieved=[],
        )
        qid = json.loads(path.read_text(encoding="utf-8"))["query_id"]
        attach_answer(tmp_path, "course-a", qid, "ans")
        items = list_queries(tmp_path, "course-a")
        assert len(items) == 1
        assert items[0]["status"] == "answered"
        assert items[0]["answered_at"] is not None

    def test_attach_answer_missing_record_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            attach_answer(tmp_path, "course-a", "q_does_not_exist", "ans")

    def test_load_record_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_record(tmp_path, "course-a", "q_does_not_exist")

    def test_index_dedupes_on_repeated_writes_with_same_id(self, tmp_path: Path):
        # Reuse the same query_id explicitly to simulate an upsert.
        write_query_record(
            repo_root=tmp_path,
            course_slug="course-a",
            query_text="q1",
            method="bm25",
            limit=10,
            retrieved=[],
            query_id="q_fixed_id_aaaaaaaa",
        )
        write_query_record(
            repo_root=tmp_path,
            course_slug="course-a",
            query_text="q1 (re-asked)",
            method="bm25",
            limit=10,
            retrieved=[],
            query_id="q_fixed_id_aaaaaaaa",
        )
        items = list_queries(tmp_path, "course-a")
        assert len(items) == 1
        assert items[0]["query_text"] == "q1 (re-asked)"


# ---------------------------------------------------------------------------
# list_queries — ordering + scope isolation
# ---------------------------------------------------------------------------


class TestListQueries:
    def test_empty_when_no_index(self, tmp_path: Path):
        assert list_queries(tmp_path, "no-such-course") == []
        assert list_queries(tmp_path, None) == []

    def test_per_course_and_cross_course_isolated(self, tmp_path: Path):
        write_query_record(
            repo_root=tmp_path, course_slug="course-a",
            query_text="course question",
            method="bm25", limit=10, retrieved=[],
        )
        write_query_record(
            repo_root=tmp_path, course_slug=None,
            query_text="cross-course question",
            method="bm25", limit=10, retrieved=[],
        )
        per = list_queries(tmp_path, "course-a")
        cross = list_queries(tmp_path, None)
        assert len(per) == 1 and per[0]["query_text"] == "course question"
        assert len(cross) == 1 and cross[0]["query_text"] == "cross-course question"

    def test_entries_sorted_by_asked_at(self, tmp_path: Path):
        for i in range(3):
            write_query_record(
                repo_root=tmp_path, course_slug="course-a",
                query_text=f"q{i}",
                method="bm25", limit=10, retrieved=[],
                query_id=f"q_{i:08d}_aaaaaaaa",
            )
        items = list_queries(tmp_path, "course-a")
        asked = [q.get("asked_at") or "" for q in items]
        assert asked == sorted(asked)


# ---------------------------------------------------------------------------
# find_answered_query — cache lookup that prevents re-synthesis on re-ask
# ---------------------------------------------------------------------------


def _write_and_answer(tmp_path: Path, course: Optional[str], qid: str,
                      query_text: str, answer: str = "the answer") -> None:
    write_query_record(
        repo_root=tmp_path, course_slug=course,
        query_text=query_text, method="bm25", limit=10, retrieved=[],
        query_id=qid,
    )
    attach_answer(tmp_path, course, qid, answer)


class TestFindAnsweredQuery:
    def test_returns_cached_answered_record(self, tmp_path: Path):
        _write_and_answer(tmp_path, "course-a", "q_aaaa_aaaaaaaa",
                          "What is RDF?", "RDF is a triple-based graph model.")
        cached = find_answered_query(tmp_path, "course-a", "What is RDF?")
        assert cached is not None
        assert cached["answer"] == "RDF is a triple-based graph model."
        assert cached["status"] == "answered"

    def test_skips_open_unanswered_records(self, tmp_path: Path):
        # Record exists, never had an answer attached → not cache-eligible.
        write_query_record(
            repo_root=tmp_path, course_slug="course-a",
            query_text="Open question", method="bm25", limit=10, retrieved=[],
            query_id="q_open_aaaaaaaa",
        )
        assert find_answered_query(tmp_path, "course-a", "Open question") is None

    def test_normalizes_case_and_whitespace(self, tmp_path: Path):
        _write_and_answer(tmp_path, "course-a", "q_norm_aaaaaaaa",
                          "What is RDF?", "ans")
        # Same query, different casing + extra spaces — should still hit.
        assert find_answered_query(tmp_path, "course-a", "  WHAT IS RDF? ") is not None
        assert find_answered_query(tmp_path, "course-a", "what is rdf?") is not None
        # Internal whitespace differences also normalize.
        assert find_answered_query(tmp_path, "course-a", "What  is\tRDF?") is not None

    def test_distinct_queries_do_not_collide(self, tmp_path: Path):
        _write_and_answer(tmp_path, "course-a", "q_one_aaaaaaaa",
                          "What is RDF?", "ans-1")
        _write_and_answer(tmp_path, "course-a", "q_two_aaaaaaaa",
                          "What is RDFS?", "ans-2")
        c1 = find_answered_query(tmp_path, "course-a", "What is RDF?")
        c2 = find_answered_query(tmp_path, "course-a", "What is RDFS?")
        assert c1 is not None and c1["answer"] == "ans-1"
        assert c2 is not None and c2["answer"] == "ans-2"

    def test_returns_most_recent_when_multiple_answered(self, tmp_path: Path):
        _write_and_answer(tmp_path, "course-a", "q_old_aaaaaaaa",
                          "Same query", "old answer")
        _write_and_answer(tmp_path, "course-a", "q_new_aaaaaaaa",
                          "Same query", "new answer")
        cached = find_answered_query(tmp_path, "course-a", "Same query")
        assert cached is not None
        # Most-recent answered_at wins; both share status='answered'
        # but q_new was answered second.
        assert cached["answer"] == "new answer"

    def test_per_course_and_cross_course_caches_isolated(self, tmp_path: Path):
        _write_and_answer(tmp_path, "course-a", "q_perc_aaaaaaaa",
                          "ambiguous query", "course-a answer")
        _write_and_answer(tmp_path, None, "q_xc_aaaaaaaa",
                          "ambiguous query", "cross-course answer")
        per = find_answered_query(tmp_path, "course-a", "ambiguous query")
        cross = find_answered_query(tmp_path, None, "ambiguous query")
        assert per["answer"] == "course-a answer"
        assert cross["answer"] == "cross-course answer"

    def test_no_index_returns_none(self, tmp_path: Path):
        # Empty repo — no queries dir, no index file.
        assert find_answered_query(tmp_path, "no-such-course", "anything") is None
        assert find_answered_query(tmp_path, None, "anything") is None
