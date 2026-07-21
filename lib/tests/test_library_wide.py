"""Tests for library-wide grounded ask (W4.2, ED4ALL_ANSWER_LIBRARY_WIDE).

Deterministic + CI-safe: NO model, NO server, NO network, NO embedding/vector
index. Retrieval is monkeypatched to a synthetic per-course fake; the LLM
compose step is driven by the shared ``FakeAnswerClient``; the citation anchor
resolver is faked so we assert per-course routing without touching the anchor
file machinery.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval import library_wide as lw
from lib.retrieval.answer_composer import RetrievedPassage
from lib.retrieval.citation_anchor import AnchorStatus, CitationAnchor
from lib.retrieval.grounded_answer import (
    STATUS_ANSWERED,
    STATUS_REFUSED_LOW_CONFIDENCE,
    Citation,
    GroundedAnswer,
)
from lib.retrieval.refusal import RefusalPolicy
from lib.tests.test_answer_composer import FakeAnswerClient, SpyCapture

_PERMISSIVE = RefusalPolicy(
    engine="lexical", min_top_score=0.0, score_floor=0.0,
    min_passages_above_floor=1, policy_version="test-permissive",
)
_STRICT = RefusalPolicy(
    engine="lexical", min_top_score=100.0, score_floor=0.5,
    min_passages_above_floor=1, policy_version="test-strict",
)


def _envelope(answer, citations, not_in_course=False):
    return json.dumps(
        {"answer": answer, "citations": citations, "not_in_course": not_in_course}
    )


class _FakeResult:
    def __init__(self, chunk_id, text, score, item_path="m/p.html"):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.item_path = item_path
        self.section_heading = "H"
        self.module_id = "m1"
        self.source = {"item_path": item_path, "section_heading": "H"}


def _make_course(libv2_root: Path, slug: str, *, domain: str = "physics") -> Path:
    course_dir = libv2_root / "courses" / slug
    (course_dir / "semantik_chunks").mkdir(parents=True, exist_ok=True)
    (course_dir / "semantik_chunks" / "chunks.jsonl").write_text(
        json.dumps({"id": f"{slug}-c0", "text": "body", "chunk_type": "explanation",
                    "word_count": 30, "concept_tags": ["alpha"]}) + "\n",
        encoding="utf-8",
    )
    (course_dir / "manifest.json").write_text(
        json.dumps({"classification": {"primary_domain": domain}}), encoding="utf-8"
    )
    return course_dir


@pytest.fixture()
def two_courses(tmp_path, monkeypatch):
    libv2_root = tmp_path / "LibV2"
    _make_course(libv2_root, "course-a", domain="physics")
    _make_course(libv2_root, "course-b", domain="chemistry")
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return libv2_root


# --------------------------------------------------------------------------- #
# Flag + enumeration
# --------------------------------------------------------------------------- #


def test_resolve_library_wide_default_off(monkeypatch):
    monkeypatch.delenv(lw.ENV_LIBRARY_WIDE, raising=False)
    assert lw.resolve_library_wide() is False
    monkeypatch.setenv(lw.ENV_LIBRARY_WIDE, "garbage")
    assert lw.resolve_library_wide() is False
    monkeypatch.setenv(lw.ENV_LIBRARY_WIDE, "1")
    assert lw.resolve_library_wide() is True
    monkeypatch.setenv(lw.ENV_LIBRARY_WIDE, "on")
    assert lw.resolve_library_wide() is True
    # explicit wins over env
    assert lw.resolve_library_wide(False) is False


def test_list_library_courses_fs_scan(two_courses):
    slugs = lw.list_library_courses(two_courses, "course-a")
    assert slugs[0] == "course-a"  # home first
    assert set(slugs) == {"course-a", "course-b"}


def test_list_library_courses_explicit_filters_missing(two_courses):
    slugs = lw.list_library_courses(
        two_courses, "course-a", explicit=["course-b", "course-ghost"]
    )
    # ghost has no chunkset → filtered; home always present.
    assert slugs[0] == "course-a"
    assert "course-b" in slugs
    assert "course-ghost" not in slugs


# --------------------------------------------------------------------------- #
# OFF path delegates (byte-identical single-course)
# --------------------------------------------------------------------------- #


def test_off_delegates_to_single_course(two_courses, monkeypatch):
    monkeypatch.delenv(lw.ENV_LIBRARY_WIDE, raising=False)
    sentinel = object()
    seen = {}

    def _spy(repo_root, course_slug, query, **kwargs):
        seen["slug"] = course_slug
        seen["query"] = query
        return sentinel

    monkeypatch.setattr(lw, "answer_course_question", _spy)
    out = lw.answer_library_question(
        two_courses, "course-a", "what is X?", client=FakeAnswerClient([""])
    )
    assert out is sentinel
    assert seen == {"slug": "course-a", "query": "what is X?"}


def test_single_course_fail_open(tmp_path, monkeypatch):
    libv2_root = tmp_path / "LibV2"
    _make_course(libv2_root, "solo", domain="physics")
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setenv(lw.ENV_LIBRARY_WIDE, "1")
    sentinel = object()
    monkeypatch.setattr(
        lw, "answer_course_question", lambda *a, **k: sentinel
    )
    cap = SpyCapture()
    out = lw.answer_library_question(
        libv2_root, "solo", "q?", client=FakeAnswerClient([""]), capture=cap
    )
    assert out is sentinel
    assert any(
        e["decision_type"] == lw.DECISION_TYPE_LIBRARY_WIDE
        and "fail_open_single_course" in e["decision"]
        for e in cap.events
    )


# --------------------------------------------------------------------------- #
# FIX A — chunk_type exclusion filter on the library-wide union seam
# --------------------------------------------------------------------------- #


class _FakeCTResult(_FakeResult):
    def __init__(self, chunk_id, chunk_type, score=0.9):
        super().__init__(chunk_id, f"body {chunk_id}", score, item_path="m/p.html")
        self.chunk_type = chunk_type


def test_retrieve_union_default_no_filter_byte_identical(two_courses, monkeypatch):
    """Empty exclude set → fetch `limit` verbatim, no drops (byte-identical)."""
    fetched = {}

    def _fake(libv2_root, slug, query, *, engine, limit):
        fetched["limit"] = limit
        return [_FakeCTResult(f"{slug}-a", "assessment_item"),
                _FakeCTResult(f"{slug}-e", "explanation")]

    monkeypatch.setattr(lw, "_retrieve", _fake)
    passages, counts, failed, excluded = lw._retrieve_union(
        two_courses, ["course-a"], "q", engine="lexical", limit=8,
    )
    assert fetched["limit"] == 8
    assert excluded == 0
    assert {p.chunk_id for p in passages} == {"course-a-a", "course-a-e"}


def test_retrieve_union_excludes_named_chunk_types_and_overfetches(
    two_courses, monkeypatch
):
    """Non-empty exclude set → over-fetch, drop excluded, count the drops."""
    fetched = {}

    def _fake(libv2_root, slug, query, *, engine, limit):
        fetched["limit"] = limit
        return [_FakeCTResult(f"{slug}-a", "assessment_item"),
                _FakeCTResult(f"{slug}-e", "explanation")]

    monkeypatch.setattr(lw, "_retrieve", _fake)
    passages, counts, failed, excluded = lw._retrieve_union(
        two_courses, ["course-a", "course-b"], "q",
        engine="lexical", limit=8, exclude_types=frozenset({"assessment_item"}),
    )
    assert fetched["limit"] == 24  # min(8*3, 50)
    assert excluded == 2  # one assessment_item per course
    assert {p.chunk_id for p in passages} == {"course-a-e", "course-b-e"}
    assert counts == {"course-a": 1, "course-b": 1}


# --------------------------------------------------------------------------- #
# Union + provenance
# --------------------------------------------------------------------------- #


def _fake_retrieve_factory():
    """Return a _retrieve stand-in giving each course one high-score passage."""
    def _fake(libv2_root, slug, query, *, engine, limit):
        return [
            _FakeResult(f"{slug}-hit", f"Passage from {slug}. relevant body.",
                        score=0.9 if slug == "course-a" else 0.7,
                        item_path=f"{slug}/page.html"),
        ]
    return _fake


def _fake_anchor_resolved():
    calls = []

    def _fake(chunk_record, course_dir, *, chunkset_kind, containment_threshold):
        calls.append({"chunk_id": chunk_record["id"], "course_dir": str(course_dir)})
        return CitationAnchor(
            chunk_id=chunk_record["id"],
            status=AnchorStatus.RESOLVED_EXACT,
            source_path=Path(course_dir) / "src.html",
            item_path=chunk_record["source"].get("item_path", ""),
            html_xpath=None,
            char_span=None,
            containment_rate=1.0,
            normalized_match=True,
        )
    _fake.calls = calls
    return _fake


def test_union_preserves_per_course_provenance(two_courses, monkeypatch):
    monkeypatch.setenv(lw.ENV_LIBRARY_WIDE, "1")
    monkeypatch.setattr(lw, "_retrieve", _fake_retrieve_factory())
    fake_anchor = _fake_anchor_resolved()
    monkeypatch.setattr(lw, "resolve_citation_anchor", fake_anchor)

    client = FakeAnswerClient([_envelope("Answer.", ["course-a-hit", "course-b-hit"])])
    cap = SpyCapture()
    out = lw.answer_library_question(
        two_courses, "course-a", "q?", client=client,
        refusal_policy=_PERMISSIVE, capture=cap,
    )
    assert isinstance(out, GroundedAnswer)
    assert out.status == STATUS_ANSWERED
    by_course = {c.chunk_id: c.course_slug for c in out.citations}
    assert by_course == {"course-a-hit": "course-a", "course-b-hit": "course-b"}
    # to_dict surfaces course_slug in library-wide mode.
    assert all("course_slug" in c.to_dict() for c in out.citations)
    # Per-course gate routing: each passage resolved against its own course dir.
    routed = {c["chunk_id"]: c["course_dir"] for c in fake_anchor.calls}
    assert routed["course-a-hit"].endswith("courses/course-a")
    assert routed["course-b-hit"].endswith("courses/course-b")
    # Capture fired with union outcome.
    assert any(
        e["decision_type"] == lw.DECISION_TYPE_LIBRARY_WIDE
        and "unioned" in e["decision"]
        for e in cap.events
    )


def test_single_course_citation_omits_course_slug_key():
    """The default single-course Citation dict stays byte-identical (no key)."""
    c = Citation(
        chunk_id="x", item_path="p", section_heading=None, module_id=None,
        page_label="P", anchor_status="resolved_exact", source_path=None,
        text_quote=None,
    )
    assert "course_slug" not in c.to_dict()


def test_per_course_retrieval_error_fail_open(two_courses, monkeypatch):
    monkeypatch.setenv(lw.ENV_LIBRARY_WIDE, "1")

    def _fake(libv2_root, slug, query, *, engine, limit):
        if slug == "course-b":
            raise RuntimeError("index missing")
        return [_FakeResult("course-a-hit", "body", 0.9, "course-a/p.html")]

    monkeypatch.setattr(lw, "_retrieve", _fake)
    sentinel = object()
    monkeypatch.setattr(lw, "answer_course_question", lambda *a, **k: sentinel)
    out = lw.answer_library_question(
        two_courses, "course-a", "q?", client=FakeAnswerClient([""]),
        refusal_policy=_PERMISSIVE,
    )
    # only course-a contributed → fail-open to single-course delegate.
    assert out is sentinel


def test_refusal_on_low_confidence_union(two_courses, monkeypatch):
    monkeypatch.setenv(lw.ENV_LIBRARY_WIDE, "1")
    monkeypatch.setattr(lw, "_retrieve", _fake_retrieve_factory())
    out = lw.answer_library_question(
        two_courses, "course-a", "q?", client=FakeAnswerClient([""]),
        refusal_policy=_STRICT,
    )
    assert out.status == STATUS_REFUSED_LOW_CONFIDENCE
    assert out.citations == []
