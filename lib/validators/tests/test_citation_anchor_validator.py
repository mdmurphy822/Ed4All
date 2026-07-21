"""Unit tests for CitationAnchorValidator (WS1.2).

Exercises the GateResult/GateIssue contract against the shared mini-course
fixture and synthetic tmp_path corpora. The validator is NOT wired into
config/workflows.yaml yet (deferred); these tests pin its behavior so the
later one-line YAML wire-up is safe.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.validators.citation_anchor import CitationAnchorValidator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MINI_COURSE = PROJECT_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"


def test_missing_input_blocks():
    result = CitationAnchorValidator().validate({})
    assert result.passed is False
    assert result.action == "block"
    assert result.issues[0].code == "CITATION_ANCHOR_MISSING_INPUT"


def test_chunks_not_found(tmp_path):
    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(tmp_path / "nope.jsonl"),
            "course_dir": str(tmp_path),
            "chunkset_kind": "dart",
        }
    )
    assert result.passed is False
    assert result.issues[0].code == "CITATION_ANCHOR_CHUNKS_NOT_FOUND"


def test_bad_kind_blocks(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n")
    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks),
            "course_dir": str(tmp_path),
            "chunkset_kind": "bogus",
        }
    )
    assert result.passed is False
    assert result.issues[0].code == "CITATION_ANCHOR_MISSING_INPUT"


def test_mini_course_passes_default_floor():
    """The fixture anchors at 1.0, so it passes even the 0.95 default floor."""
    chunks_path = MINI_COURSE / "semantik_chunks" / "chunks.jsonl"
    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(MINI_COURSE),
            "chunkset_kind": "dart",
        }
    )
    assert result.passed is True
    assert result.action is None
    report = result.metadata["citation_anchor_report"]
    assert report["anchoring_rate"] == 1.0


def test_below_floor_blocks(tmp_path):
    """A corpus where the only chunk's source page is missing → rate 0.0,
    below the 0.95 floor → critical block + a missing-page warning."""
    course_dir = tmp_path / "course"
    (course_dir / "sources" / "textbooks").mkdir(parents=True)
    chunk = {
        "id": "c1",
        "text": "some text",
        "source": {
            "course_id": "X",
            "module_id": "m",
            "lesson_id": "m",
            "item_path": "missing.html",
            "html_xpath": "/html[1]/body[1]",
            "char_span": [0, 9],
        },
    }
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n")
    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(course_dir),
            "chunkset_kind": "dart",
        }
    )
    assert result.passed is False
    assert result.action == "block"
    codes = {i.code for i in result.issues}
    assert "CITATION_ANCHOR_RATE_BELOW_FLOOR" in codes
    assert "CITATION_ANCHOR_SOURCE_PAGE_MISSING" in codes


def test_custom_floor_lets_low_rate_pass(tmp_path):
    """A relaxed floor passes a low-but-nonzero anchoring rate."""
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    page = "<html><body><h1>H</h1><p>alpha beta gamma delta epsilon zeta eta theta iota kappa</p></body></html>"
    (html_dir / "p.html").write_text(page)
    good = {
        "id": "g",
        "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        "source": {
            "course_id": "X",
            "module_id": "m",
            "lesson_id": "m",
            "item_path": "p.html",
            "html_xpath": "/html[1]/body[1]",
            "char_span": [0, 5],
        },
    }
    bad = {**good, "id": "b", "source": {**good["source"], "item_path": "gone.html"}}
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(course_dir),
            "chunkset_kind": "dart",
            "min_anchoring_rate": 0.5,
        }
    )
    # rate is 0.5 (one of two resolves) → meets the 0.5 floor → passes.
    assert result.passed is True
    report = result.metadata["citation_anchor_report"]
    assert report["anchoring_rate"] == 0.5
