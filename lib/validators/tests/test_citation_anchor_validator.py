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


def test_imscc_qti_escaped_inner_html_anchors_visible_item_text(tmp_path):
    """QTI mattext's escaped inner HTML is markup, not fabricated prose.

    The chunker emits rendered prompt/choice text while the archived cartridge
    keeps those fields as ``&lt;p&gt;...`` inside XML.  Citation comparison
    must remove the revealed tag shells without changing the 0.95 gate floor.
    """
    import zipfile

    course_dir = tmp_path / "course"
    imscc_dir = course_dir / "source" / "imscc"
    imscc_dir.mkdir(parents=True)
    member = "06_assessments/week_01_quiz.xml"
    qti = """<?xml version="1.0" encoding="UTF-8"?>
<questestinterop>
  <item>
    <presentation>
      <material><mattext texttype="text/html">&lt;p&gt;Which value is
      &lt;em&gt;positive?&lt;/em&gt;&lt;/p&gt;</mattext></material>
      <response_lid><render_choice>
        <response_label><material><mattext texttype="text/html">
          &lt;p&gt;Three&lt;/p&gt;
        </mattext></material></response_label>
        <response_label><material><mattext texttype="text/html">
          &lt;p&gt;Negative three&lt;/p&gt;
        </mattext></material></response_label>
      </render_choice></response_lid>
    </presentation>
  </item>
</questestinterop>"""
    with zipfile.ZipFile(imscc_dir / "course.imscc", "w") as zf:
        zf.writestr(member, qti)

    chunk = {
        "id": "assessment-1",
        "text": "Which value is positive? Three Negative three",
        "chunk_type": "assessment_item",
        "source": {
            "item_path": member,
            "html_xpath": "/questestinterop[1]/item[1]",
            "char_span": [0, 44],
        },
    }
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(course_dir),
            "chunkset_kind": "imscc",
        }
    )

    assert result.passed is True
    report = result.metadata["citation_anchor_report"]
    assert report["anchoring_rate"] == 1.0
    assert report["status_counts"]["span_fabricated"] == 0


def test_merged_body_runs_anchor_across_structural_heading(tmp_path):
    """Chunker-omitted section labels must not break body-prose anchoring."""
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    (html_dir / "page.html").write_text(
        "<html><body>"
        "<h2>First section</h2>"
        "<p>Alpha beta gamma delta epsilon zeta eta theta iota.</p>"
        "<h2>Second section</h2>"
        "<p>Kappa lambda mu nu xi omicron pi rho sigma.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    chunk = {
        "id": "merged-1",
        "text": (
            "Alpha beta gamma delta epsilon zeta eta theta iota. "
            "Kappa lambda mu nu xi omicron pi rho sigma."
        ),
        "source": {
            "item_path": "page.html",
            "html_xpath": "/html[1]/body[1]",
            "char_span": [0, 99],
        },
    }
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(course_dir),
            "chunkset_kind": "dart",
        }
    )

    assert result.passed is True
    report = result.metadata["citation_anchor_report"]
    assert report["anchoring_rate"] == 1.0
    assert report["status_counts"]["span_fabricated"] == 0


def test_heading_projection_does_not_admit_absent_body_claim(tmp_path):
    """Removing headings never turns source-absent prose into an anchor."""
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    (html_dir / "page.html").write_text(
        "<html><body><h2>Linear equations</h2>"
        "<p>Alpha beta gamma delta epsilon zeta eta theta.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    chunk = {
        "id": "fabricated-1",
        "text": (
            "Linear equations always have exactly three solutions and every "
            "coefficient must be positive."
        ),
        "source": {
            "item_path": "page.html",
            "html_xpath": "/html[1]/body[1]",
            "char_span": [0, 88],
        },
    }
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(course_dir),
            "chunkset_kind": "dart",
        }
    )

    assert result.passed is False
    report = result.metadata["citation_anchor_report"]
    assert report["anchoring_rate"] == 0.0
    assert report["status_counts"]["span_fabricated"] == 1


def test_qti_math_comparison_is_not_mistaken_for_nested_tag(tmp_path):
    """``<C`` is algebra, while escaped ``<p>`` is nested QTI markup."""
    import zipfile

    course_dir = tmp_path / "course"
    imscc_dir = course_dir / "source" / "imscc"
    imscc_dir.mkdir(parents=True)
    member = "assessment.xml"
    qti = (
        "<questestinterop><item><mattext texttype=\"text/html\">"
        "&lt;p&gt;The boundary separates $A x+B y&gt;C$ from "
        "$A x+B y&lt;C$.&lt;/p&gt;"
        "</mattext></item></questestinterop>"
    )
    with zipfile.ZipFile(imscc_dir / "course.imscc", "w") as zf:
        zf.writestr(member, qti)
    chunk = {
        "id": "math-1",
        "text": "The boundary separates $A x+B y > C$ from $A x+B y < C$.",
        "source": {"item_path": member},
    }
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(course_dir),
            "chunkset_kind": "imscc",
        }
    )

    assert result.passed is True
    report = result.metadata["citation_anchor_report"]
    assert report["anchoring_rate"] == 1.0
    assert report["status_counts"]["span_fabricated"] == 0


def test_ordered_body_segments_anchor_across_non_heading_label(tmp_path):
    """Exact body runs remain anchored when chunker omits a page-only label."""
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    first = "alpha beta gamma delta epsilon zeta eta theta"
    second = "iota kappa lambda mu nu xi omicron pi"
    (html_dir / "page.html").write_text(
        "<html><body><p>"
        + first
        + "</p><div class='section-label'>Practice checkpoint</div><p>"
        + second
        + "</p></body></html>",
        encoding="utf-8",
    )
    chunk = {
        "id": "segmented-1",
        "text": f"{first} {second}",
        "source": {"item_path": "page.html"},
    }
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    result = CitationAnchorValidator().validate(
        {
            "chunks_path": str(chunks_path),
            "course_dir": str(course_dir),
            "chunkset_kind": "dart",
        }
    )

    assert result.passed is True
    report = result.metadata["citation_anchor_report"]
    assert report["anchoring_rate"] == 1.0
    assert report["status_counts"]["span_fabricated"] == 0
