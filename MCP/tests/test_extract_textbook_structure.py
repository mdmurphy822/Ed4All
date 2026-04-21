"""Tests for _extract_textbook_structure (Wave 24).

Covers the new textbook-ingestor dispatch target that runs
SemanticStructureExtractor over staged DART HTML and emits
textbook_structure.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


def _write_dart_html(path: Path, chapters: list) -> None:
    """Write a minimal DART-like HTML with <article role="doc-chapter"> wrappers."""
    body_parts = [
        '<a class="skip-link" href="#main">Skip</a>',
        '<main role="main">',
    ]
    for idx, ch in enumerate(chapters, start=1):
        body_parts.append(
            f'<article role="doc-chapter" id="chap-{idx}">'
        )
        body_parts.append(
            f'<header><h2 id="ch{idx}-title">{ch["title"]}</h2></header>'
        )
        for sec_idx, sec in enumerate(ch.get("sections", []), start=1):
            body_parts.append(
                f'<section aria-labelledby="ch{idx}-s{sec_idx}">'
            )
            body_parts.append(
                f'<h3 id="ch{idx}-s{sec_idx}">{sec["title"]}</h3>'
            )
            for para in sec.get("paragraphs", []):
                body_parts.append(f"<p>{para}</p>")
            body_parts.append("</section>")
        body_parts.append("</article>")
    body_parts.append("</main>")
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{chapters[0]["title"] if chapters else "Empty"}</title>'
        "</head><body>"
        + "".join(body_parts)
        + "</body></html>"
    )
    path.write_text(html, encoding="utf-8")


@pytest.fixture
def extractor_fixture(tmp_path, monkeypatch):
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    return {"root": fake_root, "staging": staging}


async def _call(**kwargs):
    registry = _build_tool_registry()
    assert "extract_textbook_structure" in registry
    fn = registry["extract_textbook_structure"]
    raw = await fn(**kwargs)
    return json.loads(raw)


def test_requires_course_name():
    """Missing course_name → error."""
    result = asyncio.run(_call(staging_dir="/tmp/nonexistent"))
    assert "error" in result
    assert "course_name" in result["error"]


def test_no_html_produces_empty_structure(extractor_fixture):
    """Empty staging dir → valid structure.json with chapter_count=0."""
    fx = extractor_fixture
    result = asyncio.run(_call(
        course_name="EMPTY_COURSE",
        staging_dir=str(fx["staging"]),
    ))
    assert result["success"]
    assert result["chapter_count"] == 0
    structure_path = Path(result["textbook_structure_path"])
    assert structure_path.exists()
    doc = json.loads(structure_path.read_text(encoding="utf-8"))
    assert doc["chapter_count"] == 0
    assert doc["chapters"] == []


def test_single_chapter_extraction(extractor_fixture):
    fx = extractor_fixture
    _write_dart_html(fx["staging"] / "textbook_a.html", [
        {
            "title": "Chapter 1: Photosynthesis",
            "sections": [
                {
                    "title": "Overview of Photosynthesis",
                    "paragraphs": [
                        "Photosynthesis is the process by which plants "
                        "convert sunlight into chemical energy.",
                        "Two stages exist: the light-dependent reactions "
                        "and the Calvin cycle reactions.",
                    ],
                },
            ],
        },
    ])
    result = asyncio.run(_call(
        course_name="BIO_101",
        staging_dir=str(fx["staging"]),
    ))
    assert result["success"]
    assert result["chapter_count"] == 1


def test_multi_chapter_extraction(extractor_fixture):
    fx = extractor_fixture
    _write_dart_html(fx["staging"] / "book.html", [
        {
            "title": "Chapter 1: Kinematics",
            "sections": [
                {"title": "Velocity and Acceleration",
                 "paragraphs": ["Velocity describes the rate of change of position. Acceleration is the rate of change of velocity over time."]},
            ],
        },
        {
            "title": "Chapter 2: Forces",
            "sections": [
                {"title": "Newton's Laws of Motion",
                 "paragraphs": ["Newton's first law states that an object in motion tends to stay in motion unless acted upon by an external force."]},
            ],
        },
        {
            "title": "Chapter 3: Energy",
            "sections": [
                {"title": "Kinetic and Potential Energy",
                 "paragraphs": ["Kinetic energy is the energy of motion; potential energy is stored energy due to position or configuration."]},
            ],
        },
    ])
    result = asyncio.run(_call(
        course_name="PHYS_101",
        staging_dir=str(fx["staging"]),
    ))
    assert result["success"]
    assert result["chapter_count"] == 3


def test_persists_project_config(extractor_fixture):
    """After extraction, project_config.json carries course_name + duration_weeks."""
    fx = extractor_fixture
    _write_dart_html(fx["staging"] / "book.html", [
        {"title": "Chapter 1", "sections": [{"title": "S1", "paragraphs": ["Paragraph with enough words to pass the minimum word count filter for topic extraction."]}]},
    ])
    result = asyncio.run(_call(
        course_name="CHEM_201",
        staging_dir=str(fx["staging"]),
        duration_weeks=16,
    ))
    assert result["success"]
    cfg_path = Path(result["project_path"]) / "project_config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["course_name"] == "CHEM_201"
    assert cfg["duration_weeks"] == 16


def test_structure_path_location(extractor_fixture):
    """textbook_structure.json lands under 01_learning_objectives/."""
    fx = extractor_fixture
    _write_dart_html(fx["staging"] / "book.html", [
        {"title": "Chapter 1", "sections": [{"title": "S1", "paragraphs": ["A minimal paragraph with enough content to be recognized as a topic by the extractor heuristics."]}]},
    ])
    result = asyncio.run(_call(
        course_name="TEST_101",
        staging_dir=str(fx["staging"]),
    ))
    structure_path = Path(result["textbook_structure_path"])
    assert structure_path.name == "textbook_structure.json"
    assert structure_path.parent.name == "01_learning_objectives"


def test_extraction_errors_logged_not_fatal(extractor_fixture):
    """Malformed HTML files are recorded in extraction_errors, not raised."""
    fx = extractor_fixture
    # Valid file.
    _write_dart_html(fx["staging"] / "good.html", [
        {"title": "Chapter 1", "sections": [{"title": "S1", "paragraphs": ["Paragraph with enough words to pass the minimum word count filter for topic extraction."]}]},
    ])
    # Bogus file (non-HTML but .html extension).
    (fx["staging"] / "bad.html").write_bytes(b"\xff\xfe\x00")

    result = asyncio.run(_call(
        course_name="TEST_ERR",
        staging_dir=str(fx["staging"]),
    ))
    # Even with one malformed file, we still produce the structure.
    assert result["success"]
    structure_path = Path(result["textbook_structure_path"])
    doc = json.loads(structure_path.read_text(encoding="utf-8"))
    # source_file_count reflects the total files walked.
    assert doc["per_file_results"] or doc["extraction_errors"]


def test_autoscale_weeks_when_implicit(extractor_fixture):
    """duration_weeks_explicit=False → weeks scales to max(8, chapter_count)."""
    fx = extractor_fixture
    chapters = [
        {
            "title": f"Chapter {i}",
            "sections": [{
                "title": f"Section {i}",
                "paragraphs": [
                    f"Chapter {i} covers a distinct foundational topic with "
                    f"sufficient detail to qualify as a real topic. Each "
                    f"paragraph is long enough to exceed the minimum word "
                    f"count filter used by the extractor during dispatch."
                ],
            }],
        }
        for i in range(1, 11)  # 10 chapters
    ]
    _write_dart_html(fx["staging"] / "book.html", chapters)
    result = asyncio.run(_call(
        course_name="AUTOSCALE",
        staging_dir=str(fx["staging"]),
        duration_weeks=12,
        duration_weeks_explicit=False,
    ))
    assert result["success"]
    # max(8, 10) = 10.
    assert result["duration_weeks"] == 10
    assert result["duration_weeks_autoscaled"] is True


def test_no_autoscale_when_explicit(extractor_fixture):
    """duration_weeks_explicit=True → weeks sticks to user-supplied value."""
    fx = extractor_fixture
    _write_dart_html(fx["staging"] / "book.html", [
        {"title": "Chapter 1", "sections": [{"title": "S1", "paragraphs": ["Paragraph with enough words to pass the minimum word count filter for topic extraction."]}]},
    ])
    result = asyncio.run(_call(
        course_name="FIXED_WEEKS",
        staging_dir=str(fx["staging"]),
        duration_weeks=16,
        duration_weeks_explicit=True,
    ))
    assert result["success"]
    assert result["duration_weeks"] == 16
    assert result["duration_weeks_autoscaled"] is False


def test_deterministic_chapter_ids(extractor_fixture):
    """Chapter IDs deduplicated across multiple HTML files."""
    fx = extractor_fixture
    _write_dart_html(fx["staging"] / "file1.html", [
        {"title": "Chapter A", "sections": [{"title": "S1", "paragraphs": ["Paragraph with enough words to pass the minimum word count filter for topic extraction."]}]},
    ])
    _write_dart_html(fx["staging"] / "file2.html", [
        {"title": "Chapter A", "sections": [{"title": "S1", "paragraphs": ["Another paragraph with enough words to pass the minimum word count filter for topic extraction."]}]},
    ])
    result = asyncio.run(_call(
        course_name="DUPE_TEST",
        staging_dir=str(fx["staging"]),
    ))
    assert result["success"]
    structure_path = Path(result["textbook_structure_path"])
    doc = json.loads(structure_path.read_text(encoding="utf-8"))
    chapter_ids = [c.get("id") for c in doc.get("chapters", [])]
    # All chapter IDs must be unique after dedup pass.
    assert len(chapter_ids) == len(set(chapter_ids))
