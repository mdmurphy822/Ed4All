"""Course Overview Learning Objectives Map module — packaging integration.

Asserts that ``package_imscc`` renders the deterministic objectives map into
``course_overview/learning_objectives.html`` and lands it in the IMSCC under a
"Course Overview" module in the manifest organization tree, on by default,
mirrored in the zip payload. Also covers the opt-out flag, the author-copy
preservation contract, and the no-objectives skip.

Synthetic TST_910-style fixtures only — no real course identifiers.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from package_multifile_imscc import package_imscc  # noqa: E402

_CC_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"

_OBJECTIVES = {
    "course_name": "TST_910",
    "course_title": "Synthetic Course Alpha",
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Apply Newton's laws.", "bloom_level": "apply"},
    ],
    "chapter_objectives": {
        "1": [
            {
                "id": "CO-01",
                "terminal_id": "TO-01",
                "statement": "Identify forces.",
                "bloom_level": "remember",
                "section": "1.1",
            },
        ],
    },
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Project layout: 01_learning_objectives/ + 03_content_development/week_01."""
    proj = tmp_path / "fixture-objectives-project"
    content = proj / "03_content_development"
    week = content / "week_01"
    week.mkdir(parents=True)
    (week / "week_01_overview.html").write_text(
        "<html><head></head><body><h1>Week 1 Overview: Forces</h1></body></html>",
        encoding="utf-8",
    )
    obj_dir = proj / "01_learning_objectives"
    obj_dir.mkdir(parents=True)
    (obj_dir / "synthesized_objectives.json").write_text(
        json.dumps(_OBJECTIVES), encoding="utf-8"
    )
    return proj


def _content_dir(project: Path) -> Path:
    return project / "03_content_development"


def _manifest_titles(zf: zipfile.ZipFile) -> list[str]:
    xml = zf.read("imsmanifest.xml").decode("utf-8")
    root = ET.fromstring(xml)
    return [t.text for t in root.iter(f"{{{_CC_NS}}}title")]


def test_objectives_page_lands_in_manifest_and_zip_by_default(project, tmp_path):
    out = tmp_path / "course.imscc"
    package_imscc(
        _content_dir(project), out, "TST_910", "Synthetic Course Alpha",
        skip_validation=True,
    )
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        titles = _manifest_titles(zf)
        manifest = zf.read("imsmanifest.xml").decode("utf-8")

    # Zip payload carries the rendered page.
    assert "course_overview/learning_objectives.html" in names, names
    # Manifest organization carries the Course Overview module + item.
    assert "Course Overview" in titles
    assert "course_overview/learning_objectives.html" in manifest
    assert 'identifier="COURSE_OVERVIEW"' in manifest
    # The rendered page carries the canonical objectives.
    with zipfile.ZipFile(out) as zf:
        page_html = zf.read("course_overview/learning_objectives.html").decode("utf-8")
    assert "TO-01" in page_html and "CO-01" in page_html
    assert "Learning Objectives Map" in page_html


def test_course_overview_module_precedes_weeks(project, tmp_path):
    out = tmp_path / "course.imscc"
    package_imscc(
        _content_dir(project), out, "TST_910", "Synthetic Course Alpha",
        skip_validation=True,
    )
    with zipfile.ZipFile(out) as zf:
        xml = zf.read("imsmanifest.xml").decode("utf-8")
    # COURSE_OVERVIEW item must appear before WEEK_1 in document order.
    assert xml.index("COURSE_OVERVIEW") < xml.index("WEEK_1")


def test_opt_out_suppresses_objectives_page(project, tmp_path):
    out = tmp_path / "course.imscc"
    package_imscc(
        _content_dir(project), out, "TST_910", "Synthetic Course Alpha",
        skip_validation=True,
        emit_objectives_page=False,
    )
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        titles = _manifest_titles(zf)
    assert "course_overview/learning_objectives.html" not in names
    assert "Course Overview" not in titles


def test_no_objectives_source_skips_module(tmp_path):
    # content dir with a week but NO objectives json anywhere.
    content = tmp_path / "03_content_development"
    week = content / "week_01"
    week.mkdir(parents=True)
    (week / "week_01_overview.html").write_text(
        "<html><body><h1>Week 1</h1></body></html>", encoding="utf-8"
    )
    out = tmp_path / "course.imscc"
    package_imscc(
        content, out, "TST_910", "Synthetic Course Alpha",
        skip_validation=True,
    )
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "course_overview/learning_objectives.html" not in names


def test_author_copy_is_preserved(project, tmp_path):
    # Pre-place an author copy; packager must not overwrite it.
    overview_dir = _content_dir(project) / "course_overview"
    overview_dir.mkdir(parents=True)
    sentinel = "<html><body><h1>AUTHORED OVERRIDE</h1></body></html>"
    (overview_dir / "learning_objectives.html").write_text(
        sentinel, encoding="utf-8"
    )
    out = tmp_path / "course.imscc"
    package_imscc(
        _content_dir(project), out, "TST_910", "Synthetic Course Alpha",
        skip_validation=True,
    )
    with zipfile.ZipFile(out) as zf:
        page = zf.read("course_overview/learning_objectives.html").decode("utf-8")
    assert "AUTHORED OVERRIDE" in page


def test_objectives_page_present_in_outline_only_mode(project, tmp_path):
    # The objectives map is outline-tier content; it survives outline-only.
    (_content_dir(project) / "week_01" / "week_01_content_01.html").write_text(
        "<html><body><h1>Content</h1></body></html>", encoding="utf-8"
    )
    out = tmp_path / "course.imscc"
    package_imscc(
        _content_dir(project), out, "TST_910", "Synthetic Course Alpha",
        skip_validation=True,
        outline_only=True,
    )
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "course_overview/learning_objectives.html" in names
    # outline-only still drops the content page.
    assert "week_01/week_01_content_01.html" not in names
