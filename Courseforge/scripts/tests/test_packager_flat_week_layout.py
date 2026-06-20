"""Regression: packager must handle FLAT week pages, not only nested subdirs.

The two-pass rewrite tier (``MCP/tools/pipeline_tools.py::_run_content_generation_rewrite``)
writes pages FLAT under ``03_content_development/`` as ``{page_id}.html`` where
``page_id = week_NN_content_NN`` — NOT inside ``week_NN/`` subdirectories the
way the legacy single-pass ``generate_course.py`` does.

Before the fix, the packager walked ``content_dir.glob("week_*")`` and dropped
every entry failing ``is_dir()`` — so a flat-layout export produced a 2.9 KB
IMSCC carrying only the course-overview page (1 HTML), and the downstream
chunkset/vector-index had a single chunk. This locks in that flat-layout pages
are grouped by their ``week_NN`` prefix and packaged under a ``week_NN/`` zip
prefix, identical to the nested layout.
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from package_multifile_imscc import _iter_week_groups, package_imscc  # noqa: E402

_OBJECTIVES = {
    "course_title": "Flat Mini Course",
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Terminal 1", "bloomLevel": "understand"},
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1: Foundations",
            "objectives": [
                {"id": "CO-01", "statement": "Foundation 1", "bloomLevel": "understand"},
            ],
        },
        {
            "chapter": "Week 2: More",
            "objectives": [
                {"id": "CO-02", "statement": "More 1", "bloomLevel": "apply"},
            ],
        },
    ],
}


def _page_html(lo_ids):
    los = [{"id": x, "statement": f"stub for {x}"} for x in lo_ids]
    ld = json.dumps(
        {"@context": "x", "@type": "LearningResource", "learningObjectives": los}
    )
    return (
        "<!DOCTYPE html><html><head>"
        '<script type="application/ld+json">' + ld + "</script>"
        "</head><body><p>content</p></body></html>"
    )


@pytest.fixture
def flat_content_dir(tmp_path):
    """Content dir with FLAT week_NN_*.html pages (no week_NN/ subdirs)."""
    (tmp_path / "week_01_content_01.html").write_text(
        _page_html(["TO-01", "CO-01"]), encoding="utf-8"
    )
    (tmp_path / "week_01_content_02.html").write_text(
        _page_html(["TO-01", "CO-01"]), encoding="utf-8"
    )
    (tmp_path / "week_02_content_01.html").write_text(
        _page_html(["CO-02"]), encoding="utf-8"
    )
    (tmp_path / "course.json").write_text(json.dumps(_OBJECTIVES), encoding="utf-8")
    return tmp_path


def test_iter_week_groups_groups_flat_pages(flat_content_dir):
    groups = _iter_week_groups(flat_content_dir)
    assert [g[0] for g in groups] == ["week_01", "week_02"]
    # week_01 has 2 pages, week_02 has 1.
    counts = {wn: len(files) for wn, _n, _td, files in groups}
    assert counts == {"week_01": 2, "week_02": 1}


def test_flat_layout_packages_every_page(flat_content_dir, tmp_path):
    out = tmp_path / "out.imscc"
    package_imscc(flat_content_dir, out, "MINI", "Flat Mini", skip_validation=True)
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        html = [n for n in zf.namelist() if n.endswith(".html")]
    # All 3 flat pages packaged under week_NN/ prefixes (NOT just 1).
    assert "week_01/week_01_content_01.html" in html
    assert "week_01/week_01_content_02.html" in html
    assert "week_02/week_02_content_01.html" in html
    # All 3 week pages present (a course-overview page is added separately).
    week_pages = [n for n in html if n.startswith("week_")]
    assert len(week_pages) == 3


def test_nested_layout_still_works(tmp_path):
    """Backward-compat: the legacy week_NN/ subdir layout is unaffected."""
    (tmp_path / "week_01").mkdir()
    (tmp_path / "week_01" / "week_01_overview.html").write_text(
        _page_html(["TO-01", "CO-01"]), encoding="utf-8"
    )
    (tmp_path / "course.json").write_text(json.dumps(_OBJECTIVES), encoding="utf-8")
    out = tmp_path / "out.imscc"
    package_imscc(tmp_path, out, "MINI", "Mini", skip_validation=True)
    with zipfile.ZipFile(out) as zf:
        html = [n for n in zf.namelist() if n.endswith(".html")]
    assert "week_01/week_01_overview.html" in html
