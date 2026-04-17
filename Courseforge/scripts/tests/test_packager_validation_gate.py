"""
Tests for the per-week LO validation gate in package_multifile_imscc.py
(Worker I — FOLLOWUP-WORKER-H-3).

Guards against the LO-fanout defect silently reappearing: if a future
change to Courseforge's generation path reintroduces week-local IDs or
otherwise emits IDs that don't belong to a page's week, the packager
refuses to build.
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from package_multifile_imscc import (  # noqa: E402
    package_imscc,
    validate_content_objectives,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_OBJECTIVES = {
    "course_title": "Mini Course",
    "description": "Fixture",
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Terminal 1", "bloomLevel": "evaluate"},
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1-2: Foundations",
            "objectives": [
                {"id": "CO-01", "statement": "Foundation 1", "bloomLevel": "understand"},
                {"id": "CO-02", "statement": "Foundation 2", "bloomLevel": "apply"},
            ],
        },
        {
            "chapter": "Week 3-4: Advanced",
            "objectives": [
                {"id": "CO-03", "statement": "Advanced 1", "bloomLevel": "analyze"},
                {"id": "CO-04", "statement": "Advanced 2", "bloomLevel": "evaluate"},
            ],
        },
    ],
}


def _page_html(lo_ids):
    """Emit a minimal HTML page carrying one JSON-LD learningObjectives block."""
    los = [{"id": x, "statement": f"stub for {x}"} for x in lo_ids]
    ld = json.dumps({"@context": "x", "@type": "LearningResource", "learningObjectives": los})
    return (
        '<!DOCTYPE html><html><head>'
        '<script type="application/ld+json">' + ld + '</script>'
        '</head><body><p>content</p></body></html>'
    )


@pytest.fixture
def content_dir(tmp_path):
    """Build week_01 + week_03 page fixtures under a tmp content dir."""
    (tmp_path / "week_01").mkdir()
    (tmp_path / "week_03").mkdir()
    return tmp_path


@pytest.fixture
def objectives_path(tmp_path):
    p = tmp_path / "objectives.json"
    p.write_text(json.dumps(_OBJECTIVES), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidationGate:
    def test_clean_content_passes(self, content_dir, objectives_path):
        # Week 1 gets its allowed set (TO-01, CO-01, CO-02)
        (content_dir / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
        )
        # Week 3 gets its allowed set (TO-01, CO-03, CO-04)
        (content_dir / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-03", "CO-04"]), encoding="utf-8",
        )
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        assert ok, failures
        assert failures == []

    def test_cross_week_contamination_fails(self, content_dir, objectives_path):
        # Week 3 page incorrectly references CO-01 (belongs to weeks 1-2 only)
        (content_dir / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-01", "CO-03"]), encoding="utf-8",
        )
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        assert not ok
        assert len(failures) == 1
        assert "CO-01" in failures[0]

    def test_fabricated_week_local_id_fails(self, content_dir, objectives_path):
        # This is the exact defect the pre-fix builds shipped: W01-CO-01
        # doesn't exist in the canonical registry.
        (content_dir / "week_01" / "week_01_overview.html").write_text(
            _page_html(["W01-CO-01", "W01-CO-02"]), encoding="utf-8",
        )
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        assert not ok
        assert "W01-CO-01" in failures[0] or "W01-CO-02" in failures[0]

    def test_pages_without_jsonld_are_skipped(self, content_dir, objectives_path):
        (content_dir / "week_01" / "week_01_overview.html").write_text(
            "<!DOCTYPE html><html><body>plain content</body></html>", encoding="utf-8",
        )
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        assert ok
        assert failures == []

    def test_package_imscc_refuses_to_build_on_violation(
        self, content_dir, objectives_path, tmp_path,
    ):
        # Mix of valid + invalid pages. Packager must not emit the zip.
        (content_dir / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01"]), encoding="utf-8",
        )
        (content_dir / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-01"]),  # wrong — CO-01 not allowed week 3
            encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        with pytest.raises(SystemExit) as excinfo:
            package_imscc(
                content_dir, output, "TEST_101", "Test Course",
                objectives_path=objectives_path,
            )
        assert excinfo.value.code == 2
        assert not output.exists(), "packager must NOT create the zip when validation fails"

    def test_package_imscc_builds_when_valid(
        self, content_dir, objectives_path, tmp_path,
    ):
        (content_dir / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01"]), encoding="utf-8",
        )
        (content_dir / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-03"]), encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir, output, "TEST_101", "Test Course",
            objectives_path=objectives_path,
        )
        assert output.exists()
        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()
            assert "imsmanifest.xml" in names
            assert any(n.endswith("week_01_overview.html") for n in names)
            assert any(n.endswith("week_03_overview.html") for n in names)

    def test_skip_validation_bypasses_even_on_violation(
        self, content_dir, objectives_path, tmp_path,
    ):
        (content_dir / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-01"]),  # violation
            encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        # With skip_validation=True, violation is logged but packaging proceeds.
        package_imscc(
            content_dir, output, "TEST_101", "Test Course",
            objectives_path=objectives_path,
            skip_validation=True,
        )
        assert output.exists(), "--skip-validation must allow packaging to proceed"

    def test_no_objectives_arg_skips_validation_entirely(
        self, content_dir, tmp_path,
    ):
        # When caller doesn't pass --objectives, validation is silently skipped —
        # this is the legacy behavior the gate preserves for back-compat.
        (content_dir / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-01"]),  # would violate if validated
            encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(content_dir, output, "TEST_101", "Test Course")
        assert output.exists()
