"""Regression tests for REC-TAX-01 — course_metadata.json stub consume path.

Covers:
  * Stub alongside IMSCC file → classification loaded from stub.
  * CLI overrides individual fields of the stub.
  * Neither stub nor CLI → backward-compat defaults (division=STEM).
  * Courseforge emit-side fail-closed on invalid classification.

Tests build tiny on-disk fixtures rather than spinning up real IMSCC
packages — the CourseProcessor constructor only reads ``imsmanifest.xml``
when :meth:`_extract_imscc` runs, which these tests deliberately skip.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


def _make_minimal_imscc(path: Path) -> None:
    """Build a minimal IMSCC zip with only an imsmanifest.xml and one HTML."""
    manifest = """<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1" identifier="MINI">
  <metadata><schema>IMS Common Cartridge</schema><schemaversion>1.3.0</schemaversion></metadata>
  <organizations/>
  <resources/>
</manifest>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("imsmanifest.xml", manifest)
        z.writestr("week_01/week_01_overview.html", "<html><body>Hi</body></html>")


def _make_sibling_stub(imscc_path: Path, classification: dict) -> Path:
    """Drop a course_metadata.json next to the IMSCC file."""
    stub_path = imscc_path.parent / "course_metadata.json"
    stub = {
        "course_code": "MINI_101",
        "course_title": "Mini",
        "classification": classification,
        "ontology_mappings": {"acm_ccs": [], "lcsh": []},
    }
    stub_path.write_text(json.dumps(stub, indent=2), encoding="utf-8")
    return stub_path


def _make_inzip_stub(imscc_path: Path, classification: dict) -> None:
    """Add a course_metadata.json to an existing IMSCC zip."""
    stub = {
        "course_code": "MINI_101",
        "course_title": "Mini",
        "classification": classification,
        "ontology_mappings": {"acm_ccs": [], "lcsh": []},
    }
    with zipfile.ZipFile(imscc_path, "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr("course_metadata.json", json.dumps(stub, indent=2))


@pytest.mark.unit
def test_stub_driven_classification_sibling(tmp_path):
    """Sibling course_metadata.json populates classification when no CLI."""
    from Trainforge.process_course import CourseProcessor

    imscc = tmp_path / "mini.imscc"
    _make_minimal_imscc(imscc)
    _make_sibling_stub(imscc, {
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["software-engineering"],
        "topics": [],
    })

    out = tmp_path / "out"
    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(out),
        course_code="MINI_101",
        # No division/domain/subdomains/topics → stub drives
    )
    assert processor.division == "STEM"
    assert processor.domain == "computer-science"
    assert processor.subdomains == ["software-engineering"]
    assert processor.topics == []


@pytest.mark.unit
def test_stub_driven_classification_in_zip(tmp_path):
    """In-zip course_metadata.json is preferred over sibling (forward-compat)."""
    from Trainforge.process_course import CourseProcessor

    imscc = tmp_path / "mini.imscc"
    _make_minimal_imscc(imscc)
    _make_inzip_stub(imscc, {
        "division": "ARTS",
        "primary_domain": "design",
        "subdomains": [],
        "topics": [],
    })
    # Sibling with different values — in-zip must win per _load_classification_stub
    # priority order.
    _make_sibling_stub(imscc, {
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": [],
        "topics": [],
    })

    out = tmp_path / "out"
    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(out),
        course_code="MINI_101",
    )
    assert processor.division == "ARTS"
    assert processor.domain == "design"


@pytest.mark.unit
def test_cli_override_stub(tmp_path):
    """CLI flags override individual fields of the stub."""
    from Trainforge.process_course import CourseProcessor

    imscc = tmp_path / "mini.imscc"
    _make_minimal_imscc(imscc)
    _make_sibling_stub(imscc, {
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["software-engineering"],
        "topics": [],
    })

    out = tmp_path / "out"
    # Override division only; primary_domain + subdomains come from stub.
    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(out),
        course_code="MINI_101",
        division="ARTS",
    )
    assert processor.division == "ARTS"
    # Primary domain still from stub (CLI didn't override).
    assert processor.domain == "computer-science"


@pytest.mark.unit
def test_cli_override_full_replacement(tmp_path):
    """All-CLI-flag path replaces every stub field."""
    from Trainforge.process_course import CourseProcessor

    imscc = tmp_path / "mini.imscc"
    _make_minimal_imscc(imscc)
    _make_sibling_stub(imscc, {
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["software-engineering"],
        "topics": [],
    })

    out = tmp_path / "out"
    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(out),
        course_code="MINI_101",
        division="ARTS",
        domain="design",
        subdomains=["ui-design"],
        topics=[],
    )
    assert processor.division == "ARTS"
    assert processor.domain == "design"
    assert processor.subdomains == ["ui-design"]


@pytest.mark.unit
def test_no_stub_no_cli_backward_compat(tmp_path):
    """Absent stub AND absent CLI → backward-compat defaults apply."""
    from Trainforge.process_course import CourseProcessor

    imscc = tmp_path / "mini.imscc"
    _make_minimal_imscc(imscc)

    out = tmp_path / "out"
    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(out),
        course_code="MINI_101",
    )
    assert processor.division == "STEM", "default division retained"
    assert processor.domain == "", "empty primary_domain default retained"
    assert processor.subdomains == []
    assert processor.topics == []


@pytest.mark.unit
def test_stub_loader_returns_none_without_stub(tmp_path):
    """_load_classification_stub returns None when no stub anywhere."""
    from Trainforge.process_course import CourseProcessor

    imscc = tmp_path / "mini.imscc"
    _make_minimal_imscc(imscc)

    out = tmp_path / "out"
    # Must match the __init__ signature's required args; domain=something
    # arbitrary since we're just testing the stub lookup.
    processor = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(out),
        course_code="MINI_101",
        domain="bogus",
    )
    assert processor._load_classification_stub() is None


@pytest.mark.unit
def test_stub_invalid_fails_at_emit(tmp_path):
    """Bogus classification rejected by Courseforge generate_course, no files."""
    import importlib.util
    import sys

    # Path-load generate_course since it's a script, not a package module.
    repo = Path(__file__).resolve().parents[2]
    gc_path = repo / "Courseforge" / "scripts" / "generate_course.py"
    spec = importlib.util.spec_from_file_location("gc_for_tax_test", gc_path)
    gc = importlib.util.module_from_spec(spec)
    sys.modules["gc_for_tax_test"] = gc
    spec.loader.exec_module(gc)

    # Minimal course data.
    course_data = tmp_path / "course_data.json"
    course_data.write_text(json.dumps({
        "course_code": "MINI_101",
        "course_title": "Mini",
        "weeks": [
            {
                "week_number": 1,
                "title": "Kickoff",
                "objectives": [],
                "overview_text": ["Intro"],
                "readings": [],
                "content_modules": [],
            }
        ],
    }))
    out = tmp_path / "out"

    bogus = {
        "division": "BOGUS",  # invalid — not STEM/ARTS
        "primary_domain": "whatever",
        "subdomains": [],
        "topics": [],
    }
    with pytest.raises(ValueError) as exc_info:
        gc.generate_course(
            str(course_data),
            str(out),
            classification=bogus,
        )
    assert "Invalid classification" in str(exc_info.value)
    # No page files written — the fail-closed guard runs before generate_week.
    assert not (out / "week_01").exists() or not any(
        (out / "week_01").glob("*.html")
    ), "No HTML should have been written when classification is invalid"
    # Stub not written either.
    assert not (out / "course_metadata.json").exists()


@pytest.mark.unit
def test_valid_classification_emits_stub(tmp_path):
    """Valid classification triggers course_metadata.json emit + page JSON-LD."""
    import importlib.util
    import re as _re
    import sys

    repo = Path(__file__).resolve().parents[2]
    gc_path = repo / "Courseforge" / "scripts" / "generate_course.py"
    spec = importlib.util.spec_from_file_location("gc_for_tax_test2", gc_path)
    gc = importlib.util.module_from_spec(spec)
    sys.modules["gc_for_tax_test2"] = gc
    spec.loader.exec_module(gc)

    course_data = tmp_path / "course_data.json"
    course_data.write_text(json.dumps({
        "course_code": "MINI_101",
        "course_title": "Mini",
        "weeks": [
            {
                "week_number": 1,
                "title": "Kickoff",
                "objectives": [],
                "overview_text": ["Intro"],
                "readings": [],
                "content_modules": [],
            }
        ],
    }))
    out = tmp_path / "out"

    classification = {
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["software-engineering"],
        "topics": [],
    }
    gc.generate_course(
        str(course_data),
        str(out),
        classification=classification,
    )
    stub_path = out / "course_metadata.json"
    assert stub_path.exists(), "course_metadata.json must be written"
    stub = json.loads(stub_path.read_text())
    assert stub["classification"]["division"] == "STEM"
    assert stub["classification"]["primary_domain"] == "computer-science"
    assert stub["classification"]["subdomains"] == ["software-engineering"]
    assert "ontology_mappings" in stub

    # Page JSON-LD carries classification block.
    overview = out / "week_01" / "week_01_overview.html"
    assert overview.exists(), "overview page must be generated"
    html = overview.read_text()
    # JSON-LD is embedded; just grep the classification key in the blob.
    assert "\"classification\"" in html, "classification key must appear in page JSON-LD"
    assert "\"division\": \"STEM\"" in html or "\"division\":\"STEM\"" in html


@pytest.mark.unit
def test_prerequisite_pages_emit(tmp_path):
    """prerequisite_map surfaces as prerequisitePages on JSON-LD (REC-JSL-02)."""
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[2]
    gc_path = repo / "Courseforge" / "scripts" / "generate_course.py"
    spec = importlib.util.spec_from_file_location("gc_for_prereq_test", gc_path)
    gc = importlib.util.module_from_spec(spec)
    sys.modules["gc_for_prereq_test"] = gc
    spec.loader.exec_module(gc)

    course_data = tmp_path / "course_data.json"
    course_data.write_text(json.dumps({
        "course_code": "MINI_101",
        "course_title": "Mini",
        "weeks": [
            {
                "week_number": 2,
                "title": "Advanced",
                "objectives": [],
                "overview_text": ["More"],
                "readings": [],
                "content_modules": [],
            }
        ],
        "prerequisite_map": {
            "week_02_overview": ["week_01_overview"],
        },
    }))
    out = tmp_path / "out"
    gc.generate_course(str(course_data), str(out))

    overview = out / "week_02" / "week_02_overview.html"
    assert overview.exists()
    html = overview.read_text()
    assert "prerequisitePages" in html, "prerequisitePages must appear in page JSON-LD"
    assert "week_01_overview" in html
