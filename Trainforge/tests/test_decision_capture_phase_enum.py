"""CourseProcessor runs under DECISION_VALIDATION_STRICT without phase drift.

The earlier emit used ``phase="content_extraction"`` (underscore) which is
not a member of the canonical phase enum at
``schemas/events/decision_event.schema.json``. Under
``DECISION_VALIDATION_STRICT=true`` + ``VALIDATE_DECISIONS=true`` the first
``log_decision`` call raised ``ValueError`` and aborted the run.

The fix renames the literal to the canonical ``trainforge-content-analysis``.
These tests guarantee the value does not drift back and that the processor
instantiates / logs its opening decision cleanly under strict mode.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_PATH = PROJECT_ROOT / "schemas" / "events" / "decision_event.schema.json"


def _phase_enum() -> set:
    schema = json.loads(SCHEMA_PATH.read_text())
    phase = schema["properties"]["phase"]["enum"]
    # Enum includes JSON null; strip it for simple set-membership assertions.
    return {v for v in phase if isinstance(v, str)}


def _build_mini_imscc(tmp_path: Path) -> Path:
    """Create a minimal IMSCC with one HTML page so CourseProcessor has work."""
    html = (
        "<!DOCTYPE html><html><head>"
        "<title>Week 1 Overview</title>"
        "</head><body><main>"
        "<h1>Week 1 Overview</h1>"
        "<section><h2>Contrast</h2>"
        "<p>Sufficient contrast between text and background supports readers.</p>"
        "</section></main></body></html>"
    )
    page_name = "week_01_overview.html"
    manifest = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1">'
        "<organizations><organization>"
        '<item identifier="root"><item identifier="it_000" identifierref="res_000">'
        "<title>Week 1</title></item></item>"
        "</organization></organizations>"
        '<resources><resource identifier="res_000" type="webcontent" '
        f'href="{page_name}"><file href="{page_name}"/></resource></resources>'
        "</manifest>"
    )
    imscc = tmp_path / "mini.imscc"
    with zipfile.ZipFile(imscc, "w") as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr(page_name, html)
    return imscc


def test_processor_phase_is_in_canonical_enum():
    """Regression: CourseProcessor's phase MUST be a canonical enum member."""
    from Trainforge.process_course import CourseProcessor

    # __new__ sidesteps the full __init__ — we only need to read the class-
    # level phase constant, which is hard-coded inside __init__. Build a
    # real instance against a tmp path to touch the literal at runtime.
    # Use a lightweight path via tmp.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        imscc = _build_mini_imscc(Path(td))
        proc = CourseProcessor(
            imscc_path=str(imscc),
            output_dir=str(Path(td) / "out"),
            course_code="TST_101",
            division="ARTS",
            domain="accessibility",
        )
    assert proc.capture.phase in _phase_enum(), (
        f"phase={proc.capture.phase!r} not in canonical enum — strict-mode "
        f"decision validation will fail closed."
    )
    # Belt-and-suspenders: pin to the exact expected value so silent drift
    # to some other enum member also fails loudly.
    assert proc.capture.phase == "trainforge-content-analysis"


def test_processor_logs_opening_decision_under_strict_mode(monkeypatch, tmp_path):
    """Under DECISION_VALIDATION_STRICT=true, CourseProcessor.process() must
    execute the stage-1 ``imscc_extraction`` log_decision call without
    raising a ``ValueError`` from the validator.

    We don't run the full process() path here — it involves too many
    auxiliary files. Instead we explicitly drive ``log_decision`` with the
    same arguments the processor uses in ``_extract_imscc`` so the harness
    exercises exactly the strict-mode code path that previously failed.
    """
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")

    from Trainforge.process_course import CourseProcessor

    imscc = _build_mini_imscc(tmp_path)
    proc = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(tmp_path / "out"),
        course_code="TST_101",
        division="ARTS",
        domain="accessibility",
    )

    # Mirror process_course.py::_extract_imscc's opening log_decision. This
    # is the first strict-mode surface in a live run.
    proc.capture.log_decision(
        decision_type="imscc_extraction",
        decision=f"Extract {imscc.name}",
        rationale=(
            "Parse IMSCC manifest and HTML resources to build RAG corpus "
            "for LibV2 import"
        ),
    )
    # If strict mode was going to reject the phase value we'd have raised
    # above. Also assert the record landed with the canonical phase.
    assert proc.capture.decisions, "Decision record was not stored."
    record = proc.capture.decisions[-1]
    assert record.get("phase") == "trainforge-content-analysis"
    # And confirm no validation issues landed in metadata.
    metadata = record.get("metadata") or {}
    assert not metadata.get("validation_issues"), (
        f"Unexpected validation issues under strict mode: "
        f"{metadata.get('validation_issues')!r}"
    )
