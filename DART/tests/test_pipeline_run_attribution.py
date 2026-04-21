"""Wave 22 DC3 — pipeline_run_attribution capture fires on every DART run.

Pre-Wave-22, ``_raw_text_to_accessible_html`` orchestrated the entire
DART pipeline but wrote no run-attribution capture, so runs could not
be replayed from captures alone. Wave 22 emits a single
``decision_type="pipeline_run_attribution"`` record at the top of
every call with a rationale carrying:

  backend, classifier_mode, raw-text length, title, output_path state,
  figures_dir state, llm injection state, and the legacy-flag state.

This test drives a synthetic raw-text run through the pipeline and
asserts the capture was written with the expected fields.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _raw_text_to_accessible_html,
)


@pytest.fixture
def fake_pdf(tmp_path):
    """Create a minimal fake-PDF path for source_pdf attribution.

    The pipeline doesn't actually parse this — we only need the name
    for capture-file placement + rationale interpolation.
    """
    pdf = tmp_path / "sample_textbook.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return pdf


@pytest.mark.unit
def test_pipeline_run_attribution_capture_is_written(fake_pdf, tmp_path):
    """A single pipeline_run_attribution record must land on disk per call."""
    output_html = tmp_path / "out.html"
    raw_text = (
        "Chapter 1: Introduction\n\n"
        "This is body text for the attribution test.\n\n"
        "Another paragraph with enough content to survive classification.\n"
    )

    # Run the pipeline — no explicit capture passed, so the DC3 code
    # path creates + finalises a short-lived DARTDecisionCapture.
    html = _raw_text_to_accessible_html(
        raw_text,
        "Sample Textbook",
        source_pdf=str(fake_pdf),
        output_path=str(output_html),
    )

    assert html, "Pipeline returned empty HTML"

    # The capture writes to training-captures/dart/{course_code}/phase_dart-conversion/
    # where course_code is normalised from the pdf stem. For
    # "sample_textbook" that normalises to SAMPLETE_<hash>.
    from MCP.tools.dart_tools import normalize_course_code

    course_code = normalize_course_code("sample_textbook")
    capture_dir = (
        Path(__file__).resolve().parents[2]
        / "training-captures"
        / "dart"
        / course_code
        / "phase_dart-conversion"
    )

    try:
        assert capture_dir.exists(), (
            f"Capture directory missing: {capture_dir}. "
            f"DC3 run-attribution code path did not fire."
        )

        # Find the JSONL stream file(s) — there may be historical
        # entries; we only need to confirm at least one of them has a
        # pipeline_run_attribution record.
        jsonl_files = sorted(capture_dir.glob("decisions_*.jsonl"))
        assert jsonl_files, (
            f"No decisions_*.jsonl files in {capture_dir}"
        )

        found_attrib_records = []
        for jf in jsonl_files:
            for line in jf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("decision_type") == "pipeline_run_attribution":
                    found_attrib_records.append(record)

        assert found_attrib_records, (
            "No pipeline_run_attribution records were found in any "
            "JSONL file — DC3 capture regressed."
        )

        # Inspect the most recent attribution record.
        record = found_attrib_records[-1]

        # Rationale must carry the expected dynamic signals (DC3 audit).
        rationale = record.get("rationale", "")
        assert len(rationale) >= 20, (
            f"Rationale must be ≥20 chars per schema minLength, got "
            f"{len(rationale)}"
        )
        for key in ("backend", "classifier_mode", "raw_text", "title"):
            assert key in rationale, (
                f"DC3 rationale missing expected signal {key!r}: "
                f"{rationale!r}"
            )
    finally:
        # Best-effort cleanup so repeated test runs stay clean.
        import shutil

        if capture_dir.exists():
            shutil.rmtree(capture_dir.parent, ignore_errors=True)
        # Also clean LibV2 mirror.
        libv2_dir = (
            Path(__file__).resolve().parents[2]
            / "LibV2"
            / "catalog"
            / course_code
        )
        if libv2_dir.exists():
            shutil.rmtree(libv2_dir, ignore_errors=True)


@pytest.mark.unit
def test_pipeline_run_attribution_records_classifier_mode(fake_pdf, tmp_path):
    """The rationale must record whether classification ran heuristic or LLM."""
    import os

    prior = os.environ.get("DART_LLM_CLASSIFICATION")
    os.environ.pop("DART_LLM_CLASSIFICATION", None)  # heuristic mode
    try:
        output_html = tmp_path / "out.html"
        _raw_text_to_accessible_html(
            "Some minimal text.\n\nAnother paragraph.\n",
            "Mode Test",
            source_pdf=str(fake_pdf),
            output_path=str(output_html),
        )

        from MCP.tools.dart_tools import normalize_course_code

        course_code = normalize_course_code("sample_textbook")
        capture_dir = (
            Path(__file__).resolve().parents[2]
            / "training-captures"
            / "dart"
            / course_code
            / "phase_dart-conversion"
        )
        try:
            jsonl_files = sorted(capture_dir.glob("decisions_*.jsonl"))
            records = []
            for jf in jsonl_files:
                for line in jf.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("decision_type") == "pipeline_run_attribution":
                        records.append(rec)
            assert records
            assert "classifier_mode=heuristic" in records[-1]["rationale"], (
                "When DART_LLM_CLASSIFICATION is unset, the rationale "
                "must say classifier_mode=heuristic."
            )
        finally:
            import shutil

            shutil.rmtree(capture_dir.parent, ignore_errors=True)
            libv2_dir = (
                Path(__file__).resolve().parents[2]
                / "LibV2"
                / "catalog"
                / course_code
            )
            if libv2_dir.exists():
                shutil.rmtree(libv2_dir, ignore_errors=True)
    finally:
        if prior is not None:
            os.environ["DART_LLM_CLASSIFICATION"] = prior


@pytest.mark.unit
def test_pipeline_run_attribution_respects_injected_capture():
    """When a capture is injected, the emitted record uses it (no new file)."""
    from unittest.mock import MagicMock

    capture = MagicMock()
    # Drive a raw-text-only run (no source_pdf) so the DC3 code still
    # logs against the injected capture — the ``if capture is None and
    # source_pdf`` guard in the implementation means an injected
    # capture is honoured even without source_pdf.
    _raw_text_to_accessible_html(
        "Some content.\n\nMore content.\n",
        "Injected Capture Test",
        capture=capture,
    )

    assert capture.log_decision.call_count >= 1, (
        "Expected the DC3 attribution emit to land on the injected "
        "capture at least once."
    )

    # At least one of the calls must be the pipeline_run_attribution.
    matching_calls = [
        call
        for call in capture.log_decision.call_args_list
        if call.kwargs.get("decision_type") == "pipeline_run_attribution"
    ]
    assert matching_calls, (
        "No pipeline_run_attribution call was routed to the "
        "injected capture."
    )
    rationale = matching_calls[0].kwargs["rationale"]
    assert "backend=" in rationale
    assert "classifier_mode=" in rationale
