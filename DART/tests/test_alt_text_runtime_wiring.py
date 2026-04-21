"""Wave 30 Gap 1 — AltTextGenerator runtime wiring through _raw_text_to_accessible_html.

Pre-Wave-30, the DART pipeline never threaded an LLM backend through
``extract_document`` into :class:`AltTextGenerator`, so every run produced
``<figure>`` tags with ``alt=""`` + ``role="presentation"`` — a clear
WCAG 1.1.1 failure on image-heavy textbooks. Wave 30 Gap 1 threads an
optional ``llm=`` kwarg through ``_raw_text_to_accessible_html`` and
emits:

* A single ``logger.warning`` at pipeline entry when ``llm=None`` +
  ``source_pdf`` is set, naming the file that will land with
  decorative alt-text.
* One ``decision_type="alt_text_generation"`` capture per run
  summarising whether the run used LLM alt-text or fell back.

These tests exercise the wiring contract. They do NOT hit a real LLM —
we use synthetic classifiers / MagicMock captures / no source_pdf so
the extractor path is exercised without network dependencies.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.tools.pipeline_tools import _raw_text_to_accessible_html  # noqa: E402


@pytest.fixture
def fake_pdf(tmp_path):
    """Minimal fake-PDF path. The pipeline never parses its content
    because the extractor's pdftotext call fails on the stub payload
    and falls back to the raw-text-only path — exactly the code path
    Wave 30 Gap 1 instruments for the no-LLM warning + decision capture.
    """
    pdf = tmp_path / "wcag_runtime_wiring.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return pdf


@pytest.mark.unit
def test_no_llm_backend_emits_single_warning(fake_pdf, tmp_path, caplog):
    """When no LLM backend is injected, the pipeline must log exactly one
    start-of-run warning naming the PDF. Prior to Wave 30 this ran silent
    so operators had no signal that figures were falling back to decorative
    alt-text."""
    caplog.set_level(logging.WARNING, logger="MCP.tools.pipeline_tools")

    _raw_text_to_accessible_html(
        "Some pdftotext body.\n\nAnother paragraph.\n",
        "Runtime Wiring Test",
        source_pdf=str(fake_pdf),
        output_path=str(tmp_path / "out.html"),
        llm=None,
    )

    matching = [
        rec
        for rec in caplog.records
        if "Alt-text generation skipped" in rec.getMessage()
    ]
    assert len(matching) == 1, (
        f"Expected exactly one 'Alt-text generation skipped' warning, "
        f"got {len(matching)}: {[r.getMessage() for r in matching]}"
    )
    assert fake_pdf.name in matching[0].getMessage()


@pytest.mark.unit
def test_decision_capture_emitted_for_alt_text_mode_decorative_fallback(
    fake_pdf, tmp_path,
):
    """With no LLM backend, the pipeline must emit one
    ``alt_text_generation`` decision whose decision field carries
    ``decorative_fallback`` so operators can grep captures for runs
    that regressed on WCAG 1.1.1 image accessibility."""
    capture = MagicMock()

    _raw_text_to_accessible_html(
        "Some pdftotext body.\n\nAnother paragraph.\n",
        "Runtime Wiring Test",
        source_pdf=str(fake_pdf),
        output_path=str(tmp_path / "out.html"),
        llm=None,
        capture=capture,
    )

    alt_text_calls = [
        call
        for call in capture.log_decision.call_args_list
        if call.kwargs.get("decision_type") == "alt_text_generation"
    ]
    assert alt_text_calls, (
        "Expected at least one alt_text_generation decision emit; "
        "Wave 30 Gap 1 run-level summary regressed"
    )
    # The decorative-fallback emit should carry the mode explicitly
    # so downstream scrapers can filter for runs that fell back.
    rationales = [c.kwargs.get("rationale", "") for c in alt_text_calls]
    assert any("decorative_fallback" in r for r in rationales), (
        f"Expected 'decorative_fallback' in at least one alt_text_generation "
        f"rationale, got: {rationales}"
    )
    # 20-char minimum per schema — every rationale should easily clear.
    for r in rationales:
        assert len(r) >= 20


@pytest.mark.unit
def test_decision_capture_emitted_for_alt_text_mode_llm_generation(
    fake_pdf, tmp_path,
):
    """With an LLM backend injected, the decision field must carry
    ``llm_generation`` so operators can audit which runs actually
    exercised the vision-alt-text path."""
    capture = MagicMock()

    # Inject a stub backend — we only need it to be truthy; the
    # extractor may not actually call it for this synthetic PDF, but
    # the pipeline-entry decision capture fires based on backend
    # presence regardless.
    stub_backend = MagicMock(name="stub_llm_backend")

    _raw_text_to_accessible_html(
        "Some pdftotext body.\n\nAnother paragraph.\n",
        "Runtime Wiring Test",
        source_pdf=str(fake_pdf),
        output_path=str(tmp_path / "out.html"),
        llm=stub_backend,
        capture=capture,
    )

    alt_text_calls = [
        call
        for call in capture.log_decision.call_args_list
        if call.kwargs.get("decision_type") == "alt_text_generation"
    ]
    assert alt_text_calls
    rationales = [c.kwargs.get("rationale", "") for c in alt_text_calls]
    assert any("llm_generation" in r for r in rationales), (
        f"Expected 'llm_generation' in at least one alt_text_generation "
        f"rationale (backend injected), got: {rationales}"
    )


@pytest.mark.unit
def test_no_per_figure_warning_spam_on_empty_extraction(
    fake_pdf, tmp_path, caplog,
):
    """The start-of-run warning must fire exactly once per pipeline
    call regardless of how many figures exist in the PDF. This guards
    against the audit's explicit 'no per-figure warning spam' constraint
    (279 figures in a real textbook must produce 1 warning, not 279)."""
    caplog.set_level(logging.WARNING, logger="MCP.tools.pipeline_tools")

    _raw_text_to_accessible_html(
        "A\n\nB\n\nC\n\nD\n",
        "Warning Spam Test",
        source_pdf=str(fake_pdf),
        output_path=str(tmp_path / "spam_test.html"),
        llm=None,
    )

    skipped_warnings = [
        rec for rec in caplog.records
        if "Alt-text generation skipped" in rec.getMessage()
    ]
    assert len(skipped_warnings) == 1, (
        f"Expected exactly 1 alt-text-skipped warning; got {len(skipped_warnings)}"
    )


@pytest.mark.unit
def test_alt_text_generator_contract_still_works_without_capture():
    """Wave 22 contract regression check: AltTextGenerator with capture=None
    (the pre-Wave-22 API shape) must still run through the caption/generic
    fallback path without crashing. Wave 30 Gap 1 only threaded an llm=
    kwarg through — it must not have broken the direct capture=None
    construction path that other DART tooling relies on."""
    from DART.pdf_converter.alt_text_generator import AltTextGenerator
    from DART.pdf_converter.image_extractor import ExtractedImage

    gen = AltTextGenerator(use_ai=False, use_ocr_fallback=False, capture=None)

    img = ExtractedImage(
        data=b"fake-png-bytes",
        format="png",
        page=2,
        width=320,
        height=240,
        bbox=(0.0, 0.0, 320.0, 240.0),
        nearby_caption="Figure 2.1: Contract regression check",
        data_uri="",
        image_hash="",
        alt_text="",
        long_description="",
        is_vector_render=False,
    )
    result = gen.generate(img)

    assert result.success
    # Caption fallback wins when use_ai=False and use_ocr_fallback=False,
    # so the caption text rides as alt-text.
    assert result.source == "caption"
    assert "Contract regression check" in result.alt_text
