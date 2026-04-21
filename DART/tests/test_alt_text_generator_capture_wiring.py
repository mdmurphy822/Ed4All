"""Wave 22 DC1 — AltTextGenerator emits one decision per figure.

Pre-Wave-22 the per-figure Claude call in
``DART/pdf_converter/alt_text_generator.py::_call_claude_vision`` was
uninstrumented — a Bates run with dozens of figures produced zero
alt-text capture records. Wave 22 threads an optional ``capture``
kwarg through ``AltTextGenerator``; every ``generate`` call fires one
``alt_text_generation`` decision with dynamic rationale (page, bbox,
image hash, source strategy, caption presence).

These tests exercise the caption/generic fallbacks so no Claude
traffic is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from DART.pdf_converter.alt_text_generator import AltTextGenerator
from DART.pdf_converter.image_extractor import ExtractedImage


def _make_image(
    *, caption: str = "", page: int = 3, hash_seed: bytes = b"abc"
) -> ExtractedImage:
    """Build a synthetic ``ExtractedImage`` for capture wiring tests."""
    return ExtractedImage(
        data=hash_seed,
        format="png",
        page=page,
        width=640,
        height=480,
        bbox=(10.0, 20.0, 650.0, 500.0),
        nearby_caption=caption,
        data_uri="",
        image_hash="",
        alt_text="",
        long_description="",
        is_vector_render=False,
    )


@pytest.mark.unit
def test_alt_text_generator_emits_capture_on_caption_fallback():
    """Every generated alt-text must emit a dynamic decision."""
    capture = MagicMock()

    gen = AltTextGenerator(
        use_ai=False,  # force non-AI path (caption fallback wins)
        use_ocr_fallback=False,
        capture=capture,
    )
    img = _make_image(caption="Figure 3.1: Bloom's taxonomy cognitive levels.")
    result = gen.generate(img)

    assert result.success
    assert result.source == "caption"

    # One log_decision call per figure.
    assert capture.log_decision.call_count >= 1
    call = capture.log_decision.call_args_list[0]
    assert call.kwargs["decision_type"] == "alt_text_generation"

    rationale = call.kwargs["rationale"]
    # Dynamic rationale signals.
    assert f"page {img.page}" in rationale
    assert "bbox=[10,20,650,500]" in rationale
    assert "source=caption" in rationale
    assert "caption-present" in rationale
    assert "alt len=" in rationale
    assert len(rationale) >= 20


@pytest.mark.unit
def test_alt_text_generator_emits_capture_on_generic_fallback():
    """No caption + no AI + no OCR → generic fallback still logs a decision."""
    capture = MagicMock()

    gen = AltTextGenerator(
        use_ai=False,
        use_ocr_fallback=False,
        capture=capture,
    )
    img = _make_image(caption="", page=7)
    result = gen.generate(img)

    assert result.source == "generic"
    assert capture.log_decision.call_count == 1
    rationale = capture.log_decision.call_args.kwargs["rationale"]
    assert "caption-absent" in rationale
    assert "source=generic" in rationale


@pytest.mark.unit
def test_alt_text_generator_uses_darkdecisioncapture_helper_when_present():
    """When the capture carries ``log_alt_text_decision``, call it.

    ``lib/decision_capture.py::DARTDecisionCapture.log_alt_text_decision``
    existed pre-Wave-22 with zero callers. The Wave 22 DC1 fix uses it
    as the primary path and still emits the richer ``log_decision``
    record alongside for the dynamic rationale.
    """
    capture = MagicMock()
    # Simulate a DARTDecisionCapture: has log_alt_text_decision AND log_decision.
    capture.log_alt_text_decision = MagicMock()
    capture.log_decision = MagicMock()

    gen = AltTextGenerator(
        use_ai=False,
        use_ocr_fallback=False,
        capture=capture,
    )
    img = _make_image(caption="A simple caption.", page=2)
    gen.generate(img)

    # The specialised helper fired at least once.
    assert capture.log_alt_text_decision.called, (
        "The DARTDecisionCapture.log_alt_text_decision helper should "
        "have been called (Wave 22 DC1 re-uses the zero-caller helper)."
    )
    # AND the richer log_decision record fired so the 20-char minimum
    # rationale constraint is met.
    assert capture.log_decision.called


@pytest.mark.unit
def test_alt_text_generator_silent_without_capture():
    """Without an injected capture, generate() must not crash or log."""
    gen = AltTextGenerator(
        use_ai=False,
        use_ocr_fallback=False,
        capture=None,
    )
    img = _make_image(caption="no capture here", page=5)
    result = gen.generate(img)
    assert result.success is True


@pytest.mark.unit
def test_alt_text_generator_rationale_interpolates_image_hash():
    """Two distinct images (different bytes) must produce different hashes."""
    capture = MagicMock()
    gen = AltTextGenerator(
        use_ai=False,
        use_ocr_fallback=False,
        capture=capture,
    )

    img_a = _make_image(caption="A", page=1, hash_seed=b"bytes_a")
    img_b = _make_image(caption="B", page=1, hash_seed=b"bytes_b")

    gen.generate(img_a)
    gen.generate(img_b)

    rationale_a = capture.log_decision.call_args_list[0].kwargs["rationale"]
    rationale_b = capture.log_decision.call_args_list[1].kwargs["rationale"]

    # Extract the hash segment from both rationales (it's a 12-hex substring
    # after 'p0001-'). Two different inputs must produce different hashes.
    import re

    m_a = re.search(r"p\d{4}-([0-9a-f]{12})", rationale_a)
    m_b = re.search(r"p\d{4}-([0-9a-f]{12})", rationale_b)
    assert m_a and m_b, "rationale did not carry the expected image_id shape"
    assert m_a.group(1) != m_b.group(1), (
        "Different image bytes must hash to different IDs — otherwise "
        "DC1 rationale fails the 'static boilerplate forbidden' audit."
    )
