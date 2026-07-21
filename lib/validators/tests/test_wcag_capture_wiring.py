"""H3 Wave W6b — WCAGValidator decision-capture wiring.

Per `plans/h3-validator-capture-wiring-2026-05.md` §3 W6b the
`WCAGValidator` (`lib/validators/wcag.py`) MUST emit one
`wcag_compliance_check` decision per gate-shape `validate()` call.
"""

from __future__ import annotations

from typing import Any, List

from lib.validators.wcag import WCAGValidator


class _MockCapture:
    """Minimal DecisionCapture stub — records every log_decision call."""

    def __init__(self) -> None:
        self.calls: List[dict] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


_VALID_HTML = (
    "<!DOCTYPE html>\n"
    "<html lang=\"en\">\n"
    "  <head><title>Test</title></head>\n"
    "  <body>\n"
    "    <main role=\"main\"><h1>Hello</h1><p>Body.</p></main>\n"
    "  </body>\n"
    "</html>\n"
)


def test_wcag_emits_on_inline_html():
    capture = _MockCapture()
    WCAGValidator().validate({
        "html_content": _VALID_HTML,
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "wcag_compliance_check"
    metrics = call["metrics"]
    assert "score" in metrics
    assert "critical_count" in metrics
    assert "total_issues" in metrics
    assert metrics["html_present"] is True


def test_wcag_emits_on_missing_file(tmp_path: Path):
    capture = _MockCapture()
    WCAGValidator().validate({
        "html_path": str(tmp_path / "nope.html"),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "wcag_compliance_check"
    assert call["decision"] == "failed:FILE_NOT_FOUND"
    assert call["metrics"]["html_present"] is False


def test_wcag_emits_on_empty_html():
    capture = _MockCapture()
    WCAGValidator().validate({
        "html_content": "",
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "wcag_compliance_check"
    assert call["decision"] == "failed:EMPTY_HTML"


def test_wcag_capture_none_does_not_raise():
    """Back-compat: legacy callers that don't thread a capture in
    must not blow up on `if capture is None: return`."""
    result = WCAGValidator().validate({"html_content": _VALID_HTML})
    assert result is not None
