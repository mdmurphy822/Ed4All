"""H3 Wave W6b — DART WCAGValidator decision-capture wiring.

Per `plans/h3-validator-capture-wiring-2026-05.md` §3 W6b the DART
`WCAGValidator` (lives at `DART/pdf_converter/wcag_validator.py` —
outside the `lib/validators/` tree but with the same Validator
protocol) MUST emit one `wcag_compliance_check` decision per
gate-shape `validate()` call. Per the H3 review §4.1 Correction 1.2
the validator's home directory had no `tests/` sibling at the time
of the wave; per the operator's W6b dispatch instruction, the test
lives here under `lib/validators/tests/` instead of creating the
new `DART/pdf_converter/tests/` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DART_DIR = _REPO_ROOT / "DART" / "pdf_converter"
if str(_DART_DIR) not in sys.path:
    sys.path.insert(0, str(_DART_DIR))

from wcag_validator import WCAGValidator  # noqa: E402


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
