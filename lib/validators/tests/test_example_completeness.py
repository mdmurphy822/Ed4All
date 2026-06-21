"""CB5a — tests for ExampleCompletenessValidator.

Confirms the stub-example safety-net gate:

  * the stored stub (``inputs/contentgen/blocks_report.json`` example
    block — the "Find the sum of 5/6 and 7/9" body, a single ``<p>``
    problem statement with no worked solution) → FAIL with
    ``action="regenerate"`` + ``EXAMPLE_BLOCK_INCOMPLETE``;
  * a complete worked example (problem + steps with ``=`` + answer +
    check, multiple ``<p>``) → PASS;
  * a non-example block → PASS;
  * empty input → graceful PASS;
  * the ``example_completeness_check`` decision-capture fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from Courseforge.scripts.blocks import Block
from lib.validators.example_completeness import (
    DEFAULT_MIN_WORD_TOKENS,
    ExampleCompletenessValidator,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STORED_BLOCKS_REPORT = _REPO_ROOT / "inputs" / "contentgen" / "blocks_report.json"

# The stored stub body, mirrored inline so the test is hermetic even if
# the fixture moves. This is the canonical CB5a failure shape.
_STUB_EXAMPLE_HTML = (
    '<section data-cf-source-ids="week_07_content_01#example_fractions_2">\n'
    '    <h3 data-cf-content-type="example">Example: Adding Fractions with '
    "Different Denominators</h3>\n"
    "    <p>Find the sum of \\(\\frac{5}{6}\\) and \\(\\frac{7}{9}\\).</p>\n"
    "</section>"
)

_COMPLETE_EXAMPLE_HTML = (
    '<section data-cf-source-ids="week_07_content_01#example_fractions_2">\n'
    '    <h3 data-cf-content-type="example">Example: Adding Fractions with '
    "Different Denominators</h3>\n"
    "    <p>Find the sum of 5/6 and 7/9.</p>\n"
    "    <p>First find the least common denominator of 6 and 9, which is 18. "
    "Rewrite each fraction with denominator 18: 5/6 = 15/18 and "
    "7/9 = 14/18.</p>\n"
    "    <p>Now add the numerators over the common denominator: "
    "15/18 + 14/18 = 29/18.</p>\n"
    "    <p>Therefore the sum is 29/18, which as a mixed number is "
    "1 11/18. Check: 29/18 is already in lowest terms.</p>\n"
    "</section>"
)


class _SpyCapture:
    """DecisionCapture-shaped spy collecting every emitted event."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        **kw: Any,
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kw,
            }
        )


def _example_block(content: Any, block_id: str = "page#example_1") -> Block:
    return Block(
        block_id=block_id,
        block_type="example",
        page_id="page",
        sequence=1,
        content=content,
    )


def _concept_block(content: str = "<p>A concept definition.</p>") -> Block:
    return Block(
        block_id="page#concept_1",
        block_type="concept",
        page_id="page",
        sequence=0,
        content=content,
    )


def _validate(blocks: List[Block], capture: Optional[_SpyCapture] = None):
    v = ExampleCompletenessValidator()
    inputs: Dict[str, Any] = {"blocks": blocks}
    if capture is not None:
        inputs["decision_capture"] = capture
    return v.validate(inputs)


# ---------------------------------------------------------------------------
# Stub example → FAIL with action=regenerate
# ---------------------------------------------------------------------------


def test_stub_example_fails_regenerate():
    capture = _SpyCapture()
    result = _validate([_example_block(_STUB_EXAMPLE_HTML)], capture)

    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "EXAMPLE_BLOCK_INCOMPLETE" in codes
    # The single <p> stub trips both arms (length floor + no solution).
    issue = next(i for i in result.issues if i.code == "EXAMPLE_BLOCK_INCOMPLETE")
    assert issue.severity == "critical"
    assert "no worked solution" in issue.message


def test_stored_fixture_stub_example_fails():
    """The actual stored ``blocks_report.json`` example block fails."""
    if not _STORED_BLOCKS_REPORT.exists():
        pytest.skip("stored blocks_report.json fixture not present")
    data = json.loads(_STORED_BLOCKS_REPORT.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("blocks", [])
    rewrite_html: Optional[str] = None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("block_type") == "example":
            rc = entry.get("rewrite_content")
            if isinstance(rc, str) and rc.strip():
                rewrite_html = rc
                break
    if rewrite_html is None:
        pytest.skip("no str-shape example rewrite_content in fixture")

    result = _validate([_example_block(rewrite_html)])
    assert result.passed is False
    assert result.action == "regenerate"
    assert any(i.code == "EXAMPLE_BLOCK_INCOMPLETE" for i in result.issues)


# ---------------------------------------------------------------------------
# Complete worked example → PASS
# ---------------------------------------------------------------------------


def test_complete_worked_example_passes():
    capture = _SpyCapture()
    result = _validate([_example_block(_COMPLETE_EXAMPLE_HTML)], capture)

    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Non-example block → PASS
# ---------------------------------------------------------------------------


def test_non_example_block_passes():
    # A thin one-line concept block must NOT be audited by this gate.
    result = _validate([_concept_block("<p>Short.</p>")])
    assert result.passed is True
    assert result.action is None
    assert result.issues == []


def test_non_example_block_does_not_emit_capture():
    capture = _SpyCapture()
    _validate([_concept_block()], capture)
    assert capture.events == []


# ---------------------------------------------------------------------------
# Empty / no-example input → graceful PASS
# ---------------------------------------------------------------------------


def test_empty_input_graceful_pass():
    result = _validate([])
    assert result.passed is True
    assert result.action is None
    assert result.issues == []


def test_only_non_example_blocks_graceful_pass():
    result = _validate([_concept_block(), _concept_block()])
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Decision capture fires
# ---------------------------------------------------------------------------


def test_decision_capture_fires_on_audit():
    capture = _SpyCapture()
    _validate([_example_block(_STUB_EXAMPLE_HTML, "page#example_stub")], capture)

    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision_type"] == "example_completeness_check"
    assert ev["decision"] == "incomplete"
    assert len(ev["rationale"]) >= 20
    feats = ev["ml_features"]
    assert feats["block_id"] == "page#example_stub"
    assert feats["block_type"] == "example"
    assert feats["has_solution_evidence"] is False
    assert feats["p_count"] == 1
    assert feats["word_count"] < DEFAULT_MIN_WORD_TOKENS
    assert feats["passed"] is False


def test_decision_capture_fires_on_complete_example():
    capture = _SpyCapture()
    _validate([_example_block(_COMPLETE_EXAMPLE_HTML)], capture)
    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision"] == "complete"
    feats = ev["ml_features"]
    assert feats["has_solution_evidence"] is True
    assert feats["p_count"] > 1
    assert feats["passed"] is True


# ---------------------------------------------------------------------------
# Solution-evidence heuristics (the OR arms)
# ---------------------------------------------------------------------------


def test_long_body_with_equals_marker_single_p_passes():
    # Single <p> but carries an '=' result marker and enough length.
    html = (
        "<section><h3>Example</h3><p>Compute the area of a rectangle whose "
        "length is 8 units and whose width is 3 units. To find the area of "
        "any rectangle you multiply its length by its width, so here we "
        "multiply the length by the width to obtain the total area: "
        "8 * 3 = 24 square units total for this particular rectangle.</p>"
        "</section>"
    )
    result = _validate([_example_block(html)])
    assert result.passed is True


def test_long_body_no_solution_evidence_fails():
    # >= 40 tokens but no '=', no result marker, no multi-<p>, no distinct
    # final line → still a stub (problem statement only) → FAIL.
    words = " ".join(["alpha"] * 50)
    html = f"<section><h3>Example</h3><p>{words}</p></section>"
    result = _validate([_example_block(html)])
    assert result.passed is False
    assert any(i.code == "EXAMPLE_BLOCK_INCOMPLETE" for i in result.issues)


def test_dict_shape_outline_example_audited():
    # Outline-tier dict shape: a thin key_claims-only example is audited.
    content = {
        "content_type": "example",
        "key_claims": ["Find the sum of 5/6 and 7/9."],
    }
    result = _validate([_example_block(content)])
    assert result.passed is False
    assert result.action == "regenerate"


# ---------------------------------------------------------------------------
# Rewrite-template div-based worked solution (false-positive fix)
# ---------------------------------------------------------------------------


def test_solution_line_div_is_evidence():
    """A worked example whose answer lives in a ``<div class="solution-line">``
    container (with a single trailing recap ``<p>``, no literal ``=``, no
    result-marker word) is recognized as complete — the false positive
    measured on the sample-course-a 7B export (week_02_content_01#example_*)."""
    html = (
        "<section><h3>Worked Example: Least Common Multiple</h3>"
        '<div class="step-row"><span class="step-label">Step 1:</span> '
        "Factor twelve into two times two times three using a factor tree.</div>"
        '<div class="step-row"><span class="step-label">Step 2:</span> '
        "Factor eighteen into two times three times three the same way.</div>"
        '<div class="step-row"><span class="step-label">Step 3:</span> '
        "Take the highest power of each prime that appears anywhere.</div>"
        '<div class="solution-line">The least common multiple of twelve and '
        "eighteen is thirty-six.</div>"
        "<p>The prime-factorization method generalizes to any pair of whole "
        "numbers you might encounter in later chapters.</p>"
        "</section>"
    )
    # Guard: the legacy '=' / multi-<p> heuristics must NOT already pass it,
    # so this test actually exercises the new div-structure branch.
    import re as _re
    stripped = _re.sub(r"<[^>]+>", " ", html)
    assert "=" not in stripped
    result = _validate([_example_block(html)])
    assert result.passed is True, [i.code for i in result.issues]


def test_two_step_rows_are_evidence():
    """More than one ``step-row`` (no ``solution-line``) is solution
    evidence."""
    html = (
        "<section><h3>Worked Example: Identifying Coefficients</h3>"
        '<div class="step-row"><span class="step-label">Step 1:</span> '
        "Identify the first coefficient as fourteen by reading the term.</div>"
        '<div class="step-row"><span class="step-label">Step 2:</span> '
        "Identify the second coefficient as fifteen in the next term too.</div>"
        "<p>Understanding coefficients is crucial for algebra and shows up "
        "again when you simplify and combine like terms in later sections.</p>"
        "</section>"
    )
    result = _validate([_example_block(html)])
    assert result.passed is True, [i.code for i in result.issues]


def test_single_step_row_no_answer_still_fails():
    """The div-structure heuristic is PRECISE: a single ``step-row`` with no
    ``solution-line`` answer container is NOT solution evidence — a genuine
    stub still fails closed (no weakening of detection)."""
    words = " ".join(["alpha"] * 50)
    html = (
        "<section><h3>Example</h3>"
        f'<div class="step-row">{words}</div></section>'
    )
    result = _validate([_example_block(html)])
    assert result.passed is False
    assert any(i.code == "EXAMPLE_BLOCK_INCOMPLETE" for i in result.issues)
