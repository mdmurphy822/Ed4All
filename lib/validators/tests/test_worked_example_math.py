"""WorkedExampleMathValidator unit tests.

Pins the symbolic worked-example verification gate against the two REAL defects
that provenance / number-blind NLI / numeric-grounding gates could not catch
(hand-found in a 7B-authored vendor-HTML build):

* a substitution system worked example that isolates ``x - 2y = -2`` with a SIGN
  error (``x = 2y + 2``), derives ``y = 3.5``, then asserts the solution
  ``(8, 5)`` — a pair that does not satisfy the second system equation;
* a sibling block concluding ``(8, 3.5)`` — a pair that satisfies NEITHER
  equation of the system.

Both synthetic reproductions MUST be flagged ``WORKED_EXAMPLE_MATH_WRONG``. A
correct worked example passes, unparseable math skips without issues, sympy-
missing degrades to a single ``SYMPY_MISSING`` warning, the two-equation system
both-equations check fires, and the numeric simplify-chain check fires.

No course slugs anywhere — the fixtures are synthetic algebra blocks built as
plain dict rows (the validator hydrates from Block instances OR JSONL dicts).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from lib.validators import worked_example_math as wem
from lib.validators.worked_example_math import (
    WorkedExampleMathValidator,
    _CODE_MATH_WRONG,
    _CODE_SYMPY_MISSING,
    _DECISION_TYPE,
)


# --------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------- #


def _block(*, block_id: str, block_type: str, content: str) -> Dict[str, Any]:
    """A rewrite-tier block dict row (validator accepts dict OR Block)."""
    return {"block_id": block_id, "block_type": block_type, "content": content}


class _RecordingCapture:
    """Minimal DecisionCapture double — records log_decision calls."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(
        self, *, decision_type: str, decision: str, rationale: str, **_kw: Any
    ) -> None:
        self.calls.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
            }
        )


def _run(
    blocks: List[Dict[str, Any]],
    *,
    capture: Optional[_RecordingCapture] = None,
    shadow: bool = False,
) -> Any:
    inputs: Dict[str, Any] = {"blocks": blocks, "shadow": shadow}
    if capture is not None:
        inputs["decision_capture"] = capture
    return WorkedExampleMathValidator().validate(inputs)


def _wrong_codes(result: Any) -> List[str]:
    return [i.code for i in result.issues if i.code == _CODE_MATH_WRONG]


# HTML fixtures ------------------------------------------------------- #

# Defect 1: substitution system, sign error x = 2y + 2 (should be 2y - 2),
# derives y = 3.5, ASSERTS solution (8, 5). System: {x - 2y = -2, 2x + y = 8}.
# (8, 5): eq1 8-10=-2 OK; eq2 16+5=21 != 8 -> fails eq2 -> flagged.
_DEFECT_SUBSTITUTION = """
<section data-cf-content-type="example">
  <h3>Solve the System by Substitution</h3>
  <div class="worked-example">
    <div class="step-row"><span class="step-label">Step 1:</span> Given the system \\(x - 2y = -2\\) and \\(2x + y = 8\\).</div>
    <div class="step-row"><span class="step-label">Step 2:</span> Isolate x: \\(x = 2y + 2\\).</div>
    <div class="step-row"><span class="step-label">Step 3:</span> Substitute and solve to get \\(y = 3.5\\).</div>
  </div>
  <div class="solution-line">Therefore the solution is \\((8, 5)\\).</div>
</section>
"""

# Defect 2 (sibling): same system, concludes (8, 3.5) which satisfies NEITHER
# equation. eq1 8-7=1 != -2 -> flagged immediately.
_DEFECT_SIBLING = """
<section data-cf-content-type="example">
  <h3>System Solution</h3>
  <p>Consider the system \\(x - 2y = -2\\) and \\(2x + y = 8\\).</p>
  <div class="solution-line">The solution to the system is \\((8, 3.5)\\).</div>
</section>
"""

# Correct single-variable solve: 4q - 4 - 3q - 3 = 1 -> q - 7 = 1 -> q = 8.
_CORRECT_SOLVE = """
<section data-cf-content-type="example">
  <div class="worked-example">
    <div class="step-row"><span class="step-label">Step 1:</span> Solve: \\(4q - 4 - 3q - 3 = 1\\).</div>
    <div class="step-row"><span class="step-label">Step 2:</span> Combine like terms: \\(q - 7 = 1\\).</div>
    <div class="step-row"><span class="step-label">Step 3:</span> Add 7 to both sides: \\(q = 8\\).</div>
    <div class="solution-line">Solution: \\(q = 8\\).</div>
  </div>
</section>
"""

# Correct two-equation system: {x + y = 10, x - y = 2}, solution (6, 4).
_CORRECT_SYSTEM = """
<section data-cf-content-type="example">
  <p>Solve the system \\(x + y = 10\\) and \\(x - y = 2\\).</p>
  <div class="solution-line">The solution is \\((6, 4)\\).</div>
</section>
"""

# Wrong single-variable solve: 2x + 3 = 11 asserted x = 5 (should be 4).
_WRONG_SOLVE = """
<section data-cf-content-type="example">
  <div class="worked-example">
    <div class="step-row">Solve \\(2x + 3 = 11\\).</div>
    <div class="solution-line">Solution: \\(x = 5\\).</div>
  </div>
</section>
"""

# Wrong numeric simplify chain: 2 + 3 = 5 = 6.
_WRONG_CHAIN = """
<section data-cf-content-type="explanation">
  <p>Evaluate: <code>2 + 3 = 5 = 6</code>.</p>
</section>
"""

# Correct numeric simplify chain: -5 * 2 = -10 = -10.
_CORRECT_CHAIN = """
<section data-cf-content-type="explanation">
  <p>Distribute: <code>-5 * 2 = -10</code> and simplify <code>-10 + 15 = 5</code>.</p>
</section>
"""

# Unparseable math: a fragment with '=' whose left side does not parse.
_UNPARSEABLE = """
<section data-cf-content-type="example">
  <p>Consider \\(3 + = 5\\) and the slope \\(m = \\frac{y_2 - y_1}{x_2 - x_1}\\).</p>
</section>
"""


# --------------------------------------------------------------------- #
# The two REAL defects MUST flag
# --------------------------------------------------------------------- #


def test_real_defect_substitution_wrong_pair_flagged() -> None:
    result = _run([_block(
        block_id="page_01#example_substitution_0",
        block_type="example", content=_DEFECT_SUBSTITUTION,
    )])
    assert _wrong_codes(result), (
        "substitution defect (8,5) failing eq2 must flag "
        "WORKED_EXAMPLE_MATH_WRONG"
    )
    joined = " ".join(i.message for i in result.issues)
    assert "8" in joined and "system" in joined.lower()


def test_real_defect_sibling_pair_satisfies_neither_flagged() -> None:
    result = _run([_block(
        block_id="page_01#example_sibling_0",
        block_type="example", content=_DEFECT_SIBLING,
    )])
    assert _wrong_codes(result), "(8, 3.5) satisfies neither equation -> must flag"


# --------------------------------------------------------------------- #
# Correct math passes
# --------------------------------------------------------------------- #


def test_correct_single_variable_solve_passes() -> None:
    result = _run([_block(
        block_id="p#example_solve_0", block_type="example",
        content=_CORRECT_SOLVE,
    )])
    assert not _wrong_codes(result)
    assert result.passed is True
    assert result.score == 1.0


def test_correct_system_passes() -> None:
    result = _run([_block(
        block_id="p#example_system_0", block_type="example",
        content=_CORRECT_SYSTEM,
    )])
    assert not _wrong_codes(result)
    assert result.passed is True


# --------------------------------------------------------------------- #
# Additional pattern coverage
# --------------------------------------------------------------------- #


def test_wrong_single_variable_solve_flagged() -> None:
    result = _run([_block(
        block_id="p#example_wrongsolve_0", block_type="example",
        content=_WRONG_SOLVE,
    )])
    assert _wrong_codes(result), "x=5 does not satisfy 2x+3=11 -> must flag"


def test_system_both_equations_checked() -> None:
    """A pair satisfying eq1 but not eq2 is flagged (both-equation check)."""
    result = _run([_block(
        block_id="p#example_sys_0", block_type="example",
        content=_DEFECT_SUBSTITUTION,
    )])
    detail = " ".join(i.message for i in result.issues if i.code == _CODE_MATH_WRONG)
    # eq2 is 2x + y = 8; the residual mentions the offending equation.
    assert "2x + y = 8" in detail or "2*x" in detail or "system" in detail.lower()


def test_wrong_numeric_chain_flagged() -> None:
    result = _run([_block(
        block_id="p#explanation_chain_0", block_type="explanation",
        content=_WRONG_CHAIN,
    )])
    assert _wrong_codes(result), "chain 5 = 6 must flag"


def test_correct_numeric_chain_passes() -> None:
    result = _run([_block(
        block_id="p#explanation_chain_1", block_type="explanation",
        content=_CORRECT_CHAIN,
    )])
    assert not _wrong_codes(result)


def test_unparseable_math_skips_without_issue() -> None:
    result = _run([_block(
        block_id="p#example_unparse_0", block_type="example",
        content=_UNPARSEABLE,
    )])
    assert not _wrong_codes(result), "unparseable math must never flag"
    assert result.passed is True
    # The '3 + = 5' fragment split on '=' but the left side failed to parse.
    assert result.metadata is not None
    assert result.metadata.get("skipped_unparseable", 0) >= 1


# --------------------------------------------------------------------- #
# Graceful degrade + decision capture + shadow
# --------------------------------------------------------------------- #


def test_sympy_missing_degrades_to_warning(monkeypatch: Any) -> None:
    monkeypatch.setattr(wem, "_SYMPY_AVAILABLE", False)
    result = _run([_block(
        block_id="p#example_x", block_type="example", content=_WRONG_SOLVE,
    )])
    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert codes == [_CODE_SYMPY_MISSING]
    assert all(i.severity == "warning" for i in result.issues)


def test_decision_capture_fires_per_scored_block() -> None:
    capture = _RecordingCapture()
    _run(
        [
            _block(block_id="p#example_a", block_type="example",
                   content=_CORRECT_SOLVE),
            _block(block_id="p#example_b", block_type="example",
                   content=_WRONG_SOLVE),
        ],
        capture=capture,
    )
    assert len(capture.calls) == 2, "one decision per SCORED block"
    assert all(c["decision_type"] == _DECISION_TYPE for c in capture.calls)
    assert any(c["decision"].startswith("failed") for c in capture.calls)
    assert any(c["decision"] == "passed" for c in capture.calls)
    for c in capture.calls:
        assert len(c["rationale"]) >= 20


def test_non_target_block_types_skipped() -> None:
    """A wrong chain in an objective block (non-target) is not scored."""
    result = _run([_block(
        block_id="p#objective_0", block_type="objective", content=_WRONG_CHAIN,
    )])
    assert not result.issues
    assert result.metadata is not None
    assert result.metadata.get("scored_blocks", 0) == 0


def test_shadow_forces_warning_severity() -> None:
    result = _run(
        [_block(block_id="p#example_s", block_type="example",
                content=_WRONG_SOLVE)],
        shadow=True,
    )
    assert all(i.severity == "warning" for i in result.issues)
    assert result.passed is True
