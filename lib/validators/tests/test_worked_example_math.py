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


# --------------------------------------------------------------------- #
# Precision pass — five audited false-positive fixes
# (a) per-segment system scoping   (b) premise contexts
# (c) implication-chain split      (d) taught-failure suppression
# (e) "intersect at" solution keyword
# --------------------------------------------------------------------- #


# (a) Two DISTINCT problems, each with its own correct system + solution pair.
# Block-wide pooling cross-checks problem-1's pair (3,2) against problem-2's
# equation x+y=9 (3+2=5 != 9) and vice-versa -> two false positives. Scoping the
# system cross-check to one problem segment (split at "Example N:") kills both.
_TWO_CORRECT_SYSTEMS = """
<section data-cf-content-type="example">
  <h3>Two Systems</h3>
  <p>Example 1: Solve the system \\(x + y = 5\\) and \\(x - y = 1\\).</p>
  <div class="solution-line">The solution is \\((3, 2)\\).</div>
  <p>Example 2: Solve the system \\(x + y = 9\\) and \\(x - y = 3\\).</p>
  <div class="solution-line">The solution is \\((6, 3)\\).</div>
</section>
"""

# (a) TP — problem 2's stated solution (6, 5) is WRONG (6+5=11 != 9). Segmenting
# still flags it, correctly localized to problem 2 (problem 1 stays clean).
_TWO_SYSTEMS_SECOND_WRONG = """
<section data-cf-content-type="example">
  <h3>Two Systems</h3>
  <p>Example 1: Solve the system \\(x + y = 5\\) and \\(x - y = 1\\).</p>
  <div class="solution-line">The solution is \\((3, 2)\\).</div>
  <p>Example 2: Solve the system \\(x + y = 9\\) and \\(x - y = 3\\).</p>
  <div class="solution-line">The solution is \\((6, 5)\\).</div>
</section>
"""


def test_a_per_segment_two_correct_systems_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_two_sys", block_type="example",
        content=_TWO_CORRECT_SYSTEMS,
    )])
    assert not _wrong_codes(result), (
        "each problem's pair satisfies ITS OWN system; block-wide pooling would "
        "cross-attribute — per-segment scoping must not flag"
    )


def test_a_per_segment_localizes_real_defect() -> None:
    result = _run([_block(
        block_id="p#example_two_sys_wrong", block_type="example",
        content=_TWO_SYSTEMS_SECOND_WRONG,
    )])
    assert _wrong_codes(result), "problem 2's wrong pair (6,5) must still flag"
    joined = " ".join(i.message for i in result.issues)
    # Localized to problem 2's equation, not problem 1's.
    assert "6" in joined and "5" in joined


# (b) A premise: "let x = 5" under a nearby "the answer" keyword. x=5 is a
# substitution INPUT, not a claimed solution; 2x+3=11 (true x=4) must NOT flag.
_PREMISE_CONTEXT = """
<section data-cf-content-type="example">
  <p>To find the answer, let \\(x = 5\\), then substitute into \\(2x + 3 = 11\\).</p>
</section>
"""

# (b) TP — a GENUINE claimed solution (no premise marker) still flags.
_PREMISE_TP_REAL_SOLUTION = """
<section data-cf-content-type="example">
  <p>Therefore the answer is \\(x = 5\\); check \\(2x + 3 = 11\\).</p>
</section>
"""


def test_b_premise_context_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_premise", block_type="example",
        content=_PREMISE_CONTEXT,
    )])
    assert not _wrong_codes(result), (
        "'let x = 5' is a premise/substitution input, not a claimed solution"
    )


def test_b_real_stated_solution_still_flags() -> None:
    result = _run([_block(
        block_id="p#example_premise_tp", block_type="example",
        content=_PREMISE_TP_REAL_SOLUTION,
    )])
    assert _wrong_codes(result), (
        "a genuine claimed solution x=5 for 2x+3=11 (true x=4) must still flag"
    )


# (c) A CORRECT derivation chain across an implication arrow. Without splitting
# on \implies, the two true equalities collapse into a garbage all-numeric chain
# [3, 36, 12] that mis-evaluates and false-flags.
_IMPLICATION_CORRECT = """
<section data-cf-content-type="explanation">
  <p>Simplify: <code>6 / 2 = 3 \\implies 3 * 4 = 12</code>.</p>
</section>
"""

# (c) TP — the second equality is WRONG (6 + 1 = 7, not 8); the split isolates
# it so it still flags.
_IMPLICATION_WRONG = """
<section data-cf-content-type="explanation">
  <p>Chain: <code>2 * 3 = 6 \\implies 6 + 1 = 8</code>.</p>
</section>
"""


def test_c_implication_correct_chain_not_flagged() -> None:
    result = _run([_block(
        block_id="p#explanation_impl_ok", block_type="explanation",
        content=_IMPLICATION_CORRECT,
    )])
    assert not _wrong_codes(result), (
        "both equalities across \\implies are true — splitting must not flag"
    )


def test_c_implication_wrong_chain_flags() -> None:
    result = _run([_block(
        block_id="p#explanation_impl_wrong", block_type="explanation",
        content=_IMPLICATION_WRONG,
    )])
    assert _wrong_codes(result), "6 + 1 = 8 is false; the split must flag it"


# (d) The block DELIBERATELY teaches a non-solution: x=5 fails 2x+3=11 and the
# block SAYS so ("(False)" / "not a solution") right after. Intentional -> no flag.
_TAUGHT_FAILURE = """
<section data-cf-content-type="example">
  <p>Solution: \\(x = 5\\). Check \\(2x + 3 = 11\\) which gives 13 (False), so x = 5 is not a solution.</p>
</section>
"""

# (d) TP — same wrong solve WITHOUT the negation markers must still flag.
_TAUGHT_FAILURE_TP = """
<section data-cf-content-type="example">
  <p>Solution: \\(x = 5\\). Check \\(2x + 3 = 11\\).</p>
</section>
"""


def test_d_taught_failure_suppressed() -> None:
    result = _run([_block(
        block_id="p#example_taught", block_type="example",
        content=_TAUGHT_FAILURE,
    )])
    assert not _wrong_codes(result), (
        "the block itself teaches that x=5 is NOT a solution — do not flag"
    )


def test_d_wrong_solve_without_negation_still_flags() -> None:
    result = _run([_block(
        block_id="p#example_taught_tp", block_type="example",
        content=_TAUGHT_FAILURE_TP,
    )])
    assert _wrong_codes(result), "no negation marker -> the wrong solve flags"


# (e) A stated intersection point of a linear system. "intersect at" is the new
# solution keyword; the true intersection of x+y=4, x-y=0 is (2,2), so a stated
# (3, 5) does not satisfy x+y=4 and must flag.
_INTERSECT_WRONG = """
<section data-cf-content-type="example">
  <p>Find the point of intersection of \\(x + y = 4\\) and \\(x - y = 0\\). The lines intersect at \\((3, 5)\\).</p>
</section>
"""


def test_e_intersect_at_wrong_point_flags() -> None:
    result = _run([_block(
        block_id="p#example_intersect", block_type="example",
        content=_INTERSECT_WRONG,
    )])
    assert _wrong_codes(result), (
        "stated intersection (3,5) fails x+y=4 (true is (2,2)); 'intersect at' "
        "makes it a checkable claimed solution"
    )


# --------------------------------------------------------------------- #
# Radical (\sqrt / √) coverage — previously skipped wholesale by
# _UNRELIABLE_RE; now translated balanced-brace-aware (or failed closed).
# --------------------------------------------------------------------- #


# \sqrt{x} = 4 with the CORRECT claimed solution x = 16.
_RADICAL_CORRECT = """
<section data-cf-content-type="example">
  <p>Solve \\(\\sqrt{x} = 4\\).</p>
  <div class="solution-line">Solution: \\(x = 16\\).</div>
</section>
"""

# Same radical equation, WRONG claimed solution x = 15 (15 != 16).
_RADICAL_WRONG = """
<section data-cf-content-type="example">
  <p>Solve \\(\\sqrt{x} = 4\\).</p>
  <div class="solution-line">Solution: \\(x = 15\\).</div>
</section>
"""

# Nested radical \sqrt{\sqrt{x}} = 4 ; correct x = 256 (sqrt(sqrt(256))=4).
_NESTED_RADICAL_CORRECT = """
<section data-cf-content-type="example">
  <p>Solve \\(\\sqrt{\\sqrt{x}} = 4\\).</p>
  <div class="solution-line">Solution: \\(x = 256\\).</div>
</section>
"""

# Unicode √ with a numeric argument in a plain numeric identity: √27 = 5.196...
# but the block asserts the WRONG value 6 -> flagged.
_UNICODE_RADICAL_WRONG = """
<section data-cf-content-type="explanation">
  <p>Evaluate <code>√25 = 6</code>.</p>
</section>
"""

# An untranslatable radicand (unbalanced brace) must be SKIPPED, not flagged.
_RADICAL_UNTRANSLATABLE = """
<section data-cf-content-type="example">
  <p>Solve \\(\\sqrt{x = 4\\).</p>
  <div class="solution-line">Solution: \\(x = 99\\).</div>
</section>
"""


def test_radical_equation_correct_solution_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_rad_ok", block_type="example",
        content=_RADICAL_CORRECT,
    )])
    assert not _wrong_codes(result), "x=16 solves sqrt(x)=4 -> must not flag"


def test_radical_equation_wrong_solution_flagged() -> None:
    result = _run([_block(
        block_id="p#example_rad_wrong", block_type="example",
        content=_RADICAL_WRONG,
    )])
    assert _wrong_codes(result), "x=15 does not solve sqrt(x)=4 -> must flag"


def test_nested_radical_correct_solution_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_rad_nested", block_type="example",
        content=_NESTED_RADICAL_CORRECT,
    )])
    assert not _wrong_codes(result), (
        "x=256 solves sqrt(sqrt(x))=4 -> balanced-brace translation must pass"
    )


def test_unicode_radical_wrong_value_flagged() -> None:
    result = _run([_block(
        block_id="p#explanation_rad_uni", block_type="explanation",
        content=_UNICODE_RADICAL_WRONG,
    )])
    assert _wrong_codes(result), "√25 = 6 is false (√25 = 5) -> must flag"


def test_untranslatable_radical_skipped_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_rad_bad", block_type="example",
        content=_RADICAL_UNTRANSLATABLE,
    )])
    assert not _wrong_codes(result), (
        "an unbalanced \\sqrt{ radicand is failed closed (skipped), never flagged"
    )
    assert result.passed is True


# --------------------------------------------------------------------- #
# Plus-minus (±) solution-set verification
# --------------------------------------------------------------------- #


# x^2 = 4 with the CORRECT ± solution set x = ±2 (both roots satisfy).
_PM_BOTH_CORRECT = """
<section data-cf-content-type="example">
  <p>Solve \\(x^2 = 4\\).</p>
  <div class="solution-line">The solution is \\(x = \\pm 2\\).</div>
</section>
"""

# x^2 - 5x + 6 = 0 (roots 2, 3). Claimed x = 3 ± 1 -> {4, 2}; 4 is NOT a root
# (one wrong half) -> must flag.
_PM_WRONG_HALF = """
<section data-cf-content-type="example">
  <p>Solve \\(x^2 - 5x + 6 = 0\\).</p>
  <div class="solution-line">The solution is \\(x = 3 \\pm 1\\).</div>
</section>
"""

# Quadratic-formula shape with a radical: x^2 + 5x + 6 = 0,
# x = (-5 ± √1)/2 -> {-2, -3}; both roots -> must NOT flag.
_PM_QUAD_FORMULA_CORRECT = """
<section data-cf-content-type="example">
  <p>Solve \\(x^2 + 5x + 6 = 0\\).</p>
  <div class="solution-line">\\(x = \\frac{-5 \\pm \\sqrt{1}}{2}\\).</div>
</section>
"""

# Same equation, WRONG radicand \sqrt{9}: x = (-5 ± 3)/2 -> {-1, -4}; neither
# is a root of x^2+5x+6 -> must flag.
_PM_QUAD_FORMULA_WRONG = """
<section data-cf-content-type="example">
  <p>Solve \\(x^2 + 5x + 6 = 0\\).</p>
  <div class="solution-line">\\(x = \\frac{-5 \\pm \\sqrt{9}}{2}\\).</div>
</section>
"""


def test_pm_both_roots_correct_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_pm_ok", block_type="example",
        content=_PM_BOTH_CORRECT,
    )])
    assert not _wrong_codes(result), "±2 both solve x^2=4 -> must not flag"


def test_pm_wrong_half_flagged() -> None:
    result = _run([_block(
        block_id="p#example_pm_half", block_type="example",
        content=_PM_WRONG_HALF,
    )])
    assert _wrong_codes(result), (
        "x = 3 ± 1 -> {4,2}; 4 is not a root of x^2-5x+6=0 -> must flag"
    )


def test_pm_quadratic_formula_correct_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_pm_qf_ok", block_type="example",
        content=_PM_QUAD_FORMULA_CORRECT,
    )])
    assert not _wrong_codes(result), (
        "x = (-5 ± √1)/2 -> {-2,-3}; both roots of x^2+5x+6=0 -> must not flag"
    )


def test_pm_quadratic_formula_wrong_radicand_flagged() -> None:
    result = _run([_block(
        block_id="p#example_pm_qf_wrong", block_type="example",
        content=_PM_QUAD_FORMULA_WRONG,
    )])
    assert _wrong_codes(result), (
        "x = (-5 ± √9)/2 -> {-1,-4}; neither is a root -> must flag"
    )


# --------------------------------------------------------------------- #
# Gap #9 — quadratic vertex + discriminant re-derivation
# --------------------------------------------------------------------- #


# Discriminant defect (synthetic equivalent of the audited (-3)^2 - 4*3*6
# claimed as -57; actual -63). Prose claim, quadratic given separately.
_DISCRIMINANT_WRONG = """
<section data-cf-content-type="example">
  <p>For the quadratic \\(3x^2 - 3x + 6 = 0\\), the discriminant is \\(-57\\).</p>
</section>
"""

# Correct discriminant: x^2 + 5x + 6 -> 25 - 24 = 1.
_DISCRIMINANT_CORRECT = """
<section data-cf-content-type="example">
  <p>For the quadratic \\(x^2 + 5x + 6 = 0\\), the discriminant is \\(1\\).</p>
</section>
"""

# The audited arithmetic shape itself: (-3)^2 - 4*3*6 = -57 (actual -63).
_DISCRIMINANT_ARITHMETIC = """
<section data-cf-content-type="explanation">
  <p>Compute the discriminant: <code>(-3)^2 - 4*3*6 = -57</code>.</p>
</section>
"""

# Vertex defect (synthetic equivalent of the audited claimed vertex (-3, 5)).
# y = x^2 + 6x + 5 -> vertex (-3, -4); block claims (-3, 5) -> must flag.
_VERTEX_WRONG = """
<section data-cf-content-type="example">
  <p>The parabola \\(y = x^2 + 6x + 5\\) has vertex \\((-3, 5)\\).</p>
</section>
"""

# Correct vertex for the same quadratic: (-3, -4).
_VERTEX_CORRECT = """
<section data-cf-content-type="example">
  <p>The parabola \\(y = x^2 + 6x + 5\\) has vertex \\((-3, -4)\\).</p>
</section>
"""


def test_discriminant_wrong_value_flagged() -> None:
    result = _run([_block(
        block_id="p#example_disc_wrong", block_type="example",
        content=_DISCRIMINANT_WRONG,
    )])
    assert _wrong_codes(result), (
        "claimed discriminant -57 != b^2-4ac = -63 for 3x^2-3x+6 -> must flag"
    )


def test_discriminant_correct_value_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_disc_ok", block_type="example",
        content=_DISCRIMINANT_CORRECT,
    )])
    assert not _wrong_codes(result), (
        "claimed discriminant 1 == 25-24 for x^2+5x+6 -> must not flag"
    )


def test_discriminant_arithmetic_defect_flagged() -> None:
    result = _run([_block(
        block_id="p#explanation_disc_arith", block_type="explanation",
        content=_DISCRIMINANT_ARITHMETIC,
    )])
    assert _wrong_codes(result), (
        "(-3)^2 - 4*3*6 = -63, not -57 -> the numeric identity must flag"
    )


def test_vertex_wrong_value_flagged() -> None:
    result = _run([_block(
        block_id="p#example_vertex_wrong", block_type="example",
        content=_VERTEX_WRONG,
    )])
    assert _wrong_codes(result), (
        "claimed vertex (-3, 5) != computed (-3, -4) for y=x^2+6x+5 -> must flag"
    )


def test_vertex_correct_value_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_vertex_ok", block_type="example",
        content=_VERTEX_CORRECT,
    )])
    assert not _wrong_codes(result), (
        "claimed vertex (-3, -4) matches the computed vertex -> must not flag"
    )


# --------------------------------------------------------------------- #
# Genuinely-unreliable constructs stay SKIPPED (precision preserved)
# --------------------------------------------------------------------- #


# An integral is not verifiable by the deterministic translator -> skipped.
_INTEGRAL_SKIP = """
<section data-cf-content-type="example">
  <p class="solution-line">\\(\\int x^2 dx = F\\), so \\(F = 99\\).</p>
</section>
"""

# A matrix determinant construct -> skipped.
_MATRIX_SKIP = """
<section data-cf-content-type="example">
  <p class="solution-line">\\(\\det\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix} = 99\\).</p>
</section>
"""


def test_integral_construct_skipped_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_integral", block_type="example",
        content=_INTEGRAL_SKIP,
    )])
    assert not _wrong_codes(result), "calculus notation is skipped, never flagged"
    assert result.passed is True


def test_matrix_construct_skipped_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_matrix", block_type="example",
        content=_MATRIX_SKIP,
    )])
    assert not _wrong_codes(result), "matrix notation is skipped, never flagged"
    assert result.passed is True


# --------------------------------------------------------------------- #
# FP-family precision pass (residual full-corpus false positives)
# --------------------------------------------------------------------- #

# Family 1 — discriminant intermediate-capture. The discriminant is written out
# as ``b^2 - 4ac`` and THEN equated: ``9^2 - 4(2)(-5) = 121`` for the quadratic
# 2x^2 + 9x - 5 (a=2, b=9, c=-5 -> b^2-4ac = 81 + 40 = 121). The old capture
# grabbed the leading ``9`` (the b^2 first term) and "verified" it against
# b^2-4ac = 121 -> false positive. The claim must be the value after the FINAL
# ``=`` (121). Both LaTeX and plain-text renderings of the discriminant chain.
_DISC_CHAIN_LATEX_OK = (
    r'<section data-cf-content-type="example">'
    r"<p>For the quadratic \(2x^2 + 9x - 5 = 0\), the discriminant is "
    r"\(9^2 - 4(2)(-5) = 121\).</p></section>"
)
_DISC_CHAIN_LATEX_WRONG = (
    r'<section data-cf-content-type="example">'
    r"<p>For the quadratic \(2x^2 + 9x - 5 = 0\), the discriminant is "
    r"\(9^2 - 4(2)(-5) = 130\).</p></section>"
)
_DISC_CHAIN_PLAIN_OK = (
    r'<section data-cf-content-type="example">'
    r"<p>For the quadratic \(2x^2 + 9x - 5 = 0\), the discriminant is "
    r"9^2 - 4(2)(-5) = 121.</p></section>"
)
_DISC_CHAIN_PLAIN_WRONG = (
    r'<section data-cf-content-type="example">'
    r"<p>For the quadratic \(2x^2 + 9x - 5 = 0\), the discriminant is "
    r"9^2 - 4(2)(-5) = 130.</p></section>"
)


def test_f1_discriminant_chain_latex_correct_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_disc_chain_latex_ok", block_type="example",
        content=_DISC_CHAIN_LATEX_OK,
    )])
    assert not _wrong_codes(result), (
        "final equality 121 IS the discriminant of 2x^2+9x-5; the b^2 first term "
        "(9) must not be captured and mis-verified"
    )


def test_f1_discriminant_chain_latex_wrong_flagged() -> None:
    result = _run([_block(
        block_id="p#example_disc_chain_latex_wrong", block_type="example",
        content=_DISC_CHAIN_LATEX_WRONG,
    )])
    assert _wrong_codes(result), (
        "the FINAL value 130 != b^2-4ac = 121 -> the wrong discriminant flags"
    )


def test_f1_discriminant_chain_plain_correct_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_disc_chain_plain_ok", block_type="example",
        content=_DISC_CHAIN_PLAIN_OK,
    )])
    assert not _wrong_codes(result), (
        "plain-text discriminant chain resolves to 121 (correct) -> no flag"
    )


def test_f1_discriminant_chain_plain_wrong_flagged() -> None:
    result = _run([_block(
        block_id="p#example_disc_chain_plain_wrong", block_type="example",
        content=_DISC_CHAIN_PLAIN_WRONG,
    )])
    assert _wrong_codes(result), (
        "plain-text discriminant chain final value 130 != 121 -> flags"
    )


# Family 2 — taught-failure phrasing gap. A deliberate non-solution demo says
# the substitution is FALSE ("Which is false") and calls x=2 an "extraneous
# solution"; the old suppressor matched only "(false)" / "not a solution".
_TAUGHT_FAILURE_IS_FALSE = (
    r'<section data-cf-content-type="example">'
    r"<p>Test \(x = 2\): substituting into \(3x + 1 = 11\) gives \(4 = 8\). "
    r"Which is false, so x = 2 is an extraneous solution.</p></section>"
)
# TP — a genuinely wrong claimed solution with NO taught-failure phrasing flags.
_TAUGHT_FAILURE_IS_FALSE_TP = (
    r'<section data-cf-content-type="example">'
    r"<p>Therefore the solution is \(x = 5\); check \(2x + 3 = 11\).</p>"
    r"</section>"
)


def test_f2_is_false_extraneous_suppressed() -> None:
    result = _run([_block(
        block_id="p#example_extraneous", block_type="example",
        content=_TAUGHT_FAILURE_IS_FALSE,
    )])
    assert not _wrong_codes(result), (
        "'Which is false ... extraneous solution' teaches a rejected non-solution "
        "-> must not flag"
    )


def test_f2_wrong_solution_without_taught_failure_flags() -> None:
    result = _run([_block(
        block_id="p#example_extraneous_tp", block_type="example",
        content=_TAUGHT_FAILURE_IS_FALSE_TP,
    )])
    assert _wrong_codes(result), (
        "x=5 fails 2x+3=11 with no is-false / extraneous phrasing -> still flags"
    )


# Family 3 — cross-equation pooling on a \begin{cases} page. The real worked
# system lives in an (otherwise-skipped) cases environment; a foreign same-page
# self-check equation must NOT pool with the worked system's solution pair.
_CASES_FOREIGN_POOLING = (
    r'<section data-cf-content-type="example">'
    r"<h3>Solve the system by elimination</h3>"
    r"<p>Solve the system \[\begin{cases} x + y = 10 \\ x - y = 2 "
    r"\end{cases}\]</p>"
    r'<div class="solution-line">The solution to the system is \((6, 4)\).</div>'
    r"<p>Self-check: verify that \(x + 2y = 8\) and \(3x - y = 5\).</p></section>"
)
# TP — the cases system is now parsed, so a WRONG solution pair against its OWN
# system flags (a capability the wholesale cases-skip previously missed).
_CASES_WRONG_SOLUTION = (
    r'<section data-cf-content-type="example">'
    r"<h3>Solve the system</h3>"
    r"<p>Solve the system \[\begin{cases} x + y = 10 \\ x - y = 2 "
    r"\end{cases}\]</p>"
    r'<div class="solution-line">The solution to the system is \((7, 4)\).</div>'
    r"</section>"
)


def test_f3_cases_foreign_equation_not_pooled() -> None:
    result = _run([_block(
        block_id="p#example_cases_foreign", block_type="example",
        content=_CASES_FOREIGN_POOLING,
    )])
    assert not _wrong_codes(result), (
        "(6,4) solves the real cases system {x+y=10, x-y=2}; the foreign "
        "self-check equations must not pool with it"
    )


def test_f3_cases_wrong_solution_flags() -> None:
    result = _run([_block(
        block_id="p#example_cases_wrong", block_type="example",
        content=_CASES_WRONG_SOLUTION,
    )])
    assert _wrong_codes(result), (
        "(7,4) fails x+y=10 of its own cases system -> parsing the cases rows "
        "must flag it"
    )


# Family 4 — premise / chosen-point misread. ``set x = 0`` (the SET-x-to-0
# y-intercept premise) and ``choose x = 6`` (an arbitrarily chosen graphing
# point) are substitution INPUTS, not claimed solutions of a nearby equation.
_SET_X_ZERO_PREMISE = (
    r'<section data-cf-content-type="example">'
    r"<p>The solution method: so we set \(x = 0\) and check "
    r"\(2x + 5 = 11\).</p></section>"
)
_CHOOSE_POINT_PREMISE = (
    r'<section data-cf-content-type="example">'
    r"<p>To graph it, so we choose \(x = 6\); check the equation "
    r"\(x + 1 = 3\).</p></section>"
)
# TP — the SAME value stated as a genuine solution (no premise marker) flags.
_SET_X_ZERO_TP = (
    r'<section data-cf-content-type="example">'
    r"<p>Therefore the solution is \(x = 0\); check \(2x + 5 = 11\).</p>"
    r"</section>"
)


def test_f4_set_x_zero_premise_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_set_premise", block_type="example",
        content=_SET_X_ZERO_PREMISE,
    )])
    assert not _wrong_codes(result), (
        "'set x = 0' is a y-intercept premise / substitution input, not a "
        "claimed solution of 2x+5=11"
    )


def test_f4_choose_point_premise_not_flagged() -> None:
    result = _run([_block(
        block_id="p#example_choose_premise", block_type="example",
        content=_CHOOSE_POINT_PREMISE,
    )])
    assert not _wrong_codes(result), (
        "'choose x = 6' is an arbitrarily chosen graphing point, not a claimed "
        "solution of x+1=3"
    )


def test_f4_set_x_zero_real_solution_flags() -> None:
    result = _run([_block(
        block_id="p#example_set_tp", block_type="example",
        content=_SET_X_ZERO_TP,
    )])
    assert _wrong_codes(result), (
        "x=0 stated as a genuine solution (no set/choose premise) fails 2x+5=11 "
        "-> still flags"
    )
