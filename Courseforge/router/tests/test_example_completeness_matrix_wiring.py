"""Regression test for CB5 — example_completeness must run in the router.

The CB5a ``ExampleCompletenessValidator`` (lib/validators/example_completeness.py)
fires ``action="regenerate"`` on a stub ``example`` block (problem statement,
no worked solution). But the per-block-type validator matrix at
``Courseforge/config/block_routing.yaml`` filters the dispatched validator
chain down to the per-block_type allowed set
(``CourseforgeRouter._dispatch_validation_chain``): a global validator NOT
listed in the block_type's ``required``/``optional`` arrays is SILENTLY
DROPPED. Pre-CB5-fix the ``example`` matrix omitted ``example_completeness``,
so the validator never ran in the router's per-block rewrite-remediation loop
— a stubbed example shipped despite the gate's regenerate verdict.

This test asserts:

1. ``example_completeness`` is in the repo ``example`` block matrix
   (``required``), so the dispatch filter KEEPS it.
2. The dispatch filter actually keeps a live ``ExampleCompletenessValidator``
   instance for an ``example`` block (the inert-matrix regression class).
3. The dispatch filter still DROPS it for a NON-example block (scope guard —
   CB5 must not bleed into other block types).
4. The remediation builder emits the concrete CB5 worked-solution directive
   for a ``gate_id="example_completeness"`` failure (so the re-roll prompt
   tells the model to render the full solution, not a blind retry).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional  # noqa: E402

from Courseforge.router.policy import load_block_routing_policy  # noqa: E402
from Courseforge.router.remediation import (  # noqa: E402
    _REMEDIATION_DIRECTIVES_BY_GATE_ID,
    _append_remediation_for_gates,
)
from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from Courseforge.scripts.blocks import Block  # noqa: E402
from lib.validators.example_completeness import (  # noqa: E402
    ExampleCompletenessValidator,
)
from MCP.hardening.validation_gates import GateIssue, GateResult  # noqa: E402


# A stub example body: a bare problem statement, no worked solution.
_STUB_EXAMPLE_HTML = (
    '<section data-cf-source-ids="semantik:fractions#b0" '
    'data-cf-content-type="example">'
    '<h3 data-cf-content-type="example">Example: Dividing Fractions</h3>'
    "<p>Divide -2/3 by n.</p>"
    "</section>"
)

# A complete example body: problem + intermediate steps + a final
# answer, comfortably above the 40-word CB5a length floor.
_COMPLETE_EXAMPLE_HTML = (
    '<section data-cf-source-ids="semantik:fractions#b0" '
    'data-cf-content-type="example">'
    '<h3 data-cf-content-type="example">Example: Dividing Fractions</h3>'
    "<p>Divide the fraction negative two-thirds by the fraction four-fifths, "
    "showing every step of the work.</p>"
    "<p>To divide by a fraction, multiply by its reciprocal. The reciprocal "
    "of four-fifths is five-fourths, so we rewrite the problem as "
    "-2/3 * 5/4 = -10/12.</p>"
    "<p>Now simplify the result by dividing the numerator and denominator by "
    "their greatest common factor, two: -10/12 = -5/6. The final answer is "
    "-5/6.</p>"
    "</section>"
)


class _ScriptedRewriteProvider:
    """Returns scripted rewrite outputs in sequence, recording the
    ``remediation_suffix`` threaded into each call.

    First call returns a stub example (CB5a flags it -> regenerate);
    second call returns a complete example (CB5a passes -> converge).
    """

    def __init__(self, htmls: List[str], block: Block) -> None:
        self._htmls = list(htmls)
        self._block = block
        self.calls: List[Dict[str, Any]] = []

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        remediation_suffix: Optional[str] = None,
        **kwargs: Any,
    ) -> Block:
        import dataclasses

        idx = min(len(self.calls), len(self._htmls) - 1)
        self.calls.append({"remediation_suffix": remediation_suffix})
        return dataclasses.replace(self._block, content=self._htmls[idx])


def _router() -> CourseforgeRouter:
    return CourseforgeRouter(policy=load_block_routing_policy())


def test_example_matrix_lists_example_completeness_required() -> None:
    """The repo example matrix must list example_completeness as required."""
    router = _router()
    block = Block(
        block_id="p#example_x_0", block_type="example", page_id="p",
        sequence=0, content="<p>stub</p>",
    )
    required, optional, fail_action = router._resolve_validator_matrix_metadata(
        block
    )
    assert "example_completeness" in required, (
        "example_completeness must be in the example block's required matrix "
        "set or the dispatch filter silently drops the CB5 stub-example gate"
    )
    assert "example_completeness" not in optional
    assert fail_action == "regenerate"


def test_dispatch_filter_keeps_cb5a_for_example_block() -> None:
    """The dispatch filter must KEEP ExampleCompletenessValidator on example."""
    router = _router()
    block = Block(
        block_id="p#example_x_0", block_type="example", page_id="p",
        sequence=0, content="<p>stub</p>",
    )
    filtered = router._dispatch_validation_chain(
        block, [ExampleCompletenessValidator()]
    )
    kept = [type(v).__name__ for v in filtered]
    assert "ExampleCompletenessValidator" in kept, (
        "CB5 regression: the matrix filter dropped ExampleCompletenessValidator "
        "from the example-block validator chain"
    )


def test_dispatch_filter_drops_cb5a_for_nonexample_block() -> None:
    """Scope guard — CB5 must not run on non-example blocks."""
    router = _router()
    # ``concept`` matrix does not list example_completeness, so the filter
    # drops it (CB5 stays scoped to example blocks).
    block = Block(
        block_id="p#concept_x_0", block_type="concept", page_id="p",
        sequence=0, content="<p>x</p>",
    )
    filtered = router._dispatch_validation_chain(
        block, [ExampleCompletenessValidator()]
    )
    kept = [type(v).__name__ for v in filtered]
    assert "ExampleCompletenessValidator" not in kept


def test_remediation_directive_for_example_completeness_is_concrete() -> None:
    """The CB5 remediation directive must demand the full worked solution."""
    assert "example_completeness" in _REMEDIATION_DIRECTIVES_BY_GATE_ID
    directive = _REMEDIATION_DIRECTIVES_BY_GATE_ID["example_completeness"]
    low = directive.lower()
    # Concrete worked-solution language (not a generic "re-emit per contract").
    assert "worked solution" in low
    assert "final answer" in low
    # Anti-fabrication: must scope numbers to the source.
    assert "source" in low

    gr = GateResult(
        gate_id="example_completeness",
        validator_name="example_completeness",
        validator_version="0.1.0",
        passed=False,
        issues=[
            GateIssue(
                severity="critical",
                code="EXAMPLE_BLOCK_INCOMPLETE",
                message="example block looks like a problem statement",
            )
        ],
        action="regenerate",
    )
    out = _append_remediation_for_gates("PROMPT", [gr])
    assert out != "PROMPT"
    assert "worked solution" in out.lower()
    assert "[example_completeness]" in out


def test_router_rerolls_stub_example_via_cb5a(monkeypatch) -> None:
    """End-to-end: the rewrite-remediation loop re-rolls a stub example.

    Using the REAL ExampleCompletenessValidator through the REAL policy
    matrix, a stub example (problem statement, no worked solution) is
    re-rolled; once the provider returns a complete example, the loop
    converges. The second dispatch must carry the CB5 worked-solution
    remediation suffix. This is the core CB5 fix: pre-fix the matrix
    silently dropped CB5a so a stub example shipped with ONE dispatch
    and no re-roll.
    """
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)
    block = Block(
        block_id="page#example_fractions_0", block_type="example",
        page_id="page", sequence=0, content=_STUB_EXAMPLE_HTML,
    )
    provider = _ScriptedRewriteProvider(
        [_STUB_EXAMPLE_HTML, _COMPLETE_EXAMPLE_HTML], block,
    )
    router = CourseforgeRouter(
        rewrite_provider=provider, policy=load_block_routing_policy(),
    )
    out = router.route_rewrite_with_remediation(
        block,
        n_candidates=3,
        regen_budget=5,
        validators=[ExampleCompletenessValidator()],
    )
    # Two dispatches: stub flagged -> regenerate -> complete passes.
    assert len(provider.calls) == 2, (
        "CB5 regression: the router did not re-roll the stub example "
        "(expected 2 rewrite dispatches, got "
        f"{len(provider.calls)}) — the matrix likely dropped CB5a"
    )
    # The re-roll prompt carried the concrete CB5 worked-solution suffix.
    second_suffix = provider.calls[1]["remediation_suffix"]
    assert second_suffix is not None
    assert "worked solution" in second_suffix.lower()
    assert "[example_completeness]" in second_suffix
    # Converged on the complete example (no escalation marker).
    assert out.escalation_marker is None
    assert isinstance(out.content, str) and "final answer" in out.content.lower()
