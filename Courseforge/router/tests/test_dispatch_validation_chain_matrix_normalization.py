"""Regression test for follow-up #36 — matrix-lookup gate_id normalization.

Closes the inert-matrix bug discovered by follow-up #34 (commit
``61fd80c``): the per-block-type validator matrix at
``Courseforge/config/block_routing.yaml`` uses bare, seam-agnostic
gate-id tokens (``curie_anchoring``, ``source_ref``, ``content_type``,
``objective_ref``), but the production inter-tier validators expose
``name`` attributes with the ``outline_`` / ``rewrite_`` seam prefix
(``outline_curie_anchoring``, ``outline_source_refs``,
``outline_content_type``, ``outline_page_objectives``). Pre-fix,
:meth:`CourseforgeRouter._dispatch_validation_chain` did an exact-set
membership check against the matrix's allowed set, so EVERY production
validator was silently dropped from the dispatched chain on EVERY
canonical block_type — the W3.C work shipped wired but inert.

The fix lands a narrowly-scoped
:meth:`CourseforgeRouter._normalize_gate_id_for_matrix` helper that:

1. Strips the ``outline_`` / ``rewrite_`` seam prefix.
2. Applies the :attr:`CourseforgeRouter._MATRIX_GATE_ID_ALIASES` alias
   map for historical singular-vs-plural / semantic-equivalence
   shapes (``source_refs`` → ``source_ref``, ``page_objectives`` →
   ``objective_ref``).

The normalization is layered on TOP of :meth:`_validator_gate_id` and
fires ONLY at the matrix-lookup path; ``GateResult.gate_id`` and
``validator.name`` keep the seam-tagged form so the audit trail
(02_validation_report writer, decision-capture rationales,
remediation-suffix builder) keeps the seam visible.

This test instantiates the production
:class:`BlockCurieAnchoringValidator` + :class:`BlockSourceRefValidator`
and confirms that both ENGAGE on a canonical ``"objective"`` block
through the matrix filter — the inert-bug regression class.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockContentTypeValidator,
    BlockCurieAnchoringValidator,
    BlockPageObjectivesValidator,
    BlockSourceRefValidator,
)
from Courseforge.router.policy import (  # noqa: E402
    BlockRoutingPolicy,
    load_block_routing_policy,
)
from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.hardening.validation_gates import GateIssue, GateResult  # noqa: E402


_REPO_YAML_PATH = PROJECT_ROOT / "Courseforge" / "config" / "block_routing.yaml"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _block(
    *,
    block_type: str = "objective",
    block_id: Optional[str] = None,
    content: Any = "hello",
) -> Block:
    bid = block_id or f"page1#{block_type}_intro_0"
    return Block(
        block_id=bid,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content=content,
    )


class _SeamTaggedValidator:
    """Stub validator that emits a seam-tagged ``name`` attribute.

    Mirrors the production :class:`BlockCurieAnchoringValidator` /
    :class:`BlockSourceRefValidator` shape — the ``name`` attribute
    carries the ``outline_`` / ``rewrite_`` seam prefix so the
    matrix-lookup path MUST normalize the value to match the
    seam-agnostic YAML token vocabulary.
    """

    def __init__(self, *, name: str, result: GateResult) -> None:
        self.name = name
        self._result = result
        self.calls: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        self.calls.append({"inputs": inputs})
        return GateResult(
            gate_id=self._result.gate_id,
            validator_name=self._result.validator_name,
            validator_version=self._result.validator_version,
            passed=self._result.passed,
            score=self._result.score,
            issues=list(self._result.issues),
            action=self._result.action,
        )


def _gate_result(
    *,
    passed: bool,
    gate_id: str,
    action: Optional[str] = None,
    issues: Optional[List[GateIssue]] = None,
) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        validator_name=gate_id,
        validator_version="1.0.0",
        passed=passed,
        issues=list(issues or []),
        action=action,
    )


# ---------------------------------------------------------------------------
# Direct-helper unit tests
# ---------------------------------------------------------------------------


def test_normalize_strips_outline_prefix():
    """``outline_curie_anchoring`` → ``curie_anchoring`` (matrix token)."""
    assert (
        CourseforgeRouter._normalize_gate_id_for_matrix("outline_curie_anchoring")
        == "curie_anchoring"
    )


def test_normalize_strips_rewrite_prefix():
    """``rewrite_content_type`` → ``content_type`` (matrix token)."""
    assert (
        CourseforgeRouter._normalize_gate_id_for_matrix("rewrite_content_type")
        == "content_type"
    )


def test_normalize_aliases_source_refs_plural_to_singular():
    """Historical plural validator name → canonical singular YAML token.

    ``BlockSourceRefValidator.name == "outline_source_refs"`` (plural,
    historical), but the YAML token is ``source_ref`` (singular,
    operator-readable). The alias map bridges both surfaces.
    """
    assert (
        CourseforgeRouter._normalize_gate_id_for_matrix("outline_source_refs")
        == "source_ref"
    )
    assert (
        CourseforgeRouter._normalize_gate_id_for_matrix("rewrite_source_refs")
        == "source_ref"
    )


def test_normalize_aliases_page_objectives_to_objective_ref():
    """Semantic equivalence: page-objectives validator → objective_ref token.

    ``BlockPageObjectivesValidator`` validates that referenced objectives
    resolve on the page; the YAML uses ``objective_ref`` for the same
    semantic concept. The alias map routes both literal forms to a
    single matrix entry.
    """
    assert (
        CourseforgeRouter._normalize_gate_id_for_matrix("outline_page_objectives")
        == "objective_ref"
    )


def test_normalize_passes_through_already_canonical_tokens():
    """Bare YAML tokens (no seam prefix, no alias) round-trip unchanged."""
    for token in (
        "curie_anchoring",
        "source_ref",
        "content_type",
        "objective_ref",
        "bloom_alignment",
        "answerability",
    ):
        assert (
            CourseforgeRouter._normalize_gate_id_for_matrix(token) == token
        )


def test_normalize_handles_empty_and_non_str_inputs_defensively():
    """Empty / non-str inputs round-trip unchanged (defensive shape)."""
    assert CourseforgeRouter._normalize_gate_id_for_matrix("") == ""
    # Non-str types pass through; the helper is read-only against the
    # validator's name attribute and the upstream chain coerces.
    assert CourseforgeRouter._normalize_gate_id_for_matrix(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end matrix-engagement tests
# ---------------------------------------------------------------------------


def test_objective_block_engages_curie_validator_with_seam_tagged_name(monkeypatch):
    """Canonical ``objective`` block_type — curie validator MUST engage.

    The YAML matrix declares ``objective.required = [curie_anchoring,
    source_ref]``. A seam-tagged ``outline_curie_anchoring`` validator
    MUST pass the matrix filter and be invoked exactly once.

    Pre-fix this test would fail: the matrix filter dropped the
    seam-tagged validator on every production block_type because the
    name didn't literal-match a YAML token. The fix's normalization
    closes the inert-matrix bug.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="objective")

    # Stub seam-tagged validator carrying the production validator's
    # ``name`` (``outline_curie_anchoring``). Returns a passing result.
    curie_validator = _SeamTaggedValidator(
        name="outline_curie_anchoring",
        result=_gate_result(
            passed=True, gate_id="outline_curie_anchoring",
        ),
    )
    all_passed, results, _ = router._run_validator_chain(
        blk, [curie_validator],
    )
    assert curie_validator.calls, (
        "Seam-tagged outline_curie_anchoring validator MUST engage on "
        "objective blocks (matrix declares it as required); pre-fix "
        "the matrix filter silently dropped this validator."
    )
    assert all_passed is True
    assert len(results) == 1
    # The emitted GateResult.gate_id keeps the seam-tagged form so the
    # audit trail (02_validation_report, decision capture) round-trips
    # the seam — the normalization is matrix-lookup-only.
    assert results[0].gate_id == "outline_curie_anchoring"


def test_objective_block_engages_source_refs_validator_via_alias(monkeypatch):
    """Singular-vs-plural alias MUST bridge ``outline_source_refs`` → ``source_ref``.

    ``BlockSourceRefValidator.name == "outline_source_refs"`` (plural,
    historical); the YAML token is ``source_ref`` (singular). The
    alias map closes this gap so the validator engages on objective
    blocks.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="objective")

    src_validator = _SeamTaggedValidator(
        name="outline_source_refs",
        result=_gate_result(
            passed=True, gate_id="outline_source_refs",
        ),
    )
    _, results, _ = router._run_validator_chain(blk, [src_validator])
    assert src_validator.calls, (
        "outline_source_refs validator MUST engage via the "
        "source_refs → source_ref alias on objective blocks."
    )
    assert len(results) == 1
    assert results[0].gate_id == "outline_source_refs"


def test_objective_block_failed_required_gate_stamps_regenerate(monkeypatch):
    """Failed required gate via seam-tagged validator stamps fail_action.

    Pre-fix, the matrix-filter pass dropped the seam-tagged validator
    so the action-stamp pass never even ran. Post-fix, both passes
    normalize the gate_id so the matrix's ``fail_action: regenerate``
    stamps onto the GateResult exactly as the W3.C contract documents.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="objective")

    failing_curie = _SeamTaggedValidator(
        name="outline_curie_anchoring",
        result=_gate_result(
            passed=False,
            gate_id="outline_curie_anchoring",
            action=None,  # legacy validator — matrix should stamp action
        ),
    )
    all_passed, results, _ = router._run_validator_chain(
        blk, [failing_curie],
    )
    assert all_passed is False
    assert len(results) == 1
    # Per the canonical YAML: objective.fail_action == "regenerate".
    # The action-stamp pass MUST normalize the seam-tagged gate_id to
    # find the matrix entry and stamp the canonical fail_action.
    assert results[0].action == "regenerate", (
        "Required-gate failure on a seam-tagged validator MUST stamp "
        "the matrix fail_action — pre-fix the action-stamp pass "
        "skipped the validator entirely."
    )


def test_objective_block_filters_out_non_matrix_validator(monkeypatch):
    """Validators not in the objective matrix are SKIPPED (filter still works).

    Symmetric guard: confirms the normalization doesn't accidentally
    widen the filter. ``content_type`` is NOT in
    ``objective.required`` / ``objective.optional`` (only
    ``curie_anchoring`` + ``source_ref``), so a seam-tagged
    ``outline_content_type`` validator MUST be filtered out.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="objective")

    skipped = _SeamTaggedValidator(
        name="outline_content_type",
        result=_gate_result(
            passed=False,
            gate_id="outline_content_type",
            action="regenerate",
        ),
    )
    all_passed, results, _ = router._run_validator_chain(
        blk, [skipped],
    )
    # Validator was NOT invoked because objective doesn't list
    # ``content_type`` in required or optional.
    assert skipped.calls == [], (
        "Matrix filter MUST skip gates not in objective's required ∪ "
        "optional set (only curie_anchoring + source_ref are listed)."
    )
    assert all_passed is True
    assert results == []


def test_dispatch_validation_chain_filters_real_inter_tier_validators():
    """Smoke test against the real inter-tier validator instances.

    Instantiates the actual production
    :class:`BlockCurieAnchoringValidator` + :class:`BlockSourceRefValidator`
    + :class:`BlockContentTypeValidator` + :class:`BlockPageObjectivesValidator`
    and dispatches them through the matrix filter. For
    ``block_type="objective"``, only curie + source_refs MUST survive
    the filter; content_type + page_objectives MUST be filtered out
    (objective's required set is just ``curie_anchoring`` + ``source_ref``).
    """
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="objective")

    curie = BlockCurieAnchoringValidator()
    src_refs = BlockSourceRefValidator()
    content_type = BlockContentTypeValidator()
    page_obj = BlockPageObjectivesValidator()

    filtered = router._dispatch_validation_chain(
        blk, [curie, src_refs, content_type, page_obj],
    )
    surviving_names = {getattr(v, "name", None) for v in filtered}
    assert surviving_names == {
        "outline_curie_anchoring",
        "outline_source_refs",
    }, (
        f"Objective block_type matrix MUST admit curie + source_refs "
        f"only (via the source_refs → source_ref alias). Got: "
        f"{surviving_names}"
    )


def test_assessment_item_block_filters_to_matrix_set():
    """Assessment-item matrix is more permissive — confirm the filter shape.

    ``assessment_item.required = [objective_assessment_similarity,
    bloom_alignment, answerability]``; ``optional = [distractor_entropy]``.
    None of those tokens carry a seam prefix, so this test confirms
    the normalization passes bare YAML tokens through unchanged.
    """
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="assessment_item")

    sim = _SeamTaggedValidator(
        name="objective_assessment_similarity",
        result=_gate_result(
            passed=True, gate_id="objective_assessment_similarity",
        ),
    )
    bloom = _SeamTaggedValidator(
        name="bloom_alignment",
        result=_gate_result(passed=True, gate_id="bloom_alignment"),
    )
    not_in_matrix = _SeamTaggedValidator(
        name="outline_curie_anchoring",
        result=_gate_result(passed=True, gate_id="outline_curie_anchoring"),
    )
    filtered = router._dispatch_validation_chain(
        blk, [sim, bloom, not_in_matrix],
    )
    surviving_names = {getattr(v, "name", None) for v in filtered}
    # sim + bloom survive; the seam-tagged curie validator does NOT
    # (assessment_item's matrix doesn't include curie_anchoring).
    assert surviving_names == {
        "objective_assessment_similarity",
        "bloom_alignment",
    }


if __name__ == "__main__":  # pragma: no cover - manual invocation guard
    pytest.main([__file__, "-v"])
