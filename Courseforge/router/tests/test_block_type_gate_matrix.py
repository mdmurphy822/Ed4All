"""Tests for the per-block-type validator matrix (GPT Feedback v2 Wave 3 W3.C).

Surface under test:

- ``Courseforge/config/block_routing.yaml`` carries a ``validators``
  block for every entry in
  ``Courseforge.scripts.blocks.BLOCK_TYPES`` (canonical 21-value
  frozenset).
- ``Courseforge/router/policy.py::BlockRoutingPolicy.validators_by_block_type``
  is populated from the YAML by ``load_block_routing_policy``;
  ``DEFAULT_BLOCK_ROUTING`` provides the hardcoded fallback.
- ``Courseforge/router/router.py::CourseforgeRouter._dispatch_validation_chain``
  filters the global validator list down to the per-block_type
  allowed set (required ∪ optional); the validate-loop in
  ``_run_validator_chain`` stamps ``fail_action`` onto required-gate
  failures and coerces optional-gate failures to ``action="pass"``.
- Backward-compat: a block_type with no validators block falls
  through to the legacy "all gates run" behavior.

Coverage matrix:

1. Round-trip YAML → policy → router dispatch.
2. Per-block-type filtering: gates not in required-or-optional skip.
3. Required-gate failure stamps ``fail_action``.
4. Optional-gate failure emits warning only (action coerced to pass).
5. Completeness regression: every BLOCK_TYPES entry is keyed in
   ``policy.validators_by_block_type``.
6. Legacy back-compat: no matrix entry → all gates dispatched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.policy import (  # noqa: E402
    DEFAULT_BLOCK_ROUTING,
    BlockRoutingPolicy,
    load_block_routing_policy,
)
from Courseforge.router.router import (  # noqa: E402
    BlockProviderSpec,
    CourseforgeRouter,
)
from Courseforge.scripts.blocks import BLOCK_TYPES  # noqa: E402
from MCP.hardening.validation_gates import GateIssue, GateResult  # noqa: E402
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _block(
    *,
    block_type: str = "concept",
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


class _NamedValidator:
    """Stub validator that exposes a stable ``gate_id`` / ``name``
    pair for matrix-filter lookup, returning a canned GateResult."""

    def __init__(self, *, gate_id: str, result: GateResult) -> None:
        self.gate_id = gate_id
        self.name = gate_id
        self._result = result
        self.calls: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        self.calls.append({"inputs": inputs})
        # Return a fresh dataclass per call so the in-place
        # ``GateResult.action`` stamp does not bleed across calls.
        return GateResult(
            gate_id=self._result.gate_id,
            validator_name=self._result.validator_name,
            validator_version=self._result.validator_version,
            passed=self._result.passed,
            score=self._result.score,
            issues=list(self._result.issues),
            action=self._result.action,
        )


_REPO_YAML_PATH = PROJECT_ROOT / "Courseforge" / "config" / "block_routing.yaml"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_round_trip_yaml_policy_router_dispatch():
    """Test 1 — round-trip YAML → policy → router dispatch.

    Loads the canonical ``Courseforge/config/block_routing.yaml`` from
    disk; asserts the loader populates
    ``policy.validators_by_block_type`` such that the ``concept`` entry
    carries ``"source_ref"`` in ``required``.
    """
    policy = load_block_routing_policy(_REPO_YAML_PATH)

    assert isinstance(policy, BlockRoutingPolicy)
    matrix = policy.validators_by_block_type
    assert "concept" in matrix, "concept block_type must carry a matrix entry"
    concept_entry = matrix["concept"]
    assert "source_ref" in concept_entry["required"], (
        "concept.required must contain source_ref per the canonical YAML"
    )
    # Required + optional + fail_action shape contract.
    assert isinstance(concept_entry["required"], list)
    assert isinstance(concept_entry["optional"], list)
    assert concept_entry["fail_action"] in {"regenerate", "escalate", "block"}


def test_per_block_type_filtering_skips_unlisted_gates(monkeypatch):
    """Test 2 — per-block-type filtering.

    A ``concept`` block with a failing
    ``objective_assessment_similarity`` gate (NOT in concept's
    required/optional set) must be SKIPPED; the validator's
    ``validate()`` is never called and no GateResult is produced.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="concept")
    # ``objective_assessment_similarity`` is in ``assessment_item``'s
    # required set, NOT in ``concept``'s required/optional. The matrix
    # filter must drop this validator from the dispatched chain.
    skipped_validator = _NamedValidator(
        gate_id="objective_assessment_similarity",
        result=_gate_result(
            passed=False,
            gate_id="objective_assessment_similarity",
            action="regenerate",
        ),
    )
    all_passed, results, _ = router._run_validator_chain(
        blk, [skipped_validator],
    )
    # Validator was not invoked because matrix filter excluded it.
    assert skipped_validator.calls == [], (
        "Matrix filter must skip gates not in required ∪ optional"
    )
    # Empty allowed set → all_passed is True (no failures recorded).
    assert all_passed is True
    assert results == []


def test_required_gate_failure_stamps_fail_action(monkeypatch):
    """Test 3 — required gate failure stamps fail_action.

    A ``concept`` block with a failing ``source_ref`` gate (in
    concept's required set) must produce a GateResult whose ``action``
    is the matrix's ``fail_action`` (``regenerate`` for concept).
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="concept")
    failing_required = _NamedValidator(
        gate_id="source_ref",
        result=_gate_result(
            passed=False,
            gate_id="source_ref",
            action=None,  # legacy validator — matrix should stamp action
        ),
    )
    all_passed, results, _ = router._run_validator_chain(
        blk, [failing_required],
    )
    assert all_passed is False
    assert len(results) == 1
    # Per the canonical YAML, concept.fail_action == "regenerate".
    assert results[0].action == "regenerate", (
        "Required-gate failure must stamp the matrix fail_action"
    )


def test_required_gate_failure_stamps_escalate_for_assessment_item(monkeypatch):
    """Test 3b — assessment_item's fail_action is ``escalate``.

    Symmetric variant of Test 3 to confirm the matrix fail_action is
    block_type-specific (not a global router constant). assessment_item
    declares ``fail_action: escalate`` so the GateResult.action stamped
    on a failed required gate is ``escalate``.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="assessment_item")
    failing_required = _NamedValidator(
        gate_id="objective_assessment_similarity",
        result=_gate_result(
            passed=False,
            gate_id="objective_assessment_similarity",
            action=None,
        ),
    )
    _, results, _ = router._run_validator_chain(blk, [failing_required])
    assert len(results) == 1
    assert results[0].action == "escalate", (
        "assessment_item.fail_action='escalate' must stamp 'escalate'"
    )


def test_optional_gate_failure_emits_warning_only(monkeypatch):
    """Test 4 — optional gate failure emits warning only.

    A ``concept`` block with a failing ``curie_anchoring`` gate (in
    concept's optional set) must produce a GateResult whose ``action``
    is coerced to ``"pass"`` (warning-only); the validate-loop's
    ``derive_default_action`` resolution then maps that to a no-fail
    outcome — ``all_passed`` stays True.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="concept")
    failing_optional = _NamedValidator(
        gate_id="curie_anchoring",
        result=_gate_result(
            passed=False,
            gate_id="curie_anchoring",
            action=None,
            issues=[
                GateIssue(
                    severity="warning",
                    code="CURIE_NOT_RESOLVED",
                    message="optional warning",
                ),
            ],
        ),
    )
    all_passed, results, _ = router._run_validator_chain(
        blk, [failing_optional],
    )
    # GateResult is preserved (issues kept) but action coerced to pass.
    assert len(results) == 1
    assert results[0].action == "pass"
    assert results[0].issues, (
        "Optional-gate issues must be preserved for the audit trail"
    )
    # Optional-gate failures must NOT short-circuit the loop.
    assert all_passed is True


def test_completeness_every_block_type_has_matrix_entry():
    """Test 5 — completeness regression.

    Every entry in ``BLOCK_TYPES`` must be keyed in
    ``policy.validators_by_block_type``. Dropping a block_type from
    the YAML matrix or the hardcoded ``DEFAULT_BLOCK_ROUTING`` table
    fails this gate at load time.
    """
    policy = load_block_routing_policy(_REPO_YAML_PATH)
    matrix = policy.validators_by_block_type
    missing = sorted(BLOCK_TYPES - set(matrix.keys()))
    assert not missing, (
        f"BLOCK_TYPES entries missing from validators_by_block_type: "
        f"{missing}. Every canonical block_type must have a matrix "
        f"entry per the W3.C contract."
    )
    # Also assert the hardcoded DEFAULT_BLOCK_ROUTING table covers
    # every BLOCK_TYPES entry — the table is the source-of-truth
    # fallback when the YAML is absent.
    missing_in_defaults = sorted(BLOCK_TYPES - set(DEFAULT_BLOCK_ROUTING.keys()))
    assert not missing_in_defaults, (
        f"BLOCK_TYPES entries missing from DEFAULT_BLOCK_ROUTING: "
        f"{missing_in_defaults}. The hardcoded fallback must cover "
        f"every canonical block_type."
    )
    # And every matrix entry must carry the {required, optional,
    # fail_action} shape contract.
    for block_type, entry in matrix.items():
        assert "required" in entry, f"{block_type} missing required[]"
        assert "optional" in entry, f"{block_type} missing optional[]"
        assert "fail_action" in entry, f"{block_type} missing fail_action"
        assert entry["fail_action"] in {"regenerate", "escalate", "block"}, (
            f"{block_type}.fail_action must be a valid enum value"
        )


def test_legacy_back_compat_no_matrix_falls_through_to_all_gates(monkeypatch):
    """Test 6 — legacy back-compat.

    A router instantiated with ``policy=None`` (no matrix wired) must
    dispatch the FULL global validator chain — every validator's
    ``validate()`` is called, no filtering applied. This preserves the
    pre-W3.C "all gates run" contract for callers that never load a
    block_routing policy.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    # Construct a router with no policy — _dispatch_validation_chain
    # MUST return the input list unchanged so every validator runs.
    router = CourseforgeRouter(policy=None)
    blk = _block(block_type="concept")
    # Mix of "would-be required" and "would-be skipped" gates; with no
    # matrix wired, both must dispatch.
    v1 = _NamedValidator(
        gate_id="source_ref",
        result=_gate_result(passed=True, gate_id="source_ref"),
    )
    v2 = _NamedValidator(
        gate_id="objective_assessment_similarity",
        result=_gate_result(
            passed=True, gate_id="objective_assessment_similarity",
        ),
    )
    v3 = _NamedValidator(
        gate_id="some_unknown_gate_id_not_in_yaml",
        result=_gate_result(
            passed=True, gate_id="some_unknown_gate_id_not_in_yaml",
        ),
    )
    all_passed, results, _ = router._run_validator_chain(
        blk, [v1, v2, v3],
    )
    # Every validator must have been invoked exactly once.
    assert len(v1.calls) == 1
    assert len(v2.calls) == 1
    assert len(v3.calls) == 1
    assert len(results) == 3
    assert all_passed is True


def test_legacy_back_compat_block_type_outside_matrix(monkeypatch):
    """Test 6b — legacy back-compat for an unknown block_type.

    If ``block.block_type`` is NOT in
    ``policy.validators_by_block_type`` (e.g. a future block_type
    added before the matrix is updated), the dispatcher must fall
    through to the legacy "all gates run" behavior — defensive.

    Implementation check via direct ``_dispatch_validation_chain``
    invocation since constructing a Block with an unknown block_type
    would fail Block.__post_init__'s enum check.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    # Build a policy whose validators_by_block_type lacks an entry
    # for ``concept`` — the matrix-filter must return the input list
    # unchanged.
    policy = BlockRoutingPolicy(
        validators_by_block_type={
            "assessment_item": {
                "required": ["objective_assessment_similarity"],
                "optional": [],
                "fail_action": "escalate",
            },
        },
    )
    router = CourseforgeRouter(policy=policy)
    blk = _block(block_type="concept")
    v1 = _NamedValidator(
        gate_id="source_ref",
        result=_gate_result(passed=True, gate_id="source_ref"),
    )
    v2 = _NamedValidator(
        gate_id="bloom_alignment",
        result=_gate_result(passed=True, gate_id="bloom_alignment"),
    )
    filtered = router._dispatch_validation_chain(blk, [v1, v2])
    # No matrix entry for concept → both validators retained.
    assert filtered == [v1, v2]
