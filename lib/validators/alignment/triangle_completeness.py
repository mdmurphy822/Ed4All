"""IB3.5 — Triangle-completeness validator (closes the alignment-100pct delta).

Framework requirement (6.5 item 4 / Course-tier gate): each objective has ≥1
activity AND ≥1 aligned assessment (the complete triangle).

HONEST SCOPE: the block→objective half (no orphan content; every CO covered by
≥1 block) is ALREADY enforced by the planner
(``lib/generation/block_planner.py``) + the objective-membership gate
(``Courseforge/router/inter_tier_gates.py``). IB3.5 closes ONLY the other half
— for every objective referenced by ≥1 block, assert ≥1 interaction/activity
block (``activity`` / ``self_check_question`` / ``scenario`` / ``problem``)
AND ≥1 assessment block (``assessment_item``) whose RESOLVED verb shares the
objective's Bloom band (reusing the IB3.1 verb-band resolver).

Default OFF (``ED4ALL_ALIGNMENT_VERB_TRIPLE``): the validator no-ops (passes,
no issues). Misses → ``OBJECTIVE_NO_ACTIVITY`` / ``OBJECTIVE_NO_ALIGNED_ASSESSMENT``
(warning day-1).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.validators.alignment.verb_triple import (
    alignment_verb_triple_enabled,
    resolve_block_verb_level,
    resolve_objective_verb_level,
    verbs_share_band,
)

logger = logging.getLogger(__name__)

_ACTIVITY_BLOCK_TYPES: frozenset = frozenset(
    {"activity", "self_check_question", "scenario", "problem"}
)
_ASSESSMENT_BLOCK_TYPES: frozenset = frozenset({"assessment_item"})

_CODE_NO_ACTIVITY: str = "OBJECTIVE_NO_ACTIVITY"
_CODE_NO_ALIGNED_ASSESSMENT: str = "OBJECTIVE_NO_ALIGNED_ASSESSMENT"

_ISSUE_LIST_CAP: int = 50


def _block_attr(block: Any, key: str) -> Any:
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, Mapping):
        return block.get(key)
    return None


def _resolve_objective_ids(block: Any) -> List[str]:
    structural = list(_block_attr(block, "objective_ids") or ())
    if structural:
        return [o for o in structural if isinstance(o, str) and o]
    content = _block_attr(block, "content")
    if isinstance(content, Mapping):
        raw = content.get("objective_ids") or content.get("objective_refs") or []
        return [o for o in raw if isinstance(o, str) and o]
    return []


def _emit_decision(
    capture: Any,
    *,
    objective_id: str,
    has_activity: bool,
    has_aligned_assessment: bool,
    triangle_complete_rate: float,
    failure_codes: List[str],
) -> None:
    """Emit one ``alignment_triangle_check`` decision per audited objective."""
    if capture is None:
        return
    decision = "passed" if not failure_codes else f"failed:{','.join(failure_codes)}"
    rationale = (
        f"IB3.5 triangle-completeness check on objective {objective_id!r}: "
        f"has_activity={has_activity}, "
        f"has_aligned_assessment={has_aligned_assessment}, "
        f"triangle_complete_rate={triangle_complete_rate:.4f}, "
        f"failure_codes={','.join(failure_codes) or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="alignment_triangle_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("DecisionCapture.log_decision raised on alignment_triangle_check: %s", exc)


class TriangleCompletenessValidator:
    """IB3.5 — every objective has >=1 activity AND >=1 band-aligned assessment."""

    name = "triangle_completeness"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        if not alignment_verb_triple_enabled():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        raw = inputs.get("blocks")
        if raw is None or not isinstance(raw, list):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_BLOCKS_INPUT",
                    message=(
                        "inputs['blocks'] is required; expected a list of "
                        "Courseforge Block instances or block-shaped dicts."
                    ),
                )],
                action="block",
            )
        blocks = list(raw)

        objectives_map: Dict[str, Dict[str, Any]] = inputs.get("objectives") or {}
        if not isinstance(objectives_map, dict):
            objectives_map = {}

        # Build {objective_id: {"activity": bool, "aligned_assessment": bool}}
        # over all blocks. An objective's Bloom band drives the assessment
        # band-alignment test.
        coverage: Dict[str, Dict[str, bool]] = {}

        for block in blocks:
            block_type = _block_attr(block, "block_type")
            is_activity = block_type in _ACTIVITY_BLOCK_TYPES
            is_assessment = block_type in _ASSESSMENT_BLOCK_TYPES
            if not (is_activity or is_assessment):
                # Still register the objective so a concept-only objective
                # surfaces as triangle-incomplete (every objective referenced
                # by >=1 block is audited).
                for obj_id in _resolve_objective_ids(block):
                    coverage.setdefault(
                        obj_id, {"activity": False, "aligned_assessment": False}
                    )
                continue

            blk_verb, blk_level = resolve_block_verb_level(block)
            for obj_id in _resolve_objective_ids(block):
                entry = coverage.setdefault(
                    obj_id, {"activity": False, "aligned_assessment": False}
                )
                if is_activity:
                    entry["activity"] = True
                if is_assessment:
                    # Band-aligned assessment: the assessment's resolved verb
                    # shares the objective's Bloom band.
                    obj_dict = objectives_map.get(obj_id)
                    obj_level = None
                    if isinstance(obj_dict, dict):
                        _, obj_level = resolve_objective_verb_level(obj_dict)
                    if obj_level is not None and verbs_share_band(
                        obj_level, blk_level
                    ):
                        entry["aligned_assessment"] = True

        issues: List[GateIssue] = []
        emitted_codes: set = set()
        complete = 0
        total = len(coverage)

        for obj_id in sorted(coverage):
            entry = coverage[obj_id]
            has_activity = entry["activity"]
            has_aligned_assessment = entry["aligned_assessment"]
            failure_codes: List[str] = []
            if not has_activity:
                failure_codes.append(_CODE_NO_ACTIVITY)
            if not has_aligned_assessment:
                failure_codes.append(_CODE_NO_ALIGNED_ASSESSMENT)
            if not failure_codes:
                complete += 1
            for code in failure_codes:
                if code not in emitted_codes or len(issues) < _ISSUE_LIST_CAP:
                    issues.append(self._issue_for(code, obj_id))
                    emitted_codes.add(code)
            rate = round(complete / total, 4) if total else 1.0
            _emit_decision(
                capture,
                objective_id=obj_id,
                has_activity=has_activity,
                has_aligned_assessment=has_aligned_assessment,
                triangle_complete_rate=rate,
                failure_codes=failure_codes,
            )

        score = round(complete / total, 4) if total else 1.0
        action = "regenerate" if issues else None
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=score,
            issues=issues,
            action=action,
        )

    @staticmethod
    def _issue_for(code: str, objective_id: str) -> GateIssue:
        if code == _CODE_NO_ACTIVITY:
            return GateIssue(
                severity="warning",
                code=_CODE_NO_ACTIVITY,
                message=(
                    f"Objective {objective_id!r} has NO activity block "
                    f"(activity / self_check_question / scenario / problem). "
                    f"The constructive-alignment triangle requires every "
                    f"objective to have >=1 practice activity."
                ),
                location=objective_id,
                suggestion="Author an activity/practice block that exercises this objective.",
            )
        return GateIssue(
            severity="warning",
            code=_CODE_NO_ALIGNED_ASSESSMENT,
            message=(
                f"Objective {objective_id!r} has NO assessment block whose "
                f"resolved verb shares its Bloom band. The triangle requires "
                f">=1 band-aligned assessment per objective."
            ),
            location=objective_id,
            suggestion="Author an assessment_item that certifies this objective at its Bloom band.",
        )


__all__ = [
    "TriangleCompletenessValidator",
    "_ACTIVITY_BLOCK_TYPES",
    "_ASSESSMENT_BLOCK_TYPES",
    "_CODE_NO_ACTIVITY",
    "_CODE_NO_ALIGNED_ASSESSMENT",
]
