"""IB3.4 — Anchored-rubric presence validator (Evaluate/Create blocks).

Framework requirement (pp.26-33, 5.2, B14, B11): Evaluate and Create blocks
MUST ship an exemplar-anchored rubric — not a bare descriptor scale — and the
rubric MUST be published BEFORE the task. Ambiguous descriptor words
("sophisticated", "persuasive") are far too ambiguous without exemplars.

For each block whose RESOLVED objective Bloom is ``evaluate`` / ``create`` AND
which is a scored/produced-work type (``assessment_item`` / ``scenario`` /
``problem`` / ``reflection_prompt``), this validator asserts
``block.anchored_rubric`` is present, non-empty (≥1 criterion, ≥2 bands per
criterion, every band carries a non-empty ``exemplar``) and
``published_before_task=True``. Misses emit ``ANCHORED_RUBRIC_MISSING`` /
``ANCHORED_RUBRIC_NO_EXEMPLAR`` / ``ANCHORED_RUBRIC_NOT_PUBLISHED_FIRST``
(warning day-1).

Default OFF (``ED4ALL_ALIGNMENT_VERB_TRIPLE``): the field defaults None +
hash-excluded so existing block hashes / snapshots are byte-identical; the
validator no-ops (passes, emits nothing) when the flag is unset OR no
Evaluate/Create blocks exist. Reuses the IB3.1 verb-triple resolver for the
objective-Bloom lookup.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.validators.alignment.verb_triple import (
    alignment_verb_triple_enabled,
    resolve_objective_verb_level,
)

logger = logging.getLogger(__name__)

#: Block types that produce/score work whose objective, at Evaluate/Create,
#: requires an anchored rubric.
_RUBRIC_BLOCK_TYPES: frozenset = frozenset(
    {"assessment_item", "scenario", "problem", "reflection_prompt"}
)
#: Objective Bloom levels that require an anchored rubric.
_RUBRIC_BLOOM_LEVELS: frozenset = frozenset({"evaluate", "create"})

_CODE_MISSING: str = "ANCHORED_RUBRIC_MISSING"
_CODE_NO_EXEMPLAR: str = "ANCHORED_RUBRIC_NO_EXEMPLAR"
_CODE_NOT_PUBLISHED_FIRST: str = "ANCHORED_RUBRIC_NOT_PUBLISHED_FIRST"

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
    block_id: Optional[str],
    block_type: Optional[str],
    objective_bloom: Optional[str],
    rubric_present: bool,
    criteria_count: int,
    exemplar_count: int,
    published_before_task: Optional[bool],
    failure_code: Optional[str],
) -> None:
    """Emit one ``anchored_rubric_check`` decision per audited block."""
    if capture is None:
        return
    decision = "passed" if failure_code is None else f"failed:{failure_code}"
    rationale = (
        f"IB3.4 anchored-rubric check on Block {block_id or 'n/a'!r} "
        f"(block_type={block_type or 'n/a'!r}, "
        f"objective_bloom={objective_bloom or 'n/a'!r}): "
        f"rubric_present={rubric_present}, criteria_count={criteria_count}, "
        f"exemplar_count={exemplar_count}, "
        f"published_before_task={published_before_task}, "
        f"failure_code={failure_code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="anchored_rubric_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("DecisionCapture.log_decision raised on anchored_rubric_check: %s", exc)


class AnchoredRubricValidator:
    """IB3.4 — Evaluate/Create blocks must ship an exemplar-anchored rubric."""

    name = "anchored_rubric"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        # Default OFF → no-op pass (no issues), byte-identical to baseline.
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
            # Missing blocks → critical (mirror BlockObjectiveDeliveryValidator).
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

        issues: List[GateIssue] = []
        emitted_codes: set = set()
        audited = 0
        passed_blocks = 0

        for block in blocks:
            block_type = _block_attr(block, "block_type")
            if block_type not in _RUBRIC_BLOCK_TYPES:
                continue
            obj_ids = _resolve_objective_ids(block)
            if not obj_ids:
                continue

            # Resolve the highest objective Bloom this block serves.
            objective_bloom: Optional[str] = None
            for obj_id in obj_ids:
                obj_dict = objectives_map.get(obj_id)
                if not isinstance(obj_dict, dict):
                    continue
                _, level = resolve_objective_verb_level(obj_dict)
                if level in _RUBRIC_BLOOM_LEVELS:
                    objective_bloom = level
                    break

            if objective_bloom not in _RUBRIC_BLOOM_LEVELS:
                # Block's objective is below Evaluate → out of scope (pass).
                continue

            audited += 1
            block_id = _block_attr(block, "block_id")
            rubric = _block_attr(block, "anchored_rubric")
            failure_code: Optional[str] = None
            criteria_count = 0
            exemplar_count = 0
            published_before_task: Optional[bool] = None

            if not isinstance(rubric, Mapping) or not rubric:
                failure_code = _CODE_MISSING
            else:
                criteria = rubric.get("criteria")
                criteria_list = (
                    [c for c in criteria if isinstance(c, Mapping)]
                    if isinstance(criteria, list)
                    else []
                )
                criteria_count = len(criteria_list)
                published_raw = rubric.get("published_before_task")
                published_before_task = bool(published_raw) if published_raw is not None else None

                if criteria_count < 1:
                    failure_code = _CODE_MISSING
                else:
                    # Every criterion needs >=2 bands; every band a non-empty
                    # exemplar.
                    exemplar_ok = True
                    bands_ok = True
                    for crit in criteria_list:
                        bands = crit.get("bands")
                        band_list = (
                            [b for b in bands if isinstance(b, Mapping)]
                            if isinstance(bands, list)
                            else []
                        )
                        if len(band_list) < 2:
                            bands_ok = False
                        for band in band_list:
                            exemplar = band.get("exemplar")
                            if isinstance(exemplar, str) and exemplar.strip():
                                exemplar_count += 1
                            else:
                                exemplar_ok = False
                    if not bands_ok or not exemplar_ok:
                        failure_code = _CODE_NO_EXEMPLAR
                    elif published_before_task is not True:
                        failure_code = _CODE_NOT_PUBLISHED_FIRST

            if failure_code is None:
                passed_blocks += 1
            else:
                if (
                    failure_code not in emitted_codes
                    or len(issues) < _ISSUE_LIST_CAP
                ):
                    issues.append(self._issue_for(
                        failure_code, block_id, block_type, objective_bloom
                    ))
                    emitted_codes.add(failure_code)

            _emit_decision(
                capture,
                block_id=block_id,
                block_type=block_type,
                objective_bloom=objective_bloom,
                rubric_present=isinstance(rubric, Mapping) and bool(rubric),
                criteria_count=criteria_count,
                exemplar_count=exemplar_count,
                published_before_task=published_before_task,
                failure_code=failure_code,
            )

        score = 1.0 if audited == 0 else round(passed_blocks / audited, 4)
        # Warning day-1 — passed stays True (every issue is warning-severity).
        action = "regenerate" if any(
            i.severity == "warning" for i in issues
        ) else None
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
    def _issue_for(
        code: str,
        block_id: Optional[str],
        block_type: Optional[str],
        objective_bloom: Optional[str],
    ) -> GateIssue:
        msgs = {
            _CODE_MISSING: (
                f"Block {block_id!r} ({block_type!r}) serves an "
                f"{objective_bloom!r} objective but ships NO anchored rubric. "
                f"Evaluate/Create work requires an exemplar-anchored rubric "
                f"published before the task."
            ),
            _CODE_NO_EXEMPLAR: (
                f"Block {block_id!r} ({block_type!r}, {objective_bloom!r} "
                f"objective) anchored rubric is a bare descriptor scale — "
                f"every criterion needs >=2 bands and every band a non-empty "
                f"exemplar anchor (ambiguous descriptors are far too ambiguous "
                f"without exemplars)."
            ),
            _CODE_NOT_PUBLISHED_FIRST: (
                f"Block {block_id!r} ({block_type!r}, {objective_bloom!r} "
                f"objective) anchored rubric is not marked "
                f"published_before_task=True; the rubric must be published "
                f"BEFORE the task."
            ),
        }
        suggestions = {
            _CODE_MISSING: "Attach anchored_rubric={criteria:[...], published_before_task:true}.",
            _CODE_NO_EXEMPLAR: "Add >=2 bands per criterion, each with a concrete exemplar anchor.",
            _CODE_NOT_PUBLISHED_FIRST: "Set published_before_task=True and surface the rubric before the task.",
        }
        return GateIssue(
            severity="warning",
            code=code,
            message=msgs.get(code, code),
            location=block_id,
            suggestion=suggestions.get(code),
        )


__all__ = [
    "AnchoredRubricValidator",
    "_RUBRIC_BLOCK_TYPES",
    "_RUBRIC_BLOOM_LEVELS",
    "_CODE_MISSING",
    "_CODE_NO_EXEMPLAR",
    "_CODE_NOT_PUBLISHED_FIRST",
]
