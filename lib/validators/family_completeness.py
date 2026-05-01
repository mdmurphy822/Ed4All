"""Wave 137b - FamilyCompletenessValidator.

Sibling to :class:`lib.validators.eval_gating.EvalGatingValidator`:

* ``EvalGatingValidator`` reads adapter behavior from
  ``<model_dir>/eval/eval_report.json``.
* ``FamilyCompletenessValidator`` reads corpus content shape from the
  post-overlay FORM_DATA dict + the family map at
  ``schemas/training/family_map.<family>.yaml``.

Both run on the ``trainforge_train::post_training_validation`` phase,
both fail closed on regressions, both emit a decision-capture event so
gating decisions are replayable.

Algorithm: for each declared family in the family map, count how many
of its CURIEs are ``anchored_status="complete"`` vs degraded in
FORM_DATA. If both counts are >0, the family is partially complete --
asymmetric anchoring would teach the adapter one side of the cluster
(e.g. ``sh:minCount`` complete + ``sh:maxCount`` degraded -> the
adapter learns to bound below but not above). The gate fires
``FAMILY_PARTIALLY_COMPLETE`` (critical).

Singletons are evaluated independently — they belong to no family, so
no family-completeness rule applies.

The validator is a no-op (passes cleanly) on families without a
declared family map (``load_family_map`` returns None) so adding a new
course family doesn't break the gate.

Issue codes:

* ``FAMILY_PARTIALLY_COMPLETE`` (critical) - mixed complete/degraded
  within one family.
* ``FAMILY_CURIE_MISSING_FROM_FORM_DATA`` (critical) - a CURIE
  declared in the family map has no entry at all in FORM_DATA.
* ``FAMILY_MAP_NOT_FOUND`` (info) - no family map for the family;
  validator is a no-op.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


class FamilyCompletenessValidator:
    """Post-training gate enforcing the cross-CURIE coupling constraint."""

    name = "family_completeness"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "family_completeness")

        family = inputs.get("family")
        if not family:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_FAMILY",
                    message=(
                        "FamilyCompletenessValidator requires a 'family' "
                        "input (e.g. 'rdf_shacl')."
                    ),
                )],
            )

        # Load the family map. Lazy import keeps the validator
        # importable in environments without pyyaml at module load.
        from lib.ontology.family_map import (  # noqa: WPS433
            FamilyMap,
            compute_family_coverage,
            load_family_map,
        )

        try:
            fm: Optional[FamilyMap] = load_family_map(family)
        except Exception as exc:  # noqa: BLE001 - bubble as gate failure
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="FAMILY_MAP_INVALID",
                    message=(
                        f"family_map for family={family} failed to load: "
                        f"{exc}"
                    ),
                )],
            )

        # No-op when no map: adding a new course family without yet
        # authoring a family map should not break the gate.
        if fm is None:
            self._emit_decision_capture(
                inputs,
                family=family,
                passed=True,
                per_family={},
                per_curie_missing=[],
                no_op=True,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="info",
                    code="FAMILY_MAP_NOT_FOUND",
                    message=(
                        f"No family_map.{family}.yaml; validator no-ops."
                    ),
                )],
            )

        # Resolve form_data. Caller may pass it directly; otherwise we
        # load via the same pathway the backfill loop uses so the gate
        # sees the post-overlay merged dict.
        form_data: Optional[Dict[str, Any]] = inputs.get("form_data")
        if form_data is None:
            try:
                from Trainforge.generators.schema_translation_generator import (  # noqa: WPS433
                    _invalidate_form_data_cache,
                    _load_form_data,
                )
                _invalidate_form_data_cache()
                form_data = _load_form_data(family)
            except Exception as exc:  # noqa: BLE001
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="FORM_DATA_UNAVAILABLE",
                        message=(
                            f"could not load FORM_DATA for family={family}: "
                            f"{exc}"
                        ),
                    )],
                )

        issues: List[GateIssue] = []

        # Coverage report per family. compute_family_coverage gives the
        # already-classified status (complete / partial / untouched).
        coverage = compute_family_coverage(fm, form_data)

        # Per-CURIE missing-from-form-data check. A CURIE declared in
        # the family map but absent from form_data is a hard failure --
        # the manifest+map invariant guarantees every map CURIE has a
        # form_data entry.
        per_curie_missing: List[str] = []
        for fam_name, curies in fm.families.items():
            for c in curies:
                if c not in form_data:
                    per_curie_missing.append(c)
                    issues.append(GateIssue(
                        severity="critical",
                        code="FAMILY_CURIE_MISSING_FROM_FORM_DATA",
                        message=(
                            f"family='{fam_name}' references CURIE '{c}' "
                            f"which has no entry in FORM_DATA."
                        ),
                    ))

        # Per-family partial-completeness check.
        per_family_summary: Dict[str, Dict[str, Any]] = {}
        for fam_name, info in coverage.items():
            complete_count = info["complete"]
            total = info["total"]
            degraded_count = total - complete_count
            per_family_summary[fam_name] = {
                "complete": complete_count,
                "degraded": degraded_count,
                "total": total,
                "status": info["status"],
            }
            if complete_count > 0 and degraded_count > 0:
                # Build a per-CURIE breakdown for the operator message.
                breakdown_parts: List[str] = []
                for c in info["curies"]:
                    entry = form_data.get(c)
                    status = (
                        getattr(entry, "anchored_status", "complete")
                        if entry is not None
                        else "missing"
                    )
                    breakdown_parts.append(f"{c}={status}")
                breakdown = ", ".join(breakdown_parts)
                issues.append(GateIssue(
                    severity="critical",
                    code="FAMILY_PARTIALLY_COMPLETE",
                    message=(
                        f"family '{fam_name}' is partially complete "
                        f"({complete_count}/{total} complete). Asymmetric "
                        f"anchoring would teach one side of the cluster. "
                        f"Backfill the degraded CURIEs OR roll the "
                        f"complete entries back. Per-CURIE: {breakdown}."
                    ),
                ))

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        self._emit_decision_capture(
            inputs,
            family=family,
            passed=passed,
            per_family=per_family_summary,
            per_curie_missing=per_curie_missing,
            no_op=False,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    def _emit_decision_capture(
        self,
        inputs: Dict[str, Any],
        *,
        family: str,
        passed: bool,
        per_family: Dict[str, Dict[str, Any]],
        per_curie_missing: List[str],
        no_op: bool,
    ) -> None:
        """Emit a metadata-shaped decision-capture event.

        Per the project's ToS surface 6 mitigation (no prose
        rationales for content-shaped surfaces), the rationale string
        is metadata only — per-family counts plus the family slug —
        never the contents of any FORM_DATA entry.
        """
        capture = inputs.get("capture")
        if capture is None:
            return
        try:
            partial_count = sum(
                1
                for info in per_family.values()
                if info["complete"] > 0 and info["complete"] < info["total"]
            )
            complete_count = sum(
                1
                for info in per_family.values()
                if info["complete"] == info["total"] and info["total"] > 0
            )
            untouched_count = sum(
                1 for info in per_family.values() if info["complete"] == 0
            )
            rationale = (
                f"FamilyCompletenessValidator family={family} "
                f"passed={passed} no_op={no_op} "
                f"families_total={len(per_family)} "
                f"families_complete={complete_count} "
                f"families_partial={partial_count} "
                f"families_untouched={untouched_count} "
                f"missing_curies={len(per_curie_missing)}"
            )
            capture.log_decision(
                decision_type="family_completeness_decision",
                decision=(
                    "family_completeness::passed"
                    if passed
                    else "family_completeness::blocked"
                ),
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 - capture is advisory
            logger.warning("family_completeness_decision capture failed: %s", exc)
