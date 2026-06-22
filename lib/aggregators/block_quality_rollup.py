"""IB6.6 — block→module→course quality rollup aggregator.

Walks every block's IB6.1 eight-dimension rubric scores and rolls them up
three tiers using BOTH a mean averaging path AND a per-dimension minimum-floor
path (framework §6.4, §6.6, p.139) plus the three hard gates:

* **Block:** ``block_pass = mean(applicable_dims) >= 2.0 AND every CORE dim
  (alignment / cognitive_load / accessibility / coherence) >= 2.0``.
* **Module (week / page group):** computes BOTH ``module_mean`` (mean of block
  means) AND ``module_dim_minimums`` (per-dimension min across blocks);
  ``module_pass = module_mean >= 2.0 AND every required (core) dim minimum
  >= 2.0`` — so no weak block hides in a strong average.
* **Course:** ``course_pass = zero Accessibility-gate failures (no block with
  Accessibility == 0) AND 100% alignment coverage (no block scoring
  Alignment == 0)``.

Three hard gates override the mean: (1) any block with Accessibility == 0
fails that block outright AND fails the course; (2) assessment Bloom <
objective Bloom caps Alignment at 1 (applied upstream in IB6.1); (3)
interaction-without-feedback caps Feedback + Coherence at 1 (applied upstream
in IB6.1 / IB6.2).

Output: ``<libv2_course>/block_quality_rollup_report.json`` (schema 1.0;
pinned at ``schemas/aggregators/block_quality_rollup.schema.json``). Falls back
to the project / trainforge dir when archival hasn't run. Carries ``summary``,
``per_module[]``, ``per_block[]`` (scores + ``weakest_dim``), and a
``quality_decision`` enum aligned to the existing ``course_status`` 5-value
enum where it overlaps.

Best-effort posture (the aggregator contract): runs post-loop in
``WorkflowRunner.run_workflow``; failure logs a warning and does NOT change
``final_status``. Only runs when ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is on.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Core (required) dimensions whose per-module minimum must clear 2.0. Imported
# lazily-tolerant from blocks.py so the aggregator stays import-light, with a
# verbatim fallback if the Courseforge package import path is unavailable.
try:  # pragma: no cover — exercised by import
    from Courseforge.scripts.blocks import (
        CORE_QUALITY_DIMENSIONS as _CORE_DIMS,
        QUALITY_DIMENSIONS as _ALL_DIMS,
    )
except Exception:  # pragma: no cover — defensive fallback
    _ALL_DIMS = (
        "alignment",
        "cognitive_load",
        "multimedia",
        "retrieval",
        "feedback",
        "engagement",
        "accessibility",
        "coherence",
    )
    _CORE_DIMS = frozenset(
        {"alignment", "cognitive_load", "accessibility", "coherence"}
    )

_MEAN_FLOOR = 2.0


class BlockQualityRollupAggregator:
    """Roll up per-block 8-dim rubric scores to module + course tiers."""

    def __init__(
        self,
        *,
        blocks: Optional[Sequence[Any]] = None,
        phase_outputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        course_code: str = "",
        run_id: str = "",
        decision_capture: Optional[Any] = None,
    ) -> None:
        # ``blocks`` may be Block dataclasses OR plain dicts carrying
        # {block_id, page_id, framework_block, quality_rubric:[{dimension,
        # score, applicable}, ...]}. The aggregator reads scores shape-
        # tolerantly so the unit tests can pass dicts.
        self.phase_outputs = phase_outputs or {}
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        self.decision_capture = decision_capture
        # When blocks aren't passed directly, resolve scored per-block rows
        # from the stashed ``block_quality_rubric`` GateResult metadata in
        # ``phase_outputs`` (the post_rewrite_validation / inter_tier_validation
        # ``_gate_results`` chain). Each resolved row is a synthetic dict the
        # rubric reader treats like a block carrying a ``quality_rubric``.
        if blocks:
            self.blocks = list(blocks)
        else:
            self.blocks = _blocks_from_phase_outputs(self.phase_outputs)

    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        per_block: List[Dict[str, Any]] = []
        modules: Dict[str, List[Dict[str, Any]]] = {}

        for idx, block in enumerate(self.blocks):
            scores, applicable = _extract_rubric(block)
            if not applicable:
                # Non-pedagogical (chrome) or unscored block — skip from rollup.
                continue
            block_id = str(_attr(block, "block_id") or f"block-{idx}")
            page_id = str(_attr(block, "page_id") or "")
            module_key = _module_key(page_id)
            fb = _attr(block, "framework_block")

            applicable_scores = [
                scores[d] for d in applicable if scores.get(d) is not None
            ]
            mean = (
                sum(applicable_scores) / len(applicable_scores)
                if applicable_scores
                else 0.0
            )
            accessibility = scores.get("accessibility")
            alignment = scores.get("alignment")
            # Hard gate 1: Accessibility == 0 fails the block outright.
            a11y_fail = accessibility == 0
            core_ok = all(
                scores.get(d) is not None and scores[d] >= 2 for d in _CORE_DIMS
            )
            block_pass = (mean >= _MEAN_FLOOR and core_ok) and not a11y_fail

            row = {
                "block_id": block_id,
                "module": module_key,
                "framework_block": fb,
                "scores": {d: scores.get(d) for d in applicable},
                "applicable_dims": sorted(applicable),
                "mean": round(mean, 4),
                "core_ok": core_ok,
                "accessibility_gate_fail": a11y_fail,
                "alignment": alignment,
                "block_pass": block_pass,
                "weakest_dim": _weakest(scores, applicable),
            }
            per_block.append(row)
            modules.setdefault(module_key, []).append(row)

        per_module = self._roll_modules(modules)
        summary = self._roll_course(per_block, per_module)

        report = {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "summary": summary,
            "per_module": per_module,
            "per_block": sorted(per_block, key=lambda r: (r["module"], r["block_id"])),
        }
        self._emit_decision(report)
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _roll_modules(
        modules: Mapping[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for module_key in sorted(modules.keys()):
            block_rows = modules[module_key]
            block_means = [r["mean"] for r in block_rows]
            module_mean = (
                sum(block_means) / len(block_means) if block_means else 0.0
            )
            # Per-dimension minimum across blocks (the no-weak-block-hides path).
            dim_minimums: Dict[str, Optional[int]] = {}
            for dim in _ALL_DIMS:
                vals = [
                    r["scores"][dim]
                    for r in block_rows
                    if dim in r["scores"] and r["scores"][dim] is not None
                ]
                dim_minimums[dim] = min(vals) if vals else None
            # Module passes only when mean >= 2.0 AND every required-dim
            # minimum >= 2.0 (a missing core-dim minimum fails the floor).
            required_min_ok = True
            for dim in _CORE_DIMS:
                mn = dim_minimums.get(dim)
                if mn is None or mn < 2:
                    required_min_ok = False
                    break
            any_a11y_fail = any(r["accessibility_gate_fail"] for r in block_rows)
            module_pass = (
                module_mean >= _MEAN_FLOOR
                and required_min_ok
                and not any_a11y_fail
            )
            out.append(
                {
                    "module": module_key,
                    "block_count": len(block_rows),
                    "module_mean": round(module_mean, 4),
                    "module_dim_minimums": dim_minimums,
                    "required_dim_minimum_ok": required_min_ok,
                    "accessibility_gate_failures": sum(
                        1 for r in block_rows if r["accessibility_gate_fail"]
                    ),
                    "module_pass": module_pass,
                    "blocks_pass": sum(1 for r in block_rows if r["block_pass"]),
                }
            )
        return out

    @staticmethod
    def _roll_course(
        per_block: Sequence[Dict[str, Any]],
        per_module: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        a11y_failures = sum(1 for r in per_block if r["accessibility_gate_fail"])
        # 100% alignment coverage = no block scoring Alignment == 0 (an orphan
        # block with no objective). A None alignment is non-applicable (skipped).
        alignment_orphans = sum(
            1 for r in per_block if r.get("alignment") == 0
        )
        course_pass = a11y_failures == 0 and alignment_orphans == 0

        # Per-dimension course minimum (the strictest min across all blocks).
        course_dim_minimums: Dict[str, Optional[int]] = {}
        for dim in _ALL_DIMS:
            vals = [
                r["scores"][dim]
                for r in per_block
                if dim in r["scores"] and r["scores"][dim] is not None
            ]
            course_dim_minimums[dim] = min(vals) if vals else None

        # quality_decision enum overlaps the course_status 5-value enum where
        # it can: a clean course is certified_accessible, an a11y failure is
        # failed (the hard a11y gate is the accessibility floor), otherwise
        # non_certified_archive (passes a11y but not the full quality bar).
        if a11y_failures > 0 or alignment_orphans > 0:
            quality_decision = "failed"
        elif all(m["module_pass"] for m in per_module) and per_module:
            quality_decision = "certified_accessible"
        else:
            quality_decision = "non_certified_archive"

        return {
            "total_blocks": len(per_block),
            "blocks_pass": sum(1 for r in per_block if r["block_pass"]),
            "total_modules": len(per_module),
            "modules_pass": sum(1 for m in per_module if m["module_pass"]),
            "accessibility_gate_failures": a11y_failures,
            "alignment_orphan_blocks": alignment_orphans,
            "course_dim_minimums": course_dim_minimums,
            "course_pass": course_pass,
            "quality_decision": quality_decision,
        }

    # ------------------------------------------------------------------
    def write(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build()
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    def _emit_decision(self, report: Mapping[str, Any]) -> None:
        capture = self.decision_capture
        if capture is None:
            return
        try:
            s = report.get("summary") or {}
            rationale = (
                f"Block-quality rollup aggregated: total_blocks="
                f"{s.get('total_blocks')}, blocks_pass={s.get('blocks_pass')}, "
                f"modules_pass={s.get('modules_pass')}/{s.get('total_modules')}, "
                f"accessibility_gate_failures={s.get('accessibility_gate_failures')}, "
                f"alignment_orphan_blocks={s.get('alignment_orphan_blocks')}, "
                f"course_pass={s.get('course_pass')}, "
                f"quality_decision={s.get('quality_decision')}, "
                f"course_dim_minimums={s.get('course_dim_minimums')}"
            )
            capture.log_decision(
                decision_type="block_quality_rollup_aggregated",
                decision=(
                    f"quality_decision={s.get('quality_decision')} "
                    f"({s.get('blocks_pass')}/{s.get('total_blocks')} blocks pass)"
                ),
                rationale=rationale,
                context=str(
                    {
                        "course_code": report.get("course_code"),
                        "run_id": report.get("run_id"),
                        "schema_version": report.get("schema_version"),
                    }
                ),
                metrics={
                    "total_blocks": s.get("total_blocks", 0),
                    "blocks_pass": s.get("blocks_pass", 0),
                    "total_modules": s.get("total_modules", 0),
                    "modules_pass": s.get("modules_pass", 0),
                    "accessibility_gate_failures": s.get(
                        "accessibility_gate_failures", 0
                    ),
                    "alignment_orphan_blocks": s.get("alignment_orphan_blocks", 0),
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "block_quality_rollup_aggregated: %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Shape-tolerant readers
# ---------------------------------------------------------------------------


def _attr(block: Any, key: str) -> Any:
    if isinstance(block, Mapping):
        return block.get(key)
    return getattr(block, key, None)


def _extract_rubric(block: Any):
    """Return ``(scores_by_dim, applicable_dims_set)`` for a block.

    Reads the IB2 ``quality_rubric`` tuple of QualityScore-likes (dataclass OR
    dict with ``dimension`` / ``score`` / ``applicable``). A dimension is
    applicable iff its entry's ``applicable`` is truthy. Returns empty sets
    when no rubric is populated (chrome / unscored).
    """
    rubric = _attr(block, "quality_rubric")
    scores: Dict[str, int] = {}
    applicable: set = set()
    if not rubric:
        return scores, applicable
    for entry in rubric:
        dim = _attr(entry, "dimension")
        score = _attr(entry, "score")
        is_applicable = _attr(entry, "applicable")
        if is_applicable is None:
            is_applicable = True
        if dim in _ALL_DIMS and is_applicable:
            applicable.add(dim)
            if isinstance(score, int):
                scores[dim] = score
    return scores, applicable


def _module_key(page_id: str) -> str:
    """Resolve a module/week grouping key from a page_id.

    Page IDs follow ``week_NN_<type>`` — the week prefix is the module. Falls
    back to the whole page_id (or ``"module-1"`` when empty).
    """
    if not page_id:
        return "module-1"
    parts = page_id.split("_")
    if len(parts) >= 2 and parts[0] == "week":
        return f"week_{parts[1]}"
    return page_id


def _weakest(scores: Dict[str, int], applicable: set) -> Optional[str]:
    candidates = [(scores[d], d) for d in applicable if scores.get(d) is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (pair[0], _ALL_DIMS.index(pair[1])))
    return candidates[0][1]


def _blocks_from_phase_outputs(
    phase_outputs: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Reconstruct synthetic scored-block dicts from stashed gate results.

    Walks ``post_rewrite_validation`` then ``inter_tier_validation`` for the
    ``block_quality_rubric`` GateResult, reading its ``metadata.per_block`` map
    (each row carries ``scores`` / ``applicable_dims`` / ``framework_block``).
    Re-projects each row into a block-like dict carrying a ``quality_rubric``
    tuple so :meth:`_extract_rubric` reads it uniformly. Returns ``[]`` when no
    rubric gate result is present (graceful skip — the aggregator emits an
    empty rollup rather than crashing).
    """
    for phase_name in ("post_rewrite_validation", "inter_tier_validation"):
        phase = phase_outputs.get(phase_name) or {}
        gate_results = phase.get("_gate_results") or []
        for gr in gate_results:
            vname = getattr(gr, "validator_name", None)
            meta = getattr(gr, "metadata", None)
            if vname != "block_quality_rubric" or not isinstance(meta, dict):
                continue
            payload = meta.get("block_quality_rubric") or {}
            per_block = payload.get("per_block")
            if not isinstance(per_block, dict):
                continue
            out: List[Dict[str, Any]] = []
            for block_id, row in per_block.items():
                if not isinstance(row, dict):
                    continue
                applicable = set(row.get("applicable_dims") or [])
                scores = row.get("scores") or {}
                rubric = [
                    {
                        "dimension": dim,
                        "score": scores.get(dim),
                        "applicable": dim in applicable,
                    }
                    for dim in _ALL_DIMS
                    if dim in applicable
                ]
                out.append(
                    {
                        "block_id": block_id,
                        "page_id": row.get("page_id", ""),
                        "framework_block": row.get("framework_block"),
                        "quality_rubric": rubric,
                    }
                )
            if out:
                return out
    return []


__all__ = ["BlockQualityRollupAggregator", "SCHEMA_VERSION"]
