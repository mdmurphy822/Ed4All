"""Studio per-course quality scorecard — on-demand composition (T2).

Composes a single per-course quality scorecard JSON from the governance /
evaluation artifacts a completed course carries under its LibV2 course dir. It
reads everything request-time from the data root and NEVER fabricates a metric:
a section whose backing artifact is absent surfaces an explicit
``{"available": false, "status": "not yet evaluated"}`` marker rather than a
zeroed-out number, so an ungoverned / un-evaluated course reads honestly.

Sections (each present-if-available):

* ``course_status`` — the 5-value promotion-chain certification enum (reuses
  :func:`gui.services.imscc_service._read_course_status`, both landing paths).
* ``retrieval_eval`` — the latest three-arm eval scorecard
  (``retrieval_eval/eval_scorecard_<ts>.json``): per-arm key-point coverage,
  unsupported-claim rate, latency p50/p95, plus the refusal-safety headline.
* ``refusal_calibration`` — ``retrieval_eval/refusal_calibration.json``: the
  recommended refusal threshold + its precision / recall.
* ``assessment_quality`` — ``quality/trainforge_assessment_quality_report.json``:
  status, promotion decision, and the summary rates.
* ``coverage_map`` — ``coverage_map.json``: the objective-coverage summary.

Read-only, no network, no LLM, no decision capture — deterministic. Slug
validation + the 404/422 contract are reused from
:mod:`gui.services.imscc_service` (``IMSCCServiceError`` is re-exported here as
``ScorecardServiceError`` so the router maps it identically to the viewer paths).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from gui.services import imscc_service

# Re-export the shared typed error so the router maps 404 / 422 / 500 exactly as
# it does for the viewer paths (see gui.routers.library).
ScorecardServiceError = imscc_service.IMSCCServiceError

__all__ = ["build_scorecard", "ScorecardServiceError"]

# Explicit not-yet-evaluated marker for an absent section. Anti-fabrication: a
# missing artifact is stated as such, never zero-filled.
_NOT_EVALUATED = "not yet evaluated"

# Filename contract for the three-arm eval scorecard (mirrors
# lib.retrieval.eval_arms.SCORECARD_FILENAME_PREFIX; not imported to keep this
# module free of the heavy retrieval deps).
_SCORECARD_GLOB = "eval_scorecard_*.json"

_REFUSAL_CALIBRATION_NAME = "refusal_calibration.json"
_ASSESSMENT_QUALITY_REL = ("quality", "trainforge_assessment_quality_report.json")
_COVERAGE_MAP_NAME = "coverage_map.json"

# The three eval arms, in display order (matches eval_arms.ALL_ARMS).
_ARMS = ("base", "retrieval", "grounded")

# Per-arm axes surfaced on the scorecard. Copied verbatim from the arm's
# ``comparison`` block when present (``.get`` → omitted when absent, never
# fabricated).
_ARM_AXES = (
    "key_point_coverage_rate",
    "claim_level_unsupported_rate",
    "answered",
    "declined",
    "latency_ms",
)


def _section_absent() -> Dict[str, Any]:
    """The explicit not-yet-evaluated marker for an absent section."""
    return {"available": False, "status": _NOT_EVALUATED}


def _read_json(path: Path) -> Optional[Any]:
    """Best-effort JSON read; ``None`` on any failure (absent / malformed)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _latest_scorecard_path(eval_dir: Path) -> Optional[Path]:
    """Newest ``eval_scorecard_<ts>.json`` under ``eval_dir`` (ISO-ts sorts)."""
    if not eval_dir.is_dir():
        return None
    cands = sorted(eval_dir.glob(_SCORECARD_GLOB))
    return cands[-1] if cands else None


def _compose_retrieval_eval(course_dir: Path) -> Dict[str, Any]:
    """Compose the three-arm retrieval-eval section from the latest scorecard."""
    path = _latest_scorecard_path(course_dir / "retrieval_eval")
    if path is None:
        return _section_absent()
    data = _read_json(path)
    if not isinstance(data, dict):
        return _section_absent()
    comparison = data.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    arms: Dict[str, Any] = {}
    for arm in _ARMS:
        block = comparison.get(arm)
        if not isinstance(block, dict):
            continue
        arms[arm] = {k: block[k] for k in _ARM_AXES if k in block}
    section: Dict[str, Any] = {
        "available": True,
        "source_file": path.name,
    }
    for key in ("generated_at", "engine", "schema_version"):
        if key in data:
            section[key] = data[key]
    if arms:
        section["arms"] = arms
    # Refusal-safety headline (share of refusal probes each arm answered instead
    # of refusing) — echoed verbatim when present.
    refusal_safety = comparison.get("refusal_safety")
    if isinstance(refusal_safety, dict):
        section["refusal_safety"] = refusal_safety
    return section


def _compose_refusal_calibration(course_dir: Path) -> Dict[str, Any]:
    """Compose the refusal-calibration section (recommended threshold + P/R)."""
    data = _read_json(course_dir / "retrieval_eval" / _REFUSAL_CALIBRATION_NAME)
    if not isinstance(data, dict):
        return _section_absent()
    section: Dict[str, Any] = {"available": True}
    for key in ("generated_at", "engine", "fallback_policy_version", "schema_version"):
        if key in data:
            section[key] = data[key]
    recommended = data.get("recommended")
    if isinstance(recommended, dict):
        section["recommended"] = recommended
    return section


def _compose_assessment_quality(course_dir: Path) -> Dict[str, Any]:
    """Compose the assessment-quality section (status + summary rates)."""
    data = _read_json(course_dir / Path(*_ASSESSMENT_QUALITY_REL))
    if not isinstance(data, dict):
        return _section_absent()
    section: Dict[str, Any] = {"available": True}
    for key in ("generated_at", "status", "schema_version"):
        if key in data:
            section[key] = data[key]
    promotion = data.get("promotion_decision")
    if isinstance(promotion, dict):
        # Surface the resolved value + rationale, not the whole nested gate set.
        pd: Dict[str, Any] = {}
        for key in ("value", "rationale"):
            if key in promotion:
                pd[key] = promotion[key]
        if pd:
            section["promotion_decision"] = pd
    elif isinstance(promotion, str):
        section["promotion_decision"] = {"value": promotion}
    summary = data.get("summary")
    if isinstance(summary, dict):
        section["summary"] = summary
    return section


def _compose_coverage_map(course_dir: Path) -> Dict[str, Any]:
    """Compose the coverage-map section (objective-coverage summary)."""
    data = _read_json(course_dir / _COVERAGE_MAP_NAME)
    if not isinstance(data, dict):
        return _section_absent()
    section: Dict[str, Any] = {"available": True}
    for key in ("generated_at", "schema_version"):
        if key in data:
            section[key] = data[key]
    summary = data.get("summary")
    if isinstance(summary, dict):
        # ``orphan_chunks`` can be a long id list — surface its count, not the
        # raw wall of ids, and echo the rest of the summary verbatim.
        compact = dict(summary)
        orphans = compact.get("orphan_chunks")
        if isinstance(orphans, list):
            compact["orphan_chunks_count"] = len(orphans)
            compact.pop("orphan_chunks", None)
        section["summary"] = compact
    return section


def build_scorecard(course_id: str, libv2_root: Optional[Path] = None) -> Dict[str, Any]:
    """Compose the per-course quality scorecard for ``course_id``.

    Validates the slug + course existence (reusing the viewer's 422/404
    contract), then composes each section present-if-available. Every read is
    request-time from the resolved data root; a course with no eval artifacts
    returns a well-formed scorecard whose sections are all
    ``{"available": false, "status": "not yet evaluated"}``.

    Raises ``ScorecardServiceError`` (422 malformed slug / 404 unknown course).
    """
    root = Path(libv2_root) if libv2_root is not None else imscc_service._libv2_root()
    course_dir = imscc_service._validate_slug(course_id, root)

    sections: Dict[str, Any] = {
        "retrieval_eval": _compose_retrieval_eval(course_dir),
        "refusal_calibration": _compose_refusal_calibration(course_dir),
        "assessment_quality": _compose_assessment_quality(course_dir),
        "coverage_map": _compose_coverage_map(course_dir),
    }

    scorecard: Dict[str, Any] = {
        "slug": course_id,
        "sections": sections,
    }
    status = imscc_service._read_course_status(course_dir)
    if status is not None:
        scorecard["course_status"] = status
    return scorecard
