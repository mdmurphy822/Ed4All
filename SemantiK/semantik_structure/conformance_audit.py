"""Per-document conformance audit artifact (Plan 12 A1).

Every document the cascade processes — every exit mode, including
``non_certified_stamp`` — gets one machine-readable audit dict recording
*which checks ran, which passed, which were SKIPPED and why, under which
thresholds, with which models*. This is the verifiability artifact: a
third party reads it instead of the source code to confirm (or reject)
the WCAG 2.2 AA claim for that document.

Design constraints:

* **Skips are first-class.** A ``CheckOutcome`` with ``skipped=True``
  means "no measurement", not "verified safe" (see
  ``gates/types.py``). The audit surfaces every skip per region AND as
  per-check totals; a document whose text-preservation gate skipped 80%
  of regions must look different from one fully measured.
* **No silent fallback.** :func:`write_conformance_audit` raises
  :class:`ConformanceAuditError` on any failure — a document must not
  ship without its audit once the caller asked for one.
* **Compact but lossless on verdicts.** Axe violation *node* lists are
  collapsed to counts (they can be thousands of DOM nodes); rule ids,
  impacts, and every check verdict/message are kept verbatim.

The dict is embedded in the ``run_full_cascade`` result under
``"conformance_audit"`` and persisted as ``*.conformance_audit.json``
next to the HTML product by ``scripts/ops/pdf_to_html.py``.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import subprocess
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = "conformance-audit/1.0"

#: Top-level keys every audit dict must carry. Locked by
#: tests/test_conformance_audit.py::test_schema_completeness.
AUDIT_REQUIRED_KEYS = (
    "schema_version",
    "document",
    "exit",
    "gate_stage7",
    "gate_stage10",
    "theta",
    "thresholds",
    "assembly",
    "provenance",
    "wcag_coverage",
)


class ConformanceAuditError(RuntimeError):
    """Raised when the audit artifact cannot be built or persisted.

    Deliberately loud (feedback_no_silent_fallbacks): shipping a
    document without its audit would defeat the artifact's purpose.
    """


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Recursively convert enums/dataclasses/tuples into JSON-safe values."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {_jsonable(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _check_to_dict(check: Any) -> dict[str, Any]:
    """Compact one ``CheckOutcome``: verdict + skip + axe rule ids.

    Axe violation node lists are reduced to ``n_nodes`` per rule; the
    rule id and impact are preserved verbatim.
    """
    out: dict[str, Any] = {
        "check": _jsonable(check.check),
        "passed": bool(check.passed),
        "skipped": bool(getattr(check, "skipped", False)),
        "message": check.message or "",
    }
    if out["skipped"]:
        out["skip_reason"] = check.message or "unspecified"
    details = getattr(check, "details", None) or {}
    vlist = details.get("violations")
    if vlist:
        out["violations"] = [
            {
                "id": v.get("id"),
                "impact": v.get("impact"),
                "n_nodes": _node_count(v.get("nodes")),
            }
            for v in vlist
            if isinstance(v, dict)
        ]
    return out


def _node_count(nodes: Any) -> int:
    """Axe reports nodes as a list; some summaries pre-collapse to an int."""
    if isinstance(nodes, list):
        return len(nodes)
    if isinstance(nodes, int):
        return nodes
    return 0


def _gate_result_to_dict(gr: Any) -> dict[str, Any]:
    return {
        "stage": gr.stage,
        "passed": bool(gr.passed),
        "checks": [_check_to_dict(c) for c in gr.checks],
    }


def build_region_gate_log(
    results: dict[int, list[Any]],
    regions: list[Any],
) -> list[dict[str, Any]]:
    """Per-region, per-candidate Stage-7 gate log.

    ``results`` is the dict[region_index, list[GateResult]] returned by
    ``gate_per_region`` (one GateResult per candidate); ``regions`` is
    the capped region list those indices point into.
    """
    log: list[dict[str, Any]] = []
    for idx in sorted(results):
        kind = regions[idx].kind if 0 <= idx < len(regions) else "unknown"
        cands = [_gate_result_to_dict(gr) for gr in results[idx]]
        log.append(
            {
                "region_index": idx,
                "kind": kind,
                "n_candidates": len(cands),
                "n_survivors": sum(1 for c in cands if c["passed"]),
                "candidates": cands,
            }
        )
    return log


def _git_sha() -> str | None:
    """Best-effort repo SHA for provenance. None when not a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None


def _data_semantik_marker_census(html: str) -> dict[str, int]:
    """Count ``data-semantik-*`` provenance attributes in the product HTML."""
    import re

    counts: dict[str, int] = {}
    for m in re.finditer(r"data-semantik-[a-z0-9-]+", html):
        counts[m.group(0)] = counts.get(m.group(0), 0) + 1
    return counts


def _gap_slot_to_dict(slot: Any) -> dict[str, Any]:
    return {
        "kind": _jsonable(slot.kind),
        "region_index": slot.region_index,
        "detected_by": slot.detected_by,
        "used_deterministic_fallback": slot.fallback_html is not None,
    }


# ---------------------------------------------------------------------------
# Builder + writer
# ---------------------------------------------------------------------------


def build_conformance_audit(
    *,
    pdf_path: str,
    runtime_mode: str,
    k: int,
    lane_used: str,
    offline_retry_fired: bool,
    exit_action: Any,
    region_gate_results: dict[int, list[Any]],
    regions: list[Any],
    stage7_skip_counts: dict[str, int],
    stage7_violation_counts: dict[str, int],
    n_candidates_in: int,
    n_candidates_out: int,
    n_regions_no_survivor: int,
    doc_gate_result: Any,
    theta_report: Any,
    assembled: Any,
    reading_order_at_risk: list[int] | None = None,
    wcag_coverage: list[dict[str, Any]] | None = None,
    structure_review: list[dict[str, Any]] | None = None,
    ocr_repair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the audit dict from cascade state.

    Pure aggregation — no model runs, no I/O beyond a best-effort
    ``git rev-parse``. Raises :class:`ConformanceAuditError` if a
    required input is structurally unusable (the cascade handing us a
    gate result with no checks is a bug upstream, not something to
    paper over).
    """
    from .extract_shared import EXTRACT_CACHE_VERSION
    from .theta.types import floors_from_config, load_theta_config

    theta_cfg = load_theta_config()

    if doc_gate_result is None or not getattr(doc_gate_result, "checks", None):
        raise ConformanceAuditError(
            "document gate result missing or has zero checks — refusing to "
            "build an audit that cannot attest the Stage-10 verdict"
        )

    theta_payload = (
        dataclasses.asdict(theta_report)
        if dataclasses.is_dataclass(theta_report)
        else dict(theta_report)
    )
    # Surface the scoring method per dimension (model vs heuristic vs
    # stub) — it lives in each DimensionScore.breakdown["method"] when
    # the dimension sets one.
    dim_methods = {}
    for name, dim in (theta_payload.get("dimensions") or {}).items():
        breakdown = (dim or {}).get("breakdown") or {}
        dim_methods[_jsonable(name)] = breakdown.get("method", "deterministic")

    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "document": {
            "pdf": str(pdf_path),
            "wcag_status": theta_payload.get("wcag_status"),
        },
        "exit": {
            "action": _jsonable(exit_action),
            "lane_used": lane_used,
            "offline_retry_fired": bool(offline_retry_fired),
            "flags": _jsonable(theta_payload.get("flags") or []),
            "n_prior_attempts": len(theta_payload.get("retry_history") or []),
        },
        "gate_stage7": {
            "n_regions": len(region_gate_results),
            "n_candidates_in": n_candidates_in,
            "n_candidates_out": n_candidates_out,
            "n_regions_no_survivor": n_regions_no_survivor,
            "skipped_check_counts": dict(stage7_skip_counts),
            "violation_counts": dict(stage7_violation_counts),
            "regions": build_region_gate_log(region_gate_results, regions),
        },
        "gate_stage10": _gate_result_to_dict(doc_gate_result),
        "theta": {
            "report": _jsonable(theta_payload),
            "dimension_methods": dim_methods,
            # Calibrated per-dimension weights (Plan 12 A3) — surfaced
            # verbatim so the audit is self-describing about how the
            # composite was formed under this config version.
            "composite_weighting": f"config:{theta_cfg.version}",
            "composite_weights": dict(theta_cfg.weights),
        },
        "thresholds": {
            "tau_theta_confidence": theta_cfg.tau_theta_ship,
            "tau_theta_retry": theta_cfg.tau_theta_retry,
            "delta_theta_improve": theta_cfg.delta_theta_improve,
            "dimension_floors": floors_from_config(theta_cfg),
            "theta_version": theta_payload.get("theta_version"),
            # Calibration provenance (Plan 12 A3): which dataset/date
            # produced the weights+thresholds above. None only when the
            # config predates calibration — kept honest, never invented.
            "calibration": (
                {
                    "report": theta_cfg.calibration.get("report"),
                    "date": theta_cfg.calibration.get("date"),
                    "n_docs": theta_cfg.calibration.get("n_docs"),
                }
                if theta_cfg.calibration
                else None
            ),
        },
        "assembly": {
            "gaps_found": [_gap_slot_to_dict(g) for g in assembled.gaps_found],
            "gaps_resolved": [_gap_slot_to_dict(g) for g in assembled.gaps_resolved],
            "gaps_fallback": [_gap_slot_to_dict(g) for g in assembled.gaps_fallback],
            "n_headings": len(assembled.heading_tree),
            "landmarks": dict(assembled.landmarks),
            "data_semantik_markers": _data_semantik_marker_census(assembled.html),
            # Regions that fell back to deterministic emission while
            # being/adjoining a table/figure/math region — SC 1.3.2
            # risk surfaced per Plan 12 B3.
            "reading_order_at_risk": [
                {
                    "region_index": i,
                    "kind": regions[i].kind if 0 <= i < len(regions) else "unknown",
                }
                for i in (reading_order_at_risk or [])
            ],
        },
        "provenance": {
            "git_sha": _git_sha(),
            "runtime_mode": runtime_mode,
            "k": k,
            "extract_cache_version": EXTRACT_CACHE_VERSION,
        },
        # Per-SC verdicts (Plan 12 A2). None means the coverage map was
        # not supplied by the caller — distinct from an empty list.
        "wcag_coverage": wcag_coverage,
        # Stage-5d 70B structure-review verdicts (per-doc audit section).
        # OPTIONAL — deliberately NOT in AUDIT_REQUIRED_KEYS. None means the
        # reviewer did NOT run (flag off); an empty list means it ran and
        # found nothing to correct (mirrors the wcag_coverage None-vs-[]
        # distinction).
        "structure_review": structure_review,
    }
    # OCR-confusable repair audit section (channels 2+3). OPTIONAL — added ONLY
    # when the pass ran (None → key absent, byte-stable to a no-repair run,
    # deliberately NOT in AUDIT_REQUIRED_KEYS). Carries flagged/accepted/rejected
    # counts + the residual defect density + the repair-stats proxy score.
    if ocr_repair is not None:
        audit["ocr_repair"] = _jsonable(ocr_repair)
    return audit


def write_conformance_audit(audit: dict[str, Any], out_path: Path | str) -> Path:
    """Persist the audit JSON. Raises on ANY failure — never silent.

    The document product must not ship without its audit; callers that
    can tolerate a missing audit must decide that explicitly upstream.
    """
    out_path = Path(out_path)
    missing = [k for k in AUDIT_REQUIRED_KEYS if k not in audit]
    if missing:
        raise ConformanceAuditError(f"audit dict is missing required keys: {missing}")
    try:
        payload = json.dumps(_jsonable(audit), indent=1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    except ConformanceAuditError:
        raise
    except Exception as exc:
        raise ConformanceAuditError(
            f"failed to write conformance audit to {out_path}: {exc}"
        ) from exc
    return out_path


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AUDIT_REQUIRED_KEYS",
    "ConformanceAuditError",
    "build_conformance_audit",
    "build_region_gate_log",
    "write_conformance_audit",
]
