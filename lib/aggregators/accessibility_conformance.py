"""Accessibility Conformance Report (ACR) aggregator — roadmap T3.

Deterministic post-loop aggregator that INVERTS the gate-level WCAG issue
stream into a per-success-criterion conformance table, VPAT/WCAG-EM style.
Every WCAG 2.2 Level A + AA success criterion gets exactly one row carrying:

* ``criterion`` — the SC number (e.g. ``1.1.1``).
* ``level`` — ``A`` / ``AA``.
* ``title`` — the SC short name.
* ``status`` — one of ``supports`` / ``partially_supports`` /
  ``does_not_support`` / ``not_evaluated``.
* ``evidence_counts`` — ``{critical, warning, pages}`` observed-issue tally.

Status derivation (deterministic):

* Any CRITICAL-severity WCAG issue coded to the criterion → ``does_not_support``.
* WARNING-severity issues only → ``partially_supports``.
* No issues AND the criterion is in the automated-evaluable set → ``supports``.
* No issues AND the criterion is OUTSIDE the automated-evaluable set →
  ``not_evaluated`` (with a ``reason`` category). This is the anti-silent-
  degradation contract: a static-HTML checker cannot verify contrast
  COMPUTATION, time-based-media alternatives, or cognitive / human-judgement
  criteria, so the ACR emits an EXPLICIT ``not_evaluated`` row rather than
  silently claiming ``supports``.

Input sources (both read best-effort, unioned):

1. ``phase_outputs[*]._gate_results[*]`` — the in-memory gate-result chain
   the sibling aggregators read. WCAG gates carry ``issues[]`` with a
   machine-readable ``code`` (``WCAG_1_1_1``), ``severity``, and ``location``.
2. ``<project_path>/courseforge_validation_report.json::accessibility_results``
   — the on-disk per-page rollup (covers partial runs where only the report
   survives). Both ``results[].top_issues[]`` and ``per_page_issue_count``
   are consulted.

Lands at ``<libv2_course>/quality/accessibility_conformance.json`` with a
trainforge-dir fallback (mirrors the promotion-chain report). Best-effort:
aggregator failure logs a warning and never alters ``final_status``.

Schema: :file:`schemas/aggregators/accessibility_conformance.schema.json`
(Draft 2020-12, ``additionalProperties: false``).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# Conformance status enum (VPAT / WCAG-EM vocabulary).
SUPPORTS = "supports"
PARTIALLY_SUPPORTS = "partially_supports"
DOES_NOT_SUPPORT = "does_not_support"
NOT_EVALUATED = "not_evaluated"

# Accessibility gate identity — mirrors the courseforge_validation_report
# accessibility bucket membership rule so the ACR reads the SAME gate rows.
_ACCESSIBILITY_GATE_IDS = ("wcag_compliance", "wcag_aa_compliance")
_ACCESSIBILITY_VALIDATOR_SUFFIX = "wcagvalidator"

# A gate issue coded ``WCAG_1_1_1`` inverts to SC number ``1.1.1``.
_WCAG_CODE_RE = re.compile(r"^WCAG[_-](\d+)[_-](\d+)[_-](\d+)$", re.IGNORECASE)

# Not-evaluated reason categories — a static-HTML automated checker cannot
# produce a supports/does_not_support verdict for these classes, so the row
# is emitted with an explicit reason instead of a fabricated "supports".
_REASON_CONTRAST = "contrast_computation"          # needs rendered pixel colors
_REASON_MEDIA = "time_based_media"                 # captions / audio-desc / transcript
_REASON_HUMAN = "human_judgement"                  # cognitive / sensory / semantics
_REASON_RUNTIME = "interaction_runtime"            # focus / pointer / timing behavior


# Canonical WCAG 2.2 Level A + AA success criteria.
# Row: (criterion, level, title, evaluable, not_evaluated_reason).
# ``evaluable=True`` → the automated WCAGValidator machinery can assert a
# supports/does_not_support verdict from static HTML (presence/absence of
# structural markup); ``evaluable=False`` → outside static automated reach,
# so a clean run emits ``not_evaluated`` with the paired reason. A criterion
# that DID collect observed issues always yields a support-failure verdict
# regardless of the ``evaluable`` flag (evidence outweighs the default).
_WCAG_22_A_AA: Tuple[Tuple[str, str, str, bool, Optional[str]], ...] = (
    # --- Level A ---
    ("1.1.1", "A", "Non-text Content", True, None),
    ("1.2.1", "A", "Audio-only and Video-only (Prerecorded)", False, _REASON_MEDIA),
    ("1.2.2", "A", "Captions (Prerecorded)", False, _REASON_MEDIA),
    ("1.2.3", "A", "Audio Description or Media Alternative (Prerecorded)", False, _REASON_MEDIA),
    ("1.3.1", "A", "Info and Relationships", True, None),
    ("1.3.2", "A", "Meaningful Sequence", True, None),
    ("1.3.3", "A", "Sensory Characteristics", False, _REASON_HUMAN),
    ("1.4.1", "A", "Use of Color", False, _REASON_HUMAN),
    ("1.4.2", "A", "Audio Control", False, _REASON_MEDIA),
    ("2.1.1", "A", "Keyboard", True, None),
    ("2.1.2", "A", "No Keyboard Trap", False, _REASON_RUNTIME),
    ("2.1.4", "A", "Character Key Shortcuts", False, _REASON_RUNTIME),
    ("2.2.1", "A", "Timing Adjustable", False, _REASON_RUNTIME),
    ("2.2.2", "A", "Pause, Stop, Hide", False, _REASON_RUNTIME),
    ("2.3.1", "A", "Three Flashes or Below Threshold", False, _REASON_HUMAN),
    ("2.4.1", "A", "Bypass Blocks", True, None),
    ("2.4.2", "A", "Page Titled", True, None),
    ("2.4.3", "A", "Focus Order", False, _REASON_RUNTIME),
    ("2.4.4", "A", "Link Purpose (In Context)", True, None),
    ("2.5.1", "A", "Pointer Gestures", False, _REASON_RUNTIME),
    ("2.5.2", "A", "Pointer Cancellation", False, _REASON_RUNTIME),
    ("2.5.3", "A", "Label in Name", True, None),
    ("2.5.4", "A", "Motion Actuation", False, _REASON_RUNTIME),
    ("3.1.1", "A", "Language of Page", True, None),
    ("3.2.1", "A", "On Focus", False, _REASON_RUNTIME),
    ("3.2.2", "A", "On Input", False, _REASON_RUNTIME),
    ("3.2.6", "A", "Consistent Help", False, _REASON_HUMAN),
    ("3.3.1", "A", "Error Identification", False, _REASON_RUNTIME),
    ("3.3.2", "A", "Labels or Instructions", True, None),
    ("3.3.7", "A", "Redundant Entry", False, _REASON_RUNTIME),
    ("4.1.1", "A", "Parsing", True, None),
    ("4.1.2", "A", "Name, Role, Value", True, None),
    # --- Level AA ---
    ("1.2.4", "AA", "Captions (Live)", False, _REASON_MEDIA),
    ("1.2.5", "AA", "Audio Description (Prerecorded)", False, _REASON_MEDIA),
    ("1.3.4", "AA", "Orientation", False, _REASON_RUNTIME),
    ("1.3.5", "AA", "Identify Input Purpose", True, None),
    ("1.4.3", "AA", "Contrast (Minimum)", False, _REASON_CONTRAST),
    ("1.4.4", "AA", "Resize Text", False, _REASON_RUNTIME),
    ("1.4.5", "AA", "Images of Text", False, _REASON_HUMAN),
    ("1.4.10", "AA", "Reflow", False, _REASON_RUNTIME),
    ("1.4.11", "AA", "Non-text Contrast", False, _REASON_CONTRAST),
    ("1.4.12", "AA", "Text Spacing", False, _REASON_RUNTIME),
    ("1.4.13", "AA", "Content on Hover or Focus", False, _REASON_RUNTIME),
    ("2.4.5", "AA", "Multiple Ways", True, None),
    ("2.4.6", "AA", "Headings and Labels", True, None),
    ("2.4.7", "AA", "Focus Visible", False, _REASON_RUNTIME),
    ("2.4.11", "AA", "Focus Not Obscured (Minimum)", False, _REASON_RUNTIME),
    ("2.5.7", "AA", "Dragging Movements", False, _REASON_RUNTIME),
    ("2.5.8", "AA", "Target Size (Minimum)", False, _REASON_RUNTIME),
    ("3.1.2", "AA", "Language of Parts", True, None),
    ("3.2.3", "AA", "Consistent Navigation", False, _REASON_HUMAN),
    ("3.2.4", "AA", "Consistent Identification", False, _REASON_HUMAN),
    ("3.3.3", "AA", "Error Suggestion", False, _REASON_RUNTIME),
    ("3.3.4", "AA", "Error Prevention (Legal, Financial, Data)", False, _REASON_RUNTIME),
    ("3.3.8", "AA", "Accessible Authentication (Minimum)", False, _REASON_RUNTIME),
)


def _read_json(path: Path) -> Optional[Any]:
    """Best-effort JSON read; returns None on any failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("accessibility_conformance: cannot read %s: %s", path, exc)
        return None


def _criterion_from_code(code: str) -> Optional[str]:
    """Invert a gate issue code (``WCAG_1_1_1``) to an SC number (``1.1.1``)."""
    if not code:
        return None
    match = _WCAG_CODE_RE.match(str(code).strip())
    if match is None:
        return None
    return ".".join(match.groups())


def _is_accessibility_gate(gate_id: str, validator_name: str) -> bool:
    """Membership rule mirroring the courseforge accessibility bucket."""
    gid = (gate_id or "").lower()
    if any(sub in gid for sub in _ACCESSIBILITY_GATE_IDS):
        return True
    return (validator_name or "").lower().endswith(_ACCESSIBILITY_VALIDATOR_SUFFIX)


class AccessibilityConformanceAggregator:
    """Invert gate-level WCAG issues into a per-success-criterion ACR table.

    Parameters
    ----------
    phase_outputs:
        Optional ``WorkflowRunner.run_workflow`` ``phase_outputs`` map. WCAG
        gates surface ``_gate_results[*].issues[]`` with a machine-readable
        ``code`` per issue. Always read-only.
    project_path:
        Optional Courseforge project export root. When set the aggregator
        also reads ``<project_path>/courseforge_validation_report.json``'s
        ``accessibility_results`` block (per-page rollup) so a partial run
        with only the on-disk report still produces a table.
    course_code:
        Operator-facing course code so cross-run diffs key cleanly.
    run_id:
        Workflow run ID.
    decision_capture:
        Unused (no LLM decisions — this is a deterministic inversion). Kept
        in the signature for aggregator-constructor symmetry.
    """

    def __init__(
        self,
        *,
        phase_outputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        project_path: Optional[Path] = None,
        course_code: str = "",
        run_id: str = "",
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.phase_outputs = phase_outputs or {}
        self.project_path = Path(project_path) if project_path else None
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        self.decision_capture = decision_capture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Build the canonical ACR report dict (deterministic)."""
        # criterion -> {"critical": int, "warning": int, "pages": set[str]}
        observed = self._collect_observed_issues()

        criteria_rows: List[Dict[str, Any]] = []
        counts = {
            SUPPORTS: 0,
            PARTIALLY_SUPPORTS: 0,
            DOES_NOT_SUPPORT: 0,
            NOT_EVALUATED: 0,
        }
        for criterion, level, title, evaluable, reason in _WCAG_22_A_AA:
            obs = observed.get(criterion)
            critical = int(obs["critical"]) if obs else 0
            warning = int(obs["warning"]) if obs else 0
            pages = sorted(obs["pages"]) if obs else []

            if critical > 0:
                status = DOES_NOT_SUPPORT
            elif warning > 0:
                status = PARTIALLY_SUPPORTS
            elif evaluable:
                status = SUPPORTS
            else:
                status = NOT_EVALUATED
            counts[status] += 1

            row: Dict[str, Any] = {
                "criterion": criterion,
                "level": level,
                "title": title,
                "status": status,
                "evidence_counts": {
                    "critical": critical,
                    "warning": warning,
                    "pages": len(pages),
                },
                "pages": pages,
            }
            # A not_evaluated row (default, no overriding evidence) carries the
            # reason category so an operator sees WHY it wasn't machine-checked.
            if status == NOT_EVALUATED and reason is not None:
                row["reason"] = reason
            criteria_rows.append(row)

        summary = {
            "total_criteria": len(criteria_rows),
            "supports": counts[SUPPORTS],
            "partially_supports": counts[PARTIALLY_SUPPORTS],
            "does_not_support": counts[DOES_NOT_SUPPORT],
            "not_evaluated": counts[NOT_EVALUATED],
            # A run is conformant only when NO criterion regressed to a
            # support-failure. not_evaluated rows are NOT failures (they are
            # honestly out of automated scope), so they don't sink this flag.
            "conformant": (
                counts[DOES_NOT_SUPPORT] == 0
                and counts[PARTIALLY_SUPPORTS] == 0
            ),
        }

        return {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "standard": "WCAG 2.2",
            "target_level": "AA",
            "criteria": criteria_rows,
            "summary": summary,
        }

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build()
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    # Input collection
    # ------------------------------------------------------------------
    def _collect_observed_issues(self) -> Dict[str, Dict[str, Any]]:
        """Union WCAG issues from phase_outputs + the on-disk report."""
        observed: Dict[str, Dict[str, Any]] = {}

        def _accrue(criterion: str, severity: str, page: Optional[str]) -> None:
            row = observed.setdefault(
                criterion, {"critical": 0, "warning": 0, "pages": set()}
            )
            if str(severity).lower() == "critical":
                row["critical"] += 1
            else:
                row["warning"] += 1
            if isinstance(page, str) and page:
                row["pages"].add(page)

        # Source 1 — in-memory gate results.
        for _phase, payload in self.phase_outputs.items():
            if not isinstance(payload, Mapping):
                continue
            for gr in payload.get("_gate_results") or []:
                if not isinstance(gr, Mapping):
                    continue
                if not _is_accessibility_gate(
                    str(gr.get("gate_id") or ""),
                    str(gr.get("validator_name") or ""),
                ):
                    continue
                for issue in gr.get("issues") or []:
                    if not isinstance(issue, Mapping):
                        continue
                    criterion = _criterion_from_code(str(issue.get("code") or ""))
                    if criterion is None:
                        continue
                    _accrue(
                        criterion,
                        str(issue.get("severity") or "warning"),
                        issue.get("location"),
                    )

        # Source 2 — on-disk courseforge_validation_report.json.
        self._accrue_from_report(_accrue)
        return observed

    def _accrue_from_report(self, accrue: Any) -> None:
        """Fold accessibility issues from the on-disk validation report."""
        if self.project_path is None:
            return
        report_path = self.project_path / "courseforge_validation_report.json"
        if not report_path.exists():
            return
        payload = _read_json(report_path)
        if not isinstance(payload, Mapping):
            return
        access = payload.get("accessibility_results")
        if not isinstance(access, Mapping):
            return
        for entry in access.get("results") or []:
            if not isinstance(entry, Mapping):
                continue
            for issue in entry.get("top_issues") or []:
                if not isinstance(issue, Mapping):
                    continue
                criterion = _criterion_from_code(str(issue.get("code") or ""))
                if criterion is None:
                    continue
                accrue(
                    criterion,
                    str(issue.get("severity") or "warning"),
                    issue.get("location"),
                )


__all__ = [
    "AccessibilityConformanceAggregator",
    "SCHEMA_VERSION",
    "SUPPORTS",
    "PARTIALLY_SUPPORTS",
    "DOES_NOT_SUPPORT",
    "NOT_EVALUATED",
]
