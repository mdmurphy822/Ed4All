"""WS1.2 CitationAnchorValidator — corpus citation-anchoring gate.

Gates a chunkset on its citation-anchoring rate: the fraction of chunks whose
provenance (``source.item_path`` + text) resolves back to a real archived
source page containing the chunk's text. Built on the deterministic resolver
``lib.retrieval.citation_anchor.anchor_report``.

**Not wired into ``config/workflows.yaml`` yet** (the workflow YAML is
churn-fenced in the current tree). The validator ships now with the same
``GateResult`` / ``GateIssue`` shape as ``lib/validators/chunkset_manifest.py``
so a later micro-wave can add a one-line gate entry without code changes.

Inputs contract:

  * ``inputs["chunks_path"]``     — path to the chunkset ``chunks.jsonl``
                                    (required).
  * ``inputs["course_dir"]``      — LibV2 course dir the source pages are
                                    archived under (required).
  * ``inputs["chunkset_kind"]``   — ``"dart" | "imscc" | "corpus"`` (required).
  * ``inputs["min_anchoring_rate"]`` — floor; default 0.95.
  * ``inputs["gate_id"]``         — optional GateResult ID override.

Issue codes:

  * ``CITATION_ANCHOR_MISSING_INPUT``    — a required input is absent.
  * ``CITATION_ANCHOR_CHUNKS_NOT_FOUND`` — chunks_path doesn't exist.
  * ``CITATION_ANCHOR_COURSE_DIR_NOT_FOUND`` — course_dir doesn't exist.
  * ``CITATION_ANCHOR_REPORT_ERROR``     — report computation raised.
  * ``CITATION_ANCHOR_RATE_BELOW_FLOOR`` — anchoring_rate < min_anchoring_rate.
  * ``CITATION_ANCHOR_SOURCE_PAGE_MISSING`` — at least one chunk's source page
    is unresolvable (warning sub-signal; the worst-offender list rides on the
    GateResult.metadata).

Severity contract: critical for missing inputs / not-found / below-floor;
``action="block"`` on any critical issue. ``metadata`` carries the full
report dict so downstream aggregators can read per-status counts and the
worst-offender chunk_ids without re-running the report.

Deterministic; no LLM, no network, no decision capture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.retrieval.citation_anchor import AnchorStatus, anchor_report

logger = logging.getLogger(__name__)

_DEFAULT_MIN_ANCHORING_RATE = 0.95
_ALLOWED_KINDS = {"dart", "imscc", "corpus"}


class CitationAnchorValidator:
    """Citation-anchoring rate gate over a chunkset."""

    name = "citation_anchor"
    version = "0.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        issues: List[GateIssue] = []

        chunks_raw = inputs.get("chunks_path")
        course_raw = inputs.get("course_dir")
        kind = inputs.get("chunkset_kind")
        min_rate = inputs.get("min_anchoring_rate", _DEFAULT_MIN_ANCHORING_RATE)

        # ---- required inputs.
        missing = []
        if not chunks_raw:
            missing.append("chunks_path")
        if not course_raw:
            missing.append("course_dir")
        if kind not in _ALLOWED_KINDS:
            missing.append("chunkset_kind")
        if missing:
            return self._fail(
                gate_id,
                "CITATION_ANCHOR_MISSING_INPUT",
                f"CitationAnchorValidator requires inputs {missing} "
                f"(chunkset_kind must be one of {sorted(_ALLOWED_KINDS)}).",
            )

        chunks_path = Path(chunks_raw)
        course_dir = Path(course_raw)
        if not chunks_path.is_file():
            return self._fail(
                gate_id,
                "CITATION_ANCHOR_CHUNKS_NOT_FOUND",
                f"chunks.jsonl not found at {chunks_path}",
                location=str(chunks_path),
            )
        if not course_dir.is_dir():
            return self._fail(
                gate_id,
                "CITATION_ANCHOR_COURSE_DIR_NOT_FOUND",
                f"course dir not found at {course_dir}",
                location=str(course_dir),
            )

        # ---- compute the report.
        try:
            report = anchor_report(chunks_path, course_dir, chunkset_kind=kind)
        except Exception as exc:  # noqa: BLE001
            return self._fail(
                gate_id,
                "CITATION_ANCHOR_REPORT_ERROR",
                f"anchor_report raised {exc.__class__.__name__}: {exc}",
                location=str(chunks_path),
            )

        rate = float(report.get("anchoring_rate", 0.0))
        missing_pages = int(
            report.get("status_counts", {}).get(
                AnchorStatus.SOURCE_PAGE_MISSING.value, 0
            )
        )

        if missing_pages > 0:
            offenders = report.get("worst_offenders", {}).get(
                AnchorStatus.SOURCE_PAGE_MISSING.value, []
            )
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CITATION_ANCHOR_SOURCE_PAGE_MISSING",
                    message=(
                        f"{missing_pages} chunk(s) resolve to no archived "
                        f"source page (e.g. {offenders[:5]})."
                    ),
                    suggestion=(
                        "Archive the missing source pages under the course "
                        "dir, or confirm the chunkset_kind matches the "
                        "archived layout."
                    ),
                )
            )

        if rate < float(min_rate):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CITATION_ANCHOR_RATE_BELOW_FLOOR",
                    message=(
                        f"citation anchoring_rate {rate:.4f} is below the "
                        f"floor {float(min_rate):.4f} for {kind} chunkset."
                    ),
                    location=str(chunks_path),
                    suggestion=(
                        "Inspect worst_offenders in GateResult.metadata; "
                        "fabricated spans / boilerplate-strip drift lower the "
                        "rate. Re-pin the floor only if the drop is intended."
                    ),
                )
            )

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        action: Optional[str] = "block" if not passed else None
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
            metadata={"citation_anchor_report": report},
        )

    # ------------------------------------------------------------- helpers

    def _fail(
        self,
        gate_id: str,
        code: str,
        message: str,
        *,
        location: Optional[str] = None,
    ) -> GateResult:
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=False,
            issues=[
                GateIssue(
                    severity="critical",
                    code=code,
                    message=message,
                    location=location,
                )
            ],
            action="block",
        )


__all__ = ["CitationAnchorValidator"]
