"""
SemantiK Markers Validator

Validates that SemantiK-converted HTML carries the required accessibility
markers. SemantiK-produced HTML must include:
  - Skip link (<a class="skip-link">)
  - Main content landmark (<main role="main">)
  - ARIA-labelled sections (<section aria-labelledby="...">)
  - SemantiK semantic classes (semantik-section / semantik-document)

Wraps the marker-detection logic from
MCP.tools.pipeline_tools.validate_semantik_markers (the MCP tool) into the
ValidationGateManager Validator protocol so it can be wired as a
validation gate in config/workflows.yaml.

Source-provenance markers audited per <section>:
  - data-semantik-source attribute on every <section>
  - data-semantik-block-id attribute on every <section>

An attribute that is present but empty is reported at CRITICAL severity
(the "emitted-but-malformed" failure mode). Fully-absent attributes remain
at warning severity: a document that carries NO <section> elements with the
SemantiK semantic class is treated as graceful fallback and does not fail —
only pages that claim the SemantiK semantic contract and then omit
provenance are blocked.

Referenced by: config/workflows.yaml
  - textbook_to_course.semantik_conversion -> semantik_markers
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    pages_audited: int,
    markers_found: int,
    markers_missing: int,
    marker_density: Optional[float],
    sections_total: int,
    sections_without_source: int,
    sections_without_block_id: int,
    empty_source_count: int,
    empty_block_id_count: int,
) -> None:
    """Emit one ``semantik_markers_check`` decision per validate() call."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    density_str = (
        f"{marker_density:.3f}" if marker_density is not None else "n/a"
    )
    rationale = (
        f"SemantiK markers orchestration check: "
        f"pages_audited={pages_audited}, "
        f"markers_found={markers_found}, "
        f"markers_missing={markers_missing}, "
        f"marker_density={density_str}, "
        f"sections_total={sections_total}, "
        f"sections_without_source={sections_without_source}, "
        f"sections_without_block_id={sections_without_block_id}, "
        f"empty_source_count={empty_source_count}, "
        f"empty_block_id_count={empty_block_id_count}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="semantik_markers_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on semantik_markers_check: %s",
            exc,
        )

# Marker name -> tuple of literal substrings, any of which satisfies the marker.
# Kept in sync with MCP/tools/pipeline_tools.py:validate_semantik_markers.
# The semantic-class marker accepts both the ``semantik-*`` spelling (current
# emit) and the legacy ``dart-*`` spelling (pre-SemantiK corpora) — any match
# satisfies the marker.
_REQUIRED_MARKERS: Dict[str, Tuple[str, ...]] = {
    "skip_link": ('class="skip', "class='skip"),
    "main_role": ('role="main"', "role='main'"),
    "aria_sections": ('aria-labelledby="', "aria-labelledby='"),
    "semantik_structure_classes": (
        "semantik-section", "semantik-document",
        "dart-section", "dart-document",
    ),
}

# Regex for finding top-level <section> open tags. Used for the section-level
# provenance checks. Intentionally permissive (matches any attributes) — the
# presence of the section tag is what we count.
_SECTION_OPEN_RE = re.compile(r"<section\b[^>]*>", re.IGNORECASE)

# Attribute presence checks run against each section's attribute string.
# Dual-READ: accept both ``data-semantik-*`` (current emit) and the legacy
# ``data-dart-*`` spelling (pre-SemantiK corpora).
_DATA_SEMANTIK_SOURCE_RE = re.compile(
    r'\bdata-(?:dart|semantik)-source\s*=', re.IGNORECASE,
)
_DATA_SEMANTIK_BLOCK_ID_RE = re.compile(
    r'\bdata-(?:dart|semantik)-block-id\s*=', re.IGNORECASE,
)

# Critical-severity checks for malformed attributes. An attribute that is
# *present but empty* is a bug in the emit path and must block — this is the
# "emitted-but-malformed" failure mode. Fully-absent attrs remain at warning
# severity per the graceful-fallback rule documented at the top of this module.
_EMPTY_DATA_SEMANTIK_SOURCE_RE = re.compile(
    r'\bdata-(?:dart|semantik)-source\s*=\s*(["\'])\1', re.IGNORECASE,
)
_EMPTY_DATA_SEMANTIK_BLOCK_ID_RE = re.compile(
    r'\bdata-(?:dart|semantik)-block-id\s*=\s*(["\'])\1', re.IGNORECASE,
)


class SemantiKMarkersValidator:
    """Validates SemantiK HTML output for required accessibility markers."""

    name = "semantik_markers"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate SemantiK markers in HTML content.

        Expected inputs (any one of):
            html_path: Path to HTML file to validate
            html_content: Raw HTML string (alternative to html_path)
            gate_id: Optional gate_id override for the result

        Returns:
            GateResult with one critical issue per missing marker.
        """
        gate_id = inputs.get("gate_id", "semantik_markers")
        capture = inputs.get("decision_capture")
        if capture is None:
            capture = inputs.get("capture")
        content = inputs.get("html_content", "") or ""

        if not content and inputs.get("html_path"):
            path = Path(inputs["html_path"])
            if not path.exists():
                _emit_decision(
                    capture,
                    passed=False,
                    code="FILE_NOT_FOUND",
                    pages_audited=0,
                    markers_found=0,
                    markers_missing=len(_REQUIRED_MARKERS),
                    marker_density=None,
                    sections_total=0,
                    sections_without_source=0,
                    sections_without_block_id=0,
                    empty_source_count=0,
                    empty_block_id_count=0,
                )
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="FILE_NOT_FOUND",
                        message=f"SemantiK HTML file not found: {path}",
                        location=str(path),
                    )],
                )
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as e:
                _emit_decision(
                    capture,
                    passed=False,
                    code="FILE_READ_ERROR",
                    pages_audited=0,
                    markers_found=0,
                    markers_missing=len(_REQUIRED_MARKERS),
                    marker_density=None,
                    sections_total=0,
                    sections_without_source=0,
                    sections_without_block_id=0,
                    empty_source_count=0,
                    empty_block_id_count=0,
                )
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="FILE_READ_ERROR",
                        message=f"Failed to read SemantiK HTML file: {e}",
                        location=str(path),
                    )],
                )

        if not content.strip():
            _emit_decision(
                capture,
                passed=False,
                code="EMPTY_CONTENT",
                pages_audited=0,
                markers_found=0,
                markers_missing=len(_REQUIRED_MARKERS),
                marker_density=None,
                sections_total=0,
                sections_without_source=0,
                sections_without_block_id=0,
                empty_source_count=0,
                empty_block_id_count=0,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="EMPTY_CONTENT",
                    message="SemantiK HTML content is empty (no html_path or html_content supplied).",
                )],
            )

        issues: List[GateIssue] = []
        for marker_name, needles in _REQUIRED_MARKERS.items():
            if not any(needle in content for needle in needles):
                issues.append(GateIssue(
                    severity="critical",
                    code=f"MISSING_{marker_name.upper()}",
                    message=f"Required SemantiK marker missing: {marker_name}",
                    suggestion=f"Ensure SemantiK output emits one of: {needles}",
                ))

        # Source-provenance marker checks.
        #
        # Rules:
        #   - Absent attributes on every <section>           -> warning
        #     (graceful fallback for legacy HTML).
        #   - Some <section>s carry attrs, others don't      -> warning
        #     (emit has settled but coverage is incomplete).
        #   - An attr is PRESENT but the value is EMPTY      -> critical
        #     ("emitted-but-malformed" blocker).
        section_tags = _SECTION_OPEN_RE.findall(content)
        total_sections = len(section_tags)
        sections_without_source = 0
        sections_without_block_id = 0
        empty_source_count = 0
        empty_block_id_count = 0
        for tag in section_tags:
            if not _DATA_SEMANTIK_SOURCE_RE.search(tag):
                sections_without_source += 1
            elif _EMPTY_DATA_SEMANTIK_SOURCE_RE.search(tag):
                empty_source_count += 1
            if not _DATA_SEMANTIK_BLOCK_ID_RE.search(tag):
                sections_without_block_id += 1
            elif _EMPTY_DATA_SEMANTIK_BLOCK_ID_RE.search(tag):
                empty_block_id_count += 1

        if total_sections > 0 and sections_without_source > 0:
            issues.append(GateIssue(
                severity="warning",
                code="MISSING_DATA_SEMANTIK_SOURCE",
                message=(
                    f"{sections_without_source}/{total_sections} <section> elements "
                    "missing data-semantik-source attribute"
                ),
                suggestion=(
                    "Ensure SemantiK emits data-semantik-source on every <section>. "
                    "Multi-source path: data-semantik-source=\"pdfplumber\" etc. "
                    "Synthesized path: data-semantik-source=\"synthesized\"."
                ),
            ))

        if total_sections > 0 and sections_without_block_id > 0:
            issues.append(GateIssue(
                severity="warning",
                code="MISSING_DATA_SEMANTIK_BLOCK_ID",
                message=(
                    f"{sections_without_block_id}/{total_sections} <section> elements "
                    "missing data-semantik-block-id attribute"
                ),
                suggestion=(
                    "Ensure SemantiK emits data-semantik-block-id on every <section>. "
                    "Multi-source path uses \"s{index}\" or content-hash IDs."
                ),
            ))

        # Critical-severity checks: attributes emitted-but-malformed.
        if empty_source_count > 0:
            issues.append(GateIssue(
                severity="critical",
                code="EMPTY_DATA_SEMANTIK_SOURCE",
                message=(
                    f"{empty_source_count}/{total_sections} <section> elements carry "
                    "data-semantik-source but the value is empty"
                ),
                suggestion=(
                    "Emit one of the typed extractor enum values: pdftotext, "
                    "pdfplumber, ocr, synthesized, vendor."
                ),
            ))
        if empty_block_id_count > 0:
            issues.append(GateIssue(
                severity="critical",
                code="EMPTY_DATA_SEMANTIK_BLOCK_ID",
                message=(
                    f"{empty_block_id_count}/{total_sections} <section> elements carry "
                    "data-semantik-block-id but the value is empty"
                ),
                suggestion=(
                    "Populate with the synthesized-JSON block_id "
                    "(e.g. \"s3_c0\") or a 16-hex content hash."
                ),
            ))

        # Score is based only on the critical markers — warning-level
        # provenance attributes are not yet part of the score threshold.
        total_required = len(_REQUIRED_MARKERS)
        critical_issues = [i for i in issues if i.severity == "critical"]
        present = total_required - len(critical_issues)
        score = present / total_required if total_required else 1.0

        passed = len(critical_issues) == 0
        marker_density = (
            present / total_required if total_required else None
        )
        first_critical = next(
            (i.code for i in critical_issues), None
        )
        _emit_decision(
            capture,
            passed=passed,
            code=first_critical,
            pages_audited=1,
            markers_found=present,
            markers_missing=len(critical_issues),
            marker_density=marker_density,
            sections_total=total_sections,
            sections_without_source=sections_without_source,
            sections_without_block_id=sections_without_block_id,
            empty_source_count=empty_source_count,
            empty_block_id_count=empty_block_id_count,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )
