"""
DART Markers Validator

Validates that DART-processed HTML contains required accessibility markers.
DART-produced HTML must include:
  - Skip link (<a class="skip-link">)
  - Main content landmark (<main role="main">)
  - ARIA-labelled sections (<section aria-labelledby="...">)
  - DART semantic classes (dart-section / dart-document)

Wraps the marker-detection logic from
MCP.tools.pipeline_tools.validate_dart_markers (the MCP tool) into the
ValidationGateManager Validator protocol so it can be wired as a
validation gate in config/workflows.yaml.

Wave 8 addition: source-provenance markers
  - data-dart-source attribute on every <section>
  - data-dart-block-id attribute on every <section>

These are emitted by `DART/multi_source_interpreter.py::generate_html_from_synthesized`
(multi-source path) and stamped with `data-dart-source="claude_llm"` on the
legacy claude_processor path.

Wave 9 promotion: attributes that ARE emitted but are malformed / missing
on a <section> that otherwise carries the DART semantic contract are now
reported at CRITICAL severity. The graceful-fallback rule is preserved:
when a document carries NO <section> elements with the DART semantic
class (pre-Wave-8 legacy HTML), we do not fail — only pages that claim
to be DART-produced and then omit provenance are blocked. This matches
the design doc's "promote after Wave 8 emit has settled" note.

Referenced by: config/workflows.yaml
  - batch_dart.multi_source_synthesis -> dart_markers
  - textbook_to_course.dart_conversion -> dart_markers
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Marker name -> tuple of literal substrings, any of which satisfies the marker.
# Kept in sync with MCP/tools/pipeline_tools.py:validate_dart_markers.
_REQUIRED_MARKERS: Dict[str, Tuple[str, ...]] = {
    "skip_link": ('class="skip', "class='skip"),
    "main_role": ('role="main"', "role='main'"),
    "aria_sections": ('aria-labelledby="', "aria-labelledby='"),
    "dart_semantic_classes": ("dart-section", "dart-document"),
}

# Regex for finding top-level <section> open tags. Used for Wave 8
# warning-level provenance checks. Intentionally permissive (matches any
# attributes) — the presence of the section tag is what we count.
_SECTION_OPEN_RE = re.compile(r"<section\b[^>]*>", re.IGNORECASE)

# Attribute presence checks run against each section's attribute string.
_DATA_DART_SOURCE_RE = re.compile(r'\bdata-dart-source\s*=', re.IGNORECASE)
_DATA_DART_BLOCK_ID_RE = re.compile(r'\bdata-dart-block-id\s*=', re.IGNORECASE)

# Wave 9: critical-severity checks for malformed attributes. An attribute
# that is *present but empty* is a bug in the emit path and must block —
# this is the "emitted-but-malformed" failure mode. Fully-absent attrs
# remain at warning severity per the graceful-fallback rule documented
# at the top of this module.
_EMPTY_DATA_DART_SOURCE_RE = re.compile(
    r'\bdata-dart-source\s*=\s*(["\'])\1', re.IGNORECASE,
)
_EMPTY_DATA_DART_BLOCK_ID_RE = re.compile(
    r'\bdata-dart-block-id\s*=\s*(["\'])\1', re.IGNORECASE,
)


class DartMarkersValidator:
    """Validates DART HTML output for required accessibility markers."""

    name = "dart_markers"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate DART markers in HTML content.

        Expected inputs (any one of):
            html_path: Path to HTML file to validate
            html_content: Raw HTML string (alternative to html_path)
            gate_id: Optional gate_id override for the result

        Returns:
            GateResult with one critical issue per missing marker.
        """
        gate_id = inputs.get("gate_id", "dart_markers")
        content = inputs.get("html_content", "") or ""

        if not content and inputs.get("html_path"):
            path = Path(inputs["html_path"])
            if not path.exists():
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="FILE_NOT_FOUND",
                        message=f"DART HTML file not found: {path}",
                        location=str(path),
                    )],
                )
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as e:
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="FILE_READ_ERROR",
                        message=f"Failed to read DART HTML file: {e}",
                        location=str(path),
                    )],
                )

        if not content.strip():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="EMPTY_CONTENT",
                    message="DART HTML content is empty (no html_path or html_content supplied).",
                )],
            )

        issues: List[GateIssue] = []
        for marker_name, needles in _REQUIRED_MARKERS.items():
            if not any(needle in content for needle in needles):
                issues.append(GateIssue(
                    severity="critical",
                    code=f"MISSING_{marker_name.upper()}",
                    message=f"Required DART marker missing: {marker_name}",
                    suggestion=f"Ensure DART output emits one of: {needles}",
                ))

        # Wave 8 + Wave 9: source-provenance marker checks.
        #
        # Rules:
        #   - Absent attributes on every <section>           -> warning
        #     (graceful fallback for pre-Wave-8 legacy HTML).
        #   - Some <section>s carry attrs, others don't      -> warning
        #     (emit has settled but coverage is incomplete).
        #   - An attr is PRESENT but the value is EMPTY      -> critical
        #     (Wave 9 "emitted-but-malformed" blocker).
        section_tags = _SECTION_OPEN_RE.findall(content)
        total_sections = len(section_tags)
        sections_without_source = 0
        sections_without_block_id = 0
        empty_source_count = 0
        empty_block_id_count = 0
        for tag in section_tags:
            if not _DATA_DART_SOURCE_RE.search(tag):
                sections_without_source += 1
            elif _EMPTY_DATA_DART_SOURCE_RE.search(tag):
                empty_source_count += 1
            if not _DATA_DART_BLOCK_ID_RE.search(tag):
                sections_without_block_id += 1
            elif _EMPTY_DATA_DART_BLOCK_ID_RE.search(tag):
                empty_block_id_count += 1

        if total_sections > 0 and sections_without_source > 0:
            issues.append(GateIssue(
                severity="warning",
                code="MISSING_DATA_DART_SOURCE",
                message=(
                    f"{sections_without_source}/{total_sections} <section> elements "
                    "missing data-dart-source attribute"
                ),
                suggestion=(
                    "Ensure DART emits data-dart-source on every <section>. "
                    "Multi-source path: data-dart-source=\"pdfplumber\" etc. "
                    "Legacy claude_processor path: data-dart-source=\"claude_llm\"."
                ),
            ))

        if total_sections > 0 and sections_without_block_id > 0:
            issues.append(GateIssue(
                severity="warning",
                code="MISSING_DATA_DART_BLOCK_ID",
                message=(
                    f"{sections_without_block_id}/{total_sections} <section> elements "
                    "missing data-dart-block-id attribute"
                ),
                suggestion=(
                    "Ensure DART emits data-dart-block-id on every <section>. "
                    "Multi-source path uses \"s{index}\" or content-hash IDs."
                ),
            ))

        # Wave 9 critical-severity checks: attributes emitted-but-malformed.
        if empty_source_count > 0:
            issues.append(GateIssue(
                severity="critical",
                code="EMPTY_DATA_DART_SOURCE",
                message=(
                    f"{empty_source_count}/{total_sections} <section> elements carry "
                    "data-dart-source but the value is empty"
                ),
                suggestion=(
                    "Emit one of the typed extractor enum values: pdftotext, "
                    "pdfplumber, ocr, synthesized, claude_llm."
                ),
            ))
        if empty_block_id_count > 0:
            issues.append(GateIssue(
                severity="critical",
                code="EMPTY_DATA_DART_BLOCK_ID",
                message=(
                    f"{empty_block_id_count}/{total_sections} <section> elements carry "
                    "data-dart-block-id but the value is empty"
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

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=len(critical_issues) == 0,
            score=score,
            issues=issues,
        )
