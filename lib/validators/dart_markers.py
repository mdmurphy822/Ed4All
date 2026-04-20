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

Referenced by: config/workflows.yaml
  - batch_dart.multi_source_synthesis -> dart_markers
  - textbook_to_course.dart_conversion -> dart_markers
"""

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

        total_required = len(_REQUIRED_MARKERS)
        present = total_required - len(issues)
        score = present / total_required if total_required else 1.0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
        )
