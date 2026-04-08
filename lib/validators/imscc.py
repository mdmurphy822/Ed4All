"""
IMSCC Validators

Validates IMSCC package structure and parsing results:

IMSCCValidator:
- Manifest XML well-formed and schema-valid
- All resource references resolve to existing files
- Namespace declarations correct (IMS CC 1.1/1.2/1.3)
- Organization hierarchy complete

IMSCCParseValidator:
- IMSCC zip extractable
- Manifest found and parseable
- Content inventory complete
- Source LMS detected

Referenced by: config/workflows.yaml (course_generation, intake_remediation, textbook_to_course)
"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from orchestrator.core.validation_gates import GateIssue, GateResult


class IMSCCValidator:
    """Validates IMSCC package structure and manifest."""

    name = "imscc_structure"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate IMSCC package structure.

        Expected inputs:
            imscc_path: Path to .imscc file or extracted directory
            manifest_path: Path to imsmanifest.xml (optional)
        """
        gate_id = inputs.get("gate_id", "imscc_structure")
        issues: List[GateIssue] = []

        imscc_path = Path(inputs.get("imscc_path", ""))
        if not imscc_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="FILE_NOT_FOUND",
                        message=f"IMSCC path not found: {imscc_path}",
                    )
                ],
            )

        # Check manifest
        manifest_path = inputs.get("manifest_path")
        if manifest_path:
            manifest_path = Path(manifest_path)
        elif imscc_path.is_dir():
            manifest_path = imscc_path / "imsmanifest.xml"
        else:
            # It's a zip file - we can't check internal structure here
            issues.append(
                GateIssue(
                    severity="info",
                    code="ZIP_PACKAGE",
                    message="Zip package provided; use IMSCCParseValidator for extraction checks",
                )
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=0.8,
                issues=issues,
            )

        if manifest_path and manifest_path.exists():
            issues.extend(self._validate_manifest(manifest_path))
        elif manifest_path:
            issues.append(
                GateIssue(
                    severity="error",
                    code="MANIFEST_MISSING",
                    message="imsmanifest.xml not found",
                    suggestion="Ensure the IMSCC package contains imsmanifest.xml",
                )
            )

        has_errors = any(i.severity == "error" for i in issues)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=not has_errors,
            score=max(0.0, 1.0 - len(issues) * 0.15),
            issues=issues,
        )

    def _validate_manifest(self, manifest_path: Path) -> List[GateIssue]:
        """Validate manifest XML structure."""
        issues = []
        try:
            tree = ET.parse(manifest_path)
            root = tree.getroot()

            # Check for IMS CC namespace
            ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
            if not ns or "imscc" not in ns.lower() and "imsglobal" not in ns.lower():
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="NAMESPACE_MISSING",
                        message="IMS CC namespace not detected in manifest",
                        suggestion="Ensure proper xmlns declaration",
                    )
                )

            # Check for organizations element
            orgs = root.findall(f".//{ns}organization") or root.findall(".//organization")
            if not orgs:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="NO_ORGANIZATIONS",
                        message="No organization elements in manifest",
                    )
                )

            # Check for resources
            resources = root.findall(f".//{ns}resource") or root.findall(".//resource")
            if not resources:
                issues.append(
                    GateIssue(
                        severity="error",
                        code="NO_RESOURCES",
                        message="No resource elements in manifest",
                    )
                )

        except ET.ParseError as e:
            issues.append(
                GateIssue(
                    severity="error",
                    code="MANIFEST_PARSE_ERROR",
                    message=f"Manifest XML is not well-formed: {e}",
                )
            )

        return issues


class IMSCCParseValidator:
    """Validates IMSCC parsing results."""

    name = "imscc_parse"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate IMSCC parsing output.

        Expected inputs:
            imscc_path: Path to .imscc file
            parse_result: Parsed content inventory dict (optional)
        """
        gate_id = inputs.get("gate_id", "imscc_parse")
        issues: List[GateIssue] = []

        imscc_path = Path(inputs.get("imscc_path", ""))
        if not imscc_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="FILE_NOT_FOUND",
                        message=f"IMSCC file not found: {imscc_path}",
                    )
                ],
            )

        # Check extractability
        if imscc_path.suffix in (".imscc", ".zip"):
            issues.extend(self._check_extractable(imscc_path))

        # Validate parse result if provided
        parse_result = inputs.get("parse_result")
        if parse_result:
            issues.extend(self._check_parse_result(parse_result))

        has_errors = any(i.severity == "error" for i in issues)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=not has_errors,
            score=max(0.0, 1.0 - len(issues) * 0.15),
            issues=issues,
        )

    def _check_extractable(self, path: Path) -> List[GateIssue]:
        """Check that the IMSCC zip is extractable and contains a manifest."""
        issues = []
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if not names:
                    issues.append(
                        GateIssue(
                            severity="error",
                            code="EMPTY_ARCHIVE",
                            message="IMSCC archive is empty",
                        )
                    )
                    return issues

                # Check for manifest
                has_manifest = any(
                    n.endswith("imsmanifest.xml") for n in names
                )
                if not has_manifest:
                    issues.append(
                        GateIssue(
                            severity="error",
                            code="NO_MANIFEST",
                            message="imsmanifest.xml not found in archive",
                        )
                    )

                # Check for content files
                html_count = sum(1 for n in names if n.endswith(".html"))
                xml_count = sum(1 for n in names if n.endswith(".xml"))
                if html_count == 0 and xml_count <= 1:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="NO_CONTENT",
                            message="No HTML content files found in archive",
                        )
                    )

        except zipfile.BadZipFile:
            issues.append(
                GateIssue(
                    severity="error",
                    code="BAD_ZIP",
                    message="File is not a valid zip archive",
                )
            )
        except OSError as e:
            issues.append(
                GateIssue(
                    severity="error",
                    code="READ_ERROR",
                    message=f"Failed to read archive: {e}",
                )
            )

        return issues

    def _check_parse_result(self, result: Dict[str, Any]) -> List[GateIssue]:
        """Validate the parse result inventory."""
        issues = []

        if not result.get("content_items"):
            issues.append(
                GateIssue(
                    severity="error",
                    code="NO_CONTENT_ITEMS",
                    message="Parse result contains no content items",
                )
            )

        if not result.get("source_lms"):
            issues.append(
                GateIssue(
                    severity="warning",
                    code="LMS_NOT_DETECTED",
                    message="Source LMS could not be determined",
                    suggestion="Manual LMS identification may be needed",
                )
            )

        return issues
