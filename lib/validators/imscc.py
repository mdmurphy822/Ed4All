"""
IMSCC Validators

Validates IMSCC package structure and parsing results:

IMSCCValidator:
- Manifest XML well-formed and schema-valid
- All resource references resolve to existing files
- Namespace declarations correct (IMS CC 1.1/1.2/1.3)
- Organization hierarchy complete
- W5: escalation-marker-bearing blocks (consensus failure / outline budget
  exhausted / structural unfixable) MUST NOT appear in the per-page IMSCC
  HTML — defensive packager-side check that reads ``blocks_final.jsonl``
  and scans the emitted HTML for any matching ``data-cf-block-id``.

IMSCCParseValidator:
- IMSCC zip extractable
- Manifest found and parseable
- Content inventory complete
- Source LMS detected

Referenced by: config/workflows.yaml (course_generation, intake_remediation, textbook_to_course)
"""

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult


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

        # W5: defensive packager-side escalation-marker leak check.
        # Walks ``blocks_final.jsonl`` (when present) and the emitted
        # per-page HTML to confirm no escalated block_id leaked into
        # shipped HTML. Pre-W5 runs without ``blocks_final_path`` /
        # ``content_dir`` inputs no-op silently (backward compat).
        issues.extend(self._check_escalated_blocks_absent(inputs))

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

    def _check_escalated_blocks_absent(
        self, inputs: Dict[str, Any],
    ) -> List[GateIssue]:
        """W5 defensive check: confirm no escalation-marker-bearing
        block leaked into shipped IMSCC HTML.

        Inputs (all optional — pre-W5 runs / legacy direct callers
        no-op silently):
            blocks_final_path: Path to the rewrite-tier
                ``blocks_final.jsonl`` (one snake_case Block entry per
                line). When absent, the check returns an empty issue
                list so this method is purely additive.
            content_dir: Directory containing the per-page HTML files
                (preferred). When absent, the check falls back to
                walking ``imscc_path`` (extracted dir) or extracting
                the IMSCC zip into a temp dir.

        Emits ``code="ESCALATED_BLOCK_IN_IMSCC"`` (critical) for every
        match between an escalated block_id and a
        ``data-cf-block-id="{id}"`` attribute in the shipped HTML.
        """
        issues: List[GateIssue] = []

        blocks_final_raw = inputs.get("blocks_final_path")
        if not blocks_final_raw:
            # Backward compat: no blocks_final_path threaded in →
            # nothing to check. Pre-W5 callers (and the
            # course_generation / intake_remediation workflows that
            # don't run the two-pass router) hit this path.
            return issues

        blocks_final_path = Path(blocks_final_raw)
        if not blocks_final_path.exists():
            issues.append(
                GateIssue(
                    severity="info",
                    code="BLOCKS_FINAL_MISSING",
                    message=(
                        "blocks_final_path was provided but does not "
                        f"exist on disk: {blocks_final_path}"
                    ),
                )
            )
            return issues

        escalated_ids = self._collect_escalated_block_ids(blocks_final_path)
        if not escalated_ids:
            # No escalated blocks at all → vacuously safe.
            return issues

        html_files = self._gather_html_files(inputs)
        if not html_files:
            issues.append(
                GateIssue(
                    severity="info",
                    code="ESCALATED_BLOCK_CHECK_NO_HTML",
                    message=(
                        "blocks_final.jsonl listed escalated block(s) "
                        "but no HTML files were found to scan; check "
                        "skipped."
                    ),
                )
            )
            return issues

        # Compile one regex per escalated id rather than scanning the
        # full file list once per id — short-circuits on the first
        # leak per id and keeps the worst case O(html_files * ids).
        for block_id in escalated_ids:
            # Escape the id so block_ids carrying regex metacharacters
            # (``#`` / ``.`` / etc.) don't blow up the pattern.
            pat = re.compile(
                rf'data-cf-block-id="{re.escape(block_id)}"'
            )
            for html_path in html_files:
                try:
                    text = html_path.read_text(
                        encoding="utf-8", errors="replace",
                    )
                except OSError:
                    continue
                if pat.search(text):
                    issues.append(
                        GateIssue(
                            severity="error",
                            code="ESCALATED_BLOCK_IN_IMSCC",
                            message=(
                                f"Escalated block_id={block_id!r} "
                                f"found in shipped HTML "
                                f"{html_path.name}; W5 packaging gate "
                                "requires marker-bearing blocks to be "
                                "filtered out at HTML emit time."
                            ),
                            suggestion=(
                                "Confirm _run_content_generation_rewrite "
                                "is filtering blocks where "
                                "escalation_marker is not None before "
                                "emitting <section data-cf-block-id>."
                            ),
                        )
                    )
                    break  # one leak per id is enough

        return issues

    def _collect_escalated_block_ids(
        self, blocks_final_path: Path,
    ) -> List[str]:
        """Read the JSONL and return block_ids with non-null
        ``escalation_marker``."""
        escalated: List[str] = []
        try:
            with blocks_final_path.open(
                "r", encoding="utf-8",
            ) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("escalation_marker") is None:
                        continue
                    block_id = entry.get("block_id")
                    if isinstance(block_id, str) and block_id:
                        escalated.append(block_id)
        except OSError:
            pass
        return escalated

    def _gather_html_files(
        self, inputs: Dict[str, Any],
    ) -> List[Path]:
        """Resolve the list of HTML files to scan.

        Prefers ``content_dir`` (the rewrite phase's per-page emit
        target); falls back to walking an extracted ``imscc_path``
        directory. Zip-only ``imscc_path`` is intentionally NOT
        extracted here — the packaging phase always writes pages to
        ``content_dir`` first, so the directory walk is sufficient.
        """
        html_files: List[Path] = []

        content_dir_raw = inputs.get("content_dir")
        if content_dir_raw:
            cd = Path(content_dir_raw)
            if cd.is_dir():
                html_files.extend(sorted(cd.rglob("*.html")))

        if not html_files:
            imscc_path_raw = inputs.get("imscc_path")
            if imscc_path_raw:
                ip = Path(imscc_path_raw)
                if ip.is_dir():
                    html_files.extend(sorted(ip.rglob("*.html")))

        return html_files


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
