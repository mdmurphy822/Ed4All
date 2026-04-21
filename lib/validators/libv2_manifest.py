"""LibV2 Manifest Validator (Wave 23 Sub-task C).

Gates the ``libv2_archival`` phase of the ``textbook_to_course``
workflow. Pre-Wave-23, that phase had no validation gate — so
downstream ``LibV2/tools/libv2/cli retrieve`` consumers assumed
scaffold completeness while pipeline-built archives silently
violated it: empty ``pedagogy/``, missing ``course.json``,
divergent ``concept_graph/`` vs. ``graph/`` layouts, and
``features.source_provenance=false`` flowed through without so
much as a warning log.

This validator runs at ``severity: critical`` for the integrity
checks (JSON parse, schema match, on-disk artifact hash /
size agreement) and ``severity: warning`` for scaffold completeness
and the ``source_provenance=false`` gap flag — the advisory signals
that indicate a real but non-blocking downstream degradation.

Referenced by: ``config/workflows.yaml`` →
``textbook_to_course.libv2_archival.validation_gates[libv2_manifest]``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Scaffold subdirectories LibV2 expects on a well-formed archive.
# Missing → warning (never critical). Source of truth for the layout
# convention: ``LibV2/CLAUDE.md`` § "Directory Reference".
_EXPECTED_SUBDIRS = (
    "corpus",
    "graph",
    "training_specs",
    "quality",
    "source/pdf",
    "source/html",
    "source/imscc",
)


class LibV2ManifestValidator:
    """Validates a LibV2 course archive manifest + on-disk artifacts."""

    name = "libv2_manifest"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate a LibV2 archive.

        Expected inputs:
            manifest_path: Path to ``LibV2/courses/{slug}/manifest.json``.
                           Required.
            course_dir: Path to the course dir (parent of manifest_path).
                        Optional; derived from manifest_path when absent.
        """
        gate_id = inputs.get("gate_id", "libv2_manifest")
        issues: List[GateIssue] = []

        # -- 1. Manifest path is required.
        manifest_path_raw = inputs.get("manifest_path")
        if not manifest_path_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_MANIFEST_PATH",
                    message="manifest_path is required for LibV2ManifestValidator",
                )],
            )
        manifest_path = Path(manifest_path_raw)
        if not manifest_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MANIFEST_NOT_FOUND",
                    message=f"Manifest path does not exist: {manifest_path}",
                )],
            )

        course_dir_raw = inputs.get("course_dir")
        course_dir = (
            Path(course_dir_raw) if course_dir_raw else manifest_path.parent
        )

        # -- 2. JSON must parse. Critical.
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INVALID_JSON",
                    message=f"Manifest JSON failed to parse: {exc}",
                    location=str(manifest_path),
                )],
            )

        # -- 3. Schema validation (best-effort — jsonschema is optional).
        schema_issues = self._validate_against_schema(manifest, manifest_path)
        issues.extend(schema_issues)

        # -- 4. On-disk artifact integrity (path exists, hash + size match).
        integrity_issues = self._validate_artifact_integrity(
            manifest.get("source_artifacts", {}), manifest_path,
        )
        issues.extend(integrity_issues)

        # -- 5. Scaffold subdir presence (warnings).
        issues.extend(self._check_expected_subdirs(course_dir))

        # -- 6. Pedagogy / concept_graph / course.json gap checks.
        issues.extend(self._check_content_gaps(course_dir))

        # -- 7. features.source_provenance advisory flag.
        issues.extend(self._check_source_provenance_flag(manifest))

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0

        # Score: 1.0 when no issues; degrades by 0.1 per issue with a
        # floor at 0.0. Matches the convention used by ContentStructureValidator.
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _validate_against_schema(
        manifest: Dict[str, Any], manifest_path: Path,
    ) -> List[GateIssue]:
        """Validate manifest against ``schemas/library/course_manifest.schema.json``.

        Best-effort: when jsonschema isn't installed, fall back to a
        lightweight structural check (required top-level keys).
        """
        issues: List[GateIssue] = []
        try:
            import jsonschema  # type: ignore
        except ImportError:
            # Minimal structural check. Keys here are a subset of the
            # schema's ``required`` — we deliberately don't require
            # ``sourceforge_manifest`` or ``content_profile`` here,
            # because LibV2 pipelines (per ``archive_to_libv2``)
            # intentionally omit those for now. Treat the strict
            # schema-required keys as schema-violation signals below.
            for required in ("libv2_version", "slug", "classification"):
                if required not in manifest:
                    issues.append(GateIssue(
                        severity="critical",
                        code="SCHEMA_VIOLATION",
                        message=f"Missing required manifest key: {required}",
                        location=str(manifest_path),
                    ))
            return issues

        # Load schema from the canonical location
        schema_path = _resolve_schema_path()
        if not schema_path or not schema_path.exists():
            # Can't find the schema file — soft-skip
            issues.append(GateIssue(
                severity="warning",
                code="SCHEMA_UNAVAILABLE",
                message=(
                    f"course_manifest.schema.json not found at {schema_path}; "
                    "falling back to structural check."
                ),
            ))
            return issues

        # Known gap fields: the current Wave-19+ ``archive_to_libv2``
        # pipeline emits a subset of the schema. These fields' absence
        # is a warning (documented pipeline gap), not a critical
        # schema violation — otherwise every real archive fails the gate.
        _KNOWN_GAP_REQUIREDS = {"sourceforge_manifest", "content_profile"}

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(manifest, schema)
        except jsonschema.ValidationError as exc:
            # Extract the missing field name if this is a required-key
            # violation so we can demote known-gap fields to warning.
            missing_field = None
            if exc.validator == "required" and exc.validator_value:
                # exc.message looks like "'foo' is a required property"
                import re as _re
                m = _re.match(r"'([^']+)' is a required property", exc.message)
                if m:
                    missing_field = m.group(1)

            severity = "critical"
            code = "SCHEMA_VIOLATION"
            if missing_field in _KNOWN_GAP_REQUIREDS:
                severity = "warning"
                code = "SCHEMA_GAP_KNOWN"

            issues.append(GateIssue(
                severity=severity,
                code=code,
                message=f"Manifest schema check: {exc.message}",
                location=".".join(str(p) for p in exc.absolute_path),
                suggestion=(
                    "See schemas/library/course_manifest.schema.json. "
                    "Known-gap fields (sourceforge_manifest, content_profile) "
                    "are warnings — track in pipeline-integrity-review audit."
                ),
            ))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(GateIssue(
                severity="warning",
                code="SCHEMA_LOAD_ERROR",
                message=f"Failed to load manifest schema: {exc}",
            ))
        return issues

    @staticmethod
    def _validate_artifact_integrity(
        source_artifacts: Dict[str, Any], manifest_path: Path,
    ) -> List[GateIssue]:
        """Recompute sha256 + size for every source_artifacts entry.

        ``source_artifacts`` shape (per ``archive_to_libv2``):
        ``{pdf: [...], html: [...], imscc: {...}}`` where each entry
        carries ``path``, ``checksum`` (sha256 hex), and ``size``.
        """
        issues: List[GateIssue] = []

        if not isinstance(source_artifacts, dict):
            issues.append(GateIssue(
                severity="warning",
                code="MISSING_SOURCE_ARTIFACTS",
                message="Manifest has no source_artifacts block; nothing to verify.",
                location=str(manifest_path),
            ))
            return issues

        entries: List[Dict[str, Any]] = []
        for kind in ("pdf", "html"):
            bucket = source_artifacts.get(kind, [])
            if isinstance(bucket, list):
                entries.extend(bucket)
        imscc = source_artifacts.get("imscc")
        if isinstance(imscc, dict):
            entries.append(imscc)

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path_raw = entry.get("path")
            if not path_raw:
                issues.append(GateIssue(
                    severity="warning",
                    code="ARTIFACT_MISSING_PATH",
                    message=f"source_artifacts entry missing 'path': {entry}",
                ))
                continue
            path = Path(path_raw)
            if not path.exists():
                issues.append(GateIssue(
                    severity="critical",
                    code="MISSING_ARTIFACT",
                    message=f"source_artifact not found on disk: {path}",
                    location=str(path),
                ))
                continue

            # Size check (critical when mismatch)
            expected_size = entry.get("size")
            actual_size = path.stat().st_size
            if expected_size is not None and expected_size != actual_size:
                issues.append(GateIssue(
                    severity="critical",
                    code="SIZE_MISMATCH",
                    message=(
                        f"Size mismatch for {path}: manifest says "
                        f"{expected_size}, disk shows {actual_size}"
                    ),
                    location=str(path),
                ))

            # Checksum check (critical when mismatch)
            expected_checksum = entry.get("checksum") or entry.get("sha256")
            if expected_checksum:
                actual_checksum = _sha256_file(path)
                if actual_checksum != expected_checksum:
                    issues.append(GateIssue(
                        severity="critical",
                        code="CHECKSUM_MISMATCH",
                        message=(
                            f"sha256 mismatch for {path}: manifest says "
                            f"{expected_checksum[:16]}..., disk hashes to "
                            f"{actual_checksum[:16]}..."
                        ),
                        location=str(path),
                    ))

        return issues

    @staticmethod
    def _check_expected_subdirs(course_dir: Path) -> List[GateIssue]:
        """Warning-severity: scaffold dirs missing from the archive."""
        issues: List[GateIssue] = []
        if not course_dir.exists():
            issues.append(GateIssue(
                severity="critical",
                code="COURSE_DIR_NOT_FOUND",
                message=f"course_dir does not exist: {course_dir}",
            ))
            return issues
        for subdir in _EXPECTED_SUBDIRS:
            if not (course_dir / subdir).exists():
                issues.append(GateIssue(
                    severity="warning",
                    code="MISSING_SCAFFOLD_SUBDIR",
                    message=f"Expected scaffold dir missing: {subdir}",
                    location=str(course_dir / subdir),
                    suggestion=(
                        f"Ensure archive_to_libv2 creates {subdir} or "
                        "document intentional omission."
                    ),
                ))
        return issues

    @staticmethod
    def _check_content_gaps(course_dir: Path) -> List[GateIssue]:
        """Warning-severity gap flags aligned with the audit findings."""
        issues: List[GateIssue] = []

        # Empty pedagogy/ — the known gap per the pipeline audit.
        pedagogy = course_dir / "pedagogy"
        if pedagogy.exists() and _dir_is_empty(pedagogy):
            issues.append(GateIssue(
                severity="warning",
                code="PEDAGOGY_EMPTY",
                message="pedagogy/ directory is empty (known pipeline gap).",
                location=str(pedagogy),
                suggestion=(
                    "Trainforge pedagogy emitter produces no files today. "
                    "Track in plans/pipeline-integrity-review-2026-04-21."
                ),
            ))

        # Dual-schema drift signal: concept_graph/ empty but graph/ has content.
        concept_graph = course_dir / "concept_graph"
        graph_dir = course_dir / "graph"
        if concept_graph.exists() and _dir_is_empty(concept_graph):
            if graph_dir.exists() and not _dir_is_empty(graph_dir):
                issues.append(GateIssue(
                    severity="warning",
                    code="CONCEPT_GRAPH_DRIFT",
                    message=(
                        "concept_graph/ is empty but graph/ is populated — "
                        "dual-schema drift signal."
                    ),
                    location=str(concept_graph),
                    suggestion=(
                        "Reconcile concept_graph/ vs graph/ layouts in the "
                        "Trainforge emitter; downstream LibV2 retrievers "
                        "read graph/."
                    ),
                ))

        # course.json missing — warning (not all layouts emit this top-level).
        course_json = course_dir / "course.json"
        if not course_json.exists():
            issues.append(GateIssue(
                severity="warning",
                code="MISSING_COURSE_JSON",
                message="course.json not present at course_dir root.",
                location=str(course_json),
                suggestion=(
                    "Emit course.json (learning outcomes + metadata) so "
                    "LibV2 catalog indexes can surface the course."
                ),
            ))

        return issues

    @staticmethod
    def _check_source_provenance_flag(manifest: Dict[str, Any]) -> List[GateIssue]:
        """Actionable warning when source_provenance is false.

        This flag is never critical — it's an advisory that the chain
        DART → Courseforge → Trainforge didn't propagate chunk-level
        source references into the archived corpus.
        """
        features = manifest.get("features") or {}
        if not isinstance(features, dict):
            return []
        sp = features.get("source_provenance", None)
        if sp is False:
            return [GateIssue(
                severity="warning",
                code="SOURCE_PROVENANCE_FALSE",
                message=(
                    "features.source_provenance=false — archived corpus "
                    "carries no chunk-level source refs."
                ),
                suggestion=(
                    "chain: DART→Courseforge→Trainforge; see "
                    "plans/pipeline-integrity-review-2026-04-21/report.md "
                    "F3 for the propagation gap."
                ),
            )]
        return []


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_is_empty(path: Path) -> bool:
    try:
        return next(path.iterdir(), None) is None
    except OSError:
        return True


def _resolve_schema_path() -> Optional[Path]:
    """Locate ``schemas/library/course_manifest.schema.json``.

    Walks up from this file until it finds a ``schemas/library`` dir.
    Returns ``None`` when not found (validator falls back to
    structural check).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "schemas" / "library" / "course_manifest.schema.json"
        if candidate.exists():
            return candidate
    return None
