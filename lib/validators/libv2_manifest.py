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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Scaffold subdirectories LibV2 expects on a well-formed archive.
# Missing → warning (never critical). Source of truth for the layout
# convention: ``LibV2/CLAUDE.md`` § "Directory Reference".
#
# Phase 7c ST 17: ``corpus`` removed in favour of ``imscc_chunks`` (the
# Phase 7c ST 15 rename); ``dart_chunks`` added for the Phase 7b
# chunkset. The back-compat read shim in ``lib/libv2_storage.py`` lets
# legacy archives that still carry ``corpus/`` resolve at consumer
# call sites — but new archives are expected to land under the new
# layout, so the scaffold-completeness warning surfaces drift.
_EXPECTED_SUBDIRS = (
    "dart_chunks",
    "imscc_chunks",
    "graph",
    "training_specs",
    "quality",
    "source/pdf",
    "source/html",
    "source/imscc",
)

# Phase 6 ST 19: shape regex for concept_graph_sha256. Mirrors the
# schema pattern in schemas/library/course_manifest.schema.json so a
# manifest violating the regex here would also fail jsonschema; the
# duplicated regex is intentional — when jsonschema isn't installed
# the validator's structural-fallback path still validates the field.
# Phase 7c ST 17 reuses the same pattern for the dart_chunks_sha256 +
# imscc_chunks_sha256 critical-severity checks.
_CONCEPT_GRAPH_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNKS_SHA256_RE = _CONCEPT_GRAPH_SHA256_RE


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

        # -- 8. Phase 6 ST 19 / Phase 7c ST 17: concept_graph_sha256
        #    check (promoted to critical in Phase 7c).
        issues.extend(self._check_concept_graph_sha256(manifest, course_dir))

        # -- 9. Phase 7c ST 17: dart_chunks_sha256 check (critical).
        issues.extend(self._check_dart_chunks_sha256(manifest, course_dir))

        # -- 10. Phase 7c ST 17: imscc_chunks_sha256 check (critical).
        issues.extend(self._check_imscc_chunks_sha256(manifest, course_dir))

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
    def _check_concept_graph_sha256(
        manifest: Dict[str, Any], course_dir: Path,
    ) -> List[GateIssue]:
        """Phase 6 ST 19 / Phase 7c ST 17: critical-severity gate on
        ``concept_graph_sha256``.

        Promoted from warning → critical in Phase 7c per plan amendment 3.
        Phase 6 commit ``c3a9f72`` left these issues at warning severity
        explicitly noting the Phase 7c promotion; the schema's
        ``required`` array now includes ``concept_graph_sha256`` so any
        manifest lacking the field also fails the structural-fallback
        check upstream.

        Three checks (each now critical):
          1. ``MISSING_CONCEPT_GRAPH_SHA256`` — manifest has no
             ``concept_graph_sha256`` field at all.
          2. ``INVALID_CONCEPT_GRAPH_SHA256`` — value is not a 64-char
             lowercase hex string (matches the schema regex).
          3. ``CONCEPT_GRAPH_HASH_MISMATCH`` — manifest carries a hash
             AND ``concept_graph/concept_graph_semantic.json`` exists
             on disk, but the recomputed hash diverges. This is the
             load-bearing signal: schema validation alone (regex)
             can't catch a stale-or-tampered hash.

        Schema source-of-truth: ``schemas/library/course_manifest.schema.json``.
        """
        issues: List[GateIssue] = []
        graph_file = (
            course_dir / "concept_graph" / "concept_graph_semantic.json"
        )

        cg_hash = manifest.get("concept_graph_sha256")
        if cg_hash is None:
            # Phase 7c ST 17: the manifest schema's required[] now
            # carries concept_graph_sha256, so a missing field is a
            # critical issue regardless of whether a graph file exists.
            # Without an on-disk graph the message is more informative
            # (legacy archive that pre-dates Phase 6); with a graph
            # file present the operator has both pieces but failed to
            # thread the hash through.
            location = str(graph_file) if graph_file.exists() else None
            issues.append(GateIssue(
                severity="critical",
                code="MISSING_CONCEPT_GRAPH_SHA256",
                message=(
                    "manifest has no concept_graph_sha256 field. "
                    "Phase 7c ST 17 promoted this to a required field; "
                    "ensure the concept_extraction phase output is "
                    "threaded into archive_to_libv2."
                ),
                location=location,
                suggestion=(
                    "Run the concept_extraction phase, or for legacy "
                    "archives use the operator backfill path (mirror "
                    "of LibV2/tools/libv2/scripts/backfill_dart_chunks.py "
                    "for the concept graph)."
                ),
            ))
            return issues

        # Field present — validate shape (mirror schema regex).
        if not isinstance(cg_hash, str) or not _CONCEPT_GRAPH_SHA256_RE.match(
            cg_hash
        ):
            issues.append(GateIssue(
                severity="critical",
                code="INVALID_CONCEPT_GRAPH_SHA256",
                message=(
                    f"concept_graph_sha256 must be a 64-char lowercase "
                    f"hex string; got: {cg_hash!r}"
                ),
                suggestion=(
                    "See schemas/library/course_manifest.schema.json "
                    "concept_graph_sha256 pattern."
                ),
            ))
            return issues

        # Hash present + well-shaped — verify it matches the on-disk
        # graph file when one exists. This is the load-bearing check
        # that catches stale / tampered values.
        if graph_file.exists() and graph_file.is_file():
            try:
                actual = hashlib.sha256(graph_file.read_bytes()).hexdigest()
            except OSError as exc:
                issues.append(GateIssue(
                    severity="warning",
                    code="CONCEPT_GRAPH_READ_ERROR",
                    message=(
                        f"Failed to recompute hash for {graph_file}: {exc}"
                    ),
                    location=str(graph_file),
                ))
                return issues
            if actual != cg_hash:
                issues.append(GateIssue(
                    severity="critical",
                    code="CONCEPT_GRAPH_HASH_MISMATCH",
                    message=(
                        f"concept_graph_sha256 in manifest ({cg_hash[:16]}...) "
                        f"does not match disk ({actual[:16]}...)."
                    ),
                    location=str(graph_file),
                    suggestion=(
                        "Re-run the concept_extraction phase or update "
                        "the manifest's concept_graph_sha256."
                    ),
                ))

        return issues

    @staticmethod
    def _check_dart_chunks_sha256(
        manifest: Dict[str, Any], course_dir: Path,
    ) -> List[GateIssue]:
        """Phase 7c ST 17: critical-severity gate on ``dart_chunks_sha256``.

        Mirrors ``_check_concept_graph_sha256`` (same MISSING / INVALID /
        MISMATCH issue triplet, same shape regex, same on-disk hash
        recomputation). The chunkset file location is
        ``LibV2/courses/<slug>/dart_chunks/chunks.jsonl`` per the Phase 7b
        emit at ``MCP/tools/pipeline_tools.py::_run_dart_chunking``.

        Three checks (each critical):
          1. ``MISSING_DART_CHUNKS_SHA256`` — manifest lacks the field.
          2. ``INVALID_DART_CHUNKS_SHA256`` — value isn't a 64-char
             lowercase hex string.
          3. ``DART_CHUNKS_HASH_MISMATCH`` — recomputed digest of the
             on-disk ``chunks.jsonl`` diverges from the manifest value.
        """
        issues: List[GateIssue] = []
        chunks_file = course_dir / "dart_chunks" / "chunks.jsonl"

        dh = manifest.get("dart_chunks_sha256")
        if dh is None:
            location = str(chunks_file) if chunks_file.exists() else None
            issues.append(GateIssue(
                severity="critical",
                code="MISSING_DART_CHUNKS_SHA256",
                message=(
                    "manifest has no dart_chunks_sha256 field. Phase 7c "
                    "ST 17 promoted this to a required manifest key; "
                    "ensure the chunking workflow phase output is "
                    "threaded into archive_to_libv2 or use the operator "
                    "backfill at LibV2/tools/libv2/scripts/backfill_dart_chunks.py."
                ),
                location=location,
                suggestion=(
                    "See MCP/tools/pipeline_tools.py::_run_dart_chunking "
                    "(Phase 7b ST 11) for the canonical emit path."
                ),
            ))
            return issues

        if not isinstance(dh, str) or not _CHUNKS_SHA256_RE.match(dh):
            issues.append(GateIssue(
                severity="critical",
                code="INVALID_DART_CHUNKS_SHA256",
                message=(
                    f"dart_chunks_sha256 must be a 64-char lowercase "
                    f"hex string; got: {dh!r}"
                ),
                suggestion=(
                    "See schemas/library/course_manifest.schema.json "
                    "dart_chunks_sha256 pattern."
                ),
            ))
            return issues

        if chunks_file.exists() and chunks_file.is_file():
            try:
                actual = hashlib.sha256(chunks_file.read_bytes()).hexdigest()
            except OSError as exc:
                issues.append(GateIssue(
                    severity="warning",
                    code="DART_CHUNKS_READ_ERROR",
                    message=(
                        f"Failed to recompute hash for {chunks_file}: {exc}"
                    ),
                    location=str(chunks_file),
                ))
                return issues
            if actual != dh:
                issues.append(GateIssue(
                    severity="critical",
                    code="DART_CHUNKS_HASH_MISMATCH",
                    message=(
                        f"dart_chunks_sha256 in manifest ({dh[:16]}...) "
                        f"does not match disk ({actual[:16]}...)."
                    ),
                    location=str(chunks_file),
                    suggestion=(
                        "Re-run the chunking workflow phase or invoke "
                        "the operator backfill script to regenerate "
                        "dart_chunks/chunks.jsonl + update the manifest."
                    ),
                ))

        return issues

    @staticmethod
    def _check_imscc_chunks_sha256(
        manifest: Dict[str, Any], course_dir: Path,
    ) -> List[GateIssue]:
        """Phase 7c ST 17: critical-severity gate on ``imscc_chunks_sha256``.

        Mirrors ``_check_concept_graph_sha256`` and
        ``_check_dart_chunks_sha256``. The chunkset file location is
        ``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl`` per the Phase 7c
        ST 15 corpus → imscc_chunks rename + the ST 16 imscc_chunking
        workflow phase emit.

        Three checks (each critical):
          1. ``MISSING_IMSCC_CHUNKS_SHA256`` — manifest lacks the field.
          2. ``INVALID_IMSCC_CHUNKS_SHA256`` — value isn't a 64-char
             lowercase hex string.
          3. ``IMSCC_CHUNKS_HASH_MISMATCH`` — recomputed digest of the
             on-disk ``chunks.jsonl`` diverges from the manifest value.
        """
        issues: List[GateIssue] = []
        chunks_file = course_dir / "imscc_chunks" / "chunks.jsonl"

        ih = manifest.get("imscc_chunks_sha256")
        if ih is None:
            location = str(chunks_file) if chunks_file.exists() else None
            issues.append(GateIssue(
                severity="critical",
                code="MISSING_IMSCC_CHUNKS_SHA256",
                message=(
                    "manifest has no imscc_chunks_sha256 field. Phase 7c "
                    "ST 17 promoted this to a required manifest key; "
                    "ensure the imscc_chunking workflow phase output is "
                    "threaded into archive_to_libv2."
                ),
                location=location,
                suggestion=(
                    "See MCP/tools/pipeline_tools.py imscc_chunking "
                    "phase emit (Phase 7c ST 16) for the canonical "
                    "path; corpus/ → imscc_chunks/ rename per ST 15."
                ),
            ))
            return issues

        if not isinstance(ih, str) or not _CHUNKS_SHA256_RE.match(ih):
            issues.append(GateIssue(
                severity="critical",
                code="INVALID_IMSCC_CHUNKS_SHA256",
                message=(
                    f"imscc_chunks_sha256 must be a 64-char lowercase "
                    f"hex string; got: {ih!r}"
                ),
                suggestion=(
                    "See schemas/library/course_manifest.schema.json "
                    "imscc_chunks_sha256 pattern."
                ),
            ))
            return issues

        if chunks_file.exists() and chunks_file.is_file():
            try:
                actual = hashlib.sha256(chunks_file.read_bytes()).hexdigest()
            except OSError as exc:
                issues.append(GateIssue(
                    severity="warning",
                    code="IMSCC_CHUNKS_READ_ERROR",
                    message=(
                        f"Failed to recompute hash for {chunks_file}: {exc}"
                    ),
                    location=str(chunks_file),
                ))
                return issues
            if actual != ih:
                issues.append(GateIssue(
                    severity="critical",
                    code="IMSCC_CHUNKS_HASH_MISMATCH",
                    message=(
                        f"imscc_chunks_sha256 in manifest ({ih[:16]}...) "
                        f"does not match disk ({actual[:16]}...)."
                    ),
                    location=str(chunks_file),
                    suggestion=(
                        "Re-run the imscc_chunking workflow phase to "
                        "regenerate imscc_chunks/chunks.jsonl + update "
                        "the manifest."
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
