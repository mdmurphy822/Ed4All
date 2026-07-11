"""Phase 7b Subtask 13 — ChunksetManifestValidator.

Gates the per-chunkset sidecar manifest emitted alongside ``chunks.jsonl``
at ``LibV2/courses/<slug>/dart_chunks/manifest.json`` (Phase 7b) and
``LibV2/courses/<slug>/imscc_chunks/manifest.json`` (Phase 7c).

The schema is symmetric across both chunkset kinds (a single
``chunkset_kind`` discriminator drives the conditional source-SHA
requirement); see ``schemas/library/chunkset_manifest.schema.json``
(landed at commit ``626a53b`` by Subtask 12).

Per-manifest contract enforced here:

1. **Manifest path is provided** — required input
   ``chunkset_manifest_path``. Missing input → ``CHUNKSET_MANIFEST_MISSING_INPUT``
   (critical, ``action="block"``).
2. **Manifest exists on disk** — ``CHUNKSET_MANIFEST_NOT_FOUND``
   (critical, ``action="block"``).
3. **Manifest JSON parses** — ``CHUNKSET_MANIFEST_INVALID_JSON``
   (critical, ``action="block"``).
4. **Manifest validates against the canonical schema** — when
   ``jsonschema`` is installed, validates against
   ``schemas/library/chunkset_manifest.schema.json`` and emits
   ``CHUNKSET_MANIFEST_SCHEMA_VIOLATION`` (critical) per violation.
   When ``jsonschema`` isn't available the validator falls back to a
   structural check covering the schema's ``required`` fields plus the
   conditional source-SHA field for the declared chunkset_kind.
5. **Sibling ``chunks.jsonl`` exists** — sits next to ``manifest.json``
   in the same directory. Missing → ``CHUNKSET_CHUNKS_NOT_FOUND``
   (critical).
6. **Recomputed SHA-256 matches manifest's ``chunks_sha256``** —
   the load-bearing tamper-detection check. Mismatch →
   ``CHUNKSET_HASH_MISMATCH`` (critical).
7. **When ``chunks_count`` is present, line count matches** —
   recomputes the JSONL line count and compares. Mismatch →
   ``CHUNKSET_COUNT_MISMATCH`` (critical).
8. **Import-coverage gate (W1.2)** — when the operator opts in via
   ``ED4ALL_CHUNK_COVERAGE_FLOOR`` / ``ED4ALL_MIN_CHUNKS`` (both
   parse-with-fallback; unset / non-positive → OFF → byte-identical
   no-op), reads the manifest's ``source_coverage`` telemetry and fires
   WARNING issues when coverage / emitted-chunk counts fall below the
   floors: ``CHUNKSET_COVERAGE_LOW`` (``source_coverage.coverage_pct``
   below the floor) and ``CHUNKSET_TOO_THIN``
   (``source_coverage.emitted_count`` below the minimum). Warning-day-1
   with a deferred critical-flip (``# TODO(calibration)``); a missing /
   malformed ``source_coverage`` (pre-W3.H manifest) degrades gracefully
   with no issue.

Severity contract:

* The validator emits **critical-severity** GateIssues for every
  failure case so that a tampered or partially-written chunkset is
  visible to downstream consumers.
* The workflow YAML gate (Worker W10's territory) wires this validator
  as ``severity: warning`` initially per the Wave 7b-2 plan — the gate
  config tells the workflow runner whether to block. The validator
  itself is severity-honest.
* On any critical issue, ``action="block"``; otherwise ``action`` is
  ``None`` and ``passed=True``.

Inputs contract:

* ``inputs["chunkset_manifest_path"]`` — path to the manifest JSON.
  Required.
* ``inputs["gate_id"]`` — optional GateResult ID override; defaults to
  the validator name.

Cross-references:

* ``schemas/library/chunkset_manifest.schema.json`` — canonical
  schema landed by Subtask 12.
* ``lib/validators/libv2_manifest.py::_check_concept_graph_sha256``
  (Phase 6 ST 19) — closest analog: file-on-disk + SHA-256 verification
  pattern, see ``:417-521``.
* ``lib/validators/concept_graph.py`` — Phase 6 sibling validator with
  the same file-existence + JSON-load + structural-validation shape.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.utils import sha256_file

logger = logging.getLogger(__name__)


# 64-char lowercase hex — mirrors the schema regex on chunks_sha256 /
# source_*_sha256. Duplicated here so the structural-fallback path
# (when jsonschema isn't installed) still validates the SHA fields.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Chunker-version regex — mirrors the schema regex on
# chunker_version. Same fallback rationale as _SHA256_RE.
#
# Migration drift (post-Phase-8 review): the field used to carry the
# Python-package release version of the standalone chunker workspace
# (formerly ``ed4all-chunker``, now folded back into ``Trainforge.chunker``;
# e.g. ``"0.1.0"``); it now carries the chunker-schema-
# contract version emitted by ``Trainforge.chunker.
# CHUNKER_SCHEMA_VERSION`` (e.g. ``"v4"``). The alternation accepts
# BOTH shapes so any pre-migration manifest still on disk — including
# the legacy ``"0.0.0+missing"`` fallback sentinel — continues to
# validate.
_CHUNKER_VERSION_RE = re.compile(
    r"^(?:v\d+|\d+\.\d+\.\d+(?:[+-][A-Za-z0-9.+-]+)?)$"
)

# Required top-level fields per the schema (excluding the conditional
# source_*_sha256 fields driven by chunkset_kind).
_BASE_REQUIRED_FIELDS = ("chunks_sha256", "chunker_version", "chunkset_kind")

# Allowed chunkset_kind values per the schema enum.
# DART->semantik naming purge Stage 1 (dual-READ): accept the ratified
# "semantik" kind alongside legacy "dart". Emitters still write "dart".
_ALLOWED_CHUNKSET_KINDS = {"dart", "semantik", "imscc"}

# Conditional source-SHA field per chunkset_kind (driven by the schema's
# allOf / if-then clauses). The ratified "semantik" kind requires the ratified
# source-SHA field name ``source_semantik_html_sha256``.
_CONDITIONAL_SOURCE_FIELD = {
    "dart": "source_dart_html_sha256",
    "semantik": "source_semantik_html_sha256",
    "imscc": "source_imscc_sha256",
}

# Whitelist of recognised top-level keys (for the structural-fallback
# additionalProperties: false check).
_KNOWN_FIELDS = frozenset({
    "chunks_sha256",
    "chunker_version",
    "chunkset_kind",
    "source_dart_html_sha256",
    "source_semantik_html_sha256",
    "source_imscc_sha256",
    "chunks_count",
    "generated_at",
    "overlap_words",
    "source_coverage",
})


def resolve_coverage_floor() -> float:
    """Resolve ``ED4ALL_CHUNK_COVERAGE_FLOOR`` (parse-with-fallback).

    The import-coverage gate (W1.2) compares the manifest's
    ``source_coverage.coverage_pct`` against this floor. Returns a float
    in ``(0.0, 1.0]`` when the operator opts in; **``0.0`` (gate OFF)**
    for every other case — unset, empty, non-float, non-positive, or
    out of the ``(0.0, 1.0]`` range. A ``0.0`` return makes the coverage
    arm a byte-identical no-op (mirrors ``ED4ALL_ANSWER_PRUNE_MIN_OVERLAP``
    / ``ED4ALL_TO_BACKLINK_FLOOR`` parse-with-fallback posture).
    """
    raw = os.environ.get("ED4ALL_CHUNK_COVERAGE_FLOOR")
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not (0.0 < val <= 1.0):
        return 0.0
    return val


def resolve_min_chunks() -> int:
    """Resolve ``ED4ALL_MIN_CHUNKS`` (parse-with-fallback).

    The import-coverage gate (W1.2) compares the manifest's
    ``source_coverage.emitted_count`` against this floor. Returns a
    positive int when the operator opts in; **``0`` (gate OFF)** for
    every other case — unset, empty, non-int, or non-positive. A ``0``
    return makes the thin-chunkset arm a byte-identical no-op (mirrors
    ``ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE`` parse-with-fallback
    posture).
    """
    raw = os.environ.get("ED4ALL_MIN_CHUNKS")
    if raw is None:
        return 0
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 0
    if val <= 0:
        return 0
    return val


def _resolve_schema_path() -> Optional[Path]:
    """Locate ``schemas/library/chunkset_manifest.schema.json``.

    Walks up from this file until it finds a ``schemas/library`` dir.
    Returns ``None`` when not found (validator falls back to structural
    check).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = (
            parent / "schemas" / "library" / "chunkset_manifest.schema.json"
        )
        if candidate.exists():
            return candidate
    return None


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file. The chunks.jsonl emit is
    one JSON object per line; trailing empty lines (e.g. final newline)
    don't count toward chunks_count."""
    count = 0
    with path.open("rb") as fh:
        for raw in fh:
            if raw.strip():
                count += 1
    return count


def _emit_chunkset_manifest_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    metrics: Dict[str, Any],
) -> None:
    """Emit one ``chunkset_manifest_check`` decision per ``validate()`` (H3 W6b)."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    metric_strs = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
    rationale = (
        f"Chunkset manifest check verdict={decision}, "
        f"failure_code={code or 'none'}, metrics=({metric_strs})."
    )
    enriched = dict(metrics)
    enriched["passed"] = bool(passed)
    enriched["failure_code"] = code
    try:
        capture.log_decision(
            decision_type="chunkset_manifest_check",
            decision=decision,
            rationale=rationale,
            context=str(enriched),
            metrics=enriched,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on chunkset_manifest_check: %s",
            exc,
        )


class ChunksetManifestValidator:
    """Phase 7b chunkset-manifest gate.

    Validator-protocol-compatible class wired as the ``chunkset_manifest``
    gate on the new ``chunking`` phase (Worker W10's workflow YAML
    wiring). Severity warning at the gate level initially per the
    Wave 7b-2 plan; this validator emits critical-severity issues for
    schema / hash / count mismatches so the gate can promote to
    severity: critical without code changes.
    """

    name = "chunkset_manifest"
    version = "0.1.0"  # Phase 7b ST 13 PoC

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        issues: List[GateIssue] = []

        # ---- 1. manifest path required + exists.
        path_raw = inputs.get("chunkset_manifest_path")
        if not path_raw:
            _emit_chunkset_manifest_decision(
                capture,
                passed=False,
                code="CHUNKSET_MANIFEST_MISSING_INPUT",
                metrics={
                    "manifest_path_str": None,
                    "manifest_present": False,
                    "issue_count": 1,
                },
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_MANIFEST_MISSING_INPUT",
                        message=(
                            "ChunksetManifestValidator requires "
                            "inputs['chunkset_manifest_path']."
                        ),
                    )
                ],
                action="block",
            )

        manifest_path = Path(path_raw)
        if not manifest_path.exists() or not manifest_path.is_file():
            _emit_chunkset_manifest_decision(
                capture,
                passed=False,
                code="CHUNKSET_MANIFEST_NOT_FOUND",
                metrics={
                    "manifest_path_str": str(manifest_path),
                    "manifest_present": False,
                    "issue_count": 1,
                },
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_MANIFEST_NOT_FOUND",
                        message=(
                            f"chunkset manifest not found at {manifest_path}"
                        ),
                        location=str(manifest_path),
                    )
                ],
                action="block",
            )

        # ---- 2. JSON parses.
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _emit_chunkset_manifest_decision(
                capture,
                passed=False,
                code="CHUNKSET_MANIFEST_INVALID_JSON",
                metrics={
                    "manifest_path_str": str(manifest_path),
                    "manifest_present": True,
                    "json_parse_error": exc.__class__.__name__,
                    "issue_count": 1,
                },
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_MANIFEST_INVALID_JSON",
                        message=(
                            f"chunkset manifest failed to parse: "
                            f"{exc.__class__.__name__}: {exc}"
                        ),
                        location=str(manifest_path),
                    )
                ],
                action="block",
            )

        if not isinstance(manifest, dict):
            _emit_chunkset_manifest_decision(
                capture,
                passed=False,
                code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                metrics={
                    "manifest_path_str": str(manifest_path),
                    "manifest_present": True,
                    "manifest_root_type": type(manifest).__name__,
                    "issue_count": 1,
                },
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                        message=(
                            f"chunkset manifest root is not a JSON object "
                            f"(got {type(manifest).__name__})."
                        ),
                        location=str(manifest_path),
                    )
                ],
                action="block",
            )

        # ---- 3. Schema validation (canonical: jsonschema; fallback: structural).
        schema_issues = self._validate_against_schema(manifest, manifest_path)
        issues.extend(schema_issues)

        # When schema validation found critical issues that prevent
        # downstream checks from being meaningful (missing chunks_sha256,
        # invalid chunkset_kind), short-circuit. Hash/count comparisons
        # against missing fields are noise.
        critical_so_far = [i for i in issues if i.severity == "critical"]
        if critical_so_far and (
            manifest.get("chunks_sha256") is None
            or manifest.get("chunkset_kind") not in _ALLOWED_CHUNKSET_KINDS
        ):
            _emit_chunkset_manifest_decision(
                capture,
                passed=False,
                code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                metrics={
                    "manifest_path_str": str(manifest_path),
                    "manifest_present": True,
                    "schema_critical_count": len(critical_so_far),
                    "issue_count": len(issues),
                },
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=issues,
                action="block",
            )

        # ---- 4. Sibling chunks.jsonl exists.
        chunks_path = manifest_path.parent / "chunks.jsonl"
        if not chunks_path.exists() or not chunks_path.is_file():
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNKSET_CHUNKS_NOT_FOUND",
                    message=(
                        f"sibling chunks.jsonl not found at {chunks_path}; "
                        f"the chunkset manifest is orphaned."
                    ),
                    location=str(chunks_path),
                    suggestion=(
                        "Re-emit the chunkset (run the `chunking` phase) "
                        "or remove the orphaned manifest.json."
                    ),
                )
            )
            _emit_chunkset_manifest_decision(
                capture,
                passed=False,
                code="CHUNKSET_CHUNKS_NOT_FOUND",
                metrics={
                    "manifest_path_str": str(manifest_path),
                    "manifest_present": True,
                    "chunks_jsonl_present": False,
                    "issue_count": len(issues),
                },
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=issues,
                action="block",
            )

        # ---- 5. Recompute SHA-256 of chunks.jsonl + compare.
        manifest_sha = manifest.get("chunks_sha256")
        if isinstance(manifest_sha, str) and _SHA256_RE.match(manifest_sha):
            try:
                actual_sha = sha256_file(chunks_path)
            except OSError as exc:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_CHUNKS_READ_ERROR",
                        message=(
                            f"failed to recompute SHA-256 for {chunks_path}: "
                            f"{exc}"
                        ),
                        location=str(chunks_path),
                    )
                )
            else:
                if actual_sha != manifest_sha:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="CHUNKSET_HASH_MISMATCH",
                            message=(
                                f"chunks_sha256 in manifest "
                                f"({manifest_sha[:16]}...) does not match "
                                f"on-disk chunks.jsonl "
                                f"({actual_sha[:16]}...)."
                            ),
                            location=str(chunks_path),
                            suggestion=(
                                "Re-run the chunking phase or update the "
                                "manifest's chunks_sha256 to match the "
                                "current on-disk file."
                            ),
                        )
                    )

        # ---- 6. chunks_count cross-check (when present).
        manifest_count = manifest.get("chunks_count")
        if isinstance(manifest_count, int) and manifest_count >= 0:
            try:
                actual_count = _count_jsonl_lines(chunks_path)
            except OSError as exc:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_CHUNKS_READ_ERROR",
                        message=(
                            f"failed to count lines in {chunks_path}: {exc}"
                        ),
                        location=str(chunks_path),
                    )
                )
            else:
                if actual_count != manifest_count:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="CHUNKSET_COUNT_MISMATCH",
                            message=(
                                f"chunks_count in manifest "
                                f"({manifest_count}) does not match line "
                                f"count of chunks.jsonl ({actual_count})."
                            ),
                            location=str(chunks_path),
                            suggestion=(
                                "Re-run the chunking phase or update the "
                                "manifest's chunks_count to match the "
                                "actual line count."
                            ),
                        )
                    )

        # ---- 7. Import-coverage gate (W1.2). Warning-day-1 with a
        # deferred critical-flip; a strict no-op unless the operator
        # opts in via ED4ALL_CHUNK_COVERAGE_FLOOR / ED4ALL_MIN_CHUNKS.
        coverage_floor = resolve_coverage_floor()
        min_chunks = resolve_min_chunks()
        coverage_issues, coverage_metrics = self._check_import_coverage(
            manifest,
            coverage_floor=coverage_floor,
            min_chunks=min_chunks,
        )
        issues.extend(coverage_issues)

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        action: Optional[str] = "block" if not passed else None

        # Score: 1.0 when no issues; degrades by 0.1 per issue with a
        # floor at 0.0. Mirrors the convention used by
        # LibV2ManifestValidator and ContentStructureValidator.
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        first_critical_code = next(
            (i.code for i in issues if i.severity == "critical"), None
        )
        emit_metrics: Dict[str, Any] = {
            "manifest_path_str": str(manifest_path),
            "manifest_present": True,
            "chunks_jsonl_present": True,
            "score": float(round(score, 4)),
            "critical_count": int(critical_count),
            "issue_count": len(issues),
            "chunkset_kind": manifest.get("chunkset_kind"),
            "chunks_count_declared": manifest.get("chunks_count"),
        }
        emit_metrics.update(coverage_metrics)
        _emit_chunkset_manifest_decision(
            capture,
            passed=passed,
            code=first_critical_code,
            metrics=emit_metrics,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _check_import_coverage(
        manifest: Dict[str, Any],
        *,
        coverage_floor: float,
        min_chunks: int,
    ) -> Tuple[List[GateIssue], Dict[str, Any]]:
        """W1.2 import-coverage gate.

        Reads the manifest's ``source_coverage`` telemetry (canonical
        shape: ``lib/governance/source_coverage.py::build_source_coverage``)
        and fires WARNING issues when the recorded coverage / emitted
        chunk count falls below the operator-supplied floors:

        * ``CHUNKSET_COVERAGE_LOW`` — ``source_coverage.coverage_pct``
          below ``ED4ALL_CHUNK_COVERAGE_FLOOR`` (a chunkset that dropped
          too much of its upstream source).
        * ``CHUNKSET_TOO_THIN`` — ``source_coverage.emitted_count`` below
          ``ED4ALL_MIN_CHUNKS`` (a chunkset with too few chunks to be a
          useful retrieval corpus).

        Both are warning-day-1 (deferred critical-flip, calibrated once
        the floors are measured against >=2 corpora). When BOTH floors
        are off (``0.0`` / ``0``) the method is a byte-identical no-op:
        returns ``([], {})`` before touching the manifest. When on but
        ``source_coverage`` is absent / malformed / the fields are the
        wrong type, it degrades gracefully (no issue — pre-W3.H manifests
        legitimately lack the field).

        Returns ``(issues, metrics)`` where ``metrics`` feeds the
        ``chunkset_manifest_check`` decision-capture emit.
        """
        if coverage_floor <= 0.0 and min_chunks <= 0:
            # Both arms off — strict no-op (no metrics, no issues).
            return [], {}

        issues: List[GateIssue] = []
        metrics: Dict[str, Any] = {
            "coverage_floor": float(coverage_floor),
            "min_chunks_floor": int(min_chunks),
            "source_coverage_present": False,
            "coverage_pct_observed": None,
            "emitted_count_observed": None,
            "coverage_low": False,
            "too_thin": False,
        }

        source_coverage = manifest.get("source_coverage")
        if not isinstance(source_coverage, dict):
            # Optional field absent (pre-W3.H manifest) — nothing to gate.
            return issues, metrics
        metrics["source_coverage_present"] = True

        # ---- coverage_pct arm.
        if coverage_floor > 0.0:
            cov = source_coverage.get("coverage_pct")
            if isinstance(cov, (int, float)) and not isinstance(cov, bool):
                cov_f = float(cov)
                metrics["coverage_pct_observed"] = cov_f
                if cov_f < coverage_floor:
                    metrics["coverage_low"] = True
                    issues.append(
                        GateIssue(
                            # TODO(calibration): flip to critical once the
                            # coverage floor is calibrated against >=2 corpora.
                            severity="warning",
                            code="CHUNKSET_COVERAGE_LOW",
                            message=(
                                f"chunkset source coverage "
                                f"{cov_f:.4f} is below the configured floor "
                                f"{coverage_floor:.4f} "
                                f"(ED4ALL_CHUNK_COVERAGE_FLOOR); too much "
                                f"upstream source was dropped pre-emit."
                            ),
                            suggestion=(
                                "Inspect source_coverage.drop_reasons to see "
                                "why blocks were dropped, or lower "
                                "ED4ALL_CHUNK_COVERAGE_FLOOR if the drops are "
                                "expected (boilerplate/image-only)."
                            ),
                        )
                    )

        # ---- emitted_count arm.
        if min_chunks > 0:
            emitted = source_coverage.get("emitted_count")
            if isinstance(emitted, int) and not isinstance(emitted, bool):
                metrics["emitted_count_observed"] = int(emitted)
                if emitted < min_chunks:
                    metrics["too_thin"] = True
                    issues.append(
                        GateIssue(
                            # TODO(calibration): flip to critical once the
                            # min-chunk floor is calibrated against >=2 corpora.
                            severity="warning",
                            code="CHUNKSET_TOO_THIN",
                            message=(
                                f"chunkset emitted only {emitted} chunk(s), "
                                f"below the configured minimum {min_chunks} "
                                f"(ED4ALL_MIN_CHUNKS); the corpus may be too "
                                f"thin to support retrieval."
                            ),
                            suggestion=(
                                "Check that the upstream source parsed "
                                "correctly (a collapsed/empty structure emits "
                                "few chunks), or lower ED4ALL_MIN_CHUNKS."
                            ),
                        )
                    )

        return issues, metrics

    @staticmethod
    def _validate_against_schema(
        manifest: Dict[str, Any], manifest_path: Path,
    ) -> List[GateIssue]:
        """Validate the manifest dict against the canonical schema.

        Best-effort: when jsonschema isn't installed, falls back to a
        lightweight structural check covering the schema's required
        fields, the chunkset_kind enum, the SHA-256 + chunker_version
        regexes, and the conditional source_*_sha256 requirement.
        """
        issues: List[GateIssue] = []

        try:
            import jsonschema  # type: ignore
        except ImportError:
            return ChunksetManifestValidator._structural_fallback(
                manifest, manifest_path,
            )

        schema_path = _resolve_schema_path()
        if not schema_path or not schema_path.exists():
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNKSET_MANIFEST_SCHEMA_UNAVAILABLE",
                    message=(
                        f"chunkset_manifest.schema.json not found "
                        f"(searched upward from {Path(__file__).resolve()}); "
                        f"falling back to structural check."
                    ),
                )
            )
            issues.extend(
                ChunksetManifestValidator._structural_fallback(
                    manifest, manifest_path,
                )
            )
            return issues

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNKSET_MANIFEST_SCHEMA_LOAD_ERROR",
                    message=(
                        f"failed to load chunkset manifest schema "
                        f"at {schema_path}: {exc}"
                    ),
                )
            )
            issues.extend(
                ChunksetManifestValidator._structural_fallback(
                    manifest, manifest_path,
                )
            )
            return issues

        validator_cls = jsonschema.Draft7Validator
        validator = validator_cls(schema)
        for err in validator.iter_errors(manifest):
            location = ".".join(str(p) for p in err.absolute_path) or "<root>"
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                    message=f"schema check: {err.message}",
                    location=location,
                    suggestion=(
                        "See schemas/library/chunkset_manifest.schema.json "
                        "for the canonical contract."
                    ),
                )
            )
        return issues

    @staticmethod
    def _structural_fallback(
        manifest: Dict[str, Any], manifest_path: Path,
    ) -> List[GateIssue]:
        """Manual structural check covering the schema's load-bearing
        clauses. Used both when ``jsonschema`` isn't installed and as a
        belt-and-braces companion when the canonical schema file isn't
        on disk.
        """
        issues: List[GateIssue] = []

        # Required base fields.
        for required in _BASE_REQUIRED_FIELDS:
            if required not in manifest:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                        message=(
                            f"missing required field: {required!r}"
                        ),
                        location=str(manifest_path),
                    )
                )

        # additionalProperties: false enforcement.
        for key in manifest.keys():
            if key not in _KNOWN_FIELDS:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                        message=(
                            f"unknown top-level field: {key!r} "
                            f"(additionalProperties: false)"
                        ),
                        location=str(manifest_path),
                    )
                )

        # chunks_sha256: 64-char lowercase hex.
        chunks_sha = manifest.get("chunks_sha256")
        if chunks_sha is not None and not (
            isinstance(chunks_sha, str) and _SHA256_RE.match(chunks_sha)
        ):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                    message=(
                        f"chunks_sha256 must be a 64-char lowercase hex "
                        f"string; got: {chunks_sha!r}"
                    ),
                    location=str(manifest_path),
                )
            )

        # chunker_version: semver-ish.
        chunker_ver = manifest.get("chunker_version")
        if chunker_ver is not None and not (
            isinstance(chunker_ver, str)
            and _CHUNKER_VERSION_RE.match(chunker_ver)
        ):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                    message=(
                        f"chunker_version must match either the chunker-"
                        f"schema-contract pattern (e.g. 'v4') or the "
                        f"legacy semver pattern (e.g. '0.1.0', "
                        f"'0.0.0+missing'); got: {chunker_ver!r}"
                    ),
                    location=str(manifest_path),
                )
            )

        # chunkset_kind: enum.
        kind = manifest.get("chunkset_kind")
        if kind is not None and kind not in _ALLOWED_CHUNKSET_KINDS:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                    message=(
                        f"chunkset_kind must be one of "
                        f"{sorted(_ALLOWED_CHUNKSET_KINDS)}; got: {kind!r}"
                    ),
                    location=str(manifest_path),
                )
            )

        # Conditional source-SHA requirement (driven by chunkset_kind).
        if kind in _CONDITIONAL_SOURCE_FIELD:
            required_field = _CONDITIONAL_SOURCE_FIELD[kind]
            if required_field not in manifest:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                        message=(
                            f"chunkset_kind={kind!r} requires "
                            f"{required_field!r} to be present."
                        ),
                        location=str(manifest_path),
                    )
                )
            else:
                source_sha = manifest[required_field]
                if not (
                    isinstance(source_sha, str)
                    and _SHA256_RE.match(source_sha)
                ):
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                            message=(
                                f"{required_field} must be a 64-char "
                                f"lowercase hex string; got: {source_sha!r}"
                            ),
                            location=str(manifest_path),
                        )
                    )

        # chunks_count: non-negative integer when present.
        count = manifest.get("chunks_count")
        if count is not None and not (
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
        ):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNKSET_MANIFEST_SCHEMA_VIOLATION",
                    message=(
                        f"chunks_count must be a non-negative integer; "
                        f"got: {count!r}"
                    ),
                    location=str(manifest_path),
                )
            )

        return issues


__all__ = [
    "ChunksetManifestValidator",
    "resolve_coverage_floor",
    "resolve_min_chunks",
]
