"""LibV2 archive *completeness* validator ("true full course" gate).

Gates ``textbook_to_course::libv2_archival`` at WARNING severity day-1
(``# TODO(calibration)`` deferred critical-flip) with an opt-in strict
mode (``ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE``) that flips the incomplete-
archive issues to critical / blocking. Mirrors the W2.3
``ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES`` posture: default OFF →
parse-with-fallback → warning-only (non-blocking, byte-identical control
flow); only the explicit truthy tokens flip it strict.

The gate catches the incomplete-archive shapes that have historically
polluted ``LibV2/courses/`` (empty split-brain twins, timestamped
skeletons, 2-file slices, degenerate imports, truncated/fake indexes)
WITHOUT wrongly failing the two archive shapes the operator keeps:

* a **full generated course** (chunks + full scaffolding), and
* a **chunk-only retrieval import** (chunks + a real vector index +
  manifest, count ≥ floor).

Both PASS clean. Neither objectives nor content-pages are required (that
would wrongly fail a legitimate chunk-only import). Only a genuinely
broken/incomplete archive is flagged:

* ``ARCHIVE_NO_CHUNKS``      — no ``chunks.jsonl`` / 0 chunks (empties, shells).
* ``ARCHIVE_TOO_THIN``       — chunk count below the min floor (degenerate imports).
* ``ARCHIVE_NO_INDEX``       — chunks present but no ``vector_index/embeddings.npy``
                               (2-file slices / fragments).
* ``ARCHIVE_INDEX_MISMATCH`` — index vector count != chunk count (truncated index).
* ``ARCHIVE_FAKE_INDEX``     — ``embedding_provider == 'fake'`` and
                               ``ED4ALL_EMBEDDING_ALLOW_FAKE`` not set
                               (reuses the existing anti-poisoning gate).

PHASE-ORDER CAVEAT: on the ``textbook_to_course`` workflow the optional
``vector_indexing`` phase runs AFTER ``libv2_archival``, so a fresh
full-course archive legitimately has no ``vector_index/`` yet at gate
time — ``ARCHIVE_NO_INDEX`` will surface as a (non-blocking) WARNING on
that path by default. That is acceptable day-1 noise for the default
warning posture; an operator enabling the opt-in strict mode is
explicitly demanding a fully retrieval-ready archive (re-validate after
indexing, or run it as a standalone completeness check on an existing
course dir). No new gate ever blocks the default run.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Archived chunksets under a LibV2 course dir (either shape is a valid
# archive; a chunk-only import carries exactly one).
_CHUNKSET_RELPATHS = (
    ("imscc", "imscc_chunks/chunks.jsonl"),
    ("dart", "dart_chunks/chunks.jsonl"),
)

_VECTOR_INDEX_DIRNAME = "vector_index"
_EMBEDDINGS_FILENAME = "embeddings.npy"
_ID_MAP_FILENAME = "id_map.json"
_INDEX_MANIFEST_FILENAME = "manifest.json"

# Default minimum chunk floor for THIS gate. Distinct from the W1.2
# ``resolve_min_chunks`` default (0 / OFF): a degenerate import is a
# defect we WANT to flag, so this gate defaults the floor positive.
_DEFAULT_MIN_CHUNKS = 20

_TRUTHY = {"1", "true", "yes", "on"}


def resolve_require_full_course(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE`` (parse-with-fallback).

    Default OFF (warning-only, non-blocking). Only the explicit truthy
    tokens flip it on; garbage / falsey / unset → OFF. Mirrors the W2.3
    ``ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES`` posture exactly.
    """
    src = os.environ if env is None else env
    return src.get("ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE", "").strip().lower() in _TRUTHY


def resolve_archive_min_chunks(
    gate_config: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Resolve the min-chunk floor (gate config > ``ED4ALL_MIN_CHUNKS`` > default).

    Reuses the W1.2 ``ED4ALL_MIN_CHUNKS`` env knob as an operator
    override, but — unlike ``lib.validators.chunkset_manifest.
    resolve_min_chunks`` (which defaults 0 / OFF) — this gate defaults the
    floor to ``_DEFAULT_MIN_CHUNKS`` (20) so a degenerate import is
    flagged out of the box. Parse-with-fallback: garbage / non-positive →
    the default.
    """
    # 1. Explicit per-gate threshold wins.
    if gate_config:
        raw = gate_config.get("min_chunks")
        if raw is not None:
            try:
                val = int(raw)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
    # 2. ED4ALL_MIN_CHUNKS env override (reused W1.2 floor).
    src = os.environ if env is None else env
    raw_env = src.get("ED4ALL_MIN_CHUNKS")
    if raw_env is not None:
        try:
            val = int(raw_env)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    # 3. Gate default.
    return _DEFAULT_MIN_CHUNKS


def _count_jsonl_records(path: Path) -> int:
    """Count non-empty records in a JSONL file (cheap; no full parse/load).

    Counts non-blank lines that begin a JSON object/array token. Avoids
    materialising the (potentially very large) chunkset in memory — this
    is a metadata count, not a content read.
    """
    count = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    count += 1
    except OSError as exc:  # noqa: BLE001
        logger.warning("Failed to read chunkset %s: %s", path, exc)
        return 0
    return count


def _resolve_course_dir(inputs: Dict[str, Any]) -> Optional[Path]:
    """Resolve the archived course dir from ``course_dir`` or ``manifest_path``."""
    raw = inputs.get("course_dir") or inputs.get("manifest_path")
    if not raw:
        return None
    try:
        cd = Path(raw)
    except (TypeError, ValueError):
        return None
    if cd.is_file():  # a manifest_path was threaded — use its parent
        cd = cd.parent
    return cd


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    strict: bool,
    metrics: Dict[str, Any],
    codes: List[str],
) -> None:
    """Emit one ``validation_result`` decision per ``validate()`` (reused enum)."""
    if capture is None:
        return
    metric_strs = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
    verdict = "passed" if passed else "failed"
    rationale = (
        f"course_completeness gate verdict={verdict} (strict={strict}); "
        f"codes={codes or ['none']}; metrics=({metric_strs})."
    )
    enriched = dict(metrics)
    enriched["passed"] = bool(passed)
    enriched["strict"] = bool(strict)
    enriched["codes"] = list(codes)
    try:
        capture.log_decision(
            decision_type="validation_result",
            decision=f"{verdict}:{','.join(codes) or 'complete'}",
            rationale=rationale,
            context=str(enriched),
            metrics=enriched,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("DecisionCapture.log_decision raised on course_completeness: %s", exc)


class CourseCompletenessValidator:
    """WARNING-day-1 archival-completeness gate (opt-in strict mode)."""

    name = "course_completeness"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "course_completeness")
        capture = inputs.get("decision_capture")
        gate_config = inputs.get("_gate_config", {}) or {}

        strict = resolve_require_full_course()
        min_chunks = resolve_archive_min_chunks(gate_config)
        # In strict mode incompleteness blocks (critical); default is a
        # non-blocking warning (byte-identical control flow when off).
        sev = "critical" if strict else "warning"

        issues: List[GateIssue] = []
        codes: List[str] = []

        course_dir = _resolve_course_dir(inputs)
        if course_dir is None or not course_dir.exists():
            # No course dir resolvable → we cannot audit. Info-only, never
            # blocks (fail-safe: absent input is a routing gap, not a
            # broken archive).
            issues.append(GateIssue(
                severity="info",
                code="COURSE_DIR_UNRESOLVED",
                message=(
                    "course_completeness could not resolve an archived course_dir "
                    "from inputs; skipping completeness audit."
                ),
            ))
            _emit_decision(
                capture, passed=True, strict=strict,
                metrics={"course_dir": str(course_dir) if course_dir else None,
                         "min_chunks": min_chunks},
                codes=[],
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=issues,
                metadata={"strict": strict, "min_chunks": min_chunks,
                          "course_dir_resolved": False},
            )

        # -- 1. Chunk presence + counts (per chunkset). ------------------ #
        chunk_counts: Dict[str, int] = {}
        for kind, rel in _CHUNKSET_RELPATHS:
            path = course_dir / rel
            if path.is_file():
                chunk_counts[kind] = _count_jsonl_records(path)
        total_chunks = sum(chunk_counts.values())
        max_chunks = max(chunk_counts.values()) if chunk_counts else 0

        if total_chunks == 0:
            issues.append(GateIssue(
                severity=sev,
                code="ARCHIVE_NO_CHUNKS",
                message=(
                    f"Archived course {course_dir.name!r} carries no chunks "
                    "(no chunks.jsonl or 0 records) — an empty shell, not a "
                    "retrieval-ready archive."
                ),
                location=str(course_dir),
                suggestion="Re-run the chunking phase; do not archive empty courses.",
            ))
            codes.append("ARCHIVE_NO_CHUNKS")
        elif max_chunks < min_chunks:
            issues.append(GateIssue(
                severity=sev,
                code="ARCHIVE_TOO_THIN",
                message=(
                    f"Archived course {course_dir.name!r} has only {max_chunks} "
                    f"chunk(s), below the min floor {min_chunks} "
                    "(ED4ALL_MIN_CHUNKS) — a degenerate import/slice."
                ),
                location=str(course_dir),
                suggestion=(
                    "Import the full corpus, or lower ED4ALL_MIN_CHUNKS if the "
                    "course is intentionally tiny."
                ),
            ))
            codes.append("ARCHIVE_TOO_THIN")

        # -- 2. Vector-index presence + integrity. ----------------------- #
        # Only audited when chunks exist (an empty course is already
        # flagged NO_CHUNKS; don't double-flag it as NO_INDEX).
        index_dir = course_dir / _VECTOR_INDEX_DIRNAME
        embeddings_path = index_dir / _EMBEDDINGS_FILENAME
        index_present = embeddings_path.is_file()
        index_vector_count: Optional[int] = None
        embedding_provider: Optional[str] = None

        if total_chunks > 0:
            if not index_present:
                issues.append(GateIssue(
                    severity=sev,
                    code="ARCHIVE_NO_INDEX",
                    message=(
                        f"Archived course {course_dir.name!r} has chunks but no "
                        f"vector index ({_VECTOR_INDEX_DIRNAME}/{_EMBEDDINGS_FILENAME}) "
                        "— a slice/fragment, not retrieval-ready. (On the "
                        "textbook_to_course path the index is built by the "
                        "vector_indexing phase AFTER archival — see module docstring.)"
                    ),
                    location=str(index_dir),
                    suggestion="Build the vector index (libv2 vector-index build).",
                ))
                codes.append("ARCHIVE_NO_INDEX")
            else:
                mismatch_code, provider, vec_count = self._audit_index(
                    index_dir, chunk_counts, max_chunks, sev, issues,
                )
                embedding_provider = provider
                index_vector_count = vec_count
                if mismatch_code:
                    codes.append(mismatch_code)

        critical_count = sum(1 for i in issues if i.severity == "critical")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        passed = critical_count == 0

        metrics = {
            "course_dir": str(course_dir),
            "total_chunks": total_chunks,
            "max_chunks": max_chunks,
            "chunksets": ",".join(sorted(chunk_counts)) or "none",
            "min_chunks": min_chunks,
            "index_present": index_present,
            "index_vector_count": index_vector_count,
            "embedding_provider": embedding_provider,
            "critical_count": critical_count,
            "warning_count": warning_count,
        }
        _emit_decision(capture, passed=passed, strict=strict,
                       metrics=metrics, codes=codes)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            issues=issues,
            metadata={
                "strict": strict,
                "min_chunks": min_chunks,
                "total_chunks": total_chunks,
                "max_chunks": max_chunks,
                "index_present": index_present,
                "index_vector_count": index_vector_count,
                "embedding_provider": embedding_provider,
                "codes": codes,
            },
        )

    # ------------------------------------------------------------------ #
    # Vector-index integrity                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _audit_index(
        index_dir: Path,
        chunk_counts: Dict[str, int],
        max_chunks: int,
        sev: str,
        issues: List[GateIssue],
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Audit a present index for fake-provider + count-mismatch defects.

        Returns ``(mismatch_or_fake_code, embedding_provider, index_vector_count)``.
        Only the FIRST index-defect code is returned for the decision
        capture; every triggered issue is appended to ``issues``.
        """
        first_code: Optional[str] = None
        provider: Optional[str] = None

        # --- Provider (anti-poisoning fake check) ---------------------- #
        manifest_path = index_dir / _INDEX_MANIFEST_FILENAME
        chunkset_kind: Optional[str] = None
        if manifest_path.is_file():
            try:
                man = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(man, dict):
                    provider = man.get("embedding_provider")
                    ck = man.get("chunkset_kind")
                    if isinstance(ck, str):
                        chunkset_kind = ck
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to read index manifest %s: %s", manifest_path, exc)

        if provider == "fake":
            # Reuse the existing anti-poisoning opt-in gate.
            allow_fake = False
            try:
                from lib.embedding.providers import allow_fake_enabled
                allow_fake = allow_fake_enabled()
            except Exception:  # noqa: BLE001
                allow_fake = str(
                    os.environ.get("ED4ALL_EMBEDDING_ALLOW_FAKE", "")
                ).strip().lower() in _TRUTHY
            if not allow_fake:
                issues.append(GateIssue(
                    severity=sev,
                    code="ARCHIVE_FAKE_INDEX",
                    message=(
                        f"Vector index at {index_dir} was built with the 'fake' "
                        "embedding provider (deterministic test vectors) and "
                        "ED4ALL_EMBEDDING_ALLOW_FAKE is not set — not a real "
                        "retrieval index."
                    ),
                    location=str(manifest_path),
                    suggestion="Rebuild with a real provider (st / local-openai).",
                ))
                first_code = first_code or "ARCHIVE_FAKE_INDEX"

        # --- Vector-count vs chunk-count (truncated index) ------------- #
        id_map_path = index_dir / _ID_MAP_FILENAME
        index_vector_count: Optional[int] = None
        if id_map_path.is_file():
            try:
                doc = json.loads(id_map_path.read_text(encoding="utf-8"))
                if isinstance(doc, dict):
                    ids = doc.get("chunk_ids", [])
                elif isinstance(doc, list):
                    ids = doc
                else:
                    ids = []
                index_vector_count = len(ids)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to read id_map %s: %s", id_map_path, exc)

        if index_vector_count is not None:
            # Prefer the chunkset the index was built from (manifest
            # chunkset_kind); fall back to the largest present chunkset.
            if chunkset_kind in chunk_counts:
                target = chunk_counts[chunkset_kind]
            else:
                target = max_chunks
            # A truncated/stale index has a row count that disagrees with
            # its source chunkset. Tolerate an exact match against ANY
            # present chunkset (covers the no-chunkset_kind ambiguity).
            if index_vector_count != target and index_vector_count not in set(
                chunk_counts.values()
            ):
                issues.append(GateIssue(
                    severity=sev,
                    code="ARCHIVE_INDEX_MISMATCH",
                    message=(
                        f"Vector index at {index_dir} has {index_vector_count} "
                        f"vector(s) but the source chunkset has {target} chunk(s) "
                        "— a truncated / stale index."
                    ),
                    location=str(index_dir),
                    suggestion="Rebuild the vector index (libv2 vector-index build --force).",
                ))
                first_code = first_code or "ARCHIVE_INDEX_MISMATCH"

        return first_code, provider, index_vector_count
