"""GPT Feedback v2 Wave 3 W3.D — ChunksetDriftValidator.

Detects content drift between the DART evidence chunkset
(``LibV2/courses/<slug>/dart_chunks/chunks.jsonl``) and the IMSCC
packaged chunkset (``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl``).

Two complementary drift axes are tracked:

* **DART_EVIDENCE_DROPPED** (``drift_rate_out``): fraction of DART
  block IDs that the IMSCC chunkset never cited via
  ``source.source_references[].sourceId``. Surfaces the failure mode
  "the packager dropped half the textbook" — IMSCC restructuring may
  legitimately collapse multiple DART blocks into a single page, but a
  high drop rate signals lost evidence.
* **IMSCC_CLAIM_NOT_IN_DART** (``drift_rate_in``): fraction of IMSCC
  chunks that have an empty ``source.source_references[]`` list.
  Surfaces the failure mode "we authored unattributed prose" — every
  packaged claim should trace back to a DART source block.

Sidecar artifact (``<course_path>/drift_report.json``) is written
best-effort per the schema at
``schemas/library/chunkset_drift_report.schema.json``. Operator-facing
metric surface that future Wave 4+ work can promote to a critical
gate; today's gate severity is **warning**, fail behaviour is
**warn** (legitimate IMSCC restructuring synthesizes prose across DART
blocks; default is observability, not enforcement).

Decision capture: one ``chunkset_drift_check`` event per ``validate()``
call carrying the six load-bearing metrics
(``drift_rate_in``, ``drift_rate_out``, ``dropped_count``,
``unsourced_count``, ``total_dart_chunks``, ``total_imscc_chunks``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Threshold constants
# --------------------------------------------------------------------------- #

# Per Worker W3.D plan §"Failure codes". These are floors; a future
# Wave 4 promotion to critical may tighten them. Single source of
# truth lives here so the test suite + sidecar emit + decision
# rationale all read the same numbers.
_DRIFT_RATE_IN_WARN_THRESHOLD: float = 0.05
_DRIFT_RATE_IN_HIGH_THRESHOLD: float = 0.20
_DRIFT_RATE_OUT_WARN_THRESHOLD: float = 0.50

_SCHEMA_VERSION: str = "1.0"


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #


def _load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Stream the chunks.jsonl file into a list of dicts.

    Malformed lines are skipped with a debug-level log; the validator
    treats unreadable chunks as "absent" for the drift computation
    rather than failing the entire gate. Missing file → returns ``[]``
    (caller is responsible for the MISSING_CHUNKSET signal).
    """
    chunks: List[Dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return chunks
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "ChunksetDriftValidator: skipping malformed JSON "
                        "at %s:%d (%s)",
                        path,
                        line_no,
                        exc,
                    )
                    continue
                if isinstance(obj, dict):
                    chunks.append(obj)
    except OSError as exc:
        logger.warning(
            "ChunksetDriftValidator: failed to read %s (%s); treating as empty",
            path,
            exc,
        )
    return chunks


def _iter_source_ref_ids(chunk: Dict[str, Any]) -> Iterable[str]:
    """Yield each ``source.source_references[].sourceId`` for a chunk.

    Defensive against legacy chunks where ``source`` may be missing or
    where ``source_references`` is absent / non-list. Yields nothing in
    those cases (counts as "unsourced" via the empty-list path).
    """
    source = chunk.get("source")
    if not isinstance(source, dict):
        return
    refs = source.get("source_references")
    if not isinstance(refs, list):
        return
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        sid = ref.get("sourceId")
        if isinstance(sid, str) and sid:
            yield sid


def _has_source_references(chunk: Dict[str, Any]) -> bool:
    """True when the chunk carries at least one non-empty source ref."""
    for _ in _iter_source_ref_ids(chunk):
        return True
    return False


def _chunk_id(chunk: Dict[str, Any]) -> Optional[str]:
    """Resolve the chunk's unique id (canonical: ``id`` field).

    Falls back to ``chunk_id`` for legacy / fixture data. Returns
    ``None`` when neither is a non-empty string so the caller can skip
    rather than emit an empty-string identifier into the sidecar.
    """
    for key in ("id", "chunk_id"):
        val = chunk.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _collect_dart_block_ids(dart_chunks: List[Dict[str, Any]]) -> Set[str]:
    """Build the universe of DART block identifiers.

    Per W3.D plan: the union of every DART chunk's own id AND every
    ``source.source_references[].sourceId`` cited from inside that
    chunk. The union covers both id-based cross-walk identity (when
    IMSCC chunks cite DART chunks directly) and source-ref-based
    identity (when IMSCC chunks cite the upstream DART source the DART
    chunk itself was derived from).
    """
    block_ids: Set[str] = set()
    for chunk in dart_chunks:
        cid = _chunk_id(chunk)
        if cid:
            block_ids.add(cid)
        for sid in _iter_source_ref_ids(chunk):
            block_ids.add(sid)
    return block_ids


def _collect_imscc_cited_ids(imscc_chunks: List[Dict[str, Any]]) -> Set[str]:
    """Build the set of every sourceId the IMSCC chunkset cites."""
    cited: Set[str] = set()
    for chunk in imscc_chunks:
        for sid in _iter_source_ref_ids(chunk):
            cited.add(sid)
    return cited


# --------------------------------------------------------------------------- #
# Decision-capture helper
# --------------------------------------------------------------------------- #


def _emit_chunkset_drift_decision(
    capture: Any,
    *,
    passed: bool,
    failure_codes: List[str],
    metrics: Dict[str, Any],
) -> None:
    """Emit one ``chunkset_drift_check`` event per validate() call.

    Rationale interpolates the six load-bearing metrics so a regression
    test can substring-assert each one's presence.
    """
    if capture is None:
        return
    decision = "passed" if passed else "failed:" + ",".join(failure_codes or ["unknown"])
    rationale = (
        "Chunkset drift gate verdict={decision}: "
        "drift_rate_in={drift_rate_in:.4f}, "
        "drift_rate_out={drift_rate_out:.4f}, "
        "dropped_count={dropped_count}, "
        "unsourced_count={unsourced_count}, "
        "total_dart_chunks={total_dart_chunks}, "
        "total_imscc_chunks={total_imscc_chunks}."
    ).format(decision=decision, **metrics)
    enriched = dict(metrics)
    enriched["passed"] = bool(passed)
    enriched["failure_codes"] = failure_codes
    try:
        capture.log_decision(
            decision_type="chunkset_drift_check",
            decision=decision,
            rationale=rationale,
            context=str(enriched),
            metrics=enriched,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on chunkset_drift_check: %s",
            exc,
        )


# --------------------------------------------------------------------------- #
# Sidecar emit
# --------------------------------------------------------------------------- #


def _write_drift_report_sidecar(
    course_path: Path,
    *,
    dart_chunk_count: int,
    imscc_chunk_count: int,
    dart_block_ids_dropped: List[str],
    imscc_chunks_unsourced: List[str],
    drift_rate_in: float,
    drift_rate_out: float,
) -> Optional[Path]:
    """Write ``<course_path>/drift_report.json`` per the schema.

    Best-effort: any OSError logs a warning and returns ``None`` so the
    validator's gate verdict remains authoritative. Sorted lists keep
    the artifact deterministic across runs.
    """
    payload: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "dart_chunk_count": int(dart_chunk_count),
        "imscc_chunk_count": int(imscc_chunk_count),
        "dart_block_ids_dropped": sorted(dart_block_ids_dropped),
        "imscc_chunks_unsourced": sorted(imscc_chunks_unsourced),
        "drift_rate_in": float(round(drift_rate_in, 6)),
        "drift_rate_out": float(round(drift_rate_out, 6)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    target = course_path / "drift_report.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
        )
        return target
    except OSError as exc:
        logger.warning(
            "ChunksetDriftValidator: failed to write %s (%s); "
            "drift sidecar will be missing for this run.",
            target,
            exc,
        )
        return None


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class ChunksetDriftValidator:
    """Wave 3 W3.D — drift detector across DART vs. IMSCC chunksets.

    Validator-protocol-compatible class wired as the ``chunkset_drift``
    gate on the ``libv2_archival`` phase. Severity at the gate level is
    ``warning`` per the Wave 3 W3.D plan; this validator emits
    ``warning``-severity GateIssues with internally-elevated severity
    captured in the issue rationale + the drift_report.json sidecar so
    a future Wave 4 promotion to ``critical`` requires only a YAML
    flip.
    """

    name = "chunkset_drift"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        issues: List[GateIssue] = []

        # ---- 1. Required input paths.
        dart_path_raw = inputs.get("dart_chunks_path")
        imscc_path_raw = inputs.get("imscc_chunks_path")
        course_path_raw = inputs.get("course_path")

        if not dart_path_raw or not imscc_path_raw:
            metrics = self._zero_metrics()
            issues.append(
                GateIssue(
                    severity="warning",
                    code="MISSING_CHUNKSET",
                    message=(
                        "ChunksetDriftValidator requires both "
                        "inputs['dart_chunks_path'] and "
                        "inputs['imscc_chunks_path']."
                    ),
                )
            )
            _emit_chunkset_drift_decision(
                capture, passed=False, failure_codes=["MISSING_CHUNKSET"],
                metrics=metrics,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=issues,
                action="block",
            )

        dart_path = Path(dart_path_raw)
        imscc_path = Path(imscc_path_raw)
        course_path = Path(course_path_raw) if course_path_raw else None

        # ---- 2. Both chunks.jsonl files must exist (fail-closed warning).
        dart_present = dart_path.exists() and dart_path.is_file()
        imscc_present = imscc_path.exists() and imscc_path.is_file()
        if not dart_present or not imscc_present:
            metrics = self._zero_metrics()
            missing: List[str] = []
            if not dart_present:
                missing.append(str(dart_path))
            if not imscc_present:
                missing.append(str(imscc_path))
            issues.append(
                GateIssue(
                    severity="warning",
                    code="MISSING_CHUNKSET",
                    message=(
                        "chunks.jsonl missing for drift comparison: "
                        + ", ".join(missing)
                    ),
                    location=missing[0],
                    suggestion=(
                        "Re-run the chunking + imscc_chunking phases or "
                        "verify the libv2_archival staging produced both "
                        "chunksets."
                    ),
                )
            )
            _emit_chunkset_drift_decision(
                capture, passed=False, failure_codes=["MISSING_CHUNKSET"],
                metrics=metrics,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=issues,
                action="block",
            )

        # ---- 3. Load both chunksets + compute drift metrics.
        dart_chunks = _load_chunks_jsonl(dart_path)
        imscc_chunks = _load_chunks_jsonl(imscc_path)

        dart_chunk_count = len(dart_chunks)
        imscc_chunk_count = len(imscc_chunks)

        dart_block_ids = _collect_dart_block_ids(dart_chunks)
        imscc_cited_block_ids = _collect_imscc_cited_ids(imscc_chunks)

        dropped_set = dart_block_ids - imscc_cited_block_ids
        dart_block_ids_dropped: List[str] = sorted(dropped_set)

        imscc_chunks_unsourced: List[str] = []
        for chunk in imscc_chunks:
            if not _has_source_references(chunk):
                cid = _chunk_id(chunk)
                if cid:
                    imscc_chunks_unsourced.append(cid)
                else:
                    # Emit a synthetic anchor so the drift_report sidecar
                    # still records the count even when an IMSCC chunk
                    # is missing both an id and any source refs.
                    imscc_chunks_unsourced.append(
                        f"<unsourced_chunk_index_{len(imscc_chunks_unsourced)}>"
                    )

        drift_rate_in = (
            len(imscc_chunks_unsourced) / imscc_chunk_count
            if imscc_chunk_count > 0
            else 0.0
        )
        drift_rate_out = (
            len(dart_block_ids_dropped) / dart_chunk_count
            if dart_chunk_count > 0
            else 0.0
        )

        # ---- 4. Issue emission per the W3.D failure-code table.
        failure_codes: List[str] = []

        if drift_rate_in > _DRIFT_RATE_IN_WARN_THRESHOLD:
            elevated = drift_rate_in > _DRIFT_RATE_IN_HIGH_THRESHOLD
            elevated_marker = (
                "; internal severity escalated to high "
                f"(>{_DRIFT_RATE_IN_HIGH_THRESHOLD:.2f} threshold)"
                if elevated
                else ""
            )
            issues.append(
                GateIssue(
                    severity="warning",
                    code="IMSCC_CLAIM_NOT_IN_DART",
                    message=(
                        f"IMSCC chunkset has {len(imscc_chunks_unsourced)}"
                        f"/{imscc_chunk_count} chunks with no "
                        f"source.source_references[] "
                        f"(drift_rate_in={drift_rate_in:.4f}, "
                        f"warn_threshold={_DRIFT_RATE_IN_WARN_THRESHOLD:.2f}"
                        f"{elevated_marker}). "
                        "Every packaged claim should trace back to a DART "
                        "source block."
                    ),
                    location=str(imscc_path),
                    suggestion=(
                        "Inspect the unsourced chunk ids in "
                        "drift_report.json::imscc_chunks_unsourced[] and "
                        "verify the IMSCC packaging stage attaches "
                        "source.source_references[] to every authored chunk."
                    ),
                )
            )
            failure_codes.append("IMSCC_CLAIM_NOT_IN_DART")

        if drift_rate_out > _DRIFT_RATE_OUT_WARN_THRESHOLD:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="DART_EVIDENCE_DROPPED",
                    message=(
                        f"IMSCC chunkset cited only "
                        f"{len(dart_block_ids - dropped_set)}/{len(dart_block_ids)}"
                        f" DART block ids "
                        f"(drift_rate_out={drift_rate_out:.4f}, "
                        f"warn_threshold={_DRIFT_RATE_OUT_WARN_THRESHOLD:.2f})."
                        " The packager dropped more than half of the "
                        "available textbook evidence."
                    ),
                    location=str(dart_path),
                    suggestion=(
                        "Inspect drift_report.json::dart_block_ids_dropped[]"
                        " and verify the imscc_chunking phase preserves "
                        "DART source attribution. Some attrition is "
                        "expected from legitimate prose synthesis; a >50% "
                        "drop signals a packaging regression."
                    ),
                )
            )
            failure_codes.append("DART_EVIDENCE_DROPPED")

        # ---- 5. Sidecar emit (best-effort).
        if course_path is not None:
            _write_drift_report_sidecar(
                course_path,
                dart_chunk_count=dart_chunk_count,
                imscc_chunk_count=imscc_chunk_count,
                dart_block_ids_dropped=dart_block_ids_dropped,
                imscc_chunks_unsourced=imscc_chunks_unsourced,
                drift_rate_in=drift_rate_in,
                drift_rate_out=drift_rate_out,
            )
        else:
            logger.debug(
                "ChunksetDriftValidator: course_path not provided; "
                "skipping drift_report.json sidecar emit."
            )

        # ---- 6. Verdict.
        # Severity warning at the gate level → action stays None on
        # warning issues; the YAML on_fail: warn keeps the workflow
        # alive. We still set passed=False on any drift signal so the
        # phase report records the deviation.
        passed = not failure_codes
        action: Optional[str] = None  # warning gate; never blocks

        metrics = {
            "drift_rate_in": float(round(drift_rate_in, 6)),
            "drift_rate_out": float(round(drift_rate_out, 6)),
            "dropped_count": int(len(dart_block_ids_dropped)),
            "unsourced_count": int(len(imscc_chunks_unsourced)),
            "total_dart_chunks": int(dart_chunk_count),
            "total_imscc_chunks": int(imscc_chunk_count),
        }

        _emit_chunkset_drift_decision(
            capture, passed=passed, failure_codes=failure_codes,
            metrics=metrics,
        )

        # Score: 1.0 on a clean pass; halve per failure code so two
        # simultaneous drift signals score lower than one.
        score = 1.0 if passed else max(0.0, 1.0 - 0.5 * len(failure_codes))

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    @staticmethod
    def _zero_metrics() -> Dict[str, Any]:
        """Return the six-key metric shape with zero values for the
        early-return paths so the decision-capture rationale always
        interpolates the same six fields."""
        return {
            "drift_rate_in": 0.0,
            "drift_rate_out": 0.0,
            "dropped_count": 0,
            "unsourced_count": 0,
            "total_dart_chunks": 0,
            "total_imscc_chunks": 0,
        }


__all__ = [
    "ChunksetDriftValidator",
]
