"""IB4.2 — chunk-level WCAG-status gate (gates the data-only chunk fields).

``chunk_v4.schema.json`` carries ``wcag_block_status`` (enum
``passed|flagged|skipped``) + ``figure_alt`` (string), both Optional and
described as DATA-ONLY SemantiK-migration fields harvested from
``data-dart-wcag`` / ``<figcaption>``. Nothing read them as a gate. This
validator closes that gap at WARNING severity day-1:

* (a) no chunk carries ``wcag_block_status == "flagged"`` (a flagged source
  region shipped without remediation) → ``CHUNK_WCAG_FLAGGED``.
* (b) every chunk whose ``chunk_type == "figure"`` (or that carries an
  ``<img>``) has a non-empty ``figure_alt`` → ``CHUNK_FIGURE_NO_ALT``.

Legacy chunks with NEITHER field present → skip cleanly with a warning
``WCAG_FIELDS_ABSENT`` and ``passed=True`` (the byte-stable legacy-corpus path;
the fields are absent on pre-SemantiK corpora per the schema description).

Graceful, no-embedding posture (mirrors ``ChunksetManifestValidator``). Deterministic;
no LLM, no embeddings. ``passed`` is True unless a ``flagged`` chunk is present
(the deferred critical-flip moves the alt-text miss into ``passed`` too once
calibrated — see the ``# TODO(calibration)`` marker in ``config/workflows.yaml``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

# Cap per-issue list so a uniformly-flagged corpus doesn't drown the report.
_ISSUE_LIST_CAP: int = 50

_IMG_MARKERS = ("<img", "<figure")


def _emit_chunk_wcag_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    metrics: Dict[str, Any],
) -> None:
    """Emit one ``chunk_wcag_status_check`` decision per ``validate()`` (IB4.2)."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    metric_strs = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
    rationale = (
        f"Chunk-level WCAG-status gate verdict={decision}, "
        f"failure_code={code or 'none'}, metrics=({metric_strs})."
    )
    try:
        capture.log_decision(
            decision_type="chunk_wcag_status_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on chunk_wcag_status_check: %s",
            exc,
        )


def _load_chunks(inputs: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Resolve a list of chunk dicts from inputs.

    Prefers an in-memory ``inputs["chunks"]`` list; else reads the first
    present JSONL path among ``dart_chunks_path`` / ``imscc_chunks_path``.
    Returns ``None`` when no source resolves (caller emits a clean skip).
    """
    raw = inputs.get("chunks")
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]

    for key in ("dart_chunks_path", "imscc_chunks_path", "chunks_path"):
        path_raw = inputs.get(key)
        if not path_raw:
            continue
        p = Path(path_raw)
        if not p.exists() or not p.is_file():
            continue
        out: List[Dict[str, Any]] = []
        try:
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
        except OSError as exc:
            logger.debug("ChunkWcagStatusValidator could not read %s: %s", p, exc)
            continue
        return out
    return None


def _chunk_has_img(chunk: Dict[str, Any]) -> bool:
    text = chunk.get("text") or chunk.get("content") or chunk.get("html") or ""
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _IMG_MARKERS)


class ChunkWcagStatusValidator:
    """IB4.2 — gates the data-only ``wcag_block_status`` / ``figure_alt`` fields.

    Warning-severity day-1 at the ``chunking`` (DART) + ``imscc_chunking``
    (IMSCC) phases. Deterministic; no embeddings → never short-circuits on
    missing ``[embedding]`` extras.
    """

    name = "chunk_wcag_status"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        chunks = _load_chunks(inputs)
        if chunks is None:
            # No chunkset resolved — input-starved. Warning, passed=True (no
            # corpus to audit is not a defect; mirrors the legacy-absent path).
            _emit_chunk_wcag_decision(
                capture,
                passed=True,
                code="WCAG_FIELDS_ABSENT",
                metrics={"chunk_count": 0, "source": "none"},
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code="WCAG_FIELDS_ABSENT",
                    message=(
                        "No chunkset resolved (inputs['chunks'] / "
                        "dart_chunks_path / imscc_chunks_path absent); "
                        "skipping chunk-level WCAG audit."
                    ),
                )],
            )

        issues: List[GateIssue] = []
        flagged_count = 0
        figure_no_alt_count = 0
        fields_seen = False
        figure_count = 0

        for chunk in chunks:
            status = chunk.get("wcag_block_status")
            has_alt_field = "figure_alt" in chunk
            if status is not None or has_alt_field:
                fields_seen = True

            # (a) flagged → ships without remediation.
            if status == "flagged":
                flagged_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="CHUNK_WCAG_FLAGGED",
                        message=(
                            f"Chunk {chunk.get('id') or chunk.get('chunk_id') or '?'} "
                            f"carries wcag_block_status='flagged' (a flagged source "
                            f"region shipped without remediation)."
                        ),
                        location=str(chunk.get("id") or chunk.get("chunk_id") or ""),
                    ))

            # (b) figure chunk (or <img>-bearing) must carry non-empty figure_alt.
            is_figure = chunk.get("chunk_type") == "figure" or _chunk_has_img(chunk)
            if is_figure:
                figure_count += 1
                figure_alt = chunk.get("figure_alt")
                if not (isinstance(figure_alt, str) and figure_alt.strip()):
                    figure_no_alt_count += 1
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="warning",
                            code="CHUNK_FIGURE_NO_ALT",
                            message=(
                                f"Figure chunk "
                                f"{chunk.get('id') or chunk.get('chunk_id') or '?'} "
                                f"has empty/absent figure_alt (1.1.1 alt text)."
                            ),
                            location=str(chunk.get("id") or chunk.get("chunk_id") or ""),
                        ))

        # Legacy corpus — neither field present anywhere → clean skip.
        if not fields_seen:
            _emit_chunk_wcag_decision(
                capture,
                passed=True,
                code="WCAG_FIELDS_ABSENT",
                metrics={"chunk_count": len(chunks), "source": "legacy_absent"},
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code="WCAG_FIELDS_ABSENT",
                    message=(
                        f"Chunkset ({len(chunks)} chunks) carries neither "
                        f"wcag_block_status nor figure_alt (legacy / pre-SemantiK "
                        f"corpus); skipping chunk-level WCAG audit."
                    ),
                )],
            )

        # passed=True unless a flagged chunk is present (the figure-alt miss is
        # warning-only day-1; the deferred critical-flip extends this).
        passed = flagged_count == 0
        score = round(
            1.0 - (flagged_count / len(chunks)) if chunks else 1.0, 4
        )
        _emit_chunk_wcag_decision(
            capture,
            passed=passed,
            code=("CHUNK_WCAG_FLAGGED" if flagged_count else None),
            metrics={
                "chunk_count": len(chunks),
                "flagged_count": flagged_count,
                "figure_count": figure_count,
                "figure_no_alt_count": figure_no_alt_count,
                "source": "fields_present",
            },
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )


__all__ = ["ChunkWcagStatusValidator"]
