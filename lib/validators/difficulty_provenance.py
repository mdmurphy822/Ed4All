"""Anti-fabrication validator for the IRT difficulty-calibration scaffold.

Gates ``textbook_to_course::libv2_archival`` at WARNING severity day-1
(``# TODO(calibration)`` deferred critical-flip). When the scaffold is on
(``TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD``) it asserts the honest-scaffold
contract over the archived chunkset:

* every chunk carries a ``difficulty_provenance`` tag; and
* NO chunk carries a ``difficulty_irt`` block unless
  ``difficulty_provenance == 'calibrated'`` AND
  ``difficulty_irt.n_responses >= min_responses`` (real backing responses).

When the scaffold is OFF the validator no-ops with ``passed=True`` plus a
single ``DIFFICULTY_PROVENANCE_SCAFFOLD_DISABLED`` info issue, so legacy
corpora (which carry no provenance field) never fail.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.assessment.irt_difficulty import (
    DEFAULT_MIN_RESPONSES,
    DIFFICULTY_PROVENANCE_CALIBRATED,
    irt_scaffold_enabled,
)

logger = logging.getLogger(__name__)

# Where archived chunksets live under a LibV2 course dir.
_CHUNKSET_RELPATHS = (
    "imscc_chunks/chunks.jsonl",
    "dart_chunks/chunks.jsonl",
)


def _load_chunks_from_course_dir(course_dir: Path) -> List[Dict[str, Any]]:
    """Best-effort load of every archived chunkset under a course dir."""
    chunks: List[Dict[str, Any]] = []
    for rel in _CHUNKSET_RELPATHS:
        path = course_dir / rel
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(rec, dict):
                        chunks.append(rec)
        except OSError as exc:  # noqa: BLE001
            logger.warning("Failed to read chunkset %s: %s", path, exc)
    return chunks


def _resolve_chunks(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve the chunk list from an explicit ``chunks`` input or course_dir."""
    explicit = inputs.get("chunks")
    if isinstance(explicit, list):
        return [c for c in explicit if isinstance(c, dict)]
    course_dir = inputs.get("course_dir") or inputs.get("manifest_path")
    if course_dir:
        try:
            cd = Path(course_dir)
            if cd.is_file():  # a manifest_path was threaded — use its parent
                cd = cd.parent
            return _load_chunks_from_course_dir(cd)
        except (OSError, ValueError, TypeError):
            return []
    return []


class DifficultyProvenanceValidator:
    """Warning-day-1 anti-fabrication gate for difficulty provenance."""

    name = "difficulty_provenance"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "difficulty_provenance")
        gate_config = inputs.get("_gate_config", {}) or {}
        min_responses = gate_config.get("min_responses", DEFAULT_MIN_RESPONSES)
        try:
            min_responses = int(min_responses)
        except (TypeError, ValueError):
            min_responses = DEFAULT_MIN_RESPONSES
        if min_responses < 1:
            min_responses = DEFAULT_MIN_RESPONSES

        issues: List[GateIssue] = []

        # Flag-off → strict no-op (legacy corpora carry no provenance field).
        if not irt_scaffold_enabled():
            issues.append(GateIssue(
                severity="info",
                code="DIFFICULTY_PROVENANCE_SCAFFOLD_DISABLED",
                message=(
                    "TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD is off; difficulty "
                    "provenance is not enforced (legacy heuristic-only emit)."
                ),
            ))
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=issues,
                metadata={"scaffold_enabled": False},
            )

        chunks = _resolve_chunks(inputs)
        missing_provenance = 0
        irt_without_responses = 0
        calibrated_count = 0

        for chunk in chunks:
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or "<unknown>"
            provenance = chunk.get("difficulty_provenance")
            irt = chunk.get("difficulty_irt")

            if not provenance:
                missing_provenance += 1
                issues.append(GateIssue(
                    severity="warning",
                    code="DIFFICULTY_PROVENANCE_MISSING",
                    message=(
                        f"Chunk {chunk_id} carries no difficulty_provenance "
                        "tag while the IRT scaffold is on."
                    ),
                    location=str(chunk_id),
                    suggestion="Run the chunk emit with the scaffold tagging pass.",
                ))

            if irt is not None:
                n_responses = irt.get("n_responses") if isinstance(irt, dict) else None
                try:
                    n_responses = int(n_responses)
                except (TypeError, ValueError):
                    n_responses = -1
                if provenance == DIFFICULTY_PROVENANCE_CALIBRATED:
                    calibrated_count += 1
                if provenance != DIFFICULTY_PROVENANCE_CALIBRATED or n_responses < min_responses:
                    irt_without_responses += 1
                    issues.append(GateIssue(
                        severity="warning",
                        code="DIFFICULTY_IRT_WITHOUT_RESPONSES",
                        message=(
                            f"Chunk {chunk_id} carries a difficulty_irt block "
                            f"(n_responses={n_responses}, provenance={provenance!r}) "
                            "without backing calibration; an IRT block is only "
                            f"valid when provenance='calibrated' and "
                            f"n_responses>={min_responses}."
                        ),
                        location=str(chunk_id),
                        suggestion="Drop the difficulty_irt block or provide real responses.",
                    ))

        passed = sum(1 for i in issues if i.severity == "critical") == 0
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            issues=issues,
            metadata={
                "scaffold_enabled": True,
                "chunks_checked": len(chunks),
                "missing_provenance": missing_provenance,
                "irt_without_responses": irt_without_responses,
                "calibrated_count": calibrated_count,
                "min_responses": min_responses,
            },
        )
