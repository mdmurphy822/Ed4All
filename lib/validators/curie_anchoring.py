"""Wave 135c — Binary per-pair CURIE anchoring gate.

Replaces curie_preservation's mean-retention metric. Under the Wave
135 contract, the LLM provides natural-language variation and the
force-injection path provides canonical CURIE anchoring. The
preservation metric is no longer meaningful (force-injection
guarantees CURIE presence regardless of LLM behavior).

This gate is the regression-detection sentinel: if the injector
path breaks, the per-pair anchoring rate drops to whatever the
natural paraphrase rate is (~0.10 per the 2026-05-01 audit). A
0.95 floor catches that loudly. Healthy injection keeps the rate
at ~1.00 by construction.

Skips deterministic generator pairs (template_id matching
kg_metadata.* / violation_detection.* / abstention.* /
schema_translation.*) — those are oracle-grounded.

Metric
------

Binary per-pair anchoring rate:

    pair_anchoring_rate = anchored_count / total_eligible_pairs

For each eligible (paraphrase) pair sourced from a chunk that carries
≥1 CURIE, the pair is considered ``anchored`` iff its body
(``prompt + completion`` / ``prompt + chosen``) contains at least one
of the source-chunk CURIEs.

Failure semantics
-----------------

Default threshold ``min_pair_anchoring_rate=0.95``. Failures emit a
critical ``PAIR_ANCHORING_BELOW_THRESHOLD`` issue and a structured
``PAIR_ANCHORING_REPORT`` info issue carrying aggregate counts plus
the worst offenders for triage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.curie_extraction import extract_curies as _extract_curies
from lib.ontology.template_prefixes import DETERMINISTIC_TEMPLATE_PREFIXES

logger = logging.getLogger(__name__)


DEFAULT_MIN_PAIR_ANCHORING_RATE = 0.95
UNANCHORED_TOP_N = 20


def _is_deterministic(template_id: str) -> bool:
    """True when ``template_id`` matches a deterministic generator
    prefix and the pair should be skipped by this validator."""
    if not template_id:
        return False
    return any(
        template_id.startswith(prefix)
        for prefix in DETERMINISTIC_TEMPLATE_PREFIXES
    )


def _pair_body_text(row: Dict[str, Any]) -> str:
    """Concatenate the text fields a paraphrase pair could carry.

    Mirrors curie_preservation: covers ``prompt`` / ``completion``
    plus alternates (``input`` / ``output`` / ``response`` /
    ``instruction``) AND preference-pair fields (``chosen`` /
    ``rejected``) so the validator works against both instruction
    and preference shapes.
    """
    parts: List[str] = []
    for field in (
        "prompt",
        "completion",
        "input",
        "output",
        "response",
        "instruction",
        "chosen",
        "rejected",
    ):
        value = row.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)


def _resolve_paths(
    inputs: Dict[str, Any],
) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Resolve ``(instruction_pairs_path, chunks_path, error)`` from
    inputs. Mirrors curie_preservation's input shape:

    1. ``course_dir`` (canonical, mirrors synthesis_leakage).
    2. ``training_specs_dir`` (+ optional ``chunks_path`` /
       ``corpus_dir``).
    3. Explicit ``instruction_pairs_path`` + ``chunks_path``.
    """
    inst: Optional[Path] = None
    chunks: Optional[Path] = None

    raw_inst_path = inputs.get("instruction_pairs_path")
    if isinstance(raw_inst_path, str) and raw_inst_path:
        inst = Path(raw_inst_path)

    raw_chunks_path = inputs.get("chunks_path")
    if isinstance(raw_chunks_path, str) and raw_chunks_path:
        chunks = Path(raw_chunks_path)

    course_dir_raw = inputs.get("course_dir")
    if course_dir_raw:
        cd = Path(course_dir_raw)
        if inst is None:
            inst = cd / "training_specs" / "instruction_pairs.jsonl"
        if chunks is None:
            chunks = cd / "corpus" / "chunks.jsonl"

    training_specs_dir_raw = inputs.get("training_specs_dir")
    if training_specs_dir_raw and inst is None:
        inst = Path(training_specs_dir_raw) / "instruction_pairs.jsonl"

    corpus_dir_raw = inputs.get("corpus_dir")
    if corpus_dir_raw and chunks is None:
        chunks = Path(corpus_dir_raw) / "chunks.jsonl"

    if inst is None:
        return None, None, (
            "CurieAnchoringValidator requires one of: course_dir, "
            "training_specs_dir, or instruction_pairs_path."
        )
    if chunks is None:
        return None, None, (
            "CurieAnchoringValidator requires one of: course_dir, "
            "corpus_dir, or chunks_path."
        )
    return inst, chunks, None


class CurieAnchoringValidator:
    """Pre-training gate: enforce binary per-pair CURIE anchoring rate
    across paraphrase pairs is at least ``min_pair_anchoring_rate``
    (default 0.95). Sentinel for force-injection regressions.
    """

    name = "curie_anchoring"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "curie_anchoring")

        inst_path, chunks_path, path_err = _resolve_paths(inputs)
        if path_err:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_INPUTS",
                    message=path_err,
                )],
            )
        # Help static type checkers / readers: paths are non-None here.
        assert inst_path is not None and chunks_path is not None

        if not inst_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INSTRUCTION_PAIRS_NOT_FOUND",
                    message=(
                        f"instruction_pairs.jsonl not found at "
                        f"{inst_path}; run the synthesis phase before "
                        f"the curie_anchoring gate."
                    ),
                    location=str(inst_path),
                )],
            )
        if not chunks_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="CHUNKS_NOT_FOUND",
                    message=(
                        f"chunks.jsonl not found at {chunks_path}; the "
                        f"curie_anchoring gate needs the source corpus "
                        f"to compute anchoring against."
                    ),
                    location=str(chunks_path),
                )],
            )

        thresholds = inputs.get("thresholds") or inputs.get("threshold") or {}
        min_pair_anchoring_rate = float(
            thresholds.get(
                "min_pair_anchoring_rate", DEFAULT_MIN_PAIR_ANCHORING_RATE
            )
        )

        # Build chunk_id → CURIE set map.
        chunk_curies: Dict[str, Set[str]] = {}
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = chunk.get("id") or chunk.get("chunk_id") or ""
                if not cid:
                    continue
                chunk_curies[cid] = _extract_curies(
                    str(chunk.get("text") or "")
                )

        anchored_count = 0
        total_eligible = 0
        unanchored_pairs: List[Dict[str, Any]] = []
        skipped_deterministic = 0
        skipped_no_curies = 0

        with inst_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                template_id = str(row.get("template_id") or "")
                if _is_deterministic(template_id):
                    skipped_deterministic += 1
                    continue

                cid = str(row.get("chunk_id") or "")
                source_curies = chunk_curies.get(cid, set())
                if not source_curies:
                    skipped_no_curies += 1
                    continue

                pair_curies = _extract_curies(_pair_body_text(row))
                is_anchored = bool(source_curies & pair_curies)
                total_eligible += 1
                if is_anchored:
                    anchored_count += 1
                else:
                    unanchored_pairs.append({
                        "chunk_id": cid,
                        "template_id": template_id,
                        "source_curies": sorted(source_curies),
                        "pair_curies": sorted(pair_curies),
                    })

        if total_eligible == 0:
            # Nothing to audit: every pair was either deterministic or
            # came from a chunk with no CURIEs. Pass with an info note.
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="info",
                    code="NO_AUDITABLE_PAIRS",
                    message=(
                        f"No paraphrase pairs to audit "
                        f"(skipped_deterministic={skipped_deterministic}, "
                        f"skipped_no_curies={skipped_no_curies}). "
                        f"Gate passes by default."
                    ),
                    location=str(inst_path),
                )],
            )

        pair_anchoring_rate = anchored_count / total_eligible
        # Cap unanchored examples in the report.
        unanchored_pairs = unanchored_pairs[:UNANCHORED_TOP_N]

        issues: List[GateIssue] = []
        passed = pair_anchoring_rate >= min_pair_anchoring_rate
        if not passed:
            worst = ", ".join(
                p["chunk_id"] for p in unanchored_pairs[:3]
            )
            unanchored_count = total_eligible - anchored_count
            issues.append(GateIssue(
                severity="critical",
                code="PAIR_ANCHORING_BELOW_THRESHOLD",
                message=(
                    f"Per-pair CURIE anchoring rate "
                    f"{pair_anchoring_rate:.3f} across {total_eligible} "
                    f"paraphrase pairs is below the required threshold "
                    f"{min_pair_anchoring_rate:.3f}. "
                    f"{unanchored_count} pairs contain zero source-chunk "
                    f"CURIEs. Sample offenders: {worst}. Likely cause: "
                    f"the Wave 135b force-injection path regressed; "
                    f"verify the property manifest covers the chunk's "
                    f"CURIEs and that the injector is wired into the "
                    f"synthesis path."
                ),
                location=str(inst_path),
                suggestion=(
                    "Inspect the instruction_pairs.jsonl entries for "
                    "the listed chunk_ids; confirm Wave 135b "
                    "force-injection is wired into "
                    "Trainforge/generators/_local_provider.py and the "
                    "property manifest covers the chunk vocabulary."
                ),
            ))

        result = GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=round(pair_anchoring_rate, 4),
            issues=issues,
        )

        # Always emit the structured report when there are unanchored
        # pairs (regardless of pass/fail) so operators can spot
        # creeping injector drift before it crosses the threshold.
        unanchored_total = total_eligible - anchored_count
        if not passed or unanchored_total > 0:
            details_msg = json.dumps({
                "pair_anchoring_rate": round(pair_anchoring_rate, 4),
                "anchored_count": anchored_count,
                "unanchored_count": unanchored_total,
                "total_eligible_pairs": total_eligible,
                "unanchored_pairs": unanchored_pairs,
                "skipped_deterministic": skipped_deterministic,
                "skipped_no_curies": skipped_no_curies,
            }, sort_keys=True)
            result.issues.append(GateIssue(
                severity="info",
                code="PAIR_ANCHORING_REPORT",
                message=details_msg,
                location=str(inst_path),
            ))
        return result


__all__ = ["CurieAnchoringValidator"]
