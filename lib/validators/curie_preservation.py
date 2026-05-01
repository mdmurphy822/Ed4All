"""Wave 130b — CuriePreservationValidator.

Architectural backstop to the Wave 120 preserve-retry mechanism.

Wave 120's preserve-retry only enforces *exact-set* preservation of the
manifest-declared CURIE surface forms. That keeps the surface forms
the manifest knows about safe, but it cannot detect drift between the
manifest declaration and the corpus's actual CURIE vocabulary. If a
new namespace prefix lands in chunks before the manifest catches up,
the synthesis pipeline can quietly strip those CURIEs from paraphrase
pairs because the retry guard never knew to protect them.

This validator runs after the synthesis phase on the chunk's *full*
CURIE set — extracted directly from chunk text via a regex covering
the canonical W3C / standard-ontology prefixes — and computes the
mean retention rate across all paraphrase pairs that source from a
chunk carrying CURIEs. Mean retention < 0.40 fails the gate closed.
Default threshold derived from the curie-fidelity-audit-2026-05
report § 6.

Skipped pair classes
--------------------

Deterministic / oracle-grounded generators populate CURIEs from
structured manifest input rather than paraphrasing chunk text, so
their pair body's CURIE set is governed by a different contract than
the paraphrase factories. This validator skips pairs whose
``template_id`` matches the deterministic generator prefixes:

* ``kg_metadata.*``
* ``violation_detection.*``
* ``abstention.*``
* ``schema_translation.*``
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Canonical CURIE regex — restricted to a curated allowlist of W3C /
# standard-ontology prefixes so we don't sweep up arbitrary
# ``foo:bar`` substrings (URLs, code identifiers, etc.).
CURIE_PREFIXES = (
    "sh", "rdfs", "owl", "rdf", "xsd", "skos", "dcterms", "foaf",
)
CURIE_REGEX = re.compile(
    r"\b(?:" + "|".join(CURIE_PREFIXES) + r"):[A-Za-z][A-Za-z0-9_]*\b"
)

DEFAULT_MIN_MEAN_RETENTION = 0.40
LOW_RETENTION_TOP_N = 20

# Deterministic generator template_id prefixes — these pairs source
# their CURIEs from oracle-grounded manifest data, not from paraphrase
# of chunk text, so they are out of scope for this validator.
DETERMINISTIC_TEMPLATE_PREFIXES = (
    "kg_metadata.",
    "violation_detection.",
    "abstention.",
    "schema_translation.",
)


def _extract_curies(text: str) -> Set[str]:
    """Return the set of CURIEs found in ``text``."""
    if not text:
        return set()
    return set(CURIE_REGEX.findall(text))


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

    Mirrors the synthesis_leakage validator: covers ``prompt`` and
    ``completion`` plus the common alternate names (``input`` /
    ``output`` / ``response``) so the validator works against multiple
    pair-shape conventions.
    """
    parts: List[str] = []
    for field in ("prompt", "completion", "input", "output", "response", "instruction"):
        value = row.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)


def _resolve_paths(
    inputs: Dict[str, Any],
) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Resolve ``(instruction_pairs_path, chunks_path, error)`` from
    inputs. Supports three input shapes:

    1. ``course_dir`` (canonical, mirrors synthesis_leakage).
    2. ``training_specs_dir`` (+ optional ``chunks_path`` /
       ``corpus_dir``) — the Wave 130b spec shape.
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
            "CuriePreservationValidator requires one of: course_dir, "
            "training_specs_dir, or instruction_pairs_path."
        )
    if chunks is None:
        return None, None, (
            "CuriePreservationValidator requires one of: course_dir, "
            "corpus_dir, or chunks_path."
        )
    return inst, chunks, None


class CuriePreservationValidator:
    """Pre-training gate: enforce mean CURIE retention across
    paraphrase pairs is at least ``min_mean_retention`` (default 0.40).
    """

    name = "curie_preservation"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "curie_preservation")

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
                        f"the curie_preservation gate."
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
                        f"curie_preservation gate needs the source "
                        f"corpus to compute retention against."
                    ),
                    location=str(chunks_path),
                )],
            )

        thresholds = inputs.get("thresholds") or inputs.get("threshold") or {}
        min_mean_retention = float(
            thresholds.get(
                "min_mean_retention", DEFAULT_MIN_MEAN_RETENTION
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

        retentions: List[float] = []
        zero_retention_count = 0
        low_retention_pairs: List[Dict[str, Any]] = []
        # Per-CURIE counters: number of pairs that should have
        # preserved each CURIE, vs how many actually did.
        curie_expected: Counter = Counter()
        curie_preserved: Counter = Counter()
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
                preserved = source_curies & pair_curies
                retention = len(preserved) / len(source_curies)
                retentions.append(retention)
                if not preserved:
                    zero_retention_count += 1

                for curie in source_curies:
                    curie_expected[curie] += 1
                    if curie in pair_curies:
                        curie_preserved[curie] += 1

                if retention < min_mean_retention:
                    low_retention_pairs.append({
                        "chunk_id": cid,
                        "template_id": template_id,
                        "source_curies": sorted(source_curies),
                        "pair_curies": sorted(pair_curies),
                        "retention": round(retention, 4),
                    })

        pairs_audited = len(retentions)
        if pairs_audited == 0:
            # Nothing to audit: every pair was either deterministic or
            # came from a chunk with no CURIEs. Pass with an info note
            # in score so callers can see that nothing happened.
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

        mean_retention = statistics.fmean(retentions)
        median_retention = statistics.median(retentions)
        # Sort low_retention_pairs ascending by retention so the worst
        # offenders surface first; cap at LOW_RETENTION_TOP_N.
        low_retention_pairs.sort(key=lambda r: r["retention"])
        low_retention_pairs = low_retention_pairs[:LOW_RETENTION_TOP_N]

        per_curie_retention = {
            curie: round(
                curie_preserved.get(curie, 0) / curie_expected[curie], 4
            )
            for curie in curie_expected
        }

        issues: List[GateIssue] = []
        passed = mean_retention >= min_mean_retention
        if not passed:
            worst = ", ".join(
                f"{p['chunk_id']} ({p['retention']:.2f})"
                for p in low_retention_pairs[:3]
            )
            issues.append(GateIssue(
                severity="critical",
                code="CURIE_RETENTION_BELOW_THRESHOLD",
                message=(
                    f"Mean CURIE retention {mean_retention:.3f} across "
                    f"{pairs_audited} paraphrase pairs is below the "
                    f"required threshold {min_mean_retention:.3f}. "
                    f"{zero_retention_count} pairs dropped every source "
                    f"CURIE. Worst offenders: {worst}. Fix: tighten the "
                    f"paraphrase prompt's CURIE-preservation directive "
                    f"or expand the property manifest so retry-preserve "
                    f"covers the missing surface forms."
                ),
                location=str(inst_path),
                suggestion=(
                    "Inspect Trainforge/scripts/audit_pairs.py output "
                    "for the affected chunk_ids; verify the property "
                    "manifest covers every CURIE prefix in the corpus."
                ),
            ))

        result = GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=round(mean_retention, 4),
            issues=issues,
        )
        # Stash the aggregate metrics on the result via a dict-shaped
        # error-free side channel: callers that care about the full
        # report (e.g. operator audit tooling) read from issues +
        # score, but we also surface the structured payload through a
        # well-known issue when verbose. Skip when passed and quiet.
        if not passed or zero_retention_count > 0:
            details_msg = json.dumps({
                "mean_retention": round(mean_retention, 4),
                "median_retention": round(median_retention, 4),
                "pairs_audited": pairs_audited,
                "zero_retention_count": zero_retention_count,
                "low_retention_pairs": low_retention_pairs,
                "per_curie_retention": per_curie_retention,
                "skipped_deterministic": skipped_deterministic,
                "skipped_no_curies": skipped_no_curies,
            }, sort_keys=True)
            result.issues.append(GateIssue(
                severity="info",
                code="CURIE_RETENTION_REPORT",
                message=details_msg,
                location=str(inst_path),
            ))
        return result


__all__ = ["CuriePreservationValidator"]
