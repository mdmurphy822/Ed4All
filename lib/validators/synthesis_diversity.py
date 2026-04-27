"""Wave 91 Action E: SynthesisDiversityValidator.

Post-synthesis gate at ``textbook_to_course::training_synthesis``.
Reads ``instruction_pairs.jsonl`` and computes the distribution of
``template_id`` (the field instruction_factory tags every emitted
pair with). Critical-fails when:

    - The top-3 templates account for > ``max_top3_share`` (default
      0.60) of pairs.
    - A single template accounts for > ``max_single_share`` (default
      0.35) of pairs.
    - The total number of distinct templates is < ``min_distinct_templates``
      (default 8).

Warning-fails when total pair count < ``min_total_pairs`` (default
100) so a corpus that's too small to assess diversity surfaces a
visible signal without blocking the run.

Inputs:
    instruction_pairs_path: Path to ``instruction_pairs.jsonl``.
        Required.
    max_top3_share: Optional override for the top-3 concentration
        ceiling.
    max_single_share: Optional override for the single-template
        ceiling.
    min_distinct_templates: Optional override for the distinct-template
        floor.
    min_total_pairs: Optional override for the warning-only volume
        floor.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


DEFAULT_MAX_TOP3_SHARE = 0.60
DEFAULT_MAX_SINGLE_SHARE = 0.35
DEFAULT_MIN_DISTINCT_TEMPLATES = 8
DEFAULT_MIN_TOTAL_PAIRS = 100

# The instruction_factory tags every emitted pair with ``template_id``
# (e.g. "remember.explanation"). Newer corpora may also carry
# ``template_name``; fall back through alternatives so the validator
# works across emit revisions.
_TEMPLATE_ID_KEYS = ("template_id", "template_name", "template")


class SynthesisDiversityValidator:
    """Template-collapse + corpus-volume guard."""

    name = "synthesis_diversity"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "synthesis_diversity")
        issues: List[GateIssue] = []

        path_raw = inputs.get("instruction_pairs_path")
        if not path_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_INPUTS",
                    message=(
                        "SynthesisDiversityValidator requires "
                        "instruction_pairs_path."
                    ),
                )],
            )

        path = Path(path_raw)
        if not path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INSTRUCTION_PAIRS_NOT_FOUND",
                    message=f"instruction_pairs.jsonl not found at {path}",
                    location=str(path),
                )],
            )

        max_top3 = float(
            inputs.get("max_top3_share", DEFAULT_MAX_TOP3_SHARE)
            or DEFAULT_MAX_TOP3_SHARE
        )
        max_single = float(
            inputs.get("max_single_share", DEFAULT_MAX_SINGLE_SHARE)
            or DEFAULT_MAX_SINGLE_SHARE
        )
        min_distinct = int(
            inputs.get("min_distinct_templates", DEFAULT_MIN_DISTINCT_TEMPLATES)
            or DEFAULT_MIN_DISTINCT_TEMPLATES
        )
        min_total = int(
            inputs.get("min_total_pairs", DEFAULT_MIN_TOTAL_PAIRS)
            or DEFAULT_MIN_TOTAL_PAIRS
        )

        # Read JSONL, tolerating empty / blank lines.
        template_counts: Counter = Counter()
        total = 0
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        issues.append(GateIssue(
                            severity="warning",
                            code="MALFORMED_JSONL_LINE",
                            message=f"skipping malformed JSONL line: {exc}",
                            location=str(path),
                        ))
                        continue
                    total += 1
                    tid = self._extract_template_id(rec)
                    template_counts[tid] += 1
        except OSError as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INSTRUCTION_PAIRS_READ_ERROR",
                    message=f"failed to read {path}: {exc}",
                    location=str(path),
                )],
            )

        # ---------- Volume signal (warning) ----------
        if total < min_total:
            issues.append(GateIssue(
                severity="warning",
                code="LOW_TOTAL_PAIR_COUNT",
                message=(
                    f"instruction_pairs.jsonl has {total} pairs "
                    f"(< min_total_pairs={min_total}). Diversity metrics "
                    f"are unreliable on small corpora."
                ),
                location=str(path),
            ))

        # ---------- Diversity signals (critical) ----------
        if total > 0:
            distinct = len(template_counts)
            if distinct < min_distinct:
                issues.append(GateIssue(
                    severity="critical",
                    code="LOW_DISTINCT_TEMPLATES",
                    message=(
                        f"only {distinct} distinct templates emitted "
                        f"(< min_distinct_templates={min_distinct}); "
                        f"trained model will memorise {distinct} stems."
                    ),
                    location=str(path),
                    suggestion=(
                        "Verify the synthesis corpus draws from the "
                        "full template catalog; check that "
                        "instruction_factory.TEMPLATE_CATALOG cells are "
                        "all reachable for this chunk distribution."
                    ),
                ))
            top3_share = sum(
                c for _, c in template_counts.most_common(3)
            ) / total
            if top3_share > max_top3:
                top3 = template_counts.most_common(3)
                issues.append(GateIssue(
                    severity="critical",
                    code="TOP3_TEMPLATE_DOMINANCE",
                    message=(
                        f"top-3 templates account for "
                        f"{top3_share:.2%} of pairs (> max_top3_share="
                        f"{max_top3:.2%}); top-3: {top3}."
                    ),
                    location=str(path),
                    suggestion=(
                        "Template-collapse signal — most pairs trace "
                        "back to a handful of stems. Investigate chunk "
                        "metadata (bloom_level, content_type_label) and "
                        "stratify the synthesis call."
                    ),
                ))
            top1_id, top1_count = template_counts.most_common(1)[0]
            top1_share = top1_count / total
            if top1_share > max_single:
                issues.append(GateIssue(
                    severity="critical",
                    code="SINGLE_TEMPLATE_DOMINANCE",
                    message=(
                        f"single template '{top1_id}' accounts for "
                        f"{top1_share:.2%} of pairs (> max_single_share="
                        f"{max_single:.2%})."
                    ),
                    location=str(path),
                    suggestion=(
                        "One stem dominates the corpus. Check whether a "
                        "fallback path in instruction_factory is being "
                        "selected for most chunks (typically signalled by "
                        "'understand._default')."
                    ),
                ))

        critical = sum(1 for i in issues if i.severity == "critical")
        passed = critical == 0
        # Score: 1.0 - top-3 share (approximate diversity index, capped
        # at 0.0 floor). 0 pairs -> 0.0. Mirrors the convention in
        # other Trainforge validators.
        if total == 0:
            score = 0.0
        else:
            top3_pairs = sum(c for _, c in template_counts.most_common(3))
            score = round(max(0.0, 1.0 - (top3_pairs / total)), 4)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    @staticmethod
    def _extract_template_id(rec: Dict[str, Any]) -> str:
        for k in _TEMPLATE_ID_KEYS:
            v = rec.get(k)
            if v:
                return str(v)
        return "unknown"


__all__ = [
    "SynthesisDiversityValidator",
    "DEFAULT_MAX_TOP3_SHARE",
    "DEFAULT_MAX_SINGLE_SHARE",
    "DEFAULT_MIN_DISTINCT_TEMPLATES",
    "DEFAULT_MIN_TOTAL_PAIRS",
]
