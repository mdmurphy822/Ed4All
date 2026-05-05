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
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    total_pairs: int,
    n_unique_templates: int,
    top1_share: float,
    top3_share: float,
    max_top3_share: float,
    max_single_share: float,
    min_distinct_templates: int,
    min_total_pairs: int,
) -> None:
    """Emit one ``synthesis_diversity_check`` decision per validate() call.

    H3 Wave W4: every threshold-fail / pass / missing-input path emits
    one event. ml_features payload includes top-1 + top-3 share, the
    distinct-template count, and the four threshold floors so post-hoc
    replay can reconstruct the diversity verdict.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rationale = (
        f"synthesis_diversity gate verdict: total_pairs={total_pairs}, "
        f"n_unique_templates={n_unique_templates}, "
        f"top1_share={top1_share:.4f}, top3_share={top3_share:.4f}; "
        f"thresholds=(max_top3={max_top3_share:.4f}, "
        f"max_single={max_single_share:.4f}, "
        f"min_distinct={min_distinct_templates}, "
        f"min_total={min_total_pairs}); failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "total_pairs": int(total_pairs),
        "n_unique_templates": int(n_unique_templates),
        "top1_share": float(top1_share),
        "top3_share": float(top3_share),
        "max_top3_share": float(max_top3_share),
        "max_single_share": float(max_single_share),
        "min_distinct_templates": int(min_distinct_templates),
        "min_total_pairs": int(min_total_pairs),
        "passed": bool(passed),
        "failure_code": code,
    }
    try:
        capture.log_decision(
            decision_type="synthesis_diversity_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "synthesis_diversity_check: %s",
            exc,
        )


DEFAULT_MAX_TOP3_SHARE = 0.60
DEFAULT_MAX_SINGLE_SHARE = 0.35
DEFAULT_MIN_DISTINCT_TEMPLATES = 8
DEFAULT_MIN_TOTAL_PAIRS = 100

# Wave 105: prefix-bigram diversity defaults. Catches template-collapse
# at the answer level even when ``template_id`` is well distributed.
# Example: rdf-shacl-551-2 had 11 distinct template_ids but 80% of
# completions started with "the treatment" — the trained model
# memorised that phrase, not 11 templates.
DEFAULT_MAX_PREFIX_TOP1_SHARE = 0.15
DEFAULT_MAX_PREFIX_TOP3_SHARE = 0.30


_PREFIX_WORD_RE = re.compile(r"[a-zA-Z']+")

# The instruction_factory tags every emitted pair with ``template_id``
# (e.g. "remember.explanation"). Newer corpora may also carry
# ``template_name``; fall back through alternatives so the validator
# works across emit revisions.
_TEMPLATE_ID_KEYS = ("template_id", "template_name", "template")


class SynthesisDiversityValidator:
    """Template-collapse + corpus-volume guard."""

    name = "synthesis_diversity"
    version = "1.0.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "synthesis_diversity")
        issues: List[GateIssue] = []

        capture = inputs.get("decision_capture") or self._decision_capture

        path_raw = inputs.get("instruction_pairs_path")
        if not path_raw:
            _emit_decision(
                capture, passed=False, code="MISSING_INPUTS",
                total_pairs=0, n_unique_templates=0,
                top1_share=0.0, top3_share=0.0,
                max_top3_share=DEFAULT_MAX_TOP3_SHARE,
                max_single_share=DEFAULT_MAX_SINGLE_SHARE,
                min_distinct_templates=DEFAULT_MIN_DISTINCT_TEMPLATES,
                min_total_pairs=DEFAULT_MIN_TOTAL_PAIRS,
            )
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
            _emit_decision(
                capture, passed=False, code="INSTRUCTION_PAIRS_NOT_FOUND",
                total_pairs=0, n_unique_templates=0,
                top1_share=0.0, top3_share=0.0,
                max_top3_share=DEFAULT_MAX_TOP3_SHARE,
                max_single_share=DEFAULT_MAX_SINGLE_SHARE,
                min_distinct_templates=DEFAULT_MIN_DISTINCT_TEMPLATES,
                min_total_pairs=DEFAULT_MIN_TOTAL_PAIRS,
            )
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
        prefix_counts: Counter = Counter()
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
                    # Wave 105: track the prefix bigram of the
                    # completion text. If ``completion`` is missing
                    # we fall back to "answer" or "response" since
                    # different emit revisions used different keys;
                    # records that have neither contribute nothing.
                    bigram = self._extract_prefix_bigram(rec)
                    if bigram is not None:
                        prefix_counts[bigram] += 1
        except OSError as exc:
            _emit_decision(
                capture, passed=False, code="INSTRUCTION_PAIRS_READ_ERROR",
                total_pairs=total, n_unique_templates=len(template_counts),
                top1_share=0.0, top3_share=0.0,
                max_top3_share=max_top3,
                max_single_share=max_single,
                min_distinct_templates=min_distinct,
                min_total_pairs=min_total,
            )
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

            # ---------- Prefix-bigram diversity (Wave 105) ----------
            # Flags template-collapse at the COMPLETION text level: if
            # the same first 2 words show up in a large fraction of
            # responses, the trained adapter will memorise that phrase
            # instead of the underlying behaviour. The Wave 104 eval
            # of rdf-shacl-551-2 had 11 distinct template_ids (passing
            # the older check) but 80% of completions started with
            # "the treatment" — exactly the failure mode this catches.
            self._check_response_prefix_diversity(
                prefix_counts=prefix_counts,
                total_with_prefix=sum(prefix_counts.values()),
                max_top1=float(
                    inputs.get("max_prefix_top1_share",
                               DEFAULT_MAX_PREFIX_TOP1_SHARE)
                    or DEFAULT_MAX_PREFIX_TOP1_SHARE
                ),
                max_top3=float(
                    inputs.get("max_prefix_top3_share",
                               DEFAULT_MAX_PREFIX_TOP3_SHARE)
                    or DEFAULT_MAX_PREFIX_TOP3_SHARE
                ),
                issues=issues,
                path=path,
            )

        critical = sum(1 for i in issues if i.severity == "critical")
        passed = critical == 0
        # Score: 1.0 - top-3 share (approximate diversity index, capped
        # at 0.0 floor). 0 pairs -> 0.0. Mirrors the convention in
        # other Trainforge validators.
        if total == 0:
            score = 0.0
            top1_share_metric = 0.0
            top3_share_metric = 0.0
        else:
            top3_pairs = sum(c for _, c in template_counts.most_common(3))
            score = round(max(0.0, 1.0 - (top3_pairs / total)), 4)
            top3_share_metric = top3_pairs / total
            top1_count = (
                template_counts.most_common(1)[0][1] if template_counts else 0
            )
            top1_share_metric = top1_count / total if total else 0.0

        # H3 W4: surface the first critical issue's code (if any) so
        # post-hoc replay can distinguish single-template-dominance from
        # top-3-dominance from low-distinct-templates without parsing
        # GateIssue lists.
        failure_code = None
        if not passed:
            for i in issues:
                if i.severity == "critical":
                    failure_code = i.code
                    break
        _emit_decision(
            capture,
            passed=passed,
            code=failure_code,
            total_pairs=total,
            n_unique_templates=len(template_counts),
            top1_share=top1_share_metric,
            top3_share=top3_share_metric,
            max_top3_share=max_top3,
            max_single_share=max_single,
            min_distinct_templates=min_distinct,
            min_total_pairs=min_total,
        )

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

    @staticmethod
    def _extract_prefix_bigram(
        rec: Dict[str, Any],
    ) -> Tuple[str, str] | None:
        """First-2-words bigram from the completion text (Wave 105).

        Lowercased, punctuation stripped (handled via
        :data:`_PREFIX_WORD_RE`). Falls back through a few key names
        because emit revisions across waves used different fields
        (``completion`` is canonical; ``response`` / ``answer`` show
        up in older / paraphrase-style records).
        """
        for k in ("completion", "response", "answer", "output"):
            v = rec.get(k)
            if not v:
                continue
            words = _PREFIX_WORD_RE.findall(str(v).lower())
            if len(words) >= 2:
                return (words[0], words[1])
            return None
        return None

    @staticmethod
    def _check_response_prefix_diversity(
        *,
        prefix_counts: Counter,
        total_with_prefix: int,
        max_top1: float,
        max_top3: float,
        issues: List[GateIssue],
        path: Path,
    ) -> None:
        """Top-1 / top-3 prefix-bigram concentration check.

        Mirrors the template-id checks; thresholds are tighter
        because prefix bigrams are intrinsically less concentrated
        in healthy corpora (any natural-language paraphrase pass
        produces a long tail).
        """
        if total_with_prefix < 5:
            # Too few records to assess; the volume warning above
            # already surfaces the small-corpus signal.
            return
        top1_bigram, top1_count = prefix_counts.most_common(1)[0]
        top1_share = top1_count / total_with_prefix
        top3_share = (
            sum(c for _, c in prefix_counts.most_common(3))
            / total_with_prefix
        )
        if top1_share > max_top1:
            issues.append(GateIssue(
                severity="critical",
                code="PREFIX_BIGRAM_TOP1_DOMINANCE",
                message=(
                    f"top-1 completion prefix bigram "
                    f"{top1_bigram!r} accounts for "
                    f"{top1_share:.2%} of completions (> "
                    f"max_prefix_top1_share={max_top1:.2%}); "
                    f"trained model will memorise this phrase."
                ),
                location=str(path),
                suggestion=(
                    "Inspect the synthesizer's paraphrase prompt — "
                    "high top-1 prefix concentration usually means "
                    "the LLM is locking onto a single stem like "
                    "'The treatment of...' or 'The core idea...'. "
                    "Vary the request, increase temperature, or "
                    "force the model to vary its opening."
                ),
            ))
        if top3_share > max_top3:
            top3 = prefix_counts.most_common(3)
            issues.append(GateIssue(
                severity="critical",
                code="PREFIX_BIGRAM_TOP3_DOMINANCE",
                message=(
                    f"top-3 completion prefix bigrams account for "
                    f"{top3_share:.2%} of completions (> "
                    f"max_prefix_top3_share={max_top3:.2%}); "
                    f"top-3 bigrams: {top3}."
                ),
                location=str(path),
                suggestion=(
                    "Even with diverse template_ids, the answer "
                    "text is template-collapsed at the prefix "
                    "level. The trained model will pattern-match "
                    "these prefixes rather than learn the "
                    "underlying behaviour."
                ),
            ))


__all__ = [
    "SynthesisDiversityValidator",
    "DEFAULT_MAX_TOP3_SHARE",
    "DEFAULT_MAX_SINGLE_SHARE",
    "DEFAULT_MIN_DISTINCT_TEMPLATES",
    "DEFAULT_MIN_TOTAL_PAIRS",
    "DEFAULT_MAX_PREFIX_TOP1_SHARE",
    "DEFAULT_MAX_PREFIX_TOP3_SHARE",
]


def _main() -> int:  # pragma: no cover - manual CLI helper
    """Wave 105: ``python -m lib.validators.synthesis_diversity <path>``.

    Run the validator on an instruction_pairs.jsonl and print the
    issue codes + the prefix-bigram distribution. Useful for
    confirming template-collapse on an existing corpus without
    spinning up the full validation gate framework.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Run SynthesisDiversityValidator against an "
            "instruction_pairs.jsonl and print the report."
        ),
    )
    parser.add_argument(
        "instruction_pairs_path",
        type=str,
        help="Path to instruction_pairs.jsonl.",
    )
    args = parser.parse_args()

    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": args.instruction_pairs_path,
    })

    print(f"passed={result.passed}  score={result.score}")
    print(f"issues ({len(result.issues)}):")
    for i in result.issues:
        print(f"  [{i.severity}] {i.code}: {i.message}")

    # Re-read the file so we can print the prefix-bigram distribution
    # — handy for the Wave 105 verification step.
    prefix_counts: Counter = Counter()
    total = 0
    with open(args.instruction_pairs_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bigram = SynthesisDiversityValidator._extract_prefix_bigram(rec)
            if bigram is not None:
                total += 1
                prefix_counts[bigram] += 1
    print(f"\nprefix-bigram distribution (n={total}):")
    for bigram, count in prefix_counts.most_common(10):
        print(
            f"  {bigram!r}: {count} "
            f"({100 * count / max(total, 1):.1f}%)"
        )
    if total:
        top1_share = prefix_counts.most_common(1)[0][1] / total
        top3_share = (
            sum(c for _, c in prefix_counts.most_common(3)) / total
        )
        print(f"top1_share = {100*top1_share:.2f}%")
        print(f"top3_share = {100*top3_share:.2f}%")

    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
