"""Wave 122 — one-command operator audit for synthesised training pairs.

Reads ``instruction_pairs.jsonl`` + ``preference_pairs.jsonl`` +
``corpus/chunks.jsonl`` from a course directory and runs the
10-dimension poisoning audit that the inline shell scripts ran during
Waves 120-122. Wraps the runtime validators where possible
(``SynthesisLeakageValidator``) and adds the dimensions not gated yet
(force-inject saturation, fallback rate, Bloom skew, dupes, unicode,
prompt-injection markers, schema validity).

Designed for use AFTER a full uncapped synthesis run, before training.
The runtime workflow gates fire on the canonical artifacts
automatically; this script is the operator-side overlay that catches
distribution / quality dimensions the gates don't cover.

Usage:

    # Audit canonical pairs (post full-corpus run):
    python -m Trainforge.scripts.audit_pairs \\
        --course LibV2/courses/rdf-shacl-551-2

    # Audit smoke output (pre full-run sanity check):
    python -m Trainforge.scripts.audit_pairs \\
        --course LibV2/courses/rdf-shacl-551-2 --smoke

    # JSON output for downstream tooling:
    python -m Trainforge.scripts.audit_pairs \\
        --course LibV2/courses/rdf-shacl-551-2 --format json

Exit codes:
    0 — every dimension clean (no critical poisoning vector hit)
    1 — at least one dimension fails its threshold
    2 — couldn't read the inputs (missing files, parse errors)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the validator's pattern definitions so the audit and the
# runtime gate stay in lockstep.
from lib.validators.synthesis_leakage import (  # noqa: E402
    DEFAULT_LEAK_RATE_THRESHOLD,
    DEFAULT_LEAK_SPAN_CHARS,
    _ASSESSMENT_SCAFFOLD_PATTERNS,
)

# Force-inject phrasings (Wave 121). Kept in sync with the factory
# definitions; if a phrasing changes there, update here.
_PROMPT_REFERENCE_PHRASINGS = [
    " (Reference:", " (Relevant terms:", " (See:", " (In context:",
]
_COMPLETION_REFERENCE_PHRASINGS = [
    " Canonical terms:", " The relevant terms are",
    " Key vocabulary:", " This concerns ",
]

# Wave 122 rotated scaffolding phrasings (instruction + preference share
# the same set). Audit reports their distribution as a sanity check.
_SCAFFOLD_PHRASINGS = [
    "is best understood by tying",
    "should ground each idea",
    "connect each piece to",
    "is built from the interplay of",
]

# Pre-Wave-122 rigid template — must be 0 in any post-Wave-122 run.
_LEGACY_SCAFFOLD = "should be explained through the concrete RDF/SHACL role"

# Suspicious unicode ranges (zero-width, BiDi overrides, invisibles).
_SUSPICIOUS_RANGES: List[Tuple[int, int]] = [
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0xFFF0, 0xFFFF),
]

_INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all |previous |prior )?instructions", re.IGNORECASE),
    re.compile(r"<\|.*?\|>"),
    re.compile(r"</?(?:system|user|assistant|prompt)>", re.IGNORECASE),
    re.compile(r"###\s*(?:instruction|response|input)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN mode", re.IGNORECASE),
    re.compile(r"override (?:safety|the system)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _longest_common_substring(a: str, b: str, min_len: int) -> int:
    """Return length of the longest substring of ``a`` that appears in
    ``b`` and is at least ``min_len`` chars. 0 if no such substring."""
    if not a or not b:
        return 0
    a_lo, b_lo = a.lower(), b.lower()
    if len(a_lo) < min_len or len(b_lo) < min_len:
        return 0
    longest = 0
    for i in range(len(a_lo) - min_len + 1):
        if a_lo[i:i + min_len] in b_lo:
            n = min_len
            while i + n < len(a_lo) and a_lo[i:i + n + 1] in b_lo:
                n += 1
            longest = max(longest, n)
    return longest


def _has_suspicious_unicode(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _SUSPICIOUS_RANGES:
            if lo <= cp <= hi:
                return True
    return False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Dimension:
    name: str
    passed: bool
    severity: str  # "critical" | "warning" | "info"
    detail: str
    sample: Optional[List[str]] = None


@dataclass
class AuditReport:
    course_dir: str
    instruction_count: int
    preference_count: int
    chunks_count: int
    dimensions: List[Dimension] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        return all(d.passed for d in self.dimensions if d.severity == "critical")

    @property
    def critical_failures(self) -> List[Dimension]:
        return [d for d in self.dimensions if d.severity == "critical" and not d.passed]


# ---------------------------------------------------------------------------
# Dimension checks
# ---------------------------------------------------------------------------

def _check_assessment_scaffolding(
    inst: List[Dict], pref: List[Dict],
) -> Dimension:
    """Wave 122 zero-tolerance gate."""
    inst_hits = []
    for p in inst:
        text = f"{p.get('prompt','')} {p.get('completion','')}"
        for pat in _ASSESSMENT_SCAFFOLD_PATTERNS:
            m = pat.search(text)
            if m:
                inst_hits.append(f"{p.get('chunk_id','?')}: {m.group(0)[:80]}")
                break
    pref_hits = []
    for p in pref:
        text = " ".join(str(p.get(f, "")) for f in ("prompt", "chosen", "rejected"))
        for pat in _ASSESSMENT_SCAFFOLD_PATTERNS:
            m = pat.search(text)
            if m:
                pref_hits.append(f"{p.get('chunk_id','?')}: {m.group(0)[:80]}")
                break
    passed = not inst_hits and not pref_hits
    return Dimension(
        name="assessment_scaffolding",
        passed=passed,
        severity="critical",
        detail=(
            f"{len(inst_hits)}/{len(inst)} instruction + "
            f"{len(pref_hits)}/{len(pref)} preference pairs carry "
            f"assessment-outline patterns (Question N (XX-NN, Bloom: ...)). "
            f"Target: 0/0."
        ),
        sample=inst_hits[:3] + pref_hits[:3],
    )


def _check_verbatim_leakage(
    inst: List[Dict], pref: List[Dict], chunks: Dict[str, str],
    rate_threshold: float, span_threshold: int,
) -> Dimension:
    """Wave 121 leak gate."""
    inst_leaks: List[str] = []
    for p in inst:
        chunk_text = chunks.get(str(p.get("chunk_id", "")), "")
        if not chunk_text:
            continue
        for fld in ("prompt", "completion"):
            if _longest_common_substring(str(p.get(fld, "")), chunk_text, span_threshold) >= span_threshold:
                inst_leaks.append(f"{p.get('chunk_id','?')}/{fld}")
                break
    pref_leaks: List[str] = []
    for p in pref:
        chunk_text = chunks.get(str(p.get("chunk_id", "")), "")
        if not chunk_text:
            continue
        for fld in ("prompt", "chosen"):
            if _longest_common_substring(str(p.get(fld, "")), chunk_text, span_threshold) >= span_threshold:
                pref_leaks.append(f"{p.get('chunk_id','?')}/{fld}")
                break
    inst_rate = len(inst_leaks) / max(len(inst), 1)
    pref_rate = len(pref_leaks) / max(len(pref), 1)
    passed = inst_rate <= rate_threshold and pref_rate <= rate_threshold
    return Dimension(
        name="verbatim_leakage",
        passed=passed,
        severity="critical",
        detail=(
            f"Instruction: {len(inst_leaks)}/{len(inst)} ({100*inst_rate:.1f}%); "
            f"Preference: {len(pref_leaks)}/{len(pref)} ({100*pref_rate:.1f}%). "
            f"Threshold: {100*rate_threshold:.1f}% with ≥{span_threshold}-char span."
        ),
        sample=inst_leaks[:3] + pref_leaks[:3],
    )


def _check_legacy_scaffold(inst: List[Dict]) -> Dimension:
    """Wave 122 — pre-rotation rigid template must be gone."""
    hits = [p.get("chunk_id", "?") for p in inst if _LEGACY_SCAFFOLD in p.get("completion", "")]
    return Dimension(
        name="legacy_scaffold_template",
        passed=not hits,
        severity="critical",
        detail=(
            f"{len(hits)}/{len(inst)} pairs carry the pre-Wave-122 rigid "
            f"scaffolding template. Should be 0 — Wave 122 rotated to "
            f"4 phrasings."
        ),
        sample=hits[:3],
    )


def _check_phrasing_distribution(
    inst: List[Dict], phrasings: List[str], side: str, max_top1_share: float,
) -> Dimension:
    """Wave 121 force-inject saturation guard. Distribution skew is a
    warning, not critical — the runtime gate doesn't enforce this, but
    a >60% concentration on one phrasing is worth flagging."""
    dist: Counter = Counter()
    for p in inst:
        text = p.get("prompt") if side == "prompt" else p.get("completion")
        for ph in phrasings:
            if ph in (text or ""):
                dist[ph.strip()] += 1
                break
    if not dist:
        return Dimension(
            name=f"force_inject_{side}_distribution",
            passed=True,
            severity="info",
            detail=f"No force-inject {side} phrasings used (no preserve_tokens active or none missing).",
        )
    top1 = max(dist.values())
    total = sum(dist.values())
    share = top1 / total
    passed = share <= max_top1_share
    return Dimension(
        name=f"force_inject_{side}_distribution",
        passed=passed,
        severity="warning",
        detail=(
            f"{total}/{len(inst)} pairs injected on {side}; "
            f"top-1 phrasing share {100*share:.0f}% (threshold "
            f"{100*max_top1_share:.0f}%). Distribution: {dict(dist)}"
        ),
    )


def _check_schema(inst: List[Dict], pref: List[Dict]) -> Dimension:
    try:
        import jsonschema
    except ImportError:
        return Dimension(
            name="schema_validity",
            passed=True,
            severity="info",
            detail="jsonschema not installed; skipped.",
        )
    inst_schema_p = PROJECT_ROOT / "schemas/knowledge/instruction_pair.schema.json"
    pref_schema_p = PROJECT_ROOT / "schemas/knowledge/preference_pair.schema.json"
    inst_invalid: List[str] = []
    pref_invalid: List[str] = []
    if inst_schema_p.exists():
        schema = json.loads(inst_schema_p.read_text())
        for p in inst:
            try:
                jsonschema.validate(p, schema)
            except jsonschema.ValidationError as exc:
                inst_invalid.append(f"{p.get('chunk_id','?')}: {exc.message[:80]}")
    if pref_schema_p.exists():
        schema = json.loads(pref_schema_p.read_text())
        for p in pref:
            try:
                jsonschema.validate(p, schema)
            except jsonschema.ValidationError as exc:
                pref_invalid.append(f"{p.get('chunk_id','?')}: {exc.message[:80]}")
    passed = not inst_invalid and not pref_invalid
    return Dimension(
        name="schema_validity",
        passed=passed,
        severity="critical",
        detail=(
            f"{len(inst_invalid)}/{len(inst)} instruction + "
            f"{len(pref_invalid)}/{len(pref)} preference pairs failed "
            f"schema validation."
        ),
        sample=inst_invalid[:3] + pref_invalid[:3],
    )


def _check_unicode(inst: List[Dict], pref: List[Dict]) -> Dimension:
    flagged: List[str] = []
    for p in inst + pref:
        for fld in ("prompt", "completion", "chosen", "rejected"):
            if _has_suspicious_unicode(str(p.get(fld, "") or "")):
                flagged.append(f"{p.get('chunk_id','?')}/{fld}")
                break
    return Dimension(
        name="suspicious_unicode",
        passed=not flagged,
        severity="critical",
        detail=f"{len(flagged)}/{len(inst)+len(pref)} pairs contain zero-width / BiDi / invisible chars.",
        sample=flagged[:3],
    )


def _check_duplicates(inst: List[Dict], pref: List[Dict]) -> Dimension:
    p_hash = Counter(hashlib.sha256(p.get("prompt", "").encode()).hexdigest() for p in inst)
    c_hash = Counter(hashlib.sha256(p.get("completion", "").encode()).hexdigest() for p in inst)
    ch_hash = Counter(hashlib.sha256(p.get("chosen", "").encode()).hexdigest() for p in pref)
    dup_p = sum(c - 1 for c in p_hash.values() if c > 1)
    dup_c = sum(c - 1 for c in c_hash.values() if c > 1)
    dup_ch = sum(c - 1 for c in ch_hash.values() if c > 1)
    collisions = sum(1 for p in pref if p.get("chosen") == p.get("rejected"))
    passed = dup_p == 0 and dup_c == 0 and dup_ch == 0 and collisions == 0
    return Dimension(
        name="duplicates",
        passed=passed,
        severity="critical",
        detail=(
            f"{dup_p} duplicate inst prompts; {dup_c} duplicate inst "
            f"completions; {dup_ch} duplicate preference chosen; "
            f"{collisions} chosen==rejected collisions."
        ),
    )


def _check_injection(inst: List[Dict], pref: List[Dict]) -> Dimension:
    hits: List[str] = []
    for p in inst + pref:
        text = " ".join(str(p.get(f, "") or "") for f in ("prompt", "completion", "chosen", "rejected"))
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                hits.append(f"{p.get('chunk_id','?')}: {pat.pattern[:40]}")
                break
    return Dimension(
        name="prompt_injection_markers",
        passed=not hits,
        severity="critical",
        detail=f"{len(hits)}/{len(inst)+len(pref)} pairs match an injection pattern.",
        sample=hits[:3],
    )


def _check_length_bounds(inst: List[Dict], pref: List[Dict]) -> Dimension:
    oob: List[str] = []
    for p in inst:
        if not (40 <= len(p.get("prompt", "")) <= 400 and 50 <= len(p.get("completion", "")) <= 600):
            oob.append(f"{p.get('chunk_id','?')} inst")
    for p in pref:
        if not (
            40 <= len(p.get("prompt", "")) <= 400
            and 50 <= len(p.get("chosen", "")) <= 600
            and 50 <= len(p.get("rejected", "")) <= 600
        ):
            oob.append(f"{p.get('chunk_id','?')} pref")
    return Dimension(
        name="length_bounds",
        passed=not oob,
        severity="critical",
        detail=f"{len(oob)}/{len(inst)+len(pref)} pairs out of [40-400 prompt / 50-600 completion-or-chosen-or-rejected] range.",
        sample=oob[:3],
    )


def _check_diversity(inst: List[Dict]) -> Dimension:
    """Mirrors the SynthesisDiversityValidator thresholds. Warning-level
    here because production gate already enforces critical."""
    if not inst:
        return Dimension(
            name="template_diversity", passed=True, severity="info",
            detail="No instruction pairs.",
        )
    templates = Counter(p.get("template_id", "?") for p in inst)
    top3_share = sum(c for _, c in templates.most_common(3)) / len(inst)
    top1_share = templates.most_common(1)[0][1] / len(inst)
    distinct = len(templates)
    issues = []
    if top3_share > 0.60:
        issues.append(f"top-3 share {100*top3_share:.0f}% > 60%")
    if top1_share > 0.35:
        issues.append(f"top-1 share {100*top1_share:.0f}% > 35%")
    if distinct < 8:
        issues.append(f"only {distinct} distinct templates (<8)")
    return Dimension(
        name="template_diversity",
        passed=not issues,
        severity="warning",
        detail=(
            f"distinct={distinct}, top-1={100*top1_share:.0f}%, "
            f"top-3={100*top3_share:.0f}%; "
            + ("; ".join(issues) if issues else "ok")
        ),
    )


def _check_bloom_distribution(inst: List[Dict]) -> Dimension:
    if not inst:
        return Dimension(name="bloom_distribution", passed=True, severity="info", detail="No pairs.")
    blooms = Counter(p.get("bloom_level", "?") for p in inst)
    top2 = sum(c for _, c in blooms.most_common(2))
    share = top2 / len(inst)
    distinct = len([k for k, v in blooms.items() if v > 0])
    return Dimension(
        name="bloom_distribution",
        passed=share <= 0.80 and distinct >= 4,
        severity="warning",
        detail=(
            f"{dict(blooms)}; top-2 share {100*share:.0f}% (warn >80%); "
            f"distinct levels {distinct} (warn <4)."
        ),
    )


def _check_abstention_coverage(inst: List[Dict]) -> Dimension:
    """Wave 124 (audit 2026-04-30 follow-up). Counts pairs with
    ``content_type="abstention_probe"``. Warning when 0 — the cc07cc76
    corpus had no abstention pairs and scored hallucination_rate=0.63
    on the eval. Pass-through informational signal when >0 so an
    operator sees the cohort size before training."""
    count = sum(1 for p in inst if p.get("content_type") == "abstention_probe")
    if count == 0:
        return Dimension(
            name="abstention_coverage",
            passed=False,
            severity="warning",
            detail=(
                "0 abstention_probe pairs in instruction_pairs.jsonl. "
                "Wave 124 fix: re-run synthesis with --with-abstention "
                "to teach the model to say 'the source does not "
                "establish X'. Closes the cc07cc76 hallucination_rate"
                "=0.63 regression."
            ),
        )
    return Dimension(
        name="abstention_coverage",
        passed=True,
        severity="info",
        detail=(
            f"{count}/{len(inst)} pairs are abstention_probe "
            f"({100 * count / max(len(inst), 1):.1f}%). Cohort size "
            f"surfaced for operator visibility."
        ),
    )


def _check_schema_translation_coverage(inst: List[Dict]) -> Dimension:
    """Wave 124 (audit 2026-04-30 follow-up). Counts pairs with
    ``content_type="schema_translation"`` and verifies all 6 RDF/SHACL
    surface forms are covered. Warning when any of the 6 is uncovered
    — schema-to-English bridge gaps drive faithfulness=0.37 on the
    cc07cc76 corpus."""
    expected = {
        "sh:datatype", "sh:class", "sh:NodeShape", "sh:PropertyShape",
        "rdfs:subClassOf", "owl:sameAs",
    }
    seen: set = set()
    total = 0
    for p in inst:
        if p.get("content_type") != "schema_translation":
            continue
        total += 1
        tags = p.get("concept_tags") or []
        for t in tags:
            if t in expected:
                seen.add(t)
    missing = expected - seen
    passed = total > 0 and not missing
    if total == 0:
        detail = (
            "0 schema_translation pairs in instruction_pairs.jsonl. "
            "Wave 124 fix: re-run synthesis with "
            "--with-schema-translation to bridge formal CURIEs "
            "(sh:datatype, rdfs:subClassOf, owl:sameAs, ...) to "
            "plain-English meanings."
        )
    elif missing:
        detail = (
            f"{total} schema_translation pairs present, but "
            f"{len(missing)}/6 surface forms uncovered: "
            f"{sorted(missing)}. Hand-curated table in "
            f"schema_translation_generator.py may need an entry."
        )
    else:
        detail = (
            f"{total} schema_translation pairs cover all 6 "
            f"RDF/SHACL surface forms."
        )
    return Dimension(
        name="schema_translation_coverage",
        passed=passed,
        severity="warning",
        detail=detail,
        sample=sorted(missing) if missing else None,
    )


def _check_citation_coverage(inst: List[Dict]) -> Dimension:
    """Wave 124 (audit 2026-04-30 follow-up). Counts pairs with
    ``requires_source_citation=True``. Warning when the rate is below
    10% of total pairs — the cc07cc76 corpus had 0% citation-trained
    pairs (only --instruction-variants-per-chunk=3 emits the citation
    variant), and the audit gate did not exist to flag it."""
    if not inst:
        return Dimension(
            name="citation_coverage",
            passed=True,
            severity="info",
            detail="No instruction pairs.",
        )
    citations = sum(1 for p in inst if p.get("requires_source_citation"))
    rate = citations / len(inst)
    threshold = 0.10
    return Dimension(
        name="citation_coverage",
        passed=rate >= threshold,
        severity="warning",
        detail=(
            f"{citations}/{len(inst)} pairs require source citation "
            f"({100 * rate:.1f}%). Threshold: >={100 * threshold:.0f}%. "
            f"The cc07cc76 corpus shipped 0% citation-trained pairs; "
            f"raise --instruction-variants-per-chunk to 3 to emit the "
            f"citation variant."
        ),
    )


def _check_paraphrase_quality(inst: List[Dict]) -> Dimension:
    if not inst:
        return Dimension(name="paraphrase_quality", passed=True, severity="info", detail="No pairs.")
    fb = sum(1 for p in inst if p.get("paraphrase_fallback_reason"))
    inj = sum(
        1 for p in inst
        if p.get("preserve_tokens_injected") or p.get("preserve_tokens_injected_prompt")
    )
    natural = len(inst) - fb
    fb_rate = fb / len(inst)
    return Dimension(
        name="paraphrase_quality",
        passed=fb_rate <= 0.50,
        severity="warning",
        detail=(
            f"Fallback fired: {fb}/{len(inst)} ({100*fb_rate:.0f}%); "
            f"force-inject fired: {inj}/{len(inst)}; "
            f"natural preservation: {natural}/{len(inst)}. "
            f"Warn at >50% fallback (likely retries=1 smoke; full run "
            f"with retries=3 should be lower)."
        ),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_audit(
    course_dir: Path, smoke: bool = False,
    leak_rate_threshold: float = DEFAULT_LEAK_RATE_THRESHOLD,
    leak_span_chars: int = DEFAULT_LEAK_SPAN_CHARS,
) -> AuditReport:
    inst_name = "smoke_instruction_pairs.jsonl" if smoke else "instruction_pairs.jsonl"
    pref_name = "smoke_preference_pairs.jsonl" if smoke else "preference_pairs.jsonl"
    inst_path = course_dir / "training_specs" / inst_name
    pref_path = course_dir / "training_specs" / pref_name
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    if not inst_path.exists():
        raise FileNotFoundError(f"{inst_path} not found")
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found")

    inst = _load_jsonl(inst_path)
    pref = _load_jsonl(pref_path) if pref_path.exists() else []
    chunks_list = _load_jsonl(chunks_path)
    chunks = {
        str(c.get("id") or c.get("chunk_id") or ""): str(c.get("text") or "")
        for c in chunks_list
    }

    report = AuditReport(
        course_dir=str(course_dir),
        instruction_count=len(inst),
        preference_count=len(pref),
        chunks_count=len(chunks),
    )
    report.dimensions = [
        _check_assessment_scaffolding(inst, pref),
        _check_verbatim_leakage(inst, pref, chunks, leak_rate_threshold, leak_span_chars),
        _check_legacy_scaffold(inst),
        _check_phrasing_distribution(inst, _PROMPT_REFERENCE_PHRASINGS, "prompt", 0.60),
        _check_phrasing_distribution(inst, _COMPLETION_REFERENCE_PHRASINGS, "completion", 0.60),
        _check_schema(inst, pref),
        _check_unicode(inst, pref),
        _check_duplicates(inst, pref),
        _check_injection(inst, pref),
        _check_length_bounds(inst, pref),
        _check_diversity(inst),
        _check_bloom_distribution(inst),
        _check_paraphrase_quality(inst),
        # Wave 124 (audit 2026-04-30 follow-up).
        _check_abstention_coverage(inst),
        _check_schema_translation_coverage(inst),
        _check_citation_coverage(inst),
    ]
    return report


def format_report_text(report: AuditReport) -> str:
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append(f"Audit report — {report.course_dir}")
    lines.append("=" * 64)
    lines.append(
        f"Pairs: {report.instruction_count} instruction, "
        f"{report.preference_count} preference; "
        f"corpus: {report.chunks_count} chunks"
    )
    lines.append("")
    crit_pass = sum(1 for d in report.dimensions if d.severity == "critical" and d.passed)
    crit_total = sum(1 for d in report.dimensions if d.severity == "critical")
    warn_pass = sum(1 for d in report.dimensions if d.severity == "warning" and d.passed)
    warn_total = sum(1 for d in report.dimensions if d.severity == "warning")
    lines.append(f"Critical dimensions: {crit_pass}/{crit_total} pass")
    lines.append(f"Warning dimensions:  {warn_pass}/{warn_total} pass")
    lines.append("")
    for d in report.dimensions:
        status = "PASS" if d.passed else "FAIL"
        marker = {"critical": "[CRIT]", "warning": "[WARN]", "info": "[INFO]"}[d.severity]
        lines.append(f"{marker} {status}  {d.name}")
        lines.append(f"        {d.detail}")
        if d.sample and not d.passed:
            for s in d.sample:
                lines.append(f"        - {s}")
    lines.append("")
    if report.overall_passed:
        lines.append("OVERALL: PASS — no critical poisoning vector hit. Training-ready.")
    else:
        lines.append(
            f"OVERALL: FAIL — {len(report.critical_failures)} critical "
            f"dimension(s) failing. Do not train on this output."
        )
    return "\n".join(lines) + "\n"


def format_report_json(report: AuditReport) -> str:
    payload = {
        "course_dir": report.course_dir,
        "instruction_count": report.instruction_count,
        "preference_count": report.preference_count,
        "chunks_count": report.chunks_count,
        "overall_passed": report.overall_passed,
        "dimensions": [asdict(d) for d in report.dimensions],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "10-dimension poisoning audit for synthesised training pairs. "
            "Run after a full synthesis pass before launching QLoRA training."
        ),
    )
    ap.add_argument(
        "--course", required=True,
        help="Course directory (e.g. LibV2/courses/rdf-shacl-551-2).",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Read smoke_*.jsonl instead of canonical pair files.",
    )
    ap.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format (default: text).",
    )
    ap.add_argument(
        "--leak-rate-threshold", type=float, default=DEFAULT_LEAK_RATE_THRESHOLD,
        help=f"Verbatim-leak rate threshold (default {DEFAULT_LEAK_RATE_THRESHOLD}).",
    )
    ap.add_argument(
        "--leak-span-chars", type=int, default=DEFAULT_LEAK_SPAN_CHARS,
        help=f"Verbatim-leak min span chars (default {DEFAULT_LEAK_SPAN_CHARS}).",
    )
    args = ap.parse_args(argv)

    course_dir = Path(args.course)
    if not course_dir.exists():
        print(f"error: course directory not found: {course_dir}", file=sys.stderr)
        return 2

    try:
        report = run_audit(
            course_dir, smoke=args.smoke,
            leak_rate_threshold=args.leak_rate_threshold,
            leak_span_chars=args.leak_span_chars,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        sys.stdout.write(format_report_json(report))
    else:
        sys.stdout.write(format_report_text(report))

    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
