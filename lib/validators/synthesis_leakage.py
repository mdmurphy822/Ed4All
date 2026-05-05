"""Wave 121 + 122 — SynthesisLeakageValidator.

Pre-training gate that detects two distinct contamination vectors in
the synthesised training pairs:

1. **Verbatim chunk-text leakage** (Wave 121). Pairs that contain
   ≥50-char spans copied from ``chunk.text``. The 2026-04-29 audit
   found 11/20 (55%) instruction completions leaked through
   ``_build_completion``'s summary path; training on that data would
   produce a corpus-memorisation adapter.

2. **Assessment-outline scaffolding** (Wave 122). Pairs that carry
   structured patterns like ``Question 1 (CO-07, Bloom: Understand).
   Question 2 (CO-07, Bloom: Apply)...``. The 2026-04-29 follow-up
   audit (codex finding M1) caught chunk_00066 leaking this through
   chunk metadata (not chunk.text, so it bypassed the Wave 121
   verbatim check). Training on these would teach the model to emit
   quiz-outline debris in normal explanations.

Default thresholds: 5% of pairs may carry verbatim leak; 0% may
carry assessment-scaffolding (zero-tolerance — this is structural
contamination). Override via gate inputs.thresholds.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    n_pairs_audited: int,
    verbatim_leak_count: int,
    assessment_scaffold_count: int,
    rate_threshold: float,
    assessment_threshold: float,
    span_threshold: int,
) -> None:
    """Emit one ``synthesis_leakage_check`` decision per validate() call.

    H3 Wave W4: every leakage-fail / scaffold-fail / pass / missing-input
    path emits one event so post-hoc replay can distinguish the two
    contamination vectors (verbatim chunk-text leakage vs assessment-
    outline scaffolding) without re-running the gate.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    if n_pairs_audited > 0:
        verbatim_rate = verbatim_leak_count / n_pairs_audited
        scaffold_rate = assessment_scaffold_count / n_pairs_audited
    else:
        verbatim_rate = 0.0
        scaffold_rate = 0.0
    rationale = (
        f"synthesis_leakage gate verdict: n_pairs_audited="
        f"{n_pairs_audited}, verbatim_leak_count={verbatim_leak_count} "
        f"({verbatim_rate:.4f} rate), "
        f"assessment_scaffold_count={assessment_scaffold_count} "
        f"({scaffold_rate:.4f} rate); thresholds=("
        f"verbatim_rate={rate_threshold:.4f}, "
        f"assessment_rate={assessment_threshold:.4f}, "
        f"span_chars={span_threshold}); failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "n_pairs_audited": int(n_pairs_audited),
        "verbatim_leak_count": int(verbatim_leak_count),
        "assessment_scaffold_count": int(assessment_scaffold_count),
        "verbatim_leak_rate": float(verbatim_rate),
        "assessment_scaffold_rate": float(scaffold_rate),
        "rate_threshold": float(rate_threshold),
        "assessment_threshold": float(assessment_threshold),
        "span_threshold": int(span_threshold),
        "passed": bool(passed),
        "failure_code": code,
    }
    try:
        capture.log_decision(
            decision_type="synthesis_leakage_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "synthesis_leakage_check: %s",
            exc,
        )

# Same threshold the factories use. A 50-char span is long enough to
# represent a genuine quote of source material rather than coincidental
# n-gram overlap on short phrases like "is the".
DEFAULT_LEAK_SPAN_CHARS = 50
DEFAULT_LEAK_RATE_THRESHOLD = 0.05
# Wave 122: assessment-scaffolding has zero tolerance — even a single
# pair carrying the pattern is structural contamination. Tunable via
# gate config, but default 0.0 is the right default.
DEFAULT_ASSESSMENT_SCAFFOLD_THRESHOLD = 0.0

_ASSESSMENT_SCAFFOLD_PATTERNS = [
    re.compile(r'\bQuestion\s+\d+\s*\(\s*[A-Z]+-\d+\s*,?\s*Bloom\s*:', re.IGNORECASE),
    re.compile(r'\b(?:Q|Item)\s*\d+\s*\(\s*[A-Z]+-\d+\b'),
    re.compile(r'\b(?:Bloom|Cognitive)\s*:\s*(?:Remember|Understand|Apply|Analyze|Evaluate|Create)\)', re.IGNORECASE),
]


def _contains_assessment_scaffolding(text: str) -> Optional[str]:
    """Return the first matched scaffolding fragment, or None."""
    if not text:
        return None
    for pat in _ASSESSMENT_SCAFFOLD_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def _contains_verbatim_span(
    candidate: str, chunk_text: str, max_span: int = DEFAULT_LEAK_SPAN_CHARS,
) -> Optional[str]:
    """Return the first ≥``max_span``-char window in ``candidate`` that
    also appears in ``chunk_text``. None when no such span exists."""
    if not candidate or not chunk_text:
        return None
    p = candidate.lower()
    c = chunk_text.lower()
    if len(p) < max_span or len(c) < max_span:
        return None
    for i in range(0, len(p) - max_span + 1):
        window = p[i:i + max_span]
        if window in c:
            return candidate[i:i + max_span]
    return None


class SynthesisLeakageValidator:
    name = "synthesis_leakage"
    version = "1.0.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "synthesis_leakage")
        capture = inputs.get("decision_capture") or self._decision_capture
        course_dir_raw = inputs.get("course_dir")
        if not course_dir_raw:
            _emit_decision(
                capture, passed=False, code="MISSING_INPUTS",
                n_pairs_audited=0, verbatim_leak_count=0,
                assessment_scaffold_count=0,
                rate_threshold=DEFAULT_LEAK_RATE_THRESHOLD,
                assessment_threshold=DEFAULT_ASSESSMENT_SCAFFOLD_THRESHOLD,
                span_threshold=DEFAULT_LEAK_SPAN_CHARS,
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
                        "SynthesisLeakageValidator requires course_dir "
                        "input."
                    ),
                )],
            )
        course_dir = Path(course_dir_raw)
        inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
        # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
        from lib.libv2_storage import resolve_imscc_chunks_path
        chunks_path = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
        if not inst_path.exists():
            _emit_decision(
                capture, passed=False, code="INSTRUCTION_PAIRS_NOT_FOUND",
                n_pairs_audited=0, verbatim_leak_count=0,
                assessment_scaffold_count=0,
                rate_threshold=DEFAULT_LEAK_RATE_THRESHOLD,
                assessment_threshold=DEFAULT_ASSESSMENT_SCAFFOLD_THRESHOLD,
                span_threshold=DEFAULT_LEAK_SPAN_CHARS,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INSTRUCTION_PAIRS_NOT_FOUND",
                    message=(
                        f"instruction_pairs.jsonl not found at {inst_path}; "
                        "run the synthesis phase before the leakage gate."
                    ),
                    location=str(inst_path),
                )],
            )
        if not chunks_path.exists():
            _emit_decision(
                capture, passed=False, code="CHUNKS_NOT_FOUND",
                n_pairs_audited=0, verbatim_leak_count=0,
                assessment_scaffold_count=0,
                rate_threshold=DEFAULT_LEAK_RATE_THRESHOLD,
                assessment_threshold=DEFAULT_ASSESSMENT_SCAFFOLD_THRESHOLD,
                span_threshold=DEFAULT_LEAK_SPAN_CHARS,
            )
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
                        "leakage gate needs the source corpus to compare "
                        "against."
                    ),
                    location=str(chunks_path),
                )],
            )

        thresholds = inputs.get("thresholds", {}) or {}
        rate_threshold = float(
            thresholds.get("leak_rate_threshold", DEFAULT_LEAK_RATE_THRESHOLD)
        )
        span_threshold = int(
            thresholds.get("leak_span_chars", DEFAULT_LEAK_SPAN_CHARS)
        )
        assessment_threshold = float(
            thresholds.get(
                "assessment_scaffold_rate_threshold",
                DEFAULT_ASSESSMENT_SCAFFOLD_THRESHOLD,
            )
        )

        chunks_by_id: Dict[str, str] = {}
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
                if cid:
                    chunks_by_id[cid] = str(chunk.get("text") or "")

        total = 0
        leaked: List[Dict[str, Any]] = []
        scaffolded: List[Dict[str, Any]] = []
        with inst_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                cid = str(row.get("chunk_id") or "")
                # Wave 122: assessment-scaffolding check runs even when
                # chunk_text isn't available (the contamination doesn't
                # require comparison against chunk source).
                for field in ("prompt", "completion"):
                    fragment = _contains_assessment_scaffolding(
                        str(row.get(field) or "")
                    )
                    if fragment:
                        scaffolded.append({
                            "chunk_id": cid,
                            "field": field,
                            "fragment": fragment[:80],
                        })
                        break
                chunk_text = chunks_by_id.get(cid, "")
                if not chunk_text:
                    continue
                for field in ("prompt", "completion"):
                    span = _contains_verbatim_span(
                        str(row.get(field) or ""), chunk_text, span_threshold,
                    )
                    if span:
                        leaked.append({
                            "chunk_id": cid,
                            "field": field,
                            "span_chars": len(span),
                            "span_excerpt": span[:80],
                        })
                        break

        issues: List[GateIssue] = []
        if total == 0:
            issues.append(GateIssue(
                severity="critical",
                code="NO_PAIRS",
                message=(
                    "instruction_pairs.jsonl is empty; nothing to "
                    "validate."
                ),
                location=str(inst_path),
            ))
        else:
            leak_rate = len(leaked) / total
            if leak_rate > rate_threshold:
                excerpts = "; ".join(
                    f"{l['chunk_id']}/{l['field']}: {l['span_excerpt']!r}"
                    for l in leaked[:3]
                )
                issues.append(GateIssue(
                    severity="critical",
                    code="VERBATIM_LEAKAGE_ABOVE_THRESHOLD",
                    message=(
                        f"{len(leaked)}/{total} ({100*leak_rate:.1f}%) "
                        f"instruction pairs contain ≥{span_threshold}-char "
                        f"verbatim spans from chunk.text (threshold "
                        f"{100*rate_threshold:.1f}%). Training on this data "
                        f"would teach corpus memorisation rather than "
                        f"paraphrase / application. First 3 leaks: "
                        f"{excerpts}. Fix: ensure the synthesis factories "
                        f"check completion-side leakage and reject / "
                        f"rewrite leaky completions."
                    ),
                    location=str(inst_path),
                ))
            # Wave 122: assessment-scaffolding contamination check.
            scaffold_rate = len(scaffolded) / total
            if scaffold_rate > assessment_threshold:
                excerpts = "; ".join(
                    f"{s['chunk_id']}/{s['field']}: {s['fragment']!r}"
                    for s in scaffolded[:3]
                )
                issues.append(GateIssue(
                    severity="critical",
                    code="ASSESSMENT_SCAFFOLDING_ABOVE_THRESHOLD",
                    message=(
                        f"{len(scaffolded)}/{total} ({100*scaffold_rate:.1f}%) "
                        f"instruction pairs contain assessment-outline "
                        f"scaffolding patterns (e.g. 'Question N (XX-NN, "
                        f"Bloom: ...)'); threshold "
                        f"{100*assessment_threshold:.1f}%. This is "
                        f"structural contamination — training on these "
                        f"would teach the model to emit quiz-outline "
                        f"debris in normal explanations. First 3: "
                        f"{excerpts}. Fix: factory-level rejection of "
                        f"pairs whose chunk metadata carries assessment "
                        f"scaffolding."
                    ),
                    location=str(inst_path),
                ))

        passed = not [i for i in issues if i.severity == "critical"]

        # H3 W4: surface the first critical issue's code (when any).
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
            n_pairs_audited=total,
            verbatim_leak_count=len(leaked),
            assessment_scaffold_count=len(scaffolded),
            rate_threshold=rate_threshold,
            assessment_threshold=assessment_threshold,
            span_threshold=span_threshold,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - len(leaked) / max(total, 1)),
            issues=issues,
        )
