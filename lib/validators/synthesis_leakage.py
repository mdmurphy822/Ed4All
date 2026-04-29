"""Wave 121 — SynthesisLeakageValidator.

Pre-training gate that detects verbatim chunk-text leakage in the
synthesised training pairs. The 2026-04-29 smoke audit on
rdf-shacl-551-2 found 11/20 (55%) instruction completions contained
≥50-char verbatim spans copied from ``chunk.text`` — training on
that data would teach the model to memorize the source corpus
instead of paraphrasing / applying concepts. The instruction-factory
prompt-side leakage check did not cover the completion side; this
validator is the runtime backstop that catches the regression class
even after factory-level fixes land.

Default threshold: 5% of pairs may carry a leak before the gate
fails (covers occasional tolerable overlap on short technical
phrases that legitimately appear verbatim in both corpus and
completion). Override via gate inputs.thresholds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

# Same threshold the factories use. A 50-char span is long enough to
# represent a genuine quote of source material rather than coincidental
# n-gram overlap on short phrases like "is the".
DEFAULT_LEAK_SPAN_CHARS = 50
DEFAULT_LEAK_RATE_THRESHOLD = 0.05


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

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "synthesis_leakage")
        course_dir_raw = inputs.get("course_dir")
        if not course_dir_raw:
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
        chunks_path = course_dir / "corpus" / "chunks.jsonl"
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
                        f"instruction_pairs.jsonl not found at {inst_path}; "
                        "run the synthesis phase before the leakage gate."
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

        passed = not [i for i in issues if i.severity == "critical"]
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - len(leaked) / max(total, 1)),
            issues=issues,
        )
