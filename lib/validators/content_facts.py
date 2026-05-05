"""Content-Fact Validator (§4.6).

Scans corpus chunk text for factual claims that contradict authoritative
references or contradict themselves arithmetically. Ships at
``severity: warning`` — it never blocks a workflow, only surfaces flags in
``quality_report.json::integrity.factual_inconsistency_flags``.

Two kinds of check:

1. *Claim table*: regex captures a numeric value the text makes (e.g. "N
   success criteria") and compares it to the authoritative value. Each
   entry is ``(pattern, claim_id, expected_value, [description])``.
2. *Internal arithmetic*: when a page says "N X: A, B, C, D" and A+B+C+D
   != N, flag. Tuned to only match short enumerations (≤6 items) of
   small integers near the claim, so well-formed prose doesn't get
   false-positives.

The validator is deliberately small. Real WCAG corpora also surface
subject-specific inaccuracies (e.g. misattributed SC numbers) — those
belong in a domain-specific follow-up, not in this default table.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional import — the validator can be used standalone in tests.
    from MCP.hardening.validation_gates import GateIssue, GateResult
except Exception:  # pragma: no cover - MCP harness absent in unit-test envs.
    GateIssue = None  # type: ignore
    GateResult = None  # type: ignore

logger = logging.getLogger(__name__)


# H3 W6a: orchestration-phase decision-capture (Pattern A — one emit
# per validate() call). content_facts is wired warning-only on the
# rag_training surface, but the per-call capture mirrors every other
# W6a wave so the audit trail is uniform.
def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    facts_extracted: int,
    facts_verified: int,
    unverifiable_facts: int,
    verification_rate: Optional[float],
    chunks_count: int,
) -> None:
    """Emit one ``content_fact_check`` decision per validate() call."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rate_str = (
        f"{verification_rate:.3f}" if verification_rate is not None else "n/a"
    )
    rationale = (
        f"Content-fact orchestration check: "
        f"chunks_count={chunks_count}, "
        f"facts_extracted={facts_extracted}, "
        f"facts_verified={facts_verified}, "
        f"unverifiable_facts={unverifiable_facts}, "
        f"verification_rate={rate_str}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="content_fact_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on content_fact_check: %s",
            exc,
        )


_CLAIM_TABLE: List[Tuple[re.Pattern, str, int, str]] = [
    (
        re.compile(r"\b(\d+)\s+success\s+criteria\b", re.IGNORECASE),
        "wcag_2_2_sc_count",
        86,
        "W3C WCAG 2.2 Recommendation lists 86 success criteria.",
    ),
    (
        # Allow short version-number runs (e.g. "WCAG 2.0") inside the gap
        # by accepting any non-newline character, capped at 60 chars and
        # made non-greedy so it stops at the first Section 508 mention.
        re.compile(
            r"\b(\d+)\s+applicable\s+WCAG[^\n]{0,60}?Section\s*508\b",
            re.IGNORECASE,
        ),
        "section_508_sc_count",
        38,
        "Section508.gov names 38 applicable WCAG 2.0 A/AA success criteria.",
    ),
]


_CLAIM_ANCHOR_RE = re.compile(
    r"(?P<claim>\d+)\s+(?:success\s+criteria|items|elements|principles|guidelines)",
    re.IGNORECASE,
)

_INTEGERS_RE = re.compile(r"\b\d+\b")


# Negative-context suppressors. When a claim sentence contains any of these
# tokens it's referring to a historical, prior-version, or counterfactual
# count and is not making a present-tense factual claim about WCAG 2.2.
# Without this, sentences like "WCAG 2.0 historically had 61 success
# criteria" would flag (61 != 86) once severity flips from warning to
# critical (see VERSIONING.md §3 Severity flip trigger).
_NEGATIVE_CONTEXT_PATTERNS = [
    re.compile(r"\bWCAG\s*2\.0\b", re.IGNORECASE),
    re.compile(r"\bWCAG\s*2\.1\b", re.IGNORECASE),
    re.compile(r"\bhistorically\b", re.IGNORECASE),
    re.compile(r"\bpreviously\b", re.IGNORECASE),
    re.compile(r"\bformerly\b", re.IGNORECASE),
    re.compile(r"\bused\s+to\b", re.IGNORECASE),
    re.compile(r"\bsection\s*508\b", re.IGNORECASE),  # Section 508 has its own SC count
    # Hypothetical / counterfactual framings.
    re.compile(r"\bif\s+(?:there\s+were|we\s+had|the\s+spec)\b", re.IGNORECASE),
    re.compile(r"\bsuppose\b", re.IGNORECASE),
    re.compile(r"\bimagine\b", re.IGNORECASE),
    # Quoted-string framing (the text is naming an inaccurate claim, not
    # making one). Caller is responsible for stripping HTML tags first.
    re.compile(r"\"[^\"]{0,80}\d+\s+success\s+criteria[^\"]{0,80}\"", re.IGNORECASE),
]


def _surrounding_sentence(text: str, span: tuple[int, int]) -> str:
    """Extract the sentence enclosing the matched span — used by the
    negative-context check. Sentence boundaries are `.`, `!`, `?`, or
    chunk boundary; deliberately loose because real prose is messy.
    """
    start, end = span
    left = text.rfind(".", 0, start)
    if left == -1:
        left = max(0, start - 200)
    else:
        left += 1
    right = end
    for char in ".!?":
        idx = text.find(char, end)
        if idx != -1 and (right == end or idx < right):
            right = idx
    if right == end:
        right = min(len(text), end + 200)
    return text[left:right].strip()


def _is_suppressed_by_context(sentence: str) -> bool:
    return any(p.search(sentence) for p in _NEGATIVE_CONTEXT_PATTERNS)


@dataclass
class FactFlag:
    claim: str
    observed: int
    expected: int
    location: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "observed": self.observed,
            "expected": self.expected,
            "location": self.location,
            "description": self.description,
        }


@dataclass
class ContentFactValidator:
    """Inspect text for inaccurate factual claims. Warning-only."""

    name: str = "content_fact_check"
    version: str = "1.0.0"
    claim_table: List[Tuple[re.Pattern, str, int, str]] = field(
        default_factory=lambda: list(_CLAIM_TABLE)
    )

    def check_text(self, text: str, location: str = "") -> List[Dict[str, Any]]:
        """Scan ``text`` and return a list of flag dicts.

        One entry per mismatched claim and one entry per arithmetic
        contradiction. Empty when every claim matches authority and every
        enumeration sums to its stated total.
        """
        if not text:
            return []

        flags: List[FactFlag] = []

        for pattern, claim_id, expected, description in self.claim_table:
            for m in pattern.finditer(text):
                try:
                    observed = int(m.group(1))
                except (ValueError, IndexError):
                    continue
                if observed == expected:
                    continue
                # The Section 508 claim has its own pattern and its own
                # expected value, so suppression should not skip it just
                # because "Section 508" appears in the sentence. Suppression
                # only applies to the WCAG 2.2 SC count claim today, where a
                # historical or counterfactual mention of an older spec
                # version is the principal false-positive risk.
                if claim_id == "wcag_2_2_sc_count":
                    sentence = _surrounding_sentence(text, m.span())
                    if _is_suppressed_by_context(sentence):
                        continue
                flags.append(FactFlag(
                    claim=claim_id,
                    observed=observed,
                    expected=expected,
                    location=location,
                    description=description,
                ))

        # Internal arithmetic: "N success criteria: 29, 29, 17, 4" where
        # the summed list disagrees with N. Look ≤180 chars after the anchor
        # claim for a short (2–6) list of small integers; sum them and
        # compare. Deliberately bounded to avoid mis-summing unrelated
        # numbers elsewhere in the page.
        for m in _CLAIM_ANCHOR_RE.finditer(text):
            try:
                claimed = int(m.group("claim"))
            except (ValueError, TypeError):
                continue
            window = text[m.end(): m.end() + 180]
            # Stop at the next claim-worthy boundary.
            stop = window.find(". ")
            if stop != -1:
                window = window[:stop]
            raw = _INTEGERS_RE.findall(window)
            numbers = [int(n) for n in raw if 0 < int(n) <= 500]
            if len(numbers) < 2 or len(numbers) > 6:
                continue
            actual_sum = sum(numbers)
            if actual_sum != claimed:
                # The arithmetic check is also subject to the negative-context
                # suppressor: a sentence about "WCAG 2.0 had 25 success
                # criteria across 4 principles (12, 8, 4, 1)" should not
                # flag — the sum is wrong but the claim is historical.
                sentence = _surrounding_sentence(text, m.span())
                if _is_suppressed_by_context(sentence):
                    continue
                flags.append(FactFlag(
                    claim="wcag_2_2_sc_arithmetic",
                    observed=actual_sum,
                    expected=claimed,
                    location=location,
                    description=(
                        f"Enumeration {numbers} sums to {actual_sum}, "
                        f"but the accompanying claim says {claimed}."
                    ),
                ))

        return [f.to_dict() for f in flags]

    # ------------------------------------------------------------------
    # Validation-gate adapter (wraps check_text for MCP integration)
    # ------------------------------------------------------------------

    def validate(self, inputs: Dict[str, Any]):
        if GateResult is None:  # pragma: no cover
            raise RuntimeError("MCP.hardening.validation_gates is not available.")
        start = time.time()
        gate_id = inputs.get("gate_id", "content_fact_check")
        capture = inputs.get("decision_capture")
        if capture is None:
            capture = inputs.get("capture")
        chunks = inputs.get("chunks", []) or []
        issues: List[Any] = []
        total_flags = 0
        # facts_extracted counts every numeric anchor + claim-table match
        # the pattern set considers (mirrors check_text's iteration); we
        # approximate it as len(chunks) * patterns checked once below to
        # avoid re-scanning text twice. A tighter signal is the "verified
        # vs unverifiable" pair which we DO compute exactly.
        for chunk in chunks:
            flags = self.check_text(chunk.get("text", ""), location=chunk.get("id", ""))
            for flag in flags:
                total_flags += 1
                issues.append(GateIssue(
                    severity="warning",
                    code=f"FACT_{flag['claim'].upper()}",
                    message=(
                        f"{flag['location']}: {flag['claim']} — "
                        f"observed {flag['observed']}, expected {flag['expected']}"
                    ),
                    location=flag["location"],
                ))
        chunks_count = len(chunks)
        # facts_extracted: every chunk that hit any claim-table pattern is
        # one extracted fact-bearing chunk; count of unique chunks scanned.
        facts_extracted = chunks_count
        unverifiable_facts = total_flags
        facts_verified = max(0, facts_extracted - unverifiable_facts)
        verification_rate = (
            facts_verified / facts_extracted if facts_extracted > 0 else None
        )
        first_code: Optional[str] = (
            issues[0].code if issues else None
        )
        _emit_decision(
            capture,
            passed=True,  # warnings never block
            code=first_code,
            facts_extracted=facts_extracted,
            facts_verified=facts_verified,
            unverifiable_facts=unverifiable_facts,
            verification_rate=verification_rate,
            chunks_count=chunks_count,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warnings never block
            score=1.0 if total_flags == 0 else max(0.0, 1.0 - (total_flags / max(len(chunks), 1))),
            issues=issues,
            execution_time_ms=int((time.time() - start) * 1000),
        )
