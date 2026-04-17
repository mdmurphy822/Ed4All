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

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

try:  # Optional import — the validator can be used standalone in tests.
    from MCP.hardening.validation_gates import GateIssue, GateResult
except Exception:  # pragma: no cover - MCP harness absent in unit-test envs.
    GateIssue = None  # type: ignore
    GateResult = None  # type: ignore


_CLAIM_TABLE: List[Tuple[re.Pattern, str, int, str]] = [
    (
        re.compile(r"\b(\d+)\s+success\s+criteria\b", re.IGNORECASE),
        "wcag_2_2_sc_count",
        86,
        "W3C WCAG 2.2 Recommendation lists 86 success criteria.",
    ),
    (
        re.compile(
            r"\b(\d+)\s+applicable\s+WCAG[^.]{0,40}Section\s*508\b",
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
                if observed != expected:
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
        chunks = inputs.get("chunks", []) or []
        issues: List[Any] = []
        total_flags = 0
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
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warnings never block
            score=1.0 if total_flags == 0 else max(0.0, 1.0 - (total_flags / max(len(chunks), 1))),
            issues=issues,
            execution_time_ms=int((time.time() - start) * 1000),
        )
