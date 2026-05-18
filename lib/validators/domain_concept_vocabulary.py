"""Stage-3 domain-concept vocabulary validator.

Three-stage large-LLM textbook synthesis architecture, Wave A — see
``plans/textbook-llm-synthesis-3stage-2026-05.md`` §8 row 2.

``DomainConceptVocabularyValidator`` gates the
``domain_concept_vocabulary.json`` artifact emitted by Stage 3
(``_run_concept_extraction``) when ``TEXTBOOK_SYNTHESIS_PROVIDER`` is
set. Floor checks (plan §8):

1. ``>= _MIN_CONCEPTS`` (default 8 — matches ``ConceptGraphValidator``'s
   ≥10-node floor with co-occurrence-filter headroom) — code
   ``VOCAB_THIN``.
2. every concept carries a non-empty ``canonical`` — code
   ``VOCAB_CONCEPT_NO_CANONICAL``.
3. ``concept_count == len(concepts)`` — code ``VOCAB_COUNT_MISMATCH``.
4. ``chapter_synthesis_failures`` count surfaced — code
   ``VOCAB_PARTIAL_COVERAGE`` (warning-only, informational; NOT a hard
   fail per §5.4 failure-isolation contract).

**Graceful-degrade contract (the ``ABCD_MISSING`` pattern).** When no
vocabulary artifact is supplied — a default-off run where
``TEXTBOOK_SYNTHESIS_PROVIDER`` is unset and no
``domain_concept_vocabulary.json`` was written — the validator returns a
no-op ``passed=True``. Default-off runs are unaffected.

Day-1 severity is **warning**. The validator emits warning-severity
issues so the gate logs but does not block.

Inputs contract:

* ``inputs["domain_concept_vocabulary"]`` — the in-memory vocabulary
  dict. Either this or ``inputs["domain_concept_vocabulary_path"]``
  must be supplied; absence is the graceful-degrade no-op.
* ``inputs["domain_concept_vocabulary_path"]`` — alternative; an
  on-disk ``domain_concept_vocabulary.json`` the validator loads.
* ``inputs["decision_capture"]`` — optional ``DecisionCapture``; the
  validator emits one ``content_structure_check`` event per run.
* ``inputs["gate_id"]`` — optional gate-ID override.

Cross-references:

* ``schemas/knowledge/domain_concept_vocabulary.schema.json`` — the
  canonical shape this validator's floor checks mirror.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


#: Concept-count floor. Matches ConceptGraphValidator's ≥10-node floor
#: with co-occurrence-filter headroom (a few seeded concepts get
#: filtered out before the graph build). Plan §8 row 2.
_MIN_CONCEPTS: int = 8

#: Cap emitted issues so a uniformly-broken vocabulary doesn't drown
#: the report.
_ISSUE_LIST_CAP: int = 50

#: Validator-capture decision_type — an EXISTING enum member.
_DECISION_TYPE: str = "content_structure_check"


def _emit_decision(
    capture: Any,
    *,
    decision: str,
    rationale: str,
    context: Optional[str] = None,
) -> None:
    """Emit one DecisionCapture event, swallowing capture-side errors."""
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.warning(
            "DomainConceptVocabularyValidator: "
            "decision_capture.log_decision failed: %s",
            exc,
        )


def _coerce_vocabulary(
    inputs: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve the vocabulary dict from ``inputs``.

    Priority: ``inputs['domain_concept_vocabulary']`` >
    ``inputs['domain_concept_vocabulary_path']``. Returns
    ``(vocabulary, error_issue)``; ``error_issue`` is non-None only on a
    structural failure.
    """
    explicit = inputs.get("domain_concept_vocabulary")
    if explicit is not None:
        if isinstance(explicit, Mapping):
            return dict(explicit), None
        return None, GateIssue(
            severity="critical",
            code="VOCAB_MALFORMED",
            message=(
                f"inputs['domain_concept_vocabulary'] is "
                f"{type(explicit).__name__!r}, not a mapping."
            ),
        )

    path_raw = inputs.get("domain_concept_vocabulary_path")
    if not path_raw:
        return None, None

    p = Path(path_raw)
    if not p.exists():
        return None, GateIssue(
            severity="critical",
            code="VOCAB_PATH_MISSING",
            message=(
                f"domain_concept_vocabulary_path {str(p)!r} does not "
                f"exist; cannot audit Stage-3 concept vocabulary."
            ),
            location=str(p),
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, GateIssue(
            severity="critical",
            code="VOCAB_PATH_UNREADABLE",
            message=(
                f"Failed to parse domain_concept_vocabulary_path "
                f"{str(p)!r}: {exc.__class__.__name__}: {exc}"
            ),
            location=str(p),
        )
    if not isinstance(data, Mapping):
        return None, GateIssue(
            severity="critical",
            code="VOCAB_MALFORMED",
            message=(
                f"domain_concept_vocabulary_path {str(p)!r} did not "
                f"parse to a JSON object (got {type(data).__name__!r})."
            ),
            location=str(p),
        )
    return dict(data), None


class DomainConceptVocabularyValidator:
    """Stage-3 domain-concept vocabulary gate.

    Wired into ``textbook_to_course::concept_extraction::
    domain_concept_vocabulary`` (gate wiring is Wave C, not this wave).
    Skips-with-pass when no vocabulary artifact is supplied so
    default-off runs are unaffected.
    """

    name = "domain_concept_vocabulary"
    version = "0.1.0"  # Wave A — three-stage textbook synthesis

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        vocabulary, err = _coerce_vocabulary(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Graceful-degrade: no vocabulary artifact → no-op pass. A
        # default-off run (TEXTBOOK_SYNTHESIS_PROVIDER unset) writes no
        # domain_concept_vocabulary.json.
        if vocabulary is None:
            _emit_decision(
                capture,
                decision=(
                    "skip-with-pass: domain_concept_vocabulary absent"
                ),
                rationale=(
                    "DomainConceptVocabularyValidator received no "
                    "vocabulary artifact (neither "
                    "`domain_concept_vocabulary` nor "
                    "`domain_concept_vocabulary_path`) — this is a "
                    "default-off run (TEXTBOOK_SYNTHESIS_PROVIDER "
                    "unset). Returning a no-op pass per the "
                    "ABCD_MISSING graceful-degrade contract."
                ),
                context="vocabulary_absent=true",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        issues: List[GateIssue] = []

        concepts_raw = vocabulary.get("concepts")
        concepts: List[Any] = (
            concepts_raw if isinstance(concepts_raw, list) else []
        )
        concept_dicts = [c for c in concepts if isinstance(c, Mapping)]

        # --- Floor check 1: concept-count floor ------------------------
        if len(concept_dicts) < _MIN_CONCEPTS:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VOCAB_THIN",
                    message=(
                        f"Domain-concept vocabulary carries "
                        f"{len(concept_dicts)} concept(s); the floor is "
                        f"{_MIN_CONCEPTS}. A thin vocabulary will seed "
                        f"too few co-occurrence-graph DomainConcept "
                        f"nodes."
                    ),
                    location="concepts",
                    suggestion=(
                        "Re-run Stage 3 — the per-chapter concept calls "
                        "should jointly yield at least "
                        f"{_MIN_CONCEPTS} concepts for a real textbook."
                    ),
                )
            )

        # --- Floor check 2: every concept has a non-empty canonical ----
        for idx, concept in enumerate(concept_dicts):
            canonical = concept.get("canonical")
            if not isinstance(canonical, str) or not canonical.strip():
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="VOCAB_CONCEPT_NO_CANONICAL",
                            message=(
                                f"concepts[{idx}] carries no non-empty "
                                f"`canonical` (got {canonical!r}); the "
                                f"concept cannot be compiled into a "
                                f"domain-concept seed."
                            ),
                            location=f"concepts[{idx}]",
                            suggestion=(
                                "Every concept must carry a non-empty "
                                "`canonical` string (run through "
                                "canonical_slug at emit)."
                            ),
                        )
                    )

        # --- Floor check 3: concept_count == len(concepts) -------------
        declared_count = vocabulary.get("concept_count")
        if isinstance(declared_count, bool) or not isinstance(
            declared_count, int
        ):
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VOCAB_COUNT_MISMATCH",
                    message=(
                        f"`concept_count` is {declared_count!r}, not an "
                        f"integer; cannot reconcile against "
                        f"len(concepts)={len(concepts)}."
                    ),
                    location="concept_count",
                )
            )
        elif declared_count != len(concepts):
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VOCAB_COUNT_MISMATCH",
                    message=(
                        f"`concept_count`={declared_count} does not "
                        f"equal len(concepts)={len(concepts)}; the "
                        f"vocabulary artifact's asserted count is "
                        f"out of sync with its `concepts` array."
                    ),
                    location="concept_count",
                    suggestion=(
                        "Stage 3 must stamp concept_count = "
                        "len(concepts) at emit."
                    ),
                )
            )

        # --- Floor check 4: chapter_synthesis_failures surfaced --------
        # Informational warning ONLY — §5.4 failure isolation makes
        # partial coverage an expected degraded state, not a hard fail.
        failures_raw = vocabulary.get("chapter_synthesis_failures")
        failures: List[Any] = (
            failures_raw if isinstance(failures_raw, list) else []
        )
        if failures:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VOCAB_PARTIAL_COVERAGE",
                    message=(
                        f"{len(failures)} chapter(s) failed Stage-3 "
                        f"concept synthesis and contributed no concepts: "
                        f"{[str(f) for f in failures]}. The vocabulary "
                        f"reflects partial chapter coverage (§5.4 "
                        f"failure isolation)."
                    ),
                    location="chapter_synthesis_failures",
                    suggestion=(
                        "Informational — partial coverage is an "
                        "expected degraded state. Re-run Stage 3 if "
                        "the failed chapters carry important concepts."
                    ),
                )
            )

        # Day-1 warning severity: warnings never fail-close the gate.
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        warning_count = sum(1 for i in issues if i.severity == "warning")

        _emit_decision(
            capture,
            decision=(
                f"Audited Stage-3 domain-concept vocabulary: "
                f"{len(concept_dicts)} concept(s), "
                f"{len(failures)} chapter synthesis failure(s); "
                f"{warning_count} warning issue(s)."
            ),
            rationale=(
                f"DomainConceptVocabularyValidator ran the §8-row-2 "
                f"floor checks: concept-count >= {_MIN_CONCEPTS}, every "
                f"concept has a non-empty canonical, concept_count == "
                f"len(concepts), and surfaced "
                f"{len(failures)} chapter_synthesis_failures. The "
                f"vocabulary carried {len(concept_dicts)} concept(s); "
                f"emitted {warning_count} warning issue(s); "
                f"passed={passed}."
            ),
            context=(
                f"concepts={len(concept_dicts)}; "
                f"chapter_failures={len(failures)}; "
                f"warnings={warning_count}; passed={passed}"
            ),
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else 0.0,
            issues=issues,
        )


__all__ = [
    "DomainConceptVocabularyValidator",
]
