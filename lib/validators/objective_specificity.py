"""Defect B (objective-synthesis fix W2) — ObjectiveSpecificityValidator.

Vacuous chapter objectives pass every existing course_planning gate. The
``objective_entailment`` gate scores an objective's TRUTH (is the statement
entailed by its cited chunk?); it says nothing about whether the statement names
a concrete, teachable skill. So an objective like "Apply various techniques to
solve real-world problems" — grammatically valid, on-topic enough to entail — sails
through even though, once its Bloom verb and domain-agnostic scaffolding are
removed, NOTHING nameable remains. The 2026-07-06 TO/CO review found ~22 such
vacuous COs in a real full-book set.

This validator closes that gap with three DETERMINISTIC, embedding-FREE checks
over each chapter objective's STATEMENT (composes with, never re-scores,
``objective_entailment`` — truth vs. information):

* **V1 content-residual floor** — tokenize the statement, drop Bloom verbs
  (``lib/ontology/bloom.py::get_all_verbs``), generic stopwords
  (``objective_dedup._SKILL_STOPWORDS``), and the shared filler lexicon
  (``lib/objectives/filler_lexicon.py::filler_tokens``). Fewer than
  ``min_content_residual`` (default 2) content tokens remain → ``OBJECTIVE_VACUOUS``
  (this CO counts toward the headline vacuous rate).
* **V2 vague-object check** — the statement hits a vague-object phrase
  (``filler_lexicon.vague_object_regexes``) AND its residual is below
  ``min_generic_object_residual`` (default 4) → ``OBJECTIVE_GENERIC_OBJECT``
  (a thin objective whose object is a filler phrase like "various concepts").
* **V3 source-anchoring recall** — for a CO that cites chunks, the fraction of its
  residual tokens appearing in ANY cited chunk's text below
  ``min_statement_token_recall`` (default 0.5) → ``OBJECTIVE_UNANCHORED_STATEMENT``
  (a statement whose content words are absent from its own source — catches the
  factually-garbled COs; lexical, orthogonal to NLI entailment).

Gate headline: ``vacuous_rate > max_vacuous_rate`` (default 0.05) →
``OBJECTIVE_VACUOUS_RATE_HIGH``. Thresholds are gate ``config:`` keys (merged into
``inputs`` by the executor), NOT env vars.

Gating: opt-in via ``ED4ALL_OBJECTIVE_SPECIFICITY`` (default OFF → byte-identical
skip-with-pass, a single ``OBJECTIVE_SPECIFICITY_DISABLED`` info issue,
parse-with-fallback — the exact pattern of
``terminal_objective_source_grounding``). When ON, ALL issues are
**warning-severity** day-1 (``passed`` always True) with a
``# TODO(calibration)`` deferred critical-flip after a ≥2-corpus FP measurement.

Inputs (``inputs`` dict; reuses ``_build_chapter_objective_coverage_inputs``):

    synthesized_objectives_path (str) OR synthesized_objectives (dict)
        The course's synthesized objectives doc.
    chunks_by_id (dict) OR chunks / dart_chunks (list) OR dart_chunks_path (str)
        The source-chunk universe (id -> text) for V3. Absent → ``NO_CHUNK_UNIVERSE``
        warning (V3 skipped; V1/V2 still run; ``passed=True``).
    decision_capture / capture
        Injected by the executor. One summary ``validation_result`` decision.

References:
    - ``lib/validators/terminal_objective_source_grounding.py`` — the opt-in
      skip-with-pass gate template + chunk-lookup helper reused here.
    - ``lib/objectives/objective_dedup.py::_skill_keyphrase_tokens`` — the shared
      Bloom-verb/stopword residual tokenizer V1/V3 reuse.
    - ``lib/objectives/filler_lexicon.py`` — the shared filler vocabulary.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.objectives.objective_dedup import _skill_keyphrase_tokens
from lib.objectives.filler_lexicon import filler_tokens, has_vague_object
from lib.ontology.terminal_coverage import flatten_chapter_objectives
from lib.validators.terminal_objective_coverage import _coerce_dict
from lib.validators.terminal_objective_source_grounding import _build_chunks_lookup

logger = logging.getLogger(__name__)

#: Gate-config defaults (overridable via the gate ``config:`` block, merged into
#: inputs by the executor). NOT env vars.
DEFAULT_MIN_CONTENT_RESIDUAL: int = 2
DEFAULT_MIN_GENERIC_OBJECT_RESIDUAL: int = 4
DEFAULT_MIN_STATEMENT_TOKEN_RECALL: float = 0.5
DEFAULT_MAX_VACUOUS_RATE: float = 0.05

#: Cap on the emitted per-objective issue list (mirrors co_terminal_alignment).
_ISSUE_LIST_CAP: int = 50

_DECISION_TYPE: str = "validation_result"

#: Opt-in gate flag. Default OFF → byte-identical skip-with-pass.
_ENV_FLAG: str = "ED4ALL_OBJECTIVE_SPECIFICITY"
_TRUTHY: frozenset = frozenset({"1", "true", "yes", "on"})

_TOKEN_RE = re.compile(r"[a-z]+")


def _specificity_enabled() -> bool:
    """Resolve ``ED4ALL_OBJECTIVE_SPECIFICITY`` (parse-with-fallback → off)."""
    raw = os.environ.get(_ENV_FLAG, "")
    return str(raw).strip().lower() in _TRUTHY


def _statement(obj: Dict[str, Any]) -> str:
    for key in ("statement", "text", "objective", "description"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _content_residual(statement: str) -> Set[str]:
    """The statement's content residual: keyphrase tokens minus filler lexicon.

    ``_skill_keyphrase_tokens`` already drops Bloom verbs + generic stopwords;
    this additionally subtracts the shared domain-agnostic filler vocabulary so a
    "various techniques / key concepts" object contributes nothing.
    """
    return set(_skill_keyphrase_tokens(statement)) - filler_tokens()


def _co_chunk_ids(co: Dict[str, Any]) -> List[str]:
    raw = co.get("source_chunk_ids")
    if isinstance(raw, list):
        ids = [str(c).strip() for c in raw if str(c).strip()]
        if ids:
            return ids
    out: List[str] = []
    for ref in co.get("source_refs") or []:
        if isinstance(ref, dict):
            for c in ref.get("chunk_ids") or []:
                cs = str(c).strip()
                if cs:
                    out.append(cs)
    return out


class ObjectiveSpecificityValidator:
    """Defect B — CO-statement specificity / vacuity gate at ``course_planning``.

    Opt-in via ``ED4ALL_OBJECTIVE_SPECIFICITY`` (default OFF → no-op pass). When
    ON, runs V1 (content-residual floor → ``OBJECTIVE_VACUOUS``), V2 (vague-object
    → ``OBJECTIVE_GENERIC_OBJECT``), and V3 (source-anchoring recall →
    ``OBJECTIVE_UNANCHORED_STATEMENT``) over each chapter objective; flags the
    aggregate ``OBJECTIVE_VACUOUS_RATE_HIGH`` when the vacuous rate exceeds the
    floor. Warning-severity day-1; critical-flip deferred (``# TODO(calibration)``).
    """

    name = "objective_specificity"
    version = "0.1.0"  # W2 Defect B

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        # Opt-in gate: default OFF → byte-identical no-op pass.
        if not _specificity_enabled():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code="OBJECTIVE_SPECIFICITY_DISABLED",
                        message=(
                            "ED4ALL_OBJECTIVE_SPECIFICITY is off; objective "
                            "specificity/vacuity audit skipped."
                        ),
                    )
                ],
            )

        min_residual = int(
            inputs.get("min_content_residual", DEFAULT_MIN_CONTENT_RESIDUAL)
        )
        min_generic_residual = int(
            inputs.get(
                "min_generic_object_residual",
                DEFAULT_MIN_GENERIC_OBJECT_RESIDUAL,
            )
        )
        min_recall = float(
            inputs.get(
                "min_statement_token_recall", DEFAULT_MIN_STATEMENT_TOKEN_RECALL
            )
        )
        max_vacuous_rate = float(
            inputs.get("max_vacuous_rate", DEFAULT_MAX_VACUOUS_RATE)
        )

        objectives, err = _coerce_dict(
            inputs,
            explicit_key="synthesized_objectives",
            path_key="synthesized_objectives_path",
            code_prefix="OBJECTIVE_SPECIFICITY_OBJECTIVES",
        )
        if err is not None or objectives is None:
            # Unresolvable objectives → graceful no-op pass (never block).
            return self._pass(gate_id, [])

        chapter_objectives = (
            objectives.get("chapter_objectives")
            or objectives.get("component_objectives")
            or objectives.get("learning_outcomes")
            or []
        )
        cos = flatten_chapter_objectives(chapter_objectives)
        cos = [c for c in cos if isinstance(c, dict) and _statement(c)]
        if not cos:
            return self._pass(gate_id, [])

        # V3 needs the chunk text universe. Absent → NO_CHUNK_UNIVERSE warning;
        # V1/V2 still run (they don't need chunks).
        chunks_lookup = _build_chunks_lookup(inputs)
        issues: List[GateIssue] = []
        if not chunks_lookup:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="NO_CHUNK_UNIVERSE",
                    message=(
                        "No source chunkset resolved "
                        "(chunks_by_id / chunks / dart_chunks_path); the V3 "
                        "source-anchoring check is skipped (V1/V2 still run)."
                    ),
                )
            )

        # Memoize chunk token sets across COs (a chunk may back >1 CO).
        chunk_tokens_cache: Dict[str, Set[str]] = {}

        audited = 0
        vacuous_count = 0
        generic_count = 0
        unanchored_count = 0

        for co in cos:
            stmt = _statement(co)
            audited += 1
            residual = _content_residual(stmt)
            co_id = str(co.get("id") or co.get("co_id") or "").strip() or "(unminted)"

            # V1 — content-residual floor.
            if len(residual) < min_residual:
                vacuous_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="OBJECTIVE_VACUOUS",
                            message=(
                                f"Chapter objective {co_id!r} has only "
                                f"{len(residual)} content token(s) after "
                                f"removing Bloom verbs, stopwords, and filler "
                                f"(floor {min_residual}): {sorted(residual)}. The "
                                f"statement names no concrete teachable skill."
                            ),
                            location=f"chapter_objectives[id={co_id}]",
                            suggestion=(
                                "Re-synthesize the objective from a grounded "
                                "source chunk so it names a concrete skill object."
                            ),
                        )
                    )
                # A vacuous CO is not further audited for V2/V3 (already flagged).
                continue

            # V2 — vague-object phrase + thin residual.
            if len(residual) < min_generic_residual and has_vague_object(stmt):
                generic_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="OBJECTIVE_GENERIC_OBJECT",
                            message=(
                                f"Chapter objective {co_id!r} uses a vague-object "
                                f"phrase with a thin content residual "
                                f"({len(residual)} < {min_generic_residual}): "
                                f"{sorted(residual)}. Its object is filler "
                                f"(e.g. 'various concepts'), not a named skill."
                            ),
                            location=f"chapter_objectives[id={co_id}]",
                            suggestion=(
                                "Replace the generic object with the concrete "
                                "concept the source chunk actually teaches."
                            ),
                        )
                    )

            # V3 — source-anchoring recall (only for COs that cite chunks).
            if chunks_lookup and residual:
                chunk_ids = _co_chunk_ids(co)
                cited_tokens: Set[str] = set()
                have_source = False
                for cid in chunk_ids:
                    if cid not in chunks_lookup:
                        continue
                    have_source = True
                    toks = chunk_tokens_cache.get(cid)
                    if toks is None:
                        toks = set(_TOKEN_RE.findall(chunks_lookup[cid].lower()))
                        chunk_tokens_cache[cid] = toks
                    cited_tokens |= toks
                if have_source:
                    recall = len(residual & cited_tokens) / len(residual)
                    if recall < min_recall:
                        unanchored_count += 1
                        if len(issues) < _ISSUE_LIST_CAP:
                            issues.append(
                                GateIssue(
                                    severity="warning",
                                    code="OBJECTIVE_UNANCHORED_STATEMENT",
                                    message=(
                                        f"Chapter objective {co_id!r} has "
                                        f"source-token recall {recall:.2f} < "
                                        f"{min_recall:.2f}: only "
                                        f"{len(residual & cited_tokens)}/"
                                        f"{len(residual)} content token(s) appear "
                                        f"in its cited chunk text(s). The "
                                        f"statement may be factually adrift from "
                                        f"its own source."
                                    ),
                                    location=f"chapter_objectives[id={co_id}]",
                                    suggestion=(
                                        "Re-cite the chunk the statement is "
                                        "actually drawn from, or re-synthesize "
                                        "the statement from the cited source."
                                    ),
                                )
                            )

        vacuous_rate = (vacuous_count / audited) if audited else 0.0
        if audited and vacuous_rate > max_vacuous_rate:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="OBJECTIVE_VACUOUS_RATE_HIGH",
                    message=(
                        f"{vacuous_count}/{audited} chapter objective(s) "
                        f"({vacuous_rate:.2%}) are vacuous (content residual < "
                        f"{min_residual}), above the {max_vacuous_rate:.2%} "
                        f"floor. Objective synthesis is emitting scaffolding "
                        f"without a named skill object."
                    ),
                    location="chapter_objectives",
                )
            )

        self._emit_decision(
            capture,
            audited=audited,
            vacuous=vacuous_count,
            generic=generic_count,
            unanchored=unanchored_count,
            vacuous_rate=vacuous_rate,
            max_vacuous_rate=max_vacuous_rate,
            had_chunkset=bool(chunks_lookup),
        )

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0  # warning-day-1 → always True
        score = 1.0 if audited == 0 else round(1.0 - vacuous_rate, 4)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
        )

    # ---------------------------------------------------------------- helpers

    def _pass(self, gate_id: str, issues: List[GateIssue]) -> GateResult:
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=1.0,
            issues=issues,
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        audited: int,
        vacuous: int,
        generic: int,
        unanchored: int,
        vacuous_rate: float,
        max_vacuous_rate: float,
        had_chunkset: bool,
    ) -> None:
        if capture is None:
            return
        rationale = (
            f"objective_specificity audit over {audited} chapter objective(s): "
            f"vacuous={vacuous} (rate={vacuous_rate:.3f} vs floor "
            f"{max_vacuous_rate:.3f}), generic_object={generic}, "
            f"unanchored={unanchored}, chunkset_present={had_chunkset}."
        )
        try:
            capture.log_decision(
                decision_type=_DECISION_TYPE,
                decision=(
                    "passed"
                    if vacuous_rate <= max_vacuous_rate
                    else f"flagged:vacuous_rate_high({vacuous}/{audited})"
                ),
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 — capture must not break the gate
            logger.debug(
                "DecisionCapture.log_decision raised on objective_specificity: %s",
                exc,
            )


__all__ = [
    "ObjectiveSpecificityValidator",
    "DEFAULT_MIN_CONTENT_RESIDUAL",
    "DEFAULT_MIN_GENERIC_OBJECT_RESIDUAL",
    "DEFAULT_MIN_STATEMENT_TOKEN_RECALL",
    "DEFAULT_MAX_VACUOUS_RATE",
]
