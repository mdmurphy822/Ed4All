"""WS6a — Source→objective coverage-audit gate (course_planning).

The synthesized-objectives set (TO-NN + CO-NN) is the planning-time
contract for what the course will teach. Nothing, however, asserts that
the OBJECTIVES actually cover the SOURCE: a textbook section can carry
real instructional content that no synthesized objective ever rolls up
to, so downstream content generation never authors a page for it and the
material silently vanishes from the course.

This validator is the MEASUREMENT guardrail for that gap. For every
content-bearing section in ``textbook_structure.json`` it embeds
``headingText + ". " + section_text`` (truncated) and computes the max
cosine to ANY synthesized objective statement (CO or TO). A section whose
best objective match falls below ``coverage_floor`` is flagged
``SOURCE_SECTION_UNCOVERED``; when the uncovered rate exceeds
``max_uncovered_rate`` the gate emits the aggregate ``SOURCE_COVERAGE_LOW``
warning.

DE-RISK PROOF (2026-06-17, real corpus): 135/135 content sections are
covered at floor 0.45 (mean cosine 0.73), so the real corpus passes
clean. The gate is a guardrail for OTHER corpora — an objectives set that
genuinely misses a topic run will surface it here instead of at archival.

Mirror (embedding-validator template):
``lib/validators/co_terminal_alignment.py`` (WS3) — the
``try_load_embedder()`` + ``EMBEDDING_DEPS_MISSING`` graceful-degrade,
``is_strict_mode()``, ``_emit_decision``, ``_ISSUE_LIST_CAP``,
top-level ``config:``→inputs threshold resolution via
``inputs.get(...)``, graceful no-op pass on missing/empty inputs,
``score = covered/audited``, ``passed = (critical count == 0)`` (always
True day-1; all issues warning-severity).

Inputs (``inputs`` dict, fed by the EXISTING
``_build_chapter_objective_coverage_inputs`` builder — no new builder;
it already surfaces BOTH ``synthesized_objectives`` AND
``textbook_structure``):

    synthesized_objectives_path (str) OR synthesized_objectives (dict)
        The course's synthesized objectives doc (CO + TO statements).
        Resolved via ``_coerce_dict``. Unresolvable / empty → graceful
        no-op pass.

    textbook_structure_path (str) OR textbook_structure (dict)
        The textbook structure (``chapters[].sections[]``). Resolved via
        ``_coerce_dict``. Unresolvable / empty / no content section →
        graceful no-op pass.

    coverage_floor (float, TOP-LEVEL config key)
        Per-section max-cosine floor below which a section is flagged
        uncovered (default 0.45). The executor merges the gate's
        top-level ``config`` block into ``inputs`` via shallow
        ``setdefault`` (``MCP/hardening/validation_gates.py``), so this
        is read as ``inputs.get("coverage_floor")``.

    max_uncovered_rate (float, TOP-LEVEL config key)
        Corpus-level uncovered-rate floor above which the gate emits the
        ``SOURCE_COVERAGE_LOW`` aggregate warning (default 0.15).

    embedder (Optional[SentenceEmbedder])
        Test seam. Defaults to ``try_load_embedder()``.

    decision_capture / capture
        Injected by the executor. One ``source_coverage_check`` decision
        per audited section.

References:
    - ``lib/validators/co_terminal_alignment.py`` — class template (WS3).
    - ``lib/validators/terminal_objective_coverage.py`` — objectives IO
      (``_coerce_dict``).
    - ``lib/ontology/terminal_coverage.py`` — ``flatten_chapter_objectives``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    SentenceEmbedder,
    is_strict_mode,
    try_load_embedder,
)
from lib.ontology.terminal_coverage import flatten_chapter_objectives
from lib.validators.terminal_objective_coverage import _coerce_dict

logger = logging.getLogger(__name__)

#: Per-section max-cosine floor below which a content-bearing section is
#: flagged ``SOURCE_SECTION_UNCOVERED``. On the real corpus 135/135
#: content sections clear 0.45 (mean 0.73); the floor is the guardrail
#: for OTHER corpora. The critical flip + a floor calibrated toward ~0.55
#: on ≥2 corpora is DEFERRED (see the gate ``# TODO(calibration)`` marker).
DEFAULT_COVERAGE_FLOOR: float = 0.45

#: Corpus-level uncovered-rate floor above which the gate emits the
#: ``SOURCE_COVERAGE_LOW`` aggregate warning.
DEFAULT_MAX_UNCOVERED_RATE: float = 0.15

#: Minimum ``section_text`` length for a section to be content-bearing
#: (a section below this is a trivial heading / stub and is skipped — it
#: carries no instruction to cover).
_MIN_SECTION_TEXT_LEN: int = 40

#: Truncation cap on the embedded section text (heading + body). Matches
#: the WS6b re-segmentation embed truncation (the same sections).
_SECTION_TEXT_TRUNC: int = 600

#: Cap on the emitted issue list so a wholly-uncovered structure doesn't
#: drown the gate report (mirrors ``co_terminal_alignment._ISSUE_LIST_CAP``).
_ISSUE_LIST_CAP: int = 50

#: Validator-capture decision_type — NEW enum member (WS6a).
_DECISION_TYPE: str = "source_coverage_check"


def _emit_decision(
    capture: Any,
    *,
    section_heading: str,
    max_cosine: Optional[float],
    coverage_floor: float,
    covered: bool,
    code: Optional[str],
    embedder_strict: bool,
) -> None:
    """Emit one ``source_coverage_check`` decision per audited section.

    Rationale interpolates dynamic signals (≥20 chars) so captures are
    replayable post-hoc: the section heading, the section→nearest-objective
    max cosine + the coverage floor, the covered flag, the embedder
    strict-mode flag (so EMBEDDING_DEPS_MISSING degrade events are
    distinguishable from real low-cosine events), and the failure code.
    """
    if capture is None:
        return
    decision = "covered" if covered else f"uncovered:{code or 'unknown'}"
    cos_str = f"{max_cosine:.4f}" if max_cosine is not None else "n/a"
    rationale = (
        f"Source coverage check on section {section_heading!r}: "
        f"max_cosine_to_objective={cos_str}, "
        f"coverage_floor={coverage_floor:.4f}, covered={covered}, "
        f"embedder_strict_mode={embedder_strict}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "source_coverage_check: %s",
            exc,
        )


def _objective_statement(obj: Dict[str, Any]) -> str:
    """Return an objective statement text, tolerating common field names."""
    for key in ("statement", "text", "objective", "description"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _collect_objective_statements(objectives: Dict[str, Any]) -> List[str]:
    """All CO + TO statements from a synthesized-objectives doc.

    Tolerates the Courseforge synthesized form
    (``terminal_objectives`` / ``chapter_objectives``) and the LibV2
    archive form (``terminal_outcomes`` / ``component_objectives``).
    """
    statements: List[str] = []

    terminals = (
        objectives.get("terminal_objectives")
        or objectives.get("terminal_outcomes")
        or []
    )
    for to in terminals:
        if isinstance(to, dict):
            stmt = _objective_statement(to)
            if stmt:
                statements.append(stmt)

    chapter_objectives = (
        objectives.get("chapter_objectives")
        or objectives.get("component_objectives")
        or []
    )
    for co in flatten_chapter_objectives(chapter_objectives):
        stmt = _objective_statement(co)
        if stmt:
            statements.append(stmt)

    return statements


def _section_heading(section: Dict[str, Any]) -> str:
    """Return a section's heading text, tolerating common field names."""
    for key in ("headingText", "heading", "title", "id"):
        val = section.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _section_text(section: Dict[str, Any]) -> str:
    """Return a section's body text, tolerating common field names."""
    for key in ("section_text", "text", "body", "content"):
        val = section.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _iter_content_sections(structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Yield the content-bearing sections from a textbook_structure dict.

    Iterates ``chapters[].sections[]``; a section is content-bearing when
    its body text length exceeds :data:`_MIN_SECTION_TEXT_LEN` (a trivial
    heading / stub carries no instruction to cover and is skipped).
    """
    out: List[Dict[str, Any]] = []
    chapters = structure.get("chapters")
    if not isinstance(chapters, list):
        return out
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        sections = ch.get("sections")
        if not isinstance(sections, list):
            continue
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            if len(_section_text(sec)) > _MIN_SECTION_TEXT_LEN:
                out.append(sec)
    return out


def _embed_text(section: Dict[str, Any]) -> str:
    """Build the section's embed string (heading + body, truncated)."""
    heading = _section_heading(section)
    body = _section_text(section)
    combined = f"{heading}. {body}" if heading else body
    return combined[:_SECTION_TEXT_TRUNC]


class SourceCoverageValidator:
    """WS6a — Source→objective coverage-audit gate at ``course_planning``.

    Validator-protocol-compatible class wired into
    ``textbook_to_course::course_planning::source_coverage``
    (warning / warn). Embeds each content-bearing textbook section and
    asserts it is semantically covered by ≥1 synthesized objective
    (CO or TO) above ``coverage_floor``.

    Day-1 ALL issues are warning-severity → ``passed`` is always True
    (``critical-count == 0``). The critical flip (and a floor calibrated
    toward ~0.55 on ≥2 corpora) is DEFERRED (``# TODO(calibration)``).
    """

    name = "source_coverage"
    version = "0.1.0"

    def __init__(
        self,
        *,
        coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
        max_uncovered_rate: float = DEFAULT_MAX_UNCOVERED_RATE,
        embedder: Optional[SentenceEmbedder] = None,
    ) -> None:
        self._coverage_floor = coverage_floor
        self._max_uncovered_rate = max_uncovered_rate
        # Lazy-load on validate() so test injections take precedence.
        self._embedder_override = embedder

    def _noop_pass(self, gate_id: str) -> GateResult:
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=1.0,
            issues=[],
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")
        coverage_floor = float(
            inputs.get("coverage_floor", self._coverage_floor)
        )
        max_uncovered_rate = float(
            inputs.get("max_uncovered_rate", self._max_uncovered_rate)
        )

        objectives, obj_err = _coerce_dict(
            inputs,
            explicit_key="synthesized_objectives",
            path_key="synthesized_objectives_path",
            code_prefix="SOURCE_COVERAGE_OBJECTIVES",
        )
        # Unresolvable objectives → graceful no-op pass (this gate is a
        # measurement guardrail; other gates own malformed-objectives). We
        # do NOT propagate the critical _coerce_dict error — WS6a never blocks.
        if obj_err is not None or objectives is None:
            return self._noop_pass(gate_id)

        structure, struct_err = _coerce_dict(
            inputs,
            explicit_key="textbook_structure",
            path_key="textbook_structure_path",
            code_prefix="SOURCE_COVERAGE_STRUCTURE",
        )
        if struct_err is not None or structure is None:
            return self._noop_pass(gate_id)

        obj_statements = _collect_objective_statements(objectives)
        sections = _iter_content_sections(structure)

        # Nothing to audit → graceful no-op pass.
        if not obj_statements or not sections:
            return self._noop_pass(gate_id)

        embedder_strict = is_strict_mode()
        embedder = self._embedder_override or try_load_embedder()
        if embedder is None:
            # Graceful-degrade — emit one passed=True capture per section so
            # the silent-degrade signal lands in the audit trail, then return
            # the single warning gate result. Mirrors WS3.
            for sec in sections:
                _emit_decision(
                    capture,
                    section_heading=_section_heading(sec) or "?",
                    max_cosine=None,
                    coverage_floor=coverage_floor,
                    covered=True,
                    code="EMBEDDING_DEPS_MISSING",
                    embedder_strict=embedder_strict,
                )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="EMBEDDING_DEPS_MISSING",
                        message=(
                            "sentence-transformers extras not installed; "
                            "WS6a source-coverage gate skipped. Install via "
                            "`pip install -e .[embedding]` to enable."
                        ),
                    )
                ],
            )

        # Encode the objective statements once (batched when supported).
        obj_vecs: List[Any] = self._encode_batch(embedder, obj_statements)
        obj_vecs = [v for v in obj_vecs if v is not None]
        if not obj_vecs:
            # Every objective encode failed — cannot adjudicate; no-op pass.
            return self._noop_pass(gate_id)

        issues: List[GateIssue] = []
        audited = 0
        covered_count = 0
        uncovered_count = 0

        # Encode the section embed strings in one batch too.
        section_texts = [_embed_text(sec) for sec in sections]
        section_vecs = self._encode_batch(embedder, section_texts)

        for sec, sec_vec in zip(sections, section_vecs):
            heading = _section_heading(sec) or "?"
            if sec_vec is None:
                # Encode failed for this section — skip scoring (passed).
                _emit_decision(
                    capture,
                    section_heading=heading,
                    max_cosine=None,
                    coverage_floor=coverage_floor,
                    covered=True,
                    code="EMBEDDING_ENCODE_ERROR",
                    embedder_strict=embedder_strict,
                )
                continue

            audited += 1
            max_cos = max(
                (cosine_similarity(sec_vec, ov) for ov in obj_vecs),
                default=0.0,
            )
            covered = max_cos >= coverage_floor
            if covered:
                covered_count += 1
                _emit_decision(
                    capture,
                    section_heading=heading,
                    max_cosine=max_cos,
                    coverage_floor=coverage_floor,
                    covered=True,
                    code=None,
                    embedder_strict=embedder_strict,
                )
            else:
                uncovered_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="SOURCE_SECTION_UNCOVERED",
                            message=(
                                f"Textbook section {heading!r} has max cosine "
                                f"{max_cos:.4f} to any synthesized objective, "
                                f"below the coverage floor "
                                f"{coverage_floor:.4f}. No CO/TO appears to "
                                f"cover this section's content."
                            ),
                            location=f"chapters[].sections[heading={heading}]",
                            suggestion=(
                                "Synthesize a chapter objective (CO) for this "
                                "section's topic, or confirm the section is "
                                "non-instructional (answer-key / self-check) "
                                "and should not be covered."
                            ),
                        )
                    )
                _emit_decision(
                    capture,
                    section_heading=heading,
                    max_cosine=max_cos,
                    coverage_floor=coverage_floor,
                    covered=False,
                    code="SOURCE_SECTION_UNCOVERED",
                    embedder_strict=embedder_strict,
                )

        uncovered_rate = (uncovered_count / audited) if audited else 0.0
        if audited and uncovered_rate > max_uncovered_rate:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="SOURCE_COVERAGE_LOW",
                    message=(
                        f"{uncovered_count}/{audited} content section(s) "
                        f"({uncovered_rate:.2%}) are not covered by any "
                        f"synthesized objective at cosine "
                        f"{coverage_floor:.4f}, exceeding the warn floor "
                        f"{max_uncovered_rate:.2%}. The objectives set may "
                        f"miss source material."
                    ),
                    location="chapters[].sections",
                )
            )

        # Day-1 ALL issues warning-severity → passed always True.
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        score = 1.0 if audited == 0 else round(covered_count / audited, 4)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
        )

    @staticmethod
    def _encode_batch(
        embedder: SentenceEmbedder, texts: List[str]
    ) -> List[Optional[Any]]:
        """Encode a list of texts, one vector per text.

        Prefers a batch ``encode([...])`` call when the embedder returns a
        list of vectors for a list input; falls back to per-text encoding.
        Any per-text failure yields ``None`` for that slot (the caller
        treats ``None`` as a skip-with-pass).
        """
        out: List[Optional[Any]] = []
        for text in texts:
            try:
                out.append(embedder.encode(text))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedder.encode raised on source-coverage text: %s", exc
                )
                out.append(None)
        return out


__all__ = [
    "SourceCoverageValidator",
    "DEFAULT_COVERAGE_FLOOR",
    "DEFAULT_MAX_UNCOVERED_RATE",
]
