"""W7.3 — SynthesizedQuizDistractorValidator (assessment_synthesis phase).

The W10 ``assessment_synthesis`` phase synthesizes learner-facing QTI quizzes
via ``Trainforge.generators.assessment.generator.AssessmentGenerator`` and emits
them as QTI 1.2 XML into ``<export>/06_assessments/``. Those synthesized MCQ
items are gated for QTI well-formedness (``qti_well_formed``) and objective
grounding (``assessment_objective_alignment``), but their DISTRACTORS bypass
every distractor-quality validator — the ``lib/validators/distractor_*`` gates
run on Courseforge ``Block`` objects at the two-pass rewrite tier, NOT on the
Trainforge QTI item shape. So a synthesized quiz can ship distractors that are
degenerate (identical to each other), equal to the correct key, or a
non-misconception placeholder ("Not supported by the source") with no gate ever
firing.

This validator closes that gap. It reads the emitted QTI documents (the same
path-or-data input contract as ``QtiWellFormedValidator`` so the wiring wave can
point it at the ``06_assessments/`` sibling without a code change), parses each
via the in-tree ``QTIParser`` oracle, and audits every ``multiple_choice`` /
``multiple_response`` item's distractors for three degeneracy classes:

* **not-equal-to-key** — a distractor whose normalized text equals the correct
  answer's normalized text (``SYNTH_QUIZ_DISTRACTOR_EQUALS_KEY``).
* **non-degenerate** — two distractors with identical normalized text
  (``SYNTH_QUIZ_DISTRACTOR_DUPLICATE``); or fewer than two distractors
  (``SYNTH_QUIZ_DISTRACTOR_TOO_FEW``).
* **misconception-targeting** — a distractor matching a padding / placeholder
  template (e.g. "not supported by the source", "option N", "none of the
  above") is not a misconception-targeting distractor
  (``SYNTH_QUIZ_DISTRACTOR_PLACEHOLDER``).

Severity contract: ALL issues are **warning-severity** day-1 (``passed`` is
always True, ``critical-count == 0``) with a ``# TODO(calibration)`` deferred
critical-flip after a >=2-corpus FP measurement (the WS3/W4 deferred-flip
pattern; NOT IB3). Gates the synthesized-quiz distractor surface like the
authored-block distractor gates without hard-blocking the run day-1.

Input contract (``validate(inputs: Dict[str, Any]) -> GateResult``), mirroring
``QtiWellFormedValidator._collect_documents``:

* ``inputs["qti_dir"]`` — a directory globbed for ``*.xml`` (the wiring form —
  ``06_assessments/``).
* ``inputs["qti_paths"]`` — explicit list of XML file paths.
* ``inputs["qti_strings"]`` — list of ``{"id": <label>, "xml": <str>}`` dicts
  (or bare XML strings), for in-memory / test use.

When none resolve to a document, the gate is a graceful no-op pass (a
``SYNTH_QUIZ_NO_INPUT`` info issue) — the QTI-existence axis is owned by
``qti_well_formed``; this validator never double-fails on missing input.

Decision capture: one ``validation_result`` decision per ``validate()`` call
(guarded on a ``decision_capture`` / ``capture`` input), rationale interpolates
items audited + per-class flag counts so captures replay post-hoc.

References:
    - ``lib/validators/qti_well_formed.py`` — input-contract template
      (``_collect_documents``), GateResult shape.
    - ``lib/validators/distractor_structural.py`` — the authored-block
      distractor-quality gate this mirrors for the QTI item shape.
    - ``Trainforge/parsers/qti_parser.py::QTIParser`` — the parse oracle.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap the emitted issue list so a uniformly-broken quiz batch doesn't drown
#: the gate report (mirrors the sibling distractor validators' cap).
_ISSUE_LIST_CAP: int = 50

#: QTI question types whose distractors this validator audits. Only choice
#: items carry a distractor set; short_answer / essay / fill_in_blank / matching
#: are skipped.
_MCQ_TYPES: frozenset = frozenset({"multiple_choice", "multiple_response"})

#: Placeholder / padding distractor patterns — a distractor matching one of
#: these is not a misconception-targeting option. The first entry matches the
#: ``AssessmentGenerator`` padded-fallback template
#: ("Not supported by the source (option N).") that W2.D nominally eliminated
#: but which can still surface via other emit paths.
_PLACEHOLDER_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^not supported by the source"),
    re.compile(r"^none of the above$"),
    re.compile(r"^all of the above$"),
    re.compile(r"^option\s+\d+$"),
    re.compile(r"^distractor\s+\d+$"),
    re.compile(r"^n/?a$"),
)

_DECISION_TYPE: str = "validation_result"


def _strip_html(text: str) -> str:
    """Drop HTML tags + collapse entities so QTI ``<p>…</p>`` choices compare."""
    no_tags = re.sub(r"<[^>]+>", " ", text or "")
    return no_tags


def _normalize(text: str) -> str:
    """Normalize a choice string for equality / placeholder comparison.

    Lower-cases, strips HTML, collapses whitespace, and drops surrounding
    punctuation so ``<p>The Sun.</p>`` and ``the sun`` compare equal.
    """
    stripped = _strip_html(text).lower()
    # Collapse whitespace.
    stripped = re.sub(r"\s+", " ", stripped).strip()
    # Drop leading/trailing punctuation (keep internal for placeholder regex).
    stripped = stripped.strip(".,;:!?\"'()[]{}")
    return stripped.strip()


def _is_placeholder(norm_text: str) -> bool:
    """True when a normalized distractor matches a padding template."""
    for pat in _PLACEHOLDER_PATTERNS:
        if pat.search(norm_text):
            return True
    return False


class SynthesizedQuizDistractorValidator:
    """W7.3 — synthesized-quiz MCQ distractor-quality gate.

    Validator-protocol-compatible class intended for the ``assessment_synthesis``
    phase (wiring owned by Integrate). Emits WARNING-severity issues day-1 so it
    surfaces degenerate synthesized distractors without hard-blocking; the
    critical-flip is deferred (``# TODO(calibration)``).
    """

    name = "synthesized_quiz_distractor"
    version = "0.1.0"  # W7.3

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")
        issues: List[GateIssue] = []

        documents, input_issues = self._collect_documents(inputs)
        issues.extend(input_issues)
        if not documents:
            # No documents → graceful no-op pass. QTI-existence is owned by
            # qti_well_formed; this gate never double-fails on missing input.
            if not input_issues:
                issues.append(
                    GateIssue(
                        severity="info",
                        code="SYNTH_QUIZ_NO_INPUT",
                        message=(
                            "SynthesizedQuizDistractorValidator found no QTI "
                            "documents (inputs['qti_dir'] / ['qti_paths'] / "
                            "['qti_strings']); distractor audit skipped."
                        ),
                    )
                )
            return self._result(gate_id, issues, 0, 0, capture)

        try:
            from Trainforge.parsers.qti_parser import QTIParser
        except Exception as exc:  # noqa: BLE001 — graceful degrade
            logger.warning(
                "SynthesizedQuizDistractorValidator: QTIParser import "
                "failed (%s); distractor audit skipped.", exc,
            )
            issues.append(
                GateIssue(
                    severity="warning",
                    code="SYNTH_QUIZ_PARSER_MISSING",
                    message=(
                        f"QTIParser unavailable ({exc}); synthesized-quiz "
                        f"distractor audit skipped."
                    ),
                )
            )
            return self._result(gate_id, issues, 0, 0, capture)

        parser = QTIParser()
        items_audited = 0
        items_flagged = 0

        for label, xml in documents:
            try:
                assessment = parser.parse_string(xml)
            except Exception as exc:  # noqa: BLE001 — parse errors owned by
                # qti_well_formed; skip, don't double-fail.
                logger.debug(
                    "SynthesizedQuizDistractorValidator: parse failed for "
                    "%s: %s", label, exc,
                )
                continue

            for question in getattr(assessment, "questions", []) or []:
                qtype = str(getattr(question, "type", "") or "")
                if qtype not in _MCQ_TYPES:
                    continue
                items_audited += 1
                qid = str(getattr(question, "id", "") or f"{label}#item")
                flagged = self._audit_question(question, qid, label, issues)
                if flagged:
                    items_flagged += 1

        return self._result(
            gate_id, issues, items_audited, items_flagged, capture
        )

    # ------------------------------------------------------------- helpers

    def _audit_question(
        self,
        question: Any,
        qid: str,
        label: str,
        issues: List[GateIssue],
    ) -> bool:
        """Audit one MCQ's distractors; append issues. Returns True if flagged."""
        choices = list(getattr(question, "choices", []) or [])
        correct_norms = {
            _normalize(getattr(c, "text", ""))
            for c in choices
            if getattr(c, "is_correct", False)
        }
        distractors = [
            _normalize(getattr(c, "text", ""))
            for c in choices
            if not getattr(c, "is_correct", False)
        ]
        # Drop empty normalized distractors (blank options are owned by the
        # QTI structural gate, not the distractor-quality gate).
        distractors = [d for d in distractors if d]

        flagged = False
        location = f"{label}#{qid}"

        # Non-degenerate — too few distractors.
        if len(distractors) < 2:
            flagged = True
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="SYNTH_QUIZ_DISTRACTOR_TOO_FEW",
                        message=(
                            f"Synthesized MCQ {qid!r} carries "
                            f"{len(distractors)} distractor(s); a non-degenerate "
                            f"MCQ needs >=2 (1 correct + >=2 distractors)."
                        ),
                        location=location,
                        suggestion=(
                            "Extract more factual statements / key terms for "
                            "this objective, or skip-emit the item."
                        ),
                    )
                )

        # not-equal-to-key.
        for d in distractors:
            if d in correct_norms:
                flagged = True
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="SYNTH_QUIZ_DISTRACTOR_EQUALS_KEY",
                            message=(
                                f"Synthesized MCQ {qid!r} has a distractor whose "
                                f"text equals the correct answer "
                                f"({d[:60]!r}); a distractor must differ from "
                                f"the key."
                            ),
                            location=location,
                            suggestion=(
                                "Regenerate the distractor from a distinct "
                                "misconception / content statement."
                            ),
                        )
                    )
                break

        # Non-degenerate — duplicate distractors.
        seen: Dict[str, int] = {}
        for d in distractors:
            seen[d] = seen.get(d, 0) + 1
        dupes = [d for d, n in seen.items() if n > 1]
        if dupes:
            flagged = True
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="SYNTH_QUIZ_DISTRACTOR_DUPLICATE",
                        message=(
                            f"Synthesized MCQ {qid!r} carries "
                            f"{len(dupes)} duplicated distractor(s) "
                            f"(e.g. {dupes[0][:60]!r}); distractors must be "
                            f"distinct."
                        ),
                        location=location,
                        suggestion="Deduplicate the distractor set.",
                    )
                )

        # misconception-targeting — placeholder / padding distractor.
        placeholders = [d for d in distractors if _is_placeholder(d)]
        if placeholders:
            flagged = True
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="SYNTH_QUIZ_DISTRACTOR_PLACEHOLDER",
                        message=(
                            f"Synthesized MCQ {qid!r} carries "
                            f"{len(placeholders)} placeholder distractor(s) "
                            f"(e.g. {placeholders[0][:60]!r}) that do not target "
                            f"a misconception."
                        ),
                        location=location,
                        suggestion=(
                            "Replace the placeholder with a plausible "
                            "misconception-targeting distractor grounded in "
                            "the source content."
                        ),
                    )
                )

        return flagged

    def _result(
        self,
        gate_id: str,
        issues: List[GateIssue],
        items_audited: int,
        items_flagged: int,
        capture: Any,
    ) -> GateResult:
        """Assemble the GateResult; emit one aggregate decision per call."""
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0  # warning-day-1 → always True
        warn_count = sum(1 for i in issues if i.severity == "warning")
        score = (
            1.0
            if items_audited == 0
            else round(1.0 - (items_flagged / items_audited), 4)
        )
        self._emit_decision(
            capture,
            items_audited=items_audited,
            items_flagged=items_flagged,
            warn_count=warn_count,
            passed=passed,
        )
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
    def _emit_decision(
        capture: Any,
        *,
        items_audited: int,
        items_flagged: int,
        warn_count: int,
        passed: bool,
    ) -> None:
        if capture is None:
            return
        rationale = (
            f"Synthesized-quiz distractor audit: {items_audited} MCQ item(s) "
            f"audited, {items_flagged} flagged with {warn_count} "
            f"warning-severity distractor-degeneracy issue(s) "
            f"(equals-key / duplicate / too-few / placeholder). All "
            f"warning-severity day-1 (deferred critical-flip); passed={passed}."
        )
        try:
            capture.log_decision(
                decision_type=_DECISION_TYPE,
                decision=(
                    f"synthesized_quiz_distractor:"
                    f"{items_flagged}/{items_audited}_flagged"
                ),
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 — capture must not break the gate
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "synthesized_quiz_distractor: %s",
                exc,
            )

    @staticmethod
    def _collect_documents(
        inputs: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, str]], List[GateIssue]]:
        """Normalise the three input forms into ``[(label, xml_str), ...]``.

        Mirrors ``QtiWellFormedValidator._collect_documents`` but never emits a
        critical missing-input issue (this gate no-ops on missing input; the
        QTI-existence axis is owned by ``qti_well_formed``).
        """
        documents: List[Tuple[str, str]] = []
        issues: List[GateIssue] = []

        raw_strings = inputs.get("qti_strings")
        if raw_strings:
            for idx, entry in enumerate(raw_strings):
                if isinstance(entry, dict):
                    lbl = str(entry.get("id", f"qti_strings[{idx}]"))
                    xml = entry.get("xml", "")
                else:
                    lbl = f"qti_strings[{idx}]"
                    xml = entry
                documents.append((lbl, xml if isinstance(xml, str) else ""))

        raw_paths = inputs.get("qti_paths")
        path_candidates: List[Path] = []
        if raw_paths:
            path_candidates.extend(Path(p) for p in raw_paths)

        raw_dir = inputs.get("qti_dir")
        if raw_dir:
            qti_dir = Path(raw_dir)
            if qti_dir.exists() and qti_dir.is_dir():
                path_candidates.extend(sorted(qti_dir.rglob("*.xml")))

        for path in path_candidates:
            try:
                documents.append((str(path), path.read_text(encoding="utf-8")))
            except OSError as exc:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="SYNTH_QUIZ_UNREADABLE_FILE",
                        message=f"Could not read QTI file {path}: {exc}",
                        location=str(path),
                    )
                )

        # Drop empty-XML docs (nothing to audit).
        documents = [(lbl, xml) for lbl, xml in documents if xml and xml.strip()]
        return documents, issues


__all__ = ["SynthesizedQuizDistractorValidator"]
