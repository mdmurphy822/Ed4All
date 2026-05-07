"""Worker W2 — answer-text↔source-chunk grounding sentinel + Wave-3 W3.A
hardening (4 new sub-checks + 1 calibration-gated promotion).

Critical-severity gate that closes the answerability gap for
assessment-item Blocks: every assessment question's correct-answer
text must overlap the retrieved source chunks above a Jaccard
content-token threshold, OR the question is rejected. Complements
``RewriteSourceGroundingValidator`` which gates the *stem-vs-source*
axis on rewrite-tier prose blocks; W2 is the *answer-vs-source* axis
on assessment items, which the rewrite-tier validator explicitly
skips per its ``_SKIPPED_CONTENT_TYPES``.

Wave 3 W3.A extends the validator with four new authoring-quality
sub-checks (each emits its own ``GateIssue`` code) plus one severity
promotion on the existing source-attribution check. Each sub-check
ships at warning severity until the calibration signal in
``LibV2/courses/calibration_textbook/quality/trainforge_assessment_quality_report.json::summary.answerability_alignment_rate``
proves stable above the 0.85 floor; the calibration-readback helper
(``lib/governance/calibration_gate.py::resolve_severity_flip``) lifts
the promoted codes to critical when the signal is present and clears
the bar. The flip applies ONLY to the codes documented in the
"Calibration-gated severity flip" section below; the four pure-hygiene
codes stay at warning regardless of flip outcome (one of them —
``ANSWER_SPAN_MISSING`` — is hygiene that does not gate trainability).

The flip is implemented validator-side rather than via a
``config/workflows.yaml`` severity edit so the YAML stays static +
auditable, and the flip decision is captured in the validator's own
audit event (``severity_flip_applied`` / ``severity_flip_deferred``).
The two affected gates therefore stay declared at ``severity: warning``
in the YAML, but their emitted ``GateIssue.severity`` is dynamically
elevated by the validator at validate() time when the calibration
signal clears the floor.

Contract per ``plans/gpt-feedback-w2-w7-execution-2026-05.md`` §W2 +
``plans/gpt-feedback-2-wave3-wiring-telemetry-2026-05.md`` §W3.A:

- Iterate ``inputs["blocks"]`` and filter to ``block.block_type ==
  "assessment_item"``. Non-assessment_item blocks are silently
  skipped (no-op).
- Extract correct-answer text from the block's content surface:
  outline-tier (dict content) reads ``content["answer_key"]`` /
  ``content["correct_answer"]`` / the ``content["correct_answer_index"]``-
  resolved ``content["distractors"][i]["text"]``; rewrite-tier (str
  content) parses the canonical
  ``<li data-cf-correct="true">`` / ``<li data-cf-distractor-index>``
  scaffolding. Essay-style blocks (no fixed answer) emit
  ``ESSAY_SKIPPED`` info and skip the overlap check.
- Resolve referenced source chunks: walk
  ``block.source_references[].sourceId`` + ``block.source_ids``
  against ``inputs["chunks_lookup"]`` (mapping sourceId → chunk
  text). When the block declares no source IDs, **W3.A drops the
  legacy all-chunk-union fallback** — under hard-gate posture a
  block with no source attribution cannot be defended, so
  ``NO_SOURCE_ATTRIBUTION`` is now (calibration-gated) critical and
  the per-block overlap check is skipped to avoid a false-grounded
  signal.
- Compute Jaccard overlap of content-word tokens (lowercased,
  stop-word-stripped). Below the configured floor (default 0.30) →
  ``ANSWER_NOT_GROUNDED`` critical.

W3.A sub-checks (per assessment_item block, in addition to the
existing ANSWER_NOT_GROUNDED / ANSWER_TEXT_MISSING checks):

* ``STEM_NOT_SELF_CONTAINED`` (warning) — regex set on stem text
  catches dangling references ("see above", "previous figure",
  "table 3", "the figure", "see page 12") plus pronoun-only stems
  whose first content token is "It" / "This" / "These" / "Those"
  with no prior antecedent.
* ``ANSWER_LEAKED_IN_STEM`` (warning) — Jaccard token-overlap > 0.50
  between stem text and correct-answer text indicates the stem
  telegraphs the answer ("What is the capital of France? Paris is
  the capital of …").
* ``FEEDBACK_NOT_GROUNDED`` (warning) — when the block carries
  feedback / rationale / explanation prose, apply the same Jaccard
  chunk-overlap floor (default 0.30) so feedback isn't free-form
  fabrication.
* ``ANSWER_SPAN_MISSING`` (warning, even after critical flip — this
  is hygiene) — require an explicit answer-span pointer
  (``content["answer_span"] = {source_chunk_id, char_start,
  char_end}`` on outline; ``data-cf-answer-span="<chunk_id>:<start>-<end>"``
  on rewrite). The validator does not enforce the span's correctness
  beyond presence + structural shape.

Calibration-gated severity flip (W3.A):

* When ``resolve_severity_flip(calibration_signal=
  "answerability_alignment_rate", threshold=0.85)`` returns
  ``apply_flip=True``, the following codes are emitted at
  ``severity="critical"``:
  * ``NO_SOURCE_ATTRIBUTION`` (was warning).
* When the helper returns ``apply_flip=False`` (e.g. on a clean
  checkout with no calibration report yet), the codes stay at
  ``severity="warning"`` and the validator emits a
  ``severity_flip_deferred`` decision-capture event with the
  reason naming the deferral cause.

Decision-capture: emits one ``assessment_retrieval_grounding_check``
event per ``validate()`` call carrying per-block overlap scores +
threshold + verdict + per-sub-check pass/fail counts in
``ml_features`` shape, plus one ``severity_flip_applied`` /
``severity_flip_deferred`` event per validator construction (the
calibration readback runs once at __init__).

References:
    - ``lib/governance/calibration_gate.py`` — calibration-readback
      helper consumed here.
    - ``lib/validators/rewrite_source_grounding.py`` — class skeleton +
      ``_emit_decision()`` pattern this validator mirrors.
    - ``lib/validators/assessment_item_payload.py`` — sibling
      assessment_item walker (W7); same Block.content dual-shape
      (dict outline / str rewrite) handled symmetrically here.
"""

from __future__ import annotations

import logging
import re
import sys

from lib.utils import strip_html_to_text
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.governance.calibration_gate import resolve_severity_flip

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Default Jaccard floor on the (answer_tokens ∩ chunk_tokens) /
#: (answer_tokens ∪ chunk_tokens) ratio. Calibrated so a 5-token
#: answer that shares 2 content tokens with a 50-token chunk
#: (overlap = 2/53 ≈ 0.038) fails, while a 5-token answer that
#: shares 4 with a 10-token chunk (overlap = 4/11 ≈ 0.36) passes.
DEFAULT_MIN_OVERLAP_JACCARD: float = 0.30

#: Default Jaccard ceiling on (stem_tokens ∩ answer_tokens) /
#: (stem_tokens ∪ answer_tokens). Above this floor the stem
#: telegraphs the answer (W3.A ANSWER_LEAKED_IN_STEM).
DEFAULT_MAX_STEM_ANSWER_JACCARD: float = 0.50

#: Default Jaccard floor on feedback-text vs chunk overlap for the
#: W3.A FEEDBACK_NOT_GROUNDED sub-check. Mirrors the answer-grounding
#: floor (default 0.30) per W3.A spec.
DEFAULT_MIN_FEEDBACK_OVERLAP_JACCARD: float = 0.30

#: Calibration-readback signal name for the severity flip. Lives on
#: the W2.B canonical report's ``summary`` block.
CALIBRATION_SIGNAL_NAME: str = "answerability_alignment_rate"

#: Floor at or above which the calibration-gated severity flip fires.
CALIBRATION_FLIP_THRESHOLD: float = 0.85

#: Cap per-validate() issue list.
_ISSUE_LIST_CAP: int = 50

#: Stop-word set scoped to the validator. Mirrors the inline list
#: used by ``rewrite_source_grounding.py`` so the two grounding axes
#: tokenise identically.
_STOPWORDS: frozenset = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "but", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "can", "could", "may", "might", "this", "that",
        "these", "those", "with", "from", "by", "as", "it", "its",
        "their", "they", "them", "we", "our", "us", "you", "your",
    }
)

#: Word-token regex (latin alphabetic with apostrophes / hyphens).
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

#: Rewrite-tier <li data-cf-correct="true"> matcher.
_LI_CORRECT_RE = re.compile(
    r"<li[^>]*\bdata-cf-correct\s*=\s*\"true\"[^>]*>(.*?)</li>",
    re.IGNORECASE | re.DOTALL,
)

#: Rewrite-tier <li data-cf-distractor-index="N"> matcher (used to
#: locate the correct-answer index when ``data-cf-correct`` is absent
#: but the block carries an explicit
#: ``data-cf-correct-answer-index="N"`` attribute on the <ol>).
_LI_DISTRACTOR_RE = re.compile(
    r"<li[^>]*\bdata-cf-distractor-index\s*=\s*\"(\d+)\"[^>]*>(.*?)</li>",
    re.IGNORECASE | re.DOTALL,
)

#: <ol data-cf-correct-answer-index="N"> root attribute matcher.
_OL_CAI_RE = re.compile(
    r"data-cf-correct-answer-index\s*=\s*\"(\d+)\"",
    re.IGNORECASE,
)

#: Dangling-reference regex for the W3.A STEM_NOT_SELF_CONTAINED
#: sub-check. Catches "above" / "previous" / "next" / "figure N" /
#: "table N" / "see page N" / "see section N" / "the figure".
#: Whole-word match via ``\b`` boundaries; case-insensitive.
_DANGLING_REFERENCE_RE = re.compile(
    r"\b(above|previous|next|figure \d+|table \d+|"
    r"see (?:page|section)|the figure)\b",
    re.IGNORECASE,
)

#: Pronouns whose use as the first content token of a stem (no
#: antecedent) signals a non-self-contained question.
_PRONOUN_LEAD_TOKENS: frozenset = frozenset(
    {"it", "this", "these", "those"}
)

#: Rewrite-tier ``<div data-cf-feedback>`` matcher for FEEDBACK_NOT_GROUNDED.
_DIV_FEEDBACK_RE = re.compile(
    r"<(?:div|section|p)[^>]*\bdata-cf-feedback(?:\s*=\s*\"[^\"]*\")?[^>]*>(.*?)</(?:div|section|p)>",
    re.IGNORECASE | re.DOTALL,
)

#: Rewrite-tier ``data-cf-answer-span="<chunk_id>:<start>-<end>"``
#: shape matcher for ANSWER_SPAN_MISSING (presence-only check —
#: the value's correctness is out of scope).
_ANSWER_SPAN_RE = re.compile(
    r"\bdata-cf-answer-span\s*=\s*\"([^\"]+:\d+-\d+)\"",
    re.IGNORECASE,
)

#: Outline-tier feedback content keys the W3.A FEEDBACK_NOT_GROUNDED
#: sub-check inspects.
_OUTLINE_FEEDBACK_KEYS: tuple = ("feedback", "rationale", "explanation")


def _content_tokens(text: str) -> set:
    """Lowercase, stop-word-stripped, alphabetic-only token set."""
    if not text:
        return set()
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two token sets."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _coerce_blocks(inputs: Dict[str, Any]) -> Tuple[List[Any], Optional[GateIssue]]:
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _resolve_chunks_lookup(inputs: Dict[str, Any]) -> Dict[str, str]:
    """Pull the source-chunks mapping from ``inputs``.

    Accepts either ``chunks_lookup`` (canonical) or ``source_chunks``
    (legacy / sibling-validator alias) — both map sourceId → chunk
    text. Non-string values are dropped silently.
    """
    raw = inputs.get("chunks_lookup")
    if raw is None:
        raw = inputs.get("source_chunks")
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, str)}


def _block_referenced_chunk_ids(block: Any) -> List[str]:
    """Return the union of source IDs declared by the block."""
    seen: set = set()
    ids: List[str] = []
    for ref in getattr(block, "source_references", ()) or ():
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    for sid in getattr(block, "source_ids", ()) or ():
        if isinstance(sid, str) and sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def _extract_outline_answer_text(content: Dict[str, Any]) -> Optional[str]:
    """Return the correct-answer text from an outline-tier dict.

    Resolution chain:
      1. ``content["answer_key"]`` if non-empty string.
      2. ``content["correct_answer"]`` if non-empty string.
      3. ``content["distractors"][correct_answer_index]["text"]``.
      4. None — caller emits ``ANSWER_TEXT_MISSING`` or
         ``ESSAY_SKIPPED`` based on ``content_type``.
    """
    for key in ("answer_key", "correct_answer"):
        val = content.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    distractors = content.get("distractors")
    cai = content.get("correct_answer_index")
    if (
        isinstance(distractors, list)
        and isinstance(cai, int)
        and 0 <= cai < len(distractors)
    ):
        entry = distractors[cai]
        if isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _extract_rewrite_answer_text(html: str) -> Optional[str]:
    """Return the correct-answer text from a rewrite-tier HTML string.

    Resolution chain:
      1. ``<li data-cf-correct="true">…</li>`` body.
      2. ``data-cf-correct-answer-index="N"`` on the <ol> root +
         ``<li data-cf-distractor-index="N">…</li>``.
      3. None.
    """
    correct_match = _LI_CORRECT_RE.search(html)
    if correct_match:
        body = strip_html_to_text(correct_match.group(1) or "")
        if body:
            return body

    cai_match = _OL_CAI_RE.search(html)
    if cai_match:
        try:
            target_idx = int(cai_match.group(1))
        except (TypeError, ValueError):
            target_idx = None
        if target_idx is not None:
            for li_match in _LI_DISTRACTOR_RE.finditer(html):
                try:
                    idx = int(li_match.group(1))
                except (TypeError, ValueError):
                    continue
                if idx == target_idx:
                    body = strip_html_to_text(li_match.group(2) or "")
                    if body:
                        return body
    return None


def _extract_stem_text(block: Any) -> Optional[str]:
    """Return the question-stem text from either content shape.

    Outline (dict): ``content["stem"]`` / ``content["question"]`` /
    ``content["prompt"]`` (first non-empty string wins).
    Rewrite (HTML str): the first ``<p>`` element text.
    """
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        for key in ("stem", "question", "prompt"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    if isinstance(content, str):
        # Strip the answer list first so the answer text doesn't
        # leak into the stem-token set on rewrite-tier blocks.
        no_ol = re.sub(
            r"<ol[^>]*>.*?</ol>", " ", content, flags=re.IGNORECASE | re.DOTALL
        )
        # Pull the first <p>...</p>.
        p_match = re.search(
            r"<p[^>]*>(.*?)</p>", no_ol, re.IGNORECASE | re.DOTALL
        )
        if p_match:
            stripped = strip_html_to_text(p_match.group(1) or "")
            if stripped:
                return stripped
        # Fallback: strip the whole HTML.
        stripped = strip_html_to_text(no_ol)
        if stripped:
            return stripped
    return None


def _extract_feedback_text(block: Any) -> Optional[str]:
    """Return the feedback / rationale text from either content shape.

    Outline: any of ``content["feedback"|"rationale"|"explanation"]``.
    Rewrite: ``<div data-cf-feedback>...</div>`` body.
    Returns ``None`` when the block carries no feedback surface
    (in which case the FEEDBACK_NOT_GROUNDED check is no-op for it).
    """
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        for key in _OUTLINE_FEEDBACK_KEYS:
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    if isinstance(content, str):
        match = _DIV_FEEDBACK_RE.search(content)
        if match:
            stripped = strip_html_to_text(match.group(1) or "")
            if stripped:
                return stripped
    return None


def _has_answer_span(block: Any) -> bool:
    """True iff the block carries an explicit answer-span pointer.

    Outline: ``content["answer_span"]`` is a dict with the three
    canonical keys (``source_chunk_id`` / ``char_start`` /
    ``char_end``); the values are not validated beyond presence +
    type.
    Rewrite: any element carries
    ``data-cf-answer-span="<chunk_id>:<start>-<end>"``.
    """
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        span = content.get("answer_span")
        if isinstance(span, dict):
            sid = span.get("source_chunk_id")
            cs = span.get("char_start")
            ce = span.get("char_end")
            return (
                isinstance(sid, str) and bool(sid)
                and isinstance(cs, int) and cs >= 0
                and isinstance(ce, int) and ce >= cs
            )
        return False
    if isinstance(content, str):
        return _ANSWER_SPAN_RE.search(content) is not None
    return False


def _stem_starts_with_unbound_pronoun(stem_text: str) -> bool:
    """Heuristic: True when the stem opens with It/This/These/Those.

    A stem like "This is the capital of France?" has no antecedent
    inside the block prose because the only prose IS the stem itself.
    The check is intentionally first-token-only — pronouns deeper in
    the stem can be bound by earlier nouns and are out of scope.
    """
    tokens = _TOKEN_RE.findall(stem_text.strip().lower())
    if not tokens:
        return False
    return tokens[0] in _PRONOUN_LEAD_TOKENS


def _is_essay_block(block: Any) -> bool:
    """True when the block is an essay / open-ended response.

    Outline-tier marker: ``content["question_type"] == "essay"`` OR
    ``content["content_type"] == "essay"``. Rewrite-tier essays are
    rare in the canonical scaffolding; we mirror the outline check
    against ``block.content_type_label``.
    """
    label = getattr(block, "content_type_label", None)
    if isinstance(label, str) and label.lower() == "essay":
        return True
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        for key in ("question_type", "content_type"):
            val = content.get(key)
            if isinstance(val, str) and val.lower() == "essay":
                return True
    return False


def _emit_decision(
    capture: Any,
    *,
    audited: int,
    grounded: int,
    failed: int,
    skipped_essay: int,
    skipped_no_text: int,
    threshold: float,
    per_block_features: List[Dict[str, Any]],
    overall_passed: bool,
    sub_check_counts: Dict[str, int],
    severity_flip_applied: bool,
) -> None:
    """Emit one ``assessment_retrieval_grounding_check`` decision per call."""
    if capture is None:
        return
    decision = "passed" if overall_passed else "failed"
    rationale = (
        f"Assessment retrieval-grounding check: audited={audited}, "
        f"grounded={grounded}, failed={failed}, "
        f"skipped_essay={skipped_essay}, "
        f"skipped_no_text={skipped_no_text}, "
        f"min_overlap_jaccard={threshold:.3f}, "
        f"sub_checks={sub_check_counts}, "
        f"severity_flip_applied={severity_flip_applied}, "
        f"verdict={decision}."
    )
    try:
        capture.log_decision(
            decision_type="assessment_retrieval_grounding_check",
            decision=decision,
            rationale=rationale,
            ml_features={
                "min_overlap_jaccard": threshold,
                "audited_count": audited,
                "grounded_count": grounded,
                "failed_count": failed,
                "skipped_essay_count": skipped_essay,
                "skipped_no_text_count": skipped_no_text,
                "per_block": per_block_features[:_ISSUE_LIST_CAP],
                "sub_check_counts": dict(sub_check_counts),
                "severity_flip_applied": severity_flip_applied,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "assessment_retrieval_grounding_check: %s",
            exc,
        )


def _emit_severity_flip_decision(
    capture: Any, payload: Dict[str, Any]
) -> None:
    """Forward the calibration-gate decision payload to the capture handle."""
    if capture is None or not payload:
        return
    try:
        capture.log_decision(**payload)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on severity-flip event: %s",
            exc,
        )


class AssessmentRetrievalGroundingValidator:
    """Critical sentinel: every assessment-item answer text must
    overlap a referenced source chunk above the Jaccard floor + four
    new W3.A authoring-quality sub-checks.

    Inputs:
        blocks: List[Block]
            The full block batch from the calling phase. The
            validator filters internally to
            ``block.block_type == "assessment_item"``.
        chunks_lookup: Dict[str, str]
            Mapping of canonical sourceId (e.g. ``dart:slug#blk_0``)
            to the chunk's plain-text body. Alternatively spelled
            ``source_chunks`` for symmetry with sibling validators.
        min_overlap_jaccard: Optional[float]
            Override DEFAULT_MIN_OVERLAP_JACCARD (default 0.30).
        max_stem_answer_jaccard: Optional[float]
            Override DEFAULT_MAX_STEM_ANSWER_JACCARD (default 0.50).
        min_feedback_overlap_jaccard: Optional[float]
            Override DEFAULT_MIN_FEEDBACK_OVERLAP_JACCARD (default 0.30).
        decision_capture: Optional[DecisionCapture]
            When wired, one ``assessment_retrieval_grounding_check``
            event per validate() call PLUS one ``severity_flip_*``
            event when the calibration-readback fires (lazily on
            first validate() call).

    Failure-mode codes:
        ANSWER_NOT_GROUNDED (critical) — Jaccard below threshold.
        ANSWER_TEXT_MISSING (critical) — assessment_item has no
            recoverable correct-answer text.
        NO_SOURCE_ATTRIBUTION (warning, calibration-gated → critical) —
            block declares no source_references / source_ids. W3.A
            drops the legacy all-chunk-union fallback; the per-block
            grounding check is skipped to avoid a false-grounded signal.
        STEM_NOT_SELF_CONTAINED (warning) — stem carries a dangling
            reference or opens with an unbound pronoun (W3.A).
        ANSWER_LEAKED_IN_STEM (warning) — Jaccard(stem, answer) >
            ``max_stem_answer_jaccard`` (W3.A).
        FEEDBACK_NOT_GROUNDED (warning) — feedback / rationale text
            overlap with referenced chunks below
            ``min_feedback_overlap_jaccard`` (W3.A).
        ANSWER_SPAN_MISSING (warning, hygiene only) — block carries
            no explicit answer-span pointer (W3.A).
        ESSAY_SKIPPED (info) — essay-mode block; skipped from gate.
    """

    name = "assessment_retrieval_grounding"
    version = "1.1.0"

    def __init__(
        self,
        *,
        min_overlap_jaccard: float = DEFAULT_MIN_OVERLAP_JACCARD,
        max_stem_answer_jaccard: float = DEFAULT_MAX_STEM_ANSWER_JACCARD,
        min_feedback_overlap_jaccard: float = (
            DEFAULT_MIN_FEEDBACK_OVERLAP_JACCARD
        ),
        calibration_signal: str = CALIBRATION_SIGNAL_NAME,
        calibration_threshold: float = CALIBRATION_FLIP_THRESHOLD,
        calibration_libv2_root: Optional[Path] = None,
    ) -> None:
        self._min_overlap = min_overlap_jaccard
        self._max_stem_answer = max_stem_answer_jaccard
        self._min_feedback_overlap = min_feedback_overlap_jaccard
        # Resolve the calibration-gated severity flip once at
        # construction. The decision is captured per-validate() call
        # via ``_emit_severity_flip_decision`` so the audit trail
        # records the flip outcome alongside the grounding-check
        # event.
        self._severity_flip_applied, self._severity_flip_payload = (
            resolve_severity_flip(
                calibration_signal=calibration_signal,
                threshold=calibration_threshold,
                libv2_root=calibration_libv2_root,
            )
        )
        # Track whether the flip event has already fired against a
        # particular capture handle so a long-lived validator instance
        # used across multiple validate() calls only emits one
        # severity_flip_* event per capture.
        self._flip_emitted_for: set = set()

    @property
    def severity_flip_applied(self) -> bool:
        """Public read-only view of the calibration flip outcome."""
        return self._severity_flip_applied

    def _no_source_severity(self) -> str:
        """Resolve the severity for ``NO_SOURCE_ATTRIBUTION`` per flip."""
        return "critical" if self._severity_flip_applied else "warning"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        threshold = float(
            inputs.get("min_overlap_jaccard", self._min_overlap)
        )
        max_stem_answer = float(
            inputs.get("max_stem_answer_jaccard", self._max_stem_answer)
        )
        min_feedback_overlap = float(
            inputs.get(
                "min_feedback_overlap_jaccard", self._min_feedback_overlap
            )
        )

        # Emit the severity-flip decision-capture event once per
        # capture handle so the audit trail records why the validator
        # is using critical-vs-warning severity for the promoted code.
        if capture is not None and id(capture) not in self._flip_emitted_for:
            _emit_severity_flip_decision(capture, self._severity_flip_payload)
            self._flip_emitted_for.add(id(capture))

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        chunks_lookup = _resolve_chunks_lookup(inputs)

        issues: List[GateIssue] = []
        per_block_features: List[Dict[str, Any]] = []
        audited = 0
        grounded = 0
        failed = 0
        skipped_essay = 0
        skipped_no_text = 0

        # Per-sub-check counters (W3.A telemetry).
        sub_check_counts: Dict[str, int] = {
            "stem_not_self_contained": 0,
            "answer_leaked_in_stem": 0,
            "feedback_not_grounded": 0,
            "answer_span_missing": 0,
            "no_source_attribution": 0,
        }

        for block in blocks:
            block_type = getattr(block, "block_type", None)
            if block_type != "assessment_item":
                continue

            block_id = getattr(block, "block_id", "<unknown>")

            # Essay blocks have no fixed answer to ground.
            if _is_essay_block(block):
                skipped_essay += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="info",
                        code="ESSAY_SKIPPED",
                        message=(
                            f"assessment_item Block {block_id!r} is an "
                            f"essay-mode question with no fixed correct-"
                            f"answer text; skipped from grounding check."
                        ),
                        location=block_id,
                    ))
                per_block_features.append({
                    "block_id": block_id,
                    "verdict": "skipped_essay",
                    "overlap": None,
                })
                continue

            # Extract the correct-answer text per content shape.
            content = getattr(block, "content", None)
            answer_text: Optional[str] = None
            if isinstance(content, dict):
                answer_text = _extract_outline_answer_text(content)
            elif isinstance(content, str):
                answer_text = _extract_rewrite_answer_text(content)

            if not answer_text:
                skipped_no_text += 1
                failed += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="ANSWER_TEXT_MISSING",
                        message=(
                            f"assessment_item Block {block_id!r} carries "
                            f"no recoverable correct-answer text "
                            f"(checked answer_key / correct_answer / "
                            f"distractors[correct_answer_index] on outline; "
                            f"<li data-cf-correct=\"true\"> + "
                            f"data-cf-correct-answer-index on rewrite)."
                        ),
                        location=block_id,
                        suggestion=(
                            "Re-roll the block with an explicit answer "
                            "field; the grounding gate cannot evaluate "
                            "answerability without an answer string."
                        ),
                    ))
                per_block_features.append({
                    "block_id": block_id,
                    "verdict": "missing_answer_text",
                    "overlap": None,
                })
                continue

            audited += 1
            answer_tokens = _content_tokens(answer_text)

            # ---- W3.A sub-checks (warning hygiene) -----------------
            stem_text = _extract_stem_text(block)
            stem_tokens = _content_tokens(stem_text or "")

            # Sub-check #1: STEM_NOT_SELF_CONTAINED
            stem_not_self_contained = False
            if stem_text:
                if _DANGLING_REFERENCE_RE.search(stem_text):
                    stem_not_self_contained = True
                elif _stem_starts_with_unbound_pronoun(stem_text):
                    stem_not_self_contained = True
            if stem_not_self_contained:
                sub_check_counts["stem_not_self_contained"] += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="STEM_NOT_SELF_CONTAINED",
                        message=(
                            f"assessment_item Block {block_id!r}: stem "
                            f"contains a dangling reference (e.g. "
                            f"'see above', 'previous figure', 'table N') "
                            f"or opens with an unbound pronoun "
                            f"(It/This/These/Those); rewrite the stem to "
                            f"stand alone without prior context."
                        ),
                        location=block_id,
                    ))

            # Sub-check #2: ANSWER_LEAKED_IN_STEM
            stem_answer_overlap = (
                _jaccard(stem_tokens, answer_tokens)
                if stem_tokens and answer_tokens
                else 0.0
            )
            answer_leaked = (
                stem_tokens and answer_tokens
                and stem_answer_overlap > max_stem_answer
            )
            if answer_leaked:
                sub_check_counts["answer_leaked_in_stem"] += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="ANSWER_LEAKED_IN_STEM",
                        message=(
                            f"assessment_item Block {block_id!r}: "
                            f"Jaccard(stem, answer) = "
                            f"{stem_answer_overlap:.3f} > ceiling "
                            f"{max_stem_answer:.3f}; the stem appears to "
                            f"telegraph the answer."
                        ),
                        location=block_id,
                        suggestion=(
                            "Rewrite the stem so it does not reuse the "
                            "answer's content tokens verbatim."
                        ),
                    ))

            # Sub-check #3: FEEDBACK_NOT_GROUNDED
            feedback_text = _extract_feedback_text(block)
            feedback_overlap_value: Optional[float] = None
            if feedback_text:
                feedback_tokens = _content_tokens(feedback_text)
                # Build the chunk-token union (referenced chunks only;
                # falls through with empty union when source IDs are
                # absent — handled by the ANSWER_NOT_GROUNDED branch
                # plus the per-block fallback below).
                chunk_ids_for_feedback = _block_referenced_chunk_ids(block)
                feedback_chunk_tokens: set = set()
                for sid in chunk_ids_for_feedback:
                    text = chunks_lookup.get(sid)
                    if isinstance(text, str) and text.strip():
                        feedback_chunk_tokens |= _content_tokens(text)
                feedback_overlap_value = _jaccard(
                    feedback_tokens, feedback_chunk_tokens
                )
                if (
                    feedback_chunk_tokens
                    and feedback_overlap_value < min_feedback_overlap
                ):
                    sub_check_counts["feedback_not_grounded"] += 1
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="warning",
                            code="FEEDBACK_NOT_GROUNDED",
                            message=(
                                f"assessment_item Block {block_id!r}: "
                                f"feedback / rationale token overlap "
                                f"{feedback_overlap_value:.3f} < floor "
                                f"{min_feedback_overlap:.3f} against "
                                f"{len(chunk_ids_for_feedback)} referenced "
                                f"chunk(s); feedback appears to be free-form "
                                f"fabrication."
                            ),
                            location=block_id,
                            suggestion=(
                                "Anchor the feedback prose in the cited "
                                "source content or re-roll the feedback."
                            ),
                        ))

            # Sub-check #4: ANSWER_SPAN_MISSING (hygiene — stays
            # warning even if the calibration flip critical-promotes
            # NO_SOURCE_ATTRIBUTION).
            if not _has_answer_span(block):
                sub_check_counts["answer_span_missing"] += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="ANSWER_SPAN_MISSING",
                        message=(
                            f"assessment_item Block {block_id!r}: no "
                            f"explicit answer-span pointer "
                            f"(content['answer_span'] on outline; "
                            f"data-cf-answer-span=\"<chunk_id>:<start>-<end>\""
                            f" on rewrite). The grounding gate cannot pin "
                            f"the answer to a specific source range."
                        ),
                        location=block_id,
                        suggestion=(
                            "Author an answer_span object pointing at the "
                            "exact source chunk + character offsets."
                        ),
                    ))

            # ---- Existing answer-grounding axis ---------------------
            chunk_ids = _block_referenced_chunk_ids(block)
            chunk_tokens: set = set()
            referenced_count = 0
            if chunk_ids:
                for sid in chunk_ids:
                    text = chunks_lookup.get(sid)
                    if isinstance(text, str) and text.strip():
                        chunk_tokens |= _content_tokens(text)
                        referenced_count += 1
                fallback_used = False
            else:
                # W3.A drops the legacy all-chunk-union fallback. A
                # block with no source attribution cannot be defended
                # under the hard-gate posture, so we emit
                # NO_SOURCE_ATTRIBUTION at the calibration-resolved
                # severity AND skip the per-block overlap check
                # (otherwise we'd silently grade against an empty token
                # set, scoring a deceptive 0.0 overlap).
                fallback_used = True
                sub_check_counts["no_source_attribution"] += 1
                no_source_severity = self._no_source_severity()
                if no_source_severity == "critical":
                    failed += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity=no_source_severity,
                        code="NO_SOURCE_ATTRIBUTION",
                        message=(
                            f"assessment_item Block {block_id!r} declares "
                            f"no source_references / source_ids; under "
                            f"hard-gate posture the answer-grounding "
                            f"check cannot be evaluated and the block is "
                            f"rejected (W3.A dropped the legacy all-chunk-"
                            f"union fallback)."
                        ),
                        location=block_id,
                        suggestion=(
                            "Attach the canonical source chunks (via "
                            "block.source_references[].sourceId or "
                            "block.source_ids) before re-running the gate."
                        ),
                    ))
                per_block_features.append({
                    "block_id": block_id,
                    "verdict": "no_source_attribution",
                    "overlap": None,
                    "fallback_used": fallback_used,
                    "stem_answer_overlap": (
                        round(stem_answer_overlap, 4)
                        if stem_tokens and answer_tokens
                        else None
                    ),
                    "feedback_overlap": (
                        round(feedback_overlap_value, 4)
                        if feedback_overlap_value is not None
                        else None
                    ),
                })
                continue

            overlap = _jaccard(answer_tokens, chunk_tokens)

            if overlap < threshold:
                failed += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="ANSWER_NOT_GROUNDED",
                        message=(
                            f"assessment_item Block {block_id!r}: answer "
                            f"text overlap {overlap:.3f} < threshold "
                            f"{threshold:.3f} against "
                            f"{referenced_count} "
                            f"source chunk(s); answer cannot be defended "
                            f"from the corpus."
                        ),
                        location=block_id,
                        suggestion=(
                            "Either re-roll the question against the "
                            "actual source content or attach the chunks "
                            "that genuinely support the answer text."
                        ),
                    ))
                per_block_features.append({
                    "block_id": block_id,
                    "verdict": "not_grounded",
                    "overlap": round(overlap, 4),
                    "answer_token_count": len(answer_tokens),
                    "chunk_token_count": len(chunk_tokens),
                    "fallback_used": fallback_used,
                    "stem_answer_overlap": (
                        round(stem_answer_overlap, 4)
                        if stem_tokens and answer_tokens
                        else None
                    ),
                    "feedback_overlap": (
                        round(feedback_overlap_value, 4)
                        if feedback_overlap_value is not None
                        else None
                    ),
                })
            else:
                grounded += 1
                per_block_features.append({
                    "block_id": block_id,
                    "verdict": "grounded",
                    "overlap": round(overlap, 4),
                    "answer_token_count": len(answer_tokens),
                    "chunk_token_count": len(chunk_tokens),
                    "fallback_used": fallback_used,
                    "stem_answer_overlap": (
                        round(stem_answer_overlap, 4)
                        if stem_tokens and answer_tokens
                        else None
                    ),
                    "feedback_overlap": (
                        round(feedback_overlap_value, 4)
                        if feedback_overlap_value is not None
                        else None
                    ),
                })

        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        # Score: grounded / audited; 1.0 when nothing was audited.
        score = 1.0 if audited == 0 else round(grounded / audited, 4)

        _emit_decision(
            capture,
            audited=audited,
            grounded=grounded,
            failed=failed,
            skipped_essay=skipped_essay,
            skipped_no_text=skipped_no_text,
            threshold=threshold,
            per_block_features=per_block_features,
            overall_passed=passed,
            sub_check_counts=sub_check_counts,
            severity_flip_applied=self._severity_flip_applied,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


__all__ = [
    "AssessmentRetrievalGroundingValidator",
    "DEFAULT_MIN_OVERLAP_JACCARD",
    "DEFAULT_MAX_STEM_ANSWER_JACCARD",
    "DEFAULT_MIN_FEEDBACK_OVERLAP_JACCARD",
    "CALIBRATION_SIGNAL_NAME",
    "CALIBRATION_FLIP_THRESHOLD",
]
