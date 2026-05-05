"""Worker W2 — answer-text↔source-chunk grounding sentinel.

Critical-severity gate that closes the answerability gap for
assessment-item Blocks: every assessment question's correct-answer
text must overlap the retrieved source chunks above a Jaccard
content-token threshold, OR the question is rejected. Complements
``RewriteSourceGroundingValidator`` which gates the *stem-vs-source*
axis on rewrite-tier prose blocks; W2 is the *answer-vs-source* axis
on assessment items, which the rewrite-tier validator explicitly
skips per its ``_SKIPPED_CONTENT_TYPES``.

Contract per ``plans/gpt-feedback-w2-w7-execution-2026-05.md`` §W2:

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
  text). When the block declares no source IDs, fall back to the
  union of all chunks AND emit ``NO_SOURCE_ATTRIBUTION`` warning.
- Compute Jaccard overlap of content-word tokens (lowercased,
  stop-word-stripped). Below the configured floor (default 0.30) →
  ``ANSWER_NOT_GROUNDED`` critical.

Decision-capture: emits one ``assessment_retrieval_grounding_check``
event per ``validate()`` call carrying per-block overlap scores +
threshold + verdict in ``ml_features`` shape. Rationale interpolates
audited count, threshold, and overall pass/fail.

References:
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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

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


class _TextExtractor(HTMLParser):
    """Stdlib HTML parser that accumulates text from data events."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._fragments: List[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._fragments.append(data)

    def text(self) -> str:
        return " ".join(self._fragments).strip()


def _strip_html_to_text(html: str) -> str:
    """Strip HTML to plain text via stdlib parser."""
    if not html:
        return ""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.debug("HTML strip raised: %s", exc)
        return ""
    return extractor.text()


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
        body = _strip_html_to_text(correct_match.group(1) or "")
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
                    body = _strip_html_to_text(li_match.group(2) or "")
                    if body:
                        return body
    return None


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
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "assessment_retrieval_grounding_check: %s",
            exc,
        )


class AssessmentRetrievalGroundingValidator:
    """Critical sentinel: every assessment-item answer text must
    overlap a referenced source chunk above the Jaccard floor.

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
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per validate() call.

    Failure-mode codes:
        ANSWER_NOT_GROUNDED (critical) — Jaccard below threshold.
        ANSWER_TEXT_MISSING (critical) — assessment_item has no
            recoverable correct-answer text.
        NO_SOURCE_ATTRIBUTION (warning) — block declares no
            source_references / source_ids; falls back to all-
            chunk-union comparison.
        ESSAY_SKIPPED (info) — essay-mode block; skipped from gate.
    """

    name = "assessment_retrieval_grounding"
    version = "1.0.0"

    def __init__(
        self,
        *,
        min_overlap_jaccard: float = DEFAULT_MIN_OVERLAP_JACCARD,
    ) -> None:
        self._min_overlap = min_overlap_jaccard

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        threshold = float(
            inputs.get("min_overlap_jaccard", self._min_overlap)
        )

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

        # Pre-compute the union-of-all-chunks token set for the
        # NO_SOURCE_ATTRIBUTION fallback path.
        all_chunks_tokens: set = set()
        for chunk_text in chunks_lookup.values():
            all_chunks_tokens |= _content_tokens(chunk_text)

        issues: List[GateIssue] = []
        per_block_features: List[Dict[str, Any]] = []
        audited = 0
        grounded = 0
        failed = 0
        skipped_essay = 0
        skipped_no_text = 0

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

            # Resolve referenced chunks; fall back to all-chunk-union
            # when the block declares no source IDs.
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
                chunk_tokens = all_chunks_tokens
                fallback_used = True
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="NO_SOURCE_ATTRIBUTION",
                        message=(
                            f"assessment_item Block {block_id!r} declares "
                            f"no source_references / source_ids; the "
                            f"grounding gate fell back to the union of "
                            f"all chunks for the overlap check."
                        ),
                        location=block_id,
                    ))

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
                            f"{referenced_count if not fallback_used else len(chunks_lookup)} "
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
]
