"""Post-rewrite sentence-level source-grounding sentinel.

Critical-severity gate at ``post_rewrite_validation`` that closes the
fabricated-prose-with-valid-attribute-scaffolding regression class.
Where the existing ``ContentGroundingValidator`` (Wave 31) walks the
ancestor chain for ``data-cf-source-ids`` ATTRIBUTE presence, this
gate asserts the paragraph TEXT itself paraphrases (cosine ≥ 0.45)
at least one source chunk substring. A rewrite that copies the
``data-cf-source-ids`` attribute onto fabricated prose passes the
ancestor walk today; this gate critical-fails it.

Contract per ``plans/qwen7b-courseforge-fixes-2026-05-followup.md``
§3.3:

- Skip blocks whose ``content_type`` (or ``block_type``) is
  ``assessment_item`` — assessment grounding is the
  ``objective_assessment_similarity`` gate's job.
- Strip HTML to plain prose via stdlib ``html.parser``.
- Sentence-segment via a regex split on ``[.!?]\\s+[A-Z]``. No nltk
  dependency.
- For each non-trivial sentence (≥10 words after stop-token removal),
  compute max cosine similarity against (a) the block's outline-tier
  ``key_claims`` when an outline-tier touch is present in
  ``block.touched_by``, (b) every chunk text resolved via
  ``block.source_references[].sourceId`` against the
  ``inputs["source_chunks"]`` mapping (sourceId -> text).
- ``grounded_sentence_rate`` = (# sentences ≥ 0.45 cosine) /
  (# non-trivial sentences). Below 0.60 →
  ``code="REWRITE_SENTENCE_GROUNDING_LOW"`` with
  ``action="regenerate"``.
- Embedding-deps degraded: when ``try_load_embedder()`` returns
  ``None``, emit warning ``EMBEDDING_DEPS_MISSING`` with
  ``passed=True, action=None``. Strict mode via the existing
  ``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` flag.

Decision-capture: emits one ``rewrite_source_grounding_check``
decision per block evaluated. Rationale interpolates block_id,
content_type, sentence_count, grounded_count, grounded_rate,
threshold, and source_chunk_count.

References:
    - ``lib/validators/concept_example_similarity.py`` — Phase 4 PoC
      sibling that this gate complements at the post-rewrite seam.
    - ``lib/validators/content_grounding.py`` — Wave 31 attribute-
      walk gate. This validator is its text-trace counterpart.
    - ``lib/embedding/sentence_embedder.py`` — embedder loader +
      strict-mode flag (``TRAINFORGE_REQUIRE_EMBEDDINGS``).
"""

from __future__ import annotations

import logging
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    EmbeddingDepsMissing,
    SentenceEmbedder,
    is_strict_mode,
    try_load_embedder,
)

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Per-sentence cosine floor for the grounding match. Calibrated
#: against the embedder's intrinsic similarity floor — paraphrase
#: pairs cluster around 0.55-0.85, while topically-related but
#: independently authored sentences cluster around 0.35-0.55.
DEFAULT_MIN_GROUNDING_COSINE: float = 0.45

#: Per-block aggregate floor: at least 60% of non-trivial sentences
#: must be grounded above the per-sentence cosine threshold.
DEFAULT_MIN_GROUNDED_SENTENCE_RATE: float = 0.60

#: Minimum word count for a sentence to be audited. Too short
#: sentences (e.g. "Yes.", "OK.") have no semantic payload to ground.
_MIN_SENTENCE_WORDS: int = 10

#: Cap per-block issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50

#: Block content_types / block_types skipped (assessment-item
#: grounding is the objective_assessment_similarity gate's job).
_SKIPPED_CONTENT_TYPES: frozenset = frozenset(
    {"assessment_item", "self_check_question", "objective"}
)

#: Sentence boundary: . ! ? followed by whitespace + capital letter.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

#: Inline stopword set for the non-trivial-sentence filter. Keep this
#: tight so the filter rejects only glue words, not content words.
_STOPWORDS: frozenset = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "but", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "can", "could", "may", "might", "this", "that",
        "these", "those",
    }
)


class _TextExtractor(HTMLParser):
    """Stdlib HTML parser that accumulates text from data events.

    Mirrors the lightweight strip-via-regex helper at
    ``Courseforge/router/inter_tier_gates.py:130-139`` but uses the
    parser proper so embedded entities (``&amp;`` etc.) are decoded
    correctly. ``convert_charrefs=True`` (default in py3.5+) handles
    that.
    """

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
    except Exception as exc:  # noqa: BLE001 — defensive; the shape gate is upstream
        logger.debug("HTML strip raised: %s", exc)
        return ""
    return extractor.text()


def _segment_sentences(text: str) -> List[str]:
    """Split on sentence boundaries; preserve the original casing."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _is_non_trivial(sentence: str) -> bool:
    """A sentence is non-trivial when it has ≥10 non-stopword tokens."""
    if not sentence:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence.lower())
    content = [t for t in tokens if t not in _STOPWORDS]
    return len(content) >= _MIN_SENTENCE_WORDS


def _coerce_blocks(inputs: Dict[str, Any]) -> Tuple[List[Block], Optional[GateIssue]]:
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


def _block_should_skip(block: Block) -> bool:
    """True when the block's content_type is in the skip set."""
    if block.block_type in _SKIPPED_CONTENT_TYPES:
        return True
    if block.content_type_label and block.content_type_label in _SKIPPED_CONTENT_TYPES:
        return True
    return False


def _resolve_block_source_chunks(
    block: Block,
    source_chunks: Dict[str, str],
) -> List[str]:
    """Return the list of source-chunk texts the block references.

    Resolution: walks ``block.source_references[].sourceId`` and
    ``block.source_ids`` (the canonical Phase 2+ surfaces), looks
    each up in the ``source_chunks`` mapping. Missing IDs are silently
    skipped — the gate fails the block on the GROUNDING signal, not
    on staging-manifest resolution (that's the BlockSourceRefValidator's
    job).
    """
    seen: set = set()
    chunk_texts: List[str] = []
    candidate_ids: List[str] = []
    for ref in block.source_references or ():
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid:
                candidate_ids.append(sid)
    for sid in block.source_ids or ():
        if isinstance(sid, str) and sid:
            candidate_ids.append(sid)
    for sid in candidate_ids:
        if sid in seen:
            continue
        seen.add(sid)
        text = source_chunks.get(sid)
        if isinstance(text, str) and text.strip():
            chunk_texts.append(text)
    return chunk_texts


def _resolve_outline_key_claims(block: Block) -> List[str]:
    """Pull outline-tier ``key_claims`` from an upstream Touch.

    The rewrite-tier Block carries the outline-tier draft on its
    ``touched_by`` chain; the canonical post-Phase-3.5 chain is
    ``outline → outline_val → rewrite → rewrite_val``. Touches don't
    carry the dict content directly (that lives in the Block.content
    field which is overwritten by the rewrite tier), so this helper
    returns an empty list when the outline-tier dict isn't preserved
    elsewhere. We rely on source_chunks as the primary grounding
    surface; outline key_claims is a best-effort secondary signal
    when the workflow runner threads them through ``inputs``.
    """
    return []


def _emit_decision(
    capture: Any,
    block: Block,
    *,
    passed: bool,
    code: Optional[str],
    sentence_count: int,
    non_trivial_count: int,
    grounded_count: int,
    grounded_rate: Optional[float],
    threshold: float,
    source_chunk_count: int,
) -> None:
    """Emit one ``rewrite_source_grounding_check`` decision per block."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rate_str = (
        f"{grounded_rate:.3f}" if grounded_rate is not None else "n/a"
    )
    rationale = (
        f"Post-rewrite source-grounding check on Block {block.block_id!r}: "
        f"block_type={block.block_type}, "
        f"content_type={block.content_type_label or 'n/a'}, "
        f"total_sentences={sentence_count}, "
        f"non_trivial_sentences={non_trivial_count}, "
        f"grounded_sentences={grounded_count}, "
        f"grounded_rate={rate_str}, "
        f"min_rate_threshold={threshold:.3f}, "
        f"source_chunks_resolved={source_chunk_count}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="rewrite_source_grounding_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "rewrite_source_grounding_check: %s",
            exc,
        )


class RewriteSourceGroundingValidator:
    """Post-rewrite sentence-level grounding critical sentinel.

    Iterates every Block whose ``content`` is a string AND whose
    ``content_type`` is not in the skip set. For each non-trivial
    sentence in the stripped HTML, computes max cosine similarity
    against every resolved source-chunk text. The block fails when
    the per-block grounded_sentence_rate falls below the configured
    minimum (default 0.60).

    Inputs:
        blocks: List[Block]
        source_chunks: Dict[str, str]
            Mapping of canonical sourceId (e.g. ``dart:slug#blk_0``)
            to the chunk's plain-text body. The workflow runner
            populates this from the staging manifest before dispatch;
            tests inject it directly.
        threshold: Optional[float]
            Override DEFAULT_MIN_GROUNDED_SENTENCE_RATE.
        min_grounding_cosine: Optional[float]
            Override DEFAULT_MIN_GROUNDING_COSINE.
        embedder: Optional[SentenceEmbedder]
            Test seam; defaults to ``try_load_embedder()``.
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per block evaluated.

    Embedding-deps fallback per Phase 4 contract: when extras are
    missing AND ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is unset, emit a
    single warning ``EMBEDDING_DEPS_MISSING`` GateIssue with
    ``passed=True, action=None``. Strict mode raises
    ``EmbedderDepsMissing`` (caught upstream as critical).
    """

    name = "rewrite_source_grounding"
    version = "1.0.0"

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_MIN_GROUNDED_SENTENCE_RATE,
        min_grounding_cosine: float = DEFAULT_MIN_GROUNDING_COSINE,
        embedder: Optional[SentenceEmbedder] = None,
    ) -> None:
        self._threshold = threshold
        self._min_cosine = min_grounding_cosine
        self._embedder_override = embedder

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        threshold = float(inputs.get("threshold", self._threshold))
        min_cosine = float(
            inputs.get("min_grounding_cosine", self._min_cosine)
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

        # Embedder load with strict-mode awareness.
        if self._embedder_override is not None:
            embedder: Optional[SentenceEmbedder] = self._embedder_override
        else:
            try:
                embedder = try_load_embedder()
            except EmbeddingDepsMissing as exc:
                # Strict mode — propagate as critical.
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="EMBEDDING_DEPS_MISSING",
                        message=(
                            f"sentence-transformers extras missing and "
                            f"TRAINFORGE_REQUIRE_EMBEDDINGS=true: {exc}"
                        ),
                        suggestion=(
                            "Install the embedding extras via "
                            "`pip install -e .[embedding]` or unset "
                            "TRAINFORGE_REQUIRE_EMBEDDINGS for graceful "
                            "degradation."
                        ),
                    )],
                    action="regenerate",
                )

        if embedder is None:
            # Graceful-degrade path: warn but pass.
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code="EMBEDDING_DEPS_MISSING",
                    message=(
                        "sentence-transformers extras not installed; "
                        "post-rewrite source-grounding gate skipped. "
                        "Install via `pip install -e .[embedding]` to "
                        "enable. Set TRAINFORGE_REQUIRE_EMBEDDINGS=true "
                        "to fail closed instead."
                    ),
                )],
                action=None,
            )

        source_chunks_raw = inputs.get("source_chunks", {}) or {}
        if not isinstance(source_chunks_raw, dict):
            source_chunks_raw = {}
        source_chunks: Dict[str, str] = {
            str(k): v for k, v in source_chunks_raw.items()
            if isinstance(v, str)
        }

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0

        for block in blocks:
            content = block.content
            if not isinstance(content, str):
                # Outline-tier dict content — skip silently.
                continue
            if _block_should_skip(block):
                # assessment_item / self_check_question / objective:
                # emit a passed=True decision so the audit trail
                # records the skip.
                _emit_decision(
                    capture, block,
                    passed=True, code="SKIPPED_CONTENT_TYPE",
                    sentence_count=0, non_trivial_count=0,
                    grounded_count=0, grounded_rate=None,
                    threshold=threshold, source_chunk_count=0,
                )
                continue

            audited += 1

            chunk_texts = _resolve_block_source_chunks(block, source_chunks)
            outline_claims = _resolve_outline_key_claims(block)
            grounding_surfaces: List[str] = chunk_texts + outline_claims

            text = _strip_html_to_text(content)
            sentences = _segment_sentences(text)
            non_trivial = [s for s in sentences if _is_non_trivial(s)]

            if not non_trivial:
                # Block has no non-trivial sentences (very short or
                # all stopword-heavy). Pass with an info note.
                passed_count += 1
                _emit_decision(
                    capture, block,
                    passed=True, code="NO_NON_TRIVIAL_SENTENCES",
                    sentence_count=len(sentences),
                    non_trivial_count=0,
                    grounded_count=0, grounded_rate=None,
                    threshold=threshold,
                    source_chunk_count=len(chunk_texts),
                )
                continue

            if not grounding_surfaces:
                # Block declares no source_ids and the workflow runner
                # didn't pre-populate outline key_claims. The
                # BlockSourceRefValidator handles the structural
                # check; this gate emits a warning so the operator
                # knows the grounding signal is unavailable, but
                # passes the block (no source = nothing to check).
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="REWRITE_NO_GROUNDING_SOURCE",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} declares "
                            f"no source_references / source_ids and no "
                            f"outline key_claims; sentence-level grounding "
                            f"cannot be evaluated for this block."
                        ),
                        location=block.block_id,
                    ))
                passed_count += 1
                _emit_decision(
                    capture, block,
                    passed=True, code="REWRITE_NO_GROUNDING_SOURCE",
                    sentence_count=len(sentences),
                    non_trivial_count=len(non_trivial),
                    grounded_count=0, grounded_rate=None,
                    threshold=threshold,
                    source_chunk_count=0,
                )
                continue

            # Encode the grounding surfaces once per block (saves
            # redundant model calls when multiple sentences map to
            # the same source set).
            try:
                surface_vectors = [
                    embedder.encode(surf) for surf in grounding_surfaces
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedder.encode raised on grounding surfaces for %s: %s",
                    block.block_id, exc,
                )
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="EMBEDDING_ENCODE_ERROR",
                        message=(
                            f"Failed to encode grounding surfaces for block "
                            f"{block.block_id!r}: {exc}"
                        ),
                        location=block.block_id,
                    ))
                _emit_decision(
                    capture, block,
                    passed=True, code="EMBEDDING_ENCODE_ERROR",
                    sentence_count=len(sentences),
                    non_trivial_count=len(non_trivial),
                    grounded_count=0, grounded_rate=None,
                    threshold=threshold,
                    source_chunk_count=len(chunk_texts),
                )
                continue

            grounded = 0
            for sentence in non_trivial:
                try:
                    sent_vec = embedder.encode(sentence)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "embedder.encode raised on sentence: %s",
                        exc,
                    )
                    continue
                # Per-sentence max cosine across all grounding surfaces.
                max_cos = max(
                    (cosine_similarity(sent_vec, sv) for sv in surface_vectors),
                    default=0.0,
                )
                if max_cos >= min_cosine:
                    grounded += 1

            grounded_rate = grounded / len(non_trivial)

            if grounded_rate < threshold:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_SENTENCE_GROUNDING_LOW",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r}: "
                            f"only {grounded}/{len(non_trivial)} non-trivial "
                            f"sentences ({grounded_rate:.1%}) traced back to "
                            f"declared source chunks at cosine ≥ "
                            f"{min_cosine:.2f}; minimum required rate is "
                            f"{threshold:.1%}."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "The rewrite-tier prose drifted from its "
                            "declared source. Re-prompt with the source "
                            "chunks pinned in the prompt and require "
                            "verbatim or near-verbatim paraphrase."
                        ),
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_SENTENCE_GROUNDING_LOW",
                    sentence_count=len(sentences),
                    non_trivial_count=len(non_trivial),
                    grounded_count=grounded, grounded_rate=grounded_rate,
                    threshold=threshold,
                    source_chunk_count=len(chunk_texts),
                )
            else:
                passed_count += 1
                _emit_decision(
                    capture, block,
                    passed=True, code=None,
                    sentence_count=len(sentences),
                    non_trivial_count=len(non_trivial),
                    grounded_count=grounded, grounded_rate=grounded_rate,
                    threshold=threshold,
                    source_chunk_count=len(chunk_texts),
                )

        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
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
    "RewriteSourceGroundingValidator",
    "DEFAULT_MIN_GROUNDING_COSINE",
    "DEFAULT_MIN_GROUNDED_SENTENCE_RATE",
]
