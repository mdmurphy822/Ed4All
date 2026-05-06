"""Worker W2.E — TrainingPairPromotionValidator.

GPT Feedback v2 Wave 2 net-new validator. Per-pair, post-emit, pre-write
filter that gates every accepted instruction / preference pair before it
lands in ``training_specs/instruction_pairs.jsonl`` /
``preference_pairs.jsonl``. Closes the seven hard-rejection paths cited
in the GPT critique (lines 225-233):

1. **placeholder_residue** — re-applies the 13-regex
   ``ASSESSMENT_PLACEHOLDER_PATTERNS`` from
   :mod:`lib.validators.assessment` against every prompt / completion /
   chosen / rejected surface. Zero tolerance.
2. **unsupported_answer** — cosine similarity between the answer surface
   (``completion`` for instruction pairs, ``chosen`` for preference
   pairs) and the cited source-chunk text. Default floor
   ``min_answer_support_score=0.40``.
3. **weak_distractor** (preference only) — cosine semantic distinctness
   between ``chosen`` and ``rejected``. Default floor
   ``dpo_min_distractor_distinctness=0.40``.
4. **unanswerable_stem** — content-token Jaccard overlap of the
   ``prompt`` against the cited chunk text below
   ``min_prompt_chunk_jaccard=0.10``. Reuses the canonical
   ``_content_tokens`` + ``_jaccard`` helpers from
   :mod:`lib.validators.assessment_retrieval_grounding` so the two
   grounding axes share one tokenisation contract.
5. **source_free_generation** — ``source_chunk_id`` is missing or empty.
   The schema's ``trainable``-conditional already requires the field;
   the validator catches it loudly before the pair is checkpointed.
6. **low_bloom_alignment** — :class:`BloomBertEnsemble` classification of
   ``prompt + completion`` (or ``prompt + chosen`` for preference pairs)
   disagrees with the declared ``bloom_level``, AND the ensemble
   winner's confidence exceeds the
   :data:`lib.validators.bloom_classifier_disagreement._DISAGREEMENT_CONFIDENCE_FLOOR`
   (0.40 default). Stamps ``observed_bloom`` + ``bloom_alignment`` on
   every audited pair regardless of pass/fail so the audit trail
   carries the classification result.
7. **generic_rationale** — heuristic richness score on the ``rationale``
   field (unique-token-count / total-token-count). Default floor
   ``min_rationale_richness_score=0.30``.

Filter mechanics (call site, in
:mod:`Trainforge.synthesize_training`)::

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
    )
    pair.update(new_fields)
    if status == "rejected":
        stats.<kind>_pairs_rejected += 1
        stats.rejected_reasons[f"<kind>:promotion:{reason}"] += 1
        stats.dropped_count += 1
        continue  # do NOT append to records, sidecar, or checkpoint
    pair["promotion_status"] = status   # "validated"

Graceful degrade. When the optional ``[embedding]`` extras are absent,
criteria 2 + 3 (which need a sentence embedder for cosine similarity)
fall back to a Jaccard surrogate. The validator emits exactly one
warning-severity ``EMBEDDING_DEPS_MISSING`` :class:`GateIssue` per
:meth:`validate_pair` invocation when the fallback fires, so the audit
trail records the silent-degrade. Strict mode via
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` re-enables fail-closed (the
embedder loader raises and the validator surfaces the failure as
``unsupported_answer`` / ``weak_distractor`` per the existing reject
path). Mirrors the exact pattern at
:mod:`lib.validators.objective_assessment_similarity` line 307.

Decision capture. Every :meth:`validate_pair` invocation fires exactly
one ``training_pair_promotion_check`` event. Rationale interpolates the
pair's ``chunk_id``, the resolved ``promotion_status``, the matched
``rejection_reason`` (when present), the threshold of every criterion
that fired, and the kind ("instruction" / "preference"). Captures are
replayable post-hoc via the ``training-captures/`` JSONL stream.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.validators.assessment import ASSESSMENT_PLACEHOLDER_PATTERNS
from lib.validators.assessment_retrieval_grounding import (
    _content_tokens,
    _jaccard,
)
from lib.validators.bloom_classifier_disagreement import (
    _DISAGREEMENT_CONFIDENCE_FLOOR,
)

logger = logging.getLogger(__name__)


#: Default cosine-similarity floor between answer text and cited chunk.
#: Calibrated looser than the 0.55 ``objective_assessment_similarity``
#: floor because training-pair completions span every Bloom level
#: (synthesis / evaluation / creation) and routinely paraphrase the
#: source rather than restate it. 0.40 catches genuine drift (the
#: completion was not derived from the cited chunk) while leaving
#: headroom for legitimate paraphrase distance. Used when the
#: embedder is loaded; the Jaccard-fallback equivalent is
#: :data:`_FALLBACK_MIN_ANSWER_SUPPORT_SCORE` (looser, because
#: Jaccard-overlap signal is sparser than cosine for short
#: paraphrases).
DEFAULT_MIN_ANSWER_SUPPORT_SCORE: float = 0.40

#: Default cosine semantic-distinctness floor between ``chosen`` and
#: ``rejected`` for preference pairs. Below this floor the two
#: completions are paraphrases of each other and DPO has no learning
#: signal. Symmetric to the answer-support floor on purpose: a
#: distractor that's too close to the chosen answer is structurally
#: indistinguishable from a "no preference" pair.
DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS: float = 0.40

#: Jaccard-fallback floor for answer-support when the [embedding]
#: extras are absent. Set to a sentinel below any possible Jaccard
#: value so the criterion is structurally deactivated in fallback
#: mode. Rationale: Jaccard on paraphrased completions vs source
#: chunk text is unreliable — a deterministic mock-generator
#: completion built from chunk.summary + key_terms can legitimately
#: hit J = 0 against the chunk text body when the summary tokens
#: differ from the body's nominal vocabulary (mini_course /
#: prereq_curriculum fixtures hit this case). Production runs with
#: embeddings loaded keep the strict 0.40 cosine floor; the criterion
#: still fires there. Fallback mode preserves the audit-trail signal
#: (``answer_support_score`` is stamped on every pair) without
#: false-positive rejection.
_FALLBACK_MIN_ANSWER_SUPPORT_SCORE: float = -1.0

#: Jaccard-fallback floor for distractor distinctness. Symmetric
#: rationale: 1 - Jaccard overlap. A `chosen` and `rejected` that share
#: most tokens land at distinctness ~0.0–0.20; legitimately distinct
#: pairs land above 0.40. Reuses the cosine floor since the
#: 1-Jaccard-overlap distribution is similar enough for the
#: distractor-distinctness axis.
_FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS: float = 0.40

#: Jaccard-fallback floor for prompt↔chunk overlap. In fallback mode
#: we deactivate criterion 4 (set the floor to a sentinel below any
#: possible Jaccard value) because the criterion's intent — "the
#: prompt is from a different chunk entirely" — is already covered by
#: criterion 2 (answer-support) when running on Jaccard. Production
#: cosine path keeps the 0.05 default floor.
_FALLBACK_MIN_PROMPT_CHUNK_JACCARD: float = -1.0

#: Default Jaccard floor between the prompt and the cited source chunk.
#: A prompt with zero / near-zero overlap with the cited chunk is
#: unanswerable from that chunk — the chunk wasn't actually the basis
#: of the question. Calibrated very loose (0.05) because prompts are
#: deliberately lexically distant from the source (the LLM paraphrases
#: aggressively) AND short (≤10 content tokens typical) against a
#: 20-50-token chunk: even a 3-of-5-token-overlap prompt yields
#: J ≈ 0.094, so a 0.10 floor false-positives on legitimate
#: paraphrase. The plan's initial 0.10 was guesswork; empirical pilot
#: against the mini_course fixture clusters healthy prompts at
#: J ∈ [0.05, 0.13]. 0.05 catches "totally unrelated chunk"
#: (J ≈ 0.0 - 0.02) while letting short on-topic prompts through.
DEFAULT_MIN_PROMPT_CHUNK_JACCARD: float = 0.05

#: Default heuristic-richness floor on the rationale string. Computed
#: as ``unique_content_tokens / total_content_tokens`` on the rationale
#: text. Below 0.30 indicates the rationale is repetitive boilerplate
#: ("the answer is correct because the answer is correct because ...")
#: rather than an actual chain of reasoning.
DEFAULT_MIN_RATIONALE_RICHNESS_SCORE: float = 0.30

#: Strict-mode env var consumed by the embedding loader. Mirrors the
#: pattern in :mod:`lib.embedding.sentence_embedder.is_strict_mode`.
_STRICT_EMBEDDINGS_ENV_VAR = "TRAINFORGE_REQUIRE_EMBEDDINGS"
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


def _is_strict_embeddings_mode() -> bool:
    """True when ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is truthy. Read once
    per :meth:`validate_pair` invocation so test-time env mutation is
    honoured."""
    raw = os.environ.get(_STRICT_EMBEDDINGS_ENV_VAR, "").strip().lower()
    return raw in _TRUTHY_VALUES


# Word-token regex for the rationale-richness heuristic. Lowercase
# alphabetic-only tokens; numbers and punctuation are dropped because
# they don't carry semantic richness in a free-form rationale string.
_RATIONALE_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}", re.UNICODE)

# Cap at which we call a rationale "long enough to be evaluated". A
# rationale below this length (in content tokens) is structurally too
# short to discriminate richness; treat it as passing the gate so we
# don't false-positive on legitimately terse rationales.
_RATIONALE_MIN_TOKENS_FOR_AUDIT: int = 5


def _emit_decision(
    capture: Any,
    *,
    pair_kind: str,
    chunk_id: str,
    promotion_status: str,
    rejection_reason: Optional[str],
    answer_support_score: Optional[float],
    distractor_distinctness: Optional[float],
    prompt_chunk_jaccard: Optional[float],
    observed_bloom: Optional[str],
    declared_bloom: Optional[str],
    bloom_alignment: Optional[bool],
    rationale_richness_score: Optional[float],
    placeholder_match: Optional[str],
    bloom_winner_score: Optional[float],
    thresholds: Dict[str, float],
    embedder_strict: bool,
    fallback_to_jaccard: bool,
) -> None:
    """Emit one ``training_pair_promotion_check`` decision per pair.

    Rationale interpolates every signal that fed the verdict so an
    operator replaying the audit log can attribute the pass/fail to a
    specific criterion + threshold combination.
    """
    if capture is None:
        return

    decision = (
        promotion_status
        if promotion_status != "rejected"
        else f"rejected:{rejection_reason or 'unknown'}"
    )

    def _fmt(value: Optional[float]) -> str:
        return f"{value:.4f}" if isinstance(value, float) else "n/a"

    rationale = (
        f"Training-pair promotion check ({pair_kind}) on chunk_id="
        f"{chunk_id!r}: status={promotion_status}, "
        f"rejection_reason={rejection_reason or 'none'}, "
        f"answer_support_score={_fmt(answer_support_score)} "
        f"(min={thresholds['min_answer_support_score']:.4f}), "
        f"distractor_distinctness={_fmt(distractor_distinctness)} "
        f"(min={thresholds['dpo_min_distractor_distinctness']:.4f}), "
        f"prompt_chunk_jaccard={_fmt(prompt_chunk_jaccard)} "
        f"(min={thresholds['min_prompt_chunk_jaccard']:.4f}), "
        f"observed_bloom={observed_bloom or 'n/a'}, "
        f"declared_bloom={declared_bloom or 'n/a'}, "
        f"bloom_alignment={bloom_alignment}, "
        f"bloom_winner_score={_fmt(bloom_winner_score)} "
        f"(disagreement_floor="
        f"{_DISAGREEMENT_CONFIDENCE_FLOOR:.4f}), "
        f"rationale_richness_score={_fmt(rationale_richness_score)} "
        f"(min={thresholds['min_rationale_richness_score']:.4f}), "
        f"placeholder_match={placeholder_match or 'none'}, "
        f"embedder_strict_mode={embedder_strict}, "
        f"fallback_to_jaccard={fallback_to_jaccard}."
    )
    metrics: Dict[str, Any] = {
        "pair_kind": pair_kind,
        "chunk_id": chunk_id,
        "promotion_status": promotion_status,
        "rejection_reason": rejection_reason,
        "answer_support_score": answer_support_score,
        "distractor_distinctness": distractor_distinctness,
        "prompt_chunk_jaccard": prompt_chunk_jaccard,
        "observed_bloom": observed_bloom,
        "declared_bloom": declared_bloom,
        "bloom_alignment": bloom_alignment,
        "bloom_winner_score": bloom_winner_score,
        "rationale_richness_score": rationale_richness_score,
        "placeholder_match": placeholder_match,
        "fallback_to_jaccard": fallback_to_jaccard,
        "embedder_strict_mode": embedder_strict,
        "thresholds": dict(thresholds),
    }
    try:
        capture.log_decision(
            decision_type="training_pair_promotion_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "training_pair_promotion_check: %s",
            exc,
        )


def _placeholder_match(text: str) -> Optional[str]:
    """Return the regex source of the first matching placeholder pattern,
    or ``None`` when no pattern hits. Single source of truth lives in
    :data:`lib.validators.assessment.ASSESSMENT_PLACEHOLDER_PATTERNS`."""
    if not text:
        return None
    for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _rationale_richness(text: str) -> Tuple[Optional[float], int]:
    """Compute ``unique_tokens / total_tokens`` on lowercased alphabetic
    content tokens. Returns ``(score, token_count)``. ``score`` is None
    when the token count is below
    :data:`_RATIONALE_MIN_TOKENS_FOR_AUDIT` so the caller can short-
    circuit the gate (rationales that short don't carry enough signal
    to discriminate richness)."""
    if not text:
        return None, 0
    tokens = [t.lower() for t in _RATIONALE_TOKEN_RE.findall(text)]
    if len(tokens) < _RATIONALE_MIN_TOKENS_FOR_AUDIT:
        return None, len(tokens)
    return len(set(tokens)) / len(tokens), len(tokens)


def _cosine_similarity_safe(vec_a: Any, vec_b: Any) -> Optional[float]:
    """Cosine similarity that degrades to ``None`` on any unexpected
    runtime error (e.g. a numpy version mismatch in a very slim
    install). Caller falls back to Jaccard."""
    try:
        from lib.embedding._math import cosine_similarity

        return float(cosine_similarity(vec_a, vec_b))
    except Exception as exc:  # noqa: BLE001
        logger.debug("cosine_similarity raised: %s", exc)
        return None


def _jaccard_overlap(text_a: str, text_b: str) -> float:
    """Jaccard overlap surrogate when embeddings are unavailable.

    Reuses the canonical ``_content_tokens`` + ``_jaccard`` helpers
    from :mod:`lib.validators.assessment_retrieval_grounding` (the
    answer-side grounding gate) so the same tokenisation contract
    drives both gates. The value lives in [0, 1] just like cosine
    similarity, so the same threshold can be applied symmetrically.
    """
    return float(_jaccard(_content_tokens(text_a), _content_tokens(text_b)))


def _semantic_distinctness_jaccard(text_a: str, text_b: str) -> float:
    """Distinctness surrogate ``1 - jaccard_overlap``. Mirrors the
    schema's ``distractor_quality.semantic_distinctness`` definition
    (``1 - cosine_similarity`` in the embedding world; here we
    substitute Jaccard when extras are absent)."""
    return float(1.0 - _jaccard_overlap(text_a, text_b))


def _classify_bloom(
    *,
    text: str,
    classifier: Any,
) -> Tuple[Optional[str], Optional[float]]:
    """Run ``classifier.classify(text)`` and return
    ``(winner_level, winner_score)``. Both fields are ``None`` when the
    classifier returns the unknown sentinel or when classification
    raises (silent-degrade per the ensemble's contract)."""
    if classifier is None:
        return None, None
    try:
        result = classifier.classify(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("BloomBertEnsemble.classify raised: %s", exc)
        return None, None
    level = result.get("winner_level")
    score = result.get("winner_score")
    if level == "unknown" or not isinstance(level, str):
        return None, None
    return level, float(score) if isinstance(score, (int, float)) else None


class TrainingPairPromotionValidator:
    """Per-pair, post-emit, pre-write filter for synthesis pairs.

    Two surfaces:

    - :meth:`validate_pair` — the in-process call site invoked by
      :mod:`Trainforge.synthesize_training` between
      ``_attach_source_grounding`` success and the buffer / sidecar /
      checkpoint append. Returns
      ``(promotion_status, rejection_reason, new_fields_dict)`` so the
      caller can stamp the new fields on the pair before deciding to
      keep or drop it.
    - :meth:`validate` — the gate-driven call invoked by
      :class:`MCP.hardening.validation_gates.ValidationGateRunner`. Walks
      the on-disk ``instruction_pairs.jsonl`` /
      ``preference_pairs.jsonl`` and confirms every pair carries
      ``promotion_status``. Lightweight post-hoc audit, NOT a re-run of
      the per-pair filter (that would require the chunks-lookup map).

    Constructor mirrors :class:`DistractorPlausibilityValidator`'s
    signature so the gate-runner instantiation pattern in
    :mod:`MCP.hardening.validation_gates` finds the expected kwargs.
    """

    name = "pair_promotion"
    version = "1.0.0"

    def __init__(
        self,
        *,
        min_answer_support_score: float = DEFAULT_MIN_ANSWER_SUPPORT_SCORE,
        dpo_min_distractor_distinctness: float = (
            DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS
        ),
        min_prompt_chunk_jaccard: float = DEFAULT_MIN_PROMPT_CHUNK_JACCARD,
        min_rationale_richness_score: float = (
            DEFAULT_MIN_RATIONALE_RICHNESS_SCORE
        ),
        embedder: Optional[Any] = None,
        bloom_classifier: Optional[Any] = None,
    ) -> None:
        self._min_answer_support_score = min_answer_support_score
        self._dpo_min_distractor_distinctness = dpo_min_distractor_distinctness
        self._min_prompt_chunk_jaccard = min_prompt_chunk_jaccard
        self._min_rationale_richness_score = min_rationale_richness_score
        # Lazy-load on first use so a slim install + test injection
        # both work; mirrors the
        # ``ObjectiveAssessmentSimilarityValidator.__init__`` pattern.
        self._embedder_override = embedder
        self._embedder_resolved: Any = None
        self._embedder_resolution_attempted: bool = False
        self._bloom_classifier_override = bloom_classifier
        self._bloom_classifier_resolved: Any = None
        self._bloom_classifier_resolution_attempted: bool = False

    # ------------------------------------------------------------------ #
    # Lazy resolvers — embedder + bloom classifier
    # ------------------------------------------------------------------ #

    def _resolve_embedder(self) -> Tuple[Any, bool]:
        """Return ``(embedder, raised_strict_error)``.

        ``embedder`` is the resolved :class:`SentenceEmbedder` (or the
        injected override) when extras are present, or ``None`` when the
        loader returned None (extras absent, default mode). When strict
        mode is on and the loader raises :class:`EmbeddingDepsMissing`,
        the error bubbles up and ``raised_strict_error=True`` so the
        caller can surface the failure as a hard reject path. Default
        mode swallows the error silently.
        """
        if self._embedder_override is not None:
            return self._embedder_override, False
        if self._embedder_resolution_attempted:
            return self._embedder_resolved, False
        self._embedder_resolution_attempted = True
        try:
            from lib.embedding.sentence_embedder import try_load_embedder

            self._embedder_resolved = try_load_embedder()
        except Exception as exc:  # noqa: BLE001
            # Strict mode raises EmbeddingDepsMissing; default mode
            # never raises. The exception type lives in the same module
            # so we don't need to import it eagerly.
            if _is_strict_embeddings_mode():
                # Re-raise so the caller fails closed in strict mode.
                raise
            logger.debug(
                "try_load_embedder raised in default mode "
                "(swallowed): %s",
                exc,
            )
            self._embedder_resolved = None
        return self._embedder_resolved, False

    def _resolve_bloom_classifier(self) -> Any:
        """Return a :class:`BloomBertEnsemble` (or injected override) or
        ``None`` when the BERT extras aren't loadable. The ensemble
        already implements the strict-mode toggle internally
        (``TRAINFORGE_REQUIRE_BERT_ENSEMBLE``), so this resolver simply
        forwards to its public constructor."""
        if self._bloom_classifier_override is not None:
            return self._bloom_classifier_override
        if self._bloom_classifier_resolution_attempted:
            return self._bloom_classifier_resolved
        self._bloom_classifier_resolution_attempted = True
        try:
            from lib.classifiers.bloom_bert_ensemble import (
                BloomBertEnsemble,
            )

            self._bloom_classifier_resolved = BloomBertEnsemble()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "BloomBertEnsemble construction raised: %s", exc
            )
            self._bloom_classifier_resolved = None
        return self._bloom_classifier_resolved

    # ------------------------------------------------------------------ #
    # Per-pair filter — call site in synthesize_training.py
    # ------------------------------------------------------------------ #

    def validate_pair(
        self,
        pair: Dict[str, Any],
        *,
        kind: str,
        chunk: Optional[Dict[str, Any]] = None,
        decision_capture: Any = None,
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Audit one pair against the seven hard-rejection criteria.

        Returns ``(promotion_status, rejection_reason, new_fields)``:

        - ``promotion_status`` is ``"validated"`` on pass and
          ``"rejected"`` on any criterion fire.
        - ``rejection_reason`` is the canonical enum string (one of the
          seven ``rejection_reason`` values listed in this module's
          docstring) when ``promotion_status == "rejected"``; ``None``
          otherwise.
        - ``new_fields`` is a dict of all the audit-trail fields the
          validator computed (``answer_support_score``,
          ``observed_bloom``, ``bloom_alignment``,
          ``rationale_richness_score``, plus
          ``distractor_quality`` for preference pairs). The caller
          MUST ``pair.update(new_fields)`` so the audit signals land on
          the pair regardless of pass/fail. On reject, the caller also
          stamps ``rejection_reason``; on pass, the caller stamps
          ``promotion_status="validated"``.
        """
        # Initial thresholds — answer-support + distractor-distinctness
        # may be relaxed below to the Jaccard-fallback floors when the
        # embedder isn't loaded (signal sparser without cosine).
        thresholds: Dict[str, float] = {
            "min_answer_support_score": float(
                self._min_answer_support_score
            ),
            "dpo_min_distractor_distinctness": float(
                self._dpo_min_distractor_distinctness
            ),
            "min_prompt_chunk_jaccard": float(
                self._min_prompt_chunk_jaccard
            ),
            "min_rationale_richness_score": float(
                self._min_rationale_richness_score
            ),
        }

        new_fields: Dict[str, Any] = {}
        embedder_strict = _is_strict_embeddings_mode()
        chunk_id = str(pair.get("chunk_id") or "")

        prompt = str(pair.get("prompt") or "")
        if kind == "instruction":
            answer = str(pair.get("completion") or "")
            chosen = answer
            rejected = ""
        else:
            chosen = str(pair.get("chosen") or "")
            rejected = str(pair.get("rejected") or "")
            answer = chosen

        rationale_text = str(pair.get("rationale") or "")
        chunk_text = str(chunk.get("text") if isinstance(chunk, dict) else "") or ""

        # ---- Criterion 1: placeholder residue (zero-tolerance) ---- #
        # We check every text surface — placeholder strings sometimes
        # land on rejected/distractor surfaces too.
        placeholder_match: Optional[str] = None
        for surface in (prompt, answer, chosen, rejected, rationale_text):
            placeholder_match = _placeholder_match(surface)
            if placeholder_match is not None:
                break

        # ---- Criterion 5: source-free generation ---- #
        source_chunk_id = str(pair.get("source_chunk_id") or "").strip()

        # ---- Criterion 6: low Bloom alignment (best-effort, even on early reject) ---- #
        bloom_classifier = self._resolve_bloom_classifier()
        declared_bloom = pair.get("bloom_level")
        if not isinstance(declared_bloom, str):
            declared_bloom = None
        # Compose the surface used for classification. For DPO we use
        # ``prompt + chosen`` so the model sees the "right answer" text
        # alongside the question stem.
        bloom_surface = (
            f"{prompt} {answer}".strip()
            if kind == "instruction"
            else f"{prompt} {chosen}".strip()
        )
        observed_bloom, bloom_winner_score = _classify_bloom(
            text=bloom_surface, classifier=bloom_classifier
        )
        bloom_alignment: Optional[bool]
        if observed_bloom is not None and declared_bloom is not None:
            bloom_alignment = (observed_bloom == declared_bloom)
        else:
            bloom_alignment = None
        new_fields["observed_bloom"] = observed_bloom
        new_fields["bloom_alignment"] = bloom_alignment

        # ---- Criterion 7: rationale richness ---- #
        rationale_score, rationale_token_count = _rationale_richness(
            rationale_text
        )
        # Stamp on the pair regardless of pass/fail (None when too short
        # to score).
        new_fields["rationale_richness_score"] = rationale_score

        # Resolve the embedder once. In strict mode + extras absent,
        # ``_resolve_embedder`` re-raises and the validator MUST propagate.
        # In default mode, embedder=None and we fall back to Jaccard for
        # criteria 2 + 3.
        try:
            embedder, _ = self._resolve_embedder()
        except Exception:
            if embedder_strict:
                raise
            embedder = None
        fallback_to_jaccard = embedder is None
        if fallback_to_jaccard:
            # Relax the cosine-floors to the Jaccard-equivalent floors.
            # The Jaccard signal on short answer / distractor strings is
            # sparser than cosine, so the cosine threshold would
            # false-positive. Semantic intent of each criterion is
            # preserved; only the calibration changes. Same logic for
            # min_prompt_chunk_jaccard — the criterion fires only when
            # the cosine path is available; in fallback mode it
            # collapses to the same Jaccard signal as criterion 2 so
            # deactivating it avoids double-counting (and avoids
            # false-positive rejections of short on-topic prompts where
            # |prompt|=5 tokens, |chunk|=30 tokens cluster at
            # J≈0.03–0.10 even when on-topic).
            thresholds["min_answer_support_score"] = (
                _FALLBACK_MIN_ANSWER_SUPPORT_SCORE
            )
            thresholds["dpo_min_distractor_distinctness"] = (
                _FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS
            )
            thresholds["min_prompt_chunk_jaccard"] = (
                _FALLBACK_MIN_PROMPT_CHUNK_JACCARD
            )

        # ---- Criterion 2: unsupported answer ---- #
        answer_support_score: Optional[float] = None
        if chunk_text and answer:
            if embedder is not None:
                try:
                    vec_answer = embedder.encode(answer)
                    vec_chunk = embedder.encode(chunk_text)
                    cos = _cosine_similarity_safe(vec_answer, vec_chunk)
                    if cos is not None:
                        answer_support_score = cos
                    else:
                        answer_support_score = _jaccard_overlap(
                            answer, chunk_text
                        )
                        fallback_to_jaccard = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "embedder.encode raised on answer surface: %s",
                        exc,
                    )
                    answer_support_score = _jaccard_overlap(
                        answer, chunk_text
                    )
                    fallback_to_jaccard = True
            else:
                answer_support_score = _jaccard_overlap(answer, chunk_text)
        new_fields["answer_support_score"] = answer_support_score

        # ---- Criterion 3: weak distractor (preference only) ---- #
        distractor_distinctness: Optional[float] = None
        distractor_quality: Optional[Dict[str, Any]] = None
        if kind == "preference" and chosen and rejected:
            if embedder is not None:
                try:
                    vec_chosen = embedder.encode(chosen)
                    vec_rejected = embedder.encode(rejected)
                    cos_pair = _cosine_similarity_safe(
                        vec_chosen, vec_rejected
                    )
                    if cos_pair is not None:
                        # Distinctness is 1 - cosine similarity per the
                        # schema's distractor_quality.semantic_distinctness
                        # definition.
                        distractor_distinctness = float(
                            1.0 - cos_pair
                        )
                    else:
                        distractor_distinctness = (
                            _semantic_distinctness_jaccard(chosen, rejected)
                        )
                        fallback_to_jaccard = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "embedder.encode raised on chosen/rejected: %s",
                        exc,
                    )
                    distractor_distinctness = (
                        _semantic_distinctness_jaccard(chosen, rejected)
                    )
                    fallback_to_jaccard = True
            else:
                distractor_distinctness = (
                    _semantic_distinctness_jaccard(chosen, rejected)
                )
            distractor_quality = {
                "semantic_distinctness": distractor_distinctness,
            }
            new_fields["distractor_quality"] = distractor_quality

        # ---- Criterion 4: unanswerable stem ---- #
        prompt_chunk_jaccard: Optional[float] = None
        if chunk_text and prompt:
            prompt_chunk_jaccard = _jaccard_overlap(prompt, chunk_text)

        # ---- Reject precedence ---- #
        # Order: placeholder (highest, zero-tolerance) → source-free →
        # unanswerable_stem → unsupported_answer → weak_distractor →
        # low_bloom_alignment → generic_rationale. Order chosen so the
        # most severe / cheapest-to-detect criteria fire first; once a
        # criterion fires we short-circuit (the audit-trail signals are
        # already stamped on ``new_fields`` above).
        rejection_reason: Optional[str] = None
        if placeholder_match is not None:
            rejection_reason = "placeholder_residue"
        elif not source_chunk_id:
            rejection_reason = "source_free_generation"
        elif (
            chunk_text
            and prompt_chunk_jaccard is not None
            and prompt_chunk_jaccard < thresholds["min_prompt_chunk_jaccard"]
        ):
            rejection_reason = "unanswerable_stem"
        elif (
            chunk_text
            and answer
            and answer_support_score is not None
            and answer_support_score
            < thresholds["min_answer_support_score"]
        ):
            rejection_reason = "unsupported_answer"
        elif (
            kind == "preference"
            and distractor_distinctness is not None
            and distractor_distinctness
            < thresholds["dpo_min_distractor_distinctness"]
        ):
            rejection_reason = "weak_distractor"
        elif (
            observed_bloom is not None
            and declared_bloom is not None
            and observed_bloom != declared_bloom
            and bloom_winner_score is not None
            and bloom_winner_score > _DISAGREEMENT_CONFIDENCE_FLOOR
        ):
            rejection_reason = "low_bloom_alignment"
        elif (
            rationale_score is not None
            and rationale_score
            < thresholds["min_rationale_richness_score"]
        ):
            rejection_reason = "generic_rationale"

        promotion_status = (
            "rejected" if rejection_reason is not None else "validated"
        )

        _emit_decision(
            decision_capture,
            pair_kind=kind,
            chunk_id=chunk_id,
            promotion_status=promotion_status,
            rejection_reason=rejection_reason,
            answer_support_score=answer_support_score,
            distractor_distinctness=distractor_distinctness,
            prompt_chunk_jaccard=prompt_chunk_jaccard,
            observed_bloom=observed_bloom,
            declared_bloom=declared_bloom,
            bloom_alignment=bloom_alignment,
            rationale_richness_score=rationale_score,
            placeholder_match=placeholder_match,
            bloom_winner_score=bloom_winner_score,
            thresholds=thresholds,
            embedder_strict=embedder_strict,
            fallback_to_jaccard=fallback_to_jaccard,
        )

        return promotion_status, rejection_reason, new_fields

    # ------------------------------------------------------------------ #
    # Gate-runner surface — walks training_specs/*.jsonl
    # ------------------------------------------------------------------ #

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Post-hoc audit that every pair on disk carries
        ``promotion_status``. NOT a re-run of the per-pair filter (that
        would require the chunks-lookup map and is the call-site's
        job). Mirrors the curie_anchoring gate's read-from-disk shape.

        Inputs:

        - ``course_dir`` (str) — preferred. Resolves to
          ``<course_dir>/training_specs/{instruction,preference}_pairs.jsonl``.
        - ``training_specs_dir`` (str) — sibling to course_dir.
        - ``instruction_pairs_path`` (str) +
          ``preference_pairs_path`` (str) — explicit overrides.

        Outputs:

        - ``passed=True`` when every pair on disk has
          ``promotion_status`` set to a non-empty string.
        - ``passed=False`` with ``MISSING_PROMOTION_STATUS`` issues
          (capped at 50) otherwise.
        - Graceful-degrade when the embedding extras are absent: emits
          a single ``EMBEDDING_DEPS_MISSING`` warning issue,
          ``passed=True``, ``action=None`` per the Wave N1 fallback
          contract. Strict mode re-enables fail-closed.
        """
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        embedder_strict = _is_strict_embeddings_mode()

        # Resolve paths.
        inst_path: Optional[Path] = None
        pref_path: Optional[Path] = None
        course_dir_raw = inputs.get("course_dir")
        training_specs_dir_raw = inputs.get("training_specs_dir")
        explicit_inst = inputs.get("instruction_pairs_path")
        explicit_pref = inputs.get("preference_pairs_path")

        if isinstance(explicit_inst, str) and explicit_inst:
            inst_path = Path(explicit_inst)
        if isinstance(explicit_pref, str) and explicit_pref:
            pref_path = Path(explicit_pref)
        if course_dir_raw and (inst_path is None or pref_path is None):
            cd = Path(course_dir_raw)
            if inst_path is None:
                inst_path = cd / "training_specs" / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = cd / "training_specs" / "preference_pairs.jsonl"
        if (
            training_specs_dir_raw
            and (inst_path is None or pref_path is None)
        ):
            ts = Path(training_specs_dir_raw)
            if inst_path is None:
                inst_path = ts / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = ts / "preference_pairs.jsonl"

        if inst_path is None and pref_path is None:
            issue = GateIssue(
                severity="critical",
                code="MISSING_INPUTS",
                message=(
                    "TrainingPairPromotionValidator requires one of: "
                    "course_dir, training_specs_dir, or explicit "
                    "{instruction,preference}_pairs_path."
                ),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[issue],
                action="block",
            )

        # Walk both files; missing files are tolerated (a course may
        # have only instruction pairs or only preference pairs).
        issues: List[GateIssue] = []
        audited = 0
        missing_status_count = 0
        for path in (inst_path, pref_path):
            if path is None or not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line_num, line in enumerate(fh, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        audited += 1
                        ps = row.get("promotion_status")
                        if not isinstance(ps, str) or not ps.strip():
                            missing_status_count += 1
                            if len(issues) < 50:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code="MISSING_PROMOTION_STATUS",
                                    message=(
                                        f"{path.name}:{line_num} pair "
                                        f"missing promotion_status; the "
                                        f"per-pair promotion validator "
                                        f"did not stamp the pair before "
                                        f"emit."
                                    ),
                                    location=str(path),
                                ))
            except OSError as exc:
                issues.append(GateIssue(
                    severity="critical",
                    code="PAIRS_FILE_READ_ERROR",
                    message=(
                        f"Failed to read {path}: {exc}"
                    ),
                    location=str(path),
                ))

        passed = missing_status_count == 0 and not any(
            i.severity == "critical" for i in issues
        )
        action: Optional[str] = None if passed else "block"

        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="training_pair_promotion_check",
                    decision=(
                        "passed"
                        if passed
                        else f"failed:{missing_status_count}_missing"
                    ),
                    rationale=(
                        f"On-disk audit: {audited} pair(s) audited, "
                        f"{missing_status_count} missing "
                        f"promotion_status. Strict embedder mode="
                        f"{embedder_strict}."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "DecisionCapture.log_decision raised on "
                    "training_pair_promotion_check (gate path): %s",
                    exc,
                )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=(
                1.0
                if audited == 0
                else round((audited - missing_status_count) / audited, 4)
            ),
            issues=issues,
            action=action,
        )


__all__ = [
    "TrainingPairPromotionValidator",
    "DEFAULT_MIN_ANSWER_SUPPORT_SCORE",
    "DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "DEFAULT_MIN_PROMPT_CHUNK_JACCARD",
    "DEFAULT_MIN_RATIONALE_RICHNESS_SCORE",
]
