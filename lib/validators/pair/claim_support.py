"""Wave 4 W4.A — :class:`PairClaimSupportValidator`.

Per-pair, per-claim NLI entailment check that fans out every sentence
in a training pair's completion (instruction kind) or chosen surface
(preference kind) against the cited source-chunk text. Sibling of
:class:`lib.validators.claim_support.ClaimSupportValidator` (Wave 2
W2.F, Courseforge block-level) but at a different seam: training-pair
emit, immediately AFTER
:class:`lib.validators.training_pair_promotion.
TrainingPairPromotionValidator.validate_pair` returns ``"validated"``.

Wave 5 W5.D extends the per-pair fan-out: when the cited chunk carries
the new W1.5 structured ``key_claims=[{claim, source_chunk_ids[]}]``
field (admitted by ``schemas/knowledge/chunk_v4.schema.json`` since
W5.C), the validator scopes its NLI premise per-claim — each pair-side
sentence cosine-matches against the closest structured ``claim``
(threshold 0.50) and (when matched) is scored against ONLY the
chunk-text(s) listed in that claim's ``source_chunk_ids[]`` rather
than the whole cited chunk. The match attribution is stamped on each
``per_claim_support[]`` entry as ``source_chunk_ids: List[str] |
None`` so the audit trail records which chunk-ID(s) actually grounded
each pair-side sentence. Back-compat: when ``key_claims`` is absent
or carries the legacy ``List[str]`` shape, OR when no
``chunk_id_to_text_map`` lookup is threaded from the call site, the
validator falls back to whole-chunk-text scoring (byte-identical to
the W4.A behaviour).

Rationale: W2.E (`TrainingPairPromotionValidator`) checks
*answer-support* via cosine similarity between the full answer surface
and the cited chunk. That signal is a single scalar — a training pair
whose completion is mostly grounded but contains one fabricated
sentence still passes W2.E because the cosine average is dominated by
the grounded text. W4.A decomposes the completion into sentences and
runs NLI per-sentence so a single contradicted / unsupported claim can
trip the gate, mirroring the per-claim attribution win Wave 1.5
W1.5.A shipped at the Courseforge layer.

Surface contract (mirrors W2.E
:class:`TrainingPairPromotionValidator`):

1. :meth:`validate_pair` — per-pair pre-write filter, returns
   ``(promotion_status, rejection_reason, new_fields)``. The caller
   ``pair.update(new_fields)`` so the audit signals
   (``per_claim_support`` + ``claim_support_rate`` +
   ``claim_contradicted_rate``) land regardless of pass/fail.
2. :meth:`validate` — gate-runner surface. Walks
   ``training_specs/{instruction,preference}_pairs.jsonl`` post-write
   and asserts every pair carries ``per_claim_support`` +
   ``claim_support_rate``. Same shape as
   :meth:`TrainingPairPromotionValidator.validate` (the audit, not the
   per-pair filter — re-running NLI on disk would require the chunks
   lookup map, which is the call-site's job).

Algorithm (mirrors :meth:`ClaimSupportValidator._validate` lines
348-724):

1. Resolve cited chunk text via ``chunk["text"]``. The caller passes
   ``chunk`` from the call-site's chunk-by-id lookup map. Reject
   ``source_free_generation``-shaped pairs upstream — W2.E catches
   that path via the ``source_chunk_id``-empty criterion.
2. Decompose ``pair["completion"]`` (instruction) or ``pair["chosen"]``
   (preference) into sentences via
   ``re.split(r'(?<=[.!?])\\s+', text)`` (zero-deps, sufficient for
   the typical completion shape). Filter sentences shorter than
   :data:`_MIN_SENTENCE_TOKENS` content tokens — too-short fragments
   are usually structural ("Q:", "A.", " ") rather than pedagogical
   claims and would false-positive under NLI.
3. Resolve NLI classifier via
   :meth:`lib.classifiers.nli_classifier.NliClassifier.get_or_load`.
   Graceful-degrade: when ``None`` (extras absent or load failed),
   stamp ``per_claim_support=None`` + ``claim_support_rate=None`` +
   ``claim_contradicted_rate=None`` + ``deps_missing=True`` on
   ``new_fields`` and return ``("validated", None, ...)``. The W2.E
   audit gate already covers the "pair carries audit fields" check;
   the deps-missing path doesn't reject pairs.
4. For each sentence: ``nli.score_pair(premise=chunk_text,
   hypothesis=sentence)``. Bucket: ``entailed`` if entailment >=
   :data:`_DEFAULT_ENTAILMENT_FLOOR` (0.70); ``contradicted`` if
   contradiction >= :data:`_DEFAULT_CONTRADICTION_FLOOR` (0.50); else
   ``unsupported``.
5. Compute: ``unsupported_claim_rate = (unsupported + contradicted) /
   total``; ``claim_contradicted_rate = contradicted / total``.
6. Reject precedence:
   ``contradicted > _DEFAULT_MAX_CONTRADICTED_RATE`` (0.05) →
   ``rejection_reason="contradicted_claim"``;
   ``unsupported > _DEFAULT_MAX_UNSUPPORTED_RATE`` (0.20) →
   ``rejection_reason="unsupported_claim"``; else accept.
7. Stamp ``per_claim_support`` (list of
   ``{sentence, entailment, contradiction, outcome}`` dicts) +
   ``claim_support_rate`` (entailed-rate) +
   ``claim_contradicted_rate`` on ``new_fields`` regardless of
   pass/fail so the audit trail records every NLI score.

Decision capture: emits exactly one ``pair_claim_support_check``
decision per :meth:`validate_pair` invocation when
``decision_capture`` is provided. Rationale interpolates
``chunk_id``, ``pair_kind``, ``total_claims``, ``entailed_count``,
``unsupported_count``, ``contradicted_count``, ``claim_support_rate``,
``claim_contradicted_rate``, the NLI loaded flag, AND (Wave 5 W5.D)
``structured_attribution_used: bool`` + ``per_claim_match_count: int``
so an operator can see at a glance whether the pair benefited from
the per-claim attribution scoping or fell back to whole-chunk-text
scoring.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier, NliScore

logger = logging.getLogger(__name__)


# W-D7 T7.2: thresholds + GateIssue codes + sentence-split regex
# extracted into the ``_claim_support_thresholds`` sibling module.
# Re-exported here so existing
# ``from lib.validators.pair.claim_support import _DEFAULT_ENTAILMENT_FLOOR``
# (and the back-compat shim's re-export of the same name) keep
# resolving without change.
from lib.validators.pair._claim_support_thresholds import (  # noqa: F401
    _CODE_DART_DISAGREEMENT_RATE_HIGH,
    _CODE_MISSING_CLAIM_SUPPORT_RATE,
    _CODE_MISSING_INPUTS,
    _CODE_MISSING_PER_CLAIM_SUPPORT,
    _CODE_NLI_DEPS_MISSING,
    _CODE_PAIRS_FILE_READ_ERROR,
    _CONTENT_TOKEN_RE,
    _DART_DISAGREEMENT_RATE_WARN_CEILING,
    _DEFAULT_CONTRADICTION_FLOOR,
    _DEFAULT_DART_CONTRADICTION_FLOOR,
    _DEFAULT_ENTAILMENT_FLOOR,
    _DEFAULT_MAX_CONTRADICTED_RATE,
    _DEFAULT_MAX_UNSUPPORTED_RATE,
    _GATE_ISSUE_CAP,
    _ISSUE_LIST_CAP,
    _MIN_SENTENCE_TOKENS,
    _OUTCOME_DART_DISAGREEMENT,
    _PER_CLAIM_MATCH_COSINE_FLOOR,
    _REASON_CONTRADICTED_CLAIM,
    _REASON_UNSUPPORTED_CLAIM,
    _SENTENCE_SPLIT_RE,
)


def _content_token_count(text: str) -> int:
    """Count content tokens in ``text``. Used to filter too-short
    sentences before NLI fan-out."""
    if not text:
        return 0
    return len(_CONTENT_TOKEN_RE.findall(text))


def _decompose_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences via terminal-punctuation lookbehind.

    Returns a list of sentence strings (whitespace-stripped, empty
    strings filtered). Sentences shorter than
    :data:`_MIN_SENTENCE_TOKENS` content tokens are filtered out — too
    short to score meaningfully under NLI.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    raw_split = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences: List[str] = []
    for fragment in raw_split:
        candidate = fragment.strip()
        if not candidate:
            continue
        if _content_token_count(candidate) < _MIN_SENTENCE_TOKENS:
            continue
        sentences.append(candidate)
    return sentences


# W-D7 T7.2: _PER_CLAIM_MATCH_COSINE_FLOOR + Wave 9 dual-source
# constants moved to ``_claim_support_thresholds`` sibling module;
# imported at the top of this file.


def _resolve_chunk_key_claims_with_attribution(
    chunk: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Pull ``key_claims`` off a chunk dict when it carries the
    Wave 1.5 / W1.5.A structured shape.

    Mirrors :func:`lib.validators.claim_support._resolve_key_claims`
    (the Courseforge block-level template) but adapts the surface:
    here the input is a *chunk* dict (as published in
    ``schemas/knowledge/chunk_v4.schema.json``) rather than a
    ``Block`` dataclass.

    Returns ``(per_claim_entries, structured_flag)``:

    - ``per_claim_entries``: ordered list of normalised
      ``{"claim": str, "source_chunk_ids": List[str]}`` entries (only
      those carrying both keys; defensive against malformed entries).
      Empty list when ``key_claims`` is absent, malformed, the legacy
      ``List[str]`` shape, or carries no usable structured entries.
    - ``structured_flag``: ``True`` when at least one entry carried
      ``source_chunk_ids[]`` with at least one usable string ID. When
      ``False``, the caller MUST fall back to whole-chunk-text NLI
      scoring (byte-identical to the W4.A pre-W5.D behaviour).

    Defensive contract: drops non-dict entries, drops dicts missing
    ``claim``, drops empty ``source_chunk_ids[]`` (the legacy
    ``List[str]`` arm of the schema oneOf intentionally produces
    no structured entries here — those entries' attribution lives at
    the chunk level, not per-claim, so per-claim scoping is a no-op
    on legacy corpora).
    """
    if not isinstance(chunk, dict):
        return [], False
    raw = chunk.get("key_claims")
    if not isinstance(raw, list):
        return [], False

    per_claim_entries: List[Dict[str, Any]] = []
    structured_flag = False
    for entry in raw:
        if not isinstance(entry, dict):
            # Drop bare strings (legacy List[str] arm) and any other
            # malformed entries — the caller's fallback path covers
            # the legacy corpus correctly.
            continue
        claim_raw = entry.get("claim")
        if not isinstance(claim_raw, str) or not claim_raw.strip():
            continue
        attribution = entry.get("source_chunk_ids")
        if isinstance(attribution, list):
            ids = [
                sid for sid in attribution
                if isinstance(sid, str) and sid
            ]
        else:
            ids = []
        if not ids:
            # No usable structured attribution on this entry — skip
            # so the caller doesn't fall back to a degenerate empty
            # candidate-chunks list (which would score 0 entailment
            # for every sentence and false-positive the unsupported
            # rate). Per-claim scoping requires at least one named
            # source chunk.
            continue
        per_claim_entries.append({
            "claim": claim_raw.strip(),
            "source_chunk_ids": ids,
        })
        structured_flag = True
    return per_claim_entries, structured_flag


def _try_load_embedder_safe() -> Optional[Any]:
    """Wrap :func:`lib.embedding.sentence_embedder.try_load_embedder`
    in a try/except so that a missing extras / unexpected import error
    falls through to whole-chunk-text scoring rather than blowing up
    the per-pair filter. Strict-mode opt-in via
    :func:`lib.embedding.sentence_embedder.is_strict_mode` re-raises;
    the per-pair filter intentionally swallows that re-raise too,
    because ``deps_missing`` is already a graceful-degrade arm at this
    layer (see W4.A docstring §3 on the strict-mode rationale).
    """
    try:
        from lib.embedding.sentence_embedder import try_load_embedder

        return try_load_embedder()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "try_load_embedder raised in pair_claim_support per-claim "
            "attribution path (swallowed, falls back to whole-chunk-"
            "text scoring): %s", exc,
        )
        return None


def _cosine_safe(vec_a: Any, vec_b: Any) -> Optional[float]:
    """Cosine similarity that degrades to ``None`` on any error.
    Mirrors ``training_pair_promotion._cosine_similarity_safe``."""
    try:
        from lib.embedding._math import cosine_similarity

        return float(cosine_similarity(vec_a, vec_b))
    except Exception as exc:  # noqa: BLE001
        logger.debug("cosine_similarity raised in pair_claim_support: %s", exc)
        return None


def _match_sentence_to_claim(
    *,
    sentence: str,
    sentence_vec: Any,
    claim_vecs: List[Any],
    per_claim_entries: List[Dict[str, Any]],
    floor: float,
) -> Optional[int]:
    """Find the closest-matching ``per_claim_entries`` index by cosine
    similarity. Returns ``None`` when no entry clears ``floor`` or
    when an embedding error degrades the comparison.
    """
    if not per_claim_entries or not claim_vecs:
        return None
    best_idx: Optional[int] = None
    best_score: float = -1.0
    for idx, claim_vec in enumerate(claim_vecs):
        if claim_vec is None:
            continue
        score = _cosine_safe(sentence_vec, claim_vec)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None or best_score < floor:
        return None
    return best_idx


def _resolve_dart_block_texts_for_chunk(
    *,
    chunk: Optional[Dict[str, Any]],
    dart_block_text_map: Dict[str, str],
) -> List[str]:
    """Wave 9 TIGHT — resolve the union of DART block texts cited by a
    chunk via its ``source.source_references[].sourceId`` list.

    Returns the deduped list of DART block texts (preserves first-seen
    order). Empty list when:

    - ``chunk`` is None or non-dict.
    - ``chunk["source"]["source_references"]`` is missing / empty / not
      a list (legacy / non-Courseforge corpora — the dual-source check
      no-ops cleanly).
    - None of the resolved ``sourceId`` values are present in
      ``dart_block_text_map`` (chunker-drift slice on a chunk whose
      source IDs renamed post-DART; the caller's audit field will read
      "DART check did not fire" rather than "disagreement").

    The multi-block union path is the resolution called out in §2 of
    the dual-source plan: when ``merge_small_sections`` collapses
    multiple DART blocks into one IMSCC chunk, the DART premise pool
    is the union of ALL named blocks, so a sentence backed by ANY one
    of them is entailed (mirrors the W2.F max-over-candidates
    aggregation at the IMSCC layer).
    """
    if not isinstance(chunk, dict):
        return []
    source = chunk.get("source")
    if not isinstance(source, dict):
        return []
    refs = source.get("source_references")
    if not isinstance(refs, list) or not refs:
        return []
    seen: Dict[str, None] = {}
    texts: List[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        source_id = ref.get("sourceId")
        if not isinstance(source_id, str) or not source_id:
            continue
        if source_id in seen:
            continue
        seen[source_id] = None
        block_text = dart_block_text_map.get(source_id)
        if isinstance(block_text, str) and block_text.strip():
            texts.append(block_text)
    return texts


def _emit_decision(
    capture: Any,
    *,
    pair_kind: str,
    chunk_id: str,
    promotion_status: str,
    rejection_reason: Optional[str],
    total_claims: int,
    entailed_count: int,
    unsupported_count: int,
    contradicted_count: int,
    claim_support_rate: Optional[float],
    claim_contradicted_rate: Optional[float],
    nli_loaded: bool,
    deps_missing: bool,
    structured_attribution_used: bool = False,
    per_claim_match_count: int = 0,
    dart_check_fired_count: int = 0,
    dart_disagreement_count: int = 0,
) -> None:
    """Emit one ``pair_claim_support_check`` decision per
    :meth:`PairClaimSupportValidator.validate_pair` invocation.

    Rationale interpolates the dynamic signals enumerated in the W4.A
    brief: ``chunk_id``, ``pair_kind``, ``total_claims``,
    ``entailed_count``, ``unsupported_count``, ``contradicted_count``,
    ``claim_support_rate``, ``claim_contradicted_rate``, and the NLI
    loaded flag. A static rationale would violate the
    ``DECISION_VALIDATION_STRICT`` contract — every signal must be
    dynamic per call.
    """
    if capture is None:
        return
    decision = (
        f"rejected:{rejection_reason}"
        if rejection_reason is not None
        else promotion_status
    )
    rate_str = (
        f"{claim_support_rate:.4f}"
        if claim_support_rate is not None
        else "n/a"
    )
    cont_rate_str = (
        f"{claim_contradicted_rate:.4f}"
        if claim_contradicted_rate is not None
        else "n/a"
    )
    rationale = (
        f"Per-claim NLI support check on {pair_kind!r} pair "
        f"chunk_id={chunk_id!r}: total_claims={total_claims}, "
        f"entailed_count={entailed_count}, "
        f"unsupported_count={unsupported_count}, "
        f"contradicted_count={contradicted_count}, "
        f"claim_support_rate={rate_str}, "
        f"claim_contradicted_rate={cont_rate_str}, "
        f"nli_loaded={nli_loaded}, deps_missing={deps_missing}, "
        f"structured_attribution_used={structured_attribution_used}, "
        f"per_claim_match_count={per_claim_match_count}, "
        f"dart_check_fired_count={dart_check_fired_count}, "
        f"dart_disagreement_count={dart_disagreement_count}, "
        f"promotion_status={promotion_status}, "
        f"rejection_reason={rejection_reason or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="pair_claim_support_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "pair_claim_support_check: %s",
            exc,
        )


class PairClaimSupportValidator:
    """Per-pair-per-claim NLI entailment gate.

    Two surfaces (mirrors
    :class:`lib.validators.training_pair_promotion.
    TrainingPairPromotionValidator`):

    1. :meth:`validate_pair` — per-pair pre-write filter called from
       :mod:`Trainforge.synthesize_training` immediately AFTER
       :meth:`TrainingPairPromotionValidator.validate_pair` returns
       ``"validated"``. Returns
       ``(promotion_status, rejection_reason, new_fields)``.
    2. :meth:`validate` — gate-runner surface, walks
       ``training_specs/{instruction,preference}_pairs.jsonl``
       post-write, asserts every pair carries ``per_claim_support``
       + ``claim_support_rate``.

    Constructor:

    Args:
        max_unsupported_claim_rate: Per-pair ceiling on the
            ``(unsupported + contradicted) / total`` rate. Above this
            rate the pair is rejected with
            ``rejection_reason="unsupported_claim"``. Default 0.20.
        max_contradicted_claim_rate: Per-pair ceiling on the
            ``contradicted / total`` rate. Above this rate the pair is
            rejected with ``rejection_reason="contradicted_claim"``
            regardless of total unsupported rate. Default 0.05.
        entailment_floor: Per-sentence entailment-score floor for
            "entailed" bucket. Default 0.70.
        contradiction_floor: Per-sentence contradiction-score floor
            for "contradicted" bucket. Default 0.50.
        nli: Optional pre-loaded NLI classifier. Test-injection seam
            and singleton-cache override. When ``None``, calls
            :meth:`NliClassifier.get_or_load` lazily.

    Graceful-degrade: when :meth:`NliClassifier.get_or_load` returns
    ``None`` (extras absent or load failed), :meth:`validate_pair`
    returns ``("validated", None, {"per_claim_support": None,
    "claim_support_rate": None, "claim_contradicted_rate": None,
    "deps_missing": True})`` so the audit trail records the
    silent-degrade path. Strict mode is intentionally NOT wired here
    — the W2.E audit gate already covers the "pair carries audit
    fields" check; surfacing a deps-missing fail at the per-pair
    layer would block every CPU-only synthesis run.
    """

    name = "pair_claim_support"
    version = "1.0.0"

    def __init__(
        self,
        *,
        max_unsupported_claim_rate: float = _DEFAULT_MAX_UNSUPPORTED_RATE,
        max_contradicted_claim_rate: float = _DEFAULT_MAX_CONTRADICTED_RATE,
        entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
        contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
        nli: Optional[Any] = None,
    ) -> None:
        self._max_unsupported = float(max_unsupported_claim_rate)
        self._max_contradicted = float(max_contradicted_claim_rate)
        self._entailment_floor = float(entailment_floor)
        self._contradiction_floor = float(contradiction_floor)
        self._nli_override = nli

    def _get_nli(self) -> Optional[Any]:
        """Resolve the NLI classifier — explicit override wins, else
        the singleton-cached :meth:`NliClassifier.get_or_load`."""
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    # ------------------------------------------------------------------ #
    # Per-pair filter — call site in synthesize_training.py
    # ------------------------------------------------------------------ #

    def validate_pair(
        self,
        pair: Dict[str, Any],
        *,
        kind: str,
        chunk: Optional[Dict[str, Any]] = None,
        chunk_id_to_text_map: Optional[Dict[str, str]] = None,
        decision_capture: Any = None,
        dart_block_text_map: Optional[Dict[str, str]] = None,
        dual_source_severity: Literal["off", "warning"] = "off",
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Audit one pair against the per-claim NLI support criteria.

        Args:
            pair: The training pair dict. Reads ``completion`` (when
                ``kind="instruction"``) or ``chosen`` (when
                ``kind="preference"``). Also reads ``chunk_id`` for
                the decision-capture rationale.
            kind: ``"instruction"`` or ``"preference"``. Selects which
                surface to decompose into claims.
            chunk: The cited source chunk dict. Reads ``chunk["text"]``
                as the NLI premise. When ``None`` (e.g. caller couldn't
                resolve the chunk-by-id lookup), the validator
                short-circuits to a graceful-degrade pass — W2.E
                catches ``source_free_generation`` upstream, so a
                ``None`` chunk at this layer is the call-site's bug,
                not a content-quality failure.
            chunk_id_to_text_map: Optional mapping ``chunk_id -> text``
                used by the Wave 5 W5.D per-claim attribution scoping
                path. When the cited ``chunk`` carries a structured
                ``key_claims=[{claim, source_chunk_ids[]}]`` field
                (W1.5 / W5.A) AND this map is wired in, every pair-side
                sentence cosine-matches against the structured claim
                texts (threshold 0.50) and (when matched) is NLI-scored
                against ONLY the chunks listed in that claim's
                ``source_chunk_ids[]`` rather than the whole cited
                chunk. When the map is ``None``, the validator
                graceful-degrades to whole-chunk-text scoring even
                when ``key_claims`` is structured — so the validator
                works without this lookup being threaded through every
                call site (back-compat day-1).
            decision_capture: Optional :class:`DecisionCapture`
                instance. When wired, one
                ``pair_claim_support_check`` event per call.
            dart_block_text_map: Optional Wave 9 TIGHT dual-source
                cross-check map keyed by ``dart:<slug>#<block_id>``.
                When non-empty AND ``dual_source_severity != "off"``,
                every IMSCC-NLI ``"entailed"`` outcome triggers a
                second NLI pass against the union of DART block texts
                resolved from the cited IMSCC chunk's
                ``source.source_references[].sourceId``. The DART pass
                stamps ``per_claim_support[].dart_source_check =
                {entailment, contradiction, outcome}`` on the audit
                field, and bumps the per-claim ``outcome`` from
                ``"entailed"`` to ``"dart_disagreement"`` when DART
                contradicts at >=
                :data:`_DEFAULT_DART_CONTRADICTION_FLOOR`. **Day-1
                contract**: warning-severity only — does NOT reject the
                pair, no ``(status, reason)`` tuple change for
                ``dart_disagreement``. The gate-runner walk surfaces
                the aggregate :data:`_CODE_DART_DISAGREEMENT_RATE_HIGH`
                warning when > 5% of pairs carry the new outcome.
                Closes the chunker-drift slice (Finding 5 from
                ``plans/dual-source-investigation-2026-05.md``) — the
                W2.F seam fires before IMSCC chunking, so it can't see
                ``merge_small_sections`` / sentence-split sub-chunking
                drift introduced post-Courseforge.
            dual_source_severity: Wave 9 TIGHT severity dial. Values:
                ``"off"`` (default — legacy corpora bypass, no DART
                cross-check fires) or ``"warning"`` (audit-stamp only;
                no per-pair reject). A future calibration wave may
                introduce ``"critical"`` once the DART-side
                contradiction floor is calibrated against the
                rdf-shacl-551-2 corpus per §3 of the dual-source
                investigation plan.

        Returns:
            ``(promotion_status, rejection_reason, new_fields)``:

            - ``promotion_status``: ``"validated"`` on pass,
              ``"rejected"`` on contradicted/unsupported rate over
              ceiling.
            - ``rejection_reason``: ``"contradicted_claim"`` |
              ``"unsupported_claim"`` | ``None``.
            - ``new_fields``: ``{"per_claim_support": [{"sentence":
              str, "entailment": float, "contradiction": float,
              "outcome": "entailed" | "unsupported" | "contradicted",
              "source_chunk_ids": List[str] | None}],
              "claim_support_rate": float | None,
              "claim_contradicted_rate": float | None,
              "deps_missing": bool}``. Stamped on the pair regardless
              of pass/fail; the caller MUST ``pair.update(new_fields)``
              so the audit signals land on disk. ``source_chunk_ids``
              is populated from the matched W1.5 / W5.A entry when
              per-claim attribution scoping fired; ``None`` when the
              sentence didn't match any structured claim (or when
              ``chunk_id_to_text_map`` wasn't threaded so the
              validator graceful-degraded to whole-chunk-text
              scoring).
        """
        chunk_id = str(pair.get("chunk_id") or "")
        nli = self._get_nli()
        nli_loaded = nli is not None

        # ---- Resolve premise text ---- #
        chunk_text = ""
        if isinstance(chunk, dict):
            chunk_text = str(chunk.get("text") or "")

        # ---- Wave 5 W5.D: resolve structured key_claims ---- #
        # When the cited chunk carries the W1.5 / W5.A structured
        # ``key_claims=[{claim, source_chunk_ids[]}]`` field AND the
        # caller threaded a chunk_id_to_text_map lookup, every pair-
        # side sentence is routed through the per-claim attribution
        # scoping path: cosine-match sentence ↔ claim_text, then NLI
        # against ONLY the chunks named in the matched claim's
        # ``source_chunk_ids[]`` list. When either condition fails,
        # ``structured_attribution_used`` stays False and the legacy
        # whole-chunk-text scoring path runs (byte-identical to the
        # W4.A pre-W5.D behaviour).
        per_claim_entries, key_claims_structured = (
            _resolve_chunk_key_claims_with_attribution(chunk)
        )
        structured_attribution_used = (
            key_claims_structured
            and isinstance(chunk_id_to_text_map, dict)
            and bool(chunk_id_to_text_map)
        )

        # ---- Resolve hypothesis surface ---- #
        if kind == "instruction":
            text = str(pair.get("completion") or "")
        elif kind == "preference":
            text = str(pair.get("chosen") or "")
        else:
            # Unknown kind — best-effort fall-through to "chosen", but
            # don't reject; the caller is responsible for the kind
            # contract.
            text = str(pair.get("chosen") or pair.get("completion") or "")

        # ---- Graceful-degrade arms ---- #
        # Arm 1: NLI deps missing. Stamp deps_missing=True and pass.
        # Arm 2: chunk text unresolvable. Same shape as arm 1 — W2.E
        #        catches source_free upstream, this is defense in depth.
        if not nli_loaded or not chunk_text:
            new_fields: Dict[str, Any] = {
                "per_claim_support": None,
                "claim_support_rate": None,
                "claim_contradicted_rate": None,
                "deps_missing": True,
            }
            _emit_decision(
                decision_capture,
                pair_kind=kind,
                chunk_id=chunk_id,
                promotion_status="validated",
                rejection_reason=None,
                total_claims=0,
                entailed_count=0,
                unsupported_count=0,
                contradicted_count=0,
                claim_support_rate=None,
                claim_contradicted_rate=None,
                nli_loaded=nli_loaded,
                deps_missing=True,
                structured_attribution_used=False,
                per_claim_match_count=0,
            )
            return "validated", None, new_fields

        # ---- Decompose hypothesis into sentence-claims ---- #
        sentences = _decompose_sentences(text)
        if not sentences:
            # Empty / too-short hypothesis — nothing to score. Stamp
            # zeroed audit fields so the pair still carries a
            # per_claim_support array on disk (empty list, not None,
            # so the W2.E audit gate's "carries field" check stays
            # green).
            new_fields = {
                "per_claim_support": [],
                "claim_support_rate": None,
                "claim_contradicted_rate": None,
                "deps_missing": False,
            }
            _emit_decision(
                decision_capture,
                pair_kind=kind,
                chunk_id=chunk_id,
                promotion_status="validated",
                rejection_reason=None,
                total_claims=0,
                entailed_count=0,
                unsupported_count=0,
                contradicted_count=0,
                claim_support_rate=None,
                claim_contradicted_rate=None,
                nli_loaded=nli_loaded,
                deps_missing=False,
                structured_attribution_used=False,
                per_claim_match_count=0,
            )
            return "validated", None, new_fields

        # ---- Wave 5 W5.D: pre-compute per-sentence attribution ---- #
        # When structured_attribution_used is on, embed every
        # sentence + every claim_text once, then route each sentence's
        # NLI premise to the chunks named in its closest-matching
        # claim's source_chunk_ids[]. When off, every premise stays
        # anchored on the whole chunk text (legacy W4.A path).
        sentence_attributions: List[Optional[List[str]]] = [
            None for _ in sentences
        ]
        per_claim_match_count = 0
        # Per-sentence list of premise texts. Default: one premise =
        # whole chunk text. Per-claim path replaces this with the
        # union of cited chunk texts when a sentence-claim match
        # fires.
        per_sentence_premises: List[List[str]] = [
            [chunk_text] for _ in sentences
        ]
        if structured_attribution_used:
            embedder = _try_load_embedder_safe()
            if embedder is not None:
                try:
                    sentence_vecs = [
                        embedder.encode(s) for s in sentences
                    ]
                    claim_vecs = [
                        embedder.encode(entry["claim"])
                        for entry in per_claim_entries
                    ]
                except Exception as exc:  # noqa: BLE001
                    # Embedder failed mid-encode. Fall back to whole-
                    # chunk-text scoring so the run doesn't break on
                    # a transient model error. Keep
                    # ``structured_attribution_used`` True for the
                    # decision-capture rationale (so the audit trail
                    # records the intent) but flag the per-claim
                    # path as effectively off.
                    logger.warning(
                        "Embedder failed on per-claim attribution "
                        "encode (chunk_id=%r, kind=%r): %s — falling "
                        "back to whole-chunk-text NLI scoring.",
                        chunk_id, kind, exc,
                    )
                    sentence_vecs = []
                    claim_vecs = []
                # Map each sentence to a per-claim entry, then build
                # the candidate-premise list from the entry's named
                # chunks. Sentences with no match keep the default
                # whole-chunk-text premise.
                for s_idx, sent_vec in enumerate(sentence_vecs):
                    matched_idx = _match_sentence_to_claim(
                        sentence=sentences[s_idx],
                        sentence_vec=sent_vec,
                        claim_vecs=claim_vecs,
                        per_claim_entries=per_claim_entries,
                        floor=_PER_CLAIM_MATCH_COSINE_FLOOR,
                    )
                    if matched_idx is None:
                        continue
                    matched_entry = per_claim_entries[matched_idx]
                    matched_ids = matched_entry["source_chunk_ids"]
                    candidate_texts: List[str] = []
                    for sid in matched_ids:
                        cand_text = (
                            chunk_id_to_text_map.get(sid)
                            if isinstance(chunk_id_to_text_map, dict)
                            else None
                        )
                        if isinstance(cand_text, str) and cand_text.strip():
                            candidate_texts.append(cand_text)
                    if not candidate_texts:
                        # Matched entry but none of its named chunks
                        # are in the lookup map — fall back to whole-
                        # chunk-text for this sentence (don't drop
                        # the sentence; the per-claim path is best-
                        # effort).
                        continue
                    per_sentence_premises[s_idx] = candidate_texts
                    sentence_attributions[s_idx] = list(matched_ids)
                    per_claim_match_count += 1
            else:
                # Embedder couldn't load (default-mode degrade) — fall
                # back to whole-chunk-text scoring. Keep the decision
                # capture flag aligned with the actual scoring path.
                structured_attribution_used = False

        # ---- Fan-out NLI scoring ---- #
        # Build (premise, hypothesis) pairs. When per-claim attribution
        # fired for a sentence, the sentence appears once per cited
        # chunk text (so a max-over-cited-chunks aggregation can
        # decide its outcome). When the legacy whole-chunk-text path
        # is on, exactly one (chunk_text, sentence) pair per sentence.
        nli_pairs: List[Tuple[str, str]] = []
        per_sentence_pair_index_ranges: List[Tuple[int, int]] = []
        for s_idx, sentence in enumerate(sentences):
            premises = per_sentence_premises[s_idx]
            start = len(nli_pairs)
            for premise in premises:
                nli_pairs.append((premise, sentence))
            end = len(nli_pairs)
            per_sentence_pair_index_ranges.append((start, end))
        try:
            scores: List[NliScore] = nli.score_batch(pairs=nli_pairs)
        except Exception as exc:  # noqa: BLE001
            # Defensive: NLI call raised mid-fan-out. Fall back to the
            # graceful-degrade path so we don't block the synthesis
            # run on a transient model error.
            logger.warning(
                "NliClassifier.score_batch raised on pair_claim_support "
                "fan-out (chunk_id=%r, kind=%r): %s",
                chunk_id, kind, exc,
            )
            new_fields = {
                "per_claim_support": None,
                "claim_support_rate": None,
                "claim_contradicted_rate": None,
                "deps_missing": True,
            }
            _emit_decision(
                decision_capture,
                pair_kind=kind,
                chunk_id=chunk_id,
                promotion_status="validated",
                rejection_reason=None,
                total_claims=0,
                entailed_count=0,
                unsupported_count=0,
                contradicted_count=0,
                claim_support_rate=None,
                claim_contradicted_rate=None,
                nli_loaded=nli_loaded,
                deps_missing=True,
                structured_attribution_used=structured_attribution_used,
                per_claim_match_count=per_claim_match_count,
            )
            return "validated", None, new_fields

        # ---- Bucket each sentence ---- #
        # When per-claim attribution fan-out emitted multiple
        # (premise, sentence) pairs for one sentence (one per cited
        # chunk text), aggregate via max-entailment / max-
        # contradiction so a sentence backed by ANY cited chunk is
        # entailed; a sentence contradicted by ANY cited chunk is
        # contradicted. Mirrors the W2.F precedent at
        # ``lib/validators/claim_support.py`` § "rules-1.5: max-over-
        # candidate-chunks".
        per_claim_support: List[Dict[str, Any]] = []
        entailed_count = 0
        unsupported_count = 0
        contradicted_count = 0

        for s_idx, sentence in enumerate(sentences):
            start, end = per_sentence_pair_index_ranges[s_idx]
            sentence_scores = scores[start:end]
            if not sentence_scores:
                # Defensive: shouldn't happen because per_sentence_premises
                # always has at least one entry. If it does, treat the
                # sentence as unsupported.
                entailment = 0.0
                contradiction = 0.0
            else:
                entailment = max(
                    float(s.entailment) for s in sentence_scores
                )
                contradiction = max(
                    float(s.contradiction) for s in sentence_scores
                )
            if entailment >= self._entailment_floor:
                outcome = "entailed"
                entailed_count += 1
            elif contradiction >= self._contradiction_floor:
                outcome = "contradicted"
                contradicted_count += 1
            else:
                outcome = "unsupported"
                unsupported_count += 1
            per_claim_support.append({
                "sentence": sentence,
                "entailment": entailment,
                "contradiction": contradiction,
                "outcome": outcome,
                "source_chunk_ids": sentence_attributions[s_idx],
            })

        # ---- Wave 9 TIGHT: dual-source DART cross-check ---- #
        # AFTER the IMSCC NLI fan-out scores a sentence as ``entailed``,
        # AND ``dual_source_severity != "off"``, AND
        # ``dart_block_text_map`` is non-empty: re-score each entailed
        # sentence against the union of DART block text(s) resolved
        # from the cited IMSCC chunk's
        # ``source.source_references[].sourceId``. Stamp the result on
        # ``per_claim_support[].dart_source_check`` and bump the
        # per-claim ``outcome`` from ``"entailed"`` to
        # ``"dart_disagreement"`` when DART contradicts at >=
        # :data:`_DEFAULT_DART_CONTRADICTION_FLOOR`.
        #
        # **Worker A precedence (W8)**: deterministic-template pairs
        # carry ``pair_lo_resolution.skipped == "deterministic_template"``
        # and MUST NOT trigger the dual-source check — those pairs
        # don't go through the LLM paraphrase path so a DART /
        # IMSCC drift signal is meaningless on them. Skip cleanly so
        # the W8 audit stamp is the only resolution stamped on the
        # pair.
        #
        # **Reject contract**: warning-severity only. The audit field
        # is stamped, the per-claim ``outcome`` is bumped, but no
        # ``(status, reason)`` tuple change. The aggregate-rate
        # warning fires at the gate-runner walk via
        # :data:`_CODE_DART_DISAGREEMENT_RATE_HIGH`.
        dart_check_fired_count = 0
        dart_disagreement_count = 0
        _is_deterministic_template = False
        _plr = pair.get("pair_lo_resolution")
        if (
            isinstance(_plr, dict)
            and _plr.get("skipped") == "deterministic_template"
        ):
            _is_deterministic_template = True
        if (
            dual_source_severity != "off"
            and dart_block_text_map
            and not _is_deterministic_template
        ):
            dart_premise_texts = _resolve_dart_block_texts_for_chunk(
                chunk=chunk,
                dart_block_text_map=dart_block_text_map,
            )
            if dart_premise_texts:
                # Build (premise, hypothesis) pairs for every
                # currently-``entailed`` sentence × every resolved DART
                # block text. Aggregation mirrors the IMSCC fan-out
                # above: max-entailment / max-contradiction so a
                # sentence backed by ANY DART block text is entailed,
                # contradicted by ANY DART block text is contradicted.
                dart_nli_pairs: List[Tuple[str, str]] = []
                dart_pair_index_ranges: List[Optional[Tuple[int, int]]] = (
                    []
                )
                for entry in per_claim_support:
                    if entry["outcome"] != "entailed":
                        dart_pair_index_ranges.append(None)
                        continue
                    start = len(dart_nli_pairs)
                    for premise in dart_premise_texts:
                        dart_nli_pairs.append((premise, entry["sentence"]))
                    end = len(dart_nli_pairs)
                    dart_pair_index_ranges.append((start, end))
                if dart_nli_pairs:
                    try:
                        dart_scores: List[NliScore] = nli.score_batch(
                            pairs=dart_nli_pairs,
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Defensive: a transient NLI error on the DART
                        # pass MUST NOT poison the IMSCC verdict.
                        # Stamp nothing and continue — the audit field
                        # remains absent on every per_claim entry,
                        # which the gate-runner walk reads as "DART
                        # check did not fire" rather than
                        # "disagreement".
                        logger.warning(
                            "Wave 9 TIGHT: NliClassifier.score_batch "
                            "raised on dual-source DART cross-check "
                            "(chunk_id=%r, kind=%r): %s — DART check "
                            "skipped for this pair.",
                            chunk_id, kind, exc,
                        )
                        dart_scores = []
                    if dart_scores:
                        for s_idx, idx_range in enumerate(
                            dart_pair_index_ranges
                        ):
                            if idx_range is None:
                                continue
                            d_start, d_end = idx_range
                            sub = dart_scores[d_start:d_end]
                            if not sub:
                                continue
                            d_entailment = max(
                                float(s.entailment) for s in sub
                            )
                            d_contradiction = max(
                                float(s.contradiction) for s in sub
                            )
                            if (
                                d_contradiction
                                >= _DEFAULT_DART_CONTRADICTION_FLOOR
                            ):
                                d_outcome = _OUTCOME_DART_DISAGREEMENT
                                # Bump the per-claim outcome on the
                                # IMSCC audit field. The IMSCC entailed
                                # bucket count is NOT decremented —
                                # ``claim_support_rate`` retains its
                                # IMSCC-anchored meaning ("how much of
                                # this pair did the IMSCC chunk
                                # entail?"); the DART disagreement is
                                # surfaced separately via
                                # ``dart_source_check`` + the
                                # gate-runner aggregate-rate warning.
                                per_claim_support[s_idx]["outcome"] = (
                                    _OUTCOME_DART_DISAGREEMENT
                                )
                                dart_disagreement_count += 1
                            elif (
                                d_entailment >= self._entailment_floor
                            ):
                                d_outcome = "entailed"
                            else:
                                d_outcome = "unsupported"
                            per_claim_support[s_idx][
                                "dart_source_check"
                            ] = {
                                "entailment": d_entailment,
                                "contradiction": d_contradiction,
                                "outcome": d_outcome,
                            }
                            dart_check_fired_count += 1

        total_claims = (
            entailed_count + unsupported_count + contradicted_count
        )

        # ``claim_support_rate`` is the entailed-rate (fraction of
        # claims the chunk entails) — operator-readable as "how much
        # of this pair is grounded?". Distinct from
        # ``unsupported_claim_rate`` which is the rejection trigger.
        claim_support_rate: Optional[float] = (
            entailed_count / total_claims if total_claims > 0 else None
        )
        claim_contradicted_rate: Optional[float] = (
            contradicted_count / total_claims if total_claims > 0 else None
        )
        unsupported_claim_rate: Optional[float] = (
            (unsupported_count + contradicted_count) / total_claims
            if total_claims > 0
            else None
        )

        # ---- Reject precedence ---- #
        # Order: contradicted (hard signal) → unsupported. Mirrors the
        # Wave 2 W2.F sibling at lines 622-687.
        rejection_reason: Optional[str] = None
        if (
            claim_contradicted_rate is not None
            and claim_contradicted_rate > self._max_contradicted
        ):
            rejection_reason = _REASON_CONTRADICTED_CLAIM
        elif (
            unsupported_claim_rate is not None
            and unsupported_claim_rate > self._max_unsupported
        ):
            rejection_reason = _REASON_UNSUPPORTED_CLAIM

        promotion_status = (
            "rejected" if rejection_reason is not None else "validated"
        )

        new_fields = {
            "per_claim_support": per_claim_support,
            "claim_support_rate": claim_support_rate,
            "claim_contradicted_rate": claim_contradicted_rate,
            "deps_missing": False,
        }

        _emit_decision(
            decision_capture,
            pair_kind=kind,
            chunk_id=chunk_id,
            promotion_status=promotion_status,
            rejection_reason=rejection_reason,
            total_claims=total_claims,
            entailed_count=entailed_count,
            unsupported_count=unsupported_count,
            contradicted_count=contradicted_count,
            claim_support_rate=claim_support_rate,
            claim_contradicted_rate=claim_contradicted_rate,
            nli_loaded=nli_loaded,
            deps_missing=False,
            structured_attribution_used=structured_attribution_used,
            per_claim_match_count=per_claim_match_count,
            dart_check_fired_count=dart_check_fired_count,
            dart_disagreement_count=dart_disagreement_count,
        )

        return promotion_status, rejection_reason, new_fields

    # ------------------------------------------------------------------ #
    # Gate-runner surface — walks training_specs/*.jsonl
    # ------------------------------------------------------------------ #

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Post-hoc audit that every pair on disk carries
        ``per_claim_support`` + ``claim_support_rate``. NOT a re-run of
        the per-pair NLI fan-out (that would require the chunks lookup
        map and is the call-site's job — same contract as
        :meth:`TrainingPairPromotionValidator.validate`).

        Inputs:

        - ``course_dir`` (str) — preferred. Resolves to
          ``<course_dir>/training_specs/{instruction,preference}_pairs.jsonl``.
        - ``training_specs_dir`` (str) — sibling alternative.
        - ``instruction_pairs_path`` (str) +
          ``preference_pairs_path`` (str) — explicit overrides.

        Outputs:

        - ``passed=True`` when every pair on disk has
          ``per_claim_support`` (list-or-None) AND ``claim_support_rate``
          keys present (value can be ``None`` for the deps-missing
          arm).
        - ``passed=False`` with capped ``MISSING_PER_CLAIM_SUPPORT`` /
          ``MISSING_CLAIM_SUPPORT_RATE`` issues otherwise.
        - Decision-capture: emits one
          ``pair_claim_support_check`` event with
          ``decision="audit:passed"`` /
          ``decision="audit:failed:N_missing"``.
        """
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

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
                code=_CODE_MISSING_INPUTS,
                message=(
                    "PairClaimSupportValidator requires one of: "
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
        missing_field_count = 0
        # Wave 9 TIGHT: count pairs whose ``per_claim_support[*].outcome
        # == "dart_disagreement"``. Aggregate-rate warning fires when
        # the rate exceeds :data:`_DART_DISAGREEMENT_RATE_WARN_CEILING`
        # — operator-visible signal without per-pair reject. A pair
        # counts at most once toward the disagreement count regardless
        # of how many of its per-claim entries flipped.
        dart_disagreement_pairs = 0
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
                        # ``per_claim_support`` may legitimately be
                        # ``None`` (deps-missing arm) or a list (the
                        # scored arm). The audit checks key presence,
                        # not value type.
                        has_pcs = "per_claim_support" in row
                        has_csr = "claim_support_rate" in row
                        if not has_pcs:
                            missing_field_count += 1
                            if len(issues) < _GATE_ISSUE_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code=_CODE_MISSING_PER_CLAIM_SUPPORT,
                                    message=(
                                        f"{path.name}:{line_num} pair "
                                        f"missing per_claim_support; "
                                        f"the per-pair claim-support "
                                        f"validator did not stamp the "
                                        f"pair before emit."
                                    ),
                                    location=str(path),
                                ))
                        if not has_csr:
                            missing_field_count += 1
                            if len(issues) < _GATE_ISSUE_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code=_CODE_MISSING_CLAIM_SUPPORT_RATE,
                                    message=(
                                        f"{path.name}:{line_num} pair "
                                        f"missing claim_support_rate; "
                                        f"the per-pair claim-support "
                                        f"validator did not stamp the "
                                        f"pair before emit."
                                    ),
                                    location=str(path),
                                ))
                        # Wave 9 TIGHT — DART disagreement rate.
                        pcs = row.get("per_claim_support")
                        if isinstance(pcs, list):
                            for entry in pcs:
                                if (
                                    isinstance(entry, dict)
                                    and entry.get("outcome")
                                    == _OUTCOME_DART_DISAGREEMENT
                                ):
                                    dart_disagreement_pairs += 1
                                    break
            except OSError as exc:
                issues.append(GateIssue(
                    severity="critical",
                    code=_CODE_PAIRS_FILE_READ_ERROR,
                    message=(
                        f"Failed to read {path}: {exc}"
                    ),
                    location=str(path),
                ))

        # Wave 9 TIGHT — aggregate DART-disagreement rate warning.
        dart_disagreement_rate: float = (
            dart_disagreement_pairs / audited if audited > 0 else 0.0
        )
        if (
            audited > 0
            and dart_disagreement_rate
            > _DART_DISAGREEMENT_RATE_WARN_CEILING
        ):
            issues.append(GateIssue(
                severity="warning",
                code=_CODE_DART_DISAGREEMENT_RATE_HIGH,
                message=(
                    f"Wave 9 TIGHT dual-source DART cross-check: "
                    f"{dart_disagreement_pairs} of {audited} pair(s) "
                    f"({dart_disagreement_rate:.1%}) carry a "
                    f"per-claim outcome of "
                    f"'{_OUTCOME_DART_DISAGREEMENT}' — exceeds the "
                    f"day-1 warning ceiling of "
                    f"{_DART_DISAGREEMENT_RATE_WARN_CEILING:.0%}. "
                    f"Indicates chunker-drift slice (Finding 5): the "
                    f"IMSCC chunk entails the pair-side claim but at "
                    f"least one DART block contradicts at >= the "
                    f"DART-side contradiction floor. Review "
                    f"merge_small_sections / sentence-split sub-"
                    f"chunking on the affected chunks."
                ),
            ))

        passed = missing_field_count == 0 and not any(
            i.severity == "critical" for i in issues
        )
        action: Optional[str] = None if passed else "block"

        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="pair_claim_support_check",
                    decision=(
                        "audit:passed"
                        if passed
                        else f"audit:failed:{missing_field_count}_missing"
                    ),
                    rationale=(
                        f"On-disk audit: {audited} pair(s) audited, "
                        f"{missing_field_count} missing "
                        f"per_claim_support / claim_support_rate "
                        f"fields, dart_disagreement_pairs="
                        f"{dart_disagreement_pairs} "
                        f"(rate={dart_disagreement_rate:.4f}, ceiling="
                        f"{_DART_DISAGREEMENT_RATE_WARN_CEILING}). "
                        f"inst_path="
                        f"{inst_path.name if inst_path else 'n/a'}, "
                        f"pref_path="
                        f"{pref_path.name if pref_path else 'n/a'}."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "DecisionCapture.log_decision raised on "
                    "pair_claim_support_check (gate path): %s",
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
                else round((audited - missing_field_count) / audited, 4)
            ),
            issues=issues,
            action=action,
        )


__all__ = [
    "PairClaimSupportValidator",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_MAX_UNSUPPORTED_RATE",
    "_DEFAULT_MAX_CONTRADICTED_RATE",
    "_DEFAULT_DART_CONTRADICTION_FLOOR",
    "_DART_DISAGREEMENT_RATE_WARN_CEILING",
    "_MIN_SENTENCE_TOKENS",
    "_PER_CLAIM_MATCH_COSINE_FLOOR",
    "_REASON_UNSUPPORTED_CLAIM",
    "_REASON_CONTRADICTED_CLAIM",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_MISSING_PER_CLAIM_SUPPORT",
    "_CODE_MISSING_CLAIM_SUPPORT_RATE",
    "_CODE_DART_DISAGREEMENT_RATE_HIGH",
    "_OUTCOME_DART_DISAGREEMENT",
    "_resolve_chunk_key_claims_with_attribution",
    "_resolve_dart_block_texts_for_chunk",
]
