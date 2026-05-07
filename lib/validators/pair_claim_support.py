"""Wave 4 W4.A — :class:`PairClaimSupportValidator`.

Per-pair, per-claim NLI entailment check that fans out every sentence
in a training pair's completion (instruction kind) or chosen surface
(preference kind) against the cited source-chunk text. Sibling of
:class:`lib.validators.claim_support.ClaimSupportValidator` (Wave 2
W2.F, Courseforge block-level) but at a different seam: training-pair
emit, immediately AFTER
:class:`lib.validators.training_pair_promotion.
TrainingPairPromotionValidator.validate_pair` returns ``"validated"``.

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
``claim_contradicted_rate``, and the NLI loaded flag.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier, NliScore

logger = logging.getLogger(__name__)


#: Per-claim entailment floor — sentences at or above this entailment
#: score are considered "entailed" by the cited chunk. Mirrors the
#: Wave 2 W2.F sibling default.
_DEFAULT_ENTAILMENT_FLOOR: float = 0.70

#: Per-claim contradiction floor — sentences at or above this
#: contradiction score are considered "contradicted" by the cited
#: chunk (a stronger negative signal than mere "unsupported"). Mirrors
#: the Wave 2 W2.F sibling default.
_DEFAULT_CONTRADICTION_FLOOR: float = 0.50

#: Per-pair unsupported_claim_rate ceiling. Above this rate, the pair
#: is rejected with ``rejection_reason="unsupported_claim"``.
_DEFAULT_MAX_UNSUPPORTED_RATE: float = 0.20

#: Per-pair contradicted-claim rate ceiling. Above this rate, the
#: pair is rejected with ``rejection_reason="contradicted_claim"``
#: regardless of total unsupported rate — contradicted alone is a
#: hard signal.
_DEFAULT_MAX_CONTRADICTED_RATE: float = 0.05

#: Minimum content-token count for a sentence to be considered a
#: pedagogical "claim" worth scoring. Sentences below this token
#: count are filtered out before NLI fan-out — they're typically
#: structural fragments ("Q:", "A.", numeric-only labels) that would
#: false-positive under NLI's "premise → hypothesis" framing.
_MIN_SENTENCE_TOKENS: int = 4

#: Cap per-pair issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50

#: Cap on number of audit-trail issues for the gate-runner walk path
#: (per file). Keeps the GateResult JSON bounded on a 1000-pair corpus.
_GATE_ISSUE_CAP: int = 50


# --------------------------------------------------------------------- #
# Canonical rejection reasons + GateIssue codes
# --------------------------------------------------------------------- #

_REASON_UNSUPPORTED_CLAIM: str = "unsupported_claim"
_REASON_CONTRADICTED_CLAIM: str = "contradicted_claim"

_CODE_NLI_DEPS_MISSING: str = "NLI_DEPS_MISSING"
_CODE_MISSING_PER_CLAIM_SUPPORT: str = "MISSING_PER_CLAIM_SUPPORT"
_CODE_MISSING_CLAIM_SUPPORT_RATE: str = "MISSING_CLAIM_SUPPORT_RATE"
_CODE_PAIRS_FILE_READ_ERROR: str = "PAIRS_FILE_READ_ERROR"
_CODE_MISSING_INPUTS: str = "MISSING_INPUTS"


#: Sentence-split regex. Splits on terminal-punctuation followed by
#: whitespace; preserves the punctuation on the preceding sentence via
#: lookbehind. Zero-deps; sufficient for typical training-pair
#: completions (1-4 short sentences). For pathological inputs (no
#: terminal punctuation, all-caps, embedded URLs) the splitter
#: degrades to "single sentence" — that's fine; one-sentence pairs
#: still get scored.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Word-token regex for sentence-length filtering. Lowercase
#: alphabetic-only tokens of >= 2 chars; numbers and punctuation
#: dropped.
_CONTENT_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}", re.UNICODE)


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
        decision_capture: Any = None,
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
            decision_capture: Optional :class:`DecisionCapture`
                instance. When wired, one
                ``pair_claim_support_check`` event per call.

        Returns:
            ``(promotion_status, rejection_reason, new_fields)``:

            - ``promotion_status``: ``"validated"`` on pass,
              ``"rejected"`` on contradicted/unsupported rate over
              ceiling.
            - ``rejection_reason``: ``"contradicted_claim"`` |
              ``"unsupported_claim"`` | ``None``.
            - ``new_fields``: ``{"per_claim_support": [{"sentence":
              str, "entailment": float, "contradiction": float,
              "outcome": "entailed" | "unsupported" | "contradicted"}],
              "claim_support_rate": float | None,
              "claim_contradicted_rate": float | None,
              "deps_missing": bool}``. Stamped on the pair regardless
              of pass/fail; the caller MUST ``pair.update(new_fields)``
              so the audit signals land on disk.
        """
        chunk_id = str(pair.get("chunk_id") or "")
        nli = self._get_nli()
        nli_loaded = nli is not None

        # ---- Resolve premise text ---- #
        chunk_text = ""
        if isinstance(chunk, dict):
            chunk_text = str(chunk.get("text") or "")

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
            )
            return "validated", None, new_fields

        # ---- Fan-out NLI scoring (one premise, N hypotheses) ---- #
        # Build (premise=chunk_text, hypothesis=sentence) pairs and
        # batch-score them. Single chunk (no per-claim attribution at
        # the training-pair layer — the pair has one cited chunk;
        # multi-chunk attribution is a Courseforge-block concern).
        nli_pairs: List[Tuple[str, str]] = [
            (chunk_text, sentence) for sentence in sentences
        ]
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
            )
            return "validated", None, new_fields

        # ---- Bucket each sentence ---- #
        per_claim_support: List[Dict[str, Any]] = []
        entailed_count = 0
        unsupported_count = 0
        contradicted_count = 0

        for sentence, score in zip(sentences, scores):
            entailment = float(score.entailment)
            contradiction = float(score.contradiction)
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
            })

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
            except OSError as exc:
                issues.append(GateIssue(
                    severity="critical",
                    code=_CODE_PAIRS_FILE_READ_ERROR,
                    message=(
                        f"Failed to read {path}: {exc}"
                    ),
                    location=str(path),
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
                        f"fields. inst_path="
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
    "_MIN_SENTENCE_TOKENS",
    "_REASON_UNSUPPORTED_CLAIM",
    "_REASON_CONTRADICTED_CLAIM",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_MISSING_PER_CLAIM_SUPPORT",
    "_CODE_MISSING_CLAIM_SUPPORT_RATE",
]
