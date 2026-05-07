"""GPT Feedback v2 Wave 2 W2.F — ``ClaimSupportValidator``.

Per-block-per-claim NLI entailment check that fans out every
``key_claims[]`` entry against the block's cited source chunks (Wave
1.5 W1.5.A structured ``List[{claim, source_chunk_ids[]}]`` shape with
back-compat for the legacy ``List[str]`` shape).

Plan §"Worker W2.F::Algorithm" (lines 388-414):

1. Skip blocks whose ``block_type`` (or ``content_type_label``) is in
   :data:`_SKIPPED_BLOCK_TYPES` — same skip set as
   :class:`lib.validators.rewrite_source_grounding.
   RewriteSourceGroundingValidator` (assessment_item / objective /
   chrome / recap). Skip rationale per type:

     - ``assessment_item`` — questions ground via stem ↔ LO cosine,
       not via per-claim NLI against source.
     - ``objective`` — re-states a course goal; ground via paraphrase
       ↔ source-objective cosine.
     - ``chrome`` — template scaffolding with no pedagogical claims.
     - ``recap`` — week-summary chrome; per-week summary aggregates
       have their own grounding seam.

2. Strip block HTML to text via
   :func:`lib.validators.rewrite_source_grounding._strip_html_to_text`
   (single source of truth — the validator does not fork).

3. Resolve cited source chunks via
   :func:`lib.validators.rewrite_source_grounding._resolve_block_source_chunks`.

4. Read ``block.content["key_claims"]``. Two shapes per Wave 1.5
   W1.5.A oneOf:

     - **Legacy ``List[str]``** → fan out each claim against the
       union of all cited chunks (block-level fallback).
     - **Structured ``List[{claim, source_chunk_ids[]}]``** → fan out
       each claim against ONLY the chunks named in
       ``claim.source_chunk_ids[]`` (per-claim attribution scoping —
       the win Wave 1.5 shipped).

5. For each (claim, candidate_chunk) pair: NLI(premise=chunk_text,
   hypothesis=claim_text). Per claim, take ``max(entailment_score)``
   AND ``max(contradiction_score)`` across the candidate chunk
   universe.

6. Bucket each claim:

     - ``entailed`` if max_entailment >= :data:`_DEFAULT_ENTAILMENT_FLOOR`
       (0.7).
     - ``contradicted`` if max_contradiction >=
       :data:`_DEFAULT_CONTRADICTION_FLOOR` (0.5).
     - ``unsupported`` otherwise.

7. ``unsupported_claim_rate = (unsupported + contradicted) /
   total_claims`` per block. Above
   :data:`_DEFAULT_MAX_UNSUPPORTED_RATE` (0.20) emits
   :data:`_CODE_UNSUPPORTED_CLAIM` (warning, ``action="regenerate"``).
   ``contradicted_count / total_claims`` above
   :data:`_DEFAULT_MAX_CONTRADICTED_RATE` (0.05) emits
   :data:`_CODE_CONTRADICTED_CLAIM` (warning, ``action="regenerate"``)
   regardless of total unsupported rate — contradicted alone is a
   hard signal.

Graceful degrade: when :meth:`lib.classifiers.nli_classifier.
NliClassifier.get_or_load` returns ``None`` (extras absent), emits
:data:`_CODE_NLI_DEPS_MISSING` warning, ``passed=True, action=None``.
Strict mode via the existing ``TRAINFORGE_REQUIRE_EMBEDDINGS=true``
flag re-enables fail-closed (raises ``RuntimeError``).

Decision-capture: emits one ``claim_support_check`` decision per
audited block. Rationale interpolates block_id, total_claims,
entailed_count, unsupported_count, contradicted_count, and the
resolved unsupported_claim_rate.

Cross-references:

- Wave 1.5 structured ``key_claims`` shape: ``schemas/knowledge/
  courseforge_jsonld_v1.schema.json::$defs.KeyClaimWithSources`` +
  ``schemas/tests/test_wave15_per_claim_attribution_contract.py``.
- Reused helpers: ``lib/validators/rewrite_source_grounding.py``
  ``_strip_html_to_text`` + ``_resolve_block_source_chunks``.
- NLI loader contract: ``lib/classifiers/nli_classifier.py`` (Wave 2
  W2.F real loader; W1.7.C stub returned ``None``).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier, NliScore
from lib.embedding.sentence_embedder import is_strict_mode
from lib.utils import strip_html_to_text as _strip_html_to_text  # W-D6 canonical
from lib.validators.rewrite_source_grounding import (
    _resolve_block_source_chunks,
)

# Block lives at Courseforge/scripts/blocks.py — same import bridge
# as the sibling validators.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Block types skipped — same set as
#: :data:`lib.validators.rewrite_source_grounding._SKIPPED_CONTENT_TYPES`
#: (``assessment_item`` / ``objective``) plus chrome scaffolding
#: (``chrome`` / ``recap``) which carries no pedagogical claims.
#: ``self_check_question`` / ``example`` are NOT in this set — they
#: legitimately carry per-claim assertions that benefit from NLI
#: support checking even when the rewrite-tier sentence-grounding
#: gate skips them.
_SKIPPED_BLOCK_TYPES: frozenset = frozenset(
    {"assessment_item", "objective", "chrome", "recap"}
)


#: Per-claim entailment floor — claims at or above this max-across-
#: chunks entailment score are considered "entailed" by their cited
#: chunks. Mirrors the Wave 2 W2.F brief constant.
_DEFAULT_ENTAILMENT_FLOOR: float = 0.70


#: Per-claim contradiction floor — claims at or above this max-across-
#: chunks contradiction score are considered "contradicted" by their
#: cited chunks (a stronger negative signal than mere "unsupported").
_DEFAULT_CONTRADICTION_FLOOR: float = 0.50


#: Per-block unsupported_claim_rate ceiling. Above this rate, the
#: block fires :data:`_CODE_UNSUPPORTED_CLAIM`.
_DEFAULT_MAX_UNSUPPORTED_RATE: float = 0.20


#: Per-block contradicted-claim rate ceiling. Above this rate, the
#: block fires :data:`_CODE_CONTRADICTED_CLAIM` regardless of total
#: unsupported rate — contradicted alone is a hard signal.
_DEFAULT_MAX_CONTRADICTED_RATE: float = 0.05


#: Cap per-block issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50


# --------------------------------------------------------------------- #
# Canonical GateIssue codes
# --------------------------------------------------------------------- #

_CODE_UNSUPPORTED_CLAIM: str = "UNSUPPORTED_CLAIM"
_CODE_CONTRADICTED_CLAIM: str = "CONTRADICTED_CLAIM"
_CODE_NLI_DEPS_MISSING: str = "NLI_DEPS_MISSING"


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs["blocks"]``.

    Mirrors :func:`lib.validators.rewrite_source_grounding._coerce_blocks`.
    """
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
    """True when the block's block_type / content_type is in the skip set."""
    if block.block_type in _SKIPPED_BLOCK_TYPES:
        return True
    if (
        block.content_type_label
        and block.content_type_label in _SKIPPED_BLOCK_TYPES
    ):
        return True
    return False


def _resolve_key_claims(
    block: Block,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Pull ``key_claims`` off a Block (dict-content or rewrite-tier).

    Returns a tuple of ``(normalized_claims, is_structured)``. Each
    entry in ``normalized_claims`` carries ``{"claim": str,
    "source_chunk_ids": Optional[List[str]]}``. ``is_structured`` is
    True when at least one entry carried per-claim attribution (the
    Wave 1.5 W1.5.A structured shape); False when every entry was a
    bare string (legacy shape).

    For mixed shapes (a contract violation per Wave 1.5 schema) the
    helper accepts each entry on its own terms — bare strings get
    ``source_chunk_ids=None`` (block-level fallback), structured
    entries keep their attribution. The schema gate fires upstream.
    """
    content = block.content
    if not isinstance(content, dict):
        return [], False
    raw = content.get("key_claims")
    if not isinstance(raw, list):
        return [], False

    normalized: List[Dict[str, Any]] = []
    is_structured = False
    for entry in raw:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                normalized.append(
                    {"claim": text, "source_chunk_ids": None}
                )
        elif isinstance(entry, dict):
            text_raw = entry.get("claim")
            if not isinstance(text_raw, str) or not text_raw.strip():
                continue
            attribution = entry.get("source_chunk_ids")
            if isinstance(attribution, list):
                # Filter to non-empty strings; preserve order.
                ids = [
                    sid for sid in attribution
                    if isinstance(sid, str) and sid
                ]
                normalized.append(
                    {
                        "claim": text_raw.strip(),
                        "source_chunk_ids": ids if ids else None,
                    }
                )
                if ids:
                    is_structured = True
            else:
                normalized.append(
                    {
                        "claim": text_raw.strip(),
                        "source_chunk_ids": None,
                    }
                )
    return normalized, is_structured


def _candidate_chunks_for_claim(
    *,
    claim_attribution: Optional[List[str]],
    block_chunk_map: Dict[str, str],
    block_chunk_universe: List[str],
) -> List[str]:
    """Resolve the candidate chunk-text universe for one claim.

    Per-claim attribution path: when ``claim_attribution`` is a non-
    empty list, the candidate universe is just the chunks named in
    that list (intersected with the block's resolved chunk map).
    Block-level fallback path: when attribution is ``None``, the
    candidate universe is every chunk the block declared via
    ``source_references[]`` / ``source_ids``.
    """
    if claim_attribution:
        result: List[str] = []
        for sid in claim_attribution:
            text = block_chunk_map.get(sid)
            if isinstance(text, str) and text.strip():
                result.append(text)
        return result
    return list(block_chunk_universe)


def _emit_decision(
    capture: Any,
    block: Block,
    *,
    passed: bool,
    failure_codes: List[str],
    total_claims: int,
    entailed_count: int,
    unsupported_count: int,
    contradicted_count: int,
    unsupported_rate: Optional[float],
    contradicted_rate: Optional[float],
    is_structured: bool,
    nli_loaded: bool,
) -> None:
    """Emit one ``claim_support_check`` decision per audited block.

    Rationale interpolates the six required signals enumerated in the
    Wave 2 W2.F brief: block_id, total_claims, entailed_count,
    unsupported_count, contradicted_count, resolved
    unsupported_claim_rate.
    """
    if capture is None:
        return
    decision = (
        "passed" if passed else f"failed:{','.join(failure_codes) or 'unknown'}"
    )
    rate_str = (
        f"{unsupported_rate:.4f}" if unsupported_rate is not None else "n/a"
    )
    cont_rate_str = (
        f"{contradicted_rate:.4f}"
        if contradicted_rate is not None
        else "n/a"
    )
    rationale = (
        f"Per-claim NLI support check on Block {block.block_id!r}: "
        f"block_type={block.block_type!r}, "
        f"total_claims={total_claims}, "
        f"entailed_count={entailed_count}, "
        f"unsupported_count={unsupported_count}, "
        f"contradicted_count={contradicted_count}, "
        f"unsupported_claim_rate={rate_str}, "
        f"contradicted_claim_rate={cont_rate_str}, "
        f"is_structured_attribution={is_structured}, "
        f"nli_loaded={nli_loaded}, "
        f"failure_codes={','.join(failure_codes) or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="claim_support_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on claim_support_check: %s",
            exc,
        )


class ClaimSupportValidator:
    """Per-block-per-claim NLI entailment gate.

    Iterates every audited Block (skip set per
    :data:`_SKIPPED_BLOCK_TYPES`); for each claim in
    ``block.content["key_claims"]``, computes the max entailment +
    max contradiction across the candidate chunk universe (per-claim
    attribution scope when the structured shape is present; block-
    level fallback otherwise). Buckets claims into entailed /
    unsupported / contradicted; fires
    :data:`_CODE_UNSUPPORTED_CLAIM` when unsupported_claim_rate > 0.20
    and :data:`_CODE_CONTRADICTED_CLAIM` when contradicted_count rate
    > 0.05.

    Inputs:
        blocks: List[Block]
        source_chunks: Dict[str, str]
            Mapping of canonical sourceId (e.g. ``dart:slug#blk_0``)
            to the chunk's plain-text body. The workflow runner
            populates this from the staging manifest before dispatch;
            tests inject it directly.
        max_unsupported_claim_rate: Optional[float]
            Override :data:`_DEFAULT_MAX_UNSUPPORTED_RATE`.
        max_contradicted_claim_rate: Optional[float]
            Override :data:`_DEFAULT_MAX_CONTRADICTED_RATE`.
        entailment_floor: Optional[float]
            Override :data:`_DEFAULT_ENTAILMENT_FLOOR`.
        contradiction_floor: Optional[float]
            Override :data:`_DEFAULT_CONTRADICTION_FLOOR`.
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per audited block.
        nli: Optional[NliClassifier]
            Test-injection point. When None, calls
            :meth:`NliClassifier.get_or_load`.

    Graceful-degrade path: when the NLI loader returns ``None`` AND
    ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is unset, emit a single warning-
    severity :data:`_CODE_NLI_DEPS_MISSING` GateIssue with
    ``passed=True, action=None``. Strict mode raises ``RuntimeError``.
    """

    name = "claim_support"
    version = "1.0.0"

    def __init__(
        self,
        *,
        max_unsupported_claim_rate: float = _DEFAULT_MAX_UNSUPPORTED_RATE,
        max_contradicted_claim_rate: float = _DEFAULT_MAX_CONTRADICTED_RATE,
        entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
        contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
        nli: Optional[NliClassifier] = None,
    ) -> None:
        self._max_unsupported = float(max_unsupported_claim_rate)
        self._max_contradicted = float(max_contradicted_claim_rate)
        self._entailment_floor = float(entailment_floor)
        self._contradiction_floor = float(contradiction_floor)
        self._nli_override = nli

    def _get_nli(self) -> Optional[NliClassifier]:
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    def _resolve_thresholds(
        self, inputs: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        """Resolve thresholds, honoring the executor's gate.config merge.

        The executor merges ``gate.config.thresholds`` into the inputs
        dict before dispatching the gate (per the existing pattern at
        ``executor.py:1442``). Operators can override per-gate via:

        .. code-block:: yaml

            config:
              thresholds:
                max_unsupported_claim_rate: 0.30
                max_contradicted_claim_rate: 0.10
                entailment_floor: 0.65
                contradiction_floor: 0.45
        """
        return (
            float(inputs.get("max_unsupported_claim_rate", self._max_unsupported)),
            float(inputs.get("max_contradicted_claim_rate", self._max_contradicted)),
            float(inputs.get("entailment_floor", self._entailment_floor)),
            float(inputs.get("contradiction_floor", self._contradiction_floor)),
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        (
            max_unsupported,
            max_contradicted,
            entailment_floor,
            contradiction_floor,
        ) = self._resolve_thresholds(inputs)

        nli = self._get_nli()
        nli_loaded = nli is not None

        # Strict-mode + missing deps → fail-closed (RuntimeError).
        # Graceful-degrade default → warning-severity GateIssue.
        if not nli_loaded:
            if is_strict_mode():
                raise RuntimeError(
                    "ClaimSupportValidator requires the NLI classifier "
                    "(MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) "
                    "but TRAINFORGE_REQUIRE_EMBEDDINGS is set and "
                    "NliClassifier.get_or_load() returned None. "
                    "Install transformers + torch via "
                    "`pip install -e .[embedding]` or unset "
                    "TRAINFORGE_REQUIRE_EMBEDDINGS for graceful "
                    "degradation."
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
                        code=_CODE_NLI_DEPS_MISSING,
                        message=(
                            "NliClassifier.get_or_load() returned None; "
                            "Wave-2 W2.F per-claim NLI support gate "
                            "skipped. Install transformers + torch via "
                            "`pip install -e .[embedding]` to enable. "
                            "Set TRAINFORGE_REQUIRE_EMBEDDINGS=true to "
                            "fail closed instead."
                        ),
                        suggestion=(
                            "The graceful-degrade path is intentional "
                            "for CPU-only dev boxes; production runs "
                            "should install the embedding extras."
                        ),
                    )
                ],
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
        any_regenerate = False
        audited_blocks = 0
        passed_blocks = 0

        for block in blocks:
            if _block_should_skip(block):
                # Skip silently — no event, no issue. Skip set is
                # documented in the module docstring.
                continue

            normalized_claims, is_structured = _resolve_key_claims(block)
            if not normalized_claims:
                # No key_claims declared; gate is a no-op for this
                # block. Don't even fire a decision event — the gate
                # didn't audit anything.
                continue

            audited_blocks += 1
            block_chunk_universe = _resolve_block_source_chunks(
                block, source_chunks
            )
            # Also build a per-id map so the per-claim attribution
            # path can scope to the named ids only.
            block_chunk_map: Dict[str, str] = {}
            for ref in block.source_references or ():
                if isinstance(ref, dict):
                    sid = ref.get("sourceId")
                    if isinstance(sid, str) and sid:
                        text = source_chunks.get(sid)
                        if isinstance(text, str) and text.strip():
                            block_chunk_map[sid] = text
            for sid in block.source_ids or ():
                if isinstance(sid, str) and sid and sid not in block_chunk_map:
                    text = source_chunks.get(sid)
                    if isinstance(text, str) and text.strip():
                        block_chunk_map[sid] = text

            entailed_count = 0
            unsupported_count = 0
            contradicted_count = 0

            for claim_entry in normalized_claims:
                claim_text = claim_entry["claim"]
                attribution = claim_entry.get("source_chunk_ids")
                candidate_chunks = _candidate_chunks_for_claim(
                    claim_attribution=attribution,
                    block_chunk_map=block_chunk_map,
                    block_chunk_universe=block_chunk_universe,
                )

                if not candidate_chunks:
                    # No grounding surface available for this claim.
                    # Bucket as unsupported — the schema gate guards
                    # the structural shape upstream; here we score
                    # only what we can score, and an unciteable claim
                    # is unsupported by definition.
                    unsupported_count += 1
                    continue

                # Build (premise=chunk, hypothesis=claim) pairs and
                # batch-score them.
                pairs: List[Tuple[str, str]] = [
                    (chunk, claim_text) for chunk in candidate_chunks
                ]
                try:
                    scores: List[NliScore] = nli.score_batch(pairs=pairs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "NliClassifier.score_batch raised on block %r "
                        "claim fan-out: %s",
                        block.block_id, exc,
                    )
                    # Defensive: bucket as unsupported when the NLI
                    # call raises mid-fan-out. The graceful-degrade
                    # path covers the common "no model loaded" case;
                    # this branch covers per-call exceptions.
                    unsupported_count += 1
                    continue

                max_entailment = max(
                    (s.entailment for s in scores), default=0.0
                )
                max_contradiction = max(
                    (s.contradiction for s in scores), default=0.0
                )

                if max_entailment >= entailment_floor:
                    entailed_count += 1
                elif max_contradiction >= contradiction_floor:
                    contradicted_count += 1
                else:
                    unsupported_count += 1

            total_claims = (
                entailed_count + unsupported_count + contradicted_count
            )
            unsupported_rate: Optional[float] = (
                (unsupported_count + contradicted_count) / total_claims
                if total_claims > 0
                else None
            )
            contradicted_rate: Optional[float] = (
                contradicted_count / total_claims
                if total_claims > 0
                else None
            )

            failure_codes: List[str] = []

            # CONTRADICTED_CLAIM fires standalone above its rate ceiling.
            if (
                contradicted_rate is not None
                and contradicted_rate > max_contradicted
                and len(issues) < _ISSUE_LIST_CAP
            ):
                issues.append(
                    GateIssue(
                        severity="warning",
                        code=_CODE_CONTRADICTED_CLAIM,
                        message=(
                            f"Block {block.block_id!r} "
                            f"(block_type={block.block_type!r}): "
                            f"{contradicted_count}/{total_claims} "
                            f"({contradicted_rate:.1%}) of "
                            f"key_claims were contradicted by their "
                            f"cited source chunks (max_contradiction "
                            f">= {contradiction_floor:.2f}); "
                            f"per-block ceiling is "
                            f"{max_contradicted:.1%}. Contradicted "
                            f"claims are a hard signal regardless "
                            f"of total unsupported rate."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "The block prose contradicts its declared "
                            "source. Re-prompt with the source chunks "
                            "pinned and require strict paraphrase."
                        ),
                    )
                )
                failure_codes.append(_CODE_CONTRADICTED_CLAIM)
                any_regenerate = True

            # UNSUPPORTED_CLAIM fires when the (unsupported +
            # contradicted) rate crosses the ceiling.
            if (
                unsupported_rate is not None
                and unsupported_rate > max_unsupported
                and len(issues) < _ISSUE_LIST_CAP
            ):
                issues.append(
                    GateIssue(
                        severity="warning",
                        code=_CODE_UNSUPPORTED_CLAIM,
                        message=(
                            f"Block {block.block_id!r} "
                            f"(block_type={block.block_type!r}): "
                            f"{unsupported_count + contradicted_count}/"
                            f"{total_claims} ({unsupported_rate:.1%}) "
                            f"of key_claims were not entailed by "
                            f"their cited source chunks "
                            f"(max_entailment < "
                            f"{entailment_floor:.2f}); per-block "
                            f"ceiling is {max_unsupported:.1%}."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Re-prompt with the source chunks pinned "
                            "and require explicit verbatim grounding "
                            "for every key_claim."
                        ),
                    )
                )
                failure_codes.append(_CODE_UNSUPPORTED_CLAIM)
                any_regenerate = True

            block_passed = not failure_codes
            if block_passed:
                passed_blocks += 1

            _emit_decision(
                capture, block,
                passed=block_passed,
                failure_codes=failure_codes,
                total_claims=total_claims,
                entailed_count=entailed_count,
                unsupported_count=unsupported_count,
                contradicted_count=contradicted_count,
                unsupported_rate=unsupported_rate,
                contradicted_rate=contradicted_rate,
                is_structured=is_structured,
                nli_loaded=nli_loaded,
            )

        # Score = pass rate over audited blocks.
        score = (
            1.0
            if audited_blocks == 0
            else round(passed_blocks / audited_blocks, 4)
        )
        # ``passed`` stays True because every issue is warning-severity
        # by construction (Day-1 contract). Action toggles to
        # ``regenerate`` when any block fired a real failure.
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=score,
            issues=issues,
            action="regenerate" if any_regenerate else None,
        )


__all__ = [
    "ClaimSupportValidator",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_MAX_UNSUPPORTED_RATE",
    "_DEFAULT_MAX_CONTRADICTED_RATE",
    "_CODE_UNSUPPORTED_CLAIM",
    "_CODE_CONTRADICTED_CLAIM",
    "_CODE_NLI_DEPS_MISSING",
    "_SKIPPED_BLOCK_TYPES",
]
