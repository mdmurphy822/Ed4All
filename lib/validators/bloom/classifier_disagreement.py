"""Phase 4 Subtask 27 — Bloom-classifier-disagreement validator.

Wraps :class:`lib.classifiers.bloom_bert_ensemble.BloomBertEnsemble`
into the standard ``Validator`` protocol so the workflow runner can
fire it against outline-tier / rewrite-tier ``Block`` lists at the
``inter_tier_validation`` and ``post_rewrite_validation`` seams.

Per-block contract:

1. Skip blocks whose ``block_type`` is NOT in
   :data:`_AUDITED_BLOCK_TYPES` (currently ``objective`` and
   ``assessment_item`` — the only block types whose
   ``bloom_level`` is a structural authoring decision the ensemble
   can audit).
2. Skip blocks whose declared ``bloom_level`` is empty / unknown — the
   ensemble can't disagree with a level that wasn't claimed.
3. Extract a textual surface from the block via
   :func:`_extract_text_for_classification` (mirrors the shape-dispatch
   in ``Courseforge/router/inter_tier_gates.py``: dict path pulls
   ``content["key_claims"]`` / ``["statement"]`` / ``["text"]``;
   str path strips HTML).
4. Classify the surface via the ensemble. Two failure modes emit a
   ``regenerate``-action GateIssue:
   - **Disagreement**: ensemble winner != declared ``bloom_level`` AND
     the winner score is above :data:`_DISAGREEMENT_CONFIDENCE_FLOOR`.
     Emits ``BERT_ENSEMBLE_DISAGREEMENT``.
   - **High dispersion**: ensemble dispersion > :data:`_DISPERSION_THRESHOLD`
     (default 0.7 per the plan). Emits ``BERT_ENSEMBLE_DISPERSION_HIGH``
     even when the winner agrees with the declared level — high
     dispersion signals an unstable consensus that's worth re-rolling.

Both failure modes route to ``action="regenerate"`` because the
underlying issue is content-side (ambiguous wording, miscalibrated
verb choice) the rewrite tier can plausibly fix on a second draft.
The validator NEVER emits ``action="block"`` — it's a soft
warning-tier signal during the Phase 4 PoC.

Graceful degradation: missing ``transformers`` extras yield a single
warning-severity GateIssue with code ``BERT_ENSEMBLE_DEPS_MISSING``,
``passed=True``, ``action=None`` — mirrors the embedding-tier
graceful-degrade pattern in ``lib/validators/courseforge_outline_shacl.py``.
Strict mode (``TRAINFORGE_REQUIRE_BERT_ENSEMBLE=true``) is honoured
inside the ensemble's ``_load_members``; a strict-mode raise
propagates as a critical GateIssue with ``action="block"``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.bloom_bert_ensemble import (
    BertEnsembleDepsMissing,
    BloomBertEnsemble,
)

logger = logging.getLogger(__name__)


#: Block types whose declared ``bloom_level`` is a structural
#: authoring decision the ensemble can audit. ``objective`` and
#: ``assessment_item`` are the canonical pair — every other block
#: type either doesn't carry a Bloom level at all (e.g. ``chrome``,
#: ``recap``) or carries one as a derived / stylistic field (e.g.
#: ``activity``, ``misconception``) where ensemble disagreement is
#: more likely a false positive than a real authoring miss.
_AUDITED_BLOCK_TYPES: frozenset = frozenset({"objective", "assessment_item"})


#: Default dispersion threshold per the Phase 4 plan. Above this
#: value the ensemble has no clear consensus; the validator emits a
#: ``BERT_ENSEMBLE_DISPERSION_HIGH`` GateIssue regardless of whether
#: the winner agrees with the declared level.
_DISPERSION_THRESHOLD: float = 0.7


#: Confidence floor for disagreement. The ensemble's winner only
#: triggers a disagreement event when its score is above this floor —
#: low-confidence wins are noise and shouldn't override the declared
#: level.
_DISAGREEMENT_CONFIDENCE_FLOOR: float = 0.4


#: Cap the number of per-block issues emitted so a uniformly-broken
#: outline batch doesn't drown the gate report. Mirrors
#: ``Courseforge/router/inter_tier_gates.py::_ISSUE_LIST_CAP``.
_ISSUE_LIST_CAP: int = 50


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _emit_decision(
    capture: Any,
    *,
    block_id: Optional[str],
    passed: bool,
    code: Optional[str],
    declared_level: Optional[str],
    winner_level: Optional[str],
    winner_score: Optional[float],
    dispersion: Optional[float],
    dispersion_threshold: float,
    confidence_floor: float,
    member_votes: Optional[List[Any]],
    members_loaded: int,
) -> None:
    """Emit one ``bloom_classifier_disagreement_check`` decision per
    audited block (or per pre-block-loop short-circuit event when no
    block-level signal is available, e.g. graceful-degrade).

    Rationale interpolates dynamic signals — block_id, declared level,
    ensemble winner + winner_score, dispersion + threshold, confidence
    floor, member-vote summary, and the loaded-member count (so
    BERT_ENSEMBLE_DEPS_MISSING graceful-degrade events stay
    distinguishable from real disagreement / dispersion events).
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    score_str = f"{winner_score:.3f}" if winner_score is not None else "n/a"
    disp_str = f"{dispersion:.3f}" if dispersion is not None else "n/a"
    votes_summary: str
    if member_votes:
        # Summarise as a compact list of (level, score) tuples; cap at
        # 5 to keep the rationale bounded.
        capped = list(member_votes)[:5]
        votes_summary = ",".join(
            f"({lvl}:{score:.2f})" if isinstance(score, (int, float))
            else f"({lvl}:?)"
            for lvl, score in (
                (m if isinstance(m, tuple) and len(m) == 2 else (str(m), 0.0))
                for m in capped
            )
        )
    else:
        votes_summary = "n/a"
    rationale = (
        f"Bloom-classifier-disagreement check on Block {block_id or 'n/a'!r}: "
        f"declared_level={declared_level or 'n/a'}, "
        f"ensemble_winner={winner_level or 'n/a'}, "
        f"winner_score={score_str}, "
        f"dispersion={disp_str}, "
        f"dispersion_threshold={dispersion_threshold:.3f}, "
        f"confidence_floor={confidence_floor:.3f}, "
        f"member_votes={votes_summary}, "
        f"members_loaded={members_loaded}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="bloom_classifier_disagreement_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "bloom_classifier_disagreement_check: %s",
            exc,
        )


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Cheap surface extractor for the rewrite-tier (str-path) blocks.
    Mirrors the helper in ``Courseforge/router/inter_tier_gates.py``
    — kept inlined to avoid the import-cycle that pulling
    ``inter_tier_gates`` here would create (it imports from
    ``Courseforge.scripts.blocks`` which itself transitively pulls
    in the renderer surface).
    """
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _extract_text_for_classification(block: Any) -> Optional[str]:
    """Pull a textual surface from a Block (or dict-shaped Block).

    Dict path (outline tier) priority:
        1. ``content["statement"]`` — the canonical objective / assessment
           statement field.
        2. ``content["text"]`` — the canonical generic text field.
        3. ``content["key_claims"]`` — joined with ``" "`` so the
           classifier sees the union.
    Str path (rewrite tier): strips HTML and returns the text body.

    Returns ``None`` when no usable text surface is available — the
    caller short-circuits past the block (no ensemble call, no event).
    """
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        for key in ("statement", "text"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        claims = content.get("key_claims")
        if isinstance(claims, list):
            joined = " ".join(c for c in claims if isinstance(c, str))
            if joined.strip():
                return joined.strip()
        return None
    if isinstance(content, str):
        stripped = _strip_html(content)
        return stripped if stripped else None
    return None


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` (or dict-shaped Block list) out of ``inputs``.

    Mirrors the ``inputs["blocks"]`` contract used by the inter-tier
    gates in ``Courseforge/router/inter_tier_gates.py``. Returns
    ``(blocks, error_issue)``; ``error_issue`` is non-None when the
    input shape is wrong.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances or block-shaped dicts."
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


def _block_attr(block: Any, key: str) -> Any:
    """Get ``block.<key>`` for dataclass blocks OR ``block[<key>]`` for dicts.

    Lets the validator stay shape-agnostic over the two supported
    Block representations: the canonical
    :class:`Courseforge.scripts.blocks.Block` dataclass and the
    snake_case dict round-trip used by the workflow runner's JSONL
    handoff between ``content_generation_outline`` and
    ``inter_tier_validation``.
    """
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, dict):
        return block.get(key)
    return None


class BloomClassifierDisagreementValidator:
    """Phase 4 Category D — BERT ensemble disagreement gate.

    Validator-protocol-compatible class wired into both
    ``inter_tier_validation::bloom_classifier_disagreement`` and
    ``post_rewrite_validation::bloom_classifier_disagreement``.
    Emits regenerate-action GateIssues on:

    - Ensemble winner != declared ``bloom_level`` (above the
      :data:`_DISAGREEMENT_CONFIDENCE_FLOOR`).
    - Ensemble dispersion > :data:`_DISPERSION_THRESHOLD`.

    Pass-conditions:

    - No ``objective`` / ``assessment_item`` blocks in the input set.
    - Every audited block's ensemble winner == declared bloom_level
      AND dispersion <= threshold.

    Graceful-degrade:

    - Missing ``transformers`` extras (default mode) yields one
      warning issue, ``passed=True``, ``action=None``.
    - Missing ``transformers`` extras (strict mode via
      ``TRAINFORGE_REQUIRE_BERT_ENSEMBLE=true``) yields one critical
      issue, ``passed=False``, ``action="block"``.
    """

    name = "bloom_classifier_disagreement"
    version = "0.1.0"  # Phase 4 PoC

    def __init__(
        self,
        ensemble: Optional[BloomBertEnsemble] = None,
        dispersion_threshold: float = _DISPERSION_THRESHOLD,
        confidence_floor: float = _DISAGREEMENT_CONFIDENCE_FLOOR,
    ) -> None:
        self._ensemble = ensemble  # lazy-instantiated when None
        self._dispersion_threshold = float(dispersion_threshold)
        self._confidence_floor = float(confidence_floor)

    def _get_ensemble(self) -> BloomBertEnsemble:
        if self._ensemble is None:
            self._ensemble = BloomBertEnsemble()
        return self._ensemble

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

        # Empty input is a no-op pass.
        if not blocks:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        # Lazy ensemble construction — strict-mode missing-deps raises
        # propagate as a critical GateIssue (action=block).
        try:
            ensemble = self._get_ensemble()
            loaded = ensemble._load_members()
        except BertEnsembleDepsMissing as exc:
            # Strict-mode degrade — emit one capture per audited block
            # so the audit trail records the gate ran on each block.
            for block in blocks:
                if _block_attr(block, "block_type") not in _AUDITED_BLOCK_TYPES:
                    continue
                _emit_decision(
                    capture,
                    block_id=_block_attr(block, "block_id"),
                    passed=False,
                    code="BERT_ENSEMBLE_DEPS_MISSING",
                    declared_level=_block_attr(block, "bloom_level"),
                    winner_level=None, winner_score=None,
                    dispersion=None,
                    dispersion_threshold=self._dispersion_threshold,
                    confidence_floor=self._confidence_floor,
                    member_votes=None, members_loaded=0,
                )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="BERT_ENSEMBLE_DEPS_MISSING",
                        message=(
                            f"BERT ensemble dependencies are missing in "
                            f"strict mode: {exc}"
                        ),
                        suggestion=(
                            "Install transformers via `pip install -e .[bert]` "
                            "or unset TRAINFORGE_REQUIRE_BERT_ENSEMBLE."
                        ),
                    )
                ],
                action="block",
            )

        # Default-mode missing extras → graceful-degrade warning.
        if not loaded:
            # Emit one passed=True capture per audited block so the
            # audit trail records the silent-degrade C5 signal.
            for block in blocks:
                if _block_attr(block, "block_type") not in _AUDITED_BLOCK_TYPES:
                    continue
                _emit_decision(
                    capture,
                    block_id=_block_attr(block, "block_id"),
                    passed=True,
                    code="BERT_ENSEMBLE_DEPS_MISSING",
                    declared_level=_block_attr(block, "bloom_level"),
                    winner_level=None, winner_score=None,
                    dispersion=None,
                    dispersion_threshold=self._dispersion_threshold,
                    confidence_floor=self._confidence_floor,
                    member_votes=None, members_loaded=0,
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
                        code="BERT_ENSEMBLE_DEPS_MISSING",
                        message=(
                            "BERT ensemble loaded zero members "
                            "(transformers extras missing or all members "
                            "failed to load). Phase 4 PoC gate skipped."
                        ),
                        suggestion=(
                            "Install transformers via `pip install -e .[bert]` "
                            "to enable Bloom-disagreement validation."
                        ),
                    )
                ],
            )

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0
        members_loaded = len(loaded) if isinstance(loaded, list) else 1

        for block in blocks:
            block_type = _block_attr(block, "block_type")
            if block_type not in _AUDITED_BLOCK_TYPES:
                continue

            declared = _block_attr(block, "bloom_level")
            if not isinstance(declared, str) or not declared:
                # Can't disagree with an unstated level. Skip silently —
                # the page_objectives / outline_curie_anchoring gates
                # already cover declared-field-presence requirements.
                continue

            text = _extract_text_for_classification(block)
            if not text:
                continue

            audited += 1
            try:
                result = ensemble.classify(text)
            except Exception as exc:  # noqa: BLE001 — silent-degrade on per-block failure
                logger.warning(
                    "BloomBertEnsemble.classify failed for block %s: %s",
                    _block_attr(block, "block_id"),
                    exc,
                )
                _emit_decision(
                    capture,
                    block_id=_block_attr(block, "block_id"),
                    passed=True,
                    code="ENSEMBLE_CLASSIFY_ERROR",
                    declared_level=declared,
                    winner_level=None, winner_score=None,
                    dispersion=None,
                    dispersion_threshold=self._dispersion_threshold,
                    confidence_floor=self._confidence_floor,
                    member_votes=None,
                    members_loaded=members_loaded,
                )
                continue

            winner = result.get("winner_level", "unknown")
            winner_score = float(result.get("winner_score", 0.0))
            dispersion = float(result.get("dispersion", 0.0))
            per_member = result.get("per_member") or []
            block_id = _block_attr(block, "block_id")

            block_passed = True

            # 1. Disagreement check (only fires when winner score is
            #    above the confidence floor; low-confidence wins are
            #    noise).
            if (
                winner != "unknown"
                and winner != declared
                and winner_score >= self._confidence_floor
            ):
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="BERT_ENSEMBLE_DISAGREEMENT",
                            message=(
                                f"Block {block_id!r} declares "
                                f"bloom_level={declared!r} but the BERT "
                                f"ensemble winner is {winner!r} "
                                f"(score={winner_score:.3f}, "
                                f"dispersion={dispersion:.3f})."
                            ),
                            location=block_id,
                            suggestion=(
                                "Re-roll the outline-tier provider with a "
                                "prompt that nudges the verb choice toward "
                                f"{winner!r} OR confirm the declared "
                                f"{declared!r} is intentional."
                            ),
                        )
                    )
                block_passed = False

            # 2. Dispersion check (fires regardless of disagreement —
            #    high dispersion is a separate signal).
            if dispersion > self._dispersion_threshold:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="BERT_ENSEMBLE_DISPERSION_HIGH",
                            message=(
                                f"Block {block_id!r} BERT ensemble "
                                f"dispersion={dispersion:.3f} exceeds "
                                f"threshold {self._dispersion_threshold:.3f} "
                                f"(declared bloom_level={declared!r}, "
                                f"winner={winner!r})."
                            ),
                            location=block_id,
                            suggestion=(
                                "High dispersion signals an unstable "
                                "ensemble consensus. Re-roll the block "
                                "with clearer Bloom-verb anchoring in the "
                                "statement / key_claims surface."
                            ),
                        )
                    )
                block_passed = False

            # Emit one capture per audited block, after both
            # disagreement + dispersion checks have run. The failure
            # code is the most-relevant of the two (disagreement
            # dominates dispersion when both fire — disagreement
            # implies the rewrite tier should re-roll the whole block,
            # whereas dispersion alone is a softer signal).
            if not block_passed:
                if (
                    winner != "unknown"
                    and winner != declared
                    and winner_score >= self._confidence_floor
                ):
                    fail_code = "BERT_ENSEMBLE_DISAGREEMENT"
                else:
                    fail_code = "BERT_ENSEMBLE_DISPERSION_HIGH"
            else:
                fail_code = None
            _emit_decision(
                capture,
                block_id=block_id,
                passed=block_passed,
                code=fail_code,
                declared_level=declared,
                winner_level=winner,
                winner_score=winner_score,
                dispersion=dispersion,
                dispersion_threshold=self._dispersion_threshold,
                confidence_floor=self._confidence_floor,
                member_votes=per_member,
                members_loaded=members_loaded,
            )

            if block_passed:
                passed_count += 1

        # Score = pass rate over audited blocks. ``passed`` is the
        # legacy convention (no critical issues) — every issue this
        # validator emits is warning-severity by construction, so
        # ``passed`` stays True even when ``action="regenerate"`` is
        # set on the result. The Phase 3 router consumes ``action``
        # to decide whether to retry; the pass/fail bit here is for
        # backward-compat with non-router consumers.
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        # Action: regenerate when any block flagged; None otherwise.
        action: Optional[str] = "regenerate" if issues else None

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=score,
            issues=issues,
            action=action,
        )


__all__ = [
    "BloomClassifierDisagreementValidator",
    "_DISAGREEMENT_CONFIDENCE_FLOOR",
    "_DISPERSION_THRESHOLD",
]
