"""IB6.1 — per-block eight-dimension quality-rubric scoring pass.

The scoring CAPSTONE. For every :class:`Courseforge.scripts.blocks.Block`, this
validator COMPOSES already-computed validator signals into the eight anchored
0-3 dimension scores (framework §6.2-6.3) and writes them into the IB2
``Block.quality_rubric`` container. It does NOT build a new scoring engine and
it does NOT re-run NLI / embeddings — it reads the upstream ``GateResult``s
threaded in via ``inputs`` (the inter/post-rewrite validation chain output).

The eight dimensions and their reused signals (§2 reuse table)::

    1 Alignment       objective_assessment_similarity cosine + IB3 verb-triple
                      GateResult  (assessment Bloom < objective Bloom → cap 1)
    2 Cognitive-load  IB6.4 BlockCognitiveLoadValidator overflow signal
    3 Multimedia      IB4 time-based-media a11y GateResult (B04 only)
    4 Retrieval       co-located self_check presence (+ IB7 spacing, advisory)
    5 Feedback        IB6.3 interaction-feedback gate + distractor cluster
                      (interaction-without-feedback → cap 1)
    6 Engagement      IB4 UDL engagement_affordance + interaction presence
    7 Accessibility   IB4 per-block WCAG GateResult + UDL n_representations
                      (Accessibility == 0 → block FAILS — applied at rollup)
    8 Coherence       block_prose_entailment (NLI) + IB6.2 slot-presence
                      (interaction-without-feedback → cap 1, with Feedback)

Dims 1, 2, 7, 8 are the load-bearing CORE (apply to every block; the rollup's
required-dim-minimum floor is over these). The others apply only when the
block carries the relevant slot/shape (``applicable_dims``).

Three cap hard-gates are applied HERE at scoring time:

* assessment Bloom < objective Bloom → Alignment capped at 1 (reuse IB3
  ``alignment_cap_at_1``);
* interaction with no feedback slot → Feedback AND Coherence capped at 1
  (reuse the IB6.2 / IB6.3 ``caps_dims_by_block`` directive).

(The third hard gate — Accessibility == 0 fails the block outright — is a
rollup concern, applied in :mod:`lib.aggregators.block_quality_rollup`.)

Default-OFF: scoring runs only when ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is truthy.
When off → no rubric field written, ``GateResult(passed=True, action=None)``
with a ``RUBRIC_DISABLED`` info issue (byte-stable). Warning-day-1.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from Courseforge.scripts.blocks import (
    CORE_QUALITY_DIMENSIONS,
    QUALITY_DIMENSIONS,
    QualityScore,
)
from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50

# Dimension index by name for stable iteration / messages.
(
    _ALIGNMENT,
    _COGNITIVE_LOAD,
    _MULTIMEDIA,
    _RETRIEVAL,
    _FEEDBACK,
    _ENGAGEMENT,
    _ACCESSIBILITY,
    _COHERENCE,
) = QUALITY_DIMENSIONS


class BlockQualityRubricValidator:
    """IB6.1 — compose upstream signals into the 8-dim 0-3 block rubric."""

    name = "block_quality_rubric"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import (
            block_attr,
            block_quality_rubric_enabled,
            block_type_of,
            framework_block_of,
            is_interactive_block,
        )

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("rubric_enabled")
        if enabled is None:
            enabled = block_quality_rubric_enabled()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="RUBRIC_DISABLED",
                        message=(
                            "ED4ALL_BLOCK_QUALITY_RUBRIC unset — eight-dimension "
                            "block scoring skipped (no rubric field written, "
                            "byte-stable)."
                        ),
                    )
                ],
                action=None,
                metadata={"block_quality_rubric": {"enabled": False}},
            )

        blocks = list(inputs.get("blocks") or [])
        # Threaded upstream signals (validator_name -> GateResult OR a
        # pre-extracted per-block signal map). Read-only; never recomputed.
        signals = inputs.get("gate_results_by_block") or {}

        # Pre-index the per-block signal dicts from each upstream validator's
        # GateResult.metadata. Each block_id resolves to a merged signal dict.
        per_block_signals = _index_signals(signals)

        scored_blocks: List[Any] = []
        scored_count = 0
        block_pass_count = 0
        per_block_report: Dict[str, Dict[str, Any]] = {}

        import dataclasses

        for idx, block in enumerate(blocks):
            bt = block_type_of(block)
            bcode = framework_block_of(block)
            block_id = str(block_attr(block, "block_id") or f"block-{idx}")
            if not bt or bcode is None:
                # chrome / unknown — non-pedagogical, no rubric. Keep as-is.
                scored_blocks.append(block)
                continue

            sig = per_block_signals.get(block_id, {})
            scores, applicable, caps_applied = _score_block(
                block=block,
                block_id=block_id,
                framework_block=bcode,
                is_interactive=is_interactive_block(block),
                signals=sig,
            )

            rubric: List[QualityScore] = []
            for dim in QUALITY_DIMENSIONS:
                is_applicable = dim in applicable
                rubric.append(
                    QualityScore(
                        dimension=dim,
                        score=scores.get(dim) if is_applicable else None,
                        applicable=is_applicable,
                        rationale=_dim_rationale(dim, scores.get(dim), caps_applied),
                    )
                )

            # Block pass = mean(applicable) >= 2.0 AND every CORE dim >= 2.0.
            applicable_scores = [
                scores[d] for d in applicable if scores.get(d) is not None
            ]
            mean = (
                sum(applicable_scores) / len(applicable_scores)
                if applicable_scores
                else 0.0
            )
            core_ok = all(
                scores.get(d) is not None and scores[d] >= 2
                for d in CORE_QUALITY_DIMENSIONS
            )
            block_pass = mean >= 2.0 and core_ok
            if block_pass:
                block_pass_count += 1
            scored_count += 1

            weakest_dim = _weakest_dim(scores, applicable)

            per_block_report[block_id] = {
                "framework_block": bcode,
                "page_id": str(block_attr(block, "page_id") or ""),
                "scores": {d: scores.get(d) for d in applicable},
                "applicable_dims": sorted(applicable),
                "mean": round(mean, 4),
                "core_ok": core_ok,
                "block_pass": block_pass,
                "weakest_dim": weakest_dim,
                "caps_applied": sorted(caps_applied),
            }

            scored_blocks.append(
                dataclasses.replace(block, quality_rubric=tuple(rubric))
            )

        issues = self._build_issues(per_block_report)
        self._emit_decision(
            capture,
            scored=scored_count,
            passed=block_pass_count,
            report=per_block_report,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1: never blocks
            score=(round(block_pass_count / scored_count, 4) if scored_count else 1.0),
            issues=issues,
            action=None,
            metadata={
                "block_quality_rubric": {
                    "enabled": True,
                    "blocks_scored": scored_count,
                    "blocks_pass": block_pass_count,
                    "per_block": per_block_report,
                    # Threaded so a downstream caller (the IB6.6 rollup) can
                    # reuse the scored Block instances without re-parsing.
                    "scored_block_ids": list(per_block_report.keys()),
                }
            },
        )

    @staticmethod
    def _build_issues(report: Dict[str, Dict[str, Any]]) -> List[GateIssue]:
        issues: List[GateIssue] = []
        for block_id, row in report.items():
            if row.get("block_pass"):
                continue
            if len(issues) >= _ISSUE_LIST_CAP:
                break
            issues.append(
                GateIssue(
                    severity="warning",
                    code="BLOCK_QUALITY_BELOW_FLOOR",
                    message=(
                        f"Block {block_id!r} ({row.get('framework_block')}) "
                        f"fails the block rubric floor (mean="
                        f"{row.get('mean')}, core_ok={row.get('core_ok')}); "
                        f"weakest dimension: {row.get('weakest_dim')}."
                    ),
                    location=block_id,
                    suggestion=(
                        "Lift the weakest dimension to >=2 (every core "
                        "dimension must clear 2.0 and the applicable-dim mean "
                        ">=2.0)."
                    ),
                )
            )
        return issues

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        scored: int,
        passed: int,
        report: Dict[str, Dict[str, Any]],
    ) -> None:
        if capture is None:
            return
        try:
            caps = sum(1 for r in report.values() if r.get("caps_applied"))
            capture.log_decision(
                decision_type="content_selection",
                decision=f"block_rubric_pass={passed}/{scored}",
                rationale=(
                    f"IB6.1 eight-dimension rubric: scored {scored} pedagogical "
                    f"blocks by composing upstream cognitive-load / anatomy / "
                    f"interaction-feedback / alignment / WCAG-UDL / coherence "
                    f"signals (no model reload); {passed} cleared the per-block "
                    f"floor (mean>=2.0 AND core dims>=2.0); {caps} blocks had a "
                    f"hard-gate cap applied (Alignment or Feedback+Coherence)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on block_quality_rubric: %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Signal indexing — collapse upstream GateResult.metadata into per-block dicts
# ---------------------------------------------------------------------------


def _index_signals(signals: Any) -> Dict[str, Dict[str, Any]]:
    """Collapse threaded upstream signals into ``block_id -> merged signal``.

    Accepts either:

    * a mapping ``validator_name -> GateResult`` (the canonical thread shape);
    * a mapping ``validator_name -> metadata-dict``;
    * a mapping ``block_id -> per-block signal dict`` (pre-extracted form, used
      by the IB6.1 unit tests so they can score without a full validator run).

    Returns ``block_id -> {signal_key: value}`` where the signal keys are the
    namespaced metadata each leaf validator emits (``block_cognitive_load`` /
    ``anatomy_slot_presence`` / ``interaction_feedback`` / ``alignment`` /
    ``udl`` / ``wcag`` / ``coherence`` / ``retrieval``).
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(signals, dict):
        return out

    # Form C: block_id -> dict that already carries per-dim signal keys.
    if signals and all(
        isinstance(v, dict)
        and any(
            k in v
            for k in (
                "cognitive_load_overflow",
                "char_count",
                "feedback_caps",
                "feedback_status",
                "wcag_score",
                "wcag_clean",
                "alignment_score",
                "alignment_cosine",
                "verb_triple_status",
                "coherence_score",
                "alignment_cap_at_1",
                "has_colocated_retrieval",
            )
        )
        for v in signals.values()
    ):
        for block_id, v in signals.items():
            out[str(block_id)] = dict(v)
        return out

    # Form A/B: validator_name -> GateResult|metadata. Walk each validator's
    # metadata, which carries a per_block / caps_dims_by_block map.
    for _vname, result in signals.items():
        meta = getattr(result, "metadata", None)
        if meta is None and isinstance(result, dict):
            meta = result
        if not isinstance(meta, dict):
            continue
        for _ns, ns_payload in meta.items():
            if not isinstance(ns_payload, dict):
                continue
            per_block = ns_payload.get("per_block")
            if isinstance(per_block, dict):
                for block_id, row in per_block.items():
                    if isinstance(row, dict):
                        out.setdefault(str(block_id), {}).update(
                            {f"{_ns}.{k}": val for k, val in row.items()}
                        )
            caps = ns_payload.get("caps_dims_by_block")
            if isinstance(caps, dict):
                for block_id, dims in caps.items():
                    bucket = out.setdefault(str(block_id), {})
                    existing = set(bucket.get("caps_dims", []))
                    for d in dims:
                        existing.add(d)
                    bucket["caps_dims"] = sorted(existing)
    return out


# ---------------------------------------------------------------------------
# Per-block scoring — the §2 reuse table, opinionated anchors documented inline
# ---------------------------------------------------------------------------


def _score_block(
    *,
    block: Any,
    block_id: str,
    framework_block: str,
    is_interactive: bool,
    signals: Dict[str, Any],
) -> Tuple[Dict[str, int], set, set]:
    """Return ``(scores, applicable_dims, caps_applied)`` for one block.

    ``signals`` is the merged per-block signal dict from :func:`_index_signals`.
    Each dimension is scored on the anchored 0-3 scale; missing signals degrade
    to a conservative-but-not-failing default so a thin-signal block scores 2
    (warning-day-1 must not manufacture failures from absent upstream data).
    """
    scores: Dict[str, int] = {}
    applicable: set = set()
    caps: set = set()

    caps_dims = set(_first_present(signals, "caps_dims", "anatomy_slot_presence.caps_dims", default=[]) or [])
    # The IB6.2/IB6.3 caps directive may also live under "feedback_caps" (Form C).
    if signals.get("feedback_caps"):
        caps_dims |= {"feedback", "coherence"}
    alignment_cap = bool(
        signals.get("alignment_cap_at_1")
        or signals.get("alignment.alignment_cap_at_1")
    )

    # --- Dim 1: Alignment (CORE) -------------------------------------------
    applicable.add(_ALIGNMENT)
    align_score = _resolve_int(
        signals,
        "alignment_score",
        "alignment.score",
        default=None,
    )
    if align_score is None:
        cosine = _resolve_float(signals, "alignment_cosine", "alignment.cosine", default=None)
        verb_status = _first_present(signals, "verb_triple_status", "alignment.verb_triple_status")
        if cosine is None and verb_status is None:
            align_score = 2  # thin-signal default
        else:
            ok_cos = cosine is None or cosine >= 0.45
            if verb_status == "aligned" and ok_cos:
                align_score = 3
            elif verb_status == "mismatch":
                align_score = 1
            elif ok_cos:
                align_score = 2
            else:
                align_score = 1
    if alignment_cap and align_score > 1:
        align_score = 1
        caps.add("alignment")
    scores[_ALIGNMENT] = _clamp(align_score)

    # --- Dim 2: Cognitive-load fit (CORE) ----------------------------------
    applicable.add(_COGNITIVE_LOAD)
    overflow = _first_present(
        signals, "cognitive_load_overflow", "block_cognitive_load.overflow"
    )
    char_count = _resolve_int(
        signals, "char_count", "block_cognitive_load.char_count", default=None
    )
    if overflow is True:
        cl = 1
    elif char_count is not None:
        if char_count == 0:
            cl = 0
        elif char_count <= 200:
            cl = 3
        elif char_count <= 300:
            cl = 2
        elif char_count <= 400:
            cl = 2
        else:
            cl = 1
    else:
        cl = 2  # thin-signal default
    scores[_COGNITIVE_LOAD] = _clamp(cl)

    # --- Dim 3: Multimedia integrity (B04 only) ----------------------------
    if framework_block == "B04":
        applicable.add(_MULTIMEDIA)
        mm = _resolve_int(signals, "multimedia_score", "wcag.multimedia_score", default=None)
        if mm is None:
            # media_a11y stack count: 0 missing, 4 full (captions/AD/transcript/controls)
            media = _block_field(block, "media_a11y")
            n = len(media) if isinstance(media, (list, tuple)) else 0
            mm = 3 if n >= 3 else (2 if n == 2 else (1 if n == 1 else 0))
        scores[_MULTIMEDIA] = _clamp(mm)

    # --- Dim 4: Retrieval / spacing (content blocks) -----------------------
    # Applies to content-bearing exposition blocks (B03/B05/B06). A co-located
    # retrieval prompt (self_check) earns 3; spacing is advisory until IB7.
    if framework_block in ("B03", "B05", "B06"):
        applicable.add(_RETRIEVAL)
        retr = _resolve_int(signals, "retrieval_score", default=None)
        if retr is None:
            has_retrieval = bool(signals.get("has_colocated_retrieval"))
            retr = 3 if has_retrieval else 1
        scores[_RETRIEVAL] = _clamp(retr)

    # --- Dim 5: Feedback (interactive blocks) ------------------------------
    if is_interactive:
        applicable.add(_FEEDBACK)
        fb_status = _first_present(
            signals, "feedback_status", "interaction_feedback.feedback_status"
        )
        misconception = signals.get("misconception_targeted")
        if "feedback" in caps_dims or fb_status == "missing":
            fb = 1
            caps.add("feedback")
        elif fb_status == "bare":
            fb = 1
        elif fb_status == "thin":
            fb = 2
        elif fb_status == "elaborated":
            fb = 3 if misconception in (True, None) else 2
        else:
            fb = 2  # thin-signal default
        scores[_FEEDBACK] = _clamp(fb)

    # --- Dim 6: Engagement (non-core) --------------------------------------
    eng_aff = _block_field(block, "engagement_affordance")
    if eng_aff or is_interactive:
        applicable.add(_ENGAGEMENT)
        eng = _resolve_int(signals, "engagement_score", default=None)
        if eng is None:
            if eng_aff and is_interactive:
                eng = 3
            elif eng_aff or is_interactive:
                eng = 2
            else:
                eng = 1
        scores[_ENGAGEMENT] = _clamp(eng)

    # --- Dim 7: Accessibility / UDL (CORE) ---------------------------------
    applicable.add(_ACCESSIBILITY)
    wcag = _resolve_int(signals, "wcag_score", "wcag.score", default=None)
    n_repr = _block_field(block, "n_representations")
    n_repr = n_repr if isinstance(n_repr, int) else 0
    wcag_clean = signals.get("wcag_clean")
    if wcag is not None:
        acc = wcag
    elif wcag_clean is False:
        acc = 0
    elif wcag_clean is True:
        acc = 3 if n_repr >= 2 else 2
    else:
        # No per-block WCAG signal threaded — thin-signal default 2 (clean,
        # one representation). Never manufacture an Accessibility==0 from
        # absent data (that hard-gate would fail the block at rollup).
        acc = 3 if n_repr >= 2 else 2
    scores[_ACCESSIBILITY] = _clamp(acc)

    # --- Dim 8: Coherence (CORE) -------------------------------------------
    applicable.add(_COHERENCE)
    coh = _resolve_int(signals, "coherence_score", "coherence.score", default=None)
    entail = _resolve_float(signals, "entailment", "coherence.entailment", default=None)
    if "coherence" in caps_dims:
        coh = 1
        caps.add("coherence")
    elif coh is None:
        if entail is None:
            coh = 2  # thin-signal default
        elif entail >= 0.70:
            coh = 3
        elif entail >= 0.50:
            coh = 2
        else:
            coh = 0
    scores[_COHERENCE] = _clamp(coh)

    return scores, applicable, caps


# ---------------------------------------------------------------------------
# Small resolution helpers
# ---------------------------------------------------------------------------


def _clamp(v: Optional[int]) -> int:
    if v is None:
        return 0
    return max(0, min(3, int(v)))


def _first_present(signals: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in signals and signals[k] is not None:
            return signals[k]
    return default


def _resolve_int(signals: Dict[str, Any], *keys: str, default: Any = None) -> Optional[int]:
    val = _first_present(signals, *keys, default=None)
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(round(val))
    return default


def _resolve_float(signals: Dict[str, Any], *keys: str, default: Any = None) -> Optional[float]:
    val = _first_present(signals, *keys, default=None)
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return default


def _block_field(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _weakest_dim(scores: Dict[str, int], applicable: set) -> Optional[str]:
    """Return the lowest-scoring applicable dimension (localizable defect)."""
    candidates = [(scores[d], d) for d in applicable if scores.get(d) is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (pair[0], QUALITY_DIMENSIONS.index(pair[1])))
    return candidates[0][1]


def _dim_rationale(dim: str, score: Optional[int], caps: set) -> Optional[str]:
    if dim in caps:
        return f"{dim} hard-gate cap applied (capped at 1)"
    if score is None:
        return None
    return f"{dim} scored {score}/3"


__all__ = ["BlockQualityRubricValidator"]
