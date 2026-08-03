"""Stage 4 v1 — cross-BERT reranker.

This module implements the rule-based arbiter described in
``architecture.md`` §3.2 (lines 275-285). It is **pure function** over
a :class:`CouncilState` + a list of :class:`RegionCandidate`s; no torch
imports, no model loads, no IO.

Public API
----------

:func:`arbitrate` consumes the council state from
:func:`semantik_structure.council.orchestrator.run_council` and emits one
:class:`RoutingDecision` per input region. Order matches the input.

Stage 4 / Stage 5 seam
----------------------

Stage 4 v1 emits one :class:`RoutingDecision` per detected region
candidate (Table or Math). It deliberately does **NOT** emit decisions
for the per-span Structure outputs that cover prose — that is Stage
5's job (Structure-graph candidate generation per ``architecture.md``
§3.3, lines 93-95). Consumers wanting per-span structure decisions
should read ``state.outputs['structure'].signals`` directly, where
each signal's ``region_id`` is the FeatureBlock index.

Current arbitration rules
-------------------------

1. **Candidate evidence gates specialist routing.** A region only routes to
   its table or math track when the current evidence confirms it:

   * Tables: deterministic grid/border evidence is load-bearing. For weak
     candidates, Structure's binary ``table_region`` head is aggregated over
     the candidate's ``member_block_indices`` as secondary evidence.
   * Math: deterministic thresholds over
     :attr:`MathCandidate.glyph_density_features`; see
     ``region_detection.py`` for the canonical feature population site.

2. **Math wins matrix conflicts.** When a math region overlaps a table
   region by ≥0.5 along BOTH axes, math wins; the table is demoted.

3. **Ambiguous routing → prose.** Below-threshold confidences (top-1
   < 0.5) still emit a label but flag ``low_confidence`` so the soft
   reranker downstream can weigh them.

4. **Image-block demotion.** Structure's ``is_image_block`` head is a
   secondary signal. Prose-routed blocks above its threshold receive the
   ``image_block_demoted`` flag so structure-graph Pass 3b can preserve them
   as figures; deterministic ``ImageCandidate`` objects route to figures
   directly.

5. **Semantic role override.** When Structure's ``structural_role``
   and Semantic's ``doc_role`` disagree on a prose-routed region and
   Semantic is more confident, Semantic wins.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from .types import (
    BertOutput,
    CouncilState,
    ImageCandidate,
    MathCandidate,
    RegionCandidate,
    RoutingDecision,
    TableCandidate,
    TypedSignal,
)


# ---------------------------------------------------------------------------
# Tunable thresholds for the current deterministic/BERT evidence blend.
# ---------------------------------------------------------------------------

# Structure's table_region head: aggregate over member spans. A
# candidate is "confirmed" if the mean P(table_region=1) over its
# members exceeds this threshold.
_TABLE_CONFIRM_THRESHOLD = 0.5

# Structural-confidence flag (default ON). When on, a pdfplumber table
# that is STRUCTURALLY confident (a real grid: >=2 rows AND >=2 cols, OR
# explicit ruling-line/bordered evidence) is confirmed REGARDLESS of the
# Structure ``table_region`` BERT head — the deterministic geometric
# detector is load-bearing and the BERT becomes a SECONDARY signal that
# can only confirm a structurally-weak candidate (it can no longer drop a
# structurally-strong one). This fixes the 0-tables regression where the
# BERT head fires <0.5 on every real pdfplumber table and drops all of
# them. Mirrors the front-matter detector being load-bearing vs the 70B
# reviewer being secondary. Default-ON parse semantics (explicit falsey →
# off; unset / truthy / garbage → on) match
# ``deterministic_structure.resolve_structure_clean_mode``.
_TABLE_STRUCTURAL_CONFIDENCE_ENV = "SEMANTIK_TABLE_STRUCTURAL_CONFIRM"
_FALSEY = {"0", "false", "no", "off"}

# A structurally-confident grid needs at least this many rows AND cols.
_TABLE_MIN_ROWS = 2
_TABLE_MIN_COLS = 2


def resolve_table_structural_confirm() -> bool:
    """Whether a structurally-confident pdfplumber table is load-bearing.

    Reads ``SEMANTIK_TABLE_STRUCTURAL_CONFIRM``, DEFAULT ON
    (parse-with-fallback). Only an explicit falsey value (``0`` / ``false``
    / ``no`` / ``off``, case-insensitive) reverts to the legacy
    BERT-gated-only behaviour; unset / truthy / garbage keeps it on (the
    0-tables regression is a clear defect). Mirrors
    ``deterministic_structure.resolve_structure_clean_mode``.
    """
    raw = os.environ.get(_TABLE_STRUCTURAL_CONFIDENCE_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


def _structurally_confident_table(cand: TableCandidate) -> bool:
    """Deterministic structural-confidence guard for a pdfplumber table.

    A candidate is structurally confident when it is a real grid — at
    least :data:`_TABLE_MIN_ROWS` rows AND :data:`_TABLE_MIN_COLS`
    columns — OR pdfplumber found explicit ruling-line / bordered
    evidence (a high-confidence positive even when one axis is thin, e.g.
    a 2-row key/value box). The guard deliberately REJECTS page-layout
    false-positives (a 1×N or borderless single-cell "table") so the
    structural arm never over-confirms.

    Reads the row/col counts pdfplumber forwarded onto the candidate at
    detection time (``region_detection.detect_table_region_candidates``
    stamps ``evidence_features['n_rows'/'n_cols'/'bordered']`` and the
    ``bordered`` field).
    """
    if not isinstance(cand, TableCandidate):
        return False
    evidence = cand.evidence_features or {}
    try:
        n_rows = int(evidence.get("n_rows", 0) or 0)
        n_cols = int(evidence.get("n_cols", 0) or 0)
    except (TypeError, ValueError):
        n_rows = n_cols = 0
    if n_rows >= _TABLE_MIN_ROWS and n_cols >= _TABLE_MIN_COLS:
        return True
    # Ruling-line / bordered evidence is itself a strong positive.
    bordered = bool(getattr(cand, "bordered", False)
                    or evidence.get("bordered", False))
    if bordered and n_rows >= _TABLE_MIN_ROWS:
        return True
    return False

@dataclass(frozen=True)
class MathThresholds:
    """Tunable thresholds for deterministic math-candidate confirmation.

    The deterministic stand-in confirms a :class:`MathCandidate` if
    EITHER ``math_font_frac >= min_math_font_frac`` OR
    ``cid_token_count >= min_cid_token_count`` on its
    ``glyph_density_features`` dict (populated at Stage 2 by
    :func:`~semantik_structure.region_detection.detect_math_region_candidates`,
    see ``region_detection.py`` lines 432-475).

    Defaults preserve the established behavior and are intentionally
    conservative. The dataclass provides one explicit seam for evaluation and
    ablation without implying a future model dependency.
    """

    min_math_font_frac: float = 0.5
    min_cid_token_count: int = 3


# Image-block demotion threshold (Structure.is_image_block head, top-1
# probability of class "image_block").
_IMAGE_BLOCK_CONFIRM_THRESHOLD = 0.5

# Generic top-1 confidence below which we tag "low_confidence".
_LOW_CONFIDENCE_THRESHOLD = 0.5

# Math-vs-table overlap threshold (both axes, fractions of the smaller
# box) — used by the "math wins matrix conflicts" rule.
_MATH_TABLE_OVERLAP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers — kept private; arbiter is the only public entry point.
# ---------------------------------------------------------------------------


def _top1(signal: TypedSignal | None) -> tuple[str | None, float]:
    """Return ``(top_label, top_confidence)`` for a TypedSignal.

    Returns ``(None, 0.0)`` if the signal is missing or has no labels.
    """
    if signal is None:
        return (None, 0.0)
    if not signal.top_k_labels or not signal.top_k_confidences:
        return (None, 0.0)
    return (signal.top_k_labels[0], float(signal.top_k_confidences[0]))


def _get_signal(
    state: CouncilState,
    bert_name: str,
    head_name: str,
    region_id: int,
) -> TypedSignal | None:
    """Pull a single TypedSignal out of CouncilState by (bert, head, region)."""
    out: BertOutput | None = state.outputs.get(bert_name)
    if out is None:
        return None
    for sig in out.signals:
        if sig.head_name == head_name and sig.region_id == region_id:
            return sig
    return None


def _get_signals_for_head(
    state: CouncilState,
    bert_name: str,
    head_name: str,
) -> list[TypedSignal]:
    """All TypedSignals for one (bert, head) pair, in BertOutput order."""
    out: BertOutput | None = state.outputs.get(bert_name)
    if out is None:
        return []
    return [s for s in out.signals if s.head_name == head_name]


def _bbox_axis_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return (h_overlap, v_overlap) as fractions of the smaller box on
    each axis. Values in [0.0, 1.0].

    Uses (x0, y0, x1, y1) PDF-point ordering (top-left origin).
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    aw = max(0.0, ax1 - ax0)
    ah = max(0.0, ay1 - ay0)
    bw = max(0.0, bx1 - bx0)
    bh = max(0.0, by1 - by0)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return (0.0, 0.0)
    h_over = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    v_over = max(0.0, min(ay1, by1) - max(ay0, by0))
    h_frac = h_over / max(1e-9, min(aw, bw))
    v_frac = v_over / max(1e-9, min(ah, bh))
    return (h_frac, v_frac)


def _table_confirmed(
    cand: TableCandidate,
    state: CouncilState,
    *,
    threshold: float = _TABLE_CONFIRM_THRESHOLD,
    structural_confirm: bool | None = None,
) -> tuple[bool, bool]:
    """Aggregate Structure's ``table_region`` over the candidate's member spans.

    Returns ``(confirmed, signal_present)``. ``signal_present`` is False
    iff Structure's table_region signals are entirely absent for this
    candidate's members (orchestrator partial-failure path).

    Load-bearing geometric detector (default ON, see
    :func:`resolve_table_structural_confirm`): a STRUCTURALLY-CONFIDENT
    pdfplumber table (:func:`_structurally_confident_table`) is confirmed
    EITHER way — ``confirmed=True`` is returned regardless of the BERT
    head, and ``signal_present`` is reported True so the arbiter routes it
    to the table track even when the head is absent or fires low. The BERT
    can then only ADD confirmations (a structurally-weak candidate it
    rates >= threshold) — it can never DROP a structurally-strong one.
    Pass ``structural_confirm=False`` to force the legacy BERT-gated-only
    behaviour (used by the byte-stability test).
    """
    if structural_confirm is None:
        structural_confirm = resolve_table_structural_confirm()

    structurally_strong = bool(
        structural_confirm and _structurally_confident_table(cand)
    )

    member_idx = list(cand.member_block_indices or [])
    if not member_idx:
        # No members → can't aggregate the BERT. A structurally-confident
        # grid is still load-bearing (the deterministic detector owns the
        # decision); report signal_present=True so the arbiter routes it
        # to the table track. Otherwise treat as "signal missing".
        if structurally_strong:
            return (True, True)
        return (False, False)
    probs: list[float] = []
    for idx in member_idx:
        sig = _get_signal(state, "structure", "table_region", idx)
        if sig is None:
            continue
        label, conf = _top1(sig)
        if label is None:
            continue
        # P(table_region=1) — if top label is the positive class take
        # the confidence; else take 1 - conf (binary head).
        if label == "table_region":
            probs.append(conf)
        else:
            probs.append(1.0 - conf)
    if not probs:
        # BERT silent. The structural guard is the load-bearing signal.
        if structurally_strong:
            return (True, True)
        return (False, False)
    mean_p = sum(probs) / len(probs)
    bert_confirms = mean_p >= threshold
    return (bert_confirms or structurally_strong, True)


def confirm_table_candidates(
    state: CouncilState,
    candidates: Sequence[TableCandidate],
    *,
    threshold: float = _TABLE_CONFIRM_THRESHOLD,
) -> list[TableCandidate]:
    """Return the subset of ``candidates`` confirmed by the trained
    ``table_region`` head.

    Public entry point for callers that want to filter pdfplumber
    TableCandidates by Structure-council agreement before doing
    expensive downstream work (e.g. running GLM-OCR only on confirmed
    bboxes — see :func:`semantik_structure.glm_ocr_enrich.enrich_via_council_gate`).

    Candidates whose Structure signals are missing entirely
    (``signal_present=False``) are NOT included in the result. Callers
    that need the conservative "include if uncertain" behavior should
    iterate themselves over :func:`_table_confirmed` outputs.

    The aggregation rule (mean ``P(table_region=1)`` over the
    candidate's ``member_block_indices`` >= ``threshold``) matches what
    the cross-reranker uses internally; threshold defaults to
    ``_TABLE_CONFIRM_THRESHOLD`` (0.5).
    """
    out: list[TableCandidate] = []
    for cand in candidates:
        confirmed, signal_present = _table_confirmed(
            cand, state, threshold=threshold,
        )
        if confirmed and signal_present:
            out.append(cand)
    return out


def _math_candidate_confirmed(
    cand: MathCandidate,
    thresholds: MathThresholds,
) -> bool:
    """Confirm a math candidate from deterministic glyph-density evidence.

    Uses
    ``glyph_density_features`` (computed at Stage 2 by
    :func:`~semantik_structure.region_detection.detect_math_region_candidates`;
    see ``region_detection.py`` lines 432-475 for the field's
    population site on :class:`MathCandidate`).

    Defaults are carried by :class:`MathThresholds`:
        * math_font_frac >= ``thresholds.min_math_font_frac`` (default 0.5)  OR
        * cid_token_count >= ``thresholds.min_cid_token_count`` (default 3)
    """
    feats = cand.glyph_density_features or {}
    math_font_frac = float(feats.get("math_font_frac", 0.0) or 0.0)
    cid_tokens = float(feats.get("cid_token_count", 0.0) or 0.0)
    if math_font_frac >= thresholds.min_math_font_frac:
        return True
    if cid_tokens >= thresholds.min_cid_token_count:
        return True
    return False


def _aggregate_structure_role(
    cand: RegionCandidate,
    state: CouncilState,
) -> tuple[str | None, float]:
    """Pick the highest-confidence ``structural_role`` top-1 over the
    candidate's member spans. Returns ``(label, confidence)`` or
    ``(None, 0.0)`` if no signal is available.
    """
    member_idx = list(cand.member_block_indices or [])
    best_label: str | None = None
    best_conf = 0.0
    for idx in member_idx:
        sig = _get_signal(state, "structure", "structural_role", idx)
        label, conf = _top1(sig)
        if label is None:
            continue
        if conf > best_conf:
            best_conf = conf
            best_label = label
    return (best_label, best_conf)


def _aggregate_semantic_role(
    cand: RegionCandidate,
    state: CouncilState,
) -> tuple[str | None, float]:
    """Same shape as :func:`_aggregate_structure_role` but for Semantic's
    ``doc_role`` head. Returns ``(label, confidence)`` or ``(None, 0.0)``.
    """
    member_idx = list(cand.member_block_indices or [])
    best_label: str | None = None
    best_conf = 0.0
    for idx in member_idx:
        sig = _get_signal(state, "semantic", "doc_role", idx)
        label, conf = _top1(sig)
        if label is None:
            continue
        if conf > best_conf:
            best_conf = conf
            best_label = label
    return (best_label, best_conf)


def _aggregate_image_block(
    cand: RegionCandidate,
    state: CouncilState,
) -> float:
    """Mean P(image_block=1) across the candidate's member spans, per
    Structure's ``is_image_block`` head. Returns 0.0 if the head was
    suppressed (pre-Phase-3f checkpoint) or member spans missing.
    """
    member_idx = list(cand.member_block_indices or [])
    if not member_idx:
        return 0.0
    probs: list[float] = []
    for idx in member_idx:
        sig = _get_signal(state, "structure", "is_image_block", idx)
        if sig is None:
            continue
        label, conf = _top1(sig)
        if label is None:
            continue
        if label == "image_block":
            probs.append(conf)
        else:
            probs.append(1.0 - conf)
    if not probs:
        return 0.0
    return sum(probs) / len(probs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def arbitrate(
    state: CouncilState,
    regions: list[RegionCandidate],
    *,
    math_thresholds: MathThresholds | None = None,
) -> list[RoutingDecision]:
    """Apply §3.2 arbitration rules over council outputs.

    Returns one :class:`RoutingDecision` per input region; order matches
    ``regions``. Pure function — no IO, no torch.

    Parameters
    ----------
    math_thresholds:
        Optional override for deterministic math-candidate thresholds.
        ``None`` (the default) uses ``MathThresholds()``, whose defaults
        preserve the established conservative behavior. Pass a custom instance to
        force-demote all math regions (useful for ablations and the
        ``math_detector_stand_in_demoted``-flag smoke test).
    """
    if math_thresholds is None:
        math_thresholds = MathThresholds()

    decisions: list[RoutingDecision] = []

    # First pass: precompute table/math overlaps so the
    # "math wins matrix conflicts" rule has access to neighbour info
    # without N^2 scanning inside the per-region loop.
    table_indices = [
        i for i, r in enumerate(regions) if isinstance(r, TableCandidate)
    ]
    math_indices = [
        i for i, r in enumerate(regions) if isinstance(r, MathCandidate)
    ]

    # Map: table region_index -> True if any math region overlaps it.
    table_overridden_by_math: set[int] = set()
    for ti in table_indices:
        t = regions[ti]
        for mi in math_indices:
            m = regions[mi]
            # Overlap only meaningful for regions on the same page set.
            if not (set(t.pages) & set(m.pages)):
                continue
            h_frac, v_frac = _bbox_axis_overlap(t.bbox, m.bbox)
            if (h_frac >= _MATH_TABLE_OVERLAP_THRESHOLD
                    and v_frac >= _MATH_TABLE_OVERLAP_THRESHOLD):
                table_overridden_by_math.add(ti)
                break

    for region_id, region in enumerate(regions):
        flags: list[str] = []
        route: str
        final_role: str | None = None

        if isinstance(region, TableCandidate):
            # Rule 2: math wins matrix conflicts.
            if region_id in table_overridden_by_math:
                flags.append("math_matrix_override")
                # Demoted out of the table track. The overlapping math
                # region (handled in its own iteration) carries the math
                # route. This region falls back to prose.
                route = "prose"
                # Try to attach a Structure-derived role for the demoted
                # region so downstream consumers still see something.
                role_label, role_conf = _aggregate_structure_role(
                    region, state,
                )
                if role_label is None:
                    flags.append("no_signals")
                else:
                    final_role = role_label
                    if role_conf < _LOW_CONFIDENCE_THRESHOLD:
                        flags.append("low_confidence")
            else:
                # Rule 1: detector gates specialist.
                confirmed, signal_present = _table_confirmed(region, state)
                if not signal_present:
                    flags.append("detector_signal_missing")
                    route = "prose"
                elif not confirmed:
                    flags.append("detector_demoted")
                    route = "prose"
                    role_label, role_conf = _aggregate_structure_role(
                        region, state,
                    )
                    if role_label is None:
                        flags.append("no_signals")
                    else:
                        final_role = role_label
                        if role_conf < _LOW_CONFIDENCE_THRESHOLD:
                            flags.append("low_confidence")
                else:
                    route = "table"
                    final_role = "table"

        elif isinstance(region, ImageCandidate):
            # Part F — DETERMINISTIC figure route. An ImageCandidate is a
            # real PDF IMAGE page-object the deterministic extractor found;
            # the geometric detector is load-bearing, so we route to the
            # figure track unconditionally. Structure's ``is_image_block``
            # head (aggregated below for prose-routed regions) is NOT the
            # trigger — it is only a secondary strength flag. Mirrors the
            # table structural-confidence posture and the front-matter
            # detector being load-bearing vs the 70B reviewer.
            route = "figure"
            final_role = "figure"
            img_p = _aggregate_image_block(region, state)
            if img_p >= _IMAGE_BLOCK_CONFIRM_THRESHOLD:
                # Secondary confirmation: the BERT agrees. Recorded for
                # audit; it does not change the (already deterministic)
                # route.
                flags.append("image_block_confirmed")

        elif isinstance(region, MathCandidate):
            # Rule 1 (math variant): stand-in detector gates specialist.
            if _math_candidate_confirmed(region, math_thresholds):
                route = "math"
                final_role = "math"
            else:
                flags.append("math_detector_stand_in_demoted")
                route = "prose"
                role_label, role_conf = _aggregate_structure_role(
                    region, state,
                )
                if role_label is None:
                    flags.append("no_signals")
                else:
                    final_role = role_label
                    if role_conf < _LOW_CONFIDENCE_THRESHOLD:
                        flags.append("low_confidence")

        else:
            # Future region kinds (flat-text, image, etc.) — fall through
            # to the prose track using Structure's structural_role as
            # the default label.
            route = "prose"
            role_label, role_conf = _aggregate_structure_role(region, state)
            if role_label is None:
                flags.append("no_signals")
            else:
                final_role = role_label
                if role_conf < _LOW_CONFIDENCE_THRESHOLD:
                    flags.append("low_confidence")

        # Rule 4: image-block demotion. Only meaningful for prose-routed
        # regions (the table/math tracks already have a final destination).
        if route == "prose":
            img_p = _aggregate_image_block(region, state)
            if img_p >= _IMAGE_BLOCK_CONFIRM_THRESHOLD:
                if "image_block_demoted" not in flags:
                    flags.append("image_block_demoted")

        # Rule 5: Semantic role override. Only for prose-routed regions
        # where both Structure and Semantic emitted a top-1 and they
        # disagree (case-insensitive label match). Semantic wins iff its
        # top-1 confidence strictly exceeds Structure's.
        if route == "prose":
            struct_label, struct_conf = _aggregate_structure_role(
                region, state,
            )
            sem_label, sem_conf = _aggregate_semantic_role(region, state)
            if (sem_label is not None and struct_label is not None
                    and sem_label != struct_label
                    and sem_conf > struct_conf):
                final_role = sem_label
                flags.append("semantic_override")
                # Recompute low_confidence against Semantic's score.
                if "low_confidence" in flags and sem_conf >= _LOW_CONFIDENCE_THRESHOLD:
                    flags = [f for f in flags if f != "low_confidence"]
                elif sem_conf < _LOW_CONFIDENCE_THRESHOLD and "low_confidence" not in flags:
                    flags.append("low_confidence")

        decisions.append(RoutingDecision(
            region_id=region_id,
            route=route,  # type: ignore[arg-type]
            final_role=final_role,
            flags=tuple(flags),
        ))

    return decisions


__all__ = [
    "MathThresholds",
    "arbitrate",
    "confirm_table_candidates",
    "resolve_table_structural_confirm",
]
