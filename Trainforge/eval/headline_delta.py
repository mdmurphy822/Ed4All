"""Wave 103 - Headline-delta computation + ED4ALL-Bench branding.

Pulls the four numbers any procurement reader actually wants out of
the ablation report:

* hallucination_reduction_pct: ``(base - adapter+rag) / base`` measured
  in hallucination_rate.
* source_grounded_lift_x: ratio of source_match between adapter+rag
  and base.
* accuracy_lift_x: ratio of accuracy between adapter+rag and base.
* headline_sentence: a single auto-rendered marketing line that
  ships at the top of the HF README.

The marketing sentence template (locked):

    **ED4ALL-Bench v1.0**: Adapter + ED4ALL RAG reduces hallucinations
    by **{X%}** ({base} -> {final}) and increases source-grounded
    answers **{N×}** ({base} -> {final}). Run on
    ``ed4all-bench/<course_slug>`` (held-out split SHA ``<hash>``,
    scoring commit ``<sha>``).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


_HEADLINE_TEMPLATE = (
    "**ED4ALL-Bench v1.0**: Adapter + ED4ALL RAG reduces hallucinations "
    "by **{hallucination_pct}** ({base_hallucination} → "
    "{final_hallucination}) and increases source-grounded answers "
    "**{lift_x}×** ({base_source} → {final_source}). "
    "Run on `ed4all-bench/{course_slug}` "
    "(held-out split SHA `{holdout_hash}`, scoring commit `{scoring_commit}`)."
)


def _row_by_setup(
    headline_table, target: str,
) -> Optional[Dict[str, Any]]:
    for row in headline_table:
        if str(row.get("setup", "")).lower() == target.lower():
            return row
    return None


def _safe(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _fmt_3dp(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_lift(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def compute_headline_delta(
    ablation_report: Dict[str, Any],
    *,
    course_slug: str = "<course_slug>",
    holdout_hash: str = "<holdout_hash>",
    scoring_commit: str = "<scoring_commit>",
) -> Dict[str, Any]:
    """Compute the four headline numbers + the rendered sentence.

    Args:
        ablation_report: Parsed ablation_report.json dict.
        course_slug: LibV2 course slug, used as the dataset suffix in
            the headline sentence (``ed4all-bench/<slug>``).
        holdout_hash: Provenance pin for the held-out split (typically
            ``model_card.provenance.holdout_graph_hash``).
        scoring_commit: Provenance pin for the scoring commit
            (``model_card.eval_scores.scoring_commit``).

    Returns:
        Dict with keys
        ``hallucination_reduction_pct``,
        ``source_grounded_lift_x``,
        ``accuracy_lift_x``,
        ``headline_sentence``.
        Numeric values may be ``None`` when an input row is missing.
    """
    headline_table = ablation_report.get("headline_table") or []
    base_row = _row_by_setup(headline_table, "base")
    final_row = _row_by_setup(headline_table, "adapter+rag")

    if base_row is None or final_row is None:
        # Without both rows we cannot render anything meaningful;
        # return None values so the caller can decide whether to
        # render a fallback sentence.
        return {
            "hallucination_reduction_pct": None,
            "source_grounded_lift_x": None,
            "accuracy_lift_x": None,
            "headline_sentence": (
                "**ED4ALL-Bench v1.0**: ablation table incomplete; "
                "headline numbers cannot be rendered."
            ),
        }

    base_hallucination = _safe(base_row.get("hallucination_rate"))
    final_hallucination = _safe(final_row.get("hallucination_rate"))
    base_source = _safe(base_row.get("source_match"))
    final_source = _safe(final_row.get("source_match"))
    base_acc = _safe(base_row.get("accuracy"))
    final_acc = _safe(final_row.get("accuracy"))

    hallucination_reduction_pct: Optional[float] = None
    if base_hallucination is not None and final_hallucination is not None and base_hallucination > 0:
        hallucination_reduction_pct = (
            (base_hallucination - final_hallucination) / base_hallucination
        )

    source_grounded_lift_x: Optional[float] = None
    if base_source is not None and final_source is not None and base_source > 0:
        source_grounded_lift_x = final_source / base_source

    accuracy_lift_x: Optional[float] = None
    if base_acc is not None and final_acc is not None and base_acc > 0:
        accuracy_lift_x = final_acc / base_acc

    headline_sentence = _HEADLINE_TEMPLATE.format(
        hallucination_pct=_fmt_pct(hallucination_reduction_pct),
        base_hallucination=_fmt_3dp(base_hallucination),
        final_hallucination=_fmt_3dp(final_hallucination),
        lift_x=_fmt_lift(source_grounded_lift_x),
        base_source=_fmt_3dp(base_source),
        final_source=_fmt_3dp(final_source),
        course_slug=course_slug,
        holdout_hash=holdout_hash,
        scoring_commit=scoring_commit,
    )

    return {
        "hallucination_reduction_pct": (
            round(hallucination_reduction_pct, 4)
            if hallucination_reduction_pct is not None else None
        ),
        "source_grounded_lift_x": (
            round(source_grounded_lift_x, 4)
            if source_grounded_lift_x is not None else None
        ),
        "accuracy_lift_x": (
            round(accuracy_lift_x, 4)
            if accuracy_lift_x is not None else None
        ),
        "headline_sentence": headline_sentence,
    }


__all__ = ["compute_headline_delta"]
