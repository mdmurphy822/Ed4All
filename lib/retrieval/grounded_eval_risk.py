"""Risk-coverage / selective-QA view over grounded-answer eval records (E3).

Pure POST-PROCESSING: no new model calls, no network, no torch. Given a set of
per-answer records each carrying a CONFIDENCE score (the NLI groundedness rate
the eval already computed) and a binary CORRECT label, this module produces the
"knows when it doesn't know" numbers:

  * a **coverage-vs-accuracy curve** — sweep the confidence threshold; at each
    threshold the pipeline "answers" the items whose confidence clears it and
    abstains on the rest. Reports coverage, accuracy-on-answered, and selective
    risk (1 - accuracy) at every distinct threshold.
  * **AURC** (area under the risk-coverage curve) — the empirical selective-risk
    integral (lower is better; a perfect confidence ordering minimises it).
  * **abstention AUROC** — how well the confidence score discriminates correct
    from incorrect answers (rank-based Mann-Whitney; tie-corrected). ``None``
    when one class is empty (undefined, never a fabricated 0.5).
  * **ECE** (expected calibration error) — binned |confidence - accuracy| gap.

Everything is deterministic and dependency-free (no numpy — the same nearest-
rank / rank-sum arithmetic the rest of the harness uses). Every degenerate slice
(no records, one class only, no scored claims) returns ``None`` with an explicit
``basis`` marker rather than a fabricated number.

The module is usable three ways:
  1. In-process from ``run_grounded_eval`` (the always-on ``risk_coverage``
     report section, built from the per-answered-question groundedness records).
  2. Post-hoc via :func:`pairs_from_report` on a stored
     ``grounded_answer_eval_<ts>.json`` (confidence = per-question
     ``groundedness_rate``; correct = cited a gold-relevant chunk, i.e.
     ``citation_relevant_primary > 0``).
  3. As a CLI over a stored report → a matplotlib-free JSON dump of the view.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Bumped alongside ``EVAL_SCHEMA_VERSION`` growth — the selective-QA view is an
#: additive report section; this version tracks its own shape independently.
RISK_VIEW_SCHEMA_VERSION = "1.0"

#: Default number of equal-width calibration bins for ECE.
DEFAULT_ECE_BINS = 10


# --------------------------------------------------------------------------- #
# Record normalisation
# --------------------------------------------------------------------------- #

def _clean_records(
    records: Sequence[Dict[str, Any]],
) -> List[Tuple[float, bool]]:
    """Coerce raw records to ``[(confidence, correct)]`` pairs.

    A record must carry a numeric ``confidence`` in [0,1] and a ``correct``
    truthy/falsey flag. Records with a null / non-numeric confidence are dropped
    (they carry no basis for a selective decision) — never coerced to 0.0.
    """
    out: List[Tuple[float, bool]] = []
    for r in records:
        conf = r.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            continue
        c = max(0.0, min(1.0, float(conf)))
        out.append((c, bool(r.get("correct"))))
    return out


# --------------------------------------------------------------------------- #
# Core metric arithmetic (no numpy)
# --------------------------------------------------------------------------- #

def risk_coverage_curve(
    pairs: List[Tuple[float, bool]],
) -> List[Dict[str, Any]]:
    """Coverage / accuracy / risk at every distinct confidence threshold.

    At threshold ``t`` the pipeline answers every item with ``confidence >= t``.
    Thresholds are the sorted-descending distinct confidences, so the first row
    is the highest-confidence-only slice and the last row covers everything.
    Empty ``pairs`` → empty curve.
    """
    if not pairs:
        return []
    n = len(pairs)
    thresholds = sorted({c for c, _ in pairs}, reverse=True)
    curve: List[Dict[str, Any]] = []
    for t in thresholds:
        answered = [correct for c, correct in pairs if c >= t]
        k = len(answered)
        n_correct = sum(1 for correct in answered if correct)
        accuracy = (n_correct / k) if k else None
        curve.append(
            {
                "threshold": t,
                "n_answered": k,
                "coverage": k / n,
                "accuracy": accuracy,
                "risk": (1.0 - accuracy) if accuracy is not None else None,
            }
        )
    return curve


def aurc(pairs: List[Tuple[float, bool]]) -> Optional[float]:
    """Empirical Area Under the Risk-Coverage curve (lower is better).

    Sorts items by confidence descending (ties broken by original order for
    determinism) and averages the selective risk (cumulative error rate) over
    every prefix ``k = 1..N``. ``None`` for an empty input (no basis).
    """
    if not pairs:
        return None
    ordered = sorted(range(len(pairs)), key=lambda i: (-pairs[i][0], i))
    errors = 0
    risk_sum = 0.0
    for rank, idx in enumerate(ordered, start=1):
        if not pairs[idx][1]:
            errors += 1
        risk_sum += errors / rank
    return risk_sum / len(ordered)


def abstention_auroc(pairs: List[Tuple[float, bool]]) -> Optional[float]:
    """AUROC of the confidence score as a correctness classifier (tie-corrected).

    Answers "does higher confidence rank a CORRECT answer above an INCORRECT
    one?" via the rank-sum (Mann-Whitney) identity — the probability a random
    correct item outranks a random incorrect one, crediting ties 0.5. ``None``
    when either class is empty (AUROC is undefined then, never a fabricated 0.5).
    """
    pos = [c for c, correct in pairs if correct]
    neg = [c for c, correct in pairs if not correct]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    # Average-rank assignment over the pooled scores (ascending).
    scores = sorted(c for c, _ in pairs)
    # Precompute average rank for each distinct score value.
    rank_of: Dict[float, float] = {}
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        # ranks i+1 .. j (1-indexed); average rank = (i+1 + j) / 2
        avg_rank = (i + 1 + j) / 2.0
        rank_of[scores[i]] = avg_rank
        i = j
    rank_sum_pos = sum(rank_of[c] for c in pos)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return max(0.0, min(1.0, auc))


def expected_calibration_error(
    pairs: List[Tuple[float, bool]], *, bins: int = DEFAULT_ECE_BINS
) -> Dict[str, Any]:
    """Binned Expected Calibration Error (ECE) of confidence vs accuracy.

    Partitions [0,1] into ``bins`` equal-width buckets, and sums each non-empty
    bucket's population-weighted |mean_confidence - accuracy|. Returns the scalar
    ``ece`` (``None`` for empty input) plus the per-bin table for inspection.
    """
    m = bins if isinstance(bins, int) and bins > 0 else DEFAULT_ECE_BINS
    if not pairs:
        return {"ece": None, "bins": m, "bin_table": []}
    n = len(pairs)
    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(m)]
    for c, correct in pairs:
        # rightmost bin includes 1.0
        idx = min(m - 1, int(c * m))
        buckets[idx].append((c, correct))
    ece = 0.0
    table: List[Dict[str, Any]] = []
    for b, bucket in enumerate(buckets):
        if not bucket:
            continue
        cnt = len(bucket)
        mean_conf = sum(c for c, _ in bucket) / cnt
        acc = sum(1 for _, ok in bucket if ok) / cnt
        gap = abs(mean_conf - acc)
        ece += (cnt / n) * gap
        table.append(
            {
                "bin": b,
                "lo": b / m,
                "hi": (b + 1) / m,
                "count": cnt,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "gap": gap,
            }
        )
    return {"ece": ece, "bins": m, "bin_table": table}


# --------------------------------------------------------------------------- #
# Top-level view
# --------------------------------------------------------------------------- #

def selective_qa_view(
    records: Sequence[Dict[str, Any]], *, ece_bins: int = DEFAULT_ECE_BINS
) -> Dict[str, Any]:
    """Roll up the full risk-coverage / selective-QA view over ``records``.

    ``records`` = ``[{"confidence": float, "correct": bool}, ...]`` (extra keys
    ignored). Returns AURC, abstention AUROC, ECE, the coverage/accuracy curve,
    and the overall basis counts. A slice with no usable records reports
    ``basis == "none"`` and every scalar ``None`` (never fabricated).

    Confidence is the eval's per-answer NLI groundedness rate; correct is a
    binary answer-quality label supplied by the caller (default extractor:
    :func:`pairs_from_report`). Fully offline — no model calls.
    """
    pairs = _clean_records(records)
    n = len(pairs)
    n_correct = sum(1 for _, ok in pairs if ok)
    curve = risk_coverage_curve(pairs)
    ece = expected_calibration_error(pairs, bins=ece_bins)
    return {
        "schema_version": RISK_VIEW_SCHEMA_VERSION,
        "basis": "records" if n else "none",
        "n_scored": n,
        "n_correct": n_correct,
        "n_incorrect": n - n_correct,
        "base_accuracy": (n_correct / n) if n else None,
        "aurc": aurc(pairs),
        "abstention_auroc": abstention_auroc(pairs),
        "ece": ece["ece"],
        "ece_bins": ece["bins"],
        "ece_bin_table": ece["bin_table"],
        "coverage_accuracy_curve": curve,
        "_diagnostic": (
            "Selective-QA / risk-coverage view (E3): confidence = per-answer NLI "
            "groundedness rate; correct = answer-quality label (default: cited a "
            "gold-relevant chunk). AURC lower is better; abstention_auroc is "
            "confidence-as-correctness-classifier (None when one class empty); "
            "ECE is binned |confidence - accuracy|. Pure post-processing, no "
            "model calls — diagnostic, NOT a pinned milestone."
        ),
    }


# --------------------------------------------------------------------------- #
# Report extraction (post-hoc on a stored report)
# --------------------------------------------------------------------------- #

def pairs_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ``[{confidence, correct, question_id}]`` from a stored eval report.

    Confidence = each answered question's ``groundedness_rate`` (skips rows with
    a null rate — no scored claim, hence no confidence basis). Correct = the
    answer cited a gold-relevant PRIMARY chunk (``citation_relevant_primary >
    0``) — an answer-quality label INDEPENDENT of the groundedness confidence, so
    the selective-QA discrimination numbers are meaningful (not a self-referential
    threshold on the same signal).
    """
    rows = report.get("questions")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        g = row.get("groundedness_rate")
        if isinstance(g, bool) or not isinstance(g, (int, float)):
            continue
        out.append(
            {
                "confidence": float(g),
                "correct": bool(row.get("citation_relevant_primary", 0)),
                "question_id": row.get("question_id"),
            }
        )
    return out


def selective_qa_from_report(
    report: Dict[str, Any], *, ece_bins: int = DEFAULT_ECE_BINS
) -> Dict[str, Any]:
    """Convenience: :func:`pairs_from_report` → :func:`selective_qa_view`."""
    return selective_qa_view(pairs_from_report(report), ece_bins=ece_bins)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.retrieval.grounded_eval_risk",
        description=(
            "Risk-coverage / selective-QA view over a stored "
            "grounded_answer_eval_<ts>.json report (AURC, abstention AUROC, ECE, "
            "coverage-vs-accuracy curve). Pure post-processing, no model calls."
        ),
    )
    parser.add_argument("report", help="path to a grounded_answer_eval_<ts>.json")
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=DEFAULT_ECE_BINS,
        help=f"calibration bins for ECE (default: {DEFAULT_ECE_BINS})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write the view JSON here (default: stdout)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"grounded-eval-risk: cannot read report: {exc}", file=sys.stderr)
        return 2
    if not isinstance(report, dict):
        print("grounded-eval-risk: report is not a JSON object", file=sys.stderr)
        return 2
    view = selective_qa_from_report(report, ece_bins=args.ece_bins)
    payload = json.dumps(view, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RISK_VIEW_SCHEMA_VERSION",
    "DEFAULT_ECE_BINS",
    "risk_coverage_curve",
    "aurc",
    "abstention_auroc",
    "expected_calibration_error",
    "selective_qa_view",
    "pairs_from_report",
    "selective_qa_from_report",
    "main",
]
