"""Procurement evidence exporter (eval-expansion Phase 4 — backlog E4/E5/D5).

Rolls the newest ``retrieval_eval/grounded_answer_eval_<ts>.json`` grounded-QA
eval report for a LibV2 course into a versioned, self-contained
``retrieval_eval/procurement_evidence_bundle.json`` — the artifact an operator
hands to procurement as evidence that the course-pinned tutor answers grounded,
declines what it should, and does so under a recorded flag config.

Design contracts
----------------

* **Advisory only (D5).** The bundle is ADVISORY evidence in the promotion
  chain. It NEVER mutates the promotion-chain report (its schema is
  ``additionalProperties: false`` and closed) nor ``course_status``. It sits
  BESIDE the chain report, keyed to it by ``chain_hash`` + ``course_status``,
  and records a computed ``readiness`` field encoding the blocking-flip
  criterion. The gate stays advisory until an operator flips config — that flip
  is an explicit human decision, never a side effect of this exporter. See
  :func:`link_promotion_chain`.

* **Anti-silent-degradation.** A course with NO stored grounded eval report
  yields an EXPLICIT ``evaluation_status: "not_evaluated"`` bundle (never a
  fabricated all-zero headline), mirroring the ``missing_stage_report``
  sentinel precedent in
  :mod:`lib.aggregators.promotion_chain_report`.

* **No LLM call site.** This module only READS a report the harness already
  wrote; it makes no model call, so it wires no ``DecisionCapture`` (the LLM
  call-site instrumentation law applies to NEW model calls only).

* **PPI intervals (E5).** When an operator-labeled anchor slice
  (``retrieval_eval/operator_labels.json``) is present, headline CIs use
  prediction-powered inference (ARES-style) — a small labeled slice corrects
  the automated proxy estimate for a tighter, less-biased interval than
  classical inference on the labeled slice alone. Absent the labels file the
  bundle carries classical 95% Wilson intervals only. See
  :func:`compute_ppi_interval` and :data:`OPERATOR_LABELS_FILENAME`.

Blocking-flip criterion (D5, encoded — NOT auto-applied)
--------------------------------------------------------

The eval becomes a BLOCKING promotion gate only after **two consecutive
floor-passing runs on >= 2 courses** — and even then only when an operator
flips config. This module computes the per-course contribution to that
criterion (:func:`floor_pass`, :func:`this_course_readiness`) and can
aggregate per-course bundles into the cross-course readiness verdict
(:func:`aggregate_readiness`), but ``blocking_flip_ready`` is a REPORTED
signal, never an enforced gate.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Bumped when the bundle shape changes. 1.0 — initial procurement bundle
#: (pinned headline subset + phrasing/abstention/refusal breakdowns +
#: flag-config stamp + confidence intervals + readiness).
EVIDENCE_SCHEMA_VERSION = "1.0"

RETRIEVAL_EVAL_SUBDIR = "retrieval_eval"
#: Grounded-eval report glob (owned by ``lib.retrieval.grounded_eval``).
_EVAL_REPORT_GLOB = "grounded_answer_eval_*.json"
#: Stable bundle filename (overwritten in place each run — the newest bundle
#: is the procurement surface; historical reports remain the row-level
#: evidence). Disjoint from every ``lib.retrieval.grounded_eval`` filename.
EVIDENCE_BUNDLE_FILENAME = "procurement_evidence_bundle.json"
#: Operator-labeled anchor slice (E5). Operator-authored; lives beside the
#: eval reports. Absent → Wilson-only CIs.
OPERATOR_LABELS_FILENAME = "operator_labels.json"
#: Operator-labels schema this exporter reads.
OPERATOR_LABELS_SCHEMA_VERSION = "1.0"

#: Blocking-flip criterion (D5). Encoded here so the readiness computation and
#: the documented comment can never drift.
READINESS_CONSECUTIVE_RUNS = 2
READINESS_MIN_COURSES = 2

_NOT_EVALUATED = "not_evaluated"
_EVALUATED = "evaluated"

#: Headline metrics carried into the bundle (pinned floors + their companions).
#: Kept aligned with ``lib.retrieval.grounded_eval.MILESTONE_TARGETS`` +
#: the additive companion rates the report exposes.
_PINNED_HEADLINE_KEYS = (
    "answer_rate",
    "citation_resolution_rate",
    "citation_precision",
    "citation_recall",
    "citation_precision_legacy",
    "groundedness_rate_mean",
    "groundedness_rate_micro",
    "unsupported_claim_rate",
)

#: Statuses whose per-question row counts as an ANSWERED item (mirrors
#: ``grounded_eval._ANSWERED_STATUSES`` without importing the private name).
_ANSWERED_STATUSES = frozenset({"answered", "answered_with_warnings"})


# --------------------------------------------------------------------------- #
# Report discovery
# --------------------------------------------------------------------------- #

def latest_eval_report(course_dir: Path) -> Optional[Path]:
    """Return the newest ``grounded_answer_eval_<ts>.json`` under a course dir.

    Reports are timestamped ``...eval_YYYYMMDDTHHMMSSZ.json`` so lexical sort
    == chronological sort. Returns ``None`` when the ``retrieval_eval/`` dir or
    any report is absent (the caller then emits a ``not_evaluated`` bundle).
    """
    eval_dir = Path(course_dir) / RETRIEVAL_EVAL_SUBDIR
    if not eval_dir.is_dir():
        return None
    candidates = sorted(eval_dir.glob(_EVAL_REPORT_GLOB))
    return candidates[-1] if candidates else None


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("procurement_evidence: cannot read %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# Floor-pass check (mirrors the tier-2 milestone test)
# --------------------------------------------------------------------------- #

def _headline_measured(headline: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten the milestone-relevant headline metrics (incl. nested refusal)."""
    refusal = headline.get("refusal") if isinstance(headline.get("refusal"), Mapping) else {}
    return {
        "answer_rate": headline.get("answer_rate"),
        "citation_resolution_rate": headline.get("citation_resolution_rate"),
        "citation_precision": headline.get("citation_precision"),
        "groundedness_rate_mean": headline.get("groundedness_rate_mean"),
        "unsupported_claim_rate": headline.get("unsupported_claim_rate"),
        "refusal_recall": (refusal or {}).get("refusal_recall"),
        "refusal_precision": (refusal or {}).get("refusal_precision"),
    }


def floor_pass(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate a report's headline against the pinned milestone floors.

    Floors are checked via ``>=``; the ``unsupported_claim_rate`` ceiling via
    ``<=`` (mirrors ``test_stored_eval_artifacts_meet_milestone_targets``). A
    metric that is ``None`` / missing FAILS (anti-silent-degradation: a report
    that never measured a floored metric cannot pass it).

    Returns ``{"passed": bool, "metrics": {metric: {value, target, kind,
    passed}}, "missing": [metric, ...]}``.
    """
    from lib.retrieval.grounded_eval import MILESTONE_CEILINGS, MILESTONE_TARGETS

    headline = report.get("headline") if isinstance(report, Mapping) else None
    measured = _headline_measured(headline or {})
    detail: Dict[str, Any] = {}
    missing: List[str] = []
    all_pass = True
    for key, target in MILESTONE_TARGETS.items():
        value = measured.get(key)
        is_ceiling = key in MILESTONE_CEILINGS
        if value is None:
            missing.append(key)
            passed = False
        elif is_ceiling:
            passed = float(value) <= float(target)
        else:
            passed = float(value) >= float(target)
        if not passed:
            all_pass = False
        detail[key] = {
            "value": value,
            "target": target,
            "kind": "ceiling" if is_ceiling else "floor",
            "passed": passed,
        }
    return {"passed": all_pass, "metrics": detail, "missing": missing}


# --------------------------------------------------------------------------- #
# Confidence intervals — Wilson (classical) + PPI (prediction-powered)
# --------------------------------------------------------------------------- #

def wilson_ci(successes: float, n: int, z: float = 1.96) -> Dict[str, Any]:
    """95% Wilson score interval for a binomial proportion.

    ``n <= 0`` → null interval (never a fabricated 0.0). ``n < 30`` stamps
    ``basis="diagnostic"`` (too small to trust the point estimate) — mirrors
    the convention in ``lib.retrieval.grounded_eval._wilson_ci`` so the two
    surfaces agree on what a diagnostic bucket is.
    """
    n = int(n)
    if n <= 0:
        return {"method": "wilson", "point": None, "lo": None, "hi": None,
                "n": 0, "basis": "diagnostic"}
    successes = max(0.0, min(float(successes), float(n)))
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    return {
        "method": "wilson",
        "point": phat,
        "lo": max(0.0, center - margin),
        "hi": min(1.0, center + margin),
        "n": n,
        "basis": "diagnostic" if n < 30 else "sufficient",
    }


def _sample_variance(values: Sequence[float]) -> float:
    """Unbiased (n-1) sample variance; 0.0 for fewer than 2 points."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _sample_covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


def compute_ppi_interval(
    unlabeled_proxy: Sequence[float],
    labeled_proxy: Sequence[float],
    labeled_gold: Sequence[float],
    *,
    z: float = 1.96,
    power_tuning: bool = False,
) -> Optional[Dict[str, Any]]:
    """Prediction-powered inference (PPI/PPI++) interval for a mean (E5).

    ARES-style: a small operator-labeled anchor slice ``L`` (``labeled_proxy``
    = automated proxy value, ``labeled_gold`` = human gold value, per item)
    corrects an automated estimate computed over a large unlabeled set ``U``
    (``unlabeled_proxy`` = proxy value per item, no human label). The result is
    a tighter, bias-corrected interval than classical inference on ``L`` alone.

    Estimator (Angelopoulos, Duchi & Zrnic 2023, PPI++)::

        theta_hat(lambda) = mean_L(Y) + lambda * (mean_U(f) - mean_L(f))
        Var(lambda)       = lambda^2 * s2_f(U)/N  +  s2_{Y - lambda f}(L)/n
        CI                = theta_hat +/- z * sqrt(Var)

    ``lambda = 1.0`` (the default; ``power_tuning=False``) recovers classical
    PPI: ``mean_U(f) - (mean_L(f) - mean_L(Y))``. ``power_tuning=True`` uses the
    variance-minimising ``lambda*`` (clamped to ``[0, 1]``) — reported as
    ``lambda_star`` either way for transparency.

    Returns ``None`` when there is no labeled slice (``n == 0``) so the caller
    falls back to :func:`wilson_ci`. With a labeled slice but an EMPTY unlabeled
    set the estimator degrades to the classical labeled-only mean + normal CI
    (``method="ppi_labeled_only"``). ``basis="diagnostic"`` when ``n < 30``.
    """
    L_f = [float(v) for v in labeled_proxy]
    L_y = [float(v) for v in labeled_gold]
    n = len(L_y)
    if n == 0 or len(L_f) != n:
        return None
    U_f = [float(v) for v in unlabeled_proxy]
    N = len(U_f)

    mean_Ly = sum(L_y) / n
    mean_Lf = sum(L_f) / n

    # lambda* (PPI++ variance-minimiser), reported regardless of power_tuning.
    s2_f_L = _sample_variance(L_f)
    cov_fy_L = _sample_covariance(L_f, L_y)
    lambda_star: Optional[float] = None
    if N >= 2 and s2_f_L > 0.0:
        raw = cov_fy_L / (s2_f_L * (1.0 + (n / N)))
        lambda_star = max(0.0, min(1.0, raw))
    lam = (lambda_star if (power_tuning and lambda_star is not None) else 1.0)

    if N == 0:
        # No unlabeled data — classical labeled-only mean + normal CI.
        s2_y = _sample_variance(L_y)
        half = z * math.sqrt(s2_y / n) if s2_y > 0.0 else 0.0
        return {
            "method": "ppi_labeled_only",
            "point": mean_Ly,
            "lo": max(0.0, mean_Ly - half),
            "hi": min(1.0, mean_Ly + half),
            "n_labeled": n,
            "n_unlabeled": 0,
            "lambda": lam,
            "lambda_star": lambda_star,
            "half_width": half,
            "basis": "diagnostic" if n < 30 else "sufficient",
        }

    mean_Uf = sum(U_f) / N
    theta = mean_Ly + lam * (mean_Uf - mean_Lf)

    s2_f_U = _sample_variance(U_f)
    rectified = [y - lam * f for y, f in zip(L_y, L_f)]
    s2_rect_L = _sample_variance(rectified)
    var = (lam * lam) * (s2_f_U / N) + (s2_rect_L / n)
    half = z * math.sqrt(var) if var > 0.0 else 0.0
    return {
        "method": "ppi",
        "point": theta,
        "lo": max(0.0, theta - half),
        "hi": min(1.0, theta + half),
        "n_labeled": n,
        "n_unlabeled": N,
        "lambda": lam,
        "lambda_star": lambda_star,
        "half_width": half,
        "basis": "diagnostic" if n < 30 else "sufficient",
    }


# --------------------------------------------------------------------------- #
# Per-question proxy extraction (for PPI over a chosen headline metric)
# --------------------------------------------------------------------------- #

def _proxy_series(report: Mapping[str, Any], metric: str) -> Dict[str, float]:
    """Map ``question_id -> automated proxy value`` for a headline metric.

    Supported metrics:

    * ``groundedness_rate_mean`` — per-question ``groundedness_rate`` (float in
      ``[0, 1]``; ``None`` rows skipped — no groundedness was scored).
    * ``answer_rate`` — 1.0 when the row's ``status`` is an answered status,
      else 0.0 (every answerable row contributes).

    Any other metric → ``{}`` (no per-item proxy; PPI is skipped for it).
    """
    rows = report.get("questions")
    if not isinstance(rows, list):
        return {}
    out: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        if metric == "groundedness_rate_mean":
            g = row.get("groundedness_rate")
            if g is None:
                continue
            try:
                out[qid] = float(g)
            except (TypeError, ValueError):
                continue
        elif metric == "answer_rate":
            out[qid] = 1.0 if str(row.get("status") or "") in _ANSWERED_STATUSES else 0.0
    return out


def load_operator_labels(course_dir: Path) -> Optional[Dict[str, Any]]:
    """Read + validate the operator-labeled anchor slice (E5).

    Format (``retrieval_eval/operator_labels.json``)::

        {
          "schema_version": "1.0",
          "metric": "groundedness_rate_mean",   # headline metric anchored
          "labels": [
            {"question_id": "<id>", "operator_label": 1.0, "proxy_label": 0.8},
            ...
          ]
        }

    ``operator_label`` is the human gold value for the metric's per-item
    quantity (for a rate: 1.0/0.0; for a groundedness score: a value in
    ``[0, 1]``). ``proxy_label`` is OPTIONAL — when omitted the exporter joins
    the item to the report's per-question proxy by ``question_id``.

    Returns the parsed dict, or ``None`` when the file is absent / malformed /
    carries no usable ``labels`` (the caller then emits Wilson-only CIs).
    """
    path = Path(course_dir) / RETRIEVAL_EVAL_SUBDIR / OPERATOR_LABELS_FILENAME
    if not path.is_file():
        return None
    doc = _read_json(path)
    if not isinstance(doc, Mapping):
        return None
    labels = doc.get("labels")
    metric = doc.get("metric")
    if not isinstance(labels, list) or not labels or not isinstance(metric, str):
        return None
    clean: List[Dict[str, Any]] = []
    for item in labels:
        if not isinstance(item, Mapping):
            continue
        qid = item.get("question_id")
        gold = item.get("operator_label")
        if qid is None or gold is None:
            continue
        try:
            row: Dict[str, Any] = {"question_id": str(qid), "operator_label": float(gold)}
        except (TypeError, ValueError):
            continue
        if item.get("proxy_label") is not None:
            try:
                row["proxy_label"] = float(item["proxy_label"])
            except (TypeError, ValueError):
                pass
        clean.append(row)
    if not clean:
        return None
    return {
        "schema_version": str(doc.get("schema_version") or ""),
        "metric": metric,
        "labels": clean,
        "path": str(path),
    }


def _metric_ci(
    report: Mapping[str, Any],
    metric: str,
    operator_labels: Optional[Mapping[str, Any]],
    *,
    power_tuning: bool = False,
) -> Dict[str, Any]:
    """Confidence interval for one headline metric — PPI when the operator
    labels anchor THIS metric, else classical Wilson.

    The Wilson denominator is the metric's natural item count: for
    ``groundedness_rate_mean`` it is the number of per-question rows that
    carried a groundedness score; for ``answer_rate`` the answerable-item
    count. The Wilson success count is ``round(point * n)`` — an approximation
    for a mean-of-scores metric, honest for a true rate.
    """
    proxy_by_qid = _proxy_series(report, metric)

    # PPI branch — only when the operator slice anchors this exact metric.
    if (
        operator_labels is not None
        and str(operator_labels.get("metric")) == metric
    ):
        labeled = operator_labels.get("labels") or []
        labeled_ids = {str(r["question_id"]) for r in labeled}
        L_proxy: List[float] = []
        L_gold: List[float] = []
        for r in labeled:
            qid = str(r["question_id"])
            proxy = r.get("proxy_label")
            if proxy is None:
                proxy = proxy_by_qid.get(qid)
            if proxy is None:
                continue
            L_proxy.append(float(proxy))
            L_gold.append(float(r["operator_label"]))
        U_proxy = [v for qid, v in proxy_by_qid.items() if qid not in labeled_ids]
        ppi = compute_ppi_interval(
            U_proxy, L_proxy, L_gold, power_tuning=power_tuning
        )
        if ppi is not None:
            ppi["metric"] = metric
            return ppi

    # Wilson fallback over the metric's per-item population.
    headline = report.get("headline") if isinstance(report, Mapping) else {}
    point = (headline or {}).get(metric)
    n = len(proxy_by_qid)
    if point is None or n == 0:
        # No per-item series (e.g. metric not row-decomposable) — report the
        # headline point with a null interval rather than fabricating one.
        ci = {"method": "point_only", "point": point, "lo": None, "hi": None,
              "n": n, "basis": "diagnostic"}
        ci["metric"] = metric
        return ci
    ci = wilson_ci(round(float(point) * n), n)
    ci["metric"] = metric
    return ci


def _confidence_intervals(
    report: Mapping[str, Any],
    operator_labels: Optional[Mapping[str, Any]],
    *,
    power_tuning: bool = False,
) -> Dict[str, Any]:
    """CIs for the row-decomposable headline metrics (``groundedness_rate_mean``,
    ``answer_rate``). PPI when the operator slice anchors the metric, else
    Wilson."""
    out: Dict[str, Any] = {}
    for metric in ("groundedness_rate_mean", "answer_rate"):
        out[metric] = _metric_ci(
            report, metric, operator_labels, power_tuning=power_tuning
        )
    return out


# --------------------------------------------------------------------------- #
# Readiness — the blocking-flip criterion (D5, computed, never auto-applied)
# --------------------------------------------------------------------------- #

def _consecutive_passing_runs(course_dir: Path) -> int:
    """Count the trailing run of consecutive floor-passing eval reports.

    Walks every ``grounded_answer_eval_<ts>.json`` newest-first and counts how
    many of the MOST RECENT reports pass all milestone floors, stopping at the
    first failure. This is the per-course half of the two-consecutive-runs
    criterion (the >= 2 COURSES half is aggregated by
    :func:`aggregate_readiness`).
    """
    eval_dir = Path(course_dir) / RETRIEVAL_EVAL_SUBDIR
    if not eval_dir.is_dir():
        return 0
    reports = sorted(eval_dir.glob(_EVAL_REPORT_GLOB), reverse=True)
    streak = 0
    for path in reports:
        doc = _read_json(path)
        if not isinstance(doc, Mapping):
            break
        if floor_pass(doc)["passed"]:
            streak += 1
        else:
            break
    return streak


def this_course_readiness(course_dir: Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-course contribution to the blocking-flip criterion (D5).

    Returns the criterion constants, this course's current-report floor-pass,
    its trailing consecutive-passing-run count, and whether THIS course meets
    the per-course run half of the criterion. ``blocking_flip_ready`` is
    DELIBERATELY absent here — the cross-course (>= 2 courses) determination is
    made by :func:`aggregate_readiness`, and the flip itself stays an explicit
    operator config decision.
    """
    consecutive = _consecutive_passing_runs(course_dir)
    current = floor_pass(report)["passed"]
    return {
        "criterion": {
            "consecutive_floor_passing_runs": READINESS_CONSECUTIVE_RUNS,
            "min_courses": READINESS_MIN_COURSES,
            "_note": (
                "Grounded-eval becomes a BLOCKING promotion gate only after "
                f"{READINESS_CONSECUTIVE_RUNS} consecutive floor-passing runs on "
                f">= {READINESS_MIN_COURSES} courses AND an operator flips config. "
                "This bundle is ADVISORY evidence; it never enforces the gate."
            ),
        },
        "current_report_floor_pass": current,
        "consecutive_passing_runs": consecutive,
        "meets_run_criterion": consecutive >= READINESS_CONSECUTIVE_RUNS,
    }


def aggregate_readiness(bundles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-course bundles into the cross-course blocking-flip verdict.

    ``blocking_flip_ready`` is True when at least :data:`READINESS_MIN_COURSES`
    distinct EVALUATED courses each meet the per-course run criterion (>=
    :data:`READINESS_CONSECUTIVE_RUNS` consecutive floor-passing runs). Even
    when True the flip remains an explicit operator config decision — this is a
    REPORTED signal, never an enforced gate (D5)."""
    courses_meeting: List[str] = []
    for b in bundles:
        if not isinstance(b, Mapping):
            continue
        if b.get("evaluation_status") != _EVALUATED:
            continue
        readiness = b.get("readiness") if isinstance(b.get("readiness"), Mapping) else {}
        if readiness.get("meets_run_criterion"):
            slug = str(b.get("course_slug") or b.get("course_code") or len(courses_meeting))
            courses_meeting.append(slug)
    ready = len(set(courses_meeting)) >= READINESS_MIN_COURSES
    return {
        "criterion": {
            "consecutive_floor_passing_runs": READINESS_CONSECUTIVE_RUNS,
            "min_courses": READINESS_MIN_COURSES,
        },
        "courses_meeting_run_criterion": sorted(set(courses_meeting)),
        "blocking_flip_ready": ready,
        "_note": (
            "ADVISORY. blocking_flip_ready=True means the evidence supports "
            "flipping grounded-eval to a blocking gate; the flip itself stays "
            "an explicit operator config decision and is never auto-applied."
        ),
    }


# --------------------------------------------------------------------------- #
# Bundle builder
# --------------------------------------------------------------------------- #

def _headline_subset(headline: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: headline.get(k) for k in _PINNED_HEADLINE_KEYS}
    refusal = headline.get("refusal") if isinstance(headline.get("refusal"), Mapping) else {}
    out["refusal_recall"] = (refusal or {}).get("refusal_recall")
    out["refusal_precision"] = (refusal or {}).get("refusal_precision")
    return out


def build_evidence_bundle(
    course_dir: Path,
    *,
    course_code: str = "",
    course_slug: str = "",
    run_id: str = "",
    power_tuning: bool = False,
) -> Dict[str, Any]:
    """Build the procurement evidence bundle dict for a LibV2 course dir.

    Rolls the NEWEST grounded eval report into a versioned bundle: pinned
    headline subset, phrasing / abstention / refusal-by-category breakdowns,
    flag-config stamp, gold pin, confidence intervals (PPI when operator labels
    anchor a metric, else Wilson), floor-pass verdict, and the per-course
    readiness contribution.

    Missing report → an EXPLICIT ``not_evaluated`` bundle (anti-silent-
    degradation): no fabricated headline, a stated reason, and a readiness
    block that records zero passing runs.
    """
    course_dir = Path(course_dir)
    slug = course_slug or course_dir.name
    base: Dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "advisory": True,
        "advisory_note": (
            "ADVISORY procurement evidence rolled from the grounded-QA eval. "
            "It does NOT gate promotion or alter course_status; the blocking "
            "flip is an explicit operator config decision (see readiness)."
        ),
        "course_code": course_code or "",
        "course_slug": slug,
        "run_id": run_id or "",
    }

    report_path = latest_eval_report(course_dir)
    if report_path is None:
        base.update({
            "evaluation_status": _NOT_EVALUATED,
            "reason": (
                "no grounded_answer_eval_*.json under "
                f"{course_dir / RETRIEVAL_EVAL_SUBDIR}"
            ),
            "readiness": {
                "criterion": {
                    "consecutive_floor_passing_runs": READINESS_CONSECUTIVE_RUNS,
                    "min_courses": READINESS_MIN_COURSES,
                },
                "current_report_floor_pass": None,
                "consecutive_passing_runs": 0,
                "meets_run_criterion": False,
            },
        })
        return base

    report = _read_json(report_path)
    if not isinstance(report, Mapping):
        base.update({
            "evaluation_status": _NOT_EVALUATED,
            "reason": f"unreadable/malformed eval report {report_path.name}",
            "readiness": {
                "criterion": {
                    "consecutive_floor_passing_runs": READINESS_CONSECUTIVE_RUNS,
                    "min_courses": READINESS_MIN_COURSES,
                },
                "current_report_floor_pass": None,
                "consecutive_passing_runs": 0,
                "meets_run_criterion": False,
            },
        })
        return base

    headline = report.get("headline") if isinstance(report.get("headline"), Mapping) else {}
    operator_labels = load_operator_labels(course_dir)
    fp = floor_pass(report)

    base.update({
        "evaluation_status": _EVALUATED,
        "source_report": {
            "path": str(report_path),
            "filename": report_path.name,
            "eval_schema_version": report.get("schema_version"),
            "generated_at": report.get("generated_at"),
            "engine": report.get("engine"),
            "model_id": report.get("model_id"),
            "prompt_version": report.get("prompt_version"),
            "refusal_policy_version": report.get("refusal_policy_version"),
        },
        "gold": report.get("gold"),
        "headline": _headline_subset(headline),
        "phrasing_breakdown": headline.get("phrasing_breakdown"),
        "abstention": headline.get("abstention"),
        "refusal": {
            "n_probes": (headline.get("refusal") or {}).get("n_probes")
            if isinstance(headline.get("refusal"), Mapping) else None,
            "refusal_recall": (headline.get("refusal") or {}).get("refusal_recall")
            if isinstance(headline.get("refusal"), Mapping) else None,
            "refusal_precision": (headline.get("refusal") or {}).get("refusal_precision")
            if isinstance(headline.get("refusal"), Mapping) else None,
            "by_category": (headline.get("refusal") or {}).get("by_category")
            if isinstance(headline.get("refusal"), Mapping) else None,
        },
        "flag_config": report.get("flag_config"),
        "floor_pass": fp,
        "confidence_intervals": _confidence_intervals(
            report, operator_labels, power_tuning=power_tuning
        ),
        "operator_labels": (
            {
                "present": True,
                "metric": operator_labels.get("metric"),
                "n_labeled": len(operator_labels.get("labels") or []),
                "schema_version": operator_labels.get("schema_version"),
            }
            if operator_labels is not None
            else {"present": False}
        ),
        "readiness": this_course_readiness(course_dir, report),
    })
    return base


def link_promotion_chain(
    bundle: Dict[str, Any], chain_report: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Key the ADVISORY bundle to the promotion-chain report (D5 wiring seam).

    The promotion-chain report schema is ``additionalProperties: false`` and
    closed, so this exporter DELIBERATELY does not mutate it. Instead the bundle
    records a back-reference (``chain_hash`` + ``course_status`` + ``advisory``)
    so an operator (or a future consumer) can join the evidence to the exact
    chain it attests to — without the evidence ever changing ``course_status``.
    Returns the mutated ``bundle`` (also mutated in place) for chaining.
    """
    ref: Dict[str, Any] = {"advisory": True, "linked": False}
    if isinstance(chain_report, Mapping):
        ref.update({
            "linked": True,
            "chain_hash": chain_report.get("chain_hash"),
            "course_status": chain_report.get("course_status"),
            "run_id": chain_report.get("run_id"),
            "_note": (
                "ADVISORY back-reference only. The evidence does not alter "
                "course_status; the promotion-chain report is unchanged."
            ),
        })
    bundle["promotion_chain"] = ref
    return bundle


def write_evidence_bundle(
    course_dir: Path,
    *,
    course_code: str = "",
    course_slug: str = "",
    run_id: str = "",
    promotion_chain_path: Optional[Path] = None,
    power_tuning: bool = False,
) -> Optional[Path]:
    """Build + write ``retrieval_eval/procurement_evidence_bundle.json``.

    When ``promotion_chain_path`` is given (best-effort), the emitted bundle
    carries an ADVISORY back-reference to that chain report (:func:`link_
    promotion_chain`) — never mutating the chain report itself.

    Returns the written path, or ``None`` on any filesystem failure (the caller
    is best-effort — this exporter, like every post-loop aggregator, must never
    perturb the run).
    """
    try:
        bundle = build_evidence_bundle(
            course_dir,
            course_code=course_code,
            course_slug=course_slug,
            run_id=run_id,
            power_tuning=power_tuning,
        )
        chain_report: Optional[Mapping[str, Any]] = None
        if promotion_chain_path is not None:
            cp = Path(promotion_chain_path)
            if cp.is_file():
                doc = _read_json(cp)
                if isinstance(doc, Mapping):
                    chain_report = doc
        link_promotion_chain(bundle, chain_report)

        eval_dir = Path(course_dir) / RETRIEVAL_EVAL_SUBDIR
        eval_dir.mkdir(parents=True, exist_ok=True)
        out = eval_dir / EVIDENCE_BUNDLE_FILENAME
        out.write_text(
            json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return out
    except Exception as exc:  # noqa: BLE001 — exporter is best-effort
        logger.warning(
            "procurement_evidence: failed to write bundle for %s: %s",
            course_dir, exc,
        )
        return None


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_BUNDLE_FILENAME",
    "OPERATOR_LABELS_FILENAME",
    "OPERATOR_LABELS_SCHEMA_VERSION",
    "READINESS_CONSECUTIVE_RUNS",
    "READINESS_MIN_COURSES",
    "latest_eval_report",
    "floor_pass",
    "wilson_ci",
    "compute_ppi_interval",
    "load_operator_labels",
    "this_course_readiness",
    "aggregate_readiness",
    "build_evidence_bundle",
    "link_promotion_chain",
    "write_evidence_bundle",
]
