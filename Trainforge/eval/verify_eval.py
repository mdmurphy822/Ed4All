"""Wave 102 - Verifier for stored eval + ablation reports.

Re-runs the *scoring* arithmetic on the metrics stored in
``eval_report.json`` and ``ablation_report.json`` and compares them
against the model_card's ``eval_scores`` block. Exits non-zero on
drift outside the per-metric ``tolerance_band``.

This is the script that ``reproduce_eval.sh`` actually runs. It does
NOT re-run the model (no GPU, no torch). Its job is to fail closed
when:

* A metric in ``eval_report.json`` doesn't match the same metric in
  ``model_card.eval_scores`` (within tolerance).
* The ablation report's headline / retrieval tables don't match the
  model_card's optional ``headline_table`` / ``retrieval_method_table``
  shadow copies (within tolerance).

The stored ``tolerance_band`` is intentionally per-metric because some
metrics (``accuracy``, ``source_match``) are deterministic when the
underlying corpus is pinned, while others (``faithfulness``,
``hallucination_rate``) carry a small variance from the LLM grader
even at temperature=0 (token-tie ordering).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# Default tolerances when ``model_card.eval_scores.tolerance_band``
# is missing. Mirrors the bands recommended in Wave 102's spec.
_DEFAULT_TOLERANCE = {
    "accuracy": 0.0,
    "faithfulness": 0.05,
    "hallucination_rate": 0.05,
    "source_match": 0.0,
    "coverage": 0.0,
    "baseline_delta": 0.05,
    "calibration_ece": 0.05,
}


def verify(
    model_card_path: Path,
    eval_report_path: Path,
    ablation_report_path: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """Run the verifier; return (passed, list_of_drift_messages)."""
    drift: List[str] = []

    card = json.loads(Path(model_card_path).read_text(encoding="utf-8"))
    eval_report = json.loads(Path(eval_report_path).read_text(encoding="utf-8"))
    eval_scores = card.get("eval_scores") or {}
    tolerance = eval_scores.get("tolerance_band") or _DEFAULT_TOLERANCE

    # Compare top-level metrics.
    for key in ("faithfulness", "coverage", "baseline_delta",
                "calibration_ece", "source_match"):
        stored = eval_scores.get(key)
        actual = eval_report.get(key)
        if stored is None or actual is None:
            continue
        band = float(tolerance.get(key, _DEFAULT_TOLERANCE.get(key, 0.0)))
        if abs(float(stored) - float(actual)) > band:
            drift.append(
                f"DRIFT eval_scores.{key}: stored={stored} actual={actual} "
                f"|delta|={abs(float(stored) - float(actual)):.4f} > "
                f"tolerance={band}"
            )

    # Compare hallucination_rate via the inverse-faithfulness identity
    # so a model card that recorded ``hallucination_rate`` directly
    # stays consistent with the eval_report's ``metrics`` block.
    stored_hr = eval_scores.get("hallucination_rate")
    if stored_hr is not None:
        metrics_block = eval_report.get("metrics") or {}
        actual_hr = metrics_block.get("hallucination_rate")
        if actual_hr is None:
            f = eval_report.get("faithfulness")
            if f is not None:
                actual_hr = max(0.0, min(1.0, 1.0 - float(f)))
        if actual_hr is not None:
            band = float(tolerance.get(
                "hallucination_rate", _DEFAULT_TOLERANCE["hallucination_rate"],
            ))
            if abs(float(stored_hr) - float(actual_hr)) > band:
                drift.append(
                    f"DRIFT eval_scores.hallucination_rate: stored={stored_hr} "
                    f"actual={actual_hr} > tolerance={band}"
                )

    # Optional: compare ablation tables.
    if ablation_report_path and Path(ablation_report_path).exists():
        ablation = json.loads(
            Path(ablation_report_path).read_text(encoding="utf-8"),
        )
        stored_headline = eval_scores.get("headline_table") or []
        actual_headline = ablation.get("headline_table") or []
        drift.extend(_compare_table(
            stored_headline, actual_headline, "headline", tolerance,
        ))
        stored_method = eval_scores.get("retrieval_method_table") or []
        actual_method = ablation.get("retrieval_method_table") or []
        drift.extend(_compare_table(
            stored_method, actual_method, "retrieval_method", tolerance,
        ))

    return (len(drift) == 0), drift


def _compare_table(
    stored_rows: List[Dict[str, Any]],
    actual_rows: List[Dict[str, Any]],
    label: str,
    tolerance: Dict[str, float],
) -> List[str]:
    """Compare two ablation tables row-by-row and emit drift messages."""
    drifts: List[str] = []
    if not stored_rows:
        return drifts
    if len(stored_rows) != len(actual_rows):
        drifts.append(
            f"DRIFT {label}_table row count: stored={len(stored_rows)} "
            f"actual={len(actual_rows)}"
        )
        return drifts
    for i, (s, a) in enumerate(zip(stored_rows, actual_rows)):
        for key, sv in s.items():
            if key in ("setup", "method"):
                if sv != a.get(key):
                    drifts.append(
                        f"DRIFT {label}_table[{i}].{key}: stored={sv} "
                        f"actual={a.get(key)}"
                    )
                continue
            av = a.get(key)
            if sv is None or av is None:
                continue
            band = float(tolerance.get(key, _DEFAULT_TOLERANCE.get(key, 0.0)))
            if abs(float(sv) - float(av)) > band:
                drifts.append(
                    f"DRIFT {label}_table[{i}].{key}: stored={sv} "
                    f"actual={av} > tolerance={band}"
                )
    return drifts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wave 102 - Re-verify a stored Trainforge eval_report + "
            "ablation_report against the model_card's tolerance band."
        )
    )
    parser.add_argument(
        "--model-card", required=True,
        help="Path to model_card.json.",
    )
    parser.add_argument(
        "--eval-report", required=True,
        help="Path to eval_report.json.",
    )
    parser.add_argument(
        "--ablation-report", default=None,
        help="Optional path to ablation_report.json.",
    )
    args = parser.parse_args(argv)

    passed, drift = verify(
        model_card_path=Path(args.model_card),
        eval_report_path=Path(args.eval_report),
        ablation_report_path=(
            Path(args.ablation_report) if args.ablation_report else None
        ),
    )
    if passed:
        print("verify_eval: OK (no drift detected within tolerance)")
        return 0
    print("verify_eval: DRIFT detected:", file=sys.stderr)
    for line in drift:
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["verify", "main"]
