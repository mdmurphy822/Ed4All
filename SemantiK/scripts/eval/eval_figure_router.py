"""Gate the trained figure-router head on frozen evaluation labels.

Evaluates ``models/figure_router/v1/head.joblib`` against the doc-disjoint
eval labels (``data/figure_labels/labels_eval.jsonl`` — the labeled
``eval_candidates.jsonl``), reporting two views:

  * **raw** — the head's argmax predictions;
  * **runtime** — what ``figures.router.classify_subtype`` emits:
    abstain to ``other`` whenever the calibrated top-class probability is
    below the abstain threshold (0.55). This is the view the GATES apply to,
    because it represents production behavior.

The runtime view must reach accuracy >= 0.85 and macro-F1 >= 0.80. The script
writes ``data/eval_reports/figure_router_head_v1.json`` and prints the
verdict. Exit code 1 on FAIL so a queue script can stop on a bad head.

CPU, seconds, no torch.

Usage::

    .venv/bin/python scripts/eval/eval_figure_router.py \
        [--labels data/figure_labels/labels_eval.jsonl] \
        [--head models/figure_router/v1/head.joblib] \
        [--report data/eval_reports/figure_router_head_v1.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from semantik_structure.figures.router import (  # noqa: E402 — torch-free import
    SUBTYPE_ABSTAIN_THRESHOLD,
    SUBTYPES,
)
from scripts.training.train_figure_router import (  # noqa: E402
    load_embedding_index,
    load_labels,
)

GATE_ACCURACY = 0.85
GATE_MACRO_F1 = 0.80


def per_class_prf(y_true: list[str], y_pred: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cls in SUBTYPES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * prec * rec / (prec + rec)
            if prec is not None and rec is not None and (prec + rec)
            else None
        )
        out[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
        }
    return out


def summarize(y_true: list[str], y_pred: list[str]) -> dict:
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    per_class = per_class_prf(y_true, y_pred)
    # Macro-F1 over classes PRESENT in the eval truth: a class with zero
    # support has no defined F1 and must not silently inflate/deflate the
    # mean — its absence is reported in per_class.support instead.
    f1s = [
        st["f1"] if st["f1"] is not None else 0.0 for st in per_class.values() if st["support"] > 0
    ]
    confusion = Counter((t, p) for t, p in zip(y_true, y_pred))
    return {
        "accuracy": acc,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_class": per_class,
        "confusion": {f"{t}->{p}": n for (t, p), n in sorted(confusion.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/figure_labels/labels_eval.jsonl")
    ap.add_argument("--head", default="models/figure_router/v1/head.joblib")
    ap.add_argument("--report", default="data/eval_reports/figure_router_head_v1.json")
    args = ap.parse_args(argv)

    import joblib

    head_path = REPO_ROOT / args.head
    if not head_path.exists():
        raise FileNotFoundError(f"{head_path} missing — run scripts/training/train_figure_router.py first")
    head = joblib.load(head_path)

    rows = load_labels([REPO_ROOT / args.labels])
    index = load_embedding_index()
    missing = [r["image_path"] for r in rows if r["image_path"] not in index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} eval image(s) have no embedding (first 5: "
            f"{missing[:5]}) — re-run build_figure_embeddings.py"
        )

    X = np.stack([index[r["image_path"]] for r in rows], axis=0)
    y_true = [r["visual_label"] for r in rows]

    proba = head.predict_proba(X)
    classes = list(head.classes_)
    top_idx = proba.argmax(axis=1)
    y_raw = [classes[i] for i in top_idx]
    top_p = proba[np.arange(len(rows)), top_idx]
    abstained = top_p < SUBTYPE_ABSTAIN_THRESHOLD
    y_runtime = ["other" if a else pred for pred, a in zip(y_raw, abstained)]

    raw = summarize(y_true, y_raw)
    runtime = summarize(y_true, y_runtime)
    gates = {
        "accuracy": {"value": runtime["accuracy"], "gate": GATE_ACCURACY},
        "macro_f1": {"value": runtime["macro_f1"], "gate": GATE_MACRO_F1},
    }
    passed = (
        runtime["accuracy"] >= GATE_ACCURACY
        and runtime["macro_f1"] is not None
        and runtime["macro_f1"] >= GATE_MACRO_F1
    )

    report = {
        "report": "figure-router 5-class head — frozen eval gate",
        "date": date.today().isoformat(),
        "head": args.head,
        "labels": args.labels,
        "n_eval": len(rows),
        "abstain_threshold": SUBTYPE_ABSTAIN_THRESHOLD,
        "abstain_rate": float(abstained.mean()),
        "raw_argmax": raw,
        "runtime_with_abstain": runtime,
        "gates": gates,
        "verdict": "PASS" if passed else "FAIL",
    }
    report_path = REPO_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"n_eval={len(rows)} abstain_rate={float(abstained.mean()):.3f}")
    print(f"raw:     acc={raw['accuracy']:.3f} macro-F1={raw['macro_f1']:.3f}")
    print(f"runtime: acc={runtime['accuracy']:.3f} macro-F1={runtime['macro_f1']:.3f}")
    print(f"gates: acc>={GATE_ACCURACY} macro-F1>={GATE_MACRO_F1}")
    print(f"VERDICT: {report['verdict']} -> {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
