"""Train the five-class figure-router head on frozen SigLIP embeddings.

The validated recipe uses ``LogisticRegression(C=10,
class_weight="balanced")`` on frozen SigLIP-2 embeddings; its reference
five-fold cross-validation measured 0.857 accuracy and 0.801 macro-F1.
``class_weight="balanced"`` is mandatory (+0.18 macro-F1); an MLP buys
nothing and self-training did not improve the reference evaluation. The
training contract also includes:

  * ``CalibratedClassifierCV`` wrapping, because the runtime contract
    (``figures.router.classify_subtype``) abstains to ``other`` below a
    CALIBRATED p<0.55 — raw logreg scores are not calibrated enough to
    threshold;
  * a 150-row per-class cap so no dominant class overwhelms underrepresented
    classes;
  * fail-closed input guards: refuses to train when the labels are missing,
    unparseable, below ``--min-labels``, or reference images absent from the
    embedding npz files (no silent row drops).

Inputs are label JSONL files (default: ``labels_train.jsonl`` plus
``seed_labels.jsonl``), each row::

    {"image_path": "data/figure_images/...", "visual_label": "<subtype>", ...}

joined to embeddings by repo-relative ``image_path`` against
``data/figure_embeddings/{train,val,test}.npz``.

Outputs ``head.joblib`` + ``meta.json`` under ``--out``
(default ``models/figure_router/v1``).

CPU, seconds-to-minutes, no torch.

Usage::

    .venv/bin/python scripts/training/train_figure_router.py \
        [--labels data/figure_labels/labels_train.jsonl data/figure_labels/seed_labels.jsonl] \
        [--cap 150] [--min-labels 100] [--out models/figure_router/v1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from semantik_structure.figures.router import SUBTYPES  # noqa: E402 — torch-free import

EMB_DIR = REPO_ROOT / "data/figure_embeddings"
EMB_SPLITS = ("train", "val", "test")
DEFAULT_LABELS = (
    "data/figure_labels/labels_train.jsonl",
    "data/figure_labels/seed_labels.jsonl",
)


class FigureRouterTrainError(RuntimeError):
    """Training input invalid — refuse to train rather than degrade."""


def load_embedding_index() -> dict[str, np.ndarray]:
    """repo-relative image_path -> L2-normalized SigLIP embedding."""
    index: dict[str, np.ndarray] = {}
    for split in EMB_SPLITS:
        path = EMB_DIR / f"{split}.npz"
        if not path.exists():
            raise FigureRouterTrainError(
                f"{path} missing — run scripts/datasets/build_figure_embeddings.py first"
            )
        data = np.load(path, allow_pickle=True)
        for ip, emb in zip(data["image_path"], data["embedding"]):
            rel = str(Path(str(ip)).relative_to(REPO_ROOT))
            index[rel] = np.asarray(emb, dtype=np.float32)
    return index


def load_labels(paths: list[Path]) -> list[dict]:
    """Read + validate label rows; bad labels raise, they are never dropped."""
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            raise FigureRouterTrainError(
                f"labels file {path} does not exist — the labeling run has "
                "not happened yet (see data/figure_labels/README.md)"
            )
        with path.open() as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                label = d.get("visual_label")
                if label not in SUBTYPES:
                    raise FigureRouterTrainError(
                        f"{path}:{i}: visual_label {label!r} not in taxonomy {SUBTYPES}"
                    )
                if not d.get("image_path"):
                    raise FigureRouterTrainError(f"{path}:{i}: missing image_path")
                rows.append({"image_path": d["image_path"], "visual_label": label})
    # Last-write-wins dedup would hide labeling conflicts; surface them.
    by_path: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_path[r["image_path"]].add(r["visual_label"])
    conflicts = {p: sorted(ls) for p, ls in by_path.items() if len(ls) > 1}
    if conflicts:
        raise FigureRouterTrainError(
            f"{len(conflicts)} image(s) labeled inconsistently across files: "
            f"{dict(list(conflicts.items())[:5])} ..."
        )
    seen: set[str] = set()
    deduped = []
    for r in rows:
        if r["image_path"] in seen:
            continue
        seen.add(r["image_path"])
        deduped.append(r)
    return deduped


def cap_per_class(rows: list[dict], cap: int, rng) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["visual_label"]].append(r)
    kept: list[dict] = []
    for cls in SUBTYPES:
        group = by_class.get(cls, [])
        if len(group) > cap:
            rng.shuffle(group)
            group = group[:cap]
        kept.extend(group)
    return kept


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    ap.add_argument("--cap", type=int, default=150, help="max rows per class")
    ap.add_argument(
        "--min-labels",
        type=int,
        default=100,
        help="refuse to train below this many joined labels",
    )
    ap.add_argument("--out", default="models/figure_router/v1")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    import random

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate

    rng = random.Random(args.seed)
    label_paths = [REPO_ROOT / p for p in args.labels]
    rows = load_labels(label_paths)
    index = load_embedding_index()

    missing = [r["image_path"] for r in rows if r["image_path"] not in index]
    if missing:
        raise FigureRouterTrainError(
            f"{len(missing)} labeled image(s) have no embedding (first 5: "
            f"{missing[:5]}) — re-run build_figure_embeddings.py, do not drop"
        )

    rows = cap_per_class(rows, args.cap, rng)
    if len(rows) < args.min_labels:
        raise FigureRouterTrainError(
            f"only {len(rows)} usable labels after the per-class cap; "
            f"--min-labels is {args.min_labels}"
        )
    class_counts = Counter(r["visual_label"] for r in rows)

    X = np.stack([index[r["image_path"]] for r in rows], axis=0)
    y = np.array([r["visual_label"] for r in rows], dtype=object)

    # The higher iteration ceiling lets LBFGS converge on 768-dimensional
    # embeddings while preserving the validated solver and regularization.
    base = LogisticRegression(C=10, class_weight="balanced", max_iter=5000)

    n_folds = min(5, min(class_counts.values()))
    if n_folds < 2:
        raise FigureRouterTrainError(
            f"a class has <2 labels ({dict(class_counts)}) — cannot CV or "
            "calibrate; label more rows for the starved class"
        )
    cv = cross_validate(base, X, y, cv=n_folds, scoring=("accuracy", "f1_macro"), n_jobs=1)
    cv_acc = float(cv["test_accuracy"].mean())
    cv_f1 = float(cv["test_f1_macro"].mean())
    print(f"{n_folds}-fold CV: acc={cv_acc:.3f} macro-F1={cv_f1:.3f}")

    # Calibrated probabilities are the runtime contract (abstain at p<0.55).
    head = CalibratedClassifierCV(base, cv=n_folds)
    head.fit(X, y)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(head, out_dir / "head.joblib")

    import sklearn

    meta = {
        "model": "LogisticRegression(C=10, class_weight=balanced) in "
        f"CalibratedClassifierCV(cv={n_folds})",
        "embedding_model": "google/siglip2-base-patch16-224 (frozen, L2-normalized)",
        "classes": sorted(class_counts),
        "n_train": len(rows),
        "per_class": dict(class_counts),
        "cap_per_class": args.cap,
        "cv_accuracy": cv_acc,
        "cv_macro_f1": cv_f1,
        "label_files": {
            # Repo-relative when inside the repo; absolute otherwise (e.g. a
            # smoke run training from /tmp labels).
            (str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)): _sha256(p)
            for p in label_paths
        },
        "seed": args.seed,
        "sklearn_version": sklearn.__version__,
        "trained": date.today().isoformat(),
        "note": "CV numbers are train-pool internal; the shipping gate is "
        "eval_figure_router.py on the frozen eval labels "
        "(acc>=0.85 AND macro-F1>=0.80).",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"per-class (post-cap): {dict(class_counts)}")
    print(f"saved {out_dir / 'head.joblib'} + meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
