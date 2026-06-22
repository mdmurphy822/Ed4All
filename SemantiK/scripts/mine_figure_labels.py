"""Phase P2 — mine figure-label candidates for the 5-class router head.

The P2 fine-plan (2026-06-09) measured that the trained head's macro-F1 is
limited by PER-CLASS label count, not total n: ``equation_or_table_image``
and ``other`` were n-starved at 17/26 labels, and self-training was
falsified as a fix. The lever is targeted human/vision labeling. This script
selects WHICH figures to label, from the cached SigLIP-2 embedding npz
files, and writes candidate manifests for the (user-gated, deferred)
labeling run:

  * ``eval_candidates.jsonl``  — the future FROZEN eval set (default 150).
    Class-balanced over the zero-shot predicted label (macro-F1 is the gate
    metric, so per-class support matters more than deployment proportions),
    source-balanced within class, sampled RANDOMLY within each stratum — no
    uncertainty bias, so the eval is not adversarially skewed toward the
    head's hardest rows.
  * ``train_candidates.jsonl`` — labeling candidates for training (default
    500). Oversamples the starved classes (eq/table, other, map per the
    fine-plan), and within each class takes half from the LOWEST zero-shot
    confidence rows (uncertainty sampling) and half at random.
  * ``seed_labels.jsonl``      — the 60 already-adjudicated visual labels
    salvaged from ``data/eval_reports/figure_router_spotcheck_v1.json``.
    (The original 300-row truth file lived in /tmp and was lost to a host
    reboot — these 60 are what survives, persisted IN-REPO this time.)
  * ``mining_report.json``     — pool stats, quotas, shortfalls, and the
    doc-disjointness guarantee, for audit.

Leakage rules enforced here:
  * eval docs ∩ train-candidate docs = ∅ (doc-disjoint);
  * the 60 seed rows were used to tune the shipped binary-gate margin, so
    their DOCS are excluded from eval candidates entirely (they may still
    appear among train candidates — seeds are training data);
  * already-labeled seed image paths are excluded from both candidate sets
    (re-labeling them would waste budget).

No model load, no torch — pure npz + json work, runs in seconds.

Usage::

    .venv/bin/python scripts/mine_figure_labels.py [--n-train 500]
        [--n-eval 150] [--seed 13]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dart_semantic.figure_router import SUBTYPES  # noqa: E402 — torch-free import

EMB_DIR = REPO_ROOT / "data/figure_embeddings"
SPOTCHECK_REPORT = REPO_ROOT / "data/eval_reports/figure_router_spotcheck_v1.json"
OUT_DIR = REPO_ROOT / "data/figure_labels"

# The captioned dataset splits form the mining pool. The split boundaries
# belong to the CAPTIONER's train/val/test; the router defines its own frozen
# eval here, so the pool is their union (original split recorded per row).
POOL_SPLITS = ("train", "val", "test")

# Train-candidate quotas per zero-shot predicted class (sums to 500). The
# starved classes get the largest shares per the P2 fine-plan measurement;
# 'other' is also where abstain→other routing lands, so its boundary needs
# the labels. Scaled proportionally when --n-train != 500; shortfalls
# redistribute to the remaining classes (recorded in the report).
TRAIN_QUOTAS: dict[str, int] = {
    "equation_or_table_image": 120,
    "other": 120,
    "map": 60,
    "chart": 70,
    "diagram": 70,
    "photo_micrograph": 60,
}

# Fraction of each class's train quota taken from the lowest zero-shot
# confidence rows (uncertainty sampling); the rest is random within class.
UNCERTAINTY_FRACTION = 0.5


def load_pool() -> list[dict]:
    """Union of the captioned splits' npz rows, file-existence-filtered."""
    pool: list[dict] = []
    for split in POOL_SPLITS:
        path = EMB_DIR / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run scripts/build_figure_embeddings.py first"
            )
        data = np.load(path, allow_pickle=True)
        for src, doc, fi, ip, cap, lbl, conf in zip(
            data["source"],
            data["doc_id"],
            data["figure_index"],
            data["image_path"],
            data["caption"],
            data["zero_shot_label"],
            data["zero_shot_confidence"],
        ):
            if not ip or not Path(ip).exists():
                continue
            pool.append(
                {
                    "source": str(src),
                    "doc_id": str(doc),
                    "figure_index": fi,
                    "image_path": str(Path(ip).relative_to(REPO_ROOT)),
                    "caption": str(cap or ""),
                    "dataset_split": split,
                    "zero_shot_label": str(lbl),
                    "zero_shot_confidence": float(conf),
                }
            )
    # One labeling row per IMAGE: the dataset can reference the same image
    # file from more than one row (and the pool unions splits); duplicate
    # paths would waste label budget and trip the train script's per-image
    # consistency check.
    seen: set[str] = set()
    deduped = []
    for r in pool:
        if r["image_path"] in seen:
            continue
        seen.add(r["image_path"])
        deduped.append(r)
    return deduped


def load_seed_labels() -> list[dict]:
    """Salvage the 60 surviving visual labels from the spot-check report."""
    if not SPOTCHECK_REPORT.exists():
        return []
    report = json.loads(SPOTCHECK_REPORT.read_text())
    seeds: list[dict] = []
    for row in report.get("per_row", []):
        label = row.get("visual_label")
        if label not in SUBTYPES:
            continue  # e.g. a row the arbiter could not call — not a label
        path = Path(row["path"])
        rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
        seeds.append(
            {
                "image_path": rel,
                "visual_label": label,
                "visual_label_confidence": row.get("visual_label_confidence"),
                "note": row.get("note"),
                "provenance": "figure_router_spotcheck_v1",
            }
        )
    return seeds


def _source_balanced_take(rows: list[dict], n: int, rng) -> list[dict]:
    """Take up to ``n`` rows cycling round-robin across sources.

    pmc is ~75% of the pool ([[feedback-balance-dominant-category]]); plain
    random sampling would hand it ~75% of every stratum. Round-robin over
    shuffled per-source queues caps any one source at its fair share unless
    the others run dry.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for queue in by_source.values():
        rng.shuffle(queue)
    taken: list[dict] = []
    sources = sorted(by_source)
    while len(taken) < n and any(by_source[s] for s in sources):
        for s in sources:
            if len(taken) >= n:
                break
            if by_source[s]:
                taken.append(by_source[s].pop())
    return taken


def pick_eval_candidates(pool: list[dict], n: int, rng, excluded_docs: set[str]) -> list[dict]:
    """Class-balanced, source-balanced, RANDOM-within-stratum eval pick."""
    eligible = [r for r in pool if r["doc_id"] not in excluded_docs]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        by_class[r["zero_shot_label"]].append(r)
    per_class = n // len(SUBTYPES)
    picked: list[dict] = []
    for cls in SUBTYPES:
        got = _source_balanced_take(by_class.get(cls, []), per_class, rng)
        for r in got:
            picked.append({**r, "selection_reason": f"eval_balanced:{cls}"})
    # Top up to n from the leftover eligible rows if any class ran short.
    if len(picked) < n:
        chosen = {r["image_path"] for r in picked}
        leftovers = [r for r in eligible if r["image_path"] not in chosen]
        for r in _source_balanced_take(leftovers, n - len(picked), rng):
            picked.append({**r, "selection_reason": "eval_topup"})
    return picked


def pick_train_candidates(
    pool: list[dict],
    n: int,
    rng,
    excluded_docs: set[str],
    excluded_paths: set[str],
) -> tuple[list[dict], dict]:
    """Quota-driven train pick: half uncertainty, half random, per class."""
    eligible = [
        r
        for r in pool
        if r["doc_id"] not in excluded_docs and r["image_path"] not in excluded_paths
    ]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        by_class[r["zero_shot_label"]].append(r)

    scale = n / sum(TRAIN_QUOTAS.values())
    quotas = {cls: round(q * scale) for cls, q in TRAIN_QUOTAS.items()}
    shortfalls: dict[str, int] = {}
    picked: list[dict] = []
    for cls in SUBTYPES:
        quota = quotas.get(cls, 0)
        rows = by_class.get(cls, [])
        if len(rows) < quota:
            shortfalls[cls] = quota - len(rows)
            quota = len(rows)
        n_unc = int(quota * UNCERTAINTY_FRACTION)
        rows_by_conf = sorted(rows, key=lambda r: r["zero_shot_confidence"])
        uncertain = rows_by_conf[:n_unc]
        rest = rows_by_conf[n_unc:]
        random_part = _source_balanced_take(rest, quota - n_unc, rng)
        for r in uncertain:
            picked.append({**r, "selection_reason": f"train_uncertainty:{cls}"})
        for r in random_part:
            picked.append({**r, "selection_reason": f"train_random:{cls}"})
    # Redistribute any shortfall to the remaining pool (largest classes
    # absorb it), so the labeling run still gets its full budget.
    deficit = n - len(picked)
    if deficit > 0:
        chosen = {r["image_path"] for r in picked}
        leftovers = [r for r in eligible if r["image_path"] not in chosen]
        for r in _source_balanced_take(leftovers, deficit, rng):
            picked.append({**r, "selection_reason": "train_redistributed"})
    return picked, shortfalls


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    import random

    rng = random.Random(args.seed)

    pool = load_pool()
    seeds = load_seed_labels()
    seed_paths = {s["image_path"] for s in seeds}
    pool_by_path = {r["image_path"]: r for r in pool}
    seed_docs = {pool_by_path[p]["doc_id"] for p in seed_paths if p in pool_by_path}

    # Eval first (it owns its docs); seed docs are ineligible for eval —
    # the shipped binary-gate margin was tuned on them.
    eval_rows = pick_eval_candidates(pool, args.n_eval, rng, excluded_docs=seed_docs)
    eval_docs = {r["doc_id"] for r in eval_rows}

    train_rows, shortfalls = pick_train_candidates(
        pool,
        args.n_train,
        rng,
        excluded_docs=eval_docs,
        excluded_paths=seed_paths,
    )

    assert not ({r["doc_id"] for r in train_rows} & eval_docs), "doc leakage"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "eval_candidates.jsonl", eval_rows)
    write_jsonl(OUT_DIR / "train_candidates.jsonl", train_rows)
    write_jsonl(OUT_DIR / "seed_labels.jsonl", seeds)

    def class_counts(rows: list[dict]) -> dict[str, int]:
        return dict(Counter(r["zero_shot_label"] for r in rows))

    def source_counts(rows: list[dict]) -> dict[str, int]:
        return dict(Counter(r["source"] for r in rows))

    report = {
        "report": "figure label-candidate mining (P2 fine-plan)",
        "seed": args.seed,
        "pool": {
            "n": len(pool),
            "by_class": class_counts(pool),
            "by_source": source_counts(pool),
        },
        "eval_candidates": {
            "n": len(eval_rows),
            "by_class": class_counts(eval_rows),
            "by_source": source_counts(eval_rows),
        },
        "train_candidates": {
            "n": len(train_rows),
            "by_class": class_counts(train_rows),
            "by_source": source_counts(train_rows),
            "quota_shortfalls": shortfalls,
        },
        "seed_labels": {
            "n": len(seeds),
            "provenance": "figure_router_spotcheck_v1 (60 surviving of the "
            "300 visual labels; the rest were lost with /tmp)",
        },
        "doc_disjoint": True,
        "seed_docs_excluded_from_eval": len(seed_docs),
    }
    (OUT_DIR / "mining_report.json").write_text(json.dumps(report, indent=2))

    print(f"pool: {len(pool)} rows {class_counts(pool)}")
    print(f"eval candidates: {len(eval_rows)} {class_counts(eval_rows)}")
    print(f"train candidates: {len(train_rows)} {class_counts(train_rows)}")
    if shortfalls:
        print(f"quota shortfalls (redistributed): {shortfalls}")
    print(f"seed labels salvaged: {len(seeds)}")
    print(f"wrote {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
