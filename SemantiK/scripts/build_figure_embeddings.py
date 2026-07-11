"""Phase P1 — SigLIP-2 figure-router embedding pass + zero-shot subtype labels.

Embeds every figure image in ``data/figure_alt_dataset/`` (train/val/test) AND
the 437 empty-figcaption rows of ``data/figure_catalog/catalog.jsonl``, runs the
zero-shot subtype router (``semantik_structure.figure_router``), and writes per-split
``.npz`` files plus:

  * an agreement report (zero-shot vs caption-keyword weak labels) — the P1
    done-criterion is >=80% agreement overall on the keyword-matchable overlap;
  * a stratified 300-row spot-check sample (``spotcheck_sample.jsonl``) for the
    later (P2) human-labeling phase.

CPU fp32 by default (the 8GB GPU is reserved for the Qwen specialists / SmolVLM2
captioner). Pass ``--device cuda`` ONLY when the GPU is free.

Missing image files are skipped-and-counted (a data-quality count, NOT a silent
fallback). Individually-corrupt images are logged and skipped so one bad PNG
does not abort a multi-hour pass.

Usage::

    .venv/bin/python scripts/build_figure_embeddings.py [--device cpu] \
        [--batch-size 16] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "data/figure_alt_dataset"
CATALOG_PATH = REPO_ROOT / "data/figure_catalog/catalog.jsonl"
IMAGE_ROOT = REPO_ROOT / "data/figure_images"
OUT_DIR = REPO_ROOT / "data/figure_embeddings"

DATASET_SPLITS = ("train", "val", "test")
# Synthetic "split" name for the catalog's no-caption stratum.
CATALOG_SPLIT = "catalog_nocaption"


# ---------------------------------------------------------------------------
# Caption-keyword weak labels (re-derived here per the P1 brief)
# ---------------------------------------------------------------------------
# The investigation found ~4,967 / 20,416 captions keyword-matchable. These
# patterns map a caption to ONE weak subtype (the taxonomy's first five classes;
# "other" has no keyword and is never weak-labeled). A caption matching multiple
# families is assigned by the priority order below (most-specific first). They
# match the caption text only.
#
# Deliberately SPECIFIC, not loose: requiring phrasings like "map of",
# "plot of", "diagram", "photograph" (rather than bare tokens like "graph",
# "curve", "table", "image of", "versus") keeps the matchable set near the
# investigation's ~4,967 and keeps the weak label trustworthy. Loose tokens
# inflate the matchable count to ~8k and pull in false positives ("graph
# theory", "map of the quantum speed limit" — a chart, not a geographic map),
# which would make the agreement number meaningless. These weak labels are
# themselves NOISY (P2 human-labels the spot-check to settle disagreements).
_KEYWORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # map — geographic
    (
        "map",
        re.compile(
            r"\bmap of\b|\bgeographic(?:al)? (?:map|distribution|region)|"
            r"\bcartograph|\bworld map\b|\btopograph(?:ic|y)",
            re.IGNORECASE,
        ),
    ),
    # equation_or_table_image — rendered formula
    (
        "equation_or_table_image",
        re.compile(
            r"\bequations? (?:for|of|describing)\b|\bmathematical formula|"
            r"\bformula for\b",
            re.IGNORECASE,
        ),
    ),
    # chart — plotted numeric data
    (
        "chart",
        re.compile(
            r"\bchart\b|\bplot of\b|\bplot(?:s|ted)? of\b|\bplotted\b|"
            r"\bhistogram|\bscatter ?plot|\bbar (?:chart|graph|plot)|"
            r"\bline (?:chart|graph|plot)|\bpie chart|\bbox ?plot|"
            r"\bgraph of\b|\bas a function of\b|\bx[- ]axis\b|\by[- ]axis\b",
            re.IGNORECASE,
        ),
    ),
    # diagram — schematic / flowchart
    (
        "diagram",
        re.compile(
            r"\bdiagram\b|\bschematic\b|\bflow ?chart\b|\bworkflow\b|"
            r"\bpipeline of\b|\barchitecture of\b|\billustration of\b",
            re.IGNORECASE,
        ),
    ),
    # photo_micrograph — photograph / microscopy
    (
        "photo_micrograph",
        re.compile(
            r"\bphotograph|\bmicrograph|\bmicroscop(?:e|y|ic)|"
            r"\belectron microscop|\bstain(?:ed|ing)\b|\bfluorescence imag|"
            r"\bimmunohistochem|\bhistolog",
            re.IGNORECASE,
        ),
    ),
)


def keyword_weak_label(caption: str) -> str | None:
    """Return the weak subtype for ``caption`` by keyword, or None (no match).

    Priority order is the tuple order above (map > equation/table > chart >
    diagram > photo) — the first family to match wins.
    """
    if not caption:
        return None
    for label, pat in _KEYWORD_PATTERNS:
        if pat.search(caption):
            return label
    return None


# ---------------------------------------------------------------------------
# Row loading + image-path resolution
# ---------------------------------------------------------------------------
def _iter_dataset_rows(split: str):
    path = DATASET_DIR / f"{split}.jsonl"
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img = d.get("image_path")
            yield {
                "source": d.get("source"),
                "doc_id": d.get("doc_id"),
                "figure_index": d.get("figure_index"),
                "caption": d.get("caption") or "",
                "image_path": str(REPO_ROOT / img) if img else None,
            }


def _resolve_catalog_image(source: str, doc_id: str, image_ref: str) -> str | None:
    """Resolve a catalog row's image under ``data/figure_images/{source}/{doc}``.

    Catalog rows carry ``image_ref`` (a URL/CDN fragment or a bare stem like
    ``pbio.0030097.g001``); the fetched files live under the doc dir with the
    same basename stem but a fetch-determined extension. Match the stem against
    any extension; fall back to the sole file in a single-figure doc dir.

    openstax doc_ids contain ``/`` (``book-slug/section-slug``), which the
    fetcher flattened to ``_`` in the directory name, and their ``image_ref``
    is an image-CDN path whose final component is the SHA the file was saved
    under — both handled below.
    """
    doc_dir = IMAGE_ROOT / source / doc_id
    if not doc_dir.is_dir() and "/" in doc_id:
        doc_dir = IMAGE_ROOT / source / doc_id.replace("/", "_")
    if not doc_dir.is_dir():
        return None
    stem = Path(image_ref or "").name
    stem = Path(stem).stem  # drop any extension on the ref itself
    if stem:
        for cand in doc_dir.glob(f"{stem}.*"):
            if cand.is_file():
                return str(cand)
    files = [p for p in doc_dir.iterdir() if p.is_file()]
    if len(files) == 1:
        return str(files[0])
    return None


def _iter_catalog_nocaption_rows():
    with CATALOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cap = (d.get("figcaption") or "").strip()
            if cap:
                continue  # only the empty-figcaption stratum
            src = d.get("source")
            doc = d.get("doc_id")
            resolved = _resolve_catalog_image(src, doc, d.get("image_ref") or "")
            yield {
                "source": src,
                "doc_id": doc,
                "figure_index": d.get("figure_index"),
                "caption": "",  # by construction
                "image_path": resolved,
            }


def load_split_rows(split: str) -> list[dict]:
    if split == CATALOG_SPLIT:
        return list(_iter_catalog_nocaption_rows())
    return list(_iter_dataset_rows(split))


# ---------------------------------------------------------------------------
# Embedding pass per split
# ---------------------------------------------------------------------------
def embed_split(
    split: str,
    rows: list[dict],
    *,
    batch_size: int,
    prompt_embs: np.ndarray,
    embed_images_fn,
    zero_shot_fn,
) -> dict:
    """Embed one split; return a result dict with arrays + counts.

    Missing-image rows are dropped (counted). Corrupt-image batches fall back
    to per-image embedding so one bad file only loses itself.
    """
    present = [r for r in rows if r["image_path"] and Path(r["image_path"]).exists()]
    missing = len(rows) - len(present)

    embeddings: list[np.ndarray] = []
    kept_rows: list[dict] = []
    corrupt = 0

    for start in range(0, len(present), batch_size):
        chunk = present[start : start + batch_size]
        paths = [r["image_path"] for r in chunk]
        try:
            embs = embed_images_fn(paths, batch_size=batch_size)
            for r, e in zip(chunk, embs):
                kept_rows.append(r)
                embeddings.append(e)
        except Exception as exc:  # noqa: BLE001 — isolate the bad file(s)
            print(
                f"  [{split}] batch @{start} failed ({type(exc).__name__}: "
                f"{exc}); retrying per-image",
                flush=True,
            )
            for r in chunk:
                try:
                    e = embed_images_fn([r["image_path"]], batch_size=1)
                    kept_rows.append(r)
                    embeddings.append(e[0])
                except Exception as exc2:  # noqa: BLE001 — data-quality skip
                    corrupt += 1
                    print(
                        f"    skip corrupt {r['image_path']} ({type(exc2).__name__}: {exc2})",
                        flush=True,
                    )
        if (start // batch_size) % 25 == 0:
            print(
                f"  [{split}] {min(start + batch_size, len(present))}/{len(present)} images",
                flush=True,
            )

    emb_arr = np.stack(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)

    # Zero-shot subtype per row.
    labels: list[str] = []
    confs: list[float] = []
    for e in emb_arr:
        lbl, conf = zero_shot_fn(e, prompt_embs)
        labels.append(lbl)
        confs.append(conf)

    return {
        "rows": kept_rows,
        "embeddings": emb_arr,
        "labels": labels,
        "confidences": confs,
        "missing": missing,
        "corrupt": corrupt,
        "n_in": len(rows),
    }


def save_split_npz(split: str, result: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{split}.npz"
    rows = result["rows"]
    np.savez_compressed(
        out,
        source=np.array([r["source"] for r in rows], dtype=object),
        doc_id=np.array([r["doc_id"] for r in rows], dtype=object),
        figure_index=np.array([r["figure_index"] for r in rows], dtype=object),
        image_path=np.array([r["image_path"] for r in rows], dtype=object),
        caption=np.array([r["caption"] for r in rows], dtype=object),
        embedding=result["embeddings"],
        zero_shot_label=np.array(result["labels"], dtype=object),
        zero_shot_confidence=np.array(result["confidences"], dtype=np.float32),
    )
    return out


def load_npz_results(split: str) -> dict | None:
    """Load a previously-saved split npz back into the result-dict shape.

    Lets a partial re-run (``--splits catalog_nocaption``) recompute the
    agreement report and spot-check sample over ALL splits on disk, not just
    the ones embedded in this invocation. ``missing``/``corrupt`` counts are
    not persisted in the npz, so they read 0 here — the per-run log lines are
    the authoritative record of those.
    """
    path = OUT_DIR / f"{split}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    rows = [
        {
            "source": s,
            "doc_id": d,
            "figure_index": fi,
            "image_path": ip,
            "caption": c,
        }
        for s, d, fi, ip, c in zip(
            data["source"],
            data["doc_id"],
            data["figure_index"],
            data["image_path"],
            data["caption"],
        )
    ]
    return {
        "rows": rows,
        "embeddings": data["embedding"],
        "labels": list(data["zero_shot_label"]),
        "confidences": list(data["zero_shot_confidence"]),
        "missing": 0,
        "corrupt": 0,
        "n_in": len(rows),
    }


# ---------------------------------------------------------------------------
# Agreement check (zero-shot vs caption-keyword weak labels)
# ---------------------------------------------------------------------------
def compute_agreement(results: dict[str, dict]) -> dict:
    """Agreement on the keyword-matchable overlap (dataset splits only).

    The catalog no-caption stratum has no caption, so it cannot be keyword-
    weak-labeled and is excluded from the agreement denominator.
    """
    overall_match = 0
    overall_total = 0
    per_class_match: Counter[str] = Counter()
    per_class_total: Counter[str] = Counter()

    for split, res in results.items():
        if split == CATALOG_SPLIT:
            continue
        for r, pred in zip(res["rows"], res["labels"]):
            weak = keyword_weak_label(r["caption"])
            if weak is None:
                continue
            overall_total += 1
            per_class_total[weak] += 1
            if pred == weak:
                overall_match += 1
                per_class_match[weak] += 1

    per_class = {
        cls: {
            "match": per_class_match[cls],
            "total": per_class_total[cls],
            "agreement": (per_class_match[cls] / per_class_total[cls])
            if per_class_total[cls]
            else None,
        }
        for cls in sorted(per_class_total)
    }
    return {
        "overall_match": overall_match,
        "overall_total": overall_total,
        "overall_agreement": (overall_match / overall_total) if overall_total else None,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Stratified spot-check sample
# ---------------------------------------------------------------------------
def build_spotcheck(results: dict[str, dict], *, seed: int = 13) -> list[dict]:
    """300 rows stratified 120 arxiv / 120 pmc / 60 openstax, >=40 per
    predicted subtype where possible. Dataset splits only (need captions)."""
    import random

    rng = random.Random(seed)
    # source -> predicted label -> list of candidate rows
    pool: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for split, res in results.items():
        if split == CATALOG_SPLIT:
            continue
        for r, pred, conf in zip(res["rows"], res["labels"], res["confidences"]):
            pool[r["source"]][pred].append(
                {
                    "image_path": r["image_path"],
                    "predicted_label": pred,
                    "confidence": float(conf),
                    "caption": r["caption"],
                    "source": r["source"],
                    "doc_id": r["doc_id"],
                    "figure_index": r["figure_index"],
                }
            )

    quotas = {"arxiv": 120, "pmc": 120, "openstax": 60}
    per_label_target = 40
    sample: list[dict] = []
    for src, quota in quotas.items():
        by_label = pool.get(src, {})
        picked: list[dict] = []
        # First pass: up to per_label_target per predicted label.
        for lbl in sorted(by_label):
            cands = by_label[lbl][:]
            rng.shuffle(cands)
            picked.extend(cands[:per_label_target])
        # If over quota, trim; if under, top up from the remaining pool.
        rng.shuffle(picked)
        if len(picked) > quota:
            picked = picked[:quota]
        elif len(picked) < quota:
            taken = {id(x) for x in picked}
            remaining = [x for lbl in by_label for x in by_label[lbl] if id(x) not in taken]
            rng.shuffle(remaining)
            picked.extend(remaining[: quota - len(picked)])
        sample.extend(picked[:quota])
    return sample


def write_spotcheck(sample: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "spotcheck_sample.jsonl"
    with out.open("w") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


# ---------------------------------------------------------------------------
# Stale-row pruning (membership changed without re-embedding)
# ---------------------------------------------------------------------------
def prune_stale_split(split: str) -> dict | None:
    """Drop npz rows whose image_path is absent from the split's CURRENT jsonl.

    Dataset membership can shrink without any image changing — e.g. the
    2026-06-11 license correction removed 27 rows (20,416 -> 20,389) from
    rows of license-rejected docs. The embeddings of the surviving rows are
    still valid, so a multi-hour re-embed is wasted work; this re-saves the
    npz filtered to current membership instead. Rows are matched on the
    resolved absolute ``image_path`` (the npz join key downstream consumers
    use). Never ADDS rows — a row in the jsonl but not the npz is reported
    so the caller knows a real (re-)embedding run is needed.

    Returns a summary dict, or None if the split has no npz on disk.
    No model / torch import — safe to run anywhere, any time.
    """
    res = load_npz_results(split)
    if res is None:
        return None
    current = {r["image_path"] for r in load_split_rows(split) if r["image_path"]}
    keep_idx = [i for i, r in enumerate(res["rows"]) if r["image_path"] in current]
    dropped = len(res["rows"]) - len(keep_idx)
    npz_paths = {r["image_path"] for r in res["rows"]}
    unembedded = len(current - npz_paths)
    if dropped:
        pruned = {
            "rows": [res["rows"][i] for i in keep_idx],
            "embeddings": res["embeddings"][keep_idx],
            "labels": [res["labels"][i] for i in keep_idx],
            "confidences": [res["confidences"][i] for i in keep_idx],
        }
        save_split_npz(split, pruned)
    return {
        "split": split,
        "kept": len(keep_idx),
        "dropped_stale": dropped,
        "current_rows_not_in_npz": unembedded,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap rows per split (smoke); 0 = all",
    )
    ap.add_argument(
        "--splits",
        nargs="*",
        default=list(DATASET_SPLITS) + [CATALOG_SPLIT],
    )
    ap.add_argument(
        "--prune-stale",
        action="store_true",
        help="drop npz rows no longer in the split jsonl (no embedding, no model load), then exit",
    )
    ap.add_argument(
        "--embed-missing",
        action="store_true",
        help="embed ONLY rows absent from each split's npz and append (a "
        "split with no npz embeds in full); existing rows are not recomputed",
    )
    args = ap.parse_args(argv)

    if args.prune_stale:
        for split in args.splits:
            if split == CATALOG_SPLIT:
                # The catalog stratum's source-of-truth is the catalog file,
                # not a dataset jsonl; it predates (and is superseded by) the
                # captionless_* splits. Nothing to prune against.
                print(f"[{split}] skipped (catalog stratum, not dataset-backed)")
                continue
            summary = prune_stale_split(split)
            if summary is None:
                print(f"[{split}] no npz on disk, nothing to prune")
                continue
            print(
                f"[{split}] kept={summary['kept']} "
                f"dropped_stale={summary['dropped_stale']} "
                f"current_rows_not_in_npz={summary['current_rows_not_in_npz']}"
            )
        return 0

    # Honor CPU placement by hiding CUDA before torch loads, unless cuda asked.
    if args.device == "cpu":
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from semantik_structure.figure_router import (
        embed_images,
        subtype_prototype_embeddings,
        zero_shot_subtype,
    )

    t0 = time.time()
    print(f"Building subtype prototype embeddings (device={args.device})...", flush=True)
    prompt_embs = subtype_prototype_embeddings()
    print(f"  prototypes shape {prompt_embs.shape}", flush=True)

    results: dict[str, dict] = {}
    for split in args.splits:
        rows = load_split_rows(split)
        if args.limit:
            rows = rows[: args.limit]
        existing = load_npz_results(split) if args.embed_missing else None
        if existing is not None:
            have = {r["image_path"] for r in existing["rows"]}
            rows = [r for r in rows if r["image_path"] not in have]
            print(
                f"[{split}] {len(rows)} rows missing from npz ({len(have)} already embedded)",
                flush=True,
            )
        else:
            print(f"[{split}] {len(rows)} rows", flush=True)
        res = embed_split(
            split,
            rows,
            batch_size=args.batch_size,
            prompt_embs=prompt_embs,
            embed_images_fn=embed_images,
            zero_shot_fn=zero_shot_subtype,
        )
        if existing is not None and existing["rows"]:
            res = {
                "rows": existing["rows"] + res["rows"],
                "embeddings": (
                    np.concatenate([existing["embeddings"], res["embeddings"]], axis=0)
                    if len(res["rows"])
                    else existing["embeddings"]
                ),
                "labels": existing["labels"] + res["labels"],
                "confidences": existing["confidences"] + res["confidences"],
                "missing": res["missing"],
                "corrupt": res["corrupt"],
                "n_in": existing["n_in"] + res["n_in"],
            }
        out = save_split_npz(split, res)
        print(
            f"[{split}] kept={len(res['rows'])} missing={res['missing']} "
            f"corrupt={res['corrupt']} -> {out}",
            flush=True,
        )
        results[split] = res

    # Agreement check + spot-check sample over ALL splits on disk (so a
    # partial --splits re-run still regenerates complete reports).
    report_results: dict[str, dict] = {}
    for split in list(DATASET_SPLITS) + [CATALOG_SPLIT]:
        loaded = results.get(split) or load_npz_results(split)
        if loaded is not None:
            report_results[split] = loaded
    agreement = compute_agreement(report_results)
    (OUT_DIR / "agreement_report.json").write_text(json.dumps(agreement, indent=2))

    # Spot-check sample.
    sample = build_spotcheck(report_results)
    sc_path = write_spotcheck(sample)

    # Summary.
    elapsed = time.time() - t0
    print("\n=== SUMMARY ===", flush=True)
    for split, res in results.items():
        lab_counts = Counter(res["labels"])
        print(
            f"{split}: kept={len(res['rows'])} missing={res['missing']} "
            f"corrupt={res['corrupt']} labels={dict(lab_counts)}",
            flush=True,
        )
    ov = agreement["overall_agreement"]
    print(
        f"\nAGREEMENT overall: {agreement['overall_match']}/{agreement['overall_total']} = {ov:.3f}"
        if ov is not None
        else "AGREEMENT overall: n/a",
        flush=True,
    )
    for cls, st in agreement["per_class"].items():
        a = st["agreement"]
        print(
            f"  {cls}: {st['match']}/{st['total']} = {a:.3f}" if a is not None else f"  {cls}: n/a",
            flush=True,
        )
    gate = "PASS" if (ov is not None and ov >= 0.80) else "BELOW GATE"
    print(f"\nDONE-CRITERION (>=0.80 overall): {gate}", flush=True)
    print(f"spot-check sample: {len(sample)} rows -> {sc_path}", flush=True)
    print(f"elapsed: {elapsed / 60:.1f} min (device={args.device})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
