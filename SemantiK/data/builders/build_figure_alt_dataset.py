"""Assemble the (figure image -> caption) dataset from the catalog + fetched images.

Plans/09 §3a, task #22. Joins ``data/figure_catalog/catalog.jsonl`` (caption
ground truth + image_ref) with the figure images fetched to
``data/figure_images/`` and emits a doc-disjoint train/val/test split.

Ground truth = the ``figcaption`` (ar5iv ``alt`` is the "Refer to caption"
placeholder; OpenStax ``alt_raw`` carries real long descriptions and is kept as
a secondary target where present). Pair-aware split by ``doc_id`` (reusing
``data.common._splits.stable_split_for_id``) so figures from one paper never straddle
the boundary. CPU only — no model, no network.

Captionless / decorative path (Plans/09 §2 item 2, WCAG 1.1.1)
--------------------------------------------------------------
``--include-captionless`` additionally emits ``captionless_{split}.jsonl``
from catalog records carrying the ``captionless: true`` marker (produced by
``build_figure_catalog.py --include-captionless``). Each row's heuristic
``decorative_class`` is REFINED with the actual pixel dimensions (Pillow) now
that the image is on disk:

* ``decorative``           -> ``target_alt: ""`` (the WCAG-correct empty alt;
                              ``is_decorative: true``)
* ``uncaptioned_content``  -> ``target_alt: null`` + ``needs_generated_alt:
                              true`` (content-bearing, no ground truth — the
                              gated "generate" path, never alt="")

Default behavior (no flag) is byte-identical to the historical builder:
captionless rows are dropped (``dropped_no_caption``) and only
train/val/test.jsonl are written.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from data.builders.build_figure_catalog import EXTREME_ASPECT, TINY_MAX_PX
from data.common._splits import stable_split_for_id

CATALOG = Path("data/figure_catalog/catalog.jsonl")
ARXIV_LICENSE_MAP = Path("data/figure_images/_arxiv_license_map.json")
MANIFEST = Path("data/figure_images/_fetch_manifest.jsonl")
OUT = Path("data/figure_alt_dataset")
FIGURE_IMAGES = Path("data/figure_images")
SOURCES = ("arxiv", "pmc", "openstax")
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10


def _arxiv_image_path(source: str, doc_id: str, ref: str) -> Path:
    """Resolve a catalog reference to its fetched arXiv image path."""
    tail = (
        ref.split("/assets/", 1)[-1]
        if "/assets/" in ref
        else ref.rsplit("/", 1)[-1]
    )
    tail = re.sub(r"[^A-Za-z0-9._/-]", "_", tail).lstrip("/")
    safe_doc_id = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)
    return FIGURE_IMAGES / source / safe_doc_id / tail


def _manifest_index(manifest: Path = MANIFEST) -> dict[tuple[str, str, str], str]:
    """Map ``(source, doc_id, on-disk stem) -> local_path`` for fetched images.

    Used by non-arxiv sources whose ``image_ref`` is the bare stem (PMC) or a
    CDN URL whose last path segment is the stem (OpenStax). The on-disk
    filename's stem matches ``image_ref`` (PMC) or ``Path(image_ref).stem``
    (OpenStax), so we key on it.
    """
    idx: dict[tuple[str, str, str], str] = {}
    if not manifest.exists():
        return idx
    for line in manifest.open():
        if not line.strip():
            continue
        r = json.loads(line)
        lp = r.get("local_path", "")
        if r.get("status") != "ok":
            continue
        for src in ("pmc", "openstax"):
            if f"/{src}/" in lp:
                idx[(src, str(r.get("doc_id")), Path(lp).stem)] = lp
                break
    return idx


def _resolve_image(rec: dict, mf_idx: dict[tuple[str, str, str], str]) -> Path | None:
    src = rec["source"]
    doc_id = str(rec.get("doc_id", "unknown"))
    ref = rec.get("image_ref") or ""
    if src in ("pmc", "openstax"):
        # PMC ref IS the stem; OpenStax ref is a URL whose last segment is the stem.
        stem = ref if src == "pmc" else Path(ref).stem
        lp = mf_idx.get((src, doc_id, stem))
        return Path(lp) if lp else None
    return _arxiv_image_path(src, doc_id, ref)


def refine_decorative_with_pixels(
    rec: dict, img_path: Path
) -> tuple[str, list[str], int | None, int | None]:
    """Refine the catalog's offline decorative class with real pixel dims.

    The catalog classified from HTML attributes (often absent). With the image
    on disk we can apply the tiny / extreme-aspect rules to the TRUE pixels.
    Signals are additive: pixel evidence can promote uncaptioned_content ->
    decorative, never the reverse. Unreadable images (e.g. SVG) keep the
    catalog classification and get null dims.
    """
    signals = list(rec.get("decorative_signals") or [])
    w = h = None
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 — SVG/corrupt: keep offline class
        pass
    if w and h:
        if min(w, h) <= TINY_MAX_PX:
            signals.append("tiny_pixels")
        if max(w, h) / min(w, h) >= EXTREME_ASPECT:
            signals.append("extreme_aspect_pixels")
    cls = "decorative" if signals else "uncaptioned_content"
    return cls, signals, w, h


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--include-captionless",
        action="store_true",
        help="ALSO write captionless_{split}.jsonl rows (decorative -> "
        'target_alt: ""). Requires a catalog built with '
        "--include-captionless. Default OFF: output unchanged.",
    )
    ap.add_argument(
        "--arxiv-license-map",
        type=Path,
        default=ARXIV_LICENSE_MAP,
        help="per-doc arXiv license map written by fetch_ar5iv_figure_assets "
        "--verify-license. Captionless arXiv rows REQUIRE a commercial-OK "
        "entry here (fail closed): the catalog's blanket 'CC (ar5iv "
        "no-problem subset)' tag is a corpus label, not a per-paper license.",
    )
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.catalog.open() if line.strip()]
    mf_idx = _manifest_index(args.manifest)
    arxiv_lic: dict[str, dict] = {}
    if args.include_captionless and args.arxiv_license_map.exists():
        arxiv_lic = json.loads(args.arxiv_license_map.read_text(encoding="utf-8"))
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    kept: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    capless: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    capless_seen: set[str] = set()  # dedupe repeated chrome/shared images
    stats = Counter()
    cap_tokens: list[int] = []
    for r in rows:
        stats["catalog_total"] += 1
        if r.get("source") not in SOURCES:
            continue
        stats["in_source"] += 1
        cap = (r.get("figcaption") or "").strip()
        if not cap:
            stats["dropped_no_caption"] += 1
            if args.include_captionless and r.get("captionless"):
                row_license = r.get("license")
                if r["source"] == "arxiv":
                    # Per-doc veto (feedback_license_policy): only docs the
                    # OAI-verified map marks commercial-OK may contribute
                    # captionless rows. No entry -> fail closed.
                    ent = arxiv_lic.get(str(r.get("doc_id")))
                    if not (ent and ent.get("commercial_ok")):
                        stats["captionless_license_vetoed"] += 1
                        continue
                    # Stamp the VERIFIED per-paper license URL, not the
                    # blanket corpus tag.
                    row_license = ent.get("license_url") or row_license
                img = _resolve_image(r, mf_idx)
                if not (img and img.exists() and img.stat().st_size > 0):
                    stats["captionless_no_image"] += 1
                    continue
                if str(img) in capless_seen:
                    stats["captionless_dup_image"] += 1
                    continue
                capless_seen.add(str(img))
                cls, signals, w, h = refine_decorative_with_pixels(r, img)
                decorative = cls == "decorative"
                split = stable_split_for_id(str(r["doc_id"]), TRAIN_FRAC, VAL_FRAC)
                capless[split].append(
                    {
                        "source": r["source"],
                        "doc_id": r["doc_id"],
                        "figure_index": r.get("figure_index"),
                        "captionless_index": r.get("captionless_index"),
                        "image_path": str(img),
                        "caption": None,  # no caption ground truth
                        "alt_raw": r.get("alt_raw"),
                        "surrounding_context": r.get("surrounding_context", ""),
                        "license": row_license,
                        "split": split,
                        "captionless": True,
                        "standalone_img": bool(r.get("standalone_img")),
                        "decorative_class": cls,
                        "decorative_signals": signals,
                        "is_decorative": decorative,
                        # WCAG 1.1.1: decorative -> empty alt, NOT a caption.
                        "target_alt": "" if decorative else None,
                        "needs_generated_alt": not decorative,
                        "image_width": w,
                        "image_height": h,
                    }
                )
                stats["captionless_kept"] += 1
                stats[f"captionless_{r['source']}"] += 1
                stats[f"captionless_class_{cls}"] += 1
            continue
        img = _resolve_image(r, mf_idx)
        if not (img and img.exists() and img.stat().st_size > 0):
            stats["dropped_no_image"] += 1
            continue
        stats["paired"] += 1
        stats[f"paired_{r['source']}"] += 1
        cap_tokens.append(len(cap.split()))
        split = stable_split_for_id(str(r["doc_id"]), TRAIN_FRAC, VAL_FRAC)
        rec = {
            "source": r["source"],
            "doc_id": r["doc_id"],
            "figure_index": r.get("figure_index"),
            "image_path": str(img),
            "caption": cap,  # primary target (ground truth)
            "alt_raw": r.get("alt_raw"),  # secondary (real for OpenStax)
            "surrounding_context": r.get("surrounding_context", ""),
            "license": r.get("license"),
            "split": split,
        }
        kept[split].append(rec)

    for split, recs in kept.items():
        with (out / f"{split}.jsonl").open("w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if args.include_captionless:
        # Separate files: the captioned train/val/test.jsonl stay byte-stable.
        for split, recs in capless.items():
            with (out / f"captionless_{split}.jsonl").open("w") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cap_tokens.sort()

    def pct(p: float) -> int:
        return cap_tokens[int(p * (len(cap_tokens) - 1))] if cap_tokens else 0

    report = {
        "sources": list(SOURCES),
        "splits": {k: len(v) for k, v in kept.items()},
        "total_paired": stats["paired"],
        "stats": dict(stats),
        "caption_tokens": {
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": cap_tokens[-1] if cap_tokens else 0,
        },
        "doc_disjoint_split": "stable_split_for_id(doc_id, 0.80, 0.10)",
    }
    if args.include_captionless:
        report["captionless_splits"] = {k: len(v) for k, v in capless.items()}
        report["captionless_per_class"] = {
            cls: sum(1 for v in capless.values() for r in v if r["decorative_class"] == cls)
            for cls in ("decorative", "uncaptioned_content")
        }
    (out / "coverage_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
