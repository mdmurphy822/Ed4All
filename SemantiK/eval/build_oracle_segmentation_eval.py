"""Build the ORACLE-SEGMENTATION structure eval set (the classification upper bound).

The committed exact-realistic eval (``data/structure_dataset_realistic_exact``)
scores the structure head on the FRAGMENTED ``_merge_page`` extraction output:
every paragraph shatters into many PDF line-blocks, each labelled by bbox
containment against the gold element box, and the head's gold-restricted real-PDF
macro-F1 lands at **0.547**. That number conflates two error sources — the head's
*classification* error AND the upstream *segmentation* error (a multi-line
paragraph arriving as N noisy fragments instead of one clean block).

This builder isolates classification by removing segmentation error entirely:
instead of extracting + overlap-matching PDF blocks, it emits **one row per gold
DOM element** — perfect segmentation, one block per role-bearing element — using
THAT element's own rendered box for the 20-dim layout vector and its own gold
role for the labels. Scoring the SAME head on this set yields the "oracle upper
bound": what the head could achieve if segmentation were perfect.

Apples-to-apples doc set
------------------------
The oracle MUST be measured on the SAME held-out documents (and the SAME realistic
presentation) as ``structure_dataset_realistic_exact`` so the only difference is
segmentation. The doc set + per-doc render profile/seed are read DIRECTLY from
that build's ``test.jsonl`` / ``test_in_training.jsonl`` (each row carries
``source`` / ``pair`` / ``doc_id`` / ``domain`` / ``render_profile`` /
``render_seed`` / ``holdout_clean``). For every distinct ``(source, pair)`` doc we
locate its gold pair JSON under ``data/pairs/<dir>/<pair>.json``, reproduce the
EXACT augmented HTML via ``augment_html(output_html, render_profile, render_seed)``
(byte-for-byte the presentation realistic_exact rendered), render it with
per-element boxes, and emit oracle rows. The set is therefore identical to the
realistic_exact set BY CONSTRUCTION.

Reuse (no reimplementation)
---------------------------
* rendering           — ``data.augmentation.render_capture.render_with_boxes`` / ``RenderCaptureSession``
* gold-record recovery — ``data.augmentation.render_capture._gold_records_with_rcid``
                         (TAG_TO_ROLE / list-nesting / table+image ancestry, reused verbatim)
* layout features     — ``data.builders.build_structure_data.compute_span_layout_features``
                         (the SINGLE source of truth, 20-dim)
* label space         — ``ROLE_TO_ID`` filter + ``pedagogical_role_for`` /
                         ``PEDAGOGICAL_ROLE_TO_ID`` (identical to the labeler's matched branch)
* presentation        — ``augment_html`` (same profile/seed as realistic_exact)

Output schema is the EXACT eval row ``eval/measure_realpdf_structure.py`` consumes
(``--realpdf-roots``), so the scorer runs UNCHANGED:

    python -m eval.build_oracle_segmentation_eval --out-dir data/structure_dataset_oracle_seg
    python -m eval.measure_realpdf_structure --realpdf-roots data/structure_dataset_oracle_seg

Documented gaps / honesty
-------------------------
* **Font features are 0 on gold element boxes.** ``render_with_boxes`` captures
  each element's ``getBoundingClientRect`` + text but NOT its computed font-size /
  bold / italic. The synthetic span fed to ``compute_span_layout_features`` carries
  only the (pt-mapped) box + text, so the 4 font-dependent layout dims
  (``fs/12``, ``fs/median_fs``, ``is_bold``, ``is_italic``) are 0 for every oracle
  row. The dominant text signal + the 16 geometry/text-shape dims are intact, so
  the oracle is a (slight, font-axis-only) CONSERVATIVE bound — it cannot
  overstate the ceiling. The realistic_exact rows DO carry real font from PDF
  extraction; this is the one input asymmetry, recorded in ``coverage_report.json``.
* **``table_region`` derives from the gold ``<table>`` ancestry only** (no
  pdfplumber grid detection runs — there is no extracted PDF here), matching the
  labeler's matched-branch ``in_table=in_table_html`` path.
* **``SEMANTIK_COLUMN_ORDER`` is irrelevant to the oracle** (it governs the
  PDF-block reading-order re-sort, which the oracle never runs — gold elements are
  emitted in DOM order). **``SEMANTIK_RENDER_AUGMENT`` is also irrelevant**: the
  augmented presentation is reproduced PER-DOC from the dataset's stored
  ``render_profile`` / ``render_seed``, NOT from the env, so the oracle output is
  reproducible regardless of either flag's value.

CPU/Playwright only — no torch, no GPU, no model load. Scoring is a separate step.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the rendering + gold-record machinery (no reimplementation).
from data.augmentation.render_capture import (
    PAGE_W_PX,
    RenderCaptureSession,
    _gold_records_with_rcid,
)

# Reuse the SINGLE-source-of-truth feature + label logic.
from data.builders.build_structure_data import (
    PEDAGOGICAL_ROLE_TO_ID,
    ROLE_TO_ID,
    STRUCTURE_ALIGN_SCHEMA_VERSION,
    augment_html,
    compute_span_layout_features,
    pedagogical_role_for,
)

# CSS-px -> PDF-pt affine for the single-tall-page render. render_capture pins the
# viewport to PAGE_W_PX (816px == 8.5in @ 96dpi == 612pt), so the scale is exactly
# 72/96 = 0.75 (the labeler reads it back from the extracted PDF page width; here
# there is no extracted PDF, so we use the documented constant). See
# data/augmentation/render_capture.py module docstring.
_PX_TO_PT = 72.0 / 96.0

_DEFAULT_REALISTIC_EXACT_DIR = Path("data/structure_dataset_realistic_exact")
_PAIRS_ROOT = Path("data/pairs")


# ---------------------------------------------------------------------------
# Doc-set resolution (apples-to-apples with realistic_exact)
# ---------------------------------------------------------------------------


def _iter_rows(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def resolve_doc_set(realistic_exact_dir: Path) -> list[dict]:
    """Read the realistic_exact eval rows and collapse to the distinct doc set,
    preserving each doc's render profile/seed + holdout flag so the oracle renders
    the IDENTICAL presentation. ``set_kind`` mirrors the scorer's two files:
    ``heldout`` (test.jsonl) vs ``in_training`` (test_in_training.jsonl)."""
    docs: dict[tuple, dict] = {}
    for set_kind, fname in (("heldout", "test.jsonl"), ("in_training", "test_in_training.jsonl")):
        for r in _iter_rows(realistic_exact_dir / fname):
            key = (r.get("source"), r.get("pair"))
            if key in docs:
                continue
            docs[key] = {
                "source": r.get("source"),
                "pair": r.get("pair"),
                "doc_id": r.get("doc_id"),
                "domain": r.get("domain"),
                "render_profile": r.get("render_profile") or "clean",
                "render_seed": int(r.get("render_seed") or 0),
                "holdout_clean": bool(r.get("holdout_clean", set_kind == "heldout")),
                "set_kind": set_kind,
            }
    # Stable, deterministic order.
    return [docs[k] for k in sorted(docs, key=lambda k: (str(k[0]), str(k[1])))]


def _locate_pair_json(source: str, pair: str) -> Path | None:
    """Resolve a doc's gold pair JSON. Try the canonical ``data/pairs/<source>/``
    directory first, then fall back to a glob over all pair dirs (source-name vs
    dir-name can differ, e.g. forms/pdf_form), then None."""
    direct = _PAIRS_ROOT / (source or "") / f"{pair}.json"
    if direct.exists():
        return direct
    hits = sorted(_PAIRS_ROOT.glob(f"*/{pair}.json"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Oracle row emission for one doc
# ---------------------------------------------------------------------------


def oracle_rows_for_doc(meta: dict, capture) -> list[dict]:
    """Emit one oracle row per role-bearing gold element with an in-vocab role.

    Mirrors :func:`data.augmentation.render_capture.label_blocks_by_overlap`'s MATCHED branch,
    but the "block" IS the gold element (perfect segmentation): the element's own
    pt-mapped box + gold text drive ``compute_span_layout_features`` and its own
    gold role drives the labels."""
    gold = _gold_records_with_rcid(capture.attributed_html)
    box_by_rcid = {e.rc_id: e for e in capture.elements}

    # Page geometry in pt for the single tall page (matches the realistic_exact
    # per-page normalization: page dims are the rendered PDF page dims).
    page_w_pt = float(capture.page_w_px) * _PX_TO_PT
    page_h_pt = max(1.0, float(capture.scroll_height_px) * _PX_TO_PT)

    # Resolve each gold element's pt box up front so page_median_h can be a real
    # median of element heights (no font available -> page_median_fs default 12).
    resolved: list[tuple[dict, list[float]]] = []
    for rec in gold:
        eb = box_by_rcid.get(rec["rc_id"])
        if eb is None or eb.w <= 0 or eb.h <= 0:
            continue  # display:none / zero-size / uncaptured -> no row (labeler skip)
        bbox_pt = [
            eb.x * _PX_TO_PT,
            eb.y * _PX_TO_PT,
            (eb.x + eb.w) * _PX_TO_PT,
            (eb.y + eb.h) * _PX_TO_PT,
        ]
        resolved.append((rec, bbox_pt))

    heights = [b[3] - b[1] for _, b in resolved if b[3] > b[1]]
    page_median_h = sorted(heights)[len(heights) // 2] if heights else 12.0

    rows: list[dict] = []
    for rec, bbox_pt in resolved:
        text = rec["text"]
        if not text:
            continue
        role = rec["role"]
        # Only emit a row for roles in the active structural-role head vocab
        # (mirrors the labeler / fuzzy-path ROLE_TO_ID filter); others dropped.
        sr_id = ROLE_TO_ID.get(role) if role in ROLE_TO_ID else None
        if sr_id is None:
            continue

        in_table = bool(rec.get("in_table_html"))
        # Synthetic span: the element's OWN box + text. font_size/bold/italic are
        # unavailable on gold element boxes (documented gap) -> the 4 font dims = 0.
        span = {"bbox": bbox_pt, "text": text}
        layout_vec = compute_span_layout_features(
            span,
            page_w=page_w_pt,
            page_h=page_h_pt,
            page_median_fs=12.0,
            page_median_h=page_median_h,
            in_table=in_table,
        )

        tag = rec["tag"]
        labels = {
            "structural_role": sr_id,
            "is_heading": 1 if (tag.startswith("h") and tag[1:].isdigit()) else 0,
            "table_region": 1 if in_table else 0,
            "is_image_block": 1 if rec.get("in_image_block_html") else 0,
            "list_nesting": int(rec.get("list_nesting", 0)),
            "pedagogical_role": PEDAGOGICAL_ROLE_TO_ID[pedagogical_role_for(text)],
        }
        rows.append(
            {
                "text": text,
                "layout": layout_vec,
                "labels": labels,
                "html_tag": tag,
                "schema_version": STRUCTURE_ALIGN_SCHEMA_VERSION,
                "source": meta["source"],
                "pair": meta["pair"],
                "realpdf": True,
                "realistic_render": True,
                "exact_label": True,
                "oracle_segmentation": True,
                "domain": meta["domain"],
                "doc_id": meta["doc_id"],
                "holdout_clean": meta["holdout_clean"],
                # Perfectly aligned by construction (one row per gold element).
                "aligned_gold_fraction": 1.0,
                "render_profile": meta["render_profile"],
                "render_seed": meta["render_seed"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(args) -> None:
    docs = resolve_doc_set(args.realistic_exact_dir)
    if not docs:
        raise SystemExit(
            f"no docs found under {args.realistic_exact_dir} — point --realistic-exact-dir "
            "at a built structure_dataset_realistic_exact (test.jsonl)."
        )
    if args.limit:
        docs = docs[: args.limit]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_rows: list[dict] = []
    in_training_rows: list[dict] = []
    drops: Counter = Counter()
    kept_doc_ids: set[str] = set()
    role_hist: Counter = Counter()
    per_domain_docs: Counter = Counter()
    profile_counts: Counter = Counter()
    inv_role = {v: k for k, v in ROLE_TO_ID.items()}

    # One Chromium for the whole build (batched box-capture renders).
    with RenderCaptureSession(page_w_px=PAGE_W_PX) as session:
        for meta in docs:
            pf = _locate_pair_json(meta["source"], meta["pair"])
            if pf is None:
                drops["pair_json_missing"] += 1
                print(f"[oracle] pair json missing, skipping: {meta['source']}/{meta['pair']}",
                      file=sys.stderr)
                continue
            try:
                pair = json.loads(pf.read_text())
            except Exception:
                drops["read_error"] += 1
                continue
            output_html = pair.get("output_html")
            if not output_html:
                drops["no_output_html"] += 1
                continue

            # Reproduce the EXACT presentation realistic_exact rendered.
            rendered_html = augment_html(
                output_html, meta["render_profile"], meta["render_seed"]
            )
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    capture = session.render(rendered_html, tmpdir=Path(tmp))
                except Exception as exc:
                    drops["render_error"] += 1
                    print(f"[oracle] render error {meta['pair'][:42]}: {exc}", file=sys.stderr)
                    continue
                rows = oracle_rows_for_doc(meta, capture)

            if not rows:
                drops["no_rows"] += 1
                continue

            if meta["holdout_clean"]:
                heldout_rows.extend(rows)
            else:
                in_training_rows.extend(rows)
            kept_doc_ids.add(meta["doc_id"])
            per_domain_docs[meta["domain"]] += 1
            profile_counts[meta["render_profile"]] += 1
            for r in rows:
                role_hist[inv_role.get(r["labels"]["structural_role"], "?")] += 1
            print(
                f"[oracle] {meta['pair'][:42]:42}  profile={meta['render_profile']:24}  "
                f"rows={len(rows):4}  set={meta['set_kind']}",
                file=sys.stderr,
            )

    # Write the eval rows (scorer reads test.jsonl + test_in_training.jsonl).
    with (out_dir / "test.jsonl").open("w") as f:
        for r in heldout_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "test_in_training.jsonl").open("w") as f:
        for r in in_training_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_rows = len(heldout_rows) + len(in_training_rows)
    coverage = {
        "build": "oracle_segmentation",
        "schema_version": STRUCTURE_ALIGN_SCHEMA_VERSION,
        "description": (
            "Classification upper bound: one row per gold DOM element (perfect "
            "segmentation), scored by the SAME structure head as realistic_exact. "
            "Compare against the realistic_exact gold-restricted macro-F1 (0.547) to "
            "isolate classification error from segmentation error."
        ),
        "source_doc_set": str(args.realistic_exact_dir),
        "n_docs": len(kept_doc_ids),
        "n_rows": n_rows,
        "n_heldout_rows": len(heldout_rows),
        "n_in_training_rows": len(in_training_rows),
        "doc_ids": sorted(kept_doc_ids),
        "role_histogram": dict(sorted(role_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_domain_docs": dict(per_domain_docs),
        "render_profile_counts": dict(profile_counts),
        "drop_ledger": dict(drops),
        "px_to_pt_scale": _PX_TO_PT,
        "notes": {
            "presentation": (
                "Augmented HTML reproduced PER-DOC from the realistic_exact "
                "render_profile/render_seed (NOT from SEMANTIK_RENDER_AUGMENT), so the "
                "oracle layout distribution matches realistic_exact and is env-independent."
            ),
            "column_order": (
                "SEMANTIK_COLUMN_ORDER is irrelevant: the oracle emits gold elements in "
                "DOM order and never runs the PDF-block reading-order re-sort."
            ),
            "font_features_zero": (
                "render_with_boxes captures box+text but not computed font-size/weight/"
                "style, so the 4 font-dependent layout dims (fs/12, fs/median_fs, is_bold, "
                "is_italic) are 0 for every oracle row. The dominant text signal + 16 "
                "geometry/text-shape dims are intact -> a conservative (font-axis-only) "
                "bound that cannot overstate the ceiling. realistic_exact rows DO carry "
                "real font from PDF extraction (the one input asymmetry)."
            ),
            "table_region": (
                "Derived from gold <table> ancestry only (no pdfplumber grid detection), "
                "matching the labeler's matched-branch in_table=in_table_html path."
            ),
        },
    }
    (out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))

    print(
        f"\n[ORACLE] docs={len(kept_doc_ids)}  rows={n_rows} "
        f"(heldout={len(heldout_rows)}, in_training={len(in_training_rows)})  "
        f"drops={dict(drops)}\n  by_domain={dict(per_domain_docs)}  "
        f"profiles={dict(profile_counts)}",
        file=sys.stderr,
    )
    print(f"[oracle] wrote {out_dir / 'test.jsonl'} + coverage_report.json", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/structure_dataset_oracle_seg"),
        help="output root for test.jsonl + coverage_report.json",
    )
    ap.add_argument(
        "--realistic-exact-dir",
        type=Path,
        default=_DEFAULT_REALISTIC_EXACT_DIR,
        help="the built realistic_exact dataset whose held-out doc set + per-doc "
        "render profile/seed are reproduced (apples-to-apples).",
    )
    ap.add_argument("--limit", type=int, default=0, help="cap number of docs (0 = all)")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
