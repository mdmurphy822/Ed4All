"""Build REAL-PDF annotation tasks for human-grade, real-PDF-ANCHORED structure gold.

Why this exists
---------------
The committed real-PDF eval set (``data/structure_dataset_realpdf*``) is built by
PROJECTING the rendered-HTML gold labels onto real-PDF blocks via the frozen
``data/alignment/structure_align.py`` aligner. The audit found those projections match at
<0.30 text similarity on a large fraction of blocks, so the headline real-PDF F1
is partly an EVAL ARTIFACT (the head is graded against labels that don't actually
describe the block it's reading).

This builder produces ANNOTATION TASKS instead: for each real PDF it extracts the
exact candidate blocks the inference path sees (column-correct order + the
single-source 20-dim layout vector), renders each page to a PNG, and seeds a
CANDIDATE label per block with the current committed council structure head. A
strong annotator (a multimodal model reading the rendered page, or a human)
CORRECTS those candidate labels in place; ``ingest_realpdf_labels.py`` then emits
the corrected, real-PDF-ANCHORED gold in the EXACT row schema the existing eval
(``eval/measure_realpdf_structure.py``) consumes — zero eval changes.

The annotation step happens BETWEEN this script and the ingest script and is NOT
part of either tool.

Three load-bearing guarantees (carried over from ``run_realpdf_build``)
  * FEATURE PARITY — the 20-dim layout vector is built by the SAME
    ``compute_span_layout_features`` the training rows + the eval rows use, whose
    formula is byte-identical to the council INFERENCE path
    (``semantik_structure/council/structure.py::_compute_span_layout``). So a label the
    head later mispredicts is attributable to the taxonomy/target gap, never to a
    feature skew between build-time and deploy-time.
  * COLUMN-CORRECT ORDER — extraction runs with ``SEMANTIK_COLUMN_ORDER=1`` AND
    the column sort is re-applied here defensively (the extract disk-cache key does
    NOT include the column-order flag, so a pre-fix cache could otherwise return
    gutter-interleaved order). The re-applied sort reuses the SAME
    ``reading_order.column_ids_for_bboxes`` the extractor uses, so it is byte-faithful
    to a flag-on extract.
  * LICENSE PARTITION + ZERO LEAKAGE — every doc is stamped shippable/internal via
    the SAME ``_license_partition`` map (arXiv gated per-paper on the cached license
    map; OpenStax is internal and excluded by default), and ``--eval-holdout-by-docid``
    (DEFAULT ON) excludes any doc whose rendered rows touched the head's train/val
    split, so the labeled set is genuinely held out.

Usage (tiny validation slice)::

    SEMANTIK_COLUMN_ORDER=1 python -m data.builders.build_realpdf_label_tasks \
        --sources arxiv --max-docs-per-source 2 --max-pages-per-doc 3

Output layout::

    <out-dir>/<source>/<doc_id>/task.json      # the annotation task (edit in place)
    <out-dir>/<source>/<doc_id>/page-NN.png     # one render per selected page
    <out-dir>/index.json                         # manifest of built tasks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- path bootstrap: make `data.*` / `semantik_structure.*` importable however invoked
_ROOT = Path(__file__).resolve().parents[2]  # the SemantiK repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Column-correct extraction is load-bearing for candidate-block order. Set it
# BEFORE importing the extractor so a fresh extract uses it; we ALSO re-apply the
# sort below for cache-independence.
os.environ.setdefault("SEMANTIK_COLUMN_ORDER", "1")

from data.builders.build_structure_data import (  # noqa: E402
    LAYOUT_FEATURE_DIM,
    LAYOUT_FEATURE_NAMES,
    LIST_NESTING_BUCKETS,
    PEDAGOGICAL_ROLE_NAMES,
    ROLE_NAMES,
    _block_in_any_table,
    _doc_id_for,
    _domain_for,
    _extract_with_timeout,
    _filter_shared_pages,
    _license_partition,
    _load_arxiv_license_map,
    _resolve_real_pdf,
    _train_val_docids,
    _DIR_TO_SOURCE,
    compute_span_layout_features,
    pedagogical_role_for,
)
from semantik_structure.extract_shared import (  # noqa: E402
    _DEFAULT_PAGE_WIDTH,
    _pypdfium2_render_page_to_image,
    extract_shared,
    extract_shared_cached,
)
from semantik_structure.reading_order import column_ids_for_bboxes  # noqa: E402

SCHEMA_VERSION = "realpdf-label-task/1.0"

# The 6 head labels every candidate block carries (mirrors StructureModel heads).
_HEAD_LABEL_KEYS = (
    "structural_role",
    "is_heading",
    "table_region",
    "is_image_block",
    "list_nesting",
    "pedagogical_role",
)


# ---------------------------------------------------------------------------
# Candidate pre-labeling — the current committed council structure head
# ---------------------------------------------------------------------------


class _HeadPredictor:
    """Loads the committed StructureModel (ALL six heads, incl. pedagogical_role)
    and predicts a candidate label dict per block. Mirrors
    ``eval/measure_realpdf_structure._load_head`` but additionally loads the
    pedagogical head (the eval loader skips it because it only scores
    structural_role). Fail-soft: a load error degrades to a deterministic seed."""

    def __init__(self, adapter_dir: Path, base_model: str, device: str, max_length: int):
        self.ok = False
        self.device = device
        self.max_length = max_length
        try:
            import torch
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            from transformers import AutoTokenizer

            from training.train_structure import StructureModel

            self.torch = torch
            model = StructureModel(base_model, dtype=torch.float32)
            sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
            sd = {k: v.float() for k, v in sd.items()}
            set_peft_model_state_dict(model.encoder, sd)
            heads = torch.load(
                str(adapter_dir / "heads.pt"), map_location="cpu", weights_only=False
            )
            model.head_role.load_state_dict(heads["head_role.state_dict"])
            model.head_is_heading.load_state_dict(heads["head_is_heading.state_dict"])
            model.head_table_region.load_state_dict(heads["head_table_region.state_dict"])
            if "head_is_image_block.state_dict" in heads:
                model.head_is_image_block.load_state_dict(
                    heads["head_is_image_block.state_dict"]
                )
            model.head_list_nesting.load_state_dict(heads["head_list_nesting.state_dict"])
            if "head_pedagogical_role.state_dict" in heads:
                model.head_pedagogical_role.load_state_dict(
                    heads["head_pedagogical_role.state_dict"]
                )
            model.layout_norm.load_state_dict(heads["layout_norm.state_dict"])
            model.layout_mlp.load_state_dict(heads["layout_mlp.state_dict"])
            model.to(device).eval()
            # Prefer the locally-shipped tokenizer dir (offline); else the base id.
            tok_dir = adapter_dir / "tokenizer"
            self.tok = AutoTokenizer.from_pretrained(
                str(tok_dir) if tok_dir.exists() else base_model
            )
            self.model = model
            self.ok = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                f"[label-tasks] WARNING: council head unavailable "
                f"({type(exc).__name__}: {exc}); seeding deterministic candidates",
                file=sys.stderr,
            )

    def predict(self, blocks: list[dict], batch_size: int = 16) -> list[dict]:
        """Return one candidate-label dict per block (parallel list)."""
        if not self.ok:
            return [self._deterministic(b) for b in blocks]
        torch = self.torch
        out: list[dict] = []
        for i in range(0, len(blocks), batch_size):
            batch = blocks[i : i + batch_size]
            texts = [(b.get("text") or "").strip() for b in batch]
            layouts = [b["layout"] for b in batch]
            enc = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            layout = torch.tensor(layouts, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                res = self.model(ids, mask, layout)
            # StructureModel forward keys the role logits as "role"; the label
            # field is "structural_role". The other five heads match by name.
            out_key = {
                "structural_role": "role",
                "is_heading": "is_heading",
                "table_region": "table_region",
                "is_image_block": "is_image_block",
                "list_nesting": "list_nesting",
                "pedagogical_role": "pedagogical_role",
            }
            argmax = {k: res[out_key[k]].argmax(-1).tolist() for k in _HEAD_LABEL_KEYS}
            for j in range(len(batch)):
                out.append({k: int(argmax[k][j]) for k in _HEAD_LABEL_KEYS})
        return out

    @staticmethod
    def _deterministic(block: dict) -> dict:
        """Cheap deterministic seed when the head can't load — annotator corrects
        it anyway. structural_role=paragraph; pedagogical_role from the frozen
        label regex; binaries off; list_nesting 0."""
        ped = pedagogical_role_for(block.get("text") or "")
        return {
            "structural_role": 0,  # paragraph
            "is_heading": 0,
            "table_region": 1 if block.get("in_table") else 0,
            "is_image_block": 0,
            "list_nesting": 0,
            "pedagogical_role": PEDAGOGICAL_ROLE_NAMES.index(ped),
        }


# ---------------------------------------------------------------------------
# Per-doc candidate-block extraction (column-correct, inference-path layout)
# ---------------------------------------------------------------------------


def _extract_candidate_blocks(
    shared: dict, *, max_pages: int
) -> tuple[list[dict], list[dict]]:
    """Build candidate blocks + per-page metadata from an ``extract_shared`` result.

    Iterates pages in document order, re-applies the column-major sort
    (cache-independent), and computes the single-source 20-dim layout vector per
    block. Returns ``(blocks, page_meta)``; ``page_meta`` keyed by the page's
    document-enumeration index (== ``shared['pages']`` index, the same value the
    existing realpdf rows carry as ``provenance.page``)."""
    blocks: list[dict] = []
    page_meta: list[dict] = []
    pages_used = 0
    for page_idx, page in enumerate(shared.get("pages", [])):
        if max_pages and pages_used >= max_pages:
            break
        merged = page.get("merged", {}).get("text_blocks", []) or []
        merged = [b for b in merged if (b.get("text") or "").strip()]
        if not merged:
            continue

        page_w = float(page.get("width") or _DEFAULT_PAGE_WIDTH)
        page_h = float(page.get("height", 792.0) or 792.0)
        sizes = [b["font_size"] for b in merged if b.get("font_size") is not None]
        page_median_fs = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
        heights = [
            b["bbox"][3] - b["bbox"][1]
            for b in merged
            if (b.get("bbox") and b["bbox"][3] > b["bbox"][1])
        ]
        page_median_h = sorted(heights)[len(heights) // 2] if heights else 12.0

        # Defensive column-major re-sort (byte-faithful to a flag-on extract).
        table_bboxes = [
            t.get("bbox")
            for t in page.get("pdfplumber", {}).get("tables", [])
            if t.get("bbox")
        ]
        col = column_ids_for_bboxes(
            [b["bbox"] for b in merged], page_w, float_bboxes=table_bboxes or None
        )
        order = sorted(
            range(len(merged)),
            key=lambda k: (col[k], merged[k]["bbox"][1], merged[k]["bbox"][0]),
        )
        merged_sorted = [merged[k] for k in order]

        real_page_num = int(page.get("page_num", page.get("page", page_idx + 1)))
        for n, block in enumerate(merged_sorted):
            in_table = _block_in_any_table(block, page)
            layout = compute_span_layout_features(
                block,
                page_w=page_w,
                page_h=page_h,
                page_median_fs=page_median_fs,
                page_median_h=page_median_h,
                in_table=in_table,
            )
            bbox = [float(v) for v in (list(block.get("bbox") or []) + [0, 0, 0, 0])[:4]]
            blocks.append(
                {
                    "block_id": f"p{page_idx:02d}b{n:03d}",
                    "page": page_idx,
                    "pdf_page_num": real_page_num,
                    "bbox": bbox,
                    "text": (block.get("text") or "").strip(),
                    "layout": layout,
                    "in_table": bool(in_table),
                }
            )
        page_meta.append(
            {
                "page": page_idx,
                "pdf_page_num": real_page_num,
                "pdf_width": page_w,
                "pdf_height": page_h,
            }
        )
        pages_used += 1
    return blocks, page_meta


def _render_pages(
    pdf_path: Path, page_meta: list[dict], doc_dir: Path, scale: float
) -> None:
    """Render each selected page to ``<doc_dir>/page-NN.png`` and stamp the image
    path + pixel dimensions onto its ``page_meta`` entry (fail-soft per page)."""
    for pm in page_meta:
        rel = f"page-{pm['page']:02d}.png"
        dest = doc_dir / rel
        try:
            img = _pypdfium2_render_page_to_image(pdf_path, pm["pdf_page_num"], scale=scale)
            img.save(str(dest))
            pm["image_path"] = rel
            pm["width"], pm["height"] = img.size
        except Exception as exc:  # pragma: no cover - render env dependent
            pm["image_path"] = None
            pm["render_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[label-tasks]   page {pm['page']} render failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build(args) -> None:
    arxiv_lic = _load_arxiv_license_map()
    train_val = _train_val_docids() if args.eval_holdout_by_docid else set()
    predictor = (
        _HeadPredictor(args.adapter_dir, args.base_model, args.device, args.max_length)
        if not args.no_prelabel
        else None
    )
    candidate_source = "head" if (predictor and predictor.ok) else "deterministic_fallback"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    drops: dict[str, int] = {}

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    for source in args.sources:
        src_dir = Path("data/pairs") / source
        if not src_dir.exists():
            print(f"[label-tasks] source dir missing, skipping: {src_dir}", file=sys.stderr)
            continue
        kept = 0
        files = sorted(src_dir.glob("*.json"))
        # Held-out (clean) docs FIRST so the cap fills the genuinely-held-out set.
        if train_val:
            canon = _DIR_TO_SOURCE.get(source, source)
            files = sorted(
                files,
                key=lambda pf: (
                    1 if (canon, _doc_id_for(canon, pf.stem)) in train_val else 0,
                    pf.name,
                ),
            )
        for pf in files:
            if args.max_docs_per_source and kept >= args.max_docs_per_source:
                break
            try:
                pair = json.loads(pf.read_text())
            except Exception:
                _drop("read_error")
                continue
            psource = pair.get("source") or source
            domain = _domain_for(psource)
            doc_id = _doc_id_for(psource, pf.stem)
            contaminated = (psource, doc_id) in train_val
            if args.eval_holdout_by_docid and contaminated and not args.include_contaminated:
                _drop("holdout_excluded")
                continue

            license_part = _license_partition(psource, doc_id, arxiv_lic)
            if license_part == "internal" and not args.include_internal:
                _drop("internal_excluded")
                continue

            pdf_path, page_range, err = _resolve_real_pdf(
                pair, psource, doc_id, args.realpdf_cache / psource
            )
            if pdf_path is None:
                _drop("no_real_pdf")
                continue
            try:
                shared = _extract_with_timeout(pdf_path, args.extract_timeout)
            except Exception:
                _drop("extract_error")
                continue
            shared = _filter_shared_pages(shared, page_range)

            blocks, page_meta = _extract_candidate_blocks(shared, max_pages=args.max_pages_per_doc)
            if not blocks:
                _drop("no_blocks")
                continue

            if predictor is not None:
                cands = predictor.predict(blocks, batch_size=args.batch_size)
                for b, c in zip(blocks, cands):
                    b["candidate_label"] = c
            else:
                for b in blocks:
                    b["candidate_label"] = _HeadPredictor._deterministic(b)
            for b in blocks:
                cl = b["candidate_label"]
                b["candidate_role_name"] = ROLE_NAMES[cl["structural_role"]]
                b["candidate_pedagogical_role_name"] = PEDAGOGICAL_ROLE_NAMES[cl["pedagogical_role"]]
                b["corrected"] = False  # annotator flips to True when reviewed
                b.pop("in_table", None)  # internal-only signal; not part of the task contract

            doc_dir = args.out_dir / source / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            _render_pages(pdf_path, page_meta, doc_dir, args.render_scale)

            task = {
                "schema_version": SCHEMA_VERSION,
                "doc_id": doc_id,
                "source": psource,
                "domain": domain,
                "license_partition": license_part,
                "holdout_clean": not contaminated,
                "page_range": list(page_range) if page_range else None,
                "render_scale": args.render_scale,
                "candidate_source": candidate_source,
                "label_space": {
                    "structural_role": list(ROLE_NAMES),
                    "pedagogical_role": list(PEDAGOGICAL_ROLE_NAMES),
                    "list_nesting": list(LIST_NESTING_BUCKETS),
                    "binaries": ["is_heading", "table_region", "is_image_block"],
                },
                "layout_feature_names": list(LAYOUT_FEATURE_NAMES),
                "pages": page_meta,
                "blocks": blocks,
            }
            (doc_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2))
            manifest.append(
                {
                    "doc_id": doc_id,
                    "source": psource,
                    "domain": domain,
                    "license_partition": license_part,
                    "holdout_clean": not contaminated,
                    "n_pages": len(page_meta),
                    "n_blocks": len(blocks),
                    "task_path": str((doc_dir / "task.json").relative_to(args.out_dir)),
                }
            )
            kept += 1
            print(
                f"[label-tasks] {source}/{doc_id}: {len(blocks)} blocks, "
                f"{len(page_meta)} pages -> {doc_dir}",
                file=sys.stderr,
            )

    (args.out_dir / "index.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_source": candidate_source,
                "sources": list(args.sources),
                "eval_holdout_by_docid": bool(args.eval_holdout_by_docid),
                "n_tasks": len(manifest),
                "layout_feature_dim": LAYOUT_FEATURE_DIM,
                "role_names": list(ROLE_NAMES),
                "pedagogical_role_names": list(PEDAGOGICAL_ROLE_NAMES),
                "drops": drops,
                "tasks": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"[LABEL-TASKS] built {len(manifest)} tasks "
        f"(candidate_source={candidate_source}) -> {args.out_dir}  drops={drops}",
        file=sys.stderr,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=["arxiv"],
                    help="pair source dirs under data/pairs (arxiv, forms, federal_register, cfr).")
    ap.add_argument("--max-docs-per-source", type=int, default=0,
                    help="0 = no cap.")
    ap.add_argument("--max-pages-per-doc", type=int, default=4,
                    help="Cap rendered+labeled pages per doc so labeling stays tractable (0 = no cap).")
    ap.add_argument("--out-dir", type=Path, default=Path("data/realpdf_label_tasks"))
    ap.add_argument("--realpdf-cache", type=Path, default=Path("data/realpdf_cache"),
                    help="Download cache for fetchable PDFs (forms / federal_register).")
    ap.add_argument("--render-scale", type=float, default=2.0)
    ap.add_argument("--extract-timeout", type=int, default=180)
    # Holdout ON by default (we want genuinely held-out docs); --no-holdout to disable.
    ap.add_argument("--eval-holdout-by-docid", dest="eval_holdout_by_docid",
                    action="store_true", default=True)
    ap.add_argument("--no-holdout", dest="eval_holdout_by_docid", action="store_false")
    ap.add_argument("--include-contaminated", action="store_true",
                    help="Also build docs that touched the head's train/val split (off by default).")
    ap.add_argument("--include-internal", action="store_true",
                    help="Also build internal-license docs (OpenStax etc.); shippable-only by default.")
    ap.add_argument("--no-prelabel", action="store_true",
                    help="Skip the council-head candidate seed (deterministic seed instead).")
    ap.add_argument("--adapter-dir", type=Path, default=Path("models/council/structure/final"))
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
