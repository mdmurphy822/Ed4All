"""Region-level eval of Structure's table_region binary head.

Question: when we aggregate Structure's per-span ``table_region``
softmax inside a pdfplumber-detected ``TableCandidate``, does the
region-level decision agree with HTML ground truth about whether that
candidate is a real table?

If P/R/F1 at the region level is ≥ ~0.90, the standalone Phase 3d
``BERT-TableDetector`` becomes redundant: aggregating Structure's
binary head over candidate spans is equivalent to (or better than)
training a separate region-level model. If region-level eval is
weak — e.g., precision drops because Structure over-fires on
multi-column text that pdfplumber also flagged — Phase 3d stays.

Ground truth: ``html_meta["in_table_html"]`` from
``data.build_structure_data.extract_html_blocks`` (HTML-only —
"the aligned HTML element has a ``<table>`` ancestor"). This is
NOT the Phase 3b training-time label, which conflates pdfplumber
detection with HTML truth and would be circular here.

Usage:
    .venv/bin/python -m scripts.eval_table_region_at_region_level \\
        --limit 20 --source arxiv

    # Or by explicit pair id:
    .venv/bin/python -m scripts.eval_table_region_at_region_level \\
        --pair-ids 1105_4789__13_formatted_lob_snapshots ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dart_semantic.extract_shared import extract_shared, extract_shared_cached
from dart_semantic.features import featurize_from_shared
from dart_semantic.text_utils import jaccard_overlap
from dart_semantic.validate import HtmlValidator

# Phase 3b builder reuse — same html walker + alignment
from data.build_structure_data import extract_html_blocks


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def rank_pairs_by_table_density(
    per_pair_dir: Path,
    *,
    source: str | None,
    min_table_spans: int = 5,
) -> list[tuple[str, str, int, int]]:
    """Return list of (pair_id, source, total_spans, table_spans), sorted by
    table_spans desc. Reads existing Phase 3b per-pair training
    output — fast (no PDF processing)."""
    out: list[tuple[str, str, int, int]] = []
    for f in sorted(per_pair_dir.glob("*.jsonl")):
        pair_id = f.stem
        total = 0
        table = 0
        src = "?"
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                src = r.get("source", "?")
                if source and src != source:
                    break
                total += 1
                if r["labels"].get("table_region") == 1:
                    table += 1
        if source and src != source:
            continue
        if table < min_table_spans:
            continue
        out.append((pair_id, src, total, table))
    out.sort(key=lambda t: -t[3])
    return out


def find_pair_json(pair_id: str, source: str) -> Path | None:
    """Walk data/pairs/<source>/ for the json with stem matching pair_id."""
    candidate = Path(f"data/pairs/{source}/{pair_id}.json")
    if candidate.exists():
        return candidate
    # Fallback: scan all source dirs
    for sd in Path("data/pairs").iterdir():
        if not sd.is_dir():
            continue
        c = sd / f"{pair_id}.json"
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Per-pair processing — extract + structure inference + html GT
# ---------------------------------------------------------------------------


def _block_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_contains(outer: list[float], cx: float, cy: float) -> bool:
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _align_block_to_html(
    text: str,
    html_blocks: list[dict],
    cursor: int,
    *,
    window: int = 8,
    threshold: float = 0.30,
) -> tuple[int, dict | None]:
    """Mirror of Phase 3b alignment: sliding-window Jaccard. Returns
    ``(new_cursor, html_meta or None)``."""
    best_idx = -1
    best_score = 0.0
    for j in range(max(0, cursor - 2),
                   min(len(html_blocks), cursor + window)):
        s = jaccard_overlap(text, html_blocks[j]["text"])
        if s > best_score:
            best_score = s
            best_idx = j
    if best_idx < 0 or best_score < threshold:
        return cursor, None
    return max(cursor, best_idx + 1), html_blocks[best_idx]


def process_pair(
    pair_path: Path,
    structure_runtime,
    *,
    validator: HtmlValidator | None,
    tmp_dir: Path,
) -> dict:
    """Load pair, run extract_shared + Structure inference + HTML GT,
    return per-block records keyed by page+bbox plus per-page table
    candidates.

    Returns:
        {
            "pair_id": str,
            "blocks": [{text, page, bbox, gt_html, pred_p_table}],
            "tables": [{page, bbox}],   # pdfplumber TableCandidates
        }
    """
    peft_model, tok, heads, device = structure_runtime

    pair = json.loads(pair_path.read_text())
    output_html = pair.get("output_html") or ""

    # Get the PDF
    local_pdf = pair.get("local_pdf")
    if local_pdf and Path(local_pdf).exists():
        shared = extract_shared_cached(Path(local_pdf))
    elif validator is not None and output_html:
        tmp_pdf = tmp_dir / f"{pair_path.stem}.pdf"
        validator.render_pdf(output_html, tmp_pdf)
        shared = extract_shared(tmp_pdf)
    else:
        return {"pair_id": pair_path.stem, "error": "no pdf source",
                "blocks": [], "tables": []}

    # HTML ground truth (in_table_html per html-block, document order)
    html_blocks = extract_html_blocks(output_html) if output_html else []

    # Run Structure adapter on featurized blocks
    features = featurize_from_shared(shared)
    pred_probs = _run_structure(features, peft_model, tok, heads, device)
    # pred_probs[i] = {"role": [..6..], "is_heading": float (P=1),
    #                  "table_region": float (P=1)}
    feature_idx_by_pageblock: dict[tuple[int, int], int] = {}
    for i, fb in enumerate(features):
        raw = getattr(fb, "raw", None)
        page = int(getattr(raw, "page", 0) or 0)
        # Use bbox tuple as block key
        bbox = tuple(getattr(raw, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))
        feature_idx_by_pageblock[(page, bbox)] = i

    # Walk pages exactly like Phase 3b builder, align to html_blocks for GT,
    # and tag each block with prediction + bbox + page.
    blocks_out: list[dict] = []
    tables_out: list[dict] = []
    html_cursor = 0

    for page in shared.get("pages", []):
        page_num = int(page.get("page_num", page.get("number", 0)) or 0)
        merged = page.get("merged", {}).get("text_blocks", []) or []
        if not merged:
            continue

        # pdfplumber TableCandidates on this page
        for t in (page.get("pdfplumber", {}).get("tables", []) or []):
            tbb = t.get("bbox") or []
            if len(tbb) == 4:
                tables_out.append({"page": page_num, "bbox": list(tbb)})

        merged_sorted = sorted(
            merged, key=lambda b: (b["bbox"][1], b["bbox"][0])
        )

        for block in merged_sorted:
            text = (block.get("text") or "").strip()
            if not text:
                continue
            bbox = block.get("bbox") or [0, 0, 0, 0]

            # GT alignment via Jaccard
            html_cursor, html_meta = _align_block_to_html(
                text, html_blocks, html_cursor
            )
            gt_html = (
                bool(html_meta.get("in_table_html"))
                if html_meta is not None else None
            )

            # Find predicted prob for this block via featurize index
            key = (page_num, tuple(bbox))
            fidx = feature_idx_by_pageblock.get(key)
            if fidx is None:
                # Fallback: match by bbox-only (pages indexed differently)
                for (p, bb), i in feature_idx_by_pageblock.items():
                    if list(bb) == bbox:
                        fidx = i
                        break
            pred_p_table = (
                pred_probs[fidx]["table_region"] if fidx is not None
                else None
            )

            blocks_out.append({
                "text": text[:120],
                "page": page_num,
                "bbox": list(bbox),
                "gt_html": gt_html,
                "pred_p_table": pred_p_table,
            })

    return {"pair_id": pair_path.stem, "blocks": blocks_out,
            "tables": tables_out, "error": None}


# ---------------------------------------------------------------------------
# Structure inference — mirrors council/structure.run_inputs but returns
# raw probabilities per region (not TypedSignals) for aggregation.
# ---------------------------------------------------------------------------


def load_structure_runtime():
    import torch  # noqa
    from transformers import AutoTokenizer  # noqa
    from dart_semantic.council import structure as bert_structure
    from dart_semantic.council.base import (
        load_lora_adapter, load_shared_backbone,
    )

    spec = bert_structure.ADAPTER_SPEC
    backbone = load_shared_backbone()
    adapter = load_lora_adapter(spec, backbone)
    tok_dir = spec.adapter_path / "tokenizer"
    tok = AutoTokenizer.from_pretrained(
        str(tok_dir) if tok_dir.exists() else backbone.name
    )
    heads_bundle = bert_structure._load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    heads = {
        "role": heads_bundle["role"].to(device),
        "is_heading": heads_bundle["is_heading"].to(device),
        "table_region": heads_bundle["table_region"].to(device),
        "list_nesting": heads_bundle["list_nesting"].to(device),
        "layout_norm": heads_bundle["layout_norm"].to(device),
        "layout_mlp": heads_bundle["layout_mlp"].to(device),
    }
    peft_model = adapter.peft_model
    peft_model.eval()
    return peft_model, tok, heads, device


def _run_structure(features, peft_model, tok, heads, device,
                   *, batch_size: int = 64) -> list[dict]:
    """Return [{"role": list[6], "is_heading": float, "table_region": float}]
    for each FeatureBlock — softmax probabilities, P(positive class) for
    binary heads."""
    import torch  # noqa
    from dart_semantic.council.structure import (
        _compute_span_layout, _group_by_page, _page_medians,
        LAYOUT_FEATURE_DIM, _block_in_table,
    )

    span_texts: list[str] = [""] * len(features)
    span_layouts: list[list[float]] = [[0.0] * LAYOUT_FEATURE_DIM
                                        for _ in features]
    for page, idxs in _group_by_page(features):
        page_spans = [features[i] for i in idxs]
        first_raw = getattr(page_spans[0], "raw", None)
        page_w = float(getattr(first_raw, "page_width", 612.0) or 612.0)
        page_h = float(getattr(first_raw, "page_height", 792.0) or 792.0)
        median_fs, median_h = _page_medians(page_spans)
        for i in idxs:
            fb = features[i]
            raw = getattr(fb, "raw", None)
            text = (getattr(raw, "text", "") or "").strip()
            in_table = _block_in_table(fb)
            layout_vec = _compute_span_layout(
                fb, page_w=page_w, page_h=page_h,
                page_median_fs=median_fs, page_median_h=median_h,
                in_table=in_table,
            )
            span_texts[i] = text
            span_layouts[i] = layout_vec

    out: list[dict] = []
    n = len(features)
    for start in range(0, n, batch_size):
        batch_t = span_texts[start:start + batch_size]
        batch_l = span_layouts[start:start + batch_size]
        enc = tok(batch_t, padding=True, truncation=True, max_length=192,
                  return_tensors="pt").to(device)
        layout_t = torch.tensor(batch_l, dtype=torch.float32, device=device)
        with torch.no_grad():
            o = peft_model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"])
            pooled = o.last_hidden_state[:, 0, :].float()
            layout_h = heads["layout_mlp"](heads["layout_norm"](layout_t))
            h = torch.cat([pooled, layout_h], dim=-1)
            p_role = torch.softmax(heads["role"](h), dim=-1)
            p_is_h = torch.softmax(heads["is_heading"](h), dim=-1)
            p_tr = torch.softmax(heads["table_region"](h), dim=-1)
        for k in range(p_role.size(0)):
            out.append({
                "role": p_role[k].tolist(),
                "is_heading": float(p_is_h[k][1].item()),
                "table_region": float(p_tr[k][1].item()),
            })
    return out


# ---------------------------------------------------------------------------
# Region-level aggregation + scoring
# ---------------------------------------------------------------------------


def aggregate_regions(per_pair_results: list[dict],
                      *,
                      pred_threshold: float = 0.5,
                      gt_threshold: float = 0.5,
                      min_spans: int = 2) -> dict:
    """For each TableCandidate region, find spans inside, aggregate
    predicted P(table_region=1) and GT in_table_html. Return confusion
    matrix + samples."""
    region_records: list[dict] = []
    skip_no_spans = 0
    skip_no_gt = 0

    for pr in per_pair_results:
        if pr.get("error"):
            continue
        blocks = pr["blocks"]
        tables = pr["tables"]
        for t in tables:
            page = t["page"]
            bb = t["bbox"]
            spans_in = []
            for b in blocks:
                if b["page"] != page:
                    continue
                if not b["bbox"] or len(b["bbox"]) < 4:
                    continue
                cx, cy = _block_center(b["bbox"])
                if _bbox_contains(bb, cx, cy):
                    spans_in.append(b)
            if len(spans_in) < min_spans:
                skip_no_spans += 1
                continue

            # Aggregated prediction
            preds = [b["pred_p_table"] for b in spans_in
                     if b["pred_p_table"] is not None]
            if not preds:
                continue
            mean_pred = sum(preds) / len(preds)
            predicted_table = mean_pred >= pred_threshold

            # Aggregated GT
            gts = [b["gt_html"] for b in spans_in if b["gt_html"] is not None]
            if not gts:
                skip_no_gt += 1
                continue
            gt_pos_frac = sum(1 for g in gts if g) / len(gts)
            gt_table = gt_pos_frac >= gt_threshold

            region_records.append({
                "pair_id": pr["pair_id"],
                "page": page,
                "n_spans": len(spans_in),
                "n_aligned": len(gts),
                "mean_pred": round(mean_pred, 3),
                "gt_pos_frac": round(gt_pos_frac, 3),
                "predicted_table": predicted_table,
                "gt_table": gt_table,
            })

    # Confusion matrix
    tp = sum(1 for r in region_records
             if r["predicted_table"] and r["gt_table"])
    fp = sum(1 for r in region_records
             if r["predicted_table"] and not r["gt_table"])
    fn = sum(1 for r in region_records
             if not r["predicted_table"] and r["gt_table"])
    tn = sum(1 for r in region_records
             if not r["predicted_table"] and not r["gt_table"])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / max(1e-9, precision + recall))

    return {
        "n_regions_total": len(region_records),
        "skipped_no_spans": skip_no_spans,
        "skipped_no_gt": skip_no_gt,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "regions": region_records,
    }


def span_level_scores(per_pair_results: list[dict],
                      *,
                      pred_threshold: float = 0.5) -> dict:
    tp = fp = fn = tn = 0
    n_unaligned = 0
    for pr in per_pair_results:
        if pr.get("error"):
            continue
        for b in pr["blocks"]:
            if b["pred_p_table"] is None:
                continue
            if b["gt_html"] is None:
                n_unaligned += 1
                continue
            pred = b["pred_p_table"] >= pred_threshold
            gt = bool(b["gt_html"])
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "n_unaligned": n_unaligned,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair-dir", type=Path,
                    default=Path("data/structure_dataset/per_pair"))
    ap.add_argument("--source", default="arxiv",
                    help="Filter to pairs from this source (arxiv has "
                         "local_pdf so it's the fastest to eval).")
    ap.add_argument("--limit", type=int, default=20,
                    help="Top-N pairs by table density.")
    ap.add_argument("--min-table-spans", type=int, default=10,
                    help="Skip pairs with fewer than this many "
                         "table_region=1 spans in Phase 3b labels.")
    ap.add_argument("--pair-ids", nargs="+",
                    help="Override --limit with explicit pair IDs.")
    ap.add_argument("--out", type=Path,
                    default=Path("data/eval/table_region_at_region.json"))
    args = ap.parse_args()

    # Pick pairs
    if args.pair_ids:
        ranked = []
        for pid in args.pair_ids:
            ranked.append((pid, args.source, 0, 0))
    else:
        ranked = rank_pairs_by_table_density(
            args.per_pair_dir, source=args.source,
            min_table_spans=args.min_table_spans,
        )[:args.limit]
    print(f"[plan] {len(ranked)} pairs selected (source={args.source})",
          file=sys.stderr)

    # Load Structure adapter once
    print("[load] structure adapter...", file=sys.stderr)
    t0 = time.time()
    structure_runtime = load_structure_runtime()
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # HtmlValidator only used if local_pdf is missing — arxiv has it
    validator = None
    needs_render = any(s != "arxiv" for _, s, _, _ in ranked)
    if needs_render:
        validator = HtmlValidator()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/tmp/table_eval_renders")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    per_pair_results = []
    for i, (pid, src, total, table) in enumerate(ranked, 1):
        pj = find_pair_json(pid, src)
        if pj is None:
            print(f"[{i}/{len(ranked)}] {pid[:60]}  PAIR-NOT-FOUND",
                  file=sys.stderr)
            continue
        t0 = time.time()
        try:
            res = process_pair(pj, structure_runtime,
                               validator=validator, tmp_dir=tmp_dir)
        except Exception as exc:
            print(f"[{i}/{len(ranked)}] {pid[:60]}  ERR {exc}",
                  file=sys.stderr)
            continue
        n_blocks = len(res.get("blocks", []))
        n_tables = len(res.get("tables", []))
        elapsed = time.time() - t0
        print(f"[{i}/{len(ranked)}] {pid[:55]:55s}  "
              f"blocks={n_blocks:4} tables={n_tables:3}  "
              f"{elapsed:5.1f}s", file=sys.stderr)
        per_pair_results.append(res)

    # Region aggregation
    region_summary = aggregate_regions(per_pair_results)
    span_summary = span_level_scores(per_pair_results)

    # Per-pair-id confusion summary for headline
    sources_seen = Counter(s for _, s, _, _ in ranked)

    print()
    print("=" * 72)
    print("REGION-LEVEL (TableCandidate aggregation)")
    print("=" * 72)
    print(f"  regions evaluated: {region_summary['n_regions_total']}")
    print(f"  skipped (no spans): {region_summary['skipped_no_spans']}")
    print(f"  skipped (no GT alignment): {region_summary['skipped_no_gt']}")
    print(f"  TP={region_summary['tp']:4}  FP={region_summary['fp']:4}  "
          f"FN={region_summary['fn']:4}  TN={region_summary['tn']:4}")
    print(f"  precision = {region_summary['precision']:.3f}")
    print(f"  recall    = {region_summary['recall']:.3f}")
    print(f"  F1        = {region_summary['f1']:.3f}")

    print()
    print("=" * 72)
    print("SPAN-LEVEL (per-block, no aggregation) — sanity check")
    print("=" * 72)
    print(f"  TP={span_summary['tp']:5}  FP={span_summary['fp']:5}  "
          f"FN={span_summary['fn']:5}  TN={span_summary['tn']:5}")
    print(f"  precision = {span_summary['precision']:.3f}")
    print(f"  recall    = {span_summary['recall']:.3f}")
    print(f"  F1        = {span_summary['f1']:.3f}")
    print(f"  unaligned blocks (no html GT): {span_summary['n_unaligned']}")
    print()
    print(f"sources surveyed: {dict(sources_seen)}")

    out_payload = {
        "config": {"source": args.source, "limit": args.limit,
                   "min_table_spans": args.min_table_spans,
                   "pair_ids": args.pair_ids or None,
                   "n_pairs_evaluated": len(per_pair_results)},
        "region_level": {k: v for k, v in region_summary.items()
                         if k != "regions"},
        "region_records": region_summary["regions"],
        "span_level": span_summary,
        "sources_seen": dict(sources_seen),
    }
    args.out.write_text(json.dumps(out_payload, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
