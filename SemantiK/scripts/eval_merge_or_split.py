"""End-to-end evaluation of BERT-MergeOrSplit (Phase 3a).

Runs the trained adapter against:
    1. The held-out test split (per-head F1 + per-class breakdown +
       same_logical_block macro-F1 on the noisy subset).
    2. The merged-stream of the v7 sample PDF
       (eval/side_by_side/math_heavy_short_v7/input.pdf) for a
       latency measurement on adjacent-pair predictions.
    3. An adapter swap-out / swap-in determinism check (same input
       produces the same top-1 across all 4 heads).

Usage:
    .venv/bin/python scripts/eval_merge_or_split.py
    .venv/bin/python scripts/eval_merge_or_split.py --skip-pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Held-out F1
# ---------------------------------------------------------------------------


def _load_test_split(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _eval_test_split(adapter_dir: Path, test_path: Path,
                     *, max_rows: int | None = None) -> dict:
    from sklearn.metrics import classification_report, f1_score
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    import torch.nn as nn

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.build_merge_or_split_data import JOIN_TYPES, JOIN_TYPE_IGNORE
    from training.train_merge_or_split import (
        _format_input, HEADING_LABELS, HYPHEN_LABELS, SAME_LOGICAL_LABELS,
    )

    rows = _load_test_split(test_path)
    if max_rows:
        rows = rows[:max_rows]
    if not rows:
        raise SystemExit(f"empty test split at {test_path}")

    cfg = json.loads((adapter_dir / "summary.json").read_text())
    base_name = cfg["base_model"]
    print(f"[eval] base={base_name}  adapter={adapter_dir}")

    tok_dir = adapter_dir / "tokenizer"
    tok = AutoTokenizer.from_pretrained(
        str(tok_dir) if tok_dir.exists() else base_name,
    )
    base = AutoModel.from_pretrained(base_name, dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()

    head_state = torch.load(str(adapter_dir / "heads.pt"), map_location="cpu")
    hidden = base.config.hidden_size
    head_same = nn.Linear(hidden, 2)
    head_join = nn.Linear(hidden, len(JOIN_TYPES))
    head_hyphen = nn.Linear(hidden, 2)
    head_heading = nn.Linear(hidden, 2)
    head_same.load_state_dict(head_state["head_same.state_dict"])
    head_join.load_state_dict(head_state["head_join.state_dict"])
    head_hyphen.load_state_dict(head_state["head_hyphen.state_dict"])
    head_heading.load_state_dict(head_state["head_heading.state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model = peft_model.to(device)
    head_same = head_same.to(device)
    head_join = head_join.to(device)
    head_hyphen = head_hyphen.to(device)
    head_heading = head_heading.to(device)

    texts = [_format_input(r) for r in rows]
    bsz = 32
    p_same: list[int] = []
    p_join: list[int] = []
    p_hy: list[int] = []
    p_hd: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), bsz):
            chunk = texts[i:i + bsz]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=192, return_tensors="pt").to(device)
            out = peft_model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"])
            pooled = out.last_hidden_state[:, 0, :].float()
            p_same.extend(head_same(pooled).argmax(-1).tolist())
            p_join.extend(head_join(pooled).argmax(-1).tolist())
            p_hy.extend(head_hyphen(pooled).argmax(-1).tolist())
            p_hd.extend(head_heading(pooled).argmax(-1).tolist())

    t_same = [r["labels"]["same_logical_block"] for r in rows]
    t_join = [r["labels"]["join_type"] for r in rows]
    t_hy = [r["labels"]["hyphen_repair"] for r in rows]
    t_hd = [r["labels"]["heading_body_boundary"] for r in rows]
    noisy = [bool(r.get("noisy")) for r in rows]

    same_macro = f1_score(t_same, p_same, average="macro", zero_division=0)
    # join_type: only score positive-primary pairs.
    j_t = [t_join[i] for i in range(len(t_join)) if t_join[i] != JOIN_TYPE_IGNORE]
    j_p = [p_join[i] for i in range(len(p_join)) if t_join[i] != JOIN_TYPE_IGNORE]
    join_macro = (
        f1_score(j_t, j_p, average="macro", zero_division=0) if j_t else 0.0
    )
    hy_pos = f1_score(t_hy, p_hy, pos_label=1, zero_division=0)
    hd_pos = f1_score(t_hd, p_hd, pos_label=1, zero_division=0)
    n_t = [t_same[i] for i in range(len(t_same)) if noisy[i]]
    n_p = [p_same[i] for i in range(len(p_same)) if noisy[i]]
    same_noisy = (
        f1_score(n_t, n_p, average="macro", zero_division=0) if n_t else 0.0
    )

    print(f"[eval] same_logical_block macro-F1            = {same_macro:.4f}")
    print(f"[eval] same_logical_block macro-F1 (noisy)    = {same_noisy:.4f}")
    print(f"[eval] join_type macro-F1 (positive-primary)  = {join_macro:.4f}")
    print(f"[eval] hyphen_repair F1 (pos)                 = {hy_pos:.4f}")
    print(f"[eval] heading_body_boundary F1 (pos)         = {hd_pos:.4f}")

    print("[eval] same_logical_block per-class:")
    print(classification_report(
        t_same, p_same, labels=[0, 1],
        target_names=SAME_LOGICAL_LABELS, zero_division=0, digits=3,
    ))
    if j_t:
        print("[eval] join_type per-class (positive-primary only):")
        print(classification_report(
            j_t, j_p, labels=list(range(len(JOIN_TYPES))),
            target_names=JOIN_TYPES, zero_division=0, digits=3,
        ))
    print("[eval] hyphen_repair per-class:")
    print(classification_report(
        t_hy, p_hy, labels=[0, 1],
        target_names=HYPHEN_LABELS, zero_division=0, digits=3,
    ))
    print("[eval] heading_body_boundary per-class:")
    print(classification_report(
        t_hd, p_hd, labels=[0, 1],
        target_names=HEADING_LABELS, zero_division=0, digits=3,
    ))

    return {
        "same_logical_block_macro_f1": float(same_macro),
        "same_logical_block_macro_f1_noisy": float(same_noisy),
        "join_type_macro_f1": float(join_macro),
        "hyphen_repair_pos_f1": float(hy_pos),
        "heading_body_boundary_pos_f1": float(hd_pos),
        "n_rows": len(rows),
        "n_noisy": sum(noisy),
        "n_join_scored": len(j_t),
    }


# ---------------------------------------------------------------------------
# PDF latency
# ---------------------------------------------------------------------------


def _eval_pdf_latency(adapter_dir: Path, pdf_path: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dart_semantic.extract_shared import extract_shared
    from dart_semantic.features import featurize_from_shared
    from dart_semantic.council.runner import run_bert
    from dart_semantic.council.base import SharedBackbone
    SharedBackbone.reset_cache()

    if not pdf_path.exists():
        return {"skipped": True, "reason": f"no PDF at {pdf_path}"}

    print(f"[latency] extracting {pdf_path}")
    shared = extract_shared(pdf_path)
    feature_blocks = featurize_from_shared(shared)
    print(f"[latency] {len(feature_blocks)} feature blocks "
          f"(=> {max(0, len(feature_blocks)-1)} pairs)")

    if len(feature_blocks) < 2:
        return {"n_pairs": 0, "wall_seconds": 0.0, "skipped": True,
                "reason": "no pairs"}

    # Warm load (excluded from timing).
    _ = run_bert("merge_or_split", feature_blocks[:3])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    out = run_bert("merge_or_split", feature_blocks)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[latency] forward over {len(feature_blocks)} feature blocks: "
          f"{dt:.3f} s ({len(out.signals)} signals)")
    return {
        "n_feature_blocks": len(feature_blocks),
        "wall_seconds": dt,
        "n_signals": len(out.signals),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Adapter swap determinism
# ---------------------------------------------------------------------------


def _adapter_swap_test(adapter_dir: Path, test_path: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from types import SimpleNamespace

    from dart_semantic.council.base import LoRAAdapter, SharedBackbone
    from dart_semantic.council.merge_or_split import ADAPTER_SPEC, run_inputs

    SharedBackbone.reset_cache()
    backbone = SharedBackbone.get(
        json.loads((adapter_dir / "summary.json").read_text())["base_model"],
    )

    # Build 16 synthetic adjacent pairs from the test rows. Each "span"
    # is a SimpleNamespace mimicking FeatureBlock — runtime needs
    # .raw.text/.bbox/.font_size/.is_bold/.page/.page_width/.page_height.
    rows = _load_test_split(test_path)[:8]
    spans = []
    for k, r in enumerate(rows):
        feats_a = (r.get("det_features") or {}).get("a") or {}
        feats_b = (r.get("det_features") or {}).get("b") or {}
        for side, txt_key, feats in (
            ("a", "span_a_text", feats_a),
            ("b", "span_b_text", feats_b),
        ):
            raw = SimpleNamespace(
                text=r[txt_key], page=1,
                bbox=(
                    float(feats.get("x0", 0)),
                    float(feats.get("y0", 0)),
                    float(feats.get("x1", 100)),
                    float(feats.get("y1", 100) or 100),
                ),
                page_width=612.0, page_height=792.0,
                font_size=feats.get("font_size") or 11.0,
                font_name=None,
                is_bold=bool(feats.get("is_bold") or False),
                is_italic=bool(feats.get("is_italic") or False),
                confidence=1.0, source="pypdfium2",
            )
            spans.append(SimpleNamespace(raw=raw))

    spec = ADAPTER_SPEC.__class__(
        bert_name="merge_or_split",
        adapter_path=adapter_dir,
        head_kind="multi_head",
        head_specs=ADAPTER_SPEC.head_specs,
    )
    adapter = LoRAAdapter(backbone, spec)
    adapter.load()
    out_a = run_inputs(adapter, spans)
    adapter.unload()

    adapter_b = LoRAAdapter(backbone, spec)
    adapter_b.load()
    out_b = run_inputs(adapter_b, spans)

    # Compare top-1 for each head.
    same = 0
    diff = 0
    for sa, sb in zip(out_a.signals, out_b.signals):
        if sa.top_k_labels[0] == sb.top_k_labels[0]:
            same += 1
        else:
            diff += 1
    return {
        "swap_test_signals_compared": same + diff,
        "swap_test_top1_agree": same,
        "swap_test_top1_disagree": diff,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter-dir", type=Path,
        default=Path("models/council/merge_or_split/final"),
    )
    ap.add_argument(
        "--test-path", type=Path,
        default=Path("data/merge_or_split_dataset/test.jsonl"),
    )
    ap.add_argument(
        "--pdf", type=Path,
        default=Path("eval/side_by_side/math_heavy_short_v7/input.pdf"),
    )
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--skip-pdf", action="store_true")
    ap.add_argument("--skip-swap", action="store_true")
    args = ap.parse_args()

    if not args.adapter_dir.exists():
        sys.exit(f"adapter not found: {args.adapter_dir}")

    print("=" * 60)
    print("Held-out F1 (per head)")
    print("=" * 60)
    f1 = _eval_test_split(
        args.adapter_dir, args.test_path,
        max_rows=args.max_rows or None,
    )

    pdf_metrics = {"skipped": True, "reason": "skipped"}
    if not args.skip_pdf:
        print("\n" + "=" * 60)
        print("PDF latency")
        print("=" * 60)
        pdf_metrics = _eval_pdf_latency(args.adapter_dir, args.pdf)

    swap_metrics = {"skipped": True}
    if not args.skip_swap:
        print("\n" + "=" * 60)
        print("Adapter swap determinism")
        print("=" * 60)
        swap_metrics = _adapter_swap_test(args.adapter_dir, args.test_path)
        print(swap_metrics)

    out = {
        "f1": f1,
        "pdf_latency": pdf_metrics,
        "adapter_swap": swap_metrics,
    }
    out_path = args.adapter_dir / "eval_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
