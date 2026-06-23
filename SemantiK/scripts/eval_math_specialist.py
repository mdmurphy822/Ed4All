"""End-to-end evaluation of BERT-MathSpecialist.

Runs the trained adapter against:
    1. The held-out test split (math_type + eq_num_assoc macro-F1).
    2. The math-region candidate stream of the v7 sample PDF
       (eval/side_by_side/math_heavy_short_v7/input.pdf) for a
       latency measurement.
    3. An adapter swap-out / swap-in determinism check (same input
       produces the same logits modulo float-tolerance).

Usage:
    python scripts/eval_math_specialist.py
    python scripts/eval_math_specialist.py --skip-pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _load_test_split(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _eval_test_split(
    adapter_dir: Path,
    test_path: Path,
    *,
    max_rows: int | None = None,
) -> dict:
    """Score the held-out test set."""
    from sklearn.metrics import classification_report, f1_score
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.build_math_specialist_data import (
        EQ_NUM_ASSOC, EQ_NUM_TO_ID, MATH_TYPES, MATH_TYPE_TO_ID,
    )
    from training.train_math_specialist import MathSpecialistModel, _format_input
    from peft import PeftModel
    from transformers import AutoModel

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
    import torch.nn as nn
    hidden = base.config.hidden_size
    head_mt = nn.Linear(hidden, len(MATH_TYPES))
    head_en = nn.Linear(hidden, len(EQ_NUM_ASSOC))
    head_mt.load_state_dict(head_state["head_math_type.state_dict"])
    head_en.load_state_dict(head_state["head_eq_num.state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model = peft_model.to(device)
    head_mt = head_mt.to(device)
    head_en = head_en.to(device)

    texts = [_format_input(r) for r in rows]
    bsz = 32
    mt_pred: list[int] = []
    en_pred: list[int] = []
    mt_true = [MATH_TYPE_TO_ID[r["math_type"]] for r in rows]
    en_true = [EQ_NUM_TO_ID[r["eq_num_assoc"]] for r in rows]

    with torch.no_grad():
        for i in range(0, len(texts), bsz):
            chunk = texts[i:i + bsz]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=192, return_tensors="pt").to(device)
            out = peft_model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"])
            pooled = out.last_hidden_state[:, 0, :].float()
            mt_pred.extend(head_mt(pooled).argmax(-1).tolist())
            en_pred.extend(head_en(pooled).argmax(-1).tolist())

    mt_macro = f1_score(mt_true, mt_pred, average="macro", zero_division=0)
    en_macro = f1_score(en_true, en_pred, average="macro", zero_division=0)
    print(f"[eval] math_type macro-F1   = {mt_macro:.4f}")
    print(f"[eval] eq_num_assoc macro-F1 = {en_macro:.4f}")
    print("[eval] math_type per-class:")
    print(classification_report(
        mt_true, mt_pred,
        labels=list(range(len(MATH_TYPES))),
        target_names=MATH_TYPES,
        zero_division=0, digits=3,
    ))
    print("[eval] eq_num_assoc per-class:")
    print(classification_report(
        en_true, en_pred,
        labels=list(range(len(EQ_NUM_ASSOC))),
        target_names=EQ_NUM_ASSOC,
        zero_division=0, digits=3,
    ))
    return {
        "math_type_macro_f1": float(mt_macro),
        "eq_num_assoc_macro_f1": float(en_macro),
        "n_rows": len(rows),
    }


def _eval_pdf_latency(adapter_dir: Path, pdf_path: Path) -> dict:
    """Time the runner on the v7 sample PDF (math-heavy, 20-page-class)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dart_semantic.extract_shared import extract_shared
    from dart_semantic.region_detection import detect_math_region_candidates
    from dart_semantic.council.runner import run_bert
    from dart_semantic.council.base import SharedBackbone
    SharedBackbone.reset_cache()

    if not pdf_path.exists():
        return {"skipped": True, "reason": f"no PDF at {pdf_path}"}

    print(f"[latency] extracting {pdf_path}")
    shared = extract_shared(pdf_path)
    candidates = detect_math_region_candidates(shared)
    print(f"[latency] {len(candidates)} math-region candidates")

    if not candidates:
        return {
            "n_candidates": 0,
            "wall_seconds": 0.0,
            "skipped": True,
            "reason": "no math candidates on this PDF",
        }

    # Warm load (excluded from the timing).
    _ = run_bert("math_specialist", candidates[:1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    out = run_bert("math_specialist", candidates)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"[latency] forward over {len(candidates)} candidates: {dt:.3f} s")
    return {
        "n_candidates": len(candidates),
        "wall_seconds": dt,
        "n_signals": len(out.signals),
        "skipped": False,
    }


def _adapter_swap_test(adapter_dir: Path, test_path: Path) -> dict:
    """Load the adapter, run, swap-out, swap-in, run again — compare."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dart_semantic.council.base import LoRAAdapter, SharedBackbone
    from dart_semantic.council.math_specialist import ADAPTER_SPEC, run_inputs
    from dart_semantic.council.types import MathCandidate

    SharedBackbone.reset_cache()
    backbone = SharedBackbone.get(
        json.loads((adapter_dir / "summary.json").read_text())["base_model"],
    )

    rows = _load_test_split(test_path)[:8]
    candidates = [
        MathCandidate(
            kind="math",
            bbox=(0.0, 0.0, 100.0, 30.0),
            pages=[1],
            evidence_features={"src_text": r.get("alttext") or r.get("math_text", "")},
            member_block_indices=[],
            glyph_density_features={},
        )
        for r in rows
    ]

    spec = ADAPTER_SPEC.__class__(
        bert_name="math_specialist",
        adapter_path=adapter_dir,
        head_kind="multi_head",
        head_specs=ADAPTER_SPEC.head_specs,
    )
    adapter = LoRAAdapter(backbone, spec)
    adapter.load()
    out_a = run_inputs(adapter, candidates)
    adapter.unload()

    adapter_b = LoRAAdapter(backbone, spec)
    adapter_b.load()
    out_b = run_inputs(adapter_b, candidates)

    # Compare top-1 labels — they must match.
    same = 0
    diff = 0
    for sa, sb in zip(out_a.signals, out_b.signals):
        if sa.top_k_labels[0] == sb.top_k_labels[0]:
            same += 1
        else:
            diff += 1
    return {
        "swap_test_pairs_compared": same + diff,
        "swap_test_top1_agree": same,
        "swap_test_top1_disagree": diff,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter-dir", type=Path,
        default=Path("models/council/math_specialist/final"),
    )
    ap.add_argument(
        "--test-path", type=Path,
        default=Path("data/math_specialist_dataset/test.jsonl"),
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
    print("Held-out F1")
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
