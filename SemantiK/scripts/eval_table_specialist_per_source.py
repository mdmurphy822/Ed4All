"""Per-source eval for BERT-TableSpecialist (Phase 4 cascade check).

The trainer reports aggregate test macro-F1, which is dominated by
arxiv (~88% of the test split). This script breaks the same test
split out by source so we can see whether the adapter holds up on
wikipedia and openstax tables (which differ stylistically — wiki
infoboxes, multi-row headers).

Test cells in data/table_specialist_dataset/test.jsonl were already
produced by the cascade (Structure table_region gate → pdfplumber
TableCandidate aggregation → cell-level rows), so running on the
test split IS evaluating the v2 adapter under realistic cascade
inputs. What's new here is the per-source breakdown.

Usage:
    .venv/bin/python -m scripts.eval_table_specialist_per_source
    .venv/bin/python -m scripts.eval_table_specialist_per_source \\
        --adapter-dir models/council/table_specialist/final_v2 \\
        --out-path data/eval_reports/table_specialist_v2_per_source.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def _classification_report(
    y_true: list[int], y_pred: list[int],
    labels: list[int], names: list[str],
) -> dict:
    report: dict = {"per_class": {}, "support_total": len(y_true)}
    macro_p = macro_r = macro_f1 = 0.0
    weighted_p = weighted_r = weighted_f1 = 0.0
    n_total = len(y_true)
    for cls in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        report["per_class"][names[cls]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        if n_total:
            weighted_p += precision * support / n_total
            weighted_r += recall * support / n_total
            weighted_f1 += f1 * support / n_total
    n = max(1, len(labels))
    report["macro_avg"] = {
        "precision": round(macro_p / n, 4),
        "recall": round(macro_r / n, 4),
        "f1": round(macro_f1 / n, 4),
    }
    report["weighted_avg"] = {
        "precision": round(weighted_p, 4),
        "recall": round(weighted_r, 4),
        "f1": round(weighted_f1, 4),
    }
    return report


def _confusion(
    y_true: list[int], y_pred: list[int], n_classes: int,
) -> list[list[int]]:
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def _format_input(row: dict) -> str:
    text = (row.get("text") or "").strip()
    nb = row.get("neighbors") or {}
    above = (nb.get("above") or "").strip()
    below = (nb.get("below") or "").strip()
    left = (nb.get("left") or "").strip()
    right = (nb.get("right") or "").strip()
    return (
        f"[ABOVE] {above} [LEFT] {left} "
        f"[CELL] {text} "
        f"[RIGHT] {right} [BELOW] {below}"
    )


def _load_adapter(adapter_dir: Path, base_model: str):
    import torch
    import torch.nn as nn
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    tok_dir = adapter_dir / "tokenizer"
    tok = AutoTokenizer.from_pretrained(
        str(tok_dir) if tok_dir.exists() else base_model
    )
    base = AutoModel.from_pretrained(base_model, dtype=torch.bfloat16)
    encoder = PeftModel.from_pretrained(base, str(adapter_dir))
    encoder.eval()

    state = torch.load(
        str(adapter_dir / "heads.pt"), map_location="cpu",
    )
    layout_dim = int(state.get("layout_dim", 15))
    mlp_hidden = int(state.get("layout_mlp_hidden", 64))
    hidden = base.config.hidden_size

    layout_norm = nn.LayerNorm(layout_dim)
    layout_norm.load_state_dict(state["layout_norm.state_dict"])
    layout_mlp = nn.Sequential(
        nn.Linear(layout_dim, mlp_hidden),
        nn.ReLU(),
        nn.Linear(mlp_hidden, mlp_hidden),
        nn.ReLU(),
    )
    layout_mlp.load_state_dict(state["layout_mlp.state_dict"])
    head = nn.Linear(hidden + mlp_hidden, 4)
    head.load_state_dict(state["head_cell_role_scope.state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    layout_norm.to(device)
    layout_mlp.to(device)
    head.to(device)
    return {
        "tok": tok, "encoder": encoder, "layout_norm": layout_norm,
        "layout_mlp": layout_mlp, "head": head, "device": device,
        "layout_dim": layout_dim,
    }


def _predict_batch(rt, texts: list[str], layouts: list[list[float]]) -> list[int]:
    import torch
    enc = rt["tok"](
        texts, padding=True, truncation=True, max_length=192,
        return_tensors="pt",
    ).to(rt["device"])
    layout_t = torch.tensor(
        layouts, dtype=torch.float32, device=rt["device"],
    )
    with torch.no_grad():
        out = rt["encoder"](
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        pooled = out.last_hidden_state[:, 0, :].float()
        layout_h = rt["layout_mlp"](rt["layout_norm"](layout_t))
        h = torch.cat([pooled, layout_h], dim=-1)
        logits = rt["head"](h)
    return logits.argmax(dim=-1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter-dir",
        default="models/council/table_specialist/final_v2",
    )
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument(
        "--test-path",
        default="data/table_specialist_dataset/test.jsonl",
    )
    ap.add_argument(
        "--out-path",
        default="data/eval_reports/table_specialist_v2_per_source.json",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    LABEL_NAMES = ("data", "header_col", "header_row", "span")
    LABELS = list(range(len(LABEL_NAMES)))

    adapter_dir = Path(args.adapter_dir)
    test_path = Path(args.test_path)
    out_path = Path(args.out_path)

    print(f"[load] adapter: {adapter_dir}")
    rt = _load_adapter(adapter_dir, args.base_model)

    print(f"[load] test: {test_path}")
    rows: list[dict] = []
    with test_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            labels = r.get("labels") or {}
            if "cell_role_scope" not in labels:
                continue
            rows.append(r)
    print(f"[load] {len(rows)} rows")

    y_true: list[int] = []
    y_pred: list[int] = []
    sources: list[str] = []

    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = [_format_input(r) for r in batch]
        layouts = [list(r["layout"]) for r in batch]
        preds = _predict_batch(rt, texts, layouts)
        for r, p in zip(batch, preds):
            y_true.append(int(r["labels"]["cell_role_scope"]))
            y_pred.append(int(p))
            sources.append(r.get("source", "unknown"))
        if (start // args.batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = (start + len(batch)) / max(elapsed, 1e-3)
            print(
                f"[infer] {start + len(batch)}/{len(rows)}  "
                f"{rate:.1f} rows/s  elapsed={elapsed:.1f}s",
                flush=True,
            )
    elapsed = time.time() - t0
    print(f"[infer] done in {elapsed:.1f}s")

    overall = _classification_report(
        y_true, y_pred, LABELS, list(LABEL_NAMES),
    )
    overall_cm = _confusion(y_true, y_pred, len(LABEL_NAMES))

    src_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sources):
        src_to_idx[s].append(i)
    by_source: dict[str, dict] = {}
    by_source_cm: dict[str, list[list[int]]] = {}
    for src, idxs in sorted(src_to_idx.items()):
        st = [y_true[i] for i in idxs]
        sp = [y_pred[i] for i in idxs]
        by_source[src] = _classification_report(
            st, sp, LABELS, list(LABEL_NAMES),
        )
        by_source_cm[src] = _confusion(st, sp, len(LABEL_NAMES))

    report = {
        "adapter_dir": str(adapter_dir),
        "test_path": str(test_path),
        "n_rows": len(rows),
        "elapsed_s": round(elapsed, 1),
        "label_order": list(LABEL_NAMES),
        "overall": overall,
        "overall_confusion": overall_cm,
        "by_source": by_source,
        "by_source_confusion": by_source_cm,
        "source_counts": dict(Counter(sources)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[save] {out_path}")

    print()
    print(f"OVERALL macro-F1 = {overall['macro_avg']['f1']:.4f}  "
          f"(weighted = {overall['weighted_avg']['f1']:.4f})")
    print()
    print("PER SOURCE:")
    for src, rep in by_source.items():
        n = rep["support_total"]
        m = rep["macro_avg"]["f1"]
        w = rep["weighted_avg"]["f1"]
        print(f"  {src:20s} n={n:5d}  macro-F1={m:.4f}  weighted-F1={w:.4f}")


if __name__ == "__main__":
    main()
