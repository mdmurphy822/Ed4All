"""One-off: per-class cell_role_scope F1 for a Table adapter on a test split.

Plan-13 Step-3 gate. Table has no cascade, so old-vs-new on the same new
test split is a perfectly clean apples-to-apples comparison — settles
whether the macro drop vs the 0.9256 old-split baseline is a harder split
or a real regression.

    PYTHONPATH=. .venv/bin/python -m scripts._compare_table_adapter_on_test \
        --adapter models/council/table_specialist/final \
        --dataset-dir data/table_specialist_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from training.train_table_specialist import (
    CELL_ROLE_SCOPE_NAMES,
    TableSpecialistModel,
    evaluate,
    load_split,
)

BASE = "answerdotai/ModernBERT-base"


def _attach(model: TableSpecialistModel, adapter_dir: Path, device) -> None:
    from peft import PeftModel

    base = AutoModel.from_pretrained(BASE, dtype=torch.bfloat16)
    model.encoder = PeftModel.from_pretrained(base, str(adapter_dir))
    state = torch.load(str(adapter_dir / "heads.pt"), map_location="cpu")
    model.head_cell_role_scope.load_state_dict(state["head_cell_role_scope.state_dict"])
    model.layout_norm.load_state_dict(state["layout_norm.state_dict"])
    model.layout_mlp.load_state_dict(state["layout_mlp.state_dict"])
    model.to(device)
    model.eval()


def _loader(ds, tok, max_length: int) -> DataLoader:
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize, batched=True)
    coll = DataCollatorWithPadding(tok, return_tensors="pt")
    TOK = ("input_ids", "attention_mask", "token_type_ids")

    def collate(batch):
        feats = [{k: b[k] for k in TOK if k in b} for b in batch]
        out = coll(feats)
        out["y"] = torch.tensor([b["y"] for b in batch], dtype=torch.long)
        out["layout"] = torch.tensor([b["layout"] for b in batch], dtype=torch.float32)
        return out

    return DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/table_specialist_dataset"))
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE)
    test_ds = load_split(args.dataset_dir / "test.jsonl")
    loader = _loader(test_ds, tok, args.max_length)

    model = TableSpecialistModel(BASE)
    _attach(model, args.adapter, device)

    macro, y, p = evaluate(model, loader, str(device))
    print(f"\n=== {args.adapter} on {args.dataset_dir}/test.jsonl ===")
    print(f"  cell_role_scope_macro_f1  {macro:.4f}")
    labels = list(range(len(CELL_ROLE_SCOPE_NAMES)))
    print("\n[cell_role_scope per-class]")
    print(
        classification_report(
            y, p, labels=labels, target_names=list(CELL_ROLE_SCOPE_NAMES), digits=3, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
