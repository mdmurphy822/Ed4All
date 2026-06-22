"""One-off: score an arbitrary Structure adapter dir on a given test split.

Used to settle the Plan-13 comparability caveat — run the OLD
(pre-OpenStax) adapter on the NEW test split so the secondary-head
regressions can be read as genuine vs. just-a-harder-distribution.

    PYTHONPATH=. .venv/bin/python -m scripts._compare_structure_adapter_on_test \
        --adapter models/council/structure/final.prev_openstax \
        --dataset-dir data/structure_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from train_structure import (
    StructureModel,
    evaluate,
    load_split,
)


def _attach(model: StructureModel, adapter_dir: Path, device) -> None:
    from peft import PeftModel

    base = AutoModel.from_pretrained("answerdotai/ModernBERT-base", dtype=torch.bfloat16)
    model.encoder = PeftModel.from_pretrained(base, str(adapter_dir))
    state = torch.load(str(adapter_dir / "heads.pt"), map_location="cpu")
    model.head_role.load_state_dict(state["head_role.state_dict"])
    model.head_is_heading.load_state_dict(state["head_is_heading.state_dict"])
    model.head_table_region.load_state_dict(state["head_table_region.state_dict"])
    if "head_is_image_block.state_dict" in state:
        model.head_is_image_block.load_state_dict(state["head_is_image_block.state_dict"])
    model.head_list_nesting.load_state_dict(state["head_list_nesting.state_dict"])
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
        for y in ("y_role", "y_is_heading", "y_table_region", "y_is_image_block", "y_list_nesting"):
            out[y] = torch.tensor([b[y] for b in batch], dtype=torch.long)
        out["layout"] = torch.tensor([b["layout"] for b in batch], dtype=torch.float32)
        return out

    return DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/structure_dataset"))
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    test_ds = load_split(args.dataset_dir / "test.jsonl")
    loader = _loader(test_ds, tok, args.max_length)

    model = StructureModel("answerdotai/ModernBERT-base")
    _attach(model, args.adapter, device)

    m = evaluate(model, loader, device)
    print(f"\n=== {args.adapter} on {args.dataset_dir}/test.jsonl ===")
    for k in (
        "role_macro_f1",
        "is_heading_pos_f1",
        "table_region_pos_f1",
        "is_image_block_pos_f1",
        "list_nesting_mae",
    ):
        print(f"  {k:24} {m[k]:.4f}")

    ib_t, ib_p = m["raw"]["is_image_block"]
    print("\n[is_image_block]")
    print(classification_report(ib_t, ib_p, digits=3, zero_division=0))
    ln_t, ln_p = m["raw"]["list_nesting"]
    print("[list_nesting]")
    print(classification_report(ln_t, ln_p, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
