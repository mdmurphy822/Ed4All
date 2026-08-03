"""One-off: per-class doc_role F1 for a Semantic adapter on a test split.

Plan-13 Step-2 gate: the summary.json only carries macro F1, but the
comparability caveat flags minor-class collapse (legal/author/footer,
thinned by the uniform cap). This prints the full per-class report so the
gate can be read on the minor roles, not just the macro.

    PYTHONPATH=. .venv/bin/python -m scripts._compare_semantic_adapter_on_test \
        --adapter models/council/semantic/final \
        --dataset-dir data/semantic_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from training.train_semantic import (
    DOC_ROLE_NAMES,
    SemanticModel,
    evaluate,
    load_split,
)

BASE = "answerdotai/ModernBERT-base"


def _attach(model: SemanticModel, adapter_dir: Path, device) -> None:
    from peft import PeftModel

    base = AutoModel.from_pretrained(BASE, dtype=torch.bfloat16)
    model.encoder = PeftModel.from_pretrained(base, str(adapter_dir))
    state = torch.load(str(adapter_dir / "heads.pt"), map_location="cpu")
    model.head_doc_role.load_state_dict(state["head_doc_role.state_dict"])
    model.head_boilerplate.load_state_dict(state["head_boilerplate.state_dict"])
    model.side_norm.load_state_dict(state["side_norm.state_dict"])
    model.side_mlp.load_state_dict(state["side_mlp.state_dict"])
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
        out["y_doc_role"] = torch.tensor([b["y_doc_role"] for b in batch], dtype=torch.long)
        out["y_boilerplate"] = torch.tensor([b["y_boilerplate"] for b in batch], dtype=torch.long)
        layout = torch.tensor([b["layout"] for b in batch], dtype=torch.float32)
        cascade = torch.tensor([b["cascade"] for b in batch], dtype=torch.float32)
        positional = torch.tensor([b["positional"] for b in batch], dtype=torch.float32)
        out["side_channel"] = torch.cat([layout, cascade, positional], dim=-1)
        return out

    return DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/semantic_dataset"))
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE)
    test_ds = load_split(args.dataset_dir / "test.jsonl")
    loader = _loader(test_ds, tok, args.max_length)

    model = SemanticModel(BASE)
    _attach(model, args.adapter, device)

    m = evaluate(model, loader, device)
    print(f"\n=== {args.adapter} on {args.dataset_dir}/test.jsonl ===")
    print(f"  doc_role_macro_f1   {m['doc_role_macro_f1']:.4f}")
    print(f"  boilerplate_pos_f1  {m['boilerplate_pos_f1']:.4f}")

    dr_t, dr_p = m["raw"]["doc_role"]
    labels = list(range(len(DOC_ROLE_NAMES)))
    print("\n[doc_role per-class]")
    print(
        classification_report(
            dr_t, dr_p, labels=labels, target_names=list(DOC_ROLE_NAMES), digits=3, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
