"""Train the stage-3 per-block structural role classifier.

Task: multi-class text classification.
    input:  "[size=xl top caps=title] Some Title Text"
    output: one of 21 Role labels

Base: distilbert-base-uncased (66M params, Apache 2.0).
Fits comfortably on the 3070 8GB in fp32 — no QLoRA needed. Training on
~61K examples for 3 epochs takes ~10-15 min.

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 5 --base-model answerdotai/ModernBERT-base
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# The label vocabulary. Must stay aligned with classify.Role.
LABELS = [
    "title", "heading", "paragraph",
    "list", "list_item",
    "table", "table_row",
    "table_header_cell", "table_data_cell", "table_caption",
    "figure", "figure_caption",
    "blockquote", "code_block",
    "form_field", "form_label",
    "reference", "footnote", "definition",
    "metadata", "page_header", "page_footer",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}


def load_split(path: Path) -> Dataset:
    rows = []
    skipped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["label"] not in LABEL2ID:
            skipped += 1
            continue
        rows.append({
            "text": r["input"],
            "label": LABEL2ID[r["label"]],
            "source": r.get("source", "unknown"),
        })
    if skipped:
        print(f"  skipped {skipped} rows with unknown labels")
    return Dataset.from_list(rows)


def compute_metrics_factory(labels: list[str]):
    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    return compute_metrics


def compute_class_weights(train_labels: list[int], num_labels: int,
                          *, weight_cap: float) -> torch.Tensor:
    """sklearn-style 'balanced' weights with a multiplicative cap.

    Raw `balanced` yields weight_i = N / (K * count_i); for skew ratios in
    the 100s (code_block vs paragraph) that pushes rare-class weights to
    100x+ and makes the model wildly over-predict them. Cap keeps the
    rebalance strong on rare classes without flipping the distribution.

    Classes absent from training get weight 1.0 — loss never sees them.
    """
    counts = Counter(train_labels)
    n = len(train_labels)
    k = num_labels
    weights = torch.ones(num_labels, dtype=torch.float32)
    for label_id in range(num_labels):
        c = counts.get(label_id, 0)
        if c == 0:
            continue
        w = n / (k * c)
        weights[label_id] = min(w, weight_cap)
    return weights


class WeightedLossTrainer(Trainer):
    """Trainer variant with class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None,
                 oversample: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.oversample = oversample

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = nn.functional.cross_entropy(logits, labels, weight=weights)
        else:
            loss = nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self, *args, **kwargs):
        if not self.oversample or self.train_dataset is None:
            return super()._get_train_sampler(*args, **kwargs)
        # Per-sample weight = class weight at that sample's label.
        labels = self.train_dataset["label"]
        if self.class_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        cw = self.class_weights.tolist()
        sample_weights = torch.tensor([cw[int(y)] for y in labels],
                                      dtype=torch.float32)
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/classifier_dataset"))
    ap.add_argument("--output-dir", type=Path, default=Path("models/classifier"))
    ap.add_argument("--base-model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weighted", action="store_true",
                    help="Apply sklearn-balanced class weights (capped) to CE loss")
    ap.add_argument("--oversample", action="store_true",
                    help="Use WeightedRandomSampler so rare classes see more gradient steps")
    ap.add_argument("--weight-cap", type=float, default=30.0,
                    help="Max per-class weight (caps the rebalance strength for very rare classes)")
    args = ap.parse_args()

    print(f"[gpu] cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print(f"[data] loading {args.dataset_dir}")
    train_ds = load_split(args.dataset_dir / "train.jsonl")
    val_ds = load_split(args.dataset_dir / "val.jsonl")
    test_ds = load_split(args.dataset_dir / "test.jsonl")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    class_weights: torch.Tensor | None = None
    if args.class_weighted or args.oversample:
        class_weights = compute_class_weights(
            train_ds["label"], num_labels=len(LABELS),
            weight_cap=args.weight_cap,
        )
        print("[weights] per-class weights (capped at "
              f"{args.weight_cap}):")
        train_counts = Counter(train_ds["label"])
        for i, lbl in enumerate(LABELS):
            c = train_counts.get(i, 0)
            if c == 0:
                continue
            print(f"  {lbl:22} count={c:<6} weight={class_weights[i].item():.3f}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics_factory(LABELS),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
        oversample=args.oversample,
    )

    print("[train] starting...")
    trainer.train()

    print("\n[test] final evaluation on held-out test set")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    for k, v in test_metrics.items():
        print(f"  {k:30} {v}")

    # Per-class report on test set
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = np.array(preds.label_ids)
    print("\n[test] per-class report:")
    present_ids = sorted(set(y_true) | set(y_pred))
    present_labels = [ID2LABEL[i] for i in present_ids]
    print(classification_report(
        y_true, y_pred,
        labels=present_ids,
        target_names=present_labels,
        zero_division=0,
        digits=3,
    ))

    # Save the final model
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"\n[save] adapter + tokenizer -> {final_dir}")

    # Dump a small summary JSON so the sweep runner / downstream can read stats.
    summary = {
        "base_model": args.base_model,
        "labels": LABELS,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "test_metrics": test_metrics,
        "class_weighted": args.class_weighted,
        "oversample": args.oversample,
        "weight_cap": args.weight_cap,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] summary -> {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
