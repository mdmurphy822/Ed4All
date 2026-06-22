"""Confusion matrix + per-role precision/recall for a trained classifier.

Reads a JSONL with {input, label} rows, runs the model on each `input`,
and prints a confusion matrix sorted by support, plus per-role
precision / recall / F1. Designed to confirm or reject the
"heading-bias" hypothesis on classifier_v4 against classifier_dataset_v3.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True,
                    help="JSONL with {input, label} per line")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap rows for a faster smoke run (0 = all)")
    args = ap.parse_args()

    from dart_semantic.classify import load_classifier
    print(f"[load] {args.classifier}")
    m = load_classifier(args.classifier)
    tok = m["tokenizer"]
    mdl = m["model"]
    id2label = m["id2label"]
    label2id = {v: k for k, v in id2label.items()}

    rows = []
    with args.data.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["input"], r["label"]))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[data] {len(rows)} rows")

    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = Counter()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [t for t, _ in batch]
        inputs = tok(texts, return_tensors="pt", truncation=True,
                     max_length=256, padding=True)
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = mdl(**inputs).logits
            preds = logits.argmax(dim=-1).tolist()
        for (_, label), pred_id in zip(batch, preds):
            if label not in label2id:
                skipped[label] += 1
                continue
            y_true.append(label)
            y_pred.append(id2label[pred_id])
        if (i // args.batch_size) % 50 == 0:
            print(f"  [{i + len(batch)}/{len(rows)}]")

    if skipped:
        print(f"[skip] labels not in model vocab: {dict(skipped)}")

    labels = sorted(set(y_true) | set(y_pred),
                    key=lambda l: -Counter(y_true)[l])
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    width = max(len(l) for l in labels) + 1
    print()
    print("Confusion matrix (rows = true, cols = predicted):")
    header = " " * width + "  " + " ".join(f"{l[:10]:>10}" for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        line = f"{label:>{width}} | " + " ".join(f"{v:>10}" for v in row)
        support = sum(row)
        print(f"{line}    n={support}")

    print()
    print("Per-role report:")
    print(classification_report(y_true, y_pred, labels=labels,
                                digits=3, zero_division=0))

    # Heading-bias headline numbers: how often is paragraph mis-tagged
    # as heading, and vice versa.
    if "heading" in labels and "paragraph" in labels:
        h, p = "heading", "paragraph"
        n_p_true = sum(1 for t in y_true if t == p)
        n_p_to_h = sum(1 for t, q in zip(y_true, y_pred)
                       if t == p and q == h)
        n_h_true = sum(1 for t in y_true if t == h)
        n_h_to_p = sum(1 for t, q in zip(y_true, y_pred)
                       if t == h and q == p)
        print(f"[bias] paragraph→heading: {n_p_to_h}/{n_p_true} "
              f"({100 * n_p_to_h / max(1, n_p_true):.1f}%)")
        print(f"[bias] heading→paragraph: {n_h_to_p}/{n_h_true} "
              f"({100 * n_h_to_p / max(1, n_h_true):.1f}%)")


if __name__ == "__main__":
    main()
