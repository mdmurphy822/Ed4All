"""Measure the CURRENT committed structure head's REAL-PDF gold-reproduction F1.

Phase 1 headline. Loads the shipped council structure head
(``models/council/structure/final``) — the SAME (text + 20-dim layout) ->
structural_role model whose in-distribution test_role_macro_f1 is 0.894
(``summary.json``) — and scores it over the real-PDF eval rows produced by
``data/builders/build_structure_data.py --realpdf-only``.

It reports, PER DOMAIN (math / scientific / regulatory_legal / forms):
    * real-PDF structural_role macro-F1
    * alongside the in-distribution 0.894 (the gap)
    * alongside the mean alignment coverage for that slice (so a low score is
      attributable to aligner-vs-head, not a silent contamination)
split into the zero-leakage HELD-OUT set (test.jsonl) — the headline — and the
leakage-flagged in-training set (test_in_training.jsonl) — supplementary.

Runs on CPU by default (``--device cpu``) to avoid GPU contention.

Usage:
    python -m eval.measure_realpdf_structure \
        --realpdf-roots data/structure_dataset_realpdf data/structure_dataset_realpdf_internal \
        --out eval_reports/realpdf_structure_gap.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# In-distribution reference (models/council/structure/final/summary.json).
IN_DISTRIBUTION_ROLE_MACRO_F1 = 0.8709  # v3 9-class retrain (was 0.8939 for the 6-class baseline)

ROLE_NAMES = (
    "paragraph",
    "heading",
    "list_item",
    "form_label",
    "blockquote",
    "code_block",
    # v3 additive taxonomy (ids 0-5 unchanged; 6-8 appended) — see
    # training/train_structure.py:ROLE_NAMES (single source of truth).
    "definition_term",
    "definition_def",
    "caption",
)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_head(adapter_dir: Path, base_model: str, device: str):
    """Reconstruct the trainer's StructureModel and load the committed adapter +
    heads. Mirrors training/train_structure.py exactly (same model, same heads),
    so the score is comparable to the in-distribution number from the same code.
    """
    import torch
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    from transformers import AutoTokenizer

    # Import the trainer's model definition (single source of truth).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.train_structure import StructureModel

    model = StructureModel(base_model, dtype=torch.float32)
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    sd = {k: v.float() for k, v in sd.items()}
    missing = set_peft_model_state_dict(model.encoder, sd)
    heads = torch.load(str(adapter_dir / "heads.pt"), map_location="cpu", weights_only=False)
    model.head_role.load_state_dict(heads["head_role.state_dict"])
    model.head_is_heading.load_state_dict(heads["head_is_heading.state_dict"])
    model.head_table_region.load_state_dict(heads["head_table_region.state_dict"])
    if "head_is_image_block.state_dict" in heads:
        model.head_is_image_block.load_state_dict(heads["head_is_image_block.state_dict"])
    model.head_list_nesting.load_state_dict(heads["head_list_nesting.state_dict"])
    model.layout_norm.load_state_dict(heads["layout_norm.state_dict"])
    model.layout_mlp.load_state_dict(heads["layout_mlp.state_dict"])
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(base_model)
    return model, tok


def _predict_roles(model, tok, rows: list[dict], device: str, batch_size: int, max_length: int):
    """Return parallel lists (y_true, y_pred) of structural_role ids."""
    import torch

    y_true, y_pred = [], []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [(r.get("text") or "").strip() for r in batch]
        layouts = [r["layout"] for r in batch]
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        layout = torch.tensor(layouts, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(ids, mask, layout)
        preds = out["role"].argmax(-1).tolist()
        for r, p in zip(batch, preds):
            y_true.append(int(r["labels"]["structural_role"]))
            y_pred.append(int(p))
    return y_true, y_pred


def _macro_f1(y_true, y_pred, labels=None):
    from sklearn.metrics import f1_score

    if not y_true:
        return None
    return float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--realpdf-roots",
        type=Path,
        nargs="+",
        default=[
            Path("data/structure_dataset_realpdf"),
            Path("data/structure_dataset_realpdf_internal"),
        ],
    )
    ap.add_argument(
        "--adapter-dir", type=Path, default=Path("models/council/structure/final")
    )
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--out", type=Path, default=Path("eval_reports/realpdf_structure_gap.json"))
    args = ap.parse_args()

    # Gather rows from both roots, keyed by (set_kind, domain).
    # set_kind: "heldout" (test.jsonl) vs "in_training" (test_in_training.jsonl).
    sets: dict[str, list[dict]] = {"heldout": [], "in_training": []}
    coverage: dict[str, dict] = {}
    for root in args.realpdf_roots:
        sets["heldout"].extend(_load_rows(root / "test.jsonl"))
        sets["in_training"].extend(_load_rows(root / "test_in_training.jsonl"))
        cr = root / "coverage_report.json"
        if cr.exists():
            coverage[root.name] = json.loads(cr.read_text())

    total = sum(len(v) for v in sets.values())
    if total == 0:
        raise SystemExit(
            "no real-PDF rows found — run data/builders/build_structure_data.py --realpdf-only first"
        )

    print(f"[measure] loading head from {args.adapter_dir} on {args.device}", file=sys.stderr)
    model, tok = _load_head(args.adapter_dir, args.base_model, args.device)

    report = {
        "in_distribution_role_macro_f1": IN_DISTRIBUTION_ROLE_MACRO_F1,
        "adapter_dir": str(args.adapter_dir),
        "device": args.device,
        "role_names": list(ROLE_NAMES),
        "coverage_reports": coverage,
        "sets": {},
    }

    for set_kind, rows in sets.items():
        if not rows:
            continue
        # ONE forward pass over the whole set; slice predictions by domain after.
        yt, yp = _predict_roles(model, tok, rows, args.device, args.batch_size, args.max_length)
        overall_f1 = _macro_f1(yt, yp)
        # Gold-restricted macro: average ONLY over the role ids that occur in
        # the gold. A superset-taxonomy model (the v3 9-class head) predicts
        # classes the 6-class real-PDF gold can never contain (definition_term/
        # def/caption); under the union default those enter the macro at F1=0
        # and depress the score. The union number stays the comparable-to-0.271
        # baseline; this is the fair "on the classes that actually occur" view.
        overall_f1_gold = _macro_f1(yt, yp, labels=sorted(set(yt)))
        n_docs = len({(r.get("domain"), r.get("doc_id")) for r in rows})
        set_report = {
            "n_rows": len(rows),
            "n_docs": n_docs,
            "overall_role_macro_f1": overall_f1,
            "overall_role_macro_f1_gold_restricted": overall_f1_gold,
            "delta_vs_in_distribution": (
                round(overall_f1 - IN_DISTRIBUTION_ROLE_MACRO_F1, 6)
                if overall_f1 is not None
                else None
            ),
            "per_domain": {},
        }
        # Bucket the already-computed predictions by domain (no second forward).
        dom_idx: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            dom_idx[r.get("domain", "other")].append(i)
        for dom in sorted(dom_idx):
            idxs = dom_idx[dom]
            dt = [yt[i] for i in idxs]
            dp = [yp[i] for i in idxs]
            f1 = _macro_f1(dt, dp)
            covs = list(
                {
                    rows[i].get("doc_id"): float(rows[i].get("aligned_gold_fraction", 0.0))
                    for i in idxs
                }.values()
            )
            set_report["per_domain"][dom] = {
                "n_rows": len(idxs),
                "n_docs": len({rows[i].get("doc_id") for i in idxs}),
                "role_macro_f1": f1,
                "delta_vs_in_distribution": (
                    round(f1 - IN_DISTRIBUTION_ROLE_MACRO_F1, 6) if f1 is not None else None
                ),
                "mean_alignment_coverage": (
                    round(sum(covs) / len(covs), 6) if covs else None
                ),
            }
        report["sets"][set_kind] = set_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Console table.
    print(f"\nin-distribution role macro-F1 (rendered test): {IN_DISTRIBUTION_ROLE_MACRO_F1:.4f}\n")
    for set_kind, sr in report["sets"].items():
        label = "HELD-OUT (zero leakage)" if set_kind == "heldout" else "IN-TRAINING (leakage-flagged)"
        print(f"=== {label}  rows={sr['n_rows']} docs={sr['n_docs']} ===")
        print(f"  overall real-PDF role macro-F1: {sr['overall_role_macro_f1']:.4f}  "
              f"(Δ {sr['delta_vs_in_distribution']:+.4f})")
        _gr = sr.get("overall_role_macro_f1_gold_restricted")
        if _gr is not None:
            print(f"  gold-restricted macro-F1 (fair, gold classes only): {_gr:.4f}")
        print(f"  {'domain':18} {'docs':>5} {'rows':>6} {'realF1':>8} {'Δ-vs-0.894':>11} {'align_cov':>10}")
        for dom, dd in sr["per_domain"].items():
            f1 = dd["role_macro_f1"]
            cov = dd["mean_alignment_coverage"]
            print(f"  {dom:18} {dd['n_docs']:>5} {dd['n_rows']:>6} "
                  f"{f1:>8.4f} {dd['delta_vs_in_distribution']:>+11.4f} "
                  f"{(cov if cov is not None else float('nan')):>10.4f}")
        print()
    print(f"[measure] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
