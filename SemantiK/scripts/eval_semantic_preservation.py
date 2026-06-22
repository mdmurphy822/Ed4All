"""Eval Theta semantic_preservation cross-encoder (Phase 4b read-only).

Reports val/test/probe metrics using the saved calibration. Asserts the
gates from Phase 4b plan:
    - mean(predicted) on perturbation_type == "table_flatten" < 0.60
    - mean(predicted) on perturbation_type == "math_as_prose" < 0.60
    - per-domain Spearman >= 0.55 on probe_cross_domain

Usage:
    .venv/bin/python scripts/eval_semantic_preservation.py \
        --model-dir models/theta/semantic_preservation/v1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file as load_st
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoModel, DebertaV2Tokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semantic_preservation import (  # noqa: E402
    BASE_MODEL_NAME,
    DATASET_DIR,
    JsonlPairDataset,
    SemanticPreservationModel,
    make_collate,
)


def load_model(model_dir: Path, device: torch.device) -> SemanticPreservationModel:
    model = SemanticPreservationModel(BASE_MODEL_NAME)
    st = load_st(str(model_dir / "adapter_model.safetensors"))
    missing, unexpected = model.encoder.load_state_dict(st, strict=False)
    heads = torch.load(str(model_dir / "heads.pt"), map_location="cpu")
    model.head_score.load_state_dict(heads["head_score.state_dict"])
    return model.to(device)


def apply_calibration(raw: np.ndarray, cal: dict) -> np.ndarray:
    xs = np.array(cal["isotonic_X_thresholds"])
    ys = np.array(cal["isotonic_y_thresholds"])
    out = np.interp(raw, xs, ys)
    out = out + float(cal.get("offset", 0.0))
    return np.clip(out, 0.0, 1.0)


@torch.no_grad()
def predict_split(
    model: SemanticPreservationModel,
    path: Path,
    tokenizer,
    *,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    ds = JsonlPairDataset(path)
    collate = make_collate(tokenizer, max_len)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate,
    )
    raw: list[float] = []
    lab: list[float] = []
    perts: list[str] = []
    rkinds: list[str] = []
    domains: list[str] = []
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tti = batch.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device)
        logit = model(ids, mask, tti)
        s = torch.sigmoid(logit).float().cpu().numpy()
        raw.extend(s.tolist())
        lab.extend(batch["labels"].tolist())
        perts.extend(batch["perturbation_type"])
        rkinds.extend(batch["source_region_kind"])
        domains.extend(batch["source_domain"])
    return {
        "raw": np.array(raw),
        "labels": np.array(lab),
        "perturbation_type": perts,
        "source_region_kind": rkinds,
        "source_domain": domains,
    }


def summarize(preds: np.ndarray, labels: np.ndarray, prefix: str) -> dict:
    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    sp, _ = spearmanr(preds, labels)
    return {f"{prefix}_n": int(len(preds)),
            f"{prefix}_mae": mae,
            f"{prefix}_rmse": rmse,
            f"{prefix}_spearman": float(sp)}


def grouped(values: list[str], preds: np.ndarray,
            labels: np.ndarray) -> dict:
    out: dict[str, dict] = {}
    seen = defaultdict(list)
    for k, p, l in zip(values, preds, labels):
        seen[k].append((p, l))
    for k, items in seen.items():
        arr = np.array(items)
        sp = float("nan")
        try:
            if len(arr) >= 5:
                sp_, _ = spearmanr(arr[:, 0], arr[:, 1])
                sp = float(sp_)
        except Exception:
            pass
        out[k] = {
            "n": int(len(arr)),
            "mae": float(np.mean(np.abs(arr[:, 0] - arr[:, 1]))),
            "spearman": sp,
            "mean_pred": float(np.mean(arr[:, 0])),
            "mean_label": float(np.mean(arr[:, 1])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the eval JSON. Default: <model-dir>/eval_report.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu] {device}", flush=True)

    cal_path = args.model_dir / "calibration.json"
    cal = json.loads(cal_path.read_text())
    print(
        f"[calibration] loaded fit_split={cal['fit_split']} "
        f"offset={cal['offset']:.4f}",
        flush=True,
    )
    assert cal["fit_split"] == "val", (
        "Calibration must have fit_split='val'."
    )

    # Mirror the trainer: load the slow tokenizer from spm.model. We
    # save spm.model into the model_dir/tokenizer/ at train time.
    spm_candidate = args.model_dir / "tokenizer" / "spm.model"
    if not spm_candidate.exists():
        # Fallback to the HF cache.
        cache = Path.home() / ".cache/huggingface/hub/models--microsoft--deberta-v3-small/snapshots"
        snap = next(cache.glob("*"))
        spm_candidate = snap / "spm.model"
    tokenizer = DebertaV2Tokenizer(vocab_file=str(spm_candidate))
    model = load_model(args.model_dir, device)

    splits = {
        "val": DATASET_DIR / "val.jsonl",
        "test": DATASET_DIR / "test.jsonl",
        "probe_qwen_style": DATASET_DIR / "probe_qwen_style.jsonl",
        "probe_cross_domain": DATASET_DIR / "probe_cross_domain.jsonl",
    }

    report: dict = {}
    for name, path in splits.items():
        if not path.exists():
            print(f"[skip] {name}: {path} missing", flush=True)
            continue
        print(f"[predict] {name} ...", flush=True)
        out = predict_split(
            model, path, tokenizer,
            max_len=args.max_len, batch_size=args.batch_size, device=device,
        )
        cal_preds = apply_calibration(out["raw"], cal)
        agg_raw = summarize(out["raw"], out["labels"], f"{name}_raw")
        agg_cal = summarize(cal_preds, out["labels"], f"{name}_cal")
        report[name] = {
            **agg_raw,
            **agg_cal,
            "by_perturbation_type": grouped(
                out["perturbation_type"], cal_preds, out["labels"],
            ),
            "by_source_region_kind": grouped(
                out["source_region_kind"], cal_preds, out["labels"],
            ),
            "by_source_domain": grouped(
                out["source_domain"], cal_preds, out["labels"],
            ),
        }
        print(
            f"  {name}: raw_mae={agg_raw[f'{name}_raw_mae']:.4f} "
            f"cal_mae={agg_cal[f'{name}_cal_mae']:.4f} "
            f"cal_spearman={agg_cal[f'{name}_cal_spearman']:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Plan gates.
    # ------------------------------------------------------------------
    gates: dict = {}

    def _check_pert_below(name: str, pert: str, thresh: float) -> dict:
        # Look across val + test + probe_qwen_style.
        candidates = ["val", "test", "probe_qwen_style"]
        bp = {}
        for s in candidates:
            if s in report and pert in report[s]["by_perturbation_type"]:
                bp[s] = report[s]["by_perturbation_type"][pert]
        if not bp:
            return {"pass": None, "info": f"perturbation '{pert}' absent"}
        means = {s: v["mean_pred"] for s, v in bp.items()}
        ok = all(m < thresh for m in means.values())
        return {"pass": bool(ok), "thresh": thresh, "mean_pred": means}

    gates["table_flatten_below_0.60"] = _check_pert_below(
        "table_flatten", "table_flatten", 0.60,
    )
    gates["math_as_prose_below_0.60"] = _check_pert_below(
        "math_as_prose", "math_as_prose", 0.60,
    )

    # Per-domain Spearman >=0.55 on probe_cross_domain.
    pdg = {}
    if "probe_cross_domain" in report:
        for d, v in report["probe_cross_domain"]["by_source_domain"].items():
            pdg[d] = v["spearman"]
        ok = all((v >= 0.55) for v in pdg.values()) if pdg else False
        gates["probe_cross_domain_per_domain_spearman_ge_0.55"] = {
            "pass": bool(ok), "thresh": 0.55, "per_domain": pdg,
        }
    else:
        gates["probe_cross_domain_per_domain_spearman_ge_0.55"] = {
            "pass": None, "info": "probe_cross_domain absent",
        }

    report["plan_gates"] = gates
    print("[gates]", flush=True)
    for k, v in gates.items():
        print(f"  {k}: {v}", flush=True)

    out_path = args.out or (args.model_dir / "eval_report.json")
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
