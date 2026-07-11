"""End-to-end cascade eval: Structure → Semantic (Phase 4 — task #51).

Two modes:

    --mode teacher  (default, fast — ~5 min)
        Iterate the Phase 3c semantic test split, run Semantic with the
        cascade vector that was teacher-forced at data-build time
        (i.e. inferred by the Structure model that was current when the
        Phase 3c v3 dataset was built — the 4-head May-4 Structure).
        Report per-source / per-class P/R/F1, per-source confusion, and
        save a JSON report.

        Skips Structure inference entirely. Tells us whether the
        trained Semantic head generalises across sources WITHOUT having
        to re-extract PDFs. This is the per-source breakdown the
        trainer's end-of-training summary doesn't print.

    --mode endtoend  (slow — depends on --max-pairs; ~30 s/pair)
        Pick N pairs from the test stems, re-extract each pair with
        ``extract_shared_cached``, run the CURRENT 5-head Structure
        (post-Phase-3f) to derive a new cascade per span, then run
        Semantic on the inferred cascade. Match each predicted span back
        to test.jsonl rows by (pair, page, bbox) and compare predicted
        doc_role / boilerplate to the test labels. Reports cascade
        drift (cosine similarity between teacher-forced and inferred
        cascade), and per-source P/R/F1 under inferred cascade.

        If teacher-mode F1 ≫ endtoend-mode F1, cascade drift is real
        and we should rebuild Phase 3c with the post-3f Structure
        before training another Semantic generation. If they agree,
        the cascade is robust to the Structure head changes.

Usage:
    .venv/bin/python -m scripts.eval_cascade_structure_to_semantic
    .venv/bin/python -m scripts.eval_cascade_structure_to_semantic \\
        --mode endtoend --max-pairs 30 --per-source 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# numpy is already a transitive dep
import numpy as np  # type: ignore


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def _classification_report(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int],
    names: list[str],
) -> dict:
    """Per-class P/R/F1 + macro/weighted aggregates. No sklearn dep
    required — kept tiny so the script can be inlined into CI later
    without the sklearn install footprint."""
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
    y_true: list[int],
    y_pred: list[int],
    n_classes: int,
) -> list[list[int]]:
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


# ---------------------------------------------------------------------------
# Semantic runtime — direct row inference (no FeatureBlock objects)
# ---------------------------------------------------------------------------


def _load_semantic_runtime():
    import torch  # noqa
    from transformers import AutoTokenizer  # noqa

    from semantik_structure.council import semantic as bert_semantic
    from semantik_structure.council.base import (
        load_lora_adapter, load_shared_backbone,
    )

    spec = bert_semantic.ADAPTER_SPEC
    backbone = load_shared_backbone()
    adapter = load_lora_adapter(spec, backbone)
    tok_dir = spec.adapter_path / "tokenizer"
    tok = AutoTokenizer.from_pretrained(
        str(tok_dir) if tok_dir.exists() else backbone.name
    )
    heads = bert_semantic._load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    return {
        "peft_model": adapter.peft_model.eval(),
        "tok": tok,
        "head_doc_role": heads["doc_role"].to(device),
        "head_boilerplate": heads["boilerplate"].to(device),
        "side_norm": heads["side_norm"].to(device),
        "side_mlp": heads["side_mlp"].to(device),
        "device": device,
    }


def _semantic_predict_batch(
    rt: dict,
    texts: list[str],
    side_channels: list[list[float]],
) -> tuple[list[int], list[int]]:
    """Run trained Semantic on a batch of pre-built side-channel rows.

    Returns (doc_role_argmax, boilerplate_argmax) — both length len(texts).
    """
    import torch  # noqa

    device = rt["device"]
    enc = rt["tok"](
        texts, padding=True, truncation=True, max_length=192,
        return_tensors="pt",
    ).to(device)
    side_t = torch.tensor(side_channels, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = rt["peft_model"](
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        pooled = out.last_hidden_state[:, 0, :].float()
        side_h = rt["side_mlp"](rt["side_norm"](side_t))
        h = torch.cat([pooled, side_h], dim=-1)
        logits_dr = rt["head_doc_role"](h)
        logits_bp = rt["head_boilerplate"](h)
        pred_dr = logits_dr.argmax(dim=-1).tolist()
        pred_bp = logits_bp.argmax(dim=-1).tolist()
    return pred_dr, pred_bp


# ---------------------------------------------------------------------------
# Mode A — teacher-forced eval over the test split
# ---------------------------------------------------------------------------


def run_teacher_mode(
    test_path: Path,
    out_path: Path,
    *,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict:
    """Re-run Semantic on the test split using stored cascade vectors.
    No Structure inference. Adds per-source breakdown to what the
    trainer already prints."""
    from semantik_structure.council.semantic import (
        BOILERPLATE_LABELS, DOC_ROLE_NAMES,
    )

    print(f"[teacher] loading test split: {test_path}")
    rows = []
    with test_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"empty test split at {test_path}")
    print(f"[teacher] rows = {len(rows)}")

    rt = _load_semantic_runtime()

    y_true_dr: list[int] = []
    y_pred_dr: list[int] = []
    y_true_bp: list[int] = []
    y_pred_bp: list[int] = []
    sources: list[str] = []
    pair_ids: list[str] = []
    confidences_low: list[dict] = []  # for debug — top-5 lowest-confidence

    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [r["text"] for r in batch]
        side_channels = [
            list(r["layout"]) + list(r["cascade"]) + list(r["positional"])
            for r in batch
        ]
        # Sanity — every row must be 31-dim
        for sc in side_channels:
            if len(sc) != 31:
                raise SystemExit(
                    f"row side-channel dim mismatch: {len(sc)} != 31"
                )
        pred_dr, pred_bp = _semantic_predict_batch(rt, texts, side_channels)
        for r, pd, pb in zip(batch, pred_dr, pred_bp):
            y_true_dr.append(int(r["labels"]["doc_role"]))
            y_pred_dr.append(int(pd))
            y_true_bp.append(int(r["labels"]["boilerplate"]))
            y_pred_bp.append(int(pb))
            sources.append(r.get("source", "unknown"))
            pair_ids.append(r.get("pair", "?"))

        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = (start + len(batch)) / max(elapsed, 1e-3)
            print(
                f"[teacher] {start + len(batch)}/{len(rows)}  "
                f"{rate:.1f} rows/s  elapsed={elapsed:.1f}s",
                flush=True,
            )
    elapsed = time.time() - t0
    print(f"[teacher] done in {elapsed:.1f}s")

    dr_report = _classification_report(
        y_true_dr, y_pred_dr,
        list(range(len(DOC_ROLE_NAMES))), list(DOC_ROLE_NAMES),
    )
    bp_report = _classification_report(
        y_true_bp, y_pred_bp,
        list(range(len(BOILERPLATE_LABELS))), list(BOILERPLATE_LABELS),
    )
    dr_cm = _confusion(y_true_dr, y_pred_dr, len(DOC_ROLE_NAMES))

    # Per-source breakdown — NEW info beyond the trainer log
    by_source_dr: dict[str, dict] = {}
    by_source_bp: dict[str, dict] = {}
    src_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, src in enumerate(sources):
        src_to_rows[src].append(i)
    for src, idxs in sorted(src_to_rows.items()):
        st_dr = [y_true_dr[i] for i in idxs]
        sp_dr = [y_pred_dr[i] for i in idxs]
        st_bp = [y_true_bp[i] for i in idxs]
        sp_bp = [y_pred_bp[i] for i in idxs]
        by_source_dr[src] = _classification_report(
            st_dr, sp_dr,
            list(range(len(DOC_ROLE_NAMES))), list(DOC_ROLE_NAMES),
        )
        by_source_bp[src] = _classification_report(
            st_bp, sp_bp,
            list(range(len(BOILERPLATE_LABELS))),
            list(BOILERPLATE_LABELS),
        )

    report = {
        "mode": "teacher",
        "test_path": str(test_path),
        "n_rows": len(rows),
        "elapsed_s": round(elapsed, 1),
        "doc_role": dr_report,
        "boilerplate": bp_report,
        "doc_role_confusion": dr_cm,
        "doc_role_label_order": list(DOC_ROLE_NAMES),
        "by_source": {
            "doc_role": by_source_dr,
            "boilerplate": by_source_bp,
        },
        "source_counts": dict(Counter(sources)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[teacher] saved {out_path}")
    return report


# ---------------------------------------------------------------------------
# Mode B — end-to-end with newly-inferred cascade
# ---------------------------------------------------------------------------


def _structure_runtime():
    """Mirrors scripts.eval_table_region_at_region_level.load_structure_runtime
    but exposes the 5-head Structure (post-Phase-3f). Returns a dict
    with everything needed to compute cascade per span."""
    import torch  # noqa
    from transformers import AutoTokenizer  # noqa

    from semantik_structure.council import structure as bert_structure
    from semantik_structure.council.base import (
        load_lora_adapter, load_shared_backbone,
    )

    spec = bert_structure.ADAPTER_SPEC
    backbone = load_shared_backbone()
    adapter = load_lora_adapter(spec, backbone)
    tok_dir = spec.adapter_path / "tokenizer"
    tok = AutoTokenizer.from_pretrained(
        str(tok_dir) if tok_dir.exists() else backbone.name
    )
    heads = bert_structure._load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    return {
        "peft_model": adapter.peft_model.eval(),
        "tok": tok,
        "heads": {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in heads.items()},
        "device": device,
        "spec": spec,
    }


def _select_pairs_from_test(
    test_path: Path,
    *,
    max_pairs: int,
    per_source: int | None,
) -> list[tuple[str, str]]:
    """Pick (pair_id, source) tuples from test stems.

    If per_source is set, take up to that many pair_ids from each source
    (so we get cross-source coverage even when arxiv dominates the
    test set).
    """
    seen: dict[tuple[str, str], int] = {}
    by_source: dict[str, list[str]] = defaultdict(list)
    with test_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("source", "unknown"), r.get("pair", "?"))
            if key in seen:
                continue
            seen[key] = 1
            by_source[key[0]].append(key[1])
    out: list[tuple[str, str]] = []
    if per_source is not None:
        for src, pids in by_source.items():
            for pid in pids[:per_source]:
                out.append((src, pid))
                if len(out) >= max_pairs:
                    return out
    else:
        for src, pids in by_source.items():
            for pid in pids:
                out.append((src, pid))
    return out[:max_pairs]


def _find_pair_json(pair_id: str, source: str) -> Path | None:
    cand = Path(f"data/pairs/{source}/{pair_id}.json")
    if cand.exists():
        return cand
    for sd in Path("data/pairs").iterdir():
        if not sd.is_dir():
            continue
        c = sd / f"{pair_id}.json"
        if c.exists():
            return c
    return None


def run_endtoend_mode(
    test_path: Path,
    out_path: Path,
    *,
    max_pairs: int = 30,
    per_source: int | None = 5,
) -> dict:
    """Re-extract pairs, run Structure 5-head → Semantic, compare.

    For each pair we re-extract via extract_shared_cached, run the
    current Structure to derive cascade, pass that through Semantic,
    and match each predicted span back to test.jsonl rows by
    (pair, page, bbox) for ground-truth.
    """
    from semantik_structure.council.semantic import (
        BOILERPLATE_LABELS, CASCADE_DIM, DOC_ROLE_NAMES,
        compute_positional_features as _compute_positional,
    )
    from semantik_structure.council.structure import (
        LAYOUT_FEATURE_DIM, _block_in_table, _compute_span_layout,
        _group_by_page, _page_medians,
    )
    from semantik_structure.extract_shared import extract_shared_cached
    from semantik_structure.features import featurize_from_shared

    # Build (pair, page, bbox) → ground-truth labels lookup
    print(f"[endtoend] indexing test split: {test_path}")
    pair_lookup: dict[tuple[str, int, tuple], dict] = {}
    pair_rows_count: dict[str, int] = Counter()
    with test_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # No bbox stored in test rows :( — match by (pair, text-prefix)
            # as a fallback. The test rows DO carry layout + cascade +
            # positional but not bbox. We index by (pair, text) instead.
            pair = r.get("pair", "?")
            text_key = (r.get("text") or "")[:80]
            pair_lookup[(pair, 0, (text_key,))] = r
            pair_rows_count[pair] += 1
    print(f"[endtoend] indexed {len(pair_lookup)} test rows across "
          f"{len(pair_rows_count)} pairs")

    pairs = _select_pairs_from_test(
        test_path, max_pairs=max_pairs, per_source=per_source,
    )
    print(f"[endtoend] selected {len(pairs)} pairs "
          f"(target max={max_pairs}, per_source={per_source})")

    # Filter to pairs that actually have JSON files on disk
    pair_paths: list[tuple[Path, str]] = []
    for src, pid in pairs:
        p = _find_pair_json(pid, src)
        if p is not None:
            pair_paths.append((p, src))
    print(f"[endtoend] resolved {len(pair_paths)} pair files")

    sem_rt = _load_semantic_runtime()
    struct_rt = _structure_runtime()

    import torch  # noqa

    # Per-pair span buffers
    per_source_true_dr: dict[str, list[int]] = defaultdict(list)
    per_source_pred_dr: dict[str, list[int]] = defaultdict(list)
    per_source_true_bp: dict[str, list[int]] = defaultdict(list)
    per_source_pred_bp: dict[str, list[int]] = defaultdict(list)
    drift_cosines: list[float] = []
    matched_total = 0
    extracted_total = 0

    t0 = time.time()
    for pi, (pair_path, src) in enumerate(pair_paths):
        pair_data = json.loads(pair_path.read_text())
        local_pdf = pair_data.get("local_pdf")
        if not local_pdf or not Path(local_pdf).exists():
            continue
        try:
            shared = extract_shared_cached(Path(local_pdf))
        except Exception as exc:  # noqa: BLE001
            print(f"[endtoend] {pair_path.stem}: extract failed — {exc}")
            continue

        features = featurize_from_shared(shared)
        if not features:
            continue
        extracted_total += len(features)

        # === Structure 5-head forward to derive cascade ===
        # Build per-span layout (page-grouped — same as Structure runtime)
        span_layouts = [[0.0] * LAYOUT_FEATURE_DIM for _ in features]
        span_texts = [""] * len(features)
        for page, idxs in _group_by_page(features):
            page_spans = [features[i] for i in idxs]
            first_raw = getattr(page_spans[0], "raw", None)
            page_w = float(
                getattr(first_raw, "page_width", 612.0) or 612.0
            )
            page_h = float(
                getattr(first_raw, "page_height", 792.0) or 792.0
            )
            median_fs, median_h = _page_medians(page_spans)
            for i in idxs:
                fb = features[i]
                raw = getattr(fb, "raw", None)
                text = (getattr(raw, "text", "") or "").strip()
                in_table = _block_in_table(fb)
                span_texts[i] = text
                span_layouts[i] = _compute_span_layout(
                    fb, page_w=page_w, page_h=page_h,
                    page_median_fs=median_fs, page_median_h=median_h,
                    in_table=in_table,
                )

        # Forward through Structure in batches of 64
        device = struct_rt["device"]
        all_role_probs = []
        all_ih_probs = []
        all_tr_probs = []
        for s in range(0, len(features), 64):
            tx = span_texts[s:s + 64]
            lyt = span_layouts[s:s + 64]
            enc = struct_rt["tok"](
                tx, padding=True, truncation=True, max_length=192,
                return_tensors="pt",
            ).to(device)
            layout_t = torch.tensor(
                lyt, dtype=torch.float32, device=device,
            )
            with torch.no_grad():
                out = struct_rt["peft_model"](
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = out.last_hidden_state[:, 0, :].float()
                norm = struct_rt["heads"]["layout_norm"](layout_t)
                layout_h = struct_rt["heads"]["layout_mlp"](norm)
                h = torch.cat([pooled, layout_h], dim=-1)
                role_logits = struct_rt["heads"]["role"](h)
                ih_logits = struct_rt["heads"]["is_heading"](h)
                tr_logits = struct_rt["heads"]["table_region"](h)
                role_p = torch.softmax(role_logits, dim=-1).cpu().tolist()
                ih_p = torch.softmax(ih_logits, dim=-1).cpu().tolist()
                tr_p = torch.softmax(tr_logits, dim=-1).cpu().tolist()
            all_role_probs.extend(role_p)
            all_ih_probs.extend(ih_p)
            all_tr_probs.extend(tr_p)

        # Build cascade per span — [P(role)x6, P(is_heading=1), P(table=1)]
        cascades_inferred: list[list[float]] = []
        for r_p, i_p, t_p in zip(all_role_probs, all_ih_probs, all_tr_probs):
            casc = list(r_p) + [float(i_p[1])] + [float(t_p[1])]
            assert len(casc) == CASCADE_DIM, (len(casc), CASCADE_DIM)
            cascades_inferred.append(casc)

        # === Semantic forward with inferred cascade ===
        positionals = _compute_positional(features)
        side_channels = [
            list(layout) + list(cascade) + list(positional)
            for layout, cascade, positional
            in zip(span_layouts, cascades_inferred, positionals)
        ]
        # Predict in batches
        pred_dr_all: list[int] = []
        pred_bp_all: list[int] = []
        for s in range(0, len(features), 64):
            tx = span_texts[s:s + 64]
            sc = side_channels[s:s + 64]
            pdr, pbp = _semantic_predict_batch(sem_rt, tx, sc)
            pred_dr_all.extend(pdr)
            pred_bp_all.extend(pbp)

        # === Match predictions back to test rows by (pair, text-prefix) ===
        pair_id = pair_path.stem
        for i, fb in enumerate(features):
            text = span_texts[i]
            text_key = text[:80]
            gt = pair_lookup.get((pair_id, 0, (text_key,)))
            if gt is None:
                continue
            matched_total += 1
            per_source_true_dr[src].append(int(gt["labels"]["doc_role"]))
            per_source_pred_dr[src].append(int(pred_dr_all[i]))
            per_source_true_bp[src].append(int(gt["labels"]["boilerplate"]))
            per_source_pred_bp[src].append(int(pred_bp_all[i]))
            # Cascade drift — cosine sim between teacher-forced and inferred
            tf = np.asarray(gt["cascade"], dtype=np.float32)
            inf = np.asarray(cascades_inferred[i], dtype=np.float32)
            denom = float(np.linalg.norm(tf) * np.linalg.norm(inf))
            if denom > 1e-9:
                drift_cosines.append(float(np.dot(tf, inf) / denom))

        if (pi + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(
                f"[endtoend] {pi + 1}/{len(pair_paths)} pairs  "
                f"matched={matched_total}  "
                f"extracted_spans={extracted_total}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"[endtoend] done in {elapsed:.1f}s; matched={matched_total}/"
          f"{extracted_total} extracted spans")

    # Aggregate
    all_t_dr: list[int] = []
    all_p_dr: list[int] = []
    all_t_bp: list[int] = []
    all_p_bp: list[int] = []
    by_source_dr_rep: dict[str, dict] = {}
    by_source_bp_rep: dict[str, dict] = {}
    for src in sorted(per_source_true_dr.keys()):
        td = per_source_true_dr[src]
        pd = per_source_pred_dr[src]
        tb = per_source_true_bp[src]
        pb = per_source_pred_bp[src]
        all_t_dr.extend(td)
        all_p_dr.extend(pd)
        all_t_bp.extend(tb)
        all_p_bp.extend(pb)
        by_source_dr_rep[src] = _classification_report(
            td, pd,
            list(range(len(DOC_ROLE_NAMES))), list(DOC_ROLE_NAMES),
        )
        by_source_bp_rep[src] = _classification_report(
            tb, pb,
            list(range(len(BOILERPLATE_LABELS))),
            list(BOILERPLATE_LABELS),
        )

    overall_dr = _classification_report(
        all_t_dr, all_p_dr,
        list(range(len(DOC_ROLE_NAMES))), list(DOC_ROLE_NAMES),
    )
    overall_bp = _classification_report(
        all_t_bp, all_p_bp,
        list(range(len(BOILERPLATE_LABELS))), list(BOILERPLATE_LABELS),
    )

    drift_arr = np.asarray(drift_cosines, dtype=np.float32) \
        if drift_cosines else np.asarray([], dtype=np.float32)
    drift_summary = {
        "n": int(len(drift_arr)),
        "mean": float(drift_arr.mean()) if drift_arr.size else None,
        "median": float(np.median(drift_arr)) if drift_arr.size else None,
        "p10": float(np.percentile(drift_arr, 10)) if drift_arr.size else None,
        "p90": float(np.percentile(drift_arr, 90)) if drift_arr.size else None,
        "min": float(drift_arr.min()) if drift_arr.size else None,
    }

    report = {
        "mode": "endtoend",
        "n_pairs_resolved": len(pair_paths),
        "n_spans_extracted": extracted_total,
        "n_spans_matched": matched_total,
        "elapsed_s": round(elapsed, 1),
        "doc_role": overall_dr,
        "boilerplate": overall_bp,
        "by_source": {
            "doc_role": by_source_dr_rep,
            "boilerplate": by_source_bp_rep,
        },
        "cascade_drift_cosine": drift_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[endtoend] saved {out_path}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "endtoend"],
                    default="teacher")
    ap.add_argument("--test-path",
                    default="data/semantic_dataset/test.jsonl")
    ap.add_argument("--out-path", default=None,
                    help="default: data/eval_reports/cascade_<mode>.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="teacher-mode: cap on test rows")
    ap.add_argument("--max-pairs", type=int, default=30,
                    help="endtoend-mode: cap on pairs")
    ap.add_argument("--per-source", type=int, default=5,
                    help="endtoend-mode: cap pairs per source")
    args = ap.parse_args()

    # Make sure semantik_structure.council.semantic is on the registry path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_path = Path(args.test_path)
    out_path = Path(
        args.out_path
        or f"data/eval_reports/cascade_{args.mode}.json"
    )

    if args.mode == "teacher":
        run_teacher_mode(
            test_path, out_path,
            batch_size=args.batch_size, limit=args.limit,
        )
    else:
        run_endtoend_mode(
            test_path, out_path,
            max_pairs=args.max_pairs, per_source=args.per_source,
        )


if __name__ == "__main__":
    main()
