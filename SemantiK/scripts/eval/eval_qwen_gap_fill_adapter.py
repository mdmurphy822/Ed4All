"""Generative eval for the trained Qwen3-4B LoRA *gap_fill* adapter.

Mirror of ``scripts/eval/eval_qwen_prose_adapter.py`` for the 4th specialist
(trained via ``scripts/training/train_qwen_lora.py --adapter gap_fill``). Decodes
completions on the held-out gap_fill test split and scores them against
``target_html``, **stratified by ``gap_kind``** so a strong ``missing_title``
number can't mask a weak ``citation_unresolved`` (Plans/08 §5 step 3).

Unlike prose (92% one class), the gap_fill test split is balanced across the
three live kinds (citation_unresolved / missing_title / author_block), so we
evaluate the FULL test set rather than oversampling.

Metrics:
  * ``id_stripped_exact`` / ``id_stripped_sim`` — structural match vs gold
    with position/asset attribute values neutralized (the adapter can't know
    them). Primary structural signal.
  * ``wellformed`` / ``truncated`` — fragment parses as XML / hit the token cap.
  * ``citation_correct`` — FUNCTIONAL, citation rows only (Plans/08 scorer):
    gold restored an ``<a href="#anchor">`` → gen must restore the SAME anchor;
    gold left plain text (the negative / no-valid-target case) → gen must emit
    NO anchor. Measures "did it resolve the cite correctly OR correctly decline
    to hallucinate one".
  * ``axe_pass`` / ``axe_regression`` — wcag22aa over each fragment; regression
    (gold clean, gen broke it) is the meaningful a11y signal.

This eval also resolves the training ``eval_loss=nan`` flag: if the generation
scores are healthy, the nan was a loss-metric artifact, not a broken model.

Usage::

    .venv/bin/python -m scripts.eval_qwen_gap_fill_adapter            # full test
    .venv/bin/python -m scripts.eval_qwen_gap_fill_adapter --n 300 --no-axe
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GAP_DIR = REPO_ROOT / "models/qwen_specialists/gap_fill/v1"
DEFAULT_SUMMARY = DEFAULT_GAP_DIR / "summary.json"
DEFAULT_TEST = REPO_ROOT / "data/qwen_gap_fill_dataset/test.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "data/eval_reports"

_WS_RE = re.compile(r"\s+")
_POSITION_ATTR_RE = re.compile(r'\s(?:id|style|href|src|width|height)="[^"]*"')
_HREF_RE = re.compile(r'href="(#[^"]*)"')


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """All test rows when n<=0 or n>=len; else a deterministic per-kind-balanced
    subsample so every kind keeps representation."""
    if n <= 0 or n >= len(rows):
        return sorted(rows, key=lambda r: (r.get("gap_kind", ""), r.get("arxiv_id") or "",
                                           r.get("variant_idx", 0)))
    import random
    rng = random.Random(seed)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r.get("gap_kind", "?")].append(r)
    per = max(1, n // max(1, len(by_kind)))
    picked: list[dict] = []
    for _, krows in by_kind.items():
        krows = list(krows)
        rng.shuffle(krows)
        picked.extend(krows[:per])
    picked.sort(key=lambda r: (r.get("gap_kind", ""), r.get("arxiv_id") or "",
                               r.get("variant_idx", 0)))
    return picked


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _fragment(text: str) -> str:
    text = text.strip().split("<|im_end|>")[0].strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", re.sub(r">\s+<", "><", s)).strip()


def _strip_position(s: str) -> str:
    return _norm(_POSITION_ATTR_RE.sub("", s))


def _wellformed(s: str) -> bool:
    # Wrap in a root so bare-text or multi-node fragments still parse.
    try:
        ET.fromstring(f"<root>{s}</root>")
        return True
    except Exception:
        return False


def _first_href(s: str) -> str | None:
    m = _HREF_RE.search(s)
    return m.group(1) if m else None


def score_one(gen_raw: str, gold: str, gap_kind: str) -> dict:
    gen = _fragment(gen_raw)
    gold_s = gold.strip()
    truncated = bool(gen) and not gen.rstrip().endswith((">",))  and "<" in gen
    gen_ns, gold_ns = _strip_position(gen), _strip_position(gold_s)
    scores = {
        "exact": gen == gold_s,
        "norm": _norm(gen) == _norm(gold_s),
        "id_stripped_exact": gen_ns == gold_ns,
        "id_stripped_sim": SequenceMatcher(None, gen_ns, gold_ns).ratio(),
        "wellformed": _wellformed(gen),
        "truncated": truncated,
        "char_sim": SequenceMatcher(None, gen, gold_s).ratio(),
        "gen_len": len(gen),
        "citation_correct": None,
    }
    if gap_kind == "citation_unresolved":
        gold_href, gen_href = _first_href(gold_s), _first_href(gen)
        if gold_href is not None:
            # positive: must restore the SAME anchor
            scores["citation_correct"] = gen_href == gold_href
        else:
            # negative / no-valid-target: must NOT hallucinate an anchor
            scores["citation_correct"] = gen_href is None
    return scores


def aggregate(per_row: list[dict]) -> dict:
    n = len(per_row)
    if n == 0:
        return {}
    bool_keys = ("exact", "norm", "id_stripped_exact", "wellformed", "truncated")
    agg = {k: sum(1 for r in per_row if r["scores"][k]) / n for k in bool_keys}
    agg["id_stripped_sim_mean"] = sum(r["scores"]["id_stripped_sim"] for r in per_row) / n
    agg["char_sim_mean"] = sum(r["scores"]["char_sim"] for r in per_row) / n
    cit = [r for r in per_row if r["scores"]["citation_correct"] is not None]
    agg["citation_correct"] = (
        sum(1 for r in cit if r["scores"]["citation_correct"]) / len(cit) if cit else None
    )
    agg["n_citation"] = len(cit)
    if per_row and "axe_pass" in per_row[0]["scores"]:
        agg["axe_pass"] = sum(1 for r in per_row if r["scores"]["axe_pass"]) / n
        agg["axe_regression"] = sum(1 for r in per_row if r["scores"]["axe_regression"]) / n
    agg["n"] = n
    return agg


def per_class(per_row: list[dict]) -> dict:
    by_k: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_k[r["gap_kind"]].append(r)
    return {k: aggregate(rows) for k, rows in sorted(by_k.items())}


# --------------------------------------------------------------------------- #
# axe (reuse the prod WCAG 2.2 AA gate)
# --------------------------------------------------------------------------- #
def _wrap_doc(fragment: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>fragment</title></head><body>{fragment}</body></html>")


def run_axe(fragments: list[str]) -> list[dict]:
    from semantik_structure.validate import HtmlValidator
    out: list[dict] = []
    with HtmlValidator() as v:
        for frag in fragments:
            res = v.check(_wrap_doc(frag))
            out.append({"ok": res.ok, "blocking": sorted(
                {f"{vi.impact}:{vi.id}" for vi in res.violations
                 if vi.impact in {"serious", "critical"}})})
    return out


def attach_axe(per_row: list[dict], gens: list[str], golds: list[str]) -> dict:
    gold_axe, gen_axe = run_axe(golds), run_axe(gens)
    n = len(per_row)
    for row, ga, na in zip(per_row, gold_axe, gen_axe):
        row["scores"]["axe_pass"] = na["ok"]
        row["scores"]["axe_regression"] = bool(ga["ok"] and not na["ok"])
        row["axe_blocking"] = na["blocking"]
    return {"gold_pass": sum(1 for r in gold_axe if r["ok"]) / n if n else 0.0,
            "gold_blocking_rules": dict(Counter(
                r for a in gold_axe for r in a["blocking"]).most_common())}


# --------------------------------------------------------------------------- #
# Backend: safetensors (base 4-bit + LoRA, exactly as training loaded it)
# --------------------------------------------------------------------------- #
def gen_safetensors(sample, adapter_dir, base_model, max_new_tokens):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from semantik_structure.qwen_specialists.chat_format import wrap_for_qwen

    tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    print(f"[safetensors] loading base {base_model} (4-bit) + LoRA {adapter_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    outs: list[str] = []
    for i, row in enumerate(sample):
        prompt = row["request_payload"]["payload"]["prompt"]
        wrapped = wrap_for_qwen(tok, prompt, add_generation_prompt=True)
        ids = tok(wrapped, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i + 1) % 50 == 0:
            print(f"[safetensors] {i + 1}/{len(sample)}", flush=True)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return outs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir", type=Path, default=DEFAULT_GAP_DIR / "final")
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--n", type=int, default=0, help="0 = full test set; else per-kind subsample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--no-axe", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args()

    if not args.test_path.exists():
        sys.exit(f"[fatal] test split not found: {args.test_path}")
    if not (args.adapter_dir / "adapter_config.json").exists():
        sys.exit(f"[fatal] no adapter_config.json under {args.adapter_dir}")
    base_model = json.loads(args.summary.read_text())["base_model"]

    rows = _load_jsonl(args.test_path)
    sample = select_sample(rows, args.n, args.seed)
    dist = Counter(r.get("gap_kind", "?") for r in sample)
    print(f"[sample] {len(sample)} rows by gap_kind: {dict(dist)}")

    t0 = time.time()
    completions = gen_safetensors(sample, args.adapter_dir, base_model, args.max_new_tokens)
    gen_wall = time.time() - t0

    per_row = [{
        "doc_id": r.get("arxiv_id"),
        "gap_kind": r.get("gap_kind", "?"),
        "scores": score_one(c, r["target_html"], r.get("gap_kind", "?")),
    } for r, c in zip(sample, completions)]

    gold_axe_summary = None
    if not args.no_axe:
        golds = [r["target_html"] for r in sample]
        gens = [_fragment(c) for c in completions]
        gold_axe_summary = attach_axe(per_row, gens, golds)

    report = {
        "adapter_dir": str(args.adapter_dir), "base_model": base_model,
        "test_path": str(args.test_path), "n_sampled": len(sample),
        "max_new_tokens": args.max_new_tokens, "gen_wall_seconds": round(gen_wall, 1),
        "axe_enabled": not args.no_axe, "sample_dist": dict(dist),
        "overall": aggregate(per_row), "per_gap_kind": per_class(per_row),
    }
    if gold_axe_summary:
        report["gold_axe_baseline"] = gold_axe_summary

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "qwen_gap_fill_adapter_final.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    samples_path = out_path.with_suffix(".samples.jsonl")
    with samples_path.open("w", encoding="utf-8") as f:
        for r, c, pr in zip(sample, completions, per_row):
            f.write(json.dumps({"doc_id": r.get("arxiv_id"), "gap_kind": pr["gap_kind"],
                                "gold": r["target_html"], "gen": c, "scores": pr["scores"]},
                               ensure_ascii=False) + "\n")

    o = report["overall"]
    print("\n" + "=" * 64)
    print(f"  Qwen gap_fill adapter — final (n={o['n']}, {gen_wall:.0f}s gen)")
    print("=" * 64)
    print(f"  id-stripped struct exact: {o['id_stripped_exact']:.3f}  <- primary")
    print(f"  id-stripped struct sim  : {o['id_stripped_sim_mean']:.3f}")
    print(f"  XML well-formed         : {o['wellformed']:.3f}")
    print(f"  truncated at cap        : {o['truncated']:.3f}")
    if o.get("citation_correct") is not None:
        print(f"  citation anchor-correct : {o['citation_correct']:.3f}  (n_cit={o['n_citation']})")
    if "axe_pass" in o:
        gp = report.get("gold_axe_baseline", {}).get("gold_pass")
        print(f"  axe wcag22aa pass       : {o['axe_pass']:.3f}  (gold {gp:.3f})" if gp is not None
              else f"  axe wcag22aa pass       : {o['axe_pass']:.3f}")
        print(f"  axe regression vs gold  : {o['axe_regression']:.3f}")
    print("  per gap_kind:")
    for k, agg in report["per_gap_kind"].items():
        cc = f" cite_ok={agg['citation_correct']:.2f}" if agg.get("citation_correct") is not None else ""
        axe = f" axe={agg['axe_pass']:.2f}" if "axe_pass" in agg else ""
        print(f"    {k:20} n={agg['n']:<4} id_exact={agg['id_stripped_exact']:.2f} "
              f"id_sim={agg['id_stripped_sim_mean']:.2f} wf={agg['wellformed']:.2f}{cc}{axe}")
    print(f"\n  report : {out_path}\n  samples: {samples_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
