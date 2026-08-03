"""Generative eval for the trained Qwen3-4B LoRA *prose* adapter.

Mirror of ``scripts/eval/eval_qwen_math_adapter.py`` for the prose specialist
(trained via ``scripts/training/train_qwen_lora.py --adapter prose``). It decodes
completions on the held-out prose test split and scores them against
``target_html``.

Where the math harness compares two *backends* (safetensors vs gguf) over
one adapter, this harness compares two *checkpoints* of the same adapter
over one backend (safetensors) — answering "did the last training steps
help?". The default comparison is ``final`` vs ``checkpoint-7000``; both
are HF LoRA dirs loaded exactly as training loaded them (base Qwen3-4B in
4-bit + peft). Checkpoints are generated SERIALLY (8GB GPU only fits one
4-bit Qwen3-4B at a time — see memory: qwen builds run serial), each over
the identical sample so the numbers are directly comparable.

Metrics (all mirror the math harness, retargeted to accessible HTML):

  * ``id_stripped_exact`` / ``id_stripped_sim`` — PRIMARY. Position- and
    layout-encoding attribute values (id / style / href / src / width /
    height) are NOT in the prompt, so the adapter cannot reproduce them
    and a raw string match is meaningless. We drop those whole attributes
    and whitespace-normalize to score the structure the adapter controls.
  * ``wellformed`` — does the fragment parse as XML. Gold is 100%
    single-root well-formed XML, so this catches mid-decode truncation
    and tag imbalance the model introduces.
  * ``axe_pass`` — prose targets ARE accessible HTML, so we run the real
    axe-core wcag22aa ruleset (the same gate that admitted the training
    data, ``semantik_structure.validate.HtmlValidator``) over each fragment
    wrapped in a minimal lang-tagged document. Reported alongside the
    GOLD pass rate as a baseline, because some WCAG 2.2 rules (e.g.
    target-size on small inline cite links) fire on gold too without real
    CSS. The meaningful signal is ``axe_regression``: gold passed but the
    generated fragment failed — an a11y defect the model introduced.

Sampling
--------
The test split is ~92% ``paragraph``. A proportional sample would contain
~0 figure/list/blockquote rows — exactly the structured kinds the adapter
is most likely to botch. So we OVERSAMPLE rare kinds (cap each, take all
of the very rare ones) and keep a paragraph floor for the dominant class.
The headline rate is therefore NOT population-weighted — always read the
per-kind table alongside it.

Usage::

    .venv/bin/python -m scripts.eval_qwen_prose_adapter            # final vs checkpoint-7000
    .venv/bin/python -m scripts.eval_qwen_prose_adapter \\
        --checkpoint final=models/qwen_specialists/prose/v1/final \\
        --checkpoint ckpt7k=models/qwen_specialists/prose/v1/checkpoint-7000 \\
        --n 400 --no-axe        # skip the (slow) Chromium axe pass
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

DEFAULT_PROSE_DIR = REPO_ROOT / "models/qwen_specialists/prose/v1"
DEFAULT_SUMMARY = DEFAULT_PROSE_DIR / "summary.json"
DEFAULT_TEST = REPO_ROOT / "data/qwen_prose_dataset/test.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "data/eval_reports"

# Default head-to-head: the shipped adapter vs the 1000-steps-earlier
# checkpoint. name=path pairs, evaluated serially in this order.
DEFAULT_CHECKPOINTS = [
    f"final={DEFAULT_PROSE_DIR / 'final'}",
    f"checkpoint-7000={DEFAULT_PROSE_DIR / 'checkpoint-7000'}",
]

# Rare kinds we guarantee in the sample (everything that isn't the
# dominant paragraph class), with a per-kind cap so a single structured
# class can't crowd out the paragraph floor.
_RARE_KIND_CAP = 80
_PARAGRAPH_FLOOR = 120

_WS_RE = re.compile(r"\s+")
# Attribute VALUES that encode document position / PDF layout / asset
# locations — none of which are in the prompt, so the adapter cannot
# reproduce them. Drop the whole attribute (mirrors the math harness's
# id/xref strip) to score the structure + visible content it controls.
_POSITION_ATTR_RE = re.compile(r'\s(?:id|style|href|src|width|height)="[^"]*"')
_IMG_RE = re.compile(r"<img\b", re.IGNORECASE)
_ALT_RE = re.compile(r'\salt="(.*?)"', re.DOTALL)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic rare-kind-oversampled sample of size ~n.

    Takes up to ``_RARE_KIND_CAP`` rows of each non-paragraph kind (all of
    them when a kind is rarer than the cap), then fills the remainder with
    paragraph rows down to a floor. Same (rows, n, seed) => byte-identical
    selection, so every checkpoint scores the exact same items.
    """
    import random

    rng = random.Random(seed)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r.get("kind", "paragraph")].append(r)

    picked: list[dict] = []
    for kind, krows in by_kind.items():
        if kind == "paragraph":
            continue
        krows = list(krows)
        rng.shuffle(krows)
        picked.extend(krows[:_RARE_KIND_CAP])

    paragraphs = list(by_kind.get("paragraph", []))
    rng.shuffle(paragraphs)
    remaining = max(_PARAGRAPH_FLOOR, n - len(picked))
    picked.extend(paragraphs[:remaining])

    # Stable ordering keyed on content ids (not list order) so the sample
    # is invariant to upstream file reordering.
    picked.sort(key=lambda r: (r.get("arxiv_id") or "", r.get("region_idx", 0)))
    return picked


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _fragment(text: str) -> str:
    """Strip chat terminators / code fences the decoder may leave in.

    The prose root tag varies (p / ul / ol / figure / blockquote / div /
    dl), so unlike the math harness we cannot trim to a fixed tag — we
    just clean wrappers and return the fragment as emitted.
    """
    text = text.strip()
    text = text.split("<|im_end|>")[0].strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _norm(s: str) -> str:
    """Whitespace-collapse for a lenient structural comparison."""
    s = re.sub(r">\s+<", "><", s)
    return _WS_RE.sub(" ", s).strip()


def _strip_position(s: str) -> str:
    """Drop position/layout/asset attribute values, then normalize. Yields
    a comparison key that ignores the document-position info the prompt
    can't supply, isolating structure + visible content the adapter
    controls."""
    return _norm(_POSITION_ATTR_RE.sub("", s))


def _wellformed(s: str) -> bool:
    try:
        ET.fromstring(s)
        return True
    except Exception:
        return False


def score_one(gen_raw: str, gold: str) -> dict:
    gen = _fragment(gen_raw)
    gold_s = gold.strip()
    # A complete fragment ends on a closing '>'; otherwise the decode hit
    # the token cap mid-tag/mid-text (the main cause of ill-formed output).
    truncated = not gen.rstrip().endswith(">")
    gen_ns, gold_ns = _strip_position(gen), _strip_position(gold_s)
    gold_has_img = bool(_IMG_RE.search(gold_s))
    g_alt = _ALT_RE.findall(gen)
    return {
        "exact": gen == gold_s,
        "norm": _norm(gen) == _norm(gold_s),
        # structural fidelity with unknowable position/layout values
        # neutralized — the primary quality signal for this adapter.
        "id_stripped_exact": gen_ns == gold_ns,
        "id_stripped_sim": SequenceMatcher(None, gen_ns, gold_ns).ratio(),
        "wellformed": _wellformed(gen),
        "truncated": truncated,
        "char_sim": SequenceMatcher(None, gen, gold_s).ratio(),
        # alt-text presence is only meaningful where the gold has an image.
        "gold_has_img": gold_has_img,
        "alt_present": bool(gold_has_img and g_alt and any(a for a in g_alt)),
        "gen_len": len(gen),
    }


def aggregate(per_row: list[dict]) -> dict:
    n = len(per_row)
    if n == 0:
        return {}
    bool_keys = (
        "exact",
        "norm",
        "id_stripped_exact",
        "wellformed",
        "truncated",
    )
    agg = {k: sum(1 for r in per_row if r["scores"][k]) / n for k in bool_keys}
    agg["id_stripped_sim_mean"] = (
        sum(r["scores"]["id_stripped_sim"] for r in per_row) / n
    )
    agg["char_sim_mean"] = sum(r["scores"]["char_sim"] for r in per_row) / n
    # alt-text rate is conditional on the gold actually having an image.
    img_rows = [r for r in per_row if r["scores"]["gold_has_img"]]
    agg["alt_present"] = (
        sum(1 for r in img_rows if r["scores"]["alt_present"]) / len(img_rows)
        if img_rows
        else None
    )
    agg["n_img"] = len(img_rows)
    # axe metrics are attached post-hoc (see attach_axe); guard their absence.
    if per_row and "axe_pass" in per_row[0]["scores"]:
        agg["axe_pass"] = sum(1 for r in per_row if r["scores"]["axe_pass"]) / n
        agg["axe_regression"] = (
            sum(1 for r in per_row if r["scores"]["axe_regression"]) / n
        )
    agg["n"] = n
    return agg


def per_class(per_row: list[dict]) -> dict:
    out: dict[str, dict] = {}
    by_k: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_k[r["kind"]].append(r)
    for k, rows in sorted(by_k.items()):
        out[k] = aggregate(rows)
    return out


# ---------------------------------------------------------------------------
# axe-core accessibility pass (reuses the production WCAG 2.2 AA gate)
# ---------------------------------------------------------------------------
def _wrap_doc(fragment: str) -> str:
    """Wrap a fragment in a minimal lang-tagged HTML5 document so axe scores
    the FRAGMENT's accessibility, not document-chrome rules (html-has-lang,
    document-title) that aren't the adapter's responsibility."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>fragment</title></head><body>{fragment}</body></html>"
    )


def run_axe(fragments: list[str]) -> list[dict]:
    """Run the wcag22aa axe gate over each fragment. Returns one dict per
    fragment: {ok, error, blocking:[rule ids]}. Malformed HTML is reported
    as a failure rather than crashing the batch."""
    from semantik_structure.validate import HtmlValidator

    results: list[dict] = []
    with HtmlValidator() as v:
        for frag in fragments:
            res = v.check(_wrap_doc(frag))
            results.append(
                {
                    "ok": res.ok,
                    "error": res.error,
                    "blocking": sorted(
                        {f"{vi.impact}:{vi.id}" for vi in res.violations if vi.impact in {"serious", "critical"}}
                    ),
                }
            )
    return results


def attach_axe(per_row: list[dict], gens: list[str], golds: list[str]) -> dict:
    """Run axe over gold + generated fragments and fold the verdicts into
    each row's scores. Returns the gold-baseline summary."""
    gold_axe = run_axe(golds)
    gen_axe = run_axe(gens)
    n = len(per_row)
    gold_pass = sum(1 for r in gold_axe if r["ok"])
    for row, ga, na in zip(per_row, gold_axe, gen_axe):
        row["scores"]["axe_pass"] = na["ok"]
        # the meaningful WCAG signal: gold was clean, generation broke it.
        row["scores"]["axe_regression"] = bool(ga["ok"] and not na["ok"])
        row["axe_blocking"] = na["blocking"]
        row["gold_axe_blocking"] = ga["blocking"]
    return {
        "gold_pass": gold_pass / n if n else 0.0,
        "gold_blocking_rules": dict(
            Counter(rule for r in gold_axe for rule in r["blocking"]).most_common()
        ),
    }


# ---------------------------------------------------------------------------
# Backend: safetensors (base 4-bit + LoRA, exactly as training loaded it)
# ---------------------------------------------------------------------------
def gen_safetensors(
    sample: list[dict], adapter_dir: Path, base_model: str, max_new_tokens: int
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from semantik_structure.qwen_specialists.chat_format import wrap_for_qwen

    tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"[safetensors] loading base {base_model} (4-bit) + LoRA {adapter_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    outs: list[str] = []
    for i, row in enumerate(sample):
        prompt = row["request_payload"]["payload"]["prompt"]
        wrapped = wrap_for_qwen(tok, prompt, add_generation_prompt=True)
        ids = tok(wrapped, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
        new = gen[0][ids["input_ids"].shape[1] :]
        outs.append(tok.decode(new, skip_special_tokens=True))
        if (i + 1) % 25 == 0:
            print(f"[safetensors] {i + 1}/{len(sample)}", flush=True)

    # Free VRAM before the next checkpoint loads — the 8GB GPU only fits
    # one 4-bit Qwen3-4B at a time.
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return outs


# ---------------------------------------------------------------------------
# Per-checkpoint scoring
# ---------------------------------------------------------------------------
def score_checkpoint(
    name: str,
    sample: list[dict],
    completions: list[str],
    *,
    run_axe_pass: bool,
) -> tuple[dict, list[dict]]:
    per_row = []
    for row, comp in zip(sample, completions):
        per_row.append(
            {
                "arxiv_id": row.get("arxiv_id"),
                "region_idx": row.get("region_idx"),
                "kind": row.get("kind", "paragraph"),
                "scores": score_one(comp, row["target_html"]),
            }
        )
    gold_axe_summary = None
    if run_axe_pass:
        golds = [r["target_html"] for r in sample]
        gens = [_fragment(c) for c in completions]
        gold_axe_summary = attach_axe(per_row, gens, golds)

    report = {
        "checkpoint": name,
        "overall": aggregate(per_row),
        "per_class": per_class(per_row),
    }
    if gold_axe_summary is not None:
        report["gold_axe_baseline"] = gold_axe_summary
    return report, per_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_checkpoints(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for it in items:
        if "=" not in it:
            sys.exit(f"[fatal] --checkpoint expects name=path, got: {it!r}")
        name, path = it.split("=", 1)
        out.append((name, Path(path)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        help="name=path of a LoRA adapter dir; repeatable. "
        "Default: final + checkpoint-7000.",
    )
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--n", type=int, default=400, help="approx sample size")
    ap.add_argument("--seed", type=int, default=0)
    # Max gold target is 373 tokens; 512 covers the full tail with headroom.
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--no-axe",
        action="store_true",
        help="skip the (slow, Chromium-backed) axe-core accessibility pass",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args()

    if not args.test_path.exists():
        sys.exit(f"[fatal] test split not found: {args.test_path}")
    checkpoints = _parse_checkpoints(args.checkpoints or DEFAULT_CHECKPOINTS)
    for name, path in checkpoints:
        if not (path / "adapter_config.json").exists():
            sys.exit(f"[fatal] {name}: no adapter_config.json under {path}")

    base_model = json.loads(args.summary.read_text())["base_model"]
    rows = _load_jsonl(args.test_path)
    sample = select_sample(rows, args.n, args.seed)
    sample_dist = Counter(r.get("kind", "paragraph") for r in sample)
    print(
        f"[sample] {len(sample)} rows (seed={args.seed}) "
        f"by kind: {dict(sample_dist)}"
    )
    run_axe_pass = not args.no_axe

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    t_all = time.time()
    for name, path in checkpoints:
        print("\n" + "-" * 64)
        print(f"  checkpoint: {name}  ({path})")
        print("-" * 64)
        t0 = time.time()
        completions = gen_safetensors(sample, path, base_model, args.max_new_tokens)
        gen_wall = time.time() - t0
        report, per_row = score_checkpoint(
            name, sample, completions, run_axe_pass=run_axe_pass
        )
        report.update(
            {
                "adapter_dir": str(path),
                "base_model": base_model,
                "test_path": str(args.test_path),
                "n_sampled": len(sample),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "gen_wall_seconds": round(gen_wall, 1),
                "axe_enabled": run_axe_pass,
                "sample_dist": dict(sample_dist),
            }
        )
        reports[name] = report

        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        out_path = args.out_dir / f"qwen_prose_adapter_{safe}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        samples_path = out_path.with_suffix(".samples.jsonl")
        with samples_path.open("w", encoding="utf-8") as f:
            for row, comp, pr in zip(sample, completions, per_row):
                f.write(
                    json.dumps(
                        {
                            "arxiv_id": row.get("arxiv_id"),
                            "region_idx": row.get("region_idx"),
                            "kind": row.get("kind", "paragraph"),
                            "gold": row["target_html"],
                            "gen": comp,
                            "scores": pr["scores"],
                            "axe_blocking": pr.get("axe_blocking"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        _print_one(report)
        print(f"\n  report : {out_path}")
        print(f"  samples: {samples_path}")

    if len(reports) > 1:
        _print_comparison(reports)
        cmp_path = args.out_dir / "qwen_prose_adapter_comparison.json"
        cmp_path.write_text(
            json.dumps(
                {n: r["overall"] for n, r in reports.items()}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\n  comparison: {cmp_path}")
    print(f"\n[done] total wall {time.time() - t_all:.0f}s")
    return 0


def _print_one(report: dict) -> None:
    o = report["overall"]
    name = report["checkpoint"]
    print("\n" + "=" * 64)
    print(f"  Qwen prose adapter — {name}  (n={o['n']}, {report['gen_wall_seconds']:.0f}s gen)")
    print("=" * 64)
    print(f"  exact match            : {o['exact']:.3f}  (position-bearing; ~floor by design)")
    print(f"  id-stripped struct exact: {o['id_stripped_exact']:.3f}  <- primary")
    print(f"  id-stripped struct sim  : {o['id_stripped_sim_mean']:.3f}")
    print(f"  XML well-formed         : {o['wellformed']:.3f}")
    print(f"  truncated at cap        : {o['truncated']:.3f}")
    print(f"  char similarity         : {o['char_sim_mean']:.3f}")
    if o.get("alt_present") is not None:
        print(f"  img alt present         : {o['alt_present']:.3f}  (n_img={o['n_img']})")
    if "axe_pass" in o:
        gp = report.get("gold_axe_baseline", {}).get("gold_pass")
        gp_s = f"{gp:.3f}" if gp is not None else "n/a"
        print(f"  axe wcag22aa pass       : {o['axe_pass']:.3f}  (gold baseline {gp_s})")
        print(f"  axe regression vs gold  : {o['axe_regression']:.3f}  <- a11y defects introduced")
    print("  per-class (kind):")
    for k, agg in report["per_class"].items():
        axe = f" axe={agg['axe_pass']:.2f}" if "axe_pass" in agg else ""
        print(
            f"    {k:16} n={agg['n']:<4} id_exact={agg['id_stripped_exact']:.2f} "
            f"id_sim={agg['id_stripped_sim_mean']:.2f} "
            f"wf={agg['wellformed']:.2f} trunc={agg['truncated']:.2f}{axe}"
        )


def _print_comparison(reports: dict) -> None:
    names = list(reports)
    metrics = [
        ("id_stripped_exact", "id-stripped struct exact"),
        ("id_stripped_sim_mean", "id-stripped struct sim"),
        ("wellformed", "XML well-formed"),
        ("truncated", "truncated at cap"),
        ("char_sim_mean", "char similarity"),
        ("axe_pass", "axe wcag22aa pass"),
        ("axe_regression", "axe regression vs gold"),
    ]
    print("\n" + "=" * 64)
    print("  COMPARISON  (" + "  vs  ".join(names) + ")")
    print("=" * 64)
    col = "  {:<26}" + "{:>12}" * len(names) + "{:>10}"
    print(col.format("metric", *names, "Δ(last-1st)"))
    for key, label in metrics:
        vals = [reports[n]["overall"].get(key) for n in names]
        if any(v is None for v in vals):
            continue
        delta = vals[-1] - vals[0]
        print(col.format(label, *[f"{v:.3f}" for v in vals], f"{delta:+.3f}"))
    print("\n  (Δ>0 means the LAST checkpoint listed is higher on that metric;")
    print("   for 'truncated' and 'axe regression' lower is better.)")


if __name__ == "__main__":
    sys.exit(main())
