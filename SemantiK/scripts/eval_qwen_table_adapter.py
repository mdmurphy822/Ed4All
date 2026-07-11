"""Generative eval for the trained Qwen3-4B LoRA *table* adapter.

Mirror of ``scripts/eval_qwen_prose_adapter.py`` (and the math harness it
descends from), retargeted to the table specialist (trained via
``scripts/train_qwen_lora.py --adapter table``). It decodes completions on
the held-out table test split (``data/qwen_table_dataset_multi/test.jsonl``)
and scores them against ``target_html`` — a single accessible HTML5
``<table>`` with ``<caption>``, ``<thead>``/``<tbody>`` and
``<th scope=...>``.

Two things this harness does that the prose one does not:

  * **Source-stratified scoring.** The table set is 69% PMC (see Plans/06
    §6 — accepted, not diluted, because the dense gov tables that would
    balance it are too long for the 2048 cap). A single headline number
    would just report "how good at PMC". So every metric is also broken
    out ``by_source`` (pmc / arxiv / openstax / cfr) and ``by_table_type``
    (simple / scientific / matrix / dense_data / regulatory). Always read
    the breakdown, not just the headline.
  * **Table-structure metrics.** Beyond id-stripped similarity it scores
    the grid (does the generated table have the same rows × columns as
    gold — a dropped/added cell is a real defect) and the WCAG-bearing
    structural features (``<caption>`` / ``<th scope>`` / ``<thead>``),
    each conditional on gold actually having that feature.

Generation cap
--------------
The REAL Qwen-tokenized target distribution on the test split is p50=348,
p99=630, **max=780** tokens — and 54/459 rows exceed 512. So the prose
default of 512 would truncate ~12% of tables mid-grid. We default
``--max-new-tokens 1024`` (0 rows exceed it). This mirrors the production
rule that a shipped cap must clear the target-length tail; quote the
distribution if you ever lower it.

Metrics
-------
  * ``id_stripped_exact`` / ``id_stripped_sim`` — PRIMARY. Position/asset
    attribute values (id/style/href/src/width/height) aren't in the prompt,
    so we drop them and whitespace-normalize before comparing.
  * ``grid_match`` — generated (rows, cols) == gold (rows, cols), colspan
    aware. The structural-integrity signal unique to tables.
  * ``caption_present`` / ``scope_present`` / ``thead_present`` —
    conditional on gold having the feature; the accessibility-bearing
    structure the adapter must reproduce.
  * ``wellformed`` — fragment parses as XML (catches truncation / tag
    imbalance).
  * ``axe_pass`` / ``axe_regression`` — the real wcag22aa axe gate
    (``semantik_structure.validate.HtmlValidator``) over each table wrapped in a
    minimal lang-tagged doc. Tables trigger table-specific WCAG rules
    (td-headers-attr, th-has-data-cells, scope-attr-valid), so this is the
    sharpest accessibility signal of any adapter. ``axe_regression`` (gold
    clean, generation broke it) is the meaningful number.

Usage::

    .venv/bin/python -m scripts.eval_qwen_table_adapter            # final, all 459 rows
    .venv/bin/python -m scripts.eval_qwen_table_adapter --no-axe   # skip Chromium
    .venv/bin/python -m scripts.eval_qwen_table_adapter \\
        --checkpoint final=models/qwen_specialists/table/v1/final \\
        --checkpoint ckpt1200=models/qwen_specialists/table/v1/checkpoint-1200
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TABLE_DIR = REPO_ROOT / "models/qwen_specialists/table/v1"
DEFAULT_SUMMARY = DEFAULT_TABLE_DIR / "summary.json"
DEFAULT_TEST = REPO_ROOT / "data/qwen_table_dataset_multi/test.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "data/eval_reports"
# Fallback base if summary.json isn't present yet (e.g. mid-train smoke).
FALLBACK_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# First-train default: score the shipped adapter only. Add more name=path
# pairs (e.g. an earlier checkpoint) to get the prose-style comparison table.
DEFAULT_CHECKPOINTS = [f"final={DEFAULT_TABLE_DIR / 'final'}"]

# When --n caps below the test size, guarantee the rare table_types survive
# (a proportional cut would drop dense_data/regulatory to ~0).
_RARE_TYPE_CAP = 60
_COMMON_TYPE_CAP = 140

_WS_RE = re.compile(r"\s+")
_POSITION_ATTR_RE = re.compile(r'\s(?:id|style|href|src|width|height)="[^"]*"')


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


def _row_id(r: dict) -> tuple:
    """Stable identity for deterministic ordering across checkpoints."""
    return (r.get("source") or "", str(r.get("doc_id") or r.get("arxiv_id") or ""),
            int(r.get("table_idx", 0)))


def _interleave_by_source(rows: list[dict]) -> list[dict]:
    """Round-robin across sources so a sequential run (and any live peek)
    sees a representative mix early, instead of all of one source first.
    Deterministic: rows within each source are ordered by ``_row_id`` and the
    sources are cycled in sorted order."""
    from collections import deque

    buckets: dict[str, deque] = defaultdict(deque)
    for r in sorted(rows, key=_row_id):
        buckets[r.get("source") or ""].append(r)
    order = sorted(buckets)
    out: list[dict] = []
    while any(buckets[s] for s in order):
        for s in order:
            if buckets[s]:
                out.append(buckets[s].popleft())
    return out


def select_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """All rows when ``n<=0`` or ``n>=len`` (the default — the test split is
    small). Otherwise a table_type-oversampled cut so the rare types
    (dense_data/regulatory/matrix) don't vanish. Deterministic in (rows,n,seed).
    Final order is interleaved across sources (see ``_interleave_by_source``).
    """
    if n <= 0 or n >= len(rows):
        return _interleave_by_source(rows)

    import random

    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r.get("table_type", "simple")].append(r)
    picked: list[dict] = []
    # "common" = the two big classes; everything else is rare and capped high.
    common = {"simple", "scientific"}
    for ttype, trows in by_type.items():
        trows = list(trows)
        rng.shuffle(trows)
        cap = _COMMON_TYPE_CAP if ttype in common else _RARE_TYPE_CAP
        picked.extend(trows[:cap])
    rng.shuffle(picked)
    picked = picked[:n] if n < len(picked) else picked
    return _interleave_by_source(picked)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _fragment(text: str) -> str:
    """Strip chat terminators / code fences and trim to the ``<table>`` span.

    Unlike prose (variable root), a table target is always rooted at a single
    ``<table>``, so we trim to the first ``<table`` … last ``</table>`` —
    mirroring the math harness's fixed-root trim. Junk the decoder may emit
    before/after the table is discarded.
    """
    text = text.strip().split("<|im_end|>")[0].strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    i = text.find("<table")
    j = text.rfind("</table>")
    if i != -1 and j != -1 and j > i:
        text = text[i : j + len("</table>")]
    return text


def _norm(s: str) -> str:
    s = re.sub(r">\s+<", "><", s)
    return _WS_RE.sub(" ", s).strip()


def _strip_position(s: str) -> str:
    return _norm(_POSITION_ATTR_RE.sub("", s))


def _wellformed(s: str) -> bool:
    try:
        ET.fromstring(s)
        return True
    except Exception:
        return False


def _grid(html: str) -> tuple[int, int, int] | None:
    """(n_rows, widest_cols, total_cells) of a table, colspan-aware. None if
    it doesn't parse — so a malformed generation never silently scores as a
    grid match. ``total_cells`` (sum of colspans over every cell) is carried
    alongside ``widest_cols`` so a cell dropped from a non-widest row — which
    leaves the max width unchanged — still registers as a shape mismatch."""
    try:
        root = ET.fromstring(html)
    except Exception:
        return None
    rows = root.findall(".//tr")
    if not rows:
        return (0, 0, 0)
    max_cols = total = 0
    for tr in rows:
        c = 0
        for cell in tr:
            if cell.tag.lower() in ("td", "th"):
                try:
                    c += int(cell.get("colspan", "1"))
                except ValueError:
                    c += 1
        max_cols = max(max_cols, c)
        total += c
    return (len(rows), max_cols, total)


def score_one(gen_raw: str, gold: str) -> dict:
    gen = _fragment(gen_raw)
    gold_s = gold.strip()
    truncated = not gen.rstrip().endswith("</table>")
    gen_ns, gold_ns = _strip_position(gen), _strip_position(gold_s)

    g_grid, gold_grid = _grid(gen), _grid(gold_s)

    # gold-conditional structural features (only meaningful where gold has it)
    gold_caption = "<caption" in gold_s
    gold_scope = "scope=" in gold_s
    gold_thead = "<thead" in gold_s
    return {
        "exact": gen == gold_s,
        "norm": _norm(gen) == _norm(gold_s),
        "id_stripped_exact": gen_ns == gold_ns,
        "id_stripped_sim": SequenceMatcher(None, gen_ns, gold_ns).ratio(),
        "wellformed": _wellformed(gen),
        "truncated": truncated,
        "char_sim": SequenceMatcher(None, gen, gold_s).ratio(),
        # structural integrity: same shape grid (colspan-aware), gold parsed.
        "grid_match": bool(g_grid is not None and gold_grid is not None and g_grid == gold_grid),
        "gold_grid": gold_grid,
        "gen_grid": g_grid,
        # accessibility-bearing structure, conditional on gold having it.
        "gold_caption": gold_caption,
        "caption_present": bool(gold_caption and "<caption" in gen),
        "gold_scope": gold_scope,
        "scope_present": bool(gold_scope and "scope=" in gen),
        "gold_thead": gold_thead,
        "thead_present": bool(gold_thead and "<thead" in gen),
        "gen_len": len(gen),
    }


def _cond_rate(per_row: list[dict], gold_key: str, hit_key: str) -> tuple[float | None, int]:
    """Rate of ``hit_key`` over only the rows where ``gold_key`` is true."""
    rel = [r for r in per_row if r["scores"][gold_key]]
    if not rel:
        return None, 0
    return sum(1 for r in rel if r["scores"][hit_key]) / len(rel), len(rel)


def aggregate(per_row: list[dict]) -> dict:
    n = len(per_row)
    if n == 0:
        return {}
    bool_keys = ("exact", "norm", "id_stripped_exact", "wellformed", "truncated", "grid_match")
    agg = {k: sum(1 for r in per_row if r["scores"][k]) / n for k in bool_keys}
    agg["id_stripped_sim_mean"] = sum(r["scores"]["id_stripped_sim"] for r in per_row) / n
    agg["char_sim_mean"] = sum(r["scores"]["char_sim"] for r in per_row) / n
    for gold_key, hit_key, out in (
        ("gold_caption", "caption_present", "caption_present"),
        ("gold_scope", "scope_present", "scope_present"),
        ("gold_thead", "thead_present", "thead_present"),
    ):
        rate, denom = _cond_rate(per_row, gold_key, hit_key)
        agg[out] = rate
        agg[f"n_{out}"] = denom
    if per_row and "axe_pass" in per_row[0]["scores"]:
        agg["axe_pass"] = sum(1 for r in per_row if r["scores"]["axe_pass"]) / n
        agg["axe_regression"] = sum(1 for r in per_row if r["scores"]["axe_regression"]) / n
    agg["n"] = n
    return agg


def _breakdown(per_row: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    by: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by[r[key]].append(r)
    for k, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        out[k] = aggregate(rows)
    return out


# ---------------------------------------------------------------------------
# axe-core accessibility pass (reuses the production WCAG 2.2 AA gate)
# ---------------------------------------------------------------------------
def _wrap_doc(fragment: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>fragment</title></head><body>{fragment}</body></html>"
    )


def run_axe(fragments: list[str]) -> list[dict]:
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
    gold_axe = run_axe(golds)
    gen_axe = run_axe(gens)
    n = len(per_row)
    gold_pass = sum(1 for r in gold_axe if r["ok"])
    for row, ga, na in zip(per_row, gold_axe, gen_axe):
        row["scores"]["axe_pass"] = na["ok"]
        row["scores"]["axe_regression"] = bool(ga["ok"] and not na["ok"])
        row["axe_blocking"] = na["blocking"]
        row["gold_axe_blocking"] = ga["blocking"]
    return {
        "gold_pass": gold_pass / n if n else 0.0,
        "gold_blocking_rules": dict(
            Counter(rule for r in gold_axe for rule in r["blocking"]).most_common()
        ),
        "gen_blocking_rules": dict(
            Counter(rule for r in gen_axe for rule in r["blocking"]).most_common()
        ),
    }


# ---------------------------------------------------------------------------
# Backend: safetensors (base 4-bit + LoRA, exactly as training loaded it)
# ---------------------------------------------------------------------------
def gen_safetensors(
    sample: list[dict], adapter_dir: Path, base_model: str, max_new_tokens: int,
    on_row=None,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    outs: list[str] = []
    for i, row in enumerate(sample):
        prompt = row["request_payload"]["payload"]["prompt"]
        wrapped = wrap_for_qwen(tok, prompt, add_generation_prompt=True)
        ids = tok(wrapped, return_tensors="pt", add_special_tokens=False).to(model.device)
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
        if on_row is not None:
            on_row(i, row, outs[-1])
        if (i + 1) % 25 == 0:
            print(f"[safetensors] {i + 1}/{len(sample)}", flush=True)

    # Free VRAM before the next checkpoint — the 8GB GPU fits one 4-bit Qwen3-4B.
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return outs


# ---------------------------------------------------------------------------
# Per-checkpoint scoring
# ---------------------------------------------------------------------------
def score_checkpoint(
    name: str, sample: list[dict], completions: list[str], *, run_axe_pass: bool
) -> tuple[dict, list[dict]]:
    per_row = []
    for row, comp in zip(sample, completions):
        per_row.append(
            {
                "source": row.get("source", "?"),
                "table_type": row.get("table_type", "?"),
                "doc_id": row.get("doc_id") or row.get("arxiv_id"),
                "table_idx": row.get("table_idx"),
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
        "by_source": _breakdown(per_row, "source"),
        "by_table_type": _breakdown(per_row, "table_type"),
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
    ap.add_argument("--checkpoint", action="append", dest="checkpoints",
                    help="name=path of a LoRA adapter dir; repeatable. Default: final.")
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--base-model", default=None, help="override base model id")
    ap.add_argument("--n", type=int, default=0, help="sample size; 0 = all test rows")
    ap.add_argument("--seed", type=int, default=0)
    # Real target tail is max 780 tok (54 rows > 512); 1024 clears it fully.
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-axe", action="store_true",
                    help="skip the (slow, Chromium-backed) axe-core accessibility pass")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args()

    if not args.test_path.exists():
        sys.exit(f"[fatal] test split not found: {args.test_path}")
    checkpoints = _parse_checkpoints(args.checkpoints or DEFAULT_CHECKPOINTS)
    for name, path in checkpoints:
        if not (path / "adapter_config.json").exists():
            sys.exit(f"[fatal] {name}: no adapter_config.json under {path}")

    if args.base_model:
        base_model = args.base_model
    elif args.summary.exists():
        base_model = json.loads(args.summary.read_text()).get("base_model", FALLBACK_BASE_MODEL)
    else:
        base_model = FALLBACK_BASE_MODEL
    print(f"[base] {base_model}")

    rows = _load_jsonl(args.test_path)
    sample = select_sample(rows, args.n, args.seed)
    src_dist = Counter(r.get("source") for r in sample)
    type_dist = Counter(r.get("table_type") for r in sample)
    print(f"[sample] {len(sample)} rows  by source={dict(src_dist)}  by type={dict(type_dist)}")
    run_axe_pass = not args.no_axe

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    t_all = time.time()
    for name, path in checkpoints:
        print("\n" + "-" * 64)
        print(f"  checkpoint: {name}  ({path})")
        print("-" * 64)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        out_path = args.out_dir / f"qwen_table_adapter_{safe}.json"
        samples_path = out_path.with_suffix(".samples.jsonl")
        # Live stream: write each row's structural score to samples.jsonl AS it
        # generates, so a long run can be peeked mid-flight. axe is batched at
        # the end, so the live rows carry structural scores only; the final
        # write (below) overwrites this file with the complete axe-scored set.
        live = samples_path.open("w", encoding="utf-8")

        def _stream(i: int, row: dict, comp: str, _live=live) -> None:
            _live.write(
                json.dumps(
                    {
                        "i": i,
                        "source": row.get("source"),
                        "table_type": row.get("table_type"),
                        "doc_id": row.get("doc_id") or row.get("arxiv_id"),
                        "gold": row["target_html"],
                        "gen": _fragment(comp),
                        "scores": score_one(comp, row["target_html"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            _live.flush()

        t0 = time.time()
        completions = gen_safetensors(
            sample, path, base_model, args.max_new_tokens, on_row=_stream
        )
        gen_wall = time.time() - t0
        live.close()
        report, per_row = score_checkpoint(name, sample, completions, run_axe_pass=run_axe_pass)
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
                "source_dist": dict(src_dist),
                "table_type_dist": dict(type_dist),
            }
        )
        reports[name] = report

        # out_path / samples_path were computed before generation (live stream).
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Overwrite the live-streamed samples with the complete axe-scored set.
        with samples_path.open("w", encoding="utf-8") as f:
            for row, comp, pr in zip(sample, completions, per_row):
                f.write(
                    json.dumps(
                        {
                            "source": row.get("source"),
                            "table_type": row.get("table_type"),
                            "doc_id": row.get("doc_id") or row.get("arxiv_id"),
                            "table_idx": row.get("table_idx"),
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
        cmp_path = args.out_dir / "qwen_table_adapter_comparison.json"
        cmp_path.write_text(
            json.dumps({n: r["overall"] for n, r in reports.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"\n  comparison: {cmp_path}")
    print(f"\n[done] total wall {time.time() - t_all:.0f}s")
    return 0


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


def _print_one(report: dict) -> None:
    o = report["overall"]
    name = report["checkpoint"]
    print("\n" + "=" * 70)
    print(f"  Qwen table adapter — {name}  (n={o['n']}, {report['gen_wall_seconds']:.0f}s gen)")
    print("=" * 70)
    print(f"  id-stripped struct exact: {_fmt(o['id_stripped_exact'])}  <- primary")
    print(f"  id-stripped struct sim  : {_fmt(o['id_stripped_sim_mean'])}")
    print(f"  grid match (rows×cols)  : {_fmt(o['grid_match'])}  <- structural integrity")
    print(f"  XML well-formed         : {_fmt(o['wellformed'])}")
    print(f"  truncated at cap        : {_fmt(o['truncated'])}")
    print(f"  caption present (gold∩)  : {_fmt(o['caption_present'])}  (n={o['n_caption_present']})")
    print(f"  th scope present (gold∩) : {_fmt(o['scope_present'])}  (n={o['n_scope_present']})")
    print(f"  thead present (gold∩)    : {_fmt(o['thead_present'])}  (n={o['n_thead_present']})")
    if "axe_pass" in o:
        gp = report.get("gold_axe_baseline", {}).get("gold_pass")
        print(f"  axe wcag22aa pass       : {_fmt(o['axe_pass'])}  (gold baseline {_fmt(gp)})")
        print(f"  axe regression vs gold  : {_fmt(o['axe_regression'])}  <- a11y defects introduced")
    for label, key in (("by source", "by_source"), ("by table_type", "by_table_type")):
        print(f"  {label}:")
        for k, agg in report[key].items():
            axe = f" axe={_fmt(agg.get('axe_pass'))}" if "axe_pass" in agg else ""
            print(
                f"    {k:14} n={agg['n']:<4} id_sim={_fmt(agg['id_stripped_sim_mean'])} "
                f"grid={_fmt(agg['grid_match'])} wf={_fmt(agg['wellformed'])} "
                f"trunc={_fmt(agg['truncated'])}{axe}"
            )


def _print_comparison(reports: dict) -> None:
    names = list(reports)
    metrics = [
        ("id_stripped_exact", "id-stripped struct exact"),
        ("id_stripped_sim_mean", "id-stripped struct sim"),
        ("grid_match", "grid match"),
        ("wellformed", "XML well-formed"),
        ("truncated", "truncated at cap"),
        ("axe_pass", "axe wcag22aa pass"),
        ("axe_regression", "axe regression vs gold"),
    ]
    print("\n" + "=" * 70)
    print("  COMPARISON  (" + "  vs  ".join(names) + ")")
    print("=" * 70)
    col = "  {:<26}" + "{:>12}" * len(names) + "{:>10}"
    print(col.format("metric", *names, "Δ(last-1st)"))
    for key, label in metrics:
        vals = [reports[n]["overall"].get(key) for n in names]
        if any(v is None for v in vals):
            continue
        delta = vals[-1] - vals[0]
        print(col.format(label, *[f"{v:.3f}" for v in vals], f"{delta:+.3f}"))
    print("\n  (Δ>0 means the LAST checkpoint is higher; for 'truncated' and")
    print("   'axe regression' lower is better.)")


if __name__ == "__main__":
    sys.exit(main())
