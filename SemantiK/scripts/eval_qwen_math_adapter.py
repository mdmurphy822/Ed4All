"""Generative eval for the trained Qwen3-4B LoRA *math* adapter.

This is NOT ``eval_math_specialist.py`` — that one scores the retired
BERT-MathSpecialist classifier (math_type / eq_num_assoc heads). This
script evaluates the GENERATIVE Qwen LoRA adapter trained via
``scripts/train_qwen_lora.py --adapter math`` by decoding completions
on the held-out test split and scoring them against ``target_html``.

Two backends, identical sample + identical metrics so the numbers are
directly comparable:

  * ``--backend safetensors``  — load base Qwen3-4B in 4-bit + the HF
    LoRA via peft (exactly as training loaded it). Measures the trained
    weights directly, decoupled from GGUF quantization.

  * ``--backend gguf --gguf <path>`` — load the converted/quantized
    GGUF through the production ``LlamaCppRuntime``. Measures what the
    pipeline actually ships post-quantization.

Both backends apply ``wrap_for_qwen`` (the single source of truth for
the chat-template wrap; see chat_format.py) so the generated first
token matches training. Greedy decode in both (temperature 0 / n=1).

Sampling
--------
The test split is ~99% inline math. A proportional 300-row sample would
contain ~0 rare-type rows, hiding exactly the cases the adapter is most
likely to botch. So we OVERSAMPLE rare types: take every display /
numbered / multiline / matrix row, then fill the remainder with inline,
seeded for determinism. The headline match rate is therefore NOT
population-weighted — always read the per-class table alongside it.

Usage::

    .venv/bin/python -m scripts.eval_qwen_math_adapter --backend safetensors
    .venv/bin/python -m scripts.eval_qwen_math_adapter --backend gguf \\
        --gguf models/qwen_specialists/math/v1/math.q4_k_m.gguf
"""

from __future__ import annotations

import argparse
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

DEFAULT_ADAPTER_DIR = REPO_ROOT / "models/qwen_specialists/math/v1/final"
DEFAULT_SUMMARY = REPO_ROOT / "models/qwen_specialists/math/v1/summary.json"
DEFAULT_TEST = REPO_ROOT / "data/qwen_math_dataset/test.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "data/eval_reports"

# Rare math_types we guarantee in the sample (everything that isn't the
# dominant inline class).
_RARE_TYPES = ("display", "numbered", "multiline", "matrix")

_ALTTEXT_RE = re.compile(r'alttext="(.*?)"', re.DOTALL)
_WS_RE = re.compile(r"\s+")
# id / xref attribute VALUES encode document position (section / equation
# number), which is NOT in the prompt — the adapter cannot reproduce them,
# so exact string match is meaningless. Strip the values to score the
# STRUCTURE the adapter actually controls.
_ID_RE = re.compile(r'\s(?:id|xref)="[^"]*"')


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
    """Deterministic rare-type-oversampled sample of size ~n.

    Takes every rare-type row, then fills the remainder with inline
    rows. Same (rows, n, seed) => byte-identical selection, so both
    backends score the exact same items.
    """
    import random

    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r.get("math_type", "inline")].append(r)

    picked: list[dict] = []
    for t in _RARE_TYPES:
        picked.extend(by_type.get(t, []))

    inline = list(by_type.get("inline", []))
    rng.shuffle(inline)
    remaining = max(0, n - len(picked))
    picked.extend(inline[:remaining])

    # Stable, reproducible ordering keyed on a content id (not list
    # order) so the sample is invariant to upstream file reordering.
    picked.sort(key=lambda r: (r.get("arxiv_id") or "", r.get("math_idx", 0)))
    return picked


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _fragment(text: str) -> str:
    """Trim model output to the <math>...</math> fragment if delimited."""
    text = text.strip()
    # strip a trailing chat terminator if the decoder left one in
    text = text.split("<|im_end|>")[0].strip()
    start = text.find("<math")
    end = text.rfind("</math>")
    if start != -1 and end != -1 and end > start:
        return text[start : end + len("</math>")]
    return text


def _norm(s: str) -> str:
    """Whitespace-collapse for a lenient structural comparison."""
    s = re.sub(r">\s+<", "><", s)
    return _WS_RE.sub(" ", s).strip()


def _strip_ids(s: str) -> str:
    """Drop id/xref attribute values, then whitespace-normalize. Yields a
    comparison key that ignores document-position ids the prompt can't
    supply, isolating the structure + content the adapter controls."""
    return _norm(_ID_RE.sub("", s))


def _alttext(s: str) -> str | None:
    m = _ALTTEXT_RE.search(s)
    return m.group(1) if m else None


def _wellformed(s: str) -> bool:
    try:
        ET.fromstring(s)
        return True
    except Exception:
        return False


def score_one(gen_raw: str, gold: str) -> dict:
    gen = _fragment(gen_raw)
    gold_s = gold.strip()
    g_alt, t_alt = _alttext(gen), _alttext(gold_s)
    # A complete fragment ends with </math>; otherwise the decode hit the
    # token cap mid-tag (the main cause of ill-formed XML on long-id rows).
    truncated = "</math>" not in gen_raw
    gen_ns, gold_ns = _strip_ids(gen), _strip_ids(gold_s)
    return {
        "exact": gen == gold_s,
        "norm": _norm(gen) == _norm(gold_s),
        # structural fidelity with unknowable id/xref values neutralized —
        # the primary quality signal for this adapter.
        "id_stripped_exact": gen_ns == gold_ns,
        "id_stripped_sim": SequenceMatcher(None, gen_ns, gold_ns).ratio(),
        "alttext_present": g_alt is not None,
        "alttext_match": (g_alt is not None and g_alt == t_alt),
        "wellformed": _wellformed(gen),
        "truncated": truncated,
        "char_sim": SequenceMatcher(None, gen, gold_s).ratio(),
        "gen_len": len(gen),
    }


def aggregate(per_row: list[dict]) -> dict:
    n = len(per_row)
    if n == 0:
        return {}
    keys = (
        "exact",
        "norm",
        "id_stripped_exact",
        "alttext_present",
        "alttext_match",
        "wellformed",
        "truncated",
    )
    agg = {k: sum(1 for r in per_row if r["scores"][k]) / n for k in keys}
    agg["id_stripped_sim_mean"] = sum(r["scores"]["id_stripped_sim"] for r in per_row) / n
    agg["char_sim_mean"] = sum(r["scores"]["char_sim"] for r in per_row) / n
    agg["n"] = n
    return agg


def per_class(per_row: list[dict]) -> dict:
    out: dict[str, dict] = {}
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_t[r["math_type"]].append(r)
    for t, rows in sorted(by_t.items()):
        out[t] = aggregate(rows)
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def gen_safetensors(
    sample: list[dict],
    adapter_dir: Path,
    base_model: str,
    max_new_tokens: int,
    *,
    repeat_penalty: float = 1.0,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from dart_semantic.qwen_specialists.chat_format import wrap_for_qwen

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
        ids = tok(wrapped, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=repeat_penalty,
            )
        new = gen[0][ids["input_ids"].shape[1] :]
        outs.append(tok.decode(new, skip_special_tokens=True))
        if (i + 1) % 25 == 0:
            print(f"[safetensors] {i + 1}/{len(sample)}", flush=True)
    return outs


def gen_gguf(
    sample: list[dict],
    gguf_path: Path,
    max_new_tokens: int,
    *,
    server_url: str | None = None,
    adapter_dir: Path | None = None,
    repeat_penalty: float = 1.0,
) -> list[str]:
    """Generate via GGUF.

    Two transports, same llama.cpp CUDA engine:

    * ``server_url`` set — POST each prompt to an already-running
      ``llama-server`` ``/completion`` endpoint. Loads the model once
      (server-side), GPU-accelerated, no per-call reload. Used when the
      ``llama-cpp-python`` binding isn't installed (the production
      ``LlamaCppRuntime`` needs it; the CLI/server binaries don't).
    * otherwise — use the in-process ``LlamaCppRuntime`` (the production
      path; requires ``llama-cpp-python``).

    Both apply ``wrap_for_qwen`` so the chat-template wrap matches
    training, and decode greedily (temperature 0, n=1).
    """
    if server_url:
        import requests

        from transformers import AutoTokenizer

        from dart_semantic.qwen_specialists.chat_format import wrap_for_qwen

        tok = AutoTokenizer.from_pretrained(
            adapter_dir or DEFAULT_ADAPTER_DIR, trust_remote_code=True
        )
        url = server_url.rstrip("/") + "/completion"
        print(f"[gguf] server {url}")
        outs: list[str] = []
        for i, row in enumerate(sample):
            prompt = row["request_payload"]["payload"]["prompt"]
            wrapped = wrap_for_qwen(tok, prompt, add_generation_prompt=True)
            r = requests.post(
                url,
                json={
                    "prompt": wrapped,
                    "n_predict": max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 0,
                    "cache_prompt": False,
                    "repeat_penalty": repeat_penalty,
                },
                timeout=600,
            )
            r.raise_for_status()
            outs.append(r.json()["content"])
            if (i + 1) % 25 == 0:
                print(f"[gguf] {i + 1}/{len(sample)}", flush=True)
        return outs

    from dart_semantic.qwen_specialists.runtime import LlamaCppRuntime

    rt = LlamaCppRuntime()
    print(f"[gguf] loading {gguf_path}")
    rt.load(gguf_path)
    outs = []
    for i, row in enumerate(sample):
        prompt = row["request_payload"]["payload"]["prompt"]
        # n=1, temperature=0 => greedy, matching the safetensors path.
        res = rt.generate(
            prompt,
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            seed=0,
            repeat_penalty=repeat_penalty,
        )
        outs.append(res[0])
        if (i + 1) % 25 == 0:
            print(f"[gguf] {i + 1}/{len(sample)}", flush=True)
    return outs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["safetensors", "gguf"], required=True)
    ap.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--gguf", type=Path, help="GGUF path (required for gguf backend)")
    ap.add_argument(
        "--server-url",
        default=None,
        help="If set, gguf backend POSTs to this llama-server /completion "
        "endpoint instead of the in-process binding (e.g. http://127.0.0.1:8080)",
    )
    ap.add_argument("--n", type=int, default=300, help="sample size")
    ap.add_argument("--seed", type=int, default=0)
    # Real target_html runs to ~919 tokens (id/xref-heavy MathML); a cap
    # below ~960 truncates the long tail and unfairly fails those rows.
    ap.add_argument("--max-new-tokens", type=int, default=960)
    # repeat_penalty (HF repetition_penalty). Default 1.0 == no penalty,
    # matching the runtime/config default so the headline numbers are
    # unchanged until an A/B arm sets it. 1.05 is the designated arm for
    # long-target overshoot — applies to BOTH backends (gguf repeat_penalty,
    # safetensors repetition_penalty).
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--out-path", type=Path, default=None)
    args = ap.parse_args()

    if args.backend == "gguf" and not args.gguf:
        return ap.error("--gguf is required for --backend gguf")
    if not args.test_path.exists():
        sys.exit(f"[fatal] test split not found: {args.test_path}")

    base_model = json.loads(args.summary.read_text())["base_model"]
    rows = _load_jsonl(args.test_path)
    sample = select_sample(rows, args.n, args.seed)
    sample_dist = Counter(r.get("math_type", "inline") for r in sample)
    print(f"[sample] {len(sample)} rows (seed={args.seed}) by math_type: {dict(sample_dist)}")

    t0 = time.time()
    if args.backend == "safetensors":
        completions = gen_safetensors(
            sample,
            args.adapter_dir,
            base_model,
            args.max_new_tokens,
            repeat_penalty=args.repeat_penalty,
        )
    else:
        completions = gen_gguf(
            sample,
            args.gguf,
            args.max_new_tokens,
            server_url=args.server_url,
            adapter_dir=args.adapter_dir,
            repeat_penalty=args.repeat_penalty,
        )
    wall = time.time() - t0

    per_row = []
    for row, comp in zip(sample, completions):
        per_row.append(
            {
                "arxiv_id": row.get("arxiv_id"),
                "math_idx": row.get("math_idx"),
                "math_type": row.get("math_type", "inline"),
                "scores": score_one(comp, row["target_html"]),
            }
        )

    report = {
        "backend": args.backend,
        "base_model": base_model,
        "adapter_dir": str(args.adapter_dir),
        "gguf": str(args.gguf) if args.gguf else None,
        "test_path": str(args.test_path),
        "n_sampled": len(sample),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "repeat_penalty": args.repeat_penalty,
        "wall_seconds": round(wall, 1),
        "sample_dist": dict(sample_dist),
        "overall": aggregate(per_row),
        "per_class": per_class(per_row),
    }

    out_path = args.out_path or (DEFAULT_REPORT_DIR / f"qwen_math_adapter_{args.backend}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # keep the raw completions next to the report for spot-checking
    samples_path = out_path.with_suffix(".samples.jsonl")
    with samples_path.open("w", encoding="utf-8") as f:
        for row, comp in zip(sample, completions):
            f.write(
                json.dumps(
                    {
                        "arxiv_id": row.get("arxiv_id"),
                        "math_idx": row.get("math_idx"),
                        "math_type": row.get("math_type", "inline"),
                        "gold": row["target_html"],
                        "gen": comp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    o = report["overall"]
    print("\n" + "=" * 64)
    print(f"  Qwen math adapter — {args.backend}  (n={o['n']}, {wall:.0f}s)")
    print("=" * 64)
    print(f"  exact match           : {o['exact']:.3f}  (id-bearing; ~floor by design)")
    print(f"  id-stripped struct exact: {o['id_stripped_exact']:.3f}  <- primary")
    print(f"  id-stripped struct sim  : {o['id_stripped_sim_mean']:.3f}")
    print(f"  alttext (TeX) match     : {o['alttext_match']:.3f}")
    print(f"  alttext present         : {o['alttext_present']:.3f}")
    print(f"  XML well-formed         : {o['wellformed']:.3f}")
    print(f"  truncated at cap        : {o['truncated']:.3f}")
    print(f"  char similarity         : {o['char_sim_mean']:.3f}")
    print("  per-class (math_type):")
    for t, agg in report["per_class"].items():
        print(
            f"    {t:10} n={agg['n']:<4} id_exact={agg['id_stripped_exact']:.2f} "
            f"id_sim={agg['id_stripped_sim_mean']:.2f} "
            f"alttext={agg['alttext_match']:.2f} "
            f"wf={agg['wellformed']:.2f} trunc={agg['truncated']:.2f}"
        )
    print(f"\n  report : {out_path}")
    print(f"  samples: {samples_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
