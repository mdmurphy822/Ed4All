#!/usr/bin/env python3
"""Extraction-level A/B for OCR-input-quality changes (scan-conversion Phase 1).

Compares two extraction "sides" against a gold ``chunk_v4`` chunkset for one
chapter and prints 5-gram gold recall + word coverage + the deltas + the plan's
``+3 pts or stop`` gate — the cheap arbiter that must pass before a full cascade
re-convert is worth the ~2-3× slower 300-DPI OCR spend
(``plans/scan-conversion-improvements-2026-07.md`` Phase 1 A/B protocol).

Each side is EITHER:
  * a scan PDF (extracted cache-aware via ``extract_shared_cached``; SemantiK
    deps required, imported lazily), OR
  * a cached-extraction ``.json`` (the ``extract_shared`` shape, loaded
    directly — a ``.json``-vs-``.json`` run needs NO SemantiK deps).

The shingle math is REUSED from ``scripts/gold_compare.py`` by import (never
copied): ``gc.tokenize`` / ``gc.ngram_set`` / ``gc.SHINGLE_N`` / ``gc.load_gold``
/ ``gc.metric_recall`` — so the A/B and the harness measure recall identically.

Cache-key countermeasure: the extract disk cache key IGNORES
``SEMANTIK_TESSERACT_CONFIG`` / ``_USER_WORDS`` (matching the render-scale
precedent), so an A/B on the SAME pdf with different Tesseract configs would
otherwise serve side A's cache to side B. ``side_cache_dir`` puts each distinct
config in a PHYSICALLY separate cache dir so that can never happen.

Process-global caveat: ``run_side`` mutates ``os.environ`` per PDF side and
restores it in ``finally`` — single-threaded CLI use only; never import-and-call
``run_side`` from concurrent code.

Example
-------
    .venv/bin/python scripts/ocr_recall_ab.py \
        --side-a extract_200.json --side-b dart_in-300/ch02.pdf \
        --gold gold_chunks.jsonl --chapter 2 \
        --tess-config-b "--oem 1 -c preserve_interword_spaces=1" \
        --user-words-b SemantiK/data/config/algebra_user_words.txt \
        --json-out ab_ch02.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# sys.path: repo root + scripts/ + SemantiK/ so ``gold_compare`` and (lazily)
# ``dart_semantic.extract_shared`` resolve when run directly (mirrors
# semantik_rerender.py + test_gold_compare.py precedents).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SEMANTIK_DIR = _REPO_ROOT / "SemantiK"
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR), str(_SEMANTIK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gold_compare as gc  # noqa: E402

SCHEMA_VERSION = "ocr_recall_ab/1.0"
GATE_THRESHOLD = 0.03  # the plan's "+3 pts or stop"

_ENV_CONFIG = "SEMANTIK_TESSERACT_CONFIG"
_ENV_USER_WORDS = "SEMANTIK_TESSERACT_USER_WORDS"


def extraction_text(shared: dict) -> str:
    """Join per-page merged text-block text across an ``extract_shared`` doc.

    Falls back to ``tesseract`` / ``pypdfium2`` raw blocks on a page missing a
    ``merged`` view (a raw partial extraction).
    """
    parts: list[str] = []
    for page in shared.get("pages", []) or []:
        blocks = None
        merged = page.get("merged")
        if isinstance(merged, dict) and merged.get("text_blocks"):
            blocks = merged["text_blocks"]
        else:
            for fallback in ("tesseract", "pypdfium2"):
                fb = page.get(fallback)
                if isinstance(fb, dict) and fb.get("text_blocks"):
                    blocks = fb["text_blocks"]
                    break
        for b in blocks or []:
            t = b.get("text")
            if t:
                parts.append(t)
    return "\n".join(parts)


def side_cache_dir(
    cache_root: Path | None, tess_config: str, user_words: str | None
) -> Path | None:
    """Per-config cache dir so distinct Tesseract configs never share a cache.

    Returns ``None`` (the default cache) when config + user_words are both
    empty AND no ``cache_root`` is pinned; otherwise
    ``<cache_root>/ab_<sha256(config|user_words)[:12]>``. Deterministic for
    equal configs.
    """
    import hashlib

    cfg = (tess_config or "").strip()
    uw = (user_words or "").strip()
    if not cfg and not uw and cache_root is None:
        return None
    root = cache_root if cache_root is not None else (_SCRIPTS_DIR.parent / "state" / "ocr_ab_cache")
    digest = hashlib.sha256(f"{cfg}|{uw}".encode("utf-8")).hexdigest()[:12]
    return root / f"ab_{digest}"


def run_side(
    path: Path,
    tess_config: str | None,
    user_words: str | None,
    cache_root: Path | None,
) -> tuple[str, dict]:
    """Return ``(text, meta)`` for one side.

    ``.json`` → loaded directly (no env mutation, no SemantiK deps). ``.pdf`` →
    ``extract_shared_cached`` under a per-config cache, with the Tesseract-config
    env vars set around the call (the call-time resolvers make this correct) and
    RESTORED in ``finally`` even on failure.
    """
    kind = "json" if path.suffix.lower() == ".json" else "pdf"
    if kind == "json":
        shared = json.loads(path.read_text(encoding="utf-8"))
        text = extraction_text(shared)
        meta = {
            "source": str(path),
            "kind": "json",
            "cache_dir": None,
            "tess_config": tess_config or "",
            "user_words": user_words,
            "pages": len(shared.get("pages", []) or []),
            "chars": len(text),
        }
        return text, meta

    cache_dir = side_cache_dir(cache_root, tess_config or "", user_words)
    prior_config = os.environ.get(_ENV_CONFIG)
    prior_uw = os.environ.get(_ENV_USER_WORDS)
    try:
        if tess_config:
            os.environ[_ENV_CONFIG] = tess_config
        else:
            os.environ.pop(_ENV_CONFIG, None)
        if user_words:
            os.environ[_ENV_USER_WORDS] = user_words
        else:
            os.environ.pop(_ENV_USER_WORDS, None)

        from dart_semantic.extract_shared import extract_shared_cached  # noqa: E402

        shared = extract_shared_cached(path, cache_dir=cache_dir)
    finally:
        _restore_env(_ENV_CONFIG, prior_config)
        _restore_env(_ENV_USER_WORDS, prior_uw)

    text = extraction_text(shared)
    meta = {
        "source": str(path),
        "kind": "pdf",
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "tess_config": tess_config or "",
        "user_words": user_words,
        "pages": len(shared.get("pages", []) or []),
        "chars": len(text),
    }
    return text, meta


def _restore_env(name: str, prior: str | None) -> None:
    if prior is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = prior


def score_side(gch: "gc.GoldChapter", text: str) -> dict:
    """Recall (via ``gc.metric_recall``) + word coverage against a gold chapter."""
    rec = gc.metric_recall(gch, text)
    cand_vocab = set(gc.tokenize(text))
    coverage = (
        len(gch.vocab & cand_vocab) / len(gch.vocab) if gch.vocab else 0.0
    )
    return {
        "recall_mean": rec["recall_mean"],
        "recall_p10": rec["recall_p10"],
        "recall_p50": rec["recall_p50"],
        "fraction_below_0.5": rec["fraction_below_0.5"],
        "word_coverage": round(coverage, 4),
        "gold_chunks": rec["gold_chunks"],
        "worst_10": rec["worst_10"],
    }


def compare(
    gold_path: Path,
    chapter: int,
    side_a: tuple[str, dict],
    side_b: tuple[str, dict],
) -> dict:
    """Score both sides for ``chapter`` and build the delta + gate report."""
    gold = gc.load_gold(gold_path)
    gch = gold.get(chapter)
    if gch is None:
        available = sorted(c for c in gold if c is not None)
        raise SystemExit(
            f"chapter {chapter} not in gold {gold_path} (available: {available})"
        )

    text_a, meta_a = side_a
    text_b, meta_b = side_b
    score_a = score_side(gch, text_a)
    score_b = score_side(gch, text_b)

    recall_delta = round(score_b["recall_mean"] - score_a["recall_mean"], 4)
    report = {
        "schema_version": SCHEMA_VERSION,
        "chapter": chapter,
        "gold_path": str(gold_path),
        "sides": {
            "a": {"meta": meta_a, "score": score_a},
            "b": {"meta": meta_b, "score": score_b},
        },
        "deltas": {
            "recall_mean_delta": recall_delta,
            "recall_p50_delta": round(
                score_b["recall_p50"] - score_a["recall_p50"], 4
            ),
            "word_coverage_delta": round(
                score_b["word_coverage"] - score_a["word_coverage"], 4
            ),
        },
        "gate": {
            "threshold": GATE_THRESHOLD,
            "passed": recall_delta >= GATE_THRESHOLD,
        },
    }
    return report


def _print_report(report: dict) -> None:
    a = report["sides"]["a"]
    b = report["sides"]["b"]
    d = report["deltas"]
    g = report["gate"]
    print("=" * 72)
    print(f"OCR RECALL A/B  chapter={report['chapter']}")
    print("-" * 72)
    print(
        f"  A [{a['meta']['label']}]  recall_mean={a['score']['recall_mean']:.4f}"
        f"  p50={a['score']['recall_p50']:.4f}"
        f"  word_cov={a['score']['word_coverage']:.4f}  ({a['meta']['source']})"
    )
    print(
        f"  B [{b['meta']['label']}]  recall_mean={b['score']['recall_mean']:.4f}"
        f"  p50={b['score']['recall_p50']:.4f}"
        f"  word_cov={b['score']['word_coverage']:.4f}  ({b['meta']['source']})"
    )
    print("-" * 72)
    print(
        f"  delta recall_mean={d['recall_mean_delta']:+.4f}"
        f"  recall_p50={d['recall_p50_delta']:+.4f}"
        f"  word_cov={d['word_coverage_delta']:+.4f}"
    )
    verdict = "PASS" if g["passed"] else "STOP"
    print(
        f"  GATE (+{g['threshold']:.2f} or stop): {verdict}  "
        f"(recall delta {d['recall_mean_delta']:+.4f})"
    )
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extraction-level OCR recall A/B vs a gold chunkset."
    )
    ap.add_argument("--side-a", required=True, type=Path, help="PDF or extraction .json")
    ap.add_argument("--side-b", required=True, type=Path, help="PDF or extraction .json")
    ap.add_argument("--gold", required=True, type=Path, help="gold chunk_v4 chunks.jsonl")
    ap.add_argument("--chapter", required=True, type=int, help="chapter to score")
    ap.add_argument("--label-a", default="A", help="label for side A")
    ap.add_argument("--label-b", default="B", help="label for side B")
    ap.add_argument("--tess-config-a", default=None, help="SEMANTIK_TESSERACT_CONFIG for side A (PDF only)")
    ap.add_argument("--tess-config-b", default=None, help="SEMANTIK_TESSERACT_CONFIG for side B (PDF only)")
    ap.add_argument("--user-words-a", default=None, help="SEMANTIK_TESSERACT_USER_WORDS for side A (PDF only)")
    ap.add_argument("--user-words-b", default=None, help="SEMANTIK_TESSERACT_USER_WORDS for side B (PDF only)")
    ap.add_argument("--cache-root", type=Path, default=None, help="per-config extract cache root")
    ap.add_argument("--json-out", type=Path, default=None, help="write the full JSON report here")
    args = ap.parse_args(list(argv) if argv is not None else None)

    side_a = run_side(args.side_a, args.tess_config_a, args.user_words_a, args.cache_root)
    side_b = run_side(args.side_b, args.tess_config_b, args.user_words_b, args.cache_root)
    side_a[1]["label"] = args.label_a
    side_b[1]["label"] = args.label_b

    report = compare(args.gold, args.chapter, side_a, side_b)
    _print_report(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report -> {args.json_out}")

    return 0 if report["gate"]["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
