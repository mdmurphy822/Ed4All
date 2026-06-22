"""Construct Structure-head training labels for legal opinions from DART's OWN
extraction (Plan 14).

The courtlistener `output_html` ground truth is unusable: it is flat AND glues
section headings into the following body paragraph ("II. STANDARD OF REVIEW In
analyzing…"). DART's `extract_shared` DOES separate the heading line from the
body line into distinct blocks, so we pseudo-label those correctly-separated
blocks with deterministic legal-structure rules, mirroring
`build_structure_data.py`'s span extraction + layout vector so the output JSONL
is byte-compatible.

Two modes:
  --audit N   : pseudo-label N opinions and PRINT every heading/list/blockquote
                decision + a sample of paragraphs, for manual review. Writes
                nothing. Use this to iterate the rules until clean.
  --build     : write per-pair JSONL to data/structure_dataset/per_pair_legal/
                (later merged by build_structure_data.py).

The rules are AUDITED, not trusted blindly — see Plan 14 §3.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from data.build_structure_data import (
    ROLE_TO_ID,
    _block_in_any_table,
    compute_span_layout_features,
)
from dart_semantic.extract_shared import extract_shared_cached

# --- legal-heading text signals (run on DART-SEPARATED blocks) ---------------
_ROMAN = re.compile(r"^[IVXLC]{1,4}\.\s")
_LETTER = re.compile(r"^[A-Z]\.\s")
_NUM = re.compile(r"^\d{1,2}\.\s")
_BANNER = re.compile(
    r"^(DISCUSSION|BACKGROUND|CONCLUSION|ANALYSIS|OPINION|ORDER|FACTS?|"
    r"PROCEDURAL HISTORY|INTRODUCTION|STANDARD OF REVIEW|JURISDICTION|"
    r"STATEMENT OF (THE )?(CASE|FACTS)|ARGUMENT)\b",
    re.IGNORECASE,
)
_SENT_END = re.compile(r"[.!?][\"')\]]?$")
_BULLET = re.compile(r"^\s*[•‣◦▪·]\s")


def _words(t: str) -> int:
    return len(t.split())


def _ends_sentence_loose(t: str) -> bool:
    """True if the block ends a sentence (terminal . ! ? after stripping any
    trailing quotes/brackets, straight OR curly). Real section headings never
    do; sentence-y enumerated lines ("1. … probative.", "opinion.") do."""
    return t.rstrip(" \"'”’)]").endswith((".", "!", "?"))


def _is_heading(text: str) -> bool:
    """A legal section/subsection heading on a CORRECTLY-SEPARATED block:
    a short, set-off line matching a roman/lettered/numbered/allcaps signal.
    Long lines (body that merely starts with a number) are rejected by the
    word cap; a heading rarely ends mid-clause-then-runs-on because DART
    separates the heading line from the body line."""
    t = text.strip()
    if not t or _words(t) > 12:
        return False
    # A real section heading never ends a sentence; this rejects body
    # fragments ("opinion.") and sentence-y enumerated lines ("1. … probative.").
    if _ends_sentence_loose(t):
        return False
    # Banner (DISCUSSION / STANDARD OF REVIEW …). The regex is case-insensitive,
    # so common body-sentence openers ("Order that, as relevant here, affirmed…",
    # "argument is that…", "conclusion forecloses…") would match — gate on a
    # tight word cap: real banners are 1-5 words, those body FPs are 9-12.
    if _BANNER.match(t) and _words(t) <= 5:
        return True
    # Roman / lettered / numbered prefix → heading only if the REMAINDER is
    # heading-shaped: short and not a full sentence with internal punctuation.
    if _ROMAN.match(t) or _LETTER.match(t) or _NUM.match(t):
        # strip the enumerator, inspect the title
        title = re.sub(r"^([IVXLC]{1,4}|[A-Z]|\d{1,2})\.\s+", "", t)
        # reject if the title is a running sentence (internal ". " + lowercase
        # continuation, or ends with a sentence and keeps going)
        if re.search(r"[.!?]\s+[a-z]", title):
            return False
        return _words(t) <= 10
    return False


def _label_block(text: str) -> tuple[str, int]:
    """Return (structural_role, is_heading)."""
    t = (text or "").strip()
    if _is_heading(t):
        return "heading", 1
    if _BULLET.match(t):  # an ACTUAL bullet marker — rare in opinions
        return "list_item", 0
    # everything else (body, citations "65 n.3 …", inline enumerations, narrative
    # quotes) → paragraph. These are the hard negatives the head needs.
    return "paragraph", 0


def _page_examples(page: dict, stem: str) -> list[dict]:
    merged = page.get("merged", {}).get("text_blocks", []) or []
    if not merged:
        return []
    page_w = float(page.get("width") or 612.0)
    page_h = float(page.get("height") or 792.0)
    sizes = [b["font_size"] for b in merged if b.get("font_size") is not None]
    page_median_fs = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
    heights = [
        (b["bbox"][3] - b["bbox"][1]) for b in merged if b.get("bbox") and len(b["bbox"]) == 4
    ]
    page_median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
    merged_sorted = sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    out: list[dict] = []
    for block in merged_sorted:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        in_table = _block_in_any_table(block, page)
        role, is_heading = _label_block(text)
        layout = compute_span_layout_features(
            block,
            page_w=page_w,
            page_h=page_h,
            page_median_fs=page_median_fs,
            page_median_h=page_median_h,
            in_table=in_table,
        )
        out.append(
            {
                "text": text,
                "layout": layout,
                "labels": {
                    "structural_role": ROLE_TO_ID[role],
                    "is_heading": is_heading,
                    "table_region": 1 if in_table else 0,
                    "is_image_block": 0,
                    "list_nesting": 0,
                },
                "html_tag": {"heading": "h2", "list_item": "li"}.get(role, "p"),
                "source": "courtlistener_pseudo",
                "pair": stem,
            }
        )
    return out


def _pdf_examples(pdf_path: str) -> list[dict]:
    shared = extract_shared_cached(Path(pdf_path))
    stem = Path(pdf_path).stem
    out: list[dict] = []
    for page in shared.get("pages", []):
        out.extend(_page_examples(page, stem))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-glob", default="data/cache/courtlistener/*.pdf")
    ap.add_argument("--audit", type=int, default=0, help="audit N docs, write nothing")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out-dir", default="data/structure_dataset/per_pair_legal")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--para-cap",
        type=int,
        default=6000,
        help="max legal PARAGRAPH examples kept (headings/lists always kept)",
    )
    # Eval opinions measured in the loop — HELD OUT of training so the
    # real-runtime FRAGMENT eval (Plan 14 §6) stays honest.
    ap.add_argument(
        "--holdout",
        default=(
            "opinion_11297311,opinion_11298455,opinion_11320152,opinion_11313568,"
            "opinion_11315258,opinion_11314119,opinion_11297479,opinion_11311457,"
            "opinion_11314738"
        ),
    )
    args = ap.parse_args()

    holdout = {s.strip() for s in args.holdout.split(",") if s.strip()}
    pdfs = [p for p in sorted(glob.glob(args.pdf_glob)) if Path(p).stem not in holdout]
    if args.limit:
        pdfs = pdfs[: args.limit]

    if args.audit:
        for pdf in pdfs[: args.audit]:
            exs = _pdf_examples(pdf)
            heads = [e for e in exs if e["labels"]["is_heading"]]
            lists = [e for e in exs if e["labels"]["structural_role"] == ROLE_TO_ID["list_item"]]
            print(
                f"\n===== {Path(pdf).stem}: {len(exs)} blocks, "
                f"{len(heads)} heading, {len(lists)} list_item ====="
            )
            print("  -- HEADINGS (must all be REAL section headings) --")
            for e in heads:
                print(f"     H: {e['text'][:75]!r}")
            print("  -- LIST_ITEMS --")
            for e in lists:
                print(f"     L: {e['text'][:75]!r}")
            # sample paragraphs that LOOK heading-ish (start uppercase, short) to
            # check we are not MISSING real headings
            susp = [
                e
                for e in exs
                if e["labels"]["structural_role"] == ROLE_TO_ID["paragraph"]
                and _words(e["text"]) <= 8
                and e["text"][:1].isupper()
            ]
            print(f"  -- SHORT UPPERCASE PARAGRAPHS ({len(susp)}; check for MISSED headings) --")
            for e in susp[:12]:
                print(f"     p: {e['text'][:75]!r}")
        return

    if args.build:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Collect ALL examples, then balance globally: keep every heading /
        # list_item (rare + clearly-correct — the head must learn legal heading
        # patterns AND not suppress them), and seed-subsample the abundant
        # paragraph class to --para-cap (the FP-negative budget). Legal is ~99%
        # paragraph; uncapped it would swamp the 75k corpus (see
        # feedback_balance_dominant_category).
        import random
        from collections import Counter

        para_id = ROLE_TO_ID["paragraph"]
        keep: list[dict] = []
        paras: list[dict] = []
        for pdf in pdfs:
            for e in _pdf_examples(pdf):
                if e["labels"]["structural_role"] == para_id and e["labels"]["is_heading"] == 0:
                    paras.append(e)
                else:
                    keep.append(e)
        rng = random.Random(42)
        rng.shuffle(paras)
        kept_paras = paras[: args.para_cap]
        balanced = keep + kept_paras
        rng.shuffle(balanced)
        out_file = out_dir / "courtlistener_pseudo.jsonl"
        out_file.write_text("\n".join(json.dumps(e) for e in balanced))
        roles = Counter(e["labels"]["structural_role"] for e in balanced)
        names = ["paragraph", "heading", "list_item", "form_label", "blockquote", "code_block"]
        print(
            f"wrote {len(balanced)} balanced legal examples ({len(keep)} heading/list/other "
            f"+ {len(kept_paras)}/{len(paras)} paragraphs) from {len(pdfs)} opinions to {out_file}"
        )
        print(
            "  role mix:",
            {names[k] if k < len(names) else k: v for k, v in sorted(roles.items())},
        )


if __name__ == "__main__":
    main()
