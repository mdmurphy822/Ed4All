"""Generate training pairs from local Arxiiv Repo PDFs + ar5iv HTML.

Flow:
    walk local PDFs  ->  extract arxiv_id from filename
        -> fetch ar5iv HTML for that id (skip if ar5iv 404s)
        -> parse_ar5iv -> ir.Document
        -> emit_html(doc) -> axe gate
             (drop on fail)
        -> pdf_to_ocr_text(LOCAL PDF)  [real arXiv typography]
        -> emit_html(doc) -> WCAG-conformant HTML (ground truth for classifier labels)
        -> write pair

We do NOT render the ar5iv HTML to PDF for feature extraction — we use the
original arXiv PDF so the model trains on real academic typography (two-column
layouts, inline math, caption styling). The HTML gives us the ground-truth
structural labels only.

Usage:
    python scripts/pair_from_arxiv.py --limit 5 --out-dir data/pairs/arxiv
    python scripts/pair_from_arxiv.py --workers 4 --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dart_semantic import emit_html, ir  # legacy IR + emitter used for ground-truth HTML
from dart_semantic.arxiv_license import extract_arxiv_id
from dart_semantic.arxiv_sections import split_paper_into_sections
from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.parse_ar5iv import Ar5ivParseError, fetch_ar5iv_html, parse_paper
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool

# arXiv paper repo location is machine-specific; supply it via the
# SEMANTIK_ARXIV_REPO env var or the --repo flag (no hardcoded absolute path in
# tracked code). Falls back to ``~/arxiv-repo/papers``.
DEFAULT_REPO = Path(
    os.environ.get("SEMANTIK_ARXIV_REPO", str(Path.home() / "arxiv-repo" / "papers"))
).expanduser()

ARXIV_CSS = """
body { font-family: Georgia, serif; max-width: 7in; margin: 1in auto; color: #111; line-height: 1.5; }
h1 { margin: 0 0 0.6em; font-size: 1.6em; }
h2 { margin-top: 1.4em; font-size: 1.3em; }
h3 { margin-top: 1.1em; font-size: 1.1em; }
h4, h5, h6 { margin-top: 1em; }
p { text-align: justify; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #444; padding: 0.3em 0.6em; text-align: left; }
th { background: #eee; }
ul, ol { margin: 0.6em 0; padding-left: 1.6em; }
figure { margin: 1em 0; text-align: center; }
figcaption { font-style: italic; font-size: 0.9em; color: #444; }
pre { background: #f7f7f7; padding: 0.8em; white-space: pre-wrap; word-break: break-word; }
code { font-family: "DejaVu Sans Mono", monospace; }
a { padding: 2px 1px; }
math { font-style: italic; }
"""


def wrap_html(raw_html: str) -> str:
    return raw_html.replace("<head>", f"<head><style>{ARXIV_CSS}</style>", 1)


def find_local_pdfs(repo_root: Path) -> list[tuple[str, Path]]:
    """Return [(arxiv_id, pdf_path)] for every PDF with a recognizable id."""
    pairs = []
    for pdf in repo_root.rglob("*.pdf"):
        aid = extract_arxiv_id(pdf.name)
        if aid:
            pairs.append((aid, pdf))
    return pairs


def _slug(s: str, maxlen: int = 40) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in s.lower()).strip("_")
    return cleaned[:maxlen] or "section"


def process_paper(validator: HtmlValidator, work: tuple[str, str, str]) -> dict:
    """Worker: (arxiv_id, pdf_path_str, out_dir_str) -> stats dict.

    One paper yields multiple pairs — one per top-level section whose
    heading we can match to a PDF bookmark. Papers without bookmarks
    fall back to a single whole-paper pair (which build_dataset.py will
    likely drop for being over the token budget).
    """
    arxiv_id, pdf_path_str, out_dir_str = work
    pdf_path = Path(pdf_path_str)
    out_dir = Path(out_dir_str)
    stats = {"arxiv_id": arxiv_id, "ok": 0, "sections": 0,
             "fetch_error": 0, "parse_error": 0,
             "emitter_drop": 0, "axe_drop": 0, "whole_paper_fallback": 0}

    try:
        raw_html = fetch_ar5iv_html(arxiv_id)
    except Ar5ivParseError as exc:
        stats["fetch_error"] = 1
        stats["msg"] = str(exc)
        return stats

    try:
        doc = parse_paper(raw_html, arxiv_id,
                          paper_url=f"https://arxiv.org/abs/{arxiv_id}")
    except (Ar5ivParseError, ir.IRError) as exc:
        stats["parse_error"] = 1
        stats["msg"] = str(exc)
        return stats

    # Split into (section_doc, page_range) pairs. Whole-paper fallback is
    # [(doc, None)] — page_range=None means pdf_to_ocr_text uses the whole PDF.
    section_pairs = split_paper_into_sections(doc, pdf_path)
    if len(section_pairs) == 1 and section_pairs[0][1] is None:
        stats["whole_paper_fallback"] = 1

    for idx, (section_doc, page_range) in enumerate(section_pairs):
        try:
            clean_html = emit_html.emit(section_doc)
        except emit_html.EmitterError:
            stats["emitter_drop"] += 1
            continue
        html_doc = wrap_html(clean_html)

        result = validator.check(html_doc)
        if not result.ok:
            stats["axe_drop"] += 1
            continue

        input_ocr = pdf_to_ocr_text(pdf_path, page_range=page_range)
        if not input_ocr.strip():
            stats["axe_drop"] += 1  # empty page range, treat as quality failure
            continue

        # build_classifier_data.py reads output_html. Phase 3c
        # BERT-Semantic + downstream BERTs read raw_source_html (the
        # ar5iv source) for label classes our emit_html step strips
        # (ltx_authors, ltx_abstract, ltx_bibliography, …). Both
        # fields saved per pair.
        pair = {
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "section_idx": idx,
            "section_title": section_doc.title,
            "title": section_doc.title,
            "url": section_doc.source_url,
            "local_pdf": str(pdf_path),
            "page_range": list(page_range) if page_range else None,
            "input_ocr": input_ocr,
            "output_html": html_doc,
            "raw_source_html": raw_html,
        }
        section_slug = _slug(section_doc.title.split("—", 1)[-1].strip())
        out_path = out_dir / f"{arxiv_id.replace('.', '_')}__{idx:02d}_{section_slug}.json"
        out_path.write_text(json.dumps(pair, ensure_ascii=False))
        stats["ok"] += 1
        stats["sections"] += 1

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                    help="Local Arxiiv Repo root (papers/ directory)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/pairs/arxiv"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Max papers to process (default: all)")
    ap.add_argument("--shuffle", action="store_true",
                    help="Process papers in random order (better for stratification)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if not args.repo.exists():
        print(f"error: repo not found: {args.repo}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[scan] {args.repo}", file=sys.stderr)
    candidates = find_local_pdfs(args.repo)
    print(f"[scan] found {len(candidates):,} PDFs with recognizable arXiv IDs",
          file=sys.stderr)

    if args.shuffle:
        random.Random(args.seed).shuffle(candidates)
    if args.limit:
        candidates = candidates[:args.limit]

    # Skip papers we've already processed (any file starting with this arxiv_id).
    existing_ids: set[str] = set()
    for p in args.out_dir.glob("*.json"):
        stem = p.stem.split("__")[0]  # drop "__NN_section_slug" if present
        existing_ids.add(stem.replace("_", "."))
    candidates = [(aid, p) for (aid, p) in candidates if aid not in existing_ids]
    print(f"[plan] processing {len(candidates):,} papers "
          f"({len(existing_ids):,} already done)", file=sys.stderr)

    start = time.time()
    totals = {"ok": 0, "fetch_error": 0, "parse_error": 0,
              "emitter_drop": 0, "axe_drop": 0, "whole_paper_fallback": 0}

    work_items = [(aid, str(p), str(args.out_dir)) for (aid, p) in candidates]
    done = 0
    for stats in run_in_pool(process_paper, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        aid = stats.get("arxiv_id", "?")
        sections = stats.get("sections", 0)
        if stats.get("fetch_error") or stats.get("parse_error"):
            outcome = "fetch_err" if stats.get("fetch_error") else "parse_err"
        elif stats.get("ok"):
            outcome = f"ok ({sections}sec)"
        else:
            outcome = "no-sections"
        msg = stats.get("msg", "")
        fb = " [fallback]" if stats.get("whole_paper_fallback") else ""
        print(f"[{done}/{len(work_items)}] {aid:14} {outcome:18}{fb} {msg[:50]}",
              file=sys.stderr)

    elapsed = time.time() - start
    print(f"\n[summary] processed={done}  sections_kept={totals['ok']}  "
          f"fetch_err={totals['fetch_error']}  parse_err={totals['parse_error']}  "
          f"emit_drop={totals['emitter_drop']}  axe_drop={totals['axe_drop']}  "
          f"whole_paper_fallbacks={totals['whole_paper_fallback']}  "
          f"in {elapsed:.1f}s  workers={args.workers}", file=sys.stderr)


if __name__ == "__main__":
    main()
