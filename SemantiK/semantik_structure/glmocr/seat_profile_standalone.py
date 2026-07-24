"""Standalone CLI for the deterministic GLM-OCR heading-judge BOOK PROFILER +
seat-tier right-sizer (:mod:`semantik_structure.glmocr.seat_profile`).

Profiles one or more ``{stem}.glmocr_layout.json`` layout sidecars — fully
offline / deterministic / CPU-only (no GPU, no network, no LLM) — selects the
smallest seat CONTEXT tier whose judge digest budget holds each book's largest
chapter, and prints a per-book summary + the tier BUCKET map.

CLI
---
``python3 -m semantik_structure.glmocr.seat_profile_standalone \
    <layout.json> [<layout2.json> ...] [--json] [--no-chapter-mode] \
    [--no-doc-schema]``

Human (default): one line per layout —
``<basename>  chapters=<n> pending=<n> max_win=<tok> p50=<tok> p95=<tok> ->
tier=<name> (ctx=<context> seqs=<seqs>)[  OVERFLOW]`` — then the bucket map.

``--json``: a machine-readable object (per-book profiles + selection + the
bucket map) to stdout instead.

⚠️ The tier ``seqs`` above the validated 128k×8 / 250k×4 operating points are
EXTRAPOLATED — validate empirically before trusting them (see the module
warning in ``seat_profile``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .seat_profile import (
    bucket_profiles,
    profile_book,
    resolve_seat_tiers,
    resolve_select_safety,
    select_seat_tier,
)


def _human_line(basename: str, prof, tier, overflow: bool) -> str:
    return (
        f"{basename}  chapters={prof.n_chapters} pending={prof.n_pending} "
        f"max_win={prof.max_window_tokens} p50={prof.p50_window_tokens} "
        f"p95={prof.p95_window_tokens} -> tier={tier.name} "
        f"(ctx={tier.context} seqs={tier.seqs})"
        + ("  OVERFLOW" if overflow else "")
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic offline GLM-OCR heading-judge book profiler "
                    "+ seat-tier right-sizer (no GPU / network / LLM).")
    parser.add_argument("layouts", nargs="+",
                        help="one or more {stem}.glmocr_layout.json sidecars")
    parser.add_argument("--json", action="store_true",
                        help="emit a machine-readable JSON object instead of "
                             "the human summary")
    parser.add_argument("--no-chapter-mode", action="store_true",
                        help="profile without chapter-mode (default: on)")
    parser.add_argument("--no-doc-schema", action="store_true",
                        help="profile without the doc-schema preamble "
                             "(default: on)")
    args = parser.parse_args(argv)

    chapter_mode = not args.no_chapter_mode
    doc_schema = not args.no_doc_schema

    tiers = resolve_seat_tiers()
    safety = resolve_select_safety()

    rows = []  # (book_id, profile, tier, overflow)
    for layout in args.layouts:
        path = Path(layout)
        if not path.is_file():
            print(f"error: layout sidecar not found: {path}", file=sys.stderr)
            return 2
        prof = profile_book(path, chapter_mode=chapter_mode,
                            doc_schema=doc_schema)
        tier, overflow = select_seat_tier(prof, tiers, safety=safety)
        rows.append((prof.source, prof, tier, overflow))

    buckets = bucket_profiles([(book_id, prof) for book_id, prof, _, _ in rows])

    if args.json:
        out = {
            "chapter_mode": chapter_mode,
            "doc_schema": doc_schema,
            "safety": safety,
            "tiers": [
                {"name": t.name, "context": t.context, "seqs": t.seqs}
                for t in tiers
            ],
            "books": [
                {
                    "source": book_id,
                    "n_chapters": prof.n_chapters,
                    "n_pending": prof.n_pending,
                    "window_tokens": prof.window_tokens,
                    "max_window_tokens": prof.max_window_tokens,
                    "p50_window_tokens": prof.p50_window_tokens,
                    "p95_window_tokens": prof.p95_window_tokens,
                    "total_window_tokens": prof.total_window_tokens,
                    "largest_window_index": prof.largest_window_index,
                    "selected_tier": {
                        "name": tier.name,
                        "context": tier.context,
                        "seqs": tier.seqs,
                    },
                    "overflow": overflow,
                }
                for book_id, prof, tier, overflow in rows
            ],
            "buckets": {name: list(ids) for name, ids in buckets.items()},
        }
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    for book_id, prof, tier, overflow in rows:
        print(_human_line(book_id, prof, tier, overflow))

    print("\nseat-tier buckets (smallest -> largest context):")
    if not buckets:
        print("  (none)")
    for name, ids in buckets.items():
        print(f"  {name}: {len(ids)} book(s) -> {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
