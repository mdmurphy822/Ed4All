"""Crawl ar5iv HTML for arXiv IDs biased toward rare-class math density.

Phase 2 trained MathSpecialist on 331 ar5iv papers, but `math_type` macro-F1
came in at 0.766 vs target 0.80. Per-class breakdown showed the rare classes
`display` (300 train rows, F1 0.457) and `matrix` (48 rows, F1 0.667) as the
bottleneck. The existing corpus skews CS / general arXiv.

This crawler queries the arXiv Atom API for paper IDs in math-heavy and
physics-heavy categories where rare math types are denser, then downloads
ar5iv HTML for each new ID into ``data/ar5iv_html_cache/<id>.html`` so that
``build_math_specialist_data.py`` can pick them up on its next pass.

Behaviour
---------
* Skips IDs that already have an entry under the cache directory — never
  re-downloads existing papers.
* Sleeps between requests (~1 s default) and backs off exponentially on 429
  / 503. arXiv asks for 1 request / 3 s on the API; ar5iv has no documented
  rate limit but we apply the same courtesy.
* Resilient: any per-paper failure is logged and skipped, never aborts the
  whole crawl.
* Caps total wall-clock time so an overnight stall can't burn forever.

Network usage: this is the second network entry point in the repo. arXiv
API queries hit ``http://export.arxiv.org/api/query``; HTML fetches reuse
``parse_ar5iv.fetch_ar5iv_html``.

Usage::

    python scripts/crawl_ar5iv_by_category.py \\
        --target 200 \\
        --cache-dir data/ar5iv_html_cache
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


ARXIV_API = "http://export.arxiv.org/api/query"

# Rare-class-friendly category mix.
# Heavy weight on math.* (display equations + matrix usage), some physics
# (display equations / tensors), small slice of cs/stat to keep prose
# distribution diverse.
DEFAULT_CATEGORIES: list[tuple[str, float]] = [
    # math: 60%
    ("math.AG", 0.10),  # algebraic geometry — heavy display
    ("math.LA", 0.10),  # linear algebra — matrices
    ("math.AC", 0.08),  # commutative algebra
    ("math.FA", 0.08),  # functional analysis
    ("math.OC", 0.08),  # optimization & control — display + numbered
    ("math.NT", 0.08),  # number theory
    ("math.RT", 0.08),  # representation theory — matrices
    # physics: 25%
    ("math-ph", 0.10),  # math-physics
    ("hep-th", 0.08),   # high-energy theory
    ("gr-qc", 0.07),    # general relativity / tensors
    # cs/stat: 15%
    ("cs.IT", 0.07),    # information theory — numbered display
    ("cs.LG", 0.05),    # machine learning
    ("stat.ML", 0.03),
]

ARXIV_ID_RE = re.compile(
    r"http(?:s)?://arxiv\.org/abs/([\w\-./]+?)(?:v\d+)?$"
)


def _arxiv_query(
    category: str,
    *,
    start: int,
    max_results: int,
    timeout: int,
) -> list[str]:
    """Return arXiv IDs for one (category, page) query.

    The Atom feed orders results by submitted-date desc by default, which
    is fine for our purposes: we just want a wide pool of IDs per category.
    """
    params = {
        "search_query": f"cat:{category}",
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "dart-semantic/0.1 (math-specialist data crawl)"}
    r = requests.get(ARXIV_API, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    # The feed is Atom XML; grab <id>http://arxiv.org/abs/...</id> per entry.
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    ids: list[str] = []
    for entry in root.findall("atom:entry", ns):
        id_el = entry.find("atom:id", ns)
        if id_el is None or not id_el.text:
            continue
        m = ARXIV_ID_RE.match(id_el.text.strip())
        if m:
            ids.append(m.group(1))
    return ids


def _safe_cache_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_") + ".html"


def _fetch_ar5iv(
    arxiv_id: str,
    cache_dir: Path,
    *,
    timeout: int,
    sleep: float,
    max_backoff: float,
) -> str:
    """Fetch one paper's HTML to disk; return one of:
        'cached', 'fetched', 'fail-404', 'fail-other'.
    """
    out = cache_dir / _safe_cache_name(arxiv_id)
    if out.exists() and out.stat().st_size > 0:
        return "cached"

    # Reuse the canonical fetch path to keep one source of truth.
    from dart_semantic.parse_ar5iv import Ar5ivParseError, fetch_ar5iv_html

    backoff = sleep
    for attempt in range(4):
        try:
            html = fetch_ar5iv_html(arxiv_id, timeout=timeout)
        except Ar5ivParseError as exc:
            msg = str(exc)
            if "no HTML" in msg or "404" in msg:
                return "fail-404"
            # 429 / 503 / transient — retry with exponential backoff.
            if any(code in msg for code in ("429", "503", "502", "504", "timeout")):
                wait = min(max_backoff, backoff * (2 ** attempt))
                sys.stderr.write(
                    f"[backoff] {arxiv_id}: {msg} — sleep {wait:.1f}s\n"
                )
                time.sleep(wait)
                continue
            sys.stderr.write(f"[fetch fail] {arxiv_id}: {msg}\n")
            return "fail-other"
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[fetch error] {arxiv_id}: {exc}\n")
            return "fail-other"
        # Got it.
        out.write_text(html, encoding="utf-8")
        return "fetched"
    return "fail-other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/ar5iv_html_cache"),
        help="ar5iv HTML cache shared with build_math_specialist_data.py",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=200,
        help="Stop once this many *new* papers have been downloaded.",
    )
    ap.add_argument(
        "--api-page-size",
        type=int,
        default=100,
        help="Results per arXiv API call (max 2000 per arXiv docs; "
             "100 keeps each call quick).",
    )
    ap.add_argument(
        "--max-pages-per-cat",
        type=int,
        default=20,
        help="Hard ceiling on API pages per category (defends against an "
             "exhausted-feed loop).",
    )
    ap.add_argument(
        "--api-timeout",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--fetch-timeout",
        type=int,
        default=25,
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.1,
        help="Polite delay between ar5iv fetches.",
    )
    ap.add_argument(
        "--api-sleep",
        type=float,
        default=3.1,
        help="Polite delay between arXiv API calls (their docs ask 3s).",
    )
    ap.add_argument(
        "--max-backoff",
        type=float,
        default=60.0,
    )
    ap.add_argument(
        "--time-budget-min",
        type=float,
        default=90.0,
        help="Hard wall-clock cap.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("data/ar5iv_html_cache/_crawl_report.json"),
    )
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Existing cache: skip these IDs and don't count them toward the
    # target (the goal is *new* papers).
    already_cached: set[str] = set()
    for p in args.cache_dir.glob("*.html"):
        if p.stat().st_size <= 0:
            continue
        # Reverse the safe-name mapping. Old IDs (pre-2007) used the
        # form ``hep-th/0501001`` which we cached as ``hep-th_0501001``.
        # Most modern IDs have no slash so a straight stem works.
        already_cached.add(p.stem.replace("_", "/", 1) if "/" not in p.stem else p.stem)
        already_cached.add(p.stem)
    n_cached_files = sum(1 for _ in args.cache_dir.glob("*.html"))
    print(f"[cache] {n_cached_files} files already cached "
          f"(across {len(already_cached)} known IDs)", file=sys.stderr)

    # Per-category target counts based on weight.
    cats = DEFAULT_CATEGORIES
    weight_sum = sum(w for _, w in cats)
    per_cat_target = {
        c: max(1, int(round(args.target * (w / weight_sum))))
        for c, w in cats
    }
    print("[plan] per-category fetch targets:", file=sys.stderr)
    for c, n in per_cat_target.items():
        print(f"  {c:10} {n}", file=sys.stderr)

    t0 = time.time()
    deadline = t0 + args.time_budget_min * 60.0

    n_attempted = 0
    n_fetched = 0
    n_cached_hit = 0
    n_404 = 0
    n_other_fail = 0
    fetched_per_cat: dict[str, int] = {c: 0 for c in per_cat_target}
    fetched_ids: list[str] = []

    # Round-robin through categories until each hits its target or we
    # hit the global target / time budget.
    cat_pages: dict[str, int] = {c: 0 for c in per_cat_target}
    cat_buffers: dict[str, list[str]] = {c: [] for c in per_cat_target}
    cat_exhausted: set[str] = set()

    def _global_done() -> bool:
        return n_fetched >= args.target

    def _cat_done(c: str) -> bool:
        return fetched_per_cat[c] >= per_cat_target[c] or c in cat_exhausted

    while not _global_done() and time.time() < deadline:
        active = [c for c in per_cat_target if not _cat_done(c)]
        if not active:
            print("[done] all categories satisfied or exhausted", file=sys.stderr)
            break
        progress_this_round = False
        for c in active:
            if _global_done() or time.time() >= deadline:
                break

            # Refill the buffer if needed.
            if not cat_buffers[c]:
                if cat_pages[c] >= args.max_pages_per_cat:
                    cat_exhausted.add(c)
                    continue
                start = cat_pages[c] * args.api_page_size
                cat_pages[c] += 1
                try:
                    ids = _arxiv_query(
                        c,
                        start=start,
                        max_results=args.api_page_size,
                        timeout=args.api_timeout,
                    )
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(f"[api error] cat={c} start={start}: {exc}\n")
                    time.sleep(args.api_sleep)
                    continue
                time.sleep(args.api_sleep)
                if not ids:
                    cat_exhausted.add(c)
                    continue
                rng.shuffle(ids)
                cat_buffers[c] = ids

            # Pop one ID.
            aid = cat_buffers[c].pop()
            n_attempted += 1
            # Skip already-cached.
            safe_name = _safe_cache_name(aid)
            if (args.cache_dir / safe_name).exists():
                n_cached_hit += 1
                continue
            result = _fetch_ar5iv(
                aid, args.cache_dir,
                timeout=args.fetch_timeout,
                sleep=args.sleep,
                max_backoff=args.max_backoff,
            )
            if result == "fetched":
                n_fetched += 1
                fetched_per_cat[c] += 1
                fetched_ids.append(aid)
                progress_this_round = True
                if n_fetched % 10 == 0 or n_fetched == 1:
                    print(
                        f"[fetched {n_fetched}/{args.target}] {aid} cat={c} "
                        f"(elapsed {time.time()-t0:.0f}s)",
                        file=sys.stderr,
                    )
                time.sleep(args.sleep)
            elif result == "cached":
                n_cached_hit += 1
            elif result == "fail-404":
                n_404 += 1
                # 404 from ar5iv just means no HTML; small sleep to avoid
                # hammering on bad IDs.
                time.sleep(min(args.sleep, 0.5))
            else:
                n_other_fail += 1
                time.sleep(args.sleep)

        # If a full round completed with zero progress AND every active
        # category is exhausted, break out — nothing else to try.
        if not progress_this_round and all(
            (not cat_buffers[c] and cat_pages[c] >= args.max_pages_per_cat)
            for c in active
        ):
            print("[done] all active categories drained without progress",
                  file=sys.stderr)
            break

    elapsed = time.time() - t0
    summary = {
        "elapsed_s": round(elapsed, 1),
        "n_attempted": n_attempted,
        "n_fetched_new": n_fetched,
        "n_already_cached": n_cached_hit,
        "n_404": n_404,
        "n_other_fail": n_other_fail,
        "fetched_per_category": fetched_per_cat,
        "per_category_target": per_cat_target,
        "fetched_ids_sample": fetched_ids[:50],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2))

    print("\n=== Crawl summary ===")
    print(json.dumps(summary, indent=2))
    print(f"[save] crawl report -> {args.report}")


if __name__ == "__main__":
    main()
