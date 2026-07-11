"""One-time batch: warm the HTML→PDF render cache.

For every pair file without `local_pdf`, compute the deterministic
cache path (see semantik_structure.prerender_cache) and render the
`output_html` to that path with Playwright. Skips entries already in
the cache, so the script is idempotent and safely resumable.

Parallelized across N worker processes; each worker owns its own
HtmlValidator (Chromium) instance. 4-8 workers saturates a 3060-era
CPU without thrashing.

Usage:
    python scripts/prerender_pairs.py                   # default: 4 workers
    python scripts/prerender_pairs.py --workers 8
    python scripts/prerender_pairs.py --limit 100       # smoke test

After running, data-gen hits the cache and skips Playwright entirely
for every pair that this script has rendered.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from semantik_structure.prerender_cache import DEFAULT_CACHE_DIR, cache_path_for
from semantik_structure.worker_pool import run_in_pool


def iter_pairs(pair_dirs: list[Path]) -> list[tuple[str, str]]:
    """Return [(pair_path_str, html), ...] for every pair that needs rendering.

    Filters out pairs that already carry a local_pdf (arXiv, form PDFs)
    and pairs whose rendered output is already in the cache.
    """
    work: list[tuple[str, str]] = []
    for d in pair_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                pair = json.loads(p.read_text())
            except Exception:
                continue
            if pair.get("local_pdf"):
                continue
            html = pair.get("output_html")
            if not html:
                continue
            if cache_path_for(html).exists():
                continue
            work.append((str(p), html))
    return work


def render_one(validator, work: tuple[str, str]) -> dict:
    """Worker body: render one HTML→PDF into the cache, atomically."""
    from semantik_structure.prerender_cache import cache_path_for, get_or_render
    pair_path, html = work
    stats = {"pair": Path(pair_path).name, "ok": False, "error": None}
    try:
        path = cache_path_for(html)
        if path.exists():
            stats["ok"] = True
            stats["skipped"] = True
            return stats
        get_or_render(html, validator, path)
        stats["ok"] = True
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dirs", type=Path, nargs="+",
                    default=[Path("data/pairs/wikipedia"),
                             Path("data/pairs/openstax"),
                             Path("data/pairs/federal_register"),
                             Path("data/pairs/gutenberg"),
                             Path("data/pairs/forms")])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N pairs (smoke testing).")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = ap.parse_args()

    print(f"[plan] cache dir: {args.cache_dir}", file=sys.stderr)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    work = iter_pairs(args.pair_dirs)
    if args.limit:
        work = work[:args.limit]
    if not work:
        print("[done] nothing to render — cache is warm", file=sys.stderr)
        return

    print(f"[plan] {len(work)} pairs to render  workers={args.workers}",
          file=sys.stderr)

    n_done = 0
    n_ok = 0
    n_err = 0
    n_skip = 0
    start = time.time()
    for stats in run_in_pool(render_one, work, workers=args.workers):
        n_done += 1
        if stats.get("error"):
            n_err += 1
            if n_err <= 10:
                print(f"  ERR {stats['pair'][:60]}: {stats['error'][:80]}",
                      file=sys.stderr)
        elif stats.get("skipped"):
            n_skip += 1
        else:
            n_ok += 1
        if n_done % 50 == 0 or n_done == len(work):
            rate = n_done / max(0.1, time.time() - start) * 60
            print(f"[{n_done}/{len(work)}] ok={n_ok} err={n_err} "
                  f"skipped={n_skip}  {rate:.0f} pairs/min  "
                  f"elapsed={(time.time()-start)/60:.1f}min",
                  file=sys.stderr)

    total_cache_size = sum(p.stat().st_size for p in args.cache_dir.glob("*.pdf"))
    print(f"\n[done] rendered={n_ok} skipped={n_skip} errors={n_err}  "
          f"cache size: {total_cache_size / (1024*1024):.0f} MB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
