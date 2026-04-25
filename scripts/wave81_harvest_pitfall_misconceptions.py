#!/usr/bin/env python3
"""Wave 81 retroactive harvest: pull pitfall misconceptions from HTML.

The Wave 79 content-generator subagent line emits the misconception
paragraph with ``data-cf-misconception="true"`` on the <p> but does
NOT always populate the JSON-LD ``misconceptions[]`` array. Trainforge
historically harvested JSON-LD only, so for the rdf-shacl-551-2
archive (Path B regen) the per-chunk ``misconceptions[]`` count
dropped from the v1 baseline of 67 to 45.

Wave 81 forward fix: ``Courseforge/templates/chunk_templates.md``
Template 3 spec now mandates dual-emit (HTML attr AND JSON-LD entry).
Wave 81 backward bridge: ``Trainforge/parsers/html_content_parser.py``
falls back to scraping the HTML attr when JSON-LD is absent.

This script applies the same bridging logic retroactively to the
already-archived rdf-shacl-551-2 corpus so the v1 baseline is
restored without re-running content generation. It:

* Opens the IMSCC zip under ``LibV2/courses/rdf-shacl-551-2/source/imscc``.
* Walks every HTML file inside the zip looking for <section> blocks
  carrying ``data-cf-misconception="true"`` on a <p> child.
* Extracts the misconception statement (the data-cf-misconception
  paragraph) and the correction (the paragraph following the
  ``<h4>The right approach</h4>`` or ``<h4>Correct approach</h4>``
  sub-heading inside the same section).
* Loads ``corpus/chunks.jsonl``, maps each HTML file path
  (``week_NN/<page>.html``) to the chunks whose ``source.item_path``
  matches, and appends any newly-harvested misconceptions to those
  chunks' ``misconceptions[]`` arrays — without producing duplicates
  (text-equality on the misconception statement is the dedupe key).
* Backs the prior chunks file up as ``corpus/chunks.jsonl.bak``
  (``--force-bak`` overwrites; default refuses to clobber).
* Prints a per-chunk and total-count report.

Usage::

    python scripts/wave81_harvest_pitfall_misconceptions.py
    python scripts/wave81_harvest_pitfall_misconceptions.py --dry-run
    python scripts/wave81_harvest_pitfall_misconceptions.py \\
        --archive LibV2/courses/<other>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Trainforge.parsers.html_content_parser import (  # noqa: E402
    HTMLContentParser,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    text = re.sub(r"\s+", " ", s or "").strip().lower()
    # Strip surrounding ASCII or curly quotes — the HTML-attr paragraph
    # often wraps the JSON-LD's de-quoted statement with quotes plus a
    # trailing explanatory clause, so equality alone misses the overlap.
    return text.strip('"“”‘’\'')


def _scrape_misconceptions_from_html(html: str) -> List[Dict[str, Any]]:
    """Return any ``data-cf-misconception="true"`` paragraphs as
    misconception dicts, using the same bridging logic as the parser."""
    parser = HTMLContentParser()
    # Re-use the parser's private helper. We pass an empty ``existing``
    # list so every attr paragraph gets emitted; the caller then applies
    # JSON-LD-wins dedupe against the chunk's already-stored entries.
    return parser._extract_misconceptions_from_attrs(html, existing=[])


def _file_to_module_id(item_path: str) -> str:
    """Convert an IMSCC HTML path like ``week_02/week_02_pitfall_01.html``
    into a chunk's ``source.module_id`` ("week_02_pitfall_01")."""
    stem = Path(item_path).stem
    return stem


# ---------------------------------------------------------------------------
# Core harvest
# ---------------------------------------------------------------------------

def collect_html_misconceptions(
    imscc_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Walk every HTML file in the IMSCC zip and return a map of
    ``html_path`` -> list of harvested misconception dicts."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    with zipfile.ZipFile(imscc_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".html"):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            if 'data-cf-misconception="true"' not in text:
                continue
            mcs = _scrape_misconceptions_from_html(text)
            if mcs:
                out[name] = mcs
    return out


def update_chunks(
    chunks: List[Dict[str, Any]],
    html_misconceptions: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, int], int]:
    """Append HTML-attr misconceptions to chunks whose ``item_path``
    matches an HTML file the harvester collected from. Dedupe against
    each chunk's existing ``misconceptions[]`` (text equality on the
    misconception statement). Returns a per-chunk delta map plus the
    total number of misconceptions added."""
    # Build lookup: item_path -> list of chunk indices
    by_item_path: Dict[str, List[int]] = {}
    for idx, chunk in enumerate(chunks):
        item_path = (chunk.get("source") or {}).get("item_path")
        if not item_path:
            continue
        by_item_path.setdefault(item_path, []).append(idx)

    deltas: Dict[str, int] = {}
    total_added = 0

    for html_path, harvested in html_misconceptions.items():
        chunk_indices = by_item_path.get(html_path, [])
        if not chunk_indices:
            # No chunk maps to this page — skip silently.
            continue

        # The harvest target is the chunk whose section_heading or
        # html_xpath best matches a common_pitfall section. Without a
        # finer-grained signal in chunks.jsonl, attach all harvested
        # misconceptions to the FIRST chunk derived from this HTML
        # file. That matches the historical Wave 60+ chunker behavior
        # of folding pitfall sections into the lead chunk.
        target_idx = chunk_indices[0]
        chunk = chunks[target_idx]
        existing = list(chunk.get("misconceptions") or [])
        # Some pre-Wave-81 chunks store the misconception under
        # ``statement`` (the legacy field name) rather than the canonical
        # ``misconception``. Read both so the dedupe never double-counts
        # across the field-name boundary.
        existing_norm: List[str] = []
        for e in existing:
            if not isinstance(e, dict):
                continue
            stmt = e.get("misconception") or e.get("statement")
            if stmt:
                existing_norm.append(_normalize_text(stmt))

        added_for_chunk = 0
        for mc in harvested:
            statement = mc.get("misconception", "")
            norm_candidate = _normalize_text(statement)
            # Bidirectional substring containment matches the parser's
            # JSON-LD-wins dedupe so the harvest doesn't double-count
            # quoted-vs-de-quoted variants of the same misconception.
            if any(
                e and (e in norm_candidate or norm_candidate in e)
                for e in existing_norm
            ):
                continue
            existing.append(mc)
            existing_norm.append(norm_candidate)
            added_for_chunk += 1

        if added_for_chunk:
            chunk["misconceptions"] = existing
            deltas[chunk["id"]] = added_for_chunk
            total_added += added_for_chunk

    return deltas, total_added


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_archive(archive_arg: Optional[str]) -> Path:
    if archive_arg is None:
        return REPO_ROOT / "LibV2" / "courses" / "rdf-shacl-551-2"
    p = Path(archive_arg)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        default=None,
        help="Path to the LibV2 course archive root "
        "(default: LibV2/courses/rdf-shacl-551-2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delta report without writing any files.",
    )
    parser.add_argument(
        "--force-bak",
        action="store_true",
        help="Overwrite an existing chunks.jsonl.bak (default: refuse).",
    )
    args = parser.parse_args()

    archive_root = _resolve_archive(args.archive)
    if not archive_root.is_dir():
        print(f"ERROR: archive root not found: {archive_root}", file=sys.stderr)
        return 1

    # Locate the IMSCC and chunks file.
    imscc_dir = archive_root / "source" / "imscc"
    imscc_files = list(imscc_dir.glob("*.imscc"))
    if not imscc_files:
        print(
            f"ERROR: no .imscc file under {imscc_dir}", file=sys.stderr,
        )
        return 1
    imscc_path = imscc_files[0]

    chunks_path = archive_root / "corpus" / "chunks.jsonl"
    if not chunks_path.is_file():
        print(f"ERROR: chunks file not found: {chunks_path}", file=sys.stderr)
        return 1

    # Load chunks.
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    # Pre-count current misconceptions for the report.
    pre_total = sum(len(c.get("misconceptions") or []) for c in chunks)
    pre_chunks_with = sum(
        1 for c in chunks if (c.get("misconceptions") or [])
    )

    # Walk the IMSCC archive and harvest misconceptions per HTML file.
    print(f"Reading IMSCC archive: {imscc_path}")
    html_misconceptions = collect_html_misconceptions(imscc_path)
    print(
        f"  HTML files with data-cf-misconception attrs: "
        f"{len(html_misconceptions)}"
    )
    harvested_total = sum(len(v) for v in html_misconceptions.values())
    print(f"  Total misconception paragraphs harvested: {harvested_total}")

    # Apply harvest -> chunks (in-memory).
    deltas, added_total = update_chunks(chunks, html_misconceptions)

    post_total = sum(len(c.get("misconceptions") or []) for c in chunks)
    post_chunks_with = sum(
        1 for c in chunks if (c.get("misconceptions") or [])
    )

    print()
    print("=== Harvest report ===")
    print(f"  Pre  total misconceptions: {pre_total}")
    print(f"  Post total misconceptions: {post_total}")
    print(f"  Delta:                     +{added_total}")
    print(f"  Pre  chunks with mc[]:     {pre_chunks_with}")
    print(f"  Post chunks with mc[]:     {post_chunks_with}")
    if deltas:
        print()
        print("  Per-chunk additions:")
        for chunk_id, count in sorted(deltas.items()):
            print(f"    {chunk_id}: +{count}")

    if args.dry_run:
        print()
        print("(dry-run: no files modified)")
        return 0

    if added_total == 0:
        print()
        print("No new misconceptions to write. Skipping file update.")
        return 0

    # Backup + write.
    bak_path = chunks_path.with_suffix(chunks_path.suffix + ".bak")
    if bak_path.exists() and not args.force_bak:
        # Use a wave-suffixed alt to preserve history.
        alt_bak = chunks_path.with_suffix(chunks_path.suffix + ".wave81.bak")
        print(
            f"  {bak_path.name} already exists; writing alt backup as "
            f"{alt_bak.name}"
        )
        shutil.copy2(chunks_path, alt_bak)
    else:
        shutil.copy2(chunks_path, bak_path)

    with chunks_path.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {chunks_path}")
    print(f"Backup: {bak_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
