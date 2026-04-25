#!/usr/bin/env python3
"""Wave 81: reclassify existing chunks to honor ``data-cf-template-type``.

Wave 79 C added four Courseforge content-generator templates (procedure,
real_world_scenario, common_pitfall, problem_solution) which embed
``data-cf-template-type="<value>"`` on every section root. Wave 81 wires
that attribute through the chunker (``html_content_parser`` →
``process_course._merge_small_sections``) so freshly-generated chunks
carry the canonical template label as ``chunk_type``.

This script retroactively reclassifies an *existing* chunks.jsonl that
predates the Wave 81 chunker change. For each chunk it:

  1. Resolves the source IMSCC HTML file via ``chunk.source.item_path``.
  2. Locates the matching ``<section>`` element by ``data-dart-block-id``
     (if present) or by section heading match (using
     ``HTMLContentParser._extract_sections``) at the chunk's
     ``position_in_module`` index.
  3. Reads ``data-cf-template-type`` from the enclosing section root.
  4. Updates the chunk's ``chunk_type`` field if the parsed value differs
     from the existing value AND is in the canonical ChunkType enum.

The script writes ``chunks.jsonl.bak`` + ``chunks.json.bak`` before
mutating any data, then re-emits both files in lockstep.

Usage::

    python -m scripts.wave81_reclassify_chunks \\
        --course-slug rdf-shacl-551-2

Default course slug: ``rdf-shacl-551-2``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the project root importable when running as ``python -m scripts/...``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402
from Trainforge.process_course import CANONICAL_CHUNK_TYPES  # noqa: E402

logger = logging.getLogger("wave81_reclassify_chunks")


# ---------------------------------------------------------------------------
# IMSCC unpacking
# ---------------------------------------------------------------------------


def _resolve_imscc_path(course_dir: Path) -> Path:
    """Locate the IMSCC archive under ``<course_dir>/source/imscc/``."""
    imscc_dir = course_dir / "source" / "imscc"
    if not imscc_dir.exists():
        raise FileNotFoundError(f"No source/imscc directory under {course_dir}")
    candidates = sorted(imscc_dir.glob("*.imscc")) + sorted(imscc_dir.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No .imscc / .zip archive under {imscc_dir}")
    return candidates[0]


def _unpack_imscc(imscc_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(imscc_path) as zf:
        zf.extractall(dest)


# ---------------------------------------------------------------------------
# Section lookup
# ---------------------------------------------------------------------------


_SECTION_OPEN_RE = re.compile(r'<section\b([^>]*)>', re.IGNORECASE)
_TEMPLATE_TYPE_ATTR_RE = re.compile(
    r'data-cf-template-type="([^"]*)"', re.IGNORECASE
)
_DART_BLOCK_ID_ATTR_RE = re.compile(
    r'data-dart-block-id="([^"]*)"', re.IGNORECASE
)
_HEADING_RE = re.compile(r'<h([1-6])([^>]*)>([^<]+)</h\1>', re.IGNORECASE)


def _extract_subsection_text_with_markers(
    html_text: str, page_template_type: Optional[str],
) -> Optional[str]:
    """Re-derive a chunk-friendly text body that preserves H3/H4 boundaries.

    For Wave 81 template-aware extractors to fire on retroactively-
    reclassified chunks, the chunk text must include subsection headers
    (e.g. ``Your Task``, ``Approach``, ``What looks like the right
    answer``). The legacy chunker stripped these because each H4 was its
    own ``ContentSection`` and merge collapsed them without preserving
    the heading text.

    This helper walks the page-level HTML and returns a single text body
    where each subsection is prefixed with ``\\n\\n<heading>:\\n\\n``.
    The regex extractors in
    ``Trainforge/instruction_pair_extractor.py`` then pick up those
    headers via the same anchors used for naturally-labelled chunks.
    """
    if not html_text:
        return None
    # Strip <script>/<style> blocks first so their text doesn't leak.
    cleaned = re.sub(
        r'<(script|style)\b[^>]*>.*?</\1>', '', html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    headings = list(_HEADING_RE.finditer(cleaned))
    if not headings:
        return None
    parts: List[str] = []
    for i, m in enumerate(headings):
        heading_text = m.group(3).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(cleaned)
        body_html = cleaned[start:end]
        # Strip remaining tags into plain text. Keep paragraph spacing.
        body_html = re.sub(r'<br\s*/?>', '\n', body_html, flags=re.IGNORECASE)
        body_html = re.sub(
            r'</p>\s*<p[^>]*>', '\n\n', body_html, flags=re.IGNORECASE
        )
        body_text = re.sub(r'<[^>]+>', ' ', body_html)
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        if not body_text:
            continue
        parts.append(f"{heading_text}:\n\n{body_text}")
    if not parts:
        return None
    return "\n\n".join(parts)


def _extract_template_types_for_html(html_text: str) -> Dict[str, str]:
    """Return a mapping ``section_heading -> template_type``.

    Walks the HTML in section_open / heading order: every section root that
    carries ``data-cf-template-type`` propagates that value down to the
    headings that follow until the next section root opens. Multiple
    headings inside the same section therefore share the section's
    template_type — mirrors the behavior in
    ``HTMLContentParser._extract_sections``.
    """
    if not html_text:
        return {}
    # Build a flat list of (offset, kind, payload) events.
    events: List[Tuple[int, str, Any]] = []
    for m in _SECTION_OPEN_RE.finditer(html_text):
        attrs = m.group(1)
        tt_match = _TEMPLATE_TYPE_ATTR_RE.search(attrs)
        events.append(
            (m.start(), "section_open", tt_match.group(1) if tt_match else None)
        )
    heading_pattern = re.compile(
        r'<h([1-6])([^>]*)>([^<]+)</h\1>', re.IGNORECASE
    )
    for m in heading_pattern.finditer(html_text):
        events.append((m.start(), "heading", m.group(3).strip()))
    events.sort(key=lambda e: e[0])
    out: Dict[str, str] = {}
    current_template: Optional[str] = None
    for _, kind, payload in events:
        if kind == "section_open":
            current_template = payload
        elif kind == "heading":
            if current_template:
                # First-occurrence wins on duplicate headings (rare).
                out.setdefault(payload, current_template)
    return out


# ---------------------------------------------------------------------------
# Chunk reclassification
# ---------------------------------------------------------------------------


def _load_html_metadata(
    html_path: Path,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Memoize per-file metadata (heading→template_type, page text body)."""
    key = str(html_path)
    if key in cache:
        return cache[key]
    try:
        html_text = html_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read %s: %s", html_path, exc)
        cache[key] = {"mapping": {}, "page_template": None, "page_text": None}
        return cache[key]
    mapping = _extract_template_types_for_html(html_text)
    distinct = {v for v in mapping.values() if v}
    page_template = next(iter(distinct)) if len(distinct) == 1 else None
    page_text = _extract_subsection_text_with_markers(html_text, page_template)
    cache[key] = {
        "mapping": mapping,
        "page_template": page_template,
        "page_text": page_text,
    }
    return cache[key]


def _resolve_chunk_template_type(
    chunk: Dict[str, Any],
    html_root: Path,
    parser_cache: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Return the data-cf-template-type for ``chunk`` or ``None`` if absent."""
    source = chunk.get("source") or {}
    item_path = source.get("item_path")
    if not item_path:
        return None
    html_path = html_root / item_path
    if not html_path.exists():
        return None
    meta = _load_html_metadata(html_path, parser_cache)
    mapping = meta["mapping"]
    if not mapping:
        return None
    section_heading = source.get("section_heading") or ""
    if section_heading and section_heading in mapping:
        return mapping[section_heading]
    # Some chunks hold the page-level title as section_heading because the
    # page has no internal headings — fall back to "any template_type
    # present in the file" since the whole page wraps in one
    # data-cf-template-type-tagged <section>.
    if meta["page_template"]:
        return meta["page_template"]
    return None


def _reclassify_chunks(
    chunks: List[Dict[str, Any]],
    html_root: Path,
    inject_markers: bool = True,
) -> Tuple[List[Dict[str, Any]], Counter, int]:
    """Reclassify chunks in-place; return (chunks, transitions, marker_count).

    The transitions counter is keyed by ``(old_type, new_type)`` so the
    caller can print a clean before/after table. ``marker_count`` reports
    how many chunks had subsection markers injected into their text body
    so the Wave 81 template-aware extractors can fire on them.

    For chunks that get reclassified to one of the four Wave 81 template
    types AND whose existing text doesn't already carry the expected
    subsection labels (``Your Task``, ``Approach``, ``What looks like the
    right answer``, ``Inputs`` / ``Steps`` / ``Output``,
    ``Walkthrough`` / ``Common Incorrect Approach``, ...), we re-derive
    a marker-augmented text body from the source HTML so each subsection
    is prefixed with ``<heading>:\\n\\n``. Both ``chunk["text"]`` and
    ``chunk["html"]`` are updated; the legacy plain-text shape stays
    available via ``chunk["text_legacy"]`` for downstream consumers that
    pinned to the old format.
    """
    transitions: Counter = Counter()
    parser_cache: Dict[str, Dict[str, Any]] = {}
    template_aware = {
        "procedure",
        "real_world_scenario",
        "common_pitfall",
        "problem_solution",
    }
    markers_injected = 0
    # First pass: reclassify chunk_type for every chunk that needs it.
    for chunk in chunks:
        old_type = chunk.get("chunk_type", "explanation")
        template_type = _resolve_chunk_template_type(chunk, html_root, parser_cache)
        if not template_type:
            continue
        if template_type not in CANONICAL_CHUNK_TYPES:
            continue
        if template_type != old_type:
            chunk["chunk_type"] = template_type
            transitions[(old_type, template_type)] += 1
    if not inject_markers:
        return chunks, transitions, markers_injected
    # Second pass: marker injection. We pick the first chunk per page that
    # has a template-aware chunk_type and overwrite its text with the
    # page-level marker-augmented text. This guarantees that page-level
    # subsection labels (Your Task / Approach / What looks like the right
    # answer / etc.) are present in at least one chunk per page so the
    # extractors can fire — without duplicating content into trailing
    # chunks of the same page.
    seen_pages: Dict[str, str] = {}
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type")
        if chunk_type not in template_aware:
            continue
        source = chunk.get("source") or {}
        item_path = source.get("item_path")
        if not item_path:
            continue
        if item_path in seen_pages and seen_pages[item_path] == chunk_type:
            continue
        if _text_has_template_labels(chunk.get("text") or "", chunk_type):
            seen_pages[item_path] = chunk_type
            continue
        html_path = html_root / item_path
        meta = _load_html_metadata(html_path, parser_cache)
        page_text = meta.get("page_text")
        if not page_text or not page_text.strip():
            continue
        if "text" in chunk and chunk["text"] != page_text:
            chunk.setdefault("text_legacy", chunk["text"])
        chunk["text"] = page_text
        markers_injected += 1
        seen_pages[item_path] = chunk_type
    return chunks, transitions, markers_injected


# Heuristic: if the chunk text already contains at least one anchor for
# the target template type, leave it alone. Avoids overwriting
# legitimately-distinct chunks (e.g. a procedure chunk that's just one
# subsection like ``Worked Example`` page-internally).
_TEMPLATE_ANCHORS: Dict[str, Tuple[str, ...]] = {
    "procedure": ("Steps", "Inputs", "Output"),
    "real_world_scenario": ("Your Task", "Approach", "Success Criteria"),
    "common_pitfall": (
        "What looks like the right answer", "Why it's wrong",
        "The right approach",
    ),
    "problem_solution": (
        "Walkthrough", "Common Incorrect Approach", "Problem",
    ),
}


def _text_has_template_labels(text: str, template_type: str) -> bool:
    if not text:
        return False
    anchors = _TEMPLATE_ANCHORS.get(template_type, ())
    if not anchors:
        return True  # nothing to check for unknown templates
    # Require at least two of the canonical anchors so we don't false-match
    # on common words. ``Steps`` and ``Approach`` show up frequently in
    # body text; pairing anchors keeps the heuristic honest.
    hits = sum(1 for a in anchors if re.search(rf"\b{re.escape(a)}\b", text))
    return hits >= 2


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _write_jsonl(path: Path, chunks: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False))
            fh.write("\n")


def _write_json_array(path: Path, chunks: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)


def _backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        logger.info("Backed up %s -> %s", path, bak)


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _course_dir(libv2_root: Path, slug: str) -> Path:
    return libv2_root / "courses" / slug


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wave81_reclassify_chunks",
        description=(
            "Reclassify chunks.jsonl chunk_type values to honor "
            "data-cf-template-type from the source IMSCC HTML "
            "(Wave 81)."
        ),
    )
    parser.add_argument(
        "--course-slug",
        default="rdf-shacl-551-2",
        help="LibV2 course slug (default: rdf-shacl-551-2)",
    )
    parser.add_argument(
        "--libv2-root",
        default=str(PROJECT_ROOT / "LibV2"),
        help=f"LibV2 root path (default: {PROJECT_ROOT / 'LibV2'})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute reclassification but do not write any files.",
    )
    parser.add_argument(
        "--unpack-dir",
        help=(
            "Optional directory to unpack the IMSCC into. Defaults to a "
            "temporary directory created under the course's "
            "source/imscc/ folder."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    libv2_root = Path(args.libv2_root)
    course_dir = _course_dir(libv2_root, args.course_slug)
    if not course_dir.exists():
        parser.error(f"Course directory does not exist: {course_dir}")

    chunks_jsonl = course_dir / "corpus" / "chunks.jsonl"
    chunks_json = course_dir / "corpus" / "chunks.json"
    if not chunks_jsonl.exists():
        parser.error(f"chunks.jsonl not found at {chunks_jsonl}")

    imscc_path = _resolve_imscc_path(course_dir)
    logger.info("Source IMSCC: %s", imscc_path)

    if args.unpack_dir:
        unpack_dir = Path(args.unpack_dir)
    else:
        unpack_dir = course_dir / "source" / "imscc" / "_wave81_unpacked"
    if unpack_dir.exists() and unpack_dir.is_dir():
        # Wipe + re-unpack so we always reflect the current archive.
        shutil.rmtree(unpack_dir)
    _unpack_imscc(imscc_path, unpack_dir)
    logger.info("Unpacked IMSCC to %s", unpack_dir)

    chunks = _load_jsonl(chunks_jsonl)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_jsonl)

    before = Counter(c.get("chunk_type") for c in chunks)
    chunks, transitions, markers_injected = _reclassify_chunks(
        chunks, unpack_dir
    )
    after = Counter(c.get("chunk_type") for c in chunks)

    print("=== Wave 81 reclassification report ===")
    print(f"Course: {args.course_slug}")
    print(f"Total chunks: {len(chunks)}")
    print()
    print("chunk_type distribution before:")
    for k, v in sorted(before.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:>22}: {v}")
    print()
    print("chunk_type distribution after:")
    for k, v in sorted(after.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:>22}: {v}")
    print()
    print(f"Total reclassifications: {sum(transitions.values())}")
    print(
        f"Subsection markers injected (text re-derived from source HTML): "
        f"{markers_injected}"
    )
    if transitions:
        print("Transitions (old_type -> new_type: count):")
        for (old, new), count in sorted(
            transitions.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"  {old} -> {new}: {count}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return 0

    _backup(chunks_jsonl)
    _backup(chunks_json)
    _write_jsonl(chunks_jsonl, chunks)
    if chunks_json.exists() or True:  # always re-emit json array for parity
        _write_json_array(chunks_json, chunks)
    print(f"\nWrote: {chunks_jsonl}")
    print(f"Wrote: {chunks_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
