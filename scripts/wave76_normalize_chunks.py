#!/usr/bin/env python3
"""Wave 76: retroactively normalize chunk module_id + bloom_level fields.

Three data-hygiene fixes flagged by the external KG-quality review:

  (1) Module IDs occasionally leak short forms (``application``,
      ``content_01``, ``summary``) when the IMSCC inventory indexes a
      file by its bare stem instead of the prefixed
      ``week_NN_<slot>`` form. The pedagogy graph normalizes during
      build, but chunk records themselves are inconsistent. We rewrite
      ``source.module_id`` (and any top-level ``module_id``) to the
      canonical form using the chunk's own ``source.item_path`` /
      ``source.week_num`` to recover the week number.

  (2) Compound ``bloom_level`` values (``remember-apply``,
      ``understand-analyze``, ``apply-analyze``,
      ``analyze-evaluate``) violate the canonical six-value enum. We
      split on ``-`` and keep the higher Bloom level as the primary,
      storing the lower in a new optional ``bloom_level_secondary``
      field (Wave 76 schema addition). No information lost.

  (3) ``chunks.json`` and ``chunks.jsonl`` must contain the same
      chunk list in the same order. We re-emit both from the same
      normalized in-memory list, then assert parity by reading both
      back and comparing line-count + per-line dict equality.

A ``.wave76e.bak`` backup is written for both files before any
in-place edit. ``learning_outcome_refs`` (touched by Worker C's
retag pass) and every other chunk field are preserved untouched.

Usage::

    python -m scripts.wave76_normalize_chunks \
        [--course-slug rdf-shacl-550-rdf-shacl-550] \
        [--libv2-root LibV2]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import (  # noqa: E402  (after sys.path tweak)
    canonicalize_bloom_level,
    normalize_module_id,
)


def _load_chunks(jsonl_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _normalize_chunk(chunk: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (mutated chunk, per-chunk diff stats)."""
    stats: Dict[str, Any] = {
        "module_id_changed": False,
        "module_id_unresolved": False,
        "bloom_split": False,
        "bloom_compound_input": None,
        "bloom_invalid": False,
    }

    src = chunk.get("source") if isinstance(chunk.get("source"), dict) else None

    # ---------- module_id normalization ----------
    item_path = src.get("item_path") if src else None
    week_num = src.get("week_num") if src else None
    # `week_num` is not a standard chunk_v4 field but some legacy chunks may
    # carry it; gracefully fall back to None when absent.
    if src and src.get("module_id"):
        original_mid = src["module_id"]
        new_mid, changed = normalize_module_id(
            original_mid,
            item_path=item_path,
            week_num=week_num,
        )
        if changed:
            src["module_id"] = new_mid
            stats["module_id_changed"] = True
        elif new_mid and not new_mid.startswith("week_"):
            # We tried to normalize but couldn't recover a week number.
            stats["module_id_unresolved"] = True

    # Mirror the change at the top level if one happens to live there
    # (defensive — chunk_v4 keeps module_id under source, but legacy
    # exporters sometimes duplicated it at the root).
    if "module_id" in chunk and isinstance(chunk["module_id"], str):
        original_top = chunk["module_id"]
        new_top, changed = normalize_module_id(
            original_top,
            item_path=item_path,
            week_num=week_num,
        )
        if changed:
            chunk["module_id"] = new_top

    # ---------- bloom_level canonicalization ----------
    bl_value = chunk.get("bloom_level")
    if isinstance(bl_value, str) and bl_value:
        primary, secondary = canonicalize_bloom_level(bl_value)
        if secondary is not None:
            stats["bloom_split"] = True
            stats["bloom_compound_input"] = bl_value
            chunk["bloom_level"] = primary
            chunk["bloom_level_secondary"] = secondary
        elif primary is not None and primary != bl_value:
            # Lowercase / canonicalized single-form value — apply but
            # don't count as a compound split.
            chunk["bloom_level"] = primary
        elif primary is None:
            stats["bloom_invalid"] = True

    return chunk, stats


def _write_outputs(chunks: List[Dict[str, Any]], jsonl_path: Path, json_path: Path) -> None:
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, indent=2, ensure_ascii=False)


def _assert_parity(jsonl_path: Path, json_path: Path) -> None:
    """Round-trip both files and assert content equality."""
    jsonl_chunks: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                jsonl_chunks.append(json.loads(line))
    with json_path.open("r", encoding="utf-8") as fh:
        json_chunks = json.load(fh)
    if len(jsonl_chunks) != len(json_chunks):
        raise RuntimeError(
            f"parity: line count mismatch — jsonl={len(jsonl_chunks)} vs "
            f"json={len(json_chunks)}"
        )
    for idx, (a, b) in enumerate(zip(jsonl_chunks, json_chunks)):
        if a != b:
            raise RuntimeError(
                f"parity: chunk index {idx} differs "
                f"(jsonl id={a.get('id')!r}, json id={b.get('id')!r})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-slug",
        default="rdf-shacl-550-rdf-shacl-550",
        help="LibV2 course slug under <libv2-root>/courses/.",
    )
    parser.add_argument(
        "--libv2-root",
        default="LibV2",
        help="Path to LibV2 root (default: LibV2).",
    )
    args = parser.parse_args()

    libv2_root = (PROJECT_ROOT / args.libv2_root).resolve()
    corpus_dir = libv2_root / "courses" / args.course_slug / "corpus"
    jsonl_path = corpus_dir / "chunks.jsonl"
    json_path = corpus_dir / "chunks.json"

    if not jsonl_path.exists():
        print(f"chunks.jsonl not found at {jsonl_path}", file=sys.stderr)
        return 2

    # Backup before in-place edit (Worker C already left a .bak; we
    # write a Worker E-specific suffix to avoid clobbering theirs).
    jsonl_bak = jsonl_path.with_suffix(jsonl_path.suffix + ".wave76e.bak")
    json_bak = json_path.with_suffix(json_path.suffix + ".wave76e.bak")
    shutil.copy2(jsonl_path, jsonl_bak)
    if json_path.exists():
        shutil.copy2(json_path, json_bak)
    print(f"Backups: {jsonl_bak.name}, {json_bak.name}")

    chunks = _load_chunks(jsonl_path)
    print(f"Loaded {len(chunks)} chunks from {jsonl_path.relative_to(PROJECT_ROOT)}")

    module_id_changed = 0
    module_id_unresolved = 0
    bloom_split = 0
    bloom_invalid = 0
    bloom_compound_counter: Counter = Counter()
    primary_after: Counter = Counter()
    secondary_after: Counter = Counter()

    normalized: List[Dict[str, Any]] = []
    for chunk in chunks:
        new_chunk, stats = _normalize_chunk(chunk)
        normalized.append(new_chunk)
        if stats["module_id_changed"]:
            module_id_changed += 1
        if stats["module_id_unresolved"]:
            module_id_unresolved += 1
        if stats["bloom_split"]:
            bloom_split += 1
            bloom_compound_counter[stats["bloom_compound_input"]] += 1
        if stats["bloom_invalid"]:
            bloom_invalid += 1

    for chunk in normalized:
        bl = chunk.get("bloom_level")
        if isinstance(bl, str):
            primary_after[bl] += 1
        sec = chunk.get("bloom_level_secondary")
        if isinstance(sec, str):
            secondary_after[sec] += 1

    _write_outputs(normalized, jsonl_path, json_path)
    _assert_parity(jsonl_path, json_path)

    print()
    print("=== Wave 76 chunk normalization summary ===")
    print(f"Total chunks: {len(normalized)}")
    print(f"module_id rewritten to canonical week_NN_<slot>: {module_id_changed}")
    print(f"module_id un-resolvable (no week info, kept as-is): {module_id_unresolved}")
    print(f"bloom_level compound values split: {bloom_split}")
    if bloom_compound_counter:
        print("  compound-value breakdown:")
        for value, count in sorted(bloom_compound_counter.items()):
            print(f"    {value}: {count}")
    print(f"bloom_level invalid (passed through): {bloom_invalid}")
    print(f"bloom_level distribution after normalization:")
    for level in ("remember", "understand", "apply", "analyze", "evaluate", "create"):
        if level in primary_after:
            print(f"    {level}: {primary_after[level]}")
    if secondary_after:
        print("bloom_level_secondary distribution:")
        for level, count in sorted(
            secondary_after.items(), key=lambda kv: kv[0]
        ):
            print(f"    {level}: {count}")
    print()
    print("Parity check: chunks.json == chunks.jsonl (line-count + per-line equal).")
    print(f"Wrote: {jsonl_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {json_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
