"""Regression: ``_load_semantik_chunkset_for_planning`` resolves a RAW course name.

The ``course_planning`` chunkset loader is called by three sites that pass the
RAW course name (e.g. ``UNIT_TEST_ALG_XYZ``) as ``course_slug``, but the LibV2
course directory is created with the SLUGIFIED name
(``unit-test-alg-xyz`` via :meth:`LibV2Storage._generate_slug` — lower + ``_``/``
`` -> ``-``). Before the slug fix the loader looked in
``courses/UNIT_TEST_ALG_XYZ/...`` (does not exist) instead of
``courses/unit-test-alg-xyz/...`` (exists) and returned ``({}, [])`` — starving
citation reselect / the sanitizer's chunk universe and degrading grounding to
``chapter_fallback``.

These tests build a temp LibV2 root with the SLUGIFIED chunkset dir and assert:

* a RAW course name resolves to the slugified dir (the fixed behavior), and
* an already-slugified ``course_slug`` still resolves (idempotence / no
  regression for a caller that already passes a correct slug).

No real course slugs are hardcoded — the course name is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.libv2_storage import LibV2Storage
from MCP.tools.pipeline_tools import _load_semantik_chunkset_for_planning

_RAW_COURSE_NAME = "UNIT_TEST_ALG_XYZ"
_CHUNKS = [
    {"id": "chunk-0001", "text": "The associative property of addition."},
    {"id": "chunk-0002", "text": "The distributive property over addition."},
]


def _build_libv2_root(tmp_path: Path) -> Path:
    """Create ``<root>/courses/<slugified>/semantik_chunks/chunks.jsonl``."""
    slug = LibV2Storage._generate_slug(_RAW_COURSE_NAME)
    assert slug == "unit-test-alg-xyz"  # sanity: the transform under test
    chunks_dir = tmp_path / "courses" / slug / "semantik_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in _CHUNKS) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_raw_course_name_resolves_to_slugified_dir(tmp_path: Path) -> None:
    """A RAW course name must resolve to its slugified LibV2 course dir."""
    root = _build_libv2_root(tmp_path)

    chunks_by_id, all_chunks = _load_semantik_chunkset_for_planning(
        course_slug=_RAW_COURSE_NAME,
        kwargs={"libv2_root": str(root)},
    )

    assert all_chunks, "raw course name should resolve to the slugified dir"
    assert len(all_chunks) == len(_CHUNKS)
    assert set(chunks_by_id) == {"chunk-0001", "chunk-0002"}


def test_already_slugified_name_still_resolves(tmp_path: Path) -> None:
    """A caller that already passes the correct slug keeps resolving (no-reg)."""
    root = _build_libv2_root(tmp_path)
    slug = LibV2Storage._generate_slug(_RAW_COURSE_NAME)

    chunks_by_id, all_chunks = _load_semantik_chunkset_for_planning(
        course_slug=slug,
        kwargs={"libv2_root": str(root)},
    )

    assert len(all_chunks) == len(_CHUNKS)
    assert set(chunks_by_id) == {"chunk-0001", "chunk-0002"}


def test_missing_chunkset_degrades_to_empty(tmp_path: Path) -> None:
    """No chunkset on disk → ``({}, [])`` (chapter_fallback), never a crash."""
    (tmp_path / "courses").mkdir()

    chunks_by_id, all_chunks = _load_semantik_chunkset_for_planning(
        course_slug=_RAW_COURSE_NAME,
        kwargs={"libv2_root": str(tmp_path)},
    )

    assert chunks_by_id == {}
    assert all_chunks == []
