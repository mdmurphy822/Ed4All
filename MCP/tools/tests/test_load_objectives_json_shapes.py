"""Wave2c regression tests for load_objectives_json shape-handling.

Wave2b's smoke verification noted that ``_content_gen_helpers.load_objectives_json``
silently dropped every CO-NN id when ``chapter_objectives`` arrived as a
dict-of-lists (the vendor dict-of-lists shape ``{"1": [...], "2": [...]}``) — the
upstream loader iterated the dict-keys rather than the objectives.

Wave2c fixes the loader to delegate to the shared
``_normalize_chapter_objectives_to_groups`` normalizer (Wave2b,
``MCP/tools/pipeline_tools.py``) so all three supported shapes are handled
identically by the loader and the pipeline projection helper.

Tests
-----
1. dict-of-lists shape  → all CO-NN ids returned, chapter order deterministic
2. list-of-groups shape → regression: existing shape stays green
3. flat-list shape      → regression: existing shape stays green
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from MCP.tools._content_gen_helpers import load_objectives_json


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_objectives(tmp_dir: Path, payload: dict) -> str:
    p = tmp_dir / "synthesized_objectives.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1. dict-of-lists shape (Wave2c regression target)
# ---------------------------------------------------------------------------

class TestLoadObjectivesDictOfListsShape:
    """dict-of-lists chapter_objectives must surface all CO-NN entries."""

    def test_all_co_ids_are_returned(self, tmp_path: Path) -> None:
        payload = {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Terminal one."},
            ],
            "chapter_objectives": {
                "1": [
                    {"id": "CO-01", "statement": "Chapter one A."},
                    {"id": "CO-02", "statement": "Chapter one B."},
                ],
                "2": [
                    {"id": "CO-03", "statement": "Chapter two A."},
                ],
            },
        }
        path = _write_objectives(tmp_path, payload)
        terminals, chapters = load_objectives_json(path)

        terminal_ids = [e["id"] for e in terminals]
        chapter_ids = [e["id"] for e in chapters]

        assert terminal_ids == ["TO-01"], f"terminal ids wrong: {terminal_ids!r}"
        # Normalizer sorts dict keys as strings → "1" < "2".
        assert chapter_ids == ["CO-01", "CO-02", "CO-03"], (
            f"dict-of-lists dropped CO ids: {chapter_ids!r}"
        )

    def test_chapter_order_is_deterministic_by_sorted_key(self, tmp_path: Path) -> None:
        # Keys "10" < "2" lexicographically; normalizer sorts by str so "10" < "2".
        payload = {
            "terminal_objectives": [],
            "chapter_objectives": {
                "2": [{"id": "CO-02", "statement": "S2."}],
                "10": [{"id": "CO-10", "statement": "S10."}],
                "1": [{"id": "CO-01", "statement": "S1."}],
            },
        }
        path = _write_objectives(tmp_path, payload)
        _, chapters = load_objectives_json(path)
        chapter_ids = [e["id"] for e in chapters]
        # Sorted string key order: "1" < "10" < "2"
        assert chapter_ids == ["CO-01", "CO-10", "CO-02"], (
            f"key sort order wrong: {chapter_ids!r}"
        )


# ---------------------------------------------------------------------------
# 2. list-of-groups shape (regression — must stay green)
# ---------------------------------------------------------------------------

class TestLoadObjectivesListOfGroupsShape:
    """Existing Courseforge synthesized list-of-groups shape is unchanged."""

    def test_list_of_groups_returns_all_objectives(self, tmp_path: Path) -> None:
        payload = {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "T1."},
                {"id": "TO-02", "statement": "T2."},
            ],
            "chapter_objectives": [
                {
                    "chapter": "1",
                    "objectives": [
                        {"id": "CO-01", "statement": "C1."},
                        {"id": "CO-02", "statement": "C2."},
                    ],
                },
                {
                    "chapter": "2",
                    "objectives": [
                        {"id": "CO-03", "statement": "C3."},
                    ],
                },
            ],
        }
        path = _write_objectives(tmp_path, payload)
        terminals, chapters = load_objectives_json(path)

        assert [e["id"] for e in terminals] == ["TO-01", "TO-02"]
        assert [e["id"] for e in chapters] == ["CO-01", "CO-02", "CO-03"]


# ---------------------------------------------------------------------------
# 3. flat-list shape (regression — must stay green)
# ---------------------------------------------------------------------------

class TestLoadObjectivesFlatListShape:
    """Flat-list chapter_objectives (LibV2 archive component_objectives shape)."""

    def test_flat_list_returns_all_objectives(self, tmp_path: Path) -> None:
        payload = {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "T1."},
            ],
            "chapter_objectives": [
                {"id": "CO-01", "statement": "Flat one."},
                {"id": "CO-02", "statement": "Flat two."},
                {"id": "CO-03", "statement": "Flat three."},
            ],
        }
        path = _write_objectives(tmp_path, payload)
        terminals, chapters = load_objectives_json(path)

        assert [e["id"] for e in terminals] == ["TO-01"]
        assert [e["id"] for e in chapters] == ["CO-01", "CO-02", "CO-03"]
