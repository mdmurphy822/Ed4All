"""IB7.8 / IB7.1 / IB7.2 — consumer-side planner wiring in pipeline_tools.

Focused units (NO full outline-phase run):
  * IB7.1 — the per-chunk provenance detail map surfaces a non-empty
    ``heading`` (the field the consumer threads into ``_week_chunks``).
  * IB7.2 — ``_build_block_planner_provider`` honors the
    ``ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER`` seat env var (default ``nvidia``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from MCP.tools import pipeline_tools  # noqa: E402


# ---- IB7.1 — heading surfaces from the provenance detail map -------------- #
def test_detail_map_surfaces_heading(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    line = {
        "id": "c1",
        "text": "Compare proper and improper fractions across definitions.",
        "source": {
            "item_path": "module1/page1.html",
            "section_heading": "Types of Fractions",
        },
        "concept_tags": ["fraction"],
    }
    chunks_path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    detail = pipeline_tools._load_dart_chunkset_detail_map(chunks_path)
    assert "c1" in detail
    assert detail["c1"]["heading"] == "Types of Fractions"
    assert detail["c1"]["text"].startswith("Compare proper")


def test_detail_map_heading_falls_back_to_empty(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    line = {"id": "c2", "text": "Body with no heading provenance."}
    chunks_path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    detail = pipeline_tools._load_dart_chunkset_detail_map(chunks_path)
    assert detail["c2"]["heading"] == ""


# ---- IB7.2 — operator-selectable planner seat ---------------------------- #
def test_planner_seat_defaults_to_nvidia(monkeypatch):
    monkeypatch.delenv("ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER", raising=False)
    seen = {}

    def _fake_build(*, provider, capture=None):  # noqa: ARG001
        seen["provider"] = provider
        return object()

    monkeypatch.setattr(
        "lib.generation.block_planner.build_planner_provider", _fake_build
    )
    pipeline_tools._build_block_planner_provider(capture=None)
    assert seen["provider"] == "nvidia"


def test_planner_seat_honors_local_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER", "local")
    seen = {}

    def _fake_build(*, provider, capture=None):  # noqa: ARG001
        seen["provider"] = provider
        return object()

    monkeypatch.setattr(
        "lib.generation.block_planner.build_planner_provider", _fake_build
    )
    pipeline_tools._build_block_planner_provider(capture=None)
    assert seen["provider"] == "local"
