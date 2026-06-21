"""Issue I6 — deterministic palette-v2 injection (Part A).

CPU-pinned, no-GPU, no-model tests proving the I6 block types (``table`` /
``acronym`` / ``key_idea``) deterministically deploy by CONTENT SHAPE when
``ED4ALL_DYNAMIC_BLOCK_PLAN`` is on — independent of the 70B choosing them —
and that the legacy / dynamic-off path is a strict NO-OP (byte-stable).

These tests drive the planner via the FALLBACK path (a raising provider) AND
the LLM-success path (a canned-plan provider), so NO live model is needed.
"""

from __future__ import annotations

import json

import pytest

from lib.generation.block_planner import (
    plan_week_blocks,
)


# --------------------------------------------------------------------------- #
# Mock providers (no live 70B).
# --------------------------------------------------------------------------- #
class _RaisingProvider:
    """Forces the deterministic fallback path (LLM dispatch error)."""

    _model = "meta/llama-3.3-70b-instruct"

    def plan_blocks(self, prompt):  # noqa: ARG002
        raise RuntimeError("simulated 70B dispatch failure")


class _CannedProvider:
    """Returns a fixed JSON plan via ``plan_blocks`` (LLM-success path)."""

    _model = "meta/llama-3.3-70b-instruct"

    def __init__(self, blocks):
        self._payload = json.dumps({"blocks": blocks})

    def plan_blocks(self, prompt):  # noqa: ARG002
        return self._payload


_COS = [
    {"id": "CO-01", "statement": "apply the order of operations",
     "bloom_level": "apply"},
    {"id": "CO-02", "statement": "compare fraction types",
     "bloom_level": "understand"},
]
_TO = {"id": "TO-01", "statement": "Evaluate arithmetic expressions correctly"}


def _content_types(plan):
    return [bt for bt, *_ in plan.page_block_plan_for("content")]


# --------------------------------------------------------------------------- #
# (a) tabular source → a `table` block is injected on the content page.
# --------------------------------------------------------------------------- #
def test_table_injected_on_tabular_source(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    src = [
        {"id": "c1", "text": "Compare proper, improper, and mixed fractions "
                             "across their numerator and denominator.",
         "heading": "Types of fractions"},
    ]
    # Fallback path (raising provider) — the deterministic injection still fires.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=src,
        provider=_RaisingProvider(),
    )
    assert plan.fallback_used is True
    assert "table" in _content_types(plan)


def test_table_injected_on_llm_success_path(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    src = [
        {"id": "c1", "text": "There are three types of triangles; compare them "
                             "across their sides and angles.", "heading": ""},
    ]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=src,
        provider=_CannedProvider([
            {"block_type": "concept", "target_co_ids": ["CO-01"],
             "page_type": "content", "content_focus": "x"},
            {"block_type": "example", "target_co_ids": ["CO-02"],
             "page_type": "content", "content_focus": "y"},
        ]),
    )
    assert plan.fallback_used is False
    assert "table" in _content_types(plan)


# --------------------------------------------------------------------------- #
# (b) PEMDAS expansion (no literal token) → an `acronym` block is injected.
# --------------------------------------------------------------------------- #
def test_acronym_injected_on_pemdas_expansion(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    src = [
        {"id": "c1", "text": (
            "Simplify inside parentheses first, then exponents, then "
            "multiplication and division, and finally addition and "
            "subtraction."
        ), "heading": "Order of operations"},
    ]
    # The source NEVER writes the literal token "PEMDAS" — the canonical
    # mnemonic detector still fires the acronym injection.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=src,
        provider=_RaisingProvider(),
    )
    assert "acronym" in _content_types(plan)


# --------------------------------------------------------------------------- #
# (c) exactly ONE key_idea on the content page per TO (inject or promote).
# --------------------------------------------------------------------------- #
def test_exactly_one_key_idea_injected(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=[{"id": "c1", "text": "Plain prose teaches one idea.",
                        "heading": ""}],
        provider=_RaisingProvider(),
    )
    key_ideas = [bt for bt in _content_types(plan) if bt == "key_idea"]
    assert len(key_ideas) == 1


def test_key_idea_promotes_existing_callout(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=[{"id": "c1", "text": "Plain prose.", "heading": ""}],
        provider=_CannedProvider([
            {"block_type": "callout", "target_co_ids": ["CO-01"],
             "page_type": "content", "content_focus": "an important note"},
            {"block_type": "example", "target_co_ids": ["CO-02"],
             "page_type": "content", "content_focus": "worked"},
        ]),
    )
    content = _content_types(plan)
    # Exactly one key_idea, and it came from PROMOTING the callout (so no
    # callout remains on the content page, and the count is still bounded to 1).
    assert content.count("key_idea") == 1


# --------------------------------------------------------------------------- #
# Bounded: at most ONE of each type even if already partly present.
# --------------------------------------------------------------------------- #
def test_no_duplicate_when_type_already_present(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    src = [{"id": "c1", "text": "Compare types of fractions across columns.",
            "heading": ""}]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=src,
        provider=_CannedProvider([
            {"block_type": "table", "target_co_ids": ["CO-01"],
             "page_type": "content", "content_focus": "already a table"},
            {"block_type": "key_idea", "target_co_ids": ["CO-02"],
             "page_type": "content", "content_focus": "already a key idea"},
        ]),
    )
    content = _content_types(plan)
    assert content.count("table") == 1
    assert content.count("key_idea") == 1


# --------------------------------------------------------------------------- #
# Legacy / dynamic-off path is a strict NO-OP (byte-stable).
# --------------------------------------------------------------------------- #
def test_off_path_no_injection_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_DYNAMIC_BLOCK_PLAN", raising=False)
    src = [{"id": "c1", "text": "Compare proper and improper fractions; "
                                "parentheses exponents multiplication division "
                                "addition subtraction.", "heading": ""}]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=src,
        provider=_RaisingProvider(),
    )
    assert plan.fallback_used is True
    content = _content_types(plan)
    # None of the I6 types appear, and the fallback's content_co_ids stay empty
    # (byte-identical to the historic fixed-plan fallback).
    assert "table" not in content
    assert "acronym" not in content
    assert "key_idea" not in content
    for page_type in ("overview", "content", "application", "self_check",
                      "summary"):
        for entry in plan.page_block_plan_for(page_type):
            assert entry[2] == []


def test_off_path_no_injection_llm_success(monkeypatch):
    monkeypatch.delenv("ED4ALL_DYNAMIC_BLOCK_PLAN", raising=False)
    src = [{"id": "c1", "text": "Compare types of fractions across columns.",
            "heading": ""}]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=src,
        provider=_CannedProvider([
            {"block_type": "concept", "target_co_ids": ["CO-01"],
             "page_type": "content", "content_focus": "x"},
            {"block_type": "example", "target_co_ids": ["CO-02"],
             "page_type": "content", "content_focus": "y"},
        ]),
    )
    content = _content_types(plan)
    assert "table" not in content
    assert "acronym" not in content
    assert "key_idea" not in content


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
