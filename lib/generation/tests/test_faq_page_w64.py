"""W6.4 FAQ-page builder tests — deterministic, grounded, anti-fabrication.

All assertions are pure-lexical / pure-deterministic — no model, no GPU.
"""

from __future__ import annotations

import pytest

from lib.generation import faq_page as fp


# --------------------------------------------------------------------------- #
# Fixtures — synthetic in-memory chunks / objectives / misconception blocks.
# --------------------------------------------------------------------------- #

_CHUNKS = [
    {
        "id": "c1",
        "text": "A fraction is a number that represents a part of a whole. "
        "It has a numerator and a denominator.",
        "heading": "Fractions",
        "item_path": "ch1/frac.html",
    },
    {
        "id": "c2",
        "text": "TRY IT :: 1.35 Evaluate the expression 8x - 3 when x equals 2.",
        "heading": "Exercises",
        "item_path": "ch1/ex.html",
    },
    {
        "id": "c3",
        "text": "Dividing by zero is undefined because no number multiplied by "
        "zero yields a nonzero result.",
        "heading": "Division",
        "item_path": "ch1/div.html",
    },
]

_OBJECTIVES = [
    {"id": "CO-01", "statement": "Understand fractions", "keyConcepts": ["fraction"]},
    # ungrounded concept -> omitted
    {"id": "CO-02", "statement": "x", "keyConcepts": ["photosynthesis"]},
]


class _MisBlock:
    def __init__(self, content, block_type="misconception"):
        self.content = content
        self.block_type = block_type


_MISCONCEPTIONS = [
    _MisBlock(
        {
            "misconception": "You can divide by zero",
            "correction": "Dividing by zero is undefined because no number "
            "multiplied by zero yields a nonzero result.",
        }
    ),
    # ungrounded correction -> omitted
    _MisBlock(
        {
            "misconception": "made up thing",
            "correction": "totally unrelated ungrounded text about kangaroos",
        }
    ),
]


# --------------------------------------------------------------------------- #
# resolve_faq_page — parse-with-fallback gate
# --------------------------------------------------------------------------- #

def test_resolve_faq_page_default_off(monkeypatch):
    monkeypatch.delenv(fp.FAQ_PAGE_ENV, raising=False)
    assert fp.resolve_faq_page() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_resolve_faq_page_truthy(monkeypatch, val):
    monkeypatch.setenv(fp.FAQ_PAGE_ENV, val)
    assert fp.resolve_faq_page() is True


@pytest.mark.parametrize("val", ["0", "false", "garbage", ""])
def test_resolve_faq_page_falsey(monkeypatch, val):
    monkeypatch.setenv(fp.FAQ_PAGE_ENV, val)
    assert fp.resolve_faq_page() is False


def test_resolve_faq_page_explicit_override_wins(monkeypatch):
    monkeypatch.setenv(fp.FAQ_PAGE_ENV, "1")
    assert fp.resolve_faq_page(False) is False
    monkeypatch.delenv(fp.FAQ_PAGE_ENV, raising=False)
    assert fp.resolve_faq_page(True) is True


# --------------------------------------------------------------------------- #
# build_faq_blocks — flag gate (byte-identical when off)
# --------------------------------------------------------------------------- #

def test_build_blocks_empty_when_off(monkeypatch):
    monkeypatch.delenv(fp.FAQ_PAGE_ENV, raising=False)
    assert fp.build_faq_blocks(_OBJECTIVES, _MISCONCEPTIONS, _CHUNKS) == []


def test_build_blocks_empty_when_enabled_false():
    assert fp.build_faq_blocks(
        _OBJECTIVES, _MISCONCEPTIONS, _CHUNKS, enabled=False
    ) == []


# --------------------------------------------------------------------------- #
# build_faq_entries — grounding + anti-fabrication
# --------------------------------------------------------------------------- #

def test_entries_seed_from_both_sources():
    entries = fp.build_faq_entries(
        _OBJECTIVES, _MISCONCEPTIONS, _CHUNKS, course_slug="algebra-101"
    )
    seeds = {e["seed"] for e in entries}
    assert "misconception" in seeds
    assert "objective" in seeds
    # misconception entries come first.
    assert entries[0]["seed"] == "misconception"


def test_ungrounded_misconception_omitted():
    entries = fp.build_faq_entries([], _MISCONCEPTIONS, _CHUNKS)
    # only the divide-by-zero misconception is grounded; the kangaroo one is out.
    assert len(entries) == 1
    assert "divide by zero" in entries[0]["question"]


def test_ungrounded_objective_concept_omitted():
    entries = fp.build_faq_entries(_OBJECTIVES, [], _CHUNKS)
    # 'fraction' grounds to c1; 'photosynthesis' has no chunk -> omitted.
    questions = [e["question"] for e in entries]
    assert any("fraction" in q.lower() for q in questions)
    assert not any("photosynthesis" in q.lower() for q in questions)


def test_leaky_sentence_not_used_as_answer():
    # 'fraction' appears only in c1 (real def); the exercise chunk c2 (TRY IT)
    # must never be lifted as the answer.
    entries = fp.build_faq_entries(_OBJECTIVES, [], _CHUNKS)
    frac = next(e for e in entries if "fraction" in e["question"].lower())
    assert "TRY IT" not in frac["answer"]
    assert frac["answer"].startswith("A fraction is")


def test_entries_carry_source_link():
    entries = fp.build_faq_entries(
        _OBJECTIVES, _MISCONCEPTIONS, _CHUNKS, course_slug="algebra-101"
    )
    assert all("source_link" in e for e in entries)
    for e in entries:
        assert e["source_link"].startswith("/api/learn/source/algebra-101")


def test_no_source_link_without_course_slug():
    entries = fp.build_faq_entries(_OBJECTIVES, _MISCONCEPTIONS, _CHUNKS)
    # resolve_source_link returns None without a slug.
    assert all("source_link" not in e for e in entries)


def test_max_cap_respected(monkeypatch):
    monkeypatch.setenv(fp.FAQ_MAX_PER_PAGE_ENV, "1")
    entries = fp.build_faq_entries(_OBJECTIVES, _MISCONCEPTIONS, _CHUNKS)
    assert len(entries) == 1


def test_deterministic_repeat():
    a = fp.build_faq_entries(_OBJECTIVES, _MISCONCEPTIONS, _CHUNKS, course_slug="s")
    b = fp.build_faq_entries(_OBJECTIVES, _MISCONCEPTIONS, _CHUNKS, course_slug="s")
    assert [e["question"] for e in a] == [e["question"] for e in b]
    assert [e["answer"] for e in a] == [e["answer"] for e in b]


# --------------------------------------------------------------------------- #
# build_faq_blocks — Block projection
# --------------------------------------------------------------------------- #

def test_blocks_projection():
    blocks = fp.build_faq_blocks(
        _OBJECTIVES,
        _MISCONCEPTIONS,
        _CHUNKS,
        enabled=True,
        course_slug="algebra-101",
        page_id="week_01_faq",
    )
    assert len(blocks) == 2
    for b in blocks:
        assert b.block_type == "vocab_card"
        assert b.template_type == fp.FAQ_TEMPLATE_TYPE
        assert b.page_id == "week_01_faq"
        assert 'data-cf-content-type="faq"' in b.content
        # each block is grounded to a real chunk source id.
        assert b.source_ids and b.source_ids[0].startswith("dart:algebra-101#")


def test_objective_block_carries_objective_id():
    blocks = fp.build_faq_blocks(
        _OBJECTIVES, [], _CHUNKS, enabled=True, course_slug="s", page_id="p"
    )
    obj_block = next(b for b in blocks if b.objective_ids)
    assert obj_block.objective_ids == ("CO-01",)


def test_render_faq_card_escapes():
    html = fp.render_faq_card(
        question="What is <x> & y?",
        answer="A <b>bold</b> answer",
        source_link="/api/learn/source/s?item_path=a&fragment=b",
    )
    assert "&lt;x&gt;" in html
    assert "&amp; y" in html
    assert "View source" in html
    assert 'data-cf-content-type="faq"' in html


def test_render_faq_card_no_link():
    html = fp.render_faq_card(question="Q?", answer="A.")
    assert "View source" not in html
