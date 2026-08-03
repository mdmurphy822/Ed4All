"""Bloom-ladder initiative — WI-09 — Block dataclass fields + renderer emit.

Covers the two hash-excluded ``Block`` fields the WI-06 planner pass
(``lib/generation/block_planner.py::_apply_bloom_ladder_selection``) stamps on
an injected rung-representative block, and their gated projection:

* ``Block.ladder_rung`` / ``Block.mc_bloom_rung`` — unconditionally
  constructible (a ``Block`` always builds, regardless of
  ``ED4ALL_BLOOM_LADDER``); default ``None``.
* ``compute_content_hash()`` — the two fields are OUTSIDE the explicit 5-key
  allowlist, so populating them never drifts an existing block's hash.
* ``to_html_attrs()`` — an ADDITIVE ``data-cf-bloom-level`` wrapper attr,
  gated behind ``ED4ALL_BLOOM_LADDER`` AND only-when-``ladder_rung``-set;
  suppressed for the three block types (``objective`` /
  ``self_check_question`` / ``activity``) that already emit
  ``data-cf-bloom-level`` unconditionally from ``bloom_level`` (no duplicate
  attribute). A ``misconception`` card — which emits NO
  ``data-cf-bloom-level`` from its own type dispatch — gains exactly this one
  additive attribute; no new class token.
* ``to_jsonld_entry()`` — additive ``bloomLadderRung`` / ``mcBloomRung`` keys,
  gated + only-when-set, appended after dispatch so every shape (legacy
  ``misconception`` / ``objective`` / section, or the Phase-2-minimal
  default) projects them the same way.
* ``generate_course.py::_bloom_ladder_kwargs`` — the render-dispatch arm that
  threads the WI-06 planner's per-block descriptor extras onto the new
  ``Block`` fields at construction time, at the misconception /
  self-check-question / activity builder sites (mirrors the existing
  ``mc_named_concept`` / ``confidence_prompt`` threading).

All flag-off paths must be byte-identical to pre-WI-09 behavior.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402

import generate_course  # noqa: E402
from generate_course import (  # noqa: E402
    _bloom_ladder_kwargs,
    _build_activity_blocks,
    _build_misconception_blocks,
    _build_self_check_blocks,
)
from lib.generation.bloom_ladder_blocks import ENV_BLOOM_LADDER  # noqa: E402


def _block(block_type: str, **kw) -> Block:
    return Block(
        block_id=f"page_01#{block_type}_demo_0",
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=kw.pop("content", "<p>x</p>"),
        **kw,
    )


# ---------------------------------------------------------------------- #
# 1. Field defaults + hash exclusion (the keystone byte-stability assertion)
# ---------------------------------------------------------------------- #


def test_default_ladder_fields_are_none() -> None:
    b = _block("concept")
    assert b.ladder_rung is None
    assert b.mc_bloom_rung is None


def test_ladder_fields_unconditionally_constructible() -> None:
    """A Block always constructs with the new fields, regardless of the flag —
    ``ED4ALL_BLOOM_LADDER`` gates EMIT, never field assignment."""
    b = _block("misconception", ladder_rung="analyze", mc_bloom_rung="analyze")
    assert b.ladder_rung == "analyze"
    assert b.mc_bloom_rung == "analyze"


def test_content_hash_byte_identical_with_ladder_fields(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    base = _block("misconception", content={"misconception": "m", "correction": "c"})
    enriched = _block(
        "misconception",
        content={"misconception": "m", "correction": "c"},
        ladder_rung="analyze",
        mc_bloom_rung="analyze",
    )
    assert base.compute_content_hash() == enriched.compute_content_hash()


def test_content_hash_byte_identical_across_every_ladder_rung(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    base = _block("activity", content={"title": "t", "description": "d"})
    base_hash = base.compute_content_hash()
    for rung in ("remember", "understand", "apply", "analyze", "evaluate", "create"):
        variant = _block(
            "activity",
            content={"title": "t", "description": "d"},
            ladder_rung=rung,
        )
        assert variant.compute_content_hash() == base_hash


# ---------------------------------------------------------------------- #
# 2. to_html_attrs() — flag-off byte-identical, flag-on additive
# ---------------------------------------------------------------------- #


def test_flag_off_html_emit_byte_identical(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    legacy = _block(
        "misconception", content={"misconception": "m", "correction": "c"}
    )
    enriched = _block(
        "misconception",
        content={"misconception": "m", "correction": "c"},
        ladder_rung="analyze",
        mc_bloom_rung="analyze",
    )
    assert legacy.to_html_attrs() == enriched.to_html_attrs()


def test_flag_on_no_ladder_rung_byte_identical(monkeypatch) -> None:
    """Flag on but ``ladder_rung`` unset — the only-when-set guard must still
    hold (mirrors the UDL / quality-rubric tail-append posture)."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    legacy = _block("callout")
    enriched = _block("callout", mc_bloom_rung="analyze")  # ladder_rung unset
    assert legacy.to_html_attrs() == enriched.to_html_attrs()


def test_flag_on_misconception_gains_bloom_level_attr(monkeypatch) -> None:
    """The design-decision case: a misconception card's ONLY additive
    wrapper attr from the ladder is ``data-cf-bloom-level`` — no new class
    token (class tokens are rendered by generate_course.py, not
    to_html_attrs; this asserts the attr-string side of that contract)."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    off = _block("misconception", content={"misconception": "m", "correction": "c"})
    on = _block(
        "misconception",
        content={"misconception": "m", "correction": "c"},
        ladder_rung="analyze",
    )
    assert off.to_html_attrs() == ""
    assert on.to_html_attrs() == ' data-cf-bloom-level="analyze"'


def test_flag_on_callout_gains_bloom_level_attr(monkeypatch) -> None:
    """A non-misconception, non-self-check/-activity/-objective type
    (``callout``, which emits no ``data-cf-bloom-level`` today) also gains
    the additive attr — the ladder marker is not misconception-exclusive."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    b = _block("callout", ladder_rung="understand")
    assert 'data-cf-bloom-level="understand"' in b.to_html_attrs()


def test_flag_on_no_duplicate_attr_for_self_check_question(monkeypatch) -> None:
    """``self_check_question`` already stamps ``data-cf-bloom-level`` from
    ``bloom_level`` unconditionally — the ladder's own value must not
    duplicate the attribute on the same wrapper."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    b = _block(
        "self_check_question",
        content={"question": "q", "options": []},
        bloom_level="apply",
        ladder_rung="understand",  # deliberately different from bloom_level
    )
    attrs = b.to_html_attrs()
    assert attrs.count("data-cf-bloom-level") == 1
    assert 'data-cf-bloom-level="apply"' in attrs
    assert "understand" not in attrs


def test_flag_on_no_duplicate_attr_for_activity(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    b = _block(
        "activity",
        content={"title": "t", "description": "d"},
        bloom_level="apply",
        ladder_rung="create",
    )
    attrs = b.to_html_attrs()
    assert attrs.count("data-cf-bloom-level") == 1
    assert 'data-cf-bloom-level="apply"' in attrs


def test_flag_on_no_duplicate_attr_for_objective(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    b = _block(
        "objective",
        content="statement",
        objective_ids=("CO-01",),
        bloom_level="remember",
        ladder_rung="remember",
    )
    attrs = b.to_html_attrs()
    assert attrs.count("data-cf-bloom-level") == 1


# ---------------------------------------------------------------------- #
# 3. to_jsonld_entry() — flag-off byte-identical, flag-on additive
# ---------------------------------------------------------------------- #


def test_flag_off_jsonld_byte_identical_misconception(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    legacy = _block("misconception", content={"misconception": "m", "correction": "c"})
    enriched = _block(
        "misconception",
        content={"misconception": "m", "correction": "c"},
        ladder_rung="analyze",
        mc_bloom_rung="analyze",
    )
    assert legacy.to_jsonld_entry() == enriched.to_jsonld_entry()


def test_flag_off_jsonld_byte_identical_minimal_shape(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    legacy = _block("flip_card_grid")
    enriched = _block("flip_card_grid", ladder_rung="remember")
    assert legacy.to_jsonld_entry() == enriched.to_jsonld_entry()


def test_flag_on_jsonld_only_when_set(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    b = _block("misconception", content={"misconception": "m", "correction": "c"})
    entry = b.to_jsonld_entry()
    assert "bloomLadderRung" not in entry
    assert "mcBloomRung" not in entry


def test_flag_on_jsonld_adds_ladder_rung_misconception(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    b = _block(
        "misconception",
        content={"misconception": "m", "correction": "c"},
        ladder_rung="analyze",
        mc_bloom_rung="analyze",
    )
    entry = b.to_jsonld_entry()
    assert entry["bloomLadderRung"] == "analyze"
    assert entry["mcBloomRung"] == "analyze"


def test_flag_on_jsonld_adds_ladder_rung_minimal_shape(monkeypatch) -> None:
    """The universal post-dispatch append covers the Phase-2-minimal shape
    too (not just the legacy misconception shape)."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    b = _block("flip_card_grid", ladder_rung="remember")
    entry = b.to_jsonld_entry()
    assert entry["bloomLadderRung"] == "remember"
    assert "mcBloomRung" not in entry


# ---------------------------------------------------------------------- #
# 4. generate_course._bloom_ladder_kwargs — the render-dispatch arm
# ---------------------------------------------------------------------- #


def test_kwargs_flag_off_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    d = {"ladder_rung": "apply", "mc_bloom_rung": "apply"}
    assert _bloom_ladder_kwargs(d) == {}


def test_kwargs_flag_on_extracts_both(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    d = {"ladder_rung": " apply ", "mc_bloom_rung": " apply "}
    assert _bloom_ladder_kwargs(d) == {
        "ladder_rung": "apply",
        "mc_bloom_rung": "apply",
    }


def test_kwargs_flag_on_ignores_absent_empty_or_non_string(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    assert _bloom_ladder_kwargs({}) == {}
    assert _bloom_ladder_kwargs({"ladder_rung": ""}) == {}
    assert _bloom_ladder_kwargs({"ladder_rung": "   "}) == {}
    assert _bloom_ladder_kwargs({"ladder_rung": 42}) == {}
    assert _bloom_ladder_kwargs({"ladder_rung": None}) == {}


def test_kwargs_flag_on_ladder_rung_only(monkeypatch) -> None:
    """A non-misconception rung representative carries ``ladder_rung`` with
    no ``mc_bloom_rung`` (that key is misconception-only per the WI-06
    planner)."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    assert _bloom_ladder_kwargs({"ladder_rung": "create"}) == {"ladder_rung": "create"}


# ---------------------------------------------------------------------- #
# 5. Block-builder threading — misconception / activity / self-check
# ---------------------------------------------------------------------- #


def test_build_misconception_blocks_threads_ladder_rung_when_flag_on(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    misconceptions = [
        {
            "misconception": "Learners often skip the normalization step.",
            "correction": "Apply normalization before aggregation.",
            "ladder_rung": "analyze",
            "mc_bloom_rung": "analyze",
        }
    ]
    blocks = _build_misconception_blocks(misconceptions, page_id="week_01_content_01")
    assert blocks[0].ladder_rung == "analyze"
    assert blocks[0].mc_bloom_rung == "analyze"


def test_build_misconception_blocks_ignores_ladder_rung_when_flag_off(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    misconceptions = [
        {
            "misconception": "m",
            "correction": "c",
            "ladder_rung": "analyze",
            "mc_bloom_rung": "analyze",
        }
    ]
    blocks = _build_misconception_blocks(misconceptions, page_id="week_01_content_01")
    assert blocks[0].ladder_rung is None
    assert blocks[0].mc_bloom_rung is None


def test_build_activity_blocks_threads_ladder_rung_when_flag_on(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    activities = [
        {
            "title": "Practice set",
            "description": "Work the problems.",
            "bloom_level": "apply",
            "ladder_rung": "apply",
        }
    ]
    blocks = _build_activity_blocks(activities, page_id="week_01_content_01")
    assert blocks[0].ladder_rung == "apply"
    assert blocks[0].mc_bloom_rung is None


def test_build_self_check_blocks_threads_ladder_rung_when_flag_on(monkeypatch) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    questions = [
        {
            "question": "Which statement is true?",
            "options": [{"text": "A", "correct": True, "feedback": ""}],
            "bloom_level": "understand",
            "ladder_rung": "understand",
        }
    ]
    blocks = _build_self_check_blocks(questions, page_id="week_01_content_01")
    assert blocks[0].ladder_rung == "understand"


# ---------------------------------------------------------------------- #
# 6. End-to-end — generated page JSON-LD carries the ladder rung
# ---------------------------------------------------------------------- #


def _week_data_with_laddered_misconception() -> dict:
    return {
        "week_number": 1,
        "title": "Bloom-ladder smoke",
        "objectives": [
            {
                "id": "CO-01",
                "statement": "Analyze the framework correctly.",
                "bloom_level": "analyze",
            },
        ],
        "overview_text": ["Intro."],
        "readings": ["Ch. 1"],
        "content_modules": [
            {
                "title": "M",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["body."],
                    }
                ],
                "misconceptions": [
                    {
                        "misconception": "Learners often skip the normalization step.",
                        "correction": "Apply normalization before aggregation.",
                        "ladder_rung": "analyze",
                        "mc_bloom_rung": "analyze",
                    }
                ],
            }
        ],
        "activities": [],
        "key_takeaways": ["k"],
        "reflection_questions": ["q"],
    }


def _jsonld_blocks_from_html(html: str) -> list:
    blobs = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        html,
        flags=re.DOTALL,
    )
    return [json.loads(b) for b in blobs]


def test_generated_page_jsonld_carries_ladder_rung_when_flag_on(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    out = tmp_path / "out"
    generate_course.generate_week(
        _week_data_with_laddered_misconception(), out, "TST_915",
        source_module_map=None,
    )
    content_html_path = next((out / "week_01").glob("week_01_content_*.html"))
    html = content_html_path.read_text(encoding="utf-8")
    parsed = _jsonld_blocks_from_html(html)
    with_miscs = [p for p in parsed if p.get("misconceptions")]
    assert with_miscs, "No JSON-LD block carried misconceptions"
    m = with_miscs[0]["misconceptions"][0]
    assert m["bloomLadderRung"] == "analyze"
    assert m["mcBloomRung"] == "analyze"


def test_generated_page_jsonld_omits_ladder_rung_when_flag_off(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    out = tmp_path / "out"
    generate_course.generate_week(
        _week_data_with_laddered_misconception(), out, "TST_915",
        source_module_map=None,
    )
    content_html_path = next((out / "week_01").glob("week_01_content_*.html"))
    html = content_html_path.read_text(encoding="utf-8")
    parsed = _jsonld_blocks_from_html(html)
    with_miscs = [p for p in parsed if p.get("misconceptions")]
    assert with_miscs, "No JSON-LD block carried misconceptions"
    m = with_miscs[0]["misconceptions"][0]
    assert "bloomLadderRung" not in m
    assert "mcBloomRung" not in m
