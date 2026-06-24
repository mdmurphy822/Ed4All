"""C3-3 — B01 objective-block constructive-alignment enrichment.

Covers the new ``self_rating_prompt`` + ``objective_assessment_thread`` Block
fields (hash-exclusion + default-None) AND the ``_render_objectives`` render
surface: the visible Bloom chip + confidence slot + objective↔assessment
``<details>`` thread are present when ``ED4ALL_BLOCK_ANATOMY`` is on and absent
(byte-stable) when off.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402
from generate_course import _render_objectives  # noqa: E402


def _block(**kw):
    base = dict(
        block_id="p#objective_x_0", block_type="objective", page_id="p",
        sequence=0, content="Solve linear equations.",
    )
    base.update(kw)
    return Block(**base)


# --- field defaults + hash exclusion ---------------------------------------


def test_enrichment_fields_default_none():
    b = _block()
    assert b.self_rating_prompt is None
    assert b.objective_assessment_thread is None


def test_self_rating_prompt_excluded_from_hash():
    h = _block().compute_content_hash()
    assert _block(self_rating_prompt="rate yourself").compute_content_hash() == h


def test_objective_assessment_thread_excluded_from_hash():
    h = _block().compute_content_hash()
    other = _block(
        objective_assessment_thread={"prompt": "p", "assessment_refs": ["assessment_item"]}
    )
    assert other.compute_content_hash() == h


# --- render: byte-stable OFF -----------------------------------------------

_OBJS = [
    {
        "id": "CO-01",
        "statement": "Solve linear equations.",
        "bloom_level": "apply",
        "bloom_verb": "solve",
        "self_rating_prompt": "Rate your confidence.",
        "objective_assessment_thread": {
            "prompt": "How this is assessed",
            "assessment_refs": ["assessment_item", "self_check_question"],
        },
    }
]


def test_render_byte_stable_when_flag_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_BLOCK_ANATOMY", raising=False)
    html = _render_objectives(_OBJS, page_id="week_01")
    assert "objective-enrichment" not in html
    assert "objective-self-rating" not in html
    assert "objective-assessment-thread" not in html
    assert "objective-bloom-chip" not in html


def test_render_emits_enrichment_when_flag_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    html = _render_objectives(_OBJS, page_id="week_01")
    assert "objective-enrichment" in html
    # Confidence slot.
    assert "objective-self-rating" in html
    assert "Rate your confidence." in html
    # Objective↔assessment thread.
    assert "objective-assessment-thread" in html
    assert "assessment_item" in html
    assert "self_check_question" in html
    # Bloom chip (rendered from the resolved triple — verb · level · ktype).
    assert "objective-bloom-chip" in html
    assert "data-cf-bloom-triple=" in html


def test_render_chip_only_when_no_thread_data(monkeypatch):
    """Flag ON but no confidence/thread data: only the chip renders."""
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    objs = [
        {
            "id": "CO-02",
            "statement": "Define a variable.",
            "bloom_level": "remember",
            "bloom_verb": "define",
        }
    ]
    html = _render_objectives(objs, page_id="week_01")
    assert "objective-bloom-chip" in html
    assert "objective-self-rating" not in html
    assert "objective-assessment-thread" not in html
