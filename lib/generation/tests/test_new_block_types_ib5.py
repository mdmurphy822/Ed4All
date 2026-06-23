"""IB5 — missing pedagogical block types (B02 hook / B04 multimedia /
B05 worked_example / B06 diagram).

CPU-pinned, no-GPU, no-model unit tests for the four framework-aligned
pedagogical block types: token registration + catalog entry + hash-exclusion
of the new fields + HTML/JSON-LD projection gating + deterministic renderers +
the time-based-media a11y stack contract + planner selection (nudge + byte-
stability with the flag off) + the warning-day-1 validator arms.

The PRIMARY gate is byte-stability: with ED4ALL_NEW_BLOCK_TYPES unset, the four
types are never selected, never rendered, and their type-specific fields are
never projected — every existing snapshot / contentHash stays byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from Courseforge.scripts.blocks import BLOCK_TYPES, Block  # noqa: E402
from lib.generation.block_catalog import load_block_catalog  # noqa: E402
from lib.generation.new_block_types import (  # noqa: E402
    ENV_NEW_BLOCK_TYPES,
    NEW_BLOCK_TYPES,
    resolve_new_block_types,
)
from lib.ontology.framework_blocks import framework_block_for  # noqa: E402

_FRAMEWORK = {"hook": "B02", "multimedia": "B04", "worked_example": "B05", "diagram": "B06"}


# --------------------------------------------------------------------------- #
# IB5.1 — token registration + construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_in_block_types(block_type):
    assert block_type in BLOCK_TYPES


def test_block_types_count_is_twenty_eight():
    # 28 after IB5; the B15 `resources` addition brings the palette to 29.
    assert len(BLOCK_TYPES) == 29


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_constructs(block_type):
    blk = Block(
        block_id=f"week_01_content_01#{block_type}_x_0",
        block_type=block_type,
        page_id="week_01_content_01",
        sequence=0,
        content="body",
    )
    assert blk.block_type == block_type


# --------------------------------------------------------------------------- #
# IB5.2 — catalog entries + framework_block + bloom_fit
# --------------------------------------------------------------------------- #
def test_catalog_covers_new_types():
    catalog_types = {e["block_type"] for e in load_block_catalog()}
    assert set(NEW_BLOCK_TYPES) <= catalog_types
    assert catalog_types == set(BLOCK_TYPES)


@pytest.mark.parametrize("block_type,code", list(_FRAMEWORK.items()))
def test_new_type_framework_block(block_type, code):
    by_type = {e["block_type"]: e for e in load_block_catalog()}
    assert by_type[block_type]["framework_block"] == code
    # SoT resolver agrees.
    assert framework_block_for(block_type) == code


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_bloom_fit_valid(block_type):
    valid = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
    by_type = {e["block_type"]: e for e in load_block_catalog()}
    bloom_fit = by_type[block_type]["bloom_fit"]
    assert bloom_fit and all(b in valid for b in bloom_fit)


# --------------------------------------------------------------------------- #
# IB5.3 — new Block fields are Optional + hash-excluded
# --------------------------------------------------------------------------- #
def test_fade_state_excluded_from_hash():
    base = Block(
        block_id="p#worked_example_x_0", block_type="worked_example",
        page_id="p", sequence=0, content="body",
    )
    with_fade = Block(
        block_id="p#worked_example_x_0", block_type="worked_example",
        page_id="p", sequence=0, content="body", fade_state="worked",
    )
    assert base.compute_content_hash() == with_fade.compute_content_hash()


def test_long_description_and_media_a11y_excluded_from_hash():
    base = Block(
        block_id="p#diagram_x_0", block_type="diagram",
        page_id="p", sequence=0, content="body",
    )
    enriched = Block(
        block_id="p#diagram_x_0", block_type="diagram",
        page_id="p", sequence=0, content="body",
        long_description="a long structured description",
        media_a11y=("captions", "transcript"),
    )
    assert base.compute_content_hash() == enriched.compute_content_hash()


# --------------------------------------------------------------------------- #
# IB5.4 — HTML-attr + JSON-LD projection (flag-gated)
# --------------------------------------------------------------------------- #
def _mm_block():
    return Block(
        block_id="p#multimedia_x_0", block_type="multimedia",
        page_id="p", sequence=0, content="body",
        source_ids=("dart:s#b1",), source_primary="dart:s#b1",
        media_a11y=("captions", "transcript"),
    )


def test_wrapper_only_attrs_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    blk = _mm_block()
    attrs = blk.to_html_attrs()
    # Source-id attrs present; no IB5 fade-state attr (multimedia has none anyway).
    assert "data-cf-source-ids" in attrs
    assert "data-cf-fade-state" not in attrs


def test_worked_example_fade_attr_flag_on(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    blk = Block(
        block_id="p#worked_example_x_0", block_type="worked_example",
        page_id="p", sequence=0, content="body", fade_state="completion",
        source_ids=("dart:s#b1",),
    )
    attrs = blk.to_html_attrs()
    assert 'data-cf-fade-state="completion"' in attrs


def test_worked_example_fade_attr_absent_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    blk = Block(
        block_id="p#worked_example_x_0", block_type="worked_example",
        page_id="p", sequence=0, content="body", fade_state="completion",
        source_ids=("dart:s#b1",),
    )
    assert "data-cf-fade-state" not in blk.to_html_attrs()


def test_jsonld_fields_gated(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    we = Block(
        block_id="p#worked_example_x_0", block_type="worked_example",
        page_id="p", sequence=0, content="body", fade_state="worked",
    )
    dg = Block(
        block_id="p#diagram_x_0", block_type="diagram",
        page_id="p", sequence=0, content="body",
        long_description="structured long-desc",
    )
    assert we.to_jsonld_entry().get("fadeState") == "worked"
    assert dg.to_jsonld_entry().get("longDescription") == "structured long-desc"
    # Flag off → keys absent.
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    assert "fadeState" not in we.to_jsonld_entry()
    assert "longDescription" not in dg.to_jsonld_entry()


# --------------------------------------------------------------------------- #
# IB5.5 — deterministic renderers
# --------------------------------------------------------------------------- #
def test_render_worked_example_section():
    from Courseforge.scripts.generate_course import _render_worked_example_section
    blk = Block(
        block_id="p#worked_example_x_0", block_type="worked_example",
        page_id="p", sequence=0, fade_state="worked",
        content={
            "problem": "Solve 2x + 3 = 11",
            "steps": [
                {"subgoal": "Isolate the term", "body": "subtract 3", "why": "undo addition"},
                {"subgoal": "Solve", "body": "divide by 2", "why": "undo multiplication"},
            ],
        },
    )
    html = _render_worked_example_section(blk)
    assert 'class="worked-example"' in html
    assert 'data-cf-fade-state="worked"' in html
    assert 'class="subgoal-label"' in html
    assert "Why:" in html


def test_render_multimedia_section_no_url_still_has_a11y_skeleton():
    from Courseforge.scripts.generate_course import _render_multimedia_section
    blk = Block(
        block_id="p#multimedia_x_0", block_type="multimedia",
        page_id="p", sequence=0, content={},
    )
    html = _render_multimedia_section(blk)
    assert 'controls' in html
    assert '<track kind="captions"' in html
    assert 'data-cf-transcript' in html
    assert "Audio description" in html


def test_render_diagram_section_has_longdesc_and_table():
    from Courseforge.scripts.generate_course import _render_diagram_section
    blk = Block(
        block_id="p#diagram_x_0", block_type="diagram",
        page_id="p", sequence=0, long_description="the flow goes A->B->C",
        content={"caption": "Process flow", "headers": ["Node", "Next"],
                 "rows": [["A", "B"], ["B", "C"]]},
    )
    html = _render_diagram_section(blk)
    assert "<details" in html and "the flow goes A-&gt;B-&gt;C" in html
    assert "<table" in html and "<caption>" in html
    assert '<th scope="col">' in html


def test_render_hook_section():
    from Courseforge.scripts.generate_course import _render_hook_section
    blk = Block(
        block_id="p#hook_x_0", block_type="hook", page_id="p", sequence=0,
        content={"prompt": "What do you already know about fractions?",
                 "transition": "We will build on that intuition next."},
    )
    html = _render_hook_section(blk)
    assert 'class="hook"' in html
    assert "What do you already know" in html
    assert "build on that intuition" in html


# --------------------------------------------------------------------------- #
# IB5.9 — decision capture (no NEW call site; reuse existing events)
# --------------------------------------------------------------------------- #
class _RecordingCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


def test_block_plan_rationale_names_selected_new_type(monkeypatch):
    """Selecting a worked_example surfaces it in the block_plan decision."""
    from lib.generation.block_planner import plan_week_blocks
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    cap = _RecordingCapture()
    plan_week_blocks(
        terminal_objective={"id": "TO-01", "statement": "x"},
        chapter_objectives=[{"id": "CO-01", "statement": "y", "bloom_level": "apply"}],
        source_chunks=[{"text": "Step 1: isolate. Step 2: divide both sides."}],
        provider=None,
        capture=cap,
        course_code="C1",
    )
    block_plan_events = [e for e in cap.events if e.get("decision_type") == "block_plan"]
    assert block_plan_events
    blob = block_plan_events[-1]["decision"] + block_plan_events[-1]["rationale"]
    assert "worked_example" in blob


def test_failing_multimedia_a11y_emits_shape_check_decision():
    """A failing multimedia a11y check emits a rewrite_html_shape_check decision
    naming the media-stack failure (reuses the existing capture event)."""
    from lib.validators.rewrite_html_shape import RewriteHtmlShapeValidator

    cap = _RecordingCapture()
    block = Block(
        block_id="page_01#multimedia_x_0", block_type="multimedia",
        page_id="page_01", sequence=0,
        content=(
            '<figure class="multimedia" '
            'data-cf-block-id="page_01#multimedia_x_0">'
            '<video></video></figure>'  # no controls / track / transcript
        ),
    )
    RewriteHtmlShapeValidator().validate({
        "blocks": [block],
        "new_block_types_enabled": True,
        "decision_capture": cap,
    })
    shape_events = [
        e for e in cap.events
        if e.get("decision_type") == "rewrite_html_shape_check"
    ]
    # Per-piece B04 codes: the empty <video> stub is missing controls,
    # captions, audio-description, and transcript — each emits its own decision.
    decisions = " ".join(e["decision"] for e in shape_events)
    assert "MULTIMEDIA_CONTROLS_MISSING" in decisions
    assert "MULTIMEDIA_CAPTIONS_MISSING" in decisions
    assert "MULTIMEDIA_AUDIO_DESC_MISSING" in decisions
    assert "MULTIMEDIA_TRANSCRIPT_MISSING" in decisions


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
