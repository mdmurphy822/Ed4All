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
    # 28 after IB5; the B15 `resources` addition brings the palette to 29;
    # the FR-INT-02 `guided_practice` (B08) addition brings it to 30.
    assert len(BLOCK_TYPES) == 30


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
        source_ids=("semantik:s#b1",), source_primary="semantik:s#b1",
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
        source_ids=("semantik:s#b1",),
    )
    attrs = blk.to_html_attrs()
    assert 'data-cf-fade-state="completion"' in attrs


def test_worked_example_fade_attr_absent_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    blk = Block(
        block_id="p#worked_example_x_0", block_type="worked_example",
        page_id="p", sequence=0, content="body", fade_state="completion",
        source_ids=("semantik:s#b1",),
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


# --------------------------------------------------------------------------- #
# Deterministic SVG plotter wire-in (gated on ED4ALL_RICHER_VISUAL_SYSTEM).
# The plotter closes the dual-coding hole for graphing weeks — a diagram block
# carrying an equation but no image renders a real accessible SVG figure.
# --------------------------------------------------------------------------- #
def _diagram_block_with_equation():
    return Block(
        block_id="week_03_content_01#diagram_graph_02", block_type="diagram",
        page_id="week_03_content_01", sequence=2,
        content={"caption": r"Graph of \( y = x^2 - 4 \)"},
    )


def test_diagram_plot_svg_flag_off_is_placeholder(monkeypatch):
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    from Courseforge.scripts.generate_course import _render_diagram_section
    html = _render_diagram_section(_diagram_block_with_equation())
    assert "diagram-pending" in html
    assert "<svg" not in html


def test_diagram_plot_svg_flag_on_injects_accessible_svg(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    from Courseforge.scripts.generate_course import _render_diagram_section
    pytest.importorskip("sympy")
    html = _render_diagram_section(_diagram_block_with_equation())
    assert "diagram-pending" not in html
    assert "<svg" in html
    assert 'role="img"' in html
    assert "aria-labelledby=" in html
    # id_prefix is derived (sanitized) from the block id for on-page uniqueness.
    assert "plot-week-03-content-01" in html


def test_diagram_plot_svg_flag_on_is_deterministic(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    pytest.importorskip("sympy")
    from Courseforge.scripts.generate_course import _render_diagram_section
    blk = _diagram_block_with_equation()
    assert _render_diagram_section(blk) == _render_diagram_section(blk)


def test_diagram_plot_svg_fails_closed_without_equation(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    from Courseforge.scripts.generate_course import _render_diagram_section
    blk = Block(
        block_id="p#diagram_x_0", block_type="diagram", page_id="p", sequence=0,
        content={"caption": "A concept map of the water cycle"},
    )
    html = _render_diagram_section(blk)
    assert "diagram-pending" in html  # no parseable equation ⇒ placeholder
    assert "<svg" not in html


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


# --------------------------------------------------------------------------- #
# IB5 content-type fix — each IB5 renderer stamps a valid data-cf-content-type
# on its block root so the critical rewrite_content_type gate
# (BlockContentTypeValidator) + rewrite_html_shape gate PASS on the str-path.
# Regression for a real post_rewrite_validation failure: 13 IB5-type
# blocks failed for carrying no data-cf-content-type.
# --------------------------------------------------------------------------- #

import dataclasses as _dataclasses  # noqa: E402


def _ib5_block(block_type, content):
    """Minimal renderable IB5 Block (dict content -> deterministic renderer)."""
    return Block(
        block_id=f"page_01#{block_type}_x_0", block_type=block_type,
        page_id="page_01", sequence=0, content=content,
    )


def _str_block_from_render(rendered_block, html):
    """Clone a Block with its ``content`` set to the rendered HTML string.

    The rewrite-tier validators (BlockContentTypeValidator / RewriteHtmlShape
    Validator) dispatch on ``isinstance(block.content, str)`` — they scrape the
    rendered HTML. So we validate the renderer output by feeding it back as the
    block's str content.
    """
    return _dataclasses.replace(rendered_block, content=html)


# block_type -> (dict content, expected default content_type, renderer name)
_IB5_RENDER_CASES = {
    "hook": (
        {"prompt": "What do you already know about fractions?",
         "transition": "We will build on that next."},
        "overview", "_render_hook_section",
    ),
    "multimedia": (
        {"media_url": "https://x/v.mp4", "captions_url": "https://x/c.vtt",
         "transcript": "Full transcript here.", "audio_desc": "AD here.",
         "caption": "A short clip."},
        "explanation", "_render_multimedia_section",
    ),
    "worked_example": (
        {"problem": "Solve 2x + 3 = 11",
         "steps": [{"subgoal": "Isolate", "body": "subtract 3", "why": "undo add"},
                   {"subgoal": "Solve", "body": "divide by 2", "why": "undo mult"}]},
        "example", "_render_worked_example_section",
    ),
    "diagram": (
        {"caption": "Process flow", "long_description": "A->B->C",
         "headers": ["Node", "Next"], "rows": [["A", "B"], ["B", "C"]]},
        "diagram", "_render_diagram_section",
    ),
}


@pytest.mark.parametrize("block_type", sorted(_IB5_RENDER_CASES))
def test_ib5_render_stamps_valid_content_type(block_type, monkeypatch):
    """Each IB5 renderer stamps a data-cf-content-type that is a valid ChunkType.

    Asserts the rendered root carries data-cf-content-type with the per-type
    default value AND that BlockContentTypeValidator + RewriteHtmlShapeValidator
    PASS on the rendered HTML (str-path). COURSEFORGE_EMIT_BLOCKS is on (as in
    the real two-pass pipeline) so to_html_attrs() emits the universal
    data-cf-block-id that rewrite_html_shape's REQUIRED_ATTRS demands.
    """
    import importlib

    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "true")

    from lib.validators.content_type import get_valid_chunk_types
    from Courseforge.router.inter_tier_gates import BlockContentTypeValidator
    from lib.validators.rewrite_html_shape import RewriteHtmlShapeValidator

    gc = importlib.import_module("Courseforge.scripts.generate_course")
    dict_content, expected_ct, renderer_name = _IB5_RENDER_CASES[block_type]
    renderer = getattr(gc, renderer_name)

    blk = _ib5_block(block_type, dict_content)
    html = renderer(blk)
    assert "data-cf-block-id=" in html

    # The root carries a data-cf-content-type whose value is a valid ChunkType.
    assert f'data-cf-content-type="{expected_ct}"' in html
    assert expected_ct in get_valid_chunk_types()

    str_block = _str_block_from_render(blk, html)

    ct_result = BlockContentTypeValidator().validate({"blocks": [str_block]})
    assert ct_result.passed, (
        f"{block_type}: BlockContentTypeValidator failed: "
        f"{[i.code for i in ct_result.issues]}"
    )

    shape_result = RewriteHtmlShapeValidator().validate({
        "blocks": [str_block],
        "new_block_types_enabled": True,
    })
    # rewrite_html_shape's IB5 a11y arms are WARNING-severity; the gate only
    # fails on critical shape issues (not-HTML / parse-fail / missing required
    # attr). The deterministic renderer ships a clean shape, so it must pass.
    assert shape_result.passed, (
        f"{block_type}: RewriteHtmlShapeValidator failed: "
        f"{[(i.severity, i.code) for i in shape_result.issues]}"
    )


def test_ib5_render_honors_resolved_content_type_label():
    """A block with an explicit content_type_label uses it over the default."""
    import importlib

    gc = importlib.import_module("Courseforge.scripts.generate_course")
    blk = _dataclasses.replace(
        _ib5_block("worked_example", {"problem": "p", "steps": []}),
        content_type_label="procedure",
    )
    html = gc._render_worked_example_section(blk)
    assert 'data-cf-content-type="procedure"' in html
    assert 'data-cf-content-type="example"' not in html


def test_ib5_render_repairs_invalid_content_type_label():
    """An INVALID content_type_label is repaired to the per-type canonical default.

    Regression for a real ``OUTLINE_BLOCK_INVALID_CONTENT_TYPE``
    root: the planner / LLM wrote a domain term (``expression``) into the
    content-type slot. The renderer must NOT emit it (it would fail the gate) —
    it falls back to the block_type's canonical ChunkType default.
    """
    import importlib

    from lib.validators.content_type import get_valid_chunk_types

    gc = importlib.import_module("Courseforge.scripts.generate_course")
    blk = _dataclasses.replace(
        _ib5_block("worked_example", {"problem": "p", "steps": []}),
        content_type_label="expression",  # not a ChunkType member
    )
    html = gc._render_worked_example_section(blk)
    assert 'data-cf-content-type="expression"' not in html
    assert 'data-cf-content-type="example"' in html
    assert "example" in get_valid_chunk_types()


def test_scenario_and_guided_practice_roots_stamp_content_type(monkeypatch):
    """The standard scenario (B09) + guided_practice (B08) roots carry a valid
    data-cf-content-type so the post_rewrite rewrite_content_type gate passes.

    Regression for a real run's ``week_01_application#scenario`` /
    ``week_02_application#problem`` MISSING-content-type failures — these roots
    previously emitted no data-cf-content-type at all.
    """
    import importlib

    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "true")

    from Courseforge.router.inter_tier_gates import BlockContentTypeValidator
    from lib.validators.content_type import get_valid_chunk_types

    gc = importlib.import_module("Courseforge.scripts.generate_course")

    scenario_blk = _ib5_block(
        "scenario",
        {"situation": "A car accelerates.", "prompt": "Find a.", "debrief": "ok"},
    )
    scenario_html = gc._render_scenario(scenario_blk)
    assert 'data-cf-content-type="scenario"' in scenario_html
    assert "scenario" in get_valid_chunk_types()

    gp_blk = _ib5_block(
        "guided_practice", {"prompt": "Try these.", "items": ["q1", "q2"]}
    )
    gp_html = gc._render_guided_practice(gp_blk)
    assert 'data-cf-content-type="exercise"' in gp_html

    V = BlockContentTypeValidator()
    for blk, html in (
        (scenario_blk, scenario_html), (gp_blk, gp_html),
    ):
        result = V.validate({"blocks": [_str_block_from_render(blk, html)]})
        assert result.passed, (
            f"{blk.block_type}: BlockContentTypeValidator failed: "
            f"{[i.code for i in result.issues]}"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
