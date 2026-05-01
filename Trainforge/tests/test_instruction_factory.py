#!/usr/bin/env python3
"""Tests for Trainforge.generators.instruction_factory.TEMPLATE_CATALOG.

Wave 133e (Plan-2 P1#10) added 5 high-frequency content_type axes ×
6 Bloom levels = 30 tailored cells (definition / summary / overview /
real_world_scenario / common_pitfall) on top of the prior 5-axis catalog.
This test asserts the new tailored cells route through ``_select_template``
and are picked up by ``synthesize_instruction_pair`` instead of falling
back to ``("understand", "_default")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.instruction_factory import (
    TEMPLATE_CATALOG,
    synthesize_instruction_pair,
)


# ---------------------------------------------------------------------------
# Wave 133e: tailored TEMPLATE_CATALOG cells for the 5 high-frequency
# content_type axes added in Plan-2 P1#10.
# ---------------------------------------------------------------------------


_BLOOM_LEVELS = ("remember", "understand", "apply", "analyze", "evaluate", "create")

_NEW_CONTENT_TYPES = (
    "definition",
    "summary",
    "overview",
    "real_world_scenario",
    "common_pitfall",
)


def _make_chunk(content_type: str, bloom: str = "understand") -> dict:
    """Build a minimal chunk that exercises (bloom, content_type) routing."""
    return {
        "id": f"chunk_factory_{content_type}_{bloom}",
        "text": "Some pedagogical material about a focused subject area.",
        "summary": (
            "A short paraphrased summary of the chunk that does not echo "
            "the chunk text verbatim and stays well clear of any 50-char "
            "verbatim span."
        ),
        "learning_outcome_refs": ["TO-01"],
        "concept_tags": ["focused-subject"],
        "bloom_level": bloom,
        "content_type_label": content_type,
    }


def test_template_catalog_covers_30_new_cells():
    """All 5 new content types × 6 Bloom levels are present and unique."""
    for ct in _NEW_CONTENT_TYPES:
        for bloom in _BLOOM_LEVELS:
            assert (bloom, ct) in TEMPLATE_CATALOG, (
                f"Missing tailored cell ({bloom!r}, {ct!r}) added in Wave 133e"
            )

    # Every tailored cell should be a distinct string from the bloom-default
    # row — that's the whole point of the wave.
    for ct in _NEW_CONTENT_TYPES:
        for bloom in _BLOOM_LEVELS:
            tailored = TEMPLATE_CATALOG[(bloom, ct)]
            default = TEMPLATE_CATALOG[(bloom, "_default")]
            assert tailored != default, (
                f"Tailored cell ({bloom!r}, {ct!r}) is byte-identical to "
                f"the bloom-default; defeats the purpose of Wave 133e"
            )


def test_instruction_factory_uses_tailored_definition_template():
    chunk = _make_chunk("definition", bloom="understand")
    result = synthesize_instruction_pair(chunk, seed=11)
    assert result.pair is not None
    assert result.template_id == "understand.definition", (
        f"Expected tailored 'understand.definition' template, "
        f"got {result.template_id!r}"
    )


def test_instruction_factory_uses_tailored_summary_template():
    chunk = _make_chunk("summary", bloom="apply")
    result = synthesize_instruction_pair(chunk, seed=11)
    assert result.pair is not None
    assert result.template_id == "apply.summary"


def test_instruction_factory_uses_tailored_overview_template():
    chunk = _make_chunk("overview", bloom="remember")
    result = synthesize_instruction_pair(chunk, seed=11)
    assert result.pair is not None
    assert result.template_id == "remember.overview"


def test_instruction_factory_uses_tailored_real_world_scenario_template():
    chunk = _make_chunk("real_world_scenario", bloom="analyze")
    result = synthesize_instruction_pair(chunk, seed=11)
    assert result.pair is not None
    assert result.template_id == "analyze.real_world_scenario"


def test_instruction_factory_uses_tailored_common_pitfall_template():
    chunk = _make_chunk("common_pitfall", bloom="evaluate")
    result = synthesize_instruction_pair(chunk, seed=11)
    assert result.pair is not None
    assert result.template_id == "evaluate.common_pitfall"


def test_default_fallback_set_still_routes_to_bloom_default():
    """Documented Wave 133e fallback set: these content types intentionally
    have no tailored cells and must fall through to the bloom-default row."""
    fallback_types = (
        "concept",
        "page-title",
        "rationale",
        "motivation",
        "analysis",
        "orientation",
        "prerequisites",
        "exercise",
        "assessment_item",
        "problem_solution",
    )
    for ct in fallback_types:
        for bloom in _BLOOM_LEVELS:
            assert (bloom, ct) not in TEMPLATE_CATALOG, (
                f"({bloom!r}, {ct!r}) is in the deliberate _default "
                f"fallback set but has a tailored cell; either add it to "
                f"the docstring fallback list or remove the tailored cell"
            )
