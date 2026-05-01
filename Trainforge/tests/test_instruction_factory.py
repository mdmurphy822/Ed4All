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


# ---------------------------------------------------------------------------
# Wave 135b: anchored force-injection — assert the dispatch on FORM_DATA
# ``anchored_status`` actually fires for "complete" entries (real
# definition appears in the pair body) and falls back with a warning +
# decision-capture event for "degraded_placeholder" entries.
# ---------------------------------------------------------------------------


class _RecordingCapture:
    """Minimal stub for the DecisionCapture surface the factories use.

    Only the ``log_decision`` method is touched — recorded as a list of
    dicts so tests can assert on ``decision_type`` membership.
    """

    def __init__(self) -> None:
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(dict(kwargs))
        return f"event-{len(self.events)}"


def _curie_chunk(curie: str, *, chunk_id: str = "chunk_test") -> dict:
    """Build a chunk that exercises the force-injection path.

    Mock-provider templates use slugified ``concept_tags`` so colons
    become hyphens and the literal CURIE never lands in the pair body
    via the natural template path. That's exactly the scenario Wave
    135b's force-injection is designed to catch.
    """
    return {
        "id": chunk_id,
        "text": (
            f"This passage discusses {curie} as a SHACL constraint and "
            "offers extra prose so the chunk is well above the 50-char "
            "verbatim leakage threshold without trivial repetition."
        ),
        "summary": (
            "A short paraphrased summary of the chunk that does not "
            "echo any 50-char verbatim span from the chunk text and "
            "stays clear of the leakage gate."
        ),
        "learning_outcome_refs": ["TO-01"],
        "concept_tags": ["focused-shacl-topic"],
        "bloom_level": "understand",
        "content_type_label": "explanation",
    }


def test_force_injection_uses_anchored_definition_for_complete_entry():
    """Wave 135b — when the missing CURIE has anchored_status='complete'
    in FORM_DATA, the force-injection embeds an actual definition
    sentence drawn from FORM_DATA.definitions, NOT a token-stuffing
    suffix like ``(Reference: sh:datatype.)``."""
    # sh:datatype is "complete" in _RDF_SHACL_FALLBACK_FORM_DATA.
    chunk = _curie_chunk("sh:datatype")
    capture = _RecordingCapture()
    result = synthesize_instruction_pair(
        chunk, seed=42,
        preserve_tokens=["sh:datatype"],
        capture=capture,
    )
    assert result.pair is not None
    body = result.pair["prompt"] + " " + result.pair["completion"]
    # Anchored marker present in the pair (set by the factory when the
    # anchored path actually fired, not the legacy token-stuffing path).
    assert "sh:datatype" in body
    anchored = (
        result.pair.get("preserve_tokens_anchored", [])
        + result.pair.get("preserve_tokens_anchored_prompt", [])
    )
    assert "sh:datatype" in anchored, (
        "Wave 135b: complete entry must route through anchored-injection; "
        f"got pair markers={dict(result.pair)!r}"
    )
    # No degraded-fallback decision-capture event for a complete entry.
    degraded_events = [
        e for e in capture.events
        if e.get("decision_type") == "form_data_degraded_placeholder_skipped"
    ]
    assert not degraded_events
    # And the pair body should contain a substring that comes from
    # FORM_DATA.definitions — not the token-stuffing template.
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )
    entry = _RDF_SHACL_FALLBACK_FORM_DATA["sh:datatype"]
    # At least one definition chunk must appear (or its truncated
    # prefix) somewhere in the final pair body.
    matched_def = any(
        # take a recognisable phrase from the definition
        d_excerpt in body
        for d in entry.definitions
        for d_excerpt in [d[:60]] if len(d) >= 60
    )
    assert matched_def, (
        "Wave 135b: complete entry's definition text must surface in "
        f"the pair body. body={body!r}"
    )


def test_force_injection_falls_back_for_degraded_entry():
    """Wave 135b — when the missing CURIE has
    anchored_status='degraded_placeholder' in FORM_DATA, force-injection
    falls back to the legacy token-stuffing path AND emits a
    ``form_data_degraded_placeholder_skipped`` decision-capture event."""
    # sh:minCount is "degraded_placeholder" after Wave 135a.
    chunk = _curie_chunk("sh:minCount")
    capture = _RecordingCapture()
    result = synthesize_instruction_pair(
        chunk, seed=42,
        preserve_tokens=["sh:minCount"],
        capture=capture,
    )
    assert result.pair is not None
    body = result.pair["prompt"] + " " + result.pair["completion"]
    assert "sh:minCount" in body
    # Token-stuffing markers should be present (Wave 121 path).
    injected = (
        result.pair.get("preserve_tokens_injected", [])
        + result.pair.get("preserve_tokens_injected_prompt", [])
    )
    assert "sh:minCount" in injected
    # NOT in the anchored-marker list.
    anchored = (
        result.pair.get("preserve_tokens_anchored", [])
        + result.pair.get("preserve_tokens_anchored_prompt", [])
    )
    assert "sh:minCount" not in anchored
    # And the decision-capture event fires.
    degraded_events = [
        e for e in capture.events
        if e.get("decision_type") == "form_data_degraded_placeholder_skipped"
    ]
    assert degraded_events, (
        "Wave 135b: degraded-placeholder entry must emit a "
        f"form_data_degraded_placeholder_skipped event; got {capture.events!r}"
    )
    # Each event names the CURIE that triggered it.
    for ev in degraded_events:
        assert "sh:minCount" in ev.get("decision", "") or "sh:minCount" in ev.get("rationale", "")


def test_force_injection_anchors_non_manifest_curies():
    """Wave 135b — when the missing CURIE isn't in FORM_DATA at all
    (i.e. a non-manifest CURIE that the full-CURIE-set extension in
    synthesize_training.py surfaces), force-injection falls back to
    legacy token-stuffing AND emits the same degraded-fallback event."""
    # prov:wasDerivedFrom is a real W3C CURIE but isn't in FORM_DATA
    # (only the 40 rdf-shacl manifest CURIEs are).
    chunk = _curie_chunk("prov:wasDerivedFrom")
    capture = _RecordingCapture()
    result = synthesize_instruction_pair(
        chunk, seed=42,
        preserve_tokens=["prov:wasDerivedFrom"],
        capture=capture,
    )
    assert result.pair is not None
    body = result.pair["prompt"] + " " + result.pair["completion"]
    assert "prov:wasDerivedFrom" in body
    injected = (
        result.pair.get("preserve_tokens_injected", [])
        + result.pair.get("preserve_tokens_injected_prompt", [])
    )
    assert "prov:wasDerivedFrom" in injected
    # Decision-capture event fires for non-manifest CURIE just like
    # for degraded-placeholder — operator visibility is uniform.
    degraded_events = [
        e for e in capture.events
        if e.get("decision_type") == "form_data_degraded_placeholder_skipped"
    ]
    assert degraded_events
    assert any(
        "prov:wasDerivedFrom" in ev.get("decision", "")
        or "prov:wasDerivedFrom" in ev.get("rationale", "")
        for ev in degraded_events
    )
