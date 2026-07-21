"""Wave 1.6 W1.6.B — per-objective source attribution emit tests.

Asserts the contract on
:func:`MCP.tools._content_gen_helpers.synthesize_objectives_from_topics`:

* Path A (LOs extracted from explicit ``<p class="learning-objectives">``
  lists) emits ``source_refs[]`` populated from each topic's source
  block IDs in the structured ``[{ref, chunk_ids[]}]`` shape.
* Path B (heading-driven synthesis) emits single-element structured
  ``source_refs[]`` per CO, with multi-entry aggregation on terminal
  outcomes that synthesize across multiple week topics.
* :func:`_normalize_objective_entry` round-trips legacy ``List[str]``
  ``source_refs`` payloads verbatim (Wave 1.6 W1.6.A back-compat for
  ``--reuse-objectives``).
* When a topic carries no source provenance (fixtures
  with no ``dart_block_ids`` / ``source_file``), the synthesizer emits
  an EMPTY ``source_refs[]`` array — the W1.6.C validator surfaces the
  empty case downstream; the synth path itself stays graceful.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture builders
# ---------------------------------------------------------------------- #


def _mk_topic(
    heading: str,
    *,
    source_file: str = "rdf_primer_accessible",
    dart_block_ids: List[str] | None = None,
    chapter_id: str | None = None,
    extracted_lo_statements: List[str] | None = None,
    key_terms: List[str] | None = None,
) -> Dict[str, Any]:
    """Return a topic dict mirroring :func:`parse_dart_html_files` output.

    Block IDs default to a single ``s1`` so the synthesizer can resolve
    a structured source_ref. Pass ``dart_block_ids=[]`` to suppress
    attribution for the legacy-fixture / empty-refs branch.
    """
    return {
        "heading": heading,
        "paragraphs": [
            (
                f"Body text for {heading} explaining the concept in "
                "sufficient depth to satisfy the grounding validator "
                "non-trivial paragraph floor of thirty words each "
                "required for content pages to be treated as real "
                "body prose and not as an empty heading-only section."
            ),
        ],
        "key_terms": key_terms or [heading.split()[0].lower()],
        "source_file": source_file,
        "word_count": 60,
        "chapter_id": chapter_id,
        "dart_block_ids": (
            list(dart_block_ids)
            if dart_block_ids is not None
            else ["s1"]
        ),
        "extracted_lo_statements": list(extracted_lo_statements or []),
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }


# ---------------------------------------------------------------------- #
# Path A: LOs extracted from explicit <p class="learning-objectives">
# ---------------------------------------------------------------------- #


def test_path_a_populates_structured_source_refs_from_topic_dart_block_ids() -> None:
    """Path A emits one structured source_refs[] entry per LO derived
    from the topic; ``ref`` falls back to the heading slug, ``chunk_ids``
    is the topic's source block-IDs projected to ``semantik:{slug}#{block_id}``.
    """
    topics = [
        _mk_topic(
            "Triple Stores and Data Models",
            source_file="rdf_primer_accessible",
            dart_block_ids=["sec3-1-targets"],
            extracted_lo_statements=[
                "Identify the three components of an RDF triple.",
                "Describe how IRIs disambiguate resource identity.",
            ],
        ),
    ]
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=1, max_terminal=2,
    )
    # Two LOs from one topic: both terminals (max_terminal=2 admits both).
    all_los = terminal + chapter
    assert len(all_los) == 2
    for lo in all_los:
        assert "source_refs" in lo, "Path A LO MUST carry source_refs"
        assert isinstance(lo["source_refs"], list)
        assert len(lo["source_refs"]) == 1, (
            "Path A LOs from one topic emit single-element source_refs"
        )
        entry = lo["source_refs"][0]
        assert isinstance(entry, dict)
        assert set(entry.keys()) == {"ref", "chunk_ids"}
        # ref defaults to heading slug when no chapter_id is set
        assert entry["ref"]
        assert isinstance(entry["chunk_ids"], list)
        assert len(entry["chunk_ids"]) >= 1
        assert entry["chunk_ids"] == [
            "semantik:rdf_primer_accessible#sec3-1-targets",
        ]


def test_path_a_uses_chapter_id_when_present() -> None:
    """When the topic carries a ``chapter_id`` (converter output
    with ``<article role="doc-chapter">``), the synthesizer prefers it
    over the slugified-heading fallback for the ``ref`` field.
    """
    topics = [
        _mk_topic(
            "Triple Stores",
            source_file="rdf_primer_accessible",
            dart_block_ids=["s7"],
            chapter_id="ch7",
            extracted_lo_statements=[
                "Identify the components of an RDF triple.",
            ],
        ),
    ]
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=1, max_terminal=2,
    )
    all_los = terminal + chapter
    assert len(all_los) == 1
    assert all_los[0]["source_refs"] == [
        {"ref": "ch7", "chunk_ids": ["semantik:rdf_primer_accessible#s7"]},
    ]


# ---------------------------------------------------------------------- #
# Path B: heading-driven synthesis
# ---------------------------------------------------------------------- #


def test_path_b_single_heading_populates_single_element_source_refs() -> None:
    """Path B with one topic per week emits single-element structured
    source_refs[] on both the terminal and any chapter entries.
    """
    topics = [
        _mk_topic(
            "Introduction to Photosynthesis",
            source_file="biology_textbook",
            dart_block_ids=["s1"],
        ),
    ]
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=1, max_terminal=2,
    )
    assert len(terminal) == 1
    lo = terminal[0]
    assert "source_refs" in lo
    assert lo["source_refs"] == [
        {"ref": "introduction-to-photosynthesis",
         "chunk_ids": ["semantik:biology_textbook#s1"]},
    ]
    assert chapter == []


def test_path_b_multi_heading_aggregates_terminal_source_refs() -> None:
    """When MULTIPLE topics roll up under a single terminal outcome
    (the week's primary + secondary headings), the terminal's
    source_refs[] aggregates one entry per distinct heading; per-CO
    secondary entries each carry their own single-element source_refs[]
    (each secondary heading becomes its own CO).

    Verifies multi-entry aggregation is per-heading (NOT flattened into
    one entry across all chunks).
    """
    topics = [
        _mk_topic(
            "RDF Foundations",
            source_file="rdf_primer",
            dart_block_ids=["s1"],
            chapter_id="ch1",
        ),
        _mk_topic(
            "SPARQL Query Forms",
            source_file="sparql_primer",
            dart_block_ids=["s5"],
            chapter_id="ch5",
        ),
        _mk_topic(
            "SHACL Shape Authoring",
            source_file="shacl_spec",
            dart_block_ids=["s7"],
            chapter_id="ch7",
        ),
    ]
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=1, max_terminal=1,
    )
    # max_terminal=1 → first primary becomes the terminal (rolls up
    # ALL three week topics' source_refs); the two secondary topics
    # each become their own CO with single-element source_refs.
    assert len(terminal) == 1
    to_entry = terminal[0]
    assert "source_refs" in to_entry
    refs = to_entry["source_refs"]
    assert isinstance(refs, list)
    assert len(refs) == 3, (
        "Multi-heading rollup MUST emit one source_refs[] entry per "
        "distinct heading, NOT flattened into one entry"
    )
    refs_by_ref = {entry["ref"]: entry for entry in refs}
    assert refs_by_ref["ch1"]["chunk_ids"] == ["semantik:rdf_primer#s1"]
    assert refs_by_ref["ch5"]["chunk_ids"] == ["semantik:sparql_primer#s5"]
    assert refs_by_ref["ch7"]["chunk_ids"] == ["semantik:shacl_spec#s7"]
    # Each chunk_ids entry is its OWN heading's chunks — not unified.
    for entry in refs:
        assert len(entry["chunk_ids"]) == 1

    # Each secondary heading produces its own CO with single-element
    # source_refs[] (one entry per CO).
    assert len(chapter) == 2
    for co_entry in chapter:
        assert "source_refs" in co_entry
        assert len(co_entry["source_refs"]) == 1


def test_path_b_multi_block_ids_within_one_topic_collects_chunks_in_one_entry() -> None:
    """When a single topic carries multiple ``dart_block_ids`` (e.g. a
    merged-sibling-section topic), the chunks aggregate INSIDE ONE
    source_refs[] entry — NOT split across multiple entries. Mirrors
    :func:`_topic_source_references` shape.
    """
    topics = [
        _mk_topic(
            "Core Constraint Components",
            source_file="shacl_spec",
            dart_block_ids=["sec4-1-core", "sec4-2-property"],
            chapter_id="ch8",
        ),
    ]
    terminal, _chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=1, max_terminal=2,
    )
    assert len(terminal) == 1
    refs = terminal[0]["source_refs"]
    assert len(refs) == 1
    entry = refs[0]
    assert entry["ref"] == "ch8"
    assert entry["chunk_ids"] == [
        "semantik:shacl_spec#sec4-1-core",
        "semantik:shacl_spec#sec4-2-property",
    ]


# ---------------------------------------------------------------------- #
# --reuse-objectives legacy List[str] round-trip (W1.6.A pass-through)
# ---------------------------------------------------------------------- #


def test_normalize_objective_entry_preserves_legacy_list_str_source_refs() -> None:
    """``--reuse-objectives`` payloads carrying the legacy flat
    ``List[str]`` ``source_refs`` shape survive
    :func:`_normalize_objective_entry` verbatim. Asserts the W1.6.A
    pass-through landed alongside the schema bump.
    """
    raw = {
        "id": "TO-01",
        "statement": "Develop SHACL shape graphs that validate RDF data.",
        "bloom_level": "create",
        "bloom_verb": "develop",
        "source_refs": [
            "semantik:owl2_primer_accessible#s1",
            "semantik:owl2_primer_accessible#s2",
            "semantik:owl2_primer_accessible#s3",
        ],
    }
    normalized = _cgh._normalize_objective_entry(raw)
    assert normalized is not None
    assert "source_refs" in normalized
    assert normalized["source_refs"] == [
        "semantik:owl2_primer_accessible#s1",
        "semantik:owl2_primer_accessible#s2",
        "semantik:owl2_primer_accessible#s3",
    ]
    # Defensive: each entry MUST be a string (no shape coercion).
    for ref in normalized["source_refs"]:
        assert isinstance(ref, str)


def test_normalize_objective_entry_preserves_structured_source_refs() -> None:
    """The structured Wave-1.6 ``[{ref, chunk_ids[]}]`` shape ALSO
    survives the normalizer round-trip — the schema's ``oneOf`` arm
    requires both shapes pass through unchanged.
    """
    raw = {
        "id": "TO-04",
        "statement": "Author SHACL shape graphs that validate RDF data.",
        "source_refs": [
            {
                "ref": "ch7",
                "chunk_ids": ["semantik:shacl-spec#sec3-1-targets"],
            },
            {
                "ref": "ch8",
                "chunk_ids": [
                    "semantik:shacl-spec#sec4-1-core-constraints",
                    "semantik:shacl-spec#sec4-2-property-shapes",
                ],
            },
        ],
    }
    normalized = _cgh._normalize_objective_entry(raw)
    assert normalized is not None
    assert "source_refs" in normalized
    assert len(normalized["source_refs"]) == 2
    first = normalized["source_refs"][0]
    second = normalized["source_refs"][1]
    assert first == {
        "ref": "ch7",
        "chunk_ids": ["semantik:shacl-spec#sec3-1-targets"],
    }
    assert second == {
        "ref": "ch8",
        "chunk_ids": [
            "semantik:shacl-spec#sec4-1-core-constraints",
            "semantik:shacl-spec#sec4-2-property-shapes",
        ],
    }


# ---------------------------------------------------------------------- #
# Empty source_refs path: synthesizer stays graceful on legacy fixtures
# ---------------------------------------------------------------------- #


def test_synthesizer_emits_empty_source_refs_when_topic_has_no_provenance() -> None:
    """A topic carrying no ``dart_block_ids`` AND no ``source_file``
    (legacy fixture output) yields an EMPTY structured-shape
    ``source_refs[]`` rather than raising. The W1.6.C validator
    surfaces the empty case downstream — the synth path itself MUST
    stay graceful so legacy corpora continue to round-trip.
    """
    topics = [
        _mk_topic(
            "Legacy Section Without Provenance",
            source_file="",
            dart_block_ids=[],
        ),
    ]
    # No exception expected.
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=1, max_terminal=2,
    )
    all_los = terminal + chapter
    assert len(all_los) == 1
    lo = all_los[0]
    assert "source_refs" in lo
    assert lo["source_refs"] == []


def test_synthesizer_path_a_emits_empty_source_refs_for_attribution_less_topic() -> None:
    """Path A counterpart of the legacy-fixture branch: explicit LO list
    on a topic with no source block IDs still emits the LOs (statements
    are non-placeholder source material) but with empty source_refs[].
    """
    topics = [
        _mk_topic(
            "Photosynthesis",
            source_file="biology",
            dart_block_ids=[],
            extracted_lo_statements=[
                "Describe the role of chlorophyll in photosynthesis.",
            ],
        ),
    ]
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics, duration_weeks=1, max_terminal=2,
    )
    all_los = terminal + chapter
    assert len(all_los) == 1
    lo = all_los[0]
    assert "source_refs" in lo
    assert lo["source_refs"] == []
