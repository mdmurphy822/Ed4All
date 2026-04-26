"""Wave 82 regression test for pedagogy_graph_builder course_id fallback.

The rdf-shacl-551-2 audit found a shipped pedagogy_graph.json with
``course_id: ""`` despite chunks carrying the course code in their IDs
(``rdf_shacl_551_chunk_00001``). The pre-Wave-81
``_generate_pedagogy_graph`` stub didn't pass ``course_id``, and the
builder silently emitted ``""`` for the top-level field.

Wave 82 hardens the contract: when ``course_id`` is None/empty, the
builder derives a best-effort value from the first chunk's ID prefix.
This makes the failure mode (silent empty course_id) impossible without
a corresponding chunk-ID corruption.
"""

from __future__ import annotations

from Trainforge.pedagogy_graph_builder import (
    _derive_course_id_from_chunks,
    build_pedagogy_graph,
)


# ---------------------------------------------------------------------------
# _derive_course_id_from_chunks (the new helper)
# ---------------------------------------------------------------------------


class TestDeriveCourseIdFromChunks:
    def test_extracts_course_code_from_canonical_chunk_id(self):
        chunks = [{"id": "rdf_shacl_551_chunk_00001"}]
        assert _derive_course_id_from_chunks(chunks) == "RDF_SHACL_551"

    def test_uppercases_the_prefix(self):
        chunks = [{"id": "phys_101_chunk_00001"}]
        assert _derive_course_id_from_chunks(chunks) == "PHYS_101"

    def test_empty_chunks_returns_empty_string(self):
        assert _derive_course_id_from_chunks([]) == ""

    def test_chunks_without_canonical_shape_return_empty(self):
        # IDs that don't match the {code}_chunk_NNNNN shape don't yield
        # a course code. Guard against false positives.
        chunks = [
            {"id": "no-prefix"},
            {"id": "weird_id_format"},
            {"id": ""},
        ]
        assert _derive_course_id_from_chunks(chunks) == ""

    def test_picks_first_matching_chunk(self):
        # When the first chunk has a non-canonical ID but a later one does,
        # the helper still finds the course code. (Robust to test fixtures
        # mixing chunk shapes.)
        chunks = [
            {"id": "no-prefix"},
            {"id": "bio_201_chunk_00042"},
            {"id": "chem_101_chunk_00001"},
        ]
        # First match wins, even if later chunks have different prefixes.
        assert _derive_course_id_from_chunks(chunks) == "BIO_201"

    def test_handles_non_dict_entries(self):
        # Defensive: skip entries that aren't dicts or lack id field.
        chunks = [None, "string", {"no_id_field": True}, {"id": "abc_chunk_00001"}]
        assert _derive_course_id_from_chunks(chunks) == "ABC"


# ---------------------------------------------------------------------------
# build_pedagogy_graph fallback wiring
# ---------------------------------------------------------------------------


def _objectives_basic():
    """Minimal valid objectives shape — terminal objectives only."""
    return {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "First terminal objective"},
        ],
        "chapter_objectives": [],
    }


class TestBuildPedagogyGraphCourseIdFallback:
    def test_explicit_course_id_wins_over_chunk_derivation(self):
        chunks = [
            {
                "id": "rdf_shacl_551_chunk_00001",
                "chunk_type": "explanation",
                "learning_outcome_refs": ["TO-01"],
            }
        ]
        graph = build_pedagogy_graph(
            chunks, _objectives_basic(), course_id="EXPLICIT_OVERRIDE"
        )
        assert graph["course_id"] == "EXPLICIT_OVERRIDE"

    def test_missing_course_id_derives_from_chunks(self):
        # Audit reproducer: caller passes course_id=None, builder must
        # recover the course code from chunk IDs rather than emitting "".
        chunks = [
            {
                "id": "rdf_shacl_551_chunk_00001",
                "chunk_type": "explanation",
                "learning_outcome_refs": ["TO-01"],
            }
        ]
        graph = build_pedagogy_graph(chunks, _objectives_basic(), course_id=None)
        assert graph["course_id"] == "RDF_SHACL_551"

    def test_empty_string_course_id_derives_from_chunks(self):
        # Same fallback when caller passes "" (the rdf-shacl-551 actual
        # broken state).
        chunks = [
            {
                "id": "phys_101_chunk_00001",
                "chunk_type": "explanation",
                "learning_outcome_refs": ["TO-01"],
            }
        ]
        graph = build_pedagogy_graph(chunks, _objectives_basic(), course_id="")
        assert graph["course_id"] == "PHYS_101"

    def test_no_course_id_no_canonical_chunks_emits_empty(self):
        # When neither the caller nor the chunks supply a course code,
        # the builder still emits "" rather than crashing — keeps legacy
        # test fixtures (which use bare "chunk_001"-style IDs) passing.
        chunks = [
            {
                "id": "chunk_001",
                "chunk_type": "explanation",
                "learning_outcome_refs": ["TO-01"],
            }
        ]
        graph = build_pedagogy_graph(chunks, _objectives_basic(), course_id=None)
        assert graph["course_id"] == ""
