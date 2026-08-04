"""Wave 82 Phase C wiring test for tech-anchor seeding.

Pin: when TRAINFORGE_SEED_TECH_CONCEPTS=true is set, chunk text
mentioning W3C-standard surface forms (RDF, RDFS, OWL, SHACL, SPARQL,
Turtle, etc.) emits the corresponding canonical concept slug into the
chunk's concept_tags. Behaviour-flagged: when the env var is unset or
falsy the legacy tag distribution is unchanged (no surprise shifts in
existing corpora on rebuild).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from Trainforge.pipeline.process_course import CourseProcessor


def _bare_processor() -> CourseProcessor:
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TEST_101"
    proc.domain_concept_seeds = []
    return proc


class TestTechAnchorSeedingFlag:
    def test_default_off_text_with_rdf_does_not_seed(self, monkeypatch):
        monkeypatch.delenv("TRAINFORGE_SEED_TECH_CONCEPTS", raising=False)
        proc = _bare_processor()
        text = "RDF is a graph data model used in semantic-web systems."
        item: Dict[str, Any] = {"key_concepts": []}
        tags = proc._extract_concept_tags(text, item)
        assert "rdf" not in tags

    def test_flag_on_seeds_rdf_anchor(self, monkeypatch):
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        text = "RDF is a graph data model."
        tags = proc._extract_concept_tags(text, {"key_concepts": []})
        assert "rdf" in tags

    def test_flag_on_seeds_multiple_anchors(self, monkeypatch):
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        text = (
            "An RDF graph serialized as Turtle, validated with SHACL, and "
            "queried via SPARQL is the standard semantic-web stack."
        )
        tags = proc._extract_concept_tags(text, {"key_concepts": []})
        assert {"rdf", "turtle", "shacl", "sparql"} <= set(tags)

    def test_flag_on_seeds_owl_sameas_predicate(self, monkeypatch):
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        text = "Use owl:sameAs to assert identity between two IRIs."
        tags = proc._extract_concept_tags(text, {"key_concepts": []})
        assert "same-as" in tags

    def test_flag_on_does_not_double_count_existing_tags(self, monkeypatch):
        # When the chunk's key_concepts already includes 'rdf', the tech
        # anchor pass MUST NOT duplicate it.
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        text = "RDF is a graph data model."
        item = {"key_concepts": ["rdf"]}
        tags = proc._extract_concept_tags(text, item)
        assert tags.count("rdf") == 1

    def test_flag_false_value_does_not_seed(self, monkeypatch):
        # Only "true" enables. Other values stay default-off.
        for value in ["false", "0", "no", "1", "yes"]:
            monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", value)
            proc = _bare_processor()
            tags = proc._extract_concept_tags(
                "RDF is everywhere.", {"key_concepts": []}
            )
            if value == "true":
                continue
            assert "rdf" not in tags, f"value={value!r} should not enable"

    def test_owl_2_negative_case_preserved(self, monkeypatch):
        # The tech_anchors regex deliberately excludes "OWL 2" / "OWL-2"
        # so existing version-qualified concept nodes (owl-2, owl-2-dl)
        # keep their identity. Verify that wiring through the chunk
        # pipeline preserves this.
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        tags = proc._extract_concept_tags(
            "OWL 2 DL is decidable.", {"key_concepts": []}
        )
        assert "owl" not in tags


class TestTechAnchorPromotionAndDedup:
    """Tech-anchor under-seeding fix: anchors seeded FIRST + TRUSTED so
    flagship concepts survive the ``MAX_CONCEPT_TAGS`` cap and the
    Wave-76 droppable-class filter; fragment-superstring slugs of an
    admitted anchor are dropped in favour of the clean anchor.
    """

    def _eighteen_bold_fragments(self) -> List[str]:
        # 18 untrusted key_concepts that would otherwise saturate the
        # 20-tag cap and push a later-seeded anchor out.
        return [f"competing-concept-{i:02d}" for i in range(18)]

    def test_ragas_survives_cap_and_fragment_dropped(self, monkeypatch):
        # Flag ON: chunk text contains "evaluate with RAGAS" plus a flood
        # of other bold fragments AND a fragment-superstring slug
        # 'ragas-short-for-rag'. The clean 'ragas' anchor MUST be present
        # (seeded first, never capped out); the fragment MUST be absent.
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        text = (
            "To evaluate with RAGAS, score faithfulness and relevance. "
            "RAGAS is short for RAG assessment."
        )
        key_concepts = self._eighteen_bold_fragments() + [
            "ragas-is-short-for-rag",  # fragment-superstring of 'ragas'
        ]
        tags = proc._extract_concept_tags(text, {"key_concepts": key_concepts})
        assert "ragas" in tags, "clean 'ragas' anchor must survive the cap"
        assert "ragas-is-short-for-rag" not in tags, (
            "fragment-superstring of an admitted anchor must be dropped"
        )

    def test_nim_tagged_amid_competing_tags(self, monkeypatch):
        # Flag ON: "deploy on NVIDIA NIM" → 'nim' tagged even when many
        # competing untrusted tags vie for the cap.
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        text = "Deploy the model on NVIDIA NIM for low-latency inference."
        key_concepts = self._eighteen_bold_fragments()
        tags = proc._extract_concept_tags(text, {"key_concepts": key_concepts})
        assert "nim" in tags

    def test_flag_off_byte_identical_no_anchors(self, monkeypatch):
        # Flag OFF: no anchors added; output identical to the legacy
        # non-anchor pass over the same input.
        monkeypatch.delenv("TRAINFORGE_SEED_TECH_CONCEPTS", raising=False)
        proc = _bare_processor()
        text = "Deploy on NVIDIA NIM and evaluate with RAGAS using FAISS."
        item = {"key_concepts": ["vector-store", "knowledge-base"]}
        tags = proc._extract_concept_tags(text, item)
        for anchor in ("nim", "ragas", "faiss", "rag"):
            assert anchor not in tags, f"flag OFF must not seed {anchor!r}"
        # Legacy untrusted key_concepts still flow through unchanged.
        assert "vector-store" in tags
        assert "knowledge-base" in tags

    def test_real_multiword_concepts_survive_dedup(self, monkeypatch):
        # Flag ON with anchors present: legitimate multi-word concepts
        # that merely embed an anchor word are NOT dropped by the
        # fragment-superstring dedup.
        monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")
        proc = _bare_processor()
        text = "Build a RAG pipeline over a FAISS vector store knowledge base."
        item = {
            "key_concepts": [
                "vector-store",
                "knowledge-base",
                "faiss-vector-store",  # 3 words, embeds 'faiss' anchor → keep
            ]
        }
        tags = proc._extract_concept_tags(text, item)
        assert "vector-store" in tags
        assert "knowledge-base" in tags
        assert "faiss-vector-store" in tags
        # Anchors themselves still land.
        assert "faiss" in tags
        assert "rag" in tags
