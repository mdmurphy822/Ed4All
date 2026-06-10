"""Wave 82 tests for the acronym-preserving label helper."""

from __future__ import annotations

import pytest

from lib.ontology.labels import (
    KNOWN_ACRONYMS,
    slug_to_label,
    titlecase_with_acronyms,
)


# ---------------------------------------------------------------------------
# Audit-case reproductions
# ---------------------------------------------------------------------------


class TestAuditReproductions:
    def test_owl_2_rl_renders_uppercase(self):
        # Audit reported "Owl 2 Rl" — must now be "OWL 2 RL".
        assert slug_to_label("owl-2-rl") == "OWL 2 RL"

    def test_owl_2_dl_renders_uppercase(self):
        assert slug_to_label("owl-2-dl") == "OWL 2 DL"

    def test_owl_2_el_ql_render_uppercase(self):
        assert slug_to_label("owl-2-el") == "OWL 2 EL"
        assert slug_to_label("owl-2-ql") == "OWL 2 QL"

    def test_rdfs_renders_uppercase(self):
        assert slug_to_label("rdfs") == "RDFS"
        assert slug_to_label("rdfs-class") == "RDFS Class"

    def test_sparql_renders_uppercase(self):
        assert slug_to_label("sparql") == "SPARQL"
        assert slug_to_label("sparql-query") == "SPARQL Query"

    def test_shacl_renders_uppercase(self):
        assert slug_to_label("shacl") == "SHACL"
        assert slug_to_label("shacl-shape") == "SHACL Shape"


# ---------------------------------------------------------------------------
# Hyphenated tokens (json-ld is the canonical example)
# ---------------------------------------------------------------------------


class TestHyphenatedTokens:
    def test_json_ld_both_segments_uppercase(self):
        assert titlecase_with_acronyms("json-ld") == "JSON-LD"

    def test_n_triples_n_lowercased_n_not_acronym_in_set(self):
        # "n" alone isn't in KNOWN_ACRONYMS — render lowercase title.
        # "Triples" is title-cased.
        assert titlecase_with_acronyms("n-triples") == "N-Triples"

    def test_mixed_hyphen_acronym_segments(self):
        # url-shortener: URL is acronym, shortener is title.
        assert titlecase_with_acronyms("url-shortener") == "URL-Shortener"


# ---------------------------------------------------------------------------
# Plain title-case (non-acronym tokens)
# ---------------------------------------------------------------------------


class TestPlainTitleCase:
    def test_no_acronyms_normal_titlecase(self):
        assert titlecase_with_acronyms("blank node") == "Blank Node"

    def test_slug_to_label_replaces_hyphens(self):
        assert slug_to_label("blank-node") == "Blank Node"

    def test_acronym_followed_by_word(self):
        assert slug_to_label("rdf-graph") == "RDF Graph"

    def test_word_followed_by_acronym(self):
        assert slug_to_label("named-graph") == "Named Graph"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCompoundAcronymOverrides:
    """Slug-level overrides for compound acronyms whose canonical form keeps the hyphen."""

    def test_json_ld_preserves_hyphen(self):
        assert slug_to_label("json-ld") == "JSON-LD"

    def test_n_triples_preserves_hyphen(self):
        assert slug_to_label("n-triples") == "N-Triples"

    def test_n_quads_preserves_hyphen(self):
        assert slug_to_label("n-quads") == "N-Quads"

    def test_rdf_xml_preserves_slash(self):
        # RDF/XML canonical form uses a slash, not a hyphen — round-trip
        # the slug to its W3C-spec rendering.
        assert slug_to_label("rdf-xml") == "RDF/XML"

    def test_compound_prefix_does_not_match_override(self):
        # "json-ld-context" is NOT in the override table, so it falls
        # through to the standard hyphen-stripping → "JSON LD Context".
        # Exact-slug matching prevents over-broad rewrites.
        assert slug_to_label("json-ld-context") == "JSON LD Context"


class TestEdgeCases:
    def test_empty_input(self):
        assert slug_to_label("") == ""
        assert titlecase_with_acronyms("") == ""

    def test_non_string_input_returns_empty(self):
        assert titlecase_with_acronyms(None) == ""  # type: ignore[arg-type]
        assert slug_to_label(None) == ""  # type: ignore[arg-type]

    def test_already_uppercase_acronym_passes_through(self):
        assert titlecase_with_acronyms("OWL") == "OWL"

    def test_already_correctly_cased_input_idempotent(self):
        assert titlecase_with_acronyms("OWL 2 RL") == "OWL 2 RL"
        assert slug_to_label("OWL 2 RL") == "OWL 2 RL"  # no hyphens, just space-cased

    def test_multi_word_with_no_acronyms(self):
        assert slug_to_label("first-class-citizen") == "First Class Citizen"


# ---------------------------------------------------------------------------
# Acronym set integrity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Concept-label artifact fixes (rdf-shacl audit Section F follow-up)
# ---------------------------------------------------------------------------


@pytest.fixture
def normalize_labels_on(monkeypatch):
    """Enable the opt-in run-together-compound resplit path."""
    monkeypatch.setenv("TRAINFORGE_NORMALIZE_LABELS", "true")


class TestRunTogetherResplit:
    """Curated re-splitting of camelCase-collapsed compounds (flag-gated)."""

    def test_runnableassign_resplit(self, normalize_labels_on):
        assert slug_to_label("runnableassign") == "Runnable Assign"

    def test_checkpointresume_resplit(self, normalize_labels_on):
        assert slug_to_label("checkpointresume") == "Checkpoint Resume"

    def test_ragas_evaluator_chain_resplit_with_acronym(self, normalize_labels_on):
        assert slug_to_label("ragasevaluatorchain") == "RAGAS Evaluator Chain"

    def test_retrieval_augmentation_resplit(self, normalize_labels_on):
        # "retrievalaugmentation evaluation" → "Retrieval Augmentation Evaluation"
        assert (
            slug_to_label("retrievalaugmentation-evaluation")
            == "Retrieval Augmentation Evaluation"
        )

    def test_resplit_default_off_leaves_compound(self, monkeypatch):
        # Default off: compound stays one token (byte-stable). Title-cased.
        monkeypatch.delenv("TRAINFORGE_NORMALIZE_LABELS", raising=False)
        assert slug_to_label("runnableassign") == "Runnableassign"


class TestBrandResplit:
    """Curated proper-noun brand splits (RAG-course audit, flag-gated)."""

    def test_langgraph(self, normalize_labels_on):
        assert slug_to_label("langgraph") == "LangGraph"

    def test_langserve(self, normalize_labels_on):
        assert slug_to_label("langserve") == "LangServe"

    def test_llamaindex(self, normalize_labels_on):
        assert slug_to_label("llamaindex") == "LlamaIndex"

    def test_stroutputparser(self, normalize_labels_on):
        assert slug_to_label("stroutputparser") == "StrOutputParser"

    def test_retrievertool(self, normalize_labels_on):
        assert slug_to_label("retrievertool") == "RetrieverTool"

    def test_smolagents(self, normalize_labels_on):
        assert slug_to_label("smolagents") == "SmolAgents"

    def test_centralorchestrator_agent(self, normalize_labels_on):
        # Multi-word brand split feeds through acronym-aware title-casing.
        assert slug_to_label("centralorchestrator-agent") == "Central Orchestrator Agent"

    def test_dockerrouter(self, normalize_labels_on):
        assert slug_to_label("dockerrouter") == "Docker Router"

    def test_brand_resplit_default_off_byte_stable(self, monkeypatch):
        # Flag off: run-together brands stay one title-cased token.
        monkeypatch.delenv("TRAINFORGE_NORMALIZE_LABELS", raising=False)
        assert slug_to_label("langgraph") == "Langgraph"
        assert slug_to_label("stroutputparser") == "Stroutputparser"
        assert slug_to_label("dockerrouter") == "Dockerrouter"


class TestNvidiaEmbedAndAcronyms:
    """NV-Embed slug override + new PCA/VQ/NV acronyms (unconditional)."""

    def test_nvidia_nv_embed_vq_full_slug(self):
        # Slug override pins the canonical NVIDIA model name; the trailing
        # "-vq" quantization qualifier is folded into the curated label.
        assert slug_to_label("nvidia-nv-embed-vq") == "NVIDIA NV-Embed"

    def test_nv_embed_hyphen_segment_uppercase(self):
        # NV is now a known acronym, so a hyphen-preserving render of
        # "nv-embed" uppercases the NV segment and keeps the hyphen.
        # (slug_to_label strips hyphens to spaces, so the hyphen-kept
        # form is exercised via titlecase_with_acronyms, the same path
        # that backs the nvidia-nv-embed-vq slug override.)
        assert titlecase_with_acronyms("nv-embed") == "NV-Embed"

    def test_pca_acronym(self):
        assert "pca" in KNOWN_ACRONYMS
        assert slug_to_label("principal-component-analysis-pca") == (
            "Principal Component Analysis PCA"
        )

    def test_vq_acronym(self):
        assert "vq" in KNOWN_ACRONYMS
        assert slug_to_label("vector-quantization-vq") == "Vector Quantization VQ"


class TestAcronymAndBrandCasing:
    """Acronym + mixed-case brand casing applies unconditionally."""

    def test_ragas_acronym(self):
        assert slug_to_label("ragas") == "RAGAS"

    def test_faiss_vector_store(self):
        assert slug_to_label("faiss-vector-store") == "FAISS Vector Store"

    def test_ragas_short_for_rag(self):
        assert slug_to_label("ragas-short-for-rag") == "RAGAS Short For RAG"

    def test_nvidia_nim_gpu(self):
        assert slug_to_label("nvidia-nim") == "NVIDIA NIM"
        assert slug_to_label("gpu-inference") == "GPU Inference"

    def test_openai_brand_casing(self):
        assert slug_to_label("openai") == "OpenAI"

    def test_chatgpt_brand_casing(self):
        assert slug_to_label("chatgpt") == "ChatGPT"


class TestPossessiveMangling:
    """Possessive 's on a known acronym/brand renders correctly."""

    def test_openais_chatgpt(self):
        # "openais chatgpt" → "OpenAI's ChatGPT" (slug drops apostrophe).
        assert slug_to_label("openais-chatgpt") == "OpenAI's ChatGPT"

    def test_ordinary_plural_unaffected(self):
        # "graphs" is not a known acronym stem — stays a normal plural.
        assert slug_to_label("knowledge-graphs") == "Knowledge Graphs"


class TestMisspellingFix:
    """exercice → Exercise correction applies unconditionally."""

    def test_final_exercice(self):
        assert slug_to_label("final-exercice") == "Final Exercise"

    def test_bare_exercice(self):
        assert slug_to_label("exercice") == "Exercise"


class TestUnchangedNormalSlugs:
    """Regression guard: ordinary slugs render exactly as before."""

    def test_knowledge_base_unchanged(self):
        assert slug_to_label("knowledge-base") == "Knowledge Base"

    def test_knowledge_base_unchanged_with_flag(self, normalize_labels_on):
        assert slug_to_label("knowledge-base") == "Knowledge Base"

    def test_blank_node_unchanged(self):
        assert slug_to_label("blank-node") == "Blank Node"


class TestAcronymSet:
    def test_w3c_standards_present(self):
        for a in ["rdf", "rdfs", "owl", "shacl", "sparql"]:
            assert a in KNOWN_ACRONYMS, f"missing W3C acronym {a}"

    def test_owl_2_profiles_present(self):
        for a in ["rl", "el", "ql", "dl"]:
            assert a in KNOWN_ACRONYMS, f"missing OWL 2 profile {a}"

    def test_uri_iri_present(self):
        for a in ["uri", "iri", "url"]:
            assert a in KNOWN_ACRONYMS, f"missing identifier acronym {a}"

    def test_acronyms_are_lowercased_in_storage(self):
        # Storage form is lowercase so lookup matches case-insensitively.
        for a in KNOWN_ACRONYMS:
            assert a == a.lower(), f"non-lowercase entry: {a!r}"
