"""Wave 84 retriever tests: CamelCase tokenization + section-heading injection.

Pin the two changes to ``LibV2/tools/libv2/retriever.py`` made to fix the
shacl_node_shape ranking probe (chunk_00147 ranked #21 against an
unrelated logical-composition chunk because the section heading was not
indexed and ``NodeShape`` did not align with ``node shape`` in prose):

  1. ``_expand_camel_case`` splits CamelCase boundaries before tokenize
     and emits BOTH the joined and split forms so URI form ``NodeShape``
     and prose form ``node shape`` cross-match.
  2. ``LazyBM25._doc_text_for_indexing`` prepends ``source.section_heading``
     repeated ``HEADING_REPETITIONS`` times (capped at ``HEADING_TOKEN_CAP``
     words) so heading terms contribute to BM25.

Together these lifted Hit@1 from 0.467 to 0.533 (BM25 alone) and 0.600
(BM25 + chunk-type intent prior) on the 15-query rdf-shacl-551-2 probe
set; MRR from 0.576 to 0.646 (BM25) and 0.674 (intent).
"""

from __future__ import annotations

from LibV2.tools.libv2.retriever import (
    LazyBM25,
    _expand_camel_case,
    tokenize,
)


# ---------------------------------------------------------------------------
# CamelCase expansion
# ---------------------------------------------------------------------------


class TestExpandCamelCase:
    def test_camelcase_word_emits_joined_plus_split(self):
        # NodeShape → "NodeShape Node Shape" so tokenize sees both forms.
        out = _expand_camel_case("NodeShape")
        assert "NodeShape" in out
        assert "Node Shape" in out

    def test_pure_lowercase_unchanged(self):
        # No CamelCase boundary → no expansion, no double-emission.
        assert _expand_camel_case("node shape") == "node shape"
        assert _expand_camel_case("rdf-schema") == "rdf-schema"

    def test_uri_form_with_colon_handled(self):
        # The URI prefix `sh:` doesn't break the CamelCase boundary inside.
        out = _expand_camel_case("sh:PropertyShape")
        assert "sh:PropertyShape" in out
        # The expansion should split Property/Shape too.
        assert "sh:Property Shape" in out

    def test_three_word_camelcase(self):
        # subClassOf has two CamelCase boundaries: sub|Class|Of.
        out = _expand_camel_case("subClassOf")
        assert "subClassOf" in out
        assert "sub Class Of" in out

    def test_empty_string_handled(self):
        assert _expand_camel_case("") == ""

    def test_already_split_unchanged(self):
        assert _expand_camel_case("Node Shape") == "Node Shape"

    def test_plural_camelcase(self):
        # NodeShapes → "NodeShapes Node Shapes" — plural preserved.
        out = _expand_camel_case("NodeShapes")
        assert "NodeShapes" in out
        assert "Node Shapes" in out


# ---------------------------------------------------------------------------
# tokenize() — CamelCase support
# ---------------------------------------------------------------------------


class TestTokenizeCamelCase:
    def test_query_nodeshape_produces_joined_and_split_tokens(self):
        # The audit-named query case: "NodeShape" must produce
        # 'nodeshape' AND 'node' + 'shape' so it can match both
        # URI-style chunks (sh:NodeShape) and prose-style chunks
        # ("node shape").
        toks = tokenize("Define a SHACL NodeShape", structured_tokens=True)
        assert "nodeshape" in toks
        assert "node" in toks
        assert "shape" in toks
        assert "shacl" in toks

    def test_uri_form_tokenizes_with_split_forms(self):
        # sh:NodeShape in body text → both forms in the index.
        toks = tokenize("a sh:NodeShape ;", structured_tokens=True)
        assert "nodeshape" in toks
        assert "node" in toks
        assert "shape" in toks

    def test_camelcase_disabled_under_structured_tokens_false(self):
        # Back-compat path: when structured_tokens=False (legacy regex),
        # CamelCase split is NOT applied — preserves pre-Worker-J shape.
        toks = tokenize("NodeShape", structured_tokens=False)
        # legacy path lowercases + word-boundary split → single token
        assert toks == ["nodeshape"]

    def test_subclassof_tokenization(self):
        toks = tokenize("rdfs:subClassOf entailment", structured_tokens=True)
        assert "subclassof" in toks
        assert "sub" in toks or "class" in toks  # at least one split component
        assert "entailment" in toks


# ---------------------------------------------------------------------------
# LazyBM25 — section heading injection
# ---------------------------------------------------------------------------


class TestHeadingInjection:
    def test_heading_for_indexing_repeats_capped_heading(self):
        index = LazyBM25.__new__(LazyBM25)
        index.HEADING_REPETITIONS = LazyBM25.HEADING_REPETITIONS
        index.HEADING_TOKEN_CAP = LazyBM25.HEADING_TOKEN_CAP
        chunk = {"source": {"section_heading": "Node Shapes and Property Shapes"}}
        out = index._heading_for_indexing(chunk)
        # 3× repetition of the 5-word heading (under cap) = 15 words.
        words = out.split()
        assert len(words) == 5 * LazyBM25.HEADING_REPETITIONS
        assert words[:5] == ["Node", "Shapes", "and", "Property", "Shapes"]

    def test_long_heading_capped(self):
        index = LazyBM25.__new__(LazyBM25)
        index.HEADING_REPETITIONS = LazyBM25.HEADING_REPETITIONS
        index.HEADING_TOKEN_CAP = LazyBM25.HEADING_TOKEN_CAP
        long_heading = " ".join(["w"] * 20)  # 20 words, way over cap
        chunk = {"source": {"section_heading": long_heading}}
        out = index._heading_for_indexing(chunk)
        words = out.split()
        # Capped at HEADING_TOKEN_CAP (8) × HEADING_REPETITIONS (3) = 24 words.
        assert len(words) == LazyBM25.HEADING_TOKEN_CAP * LazyBM25.HEADING_REPETITIONS

    def test_missing_heading_returns_empty(self):
        index = LazyBM25.__new__(LazyBM25)
        index.HEADING_REPETITIONS = LazyBM25.HEADING_REPETITIONS
        index.HEADING_TOKEN_CAP = LazyBM25.HEADING_TOKEN_CAP
        assert index._heading_for_indexing({}) == ""
        assert index._heading_for_indexing({"source": {}}) == ""
        assert index._heading_for_indexing({"source": {"section_heading": ""}}) == ""

    def test_doc_text_for_indexing_prepends_heading(self):
        # Build a real index over a single chunk to confirm the indexed
        # text carries the heading.
        chunks = [
            {
                "id": "c1",
                "text": "body about node shapes",
                "source": {"section_heading": "Node Shapes and Property Shapes"},
            }
        ]
        index = LazyBM25(chunks, use_retrieval_text=True, structured_tokens=True)
        # The heading tokens (lowercased) must appear with extra frequency
        # in the indexed token list — 3× from injection + 1× from body
        # body says "node shapes" with `node` and `shapes` tokens.
        toks = index.doc_tokens[0]
        assert toks.count("node") >= 3, f"heading injection failed: {toks}"
        assert toks.count("shapes") >= 3 + 1  # heading × 3 + body × 1


class TestNodeShapeRankingRegression:
    """The audit's load-bearing query: 'Define a SHACL NodeShape'.

    Pin that the indexed text for chunk_00147 (the definition chunk
    whose heading is 'Node Shapes and Property Shapes') gets BOTH:
      - the heading repeated 3× (so 'node', 'shapes', 'property' get
        tf-boost from the heading)
      - the prose body's 'node shape' tokens
    AND the query 'NodeShape' tokenizes to ['nodeshape', 'node',
    'shape'] so it matches both URI form (chunk_00175 body) and prose
    form (chunk_00147 body + heading).

    This is a tokenization-and-indexing regression test, not a ranking
    test — ranking depends on global IDF over the whole corpus and is
    pinned by the live retrieval-compare run, not by unit fixtures.
    """

    def test_definition_chunk_heading_terms_present_in_index(self):
        # Realistic chunk shape mirroring chunk_00147.
        chunk = {
            "id": "rdf_shacl_551_chunk_00147",
            "text": (
                "sh:Shape is the SHACL superclass; it has exactly two "
                "subclasses, sh:NodeShape and sh:PropertyShape. Both "
                "represent constraints, both can declare targets..."
            ),
            "source": {"section_heading": "Node Shapes and Property Shapes"},
        }
        index = LazyBM25([chunk], use_retrieval_text=True, structured_tokens=True)
        toks = index.doc_tokens[0]
        # Heading must contribute (3× injection): node, shapes, property each ≥ 3
        assert toks.count("node") >= 3
        assert toks.count("shapes") >= 3
        # CamelCase split adds 'nodeshape', 'propertyshape' AND 'node', 'shape'.
        assert "nodeshape" in toks
        assert "propertyshape" in toks
        assert "shape" in toks

    def test_query_nodeshape_matches_definition_chunk_body_terms(self):
        # The query "Define a SHACL NodeShape" must produce tokens
        # that overlap chunk_00147's indexed text.
        toks = tokenize("Define a SHACL NodeShape", structured_tokens=True)
        # Query tokens
        assert "nodeshape" in toks  # matches URI form sh:NodeShape
        assert "node" in toks  # matches heading "Node Shapes" + prose "node shape"
        assert "shape" in toks  # matches prose "node shape"
        assert "shacl" in toks  # matches body
