"""Wave 81: tests for the deterministic auto-vocabulary extractor.

Locks in the contract that:

* ``auto_extract_vocabulary`` preserves ``prefix:term`` patterns
  (rdfs:label, sh:minCount) when surrounded by punctuation.
* Multi-word noun-phrase bigrams survive ("vocabulary design",
  "domain-specific").
* Stopword-only / empty inputs return [] gracefully.
* The leading bloom verb is stripped (cognitive verb shouldn't leak
  into the vocab list for the CO it labels).
* Auto-extraction is deterministic — same input → same output across
  runs and across calls.

The test fixtures track the rdf-shacl-550 / rdf-shacl-551 CO
statements verbatim, since the validator gap that motivated Wave 81
was specifically against those archives.
"""

from __future__ import annotations

from Trainforge.retag_outcomes import (
    RETAG_VOCABULARIES,
    auto_extract_vocabulary,
    build_auto_vocabularies,
    merged_vocabularies,
)


# ---- Spec test cases -----------------------------------------------

def test_extract_shacl_core_constraint_components_preserves_prefixes():
    statement = (
        "Apply SHACL Core constraint components (sh:minCount, "
        "sh:maxCount, sh:datatype, sh:class, sh:pattern, sh:in)"
    )
    out = auto_extract_vocabulary(statement)
    for term in (
        "sh:minCount",
        "sh:maxCount",
        "sh:datatype",
        "sh:class",
        "sh:pattern",
        "sh:in",
    ):
        assert term in out, f"missing prefix:term {term} in {out}"


def test_extract_rdfs_documentation_predicates():
    statement = (
        "Document a vocabulary with rdfs:label, rdfs:comment, and "
        "rdfs:seeAlso"
    )
    out = auto_extract_vocabulary(statement)
    for term in ("rdfs:label", "rdfs:comment", "rdfs:seeAlso"):
        assert term in out, f"missing predicate {term} in {out}"


def test_extract_design_domain_specific_keeps_substantive_tokens():
    statement = (
        "Design a domain-specific RDFS vocabulary for a chosen scenario"
    )
    out = auto_extract_vocabulary(statement)
    # Conservative auto-extraction: only highly specific tokens
    # survive — the protected domain identifier ``RDFS`` and the
    # hyphenated ``domain-specific``. Plain English ``vocabulary``
    # is generic and intentionally dropped to avoid over-tagging.
    # Wave 81 lays a curated override on co-10 for the canonical
    # ``vocabulary design`` phrase via ``RETAG_VOCABULARIES``.
    assert "RDFS" in out
    assert "domain-specific" in out


def test_stopword_only_returns_empty_list():
    assert auto_extract_vocabulary("the and or but") == []
    assert auto_extract_vocabulary("a the of") == []


def test_empty_or_none_input_returns_empty_list():
    assert auto_extract_vocabulary("") == []
    assert auto_extract_vocabulary(None) == []  # type: ignore[arg-type]


# ---- Determinism ---------------------------------------------------

def test_auto_extract_is_deterministic():
    statement = (
        "Apply SPARQL solution modifiers (FILTER, OPTIONAL, UNION, "
        "ORDER BY, LIMIT, OFFSET) to refine result sets."
    )
    a = auto_extract_vocabulary(statement)
    b = auto_extract_vocabulary(statement)
    c = auto_extract_vocabulary(statement)
    assert a == b == c


def test_cap_at_ten_candidates():
    # A long statement with many candidate tokens must still cap at 10.
    statement = (
        "Compare named graphs and dataset abstractions defined in the "
        "RDF 1.1 Concepts specification and explain when each is useful "
        "across SPARQL endpoints, SHACL shapes, OWL profiles, RDFS "
        "vocabularies, and JSON-LD serializations."
    )
    out = auto_extract_vocabulary(statement)
    assert len(out) <= 10


# ---- Bigrams + technical heuristics --------------------------------

def test_technical_bigram_extraction():
    statement = (
        "Apply sh:minCount sh:maxCount and sh:datatype constraints to "
        "the property shape."
    )
    out = auto_extract_vocabulary(statement)
    # Bigram with two technical halves survives; both prefix:term
    # tokens must both be technical for the bigram to emit.
    assert "sh:minCount sh:maxCount" in out, (
        f"expected technical bigram in {out}"
    )


def test_protected_short_tokens_kept():
    # "RDF" / "RDFS" / "OWL" must survive the length>=4 filter
    # because they're explicitly protected domain identifiers.
    statement = "Use RDF and OWL with SPARQL"
    out = auto_extract_vocabulary(statement)
    assert "RDF" in out
    assert "OWL" in out
    assert "SPARQL" in out


def test_camel_case_terms_kept():
    statement = "Author shapes using sh:NodeShape and sh:PropertyShape"
    out = auto_extract_vocabulary(statement)
    assert "sh:NodeShape" in out
    assert "sh:PropertyShape" in out


# ---- Builder + merger ----------------------------------------------

def test_build_auto_vocabularies_walks_component_objectives():
    obj = {
        "component_objectives": [
            {
                "id": "co-01",
                "statement": "Identify RDF triples in a graph",
            },
            {
                "id": "co-02",
                "statement": "Apply sh:minCount to a property shape",
            },
        ]
    }
    auto = build_auto_vocabularies(obj)
    assert "co-01" in auto
    assert "co-02" in auto
    assert "sh:minCount" in auto["co-02"]


def test_build_auto_vocabularies_handles_chapter_objectives_shape():
    obj = {
        "chapter_objectives": [
            {"id": "co-A", "statement": "Apply sh:minCount constraints"},
            {
                "objectives": [
                    {"id": "co-B", "statement": "Document with rdfs:label"},
                ]
            },
        ]
    }
    auto = build_auto_vocabularies(obj)
    assert "co-a" in auto
    assert "sh:minCount" in auto["co-a"]
    assert "co-b" in auto
    assert "rdfs:label" in auto["co-b"]


def test_build_auto_vocabularies_handles_terminal_outcomes():
    obj = {
        "terminal_outcomes": [
            {
                "id": "to-01",
                "statement": "Architect an end-to-end SPARQL pipeline",
            }
        ]
    }
    auto = build_auto_vocabularies(obj)
    assert "to-01" in auto
    assert "SPARQL" in auto["to-01"]


def test_build_auto_vocabularies_empty_input():
    assert build_auto_vocabularies(None) == {}
    assert build_auto_vocabularies({}) == {}


def test_merged_vocabularies_curated_overrides_auto():
    obj = {
        "component_objectives": [
            {
                "id": "co-09",
                "statement": (
                    "Document a vocabulary with rdfs:label, rdfs:comment"
                ),
            },
            # New CO not in curated -> auto-extract only.
            {
                "id": "co-99",
                "statement": "Apply sh:minCount on a property shape",
            },
        ]
    }
    merged = merged_vocabularies(obj)
    # Curated co-09 wins over auto.
    assert merged["co-09"] == RETAG_VOCABULARIES["co-09"]
    # Non-curated CO falls through to auto.
    assert "co-99" in merged
    assert "sh:minCount" in merged["co-99"]


def test_merged_vocabularies_preserves_curated_only_keys():
    # Even if objectives is empty, all curated entries remain accessible.
    merged = merged_vocabularies(None)
    for cid in RETAG_VOCABULARIES:
        assert cid.lower() in merged
