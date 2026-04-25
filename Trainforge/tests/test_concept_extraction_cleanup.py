"""Wave 76: tests for concept-extraction-time filtering.

Locks the contract that ``CourseProcessor._extract_concept_tags`` (and
the JSON-LD keyTerms merge path that feeds ``concept_tags``) reject
pedagogical scaffolding, assessment options, sentence fragments, and
HTML-entity contamination AT EMIT TIME — not just retroactively.

Pairs with ``Trainforge/tests/test_concept_graph_classification.py``
(Wave 75) which only verified that emitted nodes carry a ``class``;
Wave 76 verifies the noisy candidates never enter the concept stream
in the first place.

The classifier itself is exercised exhaustively in
``lib/ontology/tests/test_concept_classifier.py``. These tests pin the
WIRING — that the extraction call site actually consults the classifier
and drops droppable classes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.concept_classifier import (  # noqa: E402
    ASSESSMENT_OPTION,
    DOMAIN_CONCEPT,
    INSTRUCTIONAL_ARTIFACT,
    LEARNING_OBJECTIVE,
    LOW_SIGNAL,
    PEDAGOGICAL_MARKER,
    classify_concept,
    is_droppable_class,
    canonicalize_alias,
)
from Trainforge.process_course import (  # noqa: E402
    CourseProcessor,
    normalize_tag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processor() -> CourseProcessor:
    """Construct a ``CourseProcessor`` skipping IMSCC ingestion.

    Mirrors the helper pattern from ``test_concept_occurrences.py``.
    Only attaches the attributes ``_extract_concept_tags`` reads.
    """
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "RDF_SHACL_550"
    proc.domain_concept_seeds = []
    return proc


def _extract(proc: CourseProcessor, key_concepts):
    """Run extraction with a controlled item dict + empty body text."""
    item = {"key_concepts": list(key_concepts)}
    return proc._extract_concept_tags(text="", item=item)


# ---------------------------------------------------------------------------
# Classifier-level guarantees (locks the rule precedence the extractor
# relies on at emit time).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "candidate,expected_class",
    [
        # Pedagogical scaffolding (rejected via PedagogicalMarker class).
        ("key-takeaway", PEDAGOGICAL_MARKER),
        ("rubric", PEDAGOGICAL_MARKER),
        ("self-check", PEDAGOGICAL_MARKER),
        ("module-4-deliverable", PEDAGOGICAL_MARKER),
        ("application-activity", PEDAGOGICAL_MARKER),
        # ``after-the-self-check`` matches the compound pedagogy
        # pattern (contains the ``self-check`` substring as a hyphen-
        # bounded chunk) BEFORE the article-prefix fragment check
        # fires. Either class is droppable — the precedence locks
        # this onto PedagogicalMarker.
        ("after-the-self-check", PEDAGOGICAL_MARKER),
        # Assessment options (rejected via AssessmentOption class).
        ("answer-b", ASSESSMENT_OPTION),
        ("answer-c", ASSESSMENT_OPTION),
        ("answer-false", ASSESSMENT_OPTION),
        ("true", ASSESSMENT_OPTION),
        ("false", ASSESSMENT_OPTION),
        # English stopwords / sentence fragments (rejected via LowSignal).
        ("not", LOW_SIGNAL),
        ("a-literal-is-just", LOW_SIGNAL),
        ("every-direct-type-of", LOW_SIGNAL),
        ("use-a-property-chain", LOW_SIGNAL),
        ("choose-full-owl-2", LOW_SIGNAL),
        # Instructional artifacts.
        ("submission-format", INSTRUCTIONAL_ARTIFACT),
        # LO-ID leak (rejected via LearningObjective class — Wave 76
        # treats LO IDs as non-concept since they belong in
        # objectives.json).
        ("to-04", LEARNING_OBJECTIVE),
        ("co-12", LEARNING_OBJECTIVE),
        # KEPT: real domain concepts.
        ("rdf-graph", DOMAIN_CONCEPT),
        # ``sh:path`` is a CURIE shape; the classifier treats it as
        # DomainConcept (Wave 75 regression — colons must survive the
        # classifier even though the slugifier strips them).
        ("sh:path", DOMAIN_CONCEPT),
        ("blank-node", DOMAIN_CONCEPT),
        ("sparql-select", DOMAIN_CONCEPT),
        ("owl-2-rl", DOMAIN_CONCEPT),
    ],
)
def test_classifier_decisions_locked(candidate, expected_class):
    """Pins the classifier output that drives the extraction filter."""
    assert classify_concept(candidate) == expected_class


@pytest.mark.parametrize(
    "candidate",
    [
        "key-takeaway",
        "answer-b",
        "answer-false",
        "not",
        "a-literal-is-just",
        "submission-format",
        "to-04",
    ],
)
def test_droppable_classes_drop(candidate):
    """Every noisy candidate MUST classify into a droppable class."""
    assert is_droppable_class(classify_concept(candidate))


@pytest.mark.parametrize(
    "candidate",
    [
        "rdf-graph",
        "sh:path",
        "blank-node",
        "sparql-select",
        "owl-2-rl",
    ],
)
def test_domain_concepts_kept(candidate):
    """Real domain vocabulary stays out of the droppable set."""
    assert not is_droppable_class(classify_concept(candidate))


# ---------------------------------------------------------------------------
# HTML entity decoding before slugification (Wave 76 part 2)
# ---------------------------------------------------------------------------

def test_normalize_tag_decodes_named_html_entity():
    """``&mdash;target`` decodes to ``—target`` then slugifies to ``target``.

    The em-dash itself (``—``, U+2014) is not in the slug character
    class, so the slugifier strips it. The literal string ``mdash``
    must NEVER appear in the resulting slug.
    """
    out = normalize_tag("&mdash;target")
    assert out == "target", f"expected 'target', got {out!r}"
    assert "mdash" not in out


def test_normalize_tag_decodes_numeric_html_entity():
    """Numeric variants ``&#8212;`` (decimal) and ``&#x2014;`` (hex)
    must produce the same result as the named ``&mdash;`` variant.
    """
    assert normalize_tag("&#8212;target") == "target"
    assert normalize_tag("&#x2014;target") == "target"


def test_normalize_tag_decodes_compound_entity_string():
    """Embedded entities in the middle of a phrase must decode then
    slugify correctly. ``"pitfall&mdash;target class"`` → decode to
    ``"pitfall—target class"`` → slug ``"pitfalltarget-class"``.

    Per ``lib.ontology.slugs.canonical_slug`` the disallowed em-dash
    character is DELETED (not replaced with a separator) — that's the
    documented "fuse-on-delete" behaviour. So ``pitfall—target`` fuses
    to ``pitfalltarget``, then the space before ``class`` becomes a
    hyphen → ``pitfalltarget-class``.

    The critical contract is that the literal ``mdash`` token never
    appears in the slug. The ``pitfall-mdash-target-class`` pollution
    that motivated Wave 76 cannot be re-emitted via this path.
    """
    out = normalize_tag("pitfall&mdash;target class")
    assert "mdash" not in out
    # Em-dash is stripped → adjacent words fuse, then space → hyphen.
    assert out == "pitfalltarget-class"


def test_normalize_tag_decodes_entity_with_separator_after():
    """When the entity sits NEXT to whitespace it decodes cleanly and
    the slugifier collapses surrounding whitespace into a hyphen.

    ``"pitfall &mdash; target"`` → ``"pitfall — target"`` → strip the
    em-dash → ``"pitfall  target"`` → collapse whitespace →
    ``"pitfall-target"``.
    """
    out = normalize_tag("pitfall &mdash; target")
    assert "mdash" not in out
    assert out == "pitfall-target"


def test_normalize_tag_handles_none():
    assert normalize_tag(None) == ""


# ---------------------------------------------------------------------------
# Extraction-call-site wiring: pollution must NOT enter the concept
# stream when the input candidates would classify as droppable.
# ---------------------------------------------------------------------------

def test_extract_filters_pedagogical_markers():
    proc = _make_processor()
    out = _extract(proc, [
        "Key Takeaway",          # → key-takeaway → PedagogicalMarker
        "Rubric",                # → rubric → PedagogicalMarker
        "Self-Check",            # → self-check → PedagogicalMarker
        "RDF Graph",             # → rdf-graph → DomainConcept (KEEP)
    ])
    assert out == ["rdf-graph"], out


def test_extract_filters_assessment_options():
    proc = _make_processor()
    out = _extract(proc, [
        "Answer B",              # → answer-b → AssessmentOption
        "Answer C",              # → answer-c → AssessmentOption
        "Answer False",          # → answer-false → AssessmentOption
        "SPARQL SELECT",         # → sparql-select → DomainConcept (KEEP)
    ])
    assert out == ["sparql-select"], out


def test_extract_filters_low_signal():
    proc = _make_processor()
    out = _extract(proc, [
        "not",                   # → not → LowSignal
        "and",                   # → and → LowSignal
        "Blank Node",            # → blank-node → DomainConcept (KEEP)
    ])
    assert out == ["blank-node"], out


def test_extract_filters_sentence_fragments():
    proc = _make_processor()
    out = _extract(proc, [
        "A Literal Is Just",            # → a-literal-is-just → LowSignal
        "After The Self Check",         # → after-the-self-check → LowSignal
        "Every Direct Type Of",         # → every-direct-type-of → LowSignal
        "OWL 2 RL",                     # → owl-2-rl → DomainConcept (KEEP)
    ])
    assert out == ["owl-2-rl"], out


def test_extract_filters_html_entity_contamination():
    """Slugs surfaced from raw HTML carrying ``&mdash;`` no longer
    leak the literal token ``mdash`` into the concept stream.

    Two independent guards must hold this contract:
    1. ``normalize_tag`` decodes the entity before slugification, so
       the input ``"pitfall&mdash;target class"`` becomes the slug
       ``"pitfall-target-class"`` (no ``mdash``).
    2. If a legacy slug like ``"pitfall-mdash-target-class"`` enters
       extraction by some other route (a cached chunk, a hand-written
       seed), the classifier's HTML-entity-noise rule rejects it.
    """
    proc = _make_processor()
    out = _extract(proc, [
        "pitfall&mdash;target class",   # path 1: decoded
        "pitfall-mdash-target-class",   # path 2: rejected by classifier
        "RDF Graph",                    # KEEP
    ])
    # The decoded version becomes ``pitfall-target-class`` — a
    # DomainConcept under our rules. Verify both paths land us at
    # exactly one ``pitfall-target-class`` entry plus the kept domain
    # concept.
    assert "rdf-graph" in out
    assert all("mdash" not in t for t in out), out
    # The literal ``pitfall-mdash-target-class`` must NOT be present.
    assert "pitfall-mdash-target-class" not in out


def test_extract_drops_lo_id_leaks():
    """LO IDs that leak into key_concepts (e.g. when JSON-LD authoring
    places ``TO-04`` into the keyTerms list) must be filtered — they
    belong in objectives.json, not concept_graph.
    """
    proc = _make_processor()
    out = _extract(proc, [
        "TO-04",
        "CO-12",
        "Turtle",                # → turtle → DomainConcept (KEEP)
    ])
    # ``turtle`` is canonicalized via the alias map (already
    # canonical); it survives.
    assert out == ["turtle"], out


def test_extract_collapses_known_aliases():
    """``rdfxml`` and ``rdf-xml`` collapse to the same canonical slug.

    Order-dependence: we add ``rdfxml`` first; the canonicalization
    rewrites it to ``rdf-xml``. The second emit (``rdf-xml``) is
    detected as a duplicate and rejected.
    """
    proc = _make_processor()
    out = _extract(proc, [
        "RDF/XML",       # canonical_slug → "rdfxml" → alias → "rdf-xml"
        "rdf-xml",       # alias → "rdf-xml" (duplicate, dropped)
        "JSON-LD",       # → "json-ld"
    ])
    assert out == ["rdf-xml", "json-ld"], out


def test_canonicalize_alias_pass_through():
    """Slugs not in the alias map must round-trip unchanged."""
    assert canonicalize_alias("blank-node") == "blank-node"
    assert canonicalize_alias("sparql-select") == "sparql-select"


def test_extract_collapses_trivial_plurals():
    """When both ``triple`` and ``triples`` are emitted, only the
    singular survives (regardless of order).

    The first-emitted form wins by default — but the singular-form
    helper detects that ``triples`` reduces to ``triple`` and skips
    it when the singular is already present.
    """
    proc = _make_processor()
    out = _extract(proc, [
        "Triple",                # singular emitted first
        "Triples",               # plural - dropped because triple is in tags
        "Named Graph",
    ])
    assert "triple" in out
    assert "triples" not in out
    assert "named-graph" in out


def test_extract_keeps_real_domain_vocabulary():
    """Sanity: a clean candidate list passes through unchanged.

    Note that the slugifier strips colons (CURIE separators), so
    ``sh:path`` enters extraction as the slug ``shpath``. The Wave 75
    regression contract that ``classify_concept('sh:path')`` returns
    DomainConcept guards the *raw* CURIE form for non-extraction
    consumers (e.g. retrieval ranking); the extraction path
    necessarily slugifies first.
    """
    proc = _make_processor()
    out = _extract(proc, [
        "Turtle",
        "RDF Graph",
        "Blank Node",
        "SPARQL SELECT",
        "OWL 2 RL",
        "sh:path",
    ])
    # turtle is canonical via alias map; everything else passes through.
    assert "turtle" in out
    assert "rdf-graph" in out
    assert "blank-node" in out
    assert "sparql-select" in out
    assert "owl-2-rl" in out
    # CURIE → slug strips the colon; ``shpath`` is what the extractor
    # actually emits. The regression that ``classify_concept('sh:path')``
    # = DomainConcept is locked above.
    assert "shpath" in out


# ---------------------------------------------------------------------------
# Length / numeric guards (Wave 76 part 3 final tail)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("candidate", ["12", "1-5", "1", "ab"])
def test_extract_drops_short_or_numeric(candidate):
    """Pure-numeric and 1-2 char slugs are stripped pre-classifier
    by ``len(tag) < 3`` in extract; the classifier confirms LowSignal
    when those slugs do reach it via other paths.

    Note: ``normalize_tag`` rejects slugs whose first char isn't a
    letter, so ``"12"`` becomes ``""`` at the slugifier, not a
    classified slug. Still, the classifier guard is the second
    line of defence.
    """
    klass = classify_concept(candidate)
    assert klass == LOW_SIGNAL, f"{candidate!r} → {klass}"
