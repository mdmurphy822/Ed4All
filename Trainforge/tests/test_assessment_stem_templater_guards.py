"""W10 stem-templater guards — clause-grab rejection, rotation, footnote strip.

Book-1 canary defect: ``ContentExtractor.RELATIONSHIP_PATTERNS`` capture
``([^,.]+?)`` — an arbitrary clause up to the next comma/period — as each
"concept", so the "relationship between X and Y" template minted stems like::

    "Briefly explain the relationship between <full clause with a finite
     verb> and <another clause> ."

244 items carried that malformed shape; 325/765 items sat in exact-duplicate
groups across quizzes (every per-TO quiz shares the corpus-wide chunk pool and
the builders always picked ``relationships[0]``); and 98 fill-in-blank stems
retained footnote apparatus (``$^{2}$`` markers + bare URLs) from source text.

Fixes under test:

* ``_is_term_like_concept`` — only short noun-phrase terms mint a
  ``ConceptRelationship`` (length/word caps, no finite verbs / clause
  connectives, ``lib.ontology`` fragment-phrase filter);
* ``extract_relationships`` skips clause-grabs so generators fall back to a
  different template;
* ``AssessmentGenerator._rotate_relationship`` / ``_rotate_procedure`` /
  ``_rotate_example`` rotate through distinct extracted targets before reuse;
* ``_strip_footnote_apparatus`` removes ``$^{n}$`` markers + bare footnote
  URLs at the single point the assessment path quotes chunk text.

All fixture text is anonymized (no course slugs, no corpus content).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment.content_extractor import (  # noqa: E402
    ContentExtractor,
    _is_term_like_concept,
    _strip_footnote_apparatus,
    _strip_html,
)
from Trainforge.generators.assessment.generator import (  # noqa: E402
    AssessmentGenerator,
)


# --------------------------------------------------------------------------- #
# _is_term_like_concept
# --------------------------------------------------------------------------- #

class TestTermLikeConcept:
    def test_accepts_short_noun_phrases(self):
        for term in (
            "replication",
            "vector store",
            "the file system",
            "Knowledge Base",
            "distributed hash table",
        ):
            assert _is_term_like_concept(term) is True, term

    def test_rejects_clause_with_finite_verb(self):
        # Real anonymized shape of the canary clause-grab: the pattern's
        # group(1) swallowed a whole clause ending in a negated auxiliary.
        clause = (
            "ServiceX replicates your data across multiple nodes so that "
            "the loss of a single node doesn't"
        )
        assert _is_term_like_concept(clause) is False

    def test_rejects_infinitive_purpose_clause(self):
        # The canary's group(2): "all your data to be lost".
        assert _is_term_like_concept("all your data to be lost") is False

    def test_rejects_verb_bearing_fragments(self):
        for frag in (
            "the node is unavailable",
            "requests are routed to a replica",
            "you should retry the operation",
            "which handles failures",
        ):
            assert _is_term_like_concept(frag) is False, frag

    def test_rejects_over_long_phrases(self):
        assert _is_term_like_concept(
            "a very long noun phrase chain of many stacked tokens"
        ) is False

    def test_rejects_empty_and_tiny(self):
        assert _is_term_like_concept("") is False
        assert _is_term_like_concept("ab") is False


# --------------------------------------------------------------------------- #
# extract_relationships clause-grab guard
# --------------------------------------------------------------------------- #

def _chunks(text: str):
    return [{"id": "chunk_test_001", "text": text}]


class TestRelationshipExtraction:
    def test_clause_grab_sentence_mints_no_relationship(self):
        # "causes" fires inside a clause; both captures are clauses → no mint.
        text = (
            "ServiceX replicates your data across multiple nodes so that "
            "the loss of a single node doesn't cause all your data to be "
            "lost entirely."
        )
        rels = ContentExtractor().extract_relationships(_chunks(text))
        assert rels == []

    def test_genuine_term_pair_still_mints(self):
        text = "Replication causes higher storage overhead."
        rels = ContentExtractor().extract_relationships(_chunks(text))
        assert len(rels) == 1
        assert rels[0].concept_a.lower() == "replication"
        assert "storage overhead" in rels[0].concept_b.lower()

    def test_unlike_comparison_with_terms_still_mints(self):
        text = "Unlike a hash index, a tree index supports range scans."
        rels = ContentExtractor().extract_relationships(_chunks(text))
        assert len(rels) == 1
        assert "hash index" in rels[0].concept_a.lower()

    def test_generators_fall_back_when_relationships_rejected(self):
        # A chunk whose only relationship-pattern hit is a clause-grab but
        # which carries a definable key term: the short-answer builder must
        # fall back to the key-term template, never emit the salad stem.
        text = (
            "ServiceX replicates your data across nodes so that a single "
            "failure doesn't cause all your data to be lost entirely. "
            "Consistent hashing is defined as a partitioning technique "
            "that minimizes key remapping."
        )
        gen = AssessmentGenerator(capture=None, check_leaks=False)
        result = gen._generate_short_answer(
            question_id="q-test-1",
            objective_id="TO-01",
            bloom_level="understand",
            level_config={"verbs": ["explain"]},
            source_chunks=_chunks(text),
        )
        stem = getattr(result, "stem", "")
        assert "relationship between" not in stem
        assert "doesn't" not in stem


# --------------------------------------------------------------------------- #
# Rotation — distinct relationships/procedures before any reuse
# --------------------------------------------------------------------------- #

class TestRotation:
    def test_relationship_rotation_spans_distinct_pairs(self):
        text = (
            "Replication causes higher storage overhead. "
            "Sharding leads to lower query latency. "
            "Caching produces faster reads."
        )
        gen = AssessmentGenerator(capture=None, check_leaks=False)
        rels = gen._content_extractor.extract_relationships(_chunks(text))
        assert len(rels) >= 3
        picked = [gen._rotate_relationship(rels) for _ in range(3)]
        keys = {(r.concept_a.lower(), r.concept_b.lower()) for r in picked}
        assert len(keys) == 3  # no collapse onto relationships[0]

    def test_short_answer_stems_differ_across_calls(self):
        text = (
            "Replication causes higher storage overhead. "
            "Sharding leads to lower query latency."
        )
        gen = AssessmentGenerator(capture=None, check_leaks=False)
        stems = []
        for i in range(2):
            result = gen._generate_short_answer(
                question_id=f"q-test-{i}",
                objective_id="TO-01",
                bloom_level="understand",
                level_config={"verbs": ["explain"]},
                source_chunks=_chunks(text),
            )
            stems.append(getattr(result, "stem", ""))
        assert stems[0] != stems[1]


# --------------------------------------------------------------------------- #
# Footnote-apparatus strip
# --------------------------------------------------------------------------- #

class TestFootnoteStrip:
    def test_marker_removed(self):
        assert "$^{2}$" not in _strip_footnote_apparatus(
            "Durability matters.$^{2}$ Data survives restarts."
        )

    def test_tolerant_marker_spacing(self):
        out = _strip_footnote_apparatus("A claim.$ ^{ 12 } $ More text.")
        assert "^{" not in out

    def test_bare_url_removed(self):
        out = _strip_footnote_apparatus(
            "2. https://example.com/docs/durability-notes remains a footnote."
        )
        assert "https://" not in out

    def test_real_math_superscript_preserved(self):
        # $x^{2}$ has content before the caret — NOT a footnote marker.
        text = "The area grows as $x^{2}$ for side x."
        assert "$x^{2}$" in _strip_footnote_apparatus(text)

    def test_strip_html_applies_footnote_strip(self):
        out = _strip_html(
            "<p>Durability matters.$^{3}$ See https://example.com/note "
            "for details.</p>"
        )
        assert "$^{3}$" not in out
        assert "https://" not in out
        assert "Durability matters." in out

    def test_fill_in_blank_context_carries_no_apparatus(self):
        text = (
            "Consistent hashing is defined as a partitioning technique "
            "that minimizes key remapping.$^{2}$ See "
            "https://example.com/hashing for details."
        )
        gen = AssessmentGenerator(capture=None, check_leaks=False)
        result = gen._generate_fill_in_blank(
            question_id="q-test-fib",
            objective_id="TO-01",
            bloom_level="remember",
            level_config={"verbs": ["define"]},
            source_chunks=_chunks(text),
        )
        stem = getattr(result, "stem", "")
        assert "$^{" not in stem
        assert "https://" not in stem
