"""Key-term minting must require a genuine definitional SUBJECT.

Pre-fix bug: every entry in ``ContentExtractor.DEFINITION_PATTERNS`` captures
group 1 as "sentence start → first copula" and minted it as a ``KeyTerm``
verbatim. On ordinary prose that fires on any sentence that merely CONTAINS a
copula, so the capture was whatever preamble preceded it — an interrogative
("What is the value of x?"), a demonstrative ("This is the largest root"), an
existential ("There is a faster method"), a subordinate/participial opener
("Since 72 is a multiple of 8, …" / "Shown below is the table"), or an exercise
imperative ("Complete the following: _______ is the additive identity").

None of those is a term, and a key term is reused as the fill-in-the-blank
ANSWER and as the MCQ stem subject — so one fabricated term poisons every item
generated from that chunk. Measured over the archived chunksets the copula
pattern alone minted 816 copies of "What" and 775 of "This".

Two rules close it, both exercised here through the PRODUCTION entry point
``ContentExtractor.extract_key_terms``:

1. ``_is_definitional_subject`` — the capture must read as a short NOMINAL noun
   phrase (canonical shape rules) headed by a noun rather than a closed-class
   sentence opener, and must carry no label / authored-blank punctuation.
2. The one DEFINITION_PATTERN with no sentence anchor (``X, which is Y``) has
   its ``[A-Z]`` subject boundary scoped case-SENSITIVE, so it starts at a real
   capitalized term instead of swallowing the lowercase preamble. The four
   ANCHORED patterns deliberately stay case-tolerant so a lowercase
   sentence-initial glossary key is still harvested.

All fixtures are inline.
"""

import pytest

from Trainforge.generators.assessment.content_extractor import (
    ContentExtractor,
    _is_definitional_subject,
)


def _terms(text: str):
    """Run the production entry point over one inline chunk."""
    extractor = ContentExtractor()
    return [t.term for t in extractor.extract_key_terms([{"id": "c1", "text": text}])]


# --------------------------------------------------------------------------- #
# Fabricated subjects must never become terms.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,forbidden",
    [
        # Interrogative preamble.
        ("What is the value of x when the two sides balance?", "What"),
        # Bare demonstrative — no antecedent once the sentence is lifted out.
        ("This is the largest root of the polynomial shown above.", "This"),
        # Existential.
        ("There is a faster method for factoring this trinomial.", "There"),
        ("Here is the completed multiplication table for reference.", "Here"),
        # Subordinating conjunction opener.
        ("Since 72 is a multiple of 8, the quotient is a whole number.", "Since 72"),
        # Participial / adverbial opener.
        ("Shown below is the graph of the corresponding linear equation.", "Shown below"),
        # Discourse connective.
        ("But, there is a simpler way to write the same expression.", "But, there"),
        # Third-person plural pronoun.
        ("They are the two factors whose product is the constant term.", "They"),
    ],
)
def test_non_nominal_preamble_never_becomes_a_key_term(text, forbidden):
    assert forbidden not in _terms(text)


def test_exercise_imperative_with_authored_blank_is_not_a_term():
    """A fill-in-the-blank prompt is an exercise, not a definition."""
    text = "Complete the following: _______ is the additive identity for addition."
    terms = _terms(text)
    assert not any("_" in t for t in terms)
    assert "Complete the following: _______" not in terms


def test_label_prefixed_capture_is_not_a_term():
    """A colon marks a label / heading boundary, never a term."""
    text = "Efficiency : Fewer processing steps are the reason the method is faster."
    assert "Efficiency : Fewer processing steps" not in _terms(text)


def test_whole_paragraph_of_prose_mints_no_fabricated_terms():
    """The measured failure mode, end to end: clean prose, zero real definitions."""
    text = (
        "What is the value of x in this equation? This is the largest root shown. "
        "Since 72 is a multiple of 8, the quotient is an integer. "
        "Shown below is the completed table. "
        "Complete the following: _______ is the additive identity."
    )
    assert _terms(text) == []


# --------------------------------------------------------------------------- #
# Real definitions must survive.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "A rational number is a ratio of two integers with a nonzero denominator.",
            "A rational number",
        ),
        (
            "Every integer is a rational number because it can be written over one.",
            "Every integer",
        ),
        (
            "Photosynthesis is the process by which plants convert light into sugar.",
            "Photosynthesis",
        ),
    ],
)
def test_real_definitional_subjects_survive(text, expected):
    assert expected in _terms(text)


def test_head_plus_one_prepositional_qualifier_survives():
    """"The perimeter of a rectangle" is a term, not clause glue.

    The canonical fragment filter rejects any span carrying two or more
    function words, which a qualified noun phrase spends on a single
    prepositional attachment. The head is therefore tested on its own.
    """
    text = "The perimeter of a rectangle is the sum of twice the length and twice the width."
    assert "The perimeter of a rectangle" in _terms(text)


def test_clause_wearing_a_prepositional_qualifier_is_still_rejected():
    """The relaxation must not readmit relative clauses."""
    text = "A fraction in which the numerator or the denominator is a fraction is complex."
    assert "A fraction in which the numerator or the denominator" not in _terms(text)


def test_determiner_licenses_an_otherwise_blocked_head():
    """A determiner proves the head is nominal, so the opener check is skipped."""
    assert _is_definitional_subject("The second law")
    assert _is_definitional_subject("The given information")
    # …but the same words bare are sentence openers, not terms.
    assert not _is_definitional_subject("Second")
    assert not _is_definitional_subject("Given that")


# --------------------------------------------------------------------------- #
# Subject-boundary case scoping.
# --------------------------------------------------------------------------- #

def test_unanchored_pattern_starts_at_the_capitalized_term():
    """``X, which is Y`` has no sentence anchor, so its ``[A-Z]`` is its only
    left boundary. Case-insensitively it began at the first letter of the
    lowercase preamble and swallowed the whole clause; case-sensitively it
    starts at the real term.
    """
    text = (
        "the protocol version negotiated for this connection is HTTP 1.1, "
        "which is the release that introduced persistent connections."
    )
    terms = _terms(text)
    assert "HTTP 1.1" in terms
    assert not any(t.startswith("the protocol version") for t in terms)


def test_anchored_patterns_still_harvest_a_lowercase_glossary_key():
    """A flattened glossary line opens lowercase; the sentence anchor already
    bounds those patterns, so they must stay case-tolerant."""
    text = "denominator The denominator is the number below the fraction bar."
    assert "denominator The denominator" in _terms(text)


# --------------------------------------------------------------------------- #
# Predicate-level table (documents the rule, independent of the regexes).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "subject,expected",
    [
        ("What", False),
        ("This", False),
        ("It", False),
        ("There", False),
        ("Since 72", False),
        ("Shown below", False),
        ("If it", False),
        ("Complete the following: _______", False),
        ("", False),
        ("A rational number", True),
        ("Any polynomial", True),
        ("Scientific notation", True),
        ("The degree of a polynomial", True),
    ],
)
def test_is_definitional_subject_table(subject, expected):
    assert _is_definitional_subject(subject) is expected
