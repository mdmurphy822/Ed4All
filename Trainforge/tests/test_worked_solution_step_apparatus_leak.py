"""Regression net: worked-solution ``Step N:`` apparatus must never be mined.

Provenance
----------
A real production run halted on the ``assessment_quality`` gate with, among
others, these five verbatim question stems (copied out of that run's
``trainforge_assessment_checkpoint.json``)::

    Step 1: Find the opposite of 7: it is the same distance from 0 as 7 but on the opposite side
    Step 2: Since -4 is not 4 units left of 0, its opposite is 4 units right of 0.
    Step 2: Since -44 is not 44 units from 0, |-44| = 44.
    Step 2: Since -9 is 9 units from 0, |-9| = 9.
    Step 1: Find the absolute value of -44: it is not the distance from -44 to 0 on the number line

They are worked-solution steps that the harvest paths mined out of a
worked-example region and the item builders turned into true/false items (by
injecting a negation) and into fill-in-the-blank answer keys. They are not
valid assessment items and should never have reached the generator.

Root cause: ``_is_apparatus_text`` recognised only a BARE step LABEL
("Step 1", "Step 2.") as apparatus, so a complete step SENTENCE carried no
marker the guard could see. Separately, ``extract_key_terms`` screened only a
candidate's DEFINITION, never its TERM — and screened neither in the
bold/strong and concept-tag strategies — so the same scaffolding shipped as a
key term, which is reused verbatim as the fill-in-the-blank answer key.

Every marker here is generic pedagogical-label vocabulary (the same class as
``Solution:`` / ``Check:`` / ``Try It``). Nothing in this module is keyed to a
subject, a publisher or a book.
"""

from __future__ import annotations

import pytest

from Trainforge.generators.content_extractor import (
    ContentExtractor,
    _is_apparatus_text,
)

_FLAG = "ED4ALL_ASSESSMENT_APPARATUS_STRICT"

#: Verbatim from the failing checkpoint.
REAL_LEAKED_STEP_STEMS = [
    "Step 1: Find the opposite of 7: it is the same distance from 0 as 7 but on the opposite side.",
    "Step 2: Since -4 is not 4 units left of 0, its opposite is 4 units right of 0.",
    "Step 2: Since -44 is not 44 units from 0, |-44| = 44.",
    "Step 2: Since -9 is 9 units from 0, |-9| = 9.",
    "Step 1: Find the absolute value of -44: it is not the distance from -44 to 0 on the number line.",
]

#: The un-negated source sentences the item builder mutated. Both forms must
#: be rejected — the guard runs at MINING time, before any negation.
REAL_SOURCE_STEP_SENTENCES = [
    "Step 1: Find the opposite of 7: it is the same distance from 0 as 7 but on the opposite side.",
    "Step 2: Since -4 is 4 units left of 0, its opposite is 4 units right of 0.",
    "Step 2: Since -44 is 44 units from 0, |-44| = 44.",
    "Step 2: Since -9 is 9 units from 0, |-9| = 9.",
    "Step 1: Find the absolute value of -44: it is the distance from -44 to 0 on the number line.",
]

#: Ordinary prose that MENTIONS a step. Must stay minable — the guard fires on
#: the "Step N<delimiter>" label shape, not on the word "step".
NON_APPARATUS_PROSE = [
    "Step 1 is to isolate the variable on one side of the equation.",
    "The first step in solving is to isolate the variable.",
    "Each step of a proof must follow from the previous one.",
]


@pytest.fixture(autouse=True)
def _strict_on(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")


@pytest.mark.parametrize("text", REAL_LEAKED_STEP_STEMS)
def test_leaked_step_stems_are_apparatus(text: str) -> None:
    assert _is_apparatus_text(text) is True


@pytest.mark.parametrize("text", REAL_SOURCE_STEP_SENTENCES)
def test_source_step_sentences_are_apparatus(text: str) -> None:
    assert _is_apparatus_text(text) is True


@pytest.mark.parametrize("text", REAL_LEAKED_STEP_STEMS)
def test_leaked_step_stems_are_apparatus_when_html_wrapped(text: str) -> None:
    """The emitter wraps stems in ``<p>…</p>``; the ``^`` anchor must survive."""
    assert _is_apparatus_text(f"<p>{text}</p>") is True


@pytest.mark.parametrize("text", NON_APPARATUS_PROSE)
def test_prose_mentioning_a_step_is_not_apparatus(text: str) -> None:
    assert _is_apparatus_text(text) is False


def test_step_guard_is_off_without_the_strict_flag(monkeypatch) -> None:
    """Default-off stays byte-identical to the legacy marker set."""
    monkeypatch.delenv(_FLAG, raising=False)
    for text in REAL_LEAKED_STEP_STEMS:
        assert _is_apparatus_text(text) is False


# --------------------------------------------------------------------------- #
# Harvest-path coverage — the guard has to fire where the items were made.
# --------------------------------------------------------------------------- #

#: A flattened worked-example region in the shape the chunker emits it: the
#: narrative sentence, then the numbered solution steps, then the answer.
WORKED_EXAMPLE_CHUNK = {
    "id": "chunk_test_0001",
    "text": (
        "The opposite of a number is the same distance from zero but on the "
        "opposite side of the number line. "
        "Step 1: Find the opposite of 7: it is the same distance from 0 as 7 "
        "but on the opposite side. "
        "Step 2: Therefore, the opposite of 7 is -7. "
        "Solution: -7 "
        "Step 2: Since -9 is 9 units from 0, |-9| = 9."
    ),
}


def test_factual_statement_harvest_drops_worked_solution_steps() -> None:
    statements = ContentExtractor().extract_factual_statements(
        [dict(WORKED_EXAMPLE_CHUNK)]
    )
    for stmt in statements:
        assert not stmt.statement.lstrip().lower().startswith("step "), stmt


def test_key_term_harvest_drops_worked_solution_steps_as_terms() -> None:
    """A step label must never become a key TERM (= the FIB answer key)."""
    terms = ContentExtractor().extract_key_terms([dict(WORKED_EXAMPLE_CHUNK)])
    for term in terms:
        assert not term.term.lstrip().lower().startswith("step "), term
        assert not term.definition.lstrip().lower().startswith("step "), term


def test_key_term_harvest_drops_bold_apparatus_labels() -> None:
    """Strategy 2 (bold/strong) had no apparatus screen at all.

    Generated worked-example HTML bolds exactly the apparatus labels, so
    ``<strong>Solution</strong>`` shipped as a key term whose answer key was
    the literal word "Solution".
    """
    chunk = {
        "id": "chunk_test_0002",
        "text": (
            "<p>Convert 0.374 to a fraction.</p>"
            "<p><strong>Solution</strong>: 0.374 = 187/500 Check: "
            "187 / 500 = 0.374, which matches the original decimal.</p>"
        ),
    }
    terms = ContentExtractor().extract_key_terms([chunk])
    assert "solution" not in {t.term.strip().lower() for t in terms}


def test_key_term_harvest_drops_concept_tag_matches_inside_apparatus() -> None:
    """Strategy 3 (concept_tags) had no apparatus screen either."""
    chunk = {
        "id": "chunk_test_0003",
        "concept_tags": ["opposite"],
        "text": (
            "Step 1: Find the opposite of 7: it is the same distance from 0 "
            "as 7 but on the opposite side."
        ),
    }
    terms = ContentExtractor().extract_key_terms([chunk])
    for term in terms:
        assert not term.definition.lstrip().lower().startswith("step "), term


def test_real_instructional_prose_still_harvests() -> None:
    """The guard must not starve the mining pool of legitimate content."""
    chunk = {
        "id": "chunk_test_0004",
        "text": (
            "A multiple of a number is the product of that number and a "
            "counting number. The absolute value of a number is its distance "
            "from zero on the number line."
        ),
    }
    extractor = ContentExtractor()
    assert extractor.extract_factual_statements([dict(chunk)])
    assert extractor.extract_key_terms([dict(chunk)])


# =========================================================================== #
# Round 2 — the apparatus vocabulary the first widening did not cover.
#
# After the step-prefix fix landed the production run regenerated and the gate
# still failed at 0.75. Two of the four remaining VERB_LESS_STEM warnings were
# apparatus that carries a label the marker set had never seen — verbatim from
# `state/runs/<run>/checkpoints/trainforge_assessment_checkpoint.json`:
#
#   Q-cbc366e9: 'Show solution The opposite of 7 is -7 because it is the same...'
#   Q-c17ddc7e: 'Key Idea: Each digit in a whole number has a place value ...'
#
# `Show solution` is a synonym of the already-covered `Show answer` and — like
# the whole `Show <x>` family — carries NO delimiter, so a colon-anchored rule
# cannot see it. `Key Idea:` is a generic pedagogical CALLOUT label, the same
# discourse class as `Solution:` / `Check:`.
#
# A third warning was a dangling anaphoric fragment with no antecedent
# (`This is determined by its position as the third digit from the right.`) —
# unanswerable once lifted out of its paragraph, so it must never be mined.
# =========================================================================== #

#: Verbatim from the round-2 checkpoint.
REAL_SHOW_SOLUTION_STEM = (
    "Show solution The opposite of 7 is -7 because it is the same distance "
    "from 0 but on the opposite side."
)
REAL_KEY_IDEA_STEM = (
    "Key Idea: Each digit in a whole number has a place value based on its "
    "position, such as trillions, billions, millions, thousands, hundreds, "
    "tens, or ones."
)
REAL_ANAPHORIC_STATEMENT = (
    "This is determined by its position as the third digit from the right."
)


def test_show_solution_is_apparatus_without_the_strict_flag(monkeypatch) -> None:
    """The ``Show <x>`` family is a synonym set on the ALWAYS-ON marker.

    ``Show answer`` was already unconditional; ``Show solution`` / ``Show
    work`` / ``Show steps`` are the same marker emitted by the same authoring
    surface, so they are matched on the same unconditional pattern rather than
    behind the strict flag.
    """
    monkeypatch.delenv(_FLAG, raising=False)
    assert _is_apparatus_text(REAL_SHOW_SOLUTION_STEM) is True


@pytest.mark.parametrize(
    "text",
    [
        "Show answer Yes, 42 is a multiple of 7.",
        "Show solution The opposite of 7 is -7.",
        "Show work 3x + 5 = 20 so x = 5.",
        "Show steps First isolate the variable.",
        "Show step First isolate the variable.",
    ],
)
def test_show_family_synonyms_are_apparatus(text: str) -> None:
    assert _is_apparatus_text(text) is True


def test_show_family_does_not_swallow_ordinary_prose() -> None:
    """The pattern is case-SENSITIVE on ``Show``, so mid-sentence prose is safe."""
    assert _is_apparatus_text(
        "The graph will show solution sets for the inequality."
    ) is False


def test_key_idea_callout_label_is_apparatus() -> None:
    assert _is_apparatus_text(REAL_KEY_IDEA_STEM) is True


@pytest.mark.parametrize(
    "text",
    [
        "Key Idea: Place value determines a digit's worth.",
        "Key Point: The sum of two even numbers is even.",
        "Big Idea: Multiplication is repeated addition.",
        "Note: A negative times a negative is positive.",
        "Tip: Check the sign before simplifying.",
        "Warning: Do not divide by zero.",
        "Remember: The absolute value is never negative.",
        "Common wrong turn: Confusing factors and multiples.",
        "Misconception: A larger denominator means a larger fraction.",
        "Predict: The fraction will have a denominator of 1000.",
    ],
)
def test_generic_callout_labels_are_apparatus(text: str) -> None:
    assert _is_apparatus_text(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # The label WORDS used as ordinary prose — no leading label + colon.
        "Note that the sum of two even numbers is even.",
        "The important idea is that place value determines a digit's worth.",
        "Remember to check the sign before simplifying the expression.",
        "A common mistake in this chapter is dividing before multiplying.",
        # CONTENT-TYPE labels are deliberately NOT in the callout set — the
        # key-term extractor exists to mine exactly these.
        "Definition: A multiple of n is the product of n and a counting number.",
        "Theorem: The sum of the angles of a triangle is 180 degrees.",
        "Property: Addition is commutative.",
    ],
)
def test_callout_rule_does_not_swallow_content(text: str) -> None:
    assert _is_apparatus_text(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "CO-01: Identify the place value of each digit in a whole number.",
        "TO-05: Students will apply mathematical reasoning to real problems.",
        "LO-12: Explain why the absolute value is never negative.",
    ],
)
def test_learning_objective_refs_are_apparatus(text: str) -> None:
    """An objective statement is course design, not a claim to test.

    The shape is this project's OWN canonical LO-id convention
    (``lib/ontology/learning_objectives.py``), so the rule is structural.
    """
    assert _is_apparatus_text(text) is True


def test_callout_and_lo_rules_are_off_without_the_strict_flag(monkeypatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    assert _is_apparatus_text(REAL_KEY_IDEA_STEM) is False
    assert _is_apparatus_text("CO-01: Identify the place value.") is False


# --------------------------------------------------------------------------- #
# Anaphoric-subject rejection at the mining layer.
# --------------------------------------------------------------------------- #

def test_anaphoric_subject_detection() -> None:
    from Trainforge.generators.content_extractor import _has_anaphoric_subject

    for bare in ("This", "That", "It", "They", "There", "one", "Such"):
        assert _has_anaphoric_subject(bare) is True, bare
    for real in ("This process", "The value", "A multiple of n", "Absolute value"):
        assert _has_anaphoric_subject(real) is False, real


def test_dangling_anaphor_is_not_mined_as_a_factual_statement() -> None:
    """The verbatim failing stem must never re-enter the candidate pool."""
    chunk = {
        "id": "chunk_test_0005",
        "text": (
            "The place value of a digit is determined by where the digit sits "
            "in the number. "
            + REAL_ANAPHORIC_STATEMENT
        ),
    }
    statements = ContentExtractor().extract_factual_statements([chunk])
    texts = [s.statement for s in statements]
    assert REAL_ANAPHORIC_STATEMENT not in texts
    # …and the self-contained sentences from the same chunk still survive.
    assert texts, "anaphor guard must not empty the pool"


def test_round2_apparatus_never_reaches_the_statement_pool() -> None:
    """End-to-end at the harvest layer, on the real flattened shapes."""
    chunk = {
        "id": "chunk_test_0006",
        "text": (
            "Find the opposite of 7, the opposite of -10, and simplify -(-6). "
            + REAL_SHOW_SOLUTION_STEM
            + " The opposite of -10 is 10 because it is the same distance "
            "from 0 but on the opposite side."
        ),
    }
    statements = ContentExtractor().extract_factual_statements([chunk])
    for stmt in statements:
        assert not stmt.statement.startswith("Show solution"), stmt
        assert not stmt.statement.startswith("Key Idea"), stmt


def test_key_idea_and_lo_ref_never_reach_the_key_term_pool() -> None:
    chunk = {
        "id": "chunk_test_0007",
        "text": (
            "CO-01: Identify the place value of each digit in a given whole "
            "number up to trillions. "
            + REAL_KEY_IDEA_STEM
        ),
    }
    terms = ContentExtractor().extract_key_terms([chunk])
    for term in terms:
        assert not term.term.strip().lower().startswith("key idea"), term
        assert not term.term.strip().lower().startswith("co-01"), term
        assert not term.definition.strip().lower().startswith("key idea"), term
