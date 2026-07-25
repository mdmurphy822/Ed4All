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
