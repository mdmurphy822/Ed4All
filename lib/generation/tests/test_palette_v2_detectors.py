"""Issue I6 — block-planner content detectors (acronym + tabular).

CPU-pinned, no-GPU, no-model unit tests for the precision-first content
detectors that nudge the dynamic block planner toward the palette-v2 block
types:

- ``detect_acronyms`` fires on a PEMDAS-like all-caps token WHEN the source
  spells out the expansion (precision: it does NOT fire on a bare all-caps
  word with no expansion present).
- ``detect_tabular_content`` fires on comparison / multi-category framing and
  on an explicit ``<table>`` in the source, and stays quiet on plain prose.
"""

from __future__ import annotations

from lib.generation.block_planner import (
    detect_acronyms,
    detect_canonical_mnemonics,
    detect_tabular_content,
)


# --------------------------------------------------------------------------- #
# Acronym detector — precision over recall.
# --------------------------------------------------------------------------- #
def test_acronym_fires_with_expansion_present():
    src = (
        "Remember PEMDAS: Parentheses, Exponents, Multiplication, Division, "
        "Addition, Subtraction."
    )
    assert "PEMDAS" in detect_acronyms(src)


def test_acronym_fires_with_mnemonic_phrase_expansion():
    # "Please Excuse My Dear Aunt Sally" supplies every initial of PEMDAS.
    src = "PEMDAS is easy: Please Excuse My Dear Aunt Sally."
    assert "PEMDAS" in detect_acronyms(src)


def test_acronym_does_not_fire_on_bare_allcaps_word():
    # No expansion words → precision contract refuses the candidate.
    assert detect_acronyms("Just remember PEMDAS for order of operations.") == []


def test_acronym_does_not_fire_on_unrelated_allcaps_token():
    # An ID / heading-shaped capital token with no spelled-out expansion.
    assert detect_acronyms("Open the FILE and read XYZ now.") == []


def test_acronym_empty_source():
    assert detect_acronyms("") == []
    assert detect_acronyms("   ") == []


# --------------------------------------------------------------------------- #
# Canonical-mnemonic detector — fires PEMDAS even when the literal token is
# ABSENT, as long as the FULL expansion is spelled out (grounded learning aid,
# not fabrication). Precision-first: a partial expansion never fires.
# --------------------------------------------------------------------------- #
def test_canonical_mnemonic_fires_on_expansion_terms_without_token():
    # Order-of-operations source that NEVER writes "PEMDAS" but spells out the
    # full per-letter expansion → the canonical detector fires it.
    src = (
        "Follow the order of operations: first simplify expressions inside "
        "parentheses, then evaluate exponents, then do any multiplication and "
        "division left to right, and finally addition and subtraction."
    )
    assert "PEMDAS" not in detect_acronyms(src)  # literal-token detector silent
    assert "PEMDAS" in detect_canonical_mnemonics(src)


def test_canonical_mnemonic_fires_on_mnemonic_phrase_without_token():
    src = (
        "A common way to remember the order is the phrase "
        "Please Excuse My Dear Aunt Sally."
    )
    assert "PEMDAS" in detect_canonical_mnemonics(src)


def test_canonical_mnemonic_does_not_fire_on_partial_expansion():
    # Missing "subtraction" / "division" etc → precision contract refuses.
    src = "We covered parentheses and exponents and a little multiplication."
    assert detect_canonical_mnemonics(src) == []


def test_canonical_mnemonic_quiet_on_unrelated_text():
    assert detect_canonical_mnemonics("This paragraph is about photosynthesis.") == []
    assert detect_canonical_mnemonics("") == []


def test_canonical_mnemonic_bodmas():
    src = (
        "Use BODMAS-style ordering: brackets, orders, division, "
        "multiplication, addition, subtraction."
    )
    assert "BODMAS" in detect_canonical_mnemonics(src)


# --------------------------------------------------------------------------- #
# Tabular detector.
# --------------------------------------------------------------------------- #
def test_tabular_fires_on_comparison_framing():
    assert detect_tabular_content("Compare proper and improper fractions.")
    assert detect_tabular_content("Mixed numbers versus improper fractions.")
    assert detect_tabular_content("There are three types of fractions.")


def test_tabular_fires_on_explicit_table_markup():
    assert detect_tabular_content("<table><tr><td>x</td></tr></table>")


def test_tabular_quiet_on_plain_prose():
    assert not detect_tabular_content("This sentence teaches one simple idea.")
    assert not detect_tabular_content("")
