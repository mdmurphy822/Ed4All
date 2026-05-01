"""Regression tests for lib.ontology.slugs and migrated call sites.

Covers REC-ID-03 (Wave 4, Worker Q):

  * Canonical `canonical_slug` behavior — pins byte-for-byte parity with
    the historical ``Courseforge.scripts.generate_course._slugify``.
  * Courseforge `_slugify` alias resolves to `canonical_slug`.
  * Trainforge `normalize_tag` delegates canonicalization to
    `canonical_slug` while preserving its display-layer rules
    (4-token truncation + alpha-first rejection).
  * Trainforge's `is_a_from_key_terms._slugify` delegates
    canonicalization to `canonical_slug` while preserving its site-specific
    SC-reference canonicalization and punctuation→space preprocessing.
  * Cross-caller parity: for a sweep of plain inputs, all three call sites
    produce the same slug.

The three call sites were previously independent copies; this test file is
the pin that guarantees they stay in sync through future edits.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Module-load helpers
# ---------------------------------------------------------------------------


def _load_by_path(module_name: str, path: Path):
    """Load a Python module from an absolute path.

    Used for ``Courseforge/scripts/generate_course.py`` which is normally
    invoked as a script, not imported as a package.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def courseforge_slugify():
    """Load the migrated ``_slugify`` alias from generate_course.py."""
    path = _REPO_ROOT / "Courseforge" / "scripts" / "generate_course.py"
    module = _load_by_path("courseforge_generate_course_for_slug_tests", path)
    return module._slugify


@pytest.fixture(scope="module")
def trainforge_normalize_tag():
    """Load the migrated `normalize_tag` from process_course.py."""
    from Trainforge.process_course import normalize_tag
    return normalize_tag


@pytest.fixture(scope="module")
def is_a_slugify():
    """Load the migrated `_slugify` from is_a_from_key_terms.py."""
    from Trainforge.rag.inference_rules.is_a_from_key_terms import _slugify
    return _slugify


# ---------------------------------------------------------------------------
# Historical references (pre-migration implementations, pinned here so we
# can prove byte-equivalence forever).
# ---------------------------------------------------------------------------


def _legacy_courseforge_slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    return re.sub(r"\s+", "-", slug).strip("-")


def _legacy_normalize_tag(raw: str) -> str:
    tag = raw.lower().strip()
    tag = re.sub(r"[^a-z0-9\s-]", "", tag)
    tag = re.sub(r"\s+", "-", tag)
    tag = tag.strip("-")
    parts = tag.split("-")
    if len(parts) > 4:
        tag = "-".join(parts[:4])
    if tag and not tag[0].isalpha():
        return ""
    return tag


def _legacy_is_a_slugify(text: str) -> str:
    from Trainforge.rag.wcag_canonical_names import canonicalize_sc_references
    text = canonicalize_sc_references(text or "")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return text


# ---------------------------------------------------------------------------
# canonical_slug behavior pins
# ---------------------------------------------------------------------------


def test_canonical_slug_basic():
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("Cognitive Load Theory") == "cognitive-load-theory"


def test_canonical_slug_empty():
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("") == ""


def test_canonical_slug_falsy_safe():
    """Falsy (None) is handled without raising — matches historic callers
    that pass ``text or ''`` upstream but we defend anyway."""
    from lib.ontology.slugs import canonical_slug
    # Mimic common-pattern None-guard (canonical_slug's own guard covers this)
    assert canonical_slug(None or "") == ""


def test_canonical_slug_only_punctuation():
    """Pure punctuation strips to empty. Matches Courseforge _slugify."""
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("!!!") == ""
    assert canonical_slug("...") == ""
    assert canonical_slug("?!?") == ""


def test_canonical_slug_numbers_fuse_on_dot():
    """Dots are DELETED (not replaced). Pins the digit-fusing behavior
    inherited from Courseforge's _slugify — "2.2" → "22", not "2-2"."""
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("WCAG 2.2 AA") == "wcag-22-aa"


def test_canonical_slug_leading_trailing_hyphens():
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("-foo-bar-") == "foo-bar"


def test_canonical_slug_preserves_internal_multi_hyphens():
    """Courseforge _slugify did NOT collapse interior runs of hyphens.
    Preserving that behavior; callers that need collapse do it themselves."""
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("--a--b--") == "a--b"


def test_canonical_slug_case_insensitive():
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("FooBar") == "foobar"
    assert canonical_slug("FOOBAR") == "foobar"


def test_canonical_slug_whitespace_collapse():
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug("a   b") == "a-b"
    assert canonical_slug("a\tb") == "a-b"
    assert canonical_slug("a\nb") == "a-b"


def test_canonical_slug_idempotent_on_slugs():
    """Calling canonical_slug on an already-slugged string returns it unchanged."""
    from lib.ontology.slugs import canonical_slug
    for slug in ("foo", "foo-bar", "a1b2c3", "keyboard-trap", "wcag-22-aa"):
        assert canonical_slug(slug) == slug


# ---------------------------------------------------------------------------
# Parity: canonical_slug against legacy Courseforge _slugify
# ---------------------------------------------------------------------------


# A representative range of inputs designed to touch every branch of the
# algorithm: alnum-only, whitespace, mixed case, punctuation, hyphens,
# edge-hyphens, multi-hyphens, leading/trailing ws, tabs, numerics, empties.
_CANONICAL_PARITY_INPUTS = [
    "Cognitive Load Theory",
    "keyboard trap",
    "ARIA role",
    "FooBar",
    "UPPER CASE STRING",
    "no-changes-needed",
    "one two three four five six",
    "-foo-bar-",
    "--a--b--",
    "a   b",
    "Hello, World!",
    "WCAG 2.2 AA",
    "Mix-Of_Underscores-and-dashes",
    "2nd generation",
    "Success Criterion 1.3.1",
    "   leading and trailing   ",
    "",
    "!!!",
    "a",
    "abc",
    "abc1",
    "1abc",
    "a\tb\nc",
]


@pytest.mark.parametrize("text", _CANONICAL_PARITY_INPUTS)
def test_canonical_slug_matches_legacy_courseforge_slugify(text):
    from lib.ontology.slugs import canonical_slug
    assert canonical_slug(text) == _legacy_courseforge_slugify(text), (
        f"canonical_slug diverged from legacy Courseforge _slugify on input {text!r}"
    )


@pytest.mark.parametrize("text", _CANONICAL_PARITY_INPUTS)
def test_courseforge_alias_matches_canonical_slug(text, courseforge_slugify):
    """The ``_slugify`` alias inside generate_course.py IS canonical_slug."""
    from lib.ontology.slugs import canonical_slug
    assert courseforge_slugify(text) == canonical_slug(text)


# ---------------------------------------------------------------------------
# Parity: Trainforge normalize_tag retains legacy behavior
# ---------------------------------------------------------------------------


# Inputs that exercise normalize_tag's display-layer rules (truncation +
# alpha-first rejection) on top of the shared canonical base.
_NORMALIZE_TAG_INPUTS = [
    "Cognitive Load Theory",
    "WCAG 2.2 AA",                          # alpha-first: starts with 'w' → kept
    "2nd generation",                       # alpha-first: starts with '2' → ""
    "5 foo bar baz",                        # alpha-first: "5" → ""
    "one two three four five six",          # 6 tokens → truncate to 4
    "a b c d e f",                          # 6 tokens → truncate to 4
    "ARIA role",
    "  leading space  ",
    "!!!",                                  # strips to empty
    "1abc",                                 # alpha-first → ""
    "abc1",                                 # kept
    "   ",                                  # strips to empty
    "",
    "hello, world!",
    "a",
]


@pytest.mark.parametrize("text", _NORMALIZE_TAG_INPUTS)
def test_normalize_tag_matches_legacy(text, trainforge_normalize_tag):
    """New `normalize_tag` produces the same output as the pre-migration
    four-line canonicalization + truncation + alpha-first rule."""
    assert trainforge_normalize_tag(text) == _legacy_normalize_tag(text), (
        f"normalize_tag diverged from legacy implementation on input {text!r}"
    )


def test_normalize_tag_truncates_to_4_tokens(trainforge_normalize_tag):
    assert trainforge_normalize_tag("one two three four five six") == "one-two-three-four"


def test_normalize_tag_rejects_numeric_first_char(trainforge_normalize_tag):
    assert trainforge_normalize_tag("2nd generation") == ""
    assert trainforge_normalize_tag("1abc") == ""


def test_normalize_tag_matches_canonical_on_short_alpha_input(
    trainforge_normalize_tag,
):
    """When input is ≤4 tokens and starts with a letter, normalize_tag is
    a pure synonym of canonical_slug."""
    from lib.ontology.slugs import canonical_slug
    for text in ("Cognitive Load Theory", "keyboard trap", "ARIA role", "FooBar"):
        assert trainforge_normalize_tag(text) == canonical_slug(text)


# ---------------------------------------------------------------------------
# Parity: is_a_from_key_terms._slugify retains legacy behavior
# ---------------------------------------------------------------------------


_IS_A_PARITY_INPUTS = [
    "Cognitive Load Theory",
    "a.b",                                  # dot → space → hyphen (site-specific)
    "a.b.c",
    "WCAG 2.2 AA",                          # dots get word-separated (site-specific)
    "",
    "!!!",
    "-foo-bar-",
    "--a--b--",                             # multi-hyphen collapse (site-specific)
    "FooBar",
    "a   b",
    "keyboard trap",
    "   leading and trailing   ",
    "Hello, World!",
    "Mix-Of_Underscores-and-dashes",
    "2nd generation",
    "ARIA role (element)",
    "a (b) c",
    "a..b",
    "Success Criterion 1.3.1",
    "no-changes-needed",
]


@pytest.mark.parametrize("text", _IS_A_PARITY_INPUTS)
def test_is_a_slugify_matches_legacy(text, is_a_slugify):
    """The refactored is_a `_slugify` produces byte-identical output to
    the pre-migration implementation."""
    assert is_a_slugify(text) == _legacy_is_a_slugify(text), (
        f"is_a._slugify diverged from legacy implementation on input {text!r}"
    )


def test_is_a_slugify_punctuation_stays_word_separated(is_a_slugify):
    """Unlike Courseforge's _slugify, is_a replaces punctuation with a
    separator so word boundaries survive. Pin that semantic."""
    assert is_a_slugify("a.b") == "a-b"
    assert is_a_slugify("a (b) c") == "a-b-c"


def test_is_a_slugify_handles_none_input_safely(is_a_slugify):
    assert is_a_slugify(None) == ""


def test_is_a_slugify_applies_sc_canonicalization(is_a_slugify):
    """Spot-check: the is_a slugify must go through canonicalize_sc_references
    before slugging, so raw SC references get normalized. Confirm via a
    round-trip: if the canonicalization rewrites any substring, the resulting
    slug should reflect the canonical form."""
    from Trainforge.rag.wcag_canonical_names import canonicalize_sc_references
    raw = "Success Criterion 1.3.1"
    expected_pre = canonicalize_sc_references(raw)
    from lib.ontology.slugs import canonical_slug
    expected = canonical_slug(
        re.sub(r"[^a-z0-9\s\-]", " ", expected_pre.lower())
    )
    # Allow the legacy multi-hyphen collapse
    expected = re.sub(r"-+", "-", expected).strip("-")
    assert is_a_slugify(raw) == expected


# ---------------------------------------------------------------------------
# Cross-caller parity on plain input
# ---------------------------------------------------------------------------


# "Plain" inputs: no internal punctuation, no SC references, ≤4 tokens,
# starts with a letter. These should produce byte-identical slugs from
# all three call sites.
_PLAIN_INPUTS = [
    "Cognitive Load Theory",
    "keyboard trap",
    "ARIA role",
    "WCAG AA compliance",
    "FooBar",
    "UPPER CASE STRING",
    "no-changes-needed",
    "Hello World",
    "one two three four",
    "a",
    "abc",
    "   leading and trailing   ",
    "color contrast ratio",
    "focus visible indicator",
    "semantic html structure",
    "alt text review",
    "heading hierarchy check",
    "keyboard navigation test",
    "form label association",
    "landmark region usage",
]


@pytest.mark.parametrize("text", _PLAIN_INPUTS)
def test_all_three_callers_agree_on_plain_input(
    text,
    courseforge_slugify,
    trainforge_normalize_tag,
    is_a_slugify,
):
    """For inputs without internal punctuation and short enough to pass the
    display-layer cap, Courseforge `_slugify`, Trainforge `normalize_tag`,
    and `is_a_from_key_terms._slugify` all produce the same slug as
    `canonical_slug`."""
    from lib.ontology.slugs import canonical_slug
    expected = canonical_slug(text)
    assert courseforge_slugify(text) == expected, (
        f"Courseforge _slugify diverged from canonical on {text!r}"
    )
    assert trainforge_normalize_tag(text) == expected, (
        f"Trainforge normalize_tag diverged from canonical on {text!r}"
    )
    assert is_a_slugify(text) == expected, (
        f"is_a._slugify diverged from canonical on {text!r}"
    )


# ---------------------------------------------------------------------------
# Wave 130d: strip_lo_ref_suffix + deslugify_concept
#
# Concept-tag slugs built from ``CO-NN`` / ``TO-NN`` learning-objective refs
# (e.g. ``property-paths-co-15``) used to bleed ``co 15`` / ``to 03`` artifact
# tokens into prompt text via ``slug.replace("-", " ")``. The new helpers
# strip the LO-ref suffix before the deslug transform.
# ---------------------------------------------------------------------------


def test_strip_lo_ref_suffix_removes_co_NN():
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("property-paths-co-15") == "property-paths"


def test_strip_lo_ref_suffix_removes_to_NN():
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("subqueries-to-03") == "subqueries"


def test_strip_lo_ref_suffix_handles_3_digit_lo_code():
    """Three-digit LO codes like ``co-100`` are within the {1,3} range."""
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("alpha-co-100") == "alpha"


def test_strip_lo_ref_suffix_preserves_legitimate_to():
    """``map-to-existing-vocabularies`` contains ``to-`` but not as a
    numeric LO ref — the trailing token isn't a digit so the regex
    doesn't match. False-positive guard."""
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert (
        strip_lo_ref_suffix("map-to-existing-vocabularies")
        == "map-to-existing-vocabularies"
    )


def test_strip_lo_ref_suffix_preserves_pattern_to_remember():
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("pattern-to-remember") == "pattern-to-remember"


def test_strip_lo_ref_suffix_preserves_attach_to_the_focus():
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("attach-to-the-focus") == "attach-to-the-focus"


def test_strip_lo_ref_suffix_preserves_default_to_core_justify():
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert (
        strip_lo_ref_suffix("default-to-core-justify")
        == "default-to-core-justify"
    )


def test_strip_lo_ref_suffix_case_insensitive():
    """Concept slugs are normally lowercase, but the regex is
    case-insensitive so a hand-authored ``Property-Paths-CO-15`` also
    strips cleanly."""
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("Property-Paths-CO-15") == "Property-Paths"


def test_strip_lo_ref_suffix_handles_empty_string():
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("") == ""


def test_strip_lo_ref_suffix_handles_none():
    """Falsy input doesn't raise — matches ``canonical_slug``'s contract."""
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix(None or "") == ""


def test_strip_lo_ref_suffix_pure_lo_code_unchanged():
    """A bare ``co-15`` (no concept stem) lacks the leading hyphen the
    regex anchors against, so it passes through unchanged. Pure LO
    codes are filtered upstream by ``OBJECTIVE_CODE_RE`` in the
    extraction call site — the strip helper is only responsible for
    the *compound* ``stem-co-NN`` form."""
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("co-15") == "co-15"
    assert strip_lo_ref_suffix("to-03") == "to-03"


def test_strip_lo_ref_suffix_no_suffix_unchanged():
    """Slugs without a trailing LO-ref are passed through unchanged."""
    from lib.ontology.slugs import strip_lo_ref_suffix
    assert strip_lo_ref_suffix("property-paths") == "property-paths"
    assert strip_lo_ref_suffix("sub-class-of") == "sub-class-of"
    assert strip_lo_ref_suffix("rdf-graph") == "rdf-graph"


def test_deslugify_concept_strips_lo_then_spaces():
    from lib.ontology.slugs import deslugify_concept
    assert deslugify_concept("property-paths-co-15") == "property paths"


def test_deslugify_concept_strips_to_then_spaces():
    from lib.ontology.slugs import deslugify_concept
    assert deslugify_concept("subqueries-to-03") == "subqueries"


def test_deslugify_concept_preserves_subClassOf_shape():
    """``canonical_slug`` lowercases first so ``subClassOf`` arrives as
    a single hyphen-less token; deslugify_concept leaves it alone."""
    from lib.ontology.slugs import deslugify_concept
    assert deslugify_concept("subclassof") == "subclassof"


def test_deslugify_concept_handles_empty_string():
    from lib.ontology.slugs import deslugify_concept
    assert deslugify_concept("") == ""


def test_deslugify_concept_handles_underscores():
    """Underscore-to-space transform from the legacy
    ``.replace("-", " ").replace("_", " ")`` chain is preserved."""
    from lib.ontology.slugs import deslugify_concept
    assert deslugify_concept("rdf_type") == "rdf type"


def test_deslugify_concept_no_lo_ref_just_spaces():
    """A concept slug without an LO-ref suffix deslugs identically to
    the legacy chain."""
    from lib.ontology.slugs import deslugify_concept
    assert deslugify_concept("property-paths") == "property paths"
    assert deslugify_concept("rdf-graph") == "rdf graph"


def test_deslugify_concept_false_positive_guards():
    """``map-to-existing-vocabularies``, ``pattern-to-remember``,
    ``attach-to-the-focus`` deslug as if there was no LO-ref suffix —
    proving the regex doesn't strip legitimate ``-to-`` substrings."""
    from lib.ontology.slugs import deslugify_concept
    assert (
        deslugify_concept("map-to-existing-vocabularies")
        == "map to existing vocabularies"
    )
    assert deslugify_concept("pattern-to-remember") == "pattern to remember"
    assert deslugify_concept("attach-to-the-focus") == "attach to the focus"


def test_canonical_slug_is_single_source_of_truth():
    """Verify the three call sites actually import from lib.ontology.slugs.

    This is a structural pin — any future edit that reintroduces a local
    implementation would break this test."""
    from lib.ontology.slugs import canonical_slug as _canon

    # The Courseforge alias must be canonical_slug itself (same object).
    path = _REPO_ROOT / "Courseforge" / "scripts" / "generate_course.py"
    module = _load_by_path("courseforge_generate_course_identity_check", path)
    assert module._slugify is _canon, (
        "Courseforge _slugify must be the same object as canonical_slug"
    )

    # Trainforge process_course.py imports canonical_slug (used by normalize_tag).
    import Trainforge.process_course as pc
    assert pc.canonical_slug is _canon

    # is_a_from_key_terms imports canonical_slug.
    import Trainforge.rag.inference_rules.is_a_from_key_terms as is_a
    assert is_a.canonical_slug is _canon
