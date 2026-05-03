"""Plan §3.6 regression — per-error-pattern retry-directive table on
the outline tier's schema-fix retry message.

Coverage:

- ``_match_retry_directive`` returns the bloom-enum directive on the
  recorded "is not one of ['remember'..." validator string (§3.1).
- ``_match_retry_directive`` returns the CURIE-pattern directive on
  the recorded "does not match '^[a-z]..." validator string (§3.3).
- ``_match_retry_directive`` returns the key_claims compression
  directive on the recorded "['subject', 'predicate', 'object'] is
  too long" validator string (§3.4).
- ``_match_retry_directive`` returns the enum-vs-int directive on
  the "2 is not of type 'string'" validator string (§3.1 numeric-tier
  drift edge case).
- Empty / unknown error strings return ``None`` (caller falls back to
  the bare validator-error echo).
- The system prompt enumerates the canonical bloom_level allowed
  values + the empty-CURIE permission directive (§3.1 + §3.3).
- The per-block-type bounds rendering ends with the bloom_level
  allowed-values reminder (§3.1 user-prompt mirror).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._outline_provider import (  # noqa: E402
    _OUTLINE_KIND_BOUNDS,
    _OUTLINE_SYSTEM_PROMPT,
    _RETRY_DIRECTIVE_PATTERNS,
    _match_retry_directive,
)


# ---------------------------------------------------------------------------
# §3.6 retry-directive table — pattern matches
# ---------------------------------------------------------------------------


def test_match_retry_directive_bloom_enum_drift():
    """Validator emits "'2' is not one of ['remember', 'understand', ...]"
    when the model writes a numeric tier instead of the enum string.
    The directive table catches that pattern."""
    err = "'2' is not one of ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']"
    directive = _match_retry_directive(err)
    assert directive is not None
    assert "remember" in directive
    assert "lowercase" in directive
    assert "numeric tier" in directive


def test_match_retry_directive_curie_pattern_violation():
    """Validator emits "'rdf_shacl_551' does not match '^[a-z]...'"
    on the §1.3 invented-CURIE-prefix case + on the §1.5 full-IRI
    case ("rdf:https://...")."""
    invented = "'rdf_shacl_551' does not match '^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$'"
    full_iri = "'rdf:https://www.w3.org/1999/02/22-rdf-syntax-ns#' does not match '^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$'"
    for err in (invented, full_iri):
        directive = _match_retry_directive(err)
        assert directive is not None
        assert "prefix:local" in directive
        assert "[]" in directive
        assert "full IRI" in directive


def test_match_retry_directive_key_claims_too_long():
    """Validator emits "['subject', 'predicate', 'object'] is too long"
    when the model emits a 3-element key_claims list that exceeds the
    block-type's maxItems bound. Plan §3.4 / §1.4."""
    err = "['subject', 'predicate', 'object'] is too long"
    directive = _match_retry_directive(err)
    assert directive is not None
    assert "key_claims" in directive
    assert "compress" in directive.lower() or "Compress" in directive


def test_match_retry_directive_enum_vs_int():
    """Validator emits "2 is not of type 'string'" when the model
    writes an unquoted numeric tier. The directive table catches the
    generic enum-vs-int case."""
    err = "2 is not of type 'string'"
    directive = _match_retry_directive(err)
    assert directive is not None
    assert "string" in directive.lower()


def test_match_retry_directive_unknown_error_returns_none():
    """Unknown error patterns return None so the caller falls back
    to the bare validator-error echo."""
    assert _match_retry_directive("totally unrelated error") is None
    assert _match_retry_directive("") is None
    assert _match_retry_directive(None) is None  # type: ignore[arg-type]


def test_retry_directive_table_has_all_four_seed_patterns():
    """Plan §3.6 seed: four patterns required (bloom enum, CURIE
    pattern, key_claims maxItems, enum-vs-int). A regression that
    silently drops one is caught here."""
    assert len(_RETRY_DIRECTIVE_PATTERNS) >= 4
    # Each tuple is (compiled-pattern, directive-string).
    for pattern, directive in _RETRY_DIRECTIVE_PATTERNS:
        assert hasattr(pattern, "search")
        assert isinstance(directive, str)
        assert len(directive) >= 20  # plan: ≥20-char actionable directive


# ---------------------------------------------------------------------------
# §3.1 + §3.3 system prompt enumerations
# ---------------------------------------------------------------------------


def test_outline_system_prompt_enumerates_bloom_levels():
    """Plan §3.1: system prompt MUST list the canonical six string
    labels. Catches a regression that drops the enum directive."""
    prompt = _OUTLINE_SYSTEM_PROMPT
    for level in (
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    ):
        assert level in prompt, (
            f"system prompt missing canonical bloom_level label {level!r}"
        )


def test_outline_system_prompt_documents_empty_curie_permission():
    """Plan §3.3: system prompt MUST permit `curies: []` and forbid
    full-IRI / invented-prefix forms. Catches a regression that
    drops the permission directive."""
    prompt = _OUTLINE_SYSTEM_PROMPT
    assert "[]" in prompt
    assert "prefix:local" in prompt
    assert "full IRI" in prompt
    assert "invent" in prompt.lower()


# ---------------------------------------------------------------------------
# §3.4 bounds calibration
# ---------------------------------------------------------------------------


def test_assessment_item_key_claims_bound_is_one_to_four():
    """Plan §3.4 / §1.4: bumped from (1, 2) to (1, 4) so the canonical
    RDF-triple three-tuple fits the bound without forcing 7B-class
    compression the model can't produce."""
    bounds = _OUTLINE_KIND_BOUNDS["assessment_item"]
    lo, hi = bounds["key_claims"]
    assert (lo, hi) == (1, 4)


# ---------------------------------------------------------------------------
# §3.1 user-prompt mirror — bounds block ends with bloom_level enum
# ---------------------------------------------------------------------------


def test_user_prompt_bounds_block_ends_with_bloom_level_enum():
    """Plan §3.1 user-prompt mirror: the bounds block emitted by
    `_render_user_prompt` MUST include the bloom_level allowed-values
    line so the 7B-class default model sees the enum at the bottom of
    the bounds block (recency bias)."""
    from Courseforge.generators._outline_provider import OutlineProvider
    from blocks import Block

    # Construct a provider with deps stubbed (no LLM dispatch in this
    # test — we only render the user prompt).
    import os

    os.environ.pop("COURSEFORGE_OUTLINE_PROVIDER", None)
    os.environ.pop("COURSEFORGE_OUTLINE_MODEL", None)
    # Use a dummy anthropic_client so the constructor doesn't demand
    # an API key at construction time.
    p = OutlineProvider(
        provider="anthropic",
        anthropic_client=object(),
        api_key="placeholder-not-used-during-prompt-render",
    )
    block = Block(
        block_id="page1#concept_intro_0",
        block_type="concept",
        page_id="page1",
        sequence=0,
        content={},
    )
    rendered = p._render_user_prompt(
        block=block, source_chunks=[], objectives=[]
    )
    # The bounds block emits one bloom_level enum line per render.
    assert "bloom_level allowed values:" in rendered
    # All six labels appear in the line.
    for level in (
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    ):
        assert level in rendered
