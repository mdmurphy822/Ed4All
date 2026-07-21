"""Unit tests for the page-level lexical concept fallback (audit defect D1).

Covers the pure derivation + quality gates + env-gated in-place mutation of
``lib.ontology.page_concept_fallback``. No chunking / MCP surface here — the
integration test lives beside the MCP chunking tests.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.page_concept_fallback import (  # noqa: E402
    PAGE_CONCEPT_FALLBACK_ENV,
    apply_page_concept_fallback,
    derive_page_concept_seeds,
    resolve_page_concept_fallback,
)


@dataclass
class _FakeSection:
    """Duck-typed stand-in for Trainforge's ContentSection."""

    content: str


_PHOTOSYNTHESIS = (
    "The photosynthesis process converts light energy into chemical energy. "
    "Photosynthesis occurs inside the chloroplast. The chloroplast contains "
    "chlorophyll. Chlorophyll absorbs light energy. The calvin cycle fixes "
    "carbon dioxide. Carbon dioxide enters through the stomata."
)


def _markupless_html(body: str) -> str:
    """Accessible-HTML-lane page: prose only, ZERO <strong>/<b>/<dt> markup."""
    return (
        "<!doctype html><html><head><title>Biology</title></head><body>"
        f"<h1>Biology</h1><section><h2>Cells</h2><p>{body}</p></section>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# resolve_page_concept_fallback — parse-with-fallback env read
# ---------------------------------------------------------------------------

def test_resolve_default_off():
    assert resolve_page_concept_fallback(env={}) is False


def test_resolve_true_on():
    assert resolve_page_concept_fallback(env={PAGE_CONCEPT_FALLBACK_ENV: "true"})
    assert resolve_page_concept_fallback(env={PAGE_CONCEPT_FALLBACK_ENV: "TRUE"})


def test_resolve_garbage_off():
    for val in ("1", "yes", "on", "", "false", "garbage"):
        assert (
            resolve_page_concept_fallback(env={PAGE_CONCEPT_FALLBACK_ENV: val})
            is False
        ), val


# ---------------------------------------------------------------------------
# derive_page_concept_seeds — pure derivation over sections / raw_html
# ---------------------------------------------------------------------------

def test_derive_from_sections_nonempty():
    item = {"sections": [_FakeSection(content=_PHOTOSYNTHESIS)]}
    seeds = derive_page_concept_seeds(item)
    assert seeds, "expected lexical concept seeds from markup-less section text"
    assert all(isinstance(s, str) and s for s in seeds)
    # Real domain nouns surface (single-word acronyms aside, multi-word
    # noun phrases dominate this corpus).
    assert any("chloroplast" in s or "photosynthesis" in s or "carbon" in s
               for s in seeds), seeds


def test_derive_from_mapping_sections():
    """Sections may be plain mappings, not only ContentSection dataclasses."""
    item = {"sections": [{"content": _PHOTOSYNTHESIS}]}
    seeds = derive_page_concept_seeds(item)
    assert seeds


def test_derive_falls_back_to_raw_html_when_no_sections():
    item = {"sections": [], "raw_html": _markupless_html(_PHOTOSYNTHESIS)}
    seeds = derive_page_concept_seeds(item)
    assert seeds, "raw_html fallback should still yield seeds"


def test_derive_empty_when_no_text():
    assert derive_page_concept_seeds({"sections": [], "raw_html": ""}) == []
    assert derive_page_concept_seeds({}) == []


def test_derive_caps_at_20():
    item = {"sections": [_FakeSection(content=_PHOTOSYNTHESIS)]}
    seeds = derive_page_concept_seeds(item)
    assert len(seeds) <= 20


def test_derive_filters_fragments_and_scaffolding():
    """Sentence-fragment / scaffolding junk never survives the quality gates."""
    from lib.ontology.lexical_concept_seeds import is_fragment_phrase
    from lib.ontology.concept_classifier import is_scaffolding_noise

    # Prose engineered to also produce clause-shaped bigrams.
    body = (
        "Congratulations you have finished. The vector space model represents "
        "documents as vectors. The vector space model is common. Term frequency "
        "weights each term. Term frequency matters for the vector space model."
    )
    seeds = derive_page_concept_seeds({"sections": [_FakeSection(content=body)]})
    for slug in seeds:
        if "-" in slug:
            assert not is_fragment_phrase(slug.replace("-", " ")), slug
        assert not is_scaffolding_noise(slug), slug


# ---------------------------------------------------------------------------
# apply_page_concept_fallback — env-gated in-place mutation
# ---------------------------------------------------------------------------

def test_apply_off_is_byte_identical():
    """Flag off → no mutation, count 0 (item dicts unchanged)."""
    item = {"key_concepts": [], "sections": [_FakeSection(content=_PHOTOSYNTHESIS)]}
    before = dict(item)
    filled = apply_page_concept_fallback([item], env={})
    assert filled == 0
    assert item == before
    assert item["key_concepts"] == []


def test_apply_on_fills_empty_key_concepts():
    item = {"key_concepts": [], "sections": [_FakeSection(content=_PHOTOSYNTHESIS)]}
    filled = apply_page_concept_fallback(
        [item], env={PAGE_CONCEPT_FALLBACK_ENV: "true"}
    )
    assert filled == 1
    assert item["key_concepts"], "empty key_concepts should be filled on-flag"


def test_apply_on_missing_key_concepts_key():
    """An item with no key_concepts key at all is treated as empty."""
    item = {"sections": [_FakeSection(content=_PHOTOSYNTHESIS)]}
    filled = apply_page_concept_fallback(
        [item], env={PAGE_CONCEPT_FALLBACK_ENV: "true"}
    )
    assert filled == 1
    assert item["key_concepts"]


def test_apply_on_does_not_overwrite_real_concepts():
    """A page that already carries parser concepts is never touched."""
    original: List[str] = ["Photosynthesis", "Chloroplast"]
    item = {
        "key_concepts": list(original),
        "sections": [_FakeSection(content=_PHOTOSYNTHESIS)],
    }
    filled = apply_page_concept_fallback(
        [item], env={PAGE_CONCEPT_FALLBACK_ENV: "true"}
    )
    assert filled == 0
    assert item["key_concepts"] == original


def test_apply_on_empty_page_stays_empty_not_mutated():
    """A textless page keeps its empty value (never set to [] anew)."""
    item = {"key_concepts": [], "sections": [], "raw_html": ""}
    filled = apply_page_concept_fallback(
        [item], env={PAGE_CONCEPT_FALLBACK_ENV: "true"}
    )
    assert filled == 0
    assert item["key_concepts"] == []


def test_apply_page_level_identical_across_a_pages_chunks():
    """Two items sharing a page's text derive the SAME page-level list.

    Guards the page-level (not chunk-local) semantics: the fallback is a
    per-PAGE derivation, so any two pages built from the same text produce an
    identical, order-stable key-concept list.
    """
    item_a = {"key_concepts": [], "sections": [_FakeSection(content=_PHOTOSYNTHESIS)]}
    item_b = {"key_concepts": [], "sections": [_FakeSection(content=_PHOTOSYNTHESIS)]}
    apply_page_concept_fallback(
        [item_a, item_b], env={PAGE_CONCEPT_FALLBACK_ENV: "true"}
    )
    assert item_a["key_concepts"] == item_b["key_concepts"]
    assert item_a["key_concepts"]
