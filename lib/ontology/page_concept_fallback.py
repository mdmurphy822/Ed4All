"""Page-level lexical concept fallback for markup-less accessible HTML.

Motivation (audit defect D1): the SemantiK GLM-OCR accessible-HTML lane
emits ZERO ``<strong>`` / ``<b>`` / ``<dt>`` markup, so
``Trainforge.parsers.html_content_parser.HTMLContentParser._extract_concepts``
harvests NO page-level ``key_concepts``. ``extract_concept_tags`` then
falls back to the pedagogy-only ``CONCEPT_PATTERNS`` and every chunk of the
page ends up with an empty ``concept_tags`` list (measured 704/705 chunks
empty on a real GLM-OCR course, versus ~14 tags/chunk on a
vendor/publisher-HTML course whose bold/dt markup feeds the parser). That
empty substrate starves the downstream concept-graph + CURIE machinery.

Fix (behind ``TRAINFORGE_PAGE_CONCEPT_FALLBACK``, default OFF): for each
parsed item whose ``key_concepts`` is empty/absent, derive a PAGE-LEVEL
key-concept list from that page's OWN text and assign it to
``item["key_concepts"]`` so the EXISTING page-level tagging path
(``extract_concept_tags`` over the whole-page key-concept set) fires
unchanged and all chunks of the page share the same tag set — the vendor
semantics. This is deliberately PAGE-level, NOT chunk-local: chunk-local
tags fragment the KG (see ``Trainforge/CLAUDE.md`` "Recommended
graph-shaping config" + the memory note on the chunk-local trap), so this
fallback keeps ``TRAINFORGE_CHUNK_LOCAL_TAGS`` out of the picture.

Derivation is deterministic + lexical only — no embeddings, no LLM. It
routes each page's per-section texts (falling back to the tag-stripped
``raw_html`` as one doc when the item carries no sections) through the
shared ``lib.ontology.lexical_concept_seeds.derive_lexical_concept_seeds``
(``min_doc_freq=1``) and re-applies the same quality gates the chunk-local
path uses: ``is_fragment_phrase`` on multi-word slugs and
``is_scaffolding_noise``. The list is capped at 20 (the HTML parser's own
``key_concepts`` cap; ``concept_tagging.MAX_CONCEPT_TAGS`` is also 20).

Only EMPTY ``key_concepts`` is filled — a page that already carries real
parser-harvested concepts is never touched. Flag off → the item dicts are
returned byte-identical (nothing written).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from lib.ontology.concept_classifier import is_scaffolding_noise
from lib.ontology.lexical_concept_seeds import (
    derive_lexical_concept_seeds,
    is_fragment_phrase,
)

# Env flag (parse mirrors the sibling ``TRAINFORGE_*`` toggles:
# ``.lower() == "true"`` — anything else is off).
PAGE_CONCEPT_FALLBACK_ENV = "TRAINFORGE_PAGE_CONCEPT_FALLBACK"

# Cap on derived page-level concepts. Mirrors the HTML parser's own
# ``key_concepts`` cap and ``concept_tagging.MAX_CONCEPT_TAGS`` (both 20).
_MAX_PAGE_CONCEPTS = 20

# Tag stripper for the ``raw_html`` fallback doc (no sections). Purely
# lexical — collapses every tag to whitespace so the deriver sees plain text.
_TAG_RE = re.compile(r"<[^>]+>")


def resolve_page_concept_fallback(env: Mapping[str, str] | None = None) -> bool:
    """Return True iff the page-concept fallback is enabled.

    Reads ``TRAINFORGE_PAGE_CONCEPT_FALLBACK`` from ``env`` (defaults to
    ``os.environ``). Parse-with-fallback: ``"true"`` (case-insensitive) → on;
    unset / empty / anything else → off.
    """
    source = os.environ if env is None else env
    return str(source.get(PAGE_CONCEPT_FALLBACK_ENV, "")).lower() == "true"


def _section_content(section: Any) -> str | None:
    """Duck-typed per-section content extractor.

    Reads ``.content`` off a ``ContentSection`` dataclass, or ``["content"]``
    off a mapping. Returns the string when non-empty, else ``None``.
    """
    content = getattr(section, "content", None)
    if content is None and isinstance(section, Mapping):
        content = section.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def _page_docs(item: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Build the per-doc list handed to the lexical seed deriver.

    Prefers the page's per-SECTION texts (one doc per section, so a
    multi-word phrase recurring across sections earns a higher document
    frequency); falls back to the tag-stripped ``raw_html`` as ONE doc when
    the item carries no usable sections.
    """
    section_texts: List[str] = []
    for section in item.get("sections") or []:
        content = _section_content(section)
        if content is not None:
            section_texts.append(content)
    if section_texts:
        return [{"text": text} for text in section_texts]

    raw_html = item.get("raw_html")
    if isinstance(raw_html, str) and raw_html.strip():
        stripped = _TAG_RE.sub(" ", raw_html)
        if stripped.strip():
            return [{"text": stripped}]
    return []


def derive_page_concept_seeds(
    item: Mapping[str, Any], *, max_concepts: int = _MAX_PAGE_CONCEPTS
) -> List[str]:
    """Derive a page-level key-concept slug list from ``item``'s own text.

    Deterministic + lexical (no embeddings, no LLM). Routes the page's
    per-section texts (or the tag-stripped ``raw_html`` fallback) through
    ``derive_lexical_concept_seeds(min_doc_freq=1)``, then re-applies the
    chunk-local quality gates (``is_fragment_phrase`` on multi-word slugs,
    ``is_scaffolding_noise``) and caps the result at ``max_concepts``.

    Best-effort: a deriver exception returns ``[]`` (the caller keeps the
    page's existing — empty — ``key_concepts``).
    """
    docs = _page_docs(item)
    if not docs:
        return []
    try:
        seeds = derive_lexical_concept_seeds(docs, min_doc_freq=1)
    except Exception:  # noqa: BLE001 — derivation is best-effort
        return []

    out: List[str] = []
    for slug in seeds:
        if not slug:
            continue
        # Multi-word slugs get the canonical sentence-fragment gate (the same
        # filter the bold-span harvest + chunk-local path use).
        if "-" in slug and is_fragment_phrase(slug.replace("-", " ")):
            continue
        if is_scaffolding_noise(slug):
            continue
        out.append(slug)
        if len(out) >= max_concepts:
            break
    return out


def apply_page_concept_fallback(
    parsed_items: Sequence[MutableMapping[str, Any]],
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    """Fill EMPTY page-level ``key_concepts`` in place; return the fill count.

    Env-gated: a no-op returning ``0`` (item dicts byte-identical) unless
    ``TRAINFORGE_PAGE_CONCEPT_FALLBACK`` is truthy. Only items whose
    ``key_concepts`` is empty/absent are considered — a page that already
    carries real parser-harvested concepts is never overwritten. An item for
    which the deriver yields nothing keeps its original (empty) value, so a
    page is never mutated to an empty list.
    """
    if not resolve_page_concept_fallback(env):
        return 0
    filled = 0
    for item in parsed_items or []:
        if not isinstance(item, MutableMapping):
            continue
        if item.get("key_concepts"):
            # Non-empty parser-harvested key_concepts — leave untouched.
            continue
        seeds = derive_page_concept_seeds(item)
        if seeds:
            item["key_concepts"] = seeds
            filled += 1
    return filled


__all__ = [
    "PAGE_CONCEPT_FALLBACK_ENV",
    "resolve_page_concept_fallback",
    "derive_page_concept_seeds",
    "apply_page_concept_fallback",
]
