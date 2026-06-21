"""Deterministic "Key Terms" page builder (feature I5).

A fully deterministic post-pass that, per terminal objective (week), authors a
``week_NN_key_terms.html`` page of vocab cards — one card per key term with a
definition and a deep-link back to the defining source chunk. NO 7B free
authoring runs here: every definition is lifted verbatim from a source chunk or
the domain-concept vocabulary's ``definition_hint`` (anti-fabrication — a term
whose definition resolves to NEITHER is OMITTED, never invented).

The builder is intentionally GPU-free and side-effect-free so it can run as a
deterministic post-pass over an existing course export with no model server.

Term sources (unioned, deduped on canonical slug, sorted):

  1. the terminal objective's child chapter objectives' chunk ``concept_tags``,
  2. ``domain_concept_vocabulary`` surface-form matches in the TO's grounded
     chunk text (the rich source — chunk ``concept_tags`` are sparse on real
     corpora and objective ``keyConcepts`` are often empty), and
  3. objective ``keyConcepts`` (when present).

Definition resolution (anti-fabrication, in order):

  1. a definition SENTENCE from a grounding source chunk that mentions the term,
  2. the vocabulary concept's ``definition_hint``,
  3. OMIT the term (never invent a definition).

Each surviving term becomes a ``Block(block_type="vocab_card", ...)`` carrying a
pre-rendered HTML string (so the rewrite tier short-circuits — no LLM) and a
``sourceLink`` deep-link resolved via the FROZEN ``heading_slug`` algorithm
(byte-identical to ``gui/services/source_page.heading_slug`` /
``grounded_answer._fragment_for``).

Gated by ``ED4ALL_KEY_TERMS_PAGE`` (default OFF) at the call site; this module
is a pure library and reads no env directly.
"""

from __future__ import annotations

import html as _html_mod
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.ontology.slugs import canonical_slug

# The deterministic-block marker stamped onto ``Block.template_type`` so the
# rewrite tier recognises a pre-authored key-terms vocab card and SKIPS the LLM
# dispatch (the definition is already grounded + verbatim).
KEY_TERMS_TEMPLATE_TYPE = "key_terms"

# I5 BUG 2 — deterministic per-page (per-TO/week) cap on the number of vocab
# cards. The aggregation (concept_tags ∪ vocabulary surface-forms ∪
# keyConcepts) is broad and on a real corpus yields ~40 terms per TO — far too
# many for a glossary (real glossaries are ~10-20 terms per chapter). The cap
# keeps the MOST relevant terms (source-defined first, then by frequency in the
# TO's grounded chunks + CO statements). Env-overridable via
# ``ED4ALL_KEY_TERMS_MAX_PER_PAGE`` (default 15). Garbage / non-positive values
# fall back to the default (parse-with-fallback, mirroring the other knobs).
KEY_TERMS_MAX_PER_PAGE_ENV = "ED4ALL_KEY_TERMS_MAX_PER_PAGE"
KEY_TERMS_DEFAULT_MAX_PER_PAGE = 15


def resolve_max_terms_per_page(override: Optional[int] = None) -> int:
    """Resolve the per-page key-terms cap (explicit arg > env > default).

    A non-positive / garbage value at any layer falls back to the default
    ``KEY_TERMS_DEFAULT_MAX_PER_PAGE`` (15).
    """
    import os as _os

    if isinstance(override, int) and override > 0:
        return override
    raw = _os.environ.get(KEY_TERMS_MAX_PER_PAGE_ENV, "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return KEY_TERMS_DEFAULT_MAX_PER_PAGE

# Sentence splitter — deliberately simple + deterministic (no NLP model). Splits
# on sentence-final punctuation followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Strip HTML tags from chunk bodies before sentence scanning (chunk text can
# carry residual markup).
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def heading_slug(heading: str) -> str:
    """Slug a heading EXACTLY as ``gui/services/source_page.heading_slug`` does.

    FROZEN contract — must stay byte-identical to
    ``grounded_answer._fragment_for`` and ``source_page.heading_slug``:
    ``re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")``.
    """
    return re.sub(r"[^a-z0-9]+", "-", str(heading).lower()).strip("-")


def _norm_surface(text: str) -> str:
    """Lowercase + collapse whitespace for surface-form containment matching."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _strip_tags(text: str) -> str:
    return _TAG_STRIP_RE.sub(" ", str(text))


def _term_display(canonical: str) -> str:
    """Human-display form of a canonical concept name (title-ish)."""
    cleaned = str(canonical).replace("_", " ").replace("-", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _vocab_concepts(vocabulary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(vocabulary, dict):
        return []
    concepts = vocabulary.get("concepts")
    return [c for c in concepts if isinstance(c, dict)] if isinstance(
        concepts, list
    ) else []


def aggregate_terms(
    *,
    concept_tags: Sequence[str] = (),
    chunk_texts: Sequence[str] = (),
    vocabulary: Optional[Dict[str, Any]] = None,
    key_concepts: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Union the three term sources, dedup on canonical slug, return sorted.

    Returns a list of ``{slug, display, canonical, aliases, definition_hint}``
    dicts (one per unique term), sorted by ``slug``. Anti-fabrication: a term is
    only emitted when it resolves to a real surface (a concept_tag, a
    vocabulary surface-form present in the chunk text, or a keyConcept).

    Args:
        concept_tags: Chunk ``concept_tags`` from the TO's child COs' chunks.
        chunk_texts: The TO's grounded chunk bodies (used for vocabulary
            surface-form containment matching).
        vocabulary: The ``domain_concept_vocabulary`` dict (``{concepts: [...]}``).
        key_concepts: Objective ``keyConcepts`` (when present).
    """
    concepts = _vocab_concepts(vocabulary)
    # Index vocabulary concepts by every surface form (canonical + aliases).
    by_surface: Dict[str, Dict[str, Any]] = {}
    for c in concepts:
        canonical = c.get("canonical")
        if not isinstance(canonical, str) or not canonical:
            continue
        forms = [canonical, *(c.get("aliases") or [])]
        for f in forms:
            if isinstance(f, str) and f.strip():
                by_surface.setdefault(_norm_surface(f), c)

    haystack = _norm_surface(_strip_tags(" ".join(chunk_texts)))

    # Accumulate by canonical slug so the same concept reached via different
    # sources collapses to one card.
    out: Dict[str, Dict[str, Any]] = {}

    def _add(raw_name: str, concept: Optional[Dict[str, Any]]) -> None:
        canonical = (
            concept.get("canonical")
            if isinstance(concept, dict) and concept.get("canonical")
            else raw_name
        )
        slug = canonical_slug(str(canonical))
        if not slug or slug in out:
            return
        out[slug] = {
            "slug": slug,
            "display": _term_display(str(canonical)),
            "canonical": str(canonical),
            "aliases": list(
                (concept or {}).get("aliases") or []
            ) if isinstance(concept, dict) else [],
            "definition_hint": (
                (concept or {}).get("definition_hint")
                if isinstance(concept, dict) else None
            ),
        }

    # 1. concept_tags (resolve to a vocabulary concept when one exists).
    for tag in concept_tags or ():
        if not isinstance(tag, str) or not tag.strip():
            continue
        _add(tag, by_surface.get(_norm_surface(tag)))

    # 2. vocabulary surface-form matches in the grounded chunk text.
    if haystack:
        for c in concepts:
            canonical = c.get("canonical")
            if not isinstance(canonical, str) or not canonical:
                continue
            forms = [canonical, *(c.get("aliases") or [])]
            for f in forms:
                if not isinstance(f, str) or not f.strip():
                    continue
                if _norm_surface(f) in haystack:
                    _add(canonical, c)
                    break

    # 3. objective keyConcepts.
    for kc in key_concepts or ():
        if not isinstance(kc, str) or not kc.strip():
            continue
        _add(kc, by_surface.get(_norm_surface(kc)))

    return [out[k] for k in sorted(out)]


def resolve_definition(
    term: Dict[str, Any],
    *,
    chunks: Sequence[Dict[str, Any]] = (),
) -> Optional[Tuple[str, Optional[Dict[str, Any]]]]:
    """Resolve a definition for ``term``, preferring a source-chunk sentence.

    Returns ``(definition, defining_chunk_or_None)`` or ``None`` when NEITHER a
    source sentence NOR a ``definition_hint`` resolves (anti-fabrication — the
    caller OMITS the term).

    Surface forms searched: the term's canonical name + every alias.
    """
    surfaces = [term.get("canonical") or term.get("display") or ""]
    surfaces.extend(term.get("aliases") or [])
    norm_surfaces = [
        _norm_surface(s) for s in surfaces if isinstance(s, str) and s.strip()
    ]

    # 1. A definition SENTENCE from a grounding chunk that mentions the term.
    for chunk in chunks or ():
        if not isinstance(chunk, dict):
            continue
        body = _strip_tags(str(chunk.get("text") or ""))
        if not body.strip():
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(body):
            s_norm = _norm_surface(sentence)
            if not s_norm:
                continue
            if any(sf and sf in s_norm for sf in norm_surfaces):
                definition = re.sub(r"\s+", " ", sentence).strip()
                if len(definition) >= 12:
                    return definition, chunk

    # 2. definition_hint fallback.
    hint = term.get("definition_hint")
    if isinstance(hint, str) and hint.strip():
        return re.sub(r"\s+", " ", hint).strip(), None

    # 3. OMIT — never invent a definition.
    return None


def resolve_source_link(
    chunk: Optional[Dict[str, Any]],
    *,
    course_slug: str,
) -> Optional[str]:
    """Build a ``/api/learn/source/...`` deep-link for the defining chunk.

    The fragment is the chunk's heading run through the FROZEN ``heading_slug``
    algorithm so it round-trips with the learner source viewer. Returns ``None``
    when no chunk (definition_hint fallback) or no resolvable ``item_path``.
    """
    if not isinstance(chunk, dict):
        return None
    item_path = (
        chunk.get("item_path")
        or chunk.get("itemPath")
        or chunk.get("source_path")
        or ""
    )
    if not item_path:
        return None
    slug = (course_slug or "").strip()
    if not slug:
        return None
    parts = [f"/api/learn/source/{slug}", f"?item_path={item_path}"]
    heading = chunk.get("heading") or chunk.get("section_heading") or ""
    frag = heading_slug(heading) if heading else ""
    if frag:
        parts.append(f"&fragment={frag}")
    return "".join(parts)


def render_term_card(
    *,
    display: str,
    definition: str,
    source_link: Optional[str],
    slug: str,
) -> str:
    """Render ONE vocab card as deterministic HTML.

    Mirrors the Studio ``_render_key_terms_section`` shape: a
    ``data-cf-content-type="key-terms"`` vocab-card div carrying the term, its
    definition, and (when resolvable) an ``/api/learn/source/...`` deep-link.
    Reuses the existing ``.callout`` + ``.key-term`` styles.
    """
    term_html = _html_mod.escape(display)
    def_html = _html_mod.escape(definition)
    link_html = ""
    if source_link:
        link_html = (
            '\n      <p class="key-term-source">'
            f'<a href="{_html_mod.escape(source_link)}">View source</a></p>'
        )
    return (
        f'<div class="callout vocab-card" data-cf-content-type="key-terms" '
        f'data-cf-term="{_html_mod.escape(slug)}">\n'
        f'      <p><span class="key-term">{term_html}</span></p>\n'
        f'      <p>{def_html}</p>{link_html}\n'
        f'    </div>'
    )


def build_key_terms_cards(
    *,
    terms: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]] = (),
    course_slug: str,
) -> List[Dict[str, Any]]:
    """Resolve definitions + links for ``terms``; drop the un-definable ones.

    Returns a list of ``{slug, display, definition, source_link, source_chunk}``
    dicts (the OMISSION of an undefinable term is the anti-fabrication contract).
    Deterministic + order-stable (terms are already sorted by slug).
    """
    cards: List[Dict[str, Any]] = []
    for term in terms or ():
        resolved = resolve_definition(term, chunks=chunks)
        if resolved is None:
            continue  # anti-fabrication OMISSION.
        definition, defining_chunk = resolved
        link = resolve_source_link(defining_chunk, course_slug=course_slug)
        cards.append({
            "slug": term["slug"],
            "display": term["display"],
            "definition": definition,
            "source_link": link,
            "source_chunk": defining_chunk,
        })
    return cards


def _term_frequency(surface: str, haystack: str) -> int:
    """Count word-boundary occurrences of ``surface`` in ``haystack``.

    Both args are expected pre-normalised (lowercased, whitespace-collapsed)
    by the caller; the match is a simple substring-by-word-boundary count.
    """
    if not surface or not haystack:
        return 0
    pattern = re.compile(r"\b" + re.escape(surface) + r"\b")
    return len(pattern.findall(haystack))


def select_capped_cards(
    cards: Sequence[Dict[str, Any]],
    *,
    max_terms: int,
    chunk_texts: Sequence[str] = (),
    co_statements: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Cap resolved cards to the ``max_terms`` MOST relevant, deterministically.

    I5 BUG 2 — selection priority per term (descending):

      1. SOURCE-DEFINED first — a card whose definition resolved to a real
         source chunk (``source_chunk`` is a dict) outranks a card that fell
         back to a ``definition_hint`` (``source_chunk`` is ``None``). A
         source-grounded definition is the higher-confidence glossary entry.
      2. Relevance — frequency of the term's display surface in the TO's
         grounded chunk text PLUS its CO statements (the term most-discussed
         in the week's material is the most useful glossary entry).
      3. Slug — stable tiebreak so the order is deterministic across runs.

    Returns AT MOST ``max_terms`` cards, in the ORIGINAL slug-sorted order
    (the cap only DROPS the least-relevant tail; the surviving cards keep the
    deterministic page order ``build_key_terms_cards`` already produced). When
    ``len(cards) <= max_terms`` the input is returned unchanged.
    """
    cards_list = list(cards or ())
    if max_terms <= 0 or len(cards_list) <= max_terms:
        return cards_list

    haystack = _norm_surface(_strip_tags(" ".join(chunk_texts)))
    co_blob = _norm_surface(" ".join(s for s in co_statements if isinstance(s, str)))

    def _rank_key(card: Dict[str, Any]) -> Tuple[int, int, str]:
        source_defined = 1 if isinstance(card.get("source_chunk"), dict) else 0
        surface = _norm_surface(
            str(card.get("display") or card.get("slug") or "")
        )
        freq = _term_frequency(surface, haystack) + _term_frequency(
            surface, co_blob
        )
        # Pure ascending composite key: negate the desc-fields (source_defined,
        # freq) so the strongest sort first; slug ascending is the stable
        # deterministic tiebreak.
        return (-source_defined, -freq, str(card.get("slug") or ""))

    ranked = sorted(cards_list, key=_rank_key)
    kept_slugs = {str(c.get("slug") or "") for c in ranked[:max_terms]}
    # Restore the original (slug-sorted) page order among the survivors so the
    # page reads alphabetically; the cap only DROPS the least-relevant tail.
    return [c for c in cards_list if str(c.get("slug") or "") in kept_slugs]


def build_key_terms_page_html(cards: Sequence[Dict[str, Any]]) -> str:
    """Render the inner body HTML for a key-terms page from resolved cards."""
    rendered = [
        render_term_card(
            display=c["display"],
            definition=c["definition"],
            source_link=c.get("source_link"),
            slug=c["slug"],
        )
        for c in cards
    ]
    return "\n    ".join(rendered)
