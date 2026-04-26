"""Wave 82 acronym-preserving label helper.

The rdf-shacl-551 audit (Section F) found concept labels emitted as
``Owl 2 Rl`` instead of ``OWL 2 RL`` because the legacy code path used
``slug.replace("-", " ").title()`` — Python's ``.title()`` capitalizes
the first letter of each word and lowercases the rest, breaking known
acronyms.

This helper applies title-case but forces a curated set of known
acronyms (W3C standards + their profiles + general technical terms)
to uppercase. Pure function, no I/O, deterministic.
"""

from __future__ import annotations

from typing import Iterable

# Curated set of acronyms that must always render uppercase. Lowercased
# at storage time so lookup matches case-insensitively. Conservative —
# only adds entries that are unambiguously acronyms in this domain.
# Adding overly-common short tokens (like "id", "as") would over-match
# in human-readable context.
KNOWN_ACRONYMS: frozenset[str] = frozenset({
    # W3C semantic-web standards.
    "rdf",
    "rdfs",
    "owl",
    "shacl",
    "sparql",
    "ttl",     # Turtle file extension acronym
    "skos",
    "void",
    # OWL 2 profiles.
    "rl",      # Rule Language profile
    "el",      # Existential profile
    "ql",      # Query profile
    "dl",      # Description Logic profile
    # Identifiers / URIs.
    "iri",
    "uri",
    "url",
    "urn",
    # Serializations.
    "xml",
    "ld",      # JSON-LD's "LD" segment
    "json",    # less universally caps in prose, but always caps in tech contexts
    "csv",
    # Validation / query.
    "rdfa",
    "qti",
    # Generic tech.
    "api",
    "html",
    "css",
    "lms",
    "rag",
    "slm",
    "llm",
})


def titlecase_with_acronyms(text: str) -> str:
    """Title-case ``text`` while forcing :data:`KNOWN_ACRONYMS` uppercase.

    Splits on whitespace; for each token, uppercases when the
    lowercased form is in the acronym set, otherwise applies
    ``str.title()``. Non-string / empty input returns ``""``.

    Examples:
        >>> titlecase_with_acronyms("owl 2 rl")
        'OWL 2 RL'
        >>> titlecase_with_acronyms("rdf graph")
        'RDF Graph'
        >>> titlecase_with_acronyms("turtle syntax")
        'Turtle Syntax'
        >>> titlecase_with_acronyms("json-ld primer")
        'JSON-LD Primer'
    """
    if not isinstance(text, str) or not text:
        return ""
    tokens: list[str] = []
    for token in text.split():
        # Hyphenated tokens get per-segment treatment so "json-ld" → "JSON-LD".
        if "-" in token:
            parts = [_token_or_acronym(seg) for seg in token.split("-")]
            tokens.append("-".join(parts))
        else:
            tokens.append(_token_or_acronym(token))
    return " ".join(tokens)


def _token_or_acronym(token: str) -> str:
    """Uppercase if ``token.lower()`` is a known acronym, else title-case."""
    if not token:
        return ""
    lowered = token.lower()
    if lowered in KNOWN_ACRONYMS:
        return token.upper()
    return token.title()


# Slug-level overrides for compound acronyms whose canonical W3C form
# keeps the hyphen (``JSON-LD``, ``N-Triples``). Without these the
# default ``replace("-", " ")`` pass would emit ``JSON LD`` /
# ``N Triples``. Match is exact-slug; broader prefixes (e.g.
# ``json-ld-context``) fall through to the standard hyphen-stripping
# logic.
_SLUG_LABEL_OVERRIDES: dict[str, str] = {
    "json-ld": "JSON-LD",
    "n-triples": "N-Triples",
    "n-quads": "N-Quads",
    "rdf-xml": "RDF/XML",
}


def slug_to_label(slug: str) -> str:
    """Convert a concept slug to a human label, preserving acronyms.

    Replaces hyphens with spaces and applies acronym-aware title-case.
    Drop-in replacement for ``slug.replace("-", " ").title()`` at the
    label-emit sites in ``Trainforge/process_course.py`` and
    ``Trainforge/pedagogy_graph_builder.py``.

    Compound-acronym slugs (``json-ld``, ``n-triples``) consult
    :data:`_SLUG_LABEL_OVERRIDES` first to preserve their canonical
    hyphenated form (``JSON-LD``, ``N-Triples``).

    Examples:
        >>> slug_to_label("owl-2-rl")
        'OWL 2 RL'
        >>> slug_to_label("rdf-graph")
        'RDF Graph'
        >>> slug_to_label("blank-node")
        'Blank Node'
        >>> slug_to_label("json-ld")
        'JSON-LD'
    """
    if not slug:
        return ""
    override = _SLUG_LABEL_OVERRIDES.get(slug.lower())
    if override is not None:
        return override
    return titlecase_with_acronyms(slug.replace("-", " "))


__all__ = [
    "KNOWN_ACRONYMS",
    "titlecase_with_acronyms",
    "slug_to_label",
]
