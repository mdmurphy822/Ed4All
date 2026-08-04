"""Wave 82 acronym-preserving label helper.

The RDF/SHACL calibration corpus audit (Section F) found concept labels emitted as
``Owl 2 Rl`` instead of ``OWL 2 RL`` because the legacy code path used
``slug.replace("-", " ").title()`` — Python's ``.title()`` capitalizes
the first letter of each word and lowercases the rest, breaking known
acronyms.

This helper applies title-case but forces a curated set of known
acronyms (W3C standards + their profiles + general technical terms)
to uppercase. Pure function, no I/O, deterministic.
"""

from __future__ import annotations

import os
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
    "pca",     # Principal Component Analysis
    "vq",      # Vector Quantization
    # RAG / agent-framework tooling (rdf-shacl audit Section F follow-up).
    "ragas",   # RAG Assessment evaluation library
    "lcel",    # LangChain Expression Language
    "faiss",   # Facebook AI Similarity Search
    "nim",     # NVIDIA Inference Microservice
    "nv",      # NVIDIA NV-Embed family prefix
    "gpu",
    "nvidia",
    "openai",
    "chatgpt",
})


# Proper-noun / brand casing for tokens that are not pure uppercase
# acronyms (mixed-case brands). Lookup is case-insensitive; the value
# is the exact canonical rendering. Kept separate from
# :data:`KNOWN_ACRONYMS` because these are not all-caps.
_KNOWN_TOKEN_CASING: dict[str, str] = {
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "chatgpt": "ChatGPT",
}


# Run-together lowercase compounds re-split at known word boundaries.
# Conservative curated map only — no general English compound-splitting
# (too error-prone). Keyed on the fully-lowercased, hyphen-stripped
# compound; value is the space-separated form fed back through
# acronym-aware title-casing. See the rdf-shacl audit (Section F).
_LABEL_RESPLIT: dict[str, str] = {
    "runnableassign": "runnable assign",
    "checkpointresume": "checkpoint resume",
    "ragasevaluatorchain": "ragas evaluator chain",
    "retrievalaugmentation": "retrieval augmentation",
    # Run-together proper-noun brand splits (RAG-course audit).
    # Mixed-case brands keep their final casing verbatim; the
    # acronym-aware title-caser passes through any value that already
    # carries internal upper-casing (see :func:`_token_or_acronym`), so
    # these survive the downstream title-case pass unaltered.
    "langgraph": "LangGraph",
    "langserve": "LangServe",
    "llamaindex": "LlamaIndex",
    "stroutputparser": "StrOutputParser",
    "retrievertool": "RetrieverTool",
    "smolagents": "SmolAgents",
    # Multi-word brand splits — lowercase here, title-cased downstream.
    "centralorchestrator": "central orchestrator",
    "dockerrouter": "docker router",
}


# Single-token misspelling corrections applied per-token during label
# rendering. Keyed on the lowercased token; value is the corrected
# token (re-cased downstream by the acronym-aware title-caser).
_LABEL_TOKEN_FIXES: dict[str, str] = {
    "exercice": "exercise",
}


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
    """Render one token: known-acronym uppercase, brand casing, else title.

    Also applies single-token misspelling corrections
    (:data:`_LABEL_TOKEN_FIXES`) and handles a possessive ``'s`` suffix
    so ``openais`` → ``OpenAI's`` rather than ``Openais``.
    """
    if not token:
        return ""
    # Pre-cased brand pass-through: a token that already carries internal
    # mixed casing (e.g. "LangGraph", "StrOutputParser") was injected by
    # the curated resplit map and must not be re-title-cased (which would
    # flatten it to "Langgraph"). All-lower / all-upper / simple-Title
    # tokens fall through to the normal casing logic below.
    if token not in (token.lower(), token.upper(), token.title()):
        return token
    lowered = token.lower()
    # Possessive: strip a trailing bare "s" when the stem is a known
    # acronym or brand (slug form drops the apostrophe). "openais" →
    # "OpenAI's". Only fires when the stem is recognized AND the full
    # token is NOT itself a known acronym/brand — otherwise "rdfs"
    # (RDF + s) would mangle to "RDF's" and ordinary plurals would be
    # over-matched.
    if (
        lowered.endswith("s")
        and len(lowered) > 1
        and lowered not in KNOWN_ACRONYMS
        and lowered not in _KNOWN_TOKEN_CASING
    ):
        stem = lowered[:-1]
        if stem in KNOWN_ACRONYMS or stem in _KNOWN_TOKEN_CASING:
            return _token_or_acronym(stem) + "'s"
    fixed = _LABEL_TOKEN_FIXES.get(lowered)
    if fixed is not None:
        lowered = fixed
        token = fixed
    if lowered in _KNOWN_TOKEN_CASING:
        return _KNOWN_TOKEN_CASING[lowered]
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
    # NVIDIA NV-Embed embedding model: the slug carries a trailing
    # "-vq" quantization qualifier and a run-together "nv-embed"; the
    # canonical model name is "NV-Embed" (NVIDIA's own capitalization),
    # so pin the full rendering rather than reconstruct it token-wise.
    "nvidia-nv-embed-vq": "NVIDIA NV-Embed",
}


def slug_to_label(slug: str) -> str:
    """Convert a concept slug to a human label, preserving acronyms.

    Replaces hyphens with spaces and applies acronym-aware title-case.
    Drop-in replacement for ``slug.replace("-", " ").title()`` at the
    label-emit sites in ``Trainforge/process_course.py`` and
    ``Trainforge/rag/graphs/pedagogy_graph_builder.py``.

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
    spaced = slug.replace("-", " ")
    if _normalize_labels_enabled():
        spaced = _apply_label_resplit(spaced)
    return titlecase_with_acronyms(spaced)


def _normalize_labels_enabled() -> bool:
    """Return whether label normalization (resplit) is opt-in enabled.

    Gated behind ``TRAINFORGE_NORMALIZE_LABELS`` (default off) for
    byte-stability of previously-emitted labels. Acronym casing,
    brand casing, and misspelling fixes always apply; only the
    run-together compound re-splitting is flag-gated, since that is the
    behavior that re-segments existing tokens.
    """
    return os.environ.get("TRAINFORGE_NORMALIZE_LABELS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _apply_label_resplit(spaced: str) -> str:
    """Re-split run-together lowercase compounds at curated boundaries.

    Operates per whitespace-separated token; each token's lowercased
    form is looked up in :data:`_LABEL_RESPLIT`. Unmatched tokens pass
    through unchanged. Conservative: no general compound-splitting.
    """
    out: list[str] = []
    for token in spaced.split():
        replacement = _LABEL_RESPLIT.get(token.lower())
        out.append(replacement if replacement is not None else token)
    return " ".join(out)


__all__ = [
    "KNOWN_ACRONYMS",
    "titlecase_with_acronyms",
    "slug_to_label",
]


# Doctest-style examples for the new normalization behavior live in the
# test suite (``lib/ontology/tests/test_labels.py``) because the
# resplit path is gated behind ``TRAINFORGE_NORMALIZE_LABELS`` and
# cannot be exercised by a bare doctest run.
