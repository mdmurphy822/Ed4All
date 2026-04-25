"""Wave 76: vocabulary-driven LO retag + parent-outcome rollup.

External KG-quality review of the rdf-shacl-550 archive surfaced four
real coverage gaps where content exists but is mis-tagged:

    co-18 — SHACL Core constraint components
            (sh:minCount / maxCount / datatype / class / pattern / in)
    co-19 — SHACL validation report
            (sh:result / focusNode / severity, "validation report")
    co-22 — Trade-offs across SHACL Core / SHACL-SPARQL / SHACL Rules
    to-07 — Capstone integration (42 chunks already cite co-25..co-29
            but never roll up to the terminal)

This module exposes two pure-data helpers:

* ``retag_chunk_outcomes(chunk, parent_map=None)`` — apply the
  vocabulary retag pass + parent-outcome rollup to a single chunk's
  ``learning_outcome_refs`` in place. Both rules are *additive*: never
  remove an existing ref, only append.
* ``build_parent_map(objectives)`` — build the
  ``component_id -> terminal_id`` map from a loaded ``objectives.json``
  payload (handles both ``component_objectives[]`` and the legacy
  ``chapter_objectives[]`` shape).

The helpers are pure functions to keep them trivially callable from
both ``CourseProcessor._create_chunk`` (emit time) and the retroactive
regen script in ``scripts/wave76_retag_chunks.py``. They are
idempotent — running the retag twice on the same chunk does not
duplicate refs.

Wave 81 generalization
----------------------
The hand-authored ``RETAG_VOCABULARIES`` table only covered three COs
(co-18 / co-19 / co-22) — the v2 strict packet validator surfaced
co-09 + co-10 as having no teaching/assessment chunks because their
CO statements weren't represented anywhere in the curated table.

To close that gap without forcing a hand-authored entry per CO per
course, this module now also exposes:

* ``auto_extract_vocabulary(co_statement)`` — pure-data deterministic
  helper that derives keyword candidates from a single CO statement.
  Preserves ``prefix:term`` patterns (rdfs:label, sh:minCount), strips
  bloom verbs + stopwords, prefers technical tokens (anything with
  ``:``, dotted, ALLCAPS, or CamelCase). Conservative on bigrams +
  generic English singles to avoid over-tagging at substring-match
  time.
* ``build_auto_vocabularies(objectives)`` — builds the
  ``co_id -> [terms]`` map from an objectives payload by running
  ``auto_extract_vocabulary`` over every CO statement.
* ``merged_vocabularies(objectives)`` — merges
  ``RETAG_VOCABULARIES`` (curated) with the auto-extracted map. Auto
  fills in for COs that are not in curated; curated entries override
  by ID (same key in curated replaces the auto entry entirely so the
  curated list is the authoritative one for known problem cases).

The retag rule (`retag_chunk_outcomes`) accepts an optional
``vocabularies`` arg so emit-time call sites
(``CourseProcessor._create_chunk``) can pass the per-run merged map
once and reuse it for every chunk.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


# Vocabulary lists are taken verbatim from the words that appear in
# each CO's ``statement`` field, augmented with the constraint /
# property names called out in the canonical vocabulary surveys (e.g.
# the SHACL spec's Core Constraint Components table). Matching is
# substring-style on ``chunk["text"]`` — case-sensitive because the
# SHACL/SHACL-SPARQL/SHACL Rules tokens are proper nouns.
RETAG_VOCABULARIES: Dict[str, List[str]] = {
    "co-09": [
        # Wave 81 curated override: vocabulary-documentation predicates
        # surfaced by external KG-quality review (auto-extraction also
        # picks these up, but the curated entry pins the authoritative
        # list and adds the multi-word "vocabulary documentation"
        # phrase that the heuristic doesn't synthesize from the CO
        # statement alone).
        "rdfs:label",
        "rdfs:comment",
        "rdfs:seeAlso",
        "rdfs:isDefinedBy",
        "vocabulary documentation",
    ],
    "co-10": [
        # Wave 81 curated override: vocabulary-design phrases. The CO
        # statement says "design a domain-specific RDFS vocabulary" but
        # textbook chapters consistently use the noun-phrase forms
        # below; auto-extraction can't synthesize them, so we pin them.
        "vocabulary design",
        "class granularity",
        "property reuse",
        "mint policy",
        "namespace strategy",
        "domain-specific",
    ],
    "co-18": [
        # SHACL Core constraint component vocabulary.
        "sh:minCount",
        "sh:maxCount",
        "sh:datatype",
        "sh:class",
        "sh:pattern",
        "sh:in",
        "sh:minLength",
        "sh:maxLength",
        "sh:nodeKind",
        "sh:hasValue",
    ],
    "co-19": [
        # SHACL validation report shape.
        "sh:result",
        "sh:resultMessage",
        "sh:resultPath",
        "sh:focusNode",
        "sh:resultSeverity",
        "validation report",
        "Violation",
        "Warning",
        "Info",
        "sh:conforms",
    ],
    "co-22": [
        # Trade-off / comparison vocabulary.
        "SHACL-SPARQL",
        "sh:sparql",
        "SHACL Rules",
        "SHACL Advanced Features",
        "SHACL-AF",
        "vs Core",
        "vs SPARQL",
        "trade-off",
    ],
}


# ---------------------------------------------------------------------
# Wave 81: auto-extraction
# ---------------------------------------------------------------------

# Stopwords + cognitive vocabulary stripped during auto-extraction.
# Bloom verbs cover the canonical Anderson/Krathwohl cognitive domain
# revised taxonomy verbs; stopwords are an English minimal set so the
# heuristic is deterministic without an NLTK dependency.
_BLOOM_VERBS: frozenset = frozenset({
    # Remember
    "define", "list", "recall", "identify", "name", "recognize",
    "label", "match", "select", "state", "describe",
    # Understand
    "explain", "summarize", "interpret", "paraphrase", "classify",
    "compare", "contrast", "differentiate", "distinguish",
    "discuss",
    # Apply
    "apply", "demonstrate", "use", "solve", "implement", "execute",
    "construct", "write", "model", "produce", "perform", "operate",
    # Analyze
    "analyze", "examine", "deconstruct", "organize", "attribute",
    "structure", "integrate", "diagram", "outline",
    # Evaluate
    "evaluate", "judge", "justify", "critique", "assess", "defend",
    "appraise", "argue", "support", "validate", "verify",
    # Create
    "create", "design", "develop", "formulate", "compose", "plan",
    "generate", "devise", "architect", "author", "synthesize",
    # Generic process verbs that show up alongside Bloom verbs
    "consider", "predict", "choose", "declare", "show", "find",
})

_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "so", "for", "to", "of",
    "in", "on", "at", "by", "with", "from", "as", "into", "onto",
    "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "have", "has", "had", "having", "this", "that", "these",
    "those", "it", "its", "their", "they", "them", "his", "her",
    "he", "she", "we", "you", "your", "our", "i", "me", "my",
    "than", "then", "when", "where", "while", "if", "else", "not",
    "no", "yes", "such", "any", "all", "some", "each", "every",
    "given", "between", "across", "via", "per", "about", "over",
    "under", "above", "below", "before", "after", "during", "until",
    "how", "what", "why", "which", "who", "whom", "whose",
    "can", "will", "would", "should", "could", "may", "might",
    "must", "shall", "also", "only", "even", "just", "very",
    "well", "much", "many", "more", "most", "less", "least",
    "same", "different", "other", "another", "common", "basic",
    "new", "small", "big", "large", "real-world", "real",
    "appropriate", "correct",
})

# Tokens we deliberately keep even when short / lowercase because
# they're domain-specific identifiers in the rdf-shacl corpus and
# similar technical curricula. Conservative — only universally
# domain-specific.
_PROTECTED_TOKENS: frozenset = frozenset({
    "rdf", "rdfs", "owl", "sparql", "shacl", "iri", "iris",
    "xsd", "uri", "uris", "json", "xml", "ttl", "ld",
})

# Generic English single-word tokens we never want as a vocabulary
# entry — too noisy at the chunk-text substring-match step. Lives at
# module scope so tests + the bigram filter can share one list.
#
# NOTE: SPARQL keyword tokens (SELECT, CONSTRUCT, ASK, DESCRIBE,
# FILTER, OPTIONAL, UNION, ORDER, LIMIT, OFFSET, GROUP, COUNT, ...)
# are intentionally NOT blacklisted here. The retag pass uses
# case-sensitive substring matching so the ALLCAPS form will only
# hit chunks that quote the SPARQL keyword (legitimate signal),
# not chunks that contain the lowercase English verb.
_GENERIC_SINGLES: frozenset = frozenset({
    "graph", "graphs", "data", "datum", "form", "forms", "type",
    "types", "scenario", "scenarios", "thing", "things", "item",
    "items", "case", "cases", "rule", "rules", "set", "sets",
    "node", "nodes", "term", "terms", "value", "values", "name",
    "names", "kind", "kinds", "role", "roles", "fact", "facts",
    "result", "results", "shape", "shapes", "level", "levels",
    "user", "users", "step", "steps", "unit", "units", "chosen",
    "given", "specific", "general", "various", "multiple",
    "single", "several", "many", "few", "components", "component",
    "triple", "triples", "literals", "consumers", "discover",
    "vocabularies", "vocabulary", "abstractions", "defined",
    "hierarchies", "entailment", "derive", "queries", "patterns",
    "modifiers", "functions", "function", "endpoints", "endpoint",
    "characteristics", "expressions", "constraints", "constraint",
    "restrictions", "restriction", "individuals", "individual",
    "decisions", "decision", "deliverable", "audience", "artifacts",
    "documented", "coherent", "production", "produce", "produced",
    "information", "having", "average",
})

_PREFIX_TERM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*:[A-Za-z][A-Za-z0-9_\-]*$")
_CAMELCASE_RE = re.compile(r"^[a-z]+[A-Z][A-Za-z0-9]*$|^[A-Z][a-z]+[A-Z][A-Za-z0-9]*$")
_HAS_UPPER_RE = re.compile(r"[A-Z]")


def _tokenize_preserving_prefixes(text: str) -> List[str]:
    """Tokenize on whitespace + punctuation, preserving ``prefix:term``.

    ``rdfs:label`` and ``sh:minCount`` survive intact; surrounding
    commas / parentheses / periods get stripped. Hyphens within a
    token are preserved (e.g., ``domain-specific``, ``trade-off``,
    ``SHACL-SPARQL``).
    """
    if not text:
        return []
    raw = text.split()
    out: List[str] = []
    for tok in raw:
        cleaned = tok.strip(" \t\n\r,.;()[]{}\"'!?")
        if not cleaned:
            continue
        cleaned = cleaned.strip(" \t\n\r,.;()[]{}\"'!?")
        if not cleaned:
            continue
        out.append(cleaned)
    return out


def _is_technical_term(token: str) -> bool:
    """Return True if the token is a technical/domain identifier worth keeping."""
    if not token:
        return False
    if ":" in token and _PREFIX_TERM_RE.match(token):
        return True
    if "-" in token and any(_HAS_UPPER_RE.search(p) for p in token.split("-")):
        # SHACL-SPARQL style.
        return True
    if _CAMELCASE_RE.match(token):
        return True
    if token.isupper() and len(token) >= 2:
        return True
    if token.lower() in _PROTECTED_TOKENS:
        return True
    return False


def _is_stopword_or_verb(token: str) -> bool:
    """True for stopwords, bloom verbs, or single-character tokens."""
    if len(token) <= 1:
        return True
    low = token.lower()
    if low in _STOPWORDS:
        return True
    if low in _BLOOM_VERBS:
        return True
    return False


def _is_stopword(token: str) -> bool:
    """Variant that drops *only* stopwords (keeps bloom verbs)."""
    if len(token) <= 1:
        return True
    return token.lower() in _STOPWORDS


def auto_extract_vocabulary(co_statement: str) -> List[str]:
    """Derive keyword candidates from a CO statement.

    Strategy (deterministic; no LLM dependency):
      1. Tokenize on whitespace; preserve ``prefix:term`` patterns
         (rdfs:label, sh:minCount).
      2. Strip the *leading* bloom verb (the CO's stated cognitive
         verb).
      3. Drop English stopwords.
      4. Keep technical tokens — anything with ``:`` (prefix:term),
         CamelCase, ALLCAPS multi-letter, or in the protected
         domain-identifier set (RDF, OWL, RDFS, SPARQL, SHACL, ...).
      5. Keep hyphenated multi-tokens (``domain-specific``,
         ``end-to-end``, ``SHACL-SPARQL``).
      6. Keep specific multi-word bigrams: only when *both* halves
         are themselves technical (prefix:term, CamelCase, ALLCAPS,
         hyphenated, or protected). Plain-English bigrams ("RDF
         graph", "validation rules") are dropped because they
         over-tag at substring-match time.
      7. Cap at 10 candidates per CO; rank technical-singles first,
         then technical-bigrams, then hyphenated singles.

    The conservative bigram rule is the Wave 81 design choice that
    keeps auto-extraction useful without flooding curated coverage
    (otherwise auto-vocab inflates co-01 from 8 chunks to 246 in the
    rdf-shacl-551-2 corpus). For COs whose statement carries no
    technical tokens — typically generic Bloom verbs only — the
    extractor returns a short list, and curated ``RETAG_VOCABULARIES``
    overrides cover the gaps (see co-09 / co-10).

    Returns an empty list for stopword-only / empty input.
    """
    if not co_statement or not isinstance(co_statement, str):
        return []

    tokens = _tokenize_preserving_prefixes(co_statement)
    if not tokens:
        return []

    # Step 2: strip the leading bloom verb. Only the first token is
    # treated as the cognitive verb.
    if tokens and tokens[0].lower() in _BLOOM_VERBS:
        tokens = tokens[1:]
    if not tokens:
        return []

    # Phase 1: technical-term singles (highest priority).
    tech_singles: List[str] = []
    tech_seen: set = set()
    for tok in tokens:
        if _is_stopword(tok):
            continue
        if _is_technical_term(tok):
            if tok not in tech_seen:
                tech_seen.add(tok)
                tech_singles.append(tok)

    # Phase 2: hyphenated multi-tokens (single token but contains "-").
    # These often carry domain meaning ("domain-specific", "end-to-end",
    # "SHACL-SPARQL", "trade-off") — keep them after technical-singles.
    hyphenated: List[str] = []
    hyphenated_seen: set = set()
    for tok in tokens:
        if "-" not in tok:
            continue
        if len(tok) < 4:
            continue
        if _is_stopword(tok):
            continue
        if tok in tech_seen:
            continue  # already counted as technical
        low = tok.lower()
        if low in hyphenated_seen:
            continue
        hyphenated_seen.add(low)
        hyphenated.append(tok)

    # Phase 3: technical bigrams. Only emit when *both* halves are
    # themselves technical (prefix:term, CamelCase, ALLCAPS, hyphenated,
    # or protected). This is what differentiates "sh:minCount
    # sh:maxCount" (kept — both technical) from "RDF graph" (dropped —
    # half is generic).
    tech_bigrams: List[str] = []
    bigram_seen: set = set()
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if _is_stopword(a) or _is_stopword(b):
            continue
        if _is_stopword_or_verb(a) or _is_stopword_or_verb(b):
            continue
        if a.isdigit() or b.isdigit():
            continue
        if any(ch in a + b for ch in "()/\\,"):
            continue
        if not (_is_technical_term(a) and _is_technical_term(b)):
            continue
        bigram = f"{a} {b}"
        key = bigram.lower()
        if key in bigram_seen:
            continue
        bigram_seen.add(key)
        tech_bigrams.append(bigram)

    # Rank: technical singles, then technical bigrams, then hyphenated
    # singles (often capture the most specific multi-word concepts).
    # Cap at 10 candidates per CO.
    #
    # The generic-blacklist check uses the *original* token form so
    # ALLCAPS / CamelCase technical identifiers ("SELECT" the SPARQL
    # keyword) survive even when their lowercase-form ("select" the
    # English verb) lives on the blacklist. Bigrams and lowercase
    # singles still hit the blacklist via lower-cased halves.
    ranked: List[str] = []
    seen_final: set = set()
    for term in tech_singles + tech_bigrams + hyphenated:
        key = term.lower()
        if key in seen_final:
            continue
        # ALLCAPS / non-trivial-case technical tokens bypass the
        # generic blacklist; they're domain-specific identifiers
        # whose case carries signal at chunk-text match time.
        is_caseful_technical = (
            term.isupper() and len(term) >= 2
        ) or (term != term.lower() and ":" in term)
        if not is_caseful_technical and key in _GENERIC_SINGLES:
            continue
        seen_final.add(key)
        ranked.append(term)
        if len(ranked) >= 10:
            break

    return ranked


def build_auto_vocabularies(
    objectives: Optional[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Run ``auto_extract_vocabulary`` over every CO in ``objectives``.

    Returns a ``{co_id: [terms]}`` map. Empty input → empty dict. CO
    IDs are normalized to lowercase to match the
    ``RETAG_VOCABULARIES`` key style.
    """
    if not isinstance(objectives, Mapping):
        return {}

    out: Dict[str, List[str]] = {}

    def _consider(obj: Mapping[str, Any]) -> None:
        cid = obj.get("id")
        statement = obj.get("statement") or obj.get("text") or ""
        if not isinstance(cid, str) or not isinstance(statement, str):
            return
        terms = auto_extract_vocabulary(statement)
        if terms:
            out[cid.lower()] = terms

    # Canonical shape: component_objectives[].
    for entry in objectives.get("component_objectives") or []:
        if isinstance(entry, Mapping):
            _consider(entry)

    # Legacy / loader shape: chapter_objectives[] (with optional
    # nested objectives[]).
    for ch in objectives.get("chapter_objectives") or []:
        if isinstance(ch, Mapping) and "objectives" in ch:
            inner: Iterable[Any] = ch.get("objectives") or []
        else:
            inner = [ch]
        for obj in inner:
            if isinstance(obj, Mapping):
                _consider(obj)

    # Terminal outcomes — also auto-extract so retag can fire on the
    # TO-NN level when chunk text matches the terminal's vocabulary.
    for to in objectives.get("terminal_outcomes") or []:
        if isinstance(to, Mapping):
            _consider(to)

    return out


def merged_vocabularies(
    objectives: Optional[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Merge curated ``RETAG_VOCABULARIES`` with auto-extracted map.

    Auto-extracted entries cover every CO in ``objectives``; curated
    entries override by key (same CO id in curated replaces the
    auto entry entirely so the curated list stays the authoritative
    source for known problem cases).
    """
    auto = build_auto_vocabularies(objectives)
    merged: Dict[str, List[str]] = dict(auto)
    # Curated overrides win — replace whole list.
    for cid, terms in RETAG_VOCABULARIES.items():
        merged[cid.lower()] = list(terms)
    return merged


def build_parent_map(
    objectives: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    """Return a ``component_id -> terminal_id`` mapping.

    Accepts either the canonical ``objectives.json`` shape (with
    ``component_objectives[]``) or the in-memory loader shape used by
    ``CourseProcessor.objectives`` (which carries
    ``chapter_objectives[]`` with ``parent_to`` / ``parent_terminal``).
    Unknown / missing inputs return an empty dict so callers can rely
    on ``parent_map.get(co_id)`` without ``None`` checks.
    """
    if not isinstance(objectives, Mapping):
        return {}

    parent_map: Dict[str, str] = {}

    # Canonical shape: objectives.json with component_objectives[].
    for entry in objectives.get("component_objectives") or []:
        if not isinstance(entry, Mapping):
            continue
        cid = entry.get("id")
        parent = entry.get("parent_terminal") or entry.get("parent_to")
        if isinstance(cid, str) and isinstance(parent, str):
            parent_map[cid.lower()] = parent.lower()

    # Loader shape: chapter_objectives[] (sometimes wrapped in
    # {"objectives": [...]}).
    for ch in objectives.get("chapter_objectives") or []:
        if isinstance(ch, Mapping) and "objectives" in ch:
            inner: Iterable[Any] = ch.get("objectives") or []
        else:
            inner = [ch]
        for obj in inner:
            if not isinstance(obj, Mapping):
                continue
            cid = obj.get("id")
            parent = obj.get("parent_terminal") or obj.get("parent_to")
            if isinstance(cid, str) and isinstance(parent, str):
                parent_map.setdefault(cid.lower(), parent.lower())

    return parent_map


def _vocabulary_matches(
    text: str,
    vocabularies: Optional[Mapping[str, List[str]]] = None,
) -> List[str]:
    """Return the list of CO IDs whose vocabulary matches ``text``.

    ``vocabularies`` defaults to the curated ``RETAG_VOCABULARIES``
    table for backward compatibility. Wave 81 emit-time call sites
    pass the merged (curated + auto-extracted) map per run so coverage
    spans every CO present in the active objectives payload.
    """
    if not text:
        return []
    table = vocabularies if vocabularies is not None else RETAG_VOCABULARIES
    matched: List[str] = []
    for co_id, terms in table.items():
        for term in terms:
            if term and term in text:
                matched.append(co_id)
                break
    return matched


def retag_chunk_outcomes(
    chunk: Dict[str, Any],
    parent_map: Optional[Mapping[str, str]] = None,
    vocabularies: Optional[Mapping[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Apply vocabulary retag + parent rollup to ``chunk`` in place.

    Both rules are additive — existing refs are never removed. Refs
    are deduplicated case-insensitively, with the first-seen casing
    retained so callers that opt into ``TRAINFORGE_PRESERVE_LO_CASE``
    keep their authoritative casing. Returns the same chunk for
    chaining.

    Wave 81: ``vocabularies`` lets emit-time callers pass the per-run
    merged map (curated + auto-extracted). Defaults to the curated
    ``RETAG_VOCABULARIES`` table when None so legacy callers see the
    pre-Wave-81 behavior.
    """
    if not isinstance(chunk, dict):
        return chunk

    refs = chunk.get("learning_outcome_refs")
    if not isinstance(refs, list):
        refs = []

    seen: Dict[str, str] = {}
    out: List[str] = []
    for ref in refs:
        if not isinstance(ref, str):
            continue
        key = ref.lower()
        if key in seen:
            continue
        seen[key] = ref
        out.append(ref)

    def _add(ref: str) -> None:
        if not isinstance(ref, str) or not ref:
            return
        key = ref.lower()
        if key in seen:
            return
        seen[key] = ref
        out.append(ref)

    # Part 1: vocabulary-driven retag against chunk text.
    text = chunk.get("text") or ""
    if isinstance(text, str):
        for co_id in _vocabulary_matches(text, vocabularies=vocabularies):
            _add(co_id)

    # Part 2: parent-rollup. For every co-NN in the (now-extended) ref
    # list, also add its terminal parent.
    if parent_map:
        # Snapshot the keys we'll iterate over so that adding parents
        # while looping doesn't re-trigger lookups on already-added
        # parents (parents shouldn't appear in parent_map anyway, but
        # be defensive).
        for ref in list(out):
            parent = parent_map.get(ref.lower())
            if parent:
                _add(parent)

    chunk["learning_outcome_refs"] = out
    return chunk
