"""Runtime-derived learning-outcome retagging and parent rollup.

The helpers in this module add objective references without removing existing
references. Vocabulary terms are derived exclusively from objective records
supplied at runtime; no hidden course vocabulary is applied.

All transformations are deterministic and idempotent. The module supports both
the canonical ``component_objectives[]`` shape and the alternate nested
``chapter_objectives[]`` shape.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

# Stopwords and cognitive verbs excluded from runtime vocabulary extraction.
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

_PREFIX_TERM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*:[A-Za-z][A-Za-z0-9_\-]*$")
_CAMELCASE_RE = re.compile(r"^[a-z]+[A-Z][A-Za-z0-9]*$|^[A-Z][a-z]+[A-Z][A-Za-z0-9]*$")
_HAS_UPPER_RE = re.compile(r"[A-Z]")


def _tokenize_preserving_prefixes(text: str) -> List[str]:
    """Tokenize on whitespace + punctuation, preserving ``prefix:term``.

    A token such as ``schema:item`` survives intact; surrounding
    commas / parentheses / periods get stripped. Hyphens within a
    token are preserved.
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
        # Preserve case-sensitive compound identifiers.
        return True
    if _CAMELCASE_RE.match(token):
        return True
    if token.isupper() and len(token) >= 2:
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
      1. Tokenize on whitespace; preserve ``prefix:term`` patterns.
      2. Strip the *leading* bloom verb (the CO's stated cognitive
         verb).
      3. Drop English stopwords.
      4. Keep technical tokens — ``prefix:term``, CamelCase, or ALLCAPS
         multi-letter identifiers.
      5. Keep hyphenated multi-tokens.
      6. Keep specific multi-word bigrams: only when *both* halves
         are themselves technical (prefix:term, CamelCase, ALLCAPS,
         or hyphenated). Plain-English bigrams are dropped because they
         over-tag at substring-match time.
      7. Cap at 10 candidates per CO; rank technical-singles first,
         then technical-bigrams, then hyphenated singles.

    The conservative bigram rule keeps auto-extraction useful without
    flooding coverage. Statements with no technical tokens produce an
    empty or short vocabulary instead of relying on hidden defaults.

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

    # Technical single-token identifiers have the highest priority.
    tech_singles: List[str] = []
    tech_seen: set = set()
    for tok in tokens:
        if _is_stopword(tok):
            continue
        if _is_technical_term(tok):
            if tok not in tech_seen:
                tech_seen.add(tok)
                tech_singles.append(tok)

    # Hyphenated multi-tokens retain compound concepts that would otherwise
    # be split into overly broad terms.
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

    # Emit a bigram only when both halves are technical identifiers.
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

    # Rank technical singles, technical bigrams, then hyphenated terms and
    # cap the deterministic result at ten candidates per objective.
    ranked: List[str] = []
    seen_final: set = set()
    for term in tech_singles + tech_bigrams + hyphenated:
        key = term.lower()
        if key in seen_final:
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

    Returns a ``{objective_id: [terms]}`` map. Empty input returns an empty
    dict. Objective IDs are normalized to lowercase for stable matching.
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

    # Alternate loader shape: chapter_objectives[] with optional nesting.
    for ch in objectives.get("chapter_objectives") or []:
        if isinstance(ch, Mapping) and "objectives" in ch:
            inner: Iterable[Any] = ch.get("objectives") or []
        else:
            inner = [ch]
        for obj in inner:
            if isinstance(obj, Mapping):
                _consider(obj)

    # Terminal outcomes participate in the same runtime-derived matching.
    for to in objectives.get("terminal_outcomes") or []:
        if isinstance(to, Mapping):
            _consider(to)

    return out


def merged_vocabularies(
    objectives: Optional[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Return runtime-derived vocabularies for compatibility with callers."""
    return build_auto_vocabularies(objectives)


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

    No vocabulary matches when ``vocabularies`` is omitted. Callers derive
    the map from their active objective payload with ``merged_vocabularies``.
    """
    if not text:
        return []
    table = vocabularies if vocabularies is not None else {}
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

    ``vocabularies`` accepts a runtime-derived map. Omitting it performs only
    deduplication and parent rollup; no hidden vocabulary is applied.
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

    # Add objective IDs whose supplied vocabulary appears in the chunk text.
    text = chunk.get("text") or ""
    if isinstance(text, str):
        for co_id in _vocabulary_matches(text, vocabularies=vocabularies):
            _add(co_id)

    # Add the terminal parent of every mapped objective reference.
    if parent_map:
        # Iterate over a snapshot so newly added parents are not revisited.
        for ref in list(out):
            parent = parent_map.get(ref.lower())
            if parent:
                _add(parent)

    chunk["learning_outcome_refs"] = out
    return chunk


__all__ = [
    "auto_extract_vocabulary",
    "build_auto_vocabularies",
    "build_parent_map",
    "merged_vocabularies",
    "retag_chunk_outcomes",
]
