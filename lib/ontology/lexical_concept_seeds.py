"""Lexical / statistical domain-concept seed derivation.

Productionizes the proven prototype at
``runtime/kg_prototype/clean/build_clean_seeds.py`` into a reusable, domain-
agnostic helper. The motivating failure: on a *general* (non-RDF/SHACL)
textbook corpus the Stage-3 LLM vocabulary pass (gated by
``TEXTBOOK_SYNTHESIS_PROVIDER``) often does not run, so the per-course
``domain_concept_seeds`` list is empty. Chunk concept-tagging
(``lib.ontology.concept_tagging.extract_concept_tags``) then matches only the
pedagogy-term ``CONCEPT_PATTERNS``, every chunk ends up with ~2 tags, and the
downstream co-occurrence + semantic graph collapses to a degenerate 2-node
graph.

This module recovers a usable domain-concept vocabulary from the chunk text
ALONE, using LEXICAL / statistical signals only — frequency, acronym
detection, and multi-word noun-phrase candidates, hardened by a curated
function-word reject set, a generic-term stoplist, a pipeline/licence
boilerplate denylist, and plural/variant alias-folding.

Posture: **no embeddings.** The SBIR novelty claim requires that semantic /
embedding retrieval not appear in the shipped pipeline; this derivation is
deterministic n-gram + frequency analysis and stays well inside that line.

The single public entry point is :func:`derive_lexical_concept_seeds`. It
returns a ranked list of canonical slug strings suitable for feeding to
``Trainforge.process_course.compile_domain_concept_seeds`` (each slug is also
its own alias once de-hyphenated). The caller then re-tags chunks in memory
via ``extract_concept_tags`` and rebuilds the graph with the real builders.

Determinism: for a fixed chunk list the output is byte-stable — candidates are
ranked by ``(-doc_freq, slug)`` so frequency ties break alphabetically.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

# ---------------------------------------------------------------------------
# Function words. If a candidate slug contains ANY of these as a token it is a
# syntactic fragment ("accessibility-is", "easier-to", "must-be", "such-as"),
# not a noun-concept. Applied to EVERY token of a candidate. This is the
# single biggest source of noise in a naive n-gram pass.
# ---------------------------------------------------------------------------
FUNCTION_WORDS: frozenset = frozenset({
    # copulas / auxiliaries / modals
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "having",
    "do", "does", "did", "done",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must",
    # prepositions / conjunctions / particles
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "into",
    "onto", "upon", "out", "off", "over", "under", "about", "as", "than",
    "then", "and", "or", "nor", "but", "so", "if", "because", "while",
    "when", "where", "which", "who", "whom", "whose", "that", "this",
    "these", "those", "such", "via", "per", "vs",
    # articles / quantifiers / pronouns
    "a", "an", "the", "any", "all", "some", "no", "not", "more", "most",
    "less", "least", "many", "much", "few", "one", "two", "three", "only",
    "each", "every", "both", "either", "neither", "other", "another",
    "same", "own", "it", "its", "he", "she", "they", "them", "their",
    "we", "our", "you", "your", "i", "me", "my",
    # vague verbs / adverbs that surface in instructional bigrams
    "read", "hear", "default", "response", "practice", "shown", "keep",
    "add", "set", "make", "use", "used", "using", "get", "got",
    "very", "also", "just", "still", "even",
})

# ---------------------------------------------------------------------------
# Generic / pedagogical / logistics single words that should never anchor a
# domain-concept node. Domain-agnostic: these are the cross-discipline
# scaffolding nouns (chapter, section, figure, example) plus low-signal
# adjectives/quantity nouns that surface in every textbook regardless of
# subject. Subject-specific vocabulary is intentionally NOT listed — that is
# exactly what we are trying to harvest.
# ---------------------------------------------------------------------------
GENERIC: frozenset = frozenset({
    "the", "this", "that", "these", "those", "chapter", "section", "page",
    "example", "examples", "figure", "table", "introduction", "summary",
    "overview", "concept", "concepts", "content", "contents",
    "learn", "learning", "learner", "learners", "understand",
    "following", "important", "common", "various", "different", "specific",
    "general", "basic", "advanced", "first", "second", "third", "new",
    "value", "values", "way", "ways", "part",
    "parts", "list", "number", "case", "cases", "use", "using",
    "used", "make", "made", "provide", "provides", "ensure", "ensures",
    "guideline", "guidelines", "level", "levels",
    "question", "questions", "answer", "answers", "hint", "hints",
    "empty", "single", "short", "sentence", "characters", "character",
    "complicated", "normal", "large", "small", "good", "best", "better",
    "practices", "approach", "approaches", "thing", "things", "item",
    "items", "step", "steps", "note", "notes", "tip", "tips",
    "following", "above", "below", "here", "there",
})

# ---------------------------------------------------------------------------
# Pipeline / licence / attribution boilerplate. These survive the token
# filters (they read as plausible noun-phrases) but are pipeline artifacts,
# licences, attributions, or boilerplate phrases — never domain concepts.
# Stored hyphen-slugged. Matched after slugify + alias-fold.
# ---------------------------------------------------------------------------
DENY: frozenset = frozenset({
    "converted-by-dart",
    "characters-or-less",
    "open-government-licence",
    "open-government-license",
    "creative-commons",
    "all-rights-reserved",
    "table-of-contents",
    "validation-error",
    "back-link",
    "single-short-sentence",
    "document-accessibility-remediation",
})

# Plural / variant surface forms folded onto a single canonical slug so the
# graph carries one node per concept rather than "screen-reader" +
# "screen-readers" duplicates. Generic suffix-stripping handles regular
# plurals; this map covers irregulars + known multi-word variants.
ALIAS_FOLD: Mapping[str, str] = {
    "screen-readers": "screen-reader",
    "success-criterion": "success-criteria",
    # Domain synonyms with zero shared chunks (Pass A folds unconditionally).
    # key = variant surface form, value = canonical slug.
    "recursive-chunking": "chunking",
    "nvidia-nim-microservices": "nvidia-nim",
    "nvidia-ngc-service": "nvidia-ngc-catalog",
    "ragas-short-for-rag": "ragas",
    "openais-chatgpt": "chatgpt",
}

# Lowercase word tokenizer. Words are 2+ alphabetic chars (hyphen-internal
# allowed so "co-occurrence" stays one token). Multi-word noun-phrase
# candidates (bigrams + trigrams) are generated as ALL overlapping n-gram
# windows over this token stream — a single non-overlapping regex would miss
# "cell membrane" inside "the cell membrane is" once the leading article
# starts a match. Single words are harvested via the acronym pass only (a bare
# common single word floods the candidate set with low-signal nouns).
_WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

# Acronym matcher: 2-6 uppercase letters (optionally with digits), e.g.
# WCAG, ARIA, HTML, DNA, RDF, SHACL, ATP. Acronyms are high-signal domain
# anchors that the lowercase phrase pass would otherwise miss.
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6}[0-9]{0,2})\b")

# Acronyms that are generic / boilerplate, not domain concepts.
_ACRONYM_DENY: frozenset = frozenset({
    "AND", "THE", "FOR", "ARE", "NOT", "ALL", "ANY", "USE", "SEE",
    "PDF", "HTML", "CSS", "URL", "URI", "API", "FAQ", "TODO", "NOTE",
    "OK", "ID", "IDS",
})

# Default ceiling on emitted seeds. The prototype produced 60 seeds → 44
# DomainConcept nodes on a 65-chunk corpus; the cap is a guardrail against a
# pathological long-tail on very large corpora.
_DEFAULT_MAX_SEEDS = 200


def _slugify(text: str) -> str:
    """Lowercase, strip non-alnum/hyphen, collapse whitespace to hyphens."""
    text = re.sub(r"[^a-zA-Z0-9\- ]", " ", text.lower())
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _singular(slug: str) -> str:
    """Best-effort regular-plural singularization of the LAST token.

    Generic suffix rules only (no irregular dictionary) — irregulars are
    handled by :data:`ALIAS_FOLD`. Folds ``-ies`` → ``-y``, ``-sses`` →
    ``-ss``, ``-ses`` → ``-s``, and a trailing ``-s`` (not ``-ss`` / ``-us``
    / ``-is``).
    """
    if "-" in slug:
        head, _, tail = slug.rpartition("-")
        return f"{head}-{_singular_word(tail)}"
    return _singular_word(slug)


def _singular_word(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _is_noun_concept(slug: str) -> bool:
    """Heuristic: is ``slug`` a genuine noun-concept candidate?

    Rejects when ANY token is a function word or a generic non-concept, when
    the candidate is on the denylist, when it is too short, or when every
    token is a tiny (≤3-char) word.
    """
    if not slug or slug in DENY:
        return False
    if len(slug) < 5:
        return False
    tokens = slug.split("-")
    if any(t in FUNCTION_WORDS for t in tokens):
        return False
    if any(t in GENERIC for t in tokens):
        return False
    if all(len(t) <= 3 for t in tokens):
        return False
    return True


# ---------------------------------------------------------------------------
# Fragment-phrase rejection for harvested bold/strong spans.
#
# Motivation: some Courseforge/IMSCC source notebooks (e.g. NVIDIA's) bold
# whole SENTENCES rather than key terms, so a naive bold-span harvest produces
# sentence-fragment "concepts" ("Your LLM Can Have", "Congratulations We Now
# Have", "Passing Dictionaries Helps Us"). These become spurious KG nodes.
#
# A genuine key concept is a SHORT NOUN PHRASE ("Knowledge Base", "Vector
# Store", "Foundation Models"). :func:`is_fragment_phrase` is a conservative,
# domain-agnostic sentence-fragment detector that rejects the sentence junk
# while keeping real multi-word concepts. It is intentionally pure (no env
# reads, no I/O) so the caller can gate it behind a behavior flag.
# ---------------------------------------------------------------------------

# Finite-verb / sentence-opener tokens that mark a clause, not a noun phrase.
# A bold span starting with one of these is prose, not a concept. Kept
# separate from FUNCTION_WORDS (those are syntactic glue that can appear
# anywhere) — these specifically signal "this is a sentence".
_FRAGMENT_OPENERS: frozenset = frozenset({
    "congratulations", "luckily", "specifically", "passing", "try",
    "now", "let", "lets", "please", "note", "remember", "notice",
    "consider", "imagine", "suppose", "welcome", "thanks", "thank",
})

# Sentence-ending punctuation that betrays a clause rather than a term.
_SENTENCE_PUNCT = (".", "!", "?")

# Token splitter for the fragment check — keep internal hyphens/apostrophes so
# "FAISS Vector Store" and "don't" tokenize sanely, strip surrounding
# punctuation per token.
_FRAGMENT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# Max token count for a real key concept. Real multi-word concepts are short
# noun phrases; anything longer reads as a clause.
_MAX_CONCEPT_TOKENS = 5

# ---------------------------------------------------------------------------
# Leading-imperative-verb rejection. A multi-word span whose FIRST token is a
# base-form action verb ("apply to self host", "download the dataset",
# "run the cell below", "aggregating data") is an INSTRUCTION — a verb-first
# action phrase — not a noun-concept. The KG audit surfaced these leaking
# through the existing rules ("apply-to-self-host", "receive-user-input",
# "delegate-tasks", "embed-the-data", …) because none of them start with a
# FUNCTION_WORD or a _FRAGMENT_OPENER, carry < 2 function words, and stay under
# the token ceiling.
#
# Curated, domain-agnostic base-form verbs. Conservative on TWO axes: (1) only
# fires on MULTI-word phrases (a bare "apply" / "run" is handled elsewhere as
# scaffolding — a single action word is too ambiguous to reject here); (2) the
# first token must be the base form OR a de-gerunded form that maps back into
# this set. Noun-concepts never lead with one of these (vector-store,
# knowledge-base, prompt-engineering, retrieval-augmented-generation), so the
# rule can't false-flag them.
_IMPERATIVE_VERB_PREFIXES: frozenset = frozenset({
    "apply", "receive", "delegate", "download", "embed", "define", "create",
    "maintain", "synchronize", "aggregate", "run", "build", "install",
    "configure", "set", "get", "use", "add", "remove", "update", "deploy",
    "import", "load", "save", "generate", "compute", "calculate", "select",
    "pull", "push", "call", "execute", "observe", "act", "reason",
})

# Gerund (-ing) surface forms whose base maps into _IMPERATIVE_VERB_PREFIXES.
# Explicit (not a blanket "-ing"-strip) so legitimate gerund-NOUN modifiers
# that share a stem stay concepts: "running-state-chain" keeps "running" (the
# base "run" is imperative, but "running state" is a noun modifier, not an
# instruction), so "running" is deliberately ABSENT here. Only verbs that read
# as an action when they open a phrase ("aggregating data", "embedding the
# corpus") are listed. Maps surface → base purely for membership; the base is
# never re-emitted.
_IMPERATIVE_VERB_GERUNDS: frozenset = frozenset({
    "applying", "receiving", "delegating", "downloading", "embedding",
    "defining", "creating", "maintaining", "synchronizing", "aggregating",
    "installing", "configuring", "deploying", "importing", "loading",
    "saving", "generating", "computing", "calculating", "selecting",
    "pulling", "pushing", "calling", "executing", "observing", "reasoning",
})


def _leads_with_imperative_verb(tokens: Sequence[str]) -> bool:
    """Does ``tokens`` open with a base-form / gerund imperative action verb?

    Only meaningful on multi-word phrases — the caller guards on length. A
    single bare action word is intentionally NOT rejected here (it's handled
    upstream as scaffolding). De-gerunding is membership-only via the explicit
    :data:`_IMPERATIVE_VERB_GERUNDS` set so noun-modifier gerunds
    ("running-state-chain") are never swept in.
    """
    if len(tokens) < 2:
        return False
    first = tokens[0]
    return (
        first in _IMPERATIVE_VERB_PREFIXES
        or first in _IMPERATIVE_VERB_GERUNDS
    )


def is_fragment_phrase(text: str) -> bool:
    """Heuristic: is ``text`` a sentence fragment rather than a key concept?

    Conservative + domain-agnostic. Returns ``True`` (reject) when the span
    reads as prose/a clause, ``False`` (keep) when it reads as a noun-phrase
    concept. Real multi-word concepts ("Knowledge Base", "Vector Store",
    "FAISS Vector Store", "Foundation Models", "RunnableAssign") return
    ``False``; sentence fragments ("Your LLM Can Have", "Congratulations We
    Now Have", "Passing Dictionaries Helps Us") return ``True``.

    Rejection criteria (ANY triggers reject):

    * ends with sentence punctuation (``.`` / ``!`` / ``?``) or contains
      ``?`` anywhere;
    * more than :data:`_MAX_CONCEPT_TOKENS` tokens (clauses, not terms);
    * starts with a function word or a known sentence-opener / finite-verb
      token (``congratulations``, ``luckily``, ``specifically``, ``passing``,
      ``try``, …);
    * is multi-word and starts with a base-form / gerund imperative action
      verb (``apply``, ``download``, ``run``, ``aggregating``, …) — an
      instruction, not a noun-concept ("apply to self host",
      "run the cell below", "aggregating data"). Noun-phrase concepts never
      lead with one (``vector store``, ``knowledge base``,
      ``running state chain``) so they survive;
    * contains two or more :data:`FUNCTION_WORDS` (a noun phrase carries at
      most one connective; two+ marks a clause);
    * no alphabetic content noun survives — every token is a function /
      opener word.
    """
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True

    # Sentence punctuation → prose.
    if "?" in stripped:
        return True
    if stripped.endswith(_SENTENCE_PUNCT):
        return True

    tokens = _FRAGMENT_TOKEN_RE.findall(stripped.lower())
    if not tokens:
        return True

    # Too long to be a key concept (noun phrases are short).
    if len(tokens) > _MAX_CONCEPT_TOKENS:
        return True

    # Sentence-opener / finite-verb or function word in first position.
    first = tokens[0]
    if first in _FRAGMENT_OPENERS or first in FUNCTION_WORDS:
        return True

    # Leading imperative action verb on a multi-word phrase → instruction,
    # not a noun-concept ("apply to self host", "run the cell below").
    if _leads_with_imperative_verb(tokens):
        return True

    # Two or more function words → clause glue, not a noun phrase.
    function_word_count = sum(1 for t in tokens if t in FUNCTION_WORDS)
    if function_word_count >= 2:
        return True

    # No content noun survives — every token is a function / opener word.
    content_tokens = [
        t for t in tokens
        if t not in FUNCTION_WORDS and t not in _FRAGMENT_OPENERS
    ]
    if not content_tokens:
        return True

    return False


def derive_lexical_concept_seeds(
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_seeds: int = _DEFAULT_MAX_SEEDS,
    min_doc_freq: int = 3,
) -> List[str]:
    """Derive domain-concept seed slugs from chunk text — lexical only.

    Three candidate sources, all purely lexical / statistical (no embeddings):

    1. **Acronyms** — runs of 2-6 uppercase letters (``WCAG``, ``RDF``,
       ``DNA``). High-signal domain anchors; kept at ``doc_freq >= 1`` because
       a domain acronym appearing even once is almost always a real concept.
       Generic acronyms (``PDF``, ``HTML``, ``API``, …) are rejected.
    2. **Multi-word noun phrases** — 2-3 consecutive lowercase words
       (``color contrast``, ``cell membrane``). Subject to the full
       function-word / generic / denylist filter and a ``min_doc_freq`` floor
       so only recurrent phrases survive.

    Candidates are slugified, plural/variant-folded (regular-plural
    singularization + :data:`ALIAS_FOLD`), filtered, and ranked by document
    frequency. Output is deterministic: ties break alphabetically.

    Args:
        chunks: Iterable of chunk dicts; each chunk's ``"text"`` field is
            read (missing / non-string text is skipped).
        max_seeds: Hard cap on the returned list length (guards the long
            tail on large corpora). Default 200.
        min_doc_freq: Minimum number of distinct chunks a multi-word phrase
            must appear in to be kept. Default 3. Acronyms bypass this floor
            (they clear at ``doc_freq >= 1``).

    Returns:
        A ranked, deduped list of canonical slug strings (hyphenated). Empty
        when ``chunks`` is empty or yields no surviving candidate.
    """
    if not chunks:
        return []

    doc_freq: Counter = Counter()
    is_acronym: Dict[str, bool] = {}

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        text = chunk.get("text")
        if not isinstance(text, str) or not text:
            continue
        seen_in_chunk: set = set()

        # --- (1) acronyms (case-sensitive against the raw text) -----------
        for raw in _ACRONYM_RE.findall(text):
            if raw in _ACRONYM_DENY:
                continue
            slug = ALIAS_FOLD.get(_slugify(raw), _slugify(raw))
            if not slug or slug in DENY:
                continue
            sing = _singular(slug)
            if sing != slug:
                slug = sing
            seen_in_chunk.add(slug)
            # Acronyms are valid even when short; flag so the floor is
            # bypassed during ranking.
            is_acronym[slug] = True

        # --- (2) multi-word noun phrases (lowercased n-gram windows) ------
        # Sentence-bounded so candidates never straddle a period (a 3-gram
        # window must be three words of the SAME sentence to be a phrase).
        low = text.lower()
        for sentence in re.split(r"[.!?;:\n]+", low):
            words = _WORD_RE.findall(sentence)
            for n in (2, 3):
                for start in range(len(words) - n + 1):
                    phrase = " ".join(words[start:start + n])
                    slug = ALIAS_FOLD.get(_slugify(phrase), _slugify(phrase))
                    if not _is_noun_concept(slug):
                        continue
                    sing = _singular(slug)
                    if sing != slug and _is_noun_concept(sing):
                        slug = sing
                    seen_in_chunk.add(slug)
                    is_acronym.setdefault(slug, False)

        for slug in seen_in_chunk:
            doc_freq[slug] += 1

    # --- rank + threshold -------------------------------------------------
    kept: List[str] = []
    for slug, df in doc_freq.items():
        if slug in DENY:
            continue
        if is_acronym.get(slug):
            keep = df >= 1
        else:
            keep = df >= min_doc_freq and _is_noun_concept(slug)
        if keep:
            kept.append(slug)

    # Deterministic ordering: highest doc-freq first, alpha tiebreak.
    kept.sort(key=lambda s: (-doc_freq[s], s))
    return kept[:max_seeds]


__all__ = [
    "derive_lexical_concept_seeds",
    "is_fragment_phrase",
    "FUNCTION_WORDS",
    "GENERIC",
    "DENY",
    "ALIAS_FOLD",
]
