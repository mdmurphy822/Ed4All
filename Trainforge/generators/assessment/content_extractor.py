"""
Content Extractor for Assessment Generation

Extracts question-worthy elements (key terms, factual statements,
relationships, procedures, examples) from retrieved RAG chunks.
Sits between retrieval and question generation to provide structured
content that each question type can consume.
"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from lib.ontology.slugs import deslugify_concept

# TOC / page-number / chapter-heading blocklist. These patterns identify
# key-term candidates that are actually table-of-contents fragments rather
# than real course terminology. Applied BEFORE a KeyTerm is appended in
# :meth:`ContentExtractor.extract_key_terms`.
_TOC_THREE_INTS = re.compile(r"\b\d+\b.*\b\d+\b.*\b\d+\b", re.DOTALL)
# Dotted numeric followed (anywhere later) by a bare integer, e.g.
# "1.1 Structural changes ... 14" — characteristic of TOC lines with
# page numbers.
_TOC_DOTTED_PLUS_INT = re.compile(r"\b\d+\.\d+\b.*\b\d+\b", re.DOTALL)
# Leading bare integer (e.g. "42 The ..."), ".", ")", or ":" afterwards.
_TOC_LEADING_INT = re.compile(r"^\s*\d+[\.\)\:]\s*")
# Leading "Chapter 3", "Section 4", etc. — TOC title prefixes with a
# number directly following.
_TOC_TITLE_PREFIX = re.compile(
    r"^\s*(Contents|Chapter|Section|Part|Appendix)\s+\d+\b",
    re.IGNORECASE,
)
# Standalone bare-integer term like "42".
_BARE_INTEGER_ONLY = re.compile(r"^\s*\d+\s*$")


# Generic pedagogical-apparatus markers — domain-agnostic exercise/solution
# scaffolding (a "Solution:" / "Check:" label, a "Try It" prompt, an "In the
# following exercises" banner). These are the generic pedagogy-label vocabulary
# class used generically, NOT publisher-specific vocabulary, so they stay in
# code. Publisher-specific exercise banners are data loaded from the shared
# apparatus lexicon by ``_apparatus_banner_re`` rather than hardcoded here.
#
# ``Show answer`` is one member of a synonym family the same authoring surface
# emits interchangeably ("Show answer" / "Show solution" / "Show work" /
# "Show steps"); a colon never accompanies any of them, so a colon-anchored
# rule cannot see them. Matching the family keeps these apparatus labels out of
# assessment stems. Capitalised ``Show`` is required (the pattern is
# case-SENSITIVE), so mid-sentence prose — "the graph will show solution
# sets" — is untouched.
_APPARATUS_RE = re.compile(
    r"\b(?:Solution\s*:|Check\s*:|"
    r"Show\s+(?:answers?|solutions?|work|steps?)\b|Try It\b|"
    r"In the following exercises|Answers? will vary)"
)


@lru_cache(maxsize=1)
def _apparatus_banner_re() -> "re.Pattern[str]":
    """Compile the publisher exercise-banner markers from the shared lexicon.

    Publisher-specific apparatus banners (e.g. "Practice Makes Perfect",
    "Section Exercises") are vocabulary, not code: they live in
    ``schemas/taxonomies/exercise_apparatus_lexicon.json`` and are loaded via
    the canonical taxonomy loader. Returns the union of every profile's
    ``apparatus_banners`` group as one case-insensitive regex; word-internal
    whitespace is treated as ``\\s+`` so an OCR-broken banner still matches. A
    missing / malformed lexicon degrades to a never-matches sentinel (the
    generic markers above still fire).
    """
    from lib.ontology.taxonomy import load_taxonomy

    try:
        lex = load_taxonomy("exercise_apparatus_lexicon")
    except (FileNotFoundError, ValueError):
        return re.compile(r"(?!x)x")
    phrases: List[str] = []
    seen: set = set()
    for profile in (lex.get("profiles") or {}).values():
        for phrase in profile.get("apparatus_banners") or []:
            norm = str(phrase).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                phrases.append(str(phrase).strip())
    if not phrases:
        return re.compile(r"(?!x)x")
    alts = "|".join(
        r"\s+".join(re.escape(tok) for tok in p.split() if tok) for p in phrases
    )
    return re.compile(alts, re.IGNORECASE)


#: Env flag for the WIDENED generic-apparatus marker set below. Default off →
#: :func:`_is_apparatus_text` keeps the baseline marker set and therefore
#: preserves flag-off extraction output. Pipeline orchestration enables the
#: stricter mode explicitly when required.
_APPARATUS_STRICT_ENV = "ED4ALL_ASSESSMENT_APPARATUS_STRICT"


def resolve_apparatus_strict() -> bool:
    """Resolve the widened-apparatus-marker flag (parse-with-fallback).

    Truthy ``1``/``true``/``yes``/``on`` enables the widened marker set;
    anything else (unset, empty, garbage, falsey) keeps the legacy set.
    """
    return (os.environ.get(_APPARATUS_STRICT_ENV, "") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# Widened GENERIC apparatus markers (gated by ``_APPARATUS_STRICT_ENV``). Same
# vocabulary CLASS as ``_APPARATUS_RE`` — pedagogical-label / figure-apparatus
# scaffolding, never subject words — so these stay in code alongside it;
# publisher-specific banners remain data in the lexicon. The patterns recognize:
#
#   * ``Solution`` / ``Check`` WITHOUT the colon the legacy pattern requires —
#     OCR routinely drops it ("Solution A gray checkmark inside a circle").
#   * An all-caps ``HOW TO <PROCEDURE>`` banner heading.
#   * A leading ``Figure|Table|Example N.N`` caption, anchored at string start
#     so a definition that merely CITES a figure mid-sentence is not rejected.
#   * A pure glyph/icon alt-text description ("A gray checkmark inside a
#     circle, indicating correct or complete.") — image scaffolding the OCR
#     lane emits as prose. Anchored at start and required to close with an
#     ``indicating``/``showing``-class participle so real prose is untouched.
_APPARATUS_WIDE_RE = re.compile(
    r"(?:^|(?<=[\s>]))(?:Solution|Check)\s+(?=[A-Z])"
    r"|\bHOW\s+TO\s+[A-Z][A-Z\s]{3,}"
    r"|^\s*(?:Figure|Table|Example)\s+\d+(?:\.\d+)?\b",
)

#: Glyph / icon alt-text: an image description the OCR lane rendered as prose.
#: Anchored at start; requires BOTH a glyph noun and a trailing descriptive
#: participle so ordinary sentences mentioning a shape are not swept up.
_GLYPH_ALT_TEXT_RE = re.compile(
    r"^\s*(?:An?\s+)?[\w\s\-]{0,40}?"
    r"\b(?:checkmark|check\s*mark|arrow|icon|symbol|bullet|glyph|"
    r"circle|square|triangle|rectangle|box|star)\b"
    r"[\w\s\-,]{0,60}?,\s*"
    # "showing" is deliberately excluded because it is common in legitimate
    # math prose ("a box plot ... showing the median"). Glyph alt-text uses the
    # state-describing participles below.
    r"(?:indicating|representing|denoting|signifying)\b",
    re.IGNORECASE,
)


#: A worked-example step LABEL standing alone as the whole candidate
#: ("Step 1", "Step 2.") — pure procedural scaffolding, never an answer or a
#: plausible distractor. Anchored + fullmatch-shaped so ordinary prose that
#: merely MENTIONS a step ("Step 1 is to isolate the variable") is untouched.
_BARE_STEP_LABEL_RE = re.compile(r"^\s*Step\s+\d+\s*[.:)]?\s*$", re.IGNORECASE)

#: A worked-solution step that LEADS its candidate: the label plus a delimiter
#: plus the step body ("Step 2: Since -9 is 9 units from 0, |-9| = 9."). The
#: bare-label pattern above only fires when the label is the whole candidate;
#: this prefix form also excludes complete worked-solution step sentences from
#: true/false statements, fill-in-the-blank contexts, and key terms.
#:
#: The trailing delimiter is REQUIRED, which is what keeps ordinary prose
#: intact: "Step 1 is to isolate the variable" has no ``.``/``:``/``)`` after
#: the ordinal and is untouched. Same pedagogical-label vocabulary class as
#: ``Solution:`` / ``Check:`` — no subject words, no publisher phrases.
_STEP_LABEL_PREFIX_RE = re.compile(r"^\s*Step\s+\d+\s*[.:)]", re.IGNORECASE)

#: A generic pedagogical CALLOUT label opening the candidate ("Key Idea: …",
#: "Note: …", "Common wrong turn: …"). Same shape as the step-label prefix
#: above and the same vocabulary CLASS as ``Solution:`` / ``Check:``: these are
#: DISCOURSE labels the authoring surface stamps on a boxed aside, not content.
#: A sentence that still carries the label is a fragment of markup, so it must
#: never become a stem, a definition, a key term, or an answer key.
#:
#: Two deliberate scope limits keep the pool healthy:
#:   * ANCHORED at the start AND a colon is REQUIRED, so ordinary prose that
#:     merely uses one of these words ("Note that the sum is even", "the
#:     important idea here") never matches.
#:   * CONTENT-TYPE labels are excluded on purpose — "Definition:",
#:     "Example:", "Theorem:", "Property:", "Rule:", "Formula:" introduce
#:     genuinely testable material, and the key-term extractor exists to mine
#:     exactly those. Only discourse/aside labels are listed.
_CALLOUT_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"Key\s+(?:Idea|Point|Takeaway|Concept)"
    r"|Big\s+Idea|Main\s+Idea|Takeaway"
    r"|Note|Tip|Hint|Caution|Warning|Important|Remember|Reminder"
    r"|Predict|Reflect|Recall"
    r"|Common\s+(?:Error|Mistake|Wrong\s+Turn|Pitfall)|Pitfall|Misconception"
    r"|Answer|Solution|Check"
    r")\s*:",
    re.IGNORECASE,
)

#: A canonical learning-objective REF opening the candidate ("CO-01: …",
#: "TO-05: …"). The shape is this project's own LO-id convention
#: (``lib/ontology/learning_objectives.py``, pattern ``^[A-Z]{2,}-\d{2,}$``),
#: so matching it is structural, not corpus vocabulary. An objective statement
#: is a course-design artifact — a promise about what the learner will be able
#: to do — never a factual claim to test true/false or blank out.
_LO_REF_PREFIX_RE = re.compile(r"^\s*[A-Z]{2,}-\d{2,}\s*:")

#: A BARE anaphoric subject — a lone pronoun or demonstrative standing where
#: the subject noun phrase belongs ("This is determined by its position …",
#: "It is the same distance from 0 …"). Mined out of running prose, such a
#: sentence loses its antecedent and becomes unanswerable on its own: a
#: standalone assessment item cannot reference something the learner never
#: saw. Deliberately BARE-only — "This process is called clearing" keeps its
#: noun and survives — so the rule removes dangling fragments without
#: thinning legitimate prose.
_ANAPHORIC_SUBJECTS = frozenset({
    "this", "that", "these", "those", "it", "they", "them",
    "he", "she", "there", "here", "one", "both", "each", "such",
})


def _is_apparatus_text(text: str) -> bool:
    """True when the candidate text is exercise/solution APPARATUS.

    The marker set is the textbook pedagogical-label vocabulary class, never
    subject words. Such text must never become an assessment
    answer/definition/statement: on an apparatus-dense worked-example corpus
    it otherwise leaks (e.g. "Solution: r Check: ...") through every harvest
    path. Generic markers are matched in-code; publisher-specific exercise
    banners are matched from the data-driven lexicon.

    Under ``ED4ALL_ASSESSMENT_APPARATUS_STRICT`` the widened generic marker
    set additionally fires (colon-less Solution/Check, all-caps HOW-TO
    banners, leading figure/table captions, glyph alt-text) — the OCR'd-scan
    leak class the legacy patterns miss. Default off → byte-identical.
    """
    t = text or ""
    if _APPARATUS_RE.search(t) or _apparatus_banner_re().search(t):
        return True
    if not resolve_apparatus_strict():
        return False
    # Match against the TAG-STRIPPED text as well. Several widened patterns are
    # ``^``-anchored (leading figure/table caption, glyph alt-text, a bare
    # "Step N" label), while the assessment emitter wraps choice and answer
    # text in ``<p>…</p>``. Testing both forms applies the same guard to raw and
    # rendered text.
    candidates = [t]
    stripped = re.sub(r"<[^>]+>", " ", t).strip()
    if stripped and stripped != t:
        candidates.append(stripped)
    return any(
        _APPARATUS_WIDE_RE.search(c)
        or _GLYPH_ALT_TEXT_RE.search(c)
        or _BARE_STEP_LABEL_RE.match(c)
        or _STEP_LABEL_PREFIX_RE.match(c)
        or _CALLOUT_LABEL_PREFIX_RE.match(c)
        or _LO_REF_PREFIX_RE.match(c)
        for c in candidates
    )


def _has_anaphoric_subject(subject: str) -> bool:
    """True when ``subject`` is a BARE pronoun / demonstrative.

    Such a subject has no antecedent once the sentence is lifted out of its
    paragraph, so the resulting item is unanswerable by construction
    ("This is determined by its position as the third digit from the right."
     — *what* is?). This is a mining defect, not a linter artifact: the
    sentence should never have entered the candidate pool.

    Bare-only by design — a demonstrative that still carries its noun
    ("This process is called clearing the equation") is self-contained and
    survives.
    """
    tokens = re.findall(r"[A-Za-z']+", subject or "")
    if len(tokens) != 1:
        return False
    return tokens[0].lower() in _ANAPHORIC_SUBJECTS


def _is_apparatus_key_term(term: str, definition: str, context: str) -> bool:
    """True when a key-term candidate is apparatus in ANY of its three parts.

    ``extract_key_terms`` screens both the term and its definition because a
    worked-solution label can otherwise become the term itself. Key terms are
    reused as fill-in-the-blank answers and multiple-choice stem subjects, so
    apparatus on any candidate surface would contaminate multiple item types.

    Screening all three parts is the same guard applied to the whole
    candidate rather than a third of it. Domain-agnostic: the marker set is
    unchanged pedagogical-label vocabulary.
    """
    return any(
        _is_apparatus_text(part)
        for part in (term, definition, context)
        if part
    )


# --------------------------------------------------------------------------- #
# Apparatus-guard numeric-recovery (flag-gated, default OFF).
#
# The apparatus guard above strips Solution/Check/Step-N worked-example content
# from EVERY harvest path (key terms, factual statements, misconception
# corrections) because that pedagogical-label vocabulary poisons those answer
# surfaces. But a worked example's Solution/Check region is also the richest
# source of a clean, single-variable, sympy-verifiable equation — exactly what
# the numeric-FIB item builder needs. On a scanned/OCR'd corpus those equations
# arrive as PLAIN TEXT ("3x + 5 = 20") with no LaTeX / <code> markup, so the
# marked-math harvester (``worked_example_math._iter_fragments``) finds nothing
# and zero numeric items ship.
#
# ``ED4ALL_ASSESSMENT_NUMERIC_RECOVERY`` (default off) re-admits those apparatus
# regions FOR THE NUMERIC-FIB EXTRACTION PATH ONLY. Recovered candidates are
# still fully sympy-verified downstream (solve → substitute → residual == 0), so
# this only widens candidate SUPPLY; unverifiable candidates are dropped. The
# apparatus guard stays intact for every other harvest path, and the recovery
# helper returns ``[]`` when the flag is off → byte-identical.
# --------------------------------------------------------------------------- #
_NUMERIC_RECOVERY_ENV = "ED4ALL_ASSESSMENT_NUMERIC_RECOVERY"

#: Apparatus markers whose FOLLOWING window carries the worked computation.
_RECOVERY_MARKER_RE = re.compile(
    r"(?:Solution\s*:|Check\s*:|Step\s+\d+\s*[:.])",
    re.IGNORECASE,
)

#: How far past an apparatus marker to scan for a plain-text equation.
_RECOVERY_WINDOW = 240

#: A single, non-relational ``=`` (not ``==`` / ``<=`` / ``>=`` / ``!=``).
_BARE_EQUALS_RE = re.compile(r"(?<![<>=!])=(?![=])")
#: The character class a compact algebraic side is built from (digits, single-
#: letter variables, operators, parens, whitespace). A multi-letter alphabetic
#: run inside a side is NATURAL LANGUAGE ("dollars" / "so" / "find"), not math,
#: so a side is TRIMMED at the word nearest the ``=``.
_MATH_SIDE_CHAR_RE = re.compile(r"[0-9A-Za-z+\-*/^().\s]")
_WORD_RUN_RE = re.compile(r"[A-Za-z]{2,}")


def _trim_equation_side(raw: str, *, keep_suffix: bool) -> str:
    """Trim a raw equation side to its compact math core.

    ``keep_suffix=True`` (LEFT side): keep the substring AFTER the last
    multi-letter word — the math nearest the ``=``. ``keep_suffix=False``
    (RIGHT side): keep the substring BEFORE the first multi-letter word. A side
    with no word run passes through stripped.
    """
    words = list(_WORD_RUN_RE.finditer(raw))
    if not words:
        return raw.strip()
    if keep_suffix:
        return raw[words[-1].end():].strip()
    return raw[: words[0].start()].strip()


def extract_plaintext_equations(text: str) -> List[str]:
    """Harvest compact plain-text ``LHS = RHS`` algebraic equations from prose.

    A worked-example sentence embeds its equation as prose ("...for a total of
    3x = 12 dollars; find x."), so a naive grid regex sweeps the surrounding
    natural-language words into the equation sides and defeats the sympy parse.
    This helper anchors on each bare ``=``, expands each side over the math
    character class, then trims the natural-language words nearest the ``=`` so
    only the compact ``3x = 12`` core survives. Returns candidates in
    left-to-right order; each is still fully sympy-VERIFIED downstream, so a
    false candidate is dropped, never shipped.
    """
    text = text or ""
    n = len(text)
    out: List[str] = []
    for m in _BARE_EQUALS_RE.finditer(text):
        # Expand left over math chars up to (not across) the previous '='.
        li = m.start()
        while li > 0 and _MATH_SIDE_CHAR_RE.fullmatch(text[li - 1]):
            li -= 1
        left = _trim_equation_side(text[li:m.start()], keep_suffix=True)
        # Expand right over math chars up to (not across) the next '='.
        ri = m.end()
        while ri < n and _MATH_SIDE_CHAR_RE.fullmatch(text[ri]):
            ri += 1
        right = _trim_equation_side(text[m.end():ri], keep_suffix=False)
        if (
            left and right
            and re.search(r"[0-9A-Za-z]", left)
            and re.search(r"[0-9A-Za-z]", right)
        ):
            out.append(f"{left} = {right}")
    return out


def _numeric_recovery_enabled() -> bool:
    """Parse-with-fallback resolver for ``ED4ALL_ASSESSMENT_NUMERIC_RECOVERY``.

    Truthy tokens (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive) → on;
    unset / empty / garbage → off (byte-identical default).
    """
    return str(os.environ.get(_NUMERIC_RECOVERY_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_toc_fragment(term_text: str) -> bool:
    """Return True if ``term_text`` looks like a TOC/page-number fragment.

    Applied to the candidate term text (group 1 of a regex match) BEFORE that
    text becomes a ``KeyTerm.term``. Real terminology never matches these
    patterns.
    """
    if not term_text:
        return True
    # Length cap: genuine term strings are short. Long run-on matches
    # (200+ chars) are invariably paragraph fragments the regex swept in.
    if len(term_text) > 200:
        return True
    if _BARE_INTEGER_ONLY.match(term_text):
        return True
    if _TOC_LEADING_INT.match(term_text):
        return True
    if _TOC_TITLE_PREFIX.match(term_text):
        return True
    if _TOC_DOTTED_PLUS_INT.search(term_text):
        return True
    # Three standalone integers is a strong TOC signal (page runs).
    if _TOC_THREE_INTS.search(term_text):
        return True
    return False


@dataclass
class KeyTerm:
    """A defined term extracted from content."""
    term: str
    definition: str
    source_chunk_id: str
    context_sentence: str  # Full sentence containing term + definition


@dataclass
class FactualStatement:
    """A declarative statement suitable for T/F or fill-in-blank."""
    statement: str
    source_chunk_id: str
    key_subject: str  # The main subject/noun that could be blanked or negated


@dataclass
class ConceptRelationship:
    """A relationship between two concepts."""
    concept_a: str
    concept_b: str
    relationship: str  # The nature of the relationship
    full_statement: str  # Complete sentence describing the relationship
    source_chunk_id: str


@dataclass
class Procedure:
    """A sequence of steps or process."""
    title: str
    steps: List[str]
    source_chunk_id: str


@dataclass
class Example:
    """An example, case study, or application from content."""
    description: str
    context: str  # What concept it illustrates
    source_chunk_id: str


# --------------------------------------------------------------------------- #
# Footnote-apparatus strip (assessment-path only, removal-only).
#
# Source chunks quote textbook prose that carries footnote APPARATUS: LaTeX
# superscript-only footnote markers (``$^{2}$``) and the bare footnote URLs
# the marker points at. When a stem quotes that text verbatim (fill-in-blank
# context sentences, factual statements, relationship sentences) the apparatus
# ships inside the question. Strip it at the single point the assessment path
# ingests chunk text. REMOVAL-ONLY — never rewrites content; the chunker /
# conversion layers are deliberately untouched (separate fix).
#
# ``_FOOTNOTE_MARKER_RE`` matches ONLY superscript-only math spans (an
# optional-whitespace ``$^{N}$`` with a bare integer) — real math like
# ``$x^{2}$`` has content before the ``^`` and never matches.
# --------------------------------------------------------------------------- #
_FOOTNOTE_MARKER_RE = re.compile(r"\$\s*\^\s*\{\s*\d+\s*\}\s*\$")
#: A bare URL token (footnote target). Trailing sentence punctuation is left
#: for the whitespace normalizer / sentence splitter to handle.
_FOOTNOTE_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def _strip_footnote_apparatus(text: str) -> str:
    """Remove footnote markers (``$^{n}$``) and bare footnote URLs.

    Removal-only: nothing is inserted or paraphrased. Applied to every text
    the assessment generators quote from chunk content (via
    :func:`_strip_html`), so stems never retain footnote apparatus.
    """
    text = _FOOTNOTE_MARKER_RE.sub(" ", text)
    text = _FOOTNOTE_URL_RE.sub(" ", text)
    return text


def _strip_html(text: str) -> str:
    """Strip HTML tags, footnote apparatus, and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = _strip_footnote_apparatus(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# --------------------------------------------------------------------------- #
# Relationship-concept term-likeness guard.
#
# ``RELATIONSHIP_PATTERNS`` capture ``([^,.]+?)`` — an arbitrary clause up to
# the next comma/period — as each "concept". That capture can be a full clause
# rather than a concept term, which would make the "relationship between X and
# Y" stem incoherent. Only mint a
# ConceptRelationship when BOTH captures read as genuine short noun-phrase
# terms: bounded length/word count, no finite verbs / clause connectives, and
# passing the canonical fragment-phrase filter from ``lib.ontology``.
# --------------------------------------------------------------------------- #

#: Leading determiners tolerated on a prose-mined concept ("the mitochondria").
_LEADING_DETERMINER_RE = re.compile(
    r"^(?:the|a|an|its|their|your|our|this|these|those)\s+", re.IGNORECASE
)
#: Finite verbs / negations / clause connectives that mark a CLAUSE, not a
#: noun-phrase concept term. Any hit rejects the capture.
_CLAUSE_MARKER_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|"
    r"can|cannot|can'?t|will|won'?t|would|should|could|must|may|might|"
    r"shall|so\s+that|because|which|that|when|while|whereas|if|unless|"
    r"although|though)\b",
    re.IGNORECASE,
)
#: Character / word caps for a concept term (noun phrases are short).
_MAX_TERM_CHARS = 60
_MAX_TERM_WORDS = 5


def _is_term_like_concept(text: str) -> bool:
    """Is ``text`` a genuine short noun-phrase concept term?

    Used by :meth:`ContentExtractor.extract_relationships` to reject
    clause-grabs before a relationship stem template is minted. Delegates the
    fragment-phrase heuristics to the canonical
    ``lib.ontology.lexical_concept_seeds.is_fragment_phrase`` after stripping
    a tolerated leading determiner, and layers explicit length / word-count
    caps plus a finite-verb / clause-connective rejection on top.
    """
    from lib.ontology.lexical_concept_seeds import is_fragment_phrase

    if not text:
        return False
    stripped = text.strip().strip("\"'").rstrip(".,;:")
    if len(stripped) < 3 or len(stripped) > _MAX_TERM_CHARS:
        return False
    core = _LEADING_DETERMINER_RE.sub("", stripped).strip()
    if len(core) < 3:
        return False
    if len(core.split()) > _MAX_TERM_WORDS:
        return False
    if _CLAUSE_MARKER_RE.search(core):
        return False
    return not is_fragment_phrase(core)


# --------------------------------------------------------------------------- #
# Definitional-subject guard for ``DEFINITION_PATTERNS``.
#
# Every definition pattern captures group 1 as "everything from the sentence
# start up to the first copula" (``([A-Z][^.]*?)\s+(?:is|are)\s+…``). That is a
# TERM only when the sentence really is "<Term> is a <definition>". On ordinary
# prose the same shape fires on any sentence that happens to contain a copula,
# so the capture is whatever preamble preceded it: an interrogative ("What is
# the value of x?"), a demonstrative ("This is the third digit …"), an
# existential ("There is a faster method …"), a subordinate/participial opener
# ("Since 72 is a multiple of 8, …", "Shown below is the graph …"), or an
# imperative ("Complete the following: _______ is …").
#
# None of those is a term, yet each one is minted, deduplicated course-wide, and
# then reused as the fill-in-the-blank ANSWER and the MCQ stem subject — so one
# fabricated term poisons every item generated from that chunk. Recall costs one
# missed definition; precision costs a poisoned pair, so the guard is
# deliberately strict.
#
# The rule: a definitional subject must read as a genuine short NOMINAL noun
# phrase. Shape is delegated to :func:`_is_term_like_concept` (which layers the
# canonical ``lib.ontology.lexical_concept_seeds.is_fragment_phrase`` filter on
# top of length / word-count caps and a finite-verb / clause-connective
# rejection); this adds the two signals that filter cannot see, because both are
# grammatical roles rather than fragment shapes:
#
#   * a NON-NOMINAL head word — the subject position is filled by a pronoun,
#     demonstrative, interrogative, existential, or subordinating/discourse
#     opener instead of a noun;
#   * LABEL / BLANK punctuation — a colon, semicolon, question/exclamation mark,
#     dash run, or authored blank marks a label or an exercise prompt, never a
#     term.
#
# Domain-agnostic: the head set is closed-class English function vocabulary, not
# subject-matter words, so it carries across corpora unchanged.
# --------------------------------------------------------------------------- #

#: Closed-class words that can open a sentence but can never HEAD the noun
#: phrase a definition defines. Union of the bare-anaphora set (reused from
#: :data:`_ANAPHORIC_SUBJECTS` so the two guards can't drift) with the
#: interrogative, existential, subordinating-conjunction and discourse-connective
#: classes.
_NON_NOMINAL_SUBJECT_HEADS = _ANAPHORIC_SUBJECTS | frozenset({
    # Interrogatives / relatives.
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    # Subordinating conjunctions + adverbial / participial openers.
    "since", "because", "if", "unless", "although", "though", "while",
    "whereas", "after", "before", "once", "whenever", "wherever", "as",
    "until", "whether", "given", "suppose", "assume", "notice", "note",
    "recall", "consider", "shown", "using", "let", "so",
    # Discourse connectives.
    "but", "however", "therefore", "thus", "hence", "also", "then",
    "meanwhile", "moreover", "furthermore", "nevertheless", "instead",
    "otherwise", "besides", "again", "now", "yet", "still", "first",
    "second", "next", "finally", "similarly", "conversely",
})

#: Punctuation that marks the capture as a LABEL or an authored exercise BLANK
#: rather than a term ("Complete the following: _______", "Step 2 — solve",
#: the emphasis marker in "The denominator _b_"). The underscore alternatives
#: are deliberately boundary-scoped — a WORD-INTERNAL underscore is an
#: identifier character, so ``slope_of_a_line`` and ``zero_product_property``
#: stay eligible on a technical corpus.
_SUBJECT_LABEL_PUNCT_RE = re.compile(r"[:;?!]|--|—|–|(?:^|\s)_|_(?:\s|$)|__")

#: Determiners tolerated in front of a definitional subject. Wider than
#: :data:`_LEADING_DETERMINER_RE` (which is scoped to relationship concepts):
#: a definition sentence routinely opens "Every integer is …" / "Any polynomial
#: is …", and the quantifier is not the head.
_SUBJECT_LEADING_DETERMINER_RE = re.compile(
    r"^(?:the|a|an|its|their|your|our|this|these|those|each|every|any|all|"
    r"some|no|both|another|such)\s+",
    re.IGNORECASE,
)


#: Prepositions that attach a single qualifying phrase to a noun-phrase head
#: ("the perimeter OF a rectangle", "the number of decimal places IN the
#: product"). See :func:`_head_np_is_term_like`.
_SUBJECT_PP_SPLIT_RE = re.compile(r"\s+(?:of|in|for|between)\s+", re.IGNORECASE)
#: Word cap for a subject that carries a prepositional qualifier. Wider than
#: :data:`_MAX_TERM_WORDS` (which bounds a bare noun phrase) because the
#: qualifier is part of the term, not clause glue.
_MAX_SUBJECT_WORDS_WITH_PP = 8


def _head_np_is_term_like(core: str) -> bool:
    """True when ``core`` is a noun-phrase HEAD plus one prepositional qualifier.

    ``is_fragment_phrase`` rejects any span carrying two or more function words,
    which is right for clause glue but wrong for the qualified noun phrases a
    textbook defines constantly — "the perimeter of a rectangle", "the degree of
    a polynomial", "the least common multiple of two numbers". Each of those
    spends both of its function-word budget slots on ONE prepositional
    attachment.

    So the head is tested on its own (it must be a genuine term by the same
    canonical rules), and the qualifier is allowed provided the whole span stays
    short and carries no clause marker — which is what separates "the slope of a
    line" from "a fraction in which the numerator or the denominator".
    """
    if _CLAUSE_MARKER_RE.search(core):
        return False
    if len(core) > _MAX_TERM_CHARS:
        return False
    if len(core.split()) > _MAX_SUBJECT_WORDS_WITH_PP:
        return False
    parts = _SUBJECT_PP_SPLIT_RE.split(core, maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        return False
    return _is_term_like_concept(parts[0])


def _is_definitional_subject(subject: str) -> bool:
    """True when ``subject`` reads as a real definitional noun-phrase term.

    Applied to group 1 of every :data:`ContentExtractor.DEFINITION_PATTERNS`
    match before a :class:`KeyTerm` is minted. See the block comment above for
    why the copula patterns need it and why precision is weighted over recall.
    """
    if not subject:
        return False
    stripped = subject.strip().strip("\"'").rstrip(".,;:")
    if _SUBJECT_LABEL_PUNCT_RE.search(stripped):
        return False
    # Strip the leading determiner / quantifier first: it is never the head, and
    # a definition sentence routinely opens with one ("Every integer is …").
    had_determiner = bool(_SUBJECT_LEADING_DETERMINER_RE.match(stripped))
    core = _SUBJECT_LEADING_DETERMINER_RE.sub("", stripped).strip()
    head_tokens = re.findall(r"[A-Za-z']+", core)
    if not head_tokens:
        return False
    # Role: the head must be a noun, not a closed-class sentence opener. A
    # preceding determiner PROVES the head is nominal — "second"/"given"/"first"
    # open a discourse move or a participial clause only when bare, never after
    # "A" or "The" — so the head check applies to BARE subjects only. That keeps
    # "The second law is the conservation of energy" and "The given information
    # is the base and the height" while still dropping "Second, the result is …"
    # and "Given that x is a positive integer, …".
    if not had_determiner and head_tokens[0].lower() in _NON_NOMINAL_SUBJECT_HEADS:
        return False
    # Shape: short, no clause glue, passes the canonical fragment filter —
    # either as a bare noun phrase or as a head plus one prepositional
    # qualifier.
    return _is_term_like_concept(core) or _head_np_is_term_like(core)


class ContentExtractor:
    """Extract question-worthy elements from retrieved RAG chunks.

    Usage:
        extractor = ContentExtractor()
        chunks = [{"id": "c1", "text": "...", "concept_tags": [...]}]
        terms = extractor.extract_key_terms(chunks)
        statements = extractor.extract_factual_statements(chunks)
    """

    # Patterns that indicate a definition.
    #
    # ``re.IGNORECASE`` is applied pattern-wide so the CONNECTIVE matches in any
    # case ("IS DEFINED AS" in an all-caps callout, "Refers To" in a title-cased
    # heading). Under that flag ``[A-Z]`` matches lowercase too — so the subject
    # anchor is a real constraint only where it is written case-SENSITIVE.
    #
    # Four of the five patterns are already bounded on the left by
    # ``(?:^|(?<=\.\s))``, a genuine sentence-start anchor, so their ``[A-Z]``
    # is redundant with it and case-insensitivity there costs nothing: it only
    # admits lowercase sentence starts produced by OCR or flattened glossaries (an
    # OCR'd line, a flattened glossary key — "denominator The denominator is the
    # number below the fraction bar"). Those stay tolerant.
    #
    # The ``X, which is Y`` pattern has NO anchor — its ``[A-Z]`` is its ONLY
    # left boundary — so a case-insensitive ``[A-Z]`` there lets the capture
    # begin at any letter anywhere in the chunk, including mid-word, and it
    # sweeps up whatever clause precedes the nearest ", which is". That one is
    # scoped case-sensitive with ``(?-i:[A-Z])`` so the boundary means what it
    # says, while the connective stays case-insensitive.
    DEFINITION_PATTERNS = [
        # "X is defined as Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+(?:is|are)\s+defined\s+as\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        # "X refers to Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+refers?\s+to\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        # "X is the Y" / "X is a Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+(?:is|are)\s+(?:the|a|an)\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        # "X, which is Y," — unanchored, so its subject boundary is
        # case-SENSITIVE (see the note above).
        re.compile(
            r"((?-i:[A-Z])[^,]*?),\s+which\s+(?:is|are)\s+(.+?)(?:,|\.|$)",
            re.IGNORECASE,
        ),
        # "X means Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+means?\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating causal/comparative relationships
    RELATIONSHIP_PATTERNS = [
        # "X causes Y" / "X leads to Y"
        re.compile(
            r"([^,.]+?)\s+(?:causes?|leads?\s+to|results?\s+in|produces?)\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "X is related to Y"
        re.compile(
            r"([^,.]+?)\s+(?:is|are)\s+(?:related|connected|linked)\s+to\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "Unlike X, Y..."
        re.compile(
            r"Unlike\s+([^,]+?),\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "X differs from Y"
        re.compile(
            r"([^,.]+?)\s+differs?\s+from\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "While X..., Y..."
        re.compile(
            r"While\s+([^,]+?),\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "X because Y"
        re.compile(
            r"([^,.]+?)\s+because\s+([^,.]+)",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating procedural/step content
    STEP_INDICATORS = re.compile(
        r"(?:^|\n)\s*(?:"
        r"(?:step|stage|phase)\s+\d+"
        r"|(?:first|second|third|fourth|fifth|next|then|finally|lastly)"
        r"|\d+[.)]\s"
        r"|[a-z][.)]\s"
        r")",
        re.IGNORECASE,
    )

    # Patterns for examples
    EXAMPLE_PATTERNS = re.compile(
        r"(?:for\s+example|for\s+instance|such\s+as|e\.g\.|consider\s+the"
        r"|an?\s+example\s+(?:of|is)|to\s+illustrate|in\s+practice)",
        re.IGNORECASE,
    )

    def extract_from_metadata(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured content directly from chunk metadata.

        When chunks carry Courseforge metadata (key_terms, misconceptions,
        bloom_level, content_type_label), this method returns pre-structured
        data without regex parsing.

        Returns dict with keys: key_terms, misconceptions, bloom_levels.
        Empty lists for fields not present in metadata.
        """
        key_terms: List[KeyTerm] = []
        misconceptions: List[Dict[str, str]] = []
        bloom_levels: List[str] = []
        seen_terms: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))

            # Key terms from Courseforge metadata
            for kt in (chunk.get("key_terms") or []):
                if isinstance(kt, dict) and kt.get("term"):
                    term_key = kt["term"].lower()
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        key_terms.append(KeyTerm(
                            term=kt["term"],
                            definition=kt.get("definition", ""),
                            source_chunk_id=chunk_id,
                            context_sentence=f'{kt["term"]}: {kt.get("definition", "")}',
                        ))

            # Misconceptions
            for mc in (chunk.get("misconceptions") or []):
                if isinstance(mc, dict) and mc.get("misconception"):
                    misconceptions.append(mc)

            # Bloom's level
            bl = chunk.get("bloom_level")
            if bl and bl not in bloom_levels:
                bloom_levels.append(bl)

        return {
            "key_terms": key_terms,
            "misconceptions": misconceptions,
            "bloom_levels": bloom_levels,
        }

    def extract_key_terms(
        self, chunks: List[Dict[str, Any]]
    ) -> List[KeyTerm]:
        """Extract defined terms from chunk content.

        Prefers structured key_terms from chunk metadata when available,
        falls back to regex pattern matching.

        Rejects TOC fragments + page-number patterns via
        :func:`_is_toc_fragment`. When a chunk's candidate terms are all
        rejected the chunk is tagged with a ``EMPTY_TERMS_TOC_CHUNK``
        diagnostic in its ``metadata_diagnostics`` list so downstream
        generators can skip or fall back to chunk-text sampling.
        """
        # Check if any chunks have structured key_terms metadata
        metadata_result = self.extract_from_metadata(chunks)
        if metadata_result["key_terms"]:
            return metadata_result["key_terms"]

        terms: List[KeyTerm] = []
        seen_terms: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            raw_text = chunk.get("text", "")
            text = _strip_html(raw_text)
            concept_tags = chunk.get("concept_tags", [])

            # Track candidates at this chunk to detect all-rejected state
            candidates_seen = 0
            candidates_accepted = 0

            # Strategy 1: Definition patterns
            for pattern in self.DEFINITION_PATTERNS:
                for match in pattern.finditer(text):
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    if len(term) < 3 or len(definition) < 10:
                        continue
                    candidates_seen += 1
                    if _is_toc_fragment(term):
                        continue
                    # The copula patterns capture "sentence start → first
                    # copula", which is a TERM only when the sentence really is
                    # a definition. Require a genuine nominal definitional
                    # subject so an interrogative / demonstrative / existential
                    # / subordinate-clause / imperative preamble never becomes
                    # a key term — and therefore never becomes a
                    # fill-in-the-blank answer or an MCQ stem subject.
                    if not _is_definitional_subject(term):
                        continue
                    # Apparatus text must never become a definition — nor the
                    # TERM, which is reused verbatim as the fill-in-the-blank
                    # answer key and as the MCQ stem subject.
                    if _is_apparatus_key_term(
                        term, definition, match.group(0)
                    ):
                        continue
                    term_key = term.lower()
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        candidates_accepted += 1
                        terms.append(KeyTerm(
                            term=term,
                            definition=definition,
                            source_chunk_id=chunk_id,
                            context_sentence=match.group(0).strip(),
                        ))

            # Strategy 2: Bold/strong terms in HTML with surrounding context
            bold_matches = re.finditer(
                r"<(?:strong|b|em)>([^<]+)</(?:strong|b|em)>",
                raw_text,
                re.IGNORECASE,
            )
            for bold_match in bold_matches:
                term = bold_match.group(1).strip()
                if len(term) < 2 or term.lower() in seen_terms:
                    continue
                candidates_seen += 1
                # Reject TOC fragments in bold/strong terms too — textbooks
                # often bold chapter headings.
                if _is_toc_fragment(term):
                    continue
                # Get surrounding sentence context
                pos = bold_match.start()
                text_around = _strip_html(raw_text[max(0, pos - 200): pos + 300])
                sentences = _split_sentences(text_around)
                context = ""
                definition = ""
                for sent in sentences:
                    if term.lower() in sent.lower():
                        context = sent
                        # Try to extract the part after the term as definition
                        parts = re.split(
                            re.escape(term), sent, flags=re.IGNORECASE, maxsplit=1
                        )
                        if len(parts) > 1:
                            definition = parts[1].strip().lstrip("—–-:,").strip()
                        break
                # Same apparatus screen as Strategy 1: a bolded "Solution" /
                # "Step 2:" label inside a worked example is markup, not a key
                # term, and generated course HTML bolds exactly those labels.
                if _is_apparatus_key_term(term, definition, context):
                    continue
                if context and len(definition) > 10:
                    seen_terms.add(term.lower())
                    candidates_accepted += 1
                    terms.append(KeyTerm(
                        term=term,
                        definition=definition,
                        source_chunk_id=chunk_id,
                        context_sentence=context,
                    ))

            # Strategy 3: concept_tags matched in text
            for tag in concept_tags:
                # Route deslugify through the canonical helper so trailing
                # CO-NN / TO-NN learning-objective refs get stripped before
                # they bleed into key-term display text.
                tag_lower = deslugify_concept(tag.lower())
                if tag_lower in seen_terms:
                    continue
                candidates_seen += 1
                if _is_toc_fragment(tag_lower):
                    continue
                # Find sentence containing the tag. The whole sentence becomes
                # the definition AND the context, so an apparatus sentence
                # must not be adopted just because it happens to mention a
                # concept tag.
                for sentence in _split_sentences(text):
                    if tag_lower not in sentence.lower():
                        continue
                    if _is_apparatus_key_term(tag_lower, sentence, sentence):
                        continue
                    seen_terms.add(tag_lower)
                    candidates_accepted += 1
                    terms.append(KeyTerm(
                        term=deslugify_concept(tag).title(),
                        definition=sentence,
                        source_chunk_id=chunk_id,
                        context_sentence=sentence,
                    ))
                    break

            # All candidates were rejected as TOC fragments. Tag the chunk so
            # downstream callers can see that key-term extraction yielded
            # nothing for a reason.
            if candidates_seen > 0 and candidates_accepted == 0:
                diagnostics = chunk.setdefault("metadata_diagnostics", [])
                if "EMPTY_TERMS_TOC_CHUNK" not in diagnostics:
                    diagnostics.append("EMPTY_TERMS_TOC_CHUNK")

        return terms

    def extract_factual_statements(
        self, chunks: List[Dict[str, Any]]
    ) -> List[FactualStatement]:
        """Extract declarative factual statements from chunk content.

        Filters for sentences with clear subjects and predicates,
        suitable for true/false or fill-in-blank questions.
        """
        statements: List[FactualStatement] = []
        seen: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            text = _strip_html(chunk.get("text", ""))

            for sentence in _split_sentences(text):
                # Skip questions, fragments, and very long sentences
                if sentence.endswith("?") or len(sentence) < 20 or len(sentence) > 300:
                    continue

                # Skip exercise/solution APPARATUS text. On an
                # apparatus-dense worked-example corpus such fragments
                # ("Solution: r Check: If r = 20, ...") otherwise mine as
                # factual statements and ship as correct answers.
                if _is_apparatus_text(sentence):
                    continue

                # Must be declarative (contains a verb-like structure)
                if not re.search(r"\b(?:is|are|was|were|has|have|can|will|does|do|provides?|involves?|requires?|includes?|consists?|contains?|represents?)\b", sentence, re.IGNORECASE):
                    continue

                # Extract main subject (first noun phrase before first verb)
                subject_match = re.match(
                    r"^((?:The\s+|A\s+|An\s+)?[A-Z][^,;]*?)\s+(?:is|are|was|were|has|have|can|will|does|do|provides?|involves?|requires?)",
                    sentence,
                )
                subject = subject_match.group(1).strip() if subject_match else ""

                if not subject or len(subject) < 3:
                    continue

                # A BARE pronoun / demonstrative subject loses its antecedent
                # the moment the sentence leaves its paragraph, so the item
                # built from it is unanswerable on its own. Reject at the
                # source rather than let the linter flag the symptom.
                if _has_anaphoric_subject(subject):
                    continue

                norm = sentence.lower().strip()
                if norm not in seen:
                    seen.add(norm)
                    statements.append(FactualStatement(
                        statement=sentence,
                        source_chunk_id=chunk_id,
                        key_subject=subject,
                    ))

        return statements

    def extract_relationships(
        self, chunks: List[Dict[str, Any]]
    ) -> List[ConceptRelationship]:
        """Extract concept relationships (causal, comparative, associative)."""
        relationships: List[ConceptRelationship] = []
        seen: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            text = _strip_html(chunk.get("text", ""))

            for sentence in _split_sentences(text):
                for pattern in self.RELATIONSHIP_PATTERNS:
                    match = pattern.search(sentence)
                    if match:
                        a = match.group(1).strip()
                        b = match.group(2).strip()
                        if len(a) < 3 or len(b) < 3:
                            continue
                        # Clause-grab guard: only mint a relationship when
                        # BOTH captures read as short noun-phrase terms.
                        # ``([^,.]+?)`` otherwise grabs a full clause, and
                        # the "relationship between X and Y" stem template
                        # emits word salad. A rejected sentence simply
                        # yields no relationship — downstream generators
                        # fall back to their next template (procedure /
                        # example / key-term).
                        if not (
                            _is_term_like_concept(a)
                            and _is_term_like_concept(b)
                        ):
                            continue
                        # Determine relationship type from the matched pattern
                        rel = match.group(0).strip()
                        key = (a.lower(), b.lower())
                        if key not in seen:
                            seen.add(key)
                            relationships.append(ConceptRelationship(
                                concept_a=a,
                                concept_b=b,
                                relationship=rel,
                                full_statement=sentence,
                                source_chunk_id=chunk_id,
                            ))
                        break  # One relationship per sentence

        return relationships

    def extract_procedures(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Procedure]:
        """Extract step-by-step procedures from content."""
        procedures: List[Procedure] = []

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            raw_text = chunk.get("text", "")

            # Look for ordered lists in HTML
            ol_matches = re.finditer(
                r"<ol[^>]*>(.*?)</ol>", raw_text, re.DOTALL | re.IGNORECASE
            )
            for ol_match in ol_matches:
                items = re.findall(r"<li[^>]*>(.*?)</li>", ol_match.group(1), re.DOTALL)
                if len(items) >= 2:
                    steps = [_strip_html(item) for item in items]
                    # Try to find a heading before the list
                    pre_text = raw_text[: ol_match.start()]
                    heading_match = re.search(
                        r"<h[1-6][^>]*>([^<]+)</h[1-6]>(?:\s*$)",
                        pre_text[-300:],
                        re.IGNORECASE,
                    )
                    title = heading_match.group(1).strip() if heading_match else "Process"
                    procedures.append(Procedure(
                        title=title,
                        steps=steps,
                        source_chunk_id=chunk_id,
                    ))

            # Look for numbered/step text patterns
            text = _strip_html(raw_text)
            step_blocks = re.findall(
                r"(?:Step\s+\d+[:.]\s*)(.*?)(?=Step\s+\d+|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if len(step_blocks) >= 2:
                steps = [s.strip() for s in step_blocks if s.strip()]
                procedures.append(Procedure(
                    title="Process",
                    steps=steps,
                    source_chunk_id=chunk_id,
                ))

        return procedures

    def extract_examples(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Example]:
        """Extract examples, case studies, and applications from content."""
        examples: List[Example] = []

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            text = _strip_html(chunk.get("text", ""))

            for sentence in _split_sentences(text):
                match = self.EXAMPLE_PATTERNS.search(sentence)
                if match:
                    # Get surrounding context (previous sentence if available)
                    all_sentences = _split_sentences(text)
                    idx = next(
                        (i for i, s in enumerate(all_sentences) if sentence in s),
                        -1,
                    )
                    context = all_sentences[idx - 1] if idx > 0 else ""

                    examples.append(Example(
                        description=sentence,
                        context=context,
                        source_chunk_id=chunk_id,
                    ))

        return examples

    def recover_numeric_equation_candidates(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """Flag-gated recovery of plain-text equation candidates.

        Scans each chunk's text for apparatus markers (``Solution:`` /
        ``Check:`` / ``Step N:``) and, within the window that FOLLOWS each
        marker, harvests plain-text ``LHS = RHS`` equation fragments the
        marked-math harvester misses on scan corpora. Returns a list of
        ``(equation_fragment, chunk_id)`` tuples in chunk-then-position order,
        each fragment a candidate the numeric-FIB builder then sympy-VERIFIES
        before shipping.

        Gated by ``ED4ALL_ASSESSMENT_NUMERIC_RECOVERY``: returns ``[]`` when the
        flag is off (byte-identical) so no apparatus content is re-admitted on
        any legacy run. This is the ONLY path that lifts the apparatus guard,
        and only to SUPPLY sympy-verifiable candidates — never to write an
        answer/definition/statement (the guard stays everywhere else).
        """
        if not _numeric_recovery_enabled():
            return []
        out: List[Tuple[str, str]] = []
        seen: set = set()
        for chunk in chunks or []:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            text = _strip_html(chunk.get("text", "") or "")
            if not text:
                continue
            for marker in _RECOVERY_MARKER_RE.finditer(text):
                window = text[marker.end(): marker.end() + _RECOVERY_WINDOW]
                for frag in extract_plaintext_equations(window):
                    key = (str(chunk_id), frag)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((frag, str(chunk_id)))
        return out

    def extract_all(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract all content types at once.

        Returns dict with keys: key_terms, factual_statements,
        relationships, procedures, examples.
        """
        return {
            "key_terms": self.extract_key_terms(chunks),
            "factual_statements": self.extract_factual_statements(chunks),
            "relationships": self.extract_relationships(chunks),
            "procedures": self.extract_procedures(chunks),
            "examples": self.extract_examples(chunks),
        }
