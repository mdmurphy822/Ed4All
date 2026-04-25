"""Concept-graph node classifier (Wave 75).

Classifies every concept-graph node into a coarse class so retrieval can
filter pedagogical scaffolding ("key-takeaway", "rubric"), assessment
options ("answer-b"), and stop-word-like artifacts ("not", "do-not")
out of domain-concept similarity search.

The Wave 75 review surfaced that the existing
``concept_graph.json`` for the rdf-shacl-550 archive carried 459
nodes including pedagogical/assessment scaffolding that polluted
similarity search. This classifier is the deterministic, side-effect-free
labeler that lets retrieval gate by class without dropping or merging
nodes (existing edges stay intact).

Classes
-------
- ``DomainConcept`` — real subject-matter terms (turtle, rdf-graph,
  sh:path, owl-2-rl, blank-node, sparql-select).
- ``PedagogicalMarker`` — instructional scaffolding (key-takeaway,
  rubric, learning-objective, summary, application, self-check,
  practice, callout, exercise).
- ``AssessmentOption`` — quiz answer choices (answer-a..answer-d,
  option-a..option-d, correct-answer, distractor).
- ``InstructionalArtifact`` — meta-content (submission-format,
  deadline, week-overview, module-header, what-you-will-produce).
- ``LearningObjective`` — to-NN / co-NN IDs that leaked into the
  concept space.
- ``Misconception`` — flagged misconception nodes (caller-supplied
  hint).
- ``LowSignal`` — single-word negations + stop-word-like artifacts
  (not, do-not, the, a, with, by, of, ...).

The classifier is deterministic and side-effect-free; precedence is
fixed (see ``classify_concept`` docstring).
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set

# Public class enum - keep in sync with the docstring + tests.
DOMAIN_CONCEPT = "DomainConcept"
PEDAGOGICAL_MARKER = "PedagogicalMarker"
ASSESSMENT_OPTION = "AssessmentOption"
INSTRUCTIONAL_ARTIFACT = "InstructionalArtifact"
LEARNING_OBJECTIVE = "LearningObjective"
MISCONCEPTION = "Misconception"
LOW_SIGNAL = "LowSignal"

CONCEPT_CLASSES = frozenset({
    DOMAIN_CONCEPT,
    PEDAGOGICAL_MARKER,
    ASSESSMENT_OPTION,
    INSTRUCTIONAL_ARTIFACT,
    LEARNING_OBJECTIVE,
    MISCONCEPTION,
    LOW_SIGNAL,
})

# Rule 1: LO IDs that leaked into concept space.
# Mirrors the canonical LO pattern from
# ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` but case-
# insensitive because concept-graph slugs are typically lowercased.
_LO_ID_RE = re.compile(r"^(?:to|co)-\d{2,}$", re.IGNORECASE)

# Rule 2: assessment-option choices (multiple-choice answer slots).
# Wave 76 expands beyond a-d single-letter options to cover
# answer-true / answer-false / option-true variants observed in the
# rdf-shacl-550 review.
_ANSWER_OPTION_RE = re.compile(
    r"^(?:answer|option)-(?:[a-d]|true|false|yes|no)$",
    re.IGNORECASE,
)

# Wave 76: naked truth/answer tokens that escape into the concept
# stream from quiz body text. As concept slugs they're always
# quiz-answer noise rather than domain terms.
_TRUTH_VALUE_TOKENS: Set[str] = frozenset({"true", "false", "yes", "no"})

# Wave 76: HTML entity contamination — when slugification runs over
# raw HTML without entity decoding first, ``&mdash;`` becomes the
# literal string ``mdash`` embedded in the slug (e.g.
# ``pitfall-mdash-target-class``). Same shape for ``ndash`` and the
# numeric variants. Once these tokens are detected we drop the whole
# concept as a fragment — the entity glue tells us the slug spanned a
# punctuation boundary the chunker should have respected.
_HTML_ENTITY_NOISE_RE = re.compile(
    r"(?:^|-)(?:mdash|ndash|hellip|nbsp|amp|quot|lt|gt|apos|rsquo|lsquo|rdquo|ldquo)(?:-|$)",
    re.IGNORECASE,
)

# Wave 76: article / preposition / conjunction prefixes that mark
# sentence-fragment slugs. ``to-`` and ``co-`` are intentionally
# OMITTED — Rule 1 catches LO IDs first, and Wave 75 tests pin
# ``to-string`` / ``co-author`` / ``co-occurrence`` as DomainConcept.
_FRAGMENT_PREFIXES: Set[str] = frozenset({
    "a-",
    "the-",
    "an-",
    "and-",
    "or-",
    "but-",
    "by-",
    "of-",
    "in-",
    "on-",
    "at-",
    "from-",
    "for-",
    "as-",
    "if-",
    "after-",
    "before-",
    "while-",
    "during-",
    "every-",
    "each-",
    "any-",
    "some-",
    "use-",
    "choose-",
    "important-",
})

# Wave 76: pedagogical-marker pattern matchers for compound slugs that
# the static stoplist misses (``module-4-deliverable``,
# ``rubric-preview``, ``application-activity``, ``self-check-five``).
_PEDAGOGY_PATTERN_RE = re.compile(
    r"(?:^|-)(?:rubric|deliverable|self-check|key-takeaway|takeaway|"
    r"learning-objective|learning-outcome|review-question|"
    r"application-activity|practice-problem)(?:-|$)",
    re.IGNORECASE,
)

# Wave 76: ``module-NN-*`` / ``week-NN-*`` / ``content-N-*`` slugs
# are course logistics, not domain concepts. ``content-N-X-Y`` is
# Courseforge's section-numbering pattern (Section 1.1 → ``content-1``)
# and the trailing tokens are heading-fragment text.
_LOGISTICS_PREFIX_RE = re.compile(
    r"^(?:module|week|unit|lesson|chapter|section|content|pitfall|"
    r"objective|outcome)-\d+(?:-|$)",
    re.IGNORECASE,
)

# Wave 76: trailing stopword detection. A slug whose LAST hyphen-
# delimited token is a stopword is a sentence fragment that the
# 4-token slugifier truncation produced. Examples flagged in the
# rdf-shacl-550 review: ``content-1-aggregation-and``,
# ``competency-questions-are-the``, ``bring-a-shacl-sparql`` (where
# the last token is itself a tail of an unfinished phrase).
_TAIL_STOPWORDS: Set[str] = frozenset({
    "a", "an", "the",
    "and", "or", "but", "nor",
    "of", "in", "on", "at", "by", "to", "for", "from", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "as", "if", "than", "then",
    "do", "does", "did",
})

# Wave 76: starting auxiliary / wh- tokens. Slugs that begin with
# ``are-``, ``is-``, ``do-``, ``how-``, ``why-``, ``what-`` etc. are
# almost always question-fragment slugs from quiz body text or
# discussion prompts (e.g. ``are-you-enriching-rdf``).
_AUXILIARY_LEAD_TOKENS: Set[str] = frozenset({
    "are", "is", "was", "were", "do", "does", "did",
    "have", "has", "had", "will", "would", "should", "could",
    "may", "might", "must", "can",
    "how", "why", "what", "where", "when", "which", "who", "whose",
})

# Wave 76: LO IDs baked into the MIDDLE of a slug (``composition-to-03-progress``
# is a heading fragment that contains the LO reference ``to-03``).
# Match anything of the shape ``...-(to|co)-NN-...``.
_EMBEDDED_LO_RE = re.compile(
    r"-(?:to|co)-\d{2,}-",
    re.IGNORECASE,
)

# Rule 3 stoplist: pedagogical scaffolding tags.
PEDAGOGICAL_MARKERS: Set[str] = frozenset({
    "key-takeaway",
    "key-takeaways",
    "takeaway",
    "rubric",
    "rubrics",
    # ChatGPT review flagged these top-3 polluters in the rdf-shacl-550
    # concept graph; they're the meta-vocabulary that scaffolds
    # assessments rather than the domain content the assessments cover.
    "assessment",
    "assessments",
    "quiz",
    "test",
    "callout",
    "callout-box",
    "summary",
    "summary-box",
    "summary-section",
    "learning-objective",
    "learning-objectives",
    "learning-outcome",
    "learning-outcomes",
    "deliverable",
    "deliverables",
    "self-check",
    "self-assessment",
    "practice",
    "practice-problem",
    "exercise",
    "exercises",
    "application",
    "applications",
    "application-section",
    "reflection",
    "reflection-prompt",
    "review",
    "review-question",
    "review-questions",
    "discussion",
    "discussion-prompt",
    "warm-up",
    "wrap-up",
    "preview",
    "introduction",
    "intro",
    "objectives",
    "outline",
    "agenda",
    "tip",
    "tips",
    "note",
    "notes",
    "example",
    "examples",
    "feedback",
    # Wave 76 additions surfaced by the rdf-shacl-550 review: compound
    # pedagogy artifacts that masqueraded as DomainConcept under the
    # Wave 75 stoplist.
    "application-activity",
    "rubric-preview",
    "rubric-rubric",
    "deliverable-preview",
    "checkpoint",
    "milestone",
    "self-checks",
})

# Rule 4 stoplist: low-signal stop-word-like artifacts. These tend to
# appear as concept tags only because slug-extraction pulled isolated
# tokens out of body copy.
LOW_SIGNAL_TOKENS: Set[str] = frozenset({
    # negations
    "not",
    "do-not",
    "dont",
    "don-t",
    "never",
    "no",
    # determiners / articles
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    # prepositions / conjunctions
    "with",
    "without",
    "by",
    "of",
    "on",
    "in",
    "to",
    "from",
    "for",
    "as",
    "at",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "than",
    "so",
    # auxiliaries / modals
    "is",
    "was",
    "be",
    "been",
    "being",
    "are",
    "were",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "must",
    # pronouns
    "it",
    "its",
    "they",
    "them",
    "we",
    "you",
    "your",
    "our",
    # other extracted noise
    "etc",
})

# Rule 5 stoplist: instructional artifacts (course logistics + meta).
INSTRUCTIONAL_ARTIFACTS: Set[str] = frozenset({
    "submission-format",
    "submission",
    "deadline",
    "due-date",
    "week-overview",
    "module-header",
    "module-overview",
    "course-header",
    "course-overview",
    "what-you-will-produce",
    "what-you-will-learn",
    "what-you-will-do",
    "estimated-time",
    "estimated-duration",
    "duration",
    "prerequisites",
    "grading",
    "grading-criteria",
    "grading-scheme",
    "weight",
    "weighting",
    "instructions",
    "instructor-notes",
    "readings",
    "resources",
    "schedule",
    "calendar",
    # Wave 76: additional logistics terms surfaced in the rdf-shacl-550
    # cleanup pass.
    "module-overview",
    "week-overview",
    "course-introduction",
    "syllabus",
})

# Wave 76: classes that the concept-extraction pipeline should
# REJECT (drop entirely from the concept stream) rather than emit. The
# Wave 75 classifier was post-hoc — it labeled but did not filter, so
# pollution still entered chunks ``concept_tags`` and the resulting
# concept_graph nodes. Wave 76 wires :func:`is_droppable_class` at
# extraction time. Membership rationale:
#
# - ``PedagogicalMarker`` — instructional scaffolding, not domain
#   vocabulary.
# - ``AssessmentOption`` — quiz answer slots / truth values.
# - ``LowSignal`` — stopwords + sentence fragments + entity-glue
#   artifacts.
# - ``InstructionalArtifact`` — submission logistics + meta-content.
# - ``LearningObjective`` — LO IDs (TO-04, CO-12); these belong in
#   ``objectives.json`` not ``concept_graph.json``. Per the Wave 76
#   task spec, they're dropped from concept space entirely.
DROPPABLE_CLASSES: Set[str] = frozenset({
    PEDAGOGICAL_MARKER,
    ASSESSMENT_OPTION,
    LOW_SIGNAL,
    INSTRUCTIONAL_ARTIFACT,
    LEARNING_OBJECTIVE,
})


def is_droppable_class(klass: str) -> bool:
    """Return True iff ``klass`` is a class the extractor should drop.

    Used by ``Trainforge.process_course.CourseProcessor._extract_concept_tags``
    (and the Wave 76 retroactive cleanup script) to filter at emit
    time. Domain concepts and misconceptions are kept.
    """
    return klass in DROPPABLE_CLASSES


# Wave 76: serialization-format aliases. Slugifier strips ``/`` and
# ``+``, so ``RDF/XML``/``rdfxml``/``rdf-xml`` collapse to a mix of
# slugs depending on the upstream punctuation. The mapping below
# canonicalizes any equivalent variant onto a single concept slug so
# the graph doesn't carry near-duplicate nodes.
KNOWN_EQUIVALENT_ALIASES: Dict[str, str] = {
    "rdfxml": "rdf-xml",
    "rdf-xml": "rdf-xml",  # canonical
    "jsonld": "json-ld",
    "json-ld": "json-ld",  # canonical
    "ntriples": "n-triples",
    "n-triples": "n-triples",  # canonical
    "nquads": "n-quads",
    "n-quads": "n-quads",
    "turtle": "turtle",
    "ttl": "turtle",
}


def canonicalize_alias(slug: str) -> str:
    """Return the canonical slug for known equivalent variants.

    Pass-through for slugs not in :data:`KNOWN_EQUIVALENT_ALIASES`.
    """
    if not slug:
        return slug
    return KNOWN_EQUIVALENT_ALIASES.get(slug.lower(), slug)


# Wave 76: trivial English plural suffixes that slug-extraction tends
# to flip-flop on (``triple``/``triples``, ``graph``/``graphs``,
# ``ontology``/``ontologies``). The collapse helper prefers the
# singular when both forms appear.
_PLURAL_SINGULARIZATIONS = (
    ("ies", "y"),    # ontologies → ontology
    ("ses", "s"),    # classes → class
    ("xes", "x"),    # axes → ax (handled, but keep generic)
    ("s", ""),       # triples → triple, graphs → graph
)


def singular_form(slug: str) -> str:
    """Return a candidate singular form for ``slug`` (best-effort).

    Used by the duplicate-collapse pass: if both ``X`` and ``Xs`` are
    present in the concept stream, prefer ``X``. Conservative — only
    chops the suffix if the result is at least 3 chars long. Returns
    the input unchanged when no rule applies or the trim is too short.
    """
    if not slug or len(slug) < 4:
        return slug
    lowered = slug.lower()
    for suffix, replacement in _PLURAL_SINGULARIZATIONS:
        if lowered.endswith(suffix) and (
            len(lowered) - len(suffix) + len(replacement) >= 3
        ):
            return lowered[: -len(suffix)] + replacement if suffix else lowered
    return slug


def _normalize(node_id: str) -> str:
    """Lowercase + strip whitespace. Empty input → ``""``."""
    if node_id is None:
        return ""
    return str(node_id).strip().lower()


def _has_fragment_prefix(norm: str) -> bool:
    """True when ``norm`` begins with an article/preposition/conjunction
    that marks it as a sentence fragment.

    The ``to-``/``co-`` LO prefixes are deliberately not in the set —
    Rule 1 catches LO IDs first, and Wave 75 tests pin ``to-string``,
    ``co-author``, ``co-occurrence`` as DomainConcept.
    """
    for prefix in _FRAGMENT_PREFIXES:
        if norm.startswith(prefix):
            return True
    return False


def classify_concept(
    node_id: str,
    label: Optional[str] = None,
    hints: Optional[Dict[str, object]] = None,
) -> str:
    """Classify a concept-graph node.

    Returns one of: ``DomainConcept``, ``PedagogicalMarker``,
    ``AssessmentOption``, ``InstructionalArtifact``,
    ``LearningObjective``, ``Misconception``, ``LowSignal``.

    Precedence (first match wins):
      1. ``^(to|co)-NN$`` slug → ``LearningObjective``.
      2. ``^(answer|option)-(?:[a-d]|true|false|yes|no)$`` →
         ``AssessmentOption``.
      3. Slug ∈ :data:`_TRUTH_VALUE_TOKENS` (``true``/``false``/``yes``/
         ``no``) → ``AssessmentOption``.
      4. Slug in :data:`PEDAGOGICAL_MARKERS` (Wave 75 + Wave 76
         additions) → ``PedagogicalMarker``.
      5. :data:`_PEDAGOGY_PATTERN_RE` matches (compound pedagogy slugs
         like ``module-4-deliverable``, ``rubric-preview``,
         ``application-activity-week-2``) → ``PedagogicalMarker``.
      6. :data:`_LOGISTICS_PREFIX_RE` matches (``module-NN-*`` /
         ``week-NN-*`` etc.) → ``InstructionalArtifact``.
      7. Slug in :data:`INSTRUCTIONAL_ARTIFACTS` →
         ``InstructionalArtifact``.
      8. Slug in :data:`LOW_SIGNAL_TOKENS` → ``LowSignal``.
      9. Wave 76 length / numeric guards (``len < 3`` or pure-numeric)
         → ``LowSignal``.
     10. Wave 76 :data:`_HTML_ENTITY_NOISE_RE` matches (``-mdash-``,
         ``-ndash-``, etc.) → ``LowSignal``.
     11. Wave 76 fragment-prefix detection
         (:func:`_has_fragment_prefix`) → ``LowSignal``.
     12. ``hints['is_misconception']`` truthy → ``Misconception``.
     13. Empty / missing input → ``LowSignal`` (graceful default).
     14. Fallback → ``DomainConcept``.

    The ``label`` argument is accepted for symmetry with downstream
    callers but is not consulted — classification is keyed off
    ``node_id`` (the canonical slug). This keeps the function
    deterministic regardless of whether labels are populated.

    The ``hints`` dict is consulted only for the ``Misconception``
    path; callers with stronger signals (e.g. an upstream
    misconception entity table) pass ``hints={"is_misconception":
    True}``.
    """
    norm = _normalize(node_id)

    # Rule 13 (early exit): empty / null inputs collapse to LowSignal.
    if not norm:
        return LOW_SIGNAL

    # Rule 1: LO IDs.
    if _LO_ID_RE.match(norm):
        return LEARNING_OBJECTIVE

    # Rule 2: answer / option slots (incl. true/false/yes/no variants).
    if _ANSWER_OPTION_RE.match(norm):
        return ASSESSMENT_OPTION

    # Rule 3 (Wave 76): naked truth-value tokens.
    if norm in _TRUTH_VALUE_TOKENS:
        return ASSESSMENT_OPTION

    # Rule 4: pedagogical scaffolding stoplist.
    if norm in PEDAGOGICAL_MARKERS:
        return PEDAGOGICAL_MARKER

    # Rule 5 (Wave 76): compound pedagogy patterns
    # (module-4-deliverable, rubric-preview, application-activity-*).
    # Run BEFORE the logistics prefix check so ``module-3-rubric`` is
    # routed to PedagogicalMarker (rubric carries the pedagogy
    # signal) rather than to the generic ``module-NN-`` logistics
    # bucket.
    if _PEDAGOGY_PATTERN_RE.search(norm):
        return PEDAGOGICAL_MARKER

    # Rule 6 (Wave 76): module-NN / week-NN / unit-NN logistics
    # prefixes. Anything matching here is course-shell scaffolding,
    # not domain content.
    if _LOGISTICS_PREFIX_RE.match(norm):
        return INSTRUCTIONAL_ARTIFACT

    # Rule 7: instructional artifacts.
    if norm in INSTRUCTIONAL_ARTIFACTS:
        return INSTRUCTIONAL_ARTIFACT

    # Rule 8: low-signal stop-word-like artifacts.
    if norm in LOW_SIGNAL_TOKENS:
        return LOW_SIGNAL

    # Rule 9 (Wave 76): drop pure-numeric and too-short slugs.
    if len(norm) < 3:
        return LOW_SIGNAL
    if norm.replace("-", "").isdigit():
        return LOW_SIGNAL

    # Rule 10 (Wave 76): HTML-entity contamination. Slugs like
    # ``pitfall-mdash-target-class`` arise when slugification ran over
    # raw HTML without an ``html.unescape`` pre-step. Once we see the
    # entity glue token we know the slug spans a punctuation boundary
    # the chunker should have respected → drop as fragment.
    if _HTML_ENTITY_NOISE_RE.search(norm):
        return LOW_SIGNAL

    # Rule 11 (Wave 76): article/preposition/conjunction prefix
    # detection. ``a-literal-is-just``, ``after-the-self-check``,
    # ``every-direct-type-of`` — sentence fragments that escaped the
    # chunker.
    if _has_fragment_prefix(norm):
        return LOW_SIGNAL

    # Wave 76 (additional): embedded LO ID detection. Slugs containing
    # an LO reference in the middle (e.g. ``composition-to-03-progress``)
    # are heading fragments that captured the inline reference. Run
    # this before the stopword / auxiliary checks because it's the
    # strongest signal.
    if _EMBEDDED_LO_RE.search(norm):
        return LOW_SIGNAL

    # Split into hyphen-delimited tokens for the trailing-stopword and
    # auxiliary-lead checks.
    tokens = norm.split("-")
    if len(tokens) >= 3:
        # Trailing-stopword check: 3+ token slugs whose tail is a
        # stopword are fragments produced by the 4-token slugifier
        # truncation (e.g. ``content-1-aggregation-and``).
        if tokens[-1] in _TAIL_STOPWORDS:
            return LOW_SIGNAL
        # Auxiliary-lead check: slugs that begin with ``are-``, ``is-``,
        # ``how-``, ``why-`` etc. (length 3+ to avoid catching
        # legitimate 2-token domain compounds).
        if tokens[0] in _AUXILIARY_LEAD_TOKENS:
            return LOW_SIGNAL
        # ``X-are-Y-Z`` / ``X-is-Y-Z`` middle-aux check: 4+ token
        # slugs with an aux verb in position 2 are fragments
        # (``chains-are-fixed-length``, ``rdfs-cannot-express-either``).
        if len(tokens) >= 4 and tokens[1] in _AUXILIARY_LEAD_TOKENS:
            return LOW_SIGNAL

    # Rule 12: caller-supplied misconception hint.
    if hints and bool(hints.get("is_misconception")):
        return MISCONCEPTION

    # Rule 14: fallback. Real domain vocabulary lands here.
    return DOMAIN_CONCEPT


__all__ = [
    "DOMAIN_CONCEPT",
    "PEDAGOGICAL_MARKER",
    "ASSESSMENT_OPTION",
    "INSTRUCTIONAL_ARTIFACT",
    "LEARNING_OBJECTIVE",
    "MISCONCEPTION",
    "LOW_SIGNAL",
    "CONCEPT_CLASSES",
    "DROPPABLE_CLASSES",
    "PEDAGOGICAL_MARKERS",
    "LOW_SIGNAL_TOKENS",
    "INSTRUCTIONAL_ARTIFACTS",
    "KNOWN_EQUIVALENT_ALIASES",
    "classify_concept",
    "is_droppable_class",
    "canonicalize_alias",
    "singular_form",
]
