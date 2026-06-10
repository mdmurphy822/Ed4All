"""Concept-graph node classifier (Wave 75).

Classifies every concept-graph node into a coarse class so retrieval can
filter pedagogical scaffolding ("key-takeaway", "rubric"), assessment
options ("answer-b"), and stop-word-like artifacts ("not", "do-not")
out of domain-concept similarity search.

The Wave 75 review surfaced that the existing
``concept_graph.json`` for the RDF/SHACL calibration corpus carried 459
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
# RDF/SHACL calibration corpus review.
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
#
# Wave 82 (Phase D2): adds ``step-N`` to the logistics filter — the
# RDF/SHACL calibration corpus audit found ``step-1`` and ``step-2`` showing up as
# top-frequency concepts because procedural-instruction headings of
# the shape "Step 1: ..." were slugified verbatim and entered the
# concept stream.
_LOGISTICS_PREFIX_RE = re.compile(
    r"^(?:module|week|unit|lesson|chapter|section|content|pitfall|"
    r"objective|outcome|step)-\d+(?:-|$)",
    re.IGNORECASE,
)

# Wave 76: trailing stopword detection. A slug whose LAST hyphen-
# delimited token is a stopword is a sentence fragment that the
# 4-token slugifier truncation produced. Examples flagged in the
# RDF/SHACL calibration corpus review: ``content-1-aggregation-and``,
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
    # ChatGPT review flagged these top-3 polluters in the RDF/SHACL
    # calibration corpus concept graph; they're the meta-vocabulary that scaffolds
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
    # Wave 76 additions surfaced by the RDF/SHACL calibration corpus review: compound
    # pedagogy artifacts that masqueraded as DomainConcept under the
    # Wave 75 stoplist.
    "application-activity",
    "rubric-preview",
    "rubric-rubric",
    "deliverable-preview",
    "checkpoint",
    "milestone",
    "self-checks",
    # Wave 82 (Phase D2): procedural-instruction verbs the RDF/SHACL
    # calibration corpus audit caught masquerading as top-frequency domain concepts. "Plan"
    # and "Verify" are common imperative-mood headings ("Plan your
    # approach", "Verify the validation report") — pedagogical
    # scaffolding, not domain vocabulary.
    "plan",
    "verify",
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

# Change B: domain-agnostic scaffolding-noise tokens. These are
# pedagogical / procedural / comparative scaffolding words that slug-
# extraction pulls out of headings and body copy ("Pros", "Cons",
# "Creating a graph", "Mutating state", "Advanced") and which fall
# through ``classify_concept`` to ``DomainConcept`` despite carrying no
# subject-matter signal. Unlike :data:`LOW_SIGNAL_TOKENS` (stopwords),
# these are content-adjacent words, so they are only pruned behind the
# default-OFF ``TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS`` flag — read at
# the call sites in ``cooccurrence_graph`` / ``concept_tagging``, never
# inside ``classify_concept`` (which must stay pure).
SCAFFOLDING_NOISE_TOKENS: Set[str] = frozenset({
    "pros",
    "cons",
    "creating",
    "mutating",
    "consuming",
    "branch",
    "harder",
    "chain",
    "pipe",
    "task",
    "advanced",
    "basic",
    "intermediate",
    "beginner",
    "expert",
    "simpler",
    "easier",
    "faster",
    "slower",
    "better",
    "worse",
    "recap",
    "recall",
    "deleting",
    "updating",
    "reading",
    "writing",
    # RAG-course audit (W-audit): single-token generic abstractions /
    # logistics words that survived as DomainConcept and injected ~30% of
    # related-to noise. Domain-agnostic; never real RAG concepts.
    "optional",
    "objective",
    "problem",
    "insight",
    "assumption",
    "recommendation",
    "question",
    "purpose",
    "method",
    "reason",
    "menu",
    "repeat",
    "seem",
    "gathered",
    "primer",
    "exercice",
    "finished",
})

# RAG-course audit (W-audit): MULTI-TOKEN course-logistics /
# instructional scaffolding phrases (hyphen-joined slugs) that survived
# as DomainConcept. Curated + conservative: only exact-match slugs that
# are domain-agnostic logistics phrasing. Real multi-word concepts
# (``vector-store``, ``knowledge-base``, ``semantic-guardrailing``,
# ``running-state``, ``prompt-engineering``) MUST stay out of this set.
_SCAFFOLDING_NOISE_PHRASES: Set[str] = frozenset({
    "assess-task",
    "once-finished",
    "final-exercice",
    "primer-for-next-notebook",
    "plan-of-action",
    "honest-take",
    "stand-alone",
    "prompt-passing",
    "key-modifications",
    "typical-use-inspection",
    "edge-case-inspection",
})

# Change B (N2): action-verb stems for the ``-ing`` gerund-noise rule.
# A single-token slug ending ``-ing`` whose de-gerund stem is in this
# set is procedural scaffolding ("creating", "branching", "piping").
_SCAFFOLDING_ACTION_STEMS: Set[str] = frozenset({
    "create",
    "mutate",
    "consume",
    "read",
    "write",
    "update",
    "delete",
    "build",
    "run",
    "branch",
    "chain",
    "pipe",
    "recall",
})


def is_scaffolding_noise(norm: str) -> bool:
    """Return True iff ``norm`` is domain-agnostic scaffolding noise.

    Pure / side-effect-free. The default-OFF
    ``TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS`` flag is read at the call
    sites (``cooccurrence_graph`` / ``concept_tagging``), NOT here.

    Rules (any match → True):
      * **N1** — ``norm`` ∈ :data:`SCAFFOLDING_NOISE_TOKENS`.
      * **N2** — single-token slug ending ``-ing`` whose de-gerund stem
        (strip ``ing``; restore a trailing ``e`` if the bare stem isn't
        a recognised stem) is in :data:`_SCAFFOLDING_ACTION_STEMS`.
      * **N3** — single-token slug ending ``-er`` / ``-est``, no hyphen,
        length ≤ 8 (catches ``harder`` / ``simpler`` / ``faster``).
      * **N4** — multi-token slug in :data:`_SCAFFOLDING_NOISE_PHRASES`
        (curated course-logistics / instructional phrases).
    """
    if not norm:
        return False
    norm = norm.strip().lower()
    if not norm:
        return False

    # N1: explicit single-token stoplist.
    if norm in SCAFFOLDING_NOISE_TOKENS:
        return True

    # N4: curated multi-token course-logistics phrases.
    if norm in _SCAFFOLDING_NOISE_PHRASES:
        return True

    single_token = "-" not in norm

    # N2: -ing gerund of an action verb (single token only).
    if single_token and norm.endswith("ing") and len(norm) > 3:
        stem = norm[:-3]
        if stem in _SCAFFOLDING_ACTION_STEMS:
            return True
        # Restore a dropped trailing ``e`` (creating -> creat -> create).
        if (stem + "e") in _SCAFFOLDING_ACTION_STEMS:
            return True

    # N3: short comparative / superlative single tokens.
    if single_token and len(norm) <= 8 and (
        norm.endswith("er") or norm.endswith("est")
    ):
        return True

    return False


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
    # Wave 76: additional logistics terms surfaced in the RDF/SHACL
    # calibration corpus cleanup pass.
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
#
# Wave 82 (Phase C2): extended with W3C-standard surface-form aliases
# so non-canonical query terms route to the canonical anchor slugs
# emitted by ``lib/ontology/tech_anchors.py``. Pairs with the Phase C1
# wiring that seeds the canonical nodes when
# TRAINFORGE_SEED_TECH_CONCEPTS=true.
#
# Wave 83 / Phase 2.2 — RDF/SHACL enrichment:
# This dict has been demoted to a TRANSITION CACHE. The source of
# truth is now ``schemas/context/aliases.ttl`` (loaded by
# ``lib.ontology.aliases``). ``canonicalize_alias`` consults the
# Turtle path first and falls back to this dict only when the Turtle
# load returns the slug unchanged but a dict entry exists. The dict
# stays in place during the rollout to insure against rdflib import
# failures and incomplete Turtle coverage; it will be removed once
# parity is proven across all corpora — see
# ``lib/ontology/tests/test_aliases.py::test_known_aliases_dict_parity``
# which asserts every entry below is reachable via the Turtle path.
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
    # Wave 82 — W3C standards full-name → acronym slug.
    "rdf-schema": "rdfs",
    "rdfs": "rdfs",
    "web-ontology-language": "owl",
    "owl": "owl",
    "shapes-constraint-language": "shacl",
    "shacl": "shacl",
    # Wave 82 — owl:sameAs surface variants → predicate slug.
    "owlsameas": "same-as",
    "owl-sameas": "same-as",
    "sameas": "same-as",
    "same-as": "same-as",
}


def canonicalize_alias(slug: str) -> str:
    """Return the canonical slug for known equivalent variants.

    Wave 83 (Phase 2.2): consults
    ``schemas/context/aliases.ttl`` via :mod:`lib.ontology.aliases`
    first. Falls back to :data:`KNOWN_EQUIVALENT_ALIASES` only when the
    Turtle path returns the slug unchanged AND the dict has a mapping
    — this preserves the original return contract for the (legacy)
    edge case where rdflib isn't installed or the Turtle file is
    out-of-date.

    Pass-through for slugs not in either source.
    """
    if not slug:
        return slug
    # Defer the import so a missing rdflib at module load time doesn't
    # break the entire concept_classifier module — only the Turtle
    # path becomes a no-op via the aliases module's internal try/except.
    from lib.ontology import aliases as _aliases_module

    canonical = _aliases_module.canonicalize(slug)
    if canonical != slug:
        return canonical
    # Turtle-path miss (or pass-through). Fall back to the transition
    # cache for any entries the Turtle file hasn't picked up yet.
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
    "SCAFFOLDING_NOISE_TOKENS",
    "INSTRUCTIONAL_ARTIFACTS",
    "KNOWN_EQUIVALENT_ALIASES",
    "classify_concept",
    "is_droppable_class",
    "is_scaffolding_noise",
    "canonicalize_alias",
    "singular_form",
]
