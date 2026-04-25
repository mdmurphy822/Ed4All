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
_ANSWER_OPTION_RE = re.compile(r"^(?:answer|option)-[a-d]$", re.IGNORECASE)

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
})


def _normalize(node_id: str) -> str:
    """Lowercase + strip whitespace. Empty input → ``""``."""
    if node_id is None:
        return ""
    return str(node_id).strip().lower()


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
      2. ``^(answer|option)-[a-d]$`` slug → ``AssessmentOption``.
      3. Slug in :data:`PEDAGOGICAL_MARKERS` → ``PedagogicalMarker``.
      4. Slug in :data:`LOW_SIGNAL_TOKENS` → ``LowSignal``.
      5. Slug in :data:`INSTRUCTIONAL_ARTIFACTS` → ``InstructionalArtifact``.
      6. ``hints['is_misconception']`` truthy → ``Misconception``.
      7. Empty / missing input → ``LowSignal`` (graceful default).
      8. Fallback → ``DomainConcept``.

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

    # Rule 7 (early exit): empty / null inputs collapse to LowSignal.
    if not norm:
        return LOW_SIGNAL

    # Rule 1: LO IDs.
    if _LO_ID_RE.match(norm):
        return LEARNING_OBJECTIVE

    # Rule 2: answer / option slots.
    if _ANSWER_OPTION_RE.match(norm):
        return ASSESSMENT_OPTION

    # Rule 3: pedagogical scaffolding stoplist.
    if norm in PEDAGOGICAL_MARKERS:
        return PEDAGOGICAL_MARKER

    # Rule 4: low-signal stop-word-like artifacts.
    if norm in LOW_SIGNAL_TOKENS:
        return LOW_SIGNAL

    # Rule 5: instructional artifacts.
    if norm in INSTRUCTIONAL_ARTIFACTS:
        return INSTRUCTIONAL_ARTIFACT

    # Rule 6: caller-supplied misconception hint.
    if hints and bool(hints.get("is_misconception")):
        return MISCONCEPTION

    # Rule 8: fallback. Real domain vocabulary lands here.
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
    "PEDAGOGICAL_MARKERS",
    "LOW_SIGNAL_TOKENS",
    "INSTRUCTIONAL_ARTIFACTS",
    "classify_concept",
]
