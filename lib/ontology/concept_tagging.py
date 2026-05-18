"""Instance-free concept-tag extraction.

This module owns the canonical concept-tag extraction logic, lifted
out of ``Trainforge/process_course.py::CourseProcessor._extract_concept_tags``
so the same logic can run on BOTH chunk-emit surfaces:

  * the IMSCC chunk path (``CourseProcessor`` delegates here);
  * the canonical DART chunkset path (``MCP/tools/pipeline_tools.py::
    _run_dart_chunking``'s ``_create_chunk`` callback), which previously
    emitted ``concept_tags: []`` because the extraction logic was
    coupled to ``CourseProcessor`` instance state.

The empty DART ``concept_tags`` substrate starved the downstream
concept-graph + CURIE machinery. ``extract_concept_tags`` here has no
``CourseProcessor`` dependency: ``domain_concept_seeds`` is threaded as
a parameter (empty list when the caller has no objectives file), the
filter pipeline constants are module-level, and ``normalize_tag`` is
already an instance-free function.

Pipeline shape (Wave 76, preserved verbatim from the original method):
  normalize_tag (HTML-decoded, slugified)
  -> strip_lo_ref_suffix (drop ``-co-NN`` / ``-to-NN`` suffix; Wave 130d)
  -> canonicalize_alias (rdfxml -> rdf-xml, ttl -> turtle, ...)
  -> classify_concept (rule-based class assignment)
  -> is_droppable_class (reject pedagogy / assessment-options /
     instructional-artifacts / LO-leak / low-signal)
  -> tag list (dedup-aware, singular-preferred)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Pattern, Sequence, Tuple

from lib.ontology.slugs import strip_lo_ref_suffix

# Common educational concept patterns for text-based extraction.
# CONCEPT_PATTERNS covers pedagogy terms only; per-course domain
# vocabulary arrives via the ``domain_concept_seeds`` parameter.
CONCEPT_PATTERNS: Dict[str, List[str]] = {
    "learning-theory": ["learning theory", "learning theories"],
    "behaviorism": ["behaviorism", "behaviorist"],
    "cognitivism": ["cognitivism", "cognitivist", "information processing"],
    "constructivism": ["constructivism", "constructivist"],
    "connectivism": ["connectivism", "connectivist", "networked learning"],
    "instructional-design": ["instructional design"],
    "addie": ["addie model", "addie"],
    "backward-design": ["backward design", "understanding by design"],
    "cognitive-load": [
        "cognitive load", "intrinsic load", "extraneous load", "germane load",
    ],
    "multimedia-learning": [
        "multimedia learning", "multimedia principle", "mayer",
    ],
    "blooms-taxonomy": [
        "bloom's taxonomy", "blooms taxonomy", "bloom's",
        "higher-order thinking",
    ],
    "assessment": [
        "assessment", "formative assessment", "summative assessment",
    ],
    "rubric": ["rubric"],
    "accessibility": ["accessibility", "accessible", "wcag"],
    "udl": ["universal design for learning", "udl"],
    "oscqr": ["oscqr", "course quality"],
    "blended-learning": ["blended learning", "hybrid learning"],
    "online-learning": [
        "online learning", "online instruction", "distance learning",
    ],
    "synchronous": ["synchronous"],
    "asynchronous": ["asynchronous"],
    "scaffolding": ["scaffolding", "zone of proximal development"],
    "engagement": ["student engagement", "learner engagement"],
    "community-of-inquiry": ["community of inquiry", "coi framework"],
    "alignment": [
        "constructive alignment", "learning objectives", "learning outcomes",
    ],
    "feedback": ["feedback", "timely feedback"],
}

# Pattern for course/terminal/learning objective codes (CO-01, TO-08,
# LO-003, etc.).
OBJECTIVE_CODE_RE: Pattern[str] = re.compile(r'^[a-z]{2}-\d{2,3}$')
# Week prefix pattern (w01-, w02-) used by Courseforge JSON-LD but
# absent in course.json.
WEEK_PREFIX_RE: Pattern[str] = re.compile(r'^w\d{2}-', re.IGNORECASE)

# Non-concept tags to filter out (generic metadata, not knowledge
# concepts).
NON_CONCEPT_TAGS: set = {
    "estimated-time", "time", "minutes", "hours",
    # Bloom verbs (pedagogical intent, not domain concepts)
    "define", "list", "recall", "identify", "name", "state",
    "explain", "describe", "summarize", "interpret", "paraphrase",
    "apply", "demonstrate", "implement", "solve", "use", "execute",
    "analyze", "differentiate", "examine", "compare", "contrast", "organize",
    "evaluate", "assess", "critique", "judge", "justify", "argue",
    "create", "design", "develop", "construct", "produce", "formulate",
    # Course logistics
    "initial-post", "replies", "due", "guidelines",
    "correct", "incorrect", "submit", "deadline", "grading",
    "readings", "resources", "learning-objectives",
}

# Cap on emitted tags per chunk (display-layer LibV2 ceiling).
MAX_CONCEPT_TAGS = 20


def extract_concept_tags(
    text: str,
    item: Dict[str, Any],
    domain_concept_seeds: Sequence[Tuple[str, List[Pattern[str]]]] = (),
) -> List[str]:
    """Extract domain-concept tags from chunk text + parser key concepts.

    Instance-free reimplementation of the logic that previously lived on
    ``CourseProcessor._extract_concept_tags``. Threads the per-course
    ``domain_concept_seeds`` (compiled ``(canonical, [regex])`` pairs;
    see ``Trainforge.process_course.compile_domain_concept_seeds``) as a
    parameter so callers without a ``CourseProcessor`` instance — notably
    the DART chunkset ``_create_chunk`` callback — can pass ``()``.

    Args:
        text: The chunk's plain text.
        item: The parsed-item dict; ``item["key_concepts"]`` (bold terms
            / definitions from the HTML parser) is consumed when present.
        domain_concept_seeds: Optional per-course ``(canonical_tag,
            [word-boundary regex])`` seeds. Empty by default.

    Returns:
        A deduped, filter-passing list of concept-tag slugs, capped at
        ``MAX_CONCEPT_TAGS``.
    """
    # Deferred imports mirror the original method — keeps the
    # concept-classifier module out of the import graph for callers that
    # never tag.
    from lib.ontology.concept_classifier import (
        canonicalize_alias,
        classify_concept,
        is_droppable_class,
        singular_form,
    )
    # ``normalize_tag`` is a module-level function in process_course;
    # imported here to avoid a circular import at module load time.
    from Trainforge.process_course import normalize_tag

    tags: List[str] = []

    def _try_add(tag: str) -> bool:
        """Filter + dedup + add. Returns True iff appended."""
        if not tag:
            return False
        tag = strip_lo_ref_suffix(tag)
        if not tag:
            return False
        tag = canonicalize_alias(tag)
        if tag in tags:
            return False
        # Wave 76: drop classes the extractor should not emit
        # (pedagogical scaffolding, assessment options, LO leaks,
        # instructional artifacts, low-signal stopwords + fragments).
        klass = classify_concept(tag)
        if is_droppable_class(klass):
            return False
        # Plural-collapse: if the singular form is already emitted,
        # treat this tag as a duplicate.
        sing = singular_form(tag)
        if sing != tag and sing in tags:
            return False
        tags.append(tag)
        return True

    # Key concepts from HTML parser (bold terms, definitions).
    for concept in item.get("key_concepts", []):
        tag = normalize_tag(concept)
        if not tag or len(tag) < 3:
            continue
        # Collapse known WCAG SC tag-form drift onto the canonical tag
        # before any filter or dedupe (canonicalization).
        from Trainforge.rag.wcag_canonical_names import canonicalize_sc_tag
        tag = canonicalize_sc_tag(tag)
        # Skip objective codes (co-01, to-01, w01-co-01) and non-concept
        # tags.
        if (OBJECTIVE_CODE_RE.match(tag)
                or WEEK_PREFIX_RE.match(tag)
                or tag in NON_CONCEPT_TAGS):
            continue
        _try_add(tag)

    # Text-based concept detection (pedagogy-only patterns).
    text_lower = (text or "").lower()
    for tag, patterns in CONCEPT_PATTERNS.items():
        if any(p in text_lower for p in patterns):
            _try_add(tag)

    # Per-course domain concept seeds. Pedagogy filter still applies
    # below since seeds are authored per course; a well-formed seed
    # list won't collide with NON_CONCEPT_TAGS, but we defend anyway.
    for canonical, patterns in domain_concept_seeds:
        if (OBJECTIVE_CODE_RE.match(canonical)
                or WEEK_PREFIX_RE.match(canonical)
                or canonical in NON_CONCEPT_TAGS):
            continue
        if any(p.search(text or "") for p in patterns):
            _try_add(canonical)

    # Wave 82 (Phase C): W3C tech-anchor seeding. Behaviour-flagged so
    # legacy corpora don't shift their tag distributions on rebuild.
    if os.getenv("TRAINFORGE_SEED_TECH_CONCEPTS", "").lower() == "true":
        from lib.ontology.tech_anchors import detect_anchors
        for slug in detect_anchors(text or ""):
            _try_add(slug)

    return tags[:MAX_CONCEPT_TAGS]


__all__ = [
    "extract_concept_tags",
    "CONCEPT_PATTERNS",
    "OBJECTIVE_CODE_RE",
    "WEEK_PREFIX_RE",
    "NON_CONCEPT_TAGS",
    "MAX_CONCEPT_TAGS",
]
