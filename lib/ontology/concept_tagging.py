"""Instance-free concept-tag extraction.

This module owns the canonical concept-tag extraction logic, lifted
out of ``Trainforge/pipeline/process_course.py::CourseProcessor._extract_concept_tags``
so the same logic can run on BOTH chunk-emit surfaces:

  * the IMSCC chunk path (``CourseProcessor`` delegates here);
  * the canonical SemantiK chunkset path (the chunking phase's
    ``_create_chunk`` callback in ``MCP/tools/pipeline_tools.py``), which
    previously emitted ``concept_tags: []`` because the extraction logic was
    coupled to ``CourseProcessor`` instance state.

The empty ``concept_tags`` substrate starved the downstream
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


def _is_fragment_superstring(tag: str, anchor: str) -> bool:
    """True when ``tag`` is a clear sentence-fragment SUPERSTRING of ``anchor``.

    Used by the tech-anchor seeding path: once a flagship anchor slug
    (``ragas``, ``nim``, ``langchain``) is admitted, any later tag that
    merely *embeds* that anchor's word(s) inside a longer FRAGMENT-shaped
    slug (``ragas-short-for-rag``, ``nim-stands-for-nvidia-inference``) is
    a degraded paraphrase of the clean anchor and should be dropped in its
    favour.

    Conservative + asymmetric. A tag is dropped only when BOTH hold:

    * it contains the anchor as a whole hyphen-delimited word (so
      ``reranking`` does not match a chunk's unrelated ``ranking`` tag, and
      ``nim`` does not swallow ``minimum``); AND
    * it is itself fragment-shaped — MORE than four hyphen words, which is
      the signature of a sentence fragment harvested as a slug
      (``ragas-short-for-rag`` = 4 words is borderline; ``ragas-is-short-
      for-rag`` = 5 words rejects). Real multi-word domain concepts
      (``langchain-expression-language-lcel`` = 4 words, ``vector-store``,
      ``knowledge-base``, ``nvidia-nim-microservices`` = 3 words) stay
      under the ceiling and survive.

    ``is_fragment_phrase`` from ``lexical_concept_seeds`` is deliberately
    NOT reused here: it rejects on a single function word and a 5-token
    ceiling, which false-flags legitimate compound concepts that legitimately
    embed an anchor word. This helper trades that recall for precision so the
    dedup never drops a real concept.
    """
    if not tag or not anchor or tag == anchor:
        return False
    words = tag.split("-")
    if anchor not in words:
        return False
    # >4 hyphen-words is the fragment signature; clean compounds
    # (3-4 words) are kept.
    return len(words) > 4


def extract_concept_tags(
    text: str,
    item: Dict[str, Any],
    domain_concept_seeds: Sequence[Tuple[str, List[Pattern[str]]]] = (),
) -> List[str]:
    """Extract domain-concept tags from chunk text + parser key concepts.

    Instance-free reimplementation of the logic that previously lived on
    ``CourseProcessor._extract_concept_tags``. Threads the per-course
    ``domain_concept_seeds`` (compiled ``(canonical, [regex])`` pairs;
    see ``Trainforge.pipeline.process_course.compile_domain_concept_seeds``) as a
    parameter so callers without a ``CourseProcessor`` instance — notably
    the SemantiK chunkset ``_create_chunk`` callback — can pass ``()``.

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
        is_scaffolding_noise,
        singular_form,
    )
    # ``normalize_tag`` is a module-level function in process_course;
    # imported here to avoid a circular import at module load time.
    from Trainforge.pipeline.process_course import normalize_tag

    # Change B: default-OFF scaffolding-noise prune. When ON, tags that
    # ``is_scaffolding_noise`` flags are treated as droppable in
    # ``_try_add`` (pros/cons/creating/advanced/...).
    prune_scaffolding = (
        os.getenv("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS", "").lower()
        == "true"
    )

    tags: List[str] = []
    # Slugs admitted via the tech-anchor seeding path (when
    # ``TRAINFORGE_SEED_TECH_CONCEPTS`` is on). Drives the fragment-
    # superstring dedup in ``_try_add``: a later untrusted tag that is a
    # clear fragment superstring of an already-admitted anchor is dropped
    # in favour of the clean anchor slug.
    anchor_slugs: List[str] = []

    def _try_add(tag: str, trusted: bool = False) -> bool:
        """Filter + dedup + add. Returns True iff appended.

        ``trusted`` marks the tag as coming from an LLM-authored
        per-course ``domain_concept_seeds`` entry. Trusted tags BYPASS
        the Wave-76 ``is_droppable_class`` rejection — those filters are
        RDF/SHACL/pedagogy-tuned and silently drop legitimate general-
        textbook vocabulary (a concept titled "Section 1 …", a
        psychology concept "applications", a law concept "review", a
        concept "On Liberty"). Seeds were deliberately extracted AS
        domain concepts by the Stage-3 textbook-synthesis LLM, so they
        should not be re-filtered as if they were untrusted heuristic
        extraction. Everything else ``_try_add`` does — ``strip_lo_ref_
        suffix``, ``canonicalize_alias``, dedup, plural-collapse — still
        applies to trusted tags.

        Untrusted tags (``trusted=False``, the default — HTML-parser
        ``key_concepts``, ``CONCEPT_PATTERNS`` text matches, and the
        ``TRAINFORGE_SEED_TECH_CONCEPTS`` tech-anchor path) keep the
        full Wave-76 droppable-class filter.
        """
        if not tag:
            return False
        tag = strip_lo_ref_suffix(tag)
        if not tag:
            return False
        tag = canonicalize_alias(tag)
        if tag in tags:
            return False
        # Fragment-superstring dedup: a non-anchor tag that is a clear
        # sentence-fragment superstring of an already-admitted tech anchor
        # (``ragas-short-for-rag`` when ``ragas`` is present) is dropped in
        # favour of the clean anchor slug. Anchors are seeded FIRST, so the
        # anchor is already in ``anchor_slugs`` by the time the fragment
        # surfaces from the key-concept / pattern / seed passes.
        if not trusted:
            if any(_is_fragment_superstring(tag, a) for a in anchor_slugs):
                return False
        # Wave 76: drop classes the extractor should not emit
        # (pedagogical scaffolding, assessment options, LO leaks,
        # instructional artifacts, low-signal stopwords + fragments).
        # Skipped for trusted (LLM-authored seed) tags — see docstring.
        if not trusted:
            klass = classify_concept(tag)
            if is_droppable_class(klass):
                return False
            # Change B: drop domain-agnostic scaffolding noise behind the
            # default-OFF prune flag (untrusted tags only — trusted seeds
            # were deliberately LLM-authored as domain vocabulary).
            if prune_scaffolding and is_scaffolding_noise(tag):
                return False
        # Plural-collapse: if the singular form is already emitted,
        # treat this tag as a duplicate.
        sing = singular_form(tag)
        if sing != tag and sing in tags:
            return False
        tags.append(tag)
        return True

    # Wave 82 (Phase C) + tech-anchor under-seeding fix: W3C / RAG
    # flagship tech-anchor seeding. Behaviour-flagged so legacy corpora
    # don't shift their tag distributions on rebuild (flag OFF → this
    # block is skipped entirely and the remaining passes run in their
    # original order, so default-off output is byte-identical legacy).
    #
    # Promoted to run FIRST and as TRUSTED tags: flagship anchors
    # (``langgraph``, ``react``, ``ragas``, ``faiss``, ``nim``,
    # ``guardrails``, ``reranking``, ``lcel``, ``langchain``, ``rag``,
    # …) are the headline concepts a chunk teaches. Seeding them before
    # the key-concept / pattern / domain-seed passes guarantees they're
    # counted before the ``MAX_CONCEPT_TAGS`` truncation, and marking
    # them trusted means the Wave-76 ``is_droppable_class`` filter never
    # silently strips a real standalone anchor. ``detect_anchors`` is a
    # bounded, word-boundaried, case-sensitive-acronym matcher, so the
    # surface form really IS in the chunk text when an anchor lands.
    if os.getenv("TRAINFORGE_SEED_TECH_CONCEPTS", "").lower() == "true":
        from lib.ontology.tech_anchors import detect_anchors
        # Sort for deterministic emit order across runs (``detect_anchors``
        # returns a set).
        for slug in sorted(detect_anchors(text or "")):
            if _try_add(slug, trusted=True):
                # Canonicalised form is what landed in ``tags``; track it
                # so the fragment-superstring dedup matches the same slug.
                anchor_slugs.append(canonicalize_alias(strip_lo_ref_suffix(slug)))

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

    # Per-course domain concept seeds. These are LLM-authored per-course
    # vocabulary — they were deliberately extracted AS domain concepts
    # by the Stage-3 textbook-synthesis pass, so they are TRUSTED and
    # bypass the ``is_droppable_class`` drop in ``_try_add`` (those
    # filters are RDF/SHACL/pedagogy-tuned and silently strip legitimate
    # general-textbook vocabulary). The pre-``_try_add`` guards below
    # (``OBJECTIVE_CODE_RE`` / ``WEEK_PREFIX_RE`` / ``NON_CONCEPT_TAGS``)
    # are KEPT: an objective code, week prefix, or known non-concept is
    # rejected even from a seed list, since those are never real domain
    # vocabulary regardless of authorship.
    for canonical, patterns in domain_concept_seeds:
        if (OBJECTIVE_CODE_RE.match(canonical)
                or WEEK_PREFIX_RE.match(canonical)
                or canonical in NON_CONCEPT_TAGS):
            continue
        if any(p.search(text or "") for p in patterns):
            _try_add(canonical, trusted=True)

    return tags[:MAX_CONCEPT_TAGS]


__all__ = [
    "extract_concept_tags",
    "CONCEPT_PATTERNS",
    "OBJECTIVE_CODE_RE",
    "WEEK_PREFIX_RE",
    "NON_CONCEPT_TAGS",
    "MAX_CONCEPT_TAGS",
]
