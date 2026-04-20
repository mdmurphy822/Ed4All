"""Rule: derive ``is-a`` edges from key_term definitions.

Looks for phrasings in ``chunks[].key_terms[].definition`` that introduce a
parent category — for example:

    "An ARIA role is a type of accessibility attribute."
    "A keyboard trap is a form of input barrier."
    "Semantic HTML is an kind of markup."

When both the child term and the parent term resolve to nodes that exist in
the already-computed co-occurrence concept graph, the rule emits an
``<child> --is-a--> <parent>`` edge.

Deterministic: the rule scans chunks in the order they appear; outputs are
deduplicated and sorted by (source, target) before return. No randomness, no
LLM, no network.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from lib.ontology.slugs import canonical_slug
from Trainforge.rag.wcag_canonical_names import canonicalize_sc_references

RULE_NAME = "is_a_from_key_terms"
RULE_VERSION = 1
EDGE_TYPE = "is-a"

# Patterns that announce a parent-category:
#   "is a ..."        "is an ..."
#   "is a type of ..." "is a form of ..." "is a kind of ..."
#   "is one of the ..."
#   "refers to a ..." "refers to the ..."
_IS_A_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bis\s+(?:a|an)\s+(?:type|form|kind|class|category|variant|subset)\s+of\s+(?:an?\s+|the\s+)?([^.,;:\n]+)", re.IGNORECASE),
    re.compile(r"\bis\s+one\s+of\s+the\s+([^.,;:\n]+)", re.IGNORECASE),
    re.compile(r"\brefers\s+to\s+(?:an?\s+|the\s+)([^.,;:\n]+)", re.IGNORECASE),
    # Last-resort copular "is a/an X" — matches broader but only fires when
    # the captured phrase contains a known node id, so false positives are
    # filtered by the node-existence check downstream.
    re.compile(r"\bis\s+(?:a|an)\s+([^.,;:\n]+)", re.IGNORECASE),
)

# Words that must never stand as a "parent" (dead-obvious false positives).
_STOPWORDS = {
    "way", "method", "means", "tool", "thing", "example", "result",
    "process", "case", "situation", "step", "part",
}


def _slugify(text: str) -> str:
    """Normalize a term to a concept-graph-style id.

    Two site-specific preprocessing steps sit on top of the shared
    ``canonical_slug`` (REC-ID-03, Wave 4 Worker Q):

    1. ``canonicalize_sc_references`` rewrites WCAG Success Criterion variants
       (e.g. ``"1.3.1"`` ↔ ``"SC 1.3.1"``) so that downstream substring matches
       against graph node ids line up.
    2. Non-alnum/whitespace/hyphen characters are replaced with SPACE (rather
       than deleted like ``canonical_slug`` does). This preserves word
       separation — ``"a.b"`` stays slugged as ``"a-b"`` rather than fusing to
       ``"ab"`` — so phrase substrings match multi-word graph node ids.

    After those two steps the string contains only alnum, whitespace, and
    hyphens; ``canonical_slug`` then does the lowercase + kebab-case + edge
    strip. A final multi-hyphen collapse matches the historical behavior of
    this site (Courseforge's ``_slugify`` does not collapse interior runs of
    hyphens; this rule does).
    """
    text = canonicalize_sc_references(text or "")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    slug = canonical_slug(text)
    return re.sub(r"-+", "-", slug).strip("-")


def _candidate_parent_ids(
    phrase: str,
    node_ids: set,
    *,
    course_id: str | None = None,
) -> List[str]:
    """Return node ids from ``node_ids`` that appear in ``phrase``.

    Checked in descending length order so multi-word nodes ("keyboard-trap")
    win over their single-word substrings ("trap").

    REC-ID-02 (Wave 4, Worker O): when ``TRAINFORGE_SCOPE_CONCEPT_IDS=true``
    is in effect, graph node IDs are composite ``{course_id}:{slug}``. The
    phrase-to-slug output must be scoped through ``_make_concept_id`` before
    lookup so the flag-on path actually finds matches. The substring scan
    continues to work against node IDs as-is — ``re.escape(nid)`` already
    handles the ``:`` character harmlessly, and substring matches against
    the *slug* portion still fire because the slug is a suffix of the
    scoped ID.
    """
    from Trainforge.rag.typed_edge_inference import _make_concept_id

    phrase_slug = _slugify(phrase)
    if not phrase_slug:
        return []
    hits: List[str] = []

    # Try the full phrase as a scoped node ID first.
    scoped_phrase_id = _make_concept_id(phrase_slug, course_id)
    if scoped_phrase_id in node_ids and phrase_slug not in _STOPWORDS:
        hits.append(scoped_phrase_id)

    # Then scan for node ids whose slug part appears as a substring of the
    # phrase slug. Longer ids first so we don't misattribute "wcag" when
    # "wcag-compliance" is the real hit. We strip the ``{course_id}:``
    # prefix (when present) before the substring check so scope-off and
    # scope-on produce equivalent matches.
    for nid in sorted(node_ids, key=lambda n: -len(n)):
        nid_slug = nid.split(":", 1)[1] if ":" in nid else nid
        if nid_slug in _STOPWORDS:
            continue
        if nid == scoped_phrase_id:
            continue
        # Substring match at hyphen/word boundary against the slug portion.
        pattern = rf"(?:^|-){re.escape(nid_slug)}(?:-|$)"
        if re.search(pattern, phrase_slug):
            if nid not in hits:
                hits.append(nid)
    return hits


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``is-a`` edges parsed from ``key_terms[].definition`` fields.

    Args:
        chunks: Pipeline chunk dicts. Each may have ``key_terms`` as a list of
            ``{"term", "definition"}`` dicts.
        course: ``course.json`` dict (unused by this rule; kept for interface
            parity).
        concept_graph: The co-occurrence graph dict, used to resolve node ids.

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del course  # unused; interface parity
    # REC-ID-02 (Wave 4, Worker O): scope child slugs through the
    # course-scoping helper before node-id lookup. When flag is off this is
    # a no-op identity. When flag is on, node IDs in ``concept_graph`` are
    # composite ``{course_id}:{slug}`` and the per-chunk ``source.course_id``
    # supplies the scope.
    from Trainforge.rag.typed_edge_inference import _make_concept_id

    node_ids = {n["id"] for n in concept_graph.get("nodes", [])}
    if not node_ids:
        return []

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for chunk in chunks:
        course_id = (chunk.get("source") or {}).get("course_id")
        for kt in chunk.get("key_terms", []) or []:
            term = (kt.get("term") or "").strip()
            definition = (kt.get("definition") or "").strip()
            if not term or not definition:
                continue

            child_slug = _slugify(term)
            if not child_slug:
                continue
            child_id = _make_concept_id(child_slug, course_id)
            if child_id not in node_ids:
                # Child isn't a graph node — nothing to attach.
                continue

            matched_parents: List[Tuple[str, str]] = []
            for pattern in _IS_A_PATTERNS:
                m = pattern.search(definition)
                if not m:
                    continue
                phrase = m.group(1)
                parents = _candidate_parent_ids(
                    phrase, node_ids, course_id=course_id
                )
                for pid in parents:
                    if pid == child_id:
                        continue
                    matched_parents.append((pid, pattern.pattern))
                if matched_parents:
                    # First pattern wins per definition — keeps rule behavior
                    # predictable when "is a type of X" and the fallback
                    # "is a X" both fire.
                    break

            for parent_id, pattern_str in matched_parents:
                key = (child_id, parent_id)
                if key in seen:
                    continue
                seen[key] = {
                    "source": child_id,
                    "target": parent_id,
                    "type": EDGE_TYPE,
                    "confidence": 0.8,
                    "provenance": {
                        "rule": RULE_NAME,
                        "rule_version": RULE_VERSION,
                        "evidence": {
                            "chunk_id": chunk.get("id"),
                            "term": term,
                            "definition_excerpt": definition[:200],
                            "pattern": pattern_str,
                        },
                    },
                }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
