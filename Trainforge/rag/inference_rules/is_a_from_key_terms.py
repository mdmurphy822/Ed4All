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

    Lowercase, strip punctuation, collapse whitespace to a single hyphen. No
    stemming — the graph already stores canonical tag slugs, so we just need
    to match them.
    """
    text = canonicalize_sc_references(text or "")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _candidate_parent_ids(phrase: str, node_ids: set) -> List[str]:
    """Return node ids from ``node_ids`` that appear in ``phrase``.

    Checked in descending length order so multi-word nodes ("keyboard-trap")
    win over their single-word substrings ("trap").
    """
    phrase_slug = _slugify(phrase)
    if not phrase_slug:
        return []
    # Try the full phrase slug first.
    hits: List[str] = []
    if phrase_slug in node_ids and phrase_slug not in _STOPWORDS:
        hits.append(phrase_slug)

    # Then scan for node ids as substrings of the slug. Longer ids first so
    # we don't misattribute "wcag" when "wcag-compliance" is the real hit.
    for nid in sorted(node_ids, key=lambda n: -len(n)):
        if nid in _STOPWORDS:
            continue
        if nid == phrase_slug:
            continue
        # Substring match at hyphen/word boundary.
        pattern = rf"(?:^|-){re.escape(nid)}(?:-|$)"
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
    node_ids = {n["id"] for n in concept_graph.get("nodes", [])}
    if not node_ids:
        return []

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for chunk in chunks:
        for kt in chunk.get("key_terms", []) or []:
            term = (kt.get("term") or "").strip()
            definition = (kt.get("definition") or "").strip()
            if not term or not definition:
                continue

            child_id = _slugify(term)
            if not child_id or child_id not in node_ids:
                # Child isn't a graph node — nothing to attach.
                continue

            matched_parents: List[Tuple[str, str]] = []
            for pattern in _IS_A_PATTERNS:
                m = pattern.search(definition)
                if not m:
                    continue
                phrase = m.group(1)
                parents = _candidate_parent_ids(phrase, node_ids)
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
