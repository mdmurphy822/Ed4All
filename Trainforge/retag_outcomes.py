"""Wave 76: vocabulary-driven LO retag + parent-outcome rollup.

External KG-quality review of the rdf-shacl-550 archive surfaced four
real coverage gaps where content exists but is mis-tagged:

    co-18 — SHACL Core constraint components
            (sh:minCount / maxCount / datatype / class / pattern / in)
    co-19 — SHACL validation report
            (sh:result / focusNode / severity, "validation report")
    co-22 — Trade-offs across SHACL Core / SHACL-SPARQL / SHACL Rules
    to-07 — Capstone integration (42 chunks already cite co-25..co-29
            but never roll up to the terminal)

This module exposes two pure-data helpers:

* ``retag_chunk_outcomes(chunk, parent_map=None)`` — apply the
  vocabulary retag pass + parent-outcome rollup to a single chunk's
  ``learning_outcome_refs`` in place. Both rules are *additive*: never
  remove an existing ref, only append.
* ``build_parent_map(objectives)`` — build the
  ``component_id -> terminal_id`` map from a loaded ``objectives.json``
  payload (handles both ``component_objectives[]`` and the legacy
  ``chapter_objectives[]`` shape).

The helpers are pure functions to keep them trivially callable from
both ``CourseProcessor._create_chunk`` (emit time) and the retroactive
regen script in ``scripts/wave76_retag_chunks.py``. They are
idempotent — running the retag twice on the same chunk does not
duplicate refs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


# Vocabulary lists are taken verbatim from the words that appear in
# each CO's ``statement`` field, augmented with the constraint /
# property names called out in the canonical vocabulary surveys (e.g.
# the SHACL spec's Core Constraint Components table). Matching is
# substring-style on ``chunk["text"]`` — case-sensitive because the
# SHACL/SHACL-SPARQL/SHACL Rules tokens are proper nouns.
RETAG_VOCABULARIES: Dict[str, List[str]] = {
    "co-18": [
        # SHACL Core constraint component vocabulary.
        "sh:minCount",
        "sh:maxCount",
        "sh:datatype",
        "sh:class",
        "sh:pattern",
        "sh:in",
        "sh:minLength",
        "sh:maxLength",
        "sh:nodeKind",
        "sh:hasValue",
    ],
    "co-19": [
        # SHACL validation report shape.
        "sh:result",
        "sh:resultMessage",
        "sh:resultPath",
        "sh:focusNode",
        "sh:resultSeverity",
        "validation report",
        "Violation",
        "Warning",
        "Info",
        "sh:conforms",
    ],
    "co-22": [
        # Trade-off / comparison vocabulary.
        "SHACL-SPARQL",
        "sh:sparql",
        "SHACL Rules",
        "SHACL Advanced Features",
        "SHACL-AF",
        "vs Core",
        "vs SPARQL",
        "trade-off",
    ],
}


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


def _vocabulary_matches(text: str) -> List[str]:
    """Return the list of CO IDs whose vocabulary matches ``text``."""
    if not text:
        return []
    matched: List[str] = []
    for co_id, terms in RETAG_VOCABULARIES.items():
        for term in terms:
            if term and term in text:
                matched.append(co_id)
                break
    return matched


def retag_chunk_outcomes(
    chunk: Dict[str, Any],
    parent_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Apply vocabulary retag + parent rollup to ``chunk`` in place.

    Both rules are additive — existing refs are never removed. Refs
    are deduplicated case-insensitively, with the first-seen casing
    retained so callers that opt into ``TRAINFORGE_PRESERVE_LO_CASE``
    keep their authoritative casing. Returns the same chunk for
    chaining.
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

    # Part 1: vocabulary-driven retag against chunk text.
    text = chunk.get("text") or ""
    if isinstance(text, str):
        for co_id in _vocabulary_matches(text):
            _add(co_id)

    # Part 2: parent-rollup. For every co-NN in the (now-extended) ref
    # list, also add its terminal parent.
    if parent_map:
        # Snapshot the keys we'll iterate over so that adding parents
        # while looping doesn't re-trigger lookups on already-added
        # parents (parents shouldn't appear in parent_map anyway, but
        # be defensive).
        for ref in list(out):
            parent = parent_map.get(ref.lower())
            if parent:
                _add(parent)

    chunk["learning_outcome_refs"] = out
    return chunk
