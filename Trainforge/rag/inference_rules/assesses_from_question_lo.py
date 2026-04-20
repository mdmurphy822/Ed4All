"""Rule: derive ``assesses`` edges from Question→LO references.

Assessment questions in Trainforge carry ``objective_id`` declaring
which LO the question is meant to evaluate. This rule materializes that
pointer as an explicit typed edge:

    question_id --assesses--> objective_id

Federation-by-convention: ``source`` is a question ID; ``target`` is an
LO ID. No new node types are added to the concept graph.

Confidence is ``1.0`` — the reference is explicit.

**Signal-availability caveat.** Questions are not currently threaded
into ``build_semantic_graph``'s main call chain — the field travels
through ``**kwargs`` as ``questions=[...]``. When absent (current
production state), the rule emits ``[]`` gracefully. A future wave will
wire upstream pipelines to populate ``objective_id`` on questions and
thread them through the orchestrator.

Deterministic: output sorted by (source, target); duplicates on the
same (question_id, objective_id) pair are collapsed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

RULE_NAME = "assesses_from_question_lo"
# Wave 11 (Worker cc): bumped from 1 -> 2 to expose the optional
# source_references[] emit shape on AssessesEvidence.
RULE_VERSION = 2
EDGE_TYPE = "assesses"

# Wave 11: opt-in flag gates the evidence-arm source_references[] emission.
SOURCE_PROVENANCE = os.getenv("TRAINFORGE_SOURCE_PROVENANCE", "").lower() == "true"


def _build_chunk_index(chunks: List[Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    """Return a {chunk_id: chunk} lookup map."""
    if not chunks:
        return {}
    idx: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        cid = chunk.get("id")
        if not cid:
            continue
        idx[cid] = chunk
    return idx


def _lookup_source_references(
    chunk_id: str, chunk_index: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return deep-copied source_references[] for the given chunk_id."""
    chunk = chunk_index.get(chunk_id)
    if not isinstance(chunk, dict):
        return []
    source = chunk.get("source")
    if not isinstance(source, dict):
        return []
    refs = source.get("source_references")
    if not isinstance(refs, list):
        return []
    return [dict(r) for r in refs if isinstance(r, dict)]


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    *,
    questions: List[Dict[str, Any]] | None = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``assesses`` edges when questions declare an objective_id.

    Args:
        chunks: Unused; interface parity.
        course: Unused; interface parity.
        concept_graph: Unused; endpoints reference external namespaces.
        questions: Optional list of question dicts. Each may have ``id``,
            ``objective_id``, optional ``source_chunk_id``. When ``None``
            or empty, emits no edges (current pipeline state — awaiting
            upstream wiring).

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del course, concept_graph  # unused; interface parity

    if not questions:
        return []

    # Wave 11: build chunk lookup so the flag-on path can resolve
    # source_chunk_id -> chunk.source.source_references[].
    chunk_index = _build_chunk_index(chunks) if SOURCE_PROVENANCE else {}

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for q in questions:
        q_id = q.get("id")
        lo_id = q.get("objective_id")
        if not q_id or not lo_id:
            continue
        key = (q_id, lo_id)
        if key in seen:
            continue
        evidence: Dict[str, Any] = {
            "question_id": q_id,
            "objective_id": lo_id,
        }
        src_chunk = q.get("source_chunk_id")
        if src_chunk:
            evidence["source_chunk_id"] = src_chunk
            # Wave 11: flag-gated source_references emit. Only when the
            # question points at a chunk that actually exists and carries
            # refs. Legacy questions without source_chunk_id emit no
            # source_references (absence = unknown).
            if SOURCE_PROVENANCE:
                refs = _lookup_source_references(src_chunk, chunk_index)
                if refs:
                    evidence["source_references"] = refs
        seen[key] = {
            "source": q_id,
            "target": lo_id,
            "type": EDGE_TYPE,
            "confidence": 1.0,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": evidence,
            },
        }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
