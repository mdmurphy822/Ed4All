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

from typing import Any, Dict, List, Tuple

RULE_NAME = "assesses_from_question_lo"
RULE_VERSION = 1
EDGE_TYPE = "assesses"


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
    del chunks, course, concept_graph  # unused; interface parity

    if not questions:
        return []

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
