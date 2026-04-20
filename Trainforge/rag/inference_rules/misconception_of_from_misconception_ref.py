"""Rule: derive ``misconception-of`` edges from Misconception entities.

Worker R (Wave 4) added a first-class Misconception schema with
optional ``concept_id`` + ``lo_id`` fields:
``schemas/knowledge/misconception.schema.json``. When upstream data
populates ``misconception.concept_id``, this rule emits the explicit
typed edge:

    misconception_id --misconception-of--> concept_id

Federation-by-convention: ``source`` is a misconception ID
(``mc_[0-9a-f]{16}``, per schema); ``target`` is a concept node ID (flat
slug or scoped ``{course_id}:{slug}``). No new node types are added.

Confidence is ``1.0`` — the reference is explicit.

**Signal-availability caveat.** Misconceptions are not currently threaded
into ``build_semantic_graph``'s main call chain — the field travels
through ``**kwargs`` as ``misconceptions=[...]``. When absent (current
production state), the rule emits ``[]`` gracefully. A future wave will
wire upstream pipelines to populate ``concept_id`` on misconceptions and
thread them through the orchestrator.

Deterministic: output sorted by (source, target); duplicates on the
same (misconception_id, concept_id) pair are collapsed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

RULE_NAME = "misconception_of_from_misconception_ref"
RULE_VERSION = 1
EDGE_TYPE = "misconception-of"


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    *,
    misconceptions: List[Dict[str, Any]] | None = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``misconception-of`` edges when misconceptions declare a concept_id.

    Args:
        chunks: Unused; interface parity.
        course: Unused; interface parity.
        concept_graph: Unused; endpoints reference external namespaces.
        misconceptions: Optional list of misconception dicts. Each may have
            ``id`` (``mc_*``) + optional ``concept_id``. When ``None`` or
            empty, emits no edges (current pipeline state — awaiting
            upstream wiring).

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del chunks, course, concept_graph  # unused; interface parity

    if not misconceptions:
        return []

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for mc in misconceptions:
        mc_id = mc.get("id")
        concept_id = mc.get("concept_id")
        if not mc_id or not concept_id:
            continue
        key = (mc_id, concept_id)
        if key in seen:
            continue
        seen[key] = {
            "source": mc_id,
            "target": concept_id,
            "type": EDGE_TYPE,
            "confidence": 1.0,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": {
                    "misconception_id": mc_id,
                    "concept_id": concept_id,
                },
            },
        }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
