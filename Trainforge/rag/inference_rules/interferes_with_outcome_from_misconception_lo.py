"""Rule: derive ``interferes-with-outcome`` edges from Misconception entities
that declare an ``lo_id``.

``misconception.lo_id`` is an optional explicit pointer in the Misconception
schema. When an authoring or enrichment surface populates it, this rule emits
the corresponding typed edge::

    misconception_id --interferes-with-outcome--> lo_id

Federation-by-convention: ``source`` is a misconception ID
(``mc_[0-9a-f]{16}``, per
``schemas/knowledge/misconception.schema.json``); ``target`` is an LO
ID (``TO-NN`` / ``CO-NN``, lowercased per Trainforge normalisation
convention). No new node types are added to the concept-graph schema.

Confidence is ``1.0`` — the reference is explicit.

The field travels through the ``misconceptions=[...]`` keyword argument.
Missing or unpopulated input produces an empty edge list.

Deterministic: output sorted by (source, target); duplicates on the
same (misconception_id, lo_id) pair are collapsed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

RULE_NAME = "interferes_with_outcome_from_misconception_lo"
RULE_VERSION = 1
EDGE_TYPE = "interferes-with-outcome"


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    *,
    misconceptions: List[Dict[str, Any]] | None = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``interferes-with-outcome`` edges when misconceptions declare
    an ``lo_id``.

    Args:
        chunks: Unused; interface parity.
        course: Unused; interface parity.
        concept_graph: Unused; endpoints reference external namespaces.
        misconceptions: Optional list of misconception dicts. Each may
            have ``id`` (``mc_*``) + optional ``lo_id``. When ``None`` or
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
        if not isinstance(mc, dict):
            continue
        mc_id = mc.get("id")
        lo_id = mc.get("lo_id")
        if not mc_id or not lo_id:
            continue
        key = (mc_id, lo_id)
        if key in seen:
            continue
        seen[key] = {
            "source": mc_id,
            "target": lo_id,
            "type": EDGE_TYPE,
            "confidence": 1.0,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": {
                    "misconception_id": mc_id,
                    "lo_id": lo_id,
                },
            },
        }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
