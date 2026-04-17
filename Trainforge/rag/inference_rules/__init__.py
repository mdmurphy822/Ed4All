"""Typed-edge inference rules for the semantic concept graph.

Each rule module exposes:

    RULE_NAME: str
    RULE_VERSION: int
    EDGE_TYPE: str  # one of "prerequisite", "is-a", "related-to"

    def infer(chunks, course, concept_graph, **kwargs) -> list[dict]:
        ...

Rules return a list of edge dicts with shape::

    {
      "source": "<node_id>",
      "target": "<node_id>",
      "type": "<edge_type>",
      "confidence": <float in [0,1]>,
      "provenance": {
        "rule": RULE_NAME,
        "rule_version": RULE_VERSION,
        "evidence": {...},
      },
    }

The orchestrator (``typed_edge_inference.py``) consumes these lists, applies
the precedence policy ``is-a`` > ``prerequisite`` > ``related-to`` on
``(source, target)`` collisions, and writes the final artifact.
"""

from .is_a_from_key_terms import infer as infer_is_a
from .prerequisite_from_lo_order import infer as infer_prerequisite
from .related_from_cooccurrence import infer as infer_related

__all__ = [
    "infer_is_a",
    "infer_prerequisite",
    "infer_related",
]
