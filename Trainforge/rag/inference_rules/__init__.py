"""Typed-edge inference rules for the semantic concept graph.

Each rule module exposes:

    RULE_NAME: str
    RULE_VERSION: int
    EDGE_TYPE: str  # one of the values in concept_graph_semantic.schema.json
                    # edge.type enum: "prerequisite", "is-a", "related-to",
                    # "assesses", "exemplifies", "misconception-of",
                    # "derived-from-objective", "defined-by"

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
the precedence policy (``is-a`` > tier-2 edges > ``related-to``) on
``(source, target)`` collisions, and writes the final artifact.

Federation-by-convention (REC-LNK-04, Worker U Wave 5.2): edges may cross
node-type namespaces. ``source`` / ``target`` are not restricted to
concept-graph node IDs — they may also be LO IDs (``TO-NN``/``CO-NN``),
chunk IDs, misconception IDs (``mc_*``), or question IDs. Consumers
resolve endpoints by ID-namespace prefix; no new node types are added
to the concept-graph schema.
"""

from .assesses_from_question_lo import infer as infer_assesses
from .defined_by_from_first_mention import infer as infer_defined_by
from .derived_from_lo_ref import infer as infer_derived_from_objective
from .exemplifies_from_example_chunks import infer as infer_exemplifies
from .is_a_from_key_terms import infer as infer_is_a
from .misconception_of_from_misconception_ref import infer as infer_misconception_of
from .prerequisite_from_lo_order import infer as infer_prerequisite
from .related_from_cooccurrence import infer as infer_related
from .targets_concept_from_lo import infer as infer_targets_concept

__all__ = [
    "infer_assesses",
    "infer_defined_by",
    "infer_derived_from_objective",
    "infer_exemplifies",
    "infer_is_a",
    "infer_misconception_of",
    "infer_prerequisite",
    "infer_related",
    "infer_targets_concept",
]
