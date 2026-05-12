"""Typed-edge inference rules for the semantic concept graph.

Each rule module exposes:

    RULE_NAME: str
    RULE_VERSION: int
    EDGE_TYPE: str  # one of the values in concept_graph_semantic.schema.json
                    # edge.type enum: "prerequisite", "is-a", "related-to",
                    # "assesses", "exemplifies", "misconception-of",
                    # "derived-from-objective", "defined-by",
                    # "targets-concept", "broader-than", "narrower-than",
                    # "corrected-by-chunk", "detected-by-question",
                    # "interferes-with-outcome"

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

from . import assesses_from_question_lo as _assesses_mod
from . import corrected_by_from_chunk_misconception as _corrected_by_mod
from . import defined_by_from_first_mention as _defined_by_mod
from . import derived_from_lo_ref as _derived_lo_mod
from . import detected_by_from_distractor_misconception_id as _detected_by_mod
from . import exemplifies_from_example_chunks as _exemplifies_mod
from . import interferes_with_outcome_from_misconception_lo as _interferes_mod
from . import is_a_from_key_terms as _is_a_mod
from . import misconception_of_from_misconception_ref as _misconception_mod
from . import prerequisite_from_lo_order as _prereq_mod
from . import related_from_cooccurrence as _related_mod
from . import targets_concept_from_lo as _targets_concept_mod

from .assesses_from_question_lo import infer as infer_assesses
from .corrected_by_from_chunk_misconception import infer as infer_corrected_by_chunk
from .defined_by_from_first_mention import infer as infer_defined_by
from .derived_from_lo_ref import infer as infer_derived_from_objective
from .detected_by_from_distractor_misconception_id import (
    infer as infer_detected_by_question,
)
from .exemplifies_from_example_chunks import infer as infer_exemplifies
from .interferes_with_outcome_from_misconception_lo import (
    infer as infer_interferes_with_outcome,
)
from .is_a_from_key_terms import infer as infer_is_a
from .misconception_of_from_misconception_ref import infer as infer_misconception_of
from .prerequisite_from_lo_order import infer as infer_prerequisite
from .related_from_cooccurrence import infer as infer_related
from .targets_concept_from_lo import infer as infer_targets_concept


# GPT Feedback v2 (May 12 / item 3): aggregate rulepack version.
#
# Mirrors the ``CHUNKER_SCHEMA_VERSION`` pattern from
# ``Trainforge.chunker``: a single short deterministic string consumers
# can pin without enumerating every per-rule constant. Bumps
# automatically whenever any rule module bumps its ``RULE_VERSION``,
# so operators never forget to bump a top-level constant by hand.
#
# Form: ``"v" + sha256(canonical_form)[:8]`` where ``canonical_form``
# is the JSON-encoded sorted list of ``(RULE_NAME, RULE_VERSION)``
# tuples of every imported rule module above. Matches schema pattern
# ``^v[0-9a-f]+$`` (see
# ``schemas/knowledge/concept_graph_semantic.schema.json::properties.rulepack_version``).
def _compute_rulepack_version() -> str:
    import hashlib
    import json

    pairs = sorted(
        (mod.RULE_NAME, mod.RULE_VERSION)
        for mod in (
            _assesses_mod,
            _corrected_by_mod,
            _defined_by_mod,
            _derived_lo_mod,
            _detected_by_mod,
            _exemplifies_mod,
            _interferes_mod,
            _is_a_mod,
            _misconception_mod,
            _prereq_mod,
            _related_mod,
            _targets_concept_mod,
        )
    )
    canonical = json.dumps(pairs, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "v" + digest[:8]


RULEPACK_VERSION = _compute_rulepack_version()


__all__ = [
    "RULEPACK_VERSION",
    "infer_assesses",
    "infer_corrected_by_chunk",
    "infer_defined_by",
    "infer_derived_from_objective",
    "infer_detected_by_question",
    "infer_exemplifies",
    "infer_interferes_with_outcome",
    "infer_is_a",
    "infer_misconception_of",
    "infer_prerequisite",
    "infer_related",
    "infer_targets_concept",
]
