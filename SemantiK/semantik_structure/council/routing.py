"""Static DAG describing the Phase-3 council schedule.

The five council BERTs (today's lineup, post-2026-05-05 retirement of
the standalone BERT-TableDetector) and how they feed each other:

::

    merge_or_split  ──► structure ──► semantic   (cascade-from-structure)

    table_specialist  (root; no upstream in v1 — see note below)
    math_specialist   (root; per math candidate, ungated)

The ``structure ──► table_specialist`` soft-hint edge is **deferred to
v4**. The v3 table_specialist adapter has no cascade projection layer;
its head input is 832 = 768 (backbone) + 64 (cell features), with no
+8 cascade slot. Until the adapter grows that projection, declaring
the edge here would mislead readers into thinking the orchestrator
ingests a Structure cascade for tables — it does not. Once the v4
adapter ships with a cascade slot, restore the edge in :data:`ROUTING`
and the diagram above.

There is **no MathDetector** in v1: ``math_specialist`` runs ungated on
:class:`~semantik_structure.region_detection.MathCandidate` regions emitted
by Stage 2's :func:`~semantik_structure.region_detection.detect_math_region_candidates`.
Whether a candidate is "really math" is decided downstream by the
cross-BERT reranker (see ``cross_reranker.py``) using the candidate's
:attr:`~semantik_structure.region_detection.MathCandidate.glyph_density_features`
as a deterministic stand-in detector.

Per architecture.md §3.1: pass distributions (top-k + confidences), not
argmax. The orchestrator (``semantik_structure.council.orchestrator``) walks
this DAG in topological order, swapping LoRA adapters between BERTs.

Example
-------

>>> topological_order()
['math_specialist', 'merge_or_split', 'structure', 'semantic', 'table_specialist']

(Tie-break is alphabetical among ready-set siblings. The four roots —
``math_specialist``, ``merge_or_split``, ``structure``,
``table_specialist`` — sort first by name; once ``structure``
finishes, ``semantic`` joins the ready set and is re-sorted ahead of
the still-pending ``table_specialist``.)
"""
from __future__ import annotations

from typing import TypedDict


class _RouteSpec(TypedDict, total=False):
    consumes: tuple[str, ...]    # upstream BERT names whose outputs this BERT reads
    emits: tuple[str, ...]       # head_name(s) this BERT publishes
    runs_per: str                # "region" | "block" | "document" | "table" | "math_region"


# Static DAG. Edges are encoded by `consumes`. Empty consumes = root.
ROUTING: dict[str, _RouteSpec] = {
    "merge_or_split": {
        "consumes": (),
        "emits": ("same_logical_block", "join_type", "hyphen_repair"),
        "runs_per": "block",
    },
    "structure": {
        "consumes": (),
        "emits": (
            "structural_role",
            "is_heading",
            "table_region",
            "is_image_block",
            "list_nesting",
        ),
        "runs_per": "block",
    },
    "semantic": {
        # Semantic reads Structure's per-span distribution as a cascade
        # vector (8-dim: P(role)x6, P(is_heading=1), P(table_region=1)).
        "consumes": ("structure",),
        "emits": ("doc_role", "boilerplate"),
        "runs_per": "block",
    },
    "table_specialist": {
        # Per architecture.md §3.1, table_specialist *will* take
        # Structure's top-k as a soft hint, but that edge is **deferred
        # to v4**. The v3 adapter has no cascade projection layer;
        # head input is 832 = 768 (backbone) + 64 (cell features), with
        # no +8 cascade slot. Declaring the edge here would mislead
        # readers and the topological order; restore it once the v4
        # adapter ships with a cascade slot.
        "consumes": (),
        "emits": ("cell_role_scope",),
        "runs_per": "table",
    },
    "math_specialist": {
        # MathSpecialist runs ungated for v1: there is no MathDetector
        # BERT. Detector-style gating is done downstream in the
        # cross-BERT reranker using
        # MathCandidate.glyph_density_features.
        "consumes": (),
        "emits": ("math_type", "eq_num_assoc"),
        "runs_per": "math_region",
    },
}


def topological_order() -> list[str]:
    """Kahn's algorithm over ROUTING, deterministic tie-break by name.

    Returns BERT names in an order safe for sequential execution.
    Cycle in the static DAG raises :class:`ValueError`.
    """
    indegree = {name: 0 for name in ROUTING}
    for name, spec in ROUTING.items():
        for upstream in spec.get("consumes", ()):
            if upstream not in ROUTING:
                raise ValueError(
                    f"{name!r} consumes unknown BERT {upstream!r}"
                )
            indegree[name] += 1

    order: list[str] = []
    ready = sorted(n for n, d in indegree.items() if d == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for downstream, spec in ROUTING.items():
            if n in spec.get("consumes", ()):
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    ready.append(downstream)
                    ready.sort()

    if len(order) != len(ROUTING):
        raise ValueError(
            f"cycle in ROUTING; ordered {order}, "
            f"expected {sorted(ROUTING)}"
        )
    return order


__all__ = [
    "ROUTING",
    "topological_order",
]
