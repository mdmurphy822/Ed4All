"""Canonical framework B-code single-source-of-truth (IB2.2).

The instruction-blocks framework defines a canonical **15-block catalog
(B01–B15)** — Objectives, Hook/Activation, Exposition, Multimedia, Worked
Example, Visual Model/Diagram, Knowledge-Check, Guided Practice,
Case/Scenario, Discussion, Reflection, Callout, Summary, Graded Assessment,
Resources. Ed4All's 24-member ``Courseforge/scripts/blocks.py::BLOCK_TYPES``
palette is a finer-grained instance set: each Ed4All type declares its
canonical framework parent via the per-entry ``framework_block`` field in
``Courseforge/config/block_catalog.yaml`` (the IB2 reconciliation map).

This module is the ONE place validators / IB5 / IB6 ask "what is the B-code
for ``assessment_item``?" instead of re-parsing the catalog YAML. It mirrors
the SoT-loader posture of the sibling ``lib/ontology/`` modules
(``taxonomy.py`` / ``learning_objectives.py``): an ``@lru_cache`` single-shot
read, defensive lookups, fail-soft (``None``) on unknown / non-pedagogical
block types.

Surface:

    FRAMEWORK_BLOCKS: Dict[str, str]              # B01..B15 -> human label
    is_framework_block(code) -> bool
    framework_block_for(block_type) -> Optional[str]   # primary B-code or None

No runtime behavior change anywhere this wave — nothing consumes it yet
(IB5 reads the parents; IB6 reads them for the rubric). Pure additive API.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

__all__ = ["FRAMEWORK_BLOCKS", "is_framework_block", "framework_block_for"]


# The canonical 15 framework block codes -> human label, verbatim from the
# framework_requirement string (gap ``catalog-26-vs-15-reconciliation``). This
# is the authoritative B-code label table; the data model in
# ``Courseforge/scripts/blocks.py`` and the JSON-LD schema validate against
# this set.
FRAMEWORK_BLOCKS: Dict[str, str] = {
    "B01": "Objectives",
    "B02": "Hook/Activation",
    "B03": "Exposition",
    "B04": "Multimedia",
    "B05": "Worked Example",
    "B06": "Visual Model/Diagram",
    "B07": "Knowledge-Check",
    "B08": "Guided Practice",
    "B09": "Case/Scenario",
    "B10": "Discussion",
    "B11": "Reflection",
    "B12": "Callout",
    "B13": "Summary",
    "B14": "Graded Assessment",
    "B15": "Resources",
}


def is_framework_block(code: str) -> bool:
    """True iff ``code`` is one of the canonical B01–B15 codes."""
    return code in FRAMEWORK_BLOCKS


@lru_cache(maxsize=1)
def _block_type_to_framework() -> Dict[str, Optional[str]]:
    """Map every catalog ``block_type`` -> its primary ``framework_block``.

    Reads the catalog once (cached). Imported lazily so this module stays
    import-light and does not pull the Courseforge loader at module import
    for callers that only need :data:`FRAMEWORK_BLOCKS`.
    """
    from lib.generation.block_catalog import load_block_catalog

    mapping: Dict[str, Optional[str]] = {}
    for entry in load_block_catalog():
        bt = entry.get("block_type")
        if not bt:
            continue
        mapping[bt] = entry.get("framework_block")
    return mapping


def framework_block_for(block_type: str) -> Optional[str]:
    """Return the primary framework B-code for ``block_type``.

    ``None`` for the non-pedagogical ``chrome`` scaffolding (whose
    ``framework_block`` is explicitly ``null``) and for any unknown
    block_type not in the catalog. Cached via :func:`_block_type_to_framework`.
    """
    return _block_type_to_framework().get(block_type)
