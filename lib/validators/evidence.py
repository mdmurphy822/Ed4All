"""Concept-graph evidence-shape validator loader.

Thin helper over ``schemas/knowledge/concept_graph_semantic.schema.json``.
The schema ships with a ``oneOf`` discriminator on ``edges[].provenance``
keyed by the ``rule`` field. Each specific arm (``{Rule}Provenance``) binds
``rule = {name}`` to a matching evidence ``$def``; the final
``FallbackProvenance`` arm matches any rule NOT in the 9 modeled rules
(via ``not: enum``) and accepts any evidence shape.

That keeps the default validation behaviour **lenient** — preserving
backward-compat with legacy graphs and with any rule whose evidence shape
predates REC-PRV-02.

Strict mode is **opt-in**: when the caller passes ``strict=True`` or the
environment variable ``TRAINFORGE_STRICT_EVIDENCE=true`` is set, this
loader returns a deep-copied schema with the ``FallbackProvenance`` arm
removed from ``edges[].provenance.oneOf``. An edge whose rule is unknown
OR whose evidence shape drifts from its modeled ``$def`` then fails
validation under that schema.

This module intentionally does **not** wire itself into any existing
validator callsite. It exists as a building block for strict-mode
validation and is consumed directly by
``lib/tests/test_evidence_discriminator.py``.

REC-PRV-02 (Wave 6, Worker W).
"""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

__all__ = ["get_schema", "SCHEMA_PATH", "STRICT_ENV_VAR"]

STRICT_ENV_VAR = "TRAINFORGE_STRICT_EVIDENCE"

# Project root heuristic: this file lives at ``<root>/lib/validators/evidence.py``;
# ``parents[2]`` is the project root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = _REPO_ROOT / "schemas" / "knowledge" / "concept_graph_semantic.schema.json"


@lru_cache(maxsize=1)
def _load_schema_raw() -> Dict[str, Any]:
    """Load and cache the raw schema dict exactly as it sits on disk."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _strip_fallback_arm(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy with FallbackProvenance removed from the oneOf.

    Mutating the cached raw dict would corrupt other callers, so we deep-copy
    before surgery. The target path is
    ``properties.edges.items.properties.provenance.oneOf``.
    Arms are identified by their ``$ref`` value; only the FallbackProvenance
    arm is dropped. Specific-rule arms stay.
    """
    out = copy.deepcopy(schema)
    try:
        provenance = (
            out["properties"]["edges"]["items"]["properties"]["provenance"]
        )
    except KeyError:
        # Schema shape unexpected — return unchanged rather than raise; the
        # caller's validation will surface the real problem.
        return out
    arms = provenance.get("oneOf")
    if not isinstance(arms, list):
        return out
    provenance["oneOf"] = [
        arm for arm in arms
        if not (isinstance(arm, dict) and arm.get("$ref", "").endswith("/FallbackProvenance"))
    ]
    return out


def get_schema(strict: bool | None = None) -> Dict[str, Any]:
    """Return the concept_graph_semantic schema, optionally in strict mode.

    Args:
        strict: Override. When ``True``, force strict; when ``False``, force
            lenient. When ``None`` (default), read the ``TRAINFORGE_STRICT_EVIDENCE``
            env var — ``"true"`` (case-insensitive) means strict; anything else
            means lenient.

    Returns:
        A schema dict. Lenient mode returns a shallow reference to the cached
        raw schema (callers must not mutate). Strict mode returns a fresh deep
        copy safe to mutate.
    """
    if strict is None:
        strict = (os.environ.get(STRICT_ENV_VAR, "").lower() == "true")
    raw = _load_schema_raw()
    if not strict:
        return raw
    return _strip_fallback_arm(raw)
