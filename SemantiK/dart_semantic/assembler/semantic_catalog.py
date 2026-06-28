"""Gold-standard semantic-component catalog loader (gold semantic-class plan,
Phase 3 — pure, GPU-free, consumed by nobody yet).

Loads ``SemantiK/dart_semantic/config/semantic_components.yaml`` — the
machine-readable catalog of the gold accessible-component vocabulary the
Stage-5d reviewer assigns as a block's ``semantic_class`` (Phase 5 reads
``use_when`` as the 7B discriminator menu) and the assembler renders as gold
ARIA markup (Phase 7 reads ``label`` / ``html_shape`` / ``aria_markup`` /
``role`` / ``css_class``). This axis is ORTHOGONAL to the 11-member
``REGION_KINDS``.

Surface::

    load_semantic_catalog() -> list[dict]        # the catalog's ``components`` list
    valid_components() -> frozenset[str]          # the full ``component`` token set
    semantic_component_for(region_kind) -> str | None  # kind -> primary component
    label_for(component) -> str | None            # component -> authored label

Posture mirrors ``lib/generation/block_catalog.py`` VERBATIM (``@lru_cache``
single-shot read + fail-loud on a missing/malformed YAML + a defensive
per-entry copy on each ``load`` call) and the SoT-accessor posture of
``lib/ontology/framework_blocks.py::framework_block_for`` (an ``lru_cache``-d
lookup over the loaded catalog).

The ONE deliberate divergence from ``block_catalog.py``: the YAML is resolved
PACKAGE-RELATIVE (``Path(__file__).resolve().parent.parent / "config" / ...``).
A checked-in config ships in the package tree — it does NOT move under the
relocatable data-home root (that resolves models/data dirs, not package
config). The loader imports NOTHING from the Ed4All ``lib`` namespace and reads
no relocation env var: the reviewer runs out-of-process in SemantiK's bridge
venv, where an Ed4All-pathed loader would break. Imports are SemantiK-local
only (``functools`` / ``pathlib`` / ``yaml``).

Fail-loud contract: the catalog ships in-tree, so a missing or malformed file
is a packaging bug, not a recoverable runtime state — the loader raises rather
than returning an empty list.

No runtime behavior change this phase — nothing consumes it yet (the assembler
never reads it unless ``SEMANTIK_GOLD_SHELL`` is on, wired in a later phase).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import yaml

__all__ = [
    "semantic_catalog_path",
    "load_semantic_catalog",
    "valid_components",
    "semantic_component_for",
    "label_for",
]


def semantic_catalog_path() -> Path:
    """Resolve the in-tree catalog YAML path PACKAGE-RELATIVE.

    ``<this file>/../config/semantic_components.yaml`` ==
    ``SemantiK/dart_semantic/config/semantic_components.yaml``. Deliberately
    NOT keyed off any relocation env var (those relocate the models/data roots;
    a checked-in config does not move under them) and NOT off the Ed4All
    project-root helpers (the reviewer reads this in SemantiK's out-of-process
    bridge venv).
    """
    return Path(__file__).resolve().parent.parent / "config" / "semantic_components.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> Dict:
    """Read + parse the catalog YAML once (cached). Fail-loud."""
    path = semantic_catalog_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"semantic_components.yaml not found at {path} — the catalog ships "
            "in-tree; a missing file is a packaging bug."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "components" not in data:
        raise ValueError(
            f"semantic_components.yaml at {path} is malformed — expected a "
            "top-level mapping with a 'components' key."
        )
    components = data["components"]
    if not isinstance(components, list):
        raise ValueError(
            f"semantic_components.yaml at {path} 'components' must be a list, "
            f"got {type(components).__name__}."
        )
    return data


def load_semantic_catalog() -> List[Dict]:
    """Return the catalog ``components`` list (defensive copy per entry).

    Each list item is a dict carrying ``component`` / ``label`` / ``use_when``
    / ``html_shape`` / ``aria_markup`` / ``role`` / ``css_class`` /
    ``maps_from_region_kind`` / ``reconciles_with``. The per-entry shallow copy
    means a caller mutating an entry can't poison the ``lru_cache`` singleton.
    """
    components = _load_raw()["components"]
    return [dict(entry) for entry in components]


@lru_cache(maxsize=1)
def valid_components() -> FrozenSet[str]:
    """The full set of ``component`` tokens (Phase-6 validate-against-catalog)."""
    return frozenset(
        entry["component"]
        for entry in _load_raw()["components"]
        if entry.get("component")
    )


@lru_cache(maxsize=1)
def _kind_to_component() -> Dict[str, str]:
    """Map each ``REGION_KINDS`` token -> its PRIMARY semantic component.

    Multi-map policy: a ``region_kind`` can appear under several components'
    ``maps_from_region_kind`` lists (e.g. ``code_block`` -> ``worked_example``
    AND ``code_region``; ``paragraph`` -> several). The PRIMARY component is the
    FIRST one declaring that kind in catalog (YAML document) order, so the
    catalog author controls precedence by ordering — ``worked_example`` precedes
    ``code_region``, so ``code_block`` resolves to ``worked_example``. Later
    components claiming an already-mapped kind do not override the first.
    """
    mapping: Dict[str, str] = {}
    for entry in _load_raw()["components"]:
        component = entry.get("component")
        if not component:
            continue
        for kind in entry.get("maps_from_region_kind") or []:
            if kind and kind not in mapping:
                mapping[kind] = component
    return mapping


def semantic_component_for(region_kind: str) -> Optional[str]:
    """Return the PRIMARY semantic component a ``region_kind`` re-types into.

    First-match-in-catalog-order wins (see :func:`_kind_to_component`).
    ``None`` for any kind no component maps from. Cached.
    """
    return _kind_to_component().get(region_kind)


@lru_cache(maxsize=1)
def _component_to_label() -> Dict[str, str]:
    """Map each ``component`` token -> its authored ``label`` text."""
    mapping: Dict[str, str] = {}
    for entry in _load_raw()["components"]:
        component = entry.get("component")
        if not component:
            continue
        label = entry.get("label")
        if label:
            mapping[component] = label
    return mapping


def label_for(component: str) -> Optional[str]:
    """Return the authored container-heading ``label`` for a ``component`` token.

    ``None`` for an unknown token. Cached.
    """
    return _component_to_label().get(component)
