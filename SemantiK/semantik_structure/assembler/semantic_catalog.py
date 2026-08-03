"""Gold-standard semantic-component catalog loader (gold semantic-class plan,
Phase 3 — pure, GPU-free, consumed by nobody yet).

Loads ``SemantiK/semantik_structure/config/semantic_components.yaml`` — the
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
    "component_for_pedagogy_class",
    "component_for_pedagogical_role",
    "label_for",
]


# AXIS-4: map the NEW span-level ``pedagogical_role`` head tokens
# (data.builders.build_structure_data.PEDAGOGICAL_ROLE_NAMES) onto EXISTING gold
# semantic components — NO new components are invented. example/try-it/how-to/
# solution/step are all parts of a worked instance (-> worked_example);
# objectives -> objectives; exercise/practice/be-prepared are practice-item
# blocks (-> exercise). ``none`` (and any unknown token) -> None. Every target
# is asserted to be a real catalog component at lookup time (anti-fabrication).
_PEDAGOGICAL_ROLE_TO_COMPONENT = {
    "example_open": "worked_example",
    "try_it": "worked_example",
    "how_to": "worked_example",
    "solution": "worked_example",
    "step": "worked_example",
    "objectives": "objectives",
    "exercise_open": "exercise",
    "practice": "exercise",
    "be_prepared": "exercise",
}


def semantic_catalog_path() -> Path:
    """Resolve the in-tree catalog YAML path PACKAGE-RELATIVE.

    ``<this file>/../config/semantic_components.yaml`` ==
    ``SemantiK/semantik_structure/config/semantic_components.yaml``. Deliberately
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
def _pedagogy_class_to_component() -> Dict[str, str]:
    """Reverse of each component's ``reconciles_with`` -> the ``component`` token.

    The always-on deterministic structure-clean pass
    (``deterministic_structure._retag``) stamps a pedagogical CSS class on
    ``payload['css_class']`` (``pedagogy-example`` / ``pedagogy-objectives`` /
    ``pedagogy-practice`` / ...). This map lets the Stage-5d reviewer FLOOR turn
    that already-detected class into its gold ``semantic_class`` token
    (``worked_example`` / ``objectives`` / ``exercise``) WITHOUT a model call.
    First-match-in-catalog-order wins (mirrors :func:`_kind_to_component`); a
    component with ``reconciles_with: null`` contributes nothing.
    """
    mapping: Dict[str, str] = {}
    for entry in _load_raw()["components"]:
        component = entry.get("component")
        if not component:
            continue
        reconciles = entry.get("reconciles_with")
        if reconciles and str(reconciles) not in mapping:
            mapping[str(reconciles)] = component
    return mapping


def component_for_pedagogy_class(pedagogy_class: str) -> Optional[str]:
    """Return the gold ``component`` a clean-pass pedagogy CSS class reconciles to.

    The REVERSE of the catalog ``reconciles_with`` link: ``pedagogy-example``
    -> ``worked_example``, ``pedagogy-objectives`` -> ``objectives``,
    ``pedagogy-practice`` -> ``exercise``. ``None`` for any class no component
    declares (anti-fabrication — the reviewer floor never invents a token).
    Cached via :func:`_pedagogy_class_to_component`.
    """
    if not pedagogy_class:
        return None
    return _pedagogy_class_to_component().get(pedagogy_class)


def component_for_pedagogical_role(pedagogical_role: str) -> Optional[str]:
    """Return the gold ``component`` a ``pedagogical_role`` head token maps to.

    AXIS-4 bridge from the new span-level ``pedagogical_role`` head
    (``example_open`` / ``try_it`` / ``solution`` / ``objectives`` /
    ``exercise_open`` / …) to an EXISTING semantic component token
    (``worked_example`` / ``objectives`` / ``exercise``). ``None`` for
    ``none`` / unknown tokens, and ``None`` (anti-fabrication) if the mapped
    component is somehow not in the catalog — the bridge never invents a token.
    Pairs with :func:`component_for_pedagogy_class` (the clean-pass CSS-class
    bridge); both resolve to the same existing components.
    """
    if not pedagogical_role:
        return None
    component = _PEDAGOGICAL_ROLE_TO_COMPONENT.get(pedagogical_role)
    if component is None:
        return None
    return component if component in valid_components() else None


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
