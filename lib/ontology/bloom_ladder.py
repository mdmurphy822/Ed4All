"""Bloom-ladder block/probe loader (Bloom-ladder initiative, WI-01).

Loads `schemas/taxonomies/bloom_ladder_blocks.json` (the dual-use
schema+data artifact, structured as `properties.{level}.default` dicts
of `{ordinal, block_types, misconception_probe}` -- mirrors the pattern
in `schemas/taxonomies/bloom_verbs.json`) and exposes:

    RungEntry               -- frozen dataclass capturing a single rung.
    rungs_up_to(ceiling)    -- Tuple[RungEntry, ...] ordered low->high,
                               every rung whose ordinal is <= the
                               ceiling level's ordinal (i.e. the ladder
                               capped at an objective's own bloom_level).
    blocks_permitted_at(level) -- frozenset of BLOCK_TYPES tokens
                               permitted at exactly that rung.
    misconception_probe_for(level) -- Dict capturing that rung's
                               probe_strategy + the (zero-touch) class
                               token contract.

All getters validate `level` against the canonical 6-value Bloom set
(`lib.ontology.bloom.BLOOM_LEVELS`) and raise `ValueError` naming the
valid set on an unrecognized level -- fail-loud, mirroring
`lib/ontology/taxonomy.py`. `block_types` on every rung is additionally
cross-checked at load time against the canonical Courseforge BLOCK_TYPES
palette (imported lazily via `lib.generation.block_catalog` to stay
import-light, mirroring `lib/ontology/framework_blocks.py`) -- a rung
declaring an unknown block_type is a data-authoring bug, not a runtime
condition, so it raises at first load rather than silently admitting a
non-canonical token.

This module is a pure data loader: it mints no BLOCK_TYPES tokens, never
mutates `compute_content_hash` inputs, and never touches the misconception
card's existing `misconception-card` / `misconception-claim` /
`misconception-correction` class tokens -- see
`Trainforge/generators/synthesis_window_contract.py:222-238`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

from lib.ontology.bloom import BLOOM_LEVELS

__all__ = [
    "RungEntry",
    "PROBE_STRATEGIES",
    "load_bloom_ladder",
    "rungs_up_to",
    "blocks_permitted_at",
    "misconception_probe_for",
]


# ---------------------------------------------------------------------------
# Constants -- the zero-touch misconception-card class-token contract
# (Trainforge/generators/synthesis_window_contract.py:222-238), pinned
# verbatim here so a drifted schema entry fails loudly at load time.
# ---------------------------------------------------------------------------

_CARD_CLASS = "misconception-card"
_CLAIM_CLASS = "misconception-claim"
_CORRECTION_CLASS = "misconception-correction"

#: The six canonical misconception-probe strategies. One per rung; the
#: schema's `$defs.ProbeStrategy` enum mirrors this tuple.
PROBE_STRATEGIES: Tuple[str, ...] = (
    "direct_negation",
    "partial_truth",
    "off_by_one",
    "reversal",
    "false_analogy",
    "overextension",
)


@dataclass(frozen=True)
class RungEntry:
    """A single Bloom-ladder rung.

    Fields:
        level:               the Bloom level name (e.g. "remember").
        ordinal:             1-indexed rung position (remember=1 .. create=6).
        block_types:         frozenset of Courseforge BLOCK_TYPES tokens
                              permitted at this rung.
        misconception_probe: `{probe_strategy, card_class, claim_class,
                              correction_class}` -- the rung's
                              misconception-card probe guidance.
    """

    level: str
    ordinal: int
    block_types: FrozenSet[str]
    misconception_probe: Dict[str, str]


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_BLOOM_LADDER_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "bloom_ladder_blocks.json"
)

#: Level index used for validation error messages and ordinal cross-checks.
_LEVEL_ORDINAL = {level: idx + 1 for idx, level in enumerate(BLOOM_LEVELS)}


def _canonical_block_types() -> FrozenSet[str]:
    """Return the canonical 30-member BLOCK_TYPES palette.

    Lazily imports `lib.generation.block_catalog` (itself a thin loader
    over `Courseforge/config/block_catalog.yaml`, which mirrors
    `Courseforge/scripts/blocks.py::BLOCK_TYPES` 1:1) so this module stays
    import-light -- mirrors the lazy-import posture of
    `lib/ontology/framework_blocks.py::_block_type_to_framework`.
    """
    from lib.generation.block_catalog import load_block_catalog

    return frozenset(
        entry["block_type"]
        for entry in load_block_catalog()
        if entry.get("block_type")
    )


@lru_cache(maxsize=1)
def load_bloom_ladder() -> Dict[str, Dict]:
    """Read the bloom_ladder_blocks schema and return a {level: rung_dict} map.

    Only the per-level `default` dicts are extracted; the schema's `$defs`
    metadata is ignored here (it's for validators, not loaders). Raises
    `FileNotFoundError` on a missing schema and `ValueError` on a malformed
    one (missing level, missing `default`, ordinal/level mismatch, an
    unknown block_type, a probe_strategy outside the canonical set, or a
    misconception-card class token that has drifted off the zero-touch
    contract).
    """
    if not _BLOOM_LADDER_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Bloom ladder schema not found at {_BLOOM_LADDER_SCHEMA_PATH}. "
            "Expected canonical copy from the Bloom-ladder initiative (WI-01)."
        )

    with open(_BLOOM_LADDER_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    properties = schema.get("properties", {})
    canonical_blocks = _canonical_block_types()

    out: Dict[str, Dict] = {}
    for level in BLOOM_LEVELS:
        level_prop = properties.get(level)
        if not level_prop or "default" not in level_prop:
            raise ValueError(
                f"Malformed bloom_ladder_blocks schema: missing "
                f"properties.{level}.default at {_BLOOM_LADDER_SCHEMA_PATH}"
            )
        rung = level_prop["default"]

        ordinal = rung.get("ordinal")
        if ordinal != _LEVEL_ORDINAL[level]:
            raise ValueError(
                f"Malformed bloom_ladder_blocks schema: level {level!r} "
                f"declares ordinal {ordinal!r}, expected "
                f"{_LEVEL_ORDINAL[level]!r}."
            )

        block_types = rung.get("block_types") or []
        if not block_types:
            raise ValueError(
                f"Malformed bloom_ladder_blocks schema: level {level!r} "
                "declares no block_types."
            )
        unknown = set(block_types) - canonical_blocks
        if unknown:
            raise ValueError(
                f"Malformed bloom_ladder_blocks schema: level {level!r} "
                f"declares block_types outside the canonical BLOCK_TYPES "
                f"palette: {sorted(unknown)}"
            )

        probe = rung.get("misconception_probe") or {}
        strategy = probe.get("probe_strategy")
        if strategy not in PROBE_STRATEGIES:
            raise ValueError(
                f"Malformed bloom_ladder_blocks schema: level {level!r} "
                f"misconception_probe.probe_strategy {strategy!r} not in "
                f"canonical set {PROBE_STRATEGIES}."
            )
        if (
            probe.get("card_class") != _CARD_CLASS
            or probe.get("claim_class") != _CLAIM_CLASS
            or probe.get("correction_class") != _CORRECTION_CLASS
        ):
            raise ValueError(
                f"Malformed bloom_ladder_blocks schema: level {level!r} "
                "misconception_probe class tokens have drifted off the "
                f"zero-touch contract ({_CARD_CLASS!r} / {_CLAIM_CLASS!r} / "
                f"{_CORRECTION_CLASS!r})."
            )

        out[level] = rung
    return out


@lru_cache(maxsize=1)
def _build_rung_objects() -> Dict[str, RungEntry]:
    """Build immutable per-level `RungEntry` objects. Cached; getters copy."""
    raw = load_bloom_ladder()
    return {
        level: RungEntry(
            level=level,
            ordinal=raw[level]["ordinal"],
            block_types=frozenset(raw[level]["block_types"]),
            misconception_probe=dict(raw[level]["misconception_probe"]),
        )
        for level in BLOOM_LEVELS
    }


def _validate_level(level: str) -> None:
    if level not in _LEVEL_ORDINAL:
        raise ValueError(
            f"Unrecognized Bloom level {level!r}; expected one of "
            f"{BLOOM_LEVELS}."
        )


# ---------------------------------------------------------------------------
# Public getters
# ---------------------------------------------------------------------------


def rungs_up_to(ceiling: str) -> Tuple[RungEntry, ...]:
    """Return every rung whose ordinal is <= `ceiling`'s ordinal, ordered low->high.

    This is the ladder-capping primitive: an objective synthesized at
    bloom_level=`ceiling` may draw block types / misconception probes from
    its OWN rung and every rung below it, never a rung above it.

    Raises `ValueError` if `ceiling` is not one of the six canonical Bloom
    levels.

    Examples:
        >>> [r.level for r in rungs_up_to("apply")]
        ['remember', 'understand', 'apply']
        >>> [r.level for r in rungs_up_to("remember")]
        ['remember']
    """
    _validate_level(ceiling)
    built = _build_rung_objects()
    ceiling_ordinal = _LEVEL_ORDINAL[ceiling]
    return tuple(
        built[level] for level in BLOOM_LEVELS if built[level].ordinal <= ceiling_ordinal
    )


def blocks_permitted_at(level: str) -> FrozenSet[str]:
    """Return the frozenset of BLOCK_TYPES tokens permitted at exactly `level`'s rung.

    Raises `ValueError` if `level` is not one of the six canonical Bloom
    levels. Frozensets are immutable, so no defensive copy is needed.
    """
    _validate_level(level)
    return _build_rung_objects()[level].block_types


def misconception_probe_for(level: str) -> Dict[str, str]:
    """Return a defensive copy of `level`'s misconception-card probe guidance.

    Shape: `{probe_strategy, card_class, claim_class, correction_class}`.
    Raises `ValueError` if `level` is not one of the six canonical Bloom
    levels.
    """
    _validate_level(level)
    return dict(_build_rung_objects()[level].misconception_probe)
