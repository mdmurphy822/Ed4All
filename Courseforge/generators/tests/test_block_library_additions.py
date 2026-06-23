"""Wave-2 block-variety additions — registration + catalog tests.

Surface under test (Wave-2 Part 1): the five NEW canonical block types
(``scenario`` / ``problem`` / ``vocab_card`` / ``formula`` / ``checklist``)
are registered across every two-pass sync point, and the machine-readable
block catalog covers every BLOCK_TYPES member exactly.

Coverage:

1. Each new type is in ``BLOCK_TYPES``.
2. Each new type has an entry in ``_OUTLINE_KIND_BOUNDS``,
   ``_BLOCK_TYPE_OUTPUT_CONTRACTS``, and ``DEFAULT_BLOCK_ROUTING``.
3. Each new type resolves through BOTH tiers with a NON-generic contract
   and a real bounds triple.
4. Constructing a ``Block`` for each new type renders HTML attrs (str) +
   a JSON-LD entry (dict) without raising.
5-8. Catalog loads, covers BLOCK_TYPES exactly, and every entry carries
   the contract keys + valid bloom_fit levels.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ``blocks.py`` lives under Courseforge/scripts and is imported as a
# top-level module elsewhere in the suite; add that dir for the bare
# ``blocks`` import the dataclass uses internally.
SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest  # noqa: E402

from Courseforge.scripts.blocks import BLOCK_TYPES, Block  # noqa: E402
from Courseforge.generators._outline_provider import (  # noqa: E402
    _OUTLINE_KIND_BOUNDS,
)
from Courseforge.generators._rewrite_provider import (  # noqa: E402
    _BLOCK_TYPE_OUTPUT_CONTRACTS,
    _block_type_output_contract,
)
from Courseforge.router.policy import DEFAULT_BLOCK_ROUTING  # noqa: E402
from lib.generation.block_catalog import load_block_catalog  # noqa: E402


NEW_BLOCK_TYPES = [
    "scenario",
    "problem",
    "vocab_card",
    "formula",
    "checklist",
]

_BLOOM_LEVELS = {
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
}


# ---------------------------------------------------------------------------
# Test 1 — registered in BLOCK_TYPES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_in_block_types(block_type):
    assert block_type in BLOCK_TYPES


def test_block_types_count_is_twenty_eight():
    # IB5 added 4 framework-aligned types (hook / multimedia / worked_example /
    # diagram) to the 24-member palette; B15 added `resources` -> 29.
    assert len(BLOCK_TYPES) == 29


# ---------------------------------------------------------------------------
# Test 2 — registered at every sync point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_in_outline_bounds(block_type):
    assert block_type in _OUTLINE_KIND_BOUNDS


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_in_rewrite_contracts(block_type):
    assert block_type in _BLOCK_TYPE_OUTPUT_CONTRACTS


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_in_default_routing(block_type):
    assert block_type in DEFAULT_BLOCK_ROUTING
    entry = DEFAULT_BLOCK_ROUTING[block_type]
    assert "required" in entry and isinstance(entry["required"], list)
    assert "optional" in entry and isinstance(entry["optional"], list)
    assert entry["fail_action"] in {"regenerate", "escalate", "block"}


# ---------------------------------------------------------------------------
# Test 3 — resolves through both tiers (non-generic contract + real bounds)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_resolves_both_tiers(block_type):
    # Rewrite tier: the resolved contract is the registered one, not the
    # generic fallback (the fallback interpolates the bare block_type repr).
    contract = _block_type_output_contract(block_type)
    assert contract == _BLOCK_TYPE_OUTPUT_CONTRACTS[block_type]
    generic = (
        f"Emit the rendered HTML body for a block of type "
        f"{block_type!r}. Carry `data-cf-source-ids` on the top "
        f"wrapper to attribute the source chunks."
    )
    assert contract != generic
    # Outline tier: a real bounds triple exists.
    bounds = _OUTLINE_KIND_BOUNDS[block_type]
    for key in ("key_claims", "section_skeleton", "summary_chars"):
        assert key in bounds
        lo, hi = bounds[key]
        assert isinstance(lo, int) and isinstance(hi, int)
        assert lo <= hi


# ---------------------------------------------------------------------------
# Test 4 — Block construction + projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_type", NEW_BLOCK_TYPES)
def test_new_type_block_projects(block_type):
    blk = Block(
        block_id=f"week_01_content_01#{block_type}_intro_0",
        block_type=block_type,
        page_id="week_01_content_01",
        sequence=0,
        content="Sample body for the block.",
        source_ids=("dart:sample#b1",),
        source_primary="dart:sample#b1",
    )
    attrs = blk.to_html_attrs()
    assert isinstance(attrs, str)
    # Wrapper-only routing emits the source-id attrs.
    assert "data-cf-source-ids" in attrs
    entry = blk.to_jsonld_entry()
    assert isinstance(entry, dict)


# ---------------------------------------------------------------------------
# Tests 5-8 — block catalog
# ---------------------------------------------------------------------------


def test_catalog_loads_non_empty():
    catalog = load_block_catalog()
    assert isinstance(catalog, list)
    assert catalog, "block catalog must be non-empty"


def test_catalog_coverage_equals_block_types_exactly():
    catalog = load_block_catalog()
    catalog_types = {entry["block_type"] for entry in catalog}
    assert catalog_types == set(BLOCK_TYPES), (
        "catalog coverage must equal BLOCK_TYPES exactly — "
        f"missing: {sorted(set(BLOCK_TYPES) - catalog_types)}, "
        f"orphans: {sorted(catalog_types - set(BLOCK_TYPES))}"
    )


def test_catalog_entries_have_required_keys():
    required_keys = {
        "block_type",
        "label",
        "use_when",
        "conveys",
        "bloom_fit",
        "format_summary",
    }
    for entry in load_block_catalog():
        missing = required_keys - set(entry.keys())
        assert not missing, f"{entry.get('block_type')!r} missing keys {missing}"


def test_catalog_use_when_and_bloom_fit_valid():
    for entry in load_block_catalog():
        bt = entry["block_type"]
        use_when = entry["use_when"]
        assert isinstance(use_when, str) and use_when.strip(), (
            f"{bt!r} use_when must be a non-empty string"
        )
        bloom_fit = entry["bloom_fit"]
        assert isinstance(bloom_fit, list) and bloom_fit, (
            f"{bt!r} bloom_fit must be a non-empty list"
        )
        for level in bloom_fit:
            assert level in _BLOOM_LEVELS, (
                f"{bt!r} bloom_fit has invalid level {level!r}"
            )


def test_catalog_bloom_ceiling_when_present_is_valid():
    """IB7.6 — an OPTIONAL ``bloom_ceiling`` must be a valid Bloom level >= the
    entry's bloom_fit floor. Absence is allowed (advisory back-compat)."""
    for entry in load_block_catalog():
        bt = entry["block_type"]
        ceiling = entry.get("bloom_ceiling")
        if ceiling is None:
            continue
        from lib.ontology.bloom import BLOOM_LEVELS as _ORDERED_BLOOM
        assert ceiling in _BLOOM_LEVELS, (
            f"{bt!r} bloom_ceiling {ceiling!r} not a valid Bloom level"
        )
        bloom_fit = [b for b in entry["bloom_fit"] if b in _BLOOM_LEVELS]
        floor = min(bloom_fit, key=_ORDERED_BLOOM.index)
        assert _ORDERED_BLOOM.index(ceiling) >= _ORDERED_BLOOM.index(floor), (
            f"{bt!r} bloom_ceiling {ceiling!r} is below its bloom_fit floor "
            f"{floor!r}"
        )


# ---------------------------------------------------------------------------
# IB2.1 — every entry declares its canonical framework B-code parent
# ---------------------------------------------------------------------------

_VALID_FRAMEWORK_CODES = {f"B{i:02d}" for i in range(1, 16)}


def test_every_entry_declares_framework_block():
    """Each catalog entry has a framework_block key in {B01..B15} ∪ {None}."""
    for entry in load_block_catalog():
        bt = entry["block_type"]
        assert "framework_block" in entry, (
            f"{bt!r} missing framework_block key (IB2.1)"
        )
        fb = entry["framework_block"]
        assert fb is None or fb in _VALID_FRAMEWORK_CODES, (
            f"{bt!r} framework_block {fb!r} not in {{B01..B15}} ∪ {{None}}"
        )


def test_framework_block_secondary_when_present_is_valid():
    """When framework_block_secondary is present it is a valid code != primary."""
    for entry in load_block_catalog():
        if "framework_block_secondary" not in entry:
            continue
        bt = entry["block_type"]
        sec = entry["framework_block_secondary"]
        assert sec in _VALID_FRAMEWORK_CODES, (
            f"{bt!r} framework_block_secondary {sec!r} not in {{B01..B15}}"
        )
        assert sec != entry["framework_block"], (
            f"{bt!r} secondary {sec!r} must differ from primary"
        )


def test_only_chrome_maps_to_null():
    """chrome is the ONLY entry whose framework_block is None."""
    nulls = [
        entry["block_type"]
        for entry in load_block_catalog()
        if entry.get("framework_block") is None
    ]
    assert nulls == ["chrome"], (
        f"only chrome may map to null framework_block; got {nulls}"
    )


def test_framework_block_coverage_equals_block_types():
    """Every BLOCK_TYPES member has a catalog entry carrying framework_block."""
    covered = {
        entry["block_type"]
        for entry in load_block_catalog()
        if "framework_block" in entry
    }
    assert covered == set(BLOCK_TYPES), (
        "framework_block coverage must equal BLOCK_TYPES — "
        f"missing: {sorted(set(BLOCK_TYPES) - covered)}"
    )


def test_framework_map_onto_b01_through_b15():
    """The 29 primaries are ONTO the FULL canonical catalog B01–B15; every
    primary is a valid code.

    IB5 landed the dedicated B02 (hook), B04 (multimedia), B05 (worked_example),
    and B06 (diagram) first-class types; the B15 wave added `resources`, so the
    last catalog gap is closed and EVERY canonical B-code now has an Ed4All
    primary."""
    primaries = {
        entry.get("framework_block")
        for entry in load_block_catalog()
        if entry.get("framework_block") is not None
    }
    assert primaries <= _VALID_FRAMEWORK_CODES
    # B15 (`resources`) closed the last gap — the codes that have a primary are
    # now exactly the FULL canonical set B01–B15 (no gap).
    assert primaries == {
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
        "B09", "B10", "B11", "B12", "B13", "B14", "B15",
    }, f"unexpected primary code set: {sorted(primaries)}"
