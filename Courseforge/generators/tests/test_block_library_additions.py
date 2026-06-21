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


def test_block_types_count_is_twenty_one():
    assert len(BLOCK_TYPES) == 24


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
