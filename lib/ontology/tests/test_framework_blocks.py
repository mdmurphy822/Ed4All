"""IB2.2 — canonical framework B-code SoT loader tests.

Confirms ``lib/ontology/framework_blocks.py`` exposes the 15 canonical
B-codes and that ``framework_block_for`` round-trips the catalog's
``framework_block`` primaries for every ``BLOCK_TYPES`` member.
"""

from __future__ import annotations

from Courseforge.scripts.blocks import BLOCK_TYPES
from lib.generation.block_catalog import load_block_catalog
from lib.ontology.framework_blocks import (
    FRAMEWORK_BLOCKS,
    framework_block_for,
    is_framework_block,
)


def test_fifteen_codes_exact():
    assert set(FRAMEWORK_BLOCKS) == {f"B{i:02d}" for i in range(1, 16)}


def test_all_labels_non_empty():
    for code, label in FRAMEWORK_BLOCKS.items():
        assert isinstance(label, str) and label.strip(), code


def test_is_framework_block():
    assert is_framework_block("B01")
    assert is_framework_block("B15")
    assert not is_framework_block("B16")
    assert not is_framework_block("b01")
    assert not is_framework_block("")


def test_framework_block_for_roundtrips_catalog():
    by_type = {e["block_type"]: e.get("framework_block") for e in load_block_catalog()}
    for bt in BLOCK_TYPES:
        assert framework_block_for(bt) == by_type[bt], bt


def test_framework_block_for_chrome_is_none():
    assert framework_block_for("chrome") is None


def test_framework_block_for_unknown_is_none():
    assert framework_block_for("definitely_not_a_block_type") is None


def test_all_primaries_are_valid_codes():
    for entry in load_block_catalog():
        fb = entry.get("framework_block")
        if fb is not None:
            assert is_framework_block(fb), (entry["block_type"], fb)
