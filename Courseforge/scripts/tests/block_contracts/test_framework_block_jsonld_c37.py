"""C3-7 — frameworkBlock JSON-LD emit (gated by ED4ALL_BLOCK_ANATOMY).

Every block_type maps to a canonical framework B-code (B01–B15) via the SoT
loader lib/ontology/framework_blocks.py. ``_minimal_block_jsonld`` emits it as
``frameworkBlock`` ONLY when ED4ALL_BLOCK_ANATOMY is on AND the type resolves a
code (absent for the non-pedagogical ``chrome`` scaffolding). Byte-stable off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402


def _block(block_type="recap", **kw):
    base = dict(
        block_id="p#x_0", block_type=block_type, page_id="p",
        sequence=0, content="body",
    )
    base.update(kw)
    return Block(**base)


def test_framework_block_absent_when_flag_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_BLOCK_ANATOMY", raising=False)
    entry = _block("recap").to_jsonld_entry()
    assert "frameworkBlock" not in entry


# Only block types routed to ``_minimal_block_jsonld`` (the emit site). The
# legacy builders (``_objective_jsonld`` / ``_misconception_jsonld`` /
# ``_section_jsonld`` for explanation/example/concept/summary_takeaway) keep
# their dedicated shapes and don't go through the minimal path.
@pytest.mark.parametrize(
    "block_type,expected",
    [
        ("recap", "B02"),
        ("prereq_set", "B02"),
        ("self_check_question", "B07"),
        ("assessment_item", "B14"),
        ("activity", "B08"),
        ("scenario", "B09"),
        ("discussion_prompt", "B10"),
        ("reflection_prompt", "B11"),
        ("vocab_card", "B12"),
        ("checklist", "B08"),
    ],
)
def test_framework_block_resolves_correct_code(monkeypatch, block_type, expected):
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    entry = _block(block_type).to_jsonld_entry()
    assert entry.get("frameworkBlock") == expected


def test_framework_block_absent_for_chrome(monkeypatch):
    """chrome is non-pedagogical → framework_block_for returns None → no key."""
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    entry = _block("chrome").to_jsonld_entry()
    assert "frameworkBlock" not in entry


def test_framework_block_excluded_from_hash(monkeypatch):
    """Emit is resolved from block_type — never a content field → no hash drift."""
    monkeypatch.delenv("ED4ALL_BLOCK_ANATOMY", raising=False)
    h_off = _block("recap").compute_content_hash()
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    h_on = _block("recap").compute_content_hash()
    assert h_off == h_on


def test_framework_block_pattern_matches_schema(monkeypatch):
    """Emitted code matches the schema pattern ^B(0[1-9]|1[0-5])$."""
    import re

    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    pat = re.compile(r"^B(0[1-9]|1[0-5])$")
    for bt in ("recap", "assessment_item", "scenario", "discussion_prompt"):
        code = _block(bt).to_jsonld_entry()["frameworkBlock"]
        assert pat.match(code), f"{bt} -> {code}"
