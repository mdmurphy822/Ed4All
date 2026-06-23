"""B15 — `resources` block type (Resources / Further Reading).

CPU-pinned, no-GPU, no-model unit tests for the last framework catalog gap:
the B15 ``resources`` block type. Covers token registration + catalog round-trip
to B15 + the deterministic accessible-link-list renderer (gated by
ED4ALL_NEW_BLOCK_TYPES, byte-stable off) + the WCAG 2.4.4 link-purpose
validator (fires on a bad link, passes on a good one, no-op when the flag is
off) + the planner content-shape detector.

PRIMARY gate: byte-stability with ED4ALL_NEW_BLOCK_TYPES unset — the type is
never rendered (renderer returns "") and the validator no-ops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from Courseforge.scripts.blocks import BLOCK_TYPES, Block  # noqa: E402
from lib.generation.block_catalog import load_block_catalog  # noqa: E402
from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES  # noqa: E402
from lib.ontology.framework_blocks import framework_block_for  # noqa: E402
from lib.validators.resource_link_purpose import (  # noqa: E402
    ResourceLinkPurposeValidator,
)


# --------------------------------------------------------------------------- #
# B15.1 — token + catalog round-trip to B15
# --------------------------------------------------------------------------- #
def test_resources_in_block_types():
    assert "resources" in BLOCK_TYPES


def test_resources_constructs():
    blk = Block(
        block_id="week_01_summary#resources_x_0",
        block_type="resources",
        page_id="week_01_summary",
        sequence=0,
        content={"links": [{"url": "https://x", "title": "A title"}]},
    )
    assert blk.block_type == "resources"


def test_resources_maps_to_b15_via_sot_loader():
    assert framework_block_for("resources") == "B15"


def test_resources_has_catalog_entry_with_b15():
    entry = next(
        (e for e in load_block_catalog() if e["block_type"] == "resources"),
        None,
    )
    assert entry is not None
    assert entry["framework_block"] == "B15"
    assert entry.get("use_when")
    assert entry.get("bloom_fit")


def test_b15_no_longer_a_catalog_gap():
    primaries = {
        e.get("framework_block")
        for e in load_block_catalog()
        if e.get("framework_block")
    }
    assert "B15" in primaries


# --------------------------------------------------------------------------- #
# B15.2 — deterministic renderer (gated + accessible)
# --------------------------------------------------------------------------- #
def _resources_block() -> Block:
    return Block(
        block_id="week_01_summary#resources_x_0",
        block_type="resources",
        page_id="week_01_summary",
        sequence=0,
        content={
            "heading": "Further Reading",
            "links": [
                {
                    "url": "https://openstax.org/algebra",
                    "title": "OpenStax Algebra (free open textbook)",
                    "annotation": "the source textbook",
                },
                {"url": "https://example.com/notes", "title": "Course Notes"},
            ],
        },
    )


def test_render_resources_section_accessible_under_flag(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    from Courseforge.scripts.generate_course import _render_resources_section

    html = _render_resources_section(_resources_block())
    assert 'class="resources"' in html
    assert "<ul" in html and "</ul>" in html
    # descriptive anchor text — never a bare URL / "click here"
    assert "OpenStax Algebra" in html
    assert ">https://openstax.org/algebra<" not in html  # not the anchor text
    assert 'href="https://openstax.org/algebra"' in html
    assert "the source textbook" in html  # annotation rendered
    assert "click here" not in html.lower()


def test_render_resources_section_byte_stable_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    from Courseforge.scripts.generate_course import _render_resources_section

    assert _render_resources_section(_resources_block()) == ""


def test_render_resources_titleless_link_not_a_bare_url_anchor(monkeypatch):
    """A link with a URL but no title is shown as plain text, never an anchor
    whose text is the raw URL (the renderer never violates 2.4.4 itself)."""
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    from Courseforge.scripts.generate_course import _render_resources_section

    blk = Block(
        block_id="p#resources_x_0", block_type="resources", page_id="p",
        sequence=0, content={"links": [{"url": "https://bare.example.com"}]},
    )
    html = _render_resources_section(blk)
    assert '<a href="https://bare.example.com">https://bare.example.com</a>' not in html
    assert "resource-url-pending" in html


# --------------------------------------------------------------------------- #
# B15.3 — WCAG 2.4.4 link-purpose validator
# --------------------------------------------------------------------------- #
def test_validator_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    blocks = [
        {
            "block_type": "resources",
            "block_id": "b1",
            "content": {"links": [{"url": "http://x", "title": "click here"}]},
        }
    ]
    r = ResourceLinkPurposeValidator().validate({"blocks": blocks})
    assert r.passed is True
    assert r.action is None
    assert [i.code for i in r.issues] == ["RESOURCE_LINK_PURPOSE_DISABLED"]


def test_validator_fires_on_bad_link(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    blocks = [
        {
            "block_type": "resources",
            "block_id": "b1",
            "content": {
                "links": [
                    {"url": "http://a", "title": "OpenStax Algebra Chapter 2"},
                    {"url": "http://b", "title": "click here"},  # generic
                    {"url": "http://c"},  # titleless -> falls back to URL
                ]
            },
        }
    ]
    r = ResourceLinkPurposeValidator().validate({"blocks": blocks})
    md = r.metadata["resource_link_purpose"]
    assert md["links_audited"] == 3
    assert md["links_unclear"] == 2
    assert r.action == "regenerate"
    assert all(
        i.code == "RESOURCE_LINK_PURPOSE_UNCLEAR" for i in r.issues
    )


def test_validator_passes_on_good_links(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    blocks = [
        {
            "block_type": "resources",
            "block_id": "b1",
            "content": {
                "links": [
                    {"url": "http://a", "title": "Khan Academy: Linear Equations"},
                    {"url": "http://b", "title": "OpenStax Algebra (textbook)"},
                ]
            },
        }
    ]
    r = ResourceLinkPurposeValidator().validate({"blocks": blocks})
    assert r.metadata["resource_link_purpose"]["links_unclear"] == 0
    assert r.action is None
    assert not r.issues


def test_validator_str_html_path(monkeypatch):
    """Rewrite-tier HTML string content is parsed for <a> anchor text."""
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    html = (
        '<section class="resources"><ul>'
        '<li><a href="http://a">A Descriptive Resource Title</a></li>'
        '<li><a href="http://b">read more</a></li>'
        "</ul></section>"
    )
    blocks = [{"block_type": "resources", "block_id": "b2", "content": html}]
    r = ResourceLinkPurposeValidator().validate({"blocks": blocks})
    md = r.metadata["resource_link_purpose"]
    assert md["links_audited"] == 2
    assert md["links_unclear"] == 1  # only "read more"


def test_validator_ignores_non_resources_blocks(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    blocks = [
        {"block_type": "concept", "block_id": "c1",
         "content": '<a href="x">click here</a>'},
    ]
    r = ResourceLinkPurposeValidator().validate({"blocks": blocks})
    assert r.metadata["resource_link_purpose"]["resources_blocks"] == 0
    assert r.action is None


def test_validator_empty_input_graceful_pass(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    r = ResourceLinkPurposeValidator().validate({"blocks": []})
    assert r.passed is True
    assert r.action is None


# --------------------------------------------------------------------------- #
# B15.4 — planner content-shape detector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("See the Further Reading section.", True),
        ("Additional resources are listed below.", True),
        ("References\n1. Smith (2020)", True),
        ("Solve for x in the equation 2x + 3 = 11.", False),
        ("", False),
    ],
)
def test_detect_resources_reference(text, expected):
    from lib.generation.block_planner import detect_resources_reference

    assert detect_resources_reference(text) is expected
