"""Task #24 — scholarly / admonition composite units at the adapter render seam.

Synthetic block IR only (no corpus files). The ``scholarly`` and ``admonition``
lexicon profiles are opt-in overlays; the adapter reads the active lexicon's
opener roles through the ``OPENER_ROLES`` / ``OPENER_ASSOCIATION_ROLE`` snapshots
it imports from :mod:`lib.semantik.opener_classifier` at import time. Rather than
reload the whole adapter (fragile — pollutes shared module state for other
tests), these tests ``monkeypatch`` those two names with the vocabulary drawn
straight from the real lexicon profiles, so the exercised roles are exactly what
a ``SEMANTIK_LEXICON_PROFILE=…+scholarly+admonition`` run would produce.

The adapter itself is profile-AGNOSTIC (no adapter code changed) — the two new
composite-unit types flow purely from the lexicon profile + the
``composite_units.plan_units`` grammar.
"""
from __future__ import annotations

import re

import pytest

from lib.ontology.taxonomy import get_lexicon_openers
from lib.semantik import adapter as A
from lib.semantik.adapter import _AdapterBlock, _AdapterChapter, _render_chapters


def _activate_profiles(monkeypatch, spec: str) -> None:
    """Inject a profile spec's opener roles into the adapter's imported snapshots.

    Mirrors what the import-time resolution would yield under
    ``SEMANTIK_LEXICON_PROFILE=spec`` — real lexicon data, auto-restored by
    monkeypatch after the test.
    """
    openers = get_lexicon_openers(spec)
    roles = frozenset(o["role"] for o in openers)
    assoc = {o["role"]: o.get("association_role", o["role"]) for o in openers}
    monkeypatch.setattr(A, "OPENER_ROLES", A.OPENER_ROLES | roles)
    monkeypatch.setattr(
        A, "OPENER_ASSOCIATION_ROLE", {**A.OPENER_ASSOCIATION_ROLE, **assoc}
    )


def _opener(text, idx, role):
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx, heading_level=4,
        raw_text=text, heading_text=text, block_role=role,
    )


def _para(text, idx):
    return _AdapterBlock(
        html=f"<p>{text}</p>", region_kind="paragraph", raw_block_index=idx,
        raw_text=text, heading_text=None,
    )


def _section(text, idx):
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx, heading_level=3,
        raw_text=text, heading_text=text,
    )


# ---------------------------------------------------------------------------
# (a) scholarly -> theorem_block
# ---------------------------------------------------------------------------


def test_theorem_block_wraps_statement_proof_corollary(monkeypatch):
    _activate_profiles(monkeypatch, "generic-academic+scholarly")
    ch = _AdapterChapter(
        title="Groups",
        blocks=[
            _opener("Theorem 3.1", 0, "theorem"),
            _para("Every finite group has a subgroup.", 1),
            _opener("Proof", 2, "proof"),
            _para("By Lagrange's theorem.", 3),
            _opener("Corollary 3.2", 4, "corollary"),
            _para("Hence prime-order groups are cyclic.", 5),
        ],
    )
    html = _render_chapters([ch])
    assert html.count('data-dart-unit="theorem_block"') == 1
    assert 'role="group"' in html
    # aria-labelledby -> the theorem STATEMENT heading (the block lead).
    assert 'aria-labelledby="theorem-3-1"' in html
    # All three callout boxes (statement, proof, corollary) live inside the unit.
    unit = re.search(
        r'<section class="dart-unit dart-unit-theorem_block"[^>]*>(.*)',
        html, re.DOTALL,
    ).group(1)
    assert unit.count("data-dart-opener-group=") == 3


def test_lone_theorem_not_wrapped(monkeypatch):
    _activate_profiles(monkeypatch, "generic-academic+scholarly")
    ch = _AdapterChapter(
        title="Lone",
        blocks=[
            _opener("Theorem 1.1", 0, "theorem"),
            _para("A statement with no proof.", 1),
        ],
    )
    html = _render_chapters([ch])
    assert "data-dart-unit=" not in html
    assert 'data-dart-opener-group="theorem"' in html  # callout box still renders


def test_theorem_block_never_crosses_section_heading(monkeypatch):
    _activate_profiles(monkeypatch, "generic-academic+scholarly")
    ch = _AdapterChapter(
        title="Cross",
        blocks=[
            _opener("Theorem 2.1", 0, "theorem"),
            _para("statement body", 1),
            _section("2.2 New Section", 2),
            _opener("Proof", 3, "proof"),
            _para("proof body", 4),
        ],
    )
    html = _render_chapters([ch])
    assert "data-dart-unit=" not in html
    assert "<h3" in html and "New Section" in html


def test_definition_group_reuses_existing_shape(monkeypatch):
    # A scholarly Definition opener (association_role "definition") + an Example
    # forms the existing definition_group unit — no new grammar, pure reuse.
    _activate_profiles(monkeypatch, "generic-academic+scholarly")
    ch = _AdapterChapter(
        title="Defs",
        blocks=[
            _opener("Definition 1.1", 0, "definition"),
            _para("A group is a set with an associative operation.", 1),
            _opener("Example 1.2", 2, "worked_example"),
            _para("The integers under addition.", 3),
        ],
    )
    html = _render_chapters([ch])
    assert 'data-dart-unit="definition_group"' in html


# ---------------------------------------------------------------------------
# (b) admonition -> admonition (singleton)
# ---------------------------------------------------------------------------


def test_admonition_wraps_label_plus_body(monkeypatch):
    _activate_profiles(monkeypatch, "generic-academic+admonition")
    ch = _AdapterChapter(
        title="Docs",
        blocks=[
            _opener("Warning", 0, "warning"),
            _para("Never run this as root.", 1),
        ],
    )
    html = _render_chapters([ch])
    assert 'data-dart-unit="admonition"' in html
    assert 'role="group"' in html
    assert 'aria-labelledby="warning"' in html  # the label names the unit


def test_two_admonitions_stay_separate_units(monkeypatch):
    _activate_profiles(monkeypatch, "generic-academic+admonition")
    ch = _AdapterChapter(
        title="Docs",
        blocks=[
            _opener("Note", 0, "note"),
            _para("First remark.", 1),
            _opener("Tip", 2, "tip"),
            _para("A helpful hint.", 3),
        ],
    )
    html = _render_chapters([ch])
    assert html.count('data-dart-unit="admonition"') == 2


def test_bare_admonition_label_not_wrapped(monkeypatch):
    # A label with no following body carries members == 1 -> never a unit.
    _activate_profiles(monkeypatch, "generic-academic+admonition")
    ch = _AdapterChapter(
        title="Docs",
        blocks=[
            _opener("Note", 0, "note"),
            _section("Real Section", 1),
        ],
    )
    html = _render_chapters([ch])
    assert "data-dart-unit=" not in html


# ---------------------------------------------------------------------------
# (c) default profile is byte-unchanged — the new roles are inert.
# ---------------------------------------------------------------------------


def test_default_profile_ignores_scholarly_roles():
    # WITHOUT the profile activation, a "theorem"-role block is NOT a recognized
    # opener (absent from OPENER_ROLES), so it is treated as a genuine heading and
    # no theorem_block unit forms — proving the existing profiles are untouched.
    ch = _AdapterChapter(
        title="Default",
        blocks=[
            _AdapterBlock(
                html="", region_kind="heading", raw_block_index=0, heading_level=4,
                raw_text="Theorem 3.1", heading_text="Theorem 3.1", block_role="theorem",
            ),
            _para("body", 1),
        ],
    )
    html = _render_chapters([ch])
    assert "data-dart-unit=" not in html


def test_adapter_globals_are_pristine_after_monkeypatch():
    # Sanity: outside the monkeypatched tests the snapshots carry only the default
    # vocabulary (no leakage into the shared module state).
    assert "theorem" not in A.OPENER_ROLES
    assert "note" not in A.OPENER_ROLES
