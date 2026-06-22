"""IB4.6 — ED4ALL_BLOCK_A11Y flag resolver + flag-off emit byte-stability."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.generation.block_a11y import resolve_block_a11y  # noqa: E402
from blocks import Block  # noqa: E402


def test_resolve_false_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_BLOCK_A11Y", raising=False)
    assert resolve_block_a11y() is False


def test_resolve_true_on_truthy(monkeypatch) -> None:
    for token in ("1", "true", "yes", "on", "TRUE", "On"):
        monkeypatch.setenv("ED4ALL_BLOCK_A11Y", token)
        assert resolve_block_a11y() is True


def test_resolve_false_on_garbage(monkeypatch) -> None:
    for token in ("0", "false", "no", "off", "maybe", "", "2"):
        monkeypatch.setenv("ED4ALL_BLOCK_A11Y", token)
        assert resolve_block_a11y() is False


def test_resolve_explicit_override() -> None:
    assert resolve_block_a11y(True) is True
    assert resolve_block_a11y("yes") is True
    assert resolve_block_a11y("garbage") is False
    assert resolve_block_a11y(False) is False


def test_flag_off_block_emit_byte_identical(monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_BLOCK_A11Y", raising=False)
    legacy = Block(
        block_id="page_01#concept_demo_0", block_type="concept",
        page_id="page_01", sequence=0, content="<p>x</p>",
    )
    udl = Block(
        block_id="page_01#concept_demo_0", block_type="concept",
        page_id="page_01", sequence=0, content="<p>x</p>",
        n_representations=3, response_formats=("select",),
        engagement_affordance="real_world",
    )
    assert legacy.to_html_attrs() == udl.to_html_attrs()
    assert legacy.to_jsonld_entry() == udl.to_jsonld_entry()
