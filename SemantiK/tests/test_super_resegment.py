"""Unit tests for the standalone Super-120B fused-page re-segmentation stage.

Pure + FAST + NO network: the only network boundary
(``super_resegment._super_completion``) is monkeypatched in every prose test, so
no test ever POSTs. Covers table detection + deterministic conversion, the prose
split PASS / conservation-FAIL / network-error paths, and the three-valued flag
resolver.

The module lives at ``semantik_structure/super_resegment.py``; this test file is
outside the package tree, so it bootstraps ``SemantiK/`` onto ``sys.path`` the
same way the in-package ``conftest.py`` does (there is no shared conftest on this
path).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# .../SemantiK/tests/test_super_resegment.py → parents[1] == SemantiK/
_SEMANTIK_ROOT = Path(__file__).resolve().parents[1]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))

from semantik_structure import super_resegment as sr  # noqa: E402


# ---------------------------------------------------------------------------
# is_table_text
# ---------------------------------------------------------------------------
_MD_TABLE = "| Symbol | Words |\n| --- | --- |\n| = | equals |\n| < | less than |"
_TABULAR = (
    r"\begin{tabular}{cc} Inequality Symbols & Words \\ "
    r"\(a \neq b\) & \(a\) is not equal to \(b\) \\ \end{tabular}"
)
_PROSE = (
    "Introduction to Integers. In this chapter we work with positive and "
    "negative whole numbers. TRY IT Solve for x."
)


def test_is_table_text_true_on_markdown_pipe_table():
    assert sr.is_table_text(_MD_TABLE) is True


def test_is_table_text_true_on_latex_tabular():
    assert sr.is_table_text(_TABULAR) is True


def test_is_table_text_false_on_prose():
    assert sr.is_table_text(_PROSE) is False
    # A hyphenated sentence with no pipe grid must not read as a table.
    assert sr.is_table_text("A well-formed self-check is not a table.") is False


# ---------------------------------------------------------------------------
# latex_tabular_to_rows — the p19 real example.
# ---------------------------------------------------------------------------
def test_latex_tabular_to_rows_p19_example_preserves_latex():
    rows = sr.latex_tabular_to_rows(_TABULAR)
    assert rows is not None
    assert len(rows) == 2
    assert all(len(r) == 2 for r in rows)
    assert rows[0] == ["Inequality Symbols", "Words"]
    # Inline LaTeX preserved VERBATIM (backslashes intact, not decoded away).
    assert rows[1][0] == r"\(a \neq b\)"
    assert rows[1][1] == r"\(a\) is not equal to \(b\)"


def test_latex_tabular_to_rows_none_without_complete_tabular():
    assert sr.latex_tabular_to_rows(r"\begin{tabular}{cc} a & b") is None
    assert sr.latex_tabular_to_rows("just prose, no tabular") is None


# ---------------------------------------------------------------------------
# resegment_page — table branch does NOT call Super.
# ---------------------------------------------------------------------------
def test_resegment_page_table_does_not_call_super(monkeypatch):
    calls = []

    def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Super must not be called on a table input")

    monkeypatch.setattr(sr, "_super_completion", _boom)
    out = sr.resegment_page(_TABULAR, seat=sr.resolve_super_seat(), mode="on")

    assert calls == []  # the mock was never invoked
    assert out["action"] == "table"
    assert out["conserved"] is True
    assert out["elements"] is None
    assert out["rows"] == [
        ["Inequality Symbols", "Words"],
        [r"\(a \neq b\)", r"\(a\) is not equal to \(b\)"],
    ]


# ---------------------------------------------------------------------------
# resegment_page — prose PASS.
# ---------------------------------------------------------------------------
def test_resegment_page_prose_pass_returns_split(monkeypatch):
    fused = "Introduction This chapter covers integers. TRY IT Solve for x."
    tagged = (
        "<<heading>>Introduction\n"
        "<<para>>This chapter covers integers.\n"
        "<<callout>>TRY IT Solve for x.\n"
    )
    monkeypatch.setattr(
        sr,
        "_super_completion",
        lambda seat, system, user, *, timeout: {"content": tagged, "finish": "stop"},
    )
    out = sr.resegment_page(fused, seat=sr.resolve_super_seat(), mode="on")

    assert out["action"] == "split"
    assert out["conserved"] is True
    types = [e["type"] for e in out["elements"]]
    assert types == ["heading", "para", "callout"]


# ---------------------------------------------------------------------------
# resegment_page — prose conservation FAIL (a dropped word).
# ---------------------------------------------------------------------------
def test_resegment_page_prose_conservation_fail_keeps_fused(monkeypatch):
    fused = "Introduction This chapter covers integers. TRY IT Solve for x."
    # Drops the word "integers." — the concatenation no longer matches the page.
    tagged = (
        "<<heading>>Introduction\n"
        "<<para>>This chapter covers.\n"
        "<<callout>>TRY IT Solve for x.\n"
    )
    monkeypatch.setattr(
        sr,
        "_super_completion",
        lambda seat, system, user, *, timeout: {"content": tagged, "finish": "stop"},
    )
    out = sr.resegment_page(fused, seat=sr.resolve_super_seat(), mode="on")

    assert out["action"] == "keep_fused"
    assert out["conserved"] is False
    assert out["elements"] is None
    assert "conserved" in out["why"]


# ---------------------------------------------------------------------------
# resegment_page — network error never propagates.
# ---------------------------------------------------------------------------
def test_resegment_page_network_error_keeps_fused(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(sr, "_super_completion", _raise)
    out = sr.resegment_page(_PROSE, seat=sr.resolve_super_seat(), mode="on")

    assert out["action"] == "keep_fused"
    assert out["conserved"] is False
    assert "super http error" in out["why"]


def test_resegment_page_empty_completion_keeps_fused(monkeypatch):
    monkeypatch.setattr(
        sr,
        "_super_completion",
        lambda seat, system, user, *, timeout: {"content": "", "finish": "stop"},
    )
    out = sr.resegment_page(_PROSE, seat=sr.resolve_super_seat(), mode="on")
    assert out["action"] == "keep_fused"
    assert out["why"] == "empty"


# ---------------------------------------------------------------------------
# shadow flag + off guard.
# ---------------------------------------------------------------------------
def test_resegment_page_shadow_sets_flag(monkeypatch):
    fused = "Introduction This chapter covers integers."
    tagged = "<<heading>>Introduction\n<<para>>This chapter covers integers.\n"
    monkeypatch.setattr(
        sr,
        "_super_completion",
        lambda seat, system, user, *, timeout: {"content": tagged, "finish": "stop"},
    )
    out = sr.resegment_page(fused, seat=sr.resolve_super_seat(), mode="shadow")
    assert out["action"] == "split"
    assert out["shadow"] is True


def test_resegment_page_off_mode_raises():
    with pytest.raises(ValueError):
        sr.resegment_page(_PROSE, mode="off")


# ---------------------------------------------------------------------------
# resolve_super_resegment_mode.
# ---------------------------------------------------------------------------
def test_resolve_mode_default_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SUPER_RESEGMENT", raising=False)
    assert sr.resolve_super_resegment_mode() == "off"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("on", "on"),
        ("ON", "on"),
        ("  shadow  ", "shadow"),
        ("SHADOW", "shadow"),
        ("off", "off"),
        ("0", "off"),
        ("false", "off"),
        ("garbage", "off"),
        ("", "off"),
    ],
)
def test_resolve_mode_env_variants(monkeypatch, value, expected):
    monkeypatch.setenv("SEMANTIK_SUPER_RESEGMENT", value)
    assert sr.resolve_super_resegment_mode() == expected


# ---------------------------------------------------------------------------
# resolve_super_seat.
# ---------------------------------------------------------------------------
def test_resolve_super_seat_defaults(monkeypatch):
    for k in ("SEMANTIK_SUPER_BASE_URL", "SEMANTIK_SUPER_MODEL", "SEMANTIK_SUPER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    seat = sr.resolve_super_seat()
    assert seat.base_url == "http://localhost:8001/v1"
    assert seat.model == "nemotron-3-super"
    assert seat.api_key is None


def test_resolve_super_seat_env_override(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SUPER_BASE_URL", "http://gpu:9000/v1")
    monkeypatch.setenv("SEMANTIK_SUPER_MODEL", "super-120b")
    monkeypatch.setenv("SEMANTIK_SUPER_API_KEY", "tok")
    seat = sr.resolve_super_seat()
    assert seat == sr.SuperSeat("http://gpu:9000/v1", "super-120b", "tok")
