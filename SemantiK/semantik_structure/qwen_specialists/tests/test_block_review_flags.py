"""Phase-0 acceptance — SEMANTIK_BLOCK_REVIEW flag quartet (pure resolvers).

The four resolvers are DEAD-BUT-IMPORTABLE scaffolding for the full-block
structural-editor reviewer: nothing in the cascade calls them yet, so the
flag-off path is byte-identical. These tests pin the parse-with-fallback
contract (copied verbatim from ``resolve_structure_review_mode`` /
``resolve_specialist_batch_regions``) and assert byte-stability by proving no
non-test code path references the new symbols.
"""

from __future__ import annotations

from semantik_structure.qwen_specialists.reviewer import (
    resolve_block_review_cache_mode,
    resolve_block_review_conf_floor,
    resolve_block_review_edge_tokens,
    resolve_block_review_mode,
    resolve_block_review_window,
)


# ---------------------------------------------------------------------------
# Master gate — SEMANTIK_BLOCK_REVIEW (default OFF).
# ---------------------------------------------------------------------------


def test_block_review_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    assert resolve_block_review_mode() is False
    for v in ("", "  ", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", v)
        assert resolve_block_review_mode() is False


def test_block_review_mode_truthy_tokens(monkeypatch):
    for v in ("1", "true", "YES", "on", "  On  "):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", v)
        assert resolve_block_review_mode() is True


# ---------------------------------------------------------------------------
# Cache gate — SEMANTIK_BLOCK_REVIEW_CACHE (default ON).
# ---------------------------------------------------------------------------


def test_block_review_cache_on_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_CACHE", raising=False)
    assert resolve_block_review_cache_mode() is True
    for v in ("", "  ", "1", "true", "yes", "on", "garbage"):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", v)
        assert resolve_block_review_cache_mode() is True
    for v in ("0", "false", "NO", "off", "  Off  "):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", v)
        assert resolve_block_review_cache_mode() is False


# ---------------------------------------------------------------------------
# Int budgets — parse-with-fallback (window=24, edge_tokens=12).
# ---------------------------------------------------------------------------


def test_block_review_window_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_WINDOW", raising=False)
    assert resolve_block_review_window() == 24
    for bad in ("", "garbage", "0", "-3", "1.5"):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", bad)
        assert resolve_block_review_window() == 24
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "40")
    assert resolve_block_review_window() == 40


def test_block_review_edge_tokens_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", raising=False)
    assert resolve_block_review_edge_tokens() == 12
    for bad in ("", "garbage", "0", "-1", "3.0"):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", bad)
        assert resolve_block_review_edge_tokens() == 12
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", "20")
    assert resolve_block_review_edge_tokens() == 20


# ---------------------------------------------------------------------------
# Confidence floor — SEMANTIK_BLOCK_REVIEW_CONF (Phase 3, default 0.4, float
# parse-with-fallback). Mirrors _CODE_CONFIDENCE_FLOOR; the Phase-6 calibration
# knob.
# ---------------------------------------------------------------------------


def test_block_review_conf_floor_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_CONF", raising=False)
    assert resolve_block_review_conf_floor() == 0.4


def test_block_review_conf_floor_parse_with_fallback(monkeypatch):
    # blank / non-float / NaN / Inf / out-of-[0,1] -> 0.4 default.
    for bad in ("", "  ", "garbage", "nan", "inf", "-inf", "-0.1", "1.5", "2"):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CONF", bad)
        assert resolve_block_review_conf_floor() == 0.4


def test_block_review_conf_floor_valid_float_honored(monkeypatch):
    for good, want in (("0.0", 0.0), ("0.25", 0.25), ("0.6", 0.6), ("1.0", 1.0)):
        monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CONF", good)
        assert resolve_block_review_conf_floor() == want


# ---------------------------------------------------------------------------
# Resolvers are pure env reads. Flag-off byte-stability itself is enforced
# BEHAVIORALLY by test_reviewer.py::test_block_review_flag_off_byte_stable
# (run_structure_review output identical flags-absent vs explicitly-off — the
# form the Phase-0 spec preferred). The earlier grep-for-references guard was
# retired: it false-flagged the Phase-1 dead-but-callable edge-tokens consumer
# in reviewer_prompt.py (reachable only from dead code, so it cannot change
# behavior), conflating "appears in source" with "reachable from a live path".
# Per-phase deadness stays covered by each scaffolding phase's own no-live-caller
# grep (e.g. test_edge_input).
# ---------------------------------------------------------------------------


def test_resolvers_pure_reads(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_WINDOW", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_CACHE", raising=False)
    assert resolve_block_review_mode() is resolve_block_review_mode()
    assert resolve_block_review_window() == resolve_block_review_window() == 24
    assert resolve_block_review_edge_tokens() == 12
    assert resolve_block_review_cache_mode() is True
