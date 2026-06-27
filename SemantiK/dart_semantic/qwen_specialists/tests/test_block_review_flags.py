"""Phase-0 acceptance — SEMANTIK_BLOCK_REVIEW flag quartet (pure resolvers).

The four resolvers are DEAD-BUT-IMPORTABLE scaffolding for the full-block
structural-editor reviewer: nothing in the cascade calls them yet, so the
flag-off path is byte-identical. These tests pin the parse-with-fallback
contract (copied verbatim from ``resolve_structure_review_mode`` /
``resolve_specialist_batch_regions``) and assert byte-stability by proving no
non-test code path references the new symbols.
"""

from __future__ import annotations

from pathlib import Path

from dart_semantic.qwen_specialists.reviewer import (
    resolve_block_review_cache_mode,
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
# Byte-stability — the resolvers are pure reads, unreferenced by any live
# code path (chosen over a full run_structure_review byte-diff because wiring
# a real review call is heavy AND the hard contract for Phase 0 is precisely
# that NOTHING existing calls these symbols yet).
# ---------------------------------------------------------------------------


def test_flag_off_byte_stable(monkeypatch):
    # (a) pure reads: repeated calls with the env unset are stable, with no
    # import-time side effects (the import above already succeeded).
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_WINDOW", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW_CACHE", raising=False)
    assert resolve_block_review_mode() is resolve_block_review_mode()
    assert resolve_block_review_window() == resolve_block_review_window() == 24
    assert resolve_block_review_edge_tokens() == 12
    assert resolve_block_review_cache_mode() is True

    # (b) no LIVE (non-test) code path references the new resolvers or their
    # env vars — the Phase-0 byte-stability guarantee.
    pkg_root = Path(__file__).resolve().parents[3]  # SemantiK/
    new_symbols = (
        "resolve_block_review_mode",
        "resolve_block_review_window",
        "resolve_block_review_edge_tokens",
        "resolve_block_review_cache_mode",
        "SEMANTIK_BLOCK_REVIEW_WINDOW",
        "SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS",
        "SEMANTIK_BLOCK_REVIEW_CACHE",
    )
    reviewer_def = "qwen_specialists/reviewer.py"
    offenders: list[str] = []
    for py in pkg_root.rglob("*.py"):
        rel = py.relative_to(pkg_root).as_posix()
        if "/tests/" in rel or rel.endswith("test_block_review_flags.py"):
            continue
        if rel.endswith(reviewer_def):
            continue  # the definition site itself
        text = py.read_text(encoding="utf-8", errors="ignore")
        for sym in new_symbols:
            if sym in text:
                offenders.append(f"{rel}:{sym}")
    assert not offenders, f"Phase-0 resolvers must be unreferenced: {offenders}"
