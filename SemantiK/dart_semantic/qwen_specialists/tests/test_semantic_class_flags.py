"""Phase-0 acceptance — SEMANTIK_SEMANTIC_CLASS + SEMANTIK_GOLD_SHELL gates.

Two new parse-with-fallback bool resolvers for the gold-standard
semantic-class feature. Both are DEAD-BUT-IMPORTABLE scaffolding: nothing in
the cascade / assembler calls them yet, so the flag-off path is byte-identical
until Phase 4+ (reviewer) / Phase 7+ (assembler) wire them in. These tests pin
the parse-with-fallback contract (copied verbatim from
``resolve_block_review_mode`` / ``resolve_structure_review_mode``).

Byte-stability note: like ``test_block_review_flags.py``, we DON'T grep source
for references — that conflates "appears in source" with "reachable from a live
path" and would false-flag a future dead-but-callable consumer. Instead we
assert the resolvers are pure, stable env reads (no side effects, repeatable);
behavioral flag-off byte-stability is enforced once consumers exist (Phase 4+
reviewer / Phase 7+ assembler each ship their own flag-off byte-stable test).
"""

from __future__ import annotations

from dart_semantic.assembler.shell import resolve_gold_shell_mode
from dart_semantic.qwen_specialists.reviewer import resolve_semantic_class_mode


# ---------------------------------------------------------------------------
# SEMANTIK_SEMANTIC_CLASS (default OFF).
# ---------------------------------------------------------------------------


def test_semantic_class_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SEMANTIC_CLASS", raising=False)
    assert resolve_semantic_class_mode() is False
    for v in ("", "  ", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", v)
        assert resolve_semantic_class_mode() is False


def test_semantic_class_mode_truthy_tokens(monkeypatch):
    for v in ("1", "true", "YES", "on", "  On  "):
        monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", v)
        assert resolve_semantic_class_mode() is True


# ---------------------------------------------------------------------------
# SEMANTIK_GOLD_SHELL (default OFF).
# ---------------------------------------------------------------------------


def test_gold_shell_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    assert resolve_gold_shell_mode() is False
    for v in ("", "  ", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("SEMANTIK_GOLD_SHELL", v)
        assert resolve_gold_shell_mode() is False


def test_gold_shell_mode_truthy_tokens(monkeypatch):
    for v in ("1", "true", "YES", "on", "  On  "):
        monkeypatch.setenv("SEMANTIK_GOLD_SHELL", v)
        assert resolve_gold_shell_mode() is True


# ---------------------------------------------------------------------------
# Resolvers are pure, stable env reads (no side effects, repeatable).
# ---------------------------------------------------------------------------


def test_resolvers_pure_reads(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SEMANTIC_CLASS", raising=False)
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    assert resolve_semantic_class_mode() is resolve_semantic_class_mode() is False
    assert resolve_gold_shell_mode() is resolve_gold_shell_mode() is False
