"""Tests for the env-driven per-tier ``max_tokens`` DEFAULT resolvers.

Prior to this change the router constructed each tier provider with an
explicit ``max_tokens=spec.max_tokens`` kwarg that WON over the generators'
own ``COURSEFORGE_OUTLINE_MAX_TOKENS`` / ``COURSEFORGE_REWRITE_MAX_TOKENS``
resolution, and ``spec.max_tokens`` came from a hardcoded ``1200`` / ``2400``
literal in two places (``router._build_hardcoded_defaults`` +
``policy._spec_from_dict``). These tests pin the fix: the DEFAULT is now
env-driven (outline 4096 / rewrite 6144), while a per-block ``max_tokens``
override in ``block_routing.yaml`` still wins over the env default.

Precedence (high → low): YAML / per-block override > env > resolver default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.router import (  # noqa: E402
    CourseforgeRouter,
    _DEFAULT_OUTLINE_MAX_TOKENS,
    _DEFAULT_REWRITE_MAX_TOKENS,
    _build_hardcoded_defaults,
    resolve_outline_max_tokens_default,
    resolve_rewrite_max_tokens_default,
)
from Courseforge.router.policy import _spec_from_dict  # noqa: E402
from blocks import Block  # noqa: E402


_OUTLINE_ENV = "COURSEFORGE_OUTLINE_MAX_TOKENS"
_REWRITE_ENV = "COURSEFORGE_REWRITE_MAX_TOKENS"


def _block(*, block_type: str = "concept") -> Block:
    return Block(
        block_id=f"page1#{block_type}_intro_0",
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content="hello",
    )


def _clear(monkeypatch) -> None:
    monkeypatch.delenv(_OUTLINE_ENV, raising=False)
    monkeypatch.delenv(_REWRITE_ENV, raising=False)


# ---------------------------------------------------------------------------
# Resolver defaults / env / garbage
# ---------------------------------------------------------------------------


def test_resolver_defaults_when_env_unset(monkeypatch):
    _clear(monkeypatch)
    assert resolve_outline_max_tokens_default() == 4096
    assert resolve_rewrite_max_tokens_default() == 6144
    # Module constants agree with the documented defaults.
    assert _DEFAULT_OUTLINE_MAX_TOKENS == 4096
    assert _DEFAULT_REWRITE_MAX_TOKENS == 6144


def test_resolver_reads_positive_int_env(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(_OUTLINE_ENV, "6144")
    monkeypatch.setenv(_REWRITE_ENV, "8000")
    assert resolve_outline_max_tokens_default() == 6144
    assert resolve_rewrite_max_tokens_default() == 8000


def test_resolver_garbage_and_non_positive_fall_back(monkeypatch):
    _clear(monkeypatch)
    for bad in ("garbage", "0", "-5", "", "  ", "3.5"):
        monkeypatch.setenv(_OUTLINE_ENV, bad)
        monkeypatch.setenv(_REWRITE_ENV, bad)
        assert resolve_outline_max_tokens_default() == 4096
        assert resolve_rewrite_max_tokens_default() == 6144


# ---------------------------------------------------------------------------
# Hardcoded-defaults table is env-driven (read at build time)
# ---------------------------------------------------------------------------


def test_hardcoded_defaults_table_reflects_env(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(_OUTLINE_ENV, "5000")
    monkeypatch.setenv(_REWRITE_ENV, "9000")
    table = _build_hardcoded_defaults()
    assert table[("concept", "outline")].max_tokens == 5000
    assert table[("concept", "rewrite")].max_tokens == 9000
    # Anthropic-routed rewrite blocks resolve the same env default.
    assert table[("assessment_item", "rewrite")].max_tokens == 9000


def test_hardcoded_defaults_table_uses_defaults_when_unset(monkeypatch):
    _clear(monkeypatch)
    table = _build_hardcoded_defaults()
    assert table[("concept", "outline")].max_tokens == 4096
    assert table[("concept", "rewrite")].max_tokens == 6144


# ---------------------------------------------------------------------------
# policy._spec_from_dict — env default vs per-block YAML override
# ---------------------------------------------------------------------------


def test_spec_from_dict_uses_env_default_when_yaml_omits_max_tokens(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(_OUTLINE_ENV, "6144")
    monkeypatch.setenv(_REWRITE_ENV, "7777")
    outline = _spec_from_dict(
        {"provider": "local"}, block_type="concept", tier="outline"
    )
    rewrite = _spec_from_dict(
        {"provider": "local"}, block_type="concept", tier="rewrite"
    )
    assert outline.max_tokens == 6144
    assert rewrite.max_tokens == 7777


def test_spec_from_dict_per_block_max_tokens_override_wins(monkeypatch):
    """An explicit per-block ``max_tokens`` in YAML beats the env default."""
    _clear(monkeypatch)
    monkeypatch.setenv(_OUTLINE_ENV, "6144")
    spec = _spec_from_dict(
        {"provider": "local", "max_tokens": 999},
        block_type="concept",
        tier="outline",
    )
    assert spec.max_tokens == 999


# ---------------------------------------------------------------------------
# End-to-end: the router's resolved spec.max_tokens reflects the env
# (this is the value fed to the tier provider as an explicit kwarg).
# ---------------------------------------------------------------------------


def test_resolve_spec_max_tokens_reflects_env_no_policy(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(_OUTLINE_ENV, "6144")
    monkeypatch.setenv(_REWRITE_ENV, "8192")
    r = CourseforgeRouter()  # no YAML policy
    outline_spec = r._resolve_spec(_block(), "outline")
    rewrite_spec = r._resolve_spec(_block(), "rewrite")
    assert outline_spec.max_tokens == 6144
    assert rewrite_spec.max_tokens == 8192


def test_resolve_spec_max_tokens_default_when_env_unset(monkeypatch):
    _clear(monkeypatch)
    r = CourseforgeRouter()
    assert r._resolve_spec(_block(), "outline").max_tokens == 4096
    assert r._resolve_spec(_block(), "rewrite").max_tokens == 6144


class _StubPolicy:
    def __init__(self, spec) -> None:
        self._spec = spec

    def resolve(self, block_id: str, block_type: str, tier: str) -> Any:
        return self._spec


def test_resolve_spec_yaml_per_block_max_tokens_wins_over_env(monkeypatch):
    """A YAML per-block spec's max_tokens wins over the env default."""
    _clear(monkeypatch)
    monkeypatch.setenv(_OUTLINE_ENV, "6144")
    yaml_spec = _spec_from_dict(
        {"provider": "local", "max_tokens": 1500},
        block_type="concept",
        tier="outline",
    )
    r = CourseforgeRouter(policy=_StubPolicy(yaml_spec))
    spec = r._resolve_spec(_block(), "outline")
    assert spec.max_tokens == 1500
