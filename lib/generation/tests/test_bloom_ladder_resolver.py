"""Bloom-ladder initiative (WI-02) — ED4ALL_BLOOM_LADDER flag resolver.

CPU-pinned, no-GPU, no-model unit tests for the resolver choke point:
``resolve_bloom_ladder`` (default-OFF, truthy-token parse-with-fallback,
mirrors ``resolve_new_block_types`` / ``resolve_block_a11y``) and
``ladder_plan_for_objective`` (thin delegation to
``lib.ontology.bloom_ladder.rungs_up_to``, landed in WI-01).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.generation.bloom_ladder_blocks import (  # noqa: E402
    ENV_BLOOM_LADDER,
    ladder_plan_for_objective,
    resolve_bloom_ladder,
)
from lib.ontology.bloom_ladder import RungEntry, rungs_up_to  # noqa: E402


# --------------------------------------------------------------------------- #
# resolve_bloom_ladder — default OFF, truthy-token parse-with-fallback
# --------------------------------------------------------------------------- #
def test_resolve_false_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    assert resolve_bloom_ladder() is False


def test_resolve_true_on_truthy(monkeypatch) -> None:
    for token in ("1", "true", "yes", "on", "TRUE", "On", "YES"):
        monkeypatch.setenv(ENV_BLOOM_LADDER, token)
        assert resolve_bloom_ladder() is True


def test_resolve_false_on_garbage(monkeypatch) -> None:
    for token in ("0", "false", "no", "off", "maybe", "", "2"):
        monkeypatch.setenv(ENV_BLOOM_LADDER, token)
        assert resolve_bloom_ladder() is False


def test_resolve_explicit_override(monkeypatch) -> None:
    # Explicit value overrides the env var entirely, regardless of what's set.
    monkeypatch.setenv(ENV_BLOOM_LADDER, "0")
    assert resolve_bloom_ladder(True) is True
    assert resolve_bloom_ladder("yes") is True
    assert resolve_bloom_ladder("garbage") is False
    assert resolve_bloom_ladder(False) is False


def test_resolve_never_raises_on_weird_value() -> None:
    class _Unstringable:
        def __str__(self):
            raise RuntimeError("boom")

    assert resolve_bloom_ladder(_Unstringable()) is False


# --------------------------------------------------------------------------- #
# ladder_plan_for_objective — thin delegation to lib.ontology.bloom_ladder
# --------------------------------------------------------------------------- #
def test_ladder_plan_delegates_to_rungs_up_to() -> None:
    assert ladder_plan_for_objective("apply") == rungs_up_to("apply")


def test_ladder_plan_capped_at_ceiling() -> None:
    plan = ladder_plan_for_objective("apply")
    levels = [rung.level for rung in plan]
    assert levels == ["remember", "understand", "apply"]
    assert all(isinstance(rung, RungEntry) for rung in plan)


def test_ladder_plan_single_rung_at_remember() -> None:
    plan = ladder_plan_for_objective("remember")
    assert [rung.level for rung in plan] == ["remember"]


def test_ladder_plan_full_ladder_at_create() -> None:
    plan = ladder_plan_for_objective("create")
    assert [rung.level for rung in plan] == [
        "remember", "understand", "apply", "analyze", "evaluate", "create",
    ]


def test_ladder_plan_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        ladder_plan_for_objective("not_a_bloom_level")


# --------------------------------------------------------------------------- #
# Byte-identical-when-off contract: the resolver + delegation helper are pure
# reads with no side effects, so simply never being called (flag off) is the
# whole byte-stability story for this module — asserted here as a sanity
# check that neither function mutates global/module state on repeated calls.
# --------------------------------------------------------------------------- #
def test_repeated_calls_are_stable(monkeypatch) -> None:
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    assert resolve_bloom_ladder() == resolve_bloom_ladder() is False
    assert ladder_plan_for_objective("evaluate") == ladder_plan_for_objective("evaluate")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
