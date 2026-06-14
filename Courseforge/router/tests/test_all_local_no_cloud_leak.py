"""W1 (backend local wiring) — static no-cloud-leak resolution + model-agnostic tests.

These are the CI-runnable (no live server) analogues of the all-local smoke's
cloud-leak grep assertions:

- T4: walk every one of the 16 ``BLOCK_TYPES`` through both tiers (outline +
  rewrite) of the all-local routing variant the single switch selects, and
  assert NO resolved capability tier is ``anthropic`` and every resolved
  ``base_url`` is loopback. This is the static analogue of the smoke's
  ``grep ... provider_label.*anthropic`` and ``base_url ... non-loopback``
  checks (§3.4 b/c of the W1 spec).
- T5: the ``WorkflowRunner._apply_authoring_route_env`` method fills exactly
  ``AGENT_AUTHORING_PROVIDER_ENV_MAP.values()`` — a drift guard mirroring the
  GUI ``_AUTHORING_PROVIDER_ENVS`` drift guard so the two helpers can't diverge.
- T6: model-agnostic — a non-default ``LOCAL_SYNTHESIS_MODEL`` env value flows
  through the registry unchanged (no hard-coded model pin leaks through the
  resolution chain).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.policy import (  # noqa: E402
    load_block_routing_policy,
)
from Courseforge.scripts.blocks import BLOCK_TYPES  # noqa: E402

_ALL_LOCAL_PATH: Path = (
    PROJECT_ROOT / "Courseforge" / "config" / "block_routing.license_clean.yaml"
)

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1")


def _is_loopback(base_url: str) -> bool:
    if not base_url:
        # An unset base_url defaults to the local registry default
        # (http://localhost:11434/v1) downstream — not a cloud leak.
        return True
    return any(host in base_url for host in _LOOPBACK_HOSTS)


def _resolve_tier_chain(policy, block_type: str, tier: str):
    """Return the ordered list of capability-tier names for (block_type, tier).

    Per-block-type chain wins over the tier-default chain (mirrors the
    router's ``_resolve_capability_tier_chain`` priority order).
    """
    chain = policy.capability_tier_chain_by_block_type.get((block_type, tier))
    if chain:
        return chain
    return policy.capability_tier_chain_by_default_tier.get(tier, [])


# ---------------------------------------------------------------------------
# T4 — no-cloud-leak resolution over all 16 BLOCK_TYPES x both tiers
# ---------------------------------------------------------------------------


def test_all_local_routing_has_no_anthropic_or_nonloopback_tier():
    """Every capability tier in the all-local variant is non-anthropic + loopback."""
    policy = load_block_routing_policy(_ALL_LOCAL_PATH)
    assert not policy.is_empty()
    assert policy.capability_tiers, "all-local variant must define capability_tiers"

    for tier_name, spec in policy.capability_tiers.items():
        assert spec.get("provider") != "anthropic", (
            f"capability tier {tier_name!r} routes anthropic — cloud leak"
        )
        assert _is_loopback(spec.get("base_url", "")), (
            f"capability tier {tier_name!r} base_url {spec.get('base_url')!r} "
            f"is not loopback — cloud leak"
        )


def test_every_block_type_both_tiers_resolves_no_cloud():
    """Walk all 16 BLOCK_TYPES x {outline, rewrite}; no tier resolves anthropic."""
    policy = load_block_routing_policy(_ALL_LOCAL_PATH)
    tiers = policy.capability_tiers

    checked = 0
    for block_type in BLOCK_TYPES:
        for seam in ("outline", "rewrite"):
            chain = _resolve_tier_chain(policy, block_type, seam)
            assert chain, (
                f"{block_type}/{seam} resolved an empty capability-tier chain"
            )
            for tier_name in chain:
                spec = tiers.get(tier_name)
                assert spec is not None, (
                    f"{block_type}/{seam} references unknown tier {tier_name!r}"
                )
                assert spec.get("provider") != "anthropic", (
                    f"{block_type}/{seam} -> tier {tier_name!r} routes anthropic"
                )
                assert _is_loopback(spec.get("base_url", "")), (
                    f"{block_type}/{seam} -> tier {tier_name!r} base_url "
                    f"{spec.get('base_url')!r} is not loopback"
                )
                checked += 1

    # Sanity: we actually walked all 16 block types (not a silently-empty loop).
    assert len(BLOCK_TYPES) == 16
    assert checked >= 16 * 2


# ---------------------------------------------------------------------------
# T5 — drift guard for the WorkflowRunner authoring-route method
# ---------------------------------------------------------------------------


def test_workflow_runner_fills_exactly_the_authoring_env_map(
    monkeypatch: pytest.MonkeyPatch,
):
    """``_apply_authoring_route_env`` fills exactly AGENT_AUTHORING_PROVIDER_ENV_MAP.

    Mirrors the GUI ``_AUTHORING_PROVIDER_ENVS`` drift guard
    (``gui/tests/test_run_service.py``) so the CLI and GUI helpers can't
    diverge in which envs they fill.
    """
    from MCP.core.executor import AGENT_AUTHORING_PROVIDER_ENV_MAP
    from MCP.core.workflow_runner import (
        _TWO_PASS_TIER_PROVIDER_ENVS,
        WorkflowRunner,
    )

    expected = set(AGENT_AUTHORING_PROVIDER_ENV_MAP.values())
    # Clear the four authoring envs + the two-pass envs so the single-pass
    # call fills only the authoring set.
    for env_var in (
        "LLM_PROVIDER",
        "COURSEFORGE_TWO_PASS",
        "COURSEFORGE_BLOCK_ROUTING_PATH",
        *expected,
        *_TWO_PASS_TIER_PROVIDER_ENVS,
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "local")

    runner = WorkflowRunner(executor=MagicMock(), config=MagicMock())
    # Single-pass (two-pass unset) -> the helper fills ONLY the four authoring
    # envs, so `applied` is exactly the authoring map.
    applied = runner._apply_authoring_route_env("textbook_to_course", "")

    assert set(applied.keys()) == expected


# ---------------------------------------------------------------------------
# T6 — model-agnostic: a non-default local model flows through the registry
# ---------------------------------------------------------------------------


def test_local_model_env_overrides_registry_default(monkeypatch: pytest.MonkeyPatch):
    """LOCAL_SYNTHESIS_MODEL flows through the registry; no hard-coded pin leaks.

    The single switch changes only WHICH registry entry (`local`) is selected;
    the model is resolved via the registry's ``model_env`` (LOCAL_SYNTHESIS_MODEL)
    with the registry default as the floor. Setting the env to a non-default
    model id must resolve to that id, proving model-agnosticism.
    """
    from MCP.orchestrator.llm_backend import resolve_openai_compatible_backend

    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "some-other-model:99b")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    backend = resolve_openai_compatible_backend("local")

    assert backend.default_model == "some-other-model:99b", (
        f"local model did not flow through the registry; got "
        f"{backend.default_model!r}"
    )
    assert backend.provider_label == "local"
    # And the default (no env) resolves to the registry default — not anthropic.
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    default_backend = resolve_openai_compatible_backend("local")
    assert "claude" not in default_backend.default_model.lower()
    assert default_backend.provider_label != "anthropic"
