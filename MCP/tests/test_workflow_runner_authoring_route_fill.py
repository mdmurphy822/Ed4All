"""W1 (backend local wiring) — tests for the CLI single-switch authoring-route fill.

``WorkflowRunner._apply_authoring_route_env`` is the CLI-side parity of the GUI
``run_service._apply_authoring_route_env``: it fills the four blessed
``AGENT_AUTHORING_PROVIDER_ENV_MAP`` envs (Gap C) so a bare
``ed4all run --provider local`` no longer hard-fails at
``_enforce_authoring_provider_route``, and (Gap A) redirects the two-pass
block-routing policy to the all-local variant + fills the tier-default
rewrite/outline provider envs when the resolved provider is local / ToS-clean
OSS and ``COURSEFORGE_TWO_PASS=true``.

These tests drive the helper directly — no orchestrator, no LLM, no live
providers — so they run in CI with no live server.

- T1: the four authoring envs fill from the provider hint / LLM_PROVIDER /
  local default; explicit envs win; non-pipeline workflows no-op.
- T3: the two-pass block-routing redirect + tier-env fill, operator-override
  honored, and the loaded variant resolves a non-anthropic ``large`` tier.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from MCP.core.executor import AGENT_AUTHORING_PROVIDER_ENV_MAP
from MCP.core.workflow_runner import (
    _ALL_LOCAL_BLOCK_ROUTING_PATH,
    _NVIDIA_LARGE_BLOCK_ROUTING_PATH,
    _NVIDIA_LARGE_MODEL_DEFAULT,
    _NVIDIA_LARGE_MODEL_ENV,
    _TEXTBOOK_SYNTHESIS_MODEL_ROUTE_ENV,
    _TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV,
    _TWO_PASS_TIER_PROVIDER_ENVS,
    WorkflowRunner,
)

# Every env this helper can read or write — cleared per test so resolution
# sees a known baseline regardless of the ambient environment.
_ALL_ROUTE_ENVS = (
    "LLM_PROVIDER",
    "COURSEFORGE_TWO_PASS",
    "COURSEFORGE_BLOCK_ROUTING_PATH",
    _NVIDIA_LARGE_MODEL_ENV,
    _TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV,
    _TEXTBOOK_SYNTHESIS_MODEL_ROUTE_ENV,
    "TRAINFORGE_SYNTHESIS_PROVIDER",
    *AGENT_AUTHORING_PROVIDER_ENV_MAP.values(),
    *_TWO_PASS_TIER_PROVIDER_ENVS,
)


def _make_runner() -> WorkflowRunner:
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _clear_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _ALL_ROUTE_ENVS:
        monkeypatch.delenv(env_var, raising=False)


# ---------------------------------------------------------------------------
# T1 — Gap C: the four blessed authoring-route envs fill from the single switch
# ---------------------------------------------------------------------------


def test_fills_all_four_authoring_envs_from_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    """Env clean except LLM_PROVIDER=local -> all four authoring envs == local."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "")

    for env_var in AGENT_AUTHORING_PROVIDER_ENV_MAP.values():
        import os

        assert os.environ[env_var] == "local", env_var
        assert applied[env_var] == "local"


def test_fills_all_four_authoring_envs_from_local_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """No provider hint, no LLM_PROVIDER -> falls to the license-clean local default."""
    import os

    _clear_envs(monkeypatch)
    runner = _make_runner()

    runner._apply_authoring_route_env("textbook_to_course", "")

    for env_var in AGENT_AUTHORING_PROVIDER_ENV_MAP.values():
        assert os.environ[env_var] == "local", env_var


def test_provider_hint_wins_over_llm_provider(monkeypatch: pytest.MonkeyPatch):
    """provider_hint='together' with no LLM_PROVIDER -> all four == together."""
    import os

    _clear_envs(monkeypatch)
    runner = _make_runner()

    runner._apply_authoring_route_env("textbook_to_course", "together")

    for env_var in AGENT_AUTHORING_PROVIDER_ENV_MAP.values():
        assert os.environ[env_var] == "together", env_var


def test_explicit_env_is_not_overwritten(monkeypatch: pytest.MonkeyPatch):
    """A pre-set authoring env (operator/GUI) is honored verbatim (setdefault)."""
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "together")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "")

    # The explicit value wins; the helper did not touch it.
    assert os.environ["COURSEFORGE_PROVIDER"] == "together"
    assert "COURSEFORGE_PROVIDER" not in applied
    # The other three still filled from LLM_PROVIDER.
    assert os.environ["COURSEPLANNER_PROVIDER"] == "local"


def test_non_pipeline_workflow_is_noop(monkeypatch: pytest.MonkeyPatch):
    """A non-pipeline workflow (rag_training) -> no-op, returns {}."""
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("rag_training", "")

    assert applied == {}
    for env_var in AGENT_AUTHORING_PROVIDER_ENV_MAP.values():
        assert env_var not in os.environ, env_var


# ---------------------------------------------------------------------------
# T3 — Gap A: two-pass block-routing redirect + tier-env fill
# ---------------------------------------------------------------------------


def test_two_pass_redirects_block_routing_and_fills_tier_envs(
    monkeypatch: pytest.MonkeyPatch,
):
    """LLM_PROVIDER=local + COURSEFORGE_TWO_PASS=true, path unset ->
    redirect COURSEFORGE_BLOCK_ROUTING_PATH to the all-local variant AND fill
    COURSEFORGE_REWRITE_PROVIDER / COURSEFORGE_OUTLINE_PROVIDER == local.
    """
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "")

    assert os.environ["COURSEFORGE_BLOCK_ROUTING_PATH"] == _ALL_LOCAL_BLOCK_ROUTING_PATH
    assert applied["COURSEFORGE_BLOCK_ROUTING_PATH"] == _ALL_LOCAL_BLOCK_ROUTING_PATH
    for env_var in _TWO_PASS_TIER_PROVIDER_ENVS:
        assert os.environ[env_var] == "local", env_var
        assert applied[env_var] == "local"


def test_two_pass_operator_block_routing_path_is_honored(
    monkeypatch: pytest.MonkeyPatch,
):
    """A pre-set COURSEFORGE_BLOCK_ROUTING_PATH is left untouched (setdefault)."""
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", "/x.yaml")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "")

    assert os.environ["COURSEFORGE_BLOCK_ROUTING_PATH"] == "/x.yaml"
    assert "COURSEFORGE_BLOCK_ROUTING_PATH" not in applied


def test_no_two_pass_means_no_redirect(monkeypatch: pytest.MonkeyPatch):
    """COURSEFORGE_TWO_PASS unset -> single-pass path needs no routing file."""
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "")

    assert "COURSEFORGE_BLOCK_ROUTING_PATH" not in os.environ
    assert "COURSEFORGE_BLOCK_ROUTING_PATH" not in applied
    for env_var in _TWO_PASS_TIER_PROVIDER_ENVS:
        assert env_var not in os.environ, env_var


def test_cloud_provider_does_not_redirect_block_routing(
    monkeypatch: pytest.MonkeyPatch,
):
    """A cloud (anthropic) provider with two-pass on -> no all-local redirect.

    The redirect is gated on the resolved provider being local / ToS-clean
    OSS; anthropic stays on the canonical routing (unchanged cloud path).
    """
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    runner = _make_runner()

    runner._apply_authoring_route_env("textbook_to_course", "")

    assert "COURSEFORGE_BLOCK_ROUTING_PATH" not in os.environ
    for env_var in _TWO_PASS_TIER_PROVIDER_ENVS:
        assert env_var not in os.environ, env_var


def test_redirected_routing_file_resolves_non_anthropic_large_tier(
    monkeypatch: pytest.MonkeyPatch,
):
    """The redirected variant's `large` capability tier must not be anthropic.

    Closes the cloud-leak at the routing layer: load the file the single
    switch points at and assert no capability tier is anthropic.
    """
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    runner = _make_runner()
    runner._apply_authoring_route_env("textbook_to_course", "")

    from Courseforge.router.policy import load_block_routing_policy

    project_root = Path(__file__).resolve().parents[2]
    policy = load_block_routing_policy(project_root / _ALL_LOCAL_BLOCK_ROUTING_PATH)
    assert not policy.is_empty()
    for tier_name, spec in policy.capability_tiers.items():
        assert spec.get("provider") != "anthropic", tier_name


# ---------------------------------------------------------------------------
# NVIDIA-70b-everywhere — the THIRD (nvidia) branch
# ---------------------------------------------------------------------------


def test_nvidia_branch_sets_routing_model_and_synthesis(
    monkeypatch: pytest.MonkeyPatch,
):
    """provider=nvidia + two-pass on -> nvidia routing YAML + 70B model +
    TEXTBOOK_SYNTHESIS_PROVIDER=nvidia, AND leaves TRAINFORGE_SYNTHESIS_PROVIDER
    untouched (GAP-3 licensing guard).
    """
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "nvidia")

    # (1) routing YAML redirected to the nvidia-large variant.
    assert os.environ["COURSEFORGE_BLOCK_ROUTING_PATH"] == _NVIDIA_LARGE_BLOCK_ROUTING_PATH
    assert applied["COURSEFORGE_BLOCK_ROUTING_PATH"] == _NVIDIA_LARGE_BLOCK_ROUTING_PATH
    # (2) the 70B model pinned (closes the 30B-nano registry leak).
    assert os.environ[_NVIDIA_LARGE_MODEL_ENV] == _NVIDIA_LARGE_MODEL_DEFAULT
    assert "nano" not in os.environ[_NVIDIA_LARGE_MODEL_ENV].lower()
    # (3) GAP-1: the synthesis seat reaches nvidia + the 70B.
    assert os.environ[_TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV] == "nvidia"
    assert os.environ[_TEXTBOOK_SYNTHESIS_MODEL_ROUTE_ENV] == _NVIDIA_LARGE_MODEL_DEFAULT
    # (4) GAP-3: the TRAINING seat is pinned LOCAL (NEVER nvidia) so the SLM
    #     training corpus can never route through Llama-3.3.
    assert os.environ["TRAINFORGE_SYNTHESIS_PROVIDER"] == "local"
    assert applied["TRAINFORGE_SYNTHESIS_PROVIDER"] == "local"
    # The OTHER three authoring envs filled with nvidia.
    for agent, env_var in AGENT_AUTHORING_PROVIDER_ENV_MAP.items():
        if agent == "training-synthesizer":
            continue
        assert os.environ[env_var] == "nvidia", env_var


def test_nvidia_branch_setdefault_explicit_overrides_win(
    monkeypatch: pytest.MonkeyPatch,
):
    """Explicit per-phase env exports win over the nvidia branch (setdefault)."""
    import os

    _clear_envs(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    # Operator pins the routing path + model + synthesis seat explicitly.
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", "/custom.yaml")
    monkeypatch.setenv(_NVIDIA_LARGE_MODEL_ENV, "meta/llama-3.3-70b-custom")
    monkeypatch.setenv(_TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV, "local")
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "nvidia")

    assert os.environ["COURSEFORGE_BLOCK_ROUTING_PATH"] == "/custom.yaml"
    assert "COURSEFORGE_BLOCK_ROUTING_PATH" not in applied
    assert os.environ[_NVIDIA_LARGE_MODEL_ENV] == "meta/llama-3.3-70b-custom"
    assert _NVIDIA_LARGE_MODEL_ENV not in applied
    # The explicit synthesis-seat override is honored.
    assert os.environ[_TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV] == "local"
    assert _TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV not in applied


def test_nvidia_branch_noop_without_two_pass(monkeypatch: pytest.MonkeyPatch):
    """provider=nvidia but COURSEFORGE_TWO_PASS unset -> no routing redirect."""
    import os

    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_authoring_route_env("textbook_to_course", "nvidia")

    assert "COURSEFORGE_BLOCK_ROUTING_PATH" not in os.environ
    assert _NVIDIA_LARGE_MODEL_ENV not in os.environ
    assert _TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV not in os.environ
    # The four authoring envs still fill (provider routing is independent).
    # The GAP-3 licensing guard pins the training seat local even here.
    for agent, env_var in AGENT_AUTHORING_PROVIDER_ENV_MAP.items():
        expected = "local" if agent == "training-synthesizer" else "nvidia"
        assert os.environ[env_var] == expected, env_var


def test_nvidia_routing_yaml_large_tier_is_nvidia(monkeypatch: pytest.MonkeyPatch):
    """The nvidia routing YAML's `large` rewrite tier resolves provider=nvidia."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    runner = _make_runner()
    runner._apply_authoring_route_env("textbook_to_course", "nvidia")

    from Courseforge.router.policy import load_block_routing_policy

    project_root = Path(__file__).resolve().parents[2]
    policy = load_block_routing_policy(project_root / _NVIDIA_LARGE_BLOCK_ROUTING_PATH)
    assert not policy.is_empty()
    large = policy.capability_tiers.get("large", {})
    assert large.get("provider") == "nvidia"
