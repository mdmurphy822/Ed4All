"""Tests for the provider-resolution engine :mod:`lib.diagnostics.run_env`.

Covers the four public surfaces:

* :func:`load_provider_key_table` — registry-driven key table + degrade-to-{}.
* :func:`resolve_seats` — per-seat provider + source resolution, incl. the
  two-pass tier env > YAML > hardcoded chain and the honest training-synthesis
  default.
* :func:`applied_run_env` — fanout snapshot/restore (exception-safe).
* :func:`key_status` — required-missing / local / key-set.
"""

from __future__ import annotations

import os

import pytest

from lib.diagnostics import run_env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FAKE_REGISTRY = {
    "local": {
        "api_key_env": "LOCAL_SYNTHESIS_API_KEY",
        "api_key_default": "local",
        "api_key_required": False,
        "base_url_env": "LOCAL_SYNTHESIS_BASE_URL",
        "base_url": "http://localhost:11434/v1",
        "model_env": "LOCAL_SYNTHESIS_MODEL",
        "default_model": "qwen2.5:7b-instruct-q4_K_M",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_required": True,
        "base_url_env": None,
        "base_url": None,
        "model_env": "ANTHROPIC_SYNTHESIS_MODEL",
        "default_model": "claude-sonnet-4-6",
    },
    "nvidia": {
        "api_key_env": "NVIDIA_API_KEY",
        "api_key_required": True,
        "base_url_env": "NVIDIA_BASE_URL",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model_env": "NVIDIA_LARGE_MODEL",
        "default_model": "nvidia/nemotron-3-nano-30b-a3b",
    },
}


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every env var the engine reads so each test starts from a known
    baseline."""
    for var in (
        "COURSEFORGE_PROVIDER",
        "COURSEFORGE_OUTLINE_PROVIDER",
        "COURSEFORGE_OUTLINE_MODEL",
        "COURSEFORGE_REWRITE_PROVIDER",
        "COURSEFORGE_REWRITE_MODEL",
        "COURSEFORGE_BLOCK_ROUTING_PATH",
        "COURSEPLANNER_PROVIDER",
        "COURSEPLANNER_MODEL",
        "TRAINFORGE_ASSESSMENT_PROVIDER",
        "TRAINFORGE_SYNTHESIS_PROVIDER",
        "TRAINFORGE_SYNTHESIS_MODEL",
        "TEXTBOOK_SYNTHESIS_PROVIDER",
        "ED4ALL_ANSWER_PROVIDER",
        "ED4ALL_EMBEDDING_PROVIDER",
        "ED4ALL_OBJECTIVE_REVIEW_PROVIDER",
        "ED4ALL_DYNAMIC_BLOCK_PLAN",
        "ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER",
        "ANTHROPIC_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "NVIDIA_LARGE_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def _by_seat(seats):
    return {s.seat: s for s in seats}


# ---------------------------------------------------------------------------
# load_provider_key_table
# ---------------------------------------------------------------------------


def test_key_table_reads_registry_fields(monkeypatch):
    import lib.llm.endpoints as endpoints

    monkeypatch.setattr(endpoints, "load_endpoint_registry", lambda: _FAKE_REGISTRY)

    table = run_env.load_provider_key_table()

    assert table["local"]["key_required"] is False
    assert table["local"]["key_env"] == "LOCAL_SYNTHESIS_API_KEY"
    assert table["anthropic"]["key_required"] is True
    assert table["anthropic"]["key_env"] == "ANTHROPIC_API_KEY"
    assert table["nvidia"]["key_required"] is True
    assert table["nvidia"]["base_url_default"] == "https://integrate.api.nvidia.com/v1"


def test_key_table_degrades_to_empty_on_load_failure(monkeypatch):
    import lib.llm.endpoints as endpoints

    def _boom():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(endpoints, "load_endpoint_registry", _boom)

    assert run_env.load_provider_key_table() == {}


# ---------------------------------------------------------------------------
# resolve_seats
# ---------------------------------------------------------------------------


def test_resolve_seats_env_overrides(clean_env, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "Local")
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "together")
    monkeypatch.setenv("TRAINFORGE_ASSESSMENT_PROVIDER", "local")

    seats = _by_seat(run_env.resolve_seats(two_pass=False))

    assert seats["content"].effective_provider == "local"  # lowercased
    assert seats["content"].source == "env"
    assert seats["course_outline"].effective_provider == "together"
    assert seats["course_outline"].source == "env"
    assert seats["assessment"].source == "env"
    # two-pass tiers absent when two_pass=False
    assert "outline" not in seats
    assert "rewrite" not in seats


def test_resolve_seats_defaults_when_unset(clean_env):
    seats = _by_seat(run_env.resolve_seats(two_pass=False))

    # content class default reported honestly
    assert seats["content"].effective_provider == "anthropic"
    assert seats["content"].source == "class_default"

    # training_synthesis default is mock/off, NOT anthropic
    ts = seats["training_synthesis"]
    assert ts.effective_provider == "mock"
    assert ts.source == "class_default"
    assert "mock" in ts.note

    # textbook synthesis off
    assert seats["textbook_synthesis"].effective_provider is None
    assert seats["textbook_synthesis"].source == "off"

    # objective_review unset -> None/off
    assert seats["objective_review"].effective_provider is None
    assert seats["objective_review"].source == "off"

    # embedding default 'st'
    assert seats["embedding"].effective_provider == "st"

    # answer default 'local' + loopback note
    assert seats["answer"].effective_provider == "local"
    assert "LOOPBACK" in seats["answer"].note.upper()

    # block planner gated off
    bp = seats["block_planner"]
    assert bp.effective_provider is None
    assert bp.source == "off"


def test_block_planner_on_defaults_nvidia(clean_env, monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "true")
    seats = _by_seat(run_env.resolve_seats(two_pass=False))
    bp = seats["block_planner"]
    assert bp.effective_provider == "nvidia"
    assert bp.source == "class_default"

    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER", "local")
    seats = _by_seat(run_env.resolve_seats(two_pass=False))
    bp = seats["block_planner"]
    assert bp.effective_provider == "local"
    assert bp.source == "env"


def test_training_synthesis_anthropic_failclosed_note(clean_env, monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_PROVIDER", "anthropic")
    seats = _by_seat(run_env.resolve_seats(two_pass=False))
    ts = seats["training_synthesis"]
    assert ts.effective_provider == "anthropic"
    assert ts.source == "env"
    assert "TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS" in ts.note


def test_two_pass_tier_env_wins(clean_env, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "nvidia")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_PROVIDER", "local")

    seats = _by_seat(run_env.resolve_seats(two_pass=True))

    assert seats["outline"].effective_provider == "local"
    assert seats["outline"].source == "env"
    assert seats["rewrite"].effective_provider == "nvidia"
    assert seats["rewrite"].source == "env"
    # rewrite model-dead caveat always carried
    assert "dead on the cloud" in seats["rewrite"].note


def test_two_pass_tier_yaml_source(clean_env, monkeypatch, tmp_path):
    # No tier env -> the active YAML governs (source 'yaml').
    yaml_doc = """
version: 1
capability_tiers:
  small:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
  large:
    provider: nvidia
    model: meta/llama-3.3-70b-instruct
defaults:
  outline:
    capability_tier: small
  rewrite:
    capability_tier: large
blocks: {}
"""
    routing = tmp_path / "block_routing.yaml"
    routing.write_text(yaml_doc, encoding="utf-8")
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", str(routing))

    seats = _by_seat(run_env.resolve_seats(two_pass=True))

    assert seats["outline"].effective_provider == "local"
    assert seats["outline"].source == "yaml"
    assert seats["rewrite"].effective_provider == "nvidia"
    assert seats["rewrite"].source == "yaml"


def test_two_pass_tier_yaml_heterogeneity_flagged(clean_env, monkeypatch, tmp_path):
    # The heterogeneity scan reflects the STARTING tier (chain[0]) — the
    # provider the router actually dispatches on — so the per-block_type
    # pins must differ in their FIRST tier to be heterogeneous.
    yaml_doc = """
version: 1
capability_tiers:
  medium:
    provider: local
  large:
    provider: anthropic
defaults:
  outline:
    capability_tier: medium
  rewrite:
    capability_tier: medium
blocks:
  assessment_item:
    rewrite:
      capability_tier: [large, medium]
  concept:
    rewrite:
      capability_tier: medium
"""
    routing = tmp_path / "block_routing.yaml"
    routing.write_text(yaml_doc, encoding="utf-8")
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", str(routing))

    seats = _by_seat(run_env.resolve_seats(two_pass=True))
    rewrite = seats["rewrite"]
    assert rewrite.source == "yaml"
    # assessment_item STARTS on large(anthropic) while concept starts on
    # medium(local) -> heterogeneous chain[0] providers.
    assert "heterogeneous" in rewrite.note
    assert "anthropic" in rewrite.note and "local" in rewrite.note


def test_capability_tier_cascade_resolves_first_tier(clean_env, monkeypatch, tmp_path):
    # #6: a cascade list [medium, large] (medium=local, large=anthropic) must
    # resolve to the FIRST tier (local) — the router dispatches on chain[0] and
    # only escalates to large on budget exhaustion (policy.py
    # _project_capability_aware_spec). The OLD code resolved chain[-1]
    # (anthropic), falsely attributing a cloud provider to an all-local run.
    yaml_doc = """
version: 1
capability_tiers:
  medium:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
  large:
    provider: anthropic
    model: claude-sonnet-4-6
defaults:
  outline:
    capability_tier: medium
  rewrite:
    capability_tier: [medium, large]
blocks: {}
"""
    routing = tmp_path / "block_routing.yaml"
    routing.write_text(yaml_doc, encoding="utf-8")
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", str(routing))

    seats = _by_seat(run_env.resolve_seats(two_pass=True))
    rewrite = seats["rewrite"]
    # Starting tier wins — NOT anthropic.
    assert rewrite.effective_provider == "local"
    assert rewrite.source == "yaml"
    # Escalation provenance is preserved in the note (informational only).
    assert "escalates to anthropic" in rewrite.note
    # Local tier honors the YAML tier model.
    assert rewrite.model == "qwen2.5:7b-instruct-q4_K_M"


def test_plain_string_capability_tier_resolves(clean_env, monkeypatch, tmp_path):
    # A plain-string capability_tier reference still resolves correctly.
    yaml_doc = """
version: 1
capability_tiers:
  small:
    provider: local
  large:
    provider: nvidia
defaults:
  outline:
    capability_tier: small
  rewrite:
    capability_tier: large
blocks: {}
"""
    routing = tmp_path / "block_routing.yaml"
    routing.write_text(yaml_doc, encoding="utf-8")
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", str(routing))

    seats = _by_seat(run_env.resolve_seats(two_pass=True))
    assert seats["outline"].effective_provider == "local"
    assert seats["outline"].source == "yaml"
    assert seats["rewrite"].effective_provider == "nvidia"
    assert seats["rewrite"].source == "yaml"
    # No cascade -> no escalation note.
    assert "escalates" not in seats["rewrite"].note


def test_cloud_rewrite_tier_does_not_attribute_dead_model(
    clean_env, monkeypatch
):
    # #10: a cloud (nvidia) rewrite tier must NOT present COURSEFORGE_REWRITE_MODEL
    # as the effective model — the router gates that env on provider=='local',
    # so on a cloud tier the model comes from NVIDIA_LARGE_MODEL / registry.
    import lib.llm.endpoints as endpoints

    monkeypatch.setattr(endpoints, "load_endpoint_registry", lambda: _FAKE_REGISTRY)
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "nvidia")
    monkeypatch.setenv("COURSEFORGE_REWRITE_MODEL", "DEAD-should-not-appear")
    monkeypatch.setenv("NVIDIA_LARGE_MODEL", "meta/llama-3.3-70b-instruct")
    # Point at a missing YAML so only env + registry govern.
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", "/nonexistent/routing.yaml")

    seats = _by_seat(run_env.resolve_seats(two_pass=True))
    rewrite = seats["rewrite"]
    assert rewrite.effective_provider == "nvidia"
    # The dead env value is NOT attributed; the real model comes from
    # NVIDIA_LARGE_MODEL (the nvidia registry model_env).
    assert rewrite.model != "DEAD-should-not-appear"
    assert rewrite.model == "meta/llama-3.3-70b-instruct"
    assert "dead on the cloud" in rewrite.note
    assert "not COURSEFORGE_REWRITE_MODEL" in rewrite.note


def test_local_rewrite_tier_honors_rewrite_model(clean_env, monkeypatch):
    # #10 complement: a LOCAL rewrite tier DOES honor COURSEFORGE_REWRITE_MODEL
    # (the router applies the tier model env on local-provider tiers).
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_REWRITE_MODEL", "qwen2.5:14b-instruct")
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", "/nonexistent/routing.yaml")

    seats = _by_seat(run_env.resolve_seats(two_pass=True))
    rewrite = seats["rewrite"]
    assert rewrite.effective_provider == "local"
    assert rewrite.model == "qwen2.5:14b-instruct"
    # No dead-env caveat on the local tier.
    assert "dead on the cloud" not in rewrite.note


def test_two_pass_tier_hardcoded_fallback(clean_env, monkeypatch):
    # Point at a missing YAML so neither env nor YAML resolves.
    monkeypatch.setenv("COURSEFORGE_BLOCK_ROUTING_PATH", "/nonexistent/routing.yaml")
    seats = _by_seat(run_env.resolve_seats(two_pass=True))
    assert seats["outline"].effective_provider == "local"
    assert seats["outline"].source == "class_default"
    assert seats["rewrite"].effective_provider == "anthropic"
    assert seats["rewrite"].source == "class_default"


# ---------------------------------------------------------------------------
# applied_run_env
# ---------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self, executor=None, config=None):
        self.executor = executor
        self.config = config

    def _apply_corpus_generalization_defaults(self, workflow_type):
        applied = {}
        for var, val in (
            ("TEXTBOOK_SYNTHESIS_PROVIDER", "local"),
            ("TRAINFORGE_SYNTHESIS_PROVIDER", "local"),
        ):
            if not os.environ.get(var, "").strip():
                os.environ[var] = val
                applied[var] = val
        return applied

    def _apply_authoring_route_env(self, workflow_type, provider_hint=""):
        applied = {}
        resolved = (provider_hint or "").strip() or "local"
        for var in ("COURSEFORGE_PROVIDER", "COURSEPLANNER_PROVIDER"):
            if not os.environ.get(var, "").strip():
                os.environ[var] = resolved
                applied[var] = resolved
        return applied


class _FakeConfig:
    @classmethod
    def load(cls, config_dir=None):
        return cls()


def _patch_fake_runner(monkeypatch):
    import MCP.core.config as config_mod
    import MCP.core.workflow_runner as wr_mod

    monkeypatch.setattr(wr_mod, "WorkflowRunner", _FakeRunner)
    monkeypatch.setattr(config_mod, "OrchestratorConfig", _FakeConfig)


def test_applied_run_env_applies_and_restores(clean_env, monkeypatch):
    _patch_fake_runner(monkeypatch)
    before = dict(os.environ)
    assert "TEXTBOOK_SYNTHESIS_PROVIDER" not in os.environ

    with run_env.applied_run_env("textbook_to_course", provider_hint="local") as deltas:
        # inside: env mutated
        assert os.environ["TEXTBOOK_SYNTHESIS_PROVIDER"] == "local"
        assert os.environ["COURSEFORGE_PROVIDER"] == "local"
        # yielded deltas carry both helpers' fills
        assert deltas["TEXTBOOK_SYNTHESIS_PROVIDER"] == "local"
        assert deltas["COURSEFORGE_PROVIDER"] == "local"

    # after: byte-restored, no leakage
    assert dict(os.environ) == before
    assert "TEXTBOOK_SYNTHESIS_PROVIDER" not in os.environ
    assert "COURSEFORGE_PROVIDER" not in os.environ


def test_applied_run_env_restores_on_body_exception(clean_env, monkeypatch):
    _patch_fake_runner(monkeypatch)
    before = dict(os.environ)

    with pytest.raises(ValueError):
        with run_env.applied_run_env("textbook_to_course") as deltas:
            assert os.environ.get("TEXTBOOK_SYNTHESIS_PROVIDER") == "local"
            assert deltas  # non-empty
            raise ValueError("boom from body")

    # restoration still happened despite the body raise
    assert dict(os.environ) == before
    assert "TEXTBOOK_SYNTHESIS_PROVIDER" not in os.environ


def test_applied_run_env_yields_empty_and_restores_on_setup_failure(
    clean_env, monkeypatch
):
    import MCP.core.workflow_runner as wr_mod

    def _boom(*a, **k):
        raise RuntimeError("construct failed")

    monkeypatch.setattr(wr_mod, "WorkflowRunner", _boom)
    before = dict(os.environ)

    with run_env.applied_run_env("textbook_to_course") as deltas:
        assert deltas == {}

    assert dict(os.environ) == before


# ---------------------------------------------------------------------------
# key_status
# ---------------------------------------------------------------------------


def test_key_status_required_missing(clean_env, monkeypatch):
    status = run_env.key_status("anthropic", key_table=run_env_table())
    assert status["key_required"] is True
    assert status["key_present"] is False
    assert status["key_env"] == "ANTHROPIC_API_KEY"


def test_key_status_local_not_required(clean_env):
    status = run_env.key_status("local", key_table=run_env_table())
    assert status["key_required"] is False
    assert status["key_present"] is True
    assert status["base_url"] == "http://localhost:11434/v1"


def test_key_status_required_present(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    status = run_env.key_status("anthropic", key_table=run_env_table())
    assert status["key_present"] is True


def test_key_status_base_url_env_override(clean_env, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://nim.local/v1")
    status = run_env.key_status("nvidia", key_table=run_env_table())
    assert status["key_present"] is True
    assert status["base_url"] == "http://nim.local/v1"


def run_env_table():
    """The fixture key table (so key_status tests don't hit the real registry)."""
    return {
        name: {
            "key_env": row.get("api_key_env"),
            "key_required": bool(row.get("api_key_required", False)),
            "base_url_env": row.get("base_url_env"),
            "base_url_default": row.get("base_url"),
            "model_env": row.get("model_env"),
            "default_model": row.get("default_model"),
        }
        for name, row in _FAKE_REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# P0-1: local-synthesis serving topology (ollama vs a vLLM seat)
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def _clear_topology_env(monkeypatch):
    for var in (
        "ED4ALL_SEAT_BASE_URLS",
        "ED4ALL_VLLM_CONTAINERS",
        "ED4ALL_SEAT_LAUNCH_SPECS",
        "LOCAL_SYNTHESIS_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_topology_no_registry_is_ollama(monkeypatch, _clear_topology_env):
    topo = run_env.resolve_local_synthesis_topology()
    assert topo.seat_registry_configured is False
    assert topo.backend == "ollama"
    assert ":11434" in topo.base_url_root


def test_topology_registry_matching_base_url_is_vllm_seat(monkeypatch, _clear_topology_env):
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "spark-super=http://localhost:8001")
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8001/v1")
    topo = run_env.resolve_local_synthesis_topology()
    assert topo.seat_registry_configured is True
    assert topo.backend == "vllm"
    assert topo.seat_name == "spark-super"
    assert topo.base_url_root == "http://localhost:8001"


def test_topology_registry_but_ollama_default_stays_ollama(monkeypatch, _clear_topology_env):
    # Mixed topology: seats declared for OTHER roles, but local synthesis still
    # points at ollama on 11434 → backend stays ollama (legacy path).
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "spark-nano=http://localhost:8004")
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")
    topo = run_env.resolve_local_synthesis_topology()
    assert topo.backend == "ollama"
    assert topo.seat_name is None


def test_topology_registry_unregistered_nonollama_is_vllm(monkeypatch, _clear_topology_env):
    # A seat registry is configured (via containers) and LOCAL_SYNTHESIS_BASE_URL
    # points at a non-ollama port not itself in the registry → still vLLM.
    monkeypatch.setenv("ED4ALL_VLLM_CONTAINERS", "http://localhost:8001=vllm-super")
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8009/v1")
    topo = run_env.resolve_local_synthesis_topology()
    assert topo.backend == "vllm"
    assert topo.seat_name is None
    assert topo.base_url_root == "http://localhost:8009"


def test_topology_never_raises_on_broken_registry(monkeypatch, _clear_topology_env):
    def boom(*a, **k):
        raise RuntimeError("registry parse exploded")

    monkeypatch.setattr("lib.vllm_container_lifecycle.parse_seat_registry", boom)
    monkeypatch.setattr("lib.vllm_container_lifecycle.parse_container_registry", boom)
    topo = run_env.resolve_local_synthesis_topology()
    assert topo.backend == "ollama"  # degrades to legacy, no raise


def test_normalize_root_strips_v1_and_slash():
    assert run_env._normalize_root("http://localhost:8001/v1/") == "http://localhost:8001"
    assert run_env._normalize_root("http://localhost:8001/") == "http://localhost:8001"
    assert run_env._normalize_root(None) == ""


def test_probe_v1_models_live(monkeypatch):
    body = b'{"data": [{"id": "super-120b"}, {"id": "nano"}]}'
    monkeypatch.setattr(
        run_env.urllib.request, "urlopen", lambda req, timeout=0: _FakeResp(body)
    )
    live, ids, error = run_env.probe_v1_models("http://localhost:8001")
    assert live is True
    assert ids == ["super-120b", "nano"]
    assert error is None


def test_probe_v1_models_down_never_raises(monkeypatch):
    def boom(req, timeout=0):
        raise ConnectionError("refused")

    monkeypatch.setattr(run_env.urllib.request, "urlopen", boom)
    live, ids, error = run_env.probe_v1_models("http://localhost:8001")
    assert live is False
    assert ids == []
    assert "ConnectionError" in error
