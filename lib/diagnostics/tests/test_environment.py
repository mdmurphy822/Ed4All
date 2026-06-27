"""Unit tests for :mod:`lib.diagnostics.environment` — the environment group.

Hermetic + GPU-free + network-free: the ollama probe (``_fetch_ollama_tags``)
and ``importlib.util.find_spec`` are monkeypatched so the suite never touches
a real ollama server, the network, the GPU, or imports the heavy ML stack.
Asserts the per-sub-check OK/WARN logic and the never-raising contract.
"""

from __future__ import annotations

import importlib.util as _ilu

import pytest

from lib.diagnostics import environment as env
from lib.diagnostics.core import (
    CheckContext,
    Severity,
    clear_registry,
    registered_checks,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def _by_name(results, name):
    return next(r for r in results if r.name == name)


# --------------------------------------------------------------------- #
# Registration (no import-time side effect)
# --------------------------------------------------------------------- #


def test_register_environment_checks_no_import_side_effect() -> None:
    assert registered_checks() == []
    env.register_environment_checks()
    pairs = registered_checks()
    assert [g for g, _ in pairs] == ["environment"]
    assert pairs[0][1] is env.environment_checks


# --------------------------------------------------------------------- #
# 1. ollama reachable + model pulled
# --------------------------------------------------------------------- #


def test_ollama_reachable_and_model_present(monkeypatch) -> None:
    monkeypatch.setattr(
        "lib.diagnostics.serving_window.resolve_local_model",
        lambda: "qwen2.5:7b-instruct-q4_K_M",
    )
    monkeypatch.setattr(
        env,
        "_fetch_ollama_tags",
        lambda root, timeout: [
            "qwen2.5:7b-instruct-q4_K_M",
            "nomic-embed-text:latest",
        ],
    )
    results = env._check_ollama(CheckContext())
    reachable = _by_name(results, "ollama_reachable")
    pulled = _by_name(results, "ollama_model_pulled")
    assert reachable.severity is Severity.OK
    assert "qwen2.5:7b-instruct-q4_K_M" in reachable.data["tags"]
    assert pulled.severity is Severity.OK


def test_ollama_model_check_uses_shared_resolver(monkeypatch) -> None:
    # Finding #8: the model name comes from the SHARED
    # serving_window.resolve_local_model resolver, NOT a hardcoded literal.
    sentinel = "sentinel-model:tag"
    monkeypatch.setattr(
        "lib.diagnostics.serving_window.resolve_local_model", lambda: sentinel
    )
    monkeypatch.setattr(env, "_fetch_ollama_tags", lambda root, timeout: [sentinel])
    results = env._check_ollama(CheckContext())
    pulled = _by_name(results, "ollama_model_pulled")
    assert pulled.severity is Severity.OK
    assert pulled.data["model"] == sentinel
    # The duplicate literal must be gone (no re-hardcoded default).
    assert not hasattr(env, "_DEFAULT_LOCAL_MODEL")


def test_ollama_reachable_model_not_pulled(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    monkeypatch.setattr(
        env, "_fetch_ollama_tags", lambda root, timeout: ["llama3:8b"]
    )
    results = env._check_ollama(CheckContext())
    assert _by_name(results, "ollama_reachable").severity is Severity.OK
    pulled = _by_name(results, "ollama_model_pulled")
    assert pulled.severity is Severity.WARN
    assert "not pulled" in pulled.summary
    assert "ollama pull" in pulled.remediation


def test_ollama_configured_model_env_honored(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b")
    monkeypatch.setattr(
        env, "_fetch_ollama_tags", lambda root, timeout: ["qwen2.5:14b"]
    )
    results = env._check_ollama(CheckContext())
    pulled = _by_name(results, "ollama_model_pulled")
    assert pulled.severity is Severity.OK
    assert pulled.data["model"] == "qwen2.5:14b"


def test_ollama_tag_base_match(monkeypatch) -> None:
    # Configured a bare ':tag' vs a fuller pulled tag on the same base.
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(
        env,
        "_fetch_ollama_tags",
        lambda root, timeout: ["qwen2.5:7b-instruct-q4_K_M"],
    )
    results = env._check_ollama(CheckContext())
    assert _by_name(results, "ollama_model_pulled").severity is Severity.OK


def test_ollama_unreachable_warns_no_raise(monkeypatch) -> None:
    def boom(root, timeout):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(env, "_fetch_ollama_tags", boom)
    results = env._check_ollama(CheckContext())
    assert len(results) == 1
    reachable = results[0]
    assert reachable.name == "ollama_reachable"
    assert reachable.severity is Severity.WARN
    assert "not reachable" in reachable.summary
    assert "LOCAL_SYNTHESIS_BASE_URL" in reachable.remediation


# --------------------------------------------------------------------- #
# 2. Optional extras (find_spec monkeypatched)
# --------------------------------------------------------------------- #


def _fake_find_spec(present_modules):
    """Return a find_spec stub that reports only ``present_modules`` present."""
    present = set(present_modules)

    def _stub(name, *args, **kwargs):
        if name in present:
            return object()  # any non-None spec stand-in
        return None

    return _stub


@pytest.mark.parametrize(
    "name,present_set,absent_severity",
    [
        # Finding #5: load-bearing [embedding] absent -> WARN; advisory
        # single-phase [gui] / [semantik] absent -> INFO.
        ("extra_embedding", {"sentence_transformers"}, Severity.WARN),
        ("extra_gui", {"uvicorn", "fastapi"}, Severity.INFO),
        ("extra_semantik", {"pypdfium2"}, Severity.INFO),
    ],
)
def test_extra_present_and_absent(
    monkeypatch, name, present_set, absent_severity
) -> None:
    # Present (all required candidates importable).
    monkeypatch.setattr(_ilu, "find_spec", _fake_find_spec(present_set))
    res = _by_name(env._check_extras(CheckContext()), name)
    assert res.severity is Severity.OK

    # Absent (nothing present) -> the extra's configured absent severity.
    monkeypatch.setattr(_ilu, "find_spec", _fake_find_spec(set()))
    res = _by_name(env._check_extras(CheckContext()), name)
    assert res.severity is absent_severity
    assert res.remediation.startswith("pip install -e .[")


# --------------------------------------------------------------------- #
# 2a. [gui] all-of semantics (Finding #3) — needs BOTH uvicorn AND fastapi.
# --------------------------------------------------------------------- #


def test_gui_extra_requires_both_present_ok(monkeypatch) -> None:
    monkeypatch.setattr(_ilu, "find_spec", _fake_find_spec({"uvicorn", "fastapi"}))
    res = _by_name(env._check_extras(CheckContext()), "extra_gui")
    assert res.severity is Severity.OK
    assert res.data["missing"] == []


def test_gui_extra_only_uvicorn_not_present_info_names_fastapi(monkeypatch) -> None:
    # Half-installed [gui]: uvicorn present, fastapi absent -> NOT present
    # (INFO now), and the result names exactly fastapi as missing.
    monkeypatch.setattr(_ilu, "find_spec", _fake_find_spec({"uvicorn"}))
    res = _by_name(env._check_extras(CheckContext()), "extra_gui")
    assert res.severity is Severity.INFO
    assert res.data["present"] is False
    assert res.data["missing"] == ["fastapi"]
    assert "fastapi" in res.detail


def test_gui_extra_neither_present_info(monkeypatch) -> None:
    monkeypatch.setattr(_ilu, "find_spec", _fake_find_spec(set()))
    res = _by_name(env._check_extras(CheckContext()), "extra_gui")
    assert res.severity is Severity.INFO
    assert res.data["present"] is False
    assert set(res.data["missing"]) == {"uvicorn", "fastapi"}


def test_find_spec_raising_treated_as_absent(monkeypatch) -> None:
    def boom(name, *a, **k):
        raise ModuleNotFoundError("broken parent package")

    monkeypatch.setattr(_ilu, "find_spec", boom)
    # Every extra degrades to ABSENT at its configured severity; no raise.
    results = env._check_extras(CheckContext())
    expected = {
        "extra_embedding": Severity.WARN,
        "extra_gui": Severity.INFO,
        "extra_semantik": Severity.INFO,
    }
    for r in results:
        assert r.data["present"] is False
        assert r.severity is expected[r.name]


# --------------------------------------------------------------------- #
# 3. torch / NLI device
# --------------------------------------------------------------------- #


def test_torch_absent_warns(monkeypatch) -> None:
    monkeypatch.setattr(_ilu, "find_spec", _fake_find_spec(set()))
    res = _by_name(env._check_torch_nli(CheckContext()), "torch_nli")
    assert res.severity is Severity.WARN
    assert res.data["torch_present"] is False
    assert "torch missing" in res.summary


def test_torch_present_reports_device(monkeypatch) -> None:
    monkeypatch.setattr(_ilu, "find_spec", _fake_find_spec({"torch"}))
    monkeypatch.setattr(
        "lib.classifiers.nli_classifier.resolve_nli_device", lambda: "cuda"
    )
    res = _by_name(env._check_torch_nli(CheckContext()), "torch_nli")
    assert res.severity is Severity.OK
    assert res.data["torch_present"] is True
    assert res.data["nli_device"] == "cuda"
    assert "gpu group" in res.detail


# --------------------------------------------------------------------- #
# 4. local-mode dispatch trap
# --------------------------------------------------------------------- #


def test_dispatch_trap_local_without_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "local")
    monkeypatch.delenv("ED4ALL_AGENT_DISPATCH", raising=False)
    res = _by_name(env._check_dispatch_trap(CheckContext()), "local_dispatch_trap")
    assert res.severity is Severity.WARN
    assert "ED4ALL_AGENT_DISPATCH=true" in res.remediation
    assert res.data["agent_dispatch"] is False


def test_dispatch_trap_local_with_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "local")
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    res = _by_name(env._check_dispatch_trap(CheckContext()), "local_dispatch_trap")
    assert res.severity is Severity.OK
    assert res.data["agent_dispatch"] is True


def test_dispatch_trap_default_mode_is_local(monkeypatch) -> None:
    # LLM_MODE unset -> defaults to local -> trap applies.
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_AGENT_DISPATCH", raising=False)
    res = _by_name(env._check_dispatch_trap(CheckContext()), "local_dispatch_trap")
    assert res.severity is Severity.WARN


def test_dispatch_trap_api_mode_ok(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "api")
    monkeypatch.delenv("ED4ALL_AGENT_DISPATCH", raising=False)
    res = _by_name(env._check_dispatch_trap(CheckContext()), "local_dispatch_trap")
    assert res.severity is Severity.OK
    assert res.data["mode"] == "api"


# --------------------------------------------------------------------- #
# Never-raises contract — whole entry point
# --------------------------------------------------------------------- #


def test_environment_checks_never_raises_when_everything_throws(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("everything is on fire")

    # Make every sub-probe throw.
    monkeypatch.setattr(env, "_fetch_ollama_tags", boom)
    monkeypatch.setattr(_ilu, "find_spec", boom)

    def boom_root(*a, **k):
        raise RuntimeError("root resolve exploded")

    monkeypatch.setattr("lib.llm.vram_reclaim.resolve_ollama_root", boom_root)
    monkeypatch.setattr("MCP.core.executor._agent_dispatch_enabled", boom)

    # Must not raise, and must still return a non-empty list of results.
    results = env.environment_checks(CheckContext())
    assert isinstance(results, list)
    assert results  # at least the isolated-WARN results from each sub-check
    assert all(r.group == "environment" for r in results)


def test_run_subcheck_isolates_a_raising_subcheck() -> None:
    def boom(ctx):
        raise ValueError("kaboom")

    boom.__name__ = "_check_boom"
    results = env._run_subcheck(boom, CheckContext())
    assert len(results) == 1
    assert results[0].severity is Severity.WARN
    assert results[0].group == "environment"
    assert "kaboom" in results[0].summary
