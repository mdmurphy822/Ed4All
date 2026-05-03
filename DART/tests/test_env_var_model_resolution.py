"""Phase 6 Subtasks 21-23 (Phase 3c env-vars): regression tests for the
env-var-first model resolution chain.

Asserts that:

- ``DART_CLAUDE_MODEL`` resolves to the legacy default
  (``claude-sonnet-4-20250514``) when unset.
- ``DART_CLAUDE_MODEL`` env-var override propagates through every DART
  call site touched by Subtasks 21-22 (CLI, ``ClaudeProcessor``,
  ``PDFToAccessibleHTML``, ``AltTextGenerator``).
- ``MCP_ORCHESTRATOR_LLM_MODEL`` env-var override changes the imported
  ``DEFAULT_ANTHROPIC_MODEL`` constant in ``MCP/orchestrator/llm_backend``
  (Subtask 23).

Mirrors the precedent in
``Trainforge/align_chunks.py::_resolve_align_model`` + the Phase 4
Subtask 35 regression coverage.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# DART CLI (Subtask 21)
# ---------------------------------------------------------------------------


def test_cli_resolve_default_model_returns_legacy_default_when_unset(monkeypatch):
    """Subtask 21: unset env var resolves to the legacy default."""
    from DART.pdf_converter import cli as cli_mod

    monkeypatch.delenv(cli_mod.DART_CLAUDE_MODEL_ENV, raising=False)
    assert cli_mod._resolve_default_model() == cli_mod.DART_CLAUDE_MODEL_DEFAULT
    assert cli_mod._resolve_default_model() == "claude-sonnet-4-20250514"


def test_cli_resolve_default_model_honours_env_override(monkeypatch):
    """Subtask 21: ``DART_CLAUDE_MODEL`` env var wins over the default."""
    from DART.pdf_converter import cli as cli_mod

    monkeypatch.setenv(cli_mod.DART_CLAUDE_MODEL_ENV, "claude-test-model-x")
    assert cli_mod._resolve_default_model() == "claude-test-model-x"


def test_cli_resolve_default_model_treats_empty_env_as_unset(monkeypatch):
    """Subtask 21: empty-string env var falls through to the default.

    ``os.environ.get(KEY) or DEFAULT`` semantics — keeps an accidentally-
    empty `export DART_CLAUDE_MODEL=""` from bypassing the legacy default.
    """
    from DART.pdf_converter import cli as cli_mod

    monkeypatch.setenv(cli_mod.DART_CLAUDE_MODEL_ENV, "")
    assert cli_mod._resolve_default_model() == cli_mod.DART_CLAUDE_MODEL_DEFAULT


def test_cli_argparse_default_reflects_env(monkeypatch):
    """Subtask 21: ``--claude-model`` argparse default reflects the env var.

    ``parse_args`` is rebuilt per invocation, so a process that picks up the
    env var at parse time gets the override without code edits.
    """
    monkeypatch.setenv("DART_CLAUDE_MODEL", "claude-test-cli-x")
    # Force re-import so the helper rebuilds the parser with the new env state.
    if "DART.pdf_converter.cli" in sys.modules:
        del sys.modules["DART.pdf_converter.cli"]
    cli_mod = importlib.import_module("DART.pdf_converter.cli")

    parsed = cli_mod.parse_args(["dummy.pdf"])
    assert parsed.claude_model == "claude-test-cli-x"


# ---------------------------------------------------------------------------
# ClaudeProcessor + PDFToAccessibleHTML (Subtask 22)
# ---------------------------------------------------------------------------


def test_claude_processor_resolver_default_when_env_unset(monkeypatch):
    """Subtask 22: ``_resolve_dart_claude_model()`` returns the default."""
    from DART.pdf_converter import claude_processor as cp

    monkeypatch.delenv(cp.DART_CLAUDE_MODEL_ENV, raising=False)
    assert cp._resolve_dart_claude_model() == cp.DART_CLAUDE_MODEL_DEFAULT


def test_claude_processor_resolver_honours_env(monkeypatch):
    """Subtask 22: env var override propagates through the helper."""
    from DART.pdf_converter import claude_processor as cp

    monkeypatch.setenv(cp.DART_CLAUDE_MODEL_ENV, "claude-test-cp-x")
    assert cp._resolve_dart_claude_model() == "claude-test-cp-x"


def test_claude_processor_resolver_explicit_kwarg_wins_over_env(monkeypatch):
    """Subtask 22: explicit kwarg outranks the env var (precedent from
    ``Trainforge/align_chunks.py::_resolve_align_model``).
    """
    from DART.pdf_converter import claude_processor as cp

    monkeypatch.setenv(cp.DART_CLAUDE_MODEL_ENV, "claude-test-env")
    assert cp._resolve_dart_claude_model("claude-test-explicit") == "claude-test-explicit"


def test_claude_processor_init_resolves_model_from_env(monkeypatch):
    """Subtask 22: ``ClaudeProcessor()`` (no kwarg) resolves from env."""
    from DART.pdf_converter.claude_processor import ClaudeProcessor

    monkeypatch.setenv("DART_CLAUDE_MODEL", "claude-test-proc-env")
    proc = ClaudeProcessor()
    assert proc.model == "claude-test-proc-env"


def test_claude_processor_init_default_when_env_unset(monkeypatch):
    """Subtask 22: ``ClaudeProcessor()`` falls back to the legacy default."""
    from DART.pdf_converter.claude_processor import ClaudeProcessor

    monkeypatch.delenv("DART_CLAUDE_MODEL", raising=False)
    proc = ClaudeProcessor()
    assert proc.model == "claude-sonnet-4-20250514"


def test_claude_processor_init_explicit_kwarg_wins(monkeypatch):
    """Subtask 22: explicit kwarg outranks env var on the constructor."""
    from DART.pdf_converter.claude_processor import ClaudeProcessor

    monkeypatch.setenv("DART_CLAUDE_MODEL", "claude-test-env")
    proc = ClaudeProcessor(model="claude-test-explicit")
    assert proc.model == "claude-test-explicit"


def test_pdf_to_accessible_html_resolves_model_from_env(monkeypatch):
    """Subtask 22: ``PDFToAccessibleHTML()`` resolves from env via the
    canonical helper imported from ``claude_processor``.
    """
    from DART.pdf_converter.converter import PDFToAccessibleHTML

    monkeypatch.setenv("DART_CLAUDE_MODEL", "claude-test-conv-env")
    converter = PDFToAccessibleHTML()
    assert converter._claude_config["model"] == "claude-test-conv-env"


def test_pdf_to_accessible_html_default_when_env_unset(monkeypatch):
    """Subtask 22: ``PDFToAccessibleHTML()`` falls back to legacy default."""
    from DART.pdf_converter.converter import PDFToAccessibleHTML

    monkeypatch.delenv("DART_CLAUDE_MODEL", raising=False)
    converter = PDFToAccessibleHTML()
    assert converter._claude_config["model"] == "claude-sonnet-4-20250514"


def test_pdf_to_accessible_html_explicit_kwarg_wins(monkeypatch):
    """Subtask 22: explicit ``claude_model`` kwarg outranks env var."""
    from DART.pdf_converter.converter import PDFToAccessibleHTML

    monkeypatch.setenv("DART_CLAUDE_MODEL", "claude-test-env")
    converter = PDFToAccessibleHTML(claude_model="claude-test-explicit")
    assert converter._claude_config["model"] == "claude-test-explicit"


# ---------------------------------------------------------------------------
# AltTextGenerator (Subtask 22)
# ---------------------------------------------------------------------------


def test_alt_text_generator_resolves_model_from_env(monkeypatch):
    """Subtask 22: ``AltTextGenerator()`` resolves model from env var."""
    from DART.pdf_converter.alt_text_generator import AltTextGenerator

    monkeypatch.setenv("DART_CLAUDE_MODEL", "claude-test-alt-env")
    # use_ai=False sidesteps the lazy AnthropicBackend import (no API key needed
    # for the resolver assertion).
    gen = AltTextGenerator(use_ai=False)
    assert gen.model == "claude-test-alt-env"


def test_alt_text_generator_default_when_env_unset(monkeypatch):
    """Subtask 22: ``AltTextGenerator()`` falls back to legacy default."""
    from DART.pdf_converter.alt_text_generator import AltTextGenerator

    monkeypatch.delenv("DART_CLAUDE_MODEL", raising=False)
    gen = AltTextGenerator(use_ai=False)
    assert gen.model == "claude-sonnet-4-20250514"


def test_alt_text_generator_explicit_kwarg_wins(monkeypatch):
    """Subtask 22: explicit ``model`` kwarg outranks env var."""
    from DART.pdf_converter.alt_text_generator import AltTextGenerator

    monkeypatch.setenv("DART_CLAUDE_MODEL", "claude-test-env")
    gen = AltTextGenerator(use_ai=False, model="claude-test-explicit")
    assert gen.model == "claude-test-explicit"


# ---------------------------------------------------------------------------
# MCP orchestrator LLMBackend default (Subtask 23)
# ---------------------------------------------------------------------------


def test_mcp_orchestrator_default_anthropic_model_when_env_unset():
    """Subtask 23: unset env var resolves ``DEFAULT_ANTHROPIC_MODEL`` to the
    legacy default at module-import time. Run in a clean subprocess so the
    module is re-imported with the env var explicitly cleared.
    """
    env = os.environ.copy()
    env.pop("MCP_ORCHESTRATOR_LLM_MODEL", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from MCP.orchestrator.llm_backend import "
                "DEFAULT_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_MODEL_DEFAULT; "
                "assert DEFAULT_ANTHROPIC_MODEL == DEFAULT_ANTHROPIC_MODEL_DEFAULT, "
                "DEFAULT_ANTHROPIC_MODEL; "
                "assert DEFAULT_ANTHROPIC_MODEL == 'claude-opus-4-7', "
                "DEFAULT_ANTHROPIC_MODEL; "
                "print('OK')"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_mcp_orchestrator_default_anthropic_model_honours_env_override():
    """Subtask 23: ``MCP_ORCHESTRATOR_LLM_MODEL`` env var pins the default
    at module-import time. Run in a subprocess so the override takes effect.
    """
    env = os.environ.copy()
    env["MCP_ORCHESTRATOR_LLM_MODEL"] = "claude-test-orchestrator-x"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from MCP.orchestrator.llm_backend import DEFAULT_ANTHROPIC_MODEL; "
                "assert DEFAULT_ANTHROPIC_MODEL == 'claude-test-orchestrator-x', "
                "DEFAULT_ANTHROPIC_MODEL; "
                "print('OK')"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_mcp_orchestrator_anthropic_backend_uses_resolved_default():
    """Subtask 23: ``AnthropicBackend(default_model=DEFAULT)`` carries the
    env-var-resolved value through to the constructed instance.
    """
    env = os.environ.copy()
    env["MCP_ORCHESTRATOR_LLM_MODEL"] = "claude-test-backend-x"
    env["ANTHROPIC_API_KEY"] = "test-key"  # AnthropicBackend requires a key.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from MCP.orchestrator.llm_backend import AnthropicBackend; "
                "b = AnthropicBackend(); "
                "assert b.default_model == 'claude-test-backend-x', "
                "b.default_model; "
                "print('OK')"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_mcp_orchestrator_llm_model_env_outranks_orchestrator_env():
    """Subtask 23 + ``build_backend`` precedent: per-run ``LLM_MODEL`` env
    var still wins over the module-level pinned default in
    ``build_backend()`` (which reads ``LLM_MODEL`` directly). Confirms the
    env-var-first chain doesn't break the existing per-run override.
    """
    env = os.environ.copy()
    env["MCP_ORCHESTRATOR_LLM_MODEL"] = "claude-orchestrator-default"
    env["LLM_MODEL"] = "claude-per-run-override"
    env["ANTHROPIC_API_KEY"] = "test-key"
    env["LLM_MODE"] = "api"
    env["LLM_PROVIDER"] = "anthropic"
    # ``BackendSpec.mode`` defaults to ``"local"`` and outranks the
    # ``LLM_MODE`` env var inside ``build_backend``; pass mode explicitly via
    # ``overrides=`` so the env-var precedence chain is the one under test.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from MCP.orchestrator.llm_backend import build_backend; "
                "b = build_backend(mode='api', provider='anthropic'); "
                "assert b.default_model == 'claude-per-run-override', "
                "b.default_model; "
                "print('OK')"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"
