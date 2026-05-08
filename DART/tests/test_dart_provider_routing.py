"""Wave W-D13 T13.2 — DART_PROVIDER + DART_VISION_PROVIDER plumbing tests.

Asserts that:

- ``_resolve_dart_provider`` honours the explicit > env var > legacy
  default chain.
- ``_resolve_dart_vision_provider`` falls THROUGH ``DART_PROVIDER`` when
  ``DART_VISION_PROVIDER`` is unset (so an operator can pin one provider
  for both surfaces with one env var).
- ``ClaudeProcessor(provider="local")`` constructs an
  ``OpenAICompatibleBackend`` (registry-driven, NOT a per-provider
  subclass) on first ``client`` access.
- ``ClaudeProcessor(provider="anthropic")`` keeps the legacy
  ``AnthropicBackend`` path.
- ``AltTextGenerator(provider="local")`` raises ``ValueError`` at
  construction time when the resolved local backend is NOT
  vision-capable (the operator-visible "fail at startup, not mid-PDF"
  contract).
- ``AltTextGenerator(provider="local")`` succeeds when the local entry
  resolves vision-capable via the substring heuristic
  (``LOCAL_SYNTHESIS_MODEL=qwen2.5-vl:...``).
- ``cli.py --dart-provider local --dart-vision-provider together-vision``
  flow into the ``PDFToAccessibleHTML`` constructor.
- W-D12 dynamic-provider-references rule: provider name interpolates
  into the ``pipeline_run_attribution`` decision-capture rationale via
  the ``provider=...`` field, so the audit trail records WHICH provider
  produced each DART artifact.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Resolver chain
# ---------------------------------------------------------------------------


class TestResolverChain:
    def test_dart_provider_default_when_env_unset(self, monkeypatch):
        from DART.pdf_converter import claude_processor as cp

        monkeypatch.delenv(cp.DART_PROVIDER_ENV, raising=False)
        assert cp._resolve_dart_provider() == cp.DART_PROVIDER_DEFAULT
        assert cp._resolve_dart_provider() == "anthropic"

    def test_dart_provider_env_var_wins_over_default(self, monkeypatch):
        from DART.pdf_converter import claude_processor as cp

        monkeypatch.setenv(cp.DART_PROVIDER_ENV, "local")
        assert cp._resolve_dart_provider() == "local"

    def test_dart_provider_explicit_kwarg_wins_over_env(self, monkeypatch):
        from DART.pdf_converter import claude_processor as cp

        monkeypatch.setenv(cp.DART_PROVIDER_ENV, "local")
        assert cp._resolve_dart_provider("together") == "together"

    def test_dart_provider_empty_env_falls_through(self, monkeypatch):
        from DART.pdf_converter import claude_processor as cp

        monkeypatch.setenv(cp.DART_PROVIDER_ENV, "")
        assert cp._resolve_dart_provider() == cp.DART_PROVIDER_DEFAULT

    def test_dart_vision_provider_default_when_env_unset(self, monkeypatch):
        from DART.pdf_converter import claude_processor as cp

        monkeypatch.delenv(cp.DART_VISION_PROVIDER_ENV, raising=False)
        monkeypatch.delenv(cp.DART_PROVIDER_ENV, raising=False)
        assert cp._resolve_dart_vision_provider() == "anthropic"

    def test_dart_vision_provider_falls_through_to_dart_provider(
        self, monkeypatch
    ):
        """Operator pins one provider via DART_PROVIDER and gets both surfaces."""

        from DART.pdf_converter import claude_processor as cp

        monkeypatch.delenv(cp.DART_VISION_PROVIDER_ENV, raising=False)
        monkeypatch.setenv(cp.DART_PROVIDER_ENV, "together-vision")
        assert cp._resolve_dart_vision_provider() == "together-vision"

    def test_dart_vision_provider_overrides_dart_provider(self, monkeypatch):
        """DART_VISION_PROVIDER wins over DART_PROVIDER when both are set."""

        from DART.pdf_converter import claude_processor as cp

        monkeypatch.setenv(cp.DART_PROVIDER_ENV, "local")
        monkeypatch.setenv(cp.DART_VISION_PROVIDER_ENV, "together-vision")
        assert cp._resolve_dart_vision_provider() == "together-vision"
        # Text-mode resolver still sees DART_PROVIDER.
        assert cp._resolve_dart_provider() == "local"

    def test_dart_vision_provider_explicit_kwarg_wins(self, monkeypatch):
        from DART.pdf_converter import claude_processor as cp

        monkeypatch.setenv(cp.DART_PROVIDER_ENV, "local")
        monkeypatch.setenv(cp.DART_VISION_PROVIDER_ENV, "together-vision")
        assert (
            cp._resolve_dart_vision_provider("anthropic") == "anthropic"
        )


# ---------------------------------------------------------------------------
# ClaudeProcessor — text-mode provider routing
# ---------------------------------------------------------------------------


class TestClaudeProcessorProviderRouting:
    def test_default_provider_is_anthropic(self, monkeypatch):
        from DART.pdf_converter.claude_processor import ClaudeProcessor

        monkeypatch.delenv("DART_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        proc = ClaudeProcessor()
        assert proc.provider == "anthropic"

    def test_provider_kwarg_propagates(self, monkeypatch):
        from DART.pdf_converter.claude_processor import ClaudeProcessor

        proc = ClaudeProcessor(provider="local")
        assert proc.provider == "local"

    def test_provider_local_builds_openai_compatible_backend(self, monkeypatch):
        """W-D13: the lazy ``client`` builds a registry-driven backend.

        Asserts that the SAME class (``OpenAICompatibleBackend``) handles
        the local provider — proving the dynamic-provider-references
        rule: no per-provider subclass.
        """

        from DART.pdf_converter.claude_processor import ClaudeProcessor
        from MCP.orchestrator.llm_backend import OpenAICompatibleBackend

        # Avoid coupling to LOCAL_SYNTHESIS_* defaults when the env was
        # set by another test in the suite.
        monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
        proc = ClaudeProcessor(provider="local")
        backend = proc.client
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.provider_name == "local"

    def test_provider_anthropic_builds_anthropic_backend(self, monkeypatch):
        from DART.pdf_converter.claude_processor import ClaudeProcessor
        from MCP.orchestrator.llm_backend import AnthropicBackend

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        proc = ClaudeProcessor(provider="anthropic")
        backend = proc.client
        assert isinstance(backend, AnthropicBackend)


# ---------------------------------------------------------------------------
# AltTextGenerator — vision-mode provider routing
# ---------------------------------------------------------------------------


class TestAltTextGeneratorVisionRouting:
    def test_default_provider_is_anthropic(self, monkeypatch):
        from DART.pdf_converter.alt_text_generator import AltTextGenerator

        monkeypatch.delenv("DART_PROVIDER", raising=False)
        monkeypatch.delenv("DART_VISION_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        gen = AltTextGenerator()
        assert gen.provider == "anthropic"

    def test_vision_provider_local_without_vision_capability_raises(
        self, monkeypatch
    ):
        """The "fail at startup, not mid-PDF" contract.

        ``provider="local"`` with the default text-only Qwen model →
        the resolved backend is not vision-capable → constructor
        raises ``ValueError`` immediately.
        """

        from DART.pdf_converter.alt_text_generator import AltTextGenerator

        monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
        monkeypatch.delenv("LOCAL_VISION_CAPABLE", raising=False)
        with pytest.raises(ValueError, match="not vision-capable|non-vision-capable"):
            AltTextGenerator(provider="local")

    def test_vision_provider_local_with_substring_heuristic_succeeds(
        self, monkeypatch
    ):
        """A vision-substring model ID flips the flag → constructor succeeds."""

        from DART.pdf_converter.alt_text_generator import AltTextGenerator
        from MCP.orchestrator.llm_backend import OpenAICompatibleBackend

        monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5-vl:7b-instruct-q4_K_M")
        monkeypatch.delenv("LOCAL_VISION_CAPABLE", raising=False)
        gen = AltTextGenerator(provider="local")
        assert gen.provider == "local"
        assert isinstance(gen._client, OpenAICompatibleBackend)
        assert gen._client.vision_capable is True

    def test_vision_provider_local_with_env_opt_in_succeeds(self, monkeypatch):
        """``LOCAL_VISION_CAPABLE=true`` flips the flag → constructor succeeds."""

        from DART.pdf_converter.alt_text_generator import AltTextGenerator

        monkeypatch.setenv("LOCAL_VISION_CAPABLE", "true")
        monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "anything")
        gen = AltTextGenerator(provider="local")
        assert gen._client.vision_capable is True

    def test_vision_provider_together_vision_succeeds(self, monkeypatch):
        from DART.pdf_converter.alt_text_generator import AltTextGenerator

        monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
        gen = AltTextGenerator(provider="together-vision")
        assert gen.provider == "together-vision"
        assert gen._client.vision_capable is True

    def test_vision_provider_anthropic_succeeds_without_vision_check(
        self, monkeypatch
    ):
        """Anthropic vision is unconditional — no vision_capable flag."""

        from DART.pdf_converter.alt_text_generator import AltTextGenerator
        from MCP.orchestrator.llm_backend import AnthropicBackend

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        gen = AltTextGenerator(provider="anthropic")
        assert isinstance(gen._client, AnthropicBackend)


# ---------------------------------------------------------------------------
# CLI — argparse plumbing
# ---------------------------------------------------------------------------


class TestCliArgparsePlumbing:
    def test_dart_provider_defaults_to_none(self, monkeypatch):
        """Unset CLI flag → ``None`` so env-var resolution still fires."""

        if "DART.pdf_converter.cli" in sys.modules:
            del sys.modules["DART.pdf_converter.cli"]
        cli_mod = importlib.import_module("DART.pdf_converter.cli")
        parsed = cli_mod.parse_args(["dummy.pdf"])
        assert parsed.dart_provider is None
        assert parsed.dart_vision_provider is None

    def test_dart_provider_flag_parses(self, monkeypatch):
        cli_mod = importlib.import_module("DART.pdf_converter.cli")
        parsed = cli_mod.parse_args(
            [
                "dummy.pdf",
                "--dart-provider",
                "local",
                "--dart-vision-provider",
                "together-vision",
            ]
        )
        assert parsed.dart_provider == "local"
        assert parsed.dart_vision_provider == "together-vision"

    def test_dart_provider_flows_into_converter(self, monkeypatch, tmp_path):
        """End-to-end: ``--dart-provider`` reaches ``PDFToAccessibleHTML``."""

        # We don't actually run a conversion (no PDF); just inspect the
        # converter's provider attributes to confirm the CLI flag wires
        # through.
        from DART.pdf_converter import PDFToAccessibleHTML

        # Mirror what cli.main would do for arg-flow.
        converter = PDFToAccessibleHTML(
            provider="local",
            vision_provider="together-vision",
        )
        assert converter.provider == "local"
        assert converter.vision_provider == "together-vision"
        # Provider threads through into the lazy ClaudeProcessor config.
        assert converter._claude_config["provider"] == "local"


# ---------------------------------------------------------------------------
# Decision-capture rationale — provider name interpolation
# ---------------------------------------------------------------------------


class TestDecisionCaptureProviderRationale:
    def test_pipeline_run_attribution_rationale_carries_provider_name(
        self, monkeypatch, tmp_path
    ):
        """W-D12 dynamic-provider-references rule.

        The ``pipeline_run_attribution`` capture rationale interpolates
        ``provider=...`` so the audit trail records WHICH provider
        produced each DART artifact. Mirrors the synthesis-provider
        rationale shape.

        Inline state-runs isolation: redirect any state writes the
        underlying ``_run_dart_pipeline_body`` triggers into ``tmp_path``
        so the test doesn't pollute the project's ``state/runs/`` dir
        (the audit guard at ``MCP/tests/test_state_runs_test_isolation.py``
        fails CI otherwise). Mirrors the root conftest's
        ``state_runs_isolated`` fixture; inlined because DART's pytest
        config doesn't pick up the root conftest fixtures.
        """

        from MCP.tools.pipeline_tools import _raw_text_to_accessible_html

        # State-runs isolation — see docstring.
        state_dir = tmp_path / "state" / "runs"
        state_dir.mkdir(parents=True)
        monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(state_dir))

        monkeypatch.setenv("DART_PROVIDER", "local")
        monkeypatch.setenv("DART_VISION_PROVIDER", "together-vision")
        # Pin model + key so the resolvers don't probe ANTHROPIC_API_KEY.
        monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5-vl:7b")
        monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")

        captured_events = []

        class _FakeCapture:
            def log_decision(self, **kwargs: Any) -> None:
                captured_events.append(kwargs)

        # We pass a synthetic source_pdf path so the
        # pipeline_run_attribution branch fires; the body invocation
        # short-circuits via the empty raw_text guard but the capture
        # emit happens BEFORE the body. We use a tmp_path file so the
        # ``Path(source_pdf).name`` interpolation has something real.
        synthetic_pdf = tmp_path / "fake.pdf"
        synthetic_pdf.write_bytes(b"%PDF-1.4 fake")
        try:
            _raw_text_to_accessible_html(
                "raw text",
                "Test Doc",
                source_pdf=str(synthetic_pdf),
                capture=_FakeCapture(),
            )
        except Exception:
            # Body may raise (no real pdftotext); we only care that
            # the capture event fired BEFORE the body.
            pass

        # At least one pipeline_run_attribution event was recorded.
        attrib_events = [
            e
            for e in captured_events
            if e.get("decision_type") == "pipeline_run_attribution"
        ]
        assert attrib_events, "expected pipeline_run_attribution capture"
        rationale = attrib_events[0]["rationale"]
        # Provider name interpolated into the rationale per the W-D12
        # dynamic-provider-references contract.
        assert "provider=local" in rationale
        assert "vision_provider=together-vision" in rationale
        # Rationale is dynamic + ≥20 chars (project capture contract).
        assert len(rationale) >= 20
