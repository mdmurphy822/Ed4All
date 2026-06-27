"""Wave W-D5 T5.1 — tests for `_BaseSynthesisProvider`.

The base class lands in isolation (no subclass migrations yet — those
happen in T5.2-T5.6 follow-on commits). These tests exercise the base
via a minimal `_ConcreteForTest` subclass that implements only the
three abstract methods.

Coverage:

- Module-level helpers (`parse_json_lenient`, `extract_text_anthropic`,
  `extract_usage_anthropic`, `_validate_lengths`, `emit_decision_safely`)
  behave per the canonical contract.
- `_BaseSynthesisProvider` constructor resolves anthropic / together /
  local branches with the documented `explicit kwarg > env var > class
  default` precedence.
- The `_emit_decision` helper swallows + warns on capture exceptions,
  preserving the synthesis-pipeline invariant that capture failures
  never break the call path (Wave W-D5 § 6.8).
- Re-exports from the canonical leaves (`SynthesisProviderError`,
  `_Usage`, `_validate_lengths`) work so future providers can import
  from the canonical base.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._base_synthesis_provider import (  # noqa: E402
    _BaseSynthesisProvider,
    _Usage,
    _validate_lengths,
    emit_decision_safely,
    extract_text_anthropic,
    extract_usage_anthropic,
    parse_json_lenient,
    SynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Minimal concrete subclass to exercise the abstract base.
# ---------------------------------------------------------------------------


class _ConcreteForTest(_BaseSynthesisProvider):
    """Minimal `_BaseSynthesisProvider` subclass for base-class tests."""

    def _render_instruction_user(
        self, draft: Dict[str, Any], chunk_id: str, **kwargs: Any
    ) -> str:
        return f"instruction:{chunk_id}"

    def _render_preference_user(
        self, draft: Dict[str, Any], chunk_id: str, **kwargs: Any
    ) -> str:
        return f"preference:{chunk_id}"

    def _emit_per_call_decision(
        self,
        *,
        kind: str,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        self._emit_decision(
            decision_type="synthesis_provider_call",
            decision=f"test::{kind}",
            rationale=(
                f"Test rationale interpolating kind={kind}, "
                f"retry_count={retry_count}, raw_text_len={len(raw_text)}."
            ),
        )


# ---------------------------------------------------------------------------
# Constructor branch resolution
# ---------------------------------------------------------------------------


def test_constructor_resolves_anthropic_branch_with_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    provider = _ConcreteForTest(
        provider="anthropic", default_model_anthropic="claude-test"
    )
    assert provider._provider == "anthropic"
    assert provider._api_key == "sk-test-key"
    assert provider._model == "claude-test"
    assert provider._anthropic_client is None  # lazy


def test_constructor_raises_on_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _ConcreteForTest(
            provider="anthropic", default_model_anthropic="claude-test"
        )


def test_constructor_accepts_injected_anthropic_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = _ConcreteForTest(
        provider="anthropic",
        anthropic_client=MagicMock(),
        default_model_anthropic="claude-test",
    )
    assert provider._anthropic_client is not None


def test_constructor_resolves_together_branch_with_env_key(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test-key")
    provider = _ConcreteForTest(
        provider="together",
        default_model_together="llama-test",
        default_base_url_together="https://together.test/v1",
    )
    assert provider._provider == "together"
    assert provider._api_key == "tg-test-key"
    assert provider._model == "llama-test"
    assert provider._base_url == "https://together.test/v1"


def test_constructor_resolves_local_branch_no_key_required(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    provider = _ConcreteForTest(
        provider="local",
        default_model_local="qwen-test",
        default_base_url_local="http://localhost:11434/v1",
    )
    assert provider._provider == "local"
    assert provider._api_key == "local"  # placeholder
    assert provider._model == "qwen-test"
    assert provider._base_url == "http://localhost:11434/v1"


def test_constructor_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        _ConcreteForTest(provider="bogus")


def test_constructor_accepts_claude_session_provider():
    """ClaudeSession bypasses the standard branch resolution because its
    dispatch is dispatcher-routed, not LLM-API-routed."""
    provider = _ConcreteForTest(provider="claude_session")
    assert provider._provider == "claude_session"
    assert provider._base_url is None


# ---------------------------------------------------------------------------
# `_emit_decision` swallow-on-error contract (Wave W-D5 § 6.8)
# ---------------------------------------------------------------------------


def test_emit_decision_swallows_capture_exceptions(monkeypatch):
    """Capture's `log_decision` raises -> base helper logs + swallows.

    Preserves the synthesis-pipeline invariant that capture failures
    never break the call path. ClaudeSession's pre-W-D5 helper did NOT
    swallow; the migration aligns it with this contract.
    """
    class _RaisingCapture:
        def log_decision(self, **kwargs):
            raise RuntimeError("capture backend offline")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    provider = _ConcreteForTest(
        provider="anthropic",
        capture=_RaisingCapture(),
        default_model_anthropic="claude-test",
    )
    # Must not propagate.
    provider._emit_decision(
        decision_type="synthesis_provider_call",
        decision="test::instruction",
        rationale="rationale string ≥ 20 chars to satisfy contract.",
    )


def test_emit_decision_no_op_when_capture_is_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    provider = _ConcreteForTest(
        provider="anthropic",
        capture=None,
        default_model_anthropic="claude-test",
    )
    provider._emit_decision(
        decision_type="synthesis_provider_call",
        decision="x",
        rationale="rationale string ≥ 20 chars to satisfy contract.",
    )


def test_emit_decision_passes_kwargs_to_capture(monkeypatch):
    captured = []

    class _FakeCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    provider = _ConcreteForTest(
        provider="anthropic",
        capture=_FakeCapture(),
        default_model_anthropic="claude-test",
    )
    provider._emit_decision(
        decision_type="synthesis_provider_call",
        decision="d",
        rationale="rationale string ≥ 20 chars to satisfy contract.",
    )
    assert len(captured) == 1
    assert captured[0]["decision_type"] == "synthesis_provider_call"
    assert captured[0]["decision"] == "d"


# ---------------------------------------------------------------------------
# `parse_json_lenient` helper
# ---------------------------------------------------------------------------


def test_parse_json_lenient_strips_json_fence():
    text = '```json\n{"a": 1, "b": 2}\n```'
    assert parse_json_lenient(text) == {"a": 1, "b": 2}


def test_parse_json_lenient_strips_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert parse_json_lenient(text) == {"a": 1}


def test_parse_json_lenient_finds_balanced_object():
    text = 'Sure! Here you go:\n{"prompt": "x", "completion": "y"}\nhope that helps.'
    parsed = parse_json_lenient(text)
    assert parsed == {"prompt": "x", "completion": "y"}


def test_parse_json_lenient_raises_on_empty():
    with pytest.raises(ValueError, match="empty response text"):
        parse_json_lenient("")


def test_parse_json_lenient_raises_on_no_object():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_lenient("just some prose without an object")


def test_parse_json_lenient_raises_on_unbalanced():
    with pytest.raises(ValueError):
        parse_json_lenient('{"a": 1')


# ---------------------------------------------------------------------------
# Anthropic content-block helpers
# ---------------------------------------------------------------------------


def test_extract_text_anthropic_handles_typed_objects():
    block = SimpleNamespace(type="text", text="hello world")
    response = SimpleNamespace(content=[block])
    assert extract_text_anthropic(response) == "hello world"


def test_extract_text_anthropic_handles_dict_response():
    response = {"content": [{"type": "text", "text": "hello"}]}
    assert extract_text_anthropic(response) == "hello"


def test_extract_text_anthropic_skips_non_text_blocks():
    block_text = SimpleNamespace(type="text", text="A")
    block_tool = SimpleNamespace(type="tool_use", text="ignored")
    response = SimpleNamespace(content=[block_text, block_tool])
    assert extract_text_anthropic(response) == "A"


def test_extract_text_anthropic_empty_on_no_content():
    response = SimpleNamespace(content=None)
    assert extract_text_anthropic(response) == ""


def test_extract_usage_anthropic_returns_defaults_on_missing():
    response = SimpleNamespace(usage=None)
    usage = extract_usage_anthropic(response)
    assert usage == _Usage()


def test_extract_usage_anthropic_extracts_typed():
    usage_obj = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=20,
        cache_creation_input_tokens=80,
    )
    response = SimpleNamespace(usage=usage_obj)
    parsed = extract_usage_anthropic(response)
    assert parsed.input_tokens == 100
    assert parsed.output_tokens == 50
    assert parsed.cache_read_tokens == 20
    assert parsed.cache_creation_tokens == 80


# ---------------------------------------------------------------------------
# `_validate_lengths` helper
# ---------------------------------------------------------------------------


def test_validate_lengths_passes_within_bounds():
    outputs = {
        "prompt": "x" * 100,  # within 40-400
        "completion": "y" * 100,  # within 50-600
    }
    _validate_lengths(outputs, kind="instruction")  # no raise


def test_validate_lengths_raises_on_short_prompt():
    outputs = {
        "prompt": "short",
        "completion": "y" * 100,
    }
    with pytest.raises(SynthesisProviderError) as excinfo:
        _validate_lengths(outputs, kind="instruction")
    assert excinfo.value.code == "prompt_below_minimum"


def test_validate_lengths_raises_on_short_chosen_in_preference():
    outputs = {
        "prompt": "x" * 100,
        "chosen": "tiny",
        "rejected": "y" * 100,
    }
    with pytest.raises(SynthesisProviderError) as excinfo:
        _validate_lengths(outputs, kind="preference")
    assert excinfo.value.code == "chosen_below_minimum"


def test_validate_lengths_raises_on_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        _validate_lengths({}, kind="bogus")


def test_validate_lengths_raises_on_non_string_value():
    outputs = {"prompt": None, "completion": "ok" * 50}
    with pytest.raises(SynthesisProviderError) as excinfo:
        _validate_lengths(outputs, kind="instruction")
    assert excinfo.value.code == "empty_field"


# ---------------------------------------------------------------------------
# `emit_decision_safely` module-level helper
# ---------------------------------------------------------------------------


def test_emit_decision_safely_swallows_exceptions():
    class _RaisingCapture:
        def log_decision(self, **kwargs):
            raise RuntimeError("offline")

    # Must not propagate.
    emit_decision_safely(
        _RaisingCapture(),
        decision_type="synthesis_provider_call",
        decision="d",
        rationale="rationale",
    )


def test_emit_decision_safely_no_op_when_capture_is_none():
    emit_decision_safely(
        None,
        decision_type="synthesis_provider_call",
        decision="d",
        rationale="rationale",
    )


# ---------------------------------------------------------------------------
# Module-level constant preservation contract (smoke check — every name
# the plan § 8.3 enumerates is importable from its canonical leaf path).
# ---------------------------------------------------------------------------


def test_anthropic_module_constants_preserved():
    # Phase 4: the Anthropic-SDK training-pair synthesis provider was removed.
    # The slim ``_anthropic_provider`` retains ONLY the anthropic-identity
    # constants (consumed by the out-of-scope curriculum / assessment anthropic
    # backends); the shared synthesis symbols moved to ``_synthesis_common``.
    from Trainforge.generators._anthropic_provider import (
        DEFAULT_SYNTHESIS_MODEL,
        ENV_API_KEY,
        ENV_MODEL,
    )
    from Trainforge.generators._synthesis_common import (
        PROMPT_MIN,
        PROMPT_MAX,
        COMPLETION_MIN,
        COMPLETION_MAX,
        _KIND_BOUNDS,
        SynthesisProviderError as _SPE,
    )
    assert DEFAULT_SYNTHESIS_MODEL
    assert ENV_API_KEY == "ANTHROPIC_API_KEY"
    assert ENV_MODEL == "ANTHROPIC_SYNTHESIS_MODEL"
    assert PROMPT_MIN < PROMPT_MAX
    assert COMPLETION_MIN < COMPLETION_MAX
    assert "prompt" in _KIND_BOUNDS
    assert _SPE is SynthesisProviderError


def test_together_module_constants_preserved():
    from Trainforge.generators._together_provider import (
        DEFAULT_SYNTHESIS_MODEL,
        DEFAULT_BASE_URL,
        TOGETHER_API_URL,
        ENV_API_KEY,
        ENV_MODEL,
        ENV_BASE_URL,
        MAX_HTTP_RETRIES,
        MAX_PARSE_RETRIES,
        DEFAULT_TIMEOUT,
        INITIAL_BACKOFF_SECONDS,
        _RETRYABLE_STATUS,
        SynthesisProviderError as _SPE,
    )
    assert ENV_API_KEY == "TOGETHER_API_KEY"
    assert TOGETHER_API_URL == "https://api.together.xyz/v1/chat/completions"
    assert ENV_BASE_URL is None  # documented absence
    assert _SPE is SynthesisProviderError


def test_local_module_constants_preserved():
    from Trainforge.generators._local_provider import (
        DEFAULT_BASE_URL,
        DEFAULT_SYNTHESIS_MODEL,
        ENV_BASE_URL,
        ENV_MODEL,
        ENV_API_KEY,
        SynthesisProviderError as _SPE,
    )
    assert ENV_API_KEY == "LOCAL_SYNTHESIS_API_KEY"
    assert ENV_BASE_URL == "LOCAL_SYNTHESIS_BASE_URL"
    assert DEFAULT_BASE_URL.startswith("http")
    assert _SPE is SynthesisProviderError


def test_curriculum_module_constants_preserved():
    from Trainforge.generators._curriculum_provider import (
        CurriculumAlignmentProvider,
        VALID_ROLES,
        ENV_PROVIDER,
        DEFAULT_PROVIDER,
        SUPPORTED_PROVIDERS,
        SynthesisProviderError as _SPE,
    )
    assert "introduce" in VALID_ROLES
    assert ENV_PROVIDER == "CURRICULUM_ALIGNMENT_PROVIDER"
    assert DEFAULT_PROVIDER in SUPPORTED_PROVIDERS
    assert _SPE is SynthesisProviderError


def test_claude_session_module_constants_preserved():
    from Trainforge.generators._claude_session_provider import (
        ClaudeSessionProvider,
        _validate_lengths as _vl,
    )
    assert ClaudeSessionProvider
    assert _vl  # leaf re-export


def test_courseforge_base_imports_still_work():
    """Smoke: Courseforge `_BaseLLMProvider` imports the Trainforge
    defaults at module load time. Importing the class verifies every
    name in `Trainforge.generators._anthropic_provider` /
    `_local_provider` / `_together_provider` that Courseforge depends
    on is still importable from its legacy path.
    """
    from Courseforge.generators._base import _BaseLLMProvider
    assert _BaseLLMProvider is not None
