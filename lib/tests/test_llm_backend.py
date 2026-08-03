"""Tests for the LLMBackend abstraction (Wave 7)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from MCP.orchestrator.llm_backend import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicBackend,
    BackendSpec,
    LLMBackend,
    LocalBackend,
    MailboxBrokeredBackend,
    MockBackend,
    OpenAICompatibleBackend,
    _OPENAI_COMPATIBLE_PROVIDERS,
    build_backend,
    resolve_openai_compatible_backend,
)
from MCP.orchestrator.task_mailbox import TaskMailbox


class TestProtocolConformance:
    def test_mock_backend_is_llm_backend(self):
        backend = MockBackend(responses=["x"])
        assert isinstance(backend, LLMBackend)

    def test_local_backend_is_llm_backend(self):
        backend = LocalBackend()
        assert isinstance(backend, LLMBackend)

    def test_anthropic_backend_is_llm_backend(self):
        backend = AnthropicBackend(api_key="key")
        assert isinstance(backend, LLMBackend)


class TestMockBackend:
    def test_fifo_responses(self):
        backend = MockBackend(responses=["a", "b", "c"])
        assert backend.complete_sync("sys", "u1") == "a"
        assert backend.complete_sync("sys", "u2") == "b"
        assert backend.complete_sync("sys", "u3") == "c"

    def test_records_calls(self):
        backend = MockBackend(responses=["ok"])
        backend.complete_sync("sys", "hello", model="m1", max_tokens=123)
        assert len(backend.calls) == 1
        call = backend.calls[0]
        assert call.system == "sys"
        assert call.user == "hello"
        assert call.model == "m1"
        assert call.max_tokens == 123

    @pytest.mark.asyncio
    async def test_complete_async(self):
        backend = MockBackend(responses=["async-ok"])
        result = await backend.complete("sys", "user")
        assert result == "async-ok"

    @pytest.mark.asyncio
    async def test_streaming_rejected(self):
        backend = MockBackend(responses=["x"])
        with pytest.raises(NotImplementedError):
            await backend.complete("sys", "user", stream=True)

    def test_default_response(self):
        backend = MockBackend(default_response="fallback")
        assert backend.complete_sync("sys", "u") == "fallback"

    def test_response_fn(self):
        backend = MockBackend(response_fn=lambda s, u: f"echo:{u}")
        assert backend.complete_sync("sys", "hello") == "echo:hello"

    def test_fixture_dir(self, tmp_path: Path):
        # Write a fixture keyed by hash of "sys\nuser"
        key = MockBackend._fixture_key("sys", "user")
        (tmp_path / f"{key}.json").write_text(json.dumps({"text": "fx-response"}))
        backend = MockBackend(fixture_dir=tmp_path)
        assert backend.complete_sync("sys", "user") == "fx-response"

    def test_fixture_miss_falls_back(self, tmp_path: Path):
        backend = MockBackend(fixture_dir=tmp_path, default_response="def")
        # No fixture file => should return default_response
        assert backend.complete_sync("sys", "nope") == "def"


class TestLocalBackend:
    def test_complete_sync_raises(self):
        backend = LocalBackend()
        with pytest.raises(NotImplementedError, match="LocalDispatcher"):
            backend.complete_sync("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_async_raises(self):
        backend = LocalBackend()
        with pytest.raises(NotImplementedError):
            await backend.complete("sys", "user")


class TestAnthropicBackend:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            AnthropicBackend()

    def test_default_model(self):
        backend = AnthropicBackend(api_key="k")
        assert backend.default_model == DEFAULT_ANTHROPIC_MODEL

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self):
        backend = AnthropicBackend(api_key="k")
        with pytest.raises(NotImplementedError, match="O3"):
            await backend.complete("sys", "u", stream=True)

    def test_complete_sync_invokes_sdk(self):
        """Wire a mocked anthropic client and ensure the call shape is right."""
        backend = AnthropicBackend(api_key="k", default_model="test-model")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "sdk-output"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client  # inject

        result = backend.complete_sync(
            system="sys",
            user="hello",
            max_tokens=100,
            temperature=0.5,
        )
        assert result == "sdk-output"

        # Verify the SDK was called with the expected shape
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["system"] == "sys"
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "hello"

    def test_complete_sync_with_images(self):
        """Image blocks are attached correctly for vision calls."""
        backend = AnthropicBackend(api_key="k")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="alt text")]
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete_sync(
            system="",
            user="describe",
            images=[{"media_type": "image/png", "data": "b64data"}],
        )
        assert result == "alt text"

        call_kwargs = mock_client.messages.create.call_args.kwargs
        content = call_kwargs["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image"
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "describe"


class TestBuildBackend:
    def test_local_mode_default(self, monkeypatch):
        monkeypatch.delenv("LLM_MODE", raising=False)
        backend = build_backend()
        assert isinstance(backend, LocalBackend)

    def test_api_mode_anthropic(self):
        spec = BackendSpec(mode="api", provider="anthropic")
        backend = build_backend(spec, api_key="test-key")
        assert isinstance(backend, AnthropicBackend)

    def test_api_mode_mock(self):
        spec = BackendSpec(mode="api", provider="mock", mock_responses=["x"])
        backend = build_backend(spec)
        assert isinstance(backend, MockBackend)

    def test_api_mode_openai_alias(self, monkeypatch):
        # W-D12: ``provider="openai"`` is now a deprecated alias for the
        # ``local`` registry entry — the legacy stub no longer fires.
        # See TestBuildBackendRegistry::test_build_backend_openai_alias_deprecation
        # for the canonical contract; this test stays as a regression
        # guard that the spec still resolves without raising.
        monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
        spec = BackendSpec(mode="api", provider="openai")
        with pytest.warns(DeprecationWarning):
            backend = build_backend(spec, api_key="k")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.provider_name == "local"

    def test_unknown_provider_raises(self):
        spec = BackendSpec(mode="api", provider="nonsense")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_backend(spec)

    def test_env_vars_picked_up(self, monkeypatch):
        monkeypatch.setenv("LLM_MODE", "local")
        monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
        backend = build_backend()
        assert isinstance(backend, LocalBackend)

    def test_local_mode_with_run_id_builds_mailbox_backend(
        self, monkeypatch, tmp_path,
    ):
        """Wave 73: ``--mode local`` + ``ED4ALL_RUN_ID`` → MailboxBrokeredBackend.

        Previously local mode always returned the throwing LocalBackend,
        so SemantiK alt-text and block classification silently dropped to
        heuristic fallbacks even when a Claude Code operator *could*
        service completions. Presence of ED4ALL_RUN_ID + a resolvable
        mailbox base_dir now opts into the mailbox bridge.
        """
        monkeypatch.setenv("LLM_MODE", "local")
        monkeypatch.setenv("ED4ALL_RUN_ID", "TST_RUN_001")
        monkeypatch.setenv("ED4ALL_MAILBOX_BASE_DIR", str(tmp_path))

        backend = build_backend()
        assert isinstance(backend, MailboxBrokeredBackend)
        # Mailbox root is scoped under the run id.
        assert backend.mailbox.run_id == "TST_RUN_001"
        assert backend.mailbox.root == tmp_path / "TST_RUN_001" / "mailbox"

    def test_local_mode_without_run_id_still_throws(self, monkeypatch, tmp_path):
        """Without an ED4ALL_RUN_ID the stub LocalBackend path is preserved
        — callers that don't opt into the mailbox bridge still fail loudly
        if they accidentally call ``.complete()``."""
        monkeypatch.setenv("LLM_MODE", "local")
        monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
        monkeypatch.delenv("ED4ALL_MAILBOX_BASE_DIR", raising=False)

        backend = build_backend()
        assert isinstance(backend, LocalBackend)
        with pytest.raises(NotImplementedError):
            backend.complete_sync("sys", "user")


class TestMailboxBrokeredBackend:
    """Wave 73: LLM bridge between the pipeline subprocess and a Claude Code
    operator via the TaskMailbox."""

    def _mailbox(self, tmp_path, run_id: str = "TST_RUN") -> TaskMailbox:
        return TaskMailbox(run_id=run_id, base_dir=tmp_path)

    def test_protocol_conformance(self, tmp_path):
        backend = MailboxBrokeredBackend(self._mailbox(tmp_path))
        assert isinstance(backend, LLMBackend)

    def test_rejects_non_mailbox(self):
        """Typo-safety: passing a string path instead of a TaskMailbox must
        raise at construction — not mysteriously fail downstream."""
        with pytest.raises(TypeError, match="TaskMailbox"):
            MailboxBrokeredBackend("runtime/state/runs/whatever/mailbox")  # type: ignore[arg-type]

    def test_complete_sync_writes_pending_then_reads_completion(self, tmp_path):
        """Happy path: backend writes pending → operator writes completion
        → backend returns the response_text."""
        mailbox = self._mailbox(tmp_path)
        backend = MailboxBrokeredBackend(
            mailbox, timeout_seconds=5.0, poll_interval=0.02,
        )

        # Stage a completion envelope for the task id the backend will mint.
        # Backend uses "llm-0001" as the first task id.
        def operator_thread():
            # Wait for pending file to appear, then write completion.
            import time
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                pending = mailbox.list_pending()
                if pending:
                    task_id = pending[0]
                    mailbox.claim(task_id)
                    mailbox.complete(
                        task_id,
                        {
                            "success": True,
                            "result": {"response_text": "Claude says hello."},
                        },
                    )
                    return
                time.sleep(0.02)

        import threading
        op = threading.Thread(target=operator_thread, daemon=True)
        op.start()

        response = backend.complete_sync(
            system="You are a helper.",
            user="Say hello.",
        )
        op.join(timeout=2.0)

        assert response == "Claude says hello."

    def test_pending_spec_carries_kind_and_payload(self, tmp_path):
        """The pending-file payload must tag ``kind="llm_call"`` and include
        the system/user/model/max_tokens fields so the operator knows how
        to dispatch it. Guards against the operator loop confusing LLM
        tasks with phase-dispatch tasks that share the mailbox."""
        mailbox = self._mailbox(tmp_path)
        backend = MailboxBrokeredBackend(
            mailbox, timeout_seconds=0.5, poll_interval=0.02,
        )

        captured: Dict[str, Any] = {}

        def operator_thread():
            import time
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                pending = mailbox.list_pending()
                if pending:
                    task_id = pending[0]
                    spec = mailbox.claim(task_id)
                    captured.update(spec)
                    mailbox.complete(
                        task_id,
                        {"success": True, "result": {"response_text": "ok"}},
                    )
                    return
                time.sleep(0.02)

        import threading
        op = threading.Thread(target=operator_thread, daemon=True)
        op.start()
        backend.complete_sync(
            "system-msg",
            "user-msg",
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            temperature=0.2,
        )
        op.join(timeout=2.0)

        assert captured["kind"] == "llm_call"
        assert captured["system"] == "system-msg"
        assert captured["user"] == "user-msg"
        assert captured["model"] == "claude-haiku-4-5-20251001"
        assert captured["max_tokens"] == 512
        assert captured["temperature"] == 0.2

    def test_timeout_raises(self, tmp_path):
        """With no operator writing completion, the backend must raise
        TimeoutError rather than silently returning empty string."""
        mailbox = self._mailbox(tmp_path)
        backend = MailboxBrokeredBackend(
            mailbox, timeout_seconds=0.1, poll_interval=0.02,
        )
        with pytest.raises(TimeoutError):
            backend.complete_sync("sys", "user")

    def test_failure_envelope_raises(self, tmp_path):
        """``success: False`` must surface as an exception — call sites
        that catch ``Exception`` then fall back to heuristics rely on the
        exception path to know the LLM was unavailable."""
        mailbox = self._mailbox(tmp_path)
        backend = MailboxBrokeredBackend(
            mailbox, timeout_seconds=2.0, poll_interval=0.02,
        )

        def operator_thread():
            import time
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                pending = mailbox.list_pending()
                if pending:
                    task_id = pending[0]
                    mailbox.claim(task_id)
                    mailbox.complete(
                        task_id,
                        {"success": False, "error": "operator declined",
                         "error_code": "OPERATOR_REFUSED"},
                    )
                    return
                time.sleep(0.02)

        import threading
        op = threading.Thread(target=operator_thread, daemon=True)
        op.start()
        with pytest.raises(RuntimeError, match="OPERATOR_REFUSED"):
            backend.complete_sync("sys", "user")
        op.join(timeout=1.0)

    def test_envelope_shapes_accepted(self, tmp_path):
        """All three documented envelope shapes resolve to the right text.

        1. ``{"result": {"response_text": "X"}}`` — canonical
        2. ``{"result": "X"}`` — bare-string convenience
        3. ``{"raw": "X"}`` — fallback
        """
        mailbox = self._mailbox(tmp_path)
        # Shape 1 → canonical
        backend = MailboxBrokeredBackend(
            mailbox, timeout_seconds=2.0, poll_interval=0.02,
        )

        def _run(envelope_builder, expected):
            def operator_thread():
                import time
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    pending = mailbox.list_pending()
                    if pending:
                        task_id = pending[0]
                        mailbox.claim(task_id)
                        mailbox.complete(task_id, envelope_builder())
                        return
                    time.sleep(0.02)
            import threading
            op = threading.Thread(target=operator_thread, daemon=True)
            op.start()
            out = backend.complete_sync("sys", "user")
            op.join(timeout=1.0)
            assert out == expected

        _run(
            lambda: {"success": True, "result": {"response_text": "canonical"}},
            "canonical",
        )
        _run(
            lambda: {"success": True, "result": "bare-string"},
            "bare-string",
        )
        _run(
            lambda: {"success": True, "raw": "raw-fallback"},
            "raw-fallback",
        )

    def test_streaming_raises(self, tmp_path):
        """Streaming is explicitly unsupported (decision O3)."""
        import asyncio
        backend = MailboxBrokeredBackend(self._mailbox(tmp_path))
        with pytest.raises(NotImplementedError, match="streaming"):
            asyncio.run(backend.complete("sys", "user", stream=True))

    def test_task_ids_are_globally_unique_across_concurrent_backends(
        self, tmp_path,
    ):
        """Wave 73 P1: ids are globally unique, not per-instance monotonic.

        Pre-fix the backend emitted ``f"{prefix}-{counter:04d}"`` starting
        from ``llm-0001``, so two concurrent backends (which happens
        whenever ``semantik_conversion`` runs ``max_concurrent: 4`` PDFs in
        parallel, each auto-resolving its own backend in
        ``pipeline_tools._raw_text_to_accessible_html``) both emitted
        ``llm-0001`` on their first call. ``TaskMailbox.put_pending`` uses
        ``os.replace`` so the second writer silently clobbers the first,
        and both callers wait on the same ``completed/llm-0001.json`` —
        getting the same response for different figures.

        This test mints a batch of ids from two backends sharing one
        mailbox and asserts every id is unique across the union.
        """
        mailbox = self._mailbox(tmp_path)
        backend_a = MailboxBrokeredBackend(mailbox, timeout_seconds=0.01)
        backend_b = MailboxBrokeredBackend(mailbox, timeout_seconds=0.01)

        ids: set[str] = set()
        for backend in (backend_a, backend_b):
            for i in range(5):
                try:
                    backend.complete_sync("s", f"u{i}")
                except TimeoutError:
                    pass
                # Internal helper access: the public surface returns
                # text, not ids. Inspect via the monotonic counter
                # (kept for debugging) + the private minting helper so
                # we don't have to reach into the filesystem.
                ids.add(backend._next_task_id())

        # 10 mint calls (5 per backend) + 10 ids consumed in complete_sync
        # (that we didn't capture) — uniqueness holds across both paths
        # because every id carries a fresh uuid suffix.
        assert len(ids) == 10, f"duplicate ids emitted: {sorted(ids)}"
        # Still prefixed for operator-side filtering.
        assert all(tid.startswith("llm-") for tid in ids), sorted(ids)
        # Counter still advances monotonically (debug aid; no longer
        # participates in the id itself).
        assert backend_a._call_counter >= 5
        assert backend_b._call_counter >= 5
        # Cleanup ran on every complete_sync call regardless of timeout.
        assert mailbox.list_pending() == []
        assert mailbox.list_in_progress() == []

    def test_task_ids_do_not_collide_with_phase_dispatch_ids(self, tmp_path):
        """``LocalDispatcher._dispatch_via_mailbox`` mints ``{phase}-{uuid8}``
        for phase tasks. LLM-call task ids carry the ``llm-`` prefix plus
        a 12-hex uuid tail, so the two shapes never share a key even
        when both flow through the same per-run mailbox.
        """
        mailbox = self._mailbox(tmp_path)
        backend = MailboxBrokeredBackend(mailbox)
        llm_ids = {backend._next_task_id() for _ in range(20)}
        # Every LLM id has the 'llm-' prefix.
        assert all(tid.startswith("llm-") for tid in llm_ids)
        # Phase ids from LocalDispatcher use phase-name prefixes
        # (content_generation, semantik_conversion, …) — none of which can
        # equal the literal string 'llm' because phase names are
        # lowercase_snake_case and never start with that sequence in
        # the config.
        import re as _re
        llm_shape = _re.compile(r"^llm-[0-9a-f]{12}$")
        assert all(llm_shape.match(tid) for tid in llm_ids)


# =============================================================================
# Wave W-D12 — generic OpenAICompatibleBackend + provider registry
# =============================================================================


class _FakeCapture:
    """DecisionCapture-shaped stub that records ``log_decision`` calls."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _FakeChatCompletion:
    """Stand-in for ``OpenAICompatibleClient.chat_completion`` so tests
    don't need a live HTTP server. Records every call's args + payload."""

    def __init__(self, response_text: str = "ok"):
        self.response_text = response_text
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        decision_metadata: Dict[str, Any] | None = None,
        extra_payload: Dict[str, Any] | None = None,
        images: List[Dict[str, Any]] | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "decision_metadata": decision_metadata,
                "extra_payload": extra_payload,
                "images": images,
            }
        )
        return self.response_text


def _wire_fake_client(
    backend: OpenAICompatibleBackend, response_text: str = "ok"
) -> _FakeChatCompletion:
    """Replace the backend's lazy client with an in-memory stub.

    Bypasses the lazy-import path by setting ``_client`` directly to a
    minimal MagicMock whose ``chat_completion`` is the fake. Returns the
    fake so the caller can introspect its ``.calls`` list.
    """
    fake = _FakeChatCompletion(response_text=response_text)
    stub = MagicMock()
    stub.chat_completion = fake
    stub.model = backend.default_model

    # Match the override-temporarily-then-restore semantics inside
    # ``complete_sync`` — the implementation reads ``client.model`` and
    # writes ``client._model``. Provide ``_model`` as a real attribute
    # on the MagicMock so the swap is visible if the test inspects it.
    stub._model = backend.default_model
    backend._client = stub
    return fake


class TestOpenAICompatibleBackend:
    """Generic backend exercises both registered providers via mocked
    chat_completion. Confirms the SAME class handles every provider."""

    def test_protocol_conformance(self):
        backend = OpenAICompatibleBackend(
            provider_name="local",
            base_url="http://localhost:11434/v1",
            default_model="qwen2.5:7b-instruct-q4_K_M",
        )
        assert isinstance(backend, LLMBackend)

    def test_constructor_validates_args(self):
        with pytest.raises(ValueError, match="provider_name"):
            OpenAICompatibleBackend(
                provider_name="",
                base_url="http://localhost",
                default_model="m",
            )
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleBackend(
                provider_name="x", base_url="", default_model="m",
            )
        with pytest.raises(ValueError, match="default_model"):
            OpenAICompatibleBackend(
                provider_name="x", base_url="http://h", default_model="",
            )

    def test_complete_sync_local(self, monkeypatch):
        cap = _FakeCapture()
        backend = OpenAICompatibleBackend(
            provider_name="local",
            base_url="http://localhost:11434/v1",
            default_model="qwen2.5:7b-instruct-q4_K_M",
            capture=cap,
        )
        fake = _wire_fake_client(backend, response_text="local says hi")

        result = backend.complete_sync(
            "system msg", "user msg", max_tokens=128, temperature=0.3,
        )

        assert result == "local says hi"
        assert len(fake.calls) == 1
        msgs = fake.calls[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "system msg"}
        assert msgs[1] == {"role": "user", "content": "user msg"}
        assert fake.calls[0]["max_tokens"] == 128
        assert fake.calls[0]["temperature"] == 0.3

        # Capture event fired with the dynamic provider name interpolated.
        assert len(cap.events) == 1
        event = cap.events[0]
        assert event["decision_type"] == "llm_chat_call"
        # provider_label flows into the rationale's "provider=" field
        # (set from the constructor arg, so adding a registry entry
        # surfaces here unchanged).
        assert "provider=local" in event["rationale"]
        # The extras carry the explicit provider_name + redacted base_url
        # — they land in the decision string per _CaptureMixin contract.
        assert "provider_name=local" in event["decision"]
        assert "base_url=http://localhost:11434" in event["decision"]
        assert "model=qwen2.5:7b-instruct-q4_K_M" in event["rationale"]
        assert len(event["rationale"]) >= 20

    def test_complete_sync_together(self, monkeypatch):
        cap = _FakeCapture()
        backend = OpenAICompatibleBackend(
            provider_name="together",
            base_url="https://api.together.xyz/v1",
            default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key="sk-test",
            capture=cap,
        )
        fake = _wire_fake_client(backend, response_text="together says hi")

        result = backend.complete_sync("sys", "user", max_tokens=64)

        assert result == "together says hi"
        assert len(fake.calls) == 1
        # Same class, different dynamic provider_name → distinct rationale.
        event = cap.events[0]
        assert "provider=together" in event["rationale"]
        assert "provider_name=together" in event["decision"]
        assert "base_url=https://api.together.xyz" in event["decision"]
        assert "model=meta-llama/Llama-3.3-70B-Instruct-Turbo" in event["rationale"]

    def test_vision_raises_runtime_error_when_not_vision_capable(self):
        """W-D13: non-vision-capable provider raises RuntimeError on images=.

        Replaces the W-D12 NotImplementedError pre-W-D13 behaviour. The
        backend now passes ``images=`` through to the underlying client
        on vision-capable providers; non-vision-capable providers fail
        loudly BEFORE the wire round-trip with a message that points at
        every escape hatch (registry flag, vision_capable_env, sibling
        vision providers).
        """

        backend = OpenAICompatibleBackend(
            provider_name="local",
            base_url="http://localhost:11434/v1",
            default_model="qwen2.5",
            vision_capable=False,
        )
        with pytest.raises(RuntimeError, match="not vision-capable"):
            backend.complete_sync(
                "sys",
                "user",
                images=[{"media_type": "image/png", "data": "b64"}],
            )

    def test_no_system_message(self):
        """No system → only user message in the OpenAI payload."""
        backend = OpenAICompatibleBackend(
            provider_name="local",
            base_url="http://localhost:11434/v1",
            default_model="qwen2.5",
        )
        fake = _wire_fake_client(backend)
        backend.complete_sync("", "user-only")
        msgs = fake.calls[0]["messages"]
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "user-only"}

    def test_model_override_swaps_then_restores(self):
        backend = OpenAICompatibleBackend(
            provider_name="local",
            base_url="http://localhost:11434/v1",
            default_model="default-model",
        )
        fake = _wire_fake_client(backend)
        backend.complete_sync("sys", "user", model="override-model")
        # Inner client model must be restored after the call so
        # subsequent calls without an override see the default again.
        assert backend._client._model == "default-model"

    def test_capture_none_is_noop(self):
        backend = OpenAICompatibleBackend(
            provider_name="local",
            base_url="http://localhost:11434/v1",
            default_model="qwen2.5",
            capture=None,
        )
        _wire_fake_client(backend)
        # Must not raise.
        backend.complete_sync("sys", "user")

    @pytest.mark.asyncio
    async def test_streaming_rejected(self):
        backend = OpenAICompatibleBackend(
            provider_name="local",
            base_url="http://localhost:11434/v1",
            default_model="qwen2.5",
        )
        with pytest.raises(NotImplementedError, match="O3"):
            await backend.complete("sys", "user", stream=True)


class TestProviderRegistry:
    """Registry-driven dynamic resolution. The load-bearing contract:
    new providers are registry entries, NOT new classes."""

    def test_registry_seeds_local_and_together(self):
        assert "local" in _OPENAI_COMPATIBLE_PROVIDERS
        assert "together" in _OPENAI_COMPATIBLE_PROVIDERS
        # Local does not require an api_key (Ollama-style).
        assert _OPENAI_COMPATIBLE_PROVIDERS["local"]["api_key_required"] is False
        # Together requires one (cloud OSS).
        assert _OPENAI_COMPATIBLE_PROVIDERS["together"]["api_key_required"] is True

    def test_resolve_local_uses_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://my-host:9000/v1")
        monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "custom-model")
        monkeypatch.setenv("LOCAL_SYNTHESIS_API_KEY", "my-key")

        backend = resolve_openai_compatible_backend("local")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.provider_name == "local"
        assert backend.base_url == "http://my-host:9000/v1"
        assert backend.default_model == "custom-model"
        assert backend.api_key == "my-key"

    def test_resolve_local_uses_defaults_when_env_unset(self, monkeypatch):
        for k in ("LOCAL_SYNTHESIS_BASE_URL", "LOCAL_SYNTHESIS_MODEL",
                  "LOCAL_SYNTHESIS_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        backend = resolve_openai_compatible_backend("local")
        assert backend.base_url == "http://localhost:8000/v1"
        assert backend.default_model == "nemotron-3-nano-30b-a3b"
        # Local-style providers ship a placeholder so reverse-proxy
        # auth-checking servers see a stable string.
        assert backend.api_key == "local"

    def test_resolve_together_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
            resolve_openai_compatible_backend("together")

    def test_resolve_together_with_api_key(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "tg-secret")
        backend = resolve_openai_compatible_backend("together")
        assert backend.provider_name == "together"
        assert backend.base_url == "https://api.together.xyz/v1"
        assert backend.api_key == "tg-secret"

    def test_resolve_together_model_override(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "tg-secret")
        backend = resolve_openai_compatible_backend(
            "together", model_override="some/other-model",
        )
        assert backend.default_model == "some/other-model"

    def test_resolve_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown OpenAI-compatible provider"):
            resolve_openai_compatible_backend("not-a-real-provider")

    def test_resolve_unknown_provider_message_lists_registry(self):
        try:
            resolve_openai_compatible_backend("nope")
        except ValueError as exc:
            msg = str(exc)
            # Operator must see the registry surface inline so they
            # know what's available without grepping.
            assert "local" in msg
            assert "together" in msg
            assert "_OPENAI_COMPATIBLE_PROVIDERS" in msg
        else:  # pragma: no cover
            pytest.fail("ValueError not raised")

    def test_provider_registry_extension_pattern(self, monkeypatch):
        """LOAD-BEARING: synthesize a fake provider entry at runtime
        and confirm ``resolve_openai_compatible_backend("fake")`` works
        with no code change.

        This is the contract that proves W-D12's dynamic-extension
        guarantee: new OpenAI-compatible providers (Groq, Fireworks,
        DeepSeek, Mistral, hosted Gemini-shim, ...) are added by
        registry-entry edits, NOT by new subclasses. If this test
        breaks, somebody slipped a per-provider subclass in and the
        design contract is violated.
        """
        # Make a real registry mutation that the resolver can see —
        # restore via monkeypatch so the test is hermetic.
        fake_entry = {
            "base_url_env": "FAKE_PROVIDER_BASE_URL",
            "base_url_default": "https://fake-provider.example/v1",
            "api_key_env": "FAKE_PROVIDER_API_KEY",
            "api_key_default": None,
            "model_env": "FAKE_PROVIDER_MODEL",
            "model_default": "fake-model-1.5b",
            "api_key_required": True,
        }
        monkeypatch.setitem(_OPENAI_COMPATIBLE_PROVIDERS, "fake", fake_entry)
        monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "fake-secret")

        backend = resolve_openai_compatible_backend("fake")
        # Same OpenAICompatibleBackend class — NOT a subclass — handles
        # the brand-new provider end-to-end. The class never knew
        # "fake" existed at import time.
        assert isinstance(backend, OpenAICompatibleBackend)
        assert type(backend) is OpenAICompatibleBackend
        assert backend.provider_name == "fake"
        assert backend.base_url == "https://fake-provider.example/v1"
        assert backend.default_model == "fake-model-1.5b"
        assert backend.api_key == "fake-secret"

    def test_unverified_entry_logs_warning(self, monkeypatch, caplog):
        """Stub registry entries (groq / fireworks / deepseek) carry
        ``unverified: True`` so the operator sees a logged warning."""
        monkeypatch.setenv("GROQ_API_KEY", "grq-secret")
        with caplog.at_level("WARNING"):
            resolve_openai_compatible_backend("groq")
        assert any("unverified" in r.message for r in caplog.records)


class TestBuildBackendRegistry:
    """``build_backend`` dispatches dynamically via the registry — confirms
    backward compat for ``anthropic`` / ``openai`` and forward-compat for
    every registered provider name."""

    def test_build_backend_anthropic_unchanged(self):
        """Regression guard: legacy api-mode anthropic builds unchanged."""
        spec = BackendSpec(mode="api", provider="anthropic")
        backend = build_backend(spec, api_key="sk-test")
        assert isinstance(backend, AnthropicBackend)
        # Not surprisingly routed through OpenAICompatibleBackend.
        assert not isinstance(backend, OpenAICompatibleBackend)

    def test_build_backend_local_resolves_via_registry(self, monkeypatch):
        monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
        spec = BackendSpec(mode="api", provider="local")  # type: ignore[arg-type]
        backend = build_backend(spec)
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.provider_name == "local"

    def test_build_backend_together_resolves_via_registry(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "tg-secret")
        spec = BackendSpec(mode="api", provider="together")  # type: ignore[arg-type]
        backend = build_backend(spec)
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.provider_name == "together"
        assert backend.api_key == "tg-secret"

    def test_build_backend_together_missing_key_fails(self, monkeypatch):
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        spec = BackendSpec(mode="api", provider="together")  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
            build_backend(spec)

    def test_build_backend_openai_alias_deprecation(self, monkeypatch):
        """``provider="openai"`` aliases to ``local`` AND emits a
        DeprecationWarning so existing deployments keep working."""
        monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
        spec = BackendSpec(mode="api", provider="openai")
        with pytest.warns(DeprecationWarning, match="deprecated"):
            backend = build_backend(spec)
        # Aliased to the local registry entry — same OpenAICompatibleBackend
        # class, NOT the legacy stub.
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.provider_name == "local"

    def test_build_backend_unknown_provider_lists_registry(self):
        spec = BackendSpec(mode="api", provider="absolutely-not-real")  # type: ignore[arg-type]
        with pytest.raises(ValueError) as excinfo:
            build_backend(spec)
        msg = str(excinfo.value)
        # Both built-in providers AND the registry surface must appear
        # in the error message so the operator sees the full universe.
        assert "anthropic" in msg
        assert "mock" in msg
        assert "local" in msg
        assert "together" in msg


class TestDecisionCaptureProviderName:
    """The ``provider_name`` field is the dynamic audit signal — it must
    be interpolated per call, not pinned to a static label."""

    def test_decision_capture_emits_provider_name(self):
        cap = _FakeCapture()
        backend = OpenAICompatibleBackend(
            provider_name="my-custom-provider",
            base_url="https://my-custom-host/v1",
            default_model="custom-1.5b",
            capture=cap,
        )
        _wire_fake_client(backend, response_text="hi")
        backend.complete_sync("sys", "user")

        assert len(cap.events) == 1
        event = cap.events[0]
        # Dynamic interpolation — the rationale carries the literal
        # provider name string, NOT a hardcoded "openai_compatible" or
        # similar static label. A new registry entry surfaces here
        # automatically because provider_label is set from the
        # constructor arg.
        assert "provider=my-custom-provider" in event["rationale"]
        # The decision-string carries the explicit provider_name +
        # host-only base_url via the extras path.
        assert "provider_name=my-custom-provider" in event["decision"]
        assert "base_url=https://my-custom-host" in event["decision"]
        # model is the dynamic per-call value.
        assert "model=custom-1.5b" in event["rationale"]
        # provider_label flows into the leading "X chat call to model Y"
        # phrase via the _CaptureMixin contract.
        assert "my-custom-provider chat call" in event["decision"]

    def test_distinct_provider_names_emit_distinct_rationales(self):
        """Two backends, two providers → two different rationale strings.
        Confirms the interpolation is live, not memoized at class scope."""
        cap_a = _FakeCapture()
        cap_b = _FakeCapture()
        backend_a = OpenAICompatibleBackend(
            provider_name="provider-a",
            base_url="https://a/v1",
            default_model="m-a",
            capture=cap_a,
        )
        backend_b = OpenAICompatibleBackend(
            provider_name="provider-b",
            base_url="https://b/v1",
            default_model="m-b",
            capture=cap_b,
        )
        _wire_fake_client(backend_a, response_text="r-a")
        _wire_fake_client(backend_b, response_text="r-b")
        backend_a.complete_sync("sys", "u")
        backend_b.complete_sync("sys", "u")
        assert "provider=provider-a" in cap_a.events[0]["rationale"]
        assert "provider=provider-b" in cap_b.events[0]["rationale"]
        assert "provider_name=provider-a" in cap_a.events[0]["decision"]
        assert "provider_name=provider-b" in cap_b.events[0]["decision"]
        # Cross-check: provider-a's name does NOT appear in provider-b's
        # rationale or decision (no leakage between distinct backends).
        assert "provider-a" not in cap_b.events[0]["rationale"]
        assert "provider-a" not in cap_b.events[0]["decision"]
