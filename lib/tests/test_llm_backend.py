"""Tests for the LLMBackend abstraction (Wave 7)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
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
    OpenAIBackend,
    build_backend,
)
from MCP.orchestrator.task_mailbox import TaskMailbox


class TestProtocolConformance:
    def test_mock_backend_is_llm_backend(self):
        backend = MockBackend(responses=["x"])
        assert isinstance(backend, LLMBackend)

    def test_local_backend_is_llm_backend(self):
        backend = LocalBackend()
        assert isinstance(backend, LLMBackend)

    def test_openai_backend_is_llm_backend(self):
        backend = OpenAIBackend(api_key="key")
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


class TestOpenAIBackend:
    def test_stub_raises(self):
        backend = OpenAIBackend(api_key="k")
        with pytest.raises(NotImplementedError, match="stub"):
            backend.complete_sync("sys", "user")


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

    def test_api_mode_openai_stub(self):
        spec = BackendSpec(mode="api", provider="openai")
        backend = build_backend(spec, api_key="k")
        assert isinstance(backend, OpenAIBackend)

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
        so DART alt-text and block classification silently dropped to
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
            MailboxBrokeredBackend("state/runs/whatever/mailbox")  # type: ignore[arg-type]

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
        whenever ``dart_conversion`` runs ``max_concurrent: 4`` PDFs in
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
        # (content_generation, dart_conversion, …) — none of which can
        # equal the literal string 'llm' because phase names are
        # lowercase_snake_case and never start with that sequence in
        # the config.
        import re as _re
        llm_shape = _re.compile(r"^llm-[0-9a-f]{12}$")
        assert all(llm_shape.match(tid) for tid in llm_ids)
