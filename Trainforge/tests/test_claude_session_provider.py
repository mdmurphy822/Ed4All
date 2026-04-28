"""Unit tests for ClaudeSessionProvider — Wave 107 Phase A.

The provider is the third synthesis backend (alongside mock + anthropic).
It dispatches paraphrase requests to the running Claude Code session via
LocalDispatcher's mailbox bridge so users on Claude Max (no API key) can
produce real LLM-paraphrased training corpora.
"""

from __future__ import annotations

import pytest

from Trainforge.generators._claude_session_provider import ClaudeSessionProvider


def test_constructor_requires_dispatcher() -> None:
    """No dispatcher means no Claude Code session — fail loud, do not silently
    fall back to mock or anthropic."""
    with pytest.raises(RuntimeError, match="requires a LocalDispatcher"):
        ClaudeSessionProvider(dispatcher=None)
