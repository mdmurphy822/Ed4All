"""Claude-Code-session synthesis provider — Wave 107.

Mirrors AnthropicSynthesisProvider's interface but dispatches paraphrase
requests through MCP.orchestrator.local_dispatcher.LocalDispatcher's
mailbox bridge. Designed for Claude Max users who do not have an
ANTHROPIC_API_KEY but DO have a running Claude Code session that can
service subagent dispatch.

Invariants:
- Constructor fails closed if no dispatcher is supplied — there is no
  silent mock or anthropic fallback.
- All paraphrase output sets ``provider = "claude_session"`` so the
  downstream LibV2ModelValidator can audit the synthesis source.
- Every paraphrase call emits a ``synthesis_provider_call`` decision
  event when ``capture`` is wired (per CLAUDE.md instrumentation mandate).
"""

from __future__ import annotations

from typing import Any, Optional


_NO_DISPATCHER_MSG = (
    "ClaudeSessionProvider requires a LocalDispatcher; "
    "synthesize_training.py must run inside the workflow runner or MCP tool "
    "(both inject one) when --provider claude_session is set. Standalone CLI "
    "invocation has no Claude Code session to dispatch to."
)


class ClaudeSessionProvider:
    """Paraphrases mock-provider drafts via the running Claude Code session."""

    def __init__(
        self,
        *,
        dispatcher: Optional[Any] = None,
        run_id: Optional[str] = None,
        capture: Optional[Any] = None,
        provider_version: str = "v1",
    ) -> None:
        if dispatcher is None:
            raise RuntimeError(_NO_DISPATCHER_MSG)
        self._dispatcher = dispatcher
        self._run_id = run_id or "synth-standalone"
        self._capture = capture
        self._provider_version = provider_version
