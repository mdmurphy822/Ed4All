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

import asyncio
from typing import Any, Dict, List, Optional


_NO_DISPATCHER_MSG = (
    "ClaudeSessionProvider requires a LocalDispatcher; "
    "synthesize_training.py must run inside the workflow runner or MCP tool "
    "(both inject one) when --provider claude_session is set. Standalone CLI "
    "invocation has no Claude Code session to dispatch to."
)

_DISPATCH_TASK_NAME = "synthesize_training"
_AGENT_TYPE = "training-synthesizer"
_INSTRUCTION_KEYS = ["prompt", "completion"]
_PREFERENCE_KEYS = ["prompt", "chosen", "rejected"]


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

    def paraphrase_instruction(
        self, draft: Dict[str, Any], chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Paraphrase a mock-drafted instruction pair."""
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")

        outputs = asyncio.run(
            self._dispatch(
                kind="instruction",
                draft=draft,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                expected_keys=_INSTRUCTION_KEYS,
            )
        )
        out = dict(draft)
        out["prompt"] = str(outputs["prompt"])
        out["completion"] = str(outputs["completion"])
        out["provider"] = "claude_session"
        self._emit_decision(kind="instruction", draft=draft, chunk_id=chunk_id)
        return out

    def paraphrase_preference(
        self, draft: Dict[str, Any], chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Paraphrase a mock-drafted preference triple."""
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")

        outputs = asyncio.run(
            self._dispatch(
                kind="preference",
                draft=draft,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                expected_keys=_PREFERENCE_KEYS,
            )
        )
        out = dict(draft)
        out["prompt"] = str(outputs["prompt"])
        out["chosen"] = str(outputs["chosen"])
        out["rejected"] = str(outputs["rejected"])
        out["provider"] = "claude_session"
        self._emit_decision(kind="preference", draft=draft, chunk_id=chunk_id)
        return out

    def _emit_decision(
        self,
        *,
        kind: str,
        draft: Dict[str, Any],
        chunk_id: str,
    ) -> None:
        if self._capture is None:
            return
        template_id = draft.get("template_id") or "<unknown>"
        rationale = (
            f"Routed {kind} paraphrase for chunk_id={chunk_id} "
            f"template_id={template_id} via claude_session provider "
            f"(version={self._provider_version}, run_id={self._run_id})."
        )
        self._capture.log_decision(
            decision_type="synthesis_provider_call",
            decision=f"claude_session::{kind}",
            rationale=rationale,
        )

    async def _dispatch(
        self,
        *,
        kind: str,
        draft: Dict[str, Any],
        chunk_id: str,
        chunk_text: str,
        expected_keys: List[str],
    ) -> Dict[str, Any]:
        task_params = {
            "kind": kind,
            "draft": draft,
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "expected_keys": expected_keys,
        }
        result = await self._dispatcher.dispatch_task(
            task_name=_DISPATCH_TASK_NAME,
            agent_type=_AGENT_TYPE,
            task_params=task_params,
            run_id=self._run_id,
        )
        if not result.get("success"):
            raise RuntimeError(
                f"training-synthesizer dispatch failed: "
                f"code={result.get('error_code')!r} error={result.get('error')!r}"
            )
        outputs = result.get("outputs") or {}
        for key in expected_keys:
            if key not in outputs:
                raise RuntimeError(
                    f"training-synthesizer returned malformed output "
                    f"for kind={kind}: missing key {key!r}; got {sorted(outputs)!r}"
                )
        return outputs
