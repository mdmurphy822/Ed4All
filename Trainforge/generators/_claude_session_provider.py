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
- A content-addressed JSONL cache persists outputs keyed on
  ``sha256(provider_version + kind + chunk_id + draft-load-bearing-fields)``
  so re-runs against an unchanged corpus reuse paraphrases and the
  resulting ``instruction_pairs_hash`` stays stable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from Trainforge.generators._anthropic_provider import (
    COMPLETION_MAX,
    COMPLETION_MIN,
    PROMPT_MAX,
    PROMPT_MIN,
    SynthesisProviderError,
    _KIND_BOUNDS,
)
from Trainforge.generators._session_budget import (
    SynthesisBudgetExceeded,
    _BudgetTracker,
    _CircuitBreaker,
)


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

# Wave 112 Task 4: which output keys participate in the length-clamp per
# paraphrase kind. Mirrors the per-kind bounds in
# ``_anthropic_provider._KIND_BOUNDS`` (``chosen``/``rejected`` reuse the
# completion bounds). The session provider's keys map 1:1 to bound names.
_KIND_KEYS: Dict[str, List[str]] = {
    "instruction": ["prompt", "completion"],
    "preference": ["prompt", "chosen", "rejected"],
}


def _validate_lengths(
    outputs: Dict[str, Any],
    *,
    kind: str,
    chunk_id: Optional[str] = None,
) -> None:
    """Enforce per-key length bounds on a session-provider response.

    Wave 112 Task 4: parallel to ``_anthropic_provider._clamp``'s raise
    behavior. Short paraphrases must fail loud rather than silently
    landing in the cache and the JSONL writer (which would let a
    too-short prompt poison ``instruction_pairs.jsonl``).

    Args:
        outputs: The dispatcher response's ``outputs`` dict (already
            validated to have non-empty string values by ``_dispatch``).
        kind: ``"instruction"`` or ``"preference"`` — selects which
            keys to check.
        chunk_id: Optional context for the raised error.

    Raises:
        SynthesisProviderError: when any checked key's value falls
            below the minimum or exceeds the maximum for its bound.
            The ``code`` is ``f"{key}_below_minimum"`` or
            ``f"{key}_above_maximum"`` so callers can dispatch.
    """
    try:
        keys = _KIND_KEYS[kind]
    except KeyError as exc:
        raise ValueError(
            f"_validate_lengths: unknown kind={kind!r}; expected one of "
            f"{sorted(_KIND_KEYS)}"
        ) from exc
    for key in keys:
        value = outputs.get(key)
        if not isinstance(value, str):
            # _dispatch already enforced this; defensive only.
            raise SynthesisProviderError(
                f"_validate_lengths: expected string for key={key!r}, "
                f"got {type(value).__name__}",
                code="empty_field",
                chunk_id=chunk_id,
            )
        lo, hi = _KIND_BOUNDS[key]
        length = len(value.strip())
        if length < lo:
            raise SynthesisProviderError(
                f"{kind}.{key} length {length} below minimum {lo}; "
                f"refusing to ship short paraphrase. Caller should "
                f"retry the dispatch.",
                code=f"{key}_below_minimum",
                chunk_id=chunk_id,
            )
        if length > hi:
            raise SynthesisProviderError(
                f"{kind}.{key} length {length} above maximum {hi}; "
                f"subagent must constrain output.",
                code=f"{key}_above_maximum",
                chunk_id=chunk_id,
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
        cache_path: Optional[Path] = None,
        max_dispatches: Optional[int] = None,
        telemetry_path: Optional[Path] = None,
        failures_to_open: int = 3,
        failure_window_seconds: float = 60.0,
    ) -> None:
        if dispatcher is None:
            raise RuntimeError(_NO_DISPATCHER_MSG)
        self._dispatcher = dispatcher
        self._run_id = run_id or "synth-standalone"
        self._capture = capture
        self._provider_version = provider_version
        self._cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        if cache_path is not None and cache_path.exists():
            self._load_cache()
        # Wave 110 / Phase D: budget tracking + telemetry persistence.
        self._budget = _BudgetTracker(
            telemetry_path=telemetry_path,
            max_dispatches=max_dispatches,
        )
        # Wave 111 / Phase E: circuit breaker for repeated dispatcher failures.
        self._breaker = _CircuitBreaker(
            failures_to_open=failures_to_open,
            window_seconds=failure_window_seconds,
        )

    @property
    def budget(self) -> _BudgetTracker:
        """Read-only access to the budget tracker for callers that
        want to log a summary at end of run."""
        return self._budget

    def paraphrase_instruction(
        self, draft: Dict[str, Any], chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Paraphrase a mock-drafted instruction pair."""
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")
        key = self._cache_key(kind="instruction", chunk_id=chunk_id, draft=draft)
        cached = self._cache.get(key)
        if cached is not None:
            self._budget.record(
                kind="instruction", chunk_id=chunk_id,
                cached=True, elapsed_seconds=0.0,
            )
            out = dict(draft)
            out["prompt"] = str(cached["prompt"])
            out["completion"] = str(cached["completion"])
            out["provider"] = "claude_session"
            return out

        # Wave 110 / Phase D: pre-flight cap so we raise BEFORE
        # contacting the dispatcher. The cache + telemetry written so
        # far stay on disk; resume by re-running with a higher cap.
        if (
            self._budget.max_dispatches is not None
            and self._budget.dispatched >= self._budget.max_dispatches
        ):
            raise SynthesisBudgetExceeded(
                f"ClaudeSessionProvider hit max_dispatches="
                f"{self._budget.max_dispatches} (dispatched="
                f"{self._budget.dispatched}, cache_hits="
                f"{self._budget.cache_hits}). Re-run with a higher "
                f"--max-dispatches to resume from cache.",
                dispatched=self._budget.dispatched,
                cache_hits=self._budget.cache_hits,
                max_dispatches=self._budget.max_dispatches,
            )

        t0 = time.monotonic()
        outputs = asyncio.run(
            self._dispatch(
                kind="instruction",
                draft=draft,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                expected_keys=_INSTRUCTION_KEYS,
            )
        )
        elapsed = time.monotonic() - t0
        # Wave 112 Task 4: clamp lengths BEFORE persisting to cache so a
        # poisoned response (short paraphrase) never lands on disk.
        _validate_lengths(outputs, kind="instruction", chunk_id=chunk_id or None)
        self._budget.record(
            kind="instruction", chunk_id=chunk_id,
            cached=False, elapsed_seconds=elapsed,
        )
        self._cache_store(
            key=key, kind="instruction", chunk_id=chunk_id, outputs=outputs,
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
        key = self._cache_key(kind="preference", chunk_id=chunk_id, draft=draft)
        cached = self._cache.get(key)
        if cached is not None:
            self._budget.record(
                kind="preference", chunk_id=chunk_id,
                cached=True, elapsed_seconds=0.0,
            )
            out = dict(draft)
            out["prompt"] = str(cached["prompt"])
            out["chosen"] = str(cached["chosen"])
            out["rejected"] = str(cached["rejected"])
            out["provider"] = "claude_session"
            return out

        # Wave 110 / Phase D: same pre-flight cap as paraphrase_instruction.
        if (
            self._budget.max_dispatches is not None
            and self._budget.dispatched >= self._budget.max_dispatches
        ):
            raise SynthesisBudgetExceeded(
                f"ClaudeSessionProvider hit max_dispatches="
                f"{self._budget.max_dispatches} (dispatched="
                f"{self._budget.dispatched}, cache_hits="
                f"{self._budget.cache_hits}). Re-run with a higher "
                f"--max-dispatches to resume from cache.",
                dispatched=self._budget.dispatched,
                cache_hits=self._budget.cache_hits,
                max_dispatches=self._budget.max_dispatches,
            )

        t0 = time.monotonic()
        outputs = asyncio.run(
            self._dispatch(
                kind="preference",
                draft=draft,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                expected_keys=_PREFERENCE_KEYS,
            )
        )
        elapsed = time.monotonic() - t0
        # Wave 112 Task 4: clamp lengths BEFORE persisting to cache so a
        # poisoned response (short paraphrase) never lands on disk.
        _validate_lengths(outputs, kind="preference", chunk_id=chunk_id or None)
        self._budget.record(
            kind="preference", chunk_id=chunk_id,
            cached=False, elapsed_seconds=elapsed,
        )
        self._cache_store(
            key=key, kind="preference", chunk_id=chunk_id, outputs=outputs,
        )
        out = dict(draft)
        out["prompt"] = str(outputs["prompt"])
        out["chosen"] = str(outputs["chosen"])
        out["rejected"] = str(outputs["rejected"])
        out["provider"] = "claude_session"
        self._emit_decision(kind="preference", draft=draft, chunk_id=chunk_id)
        return out

    async def _dispatch(
        self,
        *,
        kind: str,
        draft: Dict[str, Any],
        chunk_id: str,
        chunk_text: str,
        expected_keys: List[str],
    ) -> Dict[str, Any]:
        # Wave 111 / Phase E: circuit breaker fail-fast on repeated timeouts.
        # Raises SynthesisCircuitOpen before contacting the dispatcher.
        self._breaker.before_dispatch()
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
            self._breaker.record_failure(
                error_code=str(result.get("error_code") or "UNKNOWN"),
            )
            raise RuntimeError(
                f"training-synthesizer dispatch failed: "
                f"code={result.get('error_code')!r} error={result.get('error')!r}"
            )
        outputs = result.get("outputs") or {}
        for key in expected_keys:
            if key not in outputs:
                self._breaker.record_failure(error_code="MALFORMED_OUTPUT")
                raise RuntimeError(
                    f"training-synthesizer returned malformed output "
                    f"for kind={kind}: missing key {key!r}; got {sorted(outputs)!r}"
                )
        # Wave 112 Task 3: tighten value validation. The key-presence loop
        # above accepted ``""``, ``"   "``, and ``None`` — all of which would
        # silently poison ``instruction_pairs.jsonl`` downstream. Reject them
        # here before the cache layer (Task 7) or the length clamp (Task 4)
        # ever sees the value.
        for key in expected_keys:
            value = outputs[key]
            if not isinstance(value, str) or not value.strip():
                self._breaker.record_failure(error_code="EMPTY_FIELD")
                raise SynthesisProviderError(
                    f"training-synthesizer returned empty/non-string value "
                    f"for kind={kind}, key={key!r}: got {value!r}",
                    code="empty_field",
                    chunk_id=chunk_id or None,
                )
        self._breaker.record_success()
        return outputs

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

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("provider_version") != self._provider_version:
                continue
            self._cache[entry["key"]] = entry["outputs"]

    def _cache_key(self, *, kind: str, chunk_id: str, draft: Dict[str, Any]) -> str:
        # Serialize draft keys deterministically; only the load-bearing fields
        # (prompt, completion / chosen / rejected, template_id) participate so
        # incidental metadata changes don't bust the cache.
        relevant = {
            k: draft.get(k)
            for k in ("prompt", "completion", "chosen", "rejected", "template_id")
            if k in draft
        }
        payload = json.dumps(
            {
                "version": self._provider_version,
                "kind": kind,
                "chunk_id": chunk_id,
                "draft": relevant,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_store(
        self,
        *,
        key: str,
        kind: str,
        chunk_id: str,
        outputs: Dict[str, Any],
    ) -> None:
        self._cache[key] = outputs
        if self._cache_path is None:
            return
        line = json.dumps(
            {
                "key": key,
                "kind": kind,
                "chunk_id": chunk_id,
                "provider_version": self._provider_version,
                "outputs": outputs,
            },
            sort_keys=True,
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
