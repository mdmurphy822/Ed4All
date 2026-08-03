"""Wave6 regression: ``claude_session`` rewrite-tier provider.

Exercises the new ``provider="claude_session"`` branch in
:class:`Courseforge.generators._rewrite_provider.RewriteProvider` that
dispatches the rewrite tier through
:class:`MCP.orchestrator.local_dispatcher.LocalDispatcher` to the
``content-generator`` Claude Code subagent instead of routing through
HTTP / the Anthropic SDK.

Mirrors the Trainforge precedent at
``Trainforge/tests/test_claude_session_provider.py`` (Wave 107 Phase A)
adapted to the rewrite-tier's Block-in / Block-out contract.

Coverage:

1. **Construction**: ``RewriteProvider(provider="claude_session",
   dispatcher=<mock>)`` succeeds; without a dispatcher, the constructor
   raises ``RuntimeError`` with a clear message.
2. **Dispatch shape**: a mock dispatcher records the call and asserts
   ``agent_type="content-generator"`` plus the expected ``task_params``
   keys (kind / system_prompt / user_prompt / prompt / model /
   max_tokens / temperature / expected_keys).
3. **Response parsing**: a structured-dict dispatcher response with
   ``outputs={"html": ...}`` lands as ``Block.content`` plus a
   ``Touch(tier="rewrite", ...)`` audit entry.
4. **W-27 directive propagation**: the ``prompt`` field passed in
   ``task_params`` carries the Wave4-W27 MANDATORY directives — verified
   via substring checks on ``data-cf-source-ids`` and ``HEADING_SKIP``
   substrings the directives reference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the same import bridge the production provider sets up so
# ``from blocks import Block`` resolves regardless of how this test is
# loaded under pytest.
_COURSEFORGE_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_COURSEFORGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_COURSEFORGE_SCRIPTS))

from Courseforge.generators._rewrite_provider import (  # noqa: E402
    ENV_PROVIDER,
    RewriteProvider,
    _CLAUDE_SESSION_AGENT_TYPE,
    _CLAUDE_SESSION_TASK_NAME,
    _NO_DISPATCHER_MSG,
    _REWRITE_SYSTEM_PROMPT,
)
from blocks import Block, Touch  # noqa: E402  (Phase 2 intermediate format)


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _FakeDispatcher:
    """Minimal stand-in for
    :class:`MCP.orchestrator.local_dispatcher.LocalDispatcher`.

    Records every call as ``(agent_type, task_name, task_params, run_id)``
    on ``self.calls`` so each test can assert the dispatch shape. The
    ``agent_tool`` callable supplies the dispatcher's return envelope.
    """

    def __init__(
        self,
        agent_tool: Callable[..., Awaitable[Dict[str, Any]]],
    ) -> None:
        self._agent_tool = agent_tool
        self.calls: List[Tuple[str, str, Dict[str, Any], str]] = []

    async def dispatch_task(
        self,
        *,
        task_name: str,
        agent_type: str,
        task_params: Dict[str, Any],
        run_id: str,
        phase_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Snapshot the params dict so subsequent mutation by the
        # production code path can't retroactively change the test
        # assertion subject.
        self.calls.append(
            (agent_type, task_name, dict(task_params), run_id)
        )
        return await self._agent_tool(
            task_name=task_name,
            agent_type=agent_type,
            task_params=task_params,
            run_id=run_id,
            phase_context=phase_context,
        )


def _outline_block(
    *,
    block_type: str = "concept",
    curies: Optional[List[str]] = None,
    escalation_marker: Optional[str] = None,
) -> Block:
    """Build a minimal outline-tier Block for rewrite-tier dispatch.

    The shape mirrors the legacy
    ``Courseforge/generators/tests/test_rewrite_provider.py::_outline_block``
    helper so this Wave6 test asserts the same field surface the rest
    of the suite exercises.
    """
    return Block(
        block_id="page#concept_intro_0",
        block_type=block_type,
        page_id="page",
        sequence=0,
        content={
            "key_claims": ["The central concept is X."],
            "curies": list(curies or []),
            "source_refs": ["semantik:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
        escalation_marker=escalation_marker,
    )


def _make_html_response(html: str) -> Dict[str, Any]:
    """Shape the dispatcher returns for a successful rewrite."""
    return {
        "success": True,
        "outputs": {"html": html},
        "artifacts": [],
    }


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


def test_construction_succeeds_with_dispatcher(monkeypatch):
    """``RewriteProvider(provider='claude_session', dispatcher=<mock>)``
    constructs without raising; no ANTHROPIC_API_KEY required."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    async def _stub(**_kwargs: Any) -> Dict[str, Any]:
        return _make_html_response("<section></section>")

    dispatcher = _FakeDispatcher(_stub)
    p = RewriteProvider(
        provider="claude_session",
        dispatcher=dispatcher,
    )

    assert p._provider == "claude_session"
    assert p._dispatcher is dispatcher
    # No HTTP / SDK plumbing should be wired for claude_session.
    assert p._oa_client is None
    assert p._anthropic_client is None


def test_construction_without_dispatcher_raises(monkeypatch):
    """``provider='claude_session'`` without a dispatcher fails loud —
    standalone scripts that don't run inside a workflow can't dispatch
    to the Claude Code session. Mirrors the Trainforge
    ``ClaudeSessionProvider`` contract."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        RewriteProvider(provider="claude_session", dispatcher=None)
    # The message names the required injection point so an operator can
    # see why their CLI invocation failed.
    msg = str(excinfo.value)
    assert "LocalDispatcher" in msg
    # Sanity: the canonical fail-closed message exposed by the module
    # is the one we raised.
    assert msg == _NO_DISPATCHER_MSG


def test_env_var_selects_claude_session(monkeypatch):
    """``COURSEFORGE_REWRITE_PROVIDER=claude_session`` resolves at
    construction time — env-var-first contract is preserved across the
    new provider value."""
    monkeypatch.setenv(ENV_PROVIDER, "claude_session")

    async def _stub(**_kwargs: Any) -> Dict[str, Any]:
        return _make_html_response("<section></section>")

    p = RewriteProvider(dispatcher=_FakeDispatcher(_stub))
    assert p._provider == "claude_session"


# ---------------------------------------------------------------------------
# 2. Dispatch shape
# ---------------------------------------------------------------------------


def test_dispatch_calls_content_generator_agent(monkeypatch):
    """``generate_rewrite`` dispatches to ``agent_type='content-generator'``
    (NOT a new agent name) with the canonical task_params surface so the
    Wave4b ``model: sonnet`` frontmatter + Wave4-W27 directives reach the
    rewrite-tier dispatch unchanged."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)

    seen_html = (
        "<section data-cf-source-ids=\"semantik:slug#blk1\">"
        "<h2 data-cf-content-type=\"concept\">Concept</h2>"
        "<p>The central concept is X.</p>"
        "</section>"
    )

    async def _agent(**_kwargs: Any) -> Dict[str, Any]:
        return _make_html_response(seen_html)

    dispatcher = _FakeDispatcher(_agent)
    p = RewriteProvider(
        provider="claude_session",
        dispatcher=dispatcher,
        run_id="run-gamma",
    )
    block = _outline_block(curies=[])
    out = p.generate_rewrite(block)

    # One dispatch call landed on the agent.
    assert len(dispatcher.calls) == 1
    agent_type, task_name, params, run_id = dispatcher.calls[0]
    assert agent_type == _CLAUDE_SESSION_AGENT_TYPE == "content-generator"
    assert task_name == _CLAUDE_SESSION_TASK_NAME
    assert run_id == "run-gamma"
    # Expected task_params surface — every key the production
    # ``_dispatch_call`` is contractually obliged to emit.
    for key in (
        "kind",
        "system_prompt",
        "user_prompt",
        "prompt",
        "model",
        "max_tokens",
        "temperature",
        "expected_keys",
    ):
        assert key in params, f"task_params missing {key!r}"
    assert params["kind"] == "rewrite"
    assert params["expected_keys"] == ["html"]
    # Block returned with the rewrite payload + Touch entry.
    assert isinstance(out, Block)
    assert isinstance(out.content, str)
    assert "central concept is X" in out.content


# ---------------------------------------------------------------------------
# 3. Response parsing
# ---------------------------------------------------------------------------


def test_response_parsing_assembles_block_with_rewrite_touch(monkeypatch):
    """A successful dispatcher response (``{"success": True,
    "outputs": {"html": ...}}``) lands as the new Block's ``content`` plus
    a cumulative ``Touch(tier='rewrite', purpose='pedagogical_depth', ...)``
    audit entry on the Block's ``touched_by`` chain."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)

    html_body = (
        "<section data-cf-source-ids=\"semantik:slug#blk1\">"
        "<h2 data-cf-content-type=\"concept\">Concept</h2>"
        "<p>Pedagogically rich prose.</p>"
        "</section>"
    )

    async def _agent(**_kwargs: Any) -> Dict[str, Any]:
        return _make_html_response(html_body)

    dispatcher = _FakeDispatcher(_agent)
    p = RewriteProvider(
        provider="claude_session",
        dispatcher=dispatcher,
    )
    block = _outline_block(curies=[])
    out = p.generate_rewrite(block)

    # Content replaced with the rewrite HTML.
    assert out.content == html_body
    # Touch chain extended by exactly one rewrite-tier Touch.
    assert len(out.touched_by) == len(block.touched_by) + 1
    new_touch = out.touched_by[-1]
    assert isinstance(new_touch, Touch)
    assert new_touch.tier == "rewrite"
    assert new_touch.purpose == "pedagogical_depth"
    assert new_touch.provider == "claude_session"


def test_response_parsing_failure_envelope_raises(monkeypatch):
    """A dispatcher response with ``success=False`` raises ``RuntimeError``
    naming the error code so the upstream router can branch on it."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)

    async def _agent(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "success": False,
            "error_code": "MAILBOX_TIMEOUT",
            "error": "subagent did not respond within 1800s",
        }

    dispatcher = _FakeDispatcher(_agent)
    p = RewriteProvider(
        provider="claude_session",
        dispatcher=dispatcher,
    )
    block = _outline_block(curies=[])
    with pytest.raises(RuntimeError) as excinfo:
        p.generate_rewrite(block)
    msg = str(excinfo.value)
    assert "MAILBOX_TIMEOUT" in msg
    assert "subagent did not respond" in msg


def test_response_parsing_empty_outputs_raises(monkeypatch):
    """A dispatcher response missing the ``html`` key (and any acceptable
    fallback key) raises so the parse-retry loop sees a hard failure
    instead of silently emitting an empty Block."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)

    async def _agent(**_kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "outputs": {}, "artifacts": []}

    dispatcher = _FakeDispatcher(_agent)
    p = RewriteProvider(
        provider="claude_session",
        dispatcher=dispatcher,
    )
    block = _outline_block(curies=[])
    with pytest.raises(RuntimeError) as excinfo:
        p.generate_rewrite(block)
    assert "empty" in str(excinfo.value).lower() or "missing" in str(
        excinfo.value
    ).lower()


# ---------------------------------------------------------------------------
# 4. W-27 directive propagation
# ---------------------------------------------------------------------------


def test_w27_directives_reach_subagent_prompt_body(monkeypatch):
    """The ``prompt`` field passed in ``task_params`` carries the Wave4-W27
    MANDATORY directives so the dispatched subagent inherits the same
    HEADING_SKIP avoidance + source-ID stamping contract the in-process
    Anthropic / local providers see via ``_REWRITE_SYSTEM_PROMPT``.

    Substring checks anchor on the two load-bearing directive tokens
    enumerated in the Wave5 W-27 propagation:

    - ``data-cf-source-ids`` (EMPTY_SOURCE_REFS / source-ID stamping)
    - ``HEADING_SKIP`` (heading-hierarchy critical gate)
    """
    monkeypatch.delenv(ENV_PROVIDER, raising=False)

    async def _agent(**_kwargs: Any) -> Dict[str, Any]:
        return _make_html_response("<section data-cf-source-ids=\"x\"></section>")

    dispatcher = _FakeDispatcher(_agent)
    p = RewriteProvider(
        provider="claude_session",
        dispatcher=dispatcher,
    )
    block = _outline_block(curies=[])
    p.generate_rewrite(block)

    assert len(dispatcher.calls) == 1
    _, _, params, _ = dispatcher.calls[0]
    prompt = params["prompt"]
    # The system prompt's load-bearing W-27 directive tokens MUST appear
    # in the prompt body the subagent receives.
    assert "data-cf-source-ids" in prompt, (
        "W-27 source-ID stamping directive missing from dispatch prompt"
    )
    assert "HEADING_SKIP" in prompt, (
        "W-27 heading-hierarchy directive missing from dispatch prompt"
    )
    # And the canonical _REWRITE_SYSTEM_PROMPT is the actual source —
    # sanity-check that the dispatch isn't sending a stripped variant.
    assert "data-cf-source-ids" in _REWRITE_SYSTEM_PROMPT
    assert "HEADING_SKIP" in _REWRITE_SYSTEM_PROMPT
    # The system_prompt field is also passed verbatim so a subagent that
    # prefers structured fields over the concatenated ``prompt`` body
    # still sees the directives.
    assert params["system_prompt"] == _REWRITE_SYSTEM_PROMPT
