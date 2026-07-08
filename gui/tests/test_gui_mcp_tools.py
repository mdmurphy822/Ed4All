"""Tests for ``MCP/tools/gui_tools.py`` — the Claude<->GUI MCP bridge.

The MCP server imports its tool modules as top-level ``tools.*`` (it runs from
inside ``MCP/``, putting that dir on ``sys.path``). We replicate that sys.path
handling here, register the tools on a REAL ``FastMCP`` instance, and call the
underlying coroutine functions to assert they round-trip through the SAME
``state/gui/`` store the GUI reads — verifying the bridge, not a stub.

State is isolated via ``state_dir`` (``ED4ALL_STATE_RUNS_DIR`` -> tmp_path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _import_register_gui_tools():
    """Import ``register_gui_tools`` the way the MCP server resolves it, then
    fully undo the global side effects.

    The server runs from inside ``MCP/`` so its tool modules are addressable as
    top-level ``tools.*``. We replicate that (put ``MCP/`` on ``sys.path``,
    import ``tools.gui_tools``), capture the function object, and then REMOVE the
    ``MCP/`` path entry + evict the cached top-level ``tools`` package. The
    captured function stays valid after its module is evicted.

    This cleanup is essential: ``MCP/tools`` is a *regular* package (it ships an
    ``__init__.py``), so leaving it bound to the top-level ``tools`` name (or
    ``MCP/`` on ``sys.path``) would permanently shadow the namespace package
    ``LibV2/tools`` that the retrieval service imports — a shared-interpreter
    collision the real ``ed4all gui`` process never hits (it imports
    ``MCP.tools.*``, never the bare ``tools`` name).
    """
    repo_root = Path(__file__).resolve().parents[2]
    mcp_dir = repo_root / "MCP"
    added = []
    for p in (str(mcp_dir), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    before_tools = {
        k: v for k, v in sys.modules.items() if k == "tools" or k.startswith("tools.")
    }
    try:
        from tools.gui_tools import register_gui_tools as fn  # noqa: PLC0415
    finally:
        # Evict whatever ``tools*`` this import introduced, restore prior state.
        for name in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
            if name not in before_tools:
                del sys.modules[name]
        for name, mod in before_tools.items():
            sys.modules[name] = mod
        # Drop only the path entries we added (the MCP dir is the dangerous one).
        if str(mcp_dir) in added and str(mcp_dir) in sys.path:
            sys.path.remove(str(mcp_dir))
    return fn


try:
    register_gui_tools = _import_register_gui_tools()
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"tools.gui_tools not importable: {exc}", allow_module_level=True)

from gui import shared_state  # noqa: E402

mcp_mod = pytest.importorskip("mcp.server.fastmcp")
FastMCP = mcp_mod.FastMCP


@pytest.fixture
def gui_tools():
    """Register the GUI tools on a real FastMCP and expose them by name.

    FastMCP's ``@mcp.tool()`` wraps each coroutine but keeps it callable; we
    capture the registered callables via a thin shim so tests can invoke the
    REAL tool bodies directly.
    """
    captured = {}

    class CapturingMCP(FastMCP):
        def tool(self, *a, **k):
            real = super().tool(*a, **k)

            def decorator(fn):
                captured[fn.__name__] = fn
                return real(fn)

            return decorator

    mcp = CapturingMCP("test-gui")
    register_gui_tools(mcp)
    return captured


async def test_set_and_get_settings_round_trip(state_dir, gui_tools):
    # Set a routing provider via the dotted-path tool.
    out = await gui_tools["gui_set_setting"]("model_routing.global.provider", "local")
    saved = json.loads(out)
    assert saved["model_routing"]["global"]["provider"] == "local"

    # gui_get_settings reflects the change (and masks secrets).
    got = json.loads(await gui_tools["gui_get_settings"]())
    assert got["model_routing"]["global"]["provider"] == "local"


async def test_set_setting_masks_secret(state_dir, gui_tools):
    out = await gui_tools["gui_set_setting"]("env.ANTHROPIC_API_KEY", "sk-secret-123")
    doc = json.loads(out)
    assert doc["env"]["ANTHROPIC_API_KEY"] == "set"
    assert "sk-secret-123" not in out
    # At rest the secret is split into the 0600 sidecar; settings.json holds no
    # plaintext. The raw value stays resolvable for run-time consumers.
    settings_text = shared_state.settings_path().read_text()
    assert "sk-secret-123" not in settings_text
    assert json.loads(settings_text)["env"]["ANTHROPIC_API_KEY"] is None
    sidecar = shared_state.settings_path().parent / "secrets.json"
    assert json.loads(sidecar.read_text())["ANTHROPIC_API_KEY"] == "sk-secret-123"


async def test_set_setting_invalid_provider_typed_error(state_dir, gui_tools):
    out = await gui_tools["gui_set_setting"]("model_routing.global.provider", "bogus")
    err = json.loads(out)
    assert "error" in err
    assert err.get("detail") == "validation_error"


async def test_post_event_readable_by_gui_side(state_dir, gui_tools):
    """A claude-sourced event posted via the tool is readable via shared_state."""
    rec = json.loads(await gui_tools["gui_post_event"]("message", '{"text": "hi gui"}'))
    assert rec["source"] == "claude"
    assert rec["kind"] == "message"

    # The GUI side reads it via shared_state.read_events (what /api/activity/events uses).
    events = shared_state.read_events(0)
    assert any(
        e["source"] == "claude" and e["payload"].get("text") == "hi gui" for e in events
    )


async def test_read_events_sees_gui_message(state_dir, gui_tools):
    """A gui-sourced human message is visible to gui_read_events (symmetry)."""
    shared_state.append_event("gui", "human_message", {"text": "from the operator"})
    out = json.loads(await gui_tools["gui_read_events"](0))
    assert any(
        e["source"] == "gui" and e["payload"].get("text") == "from the operator"
        for e in out
    )


async def test_enqueue_run_registers_requested(state_dir, gui_tools):
    out = await gui_tools["gui_enqueue_run"](
        "textbook_to_course", "PHYS_101", "/tmp/x.pdf", '{"weeks": 12}'
    )
    rec = json.loads(out)
    assert rec["status"] == "requested"
    assert rec["source"] == "claude"
    assert rec["workflow"] == "textbook_to_course"
    assert rec["options"] == {"weeks": 12}
    run_id = rec["run_id"]

    # The GUI side sees it in the run registry.
    listed = json.loads(await gui_tools["gui_list_runs"]())
    assert any(r["run_id"] == run_id and r["status"] == "requested" for r in listed)
    one = json.loads(await gui_tools["gui_get_run"](run_id))
    assert one["run_id"] == run_id


async def test_enqueue_run_bad_options_json_typed_error(state_dir, gui_tools):
    out = await gui_tools["gui_enqueue_run"]("wf", "C1", "/tmp/x.pdf", "{not json")
    err = json.loads(out)
    assert "error" in err
    assert err.get("detail") == "validation_error"


async def test_get_run_unknown_typed_error(state_dir, gui_tools):
    out = await gui_tools["gui_get_run"]("REQ-nope-000000")
    err = json.loads(out)
    assert err.get("detail") == "not_found"
