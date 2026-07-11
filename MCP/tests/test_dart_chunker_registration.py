"""
Phase 7b Subtask 9 — regression test for the chunker agent registration.

DART->semantik naming purge (task #19 Stage 3): the agent was renamed
``dart-chunker`` -> ``semantik-chunker``. The legacy ``dart-chunker`` name
is kept as a read-compat dispatch alias in ``AGENT_TOOL_MAPPING`` so old
resume states / configs still route.

Pins both registration sites so a future refactor can't silently drop
either half of the chunker wiring:

1. ``MCP/core/executor.py::AGENT_TOOL_MAPPING`` carries a
   ``"semantik-chunker": "run_dart_chunking"`` entry (canonical) plus the
   legacy ``"dart-chunker"`` alias. The actual ``run_dart_chunking`` helper
   is implemented by ST 11 and registered in
   ``MCP/tools/pipeline_tools.py::_build_tool_registry``; this test asserts
   only the mapping shape.

2. ``config/agents.yaml`` has a ``semantik-chunker`` entry under ``agents``
   with ``type: utility`` (mirroring the ``textbook-stager`` precedent —
   deterministic transformation, in-code dispatcher, no ``.md`` spec).

3. ``semantik-chunker`` does NOT leak into ``AGENT_SUBAGENT_SET`` — it is
   a deterministic chunker, not an LLM-reasoning agent, so it must
   stay on the in-process ``_invoke_tool`` path regardless of
   ``ED4ALL_AGENT_DISPATCH``.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from MCP.core.executor import AGENT_SUBAGENT_SET, AGENT_TOOL_MAPPING
except ImportError:
    pytest.skip("executor not available", allow_module_level=True)


REPO_ROOT = Path(__file__).parent.parent.parent
AGENTS_YAML = REPO_ROOT / "config" / "agents.yaml"


@pytest.mark.unit
def test_semantik_chunker_in_agent_tool_mapping():
    """``semantik-chunker`` (canonical) must map to ``run_dart_chunking``,
    and the legacy ``dart-chunker`` alias must still resolve (read-compat)."""
    assert "semantik-chunker" in AGENT_TOOL_MAPPING, (
        "task #19 Stage 3 regression — semantik-chunker missing from "
        "AGENT_TOOL_MAPPING in MCP/core/executor.py"
    )
    assert AGENT_TOOL_MAPPING["semantik-chunker"] == "run_dart_chunking", (
        "task #19 Stage 3 regression — semantik-chunker must map to "
        "run_dart_chunking (the ST 11 helper registered in "
        "MCP/tools/pipeline_tools.py::_build_tool_registry); got "
        f"{AGENT_TOOL_MAPPING['semantik-chunker']!r}"
    )
    # Legacy dart-chunker dispatch alias survives (Stage-1 read-compat).
    assert AGENT_TOOL_MAPPING.get("dart-chunker") == "run_dart_chunking", (
        "task #19 Stage 3 regression — legacy dart-chunker dispatch alias "
        "must still resolve to run_dart_chunking for old resume states"
    )


@pytest.mark.unit
def test_semantik_chunker_in_agents_yaml():
    """``semantik-chunker`` must be registered in ``config/agents.yaml``."""
    data = yaml.safe_load(AGENTS_YAML.read_text())
    agents = data.get("agents", {})
    assert "semantik-chunker" in agents, (
        "task #19 Stage 3 regression — semantik-chunker missing from "
        "config/agents.yaml::agents"
    )

    entry = agents["semantik-chunker"]
    assert entry.get("type") == "utility", (
        "task #19 Stage 3 regression — semantik-chunker must be type=utility "
        "(deterministic chunker, no LLM dispatch — mirrors "
        f"textbook-stager precedent); got type={entry.get('type')!r}"
    )
    # Capabilities list is a soft contract — assert it exists and is
    # non-empty so a future drop-by-merge-conflict gets caught, but
    # don't pin the exact strings.
    capabilities = entry.get("capabilities") or []
    assert capabilities, (
        "task #19 Stage 3 regression — semantik-chunker must declare at "
        "least one capability"
    )


@pytest.mark.unit
def test_semantik_chunker_is_not_subagent():
    """``semantik-chunker`` is a deterministic utility — must NOT be in
    ``AGENT_SUBAGENT_SET`` (which would route it through
    ``dispatcher.dispatch_task`` when ``ED4ALL_AGENT_DISPATCH=true``)."""
    assert "semantik-chunker" not in AGENT_SUBAGENT_SET, (
        "task #19 Stage 3 regression — semantik-chunker leaked into "
        "AGENT_SUBAGENT_SET. The chunker is a deterministic "
        "transformation with no LLM dispatch; it must stay on the "
        "in-process _invoke_tool path."
    )
