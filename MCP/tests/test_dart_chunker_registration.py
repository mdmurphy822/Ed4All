"""
Phase 7b Subtask 9 — regression test for ``dart-chunker`` agent
registration.

Pins both registration sites so a future refactor can't silently drop
either half of the ``dart-chunker`` wiring:

1. ``MCP/core/executor.py::AGENT_TOOL_MAPPING`` carries a
   ``"dart-chunker": "run_dart_chunking"`` entry. The actual
   ``run_dart_chunking`` helper is implemented by ST 11 and registered
   in ``MCP/tools/pipeline_tools.py::_build_tool_registry``; this test
   asserts only the mapping shape (the helper-resolution check belongs
   in ``test_pipeline_tools.py`` once ST 11 lands).

2. ``config/agents.yaml`` has a ``dart-chunker`` entry under ``agents``
   with ``type: utility`` (mirroring the ``textbook-stager``
   precedent — deterministic transformation, in-code dispatcher, no
   ``.md`` spec).

3. ``dart-chunker`` does NOT leak into ``AGENT_SUBAGENT_SET`` — it is
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
def test_dart_chunker_in_agent_tool_mapping():
    """``dart-chunker`` must map to ``run_dart_chunking``."""
    assert "dart-chunker" in AGENT_TOOL_MAPPING, (
        "Phase 7b ST 9 regression — dart-chunker missing from "
        "AGENT_TOOL_MAPPING in MCP/core/executor.py"
    )
    assert AGENT_TOOL_MAPPING["dart-chunker"] == "run_dart_chunking", (
        "Phase 7b ST 9 regression — dart-chunker must map to "
        "run_dart_chunking (the ST 11 helper registered in "
        "MCP/tools/pipeline_tools.py::_build_tool_registry); got "
        f"{AGENT_TOOL_MAPPING['dart-chunker']!r}"
    )


@pytest.mark.unit
def test_dart_chunker_in_agents_yaml():
    """``dart-chunker`` must be registered in ``config/agents.yaml``."""
    data = yaml.safe_load(AGENTS_YAML.read_text())
    agents = data.get("agents", {})
    assert "dart-chunker" in agents, (
        "Phase 7b ST 9 regression — dart-chunker missing from "
        "config/agents.yaml::agents"
    )

    entry = agents["dart-chunker"]
    assert entry.get("type") == "utility", (
        "Phase 7b ST 9 regression — dart-chunker must be type=utility "
        "(deterministic chunker, no LLM dispatch — mirrors "
        f"textbook-stager precedent); got type={entry.get('type')!r}"
    )
    # Capabilities list is a soft contract — assert it exists and is
    # non-empty so a future drop-by-merge-conflict gets caught, but
    # don't pin the exact strings.
    capabilities = entry.get("capabilities") or []
    assert capabilities, (
        "Phase 7b ST 9 regression — dart-chunker must declare at least "
        "one capability"
    )


@pytest.mark.unit
def test_dart_chunker_is_not_subagent():
    """``dart-chunker`` is a deterministic utility — must NOT be in
    ``AGENT_SUBAGENT_SET`` (which would route it through
    ``dispatcher.dispatch_task`` when ``ED4ALL_AGENT_DISPATCH=true``)."""
    assert "dart-chunker" not in AGENT_SUBAGENT_SET, (
        "Phase 7b ST 9 regression — dart-chunker leaked into "
        "AGENT_SUBAGENT_SET. The chunker is a deterministic "
        "transformation with no LLM dispatch; it must stay on the "
        "in-process _invoke_tool path."
    )
