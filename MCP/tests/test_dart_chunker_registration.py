"""Regression test for the chunker agent registration.

The canonical agent name is ``semantik-chunker``; the legacy
``dart-chunker`` name survives ONLY as a read-compat dispatch alias in
``AGENT_TOOL_MAPPING``, so paused runs resumed from an old checkpoint
still route. Removing the alias breaks resume.

Pins three sites so a refactor can't silently drop half the wiring:

1. ``MCP/core/executor.py::AGENT_TOOL_MAPPING`` maps both
   ``semantik-chunker`` (canonical) and ``dart-chunker`` (alias) to
   ``run_dart_chunking``. Only the mapping shape is asserted here; the
   helper itself is registered in
   ``MCP/tools/pipeline_tools.py::_build_tool_registry``.

2. ``config/agents.yaml`` carries a ``semantik-chunker`` entry with
   ``type: utility`` — a deterministic transformation dispatched
   in-code, with no ``.md`` agent spec.

3. ``semantik-chunker`` does NOT appear in ``AGENT_SUBAGENT_SET``. It
   runs no LLM, so it must stay on the in-process ``_invoke_tool`` path
   regardless of ``ED4ALL_AGENT_DISPATCH``.
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
        "semantik-chunker missing from AGENT_TOOL_MAPPING in "
        "MCP/core/executor.py"
    )
    assert AGENT_TOOL_MAPPING["semantik-chunker"] == "run_dart_chunking", (
        "semantik-chunker must map to run_dart_chunking (registered in "
        "MCP/tools/pipeline_tools.py::_build_tool_registry); got "
        f"{AGENT_TOOL_MAPPING['semantik-chunker']!r}"
    )
    assert AGENT_TOOL_MAPPING.get("dart-chunker") == "run_dart_chunking", (
        "the legacy dart-chunker alias must still resolve to "
        "run_dart_chunking, or runs resumed from an old checkpoint "
        "fail to dispatch"
    )


@pytest.mark.unit
def test_semantik_chunker_in_agents_yaml():
    """``semantik-chunker`` must be registered in ``config/agents.yaml``."""
    data = yaml.safe_load(AGENTS_YAML.read_text())
    agents = data.get("agents", {})
    assert "semantik-chunker" in agents, (
        "semantik-chunker missing from config/agents.yaml::agents"
    )

    entry = agents["semantik-chunker"]
    assert entry.get("type") == "utility", (
        "semantik-chunker must be type=utility (deterministic chunker, "
        f"no LLM dispatch); got type={entry.get('type')!r}"
    )
    # The capabilities list is a soft contract: assert only that it is
    # non-empty, so a drop-by-merge-conflict is caught without pinning
    # strings that legitimately evolve.
    capabilities = entry.get("capabilities") or []
    assert capabilities, (
        "semantik-chunker must declare at least one capability"
    )


@pytest.mark.unit
def test_semantik_chunker_is_not_subagent():
    """``semantik-chunker`` is a deterministic utility — must NOT be in
    ``AGENT_SUBAGENT_SET`` (which would route it through
    ``dispatcher.dispatch_task`` when ``ED4ALL_AGENT_DISPATCH=true``)."""
    assert "semantik-chunker" not in AGENT_SUBAGENT_SET, (
        "semantik-chunker leaked into AGENT_SUBAGENT_SET. The chunker "
        "is a deterministic transformation with no LLM dispatch; it "
        "must stay on the in-process _invoke_tool path."
    )
