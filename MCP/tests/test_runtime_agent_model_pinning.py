"""Wave4b-runtime-model — pin runtime-dispatched subagents to ``model: sonnet``.

The 12 agents in ``MCP/core/executor.AGENT_SUBAGENT_SET`` are dispatched as
Claude Code subagents during ``ed4all run --mode local`` runs. Without
explicit ``model: sonnet`` frontmatter on their ``.md`` spec files, the
spawned subagent inherits the operator's parent session model — so an
operator running the pipeline from an Opus session ends up paying Opus
for every runtime-dispatched subagent. Project rule: pipeline workers
run on Opus (the operator session), runtime subagents run on Sonnet.

This module pins the frontmatter contract:

1. Each of the 12 ``AGENT_SUBAGENT_SET`` agents has a corresponding
   ``.md`` file at one of the registered ``AGENT_SPEC_DIRS`` locations
   (Courseforge/agents/, Trainforge/agents/).
2. Every such ``.md`` file starts with a YAML frontmatter block.
3. The frontmatter carries ``model: sonnet``.
4. The frontmatter's ``name:`` field matches the filename stem.
5. ``LocalDispatcher._load_agent_spec`` strips the frontmatter when
   embedding the spec into the subagent prompt (so the body the
   subagent reads is just operational instructions, not a re-statement
   of its own registration metadata).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Locate project root by walking up from this test file. The test file
# lives at MCP/tests/test_runtime_agent_model_pinning.py — two parents
# from here is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# AGENT_SPEC_DIRS mirrors the registry in
# MCP/orchestrator/local_dispatcher.py. We don't import directly to keep
# this test runnable in isolation.
_AGENT_SPEC_DIRS = (
    Path("Courseforge") / "agents",
    Path("Trainforge") / "agents",
)

# Pinned membership of AGENT_SUBAGENT_SET. Mirrored here so a future
# refactor that adds a new agent to the executor's frozenset surfaces
# as a test failure rather than silently shipping a subagent that
# inherits the parent session's model.
_EXPECTED_AGENT_NAMES = frozenset({
    "course-outliner",
    "content-generator",
    "oscqr-course-evaluator",
    "quality-assurance",
    "content-analyzer",
    "accessibility-remediation",
    "content-quality-remediation",
    "intelligent-design-mapper",
    "assessment-extractor",
    "assessment-generator",
    "assessment-validator",
    "training-synthesizer",
})


def _locate_agent_spec(agent_name: str) -> Path:
    """Walk the AGENT_SPEC_DIRS in order; return the first existing path."""
    for base in _AGENT_SPEC_DIRS:
        candidate = _PROJECT_ROOT / base / f"{agent_name}.md"
        if candidate.exists():
            return candidate
    raise AssertionError(
        f"No agent spec markdown found for {agent_name!r} under any of "
        f"{[str(d) for d in _AGENT_SPEC_DIRS]}"
    )


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract a YAML-frontmatter top-block into a flat dict.

    Minimal parser — we don't pull in PyYAML because the frontmatter
    shape is fixed (``key: value`` lines, no nesting, no lists). Returns
    an empty dict if no frontmatter block is found.
    """
    if not text.startswith("---\n"):
        return {}
    rest = text[4:]
    end_idx = rest.find("\n---\n")
    if end_idx == -1:
        return {}
    block = rest[:end_idx]
    out: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


@pytest.mark.parametrize("agent_name", sorted(_EXPECTED_AGENT_NAMES))
def test_runtime_agent_has_model_sonnet_frontmatter(agent_name: str) -> None:
    """Each of the 12 runtime-dispatched agents must carry ``model: sonnet``.

    Without this, a Claude Code subagent dispatched via the Agent tool
    inherits the operator's parent session model. The project rule is
    that runtime subagents run on Sonnet, not Opus.
    """
    spec_path = _locate_agent_spec(agent_name)
    text = spec_path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), (
        f"{spec_path.relative_to(_PROJECT_ROOT)} is missing YAML "
        f"frontmatter. Runtime-dispatched agents need a frontmatter "
        f"block with ``model: sonnet`` so the spawned subagent doesn't "
        f"inherit the operator session's model."
    )
    fm = _parse_frontmatter(text)
    assert fm.get("model") == "sonnet", (
        f"{spec_path.relative_to(_PROJECT_ROOT)} frontmatter must set "
        f"``model: sonnet`` (got {fm.get('model')!r}). The 12 agents "
        f"in AGENT_SUBAGENT_SET are dispatched as runtime subagents and "
        f"the project rule pins them to Sonnet."
    )


@pytest.mark.parametrize("agent_name", sorted(_EXPECTED_AGENT_NAMES))
def test_runtime_agent_name_matches_filename(agent_name: str) -> None:
    """The frontmatter ``name`` field must match the filename stem.

    The Agent tool resolves ``subagent_type`` against the ``name``
    field. A mismatch means the Agent tool can't find the spec, falls
    back to a default agent, and silently loses the ``model: sonnet``
    pin.
    """
    spec_path = _locate_agent_spec(agent_name)
    fm = _parse_frontmatter(spec_path.read_text(encoding="utf-8"))
    assert fm.get("name") == agent_name, (
        f"{spec_path.relative_to(_PROJECT_ROOT)} frontmatter ``name`` "
        f"is {fm.get('name')!r}, expected {agent_name!r} (filename "
        f"stem). The Agent tool resolves subagent_type against this "
        f"field; a mismatch silently drops the model pin."
    )


def test_agent_subagent_set_coverage() -> None:
    """The pinned name set must equal ``executor.AGENT_SUBAGENT_SET``.

    Catches the case where a future PR adds a new agent to the
    executor's frozenset but forgets to add ``model: sonnet``
    frontmatter to the agent's spec file.
    """
    from MCP.core.executor import AGENT_SUBAGENT_SET

    assert AGENT_SUBAGENT_SET == _EXPECTED_AGENT_NAMES, (
        f"AGENT_SUBAGENT_SET has drifted from the runtime-model pin "
        f"set. Added: {sorted(AGENT_SUBAGENT_SET - _EXPECTED_AGENT_NAMES)}. "
        f"Removed: {sorted(_EXPECTED_AGENT_NAMES - AGENT_SUBAGENT_SET)}. "
        f"If a new agent was added to AGENT_SUBAGENT_SET, also add the "
        f"name to ``_EXPECTED_AGENT_NAMES`` here AND add ``model: "
        f"sonnet`` frontmatter to the agent's .md file."
    )


def test_local_dispatcher_strips_frontmatter() -> None:
    """``LocalDispatcher._load_agent_spec`` strips YAML frontmatter.

    The dispatcher embeds the agent spec body into the subagent prompt.
    Without stripping, the prompt would carry the agent's own
    registration metadata (``---\\nname: ...\\nmodel: sonnet\\n---``)
    as the first lines of the spec section — purely cosmetic noise.
    With stripping, the body the subagent reads is exactly the
    operational instructions.
    """
    from MCP.orchestrator.local_dispatcher import (
        LocalDispatcher,
        _strip_yaml_frontmatter,
    )

    # Direct helper test — confirm the canonical frontmatter shape is
    # stripped and a no-frontmatter body passes through unchanged.
    with_fm = (
        "---\n"
        "name: content-generator\n"
        "description: Test description.\n"
        "model: sonnet\n"
        "---\n"
        "\n"
        "# Content Generator\n"
        "\n"
        "Operational instructions.\n"
    )
    stripped = _strip_yaml_frontmatter(with_fm)
    assert stripped.startswith("# Content Generator\n"), (
        f"Frontmatter strip should leave the H1 as the first line; got "
        f"{stripped[:80]!r}"
    )
    assert "name: content-generator" not in stripped
    assert "model: sonnet" not in stripped

    without_fm = "# Plain Agent Spec\n\nNo frontmatter.\n"
    assert _strip_yaml_frontmatter(without_fm) == without_fm

    # Integration test — instantiate LocalDispatcher and confirm the
    # public-ish load path returns frontmatter-stripped content for one
    # of the 12 real agents.
    dispatcher = LocalDispatcher(project_root=_PROJECT_ROOT)
    spec_body = dispatcher._load_agent_spec("content-generator")
    assert "model: sonnet" not in spec_body, (
        "LocalDispatcher._load_agent_spec must strip YAML frontmatter "
        "before embedding the spec into the subagent prompt."
    )
    assert "name: content-generator" not in spec_body
    # Real body content should still be present (operational text).
    assert "Content Generator" in spec_body
