"""Marketable-v1 A2 — vector_indexing phase in textbook_to_course.

Locks the wiring that makes a freshly-built course askable (retrieval-ready)
the moment the run completes:

 1. YAML topology — vector_indexing is declared AFTER libv2_archival and
    BEFORE finalization; finalization depends on vector_indexing; the
    WorkflowRunner topological sort preserves the ordering; config loads
    without raising (meta-schema + inputs_from reference validation).
 2. Agent routing — the phase routes via the ``rag-indexer`` agent, which
    AGENT_TOOL_MAPPING resolves to ``run_vector_indexing`` (the same handler
    the rag_training ``indexing`` phase uses); no _PHASE_TOOL_MAPPING entry is
    needed (it routes by agent like an ordinary phase).
 3. inputs_from — course_name is sourced from workflow_params; course_slug is
    threaded from the libv2_archival phase output.
 4. courseforge-* stage subcommands skip it — as a post-Courseforge phase it
    falls outside every stage whitelist.
 5. Optionality / graceful-degrade — the phase is optional and skips cleanly
    when the embedding stack is unavailable UNLESS TRAINFORGE_REQUIRE_EMBEDDINGS
    is set, mirroring the embedding-stack is_strict_mode() convention.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.config import WorkflowPhase  # noqa: E402
from MCP.core.executor import (  # noqa: E402
    AGENT_TOOL_MAPPING,
    _PHASE_TOOL_MAPPING,
)
from MCP.core.workflow_runner import (  # noqa: E402
    WorkflowRunner,
    _get_phase_param_routing,
    _load_workflows_config,
)


_PHASE_NAME = "vector_indexing"


def _make_runner() -> WorkflowRunner:
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _textbook_phases() -> List[Dict[str, Any]]:
    cfg = _load_workflows_config(force_reload=True)
    return cfg["workflows"]["textbook_to_course"]["phases"]


def _phase(name: str) -> Dict[str, Any]:
    for p in _textbook_phases():
        if p["name"] == name:
            return p
    raise AssertionError(f"phase {name!r} not declared in textbook_to_course")


# ---------------------------------------------------------------------------
# 1. YAML topology
# ---------------------------------------------------------------------------


def test_phase_declared_after_libv2_archival_before_finalization():
    names = [p["name"] for p in _textbook_phases()]
    assert _PHASE_NAME in names, (
        "vector_indexing must be declared in the textbook_to_course workflow."
    )
    assert names.index("libv2_archival") < names.index(_PHASE_NAME), (
        "vector_indexing must run AFTER libv2_archival (the handler resolves "
        "the chunkset from the LibV2-archived course dir)."
    )
    assert names.index(_PHASE_NAME) < names.index("finalization"), (
        "vector_indexing must run BEFORE finalization."
    )


def test_phase_depends_on_libv2_archival():
    assert _phase(_PHASE_NAME).get("depends_on") == ["libv2_archival"]


def test_finalization_depends_on_vector_indexing():
    assert _phase("finalization").get("depends_on") == [_PHASE_NAME], (
        "finalization must depend on vector_indexing so the index build is "
        "ordered before final export (a skipped optional phase still sets "
        "_completed=True, so the dependency is satisfied either way)."
    )


def test_config_loads_without_raising():
    # force_reload re-runs the meta-schema validation +
    # _validate_inputs_from_references; a bad phase shape or dangling
    # phase_outputs reference would raise here.
    cfg = _load_workflows_config(force_reload=True)
    assert "textbook_to_course" in cfg["workflows"]


def test_phase_is_optional_with_concurrency_one():
    phase = _phase(_PHASE_NAME)
    assert phase.get("optional") is True
    assert phase.get("max_concurrent") == 1
    assert phase.get("parallel") is False


def test_topological_sort_preserves_ordering():
    phases_raw = _textbook_phases()
    phases = [
        WorkflowPhase(
            name=p["name"],
            agents=list(p.get("agents", [])),
            depends_on=list(p.get("depends_on", [])),
        )
        for p in phases_raw
    ]
    runner = object.__new__(WorkflowRunner)
    sorted_phases = runner._topological_sort(phases)
    order = [p.name for p in sorted_phases]
    assert order.index("libv2_archival") < order.index(_PHASE_NAME)
    assert order.index(_PHASE_NAME) < order.index("finalization")


# ---------------------------------------------------------------------------
# 2. Agent routing
# ---------------------------------------------------------------------------


def test_phase_routes_via_rag_indexer_to_run_vector_indexing():
    phase = _phase(_PHASE_NAME)
    assert phase.get("agents") == ["rag-indexer"], (
        "vector_indexing must dispatch via the rag-indexer agent."
    )
    assert AGENT_TOOL_MAPPING["rag-indexer"] == "run_vector_indexing", (
        "rag-indexer must resolve to the real run_vector_indexing handler."
    )


def test_phase_not_in_phase_tool_mapping():
    # vector_indexing routes by agent like an ordinary phase — it must NOT be
    # in the validator-style _PHASE_TOOL_MAPPING override.
    assert _PHASE_NAME not in _PHASE_TOOL_MAPPING


def test_handler_registered_in_tool_registry():
    from MCP.tools.pipeline_tools import _build_tool_registry

    registry = _build_tool_registry()
    assert "run_vector_indexing" in registry


# ---------------------------------------------------------------------------
# 3. inputs_from routing
# ---------------------------------------------------------------------------


def test_inputs_from_course_name_and_archived_slug():
    routing = _get_phase_param_routing(_PHASE_NAME)
    assert routing.get("course_name") == (
        "workflow_params", "course_name",
    ), "course_name must be sourced from workflow_params."
    assert routing.get("course_slug") == (
        "phase_outputs", "libv2_archival", "course_slug",
    ), "course_slug must be threaded from the libv2_archival output."


# ---------------------------------------------------------------------------
# 4. courseforge-* stage subcommands skip it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage",
    ["courseforge", "courseforge-outline", "courseforge-validate",
     "courseforge-rewrite"],
)
def test_courseforge_stage_subcommands_skip_vector_indexing(stage: str):
    runner = _make_runner()
    assert runner._should_skip_for_courseforge_stage(_PHASE_NAME, stage), (
        f"vector_indexing is a post-Courseforge phase and must be skipped by "
        f"the {stage!r} stage subcommand."
    )


# ---------------------------------------------------------------------------
# 5. Optionality / embedding-deps graceful-degrade
# ---------------------------------------------------------------------------


def test_skips_when_embedding_deps_unavailable_and_not_strict(
    monkeypatch: pytest.MonkeyPatch,
):
    """Slim install (no [embedding] extra) + strict OFF => skip cleanly."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    # Force the dep-probe to report the embedding modules as absent. The
    # method does a local ``import importlib.util`` so we patch the real
    # ``importlib.util.find_spec`` (resolved at call time).
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("lib.embedding.providers", "LibV2.tools.libv2.vector_index"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    phase = WorkflowPhase(name=_PHASE_NAME, agents=["rag-indexer"], optional=True)
    runner = _make_runner()
    assert runner._should_skip_phase(phase, {}) is True


def test_does_not_skip_when_strict_mode_set(
    monkeypatch: pytest.MonkeyPatch,
):
    """Strict mode ON => never skip; the phase runs and fails closed if the
    backend is broken (handler-level contract)."""
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")
    # Even if the dep-probe would report missing, strict mode short-circuits
    # before the probe runs — so we patch find_spec to always report absent
    # and still expect "do not skip".
    import importlib.util

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None
    )

    phase = WorkflowPhase(name=_PHASE_NAME, agents=["rag-indexer"], optional=True)
    runner = _make_runner()
    assert runner._should_skip_phase(phase, {}) is False


def test_does_not_skip_when_embedding_deps_present(
    monkeypatch: pytest.MonkeyPatch,
):
    """Full install (deps importable) + strict OFF => run (no skip)."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    phase = WorkflowPhase(name=_PHASE_NAME, agents=["rag-indexer"], optional=True)
    runner = _make_runner()
    # In this repo the embedding/vector-index modules are importable, so the
    # probe finds the specs and the phase is NOT skipped.
    assert runner._should_skip_phase(phase, {}) is False
