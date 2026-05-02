"""Phase 3 Subtask 59 — legacy single-pass regression.

Verifies that with ``COURSEFORGE_TWO_PASS`` unset (or explicitly
``false``), the legacy ``content_generation`` phase is the active
authoring path and the new two-pass router phases
(``content_generation_outline`` / ``inter_tier_validation`` /
``content_generation_rewrite``) skip cleanly.

Exercise path notes (mirrors Subtask 58's limitation note):

- The plan calls for running the ``course_generation`` workflow
  end-to-end via ``WorkflowRunner.run_workflow`` against a fixture and
  asserting byte-stable match against a pre-Phase-3 golden snapshot.
  The new phases are declared in ``config/workflows.yaml`` but their
  Python tool hooks (``_run_inter_tier_validation`` etc.) are not yet
  wired in ``MCP/tools/pipeline_tools.py``. With those hooks missing,
  the workflow runner cannot produce HTML content via
  ``WorkflowRunner.run_workflow`` without a heavy MCP-tool fixture
  surface this scope shouldn't introduce.
- The regression contract in scope here is the env-predicate gating
  (``enabled_when_env`` / ``depends_on_when_env``) — i.e. the
  legacy phase runs, the new phases skip, the packaging phase still
  depends on the legacy phase. This is the load-bearing decision
  point Subtask 59 is meant to pin: a regression that flipped the
  predicates would silently activate the new phases on a legacy run.

The test exercises that gate against the real ``config/workflows.yaml``
via ``OrchestratorConfig.load`` + ``WorkflowRunner._should_skip_phase``,
so a YAML drift OR a runner-logic regression both trip the same
assertion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.config import OrchestratorConfig, WorkflowPhase  # noqa: E402
from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _workflow_phases(workflow_id: str) -> Dict[str, WorkflowPhase]:
    """Load the real config/workflows.yaml and return the named workflow's
    phases keyed by phase.name."""
    config = OrchestratorConfig.load()
    wf = config.workflows[workflow_id]
    return {phase.name: phase for phase in wf.phases}


def _make_runner_stub() -> WorkflowRunner:
    """Build a minimal ``WorkflowRunner`` for the predicate / dependency
    helpers (``_should_skip_phase`` / ``_dependencies_met`` /
    ``_eval_enabled_when_env``).

    The constructor takes no required positional args we can't bypass,
    but ``_should_skip_phase`` and the env helpers are pure on phase /
    workflow_params + ``os.environ``. ``__new__`` produces an
    uninitialised instance suitable for these read-only helpers.
    """
    return WorkflowRunner.__new__(WorkflowRunner)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow_id", ["course_generation", "textbook_to_course"])
def test_legacy_content_generation_runs_when_two_pass_unset(
    monkeypatch, workflow_id
):
    """``COURSEFORGE_TWO_PASS`` unset → legacy ``content_generation``
    is enabled, new phases (``content_generation_outline`` /
    ``inter_tier_validation`` / ``content_generation_rewrite``) skip.

    Exercises both the ``course_generation`` and ``textbook_to_course``
    workflows because both carry the same env-predicate contract."""
    monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)

    runner = _make_runner_stub()
    phases = _workflow_phases(workflow_id)
    workflow_params: Dict[str, Any] = {}

    legacy = phases.get("content_generation")
    assert legacy is not None, (
        f"workflow {workflow_id!r} missing legacy content_generation phase"
    )
    # The legacy phase must NOT skip.
    assert runner._should_skip_phase(legacy, workflow_params) is False, (
        "legacy content_generation phase incorrectly skipped when "
        "COURSEFORGE_TWO_PASS is unset"
    )

    # New phases must skip cleanly.
    for new_name in (
        "content_generation_outline",
        "inter_tier_validation",
        "content_generation_rewrite",
    ):
        new_phase = phases.get(new_name)
        assert new_phase is not None, (
            f"workflow {workflow_id!r} missing two-pass phase {new_name!r}"
        )
        assert runner._should_skip_phase(new_phase, workflow_params) is True, (
            f"two-pass phase {new_name!r} did not skip when "
            f"COURSEFORGE_TWO_PASS is unset"
        )


@pytest.mark.parametrize("workflow_id", ["course_generation", "textbook_to_course"])
def test_legacy_content_generation_runs_when_two_pass_explicitly_false(
    monkeypatch, workflow_id
):
    """``COURSEFORGE_TWO_PASS=false`` → same outcome as unset (legacy
    enabled, new phases skip). Verifies the predicate evaluator
    handles the explicit-false case (the env predicate is
    ``COURSEFORGE_TWO_PASS!=true`` for the legacy phase, so a literal
    ``false`` value satisfies the predicate)."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "false")

    runner = _make_runner_stub()
    phases = _workflow_phases(workflow_id)

    legacy = phases.get("content_generation")
    assert legacy is not None
    assert runner._should_skip_phase(legacy, {}) is False

    for new_name in (
        "content_generation_outline",
        "inter_tier_validation",
        "content_generation_rewrite",
    ):
        new_phase = phases.get(new_name)
        assert new_phase is not None
        assert runner._should_skip_phase(new_phase, {}) is True, (
            f"phase {new_name!r} did not skip under "
            f"COURSEFORGE_TWO_PASS=false"
        )


def test_legacy_packaging_depends_on_content_generation_under_legacy_path(
    monkeypatch,
):
    """Legacy single-pass: ``packaging``'s ``depends_on`` resolves to
    ``content_generation`` because the env-predicate
    (``COURSEFORGE_TWO_PASS=true``) on
    ``depends_on_when_env_value`` is unsatisfied. Locks in the
    pre-Phase-3 dependency edge so a YAML regression that flipped the
    edges would trip this test.

    The dual-edge contract is: the runner falls through to
    ``phase.depends_on`` (legacy) unless the env predicate matches in
    which case it uses ``depends_on_when_env_value`` (the new
    rewrite-tier edge)."""
    monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)

    runner = _make_runner_stub()
    phases = _workflow_phases("course_generation")
    packaging = phases["packaging"]

    # Active depends_on under the legacy path.
    alt_pred = getattr(packaging, "depends_on_when_env", None)
    alt_value = getattr(packaging, "depends_on_when_env_value", None)
    if alt_pred and alt_value and runner._eval_enabled_when_env(alt_pred):
        active_deps = list(alt_value)
    else:
        active_deps = list(packaging.depends_on or [])
    assert "content_generation" in active_deps, (
        "packaging.depends_on did not resolve to content_generation under "
        f"the legacy path; got {active_deps!r}"
    )
    # Negative: the rewrite-tier edge MUST NOT activate under the
    # legacy path.
    assert "content_generation_rewrite" not in active_deps


def test_two_pass_predicate_flip_swaps_active_phase_set(monkeypatch):
    """Sanity check the inverse: ``COURSEFORGE_TWO_PASS=true`` enables
    the new phases AND activates the rewrite-tier dependency edge for
    packaging. Pinning this here makes any future change to the
    predicate evaluator fail one of the two parametric sides explicitly
    rather than silently regressing only one path."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")

    runner = _make_runner_stub()
    phases = _workflow_phases("course_generation")

    # Legacy phase skipped; new phases active.
    assert runner._should_skip_phase(phases["content_generation"], {}) is True
    assert runner._should_skip_phase(
        phases["content_generation_outline"], {}
    ) is False
    assert runner._should_skip_phase(
        phases["inter_tier_validation"], {}
    ) is False
    assert runner._should_skip_phase(
        phases["content_generation_rewrite"], {}
    ) is False

    packaging = phases["packaging"]
    alt_pred = getattr(packaging, "depends_on_when_env", None)
    alt_value = getattr(packaging, "depends_on_when_env_value", None)
    assert alt_pred and alt_value
    assert runner._eval_enabled_when_env(alt_pred) is True
    active_deps = list(alt_value)
    # Phase 3.5 Subtask 11: packaging now waits on post_rewrite_validation
    # (which runs after content_generation_rewrite) rather than directly
    # on content_generation_rewrite.
    assert "post_rewrite_validation" in active_deps
    # And the legacy edge does NOT activate when the predicate matches.
    assert "content_generation" not in active_deps
