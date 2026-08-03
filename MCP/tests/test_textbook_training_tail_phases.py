"""In-build training tail — training -> post_training_validation -> evaluation.

Owner decision (2026-07-25): build->training is ONE sequenced pipeline, so
``finalization`` must be genuinely LAST. These three phases were previously
only reachable through the SEPARATE ``trainforge_train`` workflow run after
``textbook_to_course`` had already finalized.

Locks:

 1. YAML topology — all three declared between ``vector_indexing`` and
    ``finalization``; ``finalization.depends_on == ["evaluation"]``; the
    topological sort preserves the ordering; config loads (meta-schema +
    inputs_from reference validation) and still validates against
    ``schemas/config/workflows_meta.schema.json``.
 2. Gate parity — ``post_training_validation`` carries the two
    ``trainforge_train`` gates VERBATIM (eval_gating + family_completeness,
    both critical / block / fail_closed).
 3. Opt-in skip semantics — skipped by default, skipped under
    ``--skip-training``, run under ``--with-training``, and ``--skip-training``
    WINS when both are set.
 4. The regression that would break every default build — with the three
    phases skipped, ``finalization``'s dependencies are still met (a skipped
    phase records ``_completed: True``).
 5. The standalone ``trainforge_train`` workflow still parses, still has its
    two phases, and its NON-optional same-named phases are NOT swept up by the
    ``--with-training`` gating.
 6. CLI plumbing — ``--with-training`` reaches ``params["with_training"]`` and
    ``--skip-training`` suppresses it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.config import WorkflowPhase  # noqa: E402
from MCP.core.workflow_runner import (  # noqa: E402
    WorkflowRunner,
    _get_phase_param_routing,
    _load_workflows_config,
)

_TAIL_PHASES = ("training", "post_training_validation", "evaluation")


def _make_runner() -> WorkflowRunner:
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _phases(workflow: str = "textbook_to_course") -> List[Dict[str, Any]]:
    cfg = _load_workflows_config(force_reload=True)
    return cfg["workflows"][workflow]["phases"]


def _phase(name: str, workflow: str = "textbook_to_course") -> Dict[str, Any]:
    for p in _phases(workflow):
        if p["name"] == name:
            return p
    raise AssertionError(f"phase {name!r} not declared in {workflow}")


def _optional_phase(name: str) -> WorkflowPhase:
    """A WorkflowPhase standing in for the OPTIONAL textbook_to_course phase."""
    return WorkflowPhase(name=name, agents=[], optional=True)


# ---------------------------------------------------------------------------
# 1. YAML topology
# ---------------------------------------------------------------------------


def test_tail_phases_declared_between_vector_indexing_and_finalization():
    names = [p["name"] for p in _phases()]
    for phase_name in _TAIL_PHASES:
        assert phase_name in names, (
            f"{phase_name} must be declared in textbook_to_course."
        )
        assert names.index("vector_indexing") < names.index(phase_name), (
            f"{phase_name} must be declared AFTER vector_indexing."
        )
        assert names.index(phase_name) < names.index("finalization"), (
            f"{phase_name} must be declared BEFORE finalization — "
            f"finalization is genuinely LAST."
        )
    assert (
        names.index("training")
        < names.index("post_training_validation")
        < names.index("evaluation")
    ), "The training tail must be ordered training -> validation -> evaluation."
    assert names[-1] == "finalization", (
        "finalization must be the LAST declared phase of textbook_to_course."
    )


def test_tail_phase_dependency_chain():
    assert _phase("training").get("depends_on") == ["vector_indexing"]
    assert _phase("post_training_validation").get("depends_on") == ["training"]
    assert _phase("evaluation").get("depends_on") == ["post_training_validation"]


def test_finalization_depends_on_evaluation():
    assert _phase("finalization").get("depends_on") == ["evaluation"], (
        "finalization must depend on evaluation (the tail of the training "
        "chain) so it is genuinely last. A skipped optional phase still sets "
        "_completed=True, so the dependency holds on a default run."
    )


@pytest.mark.parametrize("phase_name", _TAIL_PHASES)
def test_tail_phases_are_optional_and_serial(phase_name: str):
    phase = _phase(phase_name)
    assert phase.get("optional") is True, (
        f"{phase_name} must be optional — the in-build training tail is "
        f"opt-in and must never block a default build."
    )
    assert phase.get("max_concurrent") == 1
    assert phase.get("parallel") is False


def test_training_phase_has_generous_timeout():
    # Training is multi-hour; mirrors the standalone trainforge_train ceiling.
    assert _phase("training").get("timeout_minutes") == 720


@pytest.mark.parametrize("phase_name", _TAIL_PHASES)
def test_tail_phases_declare_no_agents(phase_name: str):
    # All three route by phase NAME through _PHASE_TOOL_MAPPING; ``agents: []``
    # is what makes _create_phase_tasks synthesize the virtual phase-handler
    # task (the inter_tier_validation / assessment_synthesis convention).
    assert _phase(phase_name).get("agents") == []


def test_config_loads_without_raising():
    # force_reload re-runs the meta-schema validation +
    # _validate_inputs_from_references; a bad phase shape or dangling
    # phase_outputs reference would raise here.
    cfg = _load_workflows_config(force_reload=True)
    assert "textbook_to_course" in cfg["workflows"]


def test_workflows_yaml_validates_against_meta_schema():
    jsonschema = pytest.importorskip("jsonschema")
    import yaml

    cfg = yaml.safe_load(
        (PROJECT_ROOT / "config" / "workflows.yaml").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            PROJECT_ROOT / "schemas" / "config" / "workflows_meta.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(cfg, schema)


def test_topological_sort_preserves_tail_ordering():
    phases = [
        WorkflowPhase(
            name=p["name"],
            agents=list(p.get("agents", [])),
            depends_on=list(p.get("depends_on", [])),
        )
        for p in _phases()
    ]
    runner = object.__new__(WorkflowRunner)
    order = [p.name for p in runner._topological_sort(phases)]
    assert (
        order.index("vector_indexing")
        < order.index("training")
        < order.index("post_training_validation")
        < order.index("evaluation")
        < order.index("finalization")
    )


def test_training_inputs_from_routing():
    routing = _get_phase_param_routing("training")
    assert routing.get("course_code") == ("workflow_params", "course_name")
    assert routing.get("base_model") == ("workflow_params", "base_model")


def test_evaluation_receives_the_training_run_dir():
    """The adapter dir must reach the eval arm + EvalGatingValidator."""
    routing = _get_phase_param_routing("evaluation")
    assert routing.get("model_dir") == ("phase_outputs", "training", "run_dir")
    assert routing.get("course_code") == ("workflow_params", "course_name")
    assert "run_dir" in (_phase("training").get("outputs") or [])


def test_tail_inputs_from_params_are_accepted_by_their_handlers():
    """Every routed param must exist in the tool schema — the mapper silently
    DROPS unknown keys, so a typo here is a dead route, not an error."""
    from MCP.core.tool_schemas import TOOL_SCHEMAS

    for phase_name, tool in (("training", "run_training"),
                             ("evaluation", "run_evaluation")):
        schema = TOOL_SCHEMAS[tool]
        accepted = (
            set(schema.get("required", []))
            | set(schema.get("optional", []))
            | set(schema.get("param_mapping", {}))
        )
        routed = set(_get_phase_param_routing(phase_name))
        assert routed <= accepted, (
            f"{phase_name} routes params {sorted(routed - accepted)} that "
            f"{tool} does not accept."
        )


# ---------------------------------------------------------------------------
# 2. Gate parity with trainforge_train::post_training_validation
# ---------------------------------------------------------------------------


def test_post_training_validation_gates_match_trainforge_train_verbatim():
    def _gates(wf: str) -> Dict[str, Dict[str, Any]]:
        return {
            g["gate_id"]: g
            for g in _phase("post_training_validation", wf).get(
                "validation_gates", []
            )
        }

    textbook = _gates("textbook_to_course")
    standalone = _gates("trainforge_train")

    assert set(textbook) == {"eval_gating", "family_completeness"}
    assert set(standalone) == {"eval_gating", "family_completeness"}
    for gate_id in ("eval_gating", "family_completeness"):
        assert textbook[gate_id] == standalone[gate_id], (
            f"{gate_id} must be carried VERBATIM from "
            f"trainforge_train::post_training_validation."
        )
        assert textbook[gate_id]["severity"] == "critical"
        assert textbook[gate_id]["behavior"] == {
            "on_fail": "block", "on_error": "fail_closed",
        }


# ---------------------------------------------------------------------------
# 3. Opt-in skip semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase_name", _TAIL_PHASES)
def test_skipped_by_default(phase_name: str):
    runner = _make_runner()
    assert runner._should_skip_phase(_optional_phase(phase_name), {}) is True, (
        f"{phase_name} must skip on a default run (opt-in only)."
    )


@pytest.mark.parametrize("phase_name", _TAIL_PHASES)
def test_skipped_under_skip_training(phase_name: str):
    runner = _make_runner()
    params = {"skip_training": True}
    assert runner._should_skip_phase(_optional_phase(phase_name), params) is True


@pytest.mark.parametrize("phase_name", _TAIL_PHASES)
def test_runs_under_with_training(phase_name: str):
    runner = _make_runner()
    params = {"with_training": True}
    assert runner._should_skip_phase(_optional_phase(phase_name), params) is False


@pytest.mark.parametrize("phase_name", _TAIL_PHASES)
def test_skip_training_wins_over_with_training(phase_name: str):
    """Contradictory flags resolve to the safe/cheap side: skip."""
    runner = _make_runner()
    params = {"with_training": True, "skip_training": True}
    assert runner._should_skip_phase(_optional_phase(phase_name), params) is True


def test_with_training_does_not_resurrect_training_synthesis():
    """--with-training governs the tail only; training_synthesis keeps its own
    --skip-training contract."""
    runner = _make_runner()
    phase = WorkflowPhase(name="training_synthesis", agents=[], optional=True)
    assert runner._should_skip_phase(
        phase, {"with_training": True, "skip_training": True}
    ) is True
    assert runner._should_skip_phase(phase, {"with_training": True}) is False


@pytest.mark.parametrize(
    "stage",
    ["courseforge", "courseforge-outline", "courseforge-validate",
     "courseforge-rewrite"],
)
@pytest.mark.parametrize("phase_name", _TAIL_PHASES)
def test_courseforge_stage_subcommands_skip_tail(stage: str, phase_name: str):
    runner = _make_runner()
    assert runner._should_skip_for_courseforge_stage(phase_name, stage), (
        f"{phase_name} is outside the Courseforge two-pass surface and must "
        f"be skipped by the {stage!r} stage subcommand."
    )


def test_with_training_phase_set_is_the_single_source_of_truth():
    assert WorkflowRunner._WITH_TRAINING_PHASES == frozenset(_TAIL_PHASES)


# ---------------------------------------------------------------------------
# 4. Default-build regression: finalization dependencies still met
# ---------------------------------------------------------------------------


def test_finalization_dependencies_met_when_tail_skipped():
    """THE regression this change could introduce: every default build.

    Skipped phases record ``{"_skipped": True, "_completed": True}`` and
    ``_dependencies_met`` only checks ``_completed``, so finalization's
    dependency on ``evaluation`` is satisfied even though nothing ran.
    """
    runner = _make_runner()
    phase_outputs: Dict[str, Dict[str, Any]] = {
        "vector_indexing": {"_completed": True},
    }
    for phase_name in _TAIL_PHASES:
        assert runner._should_skip_phase(_optional_phase(phase_name), {}) is True
        # Exactly what run_workflow writes on the skip branch.
        phase_outputs[phase_name] = {"_skipped": True, "_completed": True}

    finalization = WorkflowPhase(
        name="finalization",
        agents=["brightspace-packager"],
        depends_on=list(_phase("finalization")["depends_on"]),
    )
    assert runner._dependencies_met(finalization, phase_outputs) is True, (
        "finalization must still run on a default (no --with-training) build."
    )


def test_finalization_blocked_until_evaluation_completes():
    """The other half of the contract: an absent evaluation output blocks."""
    runner = _make_runner()
    finalization = WorkflowPhase(
        name="finalization",
        agents=["brightspace-packager"],
        depends_on=list(_phase("finalization")["depends_on"]),
    )
    assert runner._dependencies_met(finalization, {}) is False


# ---------------------------------------------------------------------------
# 5. Standalone trainforge_train left intact
# ---------------------------------------------------------------------------


def test_trainforge_train_still_parses_with_two_phases():
    cfg = _load_workflows_config(force_reload=True)
    assert "trainforge_train" in cfg["workflows"]
    names = [p["name"] for p in _phases("trainforge_train")]
    assert names == ["training", "post_training_validation"]


@pytest.mark.parametrize("phase_name", ["training", "post_training_validation"])
def test_trainforge_train_phases_are_not_optional(phase_name: str):
    """This is what protects the standalone workflow from the --with-training
    gate: ``_should_skip_phase`` returns early for non-optional phases."""
    phase = _phase(phase_name, "trainforge_train")
    assert phase.get("optional", False) is False

    runner = _make_runner()
    live = WorkflowPhase(
        name=phase_name,
        agents=list(phase.get("agents", [])),
        optional=False,
    )
    # No with_training param anywhere — the standalone workflow still runs.
    assert runner._should_skip_phase(live, {}) is False
    assert runner._should_skip_phase(live, {"skip_training": True}) is False


# ---------------------------------------------------------------------------
# 6. CLI flag plumbing
# ---------------------------------------------------------------------------


def _build_params(**kwargs: Any) -> Dict[str, Any]:
    from cli.commands.run import _build_workflow_params

    base = dict(
        workflow="textbook_to_course",
        corpus="corpus.pdf",
        course_name="FXTERTIARY_101",
        weeks=None,
        no_assessments=False,
        assessment_count=50,
        bloom_levels="remember,understand,apply,analyze",
        priority="normal",
        objectives_path=None,
    )
    base.update(kwargs)
    return _build_workflow_params(**base)


def test_with_training_flag_absent_by_default():
    assert "with_training" not in _build_params()


def test_with_training_flag_sets_param():
    assert _build_params(with_training=True)["with_training"] is True


def test_skip_training_suppresses_with_training_param():
    params = _build_params(with_training=True, skip_training=True)
    assert params.get("skip_training") is True
    assert "with_training" not in params, (
        "--skip-training wins: the opt-in must not even be recorded."
    )


def test_run_command_exposes_with_training_option():
    from cli.commands.run import run_command

    names = {p.name for p in run_command.params}
    assert "with_training" in names
    opts = {o for p in run_command.params for o in p.opts}
    assert "--with-training" in opts


def test_dry_run_plan_prunes_tail_unless_opted_in():
    from cli.commands.run import _dry_run_plan

    plan = _dry_run_plan(
        "textbook_to_course", _build_params(), mode="local", provider="local",
    )
    names = [p["name"] for p in plan["phases"]]
    assert names, plan
    for phase_name in _TAIL_PHASES:
        assert phase_name not in names

    opted = _dry_run_plan(
        "textbook_to_course",
        _build_params(with_training=True),
        mode="local",
        provider="local",
    )
    opted_names = [p["name"] for p in opted["phases"]]
    for phase_name in _TAIL_PHASES:
        assert phase_name in opted_names
    assert opted_names[-1] == "finalization"


def test_persist_workflow_param_writes_the_opt_in(tmp_path, monkeypatch):
    """--with-training is not a create_textbook_pipeline kwarg, so the opt-in
    is patched onto the freshly-written workflow state."""
    import lib.paths as paths
    from cli.commands import run as run_mod

    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "WF-TEST.json").write_text(
        json.dumps({"workflow_id": "WF-TEST", "params": {"course_name": "T"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "STATE_PATH", tmp_path)

    assert run_mod._persist_workflow_param("WF-TEST", "with_training", True)
    state = json.loads((wf_dir / "WF-TEST.json").read_text(encoding="utf-8"))
    assert state["params"]["with_training"] is True
    assert state["params"]["course_name"] == "T", "existing params preserved"

    # Best-effort by contract: unknown / missing ids never raise.
    assert run_mod._persist_workflow_param(None, "with_training", True) is False
    assert run_mod._persist_workflow_param("WF-NOPE", "x", 1) is False


# ---------------------------------------------------------------------------
# 7. --with-training on the --resume path
#
# ``_should_skip_phase`` reads the opt-in out of the PERSISTED params, so
# ``--with-training`` on a ``--resume`` must patch the state file or it is a
# silent no-op (the whole tail is skipped with no error). Mirrors the
# ``--stop-after`` / ``--reuse-objectives`` resume-override precedent.
# ---------------------------------------------------------------------------


def _write_resume_state(
    tmp_path: Path, monkeypatch: Any, params: Dict[str, Any],
    workflow_id: str = "WF-RESUME-TRAINING",
) -> str:
    import lib.paths as paths

    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps({"workflow_id": workflow_id, "params": dict(params)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "STATE_PATH", tmp_path)
    return workflow_id


def _read_resume_params(
    tmp_path: Path, workflow_id: str = "WF-RESUME-TRAINING",
) -> Dict[str, Any]:
    state = json.loads(
        (tmp_path / "workflows" / f"{workflow_id}.json").read_text(encoding="utf-8")
    )
    return state["params"]


def test_resume_with_training_patches_persisted_params(tmp_path, monkeypatch):
    """The gap: --with-training on --resume must reach the persisted params."""
    from cli.commands import run as run_mod

    wid = _write_resume_state(tmp_path, monkeypatch, {"course_name": "T"})
    assert run_mod._apply_resume_with_training_override(wid, True) is True

    params = _read_resume_params(tmp_path)
    assert params["with_training"] is True
    assert params["course_name"] == "T", "existing params preserved"

    # The runner honours exactly this param, so the tail now runs.
    runner = _make_runner()
    for phase_name in _TAIL_PHASES:
        assert (
            runner._should_skip_phase(_optional_phase(phase_name), params) is False
        )


def test_resume_skip_training_wins_over_with_training(tmp_path, monkeypatch):
    """Creation-path precedence mirrored: --skip-training wins, so the opt-in
    is not even recorded."""
    from cli.commands import run as run_mod

    wid = _write_resume_state(tmp_path, monkeypatch, {"course_name": "T"})
    assert (
        run_mod._apply_resume_with_training_override(wid, True, True) is None
    )

    params = _read_resume_params(tmp_path)
    assert "with_training" not in params
    runner = _make_runner()
    for phase_name in _TAIL_PHASES:
        assert runner._should_skip_phase(_optional_phase(phase_name), params) is True


def test_resume_without_flag_leaves_persisted_opt_in_intact(tmp_path, monkeypatch):
    """Stickiness: a flag ABSENT on resume never clears what creation
    persisted (the --stop-after / --reuse-objectives precedent)."""
    from cli.commands import run as run_mod

    wid = _write_resume_state(
        tmp_path, monkeypatch, {"course_name": "T", "with_training": True},
    )
    assert run_mod._apply_resume_with_training_override(wid, False) is None

    params = _read_resume_params(tmp_path)
    assert params["with_training"] is True, "persisted opt-in must survive"
    assert params["course_name"] == "T"


def test_resume_without_flag_does_not_invent_the_opt_in(tmp_path, monkeypatch):
    """The complement: no flag + nothing persisted -> still opted out."""
    from cli.commands import run as run_mod

    wid = _write_resume_state(tmp_path, monkeypatch, {"course_name": "T"})
    assert run_mod._apply_resume_with_training_override(wid, False) is None
    assert "with_training" not in _read_resume_params(tmp_path)


def test_resume_with_training_unwritable_state_file_does_not_raise(
    tmp_path, monkeypatch, capsys,
):
    """Best-effort contract: an unwritable state file never crashes the
    resume — but an explicit flag that could not be applied WARNS."""
    from cli.commands import run as run_mod

    wid = _write_resume_state(tmp_path, monkeypatch, {"course_name": "T"})

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", _boom)
    assert run_mod._apply_resume_with_training_override(wid, True) is None

    err = capsys.readouterr().err
    assert "--with-training" in err and "WARNING" in err


def test_resume_with_training_missing_state_file_warns(tmp_path, monkeypatch, capsys):
    """A missing / unreadable state file is likewise non-fatal + loud."""
    from cli.commands import run as run_mod
    import lib.paths as paths

    monkeypatch.setattr(paths, "STATE_PATH", tmp_path)
    assert run_mod._apply_resume_with_training_override("WF-NOPE", True) is None
    assert "WARNING" in capsys.readouterr().err


def test_resume_workflow_threads_the_flags(tmp_path, monkeypatch):
    """``_resume_workflow`` accepts + forwards both flags to the override."""
    import inspect

    from cli.commands import run as run_mod

    sig = inspect.signature(run_mod._resume_workflow)
    assert "with_training" in sig.parameters
    assert "skip_training" in sig.parameters
    assert sig.parameters["with_training"].default is False
    assert sig.parameters["skip_training"].default is False

    seen: List[Any] = []
    monkeypatch.setattr(
        run_mod,
        "_apply_resume_with_training_override",
        lambda *a, **kw: seen.append((a, kw)),
    )
    monkeypatch.setattr(
        run_mod, "_apply_resume_stop_after_override", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        run_mod, "_apply_resume_reuse_objectives_override", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        run_mod, "_resolve_run_id_for_workflow", lambda wid: wid,
    )

    class _Result:
        status = "ok"
        error = None

        def to_dict(self) -> Dict[str, Any]:
            return {"status": "ok"}

    class _Orch:
        async def run(self, _wid: str) -> Any:
            return _Result()

    monkeypatch.setattr(run_mod, "_build_orchestrator", lambda *a, **kw: _Orch())
    monkeypatch.setattr(run_mod, "_paused_exit_code", lambda *a, **kw: None)
    monkeypatch.setattr(run_mod, "_any_gate_failed", lambda _r: False)

    run_mod._resume_workflow(
        workflow_id="WF-X",
        mode="local",
        provider="local",
        model=None,
        output_json=True,
        watch=False,
        with_training=True,
        skip_training=False,
    )
    assert seen == [(("WF-X", True, False), {})]


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--resume", "WF-X", "--with-training"], (True, False)),
        (["--resume", "WF-X", "--with-training", "--skip-training"], (True, True)),
        (["--resume", "WF-X"], (False, False)),
    ],
)
def test_run_command_forwards_training_flags_on_resume(monkeypatch, argv, expected):
    """End of the wire: ``ed4all run --resume ... --with-training`` reaches
    ``_resume_workflow`` (and carries --skip-training for the precedence)."""
    from click.testing import CliRunner

    from cli.commands import run as run_mod

    seen: Dict[str, Any] = {}
    monkeypatch.setattr(
        run_mod, "_resume_workflow", lambda **kw: seen.update(kw),
    )

    result = CliRunner().invoke(run_mod.run_command, ["textbook-to-course", *argv])
    assert result.exit_code == 0, result.output
    assert (seen.get("with_training"), seen.get("skip_training")) == expected


def test_dry_run_plan_keeps_trainforge_train_phases():
    """The tail names are shared with the standalone workflow — pruning must
    not reach into it (its phases are not optional)."""
    from cli.commands.run import _dry_run_plan

    plan = _dry_run_plan(
        "trainforge_train",
        {"course_name": "FXTERTIARY_101"},
        mode="local",
        provider="local",
    )
    names = [p["name"] for p in plan["phases"]]
    assert names == ["training", "post_training_validation"], plan
