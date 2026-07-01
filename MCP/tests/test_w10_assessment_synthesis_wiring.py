"""W10 — workflow + gate wiring regression for the assessment_synthesis phase.

Covers the Phase-5 wiring slice of
``plans/finegrain/w10-assessments-qti-discussions-2026-06.md``:

* the new ``assessment_synthesis`` phase exists in BOTH ``textbook_to_course``
  and ``course_generation``, sits BEFORE ``packaging``, and is declared
  ``agents: []`` (the validator-only-phase / phase-name-dispatch pattern);
* the phase wires exactly the five contracted gates
  (``qti_well_formed`` critical, ``assessment_objective_alignment`` critical,
  ``discussion_assignment_grounded`` warning, ``cumulative_assessment`` warning,
  ``synthesized_quiz_distractor`` warning);
* ``packaging`` depends on ``assessment_synthesis`` (both the default and the
  two-pass ``depends_on_when_env_value`` dep list);
* ``MCP/core/executor.py::_PHASE_TOOL_MAPPING['assessment_synthesis']`` routes
  to ``run_assessment_synthesis``;
* the meta-schema still accepts the whole ``config/workflows.yaml``;
* the gate-input router resolves ``qti_dir`` to ``<export>/06_assessments`` for
  the ``QtiWellFormedValidator``.

No course slugs / device paths are hardcoded; the export root is a tmp_path.
"""

import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_YAML = PROJECT_ROOT / "config" / "workflows.yaml"
META_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "config" / "workflows_meta.schema.json"
)

PHASE_NAME = "assessment_synthesis"
EXPECTED_GATES = {
    "qti_well_formed": (
        "critical",
        "lib.validators.qti_well_formed.QtiWellFormedValidator",
    ),
    "assessment_objective_alignment": (
        "critical",
        "lib.validators.assessment_objective_alignment."
        "AssessmentObjectiveAlignmentValidator",
    ),
    # C3-2: repointed from the AssessmentObjectiveAlignmentValidator stand-in
    # to the dedicated per-item-type grounding validator. Stays warning.
    "discussion_assignment_grounded": (
        "warning",
        "lib.validators.discussion_assignment_grounding."
        "DiscussionAssignmentGroundingValidator",
    ),
    # cycle-2 Batch A (FR-COURSE-03): cumulative B14 retrieval-breadth gate,
    # warning-day-1, added to this phase at commit 808dd3c.
    "cumulative_assessment": (
        "warning",
        "lib.validators.cumulative_assessment.CumulativeAssessmentValidator",
    ),
    # W7.3: synthesized-quiz distractor-quality gate — audits the QTI item
    # distractor surface the authored-block distractor_* gates never reach.
    # Warning-day-1; reuses the qti_well_formed input builder.
    "synthesized_quiz_distractor": (
        "warning",
        "lib.validators.synthesized_quiz_distractor."
        "SynthesizedQuizDistractorValidator",
    ),
}


@pytest.fixture(scope="module")
def workflows_data():
    return yaml.safe_load(WORKFLOWS_YAML.read_text(encoding="utf-8"))


def _phases(workflows_data, wf_name):
    return workflows_data["workflows"][wf_name]["phases"]


def _phase(workflows_data, wf_name, phase_name):
    for p in _phases(workflows_data, wf_name):
        if p["name"] == phase_name:
            return p
    return None


@pytest.mark.parametrize("wf_name", ["textbook_to_course", "course_generation"])
def test_assessment_synthesis_phase_exists_before_packaging(
    workflows_data, wf_name
):
    names = [p["name"] for p in _phases(workflows_data, wf_name)]
    assert PHASE_NAME in names, f"{PHASE_NAME} missing from {wf_name}"
    assert "packaging" in names, f"packaging missing from {wf_name}"
    assert names.index(PHASE_NAME) < names.index("packaging"), (
        f"{PHASE_NAME} must sit before packaging in {wf_name}: {names}"
    )


@pytest.mark.parametrize("wf_name", ["textbook_to_course", "course_generation"])
def test_phase_is_validator_only_routed_by_name(workflows_data, wf_name):
    phase = _phase(workflows_data, wf_name, PHASE_NAME)
    assert phase is not None
    # agents: [] is the validator-only-phase signal; the phase routes by NAME
    # via _PHASE_TOOL_MAPPING, not by an agent in AGENT_TOOL_MAPPING.
    assert phase.get("agents") == [], (
        f"{PHASE_NAME} must declare agents: [] in {wf_name}"
    )


@pytest.mark.parametrize("wf_name", ["textbook_to_course", "course_generation"])
def test_phase_declares_contracted_gates(workflows_data, wf_name):
    phase = _phase(workflows_data, wf_name, PHASE_NAME)
    gates = {g["gate_id"]: g for g in phase.get("validation_gates", [])}
    assert set(gates) == set(EXPECTED_GATES), (
        f"{wf_name}.{PHASE_NAME} gate set mismatch: {set(gates)}"
    )
    for gate_id, (severity, validator) in EXPECTED_GATES.items():
        assert gates[gate_id]["severity"] == severity, (
            f"{wf_name}.{gate_id} severity {gates[gate_id]['severity']!r} "
            f"!= {severity!r}"
        )
        assert gates[gate_id]["validator"] == validator, (
            f"{wf_name}.{gate_id} validator {gates[gate_id]['validator']!r} "
            f"!= {validator!r}"
        )
    # qti_well_formed must fail-closed (critical / block).
    qti = gates["qti_well_formed"]
    assert qti["behavior"]["on_fail"] == "block"
    assert qti["behavior"]["on_error"] == "fail_closed"
    assert qti["threshold"]["max_critical_issues"] == 0


@pytest.mark.parametrize("wf_name", ["textbook_to_course", "course_generation"])
def test_packaging_depends_on_assessment_synthesis(workflows_data, wf_name):
    packaging = _phase(workflows_data, wf_name, "packaging")
    assert PHASE_NAME in packaging.get("depends_on", []), (
        f"packaging.depends_on must include {PHASE_NAME} in {wf_name}"
    )
    # The two-pass alternate dep list must also wait on assessment_synthesis.
    alt = packaging.get("depends_on_when_env_value", [])
    assert PHASE_NAME in alt, (
        f"packaging two-pass dep list must include {PHASE_NAME} in {wf_name}"
    )


def test_phase_tool_mapping_routes_to_handler():
    from MCP.core.executor import _PHASE_TOOL_MAPPING

    assert _PHASE_TOOL_MAPPING.get(PHASE_NAME) == "run_assessment_synthesis"


def test_meta_schema_accepts_workflows_yaml(workflows_data):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(META_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(workflows_data, schema)


def test_gate_input_router_resolves_qti_dir(tmp_path):
    """The qti_well_formed builder must surface qti_dir = <export>/06_assessments."""
    from MCP.hardening.gate_input_routing import default_router

    router = default_router()
    export_dir = tmp_path / "PROJ-export"
    (export_dir / "06_assessments").mkdir(parents=True)
    # The export root is resolved from objective_extraction.project_path
    # (the canonical thread used by _find_project_export_dir).
    phase_outputs = {
        "objective_extraction": {"project_path": str(export_dir)},
    }
    inputs, missing = router.build(
        "lib.validators.qti_well_formed.QtiWellFormedValidator",
        phase_outputs,
        {},
    )
    assert missing == [], f"unexpected missing inputs: {missing}"
    assert inputs["qti_dir"] == str(export_dir / "06_assessments")
