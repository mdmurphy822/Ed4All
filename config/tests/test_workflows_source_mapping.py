"""Wave 9 — config/workflows.yaml `source_mapping` phase wiring tests.

Confirms:

* The `source_mapping` phase exists in the `textbook_to_course` workflow
  with the correct dependency + outputs.
* The meta-schema at `schemas/config/workflows_meta.schema.json`
  accepts the extended workflows.yaml clean.
* Downstream phases (`course_planning`, `content_generation`) receive
  the routed inputs from `source_mapping`.
* The `source_refs` validation gate is wired on `content_generation`
  at critical severity.
* The cross-reference integrity check (from Wave 6 Worker V) still
  passes — every `phase_outputs` reference resolves to a declared
  prior-phase output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WORKFLOWS_YAML = PROJECT_ROOT / "config" / "workflows.yaml"
META_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "config" / "workflows_meta.schema.json"


@pytest.fixture(scope="module")
def workflows_data():
    with open(WORKFLOWS_YAML) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def textbook_phases(workflows_data):
    return {p["name"]: p for p in workflows_data["workflows"]["textbook_to_course"]["phases"]}


# ---------------------------------------------------------------------- #
# Phase structure
# ---------------------------------------------------------------------- #


class TestSourceMappingPhase:
    def test_phase_exists(self, textbook_phases):
        assert "source_mapping" in textbook_phases

    def test_phase_runs_source_router_agent(self, textbook_phases):
        phase = textbook_phases["source_mapping"]
        assert phase["agents"] == ["source-router"]

    def test_phase_is_sequential(self, textbook_phases):
        phase = textbook_phases["source_mapping"]
        assert phase.get("parallel", False) is False
        assert phase.get("max_concurrent", 1) == 1

    def test_phase_depends_on_objective_extraction(self, textbook_phases):
        phase = textbook_phases["source_mapping"]
        assert "objective_extraction" in phase["depends_on"]

    def test_phase_declares_expected_outputs(self, textbook_phases):
        phase = textbook_phases["source_mapping"]
        assert "source_module_map_path" in phase["outputs"]
        assert "source_chunk_ids" in phase["outputs"]

    def test_phase_inputs_from_include_staging_and_structure(self, textbook_phases):
        phase = textbook_phases["source_mapping"]
        params = {entry["param"] for entry in phase["inputs_from"]}
        assert "staging_dir" in params
        assert "textbook_structure_path" in params
        assert "project_id" in params


class TestCoursePlanningUpdated:
    def test_course_planning_receives_source_module_map(self, textbook_phases):
        phase = textbook_phases["course_planning"]
        params = {entry["param"] for entry in phase["inputs_from"]}
        assert "source_module_map_path" in params

    def test_course_planning_depends_on_concept_extraction(self, textbook_phases):
        """Phase 6 inserted ``concept_extraction`` between ``source_mapping`` and
        ``course_planning``; ``source_mapping`` is now a transitive predecessor
        via ``concept_extraction.depends_on: [source_mapping, chunking]``."""
        phase = textbook_phases["course_planning"]
        assert "concept_extraction" in phase["depends_on"]

    def test_course_planning_routes_duration_weeks_explicit(self, textbook_phases):
        """Wave 40: the flag gates ``_plan_course_structure``'s
        config-over-kwargs precedence. If the flag isn't routed, the
        Python default ``True`` wins and the Wave 40 fix is dead
        code on production workflow runs."""
        phase = textbook_phases["course_planning"]
        params = {entry["param"] for entry in phase["inputs_from"]}
        assert "duration_weeks_explicit" in params, (
            "course_planning must route duration_weeks_explicit from "
            "workflow_params so the auto-scaled project_config.json "
            "value is honored when --weeks was unset."
        )
        entry = next(
            e for e in phase["inputs_from"]
            if e["param"] == "duration_weeks_explicit"
        )
        assert entry["source"] == "workflow_params"
        assert entry["key"] == "duration_weeks_explicit"


class TestContentGenerationUpdated:
    def test_content_generation_receives_source_module_map(self, textbook_phases):
        phase = textbook_phases["content_generation"]
        params = {entry["param"] for entry in phase["inputs_from"]}
        assert "source_module_map_path" in params

    def test_content_generation_receives_staging_dir(self, textbook_phases):
        phase = textbook_phases["content_generation"]
        params = {entry["param"] for entry in phase["inputs_from"]}
        assert "staging_dir" in params

    def test_content_generation_routes_duration_weeks_explicit(self, textbook_phases):
        """Wave 40 companion: same reason as course_planning — without
        the flag routed, ``_generate_course_content``'s config-over-
        kwargs precedence falls back to the default ``True`` and
        clobbers the auto-scaled project_config value."""
        phase = textbook_phases["content_generation"]
        params = {entry["param"] for entry in phase["inputs_from"]}
        assert "duration_weeks_explicit" in params
        entry = next(
            e for e in phase["inputs_from"]
            if e["param"] == "duration_weeks_explicit"
        )
        assert entry["source"] == "workflow_params"
        assert entry["key"] == "duration_weeks_explicit"

    def test_source_refs_gate_wired_at_critical(self, textbook_phases):
        phase = textbook_phases["content_generation"]
        gates = {g["gate_id"]: g for g in (phase.get("validation_gates") or [])}
        assert "source_refs" in gates
        gate = gates["source_refs"]
        assert gate["severity"] == "critical"
        assert gate["validator"] == (
            "lib.validators.source_refs.PageSourceRefValidator"
        )
        assert gate["behavior"]["on_fail"] == "block"
        assert gate["behavior"]["on_error"] == "fail_closed"


class TestObjectiveExtractionExposesStructurePath:
    """source_mapping consumes objective_extraction.textbook_structure_path."""

    def test_outputs_include_textbook_structure_path(self, textbook_phases):
        phase = textbook_phases["objective_extraction"]
        assert "textbook_structure_path" in phase["outputs"]


# ---------------------------------------------------------------------- #
# Meta-schema validation (Wave 6 Worker V gate still passes)
# ---------------------------------------------------------------------- #


class TestMetaSchemaAcceptsExtendedWorkflows:
    def test_meta_schema_validates_clean(self, workflows_data):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(META_SCHEMA_PATH.read_text())
        jsonschema.validate(workflows_data, schema)

    def test_cross_reference_integrity_still_holds(self):
        """REC-CTR-05: every phase_outputs reference must resolve."""
        from MCP.core.workflow_runner import _validate_inputs_from_references

        with open(WORKFLOWS_YAML) as f:
            cfg = yaml.safe_load(f)
        # Should not raise.
        _validate_inputs_from_references(cfg)
