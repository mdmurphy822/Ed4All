"""Regression coverage for archive workspace and post-index gate routing."""

from pathlib import Path

import yaml


def _phases() -> dict:
    config = yaml.safe_load(
        (Path(__file__).parents[2] / "config" / "workflows.yaml").read_text(
            encoding="utf-8"
        )
    )
    return {
        phase["name"]: phase
        for phase in config["workflows"]["textbook_to_course"]["phases"]
    }


def test_archival_consumes_exact_trainforge_workspace():
    phases = _phases()
    assessment = phases["trainforge_assessment"]
    assert "trainforge_dir" in assessment["outputs"]
    routes = {
        route["param"]: route
        for route in phases["libv2_archival"]["inputs_from"]
    }
    assert routes["project_workspace"] == {
        "param": "project_workspace",
        "source": "phase_outputs",
        "phase": "trainforge_assessment",
        "output": "trainforge_dir",
    }


def test_course_completeness_runs_after_vector_indexing():
    phases = _phases()
    archival_gate_ids = {
        gate["gate_id"]
        for gate in phases["libv2_archival"].get("validation_gates", [])
    }
    vector_gate_ids = {
        gate["gate_id"]
        for gate in phases["vector_indexing"].get("validation_gates", [])
    }
    assert "course_completeness" not in archival_gate_ids
    assert "course_completeness" in vector_gate_ids
    assert phases["vector_indexing"]["depends_on"] == ["libv2_archival"]
