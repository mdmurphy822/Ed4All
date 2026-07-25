"""Tests for the full-pipeline chain service + endpoint.

``gui.services.pipeline_service.pipeline_chain`` correlates a course's Stage-A
BUILD run (``textbook_to_course`` / ``course_generation``) with its Stage-B
``trainforge_train`` run — two SEPARATE run records tied only by the course slug.
These tests pin:

- build + training both present (discovered from GUI records),
- training PLANNED-ONLY (not started) still emits the training stage with its
  ``planned_phases`` (the pipeline tail the operator sees),
- a completed training discovered ONLY via ``LibV2/courses/<slug>/models/`` with
  no run record (``present: true``, ``run_id: null``),
- a training run discovered from a bare ``state/workflows/*.json`` file with NO
  GUI record (CLI/pilot-launched),
- a non-build workflow (``rag_training``) → a single stage, no training tail,
- an unresolvable slug → single-stage chain with ``course_slug: null``,
- never-raises on a corrupt workflow-state file.

State isolated via ``state_dir`` + ``libv2_root``; the endpoint tests need
fastapi (opt-in ``gui`` extra) and skip cleanly without it. All course
slugs/names here are INVENTED synthetic fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gui import shared_state
from gui.services import pipeline_service
from lib.ontology.slugs import libv2_course_slug

# Invented synthetic course identity (never a real book/campaign name).
COURSE_NAME = "Sample Physics 101"
COURSE_SLUG = libv2_course_slug(COURSE_NAME)


@pytest.fixture(autouse=True)
def _reset_index_cache():
    """Drop the mtime-keyed workflow-index cache between tests (path-keyed)."""
    pipeline_service._WF_INDEX_CACHE["mtime"] = None
    pipeline_service._WF_INDEX_CACHE["entries"] = []
    yield


def _write_workflow_state(
    state_dir: Path,
    workflow_id: str,
    *,
    wf_type: str,
    status: str = "RUNNING",
    course_key: str = "course_name",
    course_value: str = COURSE_NAME,
    started_at: str = "2026-01-01T00:00:00",
    extra_params: Optional[Dict[str, Any]] = None,
) -> None:
    wf_dir = state_dir / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    params: Dict[str, Any] = {course_key: course_value}
    if extra_params:
        params.update(extra_params)
    doc = {
        "id": workflow_id,
        "type": wf_type,
        "status": status,
        "started_at": started_at,
        "params": params,
        "phase_outputs": {},
    }
    (wf_dir / f"{workflow_id}.json").write_text(json.dumps(doc), encoding="utf-8")


def _register_gui_run(
    run_id: str,
    workflow_id: str,
    *,
    workflow: str,
    status: str = "running",
    course_name: str = COURSE_NAME,
    started_at: str = "2026-01-01T00:00:00",
) -> None:
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow": workflow,
            "workflow_id": workflow_id,
            "course_name": course_name,
            "status": status,
            "params": {"course_name": course_name},
            "started_at": started_at,
        }
    )


def _seed_build_run(
    state_dir: Path,
    *,
    run_id: str = "GUI-build-0001",
    workflow_id: str = "WF-build-0001",
    workflow: str = "textbook_to_course",
    status: str = "running",
    started_at: str = "2026-01-01T00:00:00",
) -> str:
    _register_gui_run(
        run_id, workflow_id, workflow=workflow, status=status, started_at=started_at
    )
    _write_workflow_state(
        state_dir, workflow_id, wf_type=workflow, status=status.upper(),
        started_at=started_at,
    )
    return run_id


def _make_models(libv2_root: Path, slug: str, model_ids: List[str]) -> None:
    for mid in model_ids:
        (libv2_root / "courses" / slug / "models" / mid).mkdir(
            parents=True, exist_ok=True
        )


# ------------------------------------------------------- build + training both


def test_build_and_training_both_present(state_dir, libv2_root):
    build_id = _seed_build_run(state_dir, started_at="2026-01-01T00:00:00")
    # A trainforge_train run for the same course, launched later.
    _register_gui_run(
        "GUI-train-0001",
        "WF-train-0001",
        workflow="trainforge_train",
        status="completed",
        started_at="2026-01-02T00:00:00",
    )
    _write_workflow_state(
        state_dir, "WF-train-0001", wf_type="trainforge_train", status="COMPLETED",
        course_key="course_code", started_at="2026-01-02T00:00:00",
    )
    _make_models(libv2_root, COURSE_SLUG, ["adapter-v1"])

    chain = pipeline_service.pipeline_chain(build_id)
    assert chain is not None
    assert chain["run_id"] == build_id
    assert chain["course_slug"] == COURSE_SLUG
    assert chain["course_name"] == COURSE_NAME
    assert chain["current_stage"] == "training"
    assert [s["stage"] for s in chain["stages"]] == ["build", "training"]

    build, training = chain["stages"]
    assert build["workflow"] == "textbook_to_course"
    assert build["run_id"] == build_id
    assert build["present"] is True
    assert build["status"] == "running"
    assert "semantik_conversion" in build["planned_phases"]

    assert training["workflow"] == "trainforge_train"
    assert training["run_id"] == "GUI-train-0001"
    assert training["present"] is True
    assert training["status"] == "completed"
    assert training["planned_phases"] == ["training", "post_training_validation"]
    assert training["model_ids"] == ["adapter-v1"]


# ------------------------------------------------------- training planned only


def test_training_planned_only_still_emits_stage(state_dir, libv2_root):
    build_id = _seed_build_run(state_dir)
    chain = pipeline_service.pipeline_chain(build_id)
    assert chain is not None
    assert chain["current_stage"] == "build"
    build, training = chain["stages"]
    assert training["stage"] == "training"
    assert training["present"] is False
    assert training["run_id"] is None
    assert training["status"] is None
    assert training["model_ids"] == []
    # The planned tail is ALWAYS shown so the operator sees the full pipeline.
    assert training["planned_phases"] == ["training", "post_training_validation"]


# ---------------------------------------- training discovered via model dirs


def test_completed_training_discovered_via_model_dirs(state_dir, libv2_root):
    build_id = _seed_build_run(state_dir)
    # No trainforge_train run record anywhere — only the products on disk.
    _make_models(libv2_root, COURSE_SLUG, ["adapter-b", "adapter-a"])

    chain = pipeline_service.pipeline_chain(build_id)
    assert chain is not None
    training = chain["stages"][1]
    assert training["present"] is True  # model dirs exist
    assert training["run_id"] is None  # ...but no discoverable run record
    assert training["status"] is None
    assert training["model_ids"] == ["adapter-a", "adapter-b"]  # sorted
    assert chain["current_stage"] == "training"


# -------------------------------- training discovered from bare workflow file


def test_training_discovered_from_cli_workflow_file(state_dir, libv2_root):
    """A CLI/pilot-launched trainforge_train run has NO GUI record — only a
    state/workflows/*.json file. It must still be discovered."""
    build_id = _seed_build_run(state_dir)
    _write_workflow_state(
        state_dir, "WF-cli-train-0001", wf_type="trainforge_train",
        status="RUNNING", course_key="course_code",
        started_at="2026-02-01T00:00:00",
    )
    chain = pipeline_service.pipeline_chain(build_id)
    assert chain is not None
    training = chain["stages"][1]
    assert training["present"] is True
    assert training["run_id"] == "WF-cli-train-0001"  # the bare workflow id
    assert training["status"] == "running"


# ------------------------------------------------- non-build → single stage


def test_non_build_workflow_single_stage(state_dir, libv2_root):
    run_id = "GUI-rag-0001"
    _register_gui_run(run_id, "WF-rag-0001", workflow="rag_training", status="running")
    _write_workflow_state(
        state_dir, "WF-rag-0001", wf_type="rag_training", status="RUNNING"
    )
    chain = pipeline_service.pipeline_chain(run_id)
    assert chain is not None
    assert len(chain["stages"]) == 1  # no training tail for a non-build workflow
    assert chain["stages"][0]["workflow"] == "rag_training"
    assert chain["course_slug"] == COURSE_SLUG
    assert chain["current_stage"] == "build"


# ----------------------------------------------- unresolvable slug → 1 stage


def test_unresolvable_slug_single_stage_null_slug(state_dir, libv2_root):
    # A build run whose params carry NO course name → no slug is derivable.
    wf_dir = state_dir / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "WF-noname-0001.json").write_text(
        json.dumps(
            {
                "id": "WF-noname-0001",
                "type": "textbook_to_course",
                "status": "RUNNING",
                "params": {},  # no course_name / course_code
                "phase_outputs": {},
            }
        ),
        encoding="utf-8",
    )
    # Query it as a bare orchestrator workflow id.
    chain = pipeline_service.pipeline_chain("WF-noname-0001")
    assert chain is not None
    assert chain["course_slug"] is None
    assert chain["course_name"] is None
    assert len(chain["stages"]) == 1
    stage = chain["stages"][0]
    assert stage["stage"] == "build"  # inferred from the workflow type
    assert stage["run_id"] == "WF-noname-0001"
    assert stage["present"] is True


def test_unresolvable_slug_training_type_infers_training_stage(state_dir, libv2_root):
    wf_dir = state_dir / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "WF-noname-train.json").write_text(
        json.dumps(
            {
                "id": "WF-noname-train",
                "type": "trainforge_train",
                "status": "RUNNING",
                "params": {},
                "phase_outputs": {},
            }
        ),
        encoding="utf-8",
    )
    chain = pipeline_service.pipeline_chain("WF-noname-train")
    assert chain is not None
    assert chain["course_slug"] is None
    assert chain["current_stage"] == "training"
    assert chain["stages"][0]["stage"] == "training"
    assert chain["stages"][0]["model_ids"] == []


# ----------------------------------------------------- query the training run


def test_query_training_run_directly(state_dir, libv2_root):
    """Querying the trainforge_train run itself yields the same full chain, with
    the training stage pinned to the queried run."""
    _seed_build_run(state_dir, started_at="2026-01-01T00:00:00")
    _register_gui_run(
        "GUI-train-q",
        "WF-train-q",
        workflow="trainforge_train",
        status="running",
        started_at="2026-01-02T00:00:00",
    )
    _write_workflow_state(
        state_dir, "WF-train-q", wf_type="trainforge_train", status="RUNNING",
        course_key="course_code", started_at="2026-01-02T00:00:00",
    )
    chain = pipeline_service.pipeline_chain("GUI-train-q")
    assert chain is not None
    assert chain["run_id"] == "GUI-train-q"
    assert [s["stage"] for s in chain["stages"]] == ["build", "training"]
    build, training = chain["stages"]
    assert build["present"] is True  # sibling build discovered by slug
    assert build["run_id"] == "GUI-build-0001"
    assert training["run_id"] == "GUI-train-q"  # the queried run, pinned
    assert chain["current_stage"] == "training"


# ---------------------------------------------------------- robustness / 404


def test_unknown_run_is_none(state_dir, libv2_root):
    assert pipeline_service.pipeline_chain("GUI-does-not-exist") is None


def test_never_raises_on_corrupt_state_file(state_dir, libv2_root):
    """A corrupt/mid-write workflow file is treated as absent, never a crash."""
    build_id = _seed_build_run(state_dir)
    wf_dir = state_dir / "workflows"
    (wf_dir / "WF-corrupt-0001.json").write_text("{not valid json", encoding="utf-8")
    # The scan must skip the corrupt file and still return a valid chain.
    chain = pipeline_service.pipeline_chain(build_id)
    assert chain is not None
    assert chain["course_slug"] == COURSE_SLUG
    # A corrupt file queried directly → no state, no record → None (404).
    assert pipeline_service.pipeline_chain("WF-corrupt-0001") is None


# ----------------------------------------------------------------- endpoint


@pytest.fixture
def client(state_dir, libv2_root):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gui.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


def test_endpoint_happy_path(client, state_dir, libv2_root):
    build_id = _seed_build_run(state_dir)
    _make_models(libv2_root, COURSE_SLUG, ["adapter-v1"])
    resp = client.get(f"/api/runs/{build_id}/pipeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == build_id
    assert body["course_slug"] == COURSE_SLUG
    assert [s["stage"] for s in body["stages"]] == ["build", "training"]
    assert body["stages"][1]["model_ids"] == ["adapter-v1"]


def test_endpoint_unknown_run_404(client):
    resp = client.get("/api/runs/GUI-no-such-run/pipeline")
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown_run", "detail": "GUI-no-such-run"}
