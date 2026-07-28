from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_fresh_training_synthesis import apply_plan, build_plan
from Trainforge.synthesis_fresh_start import (
    FreshStartError,
    MARKER_NAME,
    require_fresh_start_marker,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    state_root = tmp_path / "state"
    runs = state_root / "runs"
    workflow = state_root / "workflows" / "WF-generic.json"
    specs = tmp_path / "workspace" / "trainforge" / "training_specs"
    workflow.parent.mkdir(parents=True)
    specs.mkdir(parents=True)
    workflow.write_text(json.dumps({
        "workflow_id": "WF-generic",
        "status": "PAUSED",
        "paused_phase": "training_synthesis",
        "params": {"run_id": "RUN-generic"},
        "phase_outputs": {
            "imscc_chunking": {
                "_completed": True,
                "chunks_path": "workspace/upstream/chunks.jsonl",
            },
            "trainforge_assessment": {
                "_completed": True,
                "assessments_path": "workspace/upstream/assessments.json",
            },
            "training_synthesis": {
                "_completed": False,
                "instruction_pairs": 51,
            },
            "libv2_archival": {"_completed": True},
        },
        "tasks": [
            {"phase": "imscc_chunking", "status": "COMPLETE"},
            {"phase": "training_synthesis", "status": "PAUSED"},
            {"phase": "libv2_archival", "status": "COMPLETE"},
        ],
    }), encoding="utf-8")
    for name in (
        ".synthesis_pairs_checkpoint.jsonl",
        ".synthesis_cache.jsonl",
        ".synthesis_generation_checkpoint.jsonl",
        ".synthesis_generation_checkpoint.jsonl.lock",
        ".synthesis_seat_recovery.jsonl",
        ".synthesis_telemetry.jsonl",
        "instruction_pairs.jsonl.in_progress",
        "preference_pairs.jsonl.in_progress",
        "pilot_report.md",
        "pilot_progress.json",
    ):
        (specs / name).write_text(f"derived:{name}", encoding="utf-8")
    # These are upstream inputs and must survive.
    (specs / "assessments.json").write_text("assessment-input", encoding="utf-8")
    (specs / "answer_key.json").write_text("answer-input", encoding="utf-8")
    run = runs / "RUN-generic"
    (run / "checkpoints").mkdir(parents=True)
    (run / "telemetry").mkdir()
    (run / "control").mkdir()
    (run / "checkpoints" / "training_synthesis_checkpoint.json").write_text(
        "checkpoint", encoding="utf-8",
    )
    (run / "telemetry" / "training_synthesis.json").write_text(
        "telemetry", encoding="utf-8",
    )
    (run / "control" / "STOP_REQUESTED").write_text("stop", encoding="utf-8")
    return workflow, specs, runs, state_root / "backups" / "fresh"


def test_dry_run_is_read_only_and_selects_only_derived_artifacts(
    tmp_path: Path,
) -> None:
    workflow, specs, runs, archive = _fixture(tmp_path)
    before = {
        path: _sha(path)
        for path in [workflow, *specs.iterdir()]
        if path.is_file()
    }
    plan = build_plan(
        workflow_state_path=workflow,
        training_specs_dir=specs,
        runs_dir=runs,
        archive_dir=archive,
    )
    after = {path: _sha(path) for path in before}
    assert before == after
    assert not archive.exists()
    names = {Path(path).name for path in plan["artifacts_to_archive"]}
    assert ".synthesis_pairs_checkpoint.jsonl" in names
    assert "pilot_report.md" in names
    assert "assessments.json" not in names
    assert "answer_key.json" not in names
    assert plan["phases_to_reset"] == [
        "training_synthesis", "libv2_archival",
    ]


def test_apply_archives_old_caches_and_preserves_upstream_inputs(
    tmp_path: Path,
) -> None:
    workflow, specs, runs, archive = _fixture(tmp_path)
    assessment_hash = _sha(specs / "assessments.json")
    answer_hash = _sha(specs / "answer_key.json")
    original = json.loads(workflow.read_text(encoding="utf-8"))
    plan = build_plan(
        workflow_state_path=workflow,
        training_specs_dir=specs,
        runs_dir=runs,
        archive_dir=archive,
    )
    apply_plan(plan)

    state = json.loads(workflow.read_text(encoding="utf-8"))
    assert state["status"] == "CREATED"
    assert "paused_phase" not in state
    assert list(state["phase_outputs"]) == [
        "imscc_chunking", "trainforge_assessment",
    ]
    assert state["phase_outputs"]["imscc_chunking"] == (
        original["phase_outputs"]["imscc_chunking"]
    )
    assert state["phase_outputs"]["trainforge_assessment"] == (
        original["phase_outputs"]["trainforge_assessment"]
    )
    assert [task["phase"] for task in state["tasks"]] == ["imscc_chunking"]
    assert _sha(specs / "assessments.json") == assessment_hash
    assert _sha(specs / "answer_key.json") == answer_hash
    assert not (specs / ".synthesis_pairs_checkpoint.jsonl").exists()
    assert (
        archive / "training_specs" / ".synthesis_pairs_checkpoint.jsonl"
    ).is_file()
    assert (
        archive / "run_state" / "checkpoints"
        / "training_synthesis_checkpoint.json"
    ).is_file()
    manifest = json.loads(
        (archive / "fresh_start_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["preserved_upstream_phase_count"] == 2
    assert manifest["phases_reset"] == [
        "training_synthesis", "libv2_archival",
    ]
    marker = json.loads((specs / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["fresh_start_id"] == manifest["fresh_start_id"]
    assert len(marker["fresh_start_id"]) == 32
    assert marker["preserved_input_sha256"][str(specs / "assessments.json")]
    assert require_fresh_start_marker(specs) == marker
    assert (archive / "fresh_start_manifest.json").stat().st_mode & 0o222 == 0
    assert (
        archive / "training_specs" / ".synthesis_cache.jsonl"
    ).stat().st_mode & 0o222 == 0


def test_stale_cache_reappearance_is_rejected(tmp_path: Path) -> None:
    workflow, specs, runs, archive = _fixture(tmp_path)
    plan = build_plan(
        workflow_state_path=workflow,
        training_specs_dir=specs,
        runs_dir=runs,
        archive_dir=archive,
        fresh_start_id="fresh-start-test-identity",
    )
    apply_plan(plan)
    (specs / ".synthesis_pairs_checkpoint.jsonl").write_text(
        '{"disposition":"rejected","contract_fingerprint":"stale"}\n',
        encoding="utf-8",
    )
    with pytest.raises(FreshStartError, match="stale synthesis artifacts"):
        require_fresh_start_marker(specs)


def test_apply_refuses_changed_plan_without_mutating_state(
    tmp_path: Path,
) -> None:
    workflow, specs, runs, archive = _fixture(tmp_path)
    workflow_before = _sha(workflow)
    plan = build_plan(
        workflow_state_path=workflow,
        training_specs_dir=specs,
        runs_dir=runs,
        archive_dir=archive,
    )
    checkpoint = specs / ".synthesis_pairs_checkpoint.jsonl"
    checkpoint.write_text("changed-after-review", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after dry-run"):
        apply_plan(plan)
    assert _sha(workflow) == workflow_before
    assert not archive.exists()
    assert checkpoint.read_text(encoding="utf-8") == "changed-after-review"


def test_marker_rejects_modified_preserved_input(tmp_path: Path) -> None:
    workflow, specs, runs, archive = _fixture(tmp_path)
    plan = build_plan(
        workflow_state_path=workflow,
        training_specs_dir=specs,
        runs_dir=runs,
        archive_dir=archive,
    )
    apply_plan(plan)
    (specs / "assessments.json").write_text("changed", encoding="utf-8")
    with pytest.raises(FreshStartError, match="preserved synthesis inputs changed"):
        require_fresh_start_marker(specs)


def test_marker_rejects_modified_evidence_manifest(tmp_path: Path) -> None:
    workflow, specs, runs, archive = _fixture(tmp_path)
    plan = build_plan(
        workflow_state_path=workflow,
        training_specs_dir=specs,
        runs_dir=runs,
        archive_dir=archive,
    )
    apply_plan(plan)
    manifest = archive / "fresh_start_manifest.json"
    manifest.chmod(0o644)
    manifest.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(FreshStartError, match="manifest is missing or its hash changed"):
        require_fresh_start_marker(specs)


def test_plan_manifest_and_marker_preserve_exact_holdout_identity(
    tmp_path: Path,
) -> None:
    workflow, specs, runs, archive = _fixture(tmp_path)
    identity = {
        "holdout_manifest_sha256": "a" * 64,
        "holdout_trust_chain_fingerprint": "b" * 64,
    }
    plan = build_plan(
        workflow_state_path=workflow,
        training_specs_dir=specs,
        runs_dir=runs,
        archive_dir=archive,
        holdout_identity=identity,
    )
    assert plan["synthesis_holdout_identity"] == identity
    apply_plan(plan)
    marker = json.loads((specs / MARKER_NAME).read_text(encoding="utf-8"))
    manifest = json.loads(
        (archive / "fresh_start_manifest.json").read_text(encoding="utf-8")
    )
    assert marker["synthesis_holdout_identity"] == identity
    assert manifest["synthesis_holdout_identity"] == identity
    assert require_fresh_start_marker(
        specs, expected_holdout_identity=identity
    ) == marker
