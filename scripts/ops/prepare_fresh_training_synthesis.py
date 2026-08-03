#!/usr/bin/env python3
"""Prepare an existing workflow for a genuinely fresh synthesis pass.

The operation is intentionally narrow:

* preserve every phase before ``training_synthesis`` byte-for-byte;
* reset ``training_synthesis`` and any already-observed downstream phases;
* archive, rather than delete, synthesis outputs, resume journals, pilot
  artifacts, telemetry, the phase checkpoint, and a stale stop sentinel;
* leave upstream corpus, objectives, assessments, and other Trainforge inputs
  in place.

Dry-run is the default.  ``--apply`` uses atomic state replacement and moves
each derived artifact into a timestamped evidence archive.  This tool never
starts a model service or resumes a workflow.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Trainforge.synthesis_fresh_start import (
    MARKER_NAME,
    MARKER_SCHEMA_VERSION,
    SYNTHESIS_ARTIFACT_NAMES,
    fingerprint_files,
    sha256_file,
)
from Trainforge.synthesis_holdout import (
    holdout_exclusion_enabled,
    load_synthesis_holdout_registry,
)

TARGET_PHASE = "training_synthesis"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _params(state: Mapping[str, Any]) -> dict[str, Any]:
    value = state.get("params", {})
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value) if isinstance(value, Mapping) else {}


def _run_id(state: Mapping[str, Any], workflow_id: str) -> str:
    return str(_params(state).get("run_id") or workflow_id).strip()


def _phase_names_to_reset(state: Mapping[str, Any]) -> list[str]:
    outputs = state.get("phase_outputs", {})
    if not isinstance(outputs, Mapping):
        return [TARGET_PHASE]
    names = list(outputs)
    if TARGET_PHASE not in names:
        return [TARGET_PHASE]
    return names[names.index(TARGET_PHASE):]


def _artifact_paths(training_specs_dir: Path) -> list[Path]:
    if not training_specs_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in training_specs_dir.iterdir()
            if path.is_file()
            and (
                path.name in SYNTHESIS_ARTIFACT_NAMES
                or path.name == MARKER_NAME
            )
        ),
        key=lambda path: path.name,
    )


def _preserved_input_paths(
    training_specs_dir: Path, explicit_paths: Sequence[Path],
) -> list[Path]:
    implicit = (
        path
        for path in training_specs_dir.iterdir()
        if path.is_file()
        and path.name not in SYNTHESIS_ARTIFACT_NAMES
        and path.name != MARKER_NAME
    ) if training_specs_dir.is_dir() else ()
    paths = sorted(set([*implicit, *explicit_paths]), key=lambda path: str(path))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            "preserved input does not exist or is not a file: "
            + ", ".join(str(path) for path in missing)
        )
    if not paths:
        raise ValueError(
            "no preserved inputs found; pass --upstream-input for chunks/objectives"
        )
    return paths


def build_plan(
    *,
    workflow_state_path: Path,
    training_specs_dir: Path,
    runs_dir: Path,
    archive_dir: Path,
    upstream_inputs: Sequence[Path] = (),
    fresh_start_id: Optional[str] = None,
    holdout_identity: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Return the complete fresh-start plan without changing the filesystem."""

    state = _load_json(workflow_state_path)
    workflow_id = str(
        state.get("id") or state.get("workflow_id") or workflow_state_path.stem
    )
    run_id = _run_id(state, workflow_id)
    run_root = runs_dir / run_id
    phases = _phase_names_to_reset(state)
    runtime_candidates = [
        run_root / "checkpoints" / f"{TARGET_PHASE}_checkpoint.json",
        run_root / "telemetry" / f"{TARGET_PHASE}.json",
        run_root / "control" / "STOP_REQUESTED",
    ]
    derived = _artifact_paths(training_specs_dir)
    derived.extend(path for path in runtime_candidates if path.is_file())
    preserved = _preserved_input_paths(training_specs_dir, upstream_inputs)
    artifact_sha256 = fingerprint_files(derived)
    preserved_sha256 = fingerprint_files(preserved)
    if holdout_identity is None and holdout_exclusion_enabled():
        chunks_candidates = [
            path for path in upstream_inputs if path.name == "chunks.jsonl"
        ]
        objectives_candidates = [
            path for path in upstream_inputs if path.name == "objectives.json"
        ]
        if len(chunks_candidates) != 1 or len(objectives_candidates) != 1:
            raise ValueError(
                "holdout-enabled fresh start requires exactly one chunks.jsonl "
                "and one objectives.json in --upstream-input"
            )
        registry = load_synthesis_holdout_registry(
            chunks_path=chunks_candidates[0],
            objectives_path=objectives_candidates[0],
        )
        if registry is None:  # pragma: no cover - guarded by enabled predicate
            raise ValueError("holdout identity was not produced")
        holdout_identity = registry.identity
    plan = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "fresh_start_id": fresh_start_id or uuid.uuid4().hex,
        "workflow_id": workflow_id,
        "run_id": run_id,
        "workflow_state_path": str(workflow_state_path),
        "training_specs_dir": str(training_specs_dir),
        "runs_dir": str(runs_dir),
        "archive_dir": str(archive_dir),
        "phases_to_reset": phases,
        "artifacts_to_archive": [str(path) for path in derived],
        "artifact_sha256": artifact_sha256,
        "preserved_input_sha256": preserved_sha256,
        "upstream_phase_count": _upstream_phase_count(state, phases),
    }
    if holdout_identity is not None:
        plan["synthesis_holdout_identity"] = dict(holdout_identity)
    return plan


def _upstream_phase_count(
    state: Mapping[str, Any], phases_to_reset: Sequence[str],
) -> int:
    outputs = state.get("phase_outputs", {})
    if not isinstance(outputs, Mapping):
        return 0
    reset = set(phases_to_reset)
    return sum(name not in reset for name in outputs)


def _reset_state(state: dict[str, Any], phases: Iterable[str]) -> None:
    reset = set(phases)
    outputs = state.get("phase_outputs", {})
    if isinstance(outputs, dict):
        for phase in reset:
            outputs.pop(phase, None)
    state["tasks"] = [
        task
        for task in (state.get("tasks") or [])
        if not (isinstance(task, Mapping) and task.get("phase") in reset)
    ]
    state["status"] = "CREATED"
    for marker in (
        "failed_phase", "failure_reason", "paused_phase", "stopped_after",
        "started_at", "completed_at",
    ):
        state.pop(marker, None)
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_plan(plan: Mapping[str, Any]) -> None:
    """Archive derived state and atomically reset only the synthesis tail."""

    state_path = Path(str(plan["workflow_state_path"]))
    archive_dir = Path(str(plan["archive_dir"]))

    # Preflight the complete reviewed plan before making the first mutation.
    for raw, expected_sha in plan["artifact_sha256"].items():
        source = Path(str(raw))
        if not source.is_file():
            raise ValueError(f"planned artifact disappeared after dry-run: {source}")
        actual_sha = sha256_file(source)
        if actual_sha != expected_sha:
            raise ValueError(
                f"artifact changed after dry-run: {source} "
                f"(expected {expected_sha}, found {actual_sha})"
            )
    for raw, expected_sha in plan["preserved_input_sha256"].items():
        source = Path(str(raw))
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise ValueError(f"preserved input changed after dry-run: {source}")

    archive_dir.mkdir(parents=True, exist_ok=False)
    original_state = state_path.read_bytes()
    (archive_dir / "workflow.before.json").write_bytes(original_state)

    roots = {
        Path(str(plan["training_specs_dir"])): "training_specs",
        Path(str(plan["runs_dir"])) / str(plan["run_id"]): "run_state",
    }
    moved: list[dict[str, str]] = []
    for raw in plan["artifacts_to_archive"]:
        source = Path(str(raw))
        if not source.is_file():
            continue
        actual_sha = sha256_file(source)
        relative: Optional[Path] = None
        category = "other"
        for root, label in roots.items():
            try:
                relative = source.relative_to(root)
            except ValueError:
                continue
            category = label
            break
        destination = archive_dir / category / (
            relative if relative is not None else Path(source.name)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append({
            "source": str(source),
            "archive": str(destination),
            "sha256": actual_sha,
        })

    leftovers = sorted(
        name
        for name in SYNTHESIS_ARTIFACT_NAMES
        if (Path(str(plan["training_specs_dir"])) / name).exists()
    )
    if leftovers:
        raise ValueError(
            "fresh-start reset left stale synthesis artifacts: "
            + ", ".join(leftovers)
        )

    state = json.loads(original_state)
    _reset_state(state, plan["phases_to_reset"])
    _atomic_json_write(state_path, state)
    manifest = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "fresh_start_id": plan["fresh_start_id"],
        "workflow_id": plan["workflow_id"],
        "run_id": plan["run_id"],
        "target_phase": TARGET_PHASE,
        "phases_reset": list(plan["phases_to_reset"]),
        "moved": moved,
        "preserved_upstream_phase_count": plan["upstream_phase_count"],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if "synthesis_holdout_identity" in plan:
        manifest["synthesis_holdout_identity"] = dict(
            plan["synthesis_holdout_identity"]
        )
    manifest_path = archive_dir / "fresh_start_manifest.json"
    _atomic_json_write(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    marker = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "fresh_start_id": plan["fresh_start_id"],
        "workflow_id": plan["workflow_id"],
        "run_id": plan["run_id"],
        "created_at": manifest["created_at"],
        "archive_manifest_path": str(manifest_path),
        "archive_manifest_sha256": manifest_sha256,
        "preserved_input_sha256": dict(plan["preserved_input_sha256"]),
    }
    if "synthesis_holdout_identity" in plan:
        marker["synthesis_holdout_identity"] = dict(
            plan["synthesis_holdout_identity"]
        )
    _atomic_json_write(
        Path(str(plan["training_specs_dir"])) / MARKER_NAME, marker,
    )

    # Evidence is retained locally but excluded from future synthesis and made
    # read-only after every manifest/hash write has completed.
    for path in archive_dir.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (path for path in archive_dir.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    archive_dir.chmod(0o555)


def _default_archive(workflow_id: str, state_root: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return state_root / "backups" / f"{workflow_id}-fresh-synthesis-{stamp}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-state", type=Path, required=True)
    parser.add_argument("--training-specs-dir", type=Path, required=True)
    parser.add_argument(
        "--runs-dir", type=Path, required=True,
        help="Run-state root containing <run_id>/checkpoints and telemetry.",
    )
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument(
        "--upstream-input",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional upstream file to hash into the fresh-start marker "
            "(repeat for chunks, objectives, or other inputs outside training_specs)."
        ),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    try:
        state = _load_json(args.workflow_state)
        workflow_id = str(
            state.get("id") or state.get("workflow_id")
            or args.workflow_state.stem
        )
        state_root = args.runs_dir.parent
        archive = args.archive_dir or _default_archive(workflow_id, state_root)
        plan = build_plan(
            workflow_state_path=args.workflow_state,
            training_specs_dir=args.training_specs_dir,
            runs_dir=args.runs_dir,
            archive_dir=archive,
            upstream_inputs=args.upstream_input,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        if args.apply:
            apply_plan(plan)
            print(f"applied: evidence archived under {archive}")
        else:
            print("dry-run: no files changed; pass --apply after review")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
