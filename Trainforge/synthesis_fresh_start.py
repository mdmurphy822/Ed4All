"""Fresh-start identity contract for training-pair synthesis.

The reset operation itself lives in ``scripts/ops/prepare_fresh_training_synthesis``.
This module is intentionally importable by the synthesis runtime without
depending on workflow-state implementation details.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


MARKER_NAME = ".synthesis_fresh_start.json"
MARKER_SCHEMA_VERSION = 1
RUN_CONTRACT_FIELD = "synthesis_run_contract_sha256"
RUN_CONTRACT_COMPONENTS_FIELD = "synthesis_run_contract_components"
RUN_CONTRACT_COMPONENTS_SHA_FIELD = "synthesis_run_contract_components_sha256"

# Every file here is synthesis-owned and must be absent when a fresh pass
# begins. Terminal rejections are rows in the pair checkpoint, so moving that
# file also excludes their contract fingerprints.
SYNTHESIS_ARTIFACT_NAMES = frozenset({
    ".synthesis_cache.jsonl",
    ".synthesis_generation_checkpoint.jsonl",
    ".synthesis_generation_checkpoint.jsonl.lock",
    ".synthesis_pairs_checkpoint.jsonl",
    ".synthesis_pairs_checkpoint.jsonl.lock",
    ".synthesis_seat_recovery.jsonl",
    ".synthesis_telemetry.jsonl",
    "curriculum_manifest.json",
    "dataset_config.json",
    "decontam_quarantine.jsonl",
    "instruction_pairs.jsonl",
    "instruction_pairs.jsonl.in_progress",
    "memorization_holdout.json",
    "pilot_progress.json",
    "pilot_report.md",
    "preference_pairs.jsonl",
    "preference_pairs.jsonl.in_progress",
    "smoke_instruction_pairs.jsonl",
    "smoke_pilot_report.md",
    "smoke_preference_pairs.jsonl",
    # Written by the synthesis runtime on a clean exit (the projection of the
    # checkpoint's terminal rejected/ineligible rows). Omitting it made
    # ``prepare_fresh_training_synthesis._preserved_input_paths`` classify it
    # as a preserved UPSTREAM input and hash it into
    # ``preserved_input_sha256``, so the next synthesis pass rewrote it and
    # every later resume hard-failed with "preserved synthesis inputs changed
    # after reset".
    "synthesis_dispositions.jsonl",
    "synthesis_summary.json",
})


class FreshStartError(RuntimeError):
    """The requested synthesis start is not demonstrably fresh."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[Path]) -> dict[str, str]:
    """Return stable, path-keyed hashes for regular files."""

    return {
        str(path): sha256_file(path)
        for path in sorted(set(paths), key=lambda value: str(value))
        if path.is_file()
    }


def load_fresh_start_marker(training_specs_dir: Path) -> dict[str, Any]:
    path = training_specs_dir / MARKER_NAME
    if not path.is_file():
        raise FreshStartError(
            f"fresh synthesis marker is required at {path}; run the scoped "
            "fresh-start preparation tool and review its dry-run first"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshStartError(f"invalid fresh synthesis marker at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreshStartError(f"fresh synthesis marker at {path} must be an object")
    if value.get("schema_version") != MARKER_SCHEMA_VERSION:
        raise FreshStartError(
            f"unsupported fresh synthesis marker schema at {path}: "
            f"{value.get('schema_version')!r}"
        )
    fresh_start_id = value.get("fresh_start_id")
    if not isinstance(fresh_start_id, str) or len(fresh_start_id) < 16:
        raise FreshStartError(f"fresh synthesis marker at {path} has no valid identity")
    return value


def require_fresh_start_marker(
    training_specs_dir: Path,
    *,
    expected_fresh_start_id: Optional[str] = None,
    expected_holdout_identity: Optional[Mapping[str, str]] = None,
    verify_preserved_inputs: bool = True,
    allow_resume_artifacts: bool = False,
) -> Mapping[str, Any]:
    """Validate the marker and reject every stale-cache reuse opportunity.

    Call this before loading any pair checkpoint, generation journal, terminal
    rejection, provider cache, or in-progress pair output.
    """

    marker = load_fresh_start_marker(training_specs_dir)
    actual_id = marker["fresh_start_id"]
    if expected_fresh_start_id is not None and actual_id != expected_fresh_start_id:
        raise FreshStartError(
            f"fresh synthesis identity mismatch: expected {expected_fresh_start_id}, "
            f"marker contains {actual_id}"
        )
    if expected_holdout_identity is not None:
        actual_holdout = marker.get("synthesis_holdout_identity")
        if actual_holdout != dict(expected_holdout_identity):
            raise FreshStartError(
                "fresh synthesis marker holdout identity mismatch; copied, "
                "mutated, or newly generated split state cannot be reused"
            )
    manifest_path_raw = marker.get("archive_manifest_path")
    manifest_sha = marker.get("archive_manifest_sha256")
    manifest_path = (
        Path(manifest_path_raw) if isinstance(manifest_path_raw, str) else None
    )
    if (
        manifest_path is None
        or not isinstance(manifest_sha, str)
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != manifest_sha
    ):
        raise FreshStartError(
            "fresh synthesis evidence manifest is missing or its hash changed"
        )
    stale = sorted(
        name
        for name in SYNTHESIS_ARTIFACT_NAMES
        if (training_specs_dir / name).exists()
    )
    if stale and not allow_resume_artifacts:
        raise FreshStartError(
            "stale synthesis artifacts appeared after fresh-start preparation: "
            + ", ".join(stale)
        )
    if verify_preserved_inputs:
        hashes = marker.get("preserved_input_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise FreshStartError("fresh synthesis marker has no preserved-input hashes")
        mismatches: list[str] = []
        for raw_path, expected in hashes.items():
            path = Path(raw_path)
            if not path.is_file() or sha256_file(path) != expected:
                mismatches.append(raw_path)
        if mismatches:
            raise FreshStartError(
                "preserved synthesis inputs changed after reset: "
                + ", ".join(sorted(mismatches))
            )
    return marker


def bind_fresh_start_run_contract(
    training_specs_dir: Path,
    *,
    expected_fresh_start_id: str,
    run_contract_sha256: str,
    resume_artifacts_exist: bool,
    contract_components: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Seal one fresh pass to the exact synthesis runtime contract.

    The first invocation records the contract before provider dispatch.  A
    later invocation must present the same value.  A marker without a contract
    is never upgraded when resume artifacts already exist: doing so would bless
    outputs produced by an unknown implementation.
    """

    marker_path = training_specs_dir / MARKER_NAME
    marker = dict(load_fresh_start_marker(training_specs_dir))
    if marker["fresh_start_id"] != expected_fresh_start_id:
        raise FreshStartError(
            "fresh synthesis identity changed before run-contract binding"
        )
    if (
        not isinstance(run_contract_sha256, str)
        or len(run_contract_sha256) != 64
        or any(char not in "0123456789abcdef" for char in run_contract_sha256)
    ):
        raise FreshStartError("synthesis run contract must be a lowercase SHA-256")
    stored = marker.get(RUN_CONTRACT_FIELD)
    if stored is not None:
        if stored != run_contract_sha256:
            # A generation-contract change mid-run means the code that would
            # produce the remaining rows is not the code that produced the
            # accepted ones. Resuming past this point mixes two implementations
            # into one training corpus, and nothing downstream can separate them
            # again — the pairs become a model.
            #
            # This deliberately does NOT rebind. An earlier attempt appended the
            # old and new contracts to a history list and continued, on the
            # theory that per-row fingerprints would re-run whatever the change
            # touched. That reasoning only holds if the fingerprint covers every
            # behavior-bearing input, which is exactly the assumption a contract
            # change calls into question.
            #
            # The cascade this guard was blamed for came from somewhere else:
            # VERDICT-policy files were folded into the same tuple, so editing a
            # validator re-keyed generation and forced a restart (the run that
            # motivated this carries 30 archive-and-restart directories, three
            # named *-contract-supplement). That is fixed at the source by
            # splitting GENERATION_CONTRACT_FILES from VERDICT_POLICY_FILES, so
            # reaching here now means a real generation change — rare, and worth
            # stopping for.
            changed_detail = ""
            previous = marker.get(RUN_CONTRACT_COMPONENTS_FIELD)
            if isinstance(previous, Mapping) and contract_components is not None:
                changed = sorted(
                    key for key in set(previous) | set(contract_components)
                    if previous.get(key) != contract_components.get(key)
                )
                if changed:
                    # Name the component that moved: without it the operator can
                    # only diagnose by archive-and-restart, which is how the
                    # supplement directories accumulated in the first place.
                    changed_detail = f"; changed components: {', '.join(changed)}"
            raise FreshStartError(
                "synthesis generation contract changed mid-run "
                f"({stored} -> {run_contract_sha256}){changed_detail}; "
                "resuming would mix rows from two implementations into one "
                "corpus — archive and restart from zero"
            )
        return marker
    if resume_artifacts_exist:
        raise FreshStartError(
            "fresh synthesis marker predates run-contract binding but resume "
            "artifacts already exist; archive and restart from zero"
        )

    marker[RUN_CONTRACT_FIELD] = run_contract_sha256
    if contract_components is not None:
        marker[RUN_CONTRACT_COMPONENTS_FIELD] = dict(contract_components)
        encoded_components = json.dumps(
            dict(contract_components),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        marker[RUN_CONTRACT_COMPONENTS_SHA_FIELD] = hashlib.sha256(
            encoded_components
        ).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MARKER_NAME}.", dir=training_specs_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)
    finally:
        temporary.unlink(missing_ok=True)
    return marker


__all__ = [
    "FreshStartError",
    "MARKER_NAME",
    "MARKER_SCHEMA_VERSION",
    "RUN_CONTRACT_FIELD",
    "RUN_CONTRACT_COMPONENTS_FIELD",
    "RUN_CONTRACT_COMPONENTS_SHA_FIELD",
    "SYNTHESIS_ARTIFACT_NAMES",
    "fingerprint_files",
    "bind_fresh_start_run_contract",
    "load_fresh_start_marker",
    "require_fresh_start_marker",
    "sha256_file",
]
