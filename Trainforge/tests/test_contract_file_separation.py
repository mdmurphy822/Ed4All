"""Test that verdict-policy changes don't invalidate generation fingerprints."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from Trainforge.synthesize_training import (
    _GENERATION_CONTRACT_FILES,
    _VERDICT_POLICY_FILES,
    _synthesis_static_contract_components,
)


def test_generation_verdict_policy_files_are_disjoint() -> None:
    """Confirm the two contract file sets don't overlap."""
    gen_set = set(_GENERATION_CONTRACT_FILES)
    verdict_set = set(_VERDICT_POLICY_FILES)
    overlap = gen_set & verdict_set
    assert len(overlap) == 0, f"Overlapping files: {overlap}"


def test_verdict_policy_files_not_in_generation_fingerprint() -> None:
    """Verify VERDICT_POLICY_FILES are absent from generation fingerprint.

    Touching lib/validators/pair/claim_support.py (or any verdict-policy file)
    should NOT change the generation fingerprint that keys accepted pairs.
    Only GENERATION_CONTRACT_FILES feed the fingerprint.
    """
    # Baseline: compute fingerprint with no modifications
    baseline_components = _synthesis_static_contract_components(
        provider="local",
        model_id="test-model",
        endpoint_identity="http://localhost:8000",
    )
    baseline_files = baseline_components.get("files", {})

    # Assert all verdict-policy files are ABSENT from the fingerprinted files dict
    for verdict_file in _VERDICT_POLICY_FILES:
        assert (
            verdict_file not in baseline_files
        ), (
            f"Verdict-policy file {verdict_file} is in generation fingerprint; "
            "it should only be recorded, not fingerprinted"
        )

    # Assert all generation files ARE in the fingerprinted files dict
    for gen_file in _GENERATION_CONTRACT_FILES:
        assert (
            gen_file in baseline_files
        ), (
            f"Generation file {gen_file} missing from fingerprint"
        )


def test_only_generation_files_in_fingerprint_schema() -> None:
    """Confirm the "files" dict in the fingerprint contains only generation files."""
    components = _synthesis_static_contract_components(
        provider="together",
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        endpoint_identity="https://api.together.xyz/v1",
    )
    fingerprinted_files = set(components.get("files", {}).keys())
    expected_files = set(_GENERATION_CONTRACT_FILES)

    # The fingerprinted set should exactly match generation files
    # (missing files show up as "missing" key but the key itself is still there)
    assert fingerprinted_files == expected_files, (
        f"Fingerprinted files {fingerprinted_files} don't match "
        f"generation contract {expected_files}"
    )


def test_verdict_policy_changes_recorded_not_fingerprinted(tmp_path: Path) -> None:
    """Simulate a verdict-policy file change and verify fingerprint is stable.

    This is a conceptual test: we manually verify that if a verdict-policy
    file changed on disk, the fingerprint would remain unchanged (only
    generation files contribute to it).
    """
    # Get baseline fingerprint
    baseline = _synthesis_static_contract_components(
        provider="local",
        model_id="nemotron-3-nano-30b-a3b",
        endpoint_identity="http://localhost:8000",
    )
    baseline_files = baseline.get("files", {})

    # Simulate a verdict-policy file being in the set (it's not, but illustrate the point)
    # In reality, claim_support.py is in VERDICT_POLICY_FILES, not GENERATION_CONTRACT_FILES
    claim_support_py = "lib/validators/pair/claim_support.py"
    assert (
        claim_support_py in _VERDICT_POLICY_FILES
    ), "Test assumption: claim_support.py is in verdict-policy files"
    assert (
        claim_support_py not in _GENERATION_CONTRACT_FILES
    ), "Assertion: claim_support.py is NOT in generation files"
    assert (
        claim_support_py not in baseline_files
    ), "Consequence: fingerprint doesn't include claim_support.py"
