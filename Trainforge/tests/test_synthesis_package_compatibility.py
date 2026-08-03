"""Compatibility contracts for the Trainforge synthesis package move."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import pytest


_MOVED_MODULES = (
    "synthesis_concurrency",
    "synthesis_contract_guard",
    "synthesis_eligibility",
    "synthesis_fresh_start",
    "synthesis_holdout",
    "synthesis_journal",
    "synthesis_progress",
    "synthesis_reject_mining",
    "synthesize_training",
)


@pytest.mark.parametrize("module_name", _MOVED_MODULES)
def test_legacy_module_aliases_canonical_implementation(module_name: str) -> None:
    legacy_name = f"Trainforge.{module_name}"
    canonical_name = f"Trainforge.synthesis.{module_name}"
    canonical = importlib.import_module(canonical_name)
    sys.modules.pop(legacy_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PendingDeprecationWarning)
        legacy = importlib.import_module(legacy_name)

    assert legacy is canonical
    assert any(item.category is PendingDeprecationWarning for item in caught)


def test_legacy_module_cli_preserves_help_contract() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "Trainforge.synthesize_training", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--corpus" in result.stdout
    assert "--course-code" in result.stdout
