"""Compatibility contracts for reorganized Trainforge generator packages."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


_STAGED_ALIASES = (
    ("staged_synthesis_micro", "staged.micro"),
    ("staged_synthesis_provider", "staged.provider"),
    ("synthesis_window_contract", "staged.window_contract"),
    ("objective_execution_contract", "staged.objective_contract"),
)


@pytest.mark.parametrize(("legacy_suffix", "canonical_suffix"), _STAGED_ALIASES)
def test_staged_aliases_resolve_to_canonical_module(
    legacy_suffix: str,
    canonical_suffix: str,
) -> None:
    legacy_name = f"Trainforge.generators.{legacy_suffix}"
    canonical_name = f"Trainforge.generators.{canonical_suffix}"
    canonical = importlib.import_module(canonical_name)
    sys.modules.pop(legacy_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PendingDeprecationWarning)
        legacy = importlib.import_module(legacy_name)

    assert legacy is canonical
    assert any(item.category is PendingDeprecationWarning for item in caught)
