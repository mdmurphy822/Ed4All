"""Phase-3 tests for the gold semantic-component catalog loader.

Mirrors the ``lib/generation/block_catalog.py`` posture tests: defensive-copy
isolation, fail-loud on a missing/malformed YAML, and the SoT accessors
(``valid_components`` / ``semantic_component_for`` / ``label_for``). The one
divergence under test is PACKAGE-RELATIVE pathing with NO ``lib.*`` /
``SEMANTIK_HOME`` coupling (the loader reads in SemantiK's out-of-process
bridge venv).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dart_semantic.assembler import semantic_catalog as sc

# The full authored token set (16 components — keep in sync with the YAML).
_EXPECTED_COMPONENTS = frozenset(
    {
        "section",
        "worked_example",
        "definition_region",
        "callout_note",
        "callout_tip",
        "callout_warning",
        "callout_danger",
        "figure",
        "footnote",
        "references",
        "abstract",
        "toc",
        "exercise",
        "objectives",
        "pullquote",
        "code_region",
    }
)


def test_load_returns_defensive_copy() -> None:
    """Mutating an entry in the returned list does not corrupt the cache."""
    first = sc.load_semantic_catalog()
    assert first, "catalog should be non-empty"
    first[0]["component"] = "CLOBBERED"
    first[0]["label"] = "CLOBBERED"
    first.append({"component": "injected"})

    second = sc.load_semantic_catalog()
    assert second[0]["component"] != "CLOBBERED"
    assert second[0]["label"] != "CLOBBERED"
    assert all(entry.get("component") != "injected" for entry in second)


def test_valid_components_is_frozenset() -> None:
    """Returns a frozenset carrying the full authored token set."""
    components = sc.valid_components()
    assert isinstance(components, frozenset)
    # Spot members called out by the spec.
    assert "worked_example" in components
    assert "definition_region" in components
    assert "callout_warning" in components
    # The full 16-token contract.
    assert components == _EXPECTED_COMPONENTS


def test_loader_resolves_package_relative(tmp_path, monkeypatch) -> None:
    """(a) source has NO lib.paths/SEMANTIK_HOME coupling; (b) cwd-independent."""
    source = Path(sc.__file__).read_text(encoding="utf-8")
    for forbidden in ("SEMANTIK_HOME", "semantik_home", "lib.paths", "get_project_root"):
        assert forbidden not in source, f"loader must not reference {forbidden!r}"

    # The resolved path lives under the package config/ dir.
    resolved = sc.semantic_catalog_path()
    assert resolved.name == "semantic_components.yaml"
    assert resolved.parent.name == "config"
    assert resolved.parent.parent.name == "dart_semantic"

    # A load from a DIFFERENT cwd still resolves (package-relative, not cwd-relative).
    monkeypatch.chdir(tmp_path)
    assert sc.load_semantic_catalog(), "package-relative load must work from any cwd"


def test_semantic_component_for_known_kind() -> None:
    """`code_block` resolves to the PRIMARY (first-in-catalog) component.

    `code_block` maps from both `worked_example` and `code_region`;
    catalog order makes `worked_example` the primary. A kind no component
    maps from returns None.
    """
    assert sc.semantic_component_for("code_block") == "worked_example"
    assert sc.semantic_component_for("figure") == "figure"
    assert sc.semantic_component_for("__not_a_real_kind__") is None


def test_label_for_known_component() -> None:
    """`worked_example` -> its authored label; unknown token -> None."""
    assert sc.label_for("worked_example") == "Example"
    assert sc.label_for("definition_region") == "Definition"
    assert sc.label_for("__not_a_real_component__") is None


def test_component_for_pedagogy_class() -> None:
    """The REVERSE of ``reconciles_with``: a clean-pass pedagogy CSS class ->
    its gold ``component`` token; an unmapped class -> None (no fabrication)."""
    assert sc.component_for_pedagogy_class("pedagogy-example") == "worked_example"
    assert sc.component_for_pedagogy_class("pedagogy-objectives") == "objectives"
    assert sc.component_for_pedagogy_class("pedagogy-practice") == "exercise"
    # A class no component declares ``reconciles_with`` -> None.
    assert sc.component_for_pedagogy_class("pedagogy-solution") is None
    assert sc.component_for_pedagogy_class("__not_a_real_class__") is None
    assert sc.component_for_pedagogy_class("") is None


def test_fail_loud_on_missing_yaml(monkeypatch) -> None:
    """An absent YAML raises (not a silent empty list)."""
    sc._load_raw.cache_clear()
    monkeypatch.setattr(
        sc,
        "semantic_catalog_path",
        lambda: Path(os.sep) / "no" / "such" / "semantic_components.yaml",
    )
    try:
        with pytest.raises(FileNotFoundError):
            sc._load_raw()
    finally:
        sc._load_raw.cache_clear()
