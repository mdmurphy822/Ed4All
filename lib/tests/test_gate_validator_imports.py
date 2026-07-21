"""Regression: every validation-gate ``validator:`` dotted path must import.

A validation gate in ``config/workflows.yaml`` names its validator as a fully
qualified dotted path (``pkg.module.ClassName``). When the underlying module is
renamed or retired, a stale dotted path does NOT fail loudly — the gate simply
degrades to a warning at runtime (``on_error: warn``) or fails closed at
dispatch, and in either case the validator that SHOULD run never runs. This
exact defect class shipped for the ``textbook_to_course`` packaging
``wcag_compliance`` gate, whose validator dotted path pointed at a module that
no longer existed and silently warn-degraded on every run.

This test walks every ``validation_gates[].validator`` dotted path declared in
``config/workflows.yaml`` and asserts ``importlib`` can resolve both the module
and the named attribute — so a dead validator reference fails CI instead of
silently never executing.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_YAML = PROJECT_ROOT / "config" / "workflows.yaml"


def _collect_validator_paths() -> dict[str, set[str]]:
    """Return ``{dotted_validator_path: {gate_id, ...}}`` for every gate.

    Recursively walks the parsed YAML so it finds ``validation_gates`` blocks at
    any nesting depth (per-workflow, per-phase) without hardcoding the shape.
    """
    cfg = yaml.safe_load(WORKFLOWS_YAML.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "validation_gates" and isinstance(value, list):
                    for gate in value:
                        if isinstance(gate, dict) and "validator" in gate:
                            path = str(gate["validator"])
                            found.setdefault(path, set()).add(
                                str(gate.get("gate_id"))
                            )
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(cfg)
    return found


_VALIDATOR_PATHS = _collect_validator_paths()


def test_workflows_yaml_declares_validator_gates() -> None:
    """Sanity: the walk actually found gates (guards against a silent no-op)."""
    assert _VALIDATOR_PATHS, "no validation_gates validator paths found in config"
    # There are many distinct validators across the four workflows; a tiny count
    # would mean the recursive walk regressed.
    assert len(_VALIDATOR_PATHS) >= 20


@pytest.mark.parametrize("dotted_path", sorted(_VALIDATOR_PATHS))
def test_gate_validator_dotted_path_imports(dotted_path: str) -> None:
    """Each gate's validator dotted path must resolve to an importable class."""
    module_path, _, attr = dotted_path.rpartition(".")
    assert module_path and attr, f"malformed validator path: {dotted_path!r}"
    gate_ids = ", ".join(sorted(_VALIDATOR_PATHS[dotted_path]))
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 — surface the real import error
        pytest.fail(
            f"validator module {module_path!r} for gate(s) [{gate_ids}] "
            f"failed to import: {type(exc).__name__}: {exc}"
        )
    assert hasattr(module, attr), (
        f"validator class {attr!r} not found in module {module_path!r} "
        f"(gate(s): [{gate_ids}])"
    )
