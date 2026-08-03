"""Keep MCP decision alternatives aligned with the canonical event schema."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMITTER_MODULES = (
    PROJECT_ROOT / "MCP/core/workflow_runner.py",
    PROJECT_ROOT / "MCP/tools/pipeline_tools.py",
)


def test_literal_mcp_decision_alternatives_are_objects() -> None:
    """Every inline MCP alternative is emitted as an object, not a string."""
    failures: list[str] = []
    for path in EMITTER_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keyword = next(
                (
                    item
                    for item in node.keywords
                    if item.arg == "alternatives_considered"
                    and isinstance(item.value, ast.List)
                ),
                None,
            )
            if keyword is None:
                continue
            if any(not isinstance(item, ast.Dict) for item in keyword.value.elts):
                failures.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert not failures, f"bare MCP decision alternatives at {failures}"


def test_curie_minting_dynamic_alternatives_are_objects() -> None:
    """The CURIE emitter's constructed alternatives stay object-shaped."""
    path = PROJECT_ROOT / "MCP/tools/pipeline_tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_emit_curie_minting_capture"
    )
    assignments = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "alternatives"
            for target in node.targets
        )
    ]
    appends = [
        node.args[0]
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "alternatives"
        and node.func.attr == "append"
        and node.args
    ]
    assert assignments
    assert all(
        isinstance(value, ast.List)
        and all(isinstance(item, ast.Dict) for item in value.elts)
        for value in assignments
    )
    assert appends and all(isinstance(item, ast.Dict) for item in appends)
