"""Keep production decision alternatives aligned with the event schema."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import get_args, get_type_hints

from lib.decision_capture import DecisionAlternative, DecisionCapture
from lib.streaming_capture import StreamingDecision, StreamingDecisionCapture
from Trainforge.generators.pairs.instruction import InstructionSynthesisResult
from Trainforge.generators.pairs.preference import PreferenceSynthesisResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALTERNATIVE_KEYS = {"option", "score", "reason_rejected"}

# Non-literal alternatives must have a focused runtime test that validates the
# object-producing path. Keys use stable function and expression identities so
# ordinary line movement does not require registry churn.
DYNAMIC_ALTERNATIVES_RUNTIME_TESTS = {
    ("Courseforge/generators/outline/_outline_provider.py", "_emit_per_call_decision", "alternatives"):
        "Courseforge/generators/tests/test_outline_provider.py::test_outline_success_emits_decision_event",
    ("Courseforge/router/router.py", "_emit_self_consistency_decision", "alternatives"):
        "Courseforge/router/tests/test_self_consistency.py::test_decision_event_includes_winning_candidate_index",
    ("Courseforge/router/router.py", "_emit_router_decision", "alternatives"):
        "Courseforge/router/tests/test_router.py::test_route_emits_block_outline_call_decision_event",
    ("MCP/tools/pipeline_tools.py", "_emit_curie_minting_capture", "alternatives"):
        "MCP/tests/test_outline_curie_sibling_fallback.py::test_capture_fires_with_sibling_fallback_marker",
    ("MCP/tools/pipeline_tools.py", "_plan_course_structure", "_ex_alternatives or None"):
        "MCP/tests/test_plan_course_structure.py::test_library_exemplar_capture_uses_canonical_alternatives",
    ("Trainforge/generators/assessment/generator.py", "_generate_question", "alternatives if alternatives else None"):
        "Trainforge/tests/test_assessment_generator_capture_wiring.py::test_objective_capture_uses_canonical_alternative_key",
    ("Trainforge/synthesis/synthesize_training.py", "run_synthesis", "inst_result.alternatives or None"):
        "Trainforge/tests/test_training_synthesis.py::test_decision_capture_id_resolves_for_every_pair",
    ("Trainforge/synthesis/synthesize_training.py", "run_synthesis", "pref_result.alternatives or None"):
        "Trainforge/tests/test_training_synthesis.py::test_decision_capture_id_resolves_for_every_pair",
    ("lib/objectives/block_alignment.py", "align_blocks_to_objectives", "alternatives or None"):
        "lib/objectives/tests/test_block_alignment.py::test_alignment_adds_missed_objective_ref",
    ("lib/validators/abcd_objective.py", "_emit_decision", "alternatives or None"):
        "lib/validators/tests/test_abcd_objective.py::test_emit_decision_forwards_canonical_alternatives",
    ("lib/validators/shacl_result_enricher.py", "_emit_decision_capture", "alternatives or None"):
        "lib/validators/tests/test_shacl_result_enricher.py::test_decision_capture_one_event_per_violation",
}


def _production_python_paths() -> list[Path]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.py"], cwd=PROJECT_ROOT, text=True
    ).splitlines()
    paths = []
    for relative in tracked:
        path = Path(relative)
        if path.name.startswith("test_") or {"tests", "test", "regression"} & set(path.parts):
            continue
        paths.append(PROJECT_ROOT / path)
    return paths


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "<module>"


def test_production_decision_alternatives_follow_canonical_shape() -> None:
    """Literal alternatives are schema-shaped; dynamic emitters are registered."""
    failures: list[str] = []
    observed_dynamic: set[tuple[str, str, str]] = set()

    for path in _production_python_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in {
                "log_decision", "_emit_decision"
            }:
                continue
            keyword = next(
                (item for item in node.keywords if item.arg == "alternatives_considered"),
                None,
            )
            if keyword is None:
                continue
            location = f"{relative}:{node.lineno}"
            if not isinstance(keyword.value, ast.List):
                identity = (
                    relative,
                    _enclosing_function(node, parents),
                    ast.unparse(keyword.value),
                )
                observed_dynamic.add(identity)
                if identity not in DYNAMIC_ALTERNATIVES_RUNTIME_TESTS:
                    failures.append(f"{location}: unregistered dynamic alternatives {identity[2]!r}")
                continue
            for alternative in keyword.value.elts:
                if not isinstance(alternative, ast.Dict):
                    failures.append(f"{location}: alternative must be an object")
                    continue
                keys = {
                    key.value
                    for key in alternative.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if len(keys) != len(alternative.keys):
                    failures.append(f"{location}: alternative keys must be string literals")
                if "option" not in keys:
                    failures.append(f"{location}: alternative is missing option")
                unexpected = keys - ALTERNATIVE_KEYS
                if unexpected:
                    failures.append(f"{location}: unsupported keys {sorted(unexpected)}")

    stale = set(DYNAMIC_ALTERNATIVES_RUNTIME_TESTS) - observed_dynamic
    failures.extend(f"stale dynamic registry entry {item}" for item in sorted(stale))
    assert not failures, "\n".join(failures)


def test_dynamic_registry_references_existing_test_nodes() -> None:
    """Every dynamic exception names a concrete runtime regression test."""
    failures = []
    for node_id in sorted(set(DYNAMIC_ALTERNATIVES_RUNTIME_TESTS.values())):
        relative, test_name = node_id.split("::", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file():
            failures.append(f"missing test file: {node_id}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if test_name not in names:
            failures.append(f"missing test function: {node_id}")
    assert not failures, "\n".join(failures)


def test_decision_alternative_type_hints_are_consistent() -> None:
    """Core, streaming, and pair results retain the canonical alternative type."""
    core = get_type_hints(DecisionAlternative)
    assert core == {"option": str, "score": float, "reason_rejected": str}
    for method in (
        DecisionCapture.log_decision,
        StreamingDecisionCapture.log_decision,
    ):
        annotation = get_type_hints(method)["alternatives_considered"]
        collection = next(
            member for member in get_args(annotation) if get_args(member)
        )
        assert get_args(collection) == (DecisionAlternative,)
    for result_type, field_name in (
        (StreamingDecision, "alternatives_considered"),
        (InstructionSynthesisResult, "alternatives"),
        (PreferenceSynthesisResult, "alternatives"),
    ):
        annotation = get_type_hints(result_type)[field_name]
        assert get_args(annotation) == (DecisionAlternative,)
