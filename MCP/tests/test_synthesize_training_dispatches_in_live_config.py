"""Wave 33 Bug A — synthesize_training schema matches dispatch shape.

Pre-Wave-33 ``MCP/core/tool_schemas.py::TOOL_SCHEMAS["synthesize_training"]``
listed ``corpus_dir`` as a required parameter and did NOT list
``assessments_path`` / ``chunks_path`` as aliases. The live workflow
runner dispatches ``training_synthesis`` with the shape

    course_code=..., assessments_path=..., chunks_path=..., provider=..., seed=...

(see ``config/workflows.yaml::training_synthesis.inputs_from``), so
``param_mapper.map_task_to_tool_params`` raised

    ParameterMappingError: Missing required parameters for
    synthesize_training: ['corpus_dir']. Received params: [
    'id','course_code','assessments_path','chunks_path','provider','seed']

on every real run, tripped the poison-pill detector, and the phase
never produced ``instruction_pairs.jsonl`` / ``preference_pairs.jsonl``.

The fix reshapes the schema so ``course_code`` is the only required
kwarg (the one the tool genuinely can't derive) and
``corpus_dir`` / ``assessments_path`` / ``chunks_path`` /
``trainforge_dir`` are all recognised as optional pass-through kwargs.
The tool function itself derives ``corpus_dir`` from whichever path
the dispatcher routes (see
``MCP/tools/pipeline_tools.py::_synthesize_training``).

These tests lock the dispatch-shape contract so a future regression
(dropping one of the paths, or re-requiring ``corpus_dir``) is caught
at commit-time rather than on the next live run.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

from MCP.core.param_mapper import TaskParameterMapper
from MCP.core.tool_schemas import (
    get_optional_params,
    get_required_params,
    validate_tool_params,
)


def test_validate_tool_params_accepts_live_dispatch_shape(tmp_path: Path):
    """The exact kwarg shape the workflow runner builds must validate.

    Mirrors ``training_synthesis.inputs_from`` in workflows.yaml — the
    dispatcher produces ``course_code`` + ``assessments_path`` +
    ``chunks_path``, plus the schema defaults for ``provider`` + ``seed``.
    Pre-Wave-33 this shape raised ``Missing required parameters:
    ['corpus_dir']``.
    """
    # Exactly what ``_route_params`` produces for the training_synthesis
    # phase after resolving inputs_from against trainforge_assessment's
    # outputs.
    dispatcher_kwargs = {
        "id": "T_training_synthesis_001",
        "course_code": "TESTCOURSE_101",
        "assessments_path": str(tmp_path / "trainforge" / "assessments.json"),
        "chunks_path": str(tmp_path / "trainforge" / "corpus" / "chunks.jsonl"),
        "provider": "mock",
        "seed": 7,
    }

    is_valid, missing = validate_tool_params(
        "synthesize_training", dispatcher_kwargs,
    )
    assert is_valid is True, (
        f"Live dispatch shape must satisfy the schema; got missing={missing}. "
        "Pre-Wave-33 this failed with ['corpus_dir'] because the schema "
        "required corpus_dir but the dispatcher never routes it."
    )
    assert missing == []

    # TaskParameterMapper must also accept the same shape — this is the
    # actual code path _invoke_tool takes (not just validate_tool_params).
    mapper = TaskParameterMapper(strict=False)
    mapped = mapper.map_task_to_tool_params(
        {"params": dispatcher_kwargs}, "synthesize_training",
    )
    # course_code survives; optional passthrough kwargs survive; the
    # mapper never fabricates a corpus_dir.
    assert mapped["course_code"] == "TESTCOURSE_101"
    assert "assessments_path" in mapped
    assert "chunks_path" in mapped
    assert mapped["provider"] == "mock"
    assert mapped["seed"] == 7


def test_schema_contract_reflects_dispatch_reality():
    """Required params: only ``course_code``. Everything else is optional.

    Lock the post-fix shape so a regression that re-adds ``corpus_dir``
    to required (breaking the live dispatch) is caught immediately.
    """
    required = get_required_params("synthesize_training")
    assert required == ["course_code"], (
        f"Required kwargs drifted. Wave 33 Bug A reduced required to just "
        f"course_code (the one kwarg the tool can't derive). Got: {required}"
    )

    optional = get_optional_params("synthesize_training")
    # All four path shapes the tool accepts must be surfaced as optional
    # so the mapper passes them through instead of dropping them (strict
    # mode) or renaming them (if they were in param_mapping).
    assert "corpus_dir" in optional
    assert "trainforge_dir" in optional
    assert "assessments_path" in optional
    assert "chunks_path" in optional


def test_dispatch_emits_pair_files_via_chunks_path(tmp_path: Path):
    """End-to-end: a registry dispatch with the dispatcher's exact
    kwarg shape (``assessments_path`` + ``chunks_path``) must materialise
    ``instruction_pairs.jsonl`` + ``preference_pairs.jsonl`` on disk.

    Pre-Wave-33 this path was unreachable — ``param_mapper`` raised
    before the tool function ran, so the training_synthesis phase
    never produced pair files even though the underlying
    ``run_synthesis`` function worked.
    """
    # Build a minimum-viable Trainforge corpus.
    corpus_dir = tmp_path / "trainforge"
    (corpus_dir / "corpus").mkdir(parents=True)
    (corpus_dir / "training_specs").mkdir(parents=True)
    chunk = {
        "id": "chunk_dispatch_test_01",
        "course_id": "TESTCOURSE_101",
        "section_id": "sec_01",
        "content": (
            "Evidence-based practice integrates research findings with "
            "clinical expertise and patient values. The three pillars form "
            "the foundation for clinical decision making in nursing."
        ),
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type_label": "explanation",
        "key_terms": [
            {"term": "evidence-based practice", "definition": "a decision-making framework"},
        ],
    }
    chunks_path = corpus_dir / "corpus" / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    # Mirror ``trainforge_assessment`` output layout: assessments.json
    # lives at the trainforge root (see
    # ``pipeline_tools._generate_assessments`` L3061). The tool derives
    # corpus_dir from ``assessments_path.parent`` so the file location
    # is load-bearing for the dispatch-shape contract.
    assessments_path = corpus_dir / "assessments.json"
    assessments_path.write_text(json.dumps({"questions": []}), encoding="utf-8")

    # Invoke the registry variant with the dispatcher's exact kwarg
    # shape. No explicit corpus_dir — the tool must derive it from
    # chunks_path (grandparent).
    pt = importlib.import_module("MCP.tools.pipeline_tools")
    registry = pt._build_tool_registry()
    assert "synthesize_training" in registry

    result_json = asyncio.run(registry["synthesize_training"](
        course_code="TESTCOURSE_101",
        assessments_path=str(assessments_path),
        chunks_path=str(chunks_path),
        provider="mock",
        seed=7,
    ))
    result = json.loads(result_json)

    assert result.get("success") is True, (
        f"Dispatch-shape call must succeed; envelope={result}. "
        "If you see `error: synthesize_training requires corpus_dir...` "
        "the tool's derivation logic regressed."
    )
    instr_path = Path(result["instruction_pairs_path"])
    pref_path = Path(result["preference_pairs_path"])
    assert instr_path.exists()
    assert pref_path.exists()
    # Mock provider emits at least one pair per eligible chunk.
    assert instr_path.stat().st_size > 0
    # The derived corpus_dir must match chunks_path.parent.parent.
    assert Path(result["corpus_dir"]) == corpus_dir
