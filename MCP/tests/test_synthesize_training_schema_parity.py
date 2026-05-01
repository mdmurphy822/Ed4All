"""Wave 32 Deliverable A — synthesize_training schema parity.

Pre-Wave-32 the Wave 30 PR wired ``synthesize_training`` into two of
the three required tool-wiring locations (``pipeline_tools._build_tool_registry``
and ``executor.AGENT_TOOL_MAPPING``) but missed the third:
``MCP/core/tool_schemas.py::TOOL_SCHEMAS``. That gap meant:

* ``param_mapper.get_tool_schema("synthesize_training")`` returned ``None``.
* Any dispatch via :class:`ParamMapper` raised
  ``ParameterMappingError("Unknown tool: synthesize_training")``.
* The poison-pill detector tripped on the third retry, the
  ``training_synthesis`` phase never produced
  ``instruction_pairs.jsonl`` / ``preference_pairs.jsonl``, and
  ``ed4all export-training ... --format dpo`` had nothing real to
  export.

These tests lock the schema-registration invariant so the third
wiring location cannot regress silently.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, Dict

from MCP.core.tool_schemas import (
    TOOL_SCHEMAS,
    get_param_mapping,
    get_required_params,
    get_tool_schema,
    validate_tool_params,
)


def test_synthesize_training_has_schema():
    """Schema must be registered in TOOL_SCHEMAS (third wiring location)."""
    schema = get_tool_schema("synthesize_training")
    assert schema is not None, (
        "synthesize_training missing from TOOL_SCHEMAS — this is the Wave 30 "
        "gap that caused every training_synthesis dispatch to raise "
        "ParameterMappingError and trip the poison-pill detector."
    )
    # Wave 33 Bug A: ``course_code`` is the only required kwarg the
    # tool function can't derive on its own. ``corpus_dir`` /
    # ``trainforge_dir`` / ``assessments_path`` / ``chunks_path`` are
    # all optional pass-through kwargs; the tool function picks
    # whichever is given and derives the corpus directory internally.
    # See the schema header comment in tool_schemas.py for rationale.
    required = get_required_params("synthesize_training")
    assert "course_code" in required
    optional = schema.get("optional", [])
    assert "corpus_dir" in optional
    assert "assessments_path" in optional
    assert "chunks_path" in optional


def test_synthesize_training_param_aliases_cover_pipeline_shape():
    """Registry variant accepts a wider alias surface.

    ``_synthesize_training`` in ``pipeline_tools.py`` maps
    ``trainforge_dir`` / ``output_dir`` / ``course_name`` / ``course_id``
    onto the canonical signature — the schema's param_mapping must
    register those aliases so ``param_mapper`` doesn't reject the
    aliased kwargs as unknown params.
    """
    mapping = get_param_mapping("synthesize_training")
    # Corpus dir aliases — kwargs the registry variant accepts.
    assert mapping.get("trainforge_dir") == "corpus_dir"
    assert mapping.get("output_dir") == "corpus_dir"
    # Course code aliases — mirror the registry variant's decision-capture
    # resolution order (course_code / course_name / course_id).
    assert mapping.get("course_name") == "course_code"
    assert mapping.get("course_id") == "course_code"


def test_synthesize_training_passes_validation_with_aliased_inputs(tmp_path: Path):
    """Dispatch-shape smoke: ``validate_tool_params`` resolves aliases.

    Pre-Wave-32 this raised ``ParameterMappingError("Unknown tool")`` on
    the schema lookup; now the validator must accept the aliased
    ``trainforge_dir`` + ``course_name`` kwargs as satisfying the
    ``corpus_dir`` + ``course_code`` contract.
    """
    # Alias-shape kwargs the registry variant receives from the
    # workflow runner (see PHASE_PARAM_ROUTING for training_synthesis).
    params: Dict[str, Any] = {
        "trainforge_dir": str(tmp_path / "workspace" / "trainforge"),
        "course_name": "TESTCOURSE_101",
    }
    is_valid, missing = validate_tool_params("synthesize_training", params)
    assert is_valid is True, (
        f"Aliased inputs should satisfy the schema; got missing={missing}"
    )
    assert missing == []


def test_training_synthesis_phase_emits_instruction_pairs(tmp_path: Path):
    """End-to-end: a training_synthesis dispatch emits pair files on disk.

    Builds the minimum viable Trainforge corpus layout (a single
    eligible ``chunks.jsonl`` row) and invokes the registry-variant
    ``synthesize_training`` via the pipeline's tool registry. On
    success the corpus dir contains
    ``training_specs/instruction_pairs.jsonl`` (mock provider emits
    at least one instruction pair) and the envelope carries
    ``success: true`` + both output paths.

    This test does NOT go through ``param_mapper`` — it exercises the
    registry variant directly to guarantee the end-to-end artifact
    emission. The schema-registration tests above already cover the
    param_mapper dispatch path.
    """
    # Build a minimum viable chunks.jsonl (one eligible chunk is enough
    # for the mock provider to emit instruction pairs).
    corpus_dir = tmp_path / "trainforge"
    (corpus_dir / "corpus").mkdir(parents=True)
    chunk = {
        "id": "chunk_test_0001",
        "course_id": "TESTCOURSE_101",
        "section_id": "sec_01",
        "content": (
            "Knowledge graphs organise information as nodes and edges. "
            "Nodes represent entities; edges represent typed relations "
            "between them. The semantic relations capture how concepts "
            "connect in the domain."
        ),
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type_label": "explanation",
        "key_terms": [{"term": "knowledge graph", "definition": "a structured representation"}],
    }
    chunks_path = corpus_dir / "corpus" / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    # Invoke the registry variant (same entrypoint the workflow runner
    # dispatches via AGENT_TOOL_MAPPING["training-synthesizer"]).
    pt = importlib.import_module("MCP.tools.pipeline_tools")
    registry = pt._build_tool_registry()
    assert "synthesize_training" in registry, (
        "synthesize_training missing from tool registry — first wiring "
        "location regressed."
    )

    result_json = asyncio.run(registry["synthesize_training"](
        corpus_dir=str(corpus_dir),
        course_code="TESTCOURSE_101",
        provider="mock",
        seed=7,
    ))
    result = json.loads(result_json)

    assert result.get("success") is True, (
        f"synthesize_training failed — poison-pill regression? envelope={result}"
    )
    instruction_pairs_path = Path(result["instruction_pairs_path"])
    assert instruction_pairs_path.exists()
    # Mock provider emits at least one instruction pair per eligible chunk.
    assert instruction_pairs_path.stat().st_size > 0
    # The preference-pairs file is also written (may be empty when the
    # mock provider can't synthesise a rejected arm, which is fine —
    # what matters is the phase dispatched cleanly without tripping
    # poison-pill).
    preference_pairs_path = Path(result["preference_pairs_path"])
    assert preference_pairs_path.exists()


def test_synthesize_training_schema_includes_wave_124_127_generator_kwargs():
    """Wave 129: schema must list the deterministic-generator pass-throughs.

    Pre-Wave-129 the ``synthesize_training`` schema (third wiring location)
    listed only ``corpus_dir / trainforge_dir / assessments_path /
    chunks_path / provider / seed`` as optional. The CLI had carried
    ``--with-violation-detection / --with-abstention /
    --with-schema-translation / --with-kg-metadata`` since Waves 124-125,
    but the MCP tool surface + the workflow-phase dispatch couldn't
    forward those flags — ``param_mapper`` would silently drop them as
    unknown params and the deterministic generators stayed off.

    This test pins the schema to the run_synthesis() kwarg shape at
    ``Trainforge/synthesize_training.py:677-685`` so a future contributor
    adding a new generator kwarg gets a failing test if they forget the
    third wiring location.
    """
    schema = get_tool_schema("synthesize_training")
    assert schema is not None
    optional = schema.get("optional", [])
    expected_kwargs = [
        "with_kg_metadata",
        "kg_metadata_max_pairs",
        "with_violation_detection",
        "violation_detection_max_pairs",
        "with_abstention",
        "abstention_max_pairs",
        "with_schema_translation",
        "schema_translation_max_pairs",
    ]
    for kw in expected_kwargs:
        assert kw in optional, (
            f"schema missing Wave 124-127 generator kwarg {kw!r} — Wave 129 "
            f"third-location wiring regressed."
        )

    # Defaults must mirror run_synthesis(): on-flags default False,
    # max-pairs caps default to the run_synthesis() values, and
    # violation_detection_max_pairs is intentionally absent from defaults
    # (its run_synthesis() default is None = unlimited; the schema treats
    # it as a pure pass-through).
    defaults = schema.get("defaults", {})
    assert defaults.get("with_kg_metadata") is False
    assert defaults.get("kg_metadata_max_pairs") == 2000
    assert defaults.get("with_violation_detection") is False
    assert defaults.get("with_abstention") is False
    assert defaults.get("abstention_max_pairs") == 1000
    assert defaults.get("with_schema_translation") is False
    assert defaults.get("schema_translation_max_pairs") == 50


def test_synthesize_training_registry_forwards_wave_124_127_kwargs(tmp_path: Path):
    """Wave 129: registry variant must forward generator kwargs to run_synthesis.

    End-to-end: dispatch the registry variant with
    ``with_violation_detection=True`` + an explicit cap, and assert
    ``run_synthesis`` saw the cap. We monkeypatch ``run_synthesis`` so
    the test stays fast (no real synthesis needed); the test's job is
    to lock the kwarg-forwarding contract, not re-run the synthesizer.
    """
    # Stub ``run_synthesis`` to capture the kwargs it receives.
    captured: Dict[str, Any] = {}

    def fake_run_synthesis(**kw):
        captured.update(kw)

        class _Stats:
            instruction_pairs_emitted = 0
            preference_pairs_emitted = 0
            chunks_eligible = 0
            chunks_total = 0

            def as_dict(self):
                return {}

        return _Stats()

    # Build a minimum viable corpus so the early-return / chunks-missing
    # branch doesn't fire before run_synthesis is reached.
    corpus_dir = tmp_path / "trainforge"
    (corpus_dir / "corpus").mkdir(parents=True)
    chunk = {
        "id": "chunk_test_0001",
        "course_id": "TESTCOURSE_101",
        "section_id": "sec_01",
        "content": (
            "Knowledge graphs organise information as nodes and edges."
        ),
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type_label": "explanation",
        "key_terms": [{"term": "kg", "definition": "graph"}],
    }
    (corpus_dir / "corpus" / "chunks.jsonl").write_text(
        json.dumps(chunk) + "\n", encoding="utf-8"
    )

    import Trainforge.synthesize_training as st_mod

    real_run_synthesis = st_mod.run_synthesis
    st_mod.run_synthesis = fake_run_synthesis  # type: ignore[assignment]
    try:
        pt = importlib.import_module("MCP.tools.pipeline_tools")
        registry = pt._build_tool_registry()
        asyncio.run(registry["synthesize_training"](
            corpus_dir=str(corpus_dir),
            course_code="TESTCOURSE_101",
            provider="mock",
            seed=7,
            with_violation_detection=True,
            violation_detection_max_pairs=42,
            with_abstention=True,
            abstention_max_pairs=11,
            with_schema_translation=True,
            schema_translation_max_pairs=9,
        ))
    finally:
        st_mod.run_synthesis = real_run_synthesis  # type: ignore[assignment]

    # All Wave 124-127 generator kwargs reached run_synthesis with their
    # caller-supplied values (not the schema defaults).
    assert captured["with_violation_detection"] is True
    assert captured["violation_detection_max_pairs"] == 42
    assert captured["with_abstention"] is True
    assert captured["abstention_max_pairs"] == 11
    assert captured["with_schema_translation"] is True
    assert captured["schema_translation_max_pairs"] == 9


def test_synthesize_training_registered_in_three_locations():
    """Regression guard: lock the three-location wiring invariant.

    Any future contributor who adds a new first-class pipeline tool
    must wire it in all three locations:

    1. ``MCP/core/tool_schemas.py::TOOL_SCHEMAS`` — this test's focus.
    2. ``MCP/tools/pipeline_tools.py::_build_tool_registry`` — the
       dispatcher registry.
    3. ``MCP/core/executor.py::AGENT_TOOL_MAPPING`` — the agent→tool
       resolver.

    Missing any one of the three reproduces the Wave 30 → Wave 32
    poison-pill failure mode.
    """
    from MCP.core.executor import AGENT_TOOL_MAPPING

    # Location 1: schema table.
    assert "synthesize_training" in TOOL_SCHEMAS

    # Location 2: tool registry — registered by _build_tool_registry().
    pt = importlib.import_module("MCP.tools.pipeline_tools")
    registry = pt._build_tool_registry()
    assert "synthesize_training" in registry

    # Location 3: agent→tool mapping.
    assert "training-synthesizer" in AGENT_TOOL_MAPPING
    assert AGENT_TOOL_MAPPING["training-synthesizer"] == "synthesize_training"
