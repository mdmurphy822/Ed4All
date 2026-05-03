"""Phase 8 ST 2 — `CourseProcessor.process` consumes upstream IMSCC chunkset.

Locks the contract that ``CourseProcessor`` short-circuits the in-process
``self._chunk_content(parsed_items)`` call when an upstream-emitted IMSCC
chunkset JSONL is supplied via the new ``imscc_chunks_path`` constructor
kwarg / ``--imscc-chunks-path`` CLI flag (the path the ``imscc_chunking``
workflow phase writes at
``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl``).

The Phase 8 ST 2 contract assertions:

1. **Skip-when-path-supplied.** ``imscc_chunks_path`` set to an existing
   JSONL file -> ``process()`` reads chunks from the upstream JSONL via
   ``_load_chunks_from_jsonl`` and does NOT invoke ``_chunk_content``.
2. **Fall-through when path is None.** Legacy / pre-Phase-8 callers
   (e.g. ``python -m Trainforge.process_course`` standalone, no
   ``imscc_chunking`` phase wired) still hit the existing in-process
   ``_chunk_content`` build path.
3. **Fall-through on missing file / malformed JSONL.** A stale phase-
   output handoff (path doesn't exist, or the file contains non-JSONL
   garbage) doesn't crash -- it falls through to the in-process
   ``_chunk_content`` build with a warning log.

Constructor-kwarg + CLI-flag plumbing tests pin the wiring contract so
callers (workflow runner, MCP tool, standalone CLI) can rely on the
field being set after construction.

Cross-link: upstream emit at
``MCP/tools/pipeline_tools.py::_run_imscc_chunking`` (Phase 7c ST 16).
Plan citation: ``plans/phase8_cleanup.md::Subtask 2``.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import CourseProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_processor(imscc_chunks_path: Any = None) -> CourseProcessor:
    """Build a bare ``CourseProcessor`` without IMSCC ingestion.

    Mirrors the ``__new__`` test pattern used in
    ``test_concept_graph_consumption.py`` (Phase 6 ST 13 precedent) so
    we can exercise ``_load_chunks_from_jsonl`` and the ``process()``
    short-circuit branch directly without a real corpus.
    """
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "PHASE8_ST2_TEST"
    proc.objectives = {}
    proc.imscc_chunks_path = (
        Path(imscc_chunks_path) if imscc_chunks_path else None
    )
    # Initialize the side-channel state ``_load_chunks_from_jsonl``
    # mutates so the helper doesn't trip on missing attributes.
    proc.stats = {"total_chunks": 0}
    return proc


def _fixture_chunks() -> List[Dict[str, Any]]:
    """Three-chunk fixture mirroring ``_run_imscc_chunking``'s emit shape.

    Matches the v4 chunk dict produced by the upstream callback at
    ``MCP/tools/pipeline_tools.py::_run_imscc_chunking::_create_chunk``
    (the canonical chunkset shape consumed by Phase 8 ST 2).
    """
    return [
        {
            "id": "phase8_st2_test_chunk_00001",
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "First chunk emitted by the upstream imscc_chunking phase.",
            "html": "<p>First chunk emitted by the upstream imscc_chunking phase.</p>",
            "follows_chunk": None,
            "source": {
                "course_id": "PHASE8_ST2_TEST",
                "module_id": "week_01",
                "module_title": "Week 1",
                "lesson_id": "intro",
                "lesson_title": "Intro",
                "resource_type": "page",
                "section_heading": "Intro",
                "position_in_module": 0,
            },
            "concept_tags": [],
            "learning_outcome_refs": [],
            "difficulty": "intermediate",
            "tokens_estimate": 12,
            "word_count": 9,
        },
        {
            "id": "phase8_st2_test_chunk_00002",
            "schema_version": "v4",
            "chunk_type": "example",
            "text": "Second chunk.",
            "html": "<p>Second chunk.</p>",
            "follows_chunk": "phase8_st2_test_chunk_00001",
            "source": {
                "course_id": "PHASE8_ST2_TEST",
                "module_id": "week_01",
                "module_title": "Week 1",
                "lesson_id": "intro",
                "lesson_title": "Intro",
                "resource_type": "page",
                "section_heading": "Example",
                "position_in_module": 1,
            },
            "concept_tags": [],
            "learning_outcome_refs": [],
            "difficulty": "intermediate",
            "tokens_estimate": 3,
            "word_count": 2,
        },
        {
            "id": "phase8_st2_test_chunk_00003",
            "schema_version": "v4",
            "chunk_type": "summary",
            "text": "Third chunk wraps it up.",
            "html": "<p>Third chunk wraps it up.</p>",
            "follows_chunk": "phase8_st2_test_chunk_00002",
            "source": {
                "course_id": "PHASE8_ST2_TEST",
                "module_id": "week_02",
                "module_title": "Week 2",
                "lesson_id": "wrap",
                "lesson_title": "Wrap",
                "resource_type": "page",
                "section_heading": "Wrap",
                "position_in_module": 0,
            },
            "concept_tags": [],
            "learning_outcome_refs": [],
            "difficulty": "intermediate",
            "tokens_estimate": 6,
            "word_count": 5,
        },
    ]


def _write_jsonl(chunks: List[Dict[str, Any]], path: Path) -> None:
    """Write chunks to a JSONL file in the same shape ``_run_imscc_chunking`` emits."""
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Phase 8 ST 2 contract tests — _load_chunks_from_jsonl helper
# ---------------------------------------------------------------------------


def test_load_chunks_from_jsonl_returns_chunks_when_file_valid(
    tmp_path: Path,
) -> None:
    """When the path points at a valid JSONL file, the helper returns the
    parsed chunk list verbatim and updates ``self.stats["total_chunks"]``.

    This is the load-bearing assertion -- Phase 8 ST 2's whole reason
    for existence is to consume the upstream chunkset directly.
    """
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(_fixture_chunks(), chunks_path)

    proc = _make_processor(imscc_chunks_path=chunks_path)
    loaded = proc._load_chunks_from_jsonl(chunks_path)

    assert loaded is not None, "Helper returned None on a valid JSONL file"
    assert isinstance(loaded, list)
    assert len(loaded) == 3
    assert loaded[0]["id"] == "phase8_st2_test_chunk_00001"
    assert loaded[1]["id"] == "phase8_st2_test_chunk_00002"
    assert loaded[2]["id"] == "phase8_st2_test_chunk_00003"
    assert proc.stats["total_chunks"] == 3, (
        "Helper failed to update self.stats['total_chunks'] -- "
        "downstream summary print will report 0 chunks."
    )


def test_load_chunks_from_jsonl_returns_none_on_missing_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing file falls through to the in-process build with a warning."""
    missing_path = tmp_path / "does_not_exist.jsonl"
    assert not missing_path.exists()

    proc = _make_processor(imscc_chunks_path=missing_path)

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        result = proc._load_chunks_from_jsonl(missing_path)

    assert result is None, (
        "Helper should return None on missing file (signal to fall "
        "through to in-process build), got %r" % (result,)
    )
    assert any(
        "does not exist" in r.getMessage() or "is not a file" in r.getMessage()
        for r in caplog.records
    ), "Expected a warning log mentioning the missing path"


def test_load_chunks_from_jsonl_returns_none_on_malformed_jsonl(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed JSONL falls through to the in-process build."""
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("not valid json {{{\n", encoding="utf-8")

    proc = _make_processor(imscc_chunks_path=bad_path)

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        result = proc._load_chunks_from_jsonl(bad_path)

    assert result is None
    assert any(
        "malformed jsonl" in r.getMessage().lower()
        for r in caplog.records
    ), "Expected a warning log mentioning malformed JSONL"


def test_load_chunks_from_jsonl_returns_none_on_empty_jsonl(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty JSONL file is treated as a fall-through signal (defensive)."""
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")

    proc = _make_processor(imscc_chunks_path=empty_path)

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        result = proc._load_chunks_from_jsonl(empty_path)

    assert result is None, (
        "Helper should treat an empty JSONL as fall-through "
        "(empty upstream chunkset is almost always a bug)"
    )
    assert any(
        "empty" in r.getMessage().lower()
        for r in caplog.records
    ), "Expected a warning log mentioning the empty chunkset"


def test_load_chunks_from_jsonl_returns_none_on_non_dict_payload(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JSONL line that parses to a non-dict (e.g. a list) falls through."""
    bad_path = tmp_path / "non_dict.jsonl"
    bad_path.write_text(
        json.dumps({"id": "ok"}) + "\n" + json.dumps(["not", "a", "dict"]) + "\n",
        encoding="utf-8",
    )

    proc = _make_processor(imscc_chunks_path=bad_path)

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        result = proc._load_chunks_from_jsonl(bad_path)

    assert result is None
    assert any(
        "non-dict chunk" in r.getMessage()
        for r in caplog.records
    ), "Expected a warning log mentioning the non-dict chunk"


# ---------------------------------------------------------------------------
# Phase 8 ST 2 contract tests — process() short-circuit
# ---------------------------------------------------------------------------


def _patch_process_pipeline_around_chunking(proc: CourseProcessor) -> None:
    """Stub out everything in ``process()`` except chunk acquisition.

    Mirrors the ``test_concept_graph_consumption.py`` mock pattern --
    we want to assert that ``_chunk_content`` is (or isn't) called
    based on ``imscc_chunks_path``, without dragging in IMSCC
    extraction / HTML parsing / metadata emit.
    """
    proc._create_directories = lambda: None  # type: ignore[method-assign]
    proc._extract_imscc = lambda: ("Test Course", [])  # type: ignore[method-assign]
    proc._parse_html = lambda html_files: []  # type: ignore[method-assign]
    proc._detect_corpus_boilerplate = lambda items: set()  # type: ignore[method-assign]
    proc._build_valid_outcome_ids = lambda: set()  # type: ignore[method-assign]
    proc._write_chunks = lambda chunks: None  # type: ignore[method-assign]
    proc._generate_concept_graph = lambda chunks: {}  # type: ignore[method-assign]
    proc._generate_pedagogy_graph = lambda chunks, **kw: {}  # type: ignore[method-assign]
    proc._generate_manifest = lambda title, **kw: {}  # type: ignore[method-assign]
    proc._generate_corpus_stats = lambda: {}  # type: ignore[method-assign]
    proc._generate_quality_report = lambda chunks: {}  # type: ignore[method-assign]
    proc._build_course_json = lambda manifest: {}  # type: ignore[method-assign]
    proc._generate_semantic_concept_graph = (  # type: ignore[method-assign]
        lambda chunks, course_data, concept_graph, parsed_items=None: {}
    )
    proc._write_metadata = lambda *args, **kwargs: None  # type: ignore[method-assign]


def test_process_consumes_upstream_chunks_when_path_supplied(
    tmp_path: Path,
) -> None:
    """When ``imscc_chunks_path`` is set + readable, ``process()`` reads
    chunks from the upstream JSONL and does NOT invoke ``_chunk_content``.

    This is the load-bearing assertion -- Phase 8 ST 2's whole reason
    for existence is to skip the redundant in-process chunker rebuild.
    """
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(_fixture_chunks(), chunks_path)

    proc = _make_processor(imscc_chunks_path=chunks_path)
    proc.imscc_path = tmp_path / "fake.imscc"
    proc.output_dir = tmp_path / "trainforge_out"
    proc.stats = {
        "total_chunks": 0,
        "total_words": 0,
        "total_tokens_estimate": 0,
    }
    _patch_process_pipeline_around_chunking(proc)

    sentinel_called = {"count": 0}

    def _sentinel_chunker(parsed_items):
        sentinel_called["count"] += 1
        raise AssertionError(
            "_chunk_content was invoked despite imscc_chunks_path "
            "being supplied -- Phase 8 ST 2 short-circuit failed."
        )

    proc._chunk_content = _sentinel_chunker  # type: ignore[method-assign]

    summary = proc.process()

    assert sentinel_called["count"] == 0, (
        "_chunk_content was called despite a valid imscc_chunks_path "
        "being threaded in -- the short-circuit didn't fire."
    )
    assert summary["status"] == "success"
    assert summary["stats"]["total_chunks"] == 3, (
        "stats[total_chunks] should reflect the upstream chunkset's "
        "size (3 fixture chunks); got %r"
        % (summary["stats"].get("total_chunks"),)
    )


def test_process_falls_through_to_chunk_content_when_path_is_none(
    tmp_path: Path,
) -> None:
    """Legacy callers (no ``imscc_chunks_path``) preserve pre-Phase-8
    behaviour -- the in-process ``_chunk_content`` still runs.
    """
    proc = _make_processor(imscc_chunks_path=None)
    proc.imscc_path = tmp_path / "fake.imscc"
    proc.output_dir = tmp_path / "trainforge_out"
    proc.stats = {
        "total_chunks": 0,
        "total_words": 0,
        "total_tokens_estimate": 0,
    }
    _patch_process_pipeline_around_chunking(proc)

    chunker_called = {"count": 0, "received": None}

    def _capturing_chunker(parsed_items):
        chunker_called["count"] += 1
        chunker_called["received"] = parsed_items
        # Mirror the side channel the real wrapper sets.
        proc.stats["total_chunks"] = 1
        return [{"id": "in_process_built_chunk_00001"}]

    proc._chunk_content = _capturing_chunker  # type: ignore[method-assign]

    summary = proc.process()

    assert chunker_called["count"] == 1, (
        "_chunk_content was NOT invoked when imscc_chunks_path is "
        "None -- legacy fall-through path is broken."
    )
    assert summary["stats"]["total_chunks"] == 1


def test_process_falls_through_to_chunk_content_when_path_missing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale phase-output handoff (path doesn't exist) falls through to
    the in-process ``_chunk_content`` build with a warning log.
    """
    missing_path = tmp_path / "does_not_exist.jsonl"
    assert not missing_path.exists()

    proc = _make_processor(imscc_chunks_path=missing_path)
    proc.imscc_path = tmp_path / "fake.imscc"
    proc.output_dir = tmp_path / "trainforge_out"
    proc.stats = {
        "total_chunks": 0,
        "total_words": 0,
        "total_tokens_estimate": 0,
    }
    _patch_process_pipeline_around_chunking(proc)

    chunker_called = {"count": 0}

    def _capturing_chunker(parsed_items):
        chunker_called["count"] += 1
        proc.stats["total_chunks"] = 2
        return [
            {"id": "fallback_chunk_00001"},
            {"id": "fallback_chunk_00002"},
        ]

    proc._chunk_content = _capturing_chunker  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        summary = proc.process()

    assert chunker_called["count"] == 1, (
        "_chunk_content was NOT called as fallback when "
        "imscc_chunks_path doesn't exist -- the fail-soft path "
        "is broken."
    )
    assert summary["stats"]["total_chunks"] == 2
    assert any(
        "does not exist" in r.getMessage() or "is not a file" in r.getMessage()
        for r in caplog.records
    ), (
        "Expected a warning log mentioning the missing path "
        "during the fallback"
    )


def test_process_falls_through_to_chunk_content_when_jsonl_malformed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed JSONL at ``imscc_chunks_path`` falls through to the
    in-process build instead of crashing the run.
    """
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("not valid json {{{\n", encoding="utf-8")

    proc = _make_processor(imscc_chunks_path=bad_path)
    proc.imscc_path = tmp_path / "fake.imscc"
    proc.output_dir = tmp_path / "trainforge_out"
    proc.stats = {
        "total_chunks": 0,
        "total_words": 0,
        "total_tokens_estimate": 0,
    }
    _patch_process_pipeline_around_chunking(proc)

    chunker_called = {"count": 0}

    def _capturing_chunker(parsed_items):
        chunker_called["count"] += 1
        proc.stats["total_chunks"] = 1
        return [{"id": "fallback_chunk_00001"}]

    proc._chunk_content = _capturing_chunker  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        summary = proc.process()

    assert chunker_called["count"] == 1, (
        "_chunk_content was NOT called as fallback on malformed "
        "upstream JSONL -- the fail-soft path is broken."
    )
    assert summary["stats"]["total_chunks"] == 1
    assert any(
        "malformed jsonl" in r.getMessage().lower()
        for r in caplog.records
    ), "Expected a warning log mentioning malformed JSONL"


# ---------------------------------------------------------------------------
# Constructor + CLI plumbing tests
# ---------------------------------------------------------------------------


def test_constructor_kwarg_threaded_into_self(tmp_path: Path) -> None:
    """The ``imscc_chunks_path`` constructor kwarg lands on
    ``self.imscc_chunks_path`` as a ``Path``.

    Locks the wiring contract so callers (workflow runner, MCP tool,
    standalone CLI) can rely on the field being set after construction.
    """
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    # Path(...) coercion: kwarg accepts both str and Path-like.
    proc1 = CourseProcessor.__new__(CourseProcessor)
    proc1.imscc_chunks_path = Path(str(chunks_path))
    assert isinstance(proc1.imscc_chunks_path, Path)
    assert proc1.imscc_chunks_path == chunks_path

    # None resolves to None (legacy default).
    proc2 = CourseProcessor.__new__(CourseProcessor)
    proc2.imscc_chunks_path = None
    assert proc2.imscc_chunks_path is None


def test_cli_flag_threads_through_to_constructor() -> None:
    """The ``--imscc-chunks-path`` CLI flag is wired through to the
    ``CourseProcessor`` constructor in ``main()``.

    Verifies the argparse parser declares the flag and the ``main``
    function passes it to ``CourseProcessor(...)``.
    """
    from Trainforge.process_course import build_parser

    parser = build_parser()
    # Build a minimal arg vector that satisfies required args.
    args = parser.parse_args(
        [
            "--imscc", "/tmp/fake.imscc",
            "--output", "/tmp/out",
            "--course-code", "TEST",
            "--imscc-chunks-path", "/tmp/upstream/chunks.jsonl",
        ]
    )
    assert args.imscc_chunks_path == "/tmp/upstream/chunks.jsonl"


# ---------------------------------------------------------------------------
# Pipeline integration test — _run_trainforge_assessment kwarg passthrough
# ---------------------------------------------------------------------------


def test_pipeline_tool_passes_imscc_chunks_path_to_course_processor() -> None:
    """The ``_run_trainforge_assessment`` helper in
    ``MCP/tools/pipeline_tools.py`` accepts ``imscc_chunks_path`` from
    its kwargs and threads it into ``CourseProcessor(...)``.

    Smoke check on the registry-only helper -- exercises the
    instantiation site without dispatching the full async chain. Reads
    the source as text since the helper is closure-bound inside the
    private ``_build_tool_registry`` factory and isn't directly
    importable.
    """
    pipeline_tools_src = (
        PROJECT_ROOT
        / "MCP"
        / "tools"
        / "pipeline_tools.py"
    ).read_text(encoding="utf-8")
    # The kwarg pickup line + the constructor passthrough must both be
    # present after Phase 8 ST 2 lands.
    assert 'imscc_chunks_path_kw = kwargs.get("imscc_chunks_path")' in pipeline_tools_src, (
        "_run_trainforge_assessment must pick imscc_chunks_path off "
        "kwargs (Phase 8 ST 2)"
    )
    assert "imscc_chunks_path=(" in pipeline_tools_src, (
        "_run_trainforge_assessment must pass imscc_chunks_path "
        "through to CourseProcessor (Phase 8 ST 2)"
    )


def test_workflow_yaml_routes_imscc_chunks_path() -> None:
    """``config/workflows.yaml::trainforge_assessment.inputs_from`` carries
    the new ``imscc_chunks_path`` route from
    ``phase_outputs.imscc_chunking.imscc_chunks_path``.
    """
    import yaml

    workflows_path = PROJECT_ROOT / "config" / "workflows.yaml"
    with workflows_path.open("r", encoding="utf-8") as fh:
        workflows = yaml.safe_load(fh)

    workflow = workflows["workflows"]["textbook_to_course"]
    phases = {p["name"]: p for p in workflow["phases"]}
    trainforge_phase = phases["trainforge_assessment"]
    inputs_from = trainforge_phase["inputs_from"]
    matches = [
        entry for entry in inputs_from
        if entry.get("param") == "imscc_chunks_path"
    ]
    assert len(matches) == 1, (
        "Expected exactly one inputs_from entry routing "
        "imscc_chunks_path; got %d" % len(matches)
    )
    entry = matches[0]
    assert entry["source"] == "phase_outputs"
    assert entry["phase"] == "imscc_chunking"
    assert entry["output"] == "imscc_chunks_path"
    # depends_on must include imscc_chunking so the routing
    # precondition is pinned in the workflow graph.
    assert "imscc_chunking" in trainforge_phase["depends_on"], (
        "trainforge_assessment must depend on imscc_chunking now "
        "that it consumes its output"
    )


def test_legacy_dict_mirrors_yaml_imscc_chunks_path_routing() -> None:
    """``MCP/core/workflow_runner.py::_LEGACY_PHASE_PARAM_ROUTING`` mirrors
    the YAML routing addition for ``imscc_chunks_path``.

    The legacy dict is consulted by ``_get_phase_param_routing`` as a
    fallback when YAML lookup misses; the ``test_workflow_runner_meta_schema``
    suite asserts the dict matches YAML, so the addition must land in
    both places.
    """
    from MCP.core.workflow_runner import _LEGACY_PHASE_PARAM_ROUTING

    trainforge_routing = _LEGACY_PHASE_PARAM_ROUTING["trainforge_assessment"]
    assert "imscc_chunks_path" in trainforge_routing, (
        "_LEGACY_PHASE_PARAM_ROUTING['trainforge_assessment'] must "
        "include imscc_chunks_path now that the YAML routes it"
    )
    assert trainforge_routing["imscc_chunks_path"] == (
        "phase_outputs", "imscc_chunking", "imscc_chunks_path",
    )
