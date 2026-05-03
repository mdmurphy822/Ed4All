"""Phase 5 Subtask 2 — ``_synthesize_outline_output`` synthesizer.

Locks the workflow runner's behaviour when a workflow's params carry
``outline_dir`` (set by the new ``courseforge-rewrite`` /
``courseforge-validate`` / ``courseforge-*`` stage subcommands —
plumbed in via Worker WA's CLI changes):

* ``_synthesize_outline_output`` walks the OUTLINE_DIR (Courseforge
  project export root or the ``01_outline/`` subdirectory) and
  reconstructs the per-phase ``phase_outputs`` dicts for every
  upstream phase in the canonical chain (``staging``, ``chunking``,
  ``objective_extraction``, ``source_mapping``, ``concept_extraction``,
  ``course_planning``, ``content_generation_outline``,
  ``inter_tier_validation``).
* Returned dicts carry ``_completed: True`` so the workflow runner's
  skip-already-completed check at ``run_workflow:860`` short-circuits
  every reconstructed phase.
* Per-phase keys match what ``inputs_from`` for downstream phases
  resolves against (per ``config/workflows.yaml`` per-phase
  ``inputs_from:`` blocks).
* Missing artifacts log warnings and skip the phase rather than
  silently emit a placeholder — the workflow runner's
  ``_dependencies_met`` then surfaces the gap as a normal dependency
  failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from MCP.core.workflow_runner import WorkflowRunner


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    """Minimal WorkflowRunner — _synthesize_outline_output is pure."""
    return WorkflowRunner(executor=object(), config=object())


def _make_project(tmp_path: Path, course_name: str = "TEST_101") -> Path:
    """Scaffold a minimal Courseforge project export directory.

    Mirrors the layout `_run_*` helpers in ``MCP/tools/pipeline_tools.py``
    write to: project_config.json + 01_learning_objectives/* +
    source_module_map.json + 01_outline/*.
    """
    project = tmp_path / f"PROJ-{course_name}-20260502"
    project.mkdir()
    (project / "01_learning_objectives").mkdir()
    (project / "01_outline").mkdir()

    (project / "project_config.json").write_text(
        json.dumps({
            "course_name": course_name,
            "project_id": project.name,
            "duration_weeks": 8,
            "staging_dir": str(tmp_path / "staging" / "run-001"),
        }),
        encoding="utf-8",
    )
    return project


def _make_staging(tmp_path: Path) -> Path:
    """Scaffold a minimal staging dir with a couple of HTML inputs."""
    staging = tmp_path / "staging" / "run-001"
    staging.mkdir(parents=True)
    (staging / "chapter_01_accessible.html").write_text(
        "<html><body><h1>Ch1</h1></body></html>", encoding="utf-8"
    )
    (staging / "chapter_02_accessible.html").write_text(
        "<html><body><h1>Ch2</h1></body></html>", encoding="utf-8"
    )
    (staging / "staging_manifest.json").write_text(
        json.dumps({
            "run_id": "run-001",
            "course_name": "TEST_101",
            "files": [
                {"path": "chapter_01_accessible.html", "role": "content"},
                {"path": "chapter_02_accessible.html", "role": "content"},
            ],
        }),
        encoding="utf-8",
    )
    return staging


def _make_libv2(tmp_path: Path, course_slug: str) -> Path:
    """Scaffold a LibV2 course dir with dart_chunks/ + concept_graph/.

    Patches ``PROJECT_ROOT`` so the synthesizer resolves
    ``LibV2/courses/<slug>/`` against tmp_path rather than the repo
    root.
    """
    libv2_root = tmp_path / "LibV2" / "courses" / course_slug
    chunks_dir = libv2_root / "dart_chunks"
    chunks_dir.mkdir(parents=True)
    chunks_path = chunks_dir / "chunks.jsonl"
    chunks_path.write_text(
        '{"chunk_id": "c1", "text": "..."}\n'
        '{"chunk_id": "c2", "text": "..."}\n',
        encoding="utf-8",
    )
    (chunks_dir / "manifest.json").write_text(
        json.dumps({
            "chunks_sha256": "a" * 64,
            "chunker_version": "1.0",
            "chunkset_kind": "dart",
            "source_dart_html_sha256": "b" * 64,
            "chunks_count": 2,
            "generated_at": "2026-05-02T00:00:00Z",
        }),
        encoding="utf-8",
    )

    graph_dir = libv2_root / "concept_graph"
    graph_dir.mkdir()
    (graph_dir / "concept_graph_semantic.json").write_text(
        json.dumps({"nodes": [], "edges": []}),
        encoding="utf-8",
    )
    (graph_dir / "manifest.json").write_text(
        json.dumps({
            "course_id": "TEST_101",
            "course_slug": course_slug,
            "concept_graph_path": str(graph_dir / "concept_graph_semantic.json"),
            "concept_graph_sha256": "c" * 64,
            "generated_at": "2026-05-02T00:00:00Z",
            "source_chunks": 2,
            "phase": "concept_extraction",
        }),
        encoding="utf-8",
    )
    return libv2_root


def _populate_objectives(project_path: Path) -> None:
    """Write textbook_structure.json + synthesized_objectives.json."""
    obj_dir = project_path / "01_learning_objectives"
    (obj_dir / "textbook_structure.json").write_text(
        json.dumps({
            "chapters": [
                {"title": "Ch1", "sections": []},
                {"title": "Ch2", "sections": []},
            ],
            "duration_weeks": 8,
        }),
        encoding="utf-8",
    )
    (obj_dir / "synthesized_objectives.json").write_text(
        json.dumps({
            "course_name": "TEST_101",
            "duration_weeks": 8,
            "terminal_objectives": [
                {"id": "TO-01", "statement": "T1", "bloom_level": "understand"},
                {"id": "TO-02", "statement": "T2", "bloom_level": "apply"},
            ],
            "chapter_objectives": [
                {
                    "chapter": "Week 1",
                    "objectives": [
                        {"id": "CO-01", "statement": "C1",
                         "parent_terminal": "TO-01"},
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )


def _populate_source_map(project_path: Path) -> None:
    """Write source_module_map.json with two weeks + chunk IDs."""
    (project_path / "source_module_map.json").write_text(
        json.dumps({
            "1": [
                {"chunk_id": "c1", "page": "page1.html"},
                {"chunk_id": "c2", "page": "page1.html"},
            ],
            "2": [
                {"chunk_id": "c3", "page": "page2.html"},
            ],
        }),
        encoding="utf-8",
    )


def _populate_outline(project_path: Path) -> None:
    """Write 01_outline/blocks_outline.jsonl + blocks_validated.jsonl."""
    out_dir = project_path / "01_outline"
    blocks_outline = out_dir / "blocks_outline.jsonl"
    blocks_outline.write_text(
        '{"block_id": "b1", "block_type": "objective", "week": 1}\n'
        '{"block_id": "b2", "block_type": "concept", "week": 1}\n'
        '{"block_id": "b3", "block_type": "example", "week": 2}\n',
        encoding="utf-8",
    )
    (out_dir / "blocks_validated.jsonl").write_text(
        '{"block_id": "b1", "block_type": "objective", "week": 1}\n'
        '{"block_id": "b2", "block_type": "concept", "week": 1}\n',
        encoding="utf-8",
    )
    (out_dir / "blocks_failed.jsonl").write_text(
        '{"block_id": "b3", "block_type": "example", "week": 2}\n',
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


class TestSynthesizeOutlineOutputHappyPath:
    def test_emits_all_canonical_phases(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _make_staging(tmp_path)
        _make_libv2(tmp_path, "test-101")
        _populate_objectives(project_path)
        _populate_source_map(project_path)
        _populate_outline(project_path)

        # Patch PROJECT_ROOT so LibV2 lookup resolves against tmp_path.
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        synth = runner_stub._synthesize_outline_output(project_path)

        # All canonical phases reconstructed.
        expected_phases = {
            "staging",
            "dart_conversion",
            "chunking",
            "objective_extraction",
            "source_mapping",
            "concept_extraction",
            "course_planning",
            "content_generation_outline",
            "inter_tier_validation",
        }
        assert expected_phases.issubset(set(synth.keys()))

        # Every entry carries _completed=True so the phase-loop skip
        # check at run_workflow:860 fires.
        for phase_name in expected_phases:
            assert synth[phase_name]["_completed"] is True, phase_name
            assert synth[phase_name].get("_skipped") is True, phase_name

    def test_staging_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _make_staging(tmp_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        st = synth["staging"]
        # workflows.yaml::staging.outputs: [staging_dir, staged_files,
        # file_count]
        assert "staging_dir" in st
        assert "staged_files" in st
        assert "file_count" in st
        assert st["file_count"] == 2
        assert Path(st["staging_dir"]).is_dir()

    def test_chunking_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _make_libv2(tmp_path, "test-101")
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        chk = synth["chunking"]
        # workflows.yaml::chunking.outputs: [dart_chunks_path,
        # dart_chunks_sha256]
        assert "dart_chunks_path" in chk
        assert "dart_chunks_sha256" in chk
        assert chk["dart_chunks_sha256"] == "a" * 64

    def test_concept_extraction_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _make_libv2(tmp_path, "test-101")
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        ce = synth["concept_extraction"]
        # workflows.yaml::concept_extraction.outputs:
        # [concept_graph_path, concept_graph_sha256]
        assert "concept_graph_path" in ce
        assert "concept_graph_sha256" in ce
        assert ce["concept_graph_sha256"] == "c" * 64

    def test_objective_extraction_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _populate_objectives(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        oe = synth["objective_extraction"]
        # workflows.yaml::objective_extraction.outputs: project_id,
        # project_path, textbook_structure_path, chapter_count,
        # duration_weeks, source_file_count.
        assert "project_id" in oe
        assert "project_path" in oe
        assert "textbook_structure_path" in oe
        assert oe["chapter_count"] == 2
        assert oe["duration_weeks"] == 8

    def test_source_mapping_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _populate_source_map(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        sm = synth["source_mapping"]
        # workflows.yaml::source_mapping.outputs:
        # [source_module_map_path, source_chunk_ids]
        assert "source_module_map_path" in sm
        assert "source_chunk_ids" in sm
        assert set(sm["source_chunk_ids"]) == {"c1", "c2", "c3"}

    def test_course_planning_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _populate_objectives(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        cp = synth["course_planning"]
        # workflows.yaml::course_planning.outputs: project_id,
        # synthesized_objectives_path, objective_ids, terminal_count,
        # chapter_count.
        assert "project_id" in cp
        assert "synthesized_objectives_path" in cp
        assert "objective_ids" in cp
        assert cp["terminal_count"] == 2
        assert cp["chapter_count"] == 1
        assert set(cp["objective_ids"].split(",")) == {
            "TO-01", "TO-02", "CO-01",
        }

    def test_outline_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _populate_outline(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        cgo = synth["content_generation_outline"]
        # workflows.yaml::content_generation_outline.outputs:
        # [blocks_outline_path, project_id, weeks_prepared]
        assert "blocks_outline_path" in cgo
        assert "project_id" in cgo
        assert cgo["weeks_prepared"] == 2
        assert cgo["block_count"] == 3

    def test_inter_tier_validation_keys_match_workflows_yaml(
        self, runner_stub, tmp_path, monkeypatch
    ):
        project_path = _make_project(tmp_path)
        _populate_outline(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )
        synth = runner_stub._synthesize_outline_output(project_path)

        itv = synth["inter_tier_validation"]
        # workflows.yaml::inter_tier_validation.outputs:
        # [blocks_validated_path, blocks_failed_path]
        assert "blocks_validated_path" in itv
        assert "blocks_failed_path" in itv
        assert Path(itv["blocks_validated_path"]).exists()
        assert Path(itv["blocks_failed_path"]).exists()


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


class TestSynthesizeOutlineOutputEdgeCases:
    def test_outline_subdir_resolves_to_project_path(
        self, runner_stub, tmp_path, monkeypatch
    ):
        """Operator passes ``--outline /path/to/PROJ-X/01_outline``."""
        project_path = _make_project(tmp_path)
        _populate_outline(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        # Pass the 01_outline subdir; synthesizer should walk back to
        # the project root.
        synth = runner_stub._synthesize_outline_output(
            project_path / "01_outline"
        )

        assert "content_generation_outline" in synth

    def test_missing_artifact_skips_phase_with_warning(
        self, runner_stub, tmp_path, caplog, monkeypatch
    ):
        """Missing dart_chunks/manifest.json => chunking phase omitted."""
        project_path = _make_project(tmp_path)
        # Don't make LibV2 dirs => chunking + concept_extraction
        # should be omitted.
        _populate_objectives(project_path)
        _populate_outline(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        with caplog.at_level("WARNING"):
            synth = runner_stub._synthesize_outline_output(project_path)

        assert "chunking" not in synth
        assert "concept_extraction" not in synth
        # Phases whose artifacts are present DO get reconstructed.
        assert "objective_extraction" in synth
        assert "content_generation_outline" in synth

    def test_target_phases_filter_limits_output(
        self, runner_stub, tmp_path, monkeypatch
    ):
        """``target_phases=['course_planning']`` only emits that key."""
        project_path = _make_project(tmp_path)
        _make_staging(tmp_path)
        _make_libv2(tmp_path, "test-101")
        _populate_objectives(project_path)
        _populate_source_map(project_path)
        _populate_outline(project_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        synth = runner_stub._synthesize_outline_output(
            project_path, target_phases=["course_planning"],
        )

        assert set(synth.keys()) == {"course_planning"}

    def test_unknown_phase_silently_dropped(
        self, runner_stub, tmp_path, monkeypatch
    ):
        """``target_phases=['nonexistent_phase']`` => empty dict."""
        project_path = _make_project(tmp_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        synth = runner_stub._synthesize_outline_output(
            project_path, target_phases=["nonexistent_phase"],
        )

        assert synth == {}

    def test_missing_project_dir_returns_empty(
        self, runner_stub, tmp_path, caplog
    ):
        """Bad outline_dir => empty dict + error log."""
        bogus = tmp_path / "no_such_project"

        with caplog.at_level("ERROR"):
            synth = runner_stub._synthesize_outline_output(bogus)

        assert synth == {}

    def test_dart_conversion_derived_from_staging(
        self, runner_stub, tmp_path, monkeypatch
    ):
        """When staging is reconstructed, dart_conversion follows."""
        project_path = _make_project(tmp_path)
        _make_staging(tmp_path)
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        synth = runner_stub._synthesize_outline_output(project_path)

        assert "dart_conversion" in synth
        dc = synth["dart_conversion"]
        # output_paths is comma-joined; should contain both staged HTMLs.
        assert "chapter_01_accessible.html" in dc["output_paths"]
        assert "chapter_02_accessible.html" in dc["output_paths"]

    def test_dart_conversion_omitted_when_staging_missing(
        self, runner_stub, tmp_path, monkeypatch
    ):
        """No staging => no dart_conversion either."""
        project_path = _make_project(tmp_path)
        # No staging dir.
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        synth = runner_stub._synthesize_outline_output(project_path)

        assert "dart_conversion" not in synth
        assert "staging" not in synth

    def test_partial_chain_only_reconstructs_present_artifacts(
        self, runner_stub, tmp_path, monkeypatch
    ):
        """Only objective_extraction artifact present => only that phase.

        ``_populate_objectives`` writes both textbook_structure.json AND
        synthesized_objectives.json (matching the disk layout the
        ``objective_extraction`` + ``course_planning`` phase handlers
        emit), so both phases reconstruct. Phases whose disk artifacts
        are NOT written (source_mapping, chunking, concept_extraction,
        outline) stay omitted.
        """
        project_path = _make_project(tmp_path)
        _populate_objectives(project_path)
        # No other artifacts.
        monkeypatch.setattr(
            "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
        )

        synth = runner_stub._synthesize_outline_output(project_path)

        assert "objective_extraction" in synth
        # course_planning IS present — synthesized_objectives.json
        # was written by _populate_objectives.
        assert "course_planning" in synth
        # Phases whose disk artifacts are absent are NOT in the dict.
        assert "source_mapping" not in synth
        assert "chunking" not in synth
        assert "concept_extraction" not in synth
        assert "content_generation_outline" not in synth
        assert "inter_tier_validation" not in synth
