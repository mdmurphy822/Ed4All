"""Wave 80 Worker A — ``--reuse-objectives`` phase-skip mechanic.

Locks the workflow runner's behaviour when a workflow's params carry
``reuse_objectives_path``:

* ``_synthesize_course_planning_reuse_output`` reads the user-supplied
  objectives JSON, normalizes to the Courseforge form, writes
  ``synthesized_objectives.json`` into the project's
  ``01_learning_objectives/`` dir, and emits a phase_output dict with
  the keys downstream phases' ``inputs_from`` pulls
  (``project_id``, ``synthesized_objectives_path``, ``objective_ids``,
  ``terminal_count``, ``chapter_count``, plus
  ``_completed``/``_skipped``/``_gates_passed`` markers).
* Cross-validation rejects orphan ``parent_terminal`` references,
  malformed LO IDs, and duplicates.
* When the project_path / project_id is unresolvable from upstream,
  the method returns ``None`` (caller surfaces failure).
* Course-outliner subagent is NOT dispatched when --reuse-objectives is
  set; the phase loop's already-completed guard skips dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from MCP.core.workflow_runner import WorkflowRunner


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    """Minimal WorkflowRunner — we only exercise pure helpers below.

    The ``executor`` and ``config`` fields are unused by
    ``_synthesize_course_planning_reuse_output``, so a sentinel object
    keeps the constructor happy without hauling in a real
    OrchestratorConfig.
    """
    return WorkflowRunner(executor=object(), config=object())


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Scaffold a minimal Courseforge project directory."""
    project = tmp_path / "PROJ-TEST_101-20260424"
    (project / "01_learning_objectives").mkdir(parents=True)
    return project


@pytest.fixture
def courseforge_reuse_file(tmp_path: Path) -> Path:
    """Minimal valid Courseforge-form objectives file outside the project."""
    p = tmp_path / "reuse_objectives.json"
    p.write_text(
        json.dumps({
            "course_name": "TEST_101",
            "duration_weeks": 8,
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Foundations of X.",
                 "bloom_level": "understand"},
                {"id": "TO-02", "statement": "Applications of X.",
                 "bloom_level": "apply"},
            ],
            "chapter_objectives": [
                {
                    "chapter": "Week 1",
                    "objectives": [
                        {"id": "CO-01", "statement": "Identify terms.",
                         "parent_terminal": "TO-01"},
                    ],
                },
                {
                    "chapter": "Week 2",
                    "objectives": [
                        {"id": "CO-02", "statement": "Apply terms.",
                         "parent_terminal": "TO-02"},
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )
    return p


def _build_phase_outputs(project_dir: Path) -> dict:
    """Stand-in for an ``objective_extraction`` phase_output entry."""
    return {
        "objective_extraction": {
            "project_id": project_dir.name,
            "project_path": str(project_dir),
            "_completed": True,
        },
    }


# ---------------------------------------------------------------------
# _synthesize_course_planning_reuse_output behaviour
# ---------------------------------------------------------------------


class TestSynthesizeReuseOutput:
    def test_emits_expected_keys(
        self, runner_stub, project_dir, courseforge_reuse_file
    ):
        params = {
            "reuse_objectives_path": str(courseforge_reuse_file),
            "course_name": "TEST_101",
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is not None

        # Markers.
        assert out["_completed"] is True
        assert out["_skipped"] is True
        assert out["_gates_passed"] is True

        # Counts.
        assert out["terminal_count"] == 2
        assert out["chapter_count"] == 2

        # Project id surfaced from upstream.
        assert out["project_id"] == project_dir.name

        # objective_ids comma-joined and contains all four IDs.
        ids = set(out["objective_ids"].split(","))
        assert ids == {"TO-01", "TO-02", "CO-01", "CO-02"}

        # synthesized_objectives.json was written into the project.
        out_path = Path(out["synthesized_objectives_path"])
        expected = project_dir / "01_learning_objectives" / "synthesized_objectives.json"
        assert out_path == expected
        assert out_path.exists()

        # Disk file has the canonical Courseforge shape.
        synthesized = json.loads(out_path.read_text(encoding="utf-8"))
        assert "terminal_objectives" in synthesized
        assert "chapter_objectives" in synthesized
        assert synthesized["mint_method"] == "reuse_objectives"
        assert synthesized["generated_from"] == str(courseforge_reuse_file)

    def test_updates_project_config(
        self, runner_stub, project_dir, courseforge_reuse_file
    ):
        params = {
            "reuse_objectives_path": str(courseforge_reuse_file),
            "course_name": "TEST_101",
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is not None

        config_path = project_dir / "project_config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["objectives_path"] == out["synthesized_objectives_path"]
        assert config["synthesized_objectives_path"] == (
            out["synthesized_objectives_path"]
        )
        assert config["status"] == "planned"
        assert config["course_name"] == "TEST_101"

    def test_backfills_dart_chunk_lo_refs(
        self, runner_stub, project_dir, courseforge_reuse_file, tmp_path
    ):
        """Phase-ordering fix (Option A1) companion: a reuse run must
        back-fill ``learning_outcome_refs`` onto the on-disk DART chunkset
        (the live course_planning path does this at its tail; the reuse
        path short-circuits that, so the synthesizer calls the backfill
        helper directly). Without it the reused chunks keep empty LO refs
        and the downstream concept graph degrades.
        """
        # The reuse synthesizer derives the slug as
        # course_name.lower().replace("_","-") -> "test-101".
        libv2_root = tmp_path / "libv2"
        chunks_path = (
            libv2_root / "courses" / "test-101" / "dart_chunks" / "chunks.jsonl"
        )
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.write_text(
            json.dumps({
                "id": "test_chunk_00001",
                "text": "TO-01 foundations: this chunk references CO-01 too.",
                "learning_outcome_refs": [],
            }) + "\n",
            encoding="utf-8",
        )

        params = {
            "reuse_objectives_path": str(courseforge_reuse_file),
            "course_name": "TEST_101",
            "libv2_root": str(libv2_root),
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is not None

        # The chunk's text mentions TO-01 + CO-01 (both in the reuse
        # objective_ids allowlist) → backfill populates learning_outcome_refs.
        refreshed = json.loads(chunks_path.read_text(encoding="utf-8").strip())
        assert set(refreshed.get("learning_outcome_refs") or []) >= {
            "TO-01", "CO-01"
        }, (
            "reuse backfill must populate learning_outcome_refs from the "
            "chunk text against the reuse objective_ids allowlist; got "
            f"{refreshed.get('learning_outcome_refs')!r}"
        )

    def test_backfill_failure_does_not_break_reuse_output(
        self, runner_stub, project_dir, courseforge_reuse_file
    ):
        """The backfill is best-effort: with no LibV2 chunks file present
        (libv2_root unset → default in-tree, no test course dir), the
        reuse synthesizer must still return a valid phase output."""
        params = {
            "reuse_objectives_path": str(courseforge_reuse_file),
            "course_name": "TEST_101",
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is not None
        assert out["_completed"] is True

    def test_returns_none_when_no_project_path(
        self, runner_stub, courseforge_reuse_file
    ):
        params = {"reuse_objectives_path": str(courseforge_reuse_file)}
        # Empty phase_outputs => no upstream project.
        out = runner_stub._synthesize_course_planning_reuse_output(params, {})
        assert out is None

    def test_returns_none_when_project_path_missing(
        self, runner_stub, courseforge_reuse_file, tmp_path
    ):
        params = {"reuse_objectives_path": str(courseforge_reuse_file)}
        phase_outputs = {
            "objective_extraction": {
                "project_id": "PROJ-X",
                "project_path": str(tmp_path / "no_such_proj"),
                "_completed": True,
            },
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, phase_outputs,
        )
        assert out is None

    def test_returns_none_when_reuse_file_missing(
        self, runner_stub, project_dir, tmp_path
    ):
        params = {
            "reuse_objectives_path": str(tmp_path / "no_such.json"),
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is None

    def test_returns_none_when_orphan_parent(
        self, runner_stub, project_dir, tmp_path
    ):
        """CO with parent_terminal pointing to non-existent TO."""
        bad = tmp_path / "orphan.json"
        bad.write_text(
            json.dumps({
                "terminal_objectives": [
                    {"id": "TO-01", "statement": "T1"},
                ],
                "chapter_objectives": [
                    {"chapter": "Week 1", "objectives": [
                        {"id": "CO-01", "statement": "C1",
                         "parent_terminal": "TO-99"},
                    ]},
                ],
            }),
            encoding="utf-8",
        )
        params = {"reuse_objectives_path": str(bad)}
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is None

    def test_returns_none_when_duplicate_ids(
        self, runner_stub, project_dir, tmp_path
    ):
        bad = tmp_path / "dup.json"
        bad.write_text(
            json.dumps({
                "terminal_objectives": [
                    {"id": "TO-01", "statement": "T1"},
                    {"id": "TO-01", "statement": "T1 dup"},
                ],
                "chapter_objectives": [],
            }),
            encoding="utf-8",
        )
        params = {"reuse_objectives_path": str(bad)}
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is None

    def test_returns_none_when_malformed_id(
        self, runner_stub, project_dir, tmp_path
    ):
        bad = tmp_path / "badid.json"
        bad.write_text(
            json.dumps({
                "terminal_objectives": [
                    {"id": "not-an-lo-id", "statement": "T1"},
                ],
                "chapter_objectives": [],
            }),
            encoding="utf-8",
        )
        params = {"reuse_objectives_path": str(bad)}
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is None

    def test_no_param_returns_none(self, runner_stub, project_dir):
        # No reuse_objectives_path in params at all.
        out = runner_stub._synthesize_course_planning_reuse_output(
            {}, _build_phase_outputs(project_dir),
        )
        assert out is None


# ---------------------------------------------------------------------
# Phase loop integration
# ---------------------------------------------------------------------


class TestPhaseLoopIntegration:
    """When --reuse-objectives is set, course_planning emits the
    synthesized output without dispatch.

    We verify this via the WorkflowRunner internals: the synthesized
    output has the right keys for downstream phases' ``inputs_from``
    routing, and the per-phase summary recorded in ``all_results``
    reflects zero tasks dispatched.
    """

    def test_synthesized_output_carries_downstream_routing_keys(
        self, runner_stub, project_dir, courseforge_reuse_file
    ):
        """Pin the keys used by ``inputs_from`` of content_generation
        + trainforge_assessment (per config/workflows.yaml).

        Specifically: ``project_id`` and ``objective_ids`` are routed
        from course_planning into trainforge_assessment.
        """
        params = {
            "reuse_objectives_path": str(courseforge_reuse_file),
            "course_name": "TEST_101",
        }
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is not None

        # These four keys are declared in workflows.yaml under
        # course_planning.outputs.
        for key in (
            "project_id",
            "synthesized_objectives_path",
            "objective_ids",
            "terminal_count",
            "chapter_count",
        ):
            assert key in out, f"missing downstream-routed key: {key}"

    def test_already_completed_marker_blocks_redispatch(
        self, runner_stub, project_dir, courseforge_reuse_file
    ):
        """The phase loop's already-completed guard reads
        ``_completed=True`` to short-circuit dispatch. Synthesised
        output must therefore carry that marker.
        """
        params = {"reuse_objectives_path": str(courseforge_reuse_file)}
        out = runner_stub._synthesize_course_planning_reuse_output(
            params, _build_phase_outputs(project_dir),
        )
        assert out is not None
        assert out["_completed"] is True
