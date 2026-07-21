"""W10 assessment_synthesis graceful-stop + resume-checkpoint invariants (P4).

The ``assessment_synthesis`` phase runs three sequential per-TERMINAL-OBJECTIVE
loops (quiz / discussion / assignment), then one atomic end-write
(``_write_assessment_artifacts`` → ``06_assessments/manifest.json`` + XML). This
wave gives all three loops ONE fingerprinted sidecar
(``.assessments_checkpoint.jsonl``) keyed on the composite unit
``(kind, terminal_id)``, plus a plain ``check_stop`` at each loop top.

Two axes are exercised against the SAME sidecar:

* the FULL canonical checkpoint invariant set — per-unit write, resume
  byte-equivalence, fingerprint-mismatch re-run, torn trailing line tolerated,
  family-flag opt-out;
* the three graceful-stop legs — stop after N (sidecar == N, generator == N),
  resume (total-N generator calls, byte-equivalent output), pre-armed sentinel
  (0 calls) — plus the site-specific contract that the manifest end-write did
  NOT happen on a stop and the sidecar SURVIVES for the resume.

Hermetic: a fake ``AssessmentGenerator`` (no LLM), no prose provider (the
deterministic discussion/assignment prompt path), no SemantiK chunks. Sentinel
isolation via ``state_runs_isolated`` (per-test ``ED4ALL_STATE_RUNS_DIR``) + a
synthetic ``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from Trainforge.generators.assessment_generator import (  # noqa: E402
    AssessmentData,
    QuestionData,
)

_RUN_ID = "STOP_ASSESS_TESTRUN"
_CKPT_NAME = pt._ASSESSMENTS_CHECKPOINT_NAME


# ---------------------------------------------------------------------------
# Fake AssessmentGenerator (counts + optionally arms the real sentinel).
# ---------------------------------------------------------------------------
class _FakeGen:
    calls: List[List[str]] = []
    arm_after: int = 0

    def __init__(self, *_a, capture=None, check_leaks=False, **_k) -> None:  # noqa: D401
        pass

    def generate(
        self,
        *,
        course_code: str,
        objective_ids: List[str],
        bloom_levels: List[str],
        question_count: int,
        source_chunks=None,
    ) -> AssessmentData:
        type(self).calls.append(list(objective_ids))
        if type(self).arm_after and len(type(self).calls) == type(self).arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        questions = [
            QuestionData(
                question_id=f"q-{oid}",
                question_type="multiple_choice",
                stem=f"Which best describes {oid}?",
                bloom_level="understand",
                objective_id=oid,
                choices=[
                    {"id": "A", "text": "Correct.", "is_correct": True},
                    {"id": "B", "text": "Distractor one.", "is_correct": False},
                    {"id": "C", "text": "Distractor two.", "is_correct": False},
                ],
                correct_answer="A",
                source_chunks=[],
            )
            for oid in objective_ids
        ]
        return AssessmentData(
            assessment_id=f"ASM-{objective_ids[0]}",
            title="Quiz",
            course_code=course_code,
            questions=questions,
            objectives_targeted=list(objective_ids),
            bloom_levels=list(bloom_levels),
        )


def _reset_gen(arm_after: int = 0) -> None:
    _FakeGen.calls = []
    _FakeGen.arm_after = arm_after


_OBJECTIVES_DOC: Dict[str, Any] = {
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Understand alpha concepts thoroughly."},
        {"id": "TO-02", "statement": "Understand beta concepts thoroughly."},
        {"id": "TO-03", "statement": "Understand gamma concepts thoroughly."},
    ],
    "chapter_objectives": [
        {
            "objectives": [
                {"id": "CO-01", "statement": "Explain alpha.", "terminal_id": "TO-01"},
                {"id": "CO-02", "statement": "Explain beta.", "terminal_id": "TO-02"},
                {"id": "CO-03", "statement": "Explain gamma.", "terminal_id": "TO-03"},
            ]
        }
    ],
}


def _seed_project(
    exports_root: Path, project_id: str, objectives: Optional[Dict[str, Any]] = None
) -> Path:
    proj = exports_root / project_id
    (proj / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    (proj / "project_config.json").write_text(
        json.dumps({"course_name": "DEMO_101"}), encoding="utf-8"
    )
    (proj / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps(objectives if objectives is not None else _OBJECTIVES_DOC),
        encoding="utf-8",
    )
    return proj


def _run_assess(exports_root: Path, project_id: str, monkeypatch) -> Dict[str, Any]:
    """Drive run_assessment_synthesis; return the parsed JSON envelope."""
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)
    monkeypatch.setattr(
        "Trainforge.generators.assessment_generator.AssessmentGenerator", _FakeGen
    )
    registry = pt._build_tool_registry()
    tool = registry["run_assessment_synthesis"]
    return json.loads(asyncio.run(tool(project_id=project_id)))


def _sidecar(proj: Path) -> Path:
    return proj / "06_assessments" / _CKPT_NAME


def _sidecar_records(proj: Path) -> Dict[str, Dict[str, Any]]:
    return pt._assessments_store(_sidecar(proj)).load()


def _keep_sidecars(monkeypatch) -> None:
    """Disable the post-write sidecar removal so a clean run leaves it on disk."""
    monkeypatch.setattr(pt._LlmCheckpointStore, "remove", lambda self: None)


def _output_snapshot(proj: Path) -> Dict[str, str]:
    """The emitted 06_assessments artifacts (manifest + XML), sidecar excluded."""
    out: Dict[str, str] = {}
    adir = proj / "06_assessments"
    for f in sorted(adir.glob("*.xml")):
        out[f.name] = f.read_text(encoding="utf-8")
    mf = adir / "manifest.json"
    if mf.exists():
        out["manifest.json"] = mf.read_text(encoding="utf-8")
    return out


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    monkeypatch.delenv("ED4ALL_ASSESSMENT_PROSE_PROVIDER", raising=False)
    stop_control.clear_stop(include_global=True)
    _reset_gen()
    yield
    stop_control.clear_stop(include_global=True)


# ===========================================================================
# Canonical checkpoint invariants
# ===========================================================================
def test_clean_run_removes_sidecar_and_writes_manifest(
    tmp_path, monkeypatch, _armed_env
):
    proj = _seed_project(tmp_path, "P_CLEAN")
    env = _run_assess(tmp_path, "P_CLEAN", monkeypatch)
    assert env["success"] is True
    # 3 quizzes + 3 discussions + 3 assignments.
    assert env["quiz_count"] == 3
    assert env["discussion_count"] == 3
    assert env["assignment_count"] == 3
    # Sidecar removed after the successful manifest end-write.
    assert not _sidecar(proj).exists()
    assert (proj / "06_assessments" / "manifest.json").exists()


def test_sidecar_written_per_unit(tmp_path, monkeypatch, _armed_env):
    _keep_sidecars(monkeypatch)
    proj = _seed_project(tmp_path, "P_PERUNIT")
    _run_assess(tmp_path, "P_PERUNIT", monkeypatch)
    records = _sidecar_records(proj)
    # 9 units: one per (kind, terminal_id) across the three loops.
    assert set(records) == {
        f"{kind}#{tid}"
        for kind in ("quiz", "discussion", "assignment")
        for tid in ("TO-01", "TO-02", "TO-03")
    }
    for rec in records.values():
        assert isinstance(rec["fingerprint"], str) and rec["fingerprint"]


def test_resume_reuses_all_units_no_regeneration(tmp_path, monkeypatch, _armed_env):
    _keep_sidecars(monkeypatch)
    proj = _seed_project(tmp_path, "P_REUSE")
    _run_assess(tmp_path, "P_REUSE", monkeypatch)
    assert len(_FakeGen.calls) == 3  # 3 quizzes authored on the first run

    # Resume against the populated sidecar: every fingerprint matches → the
    # quiz generator is never re-dispatched.
    _reset_gen()
    _run_assess(tmp_path, "P_REUSE", monkeypatch)
    assert _FakeGen.calls == []


def test_fingerprint_mismatch_regenerates_changed_unit(
    tmp_path, monkeypatch, _armed_env
):
    _keep_sidecars(monkeypatch)
    _seed_project(tmp_path, "P_FP")
    _run_assess(tmp_path, "P_FP", monkeypatch)
    assert len(_FakeGen.calls) == 3

    # Change TO-02's statement → its quiz fingerprint (statement sha) flips, so
    # ONLY that quiz unit re-dispatches on the next run; TO-01/TO-03 reuse.
    mutated = json.loads(json.dumps(_OBJECTIVES_DOC))
    mutated["terminal_objectives"][1]["statement"] = "A completely rewritten goal."
    _seed_project(tmp_path, "P_FP", objectives=mutated)
    _reset_gen()
    _run_assess(tmp_path, "P_FP", monkeypatch)
    assert [c[0] for c in _FakeGen.calls] == ["TO-02"]


def test_torn_trailing_line_tolerated(tmp_path, monkeypatch, _armed_env):
    _keep_sidecars(monkeypatch)
    proj = _seed_project(tmp_path, "P_TORN")
    _run_assess(tmp_path, "P_TORN", monkeypatch)
    # Append a half-written trailing line, as a crash mid-append would.
    with _sidecar(proj).open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "quiz#TO-09", "finger')
    records = _sidecar_records(proj)
    assert "quiz#TO-09" not in records  # torn line skipped
    # Resume still fully reuses (torn line ignored, no re-dispatch).
    _reset_gen()
    _run_assess(tmp_path, "P_TORN", monkeypatch)
    assert _FakeGen.calls == []


def test_family_flag_opt_out_disables(tmp_path, monkeypatch, _armed_env):
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    _keep_sidecars(monkeypatch)
    proj = _seed_project(tmp_path, "P_OPTOUT")
    _run_assess(tmp_path, "P_OPTOUT", monkeypatch)
    # No sidecar written when the family flag is off.
    assert not _sidecar(proj).exists()
    # And a "resume" re-dispatches everything (no reuse).
    _reset_gen()
    _run_assess(tmp_path, "P_OPTOUT", monkeypatch)
    assert len(_FakeGen.calls) == 3


# ===========================================================================
# Graceful-stop legs
# ===========================================================================
def test_stop_after_n_manifest_not_written_sidecar_survives(
    tmp_path, monkeypatch, _armed_env
):
    _reset_gen(arm_after=2)  # arm the sentinel after the 2nd quiz
    proj = _seed_project(tmp_path, "P_STOP_N")

    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "Trainforge.generators.assessment_generator.AssessmentGenerator", _FakeGen
    )
    registry = pt._build_tool_registry()
    tool = registry["run_assessment_synthesis"]
    with pytest.raises(GracefulStopRequested):
        asyncio.run(tool(project_id="P_STOP_N"))

    # Quiz #3's loop-top check_stop raised: exactly 2 generator calls, exactly 2
    # quiz units checkpointed, and NO manifest was written.
    assert len(_FakeGen.calls) == 2
    records = _sidecar_records(proj)
    assert set(records) == {"quiz#TO-01", "quiz#TO-02"}
    assert not (proj / "06_assessments" / "manifest.json").exists()
    # The sidecar survives for the resume (only removed after the end-write).
    assert _sidecar(proj).exists()


def test_resume_after_stop_byte_equivalent(tmp_path, monkeypatch, _armed_env):
    # Baseline (separate project, uninterrupted) for the byte-equivalence oracle.
    base_proj = _seed_project(tmp_path, "P_BASE")
    _run_assess(tmp_path, "P_BASE", monkeypatch)
    base_snapshot = _output_snapshot(base_proj)
    assert "manifest.json" in base_snapshot

    # Interrupted leg (project B): stop after 2 quizzes.
    _reset_gen(arm_after=2)
    proj = _seed_project(tmp_path, "P_RESUME")
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "Trainforge.generators.assessment_generator.AssessmentGenerator", _FakeGen
    )
    registry = pt._build_tool_registry()
    tool = registry["run_assessment_synthesis"]
    with pytest.raises(GracefulStopRequested):
        asyncio.run(tool(project_id="P_RESUME"))
    assert len(_FakeGen.calls) == 2
    assert not (proj / "06_assessments" / "manifest.json").exists()

    # Resume leg: clear the sentinel, rerun the SAME project/sidecar. Only the
    # un-authored quiz (TO-03) re-dispatches; total generator calls == 3.
    stop_control.clear_stop(include_global=True)
    _reset_gen()
    env = _run_assess(tmp_path, "P_RESUME", monkeypatch)
    assert env["success"] is True
    assert [c[0] for c in _FakeGen.calls] == ["TO-03"]

    # Byte-equivalent emitted artifacts (manifest + XML) vs the uninterrupted run.
    assert _output_snapshot(proj) == base_snapshot
    # Sidecar cleared after the successful end-write.
    assert not _sidecar(proj).exists()


def test_pre_armed_sentinel_zero_calls(tmp_path, monkeypatch, _armed_env):
    proj = _seed_project(tmp_path, "P_PREARM")
    stop_control.request_stop(scope="run", reason="test", source="test")

    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "Trainforge.generators.assessment_generator.AssessmentGenerator", _FakeGen
    )
    registry = pt._build_tool_registry()
    tool = registry["run_assessment_synthesis"]
    with pytest.raises(GracefulStopRequested):
        asyncio.run(tool(project_id="P_PREARM"))

    assert _FakeGen.calls == []  # nothing generated
    assert not (proj / "06_assessments" / "manifest.json").exists()
    assert not _sidecar(proj).exists()  # no unit checkpointed
