"""Wave 116 + 117: regression tests for ``run_synthesis`` operational
features.

Wave 116 — incremental ``.jsonl.in_progress`` sidecar writes:
  * ``test_sidecar_written_incrementally_and_cleaned_up_on_success``
  * ``test_sidecar_preserved_on_budget_exceeded``

Wave 117 — incremental ``pilot_report.md`` writes:
  * ``test_run_synthesis_writes_pilot_report_periodically``
  * ``test_run_synthesis_no_pilot_report_when_no_manifest``

All four tests use ``provider="mock"`` (or a fake LocalDispatcher) so
they're fully offline + deterministic — no LLM calls, no Ollama, no
network.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.property_manifest import (  # noqa: E402
    PropertyEntry,
    PropertyManifest,
)
from Trainforge import synthesize_training  # noqa: E402, F401
from Trainforge.synthesize_training import run_synthesis  # noqa: E402


FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


def _make_working_copy(tmp_path: Path) -> Path:
    """Copy the read-only fixture into tmp so run_synthesis can write."""
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
        dst / "training_specs" / "instruction_pairs.jsonl.in_progress",
        dst / "training_specs" / "preference_pairs.jsonl.in_progress",
        dst / "training_specs" / "pilot_report.md",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def _synthetic_manifest() -> PropertyManifest:
    """A property manifest whose surface forms are guaranteed to
    appear in some / none of the mock-provider templates so the
    coverage table has a mix of PASS / FAIL rows."""
    return PropertyManifest(
        family="mini",
        properties=[
            PropertyEntry(
                id="topic_load",
                uri="http://example.test/load",
                curie="ex:load",
                label="Cognitive load surface form",
                surface_forms=["load"],
                min_pairs=5,
            ),
            PropertyEntry(
                id="topic_zzz",
                uri="http://example.test/zzz",
                curie="ex:zzz",
                label="Sentinel surface form that never appears",
                surface_forms=["zzz_no_match_sentinel_phrase"],
                min_pairs=5,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Wave 116: sidecar incremental write
# ---------------------------------------------------------------------------


def test_sidecar_written_incrementally_and_cleaned_up_on_success(
    tmp_path: Path,
) -> None:
    """A clean ``run_synthesis`` invocation MUST leave no sidecars on
    disk after writing the final atomic JSONL artifacts."""
    working = _make_working_copy(tmp_path)
    inst_progress = (
        working / "training_specs" / "instruction_pairs.jsonl.in_progress"
    )
    pref_progress = (
        working / "training_specs" / "preference_pairs.jsonl.in_progress"
    )
    inst_final = working / "training_specs" / "instruction_pairs.jsonl"
    pref_final = working / "training_specs" / "preference_pairs.jsonl"

    assert not inst_progress.exists()
    assert not pref_progress.exists()

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    assert inst_final.exists()
    assert pref_final.exists()
    assert stats.instruction_pairs_emitted > 0
    assert stats.preference_pairs_emitted > 0
    inst_lines = [
        l for l in inst_final.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    pref_lines = [
        l for l in pref_final.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert len(inst_lines) == stats.instruction_pairs_emitted
    assert len(pref_lines) == stats.preference_pairs_emitted

    assert not inst_progress.exists(), (
        "Wave 116 contract: instruction sidecar must be deleted on a "
        f"clean run; found it at {inst_progress}"
    )
    assert not pref_progress.exists(), (
        "Wave 116 contract: preference sidecar must be deleted on a "
        f"clean run; found it at {pref_progress}"
    )

    assert stats.capped_at_max_dispatches is False


def test_sidecar_preserved_on_budget_exceeded(tmp_path: Path) -> None:
    """When the chunk loop raises ``SynthesisBudgetExceeded``, the
    sidecars MUST be preserved so the operator can inspect partial
    output."""
    from Trainforge.tests._synthesis_fakes import (
        FakeLocalDispatcher,
        make_instruction_response,
        make_preference_response,
    )

    _ok_p = "Paraphrased prompt explaining RDFS in detail for the learner."
    _ok_c = (
        "Paraphrased completion grounded in the source chunk text "
        "covering RDFS and SHACL contracts in sufficient detail."
    )

    async def agent_tool(*, task_params, **_kw):
        if task_params["kind"] == "instruction":
            return make_instruction_response(prompt=_ok_p, completion=_ok_c)
        return make_preference_response(prompt=_ok_p, chosen=_ok_c, rejected=_ok_c)

    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    working = _make_working_copy(tmp_path)
    inst_progress = (
        working / "training_specs" / "instruction_pairs.jsonl.in_progress"
    )
    pref_progress = (
        working / "training_specs" / "preference_pairs.jsonl.in_progress"
    )

    assert not inst_progress.exists()
    assert not pref_progress.exists()

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="claude_session",
        seed=11,
        dispatcher=dispatcher,
        max_dispatches=1,
    )

    assert stats.capped_at_max_dispatches is True

    progress_path = working / "training_specs" / "pilot_progress.json"
    assert progress_path.exists()

    assert inst_progress.exists() or pref_progress.exists(), (
        "Wave 116 contract: at least one sidecar must be preserved "
        "for postmortem on a SynthesisBudgetExceeded exit"
    )
    if inst_progress.exists():
        content = inst_progress.read_text(encoding="utf-8")
        if stats.instruction_pairs_emitted > 0:
            assert content.strip(), (
                "instruction sidecar exists but is empty; flush() not exercised"
            )


# ---------------------------------------------------------------------------
# Wave 117: incremental pilot_report.md
# ---------------------------------------------------------------------------


def test_run_synthesis_writes_pilot_report_periodically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_synthesis with pilot_report_every=5 should call the atomic
    writer multiple times (in-flight every 5 chunks + a final write at
    the end), and produce a pilot_report.md whose final content is the
    non-banner snapshot."""
    course_dir = _make_working_copy(tmp_path)

    manifest = _synthetic_manifest()
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *_a, **_kw: manifest,
    )

    from Trainforge.scripts import pilot_report_helpers

    write_calls: list[tuple[Path, str]] = []
    original_writer = pilot_report_helpers.write_pilot_report_atomic

    def _capturing_writer(path: Path, content: str) -> None:
        write_calls.append((Path(path), content))
        original_writer(path, content)

    monkeypatch.setattr(
        pilot_report_helpers, "write_pilot_report_atomic", _capturing_writer,
    )

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=5,
        curriculum_from_graph=False,
    )

    assert len(write_calls) >= 2, (
        f"Expected periodic + final pilot_report writes, got "
        f"{len(write_calls)}: {[p.name for p, _ in write_calls]}"
    )

    report_path = course_dir / "training_specs" / "pilot_report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    assert "In-flight snapshot" not in content
    assert "Property coverage" in content
    assert "Top 10 templates" in content
    assert "MINI_TRAINING_101" in content
    assert stats.instruction_pairs_emitted > 0


def test_run_synthesis_writes_final_pilot_report_when_pilot_every_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 119 contract: setting ``--pilot-report-every 0`` disables
    the in-flight cadence but MUST NOT disable the final post-run
    write. An operator who turned off mid-run noise should still see
    the post-run summary on disk."""
    course_dir = _make_working_copy(tmp_path)

    manifest = _synthetic_manifest()
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *_a, **_kw: manifest,
    )

    from Trainforge.scripts import pilot_report_helpers

    write_calls: list[Path] = []
    original_writer = pilot_report_helpers.write_pilot_report_atomic

    def _tracking_writer(path: Path, content: str) -> None:
        write_calls.append(Path(path))
        original_writer(path, content)

    monkeypatch.setattr(
        pilot_report_helpers, "write_pilot_report_atomic", _tracking_writer,
    )

    run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
    )

    report_path = course_dir / "training_specs" / "pilot_report.md"
    assert report_path.exists(), (
        "Wave 119: final pilot_report.md must be written even when "
        "--pilot-report-every is 0"
    )
    assert len(write_calls) == 1, (
        f"Expected exactly one (final) atomic write, got "
        f"{len(write_calls)}: {[p.name for p in write_calls]}"
    )
    content = report_path.read_text(encoding="utf-8")
    assert "In-flight snapshot" not in content
    assert "Property coverage" in content


def test_run_synthesis_pilot_report_includes_cap_banner_when_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 119 contract: when ``--max-pairs`` clips the run,
    pilot_report.md MUST carry a loud banner so an operator opening
    the file can't miss that property floors are evaluated against a
    truncated run (the failure mode that bit Wave 118)."""
    course_dir = _make_working_copy(tmp_path)

    manifest = _synthetic_manifest()
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *_a, **_kw: manifest,
    )

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        max_pairs=2,
        pilot_report_every=0,
        curriculum_from_graph=False,
    )

    assert stats.capped_at_max_pairs is True
    assert stats.max_pairs_cap == 2

    report_path = course_dir / "training_specs" / "pilot_report.md"
    content = report_path.read_text(encoding="utf-8")
    assert "WARNING" in content, (
        "Wave 119: capped run must surface a WARNING banner in "
        "pilot_report.md"
    )
    assert "cap=2" in content
    assert "--max-pairs" in content


def test_run_synthesis_logs_warning_when_max_pairs_clips_eligible_chunks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Wave 119 contract: a pre-flight WARNING fires when ``max_pairs``
    is below the eligible-chunks count, so the operator sees the
    issue at run start (not at end-of-run when 4 hours of compute have
    already burned)."""
    course_dir = _make_working_copy(tmp_path)

    with caplog.at_level(logging.WARNING, logger="Trainforge.synthesize_training"):
        run_synthesis(
            corpus_dir=course_dir,
            course_code="MINI_TRAINING_101",
            provider="mock",
            seed=11,
            max_pairs=3,
            pilot_report_every=0,
            curriculum_from_graph=False,
        )

    assert any(
        "will clip this run" in rec.message
        and "Property-coverage gates may underreport" in rec.message
        for rec in caplog.records
    ), (
        "Expected a Wave 119 pre-flight cap warning; got "
        f"{[rec.message for rec in caplog.records]}"
    )


def test_run_synthesis_no_pilot_report_when_no_manifest(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``load_property_manifest`` raises ``FileNotFoundError``, the
    pilot-report writes should be silently skipped — no pilot_report.md
    file, no atomic-writer calls, and an info-level log entry."""
    course_dir = _make_working_copy(tmp_path)

    def _raise(*_a, **_kw):
        raise FileNotFoundError("no manifest for test slug")

    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest", _raise,
    )

    from Trainforge.scripts import pilot_report_helpers

    write_calls: list[Path] = []
    original_writer = pilot_report_helpers.write_pilot_report_atomic

    def _tracking_writer(path: Path, content: str) -> None:
        write_calls.append(Path(path))
        original_writer(path, content)

    monkeypatch.setattr(
        pilot_report_helpers, "write_pilot_report_atomic", _tracking_writer,
    )

    with caplog.at_level(logging.INFO, logger="Trainforge.synthesize_training"):
        run_synthesis(
            corpus_dir=course_dir,
            course_code="bogus-course-no-manifest",
            provider="mock",
            seed=11,
            pilot_report_every=5,
            curriculum_from_graph=False,
        )

    report_path = course_dir / "training_specs" / "pilot_report.md"
    assert not report_path.exists()
    assert write_calls == []
    assert any(
        "no property manifest" in rec.message.lower()
        for rec in caplog.records
    ), "Expected info-level log about missing property manifest"


# ---------------------------------------------------------------------------
# Wave 120: schema realignment regression — zero validation_issues
# ---------------------------------------------------------------------------


def test_run_synthesis_emits_zero_validation_issues(tmp_path: Path) -> None:
    """Wave 120 schema realignment: every decision event emitted by a
    synthesis run must have an empty (or absent) ``metadata.validation_issues``
    list. Three drift points were closing on prior runs:

      * ``phase="synthesize-training"`` was missing from the schema enum.
      * ``course_id="RDF-SHACL-551-2"`` failed the underscore-only pattern.
      * ``alternatives_considered`` items were strings, schema expects objects.

    All three are now schema-clean. This test asserts the contract.
    """
    import os
    os.environ["VALIDATE_DECISIONS"] = "true"
    from lib.decision_capture import DecisionCapture

    course_dir = _make_working_copy(tmp_path)
    capture = DecisionCapture(
        course_code="rdf-shacl-551-2",
        phase="synthesize-training",
        tool="trainforge",
    )

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="mock",
        seed=11,
        capture=capture,
        pilot_report_every=0,
        curriculum_from_graph=False,
    )

    assert stats.instruction_pairs_emitted > 0
    assert capture.decisions, "synthesis emitted no decision events"
    failing: list[tuple[str, list]] = []
    for rec in capture.decisions:
        meta = rec.get("metadata") or {}
        issues = meta.get("validation_issues") or []
        if issues:
            failing.append((rec.get("decision_type", "?"), issues))

    assert not failing, (
        f"{len(failing)} of {len(capture.decisions)} decision events carry "
        f"validation_issues. First 3: {failing[:3]!r}"
    )
