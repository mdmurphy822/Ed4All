"""Wave 117: regression tests for incremental pilot_report.md writes
during ``run_synthesis``.

Coverage:
  * ``test_run_synthesis_writes_pilot_report_periodically`` —
    confirms the in-flight + final report writes fire on a
    multi-chunk run with a property manifest, by injecting a synthetic
    manifest via monkey-patch and counting calls to the atomic-write
    helper.
  * ``test_run_synthesis_no_pilot_report_when_no_manifest`` — confirms
    the feature is a no-op when the course has no manifest (the common
    case for courses that haven't authored a property manifest).
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
from Trainforge import synthesize_training  # noqa: E402
from Trainforge.synthesize_training import run_synthesis  # noqa: E402

FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


def _make_working_copy(tmp_path: Path) -> Path:
    """Copy the read-only fixture into a tmp dir so run_synthesis can
    write its outputs without polluting the source tree."""
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    # Clear any stale pairs from earlier test runs.
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
        dst / "training_specs" / "pilot_report.md",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def _synthetic_manifest() -> PropertyManifest:
    """A property manifest whose surface forms are guaranteed to
    appear in some / none of the mock-provider templates so the
    coverage table has a mix of PASS / FAIL rows. Real surface forms
    don't matter for this test — what matters is that the helpers run
    end-to-end."""
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

    write_calls: list[tuple[Path, str]] = []
    real_writer = synthesize_training.__dict__.get(
        "write_pilot_report_atomic"
    )  # placeholder; we patch the helper module instead.

    from Trainforge.scripts import pilot_report_helpers

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
        # Disable the curriculum-from-graph default since the fixture
        # has no pedagogy graph.
        curriculum_from_graph=False,
    )

    # The fixture has more than 5 eligible chunks, so we expect at
    # least one in-flight write + the final write.
    assert len(write_calls) >= 2, (
        f"Expected periodic + final pilot_report writes, got "
        f"{len(write_calls)}: {[p.name for p, _ in write_calls]}"
    )

    report_path = course_dir / "training_specs" / "pilot_report.md"
    assert report_path.exists(), "pilot_report.md should be written"
    content = report_path.read_text(encoding="utf-8")

    # Final report has in_flight=False so it should NOT carry the banner.
    assert "In-flight snapshot" not in content
    # Sanity: the final report has the expected sections + slug.
    assert "Property coverage" in content
    assert "Top 10 templates" in content
    assert "MINI_TRAINING_101" in content
    # And the synthesis actually emitted instruction pairs.
    assert stats.instruction_pairs_emitted > 0


def test_run_synthesis_no_pilot_report_when_no_manifest(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``load_property_manifest`` raises ``FileNotFoundError`` (the
    default case for courses without an authored manifest), the
    pilot-report writes should be silently skipped — no pilot_report.md
    file, no atomic-writer calls, and an info-level log entry."""
    course_dir = _make_working_copy(tmp_path)

    # Defensively force the FileNotFoundError path even if the
    # mini-fixture's slug ever ends up matching a real manifest.
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
    assert not report_path.exists(), (
        "pilot_report.md should NOT be written when no manifest exists"
    )
    assert write_calls == [], (
        f"Atomic writer should not be called without a manifest, got "
        f"{write_calls}"
    )
    # The logger.info("no property manifest...") line should fire.
    assert any(
        "no property manifest" in rec.message.lower()
        for rec in caplog.records
    ), "Expected info-level log about missing property manifest"
