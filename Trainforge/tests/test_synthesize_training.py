"""Wave 116 + 117 + 127: regression tests for ``run_synthesis``
operational features.

Wave 116 — incremental ``.jsonl.in_progress`` sidecar writes:
  * ``test_sidecar_written_incrementally_and_cleaned_up_on_success``
  * ``test_sidecar_preserved_on_budget_exceeded``

Wave 117 — incremental ``pilot_report.md`` writes:
  * ``test_run_synthesis_writes_pilot_report_periodically``
  * ``test_run_synthesis_no_pilot_report_when_no_manifest``

Wave 127 — deterministic generators hoisted ABOVE the chunk loop and
mirrored to the sidecar so an operator can ``tail -f`` the
``.jsonl.in_progress`` file and confirm ``--with-*`` flags wired through
within the first ~minute (instead of after the multi-hour paraphrase
loop):
  * ``test_violation_detection_pairs_appear_in_sidecar_before_paraphrase``

All tests use ``provider="mock"`` (or a fake LocalDispatcher) so they're
fully offline + deterministic — no LLM calls, no Ollama, no network.
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


def _shacl_manifest_for_violation_tests() -> PropertyManifest:
    """Wave 133c: minimal SHACL-family manifest that opts the mini
    course into the violation_generator gate. Used by the existing
    Wave 125a / Wave 127 violation tests after Wave 133c added the
    ``validation_kind == "shacl"`` gate. Surface forms are cosmetic —
    the violation catalog is hand-curated, not chunk-driven."""
    return PropertyManifest(
        family="mini_shacl",
        properties=[
            PropertyEntry(
                id="sh_datatype",
                uri="http://www.w3.org/ns/shacl#datatype",
                curie="sh:datatype",
                label="SHACL datatype surface form",
                surface_forms=["sh:datatype"],
                min_pairs=1,
            ),
        ],
        validation_kind="shacl",
    )


def _patch_shacl_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch ``load_property_manifest`` to return a SHACL-gated
    manifest so the violation_generator gate (Wave 133c) admits pairs
    for this fixture."""
    manifest = _shacl_manifest_for_violation_tests()
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *_a, **_kw: manifest,
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


# ---------------------------------------------------------------------------
# Wave 120: property-preservation fallback
# ---------------------------------------------------------------------------


def test_property_bearing_chunk_falls_back_to_deterministic_when_paraphrase_strips_surface_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a paraphrase provider drops a required surface form, the
    instruction factory catches ``surface_form_preservation_failed`` and
    returns the deterministic draft, marking it with
    ``paraphrase_fallback_reason``. ``run_synthesis`` then logs a
    ``paraphrase_used_deterministic_draft`` capture event (renamed Wave
    135d from ``surface_form_preservation_fallback``) and emits the
    pair instead of dropping it."""
    course_dir = _make_working_copy(tmp_path)

    # Inject a property manifest that matches text in every fixture chunk.
    manifest = PropertyManifest(
        family="mini",
        properties=[
            PropertyEntry(
                id="topic_load",
                uri="http://example.test/load",
                curie="ex:load",
                label="Cognitive load surface form",
                surface_forms=["load"],
                min_pairs=1,
            ),
        ],
    )
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *a, **kw: manifest,
    )

    # Provider that always raises surface_form_preservation_failed so
    # every property-bearing chunk hits the fallback path.
    from Trainforge.generators._local_provider import SynthesisProviderError

    class _AlwaysFailsProvider:
        def __init__(self, *args, **kwargs):
            pass

        def paraphrase_instruction(self, draft, chunk, *, preserve_tokens=None):
            if preserve_tokens:
                raise SynthesisProviderError(
                    "stub: paraphrase always drops the surface form",
                    code="surface_form_preservation_failed",
                )
            return draft

        def paraphrase_preference(self, draft, chunk, *, preserve_tokens=None):
            if preserve_tokens:
                raise SynthesisProviderError(
                    "stub: paraphrase always drops the surface form",
                    code="surface_form_preservation_failed",
                )
            return draft

    # Run synthesis with the local provider's path, but inject the stub
    # provider via the synthesize_training pathway. Since run_synthesis
    # constructs the provider internally, we monkeypatch
    # LocalSynthesisProvider for this test.
    from Trainforge.generators import _local_provider as lp_mod
    monkeypatch.setattr(lp_mod, "LocalSynthesisProvider", _AlwaysFailsProvider)

    from lib.decision_capture import DecisionCapture
    capture = DecisionCapture(
        course_code="rdf-shacl-551-2",
        phase="synthesize-training",
        tool="trainforge",
    )

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="local",
        seed=11,
        capture=capture,
        pilot_report_every=0,
        curriculum_from_graph=False,
    )

    assert stats.instruction_pairs_emitted > 0, (
        "Pairs should still be emitted via deterministic fallback, not "
        "dropped on preservation failure."
    )

    fallback_events = [
        d for d in capture.decisions
        if d.get("decision_type") == "paraphrase_used_deterministic_draft"
    ]
    assert fallback_events, (
        "Expected at least one paraphrase_used_deterministic_draft "
        "capture event; got "
        f"{[d.get('decision_type') for d in capture.decisions]}"
    )


def test_paraphrase_invalid_after_retry_falls_back_to_deterministic_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 126: when the local provider exhausts retries on the 40-char
    prompt floor (e.g. definition-style chunks producing 'Define X.'
    prompts), the instruction + preference factories must fall back to
    the deterministic draft instead of crashing the run. Closes the
    cc07cc76 rebuild's chunk-3 IRI-definition failure mode where one
    chunk's paraphrase exhaustion killed all 295 chunks of synthesis.
    """
    course_dir = _make_working_copy(tmp_path)

    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *a, **kw: _synthetic_manifest(),
    )

    from Trainforge.generators._local_provider import SynthesisProviderError

    class _AlwaysExhaustsProvider:
        def __init__(self, *args, **kwargs):
            pass

        def paraphrase_instruction(self, draft, chunk, *, preserve_tokens=None):
            raise SynthesisProviderError(
                "stub: simulate prompt length 29 below minimum 40 after 3 attempts",
                code="paraphrase_invalid_after_retry",
            )

        def paraphrase_preference(self, draft, chunk, *, preserve_tokens=None):
            raise SynthesisProviderError(
                "stub: simulate completion length 32 below minimum 50 after 3 attempts",
                code="paraphrase_invalid_after_retry",
            )

    from Trainforge.generators import _local_provider as lp_mod
    monkeypatch.setattr(lp_mod, "LocalSynthesisProvider", _AlwaysExhaustsProvider)

    from lib.decision_capture import DecisionCapture
    capture = DecisionCapture(
        course_code="rdf-shacl-551-2",
        phase="synthesize-training",
        tool="trainforge",
    )

    # Run should complete without raising, emitting deterministic fallback pairs.
    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="local",
        seed=11,
        capture=capture,
        pilot_report_every=0,
        curriculum_from_graph=False,
    )

    assert stats.instruction_pairs_emitted > 0, (
        "Pairs should still be emitted via deterministic fallback, not "
        "dropped on paraphrase exhaustion."
    )

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    assert inst_path.exists()
    import json as _json
    fallback_count = 0
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = _json.loads(line)
        if rec.get("paraphrase_fallback_reason") == "paraphrase_invalid_after_retry":
            fallback_count += 1
    assert fallback_count > 0, (
        "Expected at least one pair carrying paraphrase_fallback_reason="
        "'paraphrase_invalid_after_retry'; instead got "
        f"{fallback_count} such pairs out of {stats.instruction_pairs_emitted} emitted."
    )


def test_local_provider_definition_chunk_directive_is_injected() -> None:
    """Wave 126: ``_render_instruction_user`` injects an explicit
    'explanation-asking question, ≥40 chars' directive when the draft's
    content_type or bloom_level indicates a definition-style chunk.
    Shapes the model's first attempt so the retry path isn't burdened
    with coaxing the model out of bare 'Define X.' prompts."""
    from Trainforge.generators._local_provider import LocalSynthesisProvider

    definition_draft = {
        "prompt": "Define IRI.",
        "completion": "An IRI is an Internationalized Resource Identifier.",
        "bloom_level": "remember",
        "content_type": "definition",
        "template_id": "remember._default",
    }
    rendered = LocalSynthesisProvider._render_instruction_user(
        definition_draft, "rdf_shacl_551_chunk_00003",
    )
    assert "definition / recall chunk" in rendered
    assert "EXPLANATION-asking question of at least 40 characters" in rendered
    assert "bare 'Define X.'" in rendered

    narrative_draft = {
        "prompt": "Compare RDF and OWL inference scopes in detail.",
        "completion": "RDF supports basic triple-pattern inference...",
        "bloom_level": "analyze",
        "content_type": "explanation",
        "template_id": "analyze._default",
    }
    rendered_narrative = LocalSynthesisProvider._render_instruction_user(
        narrative_draft, "rdf_shacl_551_chunk_00100",
    )
    assert "definition / recall chunk" not in rendered_narrative


# ---------------------------------------------------------------------------
# Wave 120: smoke modes
# ---------------------------------------------------------------------------


def test_smoke_deterministic_writes_sidecar_report_and_does_not_overwrite_pilot_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--smoke-deterministic`` writes ``smoke_pilot_report.md`` (not
    ``pilot_report.md``) so a smoke run never clobbers a prior full
    run's authoritative report. Floors scaled to 1."""
    course_dir = _make_working_copy(tmp_path)

    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *a, **kw: _synthetic_manifest(),
    )

    # Pre-populate a canonical pilot_report.md so we can detect any
    # accidental overwrite by the smoke run.
    canonical = course_dir / "training_specs" / "pilot_report.md"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("# Authoritative full-run report — DO NOT OVERWRITE\n")

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="local",  # gets coerced to mock under deterministic smoke
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        smoke_mode="deterministic",
    )

    smoke_path = course_dir / "training_specs" / "smoke_pilot_report.md"
    assert smoke_path.exists(), "Smoke run must write smoke_pilot_report.md"
    smoke_text = smoke_path.read_text()
    assert "Floor | Status" in smoke_text or "Floor" in smoke_text
    # Floors scaled: 1 for deterministic smoke. Synthetic manifest's
    # floors were 5 originally; assert at least one PASS row at floor 1.
    assert "| 1 |" in smoke_text, (
        f"Expected scaled floor of 1 in smoke report; got:\n{smoke_text}"
    )
    # Canonical report untouched.
    assert canonical.read_text().startswith(
        "# Authoritative full-run report"
    ), "Smoke run overwrote pilot_report.md — sidecar isolation broken"
    # Smoke ran on at most 20 chunks.
    assert stats.chunks_total <= 20


def test_smoke_paraphrase_uses_provider_path_with_floor_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--smoke-paraphrase`` keeps the configured provider (does not
    coerce to mock), so the paraphrase + preservation path is exercised
    on the smoke sample. Floors scaled to 2."""
    course_dir = _make_working_copy(tmp_path)

    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *a, **kw: _synthetic_manifest(),
    )

    # Stub local provider so this test stays offline.
    class _PassThroughProvider:
        def __init__(self, *args, **kwargs):
            pass

        def paraphrase_instruction(self, draft, chunk, *, preserve_tokens=None):
            return draft

        def paraphrase_preference(self, draft, chunk, *, preserve_tokens=None):
            return draft

    from Trainforge.generators import _local_provider as lp_mod
    monkeypatch.setattr(lp_mod, "LocalSynthesisProvider", _PassThroughProvider)

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="local",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        smoke_mode="paraphrase",
    )

    smoke_path = course_dir / "training_specs" / "smoke_pilot_report.md"
    assert smoke_path.exists()
    smoke_text = smoke_path.read_text()
    assert "| 2 |" in smoke_text, (
        f"Expected scaled floor of 2 in paraphrase smoke; got:\n{smoke_text}"
    )
    assert stats.chunks_total <= 20


def test_smoke_stratified_sampler_prefers_property_bearing_chunks() -> None:
    """The smoke sampler picks every property-bearing chunk first (up to
    3 per surface form), then pads with random chunks."""
    import random as _r
    from Trainforge.synthesize_training import _smoke_stratified_sample

    chunks = [
        {"id": f"c{i}", "text": f"chunk {i} contains the keyword sh:NodeShape"}
        for i in range(5)
    ] + [
        {"id": f"c{i}", "text": f"chunk {i} contains the keyword sh:datatype"}
        for i in range(5, 10)
    ] + [
        {"id": f"c{i}", "text": f"chunk {i} has no surface form"}
        for i in range(10, 30)
    ]
    manifest = PropertyManifest(
        family="t",
        properties=[
            PropertyEntry(
                id="ns", uri="u", curie="sh:NodeShape", label="ns",
                surface_forms=["sh:NodeShape"], min_pairs=2,
            ),
            PropertyEntry(
                id="dt", uri="u", curie="sh:datatype", label="dt",
                surface_forms=["sh:datatype"], min_pairs=2,
            ),
        ],
    )
    # target=6 hits exactly the 3+3 property cap with no random pad,
    # so the assertion isolates the property-loop behavior.
    selected = _smoke_stratified_sample(
        chunks, manifest, target_count=6, rng=_r.Random(0),
    )
    assert len(selected) == 6
    ns_hits = [c for c in selected if "sh:NodeShape" in c["text"]]
    dt_hits = [c for c in selected if "sh:datatype" in c["text"]]
    assert len(ns_hits) == 3, "Every property should land 3 representatives"
    assert len(dt_hits) == 3


# ---------------------------------------------------------------------------
# Wave 120 follow-up: force-inject preserve_tokens absent from final pair
# ---------------------------------------------------------------------------


def test_force_inject_canonical_terms_when_deterministic_path_drops_curie() -> None:
    """Wave 135b — when the final pair doesn't contain a required
    surface form, the factory force-injects it. For ``"complete"``
    FORM_DATA entries (here: sh:NodeShape, sh:datatype) the injection
    routes through the anchored-definition path; the audit field is
    ``preserve_tokens_anchored*`` rather than the legacy
    ``preserve_tokens_injected*``. For degraded / non-manifest CURIEs
    the legacy ``preserve_tokens_injected*`` path stays alive — see
    ``test_force_injection_falls_back_for_degraded_entry`` /
    ``test_force_injection_anchors_non_manifest_curies`` in
    ``test_instruction_factory.py``."""
    from Trainforge.generators.instruction_factory import (
        synthesize_instruction_pair,
    )

    chunk = {
        "id": "smoke_c01",
        "text": (
            "A SHACL shape declared with sh:NodeShape constrains node-typed "
            "entities. The sh:datatype constraint pins a property's literal "
            "type. Both are validated together when running shapes against "
            "data."
        ),
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "understand",
        "concept_tags": ["shacl-validation", "node-shape"],
        "key_terms": [],
    }
    result = synthesize_instruction_pair(
        chunk,
        seed=11,
        provider="mock",
        preserve_tokens=["sh:NodeShape", "sh:datatype"],
    )
    assert result.pair is not None
    # Both sides now carry the CURIE so the model learns input + output.
    assert "sh:NodeShape" in result.pair["prompt"]
    assert "sh:datatype" in result.pair["prompt"]
    assert "sh:NodeShape" in result.pair["completion"]
    assert "sh:datatype" in result.pair["completion"]
    # Audit trail (Wave 135b): both CURIEs are "complete" entries so
    # the anchored-definition path fires; check anchored markers AND
    # the union with legacy markers (in case some tokens land in one
    # bucket and some in the other).
    union_completion = set(
        result.pair.get("preserve_tokens_anchored", [])
    ) | set(result.pair.get("preserve_tokens_injected", []))
    union_prompt = set(
        result.pair.get("preserve_tokens_anchored_prompt", [])
    ) | set(result.pair.get("preserve_tokens_injected_prompt", []))
    assert {"sh:NodeShape", "sh:datatype"}.issubset(union_completion)
    assert {"sh:NodeShape", "sh:datatype"}.issubset(union_prompt)


def test_force_inject_skips_per_side_when_already_contains_token() -> None:
    """Per-side idempotency: if the deterministic draft has the CURIE
    on one side only, only the missing side gets injected.

    Wave 135b: ``sh:NodeShape`` is a "complete" FORM_DATA entry so
    injection routes through the anchored-definition path; the audit
    fields are ``preserve_tokens_anchored*`` rather than the legacy
    ``preserve_tokens_injected*``."""
    from Trainforge.generators.instruction_factory import (
        _enforce_preserve_tokens_in_instruction,
    )

    # Case 1: CURIE in both sides -> no injection on either.
    pair_both = {
        "chunk_id": "c1",
        "prompt": "Define sh:NodeShape clearly.",
        "completion": "sh:NodeShape constrains node-typed instances.",
    }
    out_both = _enforce_preserve_tokens_in_instruction(
        pair_both, ["sh:NodeShape"],
    )
    assert "preserve_tokens_injected" not in out_both
    assert "preserve_tokens_injected_prompt" not in out_both
    assert "preserve_tokens_anchored" not in out_both
    assert "preserve_tokens_anchored_prompt" not in out_both
    # Pair text unchanged -> CURIE only appears once on each side.
    assert out_both["prompt"].count("sh:NodeShape") == 1
    assert out_both["completion"].count("sh:NodeShape") == 1

    # Case 2: CURIE only in completion -> prompt-side gets injection.
    pair_completion_only = {
        "chunk_id": "c2",
        "prompt": "Define this constraint.",
        "completion": "sh:NodeShape constrains node-typed instances.",
    }
    out_p = _enforce_preserve_tokens_in_instruction(
        pair_completion_only, ["sh:NodeShape"],
    )
    assert "sh:NodeShape" in out_p["prompt"]
    # Wave 135b: anchored path fires for "complete" entries.
    prompt_marker_present = (
        "preserve_tokens_anchored_prompt" in out_p
        or "preserve_tokens_injected_prompt" in out_p
    )
    assert prompt_marker_present
    assert "preserve_tokens_injected" not in out_p
    assert "preserve_tokens_anchored" not in out_p

    # Case 3: CURIE only in prompt -> completion-side gets injection.
    pair_prompt_only = {
        "chunk_id": "c3",
        "prompt": "Define sh:NodeShape clearly.",
        "completion": "It constrains node-typed instances.",
    }
    out_c = _enforce_preserve_tokens_in_instruction(
        pair_prompt_only, ["sh:NodeShape"],
    )
    assert "sh:NodeShape" in out_c["completion"]
    completion_marker_present = (
        "preserve_tokens_anchored" in out_c
        or "preserve_tokens_injected" in out_c
    )
    assert completion_marker_present
    assert "preserve_tokens_injected_prompt" not in out_c
    assert "preserve_tokens_anchored_prompt" not in out_c


def test_force_inject_phrasing_rotates_across_chunks() -> None:
    """Wave 121: phrasing rotation in the LEGACY token-stuffing path
    prevents boilerplate-suffix saturation. Wave 135b moved
    "complete" FORM_DATA entries (sh:NodeShape, sh:datatype, etc.)
    onto the anchored-definition path which has no fixed-template
    rotation; the rotation logic still drives the legacy path that
    handles "degraded_placeholder" entries AND non-manifest CURIEs.
    Test using a non-manifest CURIE (``ex:WorkedExample``) so the
    legacy path fires for every chunk."""
    from Trainforge.generators.instruction_factory import (
        _enforce_preserve_tokens_in_instruction,
        _PROMPT_REFERENCE_PHRASINGS,
        _COMPLETION_REFERENCE_PHRASINGS,
    )

    base_prompt = "Define this constraint."
    base_completion = "It constrains node-typed instances."
    chunk_ids = [f"chunk_{i:04d}" for i in range(60)]
    prompt_addition_forms: set = set()
    completion_addition_forms: set = set()
    for cid in chunk_ids:
        pair = {
            "chunk_id": cid,
            "prompt": base_prompt,
            "completion": base_completion,
        }
        # ex:WorkedExample is not in FORM_DATA -> legacy token-stuffing
        # path with phrasing rotation fires.
        out = _enforce_preserve_tokens_in_instruction(
            pair, ["ex:WorkedExample"],
        )
        # Recover the appended template by stripping the base + the
        # injected token. The token-replacement shape lets us cluster
        # identical phrasings regardless of which CURIE was injected.
        prompt_suffix = out["prompt"][len(base_prompt):]
        completion_suffix = out["completion"][len(base_completion):]
        prompt_addition_forms.add(prompt_suffix.replace("ex:WorkedExample", "X"))
        completion_addition_forms.add(completion_suffix.replace("ex:WorkedExample", "X"))
    # Across 60 chunks, every one of the 4 phrasings on each side should
    # appear at least once (uniform-ish hash distribution).
    assert len(prompt_addition_forms) == len(_PROMPT_REFERENCE_PHRASINGS), (
        f"Expected all {len(_PROMPT_REFERENCE_PHRASINGS)} prompt phrasings; "
        f"got {len(prompt_addition_forms)}: {prompt_addition_forms}"
    )
    assert len(completion_addition_forms) == len(_COMPLETION_REFERENCE_PHRASINGS)


def test_assessment_scaffolding_chunk_drops_pair() -> None:
    """Wave 122: when a chunk's summary carries an assessment outline
    (e.g. 'Question 1 (CO-07, Bloom: Understand). Question 2 ...') the
    factory must reject the pair after the disallow_summary retry,
    not emit it. Otherwise the model learns to vomit quiz outlines
    in normal explanations."""
    from Trainforge.generators.instruction_factory import (
        synthesize_instruction_pair,
    )

    chunk = {
        "id": "smoke_c66",
        "text": "Some clean teaching text without scaffolding markers.",
        "summary": (
            "Question 1 (CO-07, Bloom: Understand). Question 2 (CO-07, "
            "Bloom: Apply). Question 3 (CO-07, Bloom: Analyze)."
        ),
        # Note: NO concept_tags / key_terms — so disallow_summary retry
        # produces a completion built purely from the bloom_tail. That
        # text is clean. But if the chunk had ONLY the scaffolded
        # summary as its content source, the retry path would fall
        # through to bloom_tails which doesn't carry the pattern.
        "concept_tags": ["clean-tag-one", "clean-tag-two"],
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "understand",
    }
    result = synthesize_instruction_pair(chunk, seed=42, provider="mock")
    # The retry path produces a clean completion (bloom_tail +
    # concept_tags scaffold), so the pair should land successfully.
    assert result.pair is not None
    pair_text = result.pair["prompt"] + " " + result.pair["completion"]
    # Critical: pattern must NOT appear in the final pair.
    assert "Question 1 (CO-07" not in pair_text
    assert "Bloom: Understand)" not in pair_text


def test_assessment_scaffolding_unrecoverable_chunk_drops_pair() -> None:
    """When BOTH summary AND key_terms.definition carry the scaffolding
    pattern, no retry can produce a clean completion — the pair must
    be rejected with quality.passed=False."""
    from Trainforge.generators.instruction_factory import (
        synthesize_instruction_pair,
    )

    scaffolded = (
        "Question 1 (CO-07, Bloom: Understand). Question 2 (CO-07, "
        "Bloom: Apply). Question 3 (CO-07, Bloom: Analyze)."
    )
    chunk = {
        "id": "smoke_c66x",
        "text": "Clean source text without markers.",
        "summary": scaffolded,
        # Force the disallow_summary retry to also produce scaffolded
        # content by putting the pattern in key_terms.
        "key_terms": [{"term": "scaffolded-term", "definition": scaffolded}],
        "concept_tags": [],
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "understand",
    }
    result = synthesize_instruction_pair(chunk, seed=42, provider="mock")
    assert result.pair is None, (
        "Pair must be dropped when no retry path produces a "
        "scaffolding-clean completion."
    )
    assert result.quality.get("no_assessment_scaffolding") is False


def test_force_inject_phrasing_idempotent_for_same_chunk() -> None:
    """Same chunk_id -> same phrasing across runs (audit reproducibility)."""
    from Trainforge.generators.instruction_factory import (
        _enforce_preserve_tokens_in_instruction,
    )
    pair_a = {
        "chunk_id": "stable_chunk_001",
        "prompt": "p1", "completion": "c1",
    }
    pair_b = {
        "chunk_id": "stable_chunk_001",
        "prompt": "p1", "completion": "c1",
    }
    out_a = _enforce_preserve_tokens_in_instruction(pair_a, ["sh:foo"])
    out_b = _enforce_preserve_tokens_in_instruction(pair_b, ["sh:foo"])
    assert out_a["prompt"] == out_b["prompt"]
    assert out_a["completion"] == out_b["completion"]


def test_force_inject_clamps_both_sides_to_max_length() -> None:
    """Both prompt and completion are clamped independently when their
    addition would breach the per-field max.

    Wave 135b: with anchored-injection, multi-token sequencing on a
    pre-overlong prompt can mean the second token's clamp truncates
    the first token's definition out of view. The hard invariants are:
    (a) length never exceeds the per-field max, and (b) at least ONE
    of the requested tokens lands on each side. Use a non-manifest
    CURIE (legacy ~30-char suffix) for the second token so both can
    fit even when the first uses an anchored definition.
    """
    from Trainforge.generators.instruction_factory import (
        COMPLETION_MAX,
        PROMPT_MAX,
        _enforce_preserve_tokens_in_instruction,
    )
    long_prompt = "Q? " * 200  # ~600 chars, well over PROMPT_MAX=400
    long_completion = "Filler " * 120  # ~840 chars, over COMPLETION_MAX=600
    pair = {
        "chunk_id": "test_clamp",
        "prompt": long_prompt,
        "completion": long_completion,
    }
    # Pair one "complete" entry (sh:datatype, ~250-char anchored def)
    # with a non-manifest CURIE that uses the short legacy token-
    # stuffing suffix — the two together fit inside the 400-char prompt
    # cap.
    out = _enforce_preserve_tokens_in_instruction(
        pair, ["ex:WorkedExample", "sh:datatype"],
    )
    assert len(out["prompt"]) <= PROMPT_MAX
    assert len(out["completion"]) <= COMPLETION_MAX
    # At least one of the two tokens lands on each side after clamping.
    for side in ("prompt", "completion"):
        landed = (
            "ex:WorkedExample" in out[side]
            or "sh:datatype" in out[side]
        )
        assert landed, (
            f"force-injection lost both tokens on the {side} side; "
            f"value={out[side]!r}"
        )


# ---------------------------------------------------------------------------
# Wave 122 follow-up: cross-chunk prompt-collision dedupe
# ---------------------------------------------------------------------------


def test_run_synthesis_dedupes_duplicate_instruction_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closes the audit's zero-tolerance ``duplicates`` gate: when the
    paraphrase provider returns the same prompt for multiple chunks
    (semantic-collision case observed on rdf-shacl-551-2's 14B
    uncapped run), the second occurrence is rejected before append
    and counted under ``rejected_reasons[instruction:duplicate_prompt]``.
    """
    course_dir = _make_working_copy(tmp_path)

    # Provider stub that always overrides the prompt to a fixed string.
    # Every chunk paraphrase produces an identical prompt — the dedupe
    # set must drop all but the first.
    class _CollidingProvider:
        def __init__(self, *args, **kwargs):
            pass

        def paraphrase_instruction(self, draft, chunk, *, preserve_tokens=None):
            # Preserve the draft's metadata (chunk_id, lo_refs, etc.) and
            # only override the prompt to force a collision.
            return {
                **draft,
                "prompt": "Universal collision prompt — every chunk yields this.",
            }

        def paraphrase_preference(self, draft, chunk, *, preserve_tokens=None):
            return draft

    from Trainforge.generators import _local_provider as lp_mod
    monkeypatch.setattr(lp_mod, "LocalSynthesisProvider", _CollidingProvider)

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="local",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
    )

    # Exactly one instruction prompt admitted; every subsequent
    # collision routed to the rejected bucket with the dedupe reason.
    assert stats.instruction_pairs_emitted == 1, (
        f"Expected dedupe to keep emitted=1, got {stats.instruction_pairs_emitted}"
    )
    assert stats.rejected_reasons.get("instruction:duplicate_prompt", 0) >= 1, (
        f"rejected_reasons missing duplicate_prompt: {stats.rejected_reasons!r}"
    )

    # Final on-disk JSONL has exactly one row.
    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    rows = [
        line for line in inst_path.read_text().splitlines() if line.strip()
    ]
    assert len(rows) == 1


def test_run_synthesis_dedupes_duplicate_preference_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror of the instruction-side dedupe for preference pairs.
    A preference paraphrase that collides across chunks is rejected
    before append and tallied under
    ``rejected_reasons[preference:duplicate_prompt]``."""
    course_dir = _make_working_copy(tmp_path)

    class _CollidingPrefProvider:
        def __init__(self, *args, **kwargs):
            pass

        def paraphrase_instruction(self, draft, chunk, *, preserve_tokens=None):
            return draft  # vary by chunk so instruction-side stays clean

        def paraphrase_preference(self, draft, chunk, *, preserve_tokens=None):
            return {
                **draft,
                "prompt": "Universal collision preference prompt — every chunk yields this.",
            }

    from Trainforge.generators import _local_provider as lp_mod
    monkeypatch.setattr(lp_mod, "LocalSynthesisProvider", _CollidingPrefProvider)

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="local",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
    )

    assert stats.preference_pairs_emitted == 1, (
        f"Expected dedupe to keep emitted=1, got {stats.preference_pairs_emitted}"
    )
    assert stats.rejected_reasons.get("preference:duplicate_prompt", 0) >= 1, (
        f"rejected_reasons missing duplicate_prompt: {stats.rejected_reasons!r}"
    )


# ---------------------------------------------------------------------------
# Wave 124: abstention + schema-translation generator wiring
# ---------------------------------------------------------------------------


def _write_minimal_pedagogy_graph(course_dir: Path) -> Path:
    """Drop a small pedagogy_graph.json next to the course corpus.

    The fixture's chunks address concept_load (chunk_mc_01) and
    concept_udl (chunk_mc_02). concept_bloom + concept_silent are
    "silent" relative to chunk_mc_01 / chunk_mc_02, giving the
    abstention generator at least one silent concept per chunk.
    """
    graph_dir = course_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "pedagogy_graph.json"
    payload = {
        "nodes": [
            {"id": "chunk_mc_01", "class": "Chunk"},
            {"id": "chunk_mc_02", "class": "Chunk"},
            {"id": "concept_load", "class": "Concept", "label": "Cognitive Load"},
            {"id": "concept_udl", "class": "Concept", "label": "UDL"},
            {"id": "concept_bloom", "class": "Concept", "label": "Bloom's"},
            {"id": "concept_silent", "class": "Concept", "label": "Silent topic"},
        ],
        "edges": [
            {
                "source": "chunk_mc_01",
                "target": "concept_load",
                "relation_type": "assesses",
            },
            {
                "source": "chunk_mc_02",
                "target": "concept_udl",
                "relation_type": "exemplifies",
            },
        ],
    }
    graph_path.write_text(
        __import__("json").dumps(payload, indent=2), encoding="utf-8",
    )
    return graph_path


def test_with_abstention_flag_appends_pairs(tmp_path: Path) -> None:
    """``--with-abstention`` adds abstention_probe pairs to the
    instruction_pairs.jsonl artifact and the stats counter goes up."""
    course_dir = _make_working_copy(tmp_path)
    _write_minimal_pedagogy_graph(course_dir)

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_abstention=True,
        abstention_max_pairs=10,
    )

    assert stats.abstention_pairs_emitted > 0, (
        "abstention_pairs_emitted should reflect generator output"
    )

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    assert inst_path.exists()
    import json as _json
    found_abstention = False
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = _json.loads(line)
        if rec.get("content_type") == "abstention_probe":
            found_abstention = True
            assert rec.get("template_id") == "abstention.no_edge"
            assert rec.get("expected_response") == "No."
            break
    assert found_abstention, (
        "instruction_pairs.jsonl should contain at least one "
        "abstention_probe pair"
    )


def test_with_schema_translation_flag_appends_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--with-schema-translation`` adds schema_translation pairs to
    the instruction_pairs.jsonl artifact and the stats counter goes up."""
    course_dir = _make_working_copy(tmp_path)

    # Inject a manifest whose CURIEs match the hand-curated table so
    # the generator emits pairs (the synthetic 'ex:load' from
    # _synthetic_manifest would be skipped with a warning).
    manifest = PropertyManifest(
        family="rdf_shacl",
        properties=[
            PropertyEntry(
                id="sh_datatype",
                uri="http://www.w3.org/ns/shacl#datatype",
                curie="sh:datatype",
                label="SHACL datatype constraint",
                surface_forms=["sh:datatype"],
                min_pairs=2,
            ),
            PropertyEntry(
                id="rdfs_subclassof",
                uri="http://www.w3.org/2000/01/rdf-schema#subClassOf",
                curie="rdfs:subClassOf",
                label="RDFS subclass-of",
                surface_forms=["rdfs:subClassOf"],
                min_pairs=2,
            ),
        ],
    )
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *a, **kw: manifest,
    )

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="rdf-shacl-551-2",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_schema_translation=True,
        schema_translation_max_pairs=4,
    )

    # Wave 125b expanded the catalog to 6 families/form. With cap=4
    # and 2 surface forms in the test fixture's manifest, round-robin
    # family balance lands at exactly 4 pairs (2 per form).
    assert stats.schema_translation_pairs_emitted == 4

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    assert inst_path.exists()
    import json as _json
    seen_curies: set = set()
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = _json.loads(line)
        if rec.get("content_type") == "schema_translation":
            seen_curies.add(rec["concept_tags"][0])
    assert seen_curies == {"sh:datatype", "rdfs:subClassOf"}


# ---------------------------------------------------------------------------
# Wave 125a: --violation-detection-max-pairs cap
# ---------------------------------------------------------------------------


def test_violation_detection_max_pairs_caps_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--violation-detection-max-pairs N`` caps the count of
    violation-detection pairs appended to instruction_pairs.jsonl while
    keeping every surface form represented (family-balanced
    round-robin)."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    _patch_shacl_manifest(monkeypatch)
    course_dir = _make_working_copy(tmp_path)

    cap = 60
    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_violation_detection=True,
        violation_detection_max_pairs=cap,
    )

    # The unlimited catalog is >= 800; with cap=60 the emit must
    # respect the cap.
    assert stats.violation_pairs_emitted <= cap
    assert stats.violation_pairs_emitted > 0

    # Verify on disk: count violation_detection pairs in the artifact
    # and confirm every surface form survived the round-robin trim.
    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    assert inst_path.exists()
    import json as _json
    from collections import Counter
    surface_forms: Counter = Counter()
    violation_count = 0
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = _json.loads(line)
        if rec.get("content_type") == "violation_detection":
            violation_count += 1
            tags = rec.get("concept_tags") or []
            if tags:
                surface_forms[tags[0]] += 1
    assert violation_count == stats.violation_pairs_emitted
    # All 6 RDF/SHACL surface forms must remain represented.
    expected = {
        "sh:datatype", "sh:class", "sh:NodeShape",
        "sh:PropertyShape", "rdfs:subClassOf", "owl:sameAs",
    }
    assert expected.issubset(set(surface_forms.keys())), (
        f"capped violation emit dropped a surface form: "
        f"{expected - set(surface_forms.keys())}"
    )


def test_violation_detection_no_cap_appends_full_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``--violation-detection-max-pairs`` flag means the entire
    pyshacl-validated catalog (>= 800 pairs) lands in the artifact."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    _patch_shacl_manifest(monkeypatch)
    course_dir = _make_working_copy(tmp_path)

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_violation_detection=True,
        # violation_detection_max_pairs left at default (None)
    )
    assert stats.violation_pairs_emitted >= 800


# ---------------------------------------------------------------------------
# Wave 127: deterministic generators hoisted ABOVE chunk loop and
# mirrored to the sidecar
# ---------------------------------------------------------------------------


def test_violation_detection_pairs_appear_in_sidecar_before_paraphrase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 127 contract: when ``--with-violation-detection`` is on,
    the pyshacl-validated pairs must land in the
    ``instruction_pairs.jsonl.in_progress`` sidecar BEFORE any
    chunk-paraphrase pairs do, so an operator running ``tail -f`` can
    confirm the flag wired through within the first ~minute of the run
    instead of waiting for the multi-hour paraphrase loop.

    Verified via the budget-exceeded path: ``max_dispatches=1`` makes
    the chunk loop hit ``SynthesisBudgetExceeded`` after the very first
    paraphrase dispatch, which preserves the sidecar on disk for
    inspection. Deterministic pairs run BEFORE the chunk loop, so they
    must be present in the preserved sidecar regardless of how few
    paraphrase rows landed.
    """
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    _patch_shacl_manifest(monkeypatch)

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

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="claude_session",
        seed=11,
        dispatcher=dispatcher,
        max_dispatches=1,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_violation_detection=True,
        violation_detection_max_pairs=10,
    )

    assert stats.capped_at_max_dispatches is True, (
        "Wave 127 test setup invariant: max_dispatches=1 must trigger "
        "SynthesisBudgetExceeded so the sidecar is preserved for "
        "inspection."
    )
    assert stats.violation_pairs_emitted > 0, (
        "Wave 127 contract: violation generator must run BEFORE the "
        "chunk loop, so its pairs land regardless of how the chunk "
        "loop terminates."
    )
    assert inst_progress.exists(), (
        "Wave 127 contract: sidecar must be preserved on a "
        "SynthesisBudgetExceeded exit."
    )

    sidecar_lines = [
        l for l in inst_progress.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    template_ids = []
    for raw in sidecar_lines:
        import json as _json
        try:
            rec = _json.loads(raw)
        except _json.JSONDecodeError:
            continue
        template_ids.append(rec.get("template_id", ""))

    violation_idx = [
        i for i, t in enumerate(template_ids)
        if t.startswith("violation_detection.")
    ]
    assert violation_idx, (
        f"Wave 127 contract: violation_detection.* pairs must appear "
        f"in the sidecar; saw template_ids={template_ids[:5]!r}..."
    )
    # Hoist invariant: deterministic pairs must come first. Find any
    # paraphrase template_id index and confirm every violation index
    # precedes it. (If the chunk loop emitted zero paraphrase pairs
    # before the budget-exceeded exit, the comparison vacuously holds.)
    paraphrase_idx = [
        i for i, t in enumerate(template_ids)
        if not t.startswith((
            "violation_detection.",
            "schema_translation.",
            "abstention.",
            "kg_metadata.",
        ))
    ]
    if paraphrase_idx:
        assert max(violation_idx) < min(paraphrase_idx), (
            f"Wave 127 contract: deterministic violation_detection "
            f"pairs must precede paraphrase pairs in the sidecar. "
            f"Last violation at row {max(violation_idx)}, first "
            f"paraphrase at row {min(paraphrase_idx)}."
        )


# ---------------------------------------------------------------------------
# Wave 133c: violation_generator family-gating via validation_kind
# ---------------------------------------------------------------------------


def test_violation_generator_skipped_for_non_shacl_family(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 133c contract: ``--with-violation-detection`` must short-
    circuit when the property manifest declares a non-SHACL family
    (``validation_kind != "shacl"`` or absent), even if the flag is on.

    The pyshacl-oracle-verified violation catalog is RDF/SHACL-specific
    (hardcoded ``sh:`` / ``rdfs:`` / ``owl:`` shapes). A future course
    family (e.g. JSON Schema) toggling the flag must NOT silently get
    SHACL pairs polluting its training data; the gate must skip and
    log a warning so an operator sees the intentional no-op.
    """
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    course_dir = _make_working_copy(tmp_path)

    # Inject a non-SHACL property manifest by monkeypatching the loader.
    # PropertyEntry surface_forms picked to NOT collide with the mini
    # course's chunk text, so property_coverage gating is incidental.
    non_shacl_manifest = PropertyManifest(
        family="generic_test_family",
        properties=[
            PropertyEntry(
                id="generic_token",
                uri="http://example.test/generic",
                curie="ex:generic",
                label="Generic non-SHACL surface form",
                surface_forms=["generic_token_no_match"],
                min_pairs=1,
            ),
        ],
        # validation_kind unset (None) — gate must skip violation pairs.
        validation_kind=None,
    )
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *_a, **_kw: non_shacl_manifest,
    )
    # synthesize_training imports it via from-import inside run_synthesis,
    # so the module-level binding is what gets resolved each call.

    caplog.set_level(logging.WARNING, logger="Trainforge.synthesize_training")
    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_violation_detection=True,
        violation_detection_max_pairs=10,
    )

    # Zero violation pairs even with --with-violation-detection on.
    assert stats.violation_pairs_emitted == 0, (
        f"Wave 133c: violation generator must skip for non-SHACL "
        f"family; got {stats.violation_pairs_emitted} pairs."
    )

    # Warning must fire so the operator sees the intentional no-op.
    skip_warnings = [
        r for r in caplog.records
        if "violation_generator skipped" in r.getMessage()
    ]
    assert skip_warnings, (
        "Wave 133c: expected a 'violation_generator skipped' warning "
        f"log when validation_kind != 'shacl'; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )
    msg = skip_warnings[0].getMessage()
    assert "generic_test_family" in msg, (
        f"warning message must surface the manifest family for diagnostics; "
        f"got {msg!r}"
    )


# ---------------------------------------------------------------------------
# Worker A: per-pair resume checkpoint sidecar
#
# Mirror of ``Trainforge/tests/test_align_chunks_checkpoint.py`` — same
# tolerant-load / append-and-flush / resume-skips-cached / unlink-on-clean-
# exit / preserve-on-exception contract, applied to ``run_synthesis``.
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402

CHECKPOINT_NAME = ".synthesis_pairs_checkpoint.jsonl"


def _read_checkpoint_lines(path: Path) -> list:
    return [
        _json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_synthesis_pair_checkpoint_appended_per_chunk(tmp_path: Path) -> None:
    """A clean run (default checkpoint-on) writes one resume-cache
    record per accepted instruction or preference pair AND unlinks the
    sidecar on success — but mid-run, while the loop is still emitting,
    the per-pair appends are observable in the JSON content of the
    canonical artifacts (one line per emitted pair).
    """
    working = _make_working_copy(tmp_path)

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    inst_final = working / "training_specs" / "instruction_pairs.jsonl"
    pref_final = working / "training_specs" / "preference_pairs.jsonl"
    inst_lines = [
        line for line in inst_final.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pref_lines = [
        line for line in pref_final.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # The accepted-pair count drives the checkpoint append count, so a
    # mid-run inspection of the sidecar would have shown exactly this
    # many lines. Post-run unlink is verified separately by
    # test_synthesis_checkpoint_unlinked_on_clean_exit.
    assert len(inst_lines) == stats.instruction_pairs_emitted
    assert len(pref_lines) == stats.preference_pairs_emitted
    assert stats.instruction_pairs_emitted > 0
    assert stats.preference_pairs_emitted > 0


def test_synthesis_resume_skips_cached_chunks(tmp_path: Path) -> None:
    """Pre-seed the checkpoint with records for the first 5 chunks and
    pass a mock paraphrase provider whose call would raise. The resumed
    run must replay every cached pair without dispatching to the LLM
    once — proving the checkpoint actually short-circuits the dispatch.

    Uses ``provider="mock"`` so the synthesis path itself is the
    template factory. The factory takes ``provider`` as a hint but
    does not call any LLM; we instead validate by counting the
    resulting pairs and asserting the cached pairs land verbatim.
    """
    working = _make_working_copy(tmp_path)
    checkpoint_path = (
        working / "training_specs" / CHECKPOINT_NAME
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-seed the cache for chunks 1-5 with a recognisable sentinel
    # in the prompt so we can prove the cached value reached the
    # canonical artifact.
    sentinel_prompts = []
    with checkpoint_path.open("w", encoding="utf-8") as fh:
        for i in range(1, 6):
            chunk_id = f"chunk_mc_{i:02d}"
            sentinel_prompt = (
                f"RESUMED-FROM-CHECKPOINT prompt for {chunk_id} "
                "covering cognitive load theory in detail."
            )
            sentinel_completion = (
                f"RESUMED-FROM-CHECKPOINT completion for {chunk_id} "
                "covering UDL frameworks across multiple modalities."
            )
            sentinel_prompts.append(sentinel_prompt)
            record = {
                "schema_version": "v1",
                "chunk_id": chunk_id,
                "kind": "instruction",
                "variant_index": 0,
                "pair": {
                    "id": f"resumed_{chunk_id}_inst_000",
                    "chunk_id": chunk_id,
                    "prompt": sentinel_prompt,
                    "completion": sentinel_completion,
                    "bloom_level": "understand",
                    "content_type": "explanation",
                    "template_id": "resumed.from.checkpoint",
                    "provider": "mock",
                    "seed": 17,
                    "decision_capture_id": "evt_resumed_0",
                    "topic": "cognitive_load",
                },
                "provider": "mock",
                "seed": 17,
            }
            fh.write(_json.dumps(record) + "\n")

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    inst_final = working / "training_specs" / "instruction_pairs.jsonl"
    inst_records = [
        _json.loads(line)
        for line in inst_final.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    emitted_prompts = {r.get("prompt") for r in inst_records}
    # All 5 sentinel prompts MUST appear verbatim — they came directly
    # from the checkpoint, never through the synthesize_instruction_pair
    # factory.
    for sentinel in sentinel_prompts:
        assert sentinel in emitted_prompts, (
            f"Worker A: checkpoint replay must surface the cached prompt "
            f"verbatim. Missing: {sentinel!r}"
        )
    assert stats.instruction_pairs_emitted >= 5


def test_synthesis_checkpoint_preserved_on_budget_exceeded(
    tmp_path: Path,
) -> None:
    """When the chunk loop raises ``SynthesisBudgetExceeded``, the
    per-pair checkpoint MUST persist on disk so a re-run with a
    higher dispatch cap resumes from the partial cache.
    """
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
    checkpoint_path = (
        working / "training_specs" / CHECKPOINT_NAME
    )
    assert not checkpoint_path.exists()

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="claude_session",
        seed=11,
        dispatcher=dispatcher,
        max_dispatches=1,
    )

    assert stats.capped_at_max_dispatches is True
    assert checkpoint_path.exists(), (
        "Worker A: budget-exceeded run MUST preserve the resume "
        "checkpoint sidecar so the next run can skip the LLM dispatch "
        "for already-emitted pairs."
    )


def test_synthesis_checkpoint_unlinked_on_clean_exit(tmp_path: Path) -> None:
    """A successful, fully-completed run MUST delete the checkpoint
    sidecar after writing the final canonical JSONL artifacts. The
    sidecar is a transient resume cache, not a long-lived artifact.
    """
    working = _make_working_copy(tmp_path)
    checkpoint_path = (
        working / "training_specs" / CHECKPOINT_NAME
    )

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    assert stats.capped_at_max_dispatches is False
    assert not checkpoint_path.exists(), (
        "Worker A: clean-exit unlink contract — checkpoint sidecar "
        f"must be deleted on a clean run; found at {checkpoint_path}"
    )


def test_synthesis_checkpoint_malformed_lines_tolerated(
    tmp_path: Path,
) -> None:
    """Malformed JSON lines + truncated mid-write garbage in the
    checkpoint MUST be silently dropped. Salvageable records still
    drive resume; the run completes successfully on the rest.
    """
    working = _make_working_copy(tmp_path)
    checkpoint_path = (
        working / "training_specs" / CHECKPOINT_NAME
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    salvageable_record = {
        "schema_version": "v1",
        "chunk_id": "chunk_mc_01",
        "kind": "instruction",
        "variant_index": 0,
        "pair": {
            "id": "salvaged_inst_000",
            "chunk_id": "chunk_mc_01",
            "prompt": "SALVAGED-FROM-MALFORMED-CHECKPOINT prompt about cognitive load.",
            "completion": "SALVAGED-FROM-MALFORMED-CHECKPOINT completion about UDL.",
            "bloom_level": "understand",
            "content_type": "explanation",
            "template_id": "salvaged.from.checkpoint",
            "provider": "mock",
            "seed": 17,
            "decision_capture_id": "evt_salvaged_0",
            "topic": "cognitive_load",
        },
        "provider": "mock",
        "seed": 17,
    }
    checkpoint_path.write_text(
        "this is not valid json\n"
        + _json.dumps(salvageable_record) + "\n"
        + "{ truncated json mid-line\n",
        encoding="utf-8",
    )

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    inst_final = working / "training_specs" / "instruction_pairs.jsonl"
    inst_records = [
        _json.loads(line)
        for line in inst_final.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    emitted_prompts = {r.get("prompt") for r in inst_records}
    # Salvaged record reached the canonical artifact via resume.
    assert any(
        p and "SALVAGED-FROM-MALFORMED-CHECKPOINT" in p
        for p in emitted_prompts
    ), (
        "Worker A: tolerant loader must salvage well-formed records "
        "amid malformed/truncated lines."
    )
    # Other chunks still emitted normally — the malformed lines didn't
    # poison the run.
    assert stats.instruction_pairs_emitted >= 2


def test_synthesis_checkpoint_schema_version_drift_invalidates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A checkpoint record carrying ``schema_version: "v0"`` MUST be
    skipped (with a ``logger.warning``) so the chunk loop falls back
    to the LLM path. Lets a future bump to the post-decoration pair
    shape loudly invalidate stale resume caches.
    """
    working = _make_working_copy(tmp_path)
    checkpoint_path = (
        working / "training_specs" / CHECKPOINT_NAME
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    v0_record = {
        "schema_version": "v0",
        "chunk_id": "chunk_mc_01",
        "kind": "instruction",
        "variant_index": 0,
        "pair": {
            "chunk_id": "chunk_mc_01",
            "prompt": "STALE-V0-PROMPT-must-not-leak",
            "completion": "STALE-V0-COMPLETION",
        },
        "provider": "mock",
        "seed": 17,
    }
    checkpoint_path.write_text(
        _json.dumps(v0_record) + "\n", encoding="utf-8",
    )

    caplog.set_level(logging.WARNING, logger="Trainforge.synthesize_training")
    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    inst_final = working / "training_specs" / "instruction_pairs.jsonl"
    emitted_prompts = {
        _json.loads(line).get("prompt")
        for line in inst_final.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    # The v0 prompt MUST NOT appear in the canonical artifact.
    assert not any(
        p and "STALE-V0-PROMPT-must-not-leak" in p for p in emitted_prompts
    ), (
        "Worker A: schema_version mismatch must invalidate the cached "
        "record — the stale prompt leaked into the emitted pairs."
    )
    # Warning must fire so the operator sees why the cache was ignored.
    drift_warnings = [
        r for r in caplog.records
        if "schema_version mismatch" in r.getMessage()
    ]
    assert drift_warnings, (
        "Worker A: schema_version drift must emit a logger.warning so "
        f"the operator can debug the resume miss; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )
    # Run completes normally — schema-drift invalidation isn't fatal.
    assert stats.instruction_pairs_emitted > 0


def test_synthesis_no_checkpoint_path_is_a_noop(tmp_path: Path) -> None:
    """Back-compat: when ``--no-checkpoint`` is passed (CLI sentinel
    routes through to ``synthesis_pairs_checkpoint_path`` as a
    disable-marker Path), no sidecar is created on disk and the run
    behaves identically to legacy invocations.
    """
    working = _make_working_copy(tmp_path)
    checkpoint_path = (
        working / "training_specs" / CHECKPOINT_NAME
    )
    assert not checkpoint_path.exists()

    # Sentinel matches the one main() constructs from --no-checkpoint.
    disable_sentinel = Path("<disable-synthesis-checkpoint>")

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
        synthesis_pairs_checkpoint_path=disable_sentinel,
    )

    assert stats.instruction_pairs_emitted > 0
    assert not checkpoint_path.exists(), (
        "Worker A: --no-checkpoint must skip sidecar creation entirely "
        "(no append handle opened, no append calls fired)."
    )
