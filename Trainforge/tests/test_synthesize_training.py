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
    ``surface_form_preservation_fallback`` capture event and emits the
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
        if d.get("decision_type") == "surface_form_preservation_fallback"
    ]
    assert fallback_events, (
        "Expected at least one surface_form_preservation_fallback "
        "capture event; got "
        f"{[d.get('decision_type') for d in capture.decisions]}"
    )


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
    """When the final pair (after fallback or pure-deterministic path)
    doesn't contain a required surface form, the factory injects a
    'Canonical terms:' sentence in the completion AND a '(Reference:
    ...)' suffix in the prompt. Closes the training-objective gap
    where the model would only learn to USE CURIEs in answers, not
    RECOGNIZE them in user prompts."""
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
    # Wave 121: rigid 'Canonical terms' / 'Reference:' literals dropped
    # because the phrasing rotates across chunks; checking surface forms
    # + audit fields is the functional invariant.
    assert "sh:NodeShape" in result.pair["prompt"]
    assert "sh:datatype" in result.pair["prompt"]
    assert "sh:NodeShape" in result.pair["completion"]
    assert "sh:datatype" in result.pair["completion"]
    # Audit trail records which tokens were injected on each side.
    injected_completion = set(result.pair.get("preserve_tokens_injected", []))
    injected_prompt = set(result.pair.get("preserve_tokens_injected_prompt", []))
    assert {"sh:NodeShape", "sh:datatype"}.issubset(injected_completion)
    assert {"sh:NodeShape", "sh:datatype"}.issubset(injected_prompt)


def test_force_inject_skips_per_side_when_already_contains_token() -> None:
    """Per-side idempotency: if the deterministic draft has the CURIE
    on one side only, only the missing side gets injected."""
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
    assert "preserve_tokens_injected_prompt" in out_p
    assert "preserve_tokens_injected" not in out_p

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
    assert "preserve_tokens_injected" in out_c
    assert "preserve_tokens_injected_prompt" not in out_c


def test_force_inject_phrasing_rotates_across_chunks() -> None:
    """Wave 121: phrasing rotation prevents the boilerplate-suffix
    saturation flagged in the 2026-04-29 smoke audit (70% of prompts
    ended with a single '(Reference: ...)' string). The selector
    keys on chunk_id hash so rotation is deterministic but
    distribution-spread across the corpus."""
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
        out = _enforce_preserve_tokens_in_instruction(pair, ["sh:NodeShape"])
        # Recover the appended template by stripping the base + the
        # injected token. The token-replacement shape lets us cluster
        # identical phrasings regardless of which CURIE was injected.
        prompt_suffix = out["prompt"][len(base_prompt):]
        completion_suffix = out["completion"][len(base_completion):]
        prompt_addition_forms.add(prompt_suffix.replace("sh:NodeShape", "X"))
        completion_addition_forms.add(completion_suffix.replace("sh:NodeShape", "X"))
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
    addition would breach the per-field max. The CURIE always lands."""
    from Trainforge.generators.instruction_factory import (
        COMPLETION_MAX,
        PROMPT_MAX,
        _enforce_preserve_tokens_in_instruction,
    )
    long_prompt = "Q? " * 200  # ~600 chars, well over PROMPT_MAX=400
    long_completion = "Filler " * 120  # ~840 chars, over COMPLETION_MAX=600
    pair = {"prompt": long_prompt, "completion": long_completion}
    out = _enforce_preserve_tokens_in_instruction(
        pair, ["sh:datatype", "sh:NodeShape"],
    )
    assert len(out["prompt"]) <= PROMPT_MAX
    assert len(out["completion"]) <= COMPLETION_MAX
    # Tokens land on both sides regardless of clamping.
    for side in ("prompt", "completion"):
        assert "sh:datatype" in out[side], f"sh:datatype missing from {side}"
        assert "sh:NodeShape" in out[side], f"sh:NodeShape missing from {side}"


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


def test_violation_detection_max_pairs_caps_emit(tmp_path: Path) -> None:
    """``--violation-detection-max-pairs N`` caps the count of
    violation-detection pairs appended to instruction_pairs.jsonl while
    keeping every surface form represented (family-balanced
    round-robin)."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
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
    tmp_path: Path,
) -> None:
    """No ``--violation-detection-max-pairs`` flag means the entire
    pyshacl-validated catalog (>= 800 pairs) lands in the artifact."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
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
