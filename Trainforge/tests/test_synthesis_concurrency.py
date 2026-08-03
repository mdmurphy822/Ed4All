"""Stress and fault tests for bounded ordered synthesis concurrency."""

from __future__ import annotations

import threading
import time
import shutil
import json
import hashlib
from pathlib import Path

import pytest

import Trainforge.synthesize_training as synthesis_module
from Trainforge.synthesis_concurrency import (
    BoundedOrderedMap,
    resolve_synthesis_max_concurrent,
)
from Trainforge.synthesize_training import (
    _checkpoint_terminal_rejection,
    _run_generation_unit,
    _synthesis_runtime_policy_identity,
    run_synthesis,
)
from Trainforge.synthesis_journal import (
    GenerationJournal,
    load_generation_journal,
    summarize_generation_journal,
)
from Trainforge.generators.pairs.instruction import InstructionSynthesisResult
from Trainforge.generators.pairs.preference import PreferenceSynthesisResult
from Trainforge.generators.providers._synthesis_common import SynthesisProviderError


_FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


def _course_copy(root: Path, name: str) -> Path:
    dst = root / name
    shutil.copytree(_FIXTURE_ROOT, dst)
    specs = dst / "training_specs"
    for path in specs.glob("*pairs.jsonl*"):
        path.unlink()
    for name in (
        ".synthesis_pairs_checkpoint.jsonl",
        "dataset_config.json",
        "pilot_report.md",
        "synthesis_summary.json",
    ):
        (specs / name).unlink(missing_ok=True)
    return dst


def _fresh_marker(course: Path, fresh_id: str) -> None:
    specs = course / "training_specs"
    evidence = course / "fresh-evidence.json"
    evidence.write_text('{"fresh":true}\\n', encoding="utf-8")
    chunks = course / "corpus" / "chunks.jsonl"
    marker = {
        "schema_version": 1,
        "fresh_start_id": fresh_id,
        "archive_manifest_path": str(evidence),
        "archive_manifest_sha256": hashlib.sha256(
            evidence.read_bytes()
        ).hexdigest(),
        "preserved_input_sha256": {
            str(chunks): hashlib.sha256(chunks.read_bytes()).hexdigest(),
        },
    }
    (specs / ".synthesis_fresh_start.json").write_text(
        json.dumps(marker), encoding="utf-8",
    )


def test_generation_contract_change_aborts_and_preserves_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed GENERATION contract must abort the resume, not rebind.

    The rows already accepted were produced by one implementation; the rest
    would be produced by another. Nothing downstream can separate them again
    once they are training pairs, so the run stops here.

    This was briefly inverted to assert a rebind, on the grounds that aborting
    is what produced ~30 archive-and-restart directories for one run (three
    named ``*-contract-supplement*``). That diagnosis was right about the
    symptom and wrong about the cause: those restarts came from VERDICT-policy
    files sharing the generation tuple, so editing a validator re-keyed
    generation. That is fixed at the source by the tuple split — see
    ``test_verdict_policy_is_recorded_outside_the_generation_fingerprint``,
    which pins the other half of the contract: a verdict-policy change is
    recorded and the run continues. With that split in place, reaching this
    abort means a real generation change, which is exactly when stopping is
    correct.
    """
    course = _course_copy(tmp_path, "fresh-contract-sidecar")
    fresh_id = "fresh-contract-integration-0001"
    _fresh_marker(course, fresh_id)
    run_synthesis(
        course,
        "FRESH",
        provider="mock",
        max_pairs=1,
        max_concurrent=4,
        curriculum_from_graph=False,
        expected_fresh_start_id=fresh_id,
    )
    marker_path = course / "training_specs" / ".synthesis_fresh_start.json"
    first_contract = json.loads(marker_path.read_text(encoding="utf-8"))[
        "synthesis_run_contract_sha256"
    ]
    original = synthesis_module._synthesis_static_contract_components

    def changed_components(**kwargs):
        value = original(**kwargs)
        value["files"] = {**value["files"], "injected.py": "changed"}
        return value

    monkeypatch.setattr(
        synthesis_module,
        "_synthesis_static_contract_components",
        changed_components,
    )
    sidecar = course / "training_specs" / "instruction_pairs.jsonl.in_progress"
    before = sidecar.read_bytes() if sidecar.exists() else None

    with pytest.raises(Exception, match="generation contract changed mid-run"):
        run_synthesis(
            course,
            "FRESH",
            provider="mock",
            max_pairs=1,
            max_concurrent=4,
            curriculum_from_graph=False,
            expected_fresh_start_id=fresh_id,
        )

    # The marker still names the contract that produced the accepted rows —
    # aborting must not quietly adopt the new one.
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["synthesis_run_contract_sha256"] == first_contract

    # Failing closed must not also destroy the work already done: the sidecar
    # opens in "w" mode, so reaching it would truncate the accepted rows the
    # operator still needs to archive.
    if before is not None:
        assert sidecar.read_bytes() == before


def test_verdict_policy_is_recorded_outside_the_generation_fingerprint(
    tmp_path: Path,
) -> None:
    """Verdict-policy digests must be RECORDED but never keyed on.

    Prevents the regression where severing the verdict/generation coupling
    also deleted the record: an auditor holding an accepted pair could no
    longer tell which claim-support thresholds judged it, and re-proving a
    pair meant regenerating it.
    """
    course = _course_copy(tmp_path, "verdict-policy-record")
    fresh_id = "fresh-contract-integration-0003"
    _fresh_marker(course, fresh_id)
    run_synthesis(
        course,
        "FRESH",
        provider="mock",
        max_pairs=1,
        max_concurrent=4,
        curriculum_from_graph=False,
        expected_fresh_start_id=fresh_id,
    )
    marker = json.loads(
        (course / "training_specs" / ".synthesis_fresh_start.json").read_text(
            encoding="utf-8",
        )
    )
    components = marker["synthesis_run_contract_components"]
    verdict = components["verdict_policy"]
    assert len(verdict["sha256"]) == 64
    # Every verdict-policy file is recorded here...
    for relative in synthesis_module._VERDICT_POLICY_FILES:
        assert relative in verdict["files"]
    # ...and none of them is in the fingerprinted generation file set.
    assert not (
        set(synthesis_module._VERDICT_POLICY_FILES)
        & set(components["files"])
    )


def test_fresh_concurrent_live_edit_stops_before_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    course = _course_copy(tmp_path, "fresh-contract-live-edit")
    fresh_id = "fresh-contract-integration-0002"
    _fresh_marker(course, fresh_id)
    # Patch ``_GENERATION_CONTRACT_FILES``, not the legacy
    # ``_SYNTHESIS_REJECTION_CONTRACT_FILES`` alias.  The alias no longer
    # reaches the live drift check ``_assert_live_synthesis_contract``, which
    # reads ``_GENERATION_CONTRACT_FILES`` via
    # ``_synthesis_static_contract_components``; patching the alias made this
    # test pass vacuously (DID NOT RAISE) instead of exercising the check.
    # Entries are resolved against PROJECT_ROOT, so the watched path is
    # repo-relative rather than an absolute tmp_path.
    watched_relative = "runtime/state/watched-provider-contract.py"
    watched = synthesis_module.PROJECT_ROOT / watched_relative
    watched.parent.mkdir(parents=True, exist_ok=True)
    watched.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        synthesis_module,
        "_GENERATION_CONTRACT_FILES",
        (watched_relative,),
    )
    original = synthesis_module.synthesize_instruction_pair
    calls = 0
    call_lock = threading.Lock()

    def mutate_after_first_admitted_call(chunk, **kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
            if calls == 1:
                watched.write_text("version = 2\n", encoding="utf-8")
        return original(chunk, **kwargs)

    monkeypatch.setattr(
        synthesis_module,
        "synthesize_instruction_pair",
        mutate_after_first_admitted_call,
    )
    try:
        with pytest.raises(RuntimeError, match="contract drifted"):
            run_synthesis(
                course,
                "FRESH",
                provider="mock",
                max_concurrent=4,
                curriculum_from_graph=False,
                expected_fresh_start_id=fresh_id,
            )
    finally:
        watched.unlink(missing_ok=True)
    # At most the already-admitted C4 wave may enter; no later batch dispatches.
    assert 1 <= calls <= 4


@pytest.mark.parametrize("workers", [4, 32, 48])
def test_completed_workers_are_durable_while_sequence_zero_blocks(
    tmp_path: Path, workers: int
) -> None:
    """Fault injection: seq 1..N fsync before ordered emission can advance."""

    journal_path = tmp_path / "generation.jsonl"
    journal = GenerationJournal(journal_path)
    release_zero = threading.Event()
    siblings_done = threading.Event()
    completed = 0
    completed_lock = threading.Lock()
    calls = [0] * workers

    def result_for(value: int) -> InstructionSynthesisResult:
        calls[value] += 1
        if value == 0:
            release_zero.wait(timeout=5)
            raise RuntimeError("deterministic writer-front fault")
        return InstructionSynthesisResult(
            pair={"prompt": f"p{value}", "completion": f"c{value}"},
            quality={"valid": True},
            template_id="test",
            rationale=f"dynamic rationale for chunk-{value}",
            topic=f"topic-{value}",
        )

    def worker(value: int):
        nonlocal completed
        outcome = _run_generation_unit(
            chunk_id=f"chunk-{value}",
            kind="instruction",
            variant_index=0,
            fingerprint=f"fp-{value}",
            generation_cache={},
            journal=journal,
            call=lambda: result_for(value),
        )
        if value:
            with completed_lock:
                completed += 1
                if completed == workers - 1:
                    siblings_done.set()
        return outcome

    ordered = BoundedOrderedMap(range(workers), worker, max_concurrent=workers)
    consumer = threading.Thread(target=lambda: list(ordered), daemon=True)
    consumer.start()
    assert siblings_done.wait(timeout=10)

    # The consumer is blocked on sequence zero, but every successful sibling
    # is already durable and independently replayable.
    cache = load_generation_journal(journal_path)
    assert len(cache) == workers - 1
    assert all(
        cache[(f"chunk-{value}", "instruction", 0)]["disposition"] == "success"
        for value in range(1, workers)
    )
    release_zero.set()
    consumer.join(timeout=10)
    assert not consumer.is_alive()

    cache = load_generation_journal(journal_path)
    for value in range(1, workers):
        outcome = _run_generation_unit(
            chunk_id=f"chunk-{value}",
            kind="instruction",
            variant_index=0,
            fingerprint=f"fp-{value}",
            generation_cache=cache,
            journal=journal,
            call=lambda value=value: result_for(value),
        )
        assert (
            outcome.provider_results,
            outcome.cached_replays,
            outcome.error,
        ) == (0, 1, None)
    assert calls == [1] * workers


@pytest.mark.parametrize(
    ("kind", "result"),
    [
        (
            "instruction",
            InstructionSynthesisResult(
                pair=None,
                quality={
                    "passed": False,
                    "reason": "staged_plan_sft_invalid",
                    "rejection_evidence": {
                        "stage": "plan_sft",
                        "prompt_ref": "sha256:" + "a" * 64,
                        "response_ref": "sha256:" + "b" * 64,
                    },
                },
                template_id="test",
                rationale="bounded staged SFT repair exhausted for this unit",
                topic="unit topic",
            ),
        ),
        (
            "preference",
            PreferenceSynthesisResult(
                pair=None,
                quality={
                    "passed": False,
                    "reason": "staged_plan_dpo_invalid",
                    "rejection_evidence": {
                        "stage": "plan_dpo",
                        "prompt_ref": "sha256:" + "c" * 64,
                        "response_ref": "sha256:" + "d" * 64,
                    },
                },
                rationale="bounded staged DPO repair exhausted for this unit",
                source="misconception",
            ),
        ),
    ],
)
def test_terminal_content_rejection_replays_as_successful_provider_result(
    tmp_path: Path, kind: str, result,
) -> None:
    journal_path = tmp_path / "generation.jsonl"
    journal = GenerationJournal(journal_path)
    calls = 0

    def call():
        nonlocal calls
        calls += 1
        return result

    first = _run_generation_unit(
        chunk_id="chunk-content-rejection",
        kind=kind,
        variant_index=0,
        fingerprint="content-contract-v1",
        generation_cache={},
        journal=journal,
        call=call,
    )
    assert first.error is None
    assert first.fatal_units == 0
    assert first.result is not None and first.result.pair is None
    cache = load_generation_journal(journal_path)
    assert cache[("chunk-content-rejection", kind, 0)]["disposition"] == "success"

    replay = _run_generation_unit(
        chunk_id="chunk-content-rejection",
        kind=kind,
        variant_index=0,
        fingerprint="content-contract-v1",
        generation_cache=cache,
        journal=journal,
        call=call,
    )
    assert replay.error is None
    assert replay.cached_replays == 1
    assert replay.result is not None and replay.result.pair is None
    assert calls == 1
    assert summarize_generation_journal(journal_path)["fatal_units"] == 0


def test_terminal_pair_checkpoint_preserves_raw_rejection_refs(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pairs.jsonl"
    evidence = {
        "terminal_content_rejection": True,
        "stage": "plan_sft",
        "validation_error": "response was not a JSON object",
        "prompt_ref": "sha256:" + "a" * 64,
        "response_ref": "sha256:" + "b" * 64,
    }
    with checkpoint.open("w", encoding="utf-8") as handle:
        _checkpoint_terminal_rejection(
            handle,
            chunk_id="chunk-terminal",
            kind="instruction",
            variant_index=0,
            provider="local",
            seed=7,
            reason="staged_plan_sft_invalid",
            contract_fingerprint="contract-v1",
            rejection_evidence=evidence,
        )
    row = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert row["disposition"] == "rejected"
    assert row["reason"] == "staged_plan_sft_invalid"
    assert row["rejection_evidence"] == evidence
    assert row["pair"] is None


def test_transient_resume_retries_only_failed_unit_then_caps(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "generation.jsonl"
    journal = GenerationJournal(journal_path)
    calls = 0

    def transient() -> InstructionSynthesisResult:
        nonlocal calls
        calls += 1
        raise SynthesisProviderError(
            "temporary overload", code="503", chunk_id="chunk-transient"
        )

    for expected_attempt in (1, 2, 3):
        cache = load_generation_journal(journal_path)
        outcome = _run_generation_unit(
            chunk_id="chunk-transient",
            kind="instruction",
            variant_index=0,
            fingerprint="stable-contract",
            generation_cache=cache,
            journal=journal,
            call=transient,
        )
        assert (outcome.provider_results, outcome.cached_replays) == (1, 0)
        assert outcome.error is not None
        assert outcome.error.attempt == expected_attempt
        assert outcome.error.transient is (expected_attempt < 3)

    # The capped fatal disposition is durable and makes no fourth call.
    cache = load_generation_journal(journal_path)
    outcome = _run_generation_unit(
        chunk_id="chunk-transient",
        kind="instruction",
        variant_index=0,
        fingerprint="stable-contract",
        generation_cache=cache,
        journal=journal,
        call=transient,
    )
    assert (
        outcome.provider_results, outcome.cached_replays, calls
    ) == (0, 0, 3)
    assert outcome.error is not None and outcome.error.transient is False
    summary = summarize_generation_journal(journal_path)
    assert summary["transient_attempts"] == 3
    assert summary["exhausted_units"] == 1
    assert summary["fatal_units"] == 1


def test_fingerprint_runtime_policy_excludes_only_operational_values() -> None:
    environment = {
        "TRAINFORGE_SYNTHESIS_MAX_CONCURRENT": "48",
        "TRAINFORGE_EVAL_PROGRESS_EVERY": "1",
        "ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS": "120",
        "TRAINFORGE_AGNOSTIC_SYNTHESIS": "1",
        "TRAINFORGE_SYNTHESIS_MODEL": "model-a",
        "ED4ALL_NLI_THRESHOLD": "0.72",
        "ED4ALL_EMBEDDING_DEVICE": "cuda",
        "ED4ALL_BLOOM_POLICY": "strict",
        "UNRELATED_VALUE": "ignored",
    }

    assert _synthesis_runtime_policy_identity(environment) == {
        "ED4ALL_BLOOM_POLICY": "strict",
        "ED4ALL_EMBEDDING_DEVICE": "cuda",
        "ED4ALL_NLI_THRESHOLD": "0.72",
        "TRAINFORGE_AGNOSTIC_SYNTHESIS": "1",
        "TRAINFORGE_SYNTHESIS_MODEL": "model-a",
    }


def test_resume_max8_to_max4_reuses_terminal_contract_and_retries_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduling changes preserve cache; real policy changes invalidate it."""

    course = _course_copy(tmp_path, "max8-to-max4-contract")
    chunks_path = course / "corpus" / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:3]
    assert len(chunks) == 3
    chunks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in chunks),
        encoding="utf-8",
    )
    target_id = str(chunks[-1]["id"])
    original_instruction = synthesis_module.synthesize_instruction_pair
    original_preference = synthesis_module.synthesize_preference_pair
    inject_transient = True

    def first_instruction(chunk, **kwargs):
        nonlocal inject_transient
        if str(chunk["id"]) == target_id and inject_transient:
            inject_transient = False
            raise SynthesisProviderError("temporary overload", code="503")
        return original_instruction(chunk, **kwargs)

    monkeypatch.setattr(
        synthesis_module, "synthesize_instruction_pair", first_instruction
    )
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", "8")
    with pytest.raises(SynthesisProviderError, match="retriable"):
        run_synthesis(
            course, "MAX-SWITCH", provider="mock",
            curriculum_from_graph=False,
        )

    specs = course / "training_specs"
    journal_path = specs / ".synthesis_generation_checkpoint.jsonl"
    pair_checkpoint = specs / ".synthesis_pairs_checkpoint.jsonl"
    first_journal = load_generation_journal(journal_path)
    transient_key = (target_id, "instruction", 0)
    assert first_journal[transient_key]["disposition"] == "transient"
    terminal_success = {
        key for key, row in first_journal.items()
        if row.get("disposition") == "success"
    }
    assert terminal_success
    first_pair_rows = [
        json.loads(line)
        for line in pair_checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terminal_pair_keys = {
        (row["chunk_id"], row["kind"], int(row.get("variant_index") or 0))
        for row in first_pair_rows
        if row.get("disposition") in {"accepted", "rejected"}
    }
    assert terminal_pair_keys
    assert all(row.get("contract_fingerprint") for row in first_pair_rows)

    resumed_calls: list[tuple[str, str]] = []
    fail_resumed_transient = True

    def resumed_instruction(chunk, **kwargs):
        nonlocal fail_resumed_transient
        resumed_calls.append((str(chunk["id"]), "instruction"))
        if str(chunk["id"]) == target_id and fail_resumed_transient:
            fail_resumed_transient = False
            raise SynthesisProviderError("temporary overload again", code="503")
        return original_instruction(chunk, **kwargs)

    def resumed_preference(chunk, **kwargs):
        resumed_calls.append((str(chunk["id"]), "preference"))
        return original_preference(chunk, **kwargs)

    monkeypatch.setattr(
        synthesis_module, "synthesize_instruction_pair", resumed_instruction
    )
    monkeypatch.setattr(
        synthesis_module, "synthesize_preference_pair", resumed_preference
    )
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", "4")
    with pytest.raises(SynthesisProviderError, match="retriable"):
        run_synthesis(
            course, "MAX-SWITCH", provider="mock",
            curriculum_from_graph=False,
        )
    assert resumed_calls == [(target_id, "instruction")]
    resumed_journal = load_generation_journal(journal_path)
    assert resumed_journal[transient_key]["disposition"] == "transient"
    assert resumed_journal[transient_key]["attempt"] == 2
    assert all(
        resumed_journal[key]["disposition"] == "success"
        for key in terminal_success
    )

    # A validator runtime-policy change is content/verdict-affecting and must
    # invalidate modern accepted/rejected rows and successful generations.
    resumed_calls.clear()
    monkeypatch.setenv("ED4ALL_NLI_THRESHOLD", "0.81")
    run_synthesis(
        course, "MAX-SWITCH", provider="mock",
        curriculum_from_graph=False,
    )
    assert resumed_calls
    assert any(key in terminal_success for key in {
        (chunk_id, kind, 0) for chunk_id, kind in resumed_calls
    })


def test_opaque_legacy_checkpoint_regenerates_nonaccepted_once_at_c4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live-checkpoint shape: keep fp-less accepts, never migrate opaque rows."""

    course = _course_copy(tmp_path, "opaque-legacy-copy")
    chunks_path = course / "corpus" / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:4]
    chunks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in chunks),
        encoding="utf-8",
    )
    target_id = str(chunks[-1]["id"])
    rejected_id = str(chunks[1]["id"])
    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_RUN_ID", "WF-OPAQUE-COPY")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", "8")
    original_instruction = synthesis_module.synthesize_instruction_pair
    original_preference = synthesis_module.synthesize_preference_pair
    failures_remaining = 2

    def flaky_instruction(chunk, **kwargs):
        nonlocal failures_remaining
        if str(chunk["id"]) == rejected_id:
            return InstructionSynthesisResult(
                pair=None,
                quality={"passed": False, "reason": "injected_terminal_rejection"},
                template_id="test-rejection",
                rationale=f"Current gate rejection for {rejected_id}",
                topic="rejected",
            )
        if str(chunk["id"]) == target_id and failures_remaining:
            failures_remaining -= 1
            raise SynthesisProviderError("temporary overload", code="503")
        return original_instruction(chunk, **kwargs)

    monkeypatch.setattr(
        synthesis_module, "synthesize_instruction_pair", flaky_instruction
    )
    with pytest.raises(SynthesisProviderError, match="retriable"):
        run_synthesis(
            course, "OPAQUE", provider="mock",
            curriculum_from_graph=False,
        )

    specs = course / "training_specs"
    pair_path = specs / ".synthesis_pairs_checkpoint.jsonl"
    journal_path = specs / ".synthesis_generation_checkpoint.jsonl"
    pair_rows = [
        json.loads(line)
        for line in pair_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted_keys = {
        (row["chunk_id"], row["kind"], int(row.get("variant_index") or 0))
        for row in pair_rows if row.get("disposition") == "accepted"
    }
    rejected_keys = {
        (row["chunk_id"], row["kind"], int(row.get("variant_index") or 0))
        for row in pair_rows if row.get("disposition") == "rejected"
    }
    assert accepted_keys and rejected_keys

    # Copy the observed live artifact shape into the isolated fixture:
    # accepted v1 rows predate fingerprints; every nonaccepted hash is opaque
    # because it includes historical source bytes and must not be rewritten.
    for row in pair_rows:
        if row.get("disposition") == "accepted":
            row["contract_fingerprint"] = None
        else:
            row["contract_fingerprint"] = "opaque-historical-contract"
    pair_path.write_text(
        "".join(json.dumps(row) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    journal_rows = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in journal_rows:
        row["fingerprint"] = "opaque-historical-contract"
    journal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in journal_rows),
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    def tracked_instruction(chunk, **kwargs):
        calls.append((str(chunk["id"]), "instruction"))
        return flaky_instruction(chunk, **kwargs)

    def tracked_preference(chunk, **kwargs):
        calls.append((str(chunk["id"]), "preference"))
        return original_preference(chunk, **kwargs)

    monkeypatch.setattr(
        synthesis_module, "synthesize_instruction_pair", tracked_instruction
    )
    monkeypatch.setattr(
        synthesis_module, "synthesize_preference_pair", tracked_preference
    )
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", "4")
    with pytest.raises(SynthesisProviderError, match="retriable"):
        run_synthesis(
            course, "OPAQUE", provider="mock",
            curriculum_from_graph=False,
        )

    called_keys = {(chunk_id, kind, 0) for chunk_id, kind in calls}
    assert accepted_keys.isdisjoint(called_keys)
    assert rejected_keys <= called_keys
    assert (target_id, "instruction", 0) in called_keys

    # The c4 pass appended current-contract terminal rows. A future c4 resume
    # performs only the still-transient unit; it never regenerates those rows.
    calls.clear()
    final_stats = run_synthesis(
        course, "OPAQUE", provider="mock",
        curriculum_from_graph=False,
    )
    assert calls == [(target_id, "instruction")]
    instruction_rows = [
        json.loads(line)
        for line in (specs / "instruction_pairs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    preference_rows = [
        json.loads(line)
        for line in (specs / "preference_pairs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len({row["prompt"] for row in instruction_rows}) == len(instruction_rows)
    assert len({row["prompt"] for row in preference_rows}) == len(preference_rows)
    assert len(instruction_rows) == final_stats.instruction_pairs_emitted
    assert len(preference_rows) == final_stats.preference_pairs_emitted
    telemetry = json.loads(
        (
            runs / "WF-OPAQUE-COPY" / "telemetry" / "training_synthesis.json"
        ).read_text(encoding="utf-8")
    )
    assert telemetry["state"] == "complete"
    assert telemetry["accepted_count"] == (
        len(instruction_rows) + len(preference_rows)
    )
    assert telemetry["completed_units"] == telemetry["total_units"] == 4


def test_legacy_default_does_not_create_generation_or_telemetry_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    course = _course_copy(tmp_path, "legacy-no-new-writes")
    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_RUN_ID", "WF-LEGACY")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.delenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", raising=False)

    run_synthesis(
        course,
        "LEGACY",
        provider="mock",
        max_pairs=1,
        curriculum_from_graph=False,
    )

    assert not (
        course / "training_specs" / ".synthesis_generation_checkpoint.jsonl"
    ).exists()
    assert not (
        runs / "WF-LEGACY" / "telemetry" / "training_synthesis.json"
    ).exists()


def test_real_run_three_resumes_keep_absolute_progress_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real synthesis replay never turns logical 3 chunks into 6 or 9."""

    course = _course_copy(tmp_path, "three-resumes")
    chunks = [
        json.loads(line)
        for line in (course / "corpus" / "chunks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    eligible_ids = [
        str(row.get("id") or row.get("chunk_id"))
        for row in chunks
        if row.get("learning_outcome_refs")
    ]
    assert len(eligible_ids) >= 7
    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_RUN_ID", "WF-THREE-RESUMES")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    original_instruction = synthesis_module.synthesize_instruction_pair
    original_preference = synthesis_module.synthesize_preference_pair
    calls = 0
    fail_target: list[str | None] = [eligible_ids[1]]
    failed_this_run: set[str] = set()

    def instruction(chunk, **kwargs):
        nonlocal calls
        calls += 1
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id"))
        if chunk_id == fail_target[0] and chunk_id not in failed_this_run:
            failed_this_run.add(chunk_id)
            raise SynthesisProviderError("temporary overload", code="503")
        return original_instruction(chunk, **kwargs)

    def preference(chunk, **kwargs):
        nonlocal calls
        calls += 1
        return original_preference(chunk, **kwargs)

    monkeypatch.setattr(synthesis_module, "synthesize_instruction_pair", instruction)
    monkeypatch.setattr(synthesis_module, "synthesize_preference_pair", preference)

    with pytest.raises(SynthesisProviderError, match="retriable"):
        run_synthesis(
            course, "RESUME", provider="mock", max_concurrent=4,
            curriculum_from_graph=False,
        )
    telemetry_path = (
        runs / "WF-THREE-RESUMES" / "telemetry" / "training_synthesis.json"
    )
    failed_one = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert failed_one["state"] == "failed"
    assert failed_one["transient_count"] == 1
    assert failed_one["transient_attempts"] == 1
    assert failed_one["recovered_units"] == 0
    fail_target[0] = eligible_ids[6]
    failed_this_run.clear()
    with pytest.raises(SynthesisProviderError, match="retriable"):
        run_synthesis(
            course, "RESUME", provider="mock", max_concurrent=4,
            curriculum_from_graph=False,
        )
    failed_two = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert failed_two["state"] == "failed"
    assert failed_two["transient_count"] == 1
    assert failed_two["transient_attempts"] == 2
    assert failed_two["recovered_units"] == 1
    fail_target[0] = None
    failed_this_run.clear()
    final_stats = run_synthesis(
        course, "RESUME", provider="mock", max_concurrent=4,
        curriculum_from_graph=False,
    )

    telemetry = json.loads(
        telemetry_path.read_text(encoding="utf-8")
    )
    accepted = (
        final_stats.instruction_pairs_emitted
        + final_stats.preference_pairs_emitted
    )
    rejected = (
        final_stats.instruction_pairs_rejected
        + final_stats.preference_pairs_rejected
    )
    assert telemetry["state"] == "complete"
    assert telemetry["completed_units"] == telemetry["total_units"]
    assert telemetry["terminal_units"] == telemetry["completed_units"]
    assert telemetry["accepted_count"] == accepted
    assert telemetry["rejected_count"] == rejected
    assert telemetry["provider_results"] == calls
    assert telemetry["transient_attempts"] == 2
    assert telemetry["recovered_units"] == 2
    assert telemetry["transient_count"] == 0


@pytest.mark.parametrize("workers", [4, 32, 48])
@pytest.mark.parametrize("failure_kind", ["transient", "truncated", "fatal"])
def test_terminal_snapshot_is_written_after_all_sibling_journals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    failure_kind: str,
) -> None:
    """Real outer-finally barrier covers every started sibling at 4/32/48."""

    course = _course_copy(tmp_path, f"terminal-{failure_kind}-{workers}")
    source = json.loads(
        (course / "corpus" / "chunks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )
    chunks = []
    for index in range(workers):
        row = json.loads(json.dumps(source))
        row["id"] = f"chunk-{index:03d}"
        for claim in row.get("key_claims", []):
            claim["source_chunk_ids"] = [row["id"]]
        chunks.append(row)
    (course / "corpus" / "chunks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in chunks),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    run_id = f"WF-{failure_kind.upper()}-{workers}"
    monkeypatch.setenv("ED4ALL_RUN_ID", run_id)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    sibling_started = threading.Event()
    lock = threading.Lock()
    calls = 0
    resume_mode = False
    successful_units: set[tuple[str, str, int]] = set()

    def instruction(chunk, **kwargs):
        nonlocal calls
        with lock:
            calls += 1
        chunk_id = str(chunk["id"])
        if resume_mode and (chunk_id, "instruction", 0) in successful_units:
            raise AssertionError(f"duplicate instruction call for {chunk_id}")
        if chunk_id == "chunk-000":
            assert sibling_started.wait(timeout=10)
            if failure_kind in {"transient", "truncated"}:
                code = "503" if failure_kind == "transient" else "output_truncated"
                raise SynthesisProviderError(
                    "temporary retriable generation failure", code=code,
                )
            raise RuntimeError("deterministic injected fatal")
        sibling_started.set()
        time.sleep(0.02)
        return InstructionSynthesisResult(
            pair={"prompt": f"prompt {chunk_id}", "completion": "answer"},
            quality={"valid": True},
            template_id="test",
            rationale=f"dynamic rationale for {chunk_id}",
            topic="topic",
        )

    def preference(chunk, **kwargs):
        nonlocal calls
        chunk_id = str(chunk["id"])
        if resume_mode and (chunk_id, "preference", 0) in successful_units:
            raise AssertionError(f"duplicate preference call for {chunk_id}")
        if chunk_id != "chunk-000":
            time.sleep(0.02)
        with lock:
            calls += 1
        return PreferenceSynthesisResult(
            pair={"prompt": "prompt", "chosen": "yes", "rejected": "no"},
            quality={"valid": True},
            rationale=f"dynamic preference rationale for {chunk['id']}",
            source="rule_synthesized",
        )

    monkeypatch.setattr(synthesis_module, "synthesize_instruction_pair", instruction)
    monkeypatch.setattr(synthesis_module, "synthesize_preference_pair", preference)
    expected_error = (
        SynthesisProviderError
        if failure_kind == "transient"
        else RuntimeError
    )
    with pytest.raises(expected_error):
        run_synthesis(
            course, "DRAIN", provider="mock", max_concurrent=workers,
            curriculum_from_graph=False,
        )

    telemetry_path = (
        runs / run_id / "telemetry" / "training_synthesis.json"
    )
    journal_path = (
        course / "training_specs" / ".synthesis_generation_checkpoint.jsonl"
    )
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    summary = summarize_generation_journal(journal_path)
    if failure_kind == "truncated":
        truncated_row = load_generation_journal(journal_path)[
            ("chunk-000", "instruction", 0)
        ]
        assert truncated_row["disposition"] == "fatal"
        assert truncated_row["was_transient"] is False
        assert truncated_row["error_code"] == "output_truncated"
        assert truncated_row["truncation_kind"] == "output_truncated"
    assert calls > 1
    assert telemetry["state"] == "failed"
    assert telemetry["provider_results"] == calls
    assert telemetry["provider_results"] == summary["provider_results"]
    assert telemetry["transient_count"] == summary["transient_pending_units"]
    assert telemetry["transient_attempts"] == summary["transient_attempts"]
    assert telemetry["fatal_units"] == summary["fatal_units"]
    journal_size = journal_path.stat().st_size
    telemetry_bytes = telemetry_path.read_bytes()
    time.sleep(0.02)
    assert journal_path.stat().st_size == journal_size
    assert telemetry_path.read_bytes() == telemetry_bytes

    successful_units = {
        key
        for key, row in load_generation_journal(journal_path).items()
        if row.get("disposition") == "success"
    }
    assert successful_units
    resume_mode = True
    with pytest.raises(expected_error):
        run_synthesis(
            course, "DRAIN", provider="mock", max_concurrent=workers,
            curriculum_from_graph=False,
        )
    # The wrappers raise if resume re-calls any previously successful unit.


def test_resume_accepted_checkpoint_and_stale_fatal_at_concurrency_8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: replay owns its fingerprint callback; no local-scope leak."""

    course = _course_copy(tmp_path, "resume-stale-fatal")
    chunks_path = course / "corpus" / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:2]
    chunks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in chunks),
        encoding="utf-8",
    )
    target_id = str(chunks[1]["id"])
    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_RUN_ID", "WF-STALE-FATAL")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    original_instruction = synthesis_module.synthesize_instruction_pair
    original_preference = synthesis_module.synthesize_preference_pair
    inject_fatal = True

    def instruction(chunk, **kwargs):
        nonlocal inject_fatal
        if str(chunk["id"]) == target_id and inject_fatal:
            inject_fatal = False
            raise RuntimeError("injected deterministic fatal")
        return original_instruction(chunk, **kwargs)

    monkeypatch.setattr(
        synthesis_module, "synthesize_instruction_pair", instruction
    )
    with pytest.raises(RuntimeError, match="deterministic fatal"):
        run_synthesis(
            course, "STALE", provider="mock", max_concurrent=8,
            curriculum_from_graph=False,
        )

    pair_checkpoint = (
        course / "training_specs" / ".synthesis_pairs_checkpoint.jsonl"
    )
    generation_journal_path = (
        course / "training_specs" / ".synthesis_generation_checkpoint.jsonl"
    )
    pair_rows = [
        json.loads(line)
        for line in pair_checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("disposition") == "accepted" for row in pair_rows)
    generation_before = load_generation_journal(generation_journal_path)
    fatal_key = (target_id, "instruction", 0)
    assert generation_before[fatal_key]["disposition"] == "fatal"
    first_telemetry = json.loads(
        (
            runs
            / "WF-STALE-FATAL"
            / "telemetry"
            / "training_synthesis.json"
        ).read_text(encoding="utf-8")
    )
    assert first_telemetry["accepted_count"] > 0

    # Change only the failed chunk's source contract. Its old fatal
    # fingerprint is now stale and must retry; chunk zero's accepted pair
    # checkpoint and successful generations must not dispatch again.
    chunks[1]["text"] += " This sentence changes the source fingerprint."
    chunks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in chunks),
        encoding="utf-8",
    )
    resumed_calls: list[tuple[str, str]] = []

    def resumed_instruction(chunk, **kwargs):
        resumed_calls.append((str(chunk["id"]), "instruction"))
        return original_instruction(chunk, **kwargs)

    def resumed_preference(chunk, **kwargs):
        resumed_calls.append((str(chunk["id"]), "preference"))
        return original_preference(chunk, **kwargs)

    monkeypatch.setattr(
        synthesis_module, "synthesize_instruction_pair", resumed_instruction
    )
    monkeypatch.setattr(
        synthesis_module, "synthesize_preference_pair", resumed_preference
    )
    final_stats = run_synthesis(
        course, "STALE", provider="mock", max_concurrent=8,
        curriculum_from_graph=False,
    )

    assert resumed_calls == [
        (target_id, "instruction"),
        (target_id, "preference"),
    ]
    assert (
        final_stats.instruction_pairs_emitted
        + final_stats.preference_pairs_emitted
    ) >= first_telemetry["accepted_count"]
    final_telemetry = json.loads(
        (
            runs
            / "WF-STALE-FATAL"
            / "telemetry"
            / "training_synthesis.json"
        ).read_text(encoding="utf-8")
    )
    assert final_telemetry["state"] == "complete"
    assert final_telemetry["accepted_count"] >= first_telemetry["accepted_count"]


@pytest.mark.parametrize("workers", [32, 48])
def test_32_48_restore_input_order_and_bound_memory(workers: int) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def work(value: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # Reverse-duration skew forces physical completion out of order.
        time.sleep((workers - (value % workers)) * 0.00005)
        with lock:
            active -= 1
        return value * 10

    ordered = BoundedOrderedMap(
        range(workers * 3), work, max_concurrent=workers
    )
    samples: list[tuple[int, int, int]] = []
    monitoring = threading.Event()

    def monitor() -> None:
        while not monitoring.is_set():
            samples.append(ordered.metrics_snapshot())
            time.sleep(0.0001)

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    results = list(ordered)
    monitoring.set()
    monitor_thread.join()

    assert [row.sequence for row in results] == list(range(workers * 3))
    assert [row.value for row in results] == [
        value * 10 for value in range(workers * 3)
    ]
    assert 1 < peak <= workers
    assert ordered.peak_in_flight <= workers
    assert ordered.submitted_count == ordered.yielded_count
    assert samples
    assert all(
        active + queued == in_flight <= workers
        for active, queued, in_flight in samples
    )


def test_stop_drains_submitted_units_without_submitting_more() -> None:
    stop = threading.Event()
    completed: list[int] = []

    def work(value: int) -> int:
        time.sleep(0.001)
        completed.append(value)
        if value == 0:
            stop.set()
        return value

    ordered = BoundedOrderedMap(
        range(200),
        work,
        max_concurrent=32,
        stop_requested=stop.is_set,
    )
    rows = list(ordered)

    assert ordered.stopped_early is True
    assert ordered.submitted_count <= 32
    assert ordered.yielded_count == ordered.submitted_count
    assert [row.value for row in rows] == list(range(ordered.submitted_count))
    assert sorted(completed) == list(range(ordered.submitted_count))


def test_worker_exception_fails_loudly_without_serial_fallback() -> None:
    calls: dict[int, int] = {}
    lock = threading.Lock()

    def work(value: int) -> int:
        with lock:
            calls[value] = calls.get(value, 0) + 1
        if value == 3:
            raise RuntimeError("synthetic provider failure")
        return value

    ordered = BoundedOrderedMap(range(20), work, max_concurrent=8)
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        list(ordered)

    assert all(count == 1 for count in calls.values())


def test_default_one_uses_caller_thread_and_no_lookahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", raising=False)
    caller = threading.get_ident()
    seen: list[tuple[int, int]] = []

    def source():
        for value in range(4):
            seen.append((value, -1))
            yield value

    def work(value: int) -> int:
        # At most the current source unit has been pulled.
        assert len(seen) == value + 1
        seen[value] = (value, threading.get_ident())
        return value

    ordered = BoundedOrderedMap(
        source(),
        work,
        max_concurrent=resolve_synthesis_max_concurrent(),
    )
    assert [row.value for row in ordered] == [0, 1, 2, 3]
    assert all(thread_id == caller for _, thread_id in seen)


@pytest.mark.parametrize("raw", [None, "", "garbage", "0", "-2"])
def test_resolver_invalid_values_preserve_legacy_default(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", raising=False)
    else:
        monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", raw)
    assert resolve_synthesis_max_concurrent() == 1


@pytest.mark.parametrize("value", [2, 8, 32, 48])
def test_resolver_accepts_valid_concurrency(
    monkeypatch: pytest.MonkeyPatch, value: int
) -> None:
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", str(value))
    assert resolve_synthesis_max_concurrent() == value


def test_resolver_fails_loud_above_validated_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MAX_CONCURRENT", "49")
    with pytest.raises(ValueError, match="hard ceiling 48"):
        resolve_synthesis_max_concurrent()


@pytest.mark.parametrize("workers", [32, 48])
def test_run_synthesis_32_48_matches_serial_artifact_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workers: int,
) -> None:
    """The concurrent lane changes throughput, never artifact semantics."""

    monkeypatch.delenv("TRAINFORGE_SYNTHESIS_PROVIDER", raising=False)
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    serial = _course_copy(tmp_path, "serial")
    concurrent = _course_copy(tmp_path, f"concurrent-{workers}")

    serial_stats = run_synthesis(
        corpus_dir=serial,
        course_code="TEST_SERIAL",
        provider="mock",
        seed=17,
        max_concurrent=1,
    )
    concurrent_stats = run_synthesis(
        corpus_dir=concurrent,
        course_code="TEST_CONCURRENT",
        provider="mock",
        seed=17,
        max_concurrent=workers,
    )

    assert concurrent_stats.as_dict() == serial_stats.as_dict()
    for filename in ("instruction_pairs.jsonl", "preference_pairs.jsonl"):
        def _rows(base: Path):
            rows = [
                json.loads(line)
                for line in (
                    base / "training_specs" / filename
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in rows:
                # Capture ids are intentionally unique per run; every other
                # artifact field and the row order must match.
                row.pop("decision_capture_id", None)
            return rows

        assert _rows(concurrent) == _rows(serial)
