"""Process-level recovery regressions for the benchmark micro journal."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
from pathlib import Path

import pytest

from Trainforge.generators._synthesis_common import SynthesisProviderError
from Trainforge.generators.staged_synthesis_micro import (
    MicroResumeStore,
    micro_contract_fingerprint,
)
from Trainforge.scripts.staged_window_abcd_pilot import (
    MICRO_SYNTHESIS_CONTRACT,
    _digest,
    synthesis_contract_identity,
    summarize,
    verify_micro_journals,
)


def _append_terminal_then_block(path: str, ready: object) -> None:
    store = MicroResumeStore(Path(path), fingerprint="a" * 64)
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1, state="started",
    )
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1, state="terminal",
        artifact={"learner_task": "Compare two generic quantities."},
    )
    ready.send("terminal-fsynced")
    signal.pause()


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL requires POSIX")
def test_actual_sigkill_preserves_fsynced_terminal_for_exact_once_resume(
    tmp_path,
):
    journal = tmp_path / "cell" / "micro-journals" / "unit.jsonl"
    parent, child = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_append_terminal_then_block,
        args=(str(journal), child),
    )
    process.start()
    assert parent.poll(10)
    assert parent.recv() == "terminal-fsynced"

    rows_before = journal.read_bytes()
    assert [json.loads(line)["state"] for line in rows_before.splitlines()] == [
        "started", "terminal",
    ]
    os.kill(process.pid, signal.SIGKILL)
    process.join(10)
    assert process.exitcode == -signal.SIGKILL

    replay = MicroResumeStore(journal, fingerprint="a" * 64).load()
    provider_calls = 0
    if "instruction:A" not in replay:
        provider_calls += 1
    assert provider_calls == 0
    assert replay["instruction:A"] == {
        "learner_task": "Compare two generic quantities.",
    }
    assert journal.read_bytes() == rows_before

    # Plain restart executes only work that never acquired a durable start.
    resumed_store = MicroResumeStore(journal, fingerprint="a" * 64)
    if "instruction:B:slot-0" not in replay:
        provider_calls += 1
        resumed_store.append(
            unit="instruction:B:slot-0", stage="B", slot=0, attempt=1,
            state="started",
        )
        resumed_store.append(
            unit="instruction:B:slot-0", stage="B", slot=0, attempt=1,
            state="terminal", artifact={"claim": "A generic recorded claim."},
        )
    final = resumed_store.load()
    assert provider_calls == 1
    assert set(final) == {"instruction:A", "instruction:B:slot-0"}
    assert len(journal.read_text().splitlines()) == 4


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL requires POSIX")
def test_actual_sigkill_after_started_only_fails_ambiguous_resume(tmp_path):
    journal = tmp_path / "cell" / "micro-journals" / "unit.jsonl"
    store = MicroResumeStore(journal, fingerprint="b" * 64)
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1, state="started",
    )
    with pytest.raises(SynthesisProviderError) as exc:
        MicroResumeStore(journal, fingerprint="b" * 64).load()
    assert exc.value.code == "staged_micro_resume_ambiguous"


def test_eight_row_synthetic_af_replay_has_exact_authority_and_no_calls(
    tmp_path,
):
    """Eight recorded synthetic rows exercise A-F journal replay offline."""
    manifest_rows = []
    for index in range(8):
        kind = "instruction" if index < 4 else "preference"
        manifest_rows.append({
            "chunk_id": f"generic-chunk-{index}",
            "chunk_sha256": f"{index + 1:064x}",
            "kind": kind,
            "variant": "generic-contract",
            "repetition": 0,
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows)
    )
    cell = tmp_path / "c1"
    cell.mkdir()
    preflight = {
        "synthesis_contract": synthesis_contract_identity(
            MICRO_SYNTHESIS_CONTRACT
        ),
        "run_contract_sha256": "c" * 64,
        "telemetry_identity": {"expected_model": "served-model"},
        "pilot_run_id": "pilot-c1", "cell_id": "c1",
        "manifest_sha256": __import__("hashlib").sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "eligibility_sha256": "d" * 64,
        "micro_journal_root": "micro-journals",
    }
    intent_rows = []
    results = []
    terminal_counts = {stage: 0 for stage in "ABCDEF"}
    calls = {stage: 0 for stage in "ABDEF"}
    stage_budgets = synthesis_contract_identity(
        MICRO_SYNTHESIS_CONTRACT
    )["component_hashes"]["stage_token_budget"]["max_tokens"]
    assert stage_budgets == {
        "A": 2048, "B": 1536, "D": 1536, "E": 1280, "F": 1024,
    }
    for row in manifest_rows:
        draft_identity = {
            **row,
            "draft": {
                "provider": "local",
                "source_chunk_id": row["chunk_id"],
            },
        }
        identity = {
            "contract": micro_contract_fingerprint(),
            "kind": row["kind"],
            "chunk_id": row["chunk_id"],
            "chunk_sha256": row["chunk_sha256"],
            "draft_identity": draft_identity,
            "draft_sha256": _digest(draft_identity),
            "model": "served-model",
            "pilot_run_id": "pilot-c1",
            "cell_id": "c1",
            "manifest_sha256": preflight["manifest_sha256"],
            "eligibility_sha256": "d" * 64,
            "execution_fingerprint": "c" * 64,
        }
        identity["input_fingerprint"] = _digest({
            key: identity[key] for key in (
                "manifest_sha256", "eligibility_sha256",
                "chunk_sha256", "draft_sha256",
            )
        })
        fingerprint = _digest(identity)
        store = MicroResumeStore(
            cell / "micro-journals" / f"{fingerprint}.jsonl",
            fingerprint=fingerprint, store_identity=identity,
        )
        stages = "ABCD" if row["kind"] == "instruction" else "ABCDEF"
        for stage in stages:
            slot = 0 if stage == "B" else None
            unit = f"{row['kind']}:{stage}" + (
                ":slot-0" if slot is not None else ""
            )
            store.append(
                unit=unit, stage=stage, slot=slot, attempt=1, state="started",
            )
            store.append(
                unit=unit, stage=stage, slot=slot, attempt=1, state="terminal",
                artifact={"stage": stage, "recorded": True},
            )
            terminal_counts[stage] += 1
            if stage != "C":
                calls[stage] += 1
                intent_rows.append({
                    "stage": f"micro_{stage}_recorded",
                    "logical_attempt": 1,
                    "unit": f"{row['chunk_id']}:{row['kind']}:{stage}",
                    "max_tokens": stage_budgets[stage],
                })
        row_calls = {stage: 1 for stage in stages}
        row_metrics = {
            stage: {
                "calls": 0 if stage == "C" else 1,
                "completed_units": 1,
                "repairs": 0,
                "deterministic_events": 1 if stage == "C" else 0,
                "total_tokens": 0 if stage == "C" else 10,
            }
            for stage in stages
        }
        results.append({
            **row,
            "accepted": row["kind"] == "instruction",
            "error_code": (
                None if row["kind"] == "instruction"
                else "staged_micro_F_one_fault_rejected_invalid"
            ),
            "microstage_calls": row_calls,
            "claim_slot_calls": {"0": 1},
            "microstage_metrics": row_metrics,
        })
    (cell / "call-intents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in intent_rows)
    )
    summary = summarize(results)
    report = verify_micro_journals(
        cell, summary=summary, preflight=preflight, results=results,
    )
    assert len(report["journals"]) == 8
    assert sum(report["completed_by_microstage"].values()) == 40
    assert len(intent_rows) == 32
    assert len({
        (row["stage"], row["unit"], row["logical_attempt"])
        for row in intent_rows
    }) == 32
    tampered = [dict(row) for row in intent_rows]
    tampered[0]["max_tokens"] -= 1
    (cell / "call-intents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tampered)
    )
    with pytest.raises(ValueError, match="token budget differs"):
        verify_micro_journals(
            cell, summary=summary, preflight=preflight, results=results,
        )
    assert len({
        (row["chunk_id"], row["kind"], row["variant"])
        for row in results
    }) == 8
