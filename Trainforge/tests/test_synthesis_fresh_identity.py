from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import pytest

from Trainforge.synthesis.synthesis_journal import (
    GenerationJournal,
    load_generation_journal,
)
from Trainforge.synthesis.synthesis_fresh_start import (
    MARKER_NAME,
    FreshStartError,
    bind_fresh_start_run_contract,
)
from Trainforge.synthesis.synthesize_training import (
    _append_synthesis_pairs_checkpoint,
    _load_synthesis_pairs_checkpoint,
)
from Trainforge.generators.staged.provider import (
    StagedSynthesisProvider,
)


IDENTITY = "fresh-start-identity-0001"
DIGEST = "a" * 64
RUN_CONTRACT = "c" * 64


def _journal_row(disposition: str = "success") -> dict:
    return {
        "chunk_id": "chunk-1",
        "kind": "instruction",
        "variant_index": 0,
        "fingerprint": "fp",
        "attempt": 1,
        "disposition": disposition,
    }


@pytest.mark.parametrize("disposition", ["success", "transient", "fatal"])
def test_generation_rows_bind_exact_identity(
    tmp_path: Path, disposition: str,
) -> None:
    path = tmp_path / "generation.jsonl"
    GenerationJournal(
        path, fresh_start_id=IDENTITY, marker_digest=DIGEST,
    ).append(_journal_row(disposition))
    row = load_generation_journal(
        path,
        expected_fresh_start_id=IDENTITY,
        expected_marker_digest=DIGEST,
    )[("chunk-1", "instruction", 0)]
    assert row["fresh_start_id"] == IDENTITY
    assert row["fresh_start_marker_digest"] == DIGEST


@pytest.mark.parametrize(
    ("fresh_id", "digest"),
    [(None, None), ("wrong", DIGEST), (IDENTITY, "b" * 64)],
)
def test_generation_replay_skips_mismatched_identity_rows(
    tmp_path: Path, fresh_id: str | None, digest: str | None,
) -> None:
    # Changed behavior: mismatched identity rows are SKIPPED instead of raising.
    # The row is a cache miss and will be re-run.
    path = tmp_path / "generation.jsonl"
    payload = {"schema_version": 2, **_journal_row()}
    if fresh_id is not None:
        payload["fresh_start_id"] = fresh_id
        payload["fresh_start_marker_digest"] = digest
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    # Load with expected identity: the mismatched row is skipped silently
    result = load_generation_journal(
        path,
        expected_fresh_start_id=IDENTITY,
        expected_marker_digest=DIGEST,
    )
    # The mismatched row is not loaded; cache is empty
    assert len(result) == 0


def test_generation_concurrent_same_identity_is_serialized(tmp_path: Path) -> None:
    path = tmp_path / "generation.jsonl"
    journal = GenerationJournal(
        path, fresh_start_id=IDENTITY, marker_digest=DIGEST,
    )
    threads = [
        threading.Thread(
            target=journal.append,
            args=({**_journal_row(), "chunk_id": f"chunk-{index}"},),
        )
        for index in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(load_generation_journal(
        path,
        expected_fresh_start_id=IDENTITY,
        expected_marker_digest=DIGEST,
    )) == 12


def test_generation_replay_skips_code_contract_change(tmp_path: Path) -> None:
    # Changed behavior: run-contract mismatch rows are SKIPPED instead of raising.
    # The row is a cache miss and will be re-run with new contract.
    path = tmp_path / "generation.jsonl"
    GenerationJournal(
        path,
        fresh_start_id=IDENTITY,
        marker_digest=DIGEST,
        run_contract_sha256=RUN_CONTRACT,
    ).append(_journal_row())
    # Load with different run contract: the mismatched row is skipped
    result = load_generation_journal(
        path,
        expected_fresh_start_id=IDENTITY,
        expected_marker_digest=DIGEST,
        expected_run_contract_sha256="d" * 64,
    )
    # The mismatched row is not loaded; cache is empty (row will re-run)
    assert len(result) == 0


def test_generation_journal_rejects_duplicate_terminal_key(
    tmp_path: Path,
) -> None:
    """A unit must not reach a terminal disposition twice.

    Exactly one source-order writer closes each unit, so a second terminal
    append means double dispatch, a resume that re-queued finished work, or two
    writers racing — all of which corrupt the accepted set. This was briefly
    downgraded to log-and-return, which hid the bug at the one moment it
    surfaces; the raise is the alarm.
    """
    path = tmp_path / "generation.jsonl"
    journal = GenerationJournal(
        path,
        fresh_start_id=IDENTITY,
        marker_digest=DIGEST,
        run_contract_sha256=RUN_CONTRACT,
    )
    journal.append(_journal_row())

    with pytest.raises(RuntimeError, match="duplicate generation terminal"):
        journal.append(_journal_row())

    # The first terminal stands and the file is not double-written.
    result = load_generation_journal(
        path,
        expected_fresh_start_id=IDENTITY,
        expected_marker_digest=DIGEST,
        expected_run_contract_sha256=RUN_CONTRACT,
    )
    assert len(result) == 1
    assert result[("chunk-1", "instruction", 0)]["disposition"] == "success"


def test_pair_writer_rejects_duplicate_without_changing_bytes() -> None:
    handle = io.StringIO()
    handle._terminal_semantic_keys = set()
    handle._enforce_terminal_uniqueness = True
    kwargs = dict(
        chunk_id="chunk-duplicate",
        kind="instruction",
        variant_index=0,
        pair={"prompt": "p", "completion": "c"},
        provider="local",
    )
    _append_synthesis_pairs_checkpoint(handle, **kwargs)
    before = handle.getvalue()
    with pytest.raises(RuntimeError, match="duplicate pair terminal"):
        _append_synthesis_pairs_checkpoint(handle, **kwargs)
    assert handle.getvalue() == before


def test_code_change_after_pause_fails_closed(
    tmp_path: Path,
) -> None:
    """A generation-contract change after a pause must stop the run.

    This is the tripwire for mixed-version training data: the accepted rows
    were produced by one implementation and the remaining ones would be
    produced by another, with nothing downstream able to tell them apart once
    they are pairs. It was briefly rewritten to assert a rebind instead — the
    guard and its alarm were removed together, which is the failure this test
    now exists to prevent twice over.
    """
    marker = {
        "schema_version": 1,
        "fresh_start_id": IDENTITY,
    }
    (tmp_path / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    sealed = bind_fresh_start_run_contract(
        tmp_path,
        expected_fresh_start_id=IDENTITY,
        run_contract_sha256=RUN_CONTRACT,
        resume_artifacts_exist=False,
        contract_components={"files": {"provider.py": "original"}},
    )
    assert sealed["synthesis_run_contract_sha256"] == RUN_CONTRACT

    with pytest.raises(FreshStartError) as excinfo:
        bind_fresh_start_run_contract(
            tmp_path,
            expected_fresh_start_id=IDENTITY,
            run_contract_sha256="d" * 64,
            resume_artifacts_exist=True,
            contract_components={"files": {"provider.py": "changed"}},
        )
    # The operator has to be able to act on this without bisecting by restart.
    assert "changed components" in str(excinfo.value)


def test_contract_mismatch_fails_closed_without_touching_sidecars(
    tmp_path: Path,
) -> None:
    """Failing closed must not also destroy the work already done.

    The run stops, but the in-progress sidecar is left byte-identical so the
    operator can archive it rather than losing accepted rows to the guard.
    """
    (tmp_path / MARKER_NAME).write_text(json.dumps({
        "schema_version": 1,
        "fresh_start_id": IDENTITY,
        "synthesis_run_contract_sha256": RUN_CONTRACT,
        "synthesis_run_contract_components": {
            "files": {"provider.py": "old"},
        },
    }), encoding="utf-8")
    sidecar = tmp_path / "instruction_pairs.jsonl.in_progress"
    before = b'{"preserve":"exactly"}\n'
    sidecar.write_bytes(before)

    with pytest.raises(FreshStartError):
        bind_fresh_start_run_contract(
            tmp_path,
            expected_fresh_start_id=IDENTITY,
            run_contract_sha256="d" * 64,
            resume_artifacts_exist=True,
            contract_components={"files": {"provider.py": "new"}},
        )
    assert sidecar.read_bytes() == before


def test_unsealed_marker_with_pause_artifacts_fails_closed(
    tmp_path: Path,
) -> None:
    marker = {
        "schema_version": 1,
        "fresh_start_id": IDENTITY,
    }
    (tmp_path / MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(FreshStartError, match="archive and restart from zero"):
        bind_fresh_start_run_contract(
            tmp_path,
            expected_fresh_start_id=IDENTITY,
            run_contract_sha256=RUN_CONTRACT,
            resume_artifacts_exist=True,
        )


@pytest.mark.parametrize("disposition", ["accepted", "rejected", "ineligible"])
def test_pair_dispositions_bind_exact_identity(disposition: str) -> None:
    handle = io.StringIO()
    handle._fresh_start_id = IDENTITY
    handle._fresh_start_marker_digest = DIGEST
    _append_synthesis_pairs_checkpoint(
        handle,
        chunk_id="chunk-1",
        kind="instruction",
        variant_index=0,
        pair={"prompt": "p", "completion": "c"} if disposition == "accepted" else None,
        provider="local",
        disposition=disposition,
    )
    row = json.loads(handle.getvalue())
    assert row["fresh_start_id"] == IDENTITY
    assert row["fresh_start_marker_digest"] == DIGEST


def test_pair_replay_rejects_mixed_identity(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    base = {
        "schema_version": "v1",
        "chunk_id": "chunk-1",
        "kind": "instruction",
        "variant_index": 0,
        "disposition": "ineligible",
    }
    path.write_text(
        json.dumps({
            **base,
            "fresh_start_id": IDENTITY,
            "fresh_start_marker_digest": DIGEST,
        }) + "\n" + json.dumps({
            **base,
            "chunk_id": "chunk-2",
            "fresh_start_id": "copied-stale-identity",
            "fresh_start_marker_digest": DIGEST,
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load_synthesis_pairs_checkpoint(
            path,
            expected_fresh_start_id=IDENTITY,
            expected_marker_digest=DIGEST,
        )


def test_staged_raw_audit_refs_are_partitioned_and_identity_bound(
    tmp_path: Path,
) -> None:
    class Capture:
        output_dir = tmp_path
        fresh_start_id = IDENTITY
        fresh_start_marker_digest = DIGEST

        def __init__(self) -> None:
            self.calls = []

        def log_decision(self, **kwargs) -> None:
            self.calls.append(kwargs)

    class Base:
        _capture = Capture()
        _model = "test-model"
        _provider_name = "local"
        _max_tokens = 128

    provider = StagedSynthesisProvider(Base())
    provider._capture_call(
        stage="sft_plan",
        chunk_id="chunk-1",
        attempt=1,
        prompt="prompt bytes",
        raw_response='{"ok":true}',
        usage={"prompt_tokens": 2, "completion_tokens": 3},
        validation_error=None,
        context_headroom_tokens=1000,
    )
    call = Base._capture.calls[0]
    context = json.loads(call["context"])
    assert context["fresh_start_id"] == IDENTITY
    assert context["fresh_start_marker_digest"] == DIGEST
    assert f"fresh-{IDENTITY}" in call["prompt_ref"]
    assert Path(call["prompt_ref"]).read_text(encoding="utf-8") == "prompt bytes"
    assert Path(call["outputs"][0]["path"]).read_text(encoding="utf-8") == '{"ok":true}'
