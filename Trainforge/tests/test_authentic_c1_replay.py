import hashlib
import json
from pathlib import Path

import pytest

from Trainforge.scripts.authentic_c1_replay import (
    build_authentic_inventory,
    replay_once,
    run_replay_twice,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path):
    source = tmp_path / "source"
    raw = source / "raw"
    raw.mkdir(parents=True)
    manifest = []
    for index in range(8):
        manifest.append({
            "chunk_id": f"generic-row-{index}",
            "chunk_sha256": f"{index + 1:064x}",
            "kind": "instruction" if index < 4 else "preference",
        })
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest_path, manifest)
    graph = [
        ("dialect-preflight", "staged_synthesis:dialect_preflight", 1, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_A_task_design", 1, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_A_task_design", 2, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_B_claim_0_attempt_1", 1, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_B_claim_1_attempt_1", 1, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_B_claim_1_attempt_2", 1, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_D_dpo_chosen", 1, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_D_dpo_chosen", 2, "stop"),
        ("generic-row-0:instruction:contract:r0", "staged_synthesis:micro_D_dpo_chosen", 3, "stop"),
        ("generic-row-1:instruction:contract:r0", "staged_synthesis:micro_A_task_design", 1, "stop"),
        ("generic-row-1:instruction:contract:r0", "staged_synthesis:micro_A_task_design", 2, "stop"),
        ("generic-row-1:instruction:contract:r0", "staged_synthesis:micro_A_task_design", 3, "stop"),
        ("generic-row-4:preference:contract:r0", "staged_synthesis:micro_A_task_design", 1, "length"),
    ]
    attempts = []
    intents = []
    captures = []
    for position, (unit, stage, attempt, finish) in enumerate(graph):
        request = json.dumps({"position": position}).encode()
        if stage.endswith("dialect_preflight"):
            content = json.dumps({"probe": "ready"})
        elif "micro_B_" in stage:
            content = json.dumps({
                "claim": "A generic supported claim.",
                "evidence_quote": "A generic supported claim.",
                "source_block_id": "block-1",
            })
        elif "micro_D_" in stage:
            content = json.dumps({
                "prompt": "Analyze the generic evidence.",
                "chosen": "The evidence supports the generic claim.",
            })
        else:
            content = (
                '{"objective_id":"co-1","bloom_level":"analyze",'
                '"learner_task":"Analyze the generic evidence.",'
                '"generated_givens":[]}'
            )
            if finish == "length":
                content = '{"objective_id":"co-1"'
        response = json.dumps({
            "id": f"recorded-response-{position}",
            "choices": [{
                "message": {"content": content},
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }).encode()
        request_sha = hashlib.sha256(request).hexdigest()
        response_sha = hashlib.sha256(response).hexdigest()
        request_path = raw / f"request-{request_sha}.bin"
        response_path = raw / f"response-{response_sha}.bin"
        request_path.write_bytes(request)
        response_path.write_bytes(response)
        common = {
            "unit": unit, "stage": stage, "attempt": attempt,
            "request_sha256": request_sha, "model": "generic-model",
            "actual_timeout_seconds": 30.0,
            "endpoint": "http://replay.invalid/v1/chat/completions",
            "request_raw_ref": {
                "path": str(request_path.relative_to(tmp_path)),
                "sha256": request_sha, "bytes": len(request),
            },
        }
        attempts.append({"event": "http_attempt_started", **common})
        attempts.append({
            "event": "http_attempt_terminal", **common,
            "response_raw_ref": {
                "path": str(response_path.relative_to(tmp_path)),
                "sha256": response_sha, "bytes": len(response),
            },
            "finish_reason": finish, "http_status": 200,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        intents.append({
            "unit": unit, "stage": stage, "logical_attempt": attempt,
            "request_sha256": request_sha, "model": "generic-model",
        })
        captures.append({
            "decision_type": "synthesis_provider_call",
            "context": json.dumps({
                "intent_request_sha256": request_sha,
                "intent_unit": unit,
                "intent_stage": stage,
                "prompt_sha256": f"{position + 101:064x}",
                "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "validation_error": (
                    "staged_output_truncated" if finish == "length" else None
                ),
                "validation_evidence": {
                    "validator_stage": stage,
                    "validator_attempt": attempt,
                },
            }),
        })
    _write_jsonl(source / "http_attempts.jsonl", attempts)
    _write_jsonl(source / "call-intents.jsonl", intents)
    _write_jsonl(
        source / "audit" / "runtime/training-captures" / "decisions_generic.jsonl",
        captures,
    )
    return source, manifest_path


def test_inventory_is_exact_bijective_ordered_and_records_missing_five(tmp_path):
    source, manifest = _fixture(tmp_path)
    inventory = build_authentic_inventory(
        source_cell=source, frozen_manifest=manifest,
        repository_root=tmp_path,
    )
    assert inventory["attempt_count"] == 13
    assert inventory["authentic_row_order"] == [
        "generic-row-0", "generic-row-1", "generic-row-4",
    ]
    assert len(inventory["missing_rows"]) == 5
    assert all(
        row["status"] == "missing_by_construction"
        for row in inventory["missing_rows"]
    )
    assert inventory["attempts"][2]["parent_attempt_identity_sha256"] == (
        inventory["attempts"][1]["attempt_identity_sha256"]
    )
    assert len({row["request_sha256"] for row in inventory["attempts"]}) == 13
    assert len({row["response_sha256"] for row in inventory["attempts"]}) == 13


def test_two_clean_replays_are_byte_and_semantically_identical_and_offline(tmp_path):
    source, manifest = _fixture(tmp_path)
    evidence = tmp_path / "evidence"
    comparison = run_replay_twice(
        source_cell=source, frozen_manifest=manifest,
        repository_root=tmp_path, evidence_root=evidence,
    )
    assert comparison == {
        "status": "accepted",
        "run_1_semantic_sha256": comparison["run_1_semantic_sha256"],
        "run_2_semantic_sha256": comparison["run_2_semantic_sha256"],
        "semantic_equal": True,
        "byte_hash_maps_equal": True,
    }
    assert comparison["run_1_semantic_sha256"] == (
        comparison["run_2_semantic_sha256"]
    )
    for run in ("run-1", "run-2"):
        proof = json.loads((evidence / run / "zero-network-proof.json").read_text())
        assert proof["socket_guard_active"] is True
        assert proof["provider_call_count"] == proof["connect_call_count"] == 0
        assert proof["network_attempt_ledger"] == []
        verification = json.loads((evidence / run / "verification.json").read_text())
        assert verification["attempt_count"] == 13
        assert verification["journal_started"] == 13
        assert verification["journal_terminal"] == 13
        assert verification["checkpoint_rows"] == 3
        assert verification["published_count"] == 0


def test_replay_rejects_nonempty_output_and_source_byte_tamper(tmp_path):
    source, manifest = _fixture(tmp_path)
    inventory = build_authentic_inventory(
        source_cell=source, frozen_manifest=manifest,
        repository_root=tmp_path,
    )
    output = tmp_path / "not-empty"
    output.mkdir()
    (output / "existing").write_text("authority")
    with pytest.raises(ValueError, match="clean empty"):
        replay_once(
            inventory=inventory, source_cell=source,
            repository_root=tmp_path, output_dir=output,
        )
    terminal = [
        row for row in _read_rows(source / "http_attempts.jsonl")
        if row["event"] == "http_attempt_terminal"
    ][0]
    raw_path = tmp_path / terminal["response_raw_ref"]["path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash differs"):
        replay_once(
            inventory=inventory, source_cell=source,
            repository_root=tmp_path, output_dir=tmp_path / "tampered",
        )


def _read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]
