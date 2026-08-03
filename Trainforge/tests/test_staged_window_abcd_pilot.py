import hashlib
import json

import httpx
import pytest

from Trainforge.decision_audit_verifier import verify_decision_audit
from Trainforge.generators.http_attempt_ledger import request_sha256
from Trainforge.generators.synthesis_window_contract import objective_card
from Trainforge.scripts.harness.staged_window_abcd_pilot import (
    COHORT_SIZE,
    VARIANTS,
    _CRITICAL_CELL_ERROR_CODES,
    _digest,
    _micro_stage_coverage_complete,
    _TransportProbe,
    _provider,
    _sanitize_error_details,
    _atomic_write_bytes,
    apply_runtime_focus,
    apply_micro_transport_outcomes,
    build_pilot_manifest,
    build_pilot_rows,
    build_benchmark_rows,
    build_preflight_artifact,
    canonicalize_micro_publication_rows,
    classify_cell_outcome,
    counterbalanced_cell_rows,
    execute_pilot,
    frozen_d_c1_ordered_identity_sha256,
    load_frozen_d_c1_manifest,
    load_objectives,
    migrate_matrix_summary,
    negotiate_dialect,
    planned_benchmark_cells,
    qualification_route_binding,
    remaining_cell_budget,
    reconcile_http_audit,
    publish_cell_success,
    read_cell_publication_state,
    select_cohort,
    summarize,
    LEGACY_SYNTHESIS_CONTRACT,
    MICRO_SYNTHESIS_CONTRACT,
    resolve_synthesis_contract,
    synthesis_contract_identity,
    verify_micro_evidence_plane_rows,
    verify_micro_journals,
)


def _objectives():
    return {
        "co-01": {
            "id": "co-01",
            "statement": "Analyze the supplied evidence.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "abcd": {
                "behavior": {"action_object": "supplied evidence"},
                "condition": "given a case",
                "degree": "without unsupported claims",
            },
        }
    }


def _chunks():
    chunks = []
    for index in range(36):
        misconception = (
            f"evidence pattern {index} proves every possible conclusion"
        )
        correction = (
            f"evidence pattern {index} supports only conclusions warranted "
            "by the supplied observations"
        )
        mechanism = (
            "The correction is required because unsupported conclusions add "
            "claims beyond the observed evidence."
        )
        has_misconception = index % 2 == 0
        text = (
            f"CO-01: Analyze the supplied evidence. "
            f"Evidence sentence number {index} supports the objective. "
        )
        if has_misconception:
            text += f"{misconception}. {correction}. {mechanism} "
        text += "additional context " * (index * 15)
        chunk = {
            "id": f"chunk-{index:02d}",
            "text": text,
            "html": (
                f'<section data-cf-block-id="b-{index}" '
                f'data-cf-objective-id="co-01">{text}</section>'
            ),
            "learning_outcome_refs": ["co-01"],
            "bloom_level": "analyze",
        }
        if has_misconception:
            chunk["misconceptions"] = [{
                "misconception": misconception,
                "correction": correction,
                "mechanism_evidence": mechanism,
                "source_block_id": f"b-{index}",
            }]
        chunks.append(chunk)
    return chunks


def test_generic_fixture_has_current_preference_eligibility_capacity():
    from Trainforge.synthesis_eligibility import pair_eligibility

    focused = [
        apply_runtime_focus(chunk, _objectives()) for chunk in _chunks()
    ]
    eligible = [
        chunk for chunk in focused
        if pair_eligibility(chunk, kind="preference").eligible
    ]
    assert len(eligible) >= 14
    assert all(chunk.get("misconceptions") for chunk in eligible)


def test_objective_loader_accepts_raw_workflow_learning_outcomes(tmp_path):
    records = [
        {
            "id": " CO-01 ",
            "statement": "Analyze the supplied evidence.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "metadata": {"source": "generic-workflow", "ordinal": 1},
        },
        {
            "objective_id": "Co-02",
            "statement": "Evaluate the supported conclusion.",
            "bloom_level": "evaluate",
            "bloom_verb": "evaluate",
            "metadata": {"source": "generic-workflow", "ordinal": 2},
        },
    ]
    path = tmp_path / "raw-workflow-objectives.json"
    path.write_text(
        json.dumps({"learning_outcomes": records}), encoding="utf-8",
    )

    loaded = load_objectives(path)

    assert list(loaded) == ["co-01", "co-02"]
    assert loaded["co-01"] == records[0]
    assert loaded["co-02"] == records[1]
    focused = apply_runtime_focus(
        {
            "id": "generic-chunk",
            "text": "Analyze the supplied evidence.",
            "learning_outcome_refs": [" CO-01 "],
        },
        loaded,
    )
    assert objective_card(focused["synthesis_focus_objective"]) == (
        objective_card({
            "id": "co-01",
            "statement": "Analyze the supplied evidence.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
        })
    )


def test_objective_loader_legacy_shapes_remain_equivalent(tmp_path):
    objective = _objectives()["co-01"]
    expected = {"co-01": objective}
    shapes = {
        "list": [objective],
        "terminal": {
            "terminal_outcomes": [objective],
            "component_objectives": [],
        },
        "objectives": {"objectives": [objective]},
        "map": expected,
    }
    encoded = None
    for name, value in shapes.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        loaded = load_objectives(path)
        assert loaded == expected
        current = json.dumps(
            loaded, sort_keys=True, separators=(",", ":"),
        ).encode()
        encoded = current if encoded is None else encoded
        assert current == encoded


@pytest.mark.parametrize(
    "value",
    (
        {"learning_outcomes": "not-a-list"},
        {"learning_outcomes": ["not-an-object"]},
        {"learning_outcomes": [{"statement": "No identifier."}]},
        {
            "learning_outcomes": [
                {"id": "CO-01", "statement": "First."},
                {"id": " co-01 ", "statement": "Conflicting."},
            ],
        },
        {
            "CO-01": {"id": "co-01", "statement": "First."},
            " co-01 ": {"id": "co-01", "statement": "Conflicting."},
        },
    ),
)
def test_objective_loader_fails_loud_on_malformed_or_conflicting_input(
    tmp_path, value,
):
    path = tmp_path / "malformed-objectives.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_objectives(path)


def test_cohort_is_deterministic_stratified_and_fixed_at_28():
    forward = select_cohort(_chunks())
    reverse = select_cohort(reversed(_chunks()))
    assert forward == reverse
    assert len(forward) == COHORT_SIZE
    assert len({row["id"] for row in forward}) == COHORT_SIZE
    assert len({
        (
            "misconception" in row["text"].lower(),
            len(row["text"].split()) < 250,
        )
        for row in forward
    }) > 1


def test_manifest_is_counterbalanced_and_isolates_two_by_two_ablation():
    rows = build_pilot_rows(_chunks(), objectives=_objectives(), repetitions=2)
    assert len(rows) == COHORT_SIZE * len(VARIANTS) * 2
    for repetition in range(2):
        subset = [row for row in rows if row["repetition"] == repetition]
        for index in range(COHORT_SIZE):
            unit = [row for row in subset if row["cohort_index"] == index]
            assert {row["variant"] for row in unit} == set(VARIANTS)
            assert len({row["order_index"] for row in unit}) == 4
    first_variants = [
        row["variant"] for row in rows
        if row["repetition"] == 0 and row["order_index"] == 0
    ]
    assert max(first_variants.count(item) for item in VARIANTS) == 7
    by_variant = {
        row["variant"]: row["_chunk"]
        for row in rows if row["cohort_index"] == 0 and row["repetition"] == 0
    }
    assert "html" not in by_variant[VARIANTS[0]]
    assert "bloom_verb" not in by_variant[VARIANTS[0]]["synthesis_focus_objective"]
    assert "html" in by_variant[VARIANTS[1]]
    assert "html" not in by_variant[VARIANTS[2]]
    assert "abcd" in by_variant[VARIANTS[3]]["synthesis_focus_objective"]


def test_runtime_focus_uses_canonical_objective():
    chunk = apply_runtime_focus(_chunks()[0], _objectives())
    assert chunk["synthesis_focus_objective"]["abcd"]["degree"]
    assert chunk["learning_outcome_refs"] == ["co-01"]
    assert chunk["bloom_level"] == "analyze"


def test_manifest_uses_exact_pair_eligibility_and_reports_exclusions():
    rows, audit = build_pilot_manifest(
        _chunks(), objectives=_objectives(),
    )
    assert audit["selected_unique_chunks"] == 28
    assert audit["selected_by_kind"] == {
        "instruction": 14, "preference": 14,
    }
    assert (
        audit["exclusions"][
            "preference:preference_misconception_candidate_missing"
        ] > 0
    )
    assert len({row["chunk_id"] for row in rows}) == 28


def test_benchmark_cohort_is_eight_d_only_and_balanced_by_kind():
    rows, audit = build_benchmark_rows(_chunks(), objectives=_objectives())
    assert len(rows) == 8
    assert {row["variant"] for row in rows} == {"D_production_contract"}
    assert audit["benchmark_by_kind"] == {
        "instruction": 4, "preference": 4,
    }
    assert len({row["chunk_id"] for row in rows}) == 8
    for kind in ("instruction", "preference"):
        assert len({
            tuple(row["stratum"]) for row in rows if row["kind"] == kind
        }) >= 2


def _frozen_d_c1_fixture(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            f"{json.dumps(row, sort_keys=True, separators=(',', ':'))}\n"
            for row in _chunks()
        ),
        encoding="utf-8",
    )
    objectives_path = tmp_path / "objectives.json"
    objectives_path.write_text(
        json.dumps(_objectives(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    rows, _ = build_benchmark_rows(_chunks(), objectives=_objectives())
    frozen_rows = [
        {key: value for key, value in row.items() if key != "_chunk"}
        for row in rows
    ]
    frozen_path = tmp_path / "frozen.jsonl"
    frozen_path.write_text(
        "".join(
            f"{json.dumps(row, sort_keys=True, separators=(',', ':'))}\n"
            for row in frozen_rows
        ),
        encoding="utf-8",
    )
    return chunks_path, objectives_path, frozen_path, frozen_rows


def _frozen_eligibility_fixture(tmp_path, chunks):
    _, audit = build_benchmark_rows(chunks, objectives=_objectives())
    path = tmp_path / "frozen-eligibility.json"
    path.write_text(
        f"{json.dumps(audit, indent=2, sort_keys=True)}\n", encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_d_c1_loader_rehydrates_exact_rows_in_authoritative_order(tmp_path):
    chunks_path, objectives_path, frozen_path, frozen_rows = (
        _frozen_d_c1_fixture(tmp_path)
    )
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    ordered_sha = frozen_d_c1_ordered_identity_sha256(frozen_rows)
    eligibility_path, eligibility_sha = _frozen_eligibility_fixture(
        tmp_path, _chunks(),
    )
    rows, audit = load_frozen_d_c1_manifest(
        frozen_path,
        expected_sha256=frozen_sha,
        expected_ordered_identity_sha256=ordered_sha,
        chunks=_chunks(),
        objectives=_objectives(),
        chunks_path=chunks_path,
        objectives_path=objectives_path,
        frozen_eligibility_path=eligibility_path,
        expected_eligibility_sha256=eligibility_sha,
    )
    assert [
        {key: value for key, value in row.items() if key != "_chunk"}
        for row in rows
    ] == frozen_rows
    assert [row["chunk_id"] for row in rows] == [
        row["chunk_id"] for row in frozen_rows
    ]
    assert audit["frozen_manifest_sha256"] == frozen_sha
    assert audit["ordered_identity_sha256"] == ordered_sha
    assert audit["rehydrated_row_sha256"] == [
        row["chunk_sha256"] for row in frozen_rows
    ]


def test_frozen_loader_audit_distinguishes_full_objective_record_changes(tmp_path):
    chunks_path, objectives_path, frozen_path, frozen_rows = (
        _frozen_d_c1_fixture(tmp_path)
    )
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    ordered_sha = frozen_d_c1_ordered_identity_sha256(frozen_rows)
    eligibility_path, eligibility_sha = _frozen_eligibility_fixture(
        tmp_path, _chunks(),
    )
    baseline_rows, baseline_audit = load_frozen_d_c1_manifest(
        frozen_path,
        expected_sha256=frozen_sha,
        expected_ordered_identity_sha256=ordered_sha,
        chunks=_chunks(),
        objectives=load_objectives(objectives_path),
        chunks_path=chunks_path,
        objectives_path=objectives_path,
        frozen_eligibility_path=eligibility_path,
        expected_eligibility_sha256=eligibility_sha,
    )
    changed = _objectives()
    changed["co-01"] = {
        **changed["co-01"],
        "metadata": {"source_revision": "different-full-record"},
    }
    changed_path = tmp_path / "changed-objectives.json"
    changed_path.write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert baseline_rows
    assert baseline_audit["objectives_source_sha256"] == hashlib.sha256(
        objectives_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="canonical payload hash mismatch"):
        load_frozen_d_c1_manifest(
            frozen_path,
            expected_sha256=frozen_sha,
            expected_ordered_identity_sha256=ordered_sha,
            chunks=_chunks(),
            objectives=load_objectives(changed_path),
            chunks_path=chunks_path,
            objectives_path=changed_path,
            frozen_eligibility_path=eligibility_path,
            expected_eligibility_sha256=eligibility_sha,
        )


@pytest.mark.parametrize(
    "attack",
    (
        "wrong_expected_hash",
        "reordered",
        "duplicate_identity",
        "mutated_payload_hash",
        "mutated_objective",
        "wrong_variant",
        "wrong_repetition",
        "missing_source",
        "duplicate_source",
        "noncanonical_bytes",
        "extra_key",
        "missing_key",
    ),
)
def test_frozen_d_c1_loader_fails_closed_on_adversarial_input(tmp_path, attack):
    chunks_path, objectives_path, frozen_path, frozen_rows = (
        _frozen_d_c1_fixture(tmp_path)
    )
    chunks = _chunks()
    rows = [dict(row) for row in frozen_rows]
    expected = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    expected_ordered = frozen_d_c1_ordered_identity_sha256(frozen_rows)
    eligibility_path, eligibility_sha = _frozen_eligibility_fixture(
        tmp_path, chunks,
    )
    if attack == "wrong_expected_hash":
        expected = "0" * 64
    elif attack == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
        frozen_path.write_text(
            "".join(
                f"{json.dumps(row, sort_keys=True, separators=(',', ':'))}\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        expected = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    elif attack == "duplicate_identity":
        rows[1] = dict(rows[0])
    elif attack == "mutated_payload_hash":
        rows[0]["chunk_sha256"] = "0" * 64
    elif attack == "mutated_objective":
        rows[0]["focus_objective"] = {
            **rows[0]["focus_objective"], "statement": "Different statement."
        }
    elif attack == "wrong_variant":
        rows[0]["variant"] = "A_raw_minimal_objective"
    elif attack == "wrong_repetition":
        rows[0]["repetition"] = 1
    elif attack == "missing_source":
        chunks = chunks[1:]
    elif attack == "duplicate_source":
        chunks = [*chunks, dict(chunks[0])]
    elif attack == "noncanonical_bytes":
        frozen_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        expected = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    elif attack == "extra_key":
        rows[0]["recomputed_sha256"] = "0" * 64
    elif attack == "missing_key":
        del rows[0]["pilot_version"]
    if attack not in {
        "wrong_expected_hash", "reordered", "missing_source",
        "duplicate_source", "noncanonical_bytes",
    }:
        frozen_path.write_text(
            "".join(
                f"{json.dumps(row, sort_keys=True, separators=(',', ':'))}\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        expected = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_frozen_d_c1_manifest(
            frozen_path,
            expected_sha256=expected,
            expected_ordered_identity_sha256=expected_ordered,
            chunks=chunks,
            objectives=_objectives(),
            chunks_path=chunks_path,
            objectives_path=objectives_path,
            frozen_eligibility_path=eligibility_path,
            expected_eligibility_sha256=eligibility_sha,
        )


@pytest.mark.parametrize(
    "attack",
    (
        "wrong_sha", "mutated_count", "reordered_bytes", "extra_field",
        "null_field", "unbound_source_count", "wrong_variant",
    ),
)
def test_frozen_d_c1_eligibility_authority_fails_closed(tmp_path, attack):
    chunks_path, objectives_path, frozen_path, frozen_rows = (
        _frozen_d_c1_fixture(tmp_path)
    )
    eligibility_path, eligibility_sha = _frozen_eligibility_fixture(
        tmp_path, _chunks(),
    )
    record = json.loads(eligibility_path.read_text(encoding="utf-8"))
    if attack == "wrong_sha":
        eligibility_sha = "0" * 64
    elif attack == "mutated_count":
        record["benchmark_by_kind"]["instruction"] = 3
    elif attack == "extra_field":
        record["course_slug"] = "forbidden"
    elif attack == "null_field":
        record["selected_unique_chunks"] = None
    elif attack == "unbound_source_count":
        record["input_chunks"] += 1
    elif attack == "wrong_variant":
        record["benchmark_variant"] = "A_raw_minimal_objective"
    if attack not in {"wrong_sha", "reordered_bytes"}:
        eligibility_path.write_text(
            f"{json.dumps(record, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        eligibility_sha = hashlib.sha256(
            eligibility_path.read_bytes()
        ).hexdigest()
    elif attack == "reordered_bytes":
        eligibility_path.write_text(json.dumps(record), encoding="utf-8")
        eligibility_sha = hashlib.sha256(
            eligibility_path.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError):
        load_frozen_d_c1_manifest(
            frozen_path,
            expected_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            expected_ordered_identity_sha256=(
                frozen_d_c1_ordered_identity_sha256(frozen_rows)
            ),
            chunks=_chunks(), objectives=_objectives(),
            chunks_path=chunks_path, objectives_path=objectives_path,
            frozen_eligibility_path=eligibility_path,
            expected_eligibility_sha256=eligibility_sha,
        )


class _FakeProvider:
    def __init__(self):
        self._pilot_calls = []

    def paraphrase_instruction(self, draft, chunk):
        self._pilot_calls.extend([
            {"prompt_tokens": 100, "completion_tokens": 20, "requests": 1},
            {"prompt_tokens": 30, "completion_tokens": 10, "requests": 1},
        ])
        return {
            **draft,
            "prompt": "Analyze the evidence and explain the supported conclusion clearly.",
            "completion": "The evidence supports this conclusion for the reasons identified in the source.",
        }

    def paraphrase_preference(self, draft, chunk):
        self._pilot_calls.extend([
            {"prompt_tokens": 100, "completion_tokens": 20, "requests": 1},
            {"prompt_tokens": 30, "completion_tokens": 10, "requests": 1},
            {"prompt_tokens": 40, "completion_tokens": 10, "requests": 1},
        ])
        return {
            **draft,
            "prompt": "Analyze the evidence and explain the supported conclusion clearly.",
            "chosen": "The evidence supports this conclusion for the reasons identified in the source.",
            "rejected": "The opposite is true because the evidence can simply be ignored.",
        }


def test_execution_records_required_efficiency_and_quality_metrics():
    rows = build_pilot_rows(_chunks(), objectives=_objectives())[:4]

    def scorer(pair, chunk, kind, objectives):
        return {
            "accepted": True,
            "semantic_coverage": 1.0,
            "nli": [{"outcome": "entailed"}],
            "bloom_objective_delivery": [{"status": "delivered"}],
            "claim_support": {"status": "validated", "reason": None},
        }

    results, summary = execute_pilot(
        rows, _FakeProvider(), objectives=_objectives(), scorer=scorer,
    )
    assert len(results) == 4
    assert all(row["stage_validity"] for row in results)
    assert all(row["accepted"] for row in results)
    assert all(row["total_tokens"] > 0 for row in results)
    assert all(row["tokens_per_second"] > 0 for row in results)
    for variant in VARIANTS:
        variant_rows = [row for row in results if row["variant"] == variant]
        assert summary["variants"][variant]["calls_per_accepted_pair"] == (
            sum(row["calls"] for row in variant_rows) / len(variant_rows)
        )


def test_summary_surfaces_errors_and_truncation():
    rows = [
        {
            "variant": VARIANTS[0], "accepted": False,
            "stage_validity": False, "calls": 1, "total_tokens": 20,
            "latency_seconds": 2.0, "truncated": True,
            "error_code": "staged_output_truncated",
        }
    ]
    summary = summarize(rows)["variants"][VARIANTS[0]]
    assert summary["truncations"] == 1
    assert summary["errors"] == {"staged_output_truncated": 1}
    assert summary["calls_per_accepted_pair"] is None


def test_critical_failure_stops_admission_and_drains_active_workers():
    import threading
    import time

    rows = build_pilot_rows(_chunks(), objectives=_objectives())[:8]

    class _CriticalProvider:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def paraphrase_instruction(self, draft, chunk):
            return self._run()

        def paraphrase_preference(self, draft, chunk):
            return self._run()

        def _run(self):
            with self.lock:
                self.calls += 1
                call = self.calls
            if call == 1:
                error = RuntimeError("transport exhausted")
                error.code = "max_retries_exceeded"
                raise error
            time.sleep(0.05)
            return {
                "prompt": "Analyze the supplied evidence.",
                "completion": "The evidence supports the objective.",
                "chosen": "The evidence supports the objective.",
                "rejected": "The opposite is wrong because evidence matters.",
            }

    provider = _CriticalProvider()
    started = time.monotonic()
    try:
        execute_pilot(
            rows, provider, objectives=_objectives(),
            scorer=lambda *args: {"accepted": True},
            max_concurrent=2, request_timeout_seconds=0.1,
        )
    except RuntimeError as exc:
        assert "fail-closed" in str(exc)
    else:
        raise AssertionError("critical transport failure must stop the cell")
    assert 1 <= provider.calls <= 2
    assert time.monotonic() - started < 0.5


def test_preflight_binds_manifest_model_and_contract_hashes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"unit":1}\n')
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text('{"eligible":8}\n')
    artifact = build_preflight_artifact(
        manifest_path=manifest,
        eligibility_path=eligibility,
        base_url="http://localhost:8123/v1",
        model_snapshot={"data": [{"id": "served-model"}]},
        run_id="pilot-fixed",
        server_batch=28,
        client_concurrency=16,
        timeout_seconds=240,
        transport_attempts=1,
        initial_backoff_seconds=1,
        max_tokens=800,
        temperature=0.4,
        dialect={
            "outcome": "accepted",
            "response_dialect": "openai_json_schema_strict",
        },
    )
    assert artifact["server_batch"] == 28
    assert artifact["client_concurrency"] == 16
    assert artifact["retry_policy"] == (
        "no_reissue_without_confirmed_abort_and_drain"
    )
    assert len(artifact["manifest_sha256"]) == 64
    assert artifact["dialect_preflight"]["response_dialect"] == (
        "openai_json_schema_strict"
    )
    assert len(artifact["run_contract_sha256"]) == 64


def test_preflight_binds_frozen_selection_sources_and_rehydrated_rows(tmp_path):
    chunks_path, objectives_path, frozen_path, _ = _frozen_d_c1_fixture(
        tmp_path
    )
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    ordered_sha = frozen_d_c1_ordered_identity_sha256(
        [
            json.loads(line)
            for line in frozen_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    eligibility, eligibility_sha = _frozen_eligibility_fixture(
        tmp_path, _chunks(),
    )
    _, audit = load_frozen_d_c1_manifest(
        frozen_path,
        expected_sha256=frozen_sha,
        expected_ordered_identity_sha256=ordered_sha,
        chunks=_chunks(),
        objectives=_objectives(),
        chunks_path=chunks_path,
        objectives_path=objectives_path,
        frozen_eligibility_path=eligibility,
        expected_eligibility_sha256=eligibility_sha,
    )
    loader_audit = tmp_path / "frozen_loader_audit.json"
    loader_audit.write_text(json.dumps(audit), encoding="utf-8")
    artifact = build_preflight_artifact(
        manifest_path=frozen_path,
        eligibility_path=eligibility,
        base_url="http://localhost:8123/v1",
        model_snapshot={"data": [{"id": "served-model"}]},
        run_id="pilot-fixed", server_batch=28, client_concurrency=1,
        timeout_seconds=240, transport_attempts=1,
        initial_backoff_seconds=1, max_tokens=800, temperature=0.4,
        dialect={"outcome": "accepted"},
        frozen_loader_audit_path=loader_audit,
    )
    assert artifact["frozen_selection"] == {
        key: audit[key] for key in (
            "selection_mode", "schema_version", "loader_code_sha256",
            "frozen_manifest_sha256", "frozen_eligibility_schema",
            "frozen_eligibility_sha256", "ordered_identity_schema",
            "ordered_identity_sha256", "ordered_identity_projection",
            "chunks_source_sha256",
            "objectives_source_sha256", "rehydrated_row_sha256",
        )
    }
    assert artifact["run_contract_sha256"] == _digest({
        key: value for key, value in artifact.items()
        if key not in {"created_utc", "run_contract_sha256"}
    })


def test_micro_contract_is_explicit_authoritative_and_aliases_fail_closed():
    assert resolve_synthesis_contract("legacy") == LEGACY_SYNTHESIS_CONTRACT
    assert (
        resolve_synthesis_contract(MICRO_SYNTHESIS_CONTRACT)
        == MICRO_SYNTHESIS_CONTRACT
    )
    for value in ("micro", "v1", "staged-v4", "", "ED4ALL"):
        with pytest.raises(ValueError, match="synthesis contract"):
            resolve_synthesis_contract(value)


def test_micro_provider_requires_explicit_seed_and_binds_release_identity():
    with pytest.raises(ValueError, match="explicit --synthesis-seed"):
        _provider(synthesis_contract=MICRO_SYNTHESIS_CONTRACT)
    provider = _provider(
        synthesis_contract=MICRO_SYNTHESIS_CONTRACT,
        synthesis_seed=0,
    )
    assert provider._release_identity() == {
        "release_contract_version": "training-synthesis-release.v1.2.0",
        "synthesis_contract_version": MICRO_SYNTHESIS_CONTRACT,
        "synthesis_contract_sha256": (
            provider._pilot_synthesis_contract["fingerprint"]
        ),
        "synthesis_seed": 0,
        "projections": {
            "instruction": "ed4all-sft-chat.v2",
            "preference": "ed4all-dpo-preference.v2",
        },
        "provider": "local",
        "model": provider._model,
        "completion_cap": 600,
    }


def test_micro_preflight_binds_every_component_and_core_code_hash(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"unit":1}\n')
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text('{"eligible":8}\n')
    artifact = build_preflight_artifact(
        manifest_path=manifest,
        eligibility_path=eligibility,
        base_url="http://localhost:8123/v1",
        model_snapshot={"data": [{"id": "served-model"}]},
        run_id="pilot-fixed", server_batch=28, client_concurrency=16,
        timeout_seconds=240, transport_attempts=1,
        initial_backoff_seconds=1, max_tokens=800, temperature=0.4,
        dialect={"outcome": "accepted"},
        synthesis_contract=MICRO_SYNTHESIS_CONTRACT,
        micro_journal_root="micro-journals",
    )
    identity = artifact["synthesis_contract"]
    assert identity == synthesis_contract_identity(MICRO_SYNTHESIS_CONTRACT)
    assert identity["version"] == MICRO_SYNTHESIS_CONTRACT
    assert len(identity["fingerprint"]) == 64
    assert identity["component_hashes"]["stages"] == list("ABCDEFQ")
    assert identity["component_hashes"]["stage_token_budget"] == {
        "version": "micro-stage-max-tokens.v1",
        "max_tokens": {
            "A": 2048, "B": 1536, "D": 1536, "E": 1280, "F": 1024,
        },
        "deterministic_stages": ["A", "C"],
    }
    assert (
        "Trainforge/generators/staged_synthesis_micro.py"
        in artifact["code_hashes"]
    )
    assert artifact["micro_journal_root"] == "micro-journals"
    assert artifact["run_contract_sha256"] == _digest({
        key: value for key, value in artifact.items()
        if key not in {"created_utc", "run_contract_sha256"}
    })


@pytest.mark.parametrize("root", [None, "", "../foreign", "/foreign"])
def test_micro_preflight_requires_safe_run_local_journal_root(tmp_path, root):
    manifest = tmp_path / "manifest.jsonl"
    eligibility = tmp_path / "eligibility.json"
    manifest.write_text("")
    eligibility.write_text("{}")
    with pytest.raises(ValueError, match="journal root"):
        build_preflight_artifact(
            manifest_path=manifest,
            eligibility_path=eligibility,
            base_url="http://localhost:8123/v1",
            model_snapshot={"data": [{"id": "served-model"}]},
            run_id="pilot-fixed", server_batch=28, client_concurrency=1,
            timeout_seconds=240, transport_attempts=1,
            initial_backoff_seconds=1, max_tokens=800, temperature=0,
            dialect={"outcome": "accepted"},
            synthesis_contract=MICRO_SYNTHESIS_CONTRACT,
            micro_journal_root=root,
        )


def test_legacy_preflight_remains_default_and_excludes_micro_core(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    eligibility = tmp_path / "eligibility.json"
    manifest.write_text("")
    eligibility.write_text("{}")
    artifact = build_preflight_artifact(
        manifest_path=manifest, eligibility_path=eligibility,
        base_url="http://localhost:8123/v1",
        model_snapshot={"data": [{"id": "served-model"}]},
        run_id="legacy", server_batch=28, client_concurrency=1,
        timeout_seconds=240, transport_attempts=1,
        initial_backoff_seconds=1, max_tokens=800, temperature=0,
        dialect={"outcome": "accepted"},
    )
    assert artifact["synthesis_contract"]["version"] == "legacy"
    assert (
        "Trainforge/generators/staged_synthesis_micro.py"
        not in artifact["code_hashes"]
    )


def _micro_evidence_rows():
    hashes = ("1" * 64, "2" * 64)
    stages = (
        "staged_synthesis:micro_B_claim_0_attempt_1",
        "staged_synthesis:micro_B_claim_0_attempt_2",
    )
    intents = [
        {
            "unit": "chunk:instruction:contract:r0",
            "kind": "initial",
            "stage": stage,
            "logical_attempt": 1,
            "request_sha256": hashes[index - 1],
        }
        for index, stage in enumerate(stages, 1)
    ]
    decisions = [
        {
            "intent_unit": row["unit"],
            "intent_stage": row["stage"],
            "intent_logical_attempt": row["logical_attempt"],
            "intent_request_sha256": row["request_sha256"],
        }
        for row in intents
    ]
    http = []
    for index, row in enumerate(intents):
        common = {
            "unit": row["unit"], "stage": row["stage"],
            "attempt": row["logical_attempt"],
            "request_sha256": row["request_sha256"],
        }
        http.extend((
            {
                **common, "event": "http_attempt_started",
                "monotonic_seconds": float(index),
            },
            {
                **common, "event": "http_attempt_terminal",
                "monotonic_seconds": float(index) + 0.5,
                "http_status": 200,
                "usage": {
                    "prompt_tokens": 4, "completion_tokens": 6,
                    "total_tokens": 10,
                },
            },
        ))
    return intents, decisions, http


def test_micro_evidence_plane_separates_calls_repairs_transport_and_tokens():
    intents, decisions, http = _micro_evidence_rows()
    report = verify_micro_evidence_plane_rows(
        intent_rows=intents, decision_contexts=decisions, http_rows=http,
    )
    assert report["logical_calls"] == {"B": 2}
    assert report["repairs"] == {"B": 1}
    assert report["transport_attempts"] == {"B": 2}
    assert report["stages"]["B"]["succeeded"] == 2
    assert report["stages"]["B"]["client_usage"]["total_tokens"] == 20


def test_micro_evidence_plane_retains_timeout_without_fabricated_usage():
    intents, decisions, http = _micro_evidence_rows()
    terminal = http[-1]
    terminal.update({
        "http_status": None, "exception_class": "ReadTimeout", "usage": {},
    })
    report = verify_micro_evidence_plane_rows(
        intent_rows=intents, decision_contexts=decisions, http_rows=http,
    )
    stage = report["stages"]["B"]
    assert stage["logical_attempts"] == stage["transport_terminal"] == 2
    assert stage["succeeded"] == 1
    assert stage["failed"]["timeout"] == 1
    assert stage["unrecovered"] == 1
    assert stage["client_usage"]["available"] == 1
    assert stage["client_usage"]["unavailable"] == 1
    summary = apply_micro_transport_outcomes(
        {"microstage_metrics": {"B": {"completed_units": 1}}},
        report,
    )
    assert summary["microstage_metrics"]["B"]["calls"] == 2
    assert summary["microstage_metrics"]["B"][
        "client_sum_tokens_per_second"
    ] is None


def test_micro_evidence_plane_tracks_recovered_failure_and_partial_usage():
    intents, decisions, http = _micro_evidence_rows()
    first_terminal = http[1]
    first_terminal.update({
        "http_status": None, "exception_class": "ConnectionError",
        "usage": {"prompt_tokens": 4},
    })
    report = verify_micro_evidence_plane_rows(
        intent_rows=intents, decision_contexts=decisions, http_rows=http,
    )
    stage = report["stages"]["B"]
    assert stage["failed"]["transport"] == 1
    assert stage["recovered"] == 1
    assert stage["unrecovered"] == 0
    assert stage["client_usage"]["partial"] == 1


@pytest.mark.parametrize(
    ("target", "mutation"),
    (
        ("decision", "missing"),
        ("decision", "extra"),
        ("decision", "wrong_attempt"),
        ("decision", "cross_row"),
        ("http", "missing"),
        ("http", "extra"),
        ("http", "wrong_attempt"),
        ("http", "cross_row"),
    ),
)
def test_micro_evidence_plane_fails_closed_on_identity_tamper(target, mutation):
    intents, decisions, http = _micro_evidence_rows()
    rows = decisions if target == "decision" else http
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(dict(rows[0]))
    elif mutation == "wrong_attempt":
        key = "intent_logical_attempt" if target == "decision" else "attempt"
        rows[0][key] = 2
    else:
        key = "intent_unit" if target == "decision" else "unit"
        rows[0][key] = "foreign:instruction:contract:r0"
    with pytest.raises(ValueError, match="bijective|duplicated"):
        verify_micro_evidence_plane_rows(
            intent_rows=intents, decision_contexts=decisions, http_rows=http,
        )


def test_micro_evidence_plane_rejects_reordered_repair_sequence():
    intents, decisions, http = _micro_evidence_rows()
    with pytest.raises(ValueError, match="sequence"):
        verify_micro_evidence_plane_rows(
            intent_rows=list(reversed(intents)),
            decision_contexts=decisions,
            http_rows=http,
        )


def test_micro_evidence_plane_rejects_deterministic_transport():
    intents, decisions, http = _micro_evidence_rows()
    intents[0]["stage"] = "staged_synthesis:micro_A_task_design"
    decisions[0]["intent_stage"] = intents[0]["stage"]
    http[0]["stage"] = intents[0]["stage"]
    with pytest.raises(ValueError, match="stage identity|deterministic"):
        verify_micro_evidence_plane_rows(
            intent_rows=intents, decision_contexts=decisions, http_rows=http,
        )


def test_micro_journal_verification_binds_resume_and_telemetry(tmp_path):
    from Trainforge.generators.staged_synthesis_micro import (
        MicroResumeStore, micro_contract_fingerprint,
    )

    chunk_sha = "b" * 64
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({
            "chunk_id": "generic-chunk", "chunk_sha256": chunk_sha,
            "kind": "instruction", "variant": "generic-contract",
            "repetition": 0,
        }) + "\n"
    )
    preflight = {
        "synthesis_contract": synthesis_contract_identity(
            MICRO_SYNTHESIS_CONTRACT
        ),
        "run_contract_sha256": "c" * 64,
        "telemetry_identity": {"expected_model": "served-model"},
        "pilot_run_id": "pilot-c1",
        "cell_id": "c1",
        "manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.jsonl").read_bytes()
        ).hexdigest(),
        "eligibility_sha256": "d" * 64,
        "micro_journal_root": "micro-journals/micro_synthesis_state",
    }
    identity = {
        "contract": micro_contract_fingerprint(),
        "kind": "instruction",
        "chunk_id": "generic-chunk",
        "chunk_sha256": chunk_sha,
        "draft_identity": {
            "chunk_id": "generic-chunk", "chunk_sha256": chunk_sha,
            "kind": "instruction", "variant": "generic-contract",
            "repetition": 0,
            "draft": {
                "provider": "local", "source_chunk_id": "generic-chunk",
            },
        },
        "model": "served-model",
        "pilot_run_id": "pilot-c1",
        "cell_id": "c1",
        "manifest_sha256": preflight["manifest_sha256"],
        "eligibility_sha256": "d" * 64,
        "execution_fingerprint": "c" * 64,
    }
    identity["draft_sha256"] = _digest(identity["draft_identity"])
    identity["input_fingerprint"] = _digest({
        key: identity[key] for key in (
            "manifest_sha256", "eligibility_sha256", "chunk_sha256",
            "draft_sha256",
        )
    })
    fingerprint = _digest(identity)
    cell_dir = tmp_path / "c1"
    cell_dir.mkdir()
    (cell_dir / "call-intents.jsonl").write_text(
        "".join(json.dumps({
            "stage": stage,
            "logical_attempt": 1,
            "max_tokens": (
                synthesis_contract_identity(MICRO_SYNTHESIS_CONTRACT)
                ["component_hashes"]["stage_token_budget"]["max_tokens"][
                    next(
                        family for family in "ABDEF"
                        if f"micro_{family}" in stage
                    )
                ]
            ),
        }) + "\n" for stage in (
            "micro_A_task_design",
            "micro_B_claim_0_attempt_1",
            "micro_B_claim_0_attempt_2",
            "micro_B_claim_1_attempt_1",
            "micro_B_claim_2_attempt_1",
            "micro_D_sft",
        )),
        encoding="utf-8",
    )
    path = (
        cell_dir / "micro-journals" / "micro_synthesis_state" / "unit.jsonl"
    )
    store = MicroResumeStore(
        path, fingerprint=fingerprint, store_identity=identity,
    )
    terminal_units = [
        ("A", None), ("B", 0), ("B", 1), ("B", 2),
        ("C", None), ("D", None),
    ]
    for stage, slot in terminal_units:
        unit = f"instruction:{stage}" + (
            f":slot-{slot}" if slot is not None else ""
        )
        store.append(
            unit=unit, stage=stage, slot=slot, attempt=1,
            state="started",
        )
        store.append(
            unit=unit, stage=stage, slot=slot, attempt=1,
            state="terminal", artifact={"stage": stage},
        )
    report = verify_micro_journals(
        cell_dir,
        summary={
            "by_microstage": {"A": 1, "B": 4, "C": 1, "D": 1},
            "by_claim_slot": {"0": 1, "1": 1, "2": 1},
            "microstage_metrics": {
                "A": {"calls": 1, "completed_units": 1, "total_tokens": 10},
                "B": {
                    "calls": 4, "completed_units": 3, "repairs": 1,
                    "total_tokens": 40,
                },
                "D": {"calls": 1, "completed_units": 1, "total_tokens": 10},
                "C": {
                    "calls": 0, "completed_units": 1,
                    "total_tokens": 0, "deterministic_events": 1,
                },
            },
        },
        preflight=preflight,
    )
    assert report["journals"][0]["terminal_units"] == 6
    assert report["journals"][0]["path"] == (
        "micro-journals/micro_synthesis_state/unit.jsonl"
    )
    assert report["completed_by_microstage"]["B"] == 3
    assert report["by_microstage"]["C"] == 1
    assert report["microstage_metrics"]["C"]["calls"] == 0
    assert report["microstage_metrics"]["B"]["repairs"] == 1
    assert report["microstage_metrics"]["B"]["repair_count_source"] == (
        "call-intent-ledger"
    )
    assert len(report["report_sha256"]) == 64

    base_summary = {
        "by_microstage": {"A": 1, "B": 4, "C": 1, "D": 1},
        "by_claim_slot": {"0": 1, "1": 1, "2": 1},
        "microstage_metrics": {
            "A": {"calls": 1, "completed_units": 1, "total_tokens": 10},
            "B": {
                "calls": 4, "completed_units": 3, "repairs": 1,
                "total_tokens": 40,
            },
            "D": {"calls": 1, "completed_units": 1, "total_tokens": 10},
            "C": {
                "calls": 0, "completed_units": 1, "total_tokens": 0,
                "deterministic_events": 1,
            },
        },
    }
    for bad_summary in (
        {
            **base_summary,
            "by_microstage": {"A": 1},
            "microstage_metrics": {"A": {"calls": 1}},
        },
        {
            **base_summary,
            "by_microstage": {**base_summary["by_microstage"], "F": 1},
            "microstage_metrics": {
                **base_summary["microstage_metrics"], "F": {"calls": 1},
            },
        },
        {
            **base_summary,
            "microstage_metrics": {
                **base_summary["microstage_metrics"],
                "C": {"calls": 1, "deterministic_events": 1},
            },
        },
        {
            **base_summary,
            "microstage_metrics": {
                **base_summary["microstage_metrics"],
                "C": {"calls": 0, "deterministic_events": 0},
            },
        },
        {
            **base_summary,
            "microstage_metrics": {
                **base_summary["microstage_metrics"],
                "B": {
                    **base_summary["microstage_metrics"]["B"],
                    "completed_units": 4,
                },
            },
        },
        {
            **base_summary,
            "by_claim_slot": {"0": 1, "1": 1},
        },
    ):
        with pytest.raises(
            ValueError, match="telemetry|deterministic|claim-slot"
        ):
            verify_micro_journals(
                cell_dir, summary=bad_summary, preflight=preflight,
            )
    with pytest.raises(ValueError, match="result identity linkage"):
        verify_micro_journals(
            cell_dir, summary=base_summary, preflight=preflight, results=[],
        )

    # A durable start without a terminal can never be treated as resumable.
    store.append(
        unit="instruction:B:slot-0", stage="B", slot=0, attempt=1,
        state="started",
    )
    with pytest.raises(Exception) as exc:
        verify_micro_journals(
            cell_dir,
            summary={
                "by_microstage": {stage: 1 for stage in "ABCD"},
                "by_claim_slot": {"0": 1},
                "microstage_metrics": {
                    stage: {"calls": 1, "total_tokens": 10}
                    for stage in "ABCD"
                },
            },
            preflight=preflight,
        )
    assert getattr(exc.value, "code", None) == "staged_micro_resume_ambiguous"


def test_micro_journal_verifier_never_bootstraps_foreign_fingerprint(tmp_path):
    from Trainforge.generators.staged_synthesis_micro import MicroResumeStore

    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({
            "chunk_id": "generic-chunk", "chunk_sha256": "b" * 64,
            "kind": "instruction", "variant": "generic-contract",
            "repetition": 0,
        }) + "\n"
    )
    cell_dir = tmp_path / "c1"
    path = cell_dir / "micro-journals" / "foreign.jsonl"
    store = MicroResumeStore(path, fingerprint="a" * 64)
    for stage in "ABCD":
        store.append(
            unit=f"instruction:{stage}", stage=stage, slot=None, attempt=1,
            state="started",
        )
        store.append(
            unit=f"instruction:{stage}", stage=stage, slot=None, attempt=1,
            state="terminal", artifact={"stage": stage},
        )
    preflight = {
        "synthesis_contract": synthesis_contract_identity(
            MICRO_SYNTHESIS_CONTRACT
        ),
        "run_contract_sha256": "c" * 64,
        "telemetry_identity": {"expected_model": "served-model"},
        "pilot_run_id": "pilot-c1",
        "cell_id": "c1",
        "manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.jsonl").read_bytes()
        ).hexdigest(),
        "eligibility_sha256": "d" * 64,
        "micro_journal_root": "micro-journals",
    }
    with pytest.raises(ValueError, match="foreign|stale"):
        verify_micro_journals(
            cell_dir,
            summary={
                "by_microstage": {stage: 1 for stage in "ABCD"},
                "by_claim_slot": {"0": 1},
                "microstage_metrics": {
                    stage: {"calls": int(stage != "C")}
                    for stage in "ABCD"
                },
            },
            preflight=preflight,
        )


def test_micro_journal_verifier_ignores_decision_capture_catalog_side_effect(
    tmp_path,
):
    from Trainforge.generators.staged_synthesis_micro import MicroResumeStore

    (tmp_path / "manifest.jsonl").write_text(json.dumps({
        "chunk_id": "generic-chunk", "chunk_sha256": "b" * 64,
        "kind": "instruction", "variant": "generic-contract",
        "repetition": 0,
    }) + "\n")
    cell_dir = tmp_path / "c1"
    catalog_path = (
        cell_dir / "audit" / "runtime/training-captures" / "capture"
        / "micro_synthesis_state" / "catalog-only.jsonl"
    )
    store = MicroResumeStore(catalog_path, fingerprint="a" * 64)
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1, state="started",
    )
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1, state="terminal",
        artifact={"stage": "A"},
    )
    preflight = {
        "synthesis_contract": synthesis_contract_identity(
            MICRO_SYNTHESIS_CONTRACT
        ),
        "run_contract_sha256": "c" * 64,
        "telemetry_identity": {"expected_model": "served-model"},
        "pilot_run_id": "pilot-c1", "cell_id": "c1",
        "manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.jsonl").read_bytes()
        ).hexdigest(),
        "eligibility_sha256": "d" * 64,
        "micro_journal_root": "micro-journals",
    }
    with pytest.raises(ValueError, match="produced no resume journals"):
        verify_micro_journals(
            cell_dir,
            summary={
                "by_microstage": {"A": 1},
                "by_claim_slot": {},
                "microstage_metrics": {"A": {"calls": 1}},
            },
            preflight=preflight,
        )


def test_micro_terminal_stage_coverage_is_exact_and_kind_specific():
    assert _micro_stage_coverage_complete("instruction", "ABCD")
    assert _micro_stage_coverage_complete("preference", "ABCDEF")
    assert not _micro_stage_coverage_complete("instruction", "A")
    assert not _micro_stage_coverage_complete("preference", "ABCD")
    assert not _micro_stage_coverage_complete("instruction", "ABCDE")


def test_micro_draft_identity_is_canonical_and_manifest_bound():
    from Trainforge.scripts.harness.staged_window_abcd_pilot import (
        _canonical_micro_draft_identity,
    )

    row = {
        "chunk_id": "generic-chunk", "chunk_sha256": "b" * 64,
        "kind": "instruction", "variant": "generic-contract",
        "repetition": 2,
    }
    reordered = dict(reversed(tuple(row.items())))
    assert _digest(_canonical_micro_draft_identity(row)) == _digest(
        _canonical_micro_draft_identity(reordered)
    )
    for field, value in (
        ("variant", "foreign-contract"),
        ("repetition", 3),
        ("chunk_sha256", "c" * 64),
        ("kind", "preference"),
    ):
        mutated = {**row, field: value}
        assert _digest(_canonical_micro_draft_identity(mutated)) != _digest(
            _canonical_micro_draft_identity(row)
        )


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("pilot_run_id", "foreign-run"),
        ("cell_id", "foreign-cell"),
        ("model", "foreign-model"),
        ("execution_fingerprint", "f" * 64),
        ("manifest_sha256", "1" * 64),
        ("eligibility_sha256", "2" * 64),
        ("chunk_sha256", "3" * 64),
        ("draft_sha256", "4" * 64),
        ("draft_identity", {
            "chunk_id": "generic-chunk", "chunk_sha256": "b" * 64,
            "kind": "instruction", "variant": "stale-contract",
            "repetition": 0,
            "draft": {
                "provider": "local", "source_chunk_id": "generic-chunk",
            },
        }),
    ],
)
def test_micro_journal_rejects_self_consistent_foreign_identity(
    tmp_path, field, foreign,
):
    from Trainforge.generators.staged_synthesis_micro import (
        MicroResumeStore, micro_contract_fingerprint,
    )

    chunk_sha = "b" * 64
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({
            "chunk_id": "generic-chunk", "chunk_sha256": chunk_sha,
            "kind": "instruction", "variant": "generic-contract",
            "repetition": 0,
        }) + "\n"
    )
    preflight = {
        "synthesis_contract": synthesis_contract_identity(
            MICRO_SYNTHESIS_CONTRACT
        ),
        "run_contract_sha256": "c" * 64,
        "telemetry_identity": {"expected_model": "served-model"},
        "pilot_run_id": "pilot-c1",
        "cell_id": "c1",
        "manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.jsonl").read_bytes()
        ).hexdigest(),
        "eligibility_sha256": "d" * 64,
        "micro_journal_root": "micro-journals",
    }
    identity = {
        "contract": micro_contract_fingerprint(),
        "kind": "instruction",
        "chunk_id": "generic-chunk",
        "chunk_sha256": chunk_sha,
        "draft_identity": {
            "chunk_id": "generic-chunk", "chunk_sha256": chunk_sha,
            "kind": "instruction", "variant": "generic-contract",
            "repetition": 0,
            "draft": {
                "provider": "local", "source_chunk_id": "generic-chunk",
            },
        },
        "model": "served-model",
        "pilot_run_id": "pilot-c1",
        "cell_id": "c1",
        "manifest_sha256": preflight["manifest_sha256"],
        "eligibility_sha256": "d" * 64,
        "execution_fingerprint": "c" * 64,
    }
    identity["draft_sha256"] = _digest(identity["draft_identity"])
    identity[field] = foreign
    identity["input_fingerprint"] = _digest({
        key: identity[key] for key in (
            "manifest_sha256", "eligibility_sha256", "chunk_sha256",
            "draft_sha256",
        )
    })
    fingerprint = _digest(identity)
    cell_dir = tmp_path / "c1"
    store = MicroResumeStore(
        cell_dir / "micro-journals" / "foreign.jsonl",
        fingerprint=fingerprint, store_identity=identity,
    )
    for stage in "ABCD":
        for state in ("started", "terminal"):
            store.append(
                unit=f"instruction:{stage}", stage=stage, slot=None,
                attempt=1, state=state,
                artifact={"stage": stage} if state == "terminal" else None,
            )
    with pytest.raises(ValueError, match="foreign|stale"):
        verify_micro_journals(
            cell_dir,
            summary={
                "by_microstage": {stage: 1 for stage in "ABCD"},
                "by_claim_slot": {"0": 1},
                "microstage_metrics": {
                    stage: {"calls": int(stage != "C")}
                    for stage in "ABCD"
                },
            },
            preflight=preflight,
        )


def test_serial_dialect_preflight_requires_exact_constrained_value():
    class _Oa:
        _ollama_format_supported = False

        @staticmethod
        def _extract_json_lenient(raw):
            return __import__("json").loads(raw)

    class _BaseDialect:
        _oa_client = _Oa()
        _max_tokens = 800

    class _Provider:
        _base = _BaseDialect()

        @staticmethod
        def _structured_completion(
            messages, *, schema, meter_task, max_output_tokens,
        ):
            assert meter_task.endswith("dialect_preflight")
            assert schema["properties"]["probe"]["const"] == "ready"
            assert max_output_tokens == 800
            return '{"probe":"ready"}', {"prompt_tokens": 2}, 0

    result = negotiate_dialect(_Provider())
    assert result["outcome"] == "accepted"


def test_dialect_preflight_keeps_intent_bound_through_capture_and_http_audit(
    tmp_path,
):
    class _Capture:
        output_dir = tmp_path / "captures"
        fresh_start_id = None
        events = []

        def log_decision(self, **event):
            self.events.append({
                "event_id": f"EVT_{len(self.events) + 1:016d}",
                **event,
            })

    wire_payloads = []

    def handler(request):
        wire_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": '{"probe":"ready"}'},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        })

    manifest_path = tmp_path / "call-intents.jsonl"
    ledger_path = tmp_path / "http-attempts.jsonl"
    capture = _Capture()
    provider = _provider(
        attempt_ledger_path=ledger_path,
        raw_audit_root=tmp_path / "raw-http",
        intent_manifest_path=manifest_path,
        intent_run_id="run-generic",
        intent_cell_id="cell-generic",
        capture=capture,
    )
    ledgered_client = provider._base._oa_client._client
    ledgered_client._wrapped = httpx.Client(transport=httpx.MockTransport(handler))

    assert negotiate_dialect(provider)["outcome"] == "accepted"

    intents = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    attempts = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(intents) == 1
    assert intents[0]["kind"] == "dialect"
    assert intents[0]["response_dialect"] == "openai_json_schema_strict"
    assert len(wire_payloads) == 1
    assert "format" not in wire_payloads[0]
    assert intents[0]["request_sha256"] == request_sha256(wire_payloads[0])
    assert len(capture.events) == 1
    capture_context = json.loads(capture.events[0]["context"])
    assert capture_context["intent_request_sha256"] == intents[0]["request_sha256"]
    assert [row["event"] for row in attempts] == [
        "http_attempt_started", "http_attempt_terminal",
    ]

    decision_path = tmp_path / "decisions.jsonl"
    decision_bytes = (
        json.dumps(capture.events[0], sort_keys=True) + "\n"
    ).encode("utf-8")
    decision_path.write_bytes(decision_bytes)
    decision_report = verify_decision_audit(
        decision_path,
        expected_artifact_sha256=hashlib.sha256(decision_bytes).hexdigest(),
        intent_manifest_path=manifest_path,
        capture_closed=True,
    )
    assert decision_report["status"] == "accepted"
    assert reconcile_http_audit(
        ledger_path, intent_manifest_path=manifest_path,
    ) == {
        "status": "accepted",
        "started": 1,
        "terminal": 1,
        "errors": [],
    }


def test_strict_dialect_rejection_is_one_admitted_request_without_fallback(
    tmp_path,
):
    class _Capture:
        output_dir = tmp_path / "captures"
        fresh_start_id = None
        events = []

        def log_decision(self, **event):
            self.events.append(event)

    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "format" not in payload
        return httpx.Response(400, json={
            "detail": [{
                "type": "extra_forbidden",
                "loc": ["body", "format"],
                "msg": "Extra inputs are not permitted",
            }],
        })

    manifest_path = tmp_path / "call-intents.jsonl"
    ledger_path = tmp_path / "http-attempts.jsonl"
    capture = _Capture()
    provider = _provider(
        attempt_ledger_path=ledger_path,
        raw_audit_root=tmp_path / "raw-http",
        intent_manifest_path=manifest_path,
        intent_run_id="run-generic",
        intent_cell_id="cell-generic",
        capture=capture,
    )
    provider._base._oa_client._client._wrapped = httpx.Client(
        transport=httpx.MockTransport(handler)
    )

    with pytest.raises(Exception) as caught:
        negotiate_dialect(provider)
    assert getattr(caught.value, "code", None) == "400"
    intents = [
        json.loads(line) for line in manifest_path.read_text().splitlines()
    ]
    attempts = [
        json.loads(line) for line in ledger_path.read_text().splitlines()
    ]
    assert len(requests) == len(intents) == 1
    assert [row["event"] for row in attempts] == [
        "http_attempt_started", "http_attempt_terminal",
    ]
    assert attempts[-1]["http_status"] == 400
    assert len(capture.events) == 1


def test_transport_probe_does_not_apply_a_fixed_admission_cutoff():
    class _Client:
        calls = 0

        def post_with_usage(self, payload, *, task):
            self.calls += 1
            return {"usage": {}}, 0

        @staticmethod
        def _extract_usage(body):
            return body["usage"]

    class _Base:
        _oa_client = _Client()

    _TransportProbe(_Base())
    _Base._oa_client.post_with_usage(
        {"model": "served"}, task="staged_synthesis:plan",
    )
    assert _Base._oa_client.calls == 1


def _cell_rows(accepted_instruction=3, accepted_preference=3):
    accepted = {
        "instruction": accepted_instruction,
        "preference": accepted_preference,
    }
    seen = {"instruction": 0, "preference": 0}
    rows = []
    for index in range(8):
        kind = "instruction" if index < 4 else "preference"
        is_accepted = seen[kind] < accepted[kind]
        seen[kind] += 1
        row = {
            "chunk_id": f"chunk-{index}", "kind": kind,
            "variant": "D_production_contract", "repetition": 0,
            "accepted": is_accepted, "scoring_error": None,
            "stage_validity": is_accepted, "truncated": False,
            "leakage_passed": True, "scores": {},
            "error_code": None, "error_details": None,
            "pair_sha256": "a" * 64 if is_accepted else None,
        }
        if not is_accepted:
            row.update({
                "error_code": "staged_sft_realization_invalid",
                "error_details": {
                    "terminal_content_rejection": True,
                    "stage": "sft_realization",
                    "validation_error": "claim is not supported by evidence",
                },
            })
        rows.append(row)
    return rows


def _classify(rows, **overrides):
    def verified(schema, **fields):
        core = {"schema_version": schema, "status": "verified", **fields}
        return {
            **core,
            "report_sha256": hashlib.sha256(
                (json.dumps(
                    core, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ) + "\n").encode()
            ).hexdigest(),
        }
    audit_core = {
        "status": "accepted", "capture_closed": True,
        "provider_events_expected": 2, "provider_events_observed": 2,
        "provider_identities_verified": 2, "errors": [],
    }
    kwargs = {
        "expected_rows": rows,
        "publication_state": {"state": "committed_complete"},
        "reconciliation": verified(
            "ed4all.http-reconciliation-verification.v1",
            expected_attempts=12, started_attempts=12, terminal_attempts=12,
            errors=[],
        ),
        "telemetry_summary": verified(
            "ed4all.benchmark-telemetry-verification.v1",
            parser={"iteration_samples": 1}, errors=[],
        ),
        "decision_audit_report": {
            **audit_core,
            "report_sha256": hashlib.sha256(
                (json.dumps(
                    audit_core, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ) + "\n").encode()
            ).hexdigest(),
        },
        "active_requests": 0,
    }
    kwargs.update(overrides)
    return classify_cell_outcome(rows, **kwargs)


def test_operational_cell_advances_with_six_accepted_and_two_quality_rejections():
    outcome = _classify(_cell_rows())
    assert outcome["operationally_valid"]
    assert outcome["accepted_by_kind"] == {
        "instruction": 3, "preference": 3,
    }
    assert outcome["quality_rejections"] == {
        "staged_sft_realization_invalid": 2,
    }
    assert outcome["stage_counts"] == {"sft_realization": 2}


def test_plan_quality_rejections_use_real_typed_stages():
    rows = _cell_rows()
    for index, code, stage in (
        (0, "staged_plan_sft_invalid", "plan_sft"),
        (4, "staged_plan_dpo_invalid", "plan_dpo"),
    ):
        rows[index].update({
            "accepted": False,
            "stage_validity": False,
            "pair_sha256": None,
            "error_code": code,
            "error_details": {
                "terminal_content_rejection": True,
                "stage": stage,
                "validation_error": "required plan field is not evidenced",
            },
        })

    outcome = _classify(rows)

    assert outcome["operationally_valid"]
    assert outcome["quality_rejections"] == {
        "staged_plan_dpo_invalid": 1,
        "staged_plan_sft_invalid": 1,
        "staged_sft_realization_invalid": 2,
    }
    assert outcome["stage_counts"] == {
        "plan_dpo": 1,
        "plan_sft": 1,
        "sft_realization": 2,
    }


def test_objective_undersupport_is_explicit_post_validation_quality_rejection():
    rows = _cell_rows()
    rows[0].update({
        "accepted": False,
        "stage_validity": False,
        "pair_sha256": None,
        "error_code": "objective_statement_undersupported",
        "error_details": {
            "terminal_content_rejection": True,
            "quality_rejection": True,
            "stage": "post_validation",
            "validation_error": "objective_statement_undersupported",
        },
    })
    outcome = _classify(rows)
    assert outcome["operationally_valid"]
    assert outcome["quality_rejections"][
        "objective_statement_undersupported"
    ] == 1
    assert outcome["stage_counts"]["post_validation"] == 1


def test_objective_validation_unavailable_is_explicit_critical_failure():
    rows = _cell_rows()
    rows[0].update({
        "accepted": False,
        "stage_validity": False,
        "pair_sha256": None,
        "error_code": "objective_validation_unavailable",
        "error_details": {
            "terminal_content_rejection": False,
            "integrity_failure": True,
            "validation_error": "objective_validation_unavailable",
        },
    })
    outcome = _classify(rows)
    assert not outcome["operationally_valid"]
    assert "objective_validation_unavailable" in outcome["critical_errors"]
    assert all(item is not None for item in outcome["critical_errors"])


def test_dead_stage_aliases_cannot_classify_as_quality_rejections():
    for dead_alias in ("plan", "dialect_preflight"):
        rows = _cell_rows()
        rows[0].update({
            "accepted": False,
            "stage_validity": False,
            "pair_sha256": None,
            "error_code": "staged_plan_sft_invalid",
            "error_details": {
                "terminal_content_rejection": True,
                "stage": dead_alias,
                "validation_error": "required plan field is not evidenced",
            },
        })

        outcome = _classify(rows)

        assert not outcome["operationally_valid"], dead_alias
        assert outcome["quality_rejections"] == {
            "staged_sft_realization_invalid": 2,
        }, dead_alias
        assert outcome["critical_errors"] == [
            "unclassified:staged_plan_sft_invalid",
        ], dead_alias


def test_operational_cell_advances_with_only_one_accepted_of_each_kind():
    outcome = _classify(_cell_rows(1, 1))
    assert outcome["operationally_valid"]
    assert outcome["accepted_by_kind"] == {
        "instruction": 1, "preference": 1,
    }


def test_operational_cell_blocks_when_an_accepted_kind_is_missing():
    outcome = _classify(_cell_rows(0, 2))
    assert not outcome["operationally_valid"]
    assert "accepted_kind_missing" in outcome["integrity_errors"]


def test_every_critical_cell_class_blocks_advancement():
    for code in sorted(_CRITICAL_CELL_ERROR_CODES):
        rows = _cell_rows()
        fields = {}
        if "truncated" in code or "context_window_exceeded" in code:
            fields["truncated"] = True
        if code == "provider_output_verbatim_leakage":
            fields["leakage_passed"] = False
        rows[0].update({
            "accepted": False, "stage_validity": False,
            "error_code": code, **fields,
        })
        outcome = _classify(rows)
        assert not outcome["operationally_valid"], code
        assert outcome["critical_errors"], code
    rows = _cell_rows()
    rows[0].update({
        "accepted": False, "stage_validity": False,
        "error_code": "unexpected_failure",
    })
    assert not _classify(rows)["operationally_valid"]
    rows = _cell_rows()
    rows[0]["scoring_error"] = "RuntimeError"
    assert not _classify(rows)["operationally_valid"]


def test_operational_cell_requires_terminal_audit_and_publication_completeness():
    rows = _cell_rows()
    assert not _classify(rows[:-1], expected_rows=rows)["operationally_valid"]
    assert not _classify(
        rows, publication_state={"state": "marker_only"},
    )["operationally_valid"]
    assert not _classify(
        rows, decision_audit_report={"status": "rejected"},
    )["operationally_valid"]
    assert not _classify(rows, reconciliation={
        "status": "rejected", "started": 12, "terminal": 11,
        "errors": ["orphan"],
    })["operationally_valid"]
    assert not _classify(
        rows, telemetry_summary={"iteration_samples": []},
    )["operationally_valid"]
    assert not _classify(rows, active_requests=1)["operationally_valid"]


def test_terminal_hold_is_integrity_valid_but_never_advancing():
    outcome = _classify(
        _cell_rows(0, 0),
        publication_state={"state": "terminal_hold"},
    )

    assert outcome["publication_integrity_valid"] is True
    assert outcome["operationally_valid"] is False
    assert outcome["terminal_authority"] == "terminal_hold"
    assert outcome["nonadvancing_reasons"] == ["terminal_authority_hold"]
    assert "cell_not_committed" not in outcome["integrity_errors"]


def test_only_committed_complete_terminal_authority_advances():
    rows = _cell_rows()

    complete = _classify(
        rows, publication_state={"state": "committed_complete"},
    )
    legacy = _classify(rows, publication_state={"state": "committed"})

    assert complete["operationally_valid"] is True
    assert complete["nonadvancing_reasons"] == []
    assert legacy["publication_integrity_valid"] is True
    assert legacy["operationally_valid"] is False
    assert legacy["nonadvancing_reasons"] == ["not_committed_complete"]


def test_matrix_summary_migrates_terminal_hold_without_false_integrity_error():
    migrated = migrate_matrix_summary({
        "pilot_run_id": "generic-run",
        "outcomes": [{
            "cell": "c1",
            "operationally_valid": False,
            "terminal_authority": "terminal_hold",
            "integrity_errors": ["cell_not_committed"],
        }],
    })

    outcome = migrated["outcomes"][0]
    assert migrated["schema_version"] == "ed4all.staged-window-matrix-summary.v2"
    assert outcome["publication_integrity_valid"] is True
    assert outcome["terminal_authority"] == "terminal_hold"
    assert outcome["integrity_errors"] == []
    assert outcome["nonadvancing_reasons"] == ["terminal_authority_hold"]


def test_matrix_summary_backcompat_preserves_unknown_nonadvancing_authority():
    migrated = migrate_matrix_summary({
        "pilot_run_id": "generic-run",
        "outcomes": [{
            "cell": "c1",
            "operationally_valid": False,
            "integrity_errors": ["cell_not_committed"],
        }],
    })

    outcome = migrated["outcomes"][0]
    assert outcome["publication_integrity_valid"] is False
    assert outcome["terminal_authority"] == "unknown_legacy_authority"
    assert outcome["integrity_errors"] == []
    assert outcome["nonadvancing_reasons"] == ["not_committed_complete"]


def test_cell_order_is_deterministic_and_input_permutation_invariant():
    rows = [{
        "chunk_id": f"c-{index}", "kind": "instruction" if index % 2 else "preference",
        "repetition": 0,
    } for index in range(8)]
    first = counterbalanced_cell_rows(rows, run_id="run", cell_id="c16")
    reversed_input = counterbalanced_cell_rows(
        reversed(rows), run_id="run", cell_id="c16",
    )
    assert first == reversed_input
    assert {
        (row["chunk_id"], row["kind"]) for row in first
    } == {
        (row["chunk_id"], row["kind"]) for row in rows
    }
    assert first != counterbalanced_cell_rows(rows, run_id="run", cell_id="c20")


def test_http_reconciliation_rejects_orphan_started_and_terminal(tmp_path):
    path = tmp_path / "ledger.jsonl"
    common = {
        "unit": "u", "stage": "s", "attempt": 1, "request_sha256": "a" * 64,
    }
    intent = tmp_path / "call-intents.jsonl"
    intent.write_text(json.dumps({
        "intent_version": "ed4all.provider-call-intent.v1",
        "utc": "2026-01-01T00:00:00Z", "run_id": "run",
        "cell_id": "cell", "unit": "u", "kind": "initial", "stage": "s",
        "logical_attempt": 1, "request_sha256": "a" * 64,
        "model": "model", "contract_sha256": "b" * 64,
    }) + "\n", encoding="utf-8")
    path.write_text(__import__("json").dumps({
        **common, "event": "http_attempt_started",
    }) + "\n", encoding="utf-8")
    try:
        reconcile_http_audit(path, intent_manifest_path=intent)
    except RuntimeError:
        pass
    else:
        raise AssertionError("orphan started attempt must fail")

    path.write_text(__import__("json").dumps({
        **common, "event": "http_attempt_terminal",
    }) + "\n", encoding="utf-8")
    try:
        reconcile_http_audit(path, intent_manifest_path=intent)
    except RuntimeError:
        pass
    else:
        raise AssertionError("orphan terminal attempt must fail")


def test_error_detail_sanitizer_is_strict_recursive_allowlist():
    cleaned = _sanitize_error_details({
        "stage": "plan_sft",
        "validation_error": "missing key",
        "prompt_ref": "sha256:" + "a" * 64,
        "authorization": "Bearer secret",
        "api_key": "secret",
        "nested": {"password": "secret", "score": 1.0},
        "nli_scores": [{
            "entailment": 0.9,
            "contradiction": 0.01,
            "cookie": "secret",
            "messages": ["private"],
            "unknown": "drop",
        }],
        "body": "private",
        "headers": {"X-Secret": "private"},
    })
    assert cleaned == {
        "stage": "plan_sft",
        "validation_error": "missing key",
        "prompt_ref": "sha256:" + "a" * 64,
        "nli_scores": [{"entailment": 0.9, "contradiction": 0.01}],
    }
    serialized = __import__("json").dumps(cleaned)
    assert "secret" not in serialized
    assert "private" not in serialized

    injected = _sanitize_error_details({
        "validation_error": "Authorization: Bearer TOPSECRET",
        "path": "PRIVATE_LOCATION_SENTINEL",
        "scores": [{"score": "TOPSECRET"}],
        "nli_scores": [{
            "entailment": 0.5,
            "contradiction": 0.2,
            "validator_reason": "Bearer TOPSECRET",
        }],
    })
    assert injected == {
        "nli_scores": [{"entailment": 0.5, "contradiction": 0.2}],
    }
    assert "TOPSECRET" not in __import__("json").dumps(injected)


def _publication_rows():
    manifest = [
        {
            "chunk_id": f"chunk-{index}", "kind": "instruction",
            "variant": "contract", "repetition": 0,
        }
        for index in range(3)
    ]
    results = [
        {**row, "accepted": False, "error_code": f"reason-{index}"}
        for index, row in enumerate(manifest)
    ]
    return manifest, results


def test_micro_publication_reconciles_completion_permutation_deterministically():
    manifest, results = _publication_rows()
    permuted = [results[2], results[0], results[1]]
    first = canonicalize_micro_publication_rows(
        checkpoint_rows=permuted, result_rows=results,
        manifest_rows=manifest,
    )
    second = canonicalize_micro_publication_rows(
        checkpoint_rows=list(reversed(permuted)),
        result_rows=list(reversed(results)), manifest_rows=manifest,
    )
    assert first == second == results


def test_micro_publication_validates_frozen_row_id_and_legacy_adapter():
    manifest, results = _publication_rows()
    # Rows without row_id exercise the explicit legacy four-field adapter.
    assert canonicalize_micro_publication_rows(
        checkpoint_rows=results, result_rows=results, manifest_rows=manifest,
    ) == results
    for index, row in enumerate(manifest):
        row["row_id"] = f"cap-row-{index}"
        results[index]["row_id"] = row["row_id"]
    checkpoint = [dict(row) for row in results]
    checkpoint[0]["row_id"] = "cap-row-foreign"
    with pytest.raises(ValueError, match="frozen manifest"):
        canonicalize_micro_publication_rows(
            checkpoint_rows=checkpoint, result_rows=results,
            manifest_rows=manifest,
        )


@pytest.mark.parametrize(
    "mutation",
    ("content", "duplicate", "missing", "unknown", "identity"),
)
def test_micro_publication_reconciliation_fails_closed_on_mutation(mutation):
    manifest, results = _publication_rows()
    checkpoint = [dict(row) for row in results]
    if mutation == "content":
        checkpoint[0]["error_code"] = "changed"
    elif mutation == "duplicate":
        checkpoint[1] = dict(checkpoint[0])
    elif mutation == "missing":
        checkpoint.pop()
    elif mutation == "unknown":
        checkpoint[0]["chunk_id"] = "foreign-chunk"
    else:
        results[0]["kind"] = "preference"
    with pytest.raises(
        ValueError, match="micro publication|content hash|frozen manifest"
    ):
        canonicalize_micro_publication_rows(
            checkpoint_rows=checkpoint, result_rows=results,
            manifest_rows=manifest,
        )


def test_atomic_finalization_refuses_work_after_absolute_deadline(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.time.monotonic",
        lambda: 721.0,
    )
    target = tmp_path / "results.jsonl"
    try:
        _atomic_write_bytes(target, b"result\n", hard_deadline=720.0)
    except TimeoutError:
        pass
    else:
        raise AssertionError("post-stop bookkeeping crossed cell deadline")
    assert not target.exists()


def _fake_reconcile(_ledger, *, report_path, **_kwargs):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('{"status":"accepted"}\n', encoding="utf-8")
    return {"status": "accepted"}


def _audit_publication_kwargs(tmp_path):
    core = {
        "status": "accepted", "capture_closed": True,
        "provider_events_expected": 1, "provider_events_observed": 1,
        "provider_identities_verified": 1, "errors": [],
    }
    report = {
        **core,
        "report_sha256": hashlib.sha256(
            (json.dumps(
                core, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ) + "\n").encode()
        ).hexdigest(),
    }
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False) + "\n"
    ).encode()
    path = tmp_path / "source-decision-audit-report.json"
    path.write_bytes(payload)
    def verifier_report(name, schema):
        core = {"schema_version": schema, "status": "verified", "errors": []}
        report = {
            **core,
            "report_sha256": hashlib.sha256(
                (json.dumps(
                    core, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ) + "\n").encode()
            ).hexdigest(),
        }
        report_payload = (
            json.dumps(report, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n"
        ).encode()
        report_path = tmp_path / name
        report_path.write_bytes(report_payload)
        return report_path, hashlib.sha256(report_payload).hexdigest()
    telemetry_path, telemetry_sha = verifier_report(
        "source-telemetry-report.json",
        "ed4all.benchmark-telemetry-verification.v1",
    )
    reconciliation_path, reconciliation_sha = verifier_report(
        "source-reconciliation-report.json",
        "ed4all.http-reconciliation-verification.v1",
    )
    intent_path = tmp_path / "source-call-intents.jsonl"
    intent_path.write_text("", encoding="utf-8")
    return {
        "decision_audit_report_path": path,
        "decision_audit_report_sha256": hashlib.sha256(payload).hexdigest(),
        "telemetry_report_path": telemetry_path,
        "telemetry_report_sha256": telemetry_sha,
        "reconciliation_report_path": reconciliation_path,
        "reconciliation_report_sha256": reconciliation_sha,
        "intent_manifest_path": intent_path,
    }


def test_success_publication_commits_manifest_last(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.reconcile_http_audit",
        _fake_reconcile,
    )
    commit = publish_cell_success(
        cell_dir=tmp_path, ledger_path=tmp_path / "ledger",
        results=[{"unit": 1}], summary={"accepted": 1},
        hard_deadline=10**12,
        **_audit_publication_kwargs(tmp_path),
    )
    assert commit["status"] == "committed"
    assert (tmp_path / "success-commit.json").is_file()
    assert not (tmp_path / "finalization_in_progress.json").exists()
    for name, digest in commit["artifacts"].items():
        assert __import__("hashlib").sha256(
            (tmp_path / name).read_bytes()
        ).hexdigest() == digest


def test_micro_publication_hashes_same_checkpoint_and_journal_authority(
    tmp_path, monkeypatch,
):
    cell_dir = tmp_path / "qualification-c1"
    cell_dir.mkdir()
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.reconcile_http_audit",
        _fake_reconcile,
    )
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.verify_cell_publication",
        lambda *args, **kwargs: {"status": "verified"},
    )
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.verify_micro_journals",
        lambda *args, **kwargs: {
            "schema_version": "ed4all.staged-synthesis-micro-verification.v1",
            "journals": [{
                "path": "micro-journals/unit.jsonl",
                "sha256": "a" * 64,
                "terminal_units": 4,
            }],
            "report_sha256": "b" * 64,
        },
    )
    result = {
        "row_id": "cap-row-generic", "chunk_id": "generic-chunk",
        "kind": "instruction", "variant": "contract", "repetition": 0,
        "accepted": True,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps({
        key: result[key] for key in (
            "row_id", "chunk_id", "kind", "variant", "repetition",
        )
    }) + "\n")
    checkpoint = (
        json.dumps(
            {**result, "_checkpoint_state": "terminal"},
            sort_keys=True, separators=(",", ":"),
        ) + "\n"
    ).encode()
    (cell_dir / "checkpoint.jsonl").write_bytes(checkpoint)
    (cell_dir / "preflight.json").write_text(json.dumps({
        "synthesis_contract": synthesis_contract_identity(
            MICRO_SYNTHESIS_CONTRACT
        ),
        "micro_journal_root": "micro-journals",
        "checkpoint_path": "checkpoint.jsonl",
    }))
    commit = publish_cell_success(
        cell_dir=cell_dir, ledger_path=tmp_path / "ledger",
        results=[result], summary={"accepted": 1},
        hard_deadline=10**12,
        publication_identity={
            "synthesis_contract": synthesis_contract_identity(
                MICRO_SYNTHESIS_CONTRACT
            ),
        },
        **_audit_publication_kwargs(tmp_path),
    )
    assert set(("checkpoint.jsonl", "micro-verification.json")) <= set(
        commit["artifacts"]
    )
    assert commit["artifacts"]["checkpoint.jsonl"] == hashlib.sha256(
        checkpoint
    ).hexdigest()


def test_micro_publication_refuses_checkpoint_result_divergence(
    tmp_path, monkeypatch,
):
    cell_dir = tmp_path / "qualification-c1"
    cell_dir.mkdir()
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.reconcile_http_audit",
        _fake_reconcile,
    )
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.verify_micro_journals",
        lambda *args, **kwargs: {"journals": [{"path": "unit"}]},
    )
    identity = {
        "row_id": "cap-row-generic", "chunk_id": "generic-chunk",
        "kind": "instruction", "variant": "contract", "repetition": 0,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(identity) + "\n")
    (cell_dir / "checkpoint.jsonl").write_text(
        json.dumps({
            **identity, "accepted": False,
            "_checkpoint_state": "terminal",
        }) + "\n"
    )
    (cell_dir / "preflight.json").write_text(json.dumps({
        "synthesis_contract": synthesis_contract_identity(
            MICRO_SYNTHESIS_CONTRACT
        ),
        "micro_journal_root": "micro-journals",
        "checkpoint_path": "checkpoint.jsonl",
    }))
    with pytest.raises(ValueError, match="content hash"):
        publish_cell_success(
            cell_dir=cell_dir, ledger_path=tmp_path / "ledger",
            results=[{**identity, "accepted": True}], summary={"accepted": 1},
            hard_deadline=10**12,
            publication_identity={
                "synthesis_contract": synthesis_contract_identity(
                    MICRO_SYNTHESIS_CONTRACT
                ),
            },
            **_audit_publication_kwargs(tmp_path),
        )


def test_replace_deadline_crossing_leaves_durable_in_progress_marker(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    clock = [0.0]
    real_replace = module.os.replace
    calls = [0]

    def crossing_replace(source, target):
        calls[0] += 1
        real_replace(source, target)
        if calls[0] == 2:
            clock[0] = 11.0

    monkeypatch.setattr(module.os, "replace", crossing_replace)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    try:
        publish_cell_success(
            cell_dir=tmp_path, ledger_path=tmp_path / "ledger",
            results=[{"unit": 1}], summary={"accepted": 1},
            hard_deadline=10.0,
            **_audit_publication_kwargs(tmp_path),
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("deadline crossing during replace must fail")
    assert (tmp_path / "finalization_in_progress.json").is_file()
    assert not (tmp_path / "success-commit.json").exists()


def test_directory_fsync_deadline_crossing_never_looks_committed(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    clock = [0.0]
    real_fsync = module.os.fsync
    calls = [0]

    def crossing_fsync(fd):
        real_fsync(fd)
        calls[0] += 1
        if calls[0] == 3:
            clock[0] = 11.0

    monkeypatch.setattr(module.os, "fsync", crossing_fsync)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    try:
        publish_cell_success(
            cell_dir=tmp_path, ledger_path=tmp_path / "ledger",
            results=[{"unit": 1}], summary={"accepted": 1},
            hard_deadline=10.0,
            **_audit_publication_kwargs(tmp_path),
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("deadline crossing during fsync must fail")
    assert (tmp_path / "finalization_in_progress.json").is_file()
    assert not (tmp_path / "success-commit.json").exists()


def _publish_for_fault_test(tmp_path):
    return publish_cell_success(
        cell_dir=tmp_path, ledger_path=tmp_path / "ledger",
        results=[{"unit": 1}], summary={"accepted": 1},
        hard_deadline=10**12,
        **_audit_publication_kwargs(tmp_path),
    )


def _assert_single_authority(tmp_path):
    marker = (tmp_path / "finalization_in_progress.json").exists()
    success = (tmp_path / "success-commit.json").exists()
    assert marker ^ success


def test_commit_payload_fsync_failure_leaves_marker_only(tmp_path, monkeypatch):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    real_fsync = module.os.fsync

    def fail_prepared(fd):
        target = __import__("os").readlink(f"/proc/self/fd/{fd}")
        if ".success-commit-prepared" in target:
            raise OSError("prepared fsync fault")
        return real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", fail_prepared)
    try:
        _publish_for_fault_test(tmp_path)
    except OSError:
        pass
    else:
        raise AssertionError("commit payload fsync fault must fail")
    _assert_single_authority(tmp_path)
    assert read_cell_publication_state(tmp_path)["state"] == "marker_only"


def test_marker_content_replace_failure_leaves_marker_only(tmp_path, monkeypatch):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    real_replace = module.os.replace

    def fail_replace(source, target):
        if str(source).endswith(".success-commit-prepared.json"):
            raise OSError("marker content replace fault")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    try:
        _publish_for_fault_test(tmp_path)
    except OSError:
        pass
    else:
        raise AssertionError("marker-content replace fault must fail")
    _assert_single_authority(tmp_path)


def test_final_rename_failure_leaves_recoverable_commit_payload_at_marker(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    real_replace = module.os.replace

    def fail_final(source, target):
        if str(target).endswith("success-commit.json"):
            raise OSError("final rename fault")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_final)
    try:
        _publish_for_fault_test(tmp_path)
    except OSError:
        pass
    else:
        raise AssertionError("final rename fault must fail")
    _assert_single_authority(tmp_path)
    state = read_cell_publication_state(tmp_path)
    assert state["state"] == "marker_only"
    assert state["recoverable"] is True


def test_marker_content_directory_fsync_failure_leaves_marker_only(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    real_replace = module.os.replace
    real_fsync = module.os.fsync
    prepared_at_marker = [False]

    def observe_replace(source, target):
        result = real_replace(source, target)
        if str(source).endswith(".success-commit-prepared.json"):
            prepared_at_marker[0] = True
        return result

    def fail_marker_dir(fd):
        if prepared_at_marker[0]:
            raise OSError("marker directory fsync fault")
        return real_fsync(fd)

    monkeypatch.setattr(module.os, "replace", observe_replace)
    monkeypatch.setattr(module.os, "fsync", fail_marker_dir)
    try:
        _publish_for_fault_test(tmp_path)
    except OSError:
        pass
    else:
        raise AssertionError("marker directory fsync fault must fail")
    _assert_single_authority(tmp_path)
    assert read_cell_publication_state(tmp_path)["state"] == "marker_only"


def test_final_directory_fsync_fault_returns_commit_authoritative(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    real_fsync = module.os.fsync
    final_renamed = [False]
    real_replace = module.os.replace

    def observe_replace(source, target):
        result = real_replace(source, target)
        if str(target).endswith("success-commit.json"):
            final_renamed[0] = True
        return result

    def fail_final_dir(fd):
        if final_renamed[0]:
            raise OSError("final directory fsync fault")
        return real_fsync(fd)

    monkeypatch.setattr(module.os, "replace", observe_replace)
    monkeypatch.setattr(module.os, "fsync", fail_final_dir)
    result = _publish_for_fault_test(tmp_path)
    assert result["publication_recovery"] == "directory_fsync_required"
    _assert_single_authority(tmp_path)
    assert read_cell_publication_state(tmp_path)["state"] == "committed"


def test_success_publication_never_unlinks_marker(tmp_path, monkeypatch):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    monkeypatch.setattr(
        module.Path, "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("marker unlink is forbidden")
        ),
    )
    result = _publish_for_fault_test(tmp_path)
    assert result["status"] == "committed"
    _assert_single_authority(tmp_path)


def test_publication_reader_crash_state_precedence(tmp_path):
    assert read_cell_publication_state(tmp_path)["state"] == (
        "invalid_missing_authority"
    )
    marker = tmp_path / "finalization_in_progress.json"
    marker.write_text('{"status":"finalization_in_progress"}', encoding="utf-8")
    assert read_cell_publication_state(tmp_path) == {
        "state": "marker_only", "recoverable": False,
        "marker": {"status": "finalization_in_progress"},
    }
    marker.write_text('{"status":"committed"}', encoding="utf-8")
    assert read_cell_publication_state(tmp_path)["recoverable"] is True
    success = tmp_path / "success-commit.json"
    success.write_text('{"status":"committed"}', encoding="utf-8")
    assert read_cell_publication_state(tmp_path)["state"] == (
        "invalid_dual_authority"
    )
    marker.unlink()
    assert read_cell_publication_state(tmp_path)["state"] == "committed"


def test_preexisting_commit_is_rejected_before_new_finalization(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "Trainforge.scripts.harness.staged_window_abcd_pilot.reconcile_http_audit",
        _fake_reconcile,
    )
    (tmp_path / "success-commit.json").write_text(
        '{"status":"committed"}', encoding="utf-8",
    )
    try:
        _publish_for_fault_test(tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("preexisting authority must require recovery")
    assert not (tmp_path / "finalization_in_progress.json").exists()


def test_precommit_rejection_writes_durable_failure_without_success_authority(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    monkeypatch.setattr(
        module, "verify_cell_publication",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("candidate verification rejected")
        ),
    )
    with __import__("pytest").raises(ValueError):
        publish_cell_success(
            cell_dir=tmp_path, ledger_path=tmp_path / "ledger",
            results=[{"unit": 1}], summary={"accepted": 1},
            hard_deadline=10**12,
            publication_identity={
                "model_id": "generic-model",
                "pilot_run_id": "generic-run",
                "cell_id": "generic-cell",
                "preflight_sha256": "a" * 64,
            },
            authority_status="committed_complete",
            **_audit_publication_kwargs(tmp_path),
        )
    failure = json.loads(
        (tmp_path / "publication-integrity-failure.json").read_text()
    )
    assert failure["status"] == "publication_integrity_failed"
    assert failure["candidate_artifacts"]
    assert len(failure["candidate_commit_sha256"]) == 64
    assert failure["verifier_report"]["status"] == "rejected"
    assert (tmp_path / "finalization_in_progress.json").is_file()
    assert not (tmp_path / "committed_complete.json").exists()
    assert not (tmp_path / "terminal_hold.json").exists()
    assert not (tmp_path / "success-commit.json").exists()


def test_verified_candidate_uses_single_typed_terminal_authority(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    monkeypatch.setattr(
        module, "verify_cell_publication",
        lambda *_args, **_kwargs: {"status": "verified"},
    )
    result = publish_cell_success(
        cell_dir=tmp_path, ledger_path=tmp_path / "ledger",
        results=[{"unit": 1}], summary={"accepted": 1},
        hard_deadline=10**12,
        publication_identity={
            "model_id": "generic-model", "pilot_run_id": "generic-run",
            "cell_id": "generic-cell", "preflight_sha256": "a" * 64,
        },
        authority_status="terminal_hold",
        **_audit_publication_kwargs(tmp_path),
    )
    assert result["status"] == "terminal_hold"
    assert read_cell_publication_state(tmp_path)["state"] == "terminal_hold"
    assert (tmp_path / "terminal_hold.json").is_file()
    assert not (tmp_path / "success-commit.json").exists()


def test_typed_terminal_authority_rename_crash_leaves_only_marker(
    tmp_path, monkeypatch,
):
    module = __import__(
        "Trainforge.scripts.harness.staged_window_abcd_pilot", fromlist=["unused"],
    )
    monkeypatch.setattr(module, "reconcile_http_audit", _fake_reconcile)
    monkeypatch.setattr(
        module, "verify_cell_publication",
        lambda *_args, **_kwargs: {"status": "verified"},
    )
    real_replace = module.os.replace

    def crash_before_authority(source, target):
        if str(target).endswith("committed_complete.json"):
            raise OSError("simulated terminal authority rename crash")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", crash_before_authority)
    with __import__("pytest").raises(OSError):
        publish_cell_success(
            cell_dir=tmp_path, ledger_path=tmp_path / "ledger",
            results=[{"unit": 1}], summary={"accepted": 1},
            hard_deadline=10**12,
            publication_identity={
                "model_id": "generic-model", "pilot_run_id": "generic-run",
                "cell_id": "generic-cell", "preflight_sha256": "a" * 64,
            },
            authority_status="committed_complete",
            **_audit_publication_kwargs(tmp_path),
        )
    state = read_cell_publication_state(tmp_path)
    assert state["state"] == "marker_only"
    assert not (tmp_path / "committed_complete.json").exists()
    assert not (tmp_path / "success-commit.json").exists()


def test_planned_cells_form_maximally_even_latin_position_schedule():
    rows = [{
        "chunk_id": f"c-{index}",
        "kind": "instruction" if index % 2 else "preference",
        "repetition": 0,
    } for index in range(8)]
    cells = ("c1", "c16", "c20", "c24", "c28")
    schedules = [
        counterbalanced_cell_rows(reversed(rows), run_id="run", cell_id=cell)
        for cell in cells
    ]

    identity = lambda row: (row["chunk_id"], row["kind"], row["repetition"])
    position_counts = {
        identity(row): [0] * len(rows)
        for row in rows
    }
    for schedule in schedules:
        assert len({identity(row) for row in schedule}) == len(rows)
        for position, row in enumerate(schedule):
            position_counts[identity(row)][position] += 1

    # Five placements across eight positions can only be balanced at 0 or 1.
    assert all(
        max(counts) - min(counts) == 1
        and sum(counts) == len(cells)
        for counts in position_counts.values()
    )
    assert all(
        len({
            next(
                position
                for position, row in enumerate(schedule)
                if identity(row) == unit
            )
            for schedule in schedules
        }) == len(cells)
        for unit in position_counts
    )


def test_benchmark_stop_after_c1_constructs_only_c1():
    assert planned_benchmark_cells(1) == [(1, "c1")]


def test_benchmark_default_preserves_legacy_advancement_cells():
    assert planned_benchmark_cells() == [
        (1, "c1"), (16, "c16"), (20, "c20"), (24, "c24"), (28, "c28"),
    ]


def test_qualification_route_binding_requires_explicit_env_corroboration():
    expected = {
        "LOCAL_SYNTHESIS_BASE_URL": "http://127.0.0.1:8123/v1",
        "LOCAL_SYNTHESIS_MODEL": "served-revision",
    }
    assert qualification_route_binding(
        explicit_base_url=expected["LOCAL_SYNTHESIS_BASE_URL"],
        explicit_model=expected["LOCAL_SYNTHESIS_MODEL"],
        environ=expected,
    ) == {
        "base_url": "http://127.0.0.1:8123/v1",
        "served_model": "served-revision",
    }
    for environ in (
        {},
        {**expected, "LOCAL_SYNTHESIS_BASE_URL": "http://localhost:8000/v1"},
        {**expected, "LOCAL_SYNTHESIS_MODEL": "foreign-revision"},
    ):
        with pytest.raises(ValueError, match="route binding"):
            qualification_route_binding(
                explicit_base_url="http://127.0.0.1:8123/v1",
                explicit_model="served-revision",
                environ=environ,
            )


def test_latin_schedule_is_invariant_to_every_input_permutation():
    rows = [{
        "chunk_id": f"c-{index}",
        "kind": "instruction" if index % 2 else "preference",
        "repetition": 0,
    } for index in range(8)]
    permutations = (
        rows,
        list(reversed(rows)),
        rows[3:] + rows[:3],
        rows[::2] + rows[1::2],
    )
    for cell in ("c1", "c16", "c20", "c24", "c28"):
        expected = counterbalanced_cell_rows(
            permutations[0], run_id="run", cell_id=cell,
        )
        assert all(
            counterbalanced_cell_rows(
                permutation, run_id="run", cell_id=cell,
            ) == expected
            for permutation in permutations[1:]
        )


def test_240_second_dialect_time_is_charged_to_12_minute_cell_budget():
    assert remaining_cell_budget(
        cell_started=1000.0,
        global_deadline=10000.0,
        now=1240.0,
    ) == 480.0
    assert remaining_cell_budget(
        cell_started=1000.0,
        global_deadline=1300.0,
        now=1240.0,
    ) == 60.0


def test_every_stable_preflight_contract_field_changes_digest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    eligibility = tmp_path / "eligibility.json"
    manifest.write_text('{"unit":1}\n')
    eligibility.write_text('{"eligible":8}\n')
    base = {
        "manifest_path": manifest,
        "eligibility_path": eligibility,
        "base_url": "http://user:secret@localhost:8123/v1?secret=yes",
        "model_snapshot": {"data": [{"id": "model-a"}]},
        "run_id": "pilot-a",
        "server_batch": 28,
        "client_concurrency": 16,
        "timeout_seconds": 240,
        "transport_attempts": 1,
        "initial_backoff_seconds": 1,
        "max_tokens": 800,
        "temperature": 0.4,
        "dialect": {
            "outcome": "accepted", "schema_sha256": "a" * 64,
        },
    }
    baseline = build_preflight_artifact(**base)
    mutations = (
        ("base_url", "http://localhost:9000/v1"),
        ("model_snapshot", {"data": [{"id": "model-b"}]}),
        ("run_id", "pilot-b"),
        ("server_batch", 16),
        ("client_concurrency", 20),
        ("timeout_seconds", 239),
        ("transport_attempts", 2),
        ("initial_backoff_seconds", 2),
        ("max_tokens", 700),
        ("temperature", 0.2),
        ("dialect", {"outcome": "accepted", "schema_sha256": "b" * 64}),
    )
    for field, value in mutations:
        changed = build_preflight_artifact(**{**base, field: value})
        assert changed["run_contract_sha256"] != baseline["run_contract_sha256"]
    manifest.write_text('{"unit":2}\n')
    assert build_preflight_artifact(**base)["run_contract_sha256"] != (
        baseline["run_contract_sha256"]
    )
    manifest.write_text('{"unit":1}\n')
    eligibility.write_text('{"eligible":7}\n')
    assert build_preflight_artifact(**base)["run_contract_sha256"] != (
        baseline["run_contract_sha256"]
    )
    assert baseline["base_url"] == "http://localhost:8123/v1"


def test_simultaneous_critical_and_success_are_both_checkpointed(tmp_path):
    import threading

    rows = build_pilot_rows(_chunks(), objectives=_objectives())[:2]
    barrier = threading.Barrier(2)

    class _Provider:
        def _run(self):
            barrier.wait()
            if threading.current_thread().name.endswith("_0"):
                error = RuntimeError("critical")
                error.code = "max_retries_exceeded"
                raise error
            return {
                "prompt": "Analyze evidence.",
                "completion": "Evidence supports the objective.",
                "chosen": "Evidence supports the objective.",
                "rejected": "Wrong because evidence was ignored.",
            }

        paraphrase_instruction = lambda self, draft, chunk: self._run()
        paraphrase_preference = lambda self, draft, chunk: self._run()

    checkpoint = tmp_path / "simultaneous.jsonl"
    try:
        execute_pilot(
            rows, _Provider(), objectives=_objectives(),
            scorer=lambda *args: {"accepted": True},
            max_concurrent=2, checkpoint_path=checkpoint,
            request_timeout_seconds=0.1,
        )
    except RuntimeError:
        pass
    saved = [__import__("json").loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(saved) == 2
    assert len({(row["chunk_id"], row["variant"]) for row in saved}) == 2


def test_execute_pilot_resumes_terminal_checkpoint_without_reentry(tmp_path):
    rows = build_pilot_rows(_chunks(), objectives=_objectives())[:1]

    class _Provider:
        calls = 0

        def paraphrase_instruction(self, draft, chunk):
            self.calls += 1
            return {
                "prompt": "Analyze evidence.",
                "completion": "Evidence supports the objective.",
            }

        paraphrase_preference = paraphrase_instruction

    checkpoint = tmp_path / "resume.jsonl"
    first = _Provider()
    expected, _ = execute_pilot(
        rows, first, objectives=_objectives(),
        scorer=lambda *args: {"accepted": True},
        checkpoint_path=checkpoint,
    )
    assert first.calls == 1
    second = _Provider()
    resumed, _ = execute_pilot(
        rows, second, objectives=_objectives(),
        scorer=lambda *args: {"accepted": True},
        checkpoint_path=checkpoint,
    )
    assert second.calls == 0
    assert resumed == expected


def test_execute_pilot_replays_terminal_quality_rejection_without_new_intent(
    tmp_path,
):
    row = build_pilot_rows(_chunks(), objectives=_objectives())[0]
    checkpoint = tmp_path / "quality-hold.jsonl"
    terminal = {
        key: value for key, value in row.items() if key != "_chunk"
    }
    terminal.update({
        "accepted": False,
        "stage_validity": False,
        "error_code": "staged_plan_sft_invalid",
        "error_details": {
            "terminal_content_rejection": True,
            "stage": "plan_sft",
            "validation_error": "objective obligation omitted",
        },
        "scoring_error": None,
        "truncated": False,
        "calls": 1,
        "requests": 1,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "latency_seconds": 1.0,
        "tokens_per_second": 30.0,
        "pair_sha256": None,
    })
    checkpoint.write_text(
        json.dumps(
            {**terminal, "_checkpoint_state": "terminal"},
            sort_keys=True, separators=(",", ":"),
        ) + "\n"
    )

    class _NeverCalled:
        calls = 0

        def paraphrase_instruction(self, draft, chunk):
            self.calls += 1
            raise AssertionError("terminal quality hold must replay")

        paraphrase_preference = paraphrase_instruction

    provider = _NeverCalled()
    results, _summary = execute_pilot(
        [row], provider, objectives=_objectives(),
        checkpoint_path=checkpoint,
    )
    assert provider.calls == 0
    assert results == [terminal]
    assert checkpoint.read_text().count("\n") == 1


def test_execute_pilot_rejects_ambiguous_checkpoint_start(tmp_path):
    row = build_pilot_rows(_chunks(), objectives=_objectives())[0]
    checkpoint = tmp_path / "ambiguous.jsonl"
    checkpoint.write_text(json.dumps({
        **{key: value for key, value in row.items() if key != "_chunk"},
        "_checkpoint_state": "started",
    }) + "\n")
    with pytest.raises(ValueError, match="started rows without terminal"):
        execute_pilot(
            [row], object(), objectives=_objectives(),
            checkpoint_path=checkpoint,
        )


def test_fail_close_persists_bounded_drained_future_without_duplicates(tmp_path):
    import threading
    import time

    rows = build_pilot_rows(_chunks(), objectives=_objectives())[:2]
    barrier = threading.Barrier(2)
    counter = iter((0, 1))
    lock = threading.Lock()

    class _Provider:
        def _run(self):
            with lock:
                index = next(counter)
            barrier.wait()
            if index == 0:
                error = RuntimeError("critical")
                error.code = "max_retries_exceeded"
                raise error
            time.sleep(0.05)
            return {
                "prompt": "Analyze evidence.",
                "completion": "Evidence supports the objective.",
                "chosen": "Evidence supports the objective.",
                "rejected": "Wrong because evidence was ignored.",
            }

        paraphrase_instruction = lambda self, draft, chunk: self._run()
        paraphrase_preference = lambda self, draft, chunk: self._run()

    checkpoint = tmp_path / "drain.jsonl"
    started = time.monotonic()
    try:
        execute_pilot(
            rows, _Provider(), objectives=_objectives(),
            scorer=lambda *args: {"accepted": True},
            max_concurrent=2, checkpoint_path=checkpoint,
            request_timeout_seconds=0.2,
        )
    except RuntimeError:
        pass
    assert time.monotonic() - started < 0.5
    saved = [__import__("json").loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(saved) == 2
    assert len({(row["chunk_id"], row["variant"]) for row in saved}) == 2
