from __future__ import annotations

from copy import deepcopy
import pytest

from Trainforge.generators.objective_execution_contract import (
    CANDIDATE_COMPLETION_CAPS,
)
from Trainforge.scripts.harness import completion_cap_qualification as capq


def _row(key, ordinal):
    kind, bloom, requirement_bucket, task_family, source_cardinality = key
    objective = {
        "id": f"objective-{ordinal}",
        "statement": f"Explain qualification objective {ordinal}.",
        "bloom_level": bloom,
    }
    core = {
        "contract": capq.CAP_QUALIFICATION_CONTRACT,
        "row_id": f"cap-row-{ordinal:04d}",
        "kind": kind,
        "variant": "D_production_contract",
        "repetition": 0,
        "cohort_index": ordinal,
        "order_index": 0,
        "chunk_id": f"chunk-{ordinal:04d}",
        "chunk_sha256": f"{ordinal:064x}",
        "focus_objective": objective,
        "objective_contract_sha256": f"{ordinal + 1:064x}",
        "requirement_count": 1,
        "stratum": [
            bloom, requirement_bucket, task_family, source_cardinality,
        ],
        "evidence_window_sha256": f"{ordinal + 2:064x}",
        "_chunk": {"id": f"chunk-{ordinal:04d}"},
    }
    return core


def _rows():
    rows = [
        _row(key, index) for index, key in enumerate(capq.required_strata())
    ]
    for index, row in enumerate(rows):
        row["cohort_index"] = index
        row["order_index"] = 0
    return rows


def _runner_rows():
    all_rows = _rows()
    rows = [
        *[row for row in all_rows if row["kind"] == "instruction"][:10],
        *[row for row in all_rows if row["kind"] == "preference"][:9],
    ]
    for index, row in enumerate(rows):
        row["cohort_index"] = index
        row["order_index"] = 0
    return rows


def _paired_candidate_fixture():
    non_kind = list(dict.fromkeys(
        key[1:] for key in capq.required_strata()
    ))
    chunks = [{"id": f"source-{index:04d}"} for index in range(len(non_kind))]
    candidates = {}
    ordinal = 0
    for index, stratum in enumerate(non_kind):
        for kind in capq.PAIR_KINDS:
            row = _row((kind, *stratum), ordinal)
            row["chunk_id"] = chunks[index]["id"]
            row["_chunk"] = deepcopy(chunks[index])
            candidates[(chunks[index]["id"], kind)] = row
            ordinal += 1
    return chunks, candidates


def _observations(rows, *, lengths=None):
    manifests = capq.cap_manifests(rows)
    result = []
    lengths = lengths or {}
    for cap in CANDIDATE_COMPLETION_CAPS:
        for row in rows:
            chars = lengths.get(cap, 400)
            result.append({
                "completion_cap": cap,
                "base_manifest_sha256": manifests[cap][
                    "base_manifest_sha256"
                ],
                "cap_manifest_sha256": manifests[cap]["cap_manifest_sha256"],
                "row_id": row["row_id"],
                "status": "accepted",
                "reason_code": "accepted",
                "critical_error": False,
                "completion_chars": chars,
                "completion_tokens": chars // 4,
                "finish_reason": "stop",
                "truncated": False,
                "requirement_pass_rate": 1.0,
                "result_coverage_rate": 1.0,
                "claim_support_rate": 1.0,
                "contradiction_rate": 0.0,
                "unsupported_added_claim_rate": 0.0,
                "reconciliation_rate": 1.0,
                "latency_seconds": 1.5,
                "output_tokens": chars // 4,
                "throughput_tokens_per_second": 40.0,
            })
    return result


def test_required_strata_is_complete_canonical_cross_product():
    strata = capq.required_strata()
    assert len(strata) == 2 * 6 * 4 * 4 * 2
    assert len(set(strata)) == len(strata)
    assert strata[0] == (
        "instruction", "remember", "1-2", "declarative", "single",
    )
    assert strata[-1] == (
        "preference", "create", "7+", "computational", "multi",
    )
    assert len(capq.qualification_contract_fingerprint()) == 64


def test_frozen_manifest_is_order_independent_and_covers_every_stratum(
    monkeypatch,
):
    chunks, by_chunk_kind = _paired_candidate_fixture()

    def fake_candidate(chunk, *, kind, objectives):
        del objectives
        return deepcopy(by_chunk_kind[(chunk["id"], kind)])

    monkeypatch.setattr(capq, "_candidate", fake_candidate)
    forward, forward_audit = capq.build_frozen_manifest(
        chunks, objectives={},
    )
    reverse, reverse_audit = capq.build_frozen_manifest(
        reversed(chunks), objectives={},
    )
    assert [row["row_id"] for row in forward] == [
        row["row_id"] for row in reverse
    ]
    assert forward_audit["manifest_sha256"] == reverse_audit["manifest_sha256"]
    assert forward_audit["selected_rows"] == 384
    assert forward_audit["required_strata"] == 384


def test_frozen_manifest_fails_closed_without_every_stratum(monkeypatch):
    chunks, by_chunk_kind = _paired_candidate_fixture()
    by_chunk_kind.pop((chunks[-1]["id"], "preference"))

    def fake_candidate(chunk, *, kind, objectives):
        del objectives
        try:
            return deepcopy(by_chunk_kind[(chunk["id"], kind)])
        except KeyError as exc:
            raise capq.CapQualificationError("ineligible") from exc

    monkeypatch.setattr(capq, "_candidate", fake_candidate)
    with pytest.raises(
        capq.CapQualificationError, match="lacks required strata",
    ):
        capq.build_frozen_manifest(chunks, objectives={})


def test_empirical_manifest_covers_observed_axes_and_pairwise_combinations(
    monkeypatch,
):
    chunks, by_chunk_kind = _paired_candidate_fixture()
    # Keep a bounded, non-Cartesian observed universe.
    chunks = chunks[:24]

    def fake_candidate(chunk, *, kind, objectives):
        del objectives
        return deepcopy(by_chunk_kind[(chunk["id"], kind)])

    monkeypatch.setattr(capq, "_candidate", fake_candidate)
    rows, audit = capq.build_empirical_set_cover_manifest(
        reversed(chunks), objectives={},
    )
    observed_tokens = set().union(*(
        capq._coverage_tokens((row["kind"], *tuple(row["stratum"])))
        for row in by_chunk_kind.values()
        if row["chunk_id"] in {chunk["id"] for chunk in chunks}
    ))
    selected_tokens = set().union(*(
        capq._coverage_tokens((row["kind"], *tuple(row["stratum"])))
        for row in rows
    ))
    assert selected_tokens == observed_tokens
    assert len(rows) <= 48
    assert audit["coverage_mode"] == (
        "observed-axis-risk-pairwise-irreducible-greedy"
    )
    assert audit["preflight_status"] == "ready"
    assert 16 <= len(rows) <= 32
    assert sum(row["kind"] == "instruction" for row in rows) >= 6
    assert sum(row["kind"] == "preference" for row in rows) >= 6
    assert audit["five_cap_units"] <= 160
    assert set(audit["mandatory_stress_rows"]) >= {
        "max_requirement_count", "max_objective_chars",
        "max_evidence_chars", "max_source_blocks", "longest_real_source",
    }
    assert audit["absent_required_strata"]
    assert len(audit["manifest_sha256"]) == 64
    assert len(capq.cap_manifests(rows)) == 5


def test_production_new_stratum_requires_requalification():
    rows = _rows()[:2]
    audit = {
        "selected_combined_strata": [
            [row["kind"], *row["stratum"]] for row in rows
        ],
    }
    capq.require_qualified_production_strata(audit, rows)
    with pytest.raises(
        capq.CapQualificationError, match="unqualified strata",
    ):
        capq.require_qualified_production_strata(audit, [_rows()[2]])


def test_cap_manifests_bind_identical_ordered_rows_across_all_five_caps():
    manifests = capq.cap_manifests(_runner_rows())
    assert tuple(manifests) == CANDIDATE_COMPLETION_CAPS
    assert len({
        value["base_manifest_sha256"] for value in manifests.values()
    }) == 1
    assert len({
        tuple(value["ordered_row_ids"]) for value in manifests.values()
    }) == 1
    assert len({
        value["cap_manifest_sha256"] for value in manifests.values()
    }) == 5
    assert len({
        value["runner_identity_sha256"] for value in manifests.values()
    }) == 1


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "reorder"))
def test_runner_manifest_identity_fails_before_traffic(mutation):
    rows = _runner_rows()
    if mutation == "missing":
        rows[0].pop("variant")
    elif mutation == "duplicate":
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
        rows[1]["kind"] = rows[0]["kind"]
        rows[1]["variant"] = rows[0]["variant"]
        rows[1]["repetition"] = rows[0]["repetition"]
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(
        (capq.CapQualificationError, KeyError),
    ):
        capq.cap_manifests(rows)


def test_adjudication_selects_smallest_cap_with_per_bucket_headroom():
    rows = _runner_rows()
    result = capq.adjudicate_caps(
        rows,
        _observations(rows, lengths={
            600: 550,
            800: 700,
            1000: 700,
            1200: 700,
            1600: 700,
        }),
    )
    assert result["caps"][600]["failure_reasons"] == [
        "accepted_utilization_not_below_90_percent",
        "pooled_p95_above_80_percent",
    ]
    assert result["caps"][800]["failure_reasons"] == [
        "pooled_p95_above_80_percent",
    ]
    assert result["caps"][1000]["qualifies"] is True
    assert result["selected_cap"] == 1000
    assert result["decision"] == "selected"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ({"finish_reason": "length"}, "length_finish"),
        ({"truncated": True}, "truncation"),
        ({"requirement_pass_rate": 0.99}, "accepted_execution_incomplete"),
        ({"result_coverage_rate": 0.0}, "accepted_execution_incomplete"),
        ({"reconciliation_rate": 0.99}, "accepted_execution_incomplete"),
    ),
)
def test_adjudication_rejects_critical_completion_failures(mutation, reason):
    rows = _runner_rows()
    observations = _observations(rows, lengths={cap: 100 for cap in CANDIDATE_COMPLETION_CAPS})
    target = next(item for item in observations if item["completion_cap"] == 600)
    target.update(mutation)
    result = capq.adjudicate_caps(rows, observations)
    assert reason in result["caps"][600]["failure_reasons"]
    assert result["selected_cap"] != 600


def test_adjudication_rejects_regression_against_600_control():
    rows = _runner_rows()
    observations = _observations(
        rows, lengths={cap: 100 for cap in CANDIDATE_COMPLETION_CAPS},
    )
    target = next(item for item in observations if item["completion_cap"] == 800)
    target["unsupported_added_claim_rate"] = 1.0
    target["contradiction_rate"] = 1.0
    result = capq.adjudicate_caps(rows, observations)
    assert result["caps"][800]["failure_reasons"] == [
        "unsupported_claim_regression",
        "contradiction_regression",
    ]


def test_adjudication_fails_closed_on_missing_duplicate_or_wrong_manifest():
    rows = _runner_rows()
    observations = _observations(rows)
    with pytest.raises(capq.CapQualificationError, match="incomplete"):
        capq.adjudicate_caps(rows, observations[:-1])
    with pytest.raises(capq.CapQualificationError, match="duplicate"):
        capq.adjudicate_caps(rows, [*observations, observations[0]])
    tampered = deepcopy(observations)
    tampered[0]["base_manifest_sha256"] = "0" * 64
    with pytest.raises(capq.CapQualificationError, match="binding mismatch"):
        capq.adjudicate_caps(rows, tampered)
    cross_cap = deepcopy(observations)
    manifests = capq.cap_manifests(rows)
    cross_cap[0]["cap_manifest_sha256"] = manifests[800][
        "cap_manifest_sha256"
    ]
    with pytest.raises(capq.CapQualificationError, match="cap binding mismatch"):
        capq.adjudicate_caps(rows, cross_cap)
