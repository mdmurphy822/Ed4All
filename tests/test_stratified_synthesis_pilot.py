from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.stratified_synthesis_pilot import (
    compare_reports,
    fallback_provenance_summary,
    isolated_runtime_environment,
    prepare_workspace,
    rejection_efficiency_ablation,
    select_sample,
)


def _chunk(index: int, kind: str, bloom: str) -> dict:
    return {
        "id": f"chunk-{index:03d}",
        "chunk_type": kind,
        "bloom_level": bloom,
        "learning_outcome_refs": [f"co-{index:03d}"],
        "correct_answer": f"The supported rule for concept {index}.",
        "text": (
            f"Instructional source text {index} explains concept {index} and "
            "a concrete relationship because learners can compare the correct "
            "rule with a common misconception in an example."
        ),
    }


def test_select_sample_is_fixed_seed_tail_only_and_covers_sparse_strata() -> None:
    chunks = [_chunk(i, "example", "apply") for i in range(20)]
    for i in range(20, 28):
        chunks.append(_chunk(i, "assessment_item", "analyze"))
    for i in range(28, 36):
        chunks.append(_chunk(i, "explanation", "evaluate"))
    for i in range(36, 40):
        chunks.append(_chunk(i, "assessment_item", "create"))

    selected_a, manifest_a = select_sample(
        chunks, seed=19, per_stratum=6, tail_fraction=0.5
    )
    selected_b, manifest_b = select_sample(
        chunks, seed=19, per_stratum=6, tail_fraction=0.5
    )

    assert selected_a == selected_b
    assert manifest_a == manifest_b
    assert all(int(row["id"].split("-")[1]) >= 20 for row in selected_a)
    assert manifest_a["strata"]["assessment_item"]["selected"] == 6
    assert manifest_a["strata"]["explanation"]["selected"] == 6
    assert manifest_a["strata"]["bloom_analyze"]["selected"] == 6
    assert manifest_a["strata"]["bloom_evaluate"]["selected"] == 6
    assert manifest_a["strata"]["bloom_create"]["selected"] == 4
    assert manifest_a["unique_selected_chunks"] < 28  # overlap is intentional


def test_prepare_workspace_is_isolated_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    chunks = [
        _chunk(i, "assessment_item" if i % 2 else "explanation", "analyze")
        for i in range(20)
    ]
    chunks_path = source / "imscc_chunks" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in chunks), encoding="utf-8"
    )
    (source / "objectives.json").write_text(json.dumps({
        "learning_outcomes": [
            {
                "id": f"co-{i:03d}",
                "statement": (
                    f"Analyze concept {i} and compare its correct rule with "
                    "a common misconception."
                ),
                "bloom_level": "analyze",
                "bloom_verb": "analyze",
            }
            for i in range(20)
        ],
    }) + "\n")
    workspace = tmp_path / "pilot"

    manifest = prepare_workspace(
        source_course=source,
        workspace=workspace,
        seed=7,
        per_stratum=2,
        tail_fraction=0.5,
    )

    assert (workspace / "course" / "imscc_chunks" / "chunks.jsonl").is_file()
    assert (workspace / "course" / "objectives.json").read_text() == (
        source / "objectives.json"
    ).read_text()
    assert manifest["source_chunk_count"] == 20
    assert not (source / "training_specs").exists()
    with pytest.raises(FileExistsError):
        prepare_workspace(
            source_course=source,
            workspace=workspace,
            seed=7,
            per_stratum=2,
            tail_fraction=0.5,
        )


def test_compare_requires_identical_sample_and_calculates_deltas() -> None:
    before = {
        "sample_manifest_sha256": "same",
        "label": "before",
        "model_calls": 20,
        "retry_counts": {"leakage_retries": 8},
        "truncated_responses": 2,
        "accepted": {"sft": 3, "dpo": 4},
        "rejected": {"total": 13},
        "template_diversity": {"distinct_templates": 4},
    }
    after = {
        "sample_manifest_sha256": "same",
        "label": "after",
        "model_calls": 12,
        "retry_counts": {"leakage_retries": 1},
        "truncated_responses": 0,
        "accepted": {"sft": 5, "dpo": 5},
        "rejected": {"total": 2},
        "template_diversity": {"distinct_templates": 7},
    }

    comparison = compare_reports(before, after)

    assert comparison["metrics"]["model_calls"]["delta"] == -8
    assert comparison["metrics"]["accepted_sft"]["delta"] == 2
    assert comparison["metrics"]["rejected_total"]["delta"] == -11

    after["sample_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="same sample"):
        compare_reports(before, after)


def test_isolated_runtime_environment_restores_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ED4ALL_RUN_ID", "production-run")
    monkeypatch.setenv("ED4ALL_SEAT_SCHEDULE", "true")

    with isolated_runtime_environment(tmp_path):
        assert "ED4ALL_RUN_ID" not in os.environ
        assert os.environ["ED4ALL_SEAT_SCHEDULE"] == "false"
        assert os.environ["ED4ALL_LIBV2_ROOT"].startswith(str(tmp_path))

    assert os.environ["ED4ALL_RUN_ID"] == "production-run"
    assert os.environ["ED4ALL_SEAT_SCHEDULE"] == "true"


def test_terminal_rejection_replay_ablation_is_scoped() -> None:
    from Trainforge import synthesize_training

    fingerprint = "a" * 64
    record = {
        "disposition": "rejected",
        "contract_fingerprint": fingerprint,
    }
    assert synthesize_training._checkpoint_rejection_matches_contract(
        record, fingerprint
    )
    with rejection_efficiency_ablation(
        "terminal-rejection-replay-disabled"
    ):
        assert not synthesize_training._checkpoint_rejection_matches_contract(
            record, fingerprint
        )

    assert synthesize_training._checkpoint_rejection_matches_contract(
        record, fingerprint
    )


def test_modeled_old_leakage_policy_budget_is_scoped() -> None:
    from Trainforge.generators import _synthesis_provider
    from Trainforge.generators._synthesis_common import SynthesisProviderError

    current = _synthesis_provider.MAX_LEAKAGE_REWRITE_RETRIES
    assert current == 2
    provider_class = _synthesis_provider.SynthesisProvider
    original = provider_class.paraphrase_instruction

    def exhausted(*_args, **_kwargs):
        raise SynthesisProviderError(
            "leakage exhausted", code="provider_output_verbatim_leakage"
        )

    provider_class.paraphrase_instruction = exhausted
    try:
        with rejection_efficiency_ablation("modeled-old-leakage-policy"):
            assert _synthesis_provider.MAX_LEAKAGE_REWRITE_RETRIES == 9
            result = provider_class.paraphrase_instruction(
                object(),
                {"prompt": "draft", "completion": "answer"},
                {"text": "source"},
            )
            assert result["prompt"] == "draft"
            assert result["paraphrase_fallback_reason"] == (
                "paraphrase_invalid_after_retry"
            )
    finally:
        provider_class.paraphrase_instruction = original

    assert _synthesis_provider.MAX_LEAKAGE_REWRITE_RETRIES == current


def test_fallback_provenance_requires_explicit_emitted_reason() -> None:
    pairs = [
        {"prompt": "normal stochastic provider success", "ablation": "modeled-old"},
        {
            "prompt": "explicit deterministic fallback",
            "paraphrase_fallback_reason": "paraphrase_invalid_after_retry",
        },
    ]

    summary = fallback_provenance_summary(pairs)

    assert summary == {
        "explicit_fallback_pairs": 1,
        "reasons": {"paraphrase_invalid_after_retry": 1},
    }
