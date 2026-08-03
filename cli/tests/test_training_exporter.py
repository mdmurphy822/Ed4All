"""Decision-capture compatibility tests for the training exporter."""

from cli.exporters.training_exporter import TrainingExporter


def _event(alternative):
    return {
        "decision_type": "assessment_planning",
        "decision": "Use the selected assessment plan.",
        "rationale": "The selected plan balances coverage and learner time.",
        "alternatives_considered": [alternative],
        "event_id": "event-1",
    }


def test_dpo_exporter_reads_canonical_rejection_reason(tmp_path):
    """DPO output includes the canonical alternative rejection reason."""
    exporter = TrainingExporter("run-1", runs_root=tmp_path)

    pairs = exporter._create_dpo_pairs(_event({
        "option": "shorter_plan",
        "reason_rejected": "It would leave required objectives unassessed.",
    }))

    assert len(pairs) == 1
    assert "leave required objectives unassessed" in pairs[0]["rejected"]


def test_dpo_exporter_reads_legacy_rejection_reason(tmp_path):
    """DPO output remains usable for captures written with the legacy key."""
    exporter = TrainingExporter("run-1", runs_root=tmp_path)

    pairs = exporter._create_dpo_pairs(_event({
        "option": "shorter_plan",
        "rejected_because": "Legacy captures still need export support.",
    }))

    assert len(pairs) == 1
    assert "Legacy captures still need export support" in pairs[0]["rejected"]
