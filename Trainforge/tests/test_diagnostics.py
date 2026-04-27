"""Wave 103 - tests for the diagnostic-finding auto-detector.

Three rules; each gets a triggering and a non-triggering case.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_report(*, base_acc, base_faith, adapter_acc, adapter_faith):
    return {
        "headline_table": [
            {"setup": "base", "accuracy": base_acc,
             "faithfulness": base_faith, "hallucination_rate": 1 - base_faith,
             "source_match": 0.1, "qualitative_score": None},
            {"setup": "base+rag", "accuracy": 0.55, "faithfulness": 0.65,
             "hallucination_rate": 0.35, "source_match": 0.4,
             "qualitative_score": None},
            {"setup": "adapter", "accuracy": adapter_acc,
             "faithfulness": adapter_faith,
             "hallucination_rate": 1 - adapter_faith, "source_match": 0.2,
             "qualitative_score": None},
            {"setup": "adapter+rag", "accuracy": 0.85, "faithfulness": 0.88,
             "hallucination_rate": 0.12, "source_match": 0.6,
             "qualitative_score": None},
        ],
        "retrieval_method_table": [],
    }


def _make_traces(*, n_total: int, n_empty: int) -> List:
    from Trainforge.eval.evidence_trace import EvidenceTrace

    traces = []
    for i in range(n_total - n_empty):
        traces.append(EvidenceTrace(
            probe_id=f"p{i}", setup="adapter+rag",
            retrieval_method="bm25", prompt="q",
            extracted_citations=["c1"],
        ))
    for i in range(n_empty):
        traces.append(EvidenceTrace(
            probe_id=f"empty{i}", setup="adapter+rag",
            retrieval_method="bm25", prompt="q",
            extracted_citations=[],
        ))
    return traces


def test_adapter_tone_only_triggers_when_accuracy_close_faithfulness_lifts():
    from Trainforge.eval.diagnostics import detect_findings

    report = _build_report(
        base_acc=0.50, base_faith=0.60,
        adapter_acc=0.52, adapter_faith=0.80,  # +0.02 acc, +0.20 faith
    )
    findings = detect_findings(report, traces=[])
    labels = [f["finding"] for f in findings]
    assert "adapter_tone_only" in labels
    # Rationale must be substantive
    tone = [f for f in findings if f["finding"] == "adapter_tone_only"][0]
    assert "tone" in tone["rationale"].lower() or "knowledge" in tone["rationale"].lower()


def test_adapter_tone_only_quiet_when_accuracy_actually_lifts():
    from Trainforge.eval.diagnostics import detect_findings

    report = _build_report(
        base_acc=0.50, base_faith=0.60,
        adapter_acc=0.70, adapter_faith=0.80,  # +0.20 acc, +0.20 faith
    )
    findings = detect_findings(report, traces=[])
    assert "adapter_tone_only" not in [f["finding"] for f in findings]


def test_prompting_failure_triggers_above_threshold():
    from Trainforge.eval.diagnostics import detect_findings

    report = _build_report(
        base_acc=0.40, base_faith=0.50,
        adapter_acc=0.65, adapter_faith=0.70,
    )
    # 5/10 = 50% empty citations; threshold is 30%
    traces = _make_traces(n_total=10, n_empty=5)
    findings = detect_findings(report, traces=traces)
    assert "prompting_failure" in [f["finding"] for f in findings]


def test_prompting_failure_quiet_below_threshold():
    from Trainforge.eval.diagnostics import detect_findings

    report = _build_report(
        base_acc=0.40, base_faith=0.50,
        adapter_acc=0.65, adapter_faith=0.70,
    )
    # 1/10 = 10% empty; threshold is 30%
    traces = _make_traces(n_total=10, n_empty=1)
    findings = detect_findings(report, traces=traces)
    assert "prompting_failure" not in [f["finding"] for f in findings]


def test_dataset_too_easy_triggers_when_base_accuracy_high():
    from Trainforge.eval.diagnostics import detect_findings

    report = _build_report(
        base_acc=0.85, base_faith=0.80,
        adapter_acc=0.88, adapter_faith=0.85,
    )
    findings = detect_findings(report, traces=[])
    assert "dataset_too_easy" in [f["finding"] for f in findings]


def test_dataset_too_easy_quiet_when_base_accuracy_low():
    from Trainforge.eval.diagnostics import detect_findings

    report = _build_report(
        base_acc=0.40, base_faith=0.50,
        adapter_acc=0.65, adapter_faith=0.70,
    )
    findings = detect_findings(report, traces=[])
    assert "dataset_too_easy" not in [f["finding"] for f in findings]


def test_no_rules_fire_on_clean_report():
    from Trainforge.eval.diagnostics import detect_findings

    # acc lift + low base accuracy + few empty citations
    report = _build_report(
        base_acc=0.40, base_faith=0.50,
        adapter_acc=0.65, adapter_faith=0.70,
    )
    traces = _make_traces(n_total=10, n_empty=1)
    findings = detect_findings(report, traces=traces)
    assert findings == []


def test_finding_has_required_shape():
    from Trainforge.eval.diagnostics import detect_findings

    report = _build_report(
        base_acc=0.85, base_faith=0.80,
        adapter_acc=0.88, adapter_faith=0.85,
    )
    findings = detect_findings(report, traces=[])
    for finding in findings:
        assert set(finding.keys()) >= {"finding", "severity", "rationale"}
        assert isinstance(finding["rationale"], str)
        assert len(finding["rationale"]) >= 20
