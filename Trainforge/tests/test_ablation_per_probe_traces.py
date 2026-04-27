"""Wave 104 - per-probe trace emission + consolidated eval_report.json.

The Wave 103 ablation runner emitted only one synthetic trace row per
setup (the aggregate fallback). After Wave 104, when the harness's
eval_report carries a `per_question` array, the runner emits one
EvidenceTrace per probe per setup, plus a consolidated
`eval_report.json` next to `ablation_report.json`.

Tests:
1. Per-probe traces - one row per probe per setup, with non-empty
   model_output and probe-derived ground_truth_chunk_id.
2. eval_report.json emission - the runner writes a top-level
   eval_report.json carrying the canonical EvalReport shape for the
   adapter+rag setup, with per_setup metadata for the rest.
3. Latency surfacing - when the harness persists mean_latency_ms,
   the retrieval-method table picks it up even when the callable
   itself doesn't expose the attribute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _per_question_payload(
    n: int = 5,
    *,
    coverage: float = 0.6,
    faithfulness: float = 0.7,
    source_match: float = 0.4,
    setup_label: str = "base",
    mean_latency_ms: float = None,
) -> Dict[str, Any]:
    """Build a fake eval_report payload carrying per_question records."""
    per_question = []
    for i in range(n):
        per_question.append({
            "probe": f"What is concept {i} for {setup_label}?",
            "response": (
                f"The answer about concept {i} cites [chunk_{i:04d}]."
                if i % 2 == 0 else
                f"Generic response without citation for probe {i}."
            ),
            "ground_truth_chunk_id": f"chunk_{i:04d}",
            "outcome": "pass" if i % 2 == 0 else "fail",
            "correct": i % 2 == 0,
        })
    payload: Dict[str, Any] = {
        "faithfulness": faithfulness,
        "coverage": coverage,
        "source_match": source_match,
        "metrics": {
            "hallucination_rate": round(1.0 - faithfulness, 4),
            "source_match": source_match,
        },
        "per_question": per_question,
        "per_tier": {
            "faithfulness": {"accuracy": faithfulness, "scored": n, "correct": n // 2},
        },
        "per_invariant": {},
        "profile": "rdf_shacl",
    }
    if mean_latency_ms is not None:
        payload["metrics"]["mean_latency_ms"] = mean_latency_ms
    return payload


def _build_factory(payloads_by_key: Dict[str, Dict[str, Any]]):
    class _FakeHarness:
        def __init__(self, course_path, model_callable, key):
            self.course_path = course_path
            self.model_callable = model_callable
            self.key = key

        def run_all(self, output_path: Path) -> Path:
            payload = payloads_by_key[self.key]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            return output_path

    def factory(course_path, model_callable):
        key = getattr(model_callable, "_ablation_key", "default")
        return _FakeHarness(course_path, model_callable, key)

    return factory


def _tagged_callable(key: str, *, mean_latency_ms: float = None):
    def _call(prompt: str) -> str:
        return f"reply for {prompt}"
    _call._ablation_key = key  # type: ignore[attr-defined]
    if mean_latency_ms is not None:
        _call.mean_latency_ms = mean_latency_ms  # type: ignore[attr-defined]
    return _call


def test_per_probe_traces_emitted_one_per_probe_per_setup(tmp_path):
    """One EvidenceTrace per probe per setup; not just one aggregate."""
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup
    from Trainforge.eval.evidence_trace import load_traces

    n = 5
    payloads = {
        "base":         _per_question_payload(n=n, setup_label="base"),
        "base+rag":     _per_question_payload(n=n, setup_label="base+rag"),
        "adapter":      _per_question_payload(n=n, setup_label="adapter"),
        "adapter+rag":  _per_question_payload(n=n, setup_label="adapter+rag"),
    }
    factory = _build_factory(payloads)
    setups = [
        AblationSetup(setup="base",        callable=_tagged_callable("base")),
        AblationSetup(setup="base+rag",    callable=_tagged_callable("base+rag")),
        AblationSetup(setup="adapter",     callable=_tagged_callable("adapter")),
        AblationSetup(setup="adapter+rag", callable=_tagged_callable("adapter+rag")),
    ]
    runner = AblationRunner(
        course_path=tmp_path,
        setups=setups,
        harness_factory=factory,
    )
    runner.run()

    traces = load_traces(tmp_path / "eval" / "eval_traces.jsonl")
    # Expect 5 probes x 4 setups = 20 traces.
    assert len(traces) == n * 4, (
        f"expected {n*4} per-probe traces, got {len(traces)}"
    )
    # Every trace carries non-empty model_output and a ground-truth.
    non_empty_outputs = [t for t in traces if t.model_output]
    assert len(non_empty_outputs) == n * 4
    assert all(
        t.ground_truth_chunk_id and t.ground_truth_chunk_id.startswith("chunk_")
        for t in traces
    )
    # Every setup label is represented.
    setup_set = {t.setup for t in traces}
    assert setup_set == {"base", "base+rag", "adapter", "adapter+rag"}


def test_consolidated_eval_report_json_written_next_to_ablation(tmp_path):
    """eval_report.json must be emitted next to ablation_report.json."""
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    payloads = {
        "base":         _per_question_payload(n=3, coverage=0.4, setup_label="base"),
        "base+rag":     _per_question_payload(n=3, coverage=0.5, setup_label="base+rag"),
        "adapter":      _per_question_payload(n=3, coverage=0.6, setup_label="adapter"),
        "adapter+rag":  _per_question_payload(n=3, coverage=0.7, setup_label="adapter+rag"),
    }
    factory = _build_factory(payloads)
    setups = [
        AblationSetup(setup=k, callable=_tagged_callable(k))
        for k in ("base", "base+rag", "adapter", "adapter+rag")
    ]
    runner = AblationRunner(
        course_path=tmp_path,
        setups=setups,
        harness_factory=factory,
    )
    out = runner.run()

    eval_report_path = out.parent / "eval_report.json"
    assert eval_report_path.exists(), "eval_report.json must be written"

    consolidated = json.loads(eval_report_path.read_text(encoding="utf-8"))
    # The runner picks adapter+rag when present.
    assert consolidated["selected_setup"] == "adapter+rag"
    # Per-setup payloads under per_setup carry the canonical metrics.
    assert set(consolidated["per_setup"].keys()) == {
        "base", "base+rag", "adapter", "adapter+rag",
    }
    assert (
        consolidated["per_setup"]["adapter+rag"]["coverage"] == pytest.approx(0.7)
    )
    # Top-level fields conform to the EvalReport shape.
    assert "faithfulness" in consolidated
    assert "coverage" in consolidated


def test_latency_picked_up_from_eval_report_when_callable_lacks_it(tmp_path):
    """Latency surfaces from harness output when the callable lacks it."""
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    headline_payload = _per_question_payload(n=2, setup_label="adapter+rag")
    method_payload = _per_question_payload(
        n=2, setup_label="method", mean_latency_ms=25.0,
    )
    payloads = {
        "default": headline_payload,
        **{f"method:{m}": method_payload for m in (
            "bm25", "bm25+intent", "bm25+graph", "bm25+tag", "hybrid",
        )},
    }
    factory = _build_factory(payloads)

    def method_factory(method: str):
        # Deliberately omit mean_latency_ms attribute on the callable.
        return _tagged_callable(f"method:{method}")

    runner = AblationRunner(
        course_path=tmp_path,
        setups=[
            AblationSetup(setup="adapter+rag", callable=_tagged_callable("default")),
        ],
        retrieval_method_setup=AblationSetup(
            setup="adapter+rag-method-sweep",
            callable=_tagged_callable("default"),
        ),
        retrieval_method_factory=method_factory,
        harness_factory=factory,
    )
    out = runner.run()
    payload = json.loads(out.read_text(encoding="utf-8"))
    rm = payload["retrieval_method_table"]
    assert len(rm) == 5
    for row in rm:
        # Latency comes from the harness payload's metrics.mean_latency_ms.
        assert row["mean_latency_ms"] == pytest.approx(25.0)


def test_aggregate_fallback_still_used_when_no_per_question(tmp_path):
    """Backwards compat: legacy fixtures emit one synthetic row per setup."""
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup
    from Trainforge.eval.evidence_trace import load_traces

    aggregate_only = {
        "faithfulness": 0.5,
        "coverage": 0.4,
        "source_match": 0.2,
        "metrics": {"hallucination_rate": 0.5, "source_match": 0.2},
        "profile": "rdf_shacl",
        "per_tier": {},
        "per_invariant": {},
    }
    payloads = {"base": aggregate_only}
    factory = _build_factory(payloads)
    setups = [AblationSetup(setup="base", callable=_tagged_callable("base"))]
    runner = AblationRunner(
        course_path=tmp_path,
        setups=setups,
        harness_factory=factory,
    )
    runner.run()

    traces = load_traces(tmp_path / "eval" / "eval_traces.jsonl")
    # Aggregate fallback emits one row per setup.
    assert len(traces) == 1
    assert traces[0].probe_id == "base:aggregate"
