"""Wave 102 - AblationRunner tests.

Mocks the harness factory + four model callables so no real eval
runs. Asserts:

* Both tables (headline 4-row + retrieval-method 5-row) appear in the
  emitted ablation_report.json with the expected shape.
* qualitative_score is None on every row when judge=none and a number
  on every row when an anthropic-shaped judge is wired up.
* Mean retrieval latency lands in the retrieval-method table when the
  factory yields callables exposing ``mean_latency_ms``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------- #
# Fakes                                                                   #
# ---------------------------------------------------------------------- #


def _build_fake_harness_factory(metric_table: Dict[str, Dict[str, float]]):
    """Yield a factory that fakes per-setup metric output.

    ``metric_table`` is keyed by a probe-string we shove into the
    callable on construction; the factory inspects the captured key
    and writes the corresponding metrics into the scratch report.
    """
    class _FakeHarness:
        def __init__(self, course_path: Path, model_callable: Callable, key: str):
            self.course_path = course_path
            self.model_callable = model_callable
            self.key = key

        def run_all(self, output_path: Path) -> Path:
            payload = metric_table[self.key]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            return output_path

    def factory(course_path, model_callable):
        # Read the marker the runner stamps on the callable so we know
        # which setup we're in.
        key = getattr(model_callable, "_ablation_key", "default")
        return _FakeHarness(course_path, model_callable, key)

    return factory


def _tagged_callable(key: str, *, mean_latency_ms: float = None):
    """Return a callable carrying an ``_ablation_key`` marker."""
    def _call(prompt: str) -> str:
        return f"reply for {prompt}"
    _call._ablation_key = key  # type: ignore[attr-defined]
    if mean_latency_ms is not None:
        _call.mean_latency_ms = mean_latency_ms  # type: ignore[attr-defined]
    return _call


def _metric_payload(
    coverage: float, faithfulness: float, source_match: float,
) -> Dict[str, Any]:
    return {
        "faithfulness": faithfulness,
        "coverage": coverage,
        "source_match": source_match,
        "metrics": {
            "hallucination_rate": round(1.0 - faithfulness, 4),
            "source_match": source_match,
        },
        "profile": "rdf_shacl",
    }


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #


def test_headline_table_emits_four_rows(tmp_path):
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    metrics = {
        "base":         _metric_payload(0.40, 0.50, 0.10),
        "base+rag":     _metric_payload(0.55, 0.65, 0.40),
        "adapter":      _metric_payload(0.65, 0.70, 0.20),
        "adapter+rag":  _metric_payload(0.80, 0.85, 0.55),
    }
    factory = _build_fake_harness_factory(metrics)
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
    out = runner.run()
    payload = json.loads(out.read_text(encoding="utf-8"))

    rows = payload["headline_table"]
    assert len(rows) == 4
    assert [r["setup"] for r in rows] == [
        "base", "base+rag", "adapter", "adapter+rag",
    ]
    # Every row gets the four core metric columns + qualitative=None
    for row in rows:
        assert "accuracy" in row
        assert "faithfulness" in row
        assert "hallucination_rate" in row
        assert "source_match" in row
        assert "qualitative_score" in row
    # Adapter+RAG should be the strongest accuracy
    assert rows[3]["accuracy"] == pytest.approx(0.80)


def test_qualitative_column_omitted_when_judge_none(tmp_path):
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    metrics = {"base": _metric_payload(0.4, 0.5, 0.1)}
    factory = _build_fake_harness_factory(metrics)
    runner = AblationRunner(
        course_path=tmp_path,
        setups=[
            AblationSetup(
                setup="base",
                callable=_tagged_callable("base"),
                qualitative_judge=None,
            ),
        ],
        harness_factory=factory,
    )
    out = runner.run()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["headline_table"][0]["qualitative_score"] is None


def test_qualitative_column_populated_when_judge_enabled(tmp_path):
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    class _FakeJudge:
        enabled = True
        def score(self, prompt, model_output, ground_truth):
            return 4.5

    metrics = {"base": _metric_payload(0.4, 0.5, 0.1)}
    factory = _build_fake_harness_factory(metrics)
    runner = AblationRunner(
        course_path=tmp_path,
        setups=[
            AblationSetup(
                setup="base",
                callable=_tagged_callable("base"),
                qualitative_judge=_FakeJudge(),
            ),
        ],
        harness_factory=factory,
    )
    out = runner.run()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["headline_table"][0]["qualitative_score"] == 4.5


def test_retrieval_method_table_emits_five_rows(tmp_path):
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    # All five methods share metrics for simplicity; latency varies.
    method_payload = _metric_payload(0.7, 0.8, 0.4)
    metrics = {f"method:{m}": method_payload for m in (
        "bm25", "bm25+intent", "bm25+graph", "bm25+tag", "hybrid",
    )}
    factory = _build_fake_harness_factory(metrics)

    def method_factory(method: str):
        cb = _tagged_callable(f"method:{method}", mean_latency_ms=12.5 + len(method))
        return cb

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
    # The default-tagged callable also needs a metric entry for the
    # headline pass.
    metrics["default"] = method_payload
    out = runner.run()
    payload = json.loads(out.read_text(encoding="utf-8"))

    rm = payload["retrieval_method_table"]
    assert len(rm) == 5
    assert [r["method"] for r in rm] == [
        "bm25", "bm25+intent", "bm25+graph", "bm25+tag", "hybrid",
    ]
    for row in rm:
        assert row["mean_latency_ms"] is not None
        assert row["accuracy"] == pytest.approx(0.7)


def test_retrieval_method_table_skipped_when_factory_missing(tmp_path):
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    metrics = {"base": _metric_payload(0.4, 0.5, 0.1)}
    factory = _build_fake_harness_factory(metrics)
    runner = AblationRunner(
        course_path=tmp_path,
        setups=[AblationSetup(setup="base", callable=_tagged_callable("base"))],
        harness_factory=factory,
    )
    out = runner.run()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["retrieval_method_table"] == []


def test_headline_delta_in_ablation_and_eval_report(tmp_path):
    """Both ablation_report.json and the consolidated eval_report.json
    carry the headline_delta block so the HF README writer + any
    downstream consumer (audit, dashboard) see the procurement claim
    without a second JSON load."""
    from Trainforge.eval.ablation_runner import AblationRunner, AblationSetup

    # base hallucination 0.50 -> adapter+rag hallucination 0.12 = 76% reduction
    metrics = {
        "base":         _metric_payload(0.40, 0.50, 0.10),
        "base+rag":     _metric_payload(0.55, 0.65, 0.40),
        "adapter":      _metric_payload(0.65, 0.70, 0.20),
        "adapter+rag":  _metric_payload(0.85, 0.88, 0.60),
    }
    factory = _build_fake_harness_factory(metrics)
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
    abl_path = runner.run()
    eval_path = abl_path.parent / "eval_report.json"

    abl_payload = json.loads(abl_path.read_text(encoding="utf-8"))
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))

    for payload in (abl_payload, eval_payload):
        delta = payload.get("headline_delta")
        assert delta is not None, "headline_delta missing from report"
        # 1.0 - 0.88 = 0.12 hallucination on adapter+rag, 1.0 - 0.50 = 0.50 base
        # reduction = (0.50 - 0.12) / 0.50 = 0.76
        assert delta["hallucination_reduction_pct"] == pytest.approx(0.76, abs=1e-3)
        assert "headline_sentence" in delta
        assert "ED4ALL-Bench v1.0" in delta["headline_sentence"]
