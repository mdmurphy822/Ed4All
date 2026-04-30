"""2026-04-30 — SLM eval-harness smoke-mode regression tests.

Exercises the ``--smoke`` flag end-to-end against a synthetic LibV2
course tree. We never actually load a model — every test substitutes
a mock callable for ``AdapterCallable`` (or constructs the harness
directly), so the suite stays CPU-only and fast.

The properties under test:

1. ``--smoke`` writes ``smoke_eval_report.json`` (NOT
   ``eval_report.json``).
2. The emitted report carries ``smoke_mode: true`` at the top level.
3. ``--smoke --with-ablation`` does NOT emit ``ablation_report.json``
   (smoke forces ablation off regardless of operator intent).
4. Smoke mode caps every gated evaluator at N=3 prompts.
5. ``--smoke --stub`` aborts via ``parser.error`` (mutually exclusive
   modes — the combination is meaningless).
6. ``EvalGatingValidator`` refuses to gate a report carrying
   ``smoke_mode: true`` (defensive against operator-renamed reports).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.slm_eval_harness import SLMEvalHarness, main as harness_main  # noqa: E402


def _build_course(tmp_path: Path) -> Path:
    """Reproduces the rdf_shacl-shaped synthetic course from
    test_eval_harness_integration.py with extra graph edges so the
    holdout split has enough probes to demonstrate the N=3 cap.
    """
    course = tmp_path / "tst-smoke"
    (course / "graph").mkdir(parents=True)
    (course / "corpus").mkdir(parents=True)

    (course / "manifest.json").write_text(
        json.dumps({"classification": {
            "subdomains": ["semantic web"],
            "topics": ["rdf and shacl"],
        }}),
        encoding="utf-8",
    )

    nodes = [
        {"id": "bloom:remember", "class": "BloomLevel", "level": "remember"},
        {"id": "bloom:apply", "class": "BloomLevel", "level": "apply"},
        {
            "id": "mc_001", "class": "Misconception",
            "label": "wrong idea",
            "statement": "This is a misconception that should be rejected.",
        },
        {"id": "concept_x", "class": "Concept", "label": "Concept X"},
        {"id": "concept_y", "class": "Concept", "label": "Concept Y"},
        {"id": "concept_z", "class": "Concept", "label": "Concept Z"},
    ]
    # Provide >3 edges of each interesting type so smoke caps demonstrably bite.
    edges = [
        {"source": "concept_x", "target": "concept_y", "relation_type": "prerequisite_of"},
        {"source": "concept_y", "target": "concept_z", "relation_type": "prerequisite_of"},
        {"source": "concept_x", "target": "concept_z", "relation_type": "prerequisite_of"},
        {"source": "concept_y", "target": "concept_x", "relation_type": "related_to"},
        {"source": "chunk_01", "target": "bloom:remember", "relation_type": "at_bloom_level"},
        {"source": "chunk_02", "target": "bloom:apply", "relation_type": "at_bloom_level"},
        {"source": "chunk_03", "target": "bloom:apply", "relation_type": "at_bloom_level"},
        {"source": "chunk_04", "target": "bloom:remember", "relation_type": "at_bloom_level"},
        {"source": "mc_001", "target": "concept_x", "relation_type": "interferes_with"},
    ]
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}),
        encoding="utf-8",
    )

    chunks = [
        {
            "id": f"c_00{i}",
            "key_terms": [
                {"term": f"concept {chr(120 + i)}", "definition": "Definition stub for testing."},
            ],
            "misconceptions": [
                {
                    "misconception": "This is a misconception that should be rejected.",
                    "correction": "The actual correct understanding is rigorous.",
                },
            ],
        }
        for i in range(1, 5)
    ]
    with (course / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    return course


def _mock_model(prompt: str) -> str:
    return (
        "yes, that holds. confidence: 80%. "
        "First, define the concept; rather than relying on rows, "
        "we have first-class facts. Actually false - misconception."
    )


# ---------------------------------------------------------------------------
# CLI-level tests (cover --smoke flag plumbing in main())
# ---------------------------------------------------------------------------


def _stub_adapter_module(monkeypatch):
    """Patch AdapterCallable + RAGCallable + base-model registry so
    main() can run end-to-end without torch / transformers / a real
    adapter directory.
    """
    import Trainforge.eval.slm_eval_harness as harness_mod  # noqa: F401

    fake_callable = _mock_model

    class _FakeRegistry:
        @staticmethod
        def resolve(name):
            class _Spec:
                huggingface_repo = "Qwen/Qwen2.5-1.5B"
                default_revision = "main"
            return _Spec()

    class _FakeAdapter:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt: str) -> str:
            return fake_callable(prompt)

    def _fake_load_eval_config(course_path):
        class _Loaded:
            config = {
                "max_new_tokens": 16,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
            }
        return _Loaded()

    # Patch the late imports inside main().
    monkeypatch.setattr(
        "Trainforge.eval.adapter_callable.AdapterCallable",
        _FakeAdapter,
        raising=True,
    )
    monkeypatch.setattr(
        "Trainforge.eval.eval_config.load_eval_config",
        _fake_load_eval_config,
        raising=True,
    )
    monkeypatch.setattr(
        "Trainforge.training.base_models.BaseModelRegistry",
        _FakeRegistry,
        raising=True,
    )


def test_smoke_flag_writes_smoke_eval_report_not_eval_report(tmp_path, monkeypatch):
    course = _build_course(tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    _stub_adapter_module(monkeypatch)

    argv = [
        "slm_eval_harness",
        "--course-path", str(course),
        "--adapter-path", str(adapter_dir),
        "--base-model", "qwen2.5-1.5b",
        "--smoke",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    harness_main()

    smoke_report = adapter_dir / "eval" / "smoke_eval_report.json"
    eval_report = adapter_dir / "eval" / "eval_report.json"
    assert smoke_report.exists(), "smoke_eval_report.json was not written"
    assert not eval_report.exists(), (
        "eval_report.json should NOT be created in smoke mode -- it would "
        "be mistaken for / overwrite a real promotion-gated report."
    )


def test_smoke_mode_field_set_to_true_in_report(tmp_path, monkeypatch):
    course = _build_course(tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    _stub_adapter_module(monkeypatch)

    argv = [
        "slm_eval_harness",
        "--course-path", str(course),
        "--adapter-path", str(adapter_dir),
        "--base-model", "qwen2.5-1.5b",
        "--smoke",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    harness_main()

    smoke_report = adapter_dir / "eval" / "smoke_eval_report.json"
    payload = json.loads(smoke_report.read_text(encoding="utf-8"))
    assert payload.get("smoke_mode") is True, (
        "Report must carry smoke_mode: true so EvalGatingValidator + "
        "hf_model_index can short-circuit on a renamed smoke report."
    )


def test_smoke_mode_forces_with_ablation_off(tmp_path, monkeypatch):
    course = _build_course(tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    _stub_adapter_module(monkeypatch)

    argv = [
        "slm_eval_harness",
        "--course-path", str(course),
        "--adapter-path", str(adapter_dir),
        "--base-model", "qwen2.5-1.5b",
        "--smoke",
        "--with-ablation",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    harness_main()

    ablation_report = adapter_dir / "eval" / "ablation_report.json"
    assert not ablation_report.exists(), (
        "Smoke mode must override --with-ablation so the second model "
        "load + 4-setup loop never runs."
    )


def test_smoke_mode_caps_to_three_prompts(tmp_path):
    """Verify the harness caps every gated evaluator at N=3 in smoke mode.

    Approach: count model calls and assert each major stage
    (faithfulness, negative_grounding, source_match) saw at most 3 +
    a small buffer for non-capped invariants. The harness's own
    eval_progress.jsonl emits stage_start / stage_end events that pin
    per-stage call counts.
    """
    course = _build_course(tmp_path)

    call_count = {"n": 0}

    def _counting_model(prompt: str) -> str:
        call_count["n"] += 1
        return _mock_model(prompt)

    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_counting_model,
        max_holdout_questions=3,
        smoke_mode=True,
    )
    report_path = harness.run_all()
    progress_path = report_path.parent / "eval_progress.jsonl"
    records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Per-stage call counts: every stage_end event records
    # `stage_calls` since stage_start. Capped evaluators must be <= 3.
    capped_stages = {"faithfulness", "negative_grounding", "source_match"}
    for r in records:
        if r.get("event") != "stage_end":
            continue
        if r.get("stage") in capped_stages:
            assert r["stage_calls"] <= 3, (
                f"Stage {r['stage']} ran {r['stage_calls']} model calls in "
                f"smoke mode; expected <= 3."
            )


def test_smoke_and_stub_are_mutually_exclusive(tmp_path, monkeypatch, capsys):
    course = _build_course(tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    argv = [
        "slm_eval_harness",
        "--course-path", str(course),
        "--adapter-path", str(adapter_dir),
        "--base-model", "qwen2.5-1.5b",
        "--smoke",
        "--stub",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        harness_main()
    # argparse.error -> SystemExit(2)
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in (captured.err + captured.out).lower()


# ---------------------------------------------------------------------------
# EvalGatingValidator integration: smoke reports must NEVER gate.
# ---------------------------------------------------------------------------


def test_eval_gating_validator_rejects_smoke_mode_report(tmp_path):
    from lib.validators.eval_gating import EvalGatingValidator

    model_dir = tmp_path / "model"
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True)

    # Write a report whose metric values would otherwise pass every
    # threshold -- the only critical-fail signal must be smoke_mode.
    report = {
        "faithfulness": 0.95,
        "coverage": 0.95,
        "source_match": 0.85,
        "baseline_delta": 0.10,
        "negative_grounding_accuracy": 0.90,
        "yes_rate": 0.50,
        "metrics": {"hallucination_rate": 0.05},
        "calibration_ece": 0.05,
        "smoke_mode": True,
    }
    (eval_dir / "eval_report.json").write_text(
        json.dumps(report), encoding="utf-8",
    )

    validator = EvalGatingValidator()
    result = validator.validate({"model_dir": str(model_dir)})

    assert not result.passed, (
        "Validator must refuse to gate a report carrying smoke_mode=true."
    )
    codes = [i.code for i in result.issues]
    assert "EVAL_REPORT_IS_SMOKE" in codes, (
        f"Expected EVAL_REPORT_IS_SMOKE issue, got {codes}"
    )
    smoke_issue = next(i for i in result.issues if i.code == "EVAL_REPORT_IS_SMOKE")
    assert "smoke_mode" in smoke_issue.message.lower()
    assert smoke_issue.severity == "critical"
