"""``ed4all run trainforge_train`` must actually train — dispatch + carve-outs.

Before this wiring the ``training`` phase declared the ``training-synthesizer``
agent, whose ``AGENT_TOOL_MAPPING`` entry is ``synthesize_training`` — the
instruction/preference PAIR-SYNTHESIS tool. The workflow re-synthesized training
pairs, stamped the phase complete, and no trainer was ever called.

The tests pin, in order of how much a regression costs:

1. ``GracefulStopRequested`` propagates OUT of both new tools. It subclasses
   ``RuntimeError``, so a broad ``except Exception`` converts a PAUSE into an
   error envelope; the phase then runs its gates and stamps ``completed``, and a
   ``--resume`` re-enters it as if done. That is the bug class this wave fixed
   in ``_synthesize_training`` and it cost a real run.
2. An unknown base model fails LOUD (the registry's supported-name list) rather
   than substituting a model the operator never selected.
3. The registry keys + the phase-name routing exist at all.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.executor import (  # noqa: E402
    _DETERMINISTIC_TRAINING_TOOLS,
    _PHASE_TOOL_MAPPING,
)
from MCP.core.tool_schemas import TOOL_SCHEMAS  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return _build_tool_registry()


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Wiring                                                                       #
# --------------------------------------------------------------------------- #


class TestWiring:
    @pytest.mark.parametrize("tool_name", ["run_training", "run_evaluation"])
    def test_registry_key_exists(self, registry, tool_name):
        assert tool_name in registry, (
            f"{tool_name} missing from _build_tool_registry(); the phase would "
            f"fail with 'Tool not registered'."
        )

    @pytest.mark.parametrize("tool_name", ["run_training", "run_evaluation"])
    def test_tool_schema_exists(self, tool_name):
        # Registry ⊆ TOOL_SCHEMAS: without an entry the param mapper raises
        # ParameterMappingError before the executor reaches the registry.
        assert tool_name in TOOL_SCHEMAS

    @pytest.mark.parametrize(
        "phase_name,tool_name",
        [("training", "run_training"), ("evaluation", "run_evaluation")],
    )
    def test_phase_routes_by_name(self, phase_name, tool_name):
        assert _PHASE_TOOL_MAPPING.get(phase_name) == tool_name

    def test_training_phase_no_longer_routes_to_pair_synthesis(self):
        # The precise defect: phase-name routing must WIN over the agent
        # mapping, which still (correctly) points training-synthesizer at the
        # pair synthesizer for textbook_to_course's training_synthesis phase.
        from MCP.core.executor import AGENT_TOOL_MAPPING

        assert AGENT_TOOL_MAPPING["training-synthesizer"] == "synthesize_training"
        assert _PHASE_TOOL_MAPPING["training"] != "synthesize_training"

    def test_post_training_validation_stays_validator_only(self):
        # It has no Python handler — its gates run through the gate manager.
        # A mapping here would synthesize a task pointing at nothing.
        assert "post_training_validation" not in _PHASE_TOOL_MAPPING

    def test_deterministic_tools_never_subagent_fork(self):
        assert _DETERMINISTIC_TRAINING_TOOLS == {"run_training", "run_evaluation"}


# --------------------------------------------------------------------------- #
# Base-model resolution                                                        #
# --------------------------------------------------------------------------- #


class TestBaseModelResolution:
    def test_unknown_base_model_fails_loud(self, registry, monkeypatch):
        """No silent substitution: the envelope must carry the registry's
        supported-name list so the operator sees the right spelling."""
        constructed = []

        import Trainforge.training as training_pkg

        class _ExplodingRunner:
            def __init__(self, **kwargs):
                constructed.append(kwargs)

        monkeypatch.setattr(
            training_pkg, "TrainingRunner", _ExplodingRunner, raising=True
        )

        out = json.loads(_run(registry["run_training"](
            course_name="FXTERTIARY_101",
            base_model="definitely-not-a-real-base",
        )))
        assert out["success"] is False
        assert out["error_type"] == "unknown_base_model"
        assert "Supported bases" in out["error"]
        assert not constructed, (
            "TrainingRunner was constructed for an unknown base model — the "
            "registry check must short-circuit BEFORE any runner is built."
        )

    def test_unknown_base_model_fails_loud_on_eval_rerun(
        self, registry, monkeypatch, tmp_path
    ):
        """The eval arm re-runs the harness when no report exists, so it needs
        a base too — and must fail the same loud way, not substitute."""
        (tmp_path / "courses" / "fxtertiary-101" / "models" / "m1").mkdir(parents=True)

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )
        out = json.loads(_run(registry["run_evaluation"](
            course_name="FXTERTIARY_101", base_model="nope-not-a-base",
        )))
        assert out["success"] is False
        assert out["error_type"] == "unknown_base_model"
        assert "Supported bases" in out["error"]

    def test_env_default_is_used_when_no_kwarg(self, registry, monkeypatch):
        monkeypatch.setenv("ED4ALL_CAMPAIGN_BASE_MODEL", "also-not-real")
        out = json.loads(_run(registry["run_training"](course_name="FXTERTIARY_101")))
        assert out["base_model"] == "also-not-real"
        assert out["error_type"] == "unknown_base_model"


# --------------------------------------------------------------------------- #
# Graceful stop — the highest-value invariant                                  #
# --------------------------------------------------------------------------- #


class TestGracefulStopPropagates:
    def test_run_training_propagates_graceful_stop(self, registry, monkeypatch):
        import Trainforge.training as training_pkg

        class _StoppingRunner:
            def __init__(self, **kwargs):
                pass

            def run(self):
                raise GracefulStopRequested("training", 3)

        monkeypatch.setattr(
            training_pkg, "TrainingRunner", _StoppingRunner, raising=True
        )
        monkeypatch.setattr(
            training_pkg, "LocalBackend", lambda **kw: object(), raising=True
        )

        with pytest.raises(GracefulStopRequested):
            _run(registry["run_training"](
                course_name="FXTERTIARY_101",
                base_model="qwen2.5-1.5b",
            ))

    def test_run_evaluation_propagates_graceful_stop(
        self, registry, monkeypatch, tmp_path
    ):
        course_dir = tmp_path / "courses" / "fxtertiary-101"
        (course_dir / "models" / "m1").mkdir(parents=True)

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )

        # The held-out arm has no report to reuse, so it re-runs the harness —
        # which is where a mid-eval stop sentinel trips.
        import Trainforge.eval.slm_eval_harness as harness_mod

        class _StoppingHarness:
            def __init__(self, **kwargs):
                pass

            def run_all(self, output_path=None):
                raise GracefulStopRequested("eval_heldout", 1)

        monkeypatch.setattr(
            harness_mod, "SLMEvalHarness", _StoppingHarness, raising=True
        )
        monkeypatch.setattr(
            "Trainforge.eval.adapter_callable.AdapterCallable",
            lambda **kw: (lambda prompt: ""),
            raising=True,
        )
        monkeypatch.setattr(
            "Trainforge.eval.eval_config.load_eval_config",
            lambda course_path: type("_C", (), {"config": {}})(),
            raising=True,
        )

        with pytest.raises(GracefulStopRequested):
            _run(registry["run_evaluation"](
                course_name="FXTERTIARY_101",
                base_model="qwen2.5-1.5b",
            ))

    @pytest.mark.parametrize("tool_name", ["run_training", "run_evaluation"])
    def test_graceful_stop_is_a_runtime_error_subclass(self, tool_name):
        """Documents WHY the carve-out is load-bearing: a broad
        ``except Exception`` catches GracefulStopRequested."""
        assert issubclass(GracefulStopRequested, RuntimeError)


# --------------------------------------------------------------------------- #
# Evaluation envelope                                                          #
# --------------------------------------------------------------------------- #


class TestEvaluationEnvelope:
    def test_missing_course_fails_closed(self, registry, monkeypatch, tmp_path):
        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )
        out = json.loads(_run(registry["run_evaluation"](course_name="FXTERTIARY_101")))
        assert out["success"] is False
        assert out["error_type"] == "course_missing"

    def test_missing_adapter_fails_closed(self, registry, monkeypatch, tmp_path):
        (tmp_path / "courses" / "fxtertiary-101").mkdir(parents=True)

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )
        out = json.loads(_run(registry["run_evaluation"](course_name="FXTERTIARY_101")))
        assert out["success"] is False
        assert out["error_type"] == "model_dir_missing"

    def test_dry_run_holds_and_writes_nothing(
        self, registry, monkeypatch, tmp_path
    ):
        course_dir = tmp_path / "courses" / "fxtertiary-101"
        model_dir = course_dir / "models" / "m1"
        model_dir.mkdir(parents=True)

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )
        out = json.loads(_run(registry["run_evaluation"](
            course_name="FXTERTIARY_101", dry_run=True,
        )))
        assert out["verdict"] == "hold"
        assert not (model_dir / "eval" / "eval_report.json").exists(), (
            "a dry run must never write a report — an unscored verdict is a "
            "fabricated metric."
        )

    def test_degraded_arms_hold_and_never_promote(
        self, registry, monkeypatch, tmp_path
    ):
        """Both arms unavailable ⇒ HOLD, a distinct warning per arm, and no
        fabricated metric anywhere in the merged report."""
        course_dir = tmp_path / "courses" / "fxtertiary-101"
        model_dir = course_dir / "models" / "m1"
        (model_dir / "eval").mkdir(parents=True)
        # A real harness report so the held-out arm is "reused" and the gate
        # probe has something conclusive to read.
        (model_dir / "eval" / "eval_report.json").write_text(
            json.dumps({"faithfulness": 0.9, "coverage": 0.8}), encoding="utf-8"
        )

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )

        import lib.retrieval.grounded_eval as ge

        def _unavailable(*a, **kw):
            raise NotImplementedError("answer pipeline not importable")

        monkeypatch.setattr(ge, "run_grounded_eval", _unavailable, raising=True)

        out = json.loads(_run(registry["run_evaluation"](course_name="FXTERTIARY_101")))
        assert out["success"] is True
        assert out["verdict"] == "hold", (
            "a missing arm can never promote — the owner's contract is ONE "
            "verdict over BOTH arms."
        )
        assert "EVAL_GROUNDED_PIPELINE_UNAVAILABLE" in out["warnings"]
        assert out["degraded"] is True

        merged = json.loads(
            (model_dir / "eval" / "eval_report.json").read_text(encoding="utf-8")
        )
        # Additive merge: the harness-owned keys survive untouched.
        assert merged["faithfulness"] == 0.9
        assert merged["coverage"] == 0.8
        assert merged["evaluation"]["verdict"] == "hold"
        assert "grounded_answer" not in merged, (
            "no grounded headline was produced, so none may appear."
        )

    def test_both_arms_clean_promotes(self, registry, monkeypatch, tmp_path):
        course_dir = tmp_path / "courses" / "fxtertiary-101"
        model_dir = course_dir / "models" / "m1"
        (model_dir / "eval").mkdir(parents=True)
        (model_dir / "eval" / "eval_report.json").write_text(
            json.dumps({
                "faithfulness": 0.9,
                "coverage": 0.8,
                "source_match": 0.7,
                "negative_grounding_accuracy": 0.9,
                "yes_rate": 0.4,
                "baseline_delta": 0.1,
            }),
            encoding="utf-8",
        )

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )

        import lib.retrieval.grounded_eval as ge

        clean_headline = {
            "answer_rate": 1.0,
            "citation_resolution_rate": 1.0,
            "citation_precision": 0.9,
            "groundedness_rate_mean": 0.9,
            "unsupported_claim_rate": 0.0,
            "refusal": {"refusal_recall": 0.9, "refusal_precision": 1.0},
        }
        monkeypatch.setattr(
            ge,
            "run_grounded_eval",
            lambda *a, **kw: {
                "headline": clean_headline,
                "_written": {"report_path": "/tmp/grounded.json"},
            },
            raising=True,
        )

        out = json.loads(_run(registry["run_evaluation"](course_name="FXTERTIARY_101")))
        assert out["verdict"] == "promote", out["verdict_reasons"]
        merged = json.loads(
            (model_dir / "eval" / "eval_report.json").read_text(encoding="utf-8")
        )
        assert merged["grounded_answer"]["answer_rate"] == 1.0
        assert merged["faithfulness"] == 0.9

    def test_heldout_threshold_breach_rejects(
        self, registry, monkeypatch, tmp_path
    ):
        course_dir = tmp_path / "courses" / "fxtertiary-101"
        model_dir = course_dir / "models" / "m1"
        (model_dir / "eval").mkdir(parents=True)
        # yes_rate above the gate's 0.85 ceiling: a CONCLUSIVE held-out failure.
        (model_dir / "eval" / "eval_report.json").write_text(
            json.dumps({"faithfulness": 0.9, "coverage": 0.8, "yes_rate": 0.99}),
            encoding="utf-8",
        )

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )

        import lib.retrieval.grounded_eval as ge

        monkeypatch.setattr(
            ge,
            "run_grounded_eval",
            lambda *a, **kw: {"headline": {}, "_written": {}},
            raising=True,
        )

        out = json.loads(_run(registry["run_evaluation"](course_name="FXTERTIARY_101")))
        assert out["verdict"] == "reject"
        assert "EVAL_YES_BIAS_DETECTED" in out["verdict_reasons"]

    def test_smoke_report_holds_rather_than_rejects(
        self, registry, monkeypatch, tmp_path
    ):
        """A smoke report is "could not judge", not "the adapter is bad"."""
        course_dir = tmp_path / "courses" / "fxtertiary-101"
        model_dir = course_dir / "models" / "m1"
        (model_dir / "eval").mkdir(parents=True)
        (model_dir / "eval" / "eval_report.json").write_text(
            json.dumps({"faithfulness": 0.9, "coverage": 0.8, "smoke_mode": True}),
            encoding="utf-8",
        )

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(
            pt, "_resolve_libv2_root", lambda explicit=None: tmp_path
        )

        import lib.retrieval.grounded_eval as ge

        monkeypatch.setattr(
            ge,
            "run_grounded_eval",
            lambda *a, **kw: {"headline": {}, "_written": {}},
            raising=True,
        )

        out = json.loads(_run(registry["run_evaluation"](course_name="FXTERTIARY_101")))
        assert out["verdict"] == "hold"
        assert "EVAL_REPORT_IS_SMOKE" in out["verdict_reasons"]
