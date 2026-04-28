"""Wave 111 / Phase E — end-to-end smoke covering Phase A→D contracts.

Exercises:
  1. ClaudeSessionProvider with FakeLocalDispatcher (Phase A)
  2. yes_rate + negative_grounding_accuracy in eval_report (Phase B)
  3. per_property_accuracy in eval_report (Phase C)
  4. _BudgetTracker telemetry persistence (Phase D)
  5. EvalGatingValidator threshold checks (Phase B + C)

A regression in any layer's contract surfaces here as a single failing
assertion, with enough context to point at which layer broke.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import run_synthesis  # noqa: E402
from Trainforge.tests._synthesis_fakes import (  # noqa: E402
    FakeLocalDispatcher,
    make_instruction_response,
    make_preference_response,
)
from lib.validators.eval_gating import EvalGatingValidator  # noqa: E402


FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


def _make_working_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def test_full_chain_synthesis_to_eval_gating(tmp_path: Path) -> None:
    # Phase A: synthesis dispatches via fake LocalDispatcher and
    # records telemetry under the corpus.
    async def agent_tool(*, task_params, **_kw):
        kind = task_params["kind"]
        # Wave 112 Task 4: outputs must respect _validate_lengths floors
        # (PROMPT_MIN=40, COMPLETION_MIN=50). These fixtures are kept just
        # above the floor so a regression in the validator surfaces here
        # rather than silently passing through with poisoned shorts.
        if kind == "instruction":
            return make_instruction_response(
                prompt=(
                    "What does sh:datatype constrain about literal "
                    "values in SHACL shape definitions?"
                ),
                completion=(
                    "sh:datatype constrains literal types in SHACL "
                    "shapes, requiring values to match the named "
                    f"datatype IRI. [{task_params.get('chunk_id', 'chunk_001')}]"
                ),
            )
        return make_preference_response(
            prompt=(
                "Which statement about sh:datatype in SHACL is "
                "correct with respect to literal binding?"
            ),
            chosen=(
                "sh:datatype binds a literal type to the constrained "
                "value, requiring an XSD datatype IRI."
            ),
            rejected=(
                "sh:datatype binds a class to the constrained value, "
                "requiring instance graph membership in that class."
            ),
        )

    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    working = _make_working_copy(tmp_path)

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="claude_session",
        seed=11,
        dispatcher=dispatcher,
        max_dispatches=200,
    )

    # Phase A assertions: rows tagged claude_session, telemetry written.
    inst_path = working / "training_specs" / "instruction_pairs.jsonl"
    rows = [json.loads(l) for l in inst_path.read_text().splitlines() if l.strip()]
    assert rows, "no instruction pairs emitted"
    assert all(r["provider"] == "claude_session" for r in rows)

    # Phase D assertion: telemetry file present.
    telemetry = working / "training_specs" / ".synthesis_telemetry.jsonl"
    assert telemetry.exists()
    tel_rows = [json.loads(l) for l in telemetry.read_text().splitlines() if l.strip()]
    assert tel_rows, "telemetry file is empty"
    assert all("kind" in r and "cached" in r for r in tel_rows)

    # Phase D assertion: budget summary populated on stats.
    assert stats.dispatched_count + stats.cache_hits_count == len(tel_rows)
    # Phase E assertion: capped flag is False on a healthy run.
    assert stats.capped_at_max_dispatches is False

    # Construct a synthetic eval_report.json that the gating validator can
    # consume. (Running the full eval harness end-to-end requires a real
    # model; this test asserts the gating contract honors the new fields.)
    model_dir = tmp_path / "models" / "test-phase-e"
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True)
    eval_report = {
        "faithfulness": 0.80,
        "coverage": 0.85,
        "profile": "rdf_shacl",
        "per_tier": {},
        "per_invariant": {},
        "baseline_delta": 0.10,
        "source_match": 0.65,
        "negative_grounding_accuracy": 0.70,
        "yes_rate": 0.50,
        "metrics": {"hallucination_rate": 0.20},
        "per_property_accuracy": {
            "sh_datatype": 0.70,
            "sh_class": 0.65,
            "sh_nodeshape": 0.55,
            "sh_propertyshape": None,
            "rdfs_subclassof": 0.60,
            "owl_sameas": 0.50,
        },
    }
    (eval_dir / "eval_report.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8",
    )

    # Phase B + C assertions: gating validator passes when all signals
    # are above floors.
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is True, [
        (i.severity, i.code, i.message) for i in result.issues
    ]

    # Phase B assertion: flipping yes_rate above ceiling fails closed.
    eval_report["yes_rate"] = 0.95
    (eval_dir / "eval_report.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8",
    )
    result_yb = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result_yb.passed is False
    assert any(i.code == "EVAL_YES_BIAS_DETECTED" for i in result_yb.issues)

    # Phase C assertion: flipping a property below floor fails closed.
    eval_report["yes_rate"] = 0.50  # restore
    eval_report["per_property_accuracy"]["sh_class"] = 0.10
    (eval_dir / "eval_report.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8",
    )
    result_pp = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result_pp.passed is False
    assert any(
        i.code == "EVAL_PER_PROPERTY_BELOW_THRESHOLD" for i in result_pp.issues
    )
