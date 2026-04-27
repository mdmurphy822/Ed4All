"""Trainforge eval submodule (Wave 92 — slm-training-2026-04-26).

End-to-end eval surface for trained SLM adapters. Composed of five
generic layers (faithfulness, behavioral invariants, calibration,
comparative delta, regression) plus three corpus-aware tiers
(machine-verifiable, graph-derived, semantic) so a single
``SLMEvalHarness`` produces an ``eval_report.json`` whose shape drops
into ``model_card.json::eval_scores``.

Public API:

    from Trainforge.eval import (
        HoldoutBuilder,
        load_holdout_split,
        FaithfulnessEvaluator,
        PrerequisiteOrderInvariant,
        BloomLevelInvariant,
        MisconceptionRejectionInvariant,
        CalibrationEvaluator,
        BaselineComparator,
        RegressionEvaluator,
        KeyTermPrecisionEvaluator,
        DisambiguationEvaluator,
        SLMEvalHarness,
        EvalReport,
        evaluate_turtle,
        evaluate_sparql,
        evaluate_shacl_shape,
        evaluate_shacl_validation,
        evaluate_owl_entailment,
    )
"""
from Trainforge.eval.baseline_compare import BaselineComparator  # noqa: F401
from Trainforge.eval.calibration import CalibrationEvaluator  # noqa: F401
from Trainforge.eval.disambiguation import DisambiguationEvaluator  # noqa: F401
from Trainforge.eval.faithfulness import FaithfulnessEvaluator  # noqa: F401
from Trainforge.eval.holdout_builder import (  # noqa: F401
    HoldoutBuilder,
    load_holdout_split,
)
from Trainforge.eval.invariants import (  # noqa: F401
    BloomLevelInvariant,
    MisconceptionRejectionInvariant,
    PrerequisiteOrderInvariant,
)
from Trainforge.eval.key_term_precision import KeyTermPrecisionEvaluator  # noqa: F401
from Trainforge.eval.regression import RegressionEvaluator  # noqa: F401
from Trainforge.eval.slm_eval_harness import EvalReport, SLMEvalHarness  # noqa: F401
from Trainforge.eval.syntactic import (  # noqa: F401
    evaluate_owl_entailment,
    evaluate_shacl_shape,
    evaluate_shacl_validation,
    evaluate_sparql,
    evaluate_turtle,
)


__all__ = [
    "BaselineComparator",
    "BloomLevelInvariant",
    "CalibrationEvaluator",
    "DisambiguationEvaluator",
    "EvalReport",
    "FaithfulnessEvaluator",
    "HoldoutBuilder",
    "KeyTermPrecisionEvaluator",
    "MisconceptionRejectionInvariant",
    "PrerequisiteOrderInvariant",
    "RegressionEvaluator",
    "SLMEvalHarness",
    "evaluate_owl_entailment",
    "evaluate_shacl_shape",
    "evaluate_shacl_validation",
    "evaluate_sparql",
    "evaluate_turtle",
    "load_holdout_split",
]
