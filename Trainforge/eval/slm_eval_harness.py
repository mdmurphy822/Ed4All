"""Wave 92 — SLM Eval Harness.

End-to-end orchestrator that composes the holdout builder + each
generic-layer + corpus-aware-tier evaluator into a single
``eval_report.json`` whose shape conforms to
``model_card.json::eval_scores``.

Required scores in the output report:

* ``faithfulness`` (0..1) — Tier-1 / Tier-3 weighted accuracy.
* ``coverage`` — proxy = invariant pass-rate (Tier 2) × Tier-1 syntactic
  pass rate when applicable; otherwise just invariant pass rate.
* ``baseline_delta`` — paired-bootstrap mean delta from the trained
  model vs the base model. Optional, populated when a base callable
  is wired in.

Optional sub-blocks (for richer reporting beyond the canonical
eval_scores schema): per-tier breakdowns, per-invariant pass rates,
calibration ECE.

The harness is wired into the training runner's post-train hook
(``Trainforge.training.runner._run_eval``); it can also be invoked
standalone via the module's CLI.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _load_profile(name: str) -> Dict[str, Any]:
    p = _CONFIG_DIR / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Eval profile not found: {p}. Available: "
            f"{sorted(c.stem for c in _CONFIG_DIR.glob('*.yaml'))}"
        )
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _resolve_default_profile(course_path: Path) -> str:
    """Pick a default profile from the course manifest classification."""
    manifest_path = course_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "generic"
        cls = manifest.get("classification") or {}
        subs = [s.lower() for s in cls.get("subdomains", []) or []]
        topics = [t.lower() for t in cls.get("topics", []) or []]
        if "semantic web" in subs or any("rdf" in t or "shacl" in t for t in topics):
            return "rdf_shacl"
    return "generic"


@dataclass
class EvalReport:
    faithfulness: float
    coverage: float
    baseline_delta: Optional[float]
    per_tier: Dict[str, Any]
    per_invariant: Dict[str, Any]
    calibration_ece: Optional[float]
    profile: str
    # Wave 102 additive: source-match precision + named hallucination rate.
    source_match: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "faithfulness": round(self.faithfulness, 4),
            "coverage": round(self.coverage, 4),
            "profile": self.profile,
            "per_tier": self.per_tier,
            "per_invariant": self.per_invariant,
        }
        if self.baseline_delta is not None:
            out["baseline_delta"] = round(self.baseline_delta, 4)
        if self.calibration_ece is not None:
            out["calibration_ece"] = round(self.calibration_ece, 4)
        if self.source_match is not None:
            out["source_match"] = round(self.source_match, 4)
        # Wave 102: hallucination_rate is the named inverse of
        # faithfulness so the ablation renderer can show it as its own
        # column without recomputing.
        out.setdefault("metrics", {})
        out["metrics"]["hallucination_rate"] = round(
            max(0.0, min(1.0, 1.0 - float(self.faithfulness))), 4,
        )
        if self.source_match is not None:
            out["metrics"]["source_match"] = round(self.source_match, 4)
        return out


class SLMEvalHarness:
    """Run the full eval suite for a trained model.

    Args:
        course_path: Path to ``LibV2/courses/<slug>/``.
        model_callable: Trained model callable.
        base_callable: Optional base-model callable for the
            comparative-delta layer. When None, ``baseline_delta``
            is omitted from the report.
        profile: Optional explicit profile name. When None, picked
            from the course manifest classification.
        max_holdout_questions: Optional override on probe count.
    """

    def __init__(
        self,
        course_path: Path,
        model_callable: Callable[[str], str],
        base_callable: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
        max_holdout_questions: Optional[int] = None,
    ) -> None:
        self.course_path = Path(course_path)
        if not self.course_path.exists():
            raise FileNotFoundError(
                f"course_path does not exist: {self.course_path}"
            )
        self.model_callable = model_callable
        self.base_callable = base_callable
        profile_name = profile or _resolve_default_profile(self.course_path)
        self.profile_name = profile_name
        self.profile = _load_profile(profile_name)
        self.max_holdout_questions = max_holdout_questions

    def run_all(self, output_path: Optional[Path] = None) -> Path:
        """Run every enabled evaluator and emit ``eval_report.json``.

        Args:
            output_path: Override for the output file. Defaults to
                ``<course>/eval/eval_report.json``.
        """
        from Trainforge.eval.holdout_builder import HoldoutBuilder
        from Trainforge.eval.faithfulness import FaithfulnessEvaluator
        from Trainforge.eval.invariants import (
            BloomLevelInvariant,
            MisconceptionRejectionInvariant,
            PrerequisiteOrderInvariant,
        )
        from Trainforge.eval.calibration import CalibrationEvaluator
        from Trainforge.eval.key_term_precision import KeyTermPrecisionEvaluator
        from Trainforge.eval.disambiguation import DisambiguationEvaluator
        from Trainforge.eval.source_match import SourceMatchEvaluator

        evaluators = self.profile.get("evaluators", {})
        caps = self.profile.get("caps", {})

        holdout_path = self.course_path / "eval" / "holdout_split.json"
        if not holdout_path.exists():
            HoldoutBuilder(self.course_path).build()

        per_tier: Dict[str, Any] = {}
        per_invariant: Dict[str, Any] = {}

        # --- Faithfulness (Layer 1) -------------------------------- #
        faithfulness_score = 0.0
        faithfulness_per_question: List[Dict[str, Any]] = []
        if evaluators.get("faithfulness"):
            cap = self.max_holdout_questions or caps.get("max_holdout_questions")
            fr = FaithfulnessEvaluator(
                holdout_split=holdout_path,
                model_callable=self.model_callable,
                max_questions=cap,
            ).evaluate()
            per_tier["faithfulness"] = {
                "accuracy": fr["accuracy"],
                "scored": fr["scored_total"],
                "correct": fr["correct"],
            }
            faithfulness_score = fr["accuracy"]
            # Wave 104: surface per-question records for the trace
            # writer in the ablation runner. Each row carries the
            # probe text, model response, ground-truth chunk id (for
            # chunk-anchored edges), and pass/fail outcome.
            for r in fr.get("per_question_results", []) or []:
                edge = r.get("edge", {}) or {}
                source = edge.get("source")
                gt_chunk = source if isinstance(source, str) and source.startswith("chunk_") else None
                faithfulness_per_question.append({
                    "probe": r.get("probe", ""),
                    "response": r.get("response") or "",
                    "ground_truth_chunk_id": gt_chunk,
                    "edge": edge,
                    "outcome": r.get("outcome", "ambiguous"),
                    "correct": r.get("outcome") == "correct",
                })

        # --- Behavioral invariants (Layer 2) ---------------------- #
        invariant_pass_rates: List[float] = []
        inv_cfg = evaluators.get("invariants") or {}
        # Wave 104: collect per-prompt records across invariants so the
        # ablation runner can emit per-probe traces. We retain the
        # invariant name as the prefix for probe_id disambiguation.
        invariant_per_prompt: List[Dict[str, Any]] = []

        def _collect_invariant_probes(invariant_name: str, result: Dict[str, Any]) -> None:
            for i, p in enumerate(result.get("per_prompt", []) or []):
                edge = p.get("edge") or {}
                source = edge.get("source") if isinstance(edge, dict) else None
                gt_chunk = (
                    source if isinstance(source, str) and source.startswith("chunk_")
                    else (
                        p.get("chunk_id")
                        if isinstance(p.get("chunk_id"), str) and p["chunk_id"].startswith("chunk_")
                        else None
                    )
                )
                invariant_per_prompt.append({
                    "probe_id": f"{invariant_name}:{i}",
                    "probe": p.get("prompt", ""),
                    "response": p.get("response") or "",
                    "ground_truth_chunk_id": gt_chunk,
                    "outcome": p.get("outcome", "ambiguous"),
                    "correct": p.get("outcome") == "pass",
                    "invariant": invariant_name,
                })

        if inv_cfg.get("prerequisite_order"):
            r = PrerequisiteOrderInvariant(
                self.course_path,
                max_prompts=caps.get("max_invariant_prompts", 30),
            ).evaluate(self.model_callable)
            per_invariant["prerequisite_order"] = r
            invariant_pass_rates.append(r["pass_rate"])
            _collect_invariant_probes("prerequisite_order", r)
        if inv_cfg.get("bloom_level"):
            r = BloomLevelInvariant(
                self.course_path,
                max_per_level=max(2, caps.get("max_invariant_prompts", 30) // 6),
            ).evaluate(self.model_callable)
            per_invariant["bloom_level"] = r
            invariant_pass_rates.append(r["pass_rate"])
            _collect_invariant_probes("bloom_level", r)
        if inv_cfg.get("misconception_rejection"):
            r = MisconceptionRejectionInvariant(self.course_path).evaluate(self.model_callable)
            per_invariant["misconception_rejection"] = r
            invariant_pass_rates.append(r["pass_rate"])
            _collect_invariant_probes("misconception_rejection", r)

        avg_invariant_pass = (
            sum(invariant_pass_rates) / len(invariant_pass_rates)
            if invariant_pass_rates else 0.0
        )

        # --- Calibration (Layer 3) -------------------------------- #
        calibration_ece: Optional[float] = None
        if evaluators.get("calibration"):
            cap = self.max_holdout_questions or caps.get("max_holdout_questions")
            ce = CalibrationEvaluator(
                holdout_split=holdout_path,
                model_callable=self.model_callable,
                max_questions=cap,
            ).evaluate()
            calibration_ece = ce["ece"]
            per_tier["calibration"] = {
                "ece": ce["ece"],
                "scored": ce["scored"],
                "total": ce["total"],
            }

        # --- Baseline comparator (Layer 4) ------------------------ #
        baseline_delta: Optional[float] = None
        if evaluators.get("baseline_compare") and self.base_callable is not None:
            baseline_delta = self._run_baseline_compare(holdout_path)
            per_tier["baseline_delta"] = baseline_delta

        # --- Tier 3: key-term precision --------------------------- #
        if evaluators.get("key_term_precision"):
            kt = KeyTermPrecisionEvaluator(
                course_path=self.course_path,
                model_callable=self.model_callable,
                max_terms=caps.get("max_key_terms", 50),
            ).evaluate()
            per_tier["key_term_precision"] = {
                "avg_similarity": kt["avg_similarity"],
                "required_element_precision": kt["required_element_precision"],
                "scoring_method": kt["scoring_method"],
                "total": kt["total"],
            }
            invariant_pass_rates.append(kt["required_element_precision"])

        # --- Tier 3: disambiguation ------------------------------- #
        if evaluators.get("disambiguation"):
            dis = DisambiguationEvaluator(
                course_path=self.course_path,
                model_callable=self.model_callable,
                max_pairs=caps.get("max_disambiguation_pairs", 50),
            ).evaluate()
            per_invariant["disambiguation"] = dis
            invariant_pass_rates.append(dis["pass_rate"])

        # --- Source-match (Wave 102 - precision companion to faithfulness)
        source_match_score: Optional[float] = None
        source_match_per_question: List[Dict[str, Any]] = []
        if evaluators.get("source_match"):
            cap = self.max_holdout_questions or caps.get("max_holdout_questions")
            sm = SourceMatchEvaluator(
                holdout_split=holdout_path,
                model_callable=self.model_callable,
                max_questions=cap,
            ).evaluate()
            source_match_score = sm["source_match_rate"]
            per_tier["source_match"] = {
                "rate": sm["source_match_rate"],
                "scored": sm["scored_total"],
                "matches": sm["matches"],
            }
            for r in sm.get("per_question", []) or []:
                source_match_per_question.append({
                    "probe": r.get("probe", ""),
                    "response": r.get("response") or "",
                    "ground_truth_chunk_id": r.get("ground_truth_chunk_id"),
                    "cited_chunk_ids": r.get("cited_chunk_ids", []),
                    "outcome": r.get("outcome", "miss"),
                    "correct": r.get("outcome") == "match",
                })

        # Recompute coverage proxy with the Tier-3 contributions
        if invariant_pass_rates:
            avg_invariant_pass = sum(invariant_pass_rates) / len(invariant_pass_rates)

        coverage = avg_invariant_pass
        if evaluators.get("syntactic") and per_tier.get("syntactic_pass_rate") is not None:
            coverage = avg_invariant_pass * per_tier["syntactic_pass_rate"]

        report = EvalReport(
            faithfulness=faithfulness_score,
            coverage=coverage,
            baseline_delta=baseline_delta,
            per_tier=per_tier,
            per_invariant=per_invariant,
            calibration_ece=calibration_ece,
            profile=self.profile_name,
            source_match=source_match_score,
        )

        # Wave 104: aggregate per-question records into a single
        # `per_question` array so the ablation runner can emit one
        # trace per probe per setup. Records carry probe text, model
        # response, ground-truth chunk id (where known), and a
        # boolean correctness signal for failure-mode classification.
        per_question_all: List[Dict[str, Any]] = []
        per_question_all.extend(faithfulness_per_question)
        per_question_all.extend(invariant_per_prompt)
        per_question_all.extend(source_match_per_question)

        # Wave 104: surface mean retrieval latency when the model
        # callable is RAG-backed. Both BaseOnlyCallable / AdapterCallable
        # leave this attribute unset; RAGCallable exposes it as a
        # rolling mean over its retrieval calls.
        mean_latency = getattr(self.model_callable, "mean_latency_ms", None)

        out_dict = report.to_dict()
        if per_question_all:
            out_dict["per_question"] = per_question_all
        if faithfulness_per_question:
            out_dict["faithfulness_per_question"] = faithfulness_per_question
        if mean_latency is not None:
            out_dict.setdefault("metrics", {})
            out_dict["metrics"]["mean_latency_ms"] = round(float(mean_latency), 2)

        if output_path is None:
            output_path = self.course_path / "eval" / "eval_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(out_dict, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def _run_baseline_compare(self, holdout_path: Path) -> float:
        """Compose probes from the holdout split and run paired delta."""
        from Trainforge.eval.baseline_compare import BaselineComparator
        from Trainforge.eval.faithfulness import _classify_response, _format_probe
        from Trainforge.eval.holdout_builder import load_holdout_split

        split = load_holdout_split(holdout_path)
        edges = split.get("withheld_edges", [])
        cap = self.max_holdout_questions or self.profile.get("caps", {}).get(
            "max_holdout_questions", 100,
        )
        edges = edges[:cap]

        def _score(resp: str) -> float:
            return 1.0 if _classify_response(resp) == "affirm" else 0.0

        prompts = [(_format_probe(e), _score) for e in edges]
        cmp_result = BaselineComparator(
            base_callable=self.base_callable,  # type: ignore[arg-type]
            trained_callable=self.model_callable,
            prompts=prompts,
            bootstrap_iterations=self.profile.get("caps", {}).get(
                "bootstrap_iterations", 1000,
            ),
        ).evaluate()
        return float(cmp_result["mean_delta"])


def main() -> None:  # pragma: no cover — CLI passthrough
    import argparse

    parser = argparse.ArgumentParser(
        description="Wave 92 — run the SLM eval harness on a trained adapter."
    )
    parser.add_argument("--course-path", required=True, help="LibV2 course path.")
    parser.add_argument("--profile", default=None, help="Eval profile name.")
    parser.add_argument(
        "--output", default=None, help="Override output path (default: eval/eval_report.json)."
    )
    parser.add_argument(
        "--max-prompts", type=int, default=None, help="Cap holdout questions."
    )
    args = parser.parse_args()

    def _stub(prompt: str) -> str:
        return "yes (stub)"
    harness = SLMEvalHarness(
        course_path=Path(args.course_path),
        model_callable=_stub,
        profile=args.profile,
        max_holdout_questions=args.max_prompts,
    )
    out = harness.run_all(
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["SLMEvalHarness", "EvalReport"]
