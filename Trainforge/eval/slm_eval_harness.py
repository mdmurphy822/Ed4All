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

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# Wave 105: SHA-256 of empty bytes — used as a placeholder marker in
# legacy / stub holdout_split.json files. When the harness sees this
# hash it must refuse to score Tier-2 evaluators because the holdout
# set is untrustworthy (running them anyway risks train-on-test leak).
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _progress_interval() -> int:
    raw = os.environ.get("TRAINFORGE_EVAL_PROGRESS_EVERY", "25")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "SLMEvalHarness: invalid TRAINFORGE_EVAL_PROGRESS_EVERY=%r; "
            "falling back to 25.",
            raw,
        )
        return 25


class _EvalProgressTracker:
    """Small JSONL + logger progress sink for long adapter eval runs."""

    def __init__(self, progress_path: Path, *, log_every: int) -> None:
        self.progress_path = Path(progress_path)
        self.log_every = max(1, int(log_every))
        self.started_at = time.monotonic()
        self.total_calls = 0
        self.stage: Optional[str] = None
        self.stage_started_at = self.started_at
        self.stage_calls_started = 0
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text("", encoding="utf-8")

    def emit(self, event: str, **payload: Any) -> None:
        row = {
            "event": event,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
            "total_calls": self.total_calls,
            **payload,
        }
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def begin_stage(self, name: str, *, expected_calls: Optional[int] = None) -> None:
        self.stage = name
        self.stage_started_at = time.monotonic()
        self.stage_calls_started = self.total_calls
        logger.info(
            "SLMEvalHarness: starting %s%s",
            name,
            f" (~{expected_calls} model calls)" if expected_calls is not None else "",
        )
        self.emit("stage_start", stage=name, expected_calls=expected_calls)

    def record_call(self) -> None:
        self.total_calls += 1
        stage_calls = self.total_calls - self.stage_calls_started
        if self.total_calls == 1 or self.total_calls % self.log_every == 0:
            elapsed = max(0.001, time.monotonic() - self.started_at)
            calls_per_minute = self.total_calls / (elapsed / 60.0)
            logger.info(
                "SLMEvalHarness: eval progress %d model calls complete "
                "(stage=%s, %.2f calls/min).",
                self.total_calls,
                self.stage or "unknown",
                calls_per_minute,
            )
            self.emit(
                "model_call",
                stage=self.stage,
                stage_calls=stage_calls,
                calls_per_minute=round(calls_per_minute, 3),
            )

    def end_stage(self, name: str) -> None:
        elapsed = time.monotonic() - self.stage_started_at
        calls = self.total_calls - self.stage_calls_started
        logger.info(
            "SLMEvalHarness: finished %s (%d model calls, %.1fs).",
            name,
            calls,
            elapsed,
        )
        self.emit(
            "stage_end",
            stage=name,
            stage_calls=calls,
            stage_elapsed_seconds=round(elapsed, 3),
        )
        self.stage = None

    def finish(self) -> None:
        elapsed = time.monotonic() - self.started_at
        logger.info(
            "SLMEvalHarness: eval complete (%d model calls, %.1fs).",
            self.total_calls,
            elapsed,
        )
        self.emit(
            "run_end",
            total_elapsed_seconds=round(elapsed, 3),
        )


class _ProgressModelCallable:
    """Callable wrapper that records model-call progress."""

    def __init__(self, wrapped: Callable[[str], str], tracker: _EvalProgressTracker) -> None:
        self._wrapped = wrapped
        self._tracker = tracker

    def __call__(self, prompt: str) -> str:
        response = self._wrapped(prompt)
        self._tracker.record_call()
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


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
    # Wave 108 / Phase B additive: negative-grounding signals.
    negative_grounding_accuracy: Optional[float] = None
    yes_rate: Optional[float] = None
    # Wave 109 / Phase C additive: per-property accuracy when the
    # course has a property manifest. None elsewhere; keys are property IDs.
    per_property_accuracy: Optional[Dict[str, Optional[float]]] = None

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
        if self.negative_grounding_accuracy is not None:
            out["negative_grounding_accuracy"] = round(
                self.negative_grounding_accuracy, 4
            )
        if self.yes_rate is not None:
            out["yes_rate"] = round(self.yes_rate, 4)
        # Wave 102: hallucination_rate is the named inverse of
        # faithfulness so the ablation renderer can show it as its own
        # column without recomputing.
        out.setdefault("metrics", {})
        out["metrics"]["hallucination_rate"] = round(
            max(0.0, min(1.0, 1.0 - float(self.faithfulness))), 4,
        )
        if self.source_match is not None:
            out["metrics"]["source_match"] = round(self.source_match, 4)
        if self.negative_grounding_accuracy is not None:
            out["metrics"]["negative_grounding_accuracy"] = round(
                self.negative_grounding_accuracy, 4
            )
        if self.per_property_accuracy:
            # Round each scored property; preserve None for unscored.
            rounded: Dict[str, Optional[float]] = {}
            for k, v in self.per_property_accuracy.items():
                rounded[k] = round(float(v), 4) if v is not None else None
            out["per_property_accuracy"] = rounded
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
        from Trainforge.eval.chunk_ids import is_chunk_id

        evaluators = self.profile.get("evaluators", {})
        caps = self.profile.get("caps", {})
        if output_path is None:
            output_path = self.course_path / "eval" / "eval_report.json"
        output_path = Path(output_path)
        progress = _EvalProgressTracker(
            output_path.parent / "eval_progress.jsonl",
            log_every=_progress_interval(),
        )
        model_callable = _ProgressModelCallable(self.model_callable, progress)
        progress.emit(
            "run_start",
            profile=self.profile_name,
            output_path=str(output_path),
            progress_path=str(progress.progress_path),
        )

        holdout_path = self.course_path / "eval" / "holdout_split.json"
        if not holdout_path.exists():
            HoldoutBuilder(self.course_path).build()

        # Wave 105: refuse to score Tier-2 (graph-derived) evaluators
        # when the holdout split is a placeholder. SHA-256(b"") is
        # the canonical "empty content" hash — when the holdout
        # builder was a stub, the file landed on disk with this
        # hash. Running Tier-2 against an empty / unverified split
        # risks train-on-test contamination, so we drop those
        # evaluators and stamp the report's ``tier_2_status`` field
        # so the model card reviewer sees the gap.
        tier_2_status: Optional[str] = None
        try:
            holdout_payload = json.loads(
                holdout_path.read_text(encoding="utf-8"),
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.critical(
                "SLMEvalHarness: cannot read holdout_split at %s "
                "(%s); skipping Tier-2 evaluators.", holdout_path, exc,
            )
            holdout_payload = {}
            tier_2_status = "skipped: holdout_split unreadable"

        declared_hash = (
            (holdout_payload or {}).get("holdout_graph_hash") or ""
        )
        if declared_hash in ("", _EMPTY_SHA256):
            logger.critical(
                "SLMEvalHarness: holdout_split.json at %s carries an "
                "empty-bytes hash (%r); refusing to score Tier-2 "
                "evaluators. Rebuild the holdout split with "
                "HoldoutBuilder before re-running.",
                holdout_path, declared_hash,
            )
            # Drop Tier-2 evaluators (faithfulness, invariants,
            # source-match) so they don't run against an
            # untrustworthy holdout. Tier-1 syntactic + Tier-3
            # semantic checks remain available because they don't
            # depend on the holdout split.
            evaluators = dict(evaluators)
            for k in ("faithfulness", "invariants", "source_match",
                      "calibration", "baseline_compare"):
                if k in evaluators:
                    evaluators[k] = (
                        {} if isinstance(evaluators[k], dict) else False
            )
            tier_2_status = "skipped: holdout_split is placeholder"

        withheld_edges = (holdout_payload or {}).get("withheld_edges", []) or []

        def _capped_count(items: List[Any], cap: Optional[int]) -> int:
            if cap is None:
                return len(items)
            return min(len(items), int(cap))

        def _run_stage(
            name: str,
            expected_calls: Optional[int],
            fn: Callable[[], Any],
        ) -> Any:
            progress.begin_stage(name, expected_calls=expected_calls)
            try:
                return fn()
            finally:
                progress.end_stage(name)

        per_tier: Dict[str, Any] = {}
        per_invariant: Dict[str, Any] = {}

        # --- Faithfulness (Layer 1) -------------------------------- #
        faithfulness_score = 0.0
        faithfulness_yes_rate: Optional[float] = None
        faithfulness_per_question: List[Dict[str, Any]] = []
        if evaluators.get("faithfulness"):
            cap = self.max_holdout_questions or caps.get("max_holdout_questions")
            fr = _run_stage(
                "faithfulness",
                _capped_count(withheld_edges, cap),
                lambda: FaithfulnessEvaluator(
                    holdout_split=holdout_path,
                    model_callable=model_callable,
                    max_questions=cap,
                ).evaluate(),
            )
            per_tier["faithfulness"] = {
                "accuracy": fr["accuracy"],
                "scored": fr["scored_total"],
                "correct": fr["correct"],
            }
            faithfulness_score = fr["accuracy"]
            # Wave 108 / Phase B: yes_rate surfaces yes-bias even when
            # accuracy is high (every probe is a TRUE statement, so a
            # 'yes always' model trivially scores 1.0).
            faithfulness_yes_rate = fr.get("yes_rate")
            # Wave 104: surface per-question records for the trace
            # writer in the ablation runner. Each row carries the
            # probe text, model response, ground-truth chunk id (for
            # chunk-anchored edges), and pass/fail outcome.
            for r in fr.get("per_question_results", []) or []:
                edge = r.get("edge", {}) or {}
                source = edge.get("source")
                gt_chunk = source if is_chunk_id(source) else None
                faithfulness_per_question.append({
                    "probe": r.get("probe", ""),
                    "response": r.get("response") or "",
                    "ground_truth_chunk_id": gt_chunk,
                    "edge": edge,
                    "outcome": r.get("outcome", "ambiguous"),
                    "correct": r.get("outcome") == "correct",
                })

        # --- Negative grounding (Wave 108 / Phase B) ---------------- #
        # Same probe-template machinery as faithfulness, but ground-truth
        # is "no". Catches yes-biased template-recognizer adapters that
        # answer "yes" to everything (fail open on positive-only probes).
        negative_grounding_score: Optional[float] = None
        if evaluators.get("faithfulness"):
            from Trainforge.eval.negative_grounding import (
                NegativeGroundingEvaluator,
            )
            cap = self.max_holdout_questions or caps.get("max_holdout_questions")
            negative_probes = (holdout_payload or {}).get("negative_probes", []) or []
            ng = _run_stage(
                "negative_grounding",
                _capped_count(negative_probes, cap),
                lambda: NegativeGroundingEvaluator(
                    holdout_split=holdout_path,
                    model_callable=model_callable,
                    max_questions=cap,
                ).evaluate(),
            )
            per_tier["negative_grounding"] = {
                "accuracy": ng.get("negative_grounding_accuracy"),
                "false_yes_rate": ng.get("false_yes_rate"),
                "scored": ng.get("scored_total"),
            }
            negative_grounding_score = ng.get("negative_grounding_accuracy")

        # --- Per-property eval (Wave 109 / Phase C) ---------------- #
        # No-ops for courses without a property manifest. Surface
        # per-property accuracy in eval_report.json so the
        # EvalGatingValidator can apply per-property thresholds.
        per_property_accuracy: Optional[Dict[str, Optional[float]]] = None
        if evaluators.get("faithfulness"):
            from Trainforge.eval.property_eval import PerPropertyEvaluator
            try:
                pp_result = _run_stage(
                    "per_property",
                    None,
                    lambda: PerPropertyEvaluator(
                        holdout_split=holdout_path,
                        course_slug=self.course_path.name,
                        model_callable=model_callable,
                    ).evaluate(),
                )
                per_tier["per_property"] = pp_result
                pa = pp_result.get("per_property_accuracy")
                if pa:
                    per_property_accuracy = pa
            except Exception as exc:  # noqa: BLE001 — advisory
                logger.warning("PerPropertyEvaluator failed: %s", exc)

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
                    source if is_chunk_id(source)
                    else (
                        p.get("chunk_id")
                        if is_chunk_id(p.get("chunk_id"))
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
            r = _run_stage(
                "invariant:prerequisite_order",
                None,
                lambda: PrerequisiteOrderInvariant(
                    self.course_path,
                    max_prompts=caps.get("max_invariant_prompts", 30),
                ).evaluate(model_callable),
            )
            per_invariant["prerequisite_order"] = r
            invariant_pass_rates.append(r["pass_rate"])
            _collect_invariant_probes("prerequisite_order", r)
        if inv_cfg.get("bloom_level"):
            r = _run_stage(
                "invariant:bloom_level",
                None,
                lambda: BloomLevelInvariant(
                    self.course_path,
                    max_per_level=max(2, caps.get("max_invariant_prompts", 30) // 6),
                ).evaluate(model_callable),
            )
            per_invariant["bloom_level"] = r
            invariant_pass_rates.append(r["pass_rate"])
            _collect_invariant_probes("bloom_level", r)
        if inv_cfg.get("misconception_rejection"):
            r = _run_stage(
                "invariant:misconception_rejection",
                None,
                lambda: MisconceptionRejectionInvariant(self.course_path).evaluate(
                    model_callable,
                ),
            )
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
            ce = _run_stage(
                "calibration",
                _capped_count(withheld_edges, cap),
                lambda: CalibrationEvaluator(
                    holdout_split=holdout_path,
                    model_callable=model_callable,
                    max_questions=cap,
                ).evaluate(),
            )
            calibration_ece = ce["ece"]
            per_tier["calibration"] = {
                "ece": ce["ece"],
                "scored": ce["scored"],
                "total": ce["total"],
            }

        # --- Baseline comparator (Layer 4) ------------------------ #
        baseline_delta: Optional[float] = None
        if evaluators.get("baseline_compare") and self.base_callable is not None:
            cap = self.max_holdout_questions or caps.get("max_holdout_questions")
            baseline_delta = _run_stage(
                "baseline_compare",
                _capped_count(withheld_edges, cap),
                lambda: self._run_baseline_compare(
                    holdout_path,
                    model_callable=model_callable,
                ),
            )
            per_tier["baseline_delta"] = baseline_delta

        # --- Tier 3: key-term precision --------------------------- #
        if evaluators.get("key_term_precision"):
            kt = _run_stage(
                "key_term_precision",
                caps.get("max_key_terms", 50),
                lambda: KeyTermPrecisionEvaluator(
                    course_path=self.course_path,
                    model_callable=model_callable,
                    max_terms=caps.get("max_key_terms", 50),
                ).evaluate(),
            )
            per_tier["key_term_precision"] = {
                "avg_similarity": kt["avg_similarity"],
                "required_element_precision": kt["required_element_precision"],
                "scoring_method": kt["scoring_method"],
                "total": kt["total"],
            }
            invariant_pass_rates.append(kt["required_element_precision"])

        # --- Tier 3: disambiguation ------------------------------- #
        if evaluators.get("disambiguation"):
            dis = _run_stage(
                "disambiguation",
                caps.get("max_disambiguation_pairs", 50),
                lambda: DisambiguationEvaluator(
                    course_path=self.course_path,
                    model_callable=model_callable,
                    max_pairs=caps.get("max_disambiguation_pairs", 50),
                ).evaluate(),
            )
            per_invariant["disambiguation"] = dis
            invariant_pass_rates.append(dis["pass_rate"])

        # --- Source-match (Wave 102 - precision companion to faithfulness)
        source_match_score: Optional[float] = None
        source_match_per_question: List[Dict[str, Any]] = []
        if evaluators.get("source_match"):
            cap = self.max_holdout_questions or caps.get("max_holdout_questions")
            chunk_edges = [e for e in withheld_edges if is_chunk_id(e.get("source"))]
            sm = _run_stage(
                "source_match",
                _capped_count(chunk_edges, cap),
                lambda: SourceMatchEvaluator(
                    holdout_split=holdout_path,
                    model_callable=model_callable,
                    max_questions=cap,
                ).evaluate(),
            )
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
            negative_grounding_accuracy=negative_grounding_score,
            yes_rate=faithfulness_yes_rate,
            per_property_accuracy=per_property_accuracy,
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
        mean_latency = getattr(model_callable, "mean_latency_ms", None)

        out_dict = report.to_dict()
        if per_question_all:
            out_dict["per_question"] = per_question_all
        if faithfulness_per_question:
            out_dict["faithfulness_per_question"] = faithfulness_per_question
        if mean_latency is not None:
            out_dict.setdefault("metrics", {})
            out_dict["metrics"]["mean_latency_ms"] = round(float(mean_latency), 2)
        # Wave 105: surface the Tier-2 holdout status so reviewers see
        # exactly why those metrics may be absent. Carries either
        # "ok" (default) or a "skipped: ..." reason.
        out_dict["tier_2_status"] = tier_2_status or "ok"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(out_dict, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        progress.finish()
        return output_path

    def _run_baseline_compare(
        self,
        holdout_path: Path,
        *,
        model_callable: Optional[Callable[[str], str]] = None,
    ) -> float:
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
            trained_callable=model_callable or self.model_callable,
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
