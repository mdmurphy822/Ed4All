"""Phase 4 Subtask 32 — threshold-calibration CLI for embedding +
BERT-ensemble validators.

Sweeps the per-gate threshold across a configurable range against a
holdout corpus, computes precision / recall / F1 per threshold, and
persists the best-F1 threshold (plus the full sweep table for audit)
to ``LibV2/courses/<slug>/eval/calibrated_thresholds.yaml``.

Holdout corpus shape (JSONL, one row per block):

    {
        "block_id": "<id>",                       # required, str
        "block_type": "<type>",                   # required, str — one of
                                                  #   "objective" | "assessment_item" | "example"
        "content": <dict|str>,                    # required — block.content surface
        "expected_action": "pass"|"regenerate"|"block",  # required, str
        "expected_bloom_level": "<level>",        # optional, str — required for the
                                                  #   bert_ensemble gate (one of the 6
                                                  #   canonical Bloom's levels)
        # Optional per-block hints consumed by specific gates:
        "objective_ids": ["TO-01", ...],          # used by objective_assessment_similarity
        "concept_refs": ["ed4all:Foo", ...],      # used by concept_example_similarity
        "bloom_level": "<level>"                  # used by bert_ensemble (declared bloom)
    }

The calibration script is read-only against the holdout corpus and
writes only to the per-course ``eval/calibrated_thresholds.yaml``
sidecar. The output YAML carries:

    - ``gate``: the gate name being calibrated.
    - ``best_threshold``: the threshold value with the highest F1.
    - ``best_f1``: the F1 score at ``best_threshold``.
    - ``sweep``: list of ``{threshold, precision, recall, f1, tp, fp, fn, tn}``
      rows over the entire sweep so an operator can inspect the
      precision-recall trade-off.

Subtask 33 (temperature scaling for the BERT ensemble) and Subtask 34
(dispersion-threshold sweep) extend this script in-place by appending
top-level keys to the same YAML file.

CLI invocation::

    python scripts/calibrate_phase4_thresholds.py \\
        --course-slug mini-101 \\
        --gate objective_assessment \\
        --sweep-from 0.30 \\
        --sweep-to 0.80 \\
        --steps 11

Verification: ``python scripts/calibrate_phase4_thresholds.py --help`` exits 0.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Ensure we can import lib.* + Courseforge.scripts.blocks regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # type: ignore  # noqa: E402

logger = logging.getLogger("calibrate_phase4_thresholds")


# ---------------------------------------------------------------------------
# Constants + canonical gate registry
# ---------------------------------------------------------------------------

GATE_OBJECTIVE_ASSESSMENT = "objective_assessment"
GATE_CONCEPT_EXAMPLE = "concept_example"
GATE_OBJECTIVE_ROUNDTRIP = "objective_roundtrip"
GATE_BERT_ENSEMBLE = "bert_ensemble"

GATE_CHOICES: Tuple[str, ...] = (
    GATE_OBJECTIVE_ASSESSMENT,
    GATE_CONCEPT_EXAMPLE,
    GATE_OBJECTIVE_ROUNDTRIP,
    GATE_BERT_ENSEMBLE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HoldoutRow:
    """One row from the holdout JSONL corpus."""

    block_id: str
    block_type: str
    content: Any  # dict or str
    expected_action: str  # "pass" | "regenerate" | "block"
    expected_bloom_level: Optional[str] = None
    objective_ids: Optional[List[str]] = None
    concept_refs: Optional[List[str]] = None
    bloom_level: Optional[str] = None
    objective_statements: Optional[Dict[str, str]] = None
    concept_definitions: Optional[Dict[str, str]] = None
    paraphrase: Optional[str] = None  # pre-recorded paraphrase for roundtrip gate


@dataclass
class SweepRow:
    """One threshold's confusion matrix + derived metrics."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _holdout_path(course_slug: str) -> Path:
    return _REPO_ROOT / "LibV2" / "courses" / course_slug / "eval" / "phase4_holdout.jsonl"


def _output_path(course_slug: str) -> Path:
    return _REPO_ROOT / "LibV2" / "courses" / course_slug / "eval" / "calibrated_thresholds.yaml"


def load_holdout(path: Path) -> List[HoldoutRow]:
    """Read the JSONL holdout corpus and coerce into ``HoldoutRow`` rows.

    Rows missing required fields are skipped with a warning so a
    partially-malformed corpus still calibrates against the well-formed
    rows. Returns an empty list when the file doesn't exist.
    """
    if not path.exists():
        logger.warning("Holdout corpus not found at %s; returning empty list.", path)
        return []

    rows: List[HoldoutRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON at %s:%d (%s)", path, lineno, exc
                )
                continue
            if not all(k in payload for k in ("block_id", "block_type", "content", "expected_action")):
                logger.warning(
                    "Skipping row at %s:%d (missing one of block_id/block_type/content/expected_action).",
                    path,
                    lineno,
                )
                continue
            try:
                rows.append(
                    HoldoutRow(
                        block_id=str(payload["block_id"]),
                        block_type=str(payload["block_type"]),
                        content=payload["content"],
                        expected_action=str(payload["expected_action"]),
                        expected_bloom_level=payload.get("expected_bloom_level"),
                        objective_ids=payload.get("objective_ids"),
                        concept_refs=payload.get("concept_refs"),
                        bloom_level=payload.get("bloom_level"),
                        objective_statements=payload.get("objective_statements"),
                        concept_definitions=payload.get("concept_definitions"),
                        paraphrase=payload.get("paraphrase"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping row at %s:%d (failed to coerce: %s)",
                    path,
                    lineno,
                    exc,
                )
    return rows


def write_yaml(payload: Dict[str, Any], path: Path) -> None:
    """Write a YAML payload to disk, creating parent dirs as needed.

    Existing files are merged at the top-level dict layer so subsequent
    Subtask 33 / 34 invocations can append ``ensemble_temperatures`` /
    ``dispersion_threshold`` keys without overwriting the per-gate
    sweep results from Subtask 32.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(existing, dict):
                existing = {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Existing YAML at %s couldn't be parsed (%s); overwriting.",
                path,
                exc,
            )
            existing = {}
    existing.update(payload)
    path.write_text(yaml.safe_dump(existing, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Block construction adapter
# ---------------------------------------------------------------------------


def _make_block(row: HoldoutRow) -> Any:
    """Construct a Block-shaped object from a holdout row.

    Blocks come from ``Courseforge/scripts/blocks.py``; we lazy-import
    so a slim install / CLI smoke test that only exercises ``--help``
    doesn't pay the import cost. Returns the canonical Block dataclass.
    """
    scripts_dir = _REPO_ROOT / "Courseforge" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from blocks import Block  # type: ignore[import-not-found]

    content = row.content
    # If the row provided objective_ids / concept_refs at the top level,
    # fold them into a dict-shaped content surface so the gate's
    # extractors find them. Existing dict content takes precedence.
    if isinstance(content, dict):
        content = dict(content)
        if row.objective_ids and "objective_ids" not in content:
            content["objective_ids"] = list(row.objective_ids)
        if row.concept_refs and "concept_refs" not in content:
            content["concept_refs"] = list(row.concept_refs)

    return Block(
        block_id=row.block_id,
        block_type=row.block_type,
        page_id=row.block_id.split("#")[0] if "#" in row.block_id else row.block_id,
        sequence=0,
        content=content,
        bloom_level=row.bloom_level or None,
        objective_ids=tuple(row.objective_ids or ()),
    )


# ---------------------------------------------------------------------------
# Per-gate evaluation: returns a callable that, given a row + threshold,
# decides whether the gate would fire ``regenerate`` for that row.
# ---------------------------------------------------------------------------


def _build_objective_assessment_evaluator(rows: List[HoldoutRow]) -> Callable[[float], List[str]]:
    """Return a function ``(threshold) -> [predicted_action per row]``."""
    from lib.validators.objective_assessment_similarity import (
        ObjectiveAssessmentSimilarityValidator,
    )

    # Build the objective_statements map from any row hints.
    obj_statements: Dict[str, str] = {}
    for r in rows:
        if r.objective_statements:
            obj_statements.update(r.objective_statements)

    blocks = [_make_block(r) for r in rows]

    def _eval(threshold: float) -> List[str]:
        validator = ObjectiveAssessmentSimilarityValidator(threshold=threshold)
        # Per-row evaluation: we pass blocks one at a time so the
        # confusion matrix is per-row, not aggregated.
        per_row_actions: List[str] = []
        for block in blocks:
            result = validator.validate(
                {
                    "blocks": [block],
                    "threshold": threshold,
                    "objective_statements": obj_statements,
                }
            )
            action = result.action or "pass"
            per_row_actions.append(action)
        return per_row_actions

    return _eval


def _build_concept_example_evaluator(rows: List[HoldoutRow]) -> Callable[[float], List[str]]:
    from lib.validators.concept_example_similarity import (
        ConceptExampleSimilarityValidator,
    )

    concept_defs: Dict[str, str] = {}
    for r in rows:
        if r.concept_definitions:
            concept_defs.update(r.concept_definitions)

    blocks = [_make_block(r) for r in rows]

    def _eval(threshold: float) -> List[str]:
        validator = ConceptExampleSimilarityValidator(threshold=threshold)
        per_row_actions: List[str] = []
        for block in blocks:
            result = validator.validate(
                {
                    "blocks": [block],
                    "threshold": threshold,
                    "concept_definitions": concept_defs,
                }
            )
            action = result.action or "pass"
            per_row_actions.append(action)
        return per_row_actions

    return _eval


def _build_objective_roundtrip_evaluator(rows: List[HoldoutRow]) -> Callable[[float], List[str]]:
    from lib.validators.objective_roundtrip_similarity import (
        ObjectiveRoundtripSimilarityValidator,
    )

    blocks = [_make_block(r) for r in rows]
    # Pre-recorded paraphrase per block_id so the calibration is
    # deterministic across runs (no LLM dispatch). Rows without a
    # paraphrase fall back to echoing the statement (cosine ~= 1.0,
    # always passes — used to establish the negative class).
    paraphrase_map: Dict[str, Optional[str]] = {r.block_id: r.paraphrase for r in rows}

    def _paraphrase_fn_factory(block_id: str) -> Callable[[str], Optional[str]]:
        def _fn(text: str) -> Optional[str]:
            return paraphrase_map.get(block_id) or text
        return _fn

    def _eval(threshold: float) -> List[str]:
        per_row_actions: List[str] = []
        for block in blocks:
            validator = ObjectiveRoundtripSimilarityValidator(
                threshold=threshold,
                paraphrase_fn=_paraphrase_fn_factory(block.block_id),
            )
            result = validator.validate(
                {
                    "blocks": [block],
                    "threshold": threshold,
                }
            )
            action = result.action or "pass"
            per_row_actions.append(action)
        return per_row_actions

    return _eval


def _build_bert_ensemble_evaluator(
    rows: List[HoldoutRow],
    ensemble: Optional[Any] = None,
) -> Callable[[float], List[str]]:
    """Calibrate the BERT-ensemble dispersion gate.

    Per-block: classify via the ensemble; predict ``regenerate`` when the
    ensemble winner disagrees with the declared bloom_level above the
    confidence floor OR when dispersion exceeds ``threshold``. The
    ``threshold`` swept here IS the dispersion threshold.

    When ``transformers`` isn't available, a mock ensemble is used —
    the calibration loop still produces a confusion matrix shape so a
    fixture-driven CI smoke test can exercise the script end-to-end.
    """
    from lib.validators.bloom_classifier_disagreement import (
        BloomClassifierDisagreementValidator,
    )

    blocks = [_make_block(r) for r in rows]

    def _eval(threshold: float) -> List[str]:
        validator = BloomClassifierDisagreementValidator(
            ensemble=ensemble,
            dispersion_threshold=threshold,
        )
        per_row_actions: List[str] = []
        for block in blocks:
            result = validator.validate({"blocks": [block]})
            action = result.action or "pass"
            per_row_actions.append(action)
        return per_row_actions

    return _eval


GATE_BUILDERS: Dict[str, Callable[[List[HoldoutRow]], Callable[[float], List[str]]]] = {
    GATE_OBJECTIVE_ASSESSMENT: _build_objective_assessment_evaluator,
    GATE_CONCEPT_EXAMPLE: _build_concept_example_evaluator,
    GATE_OBJECTIVE_ROUNDTRIP: _build_objective_roundtrip_evaluator,
    GATE_BERT_ENSEMBLE: _build_bert_ensemble_evaluator,
}


# ---------------------------------------------------------------------------
# Confusion matrix + sweep
# ---------------------------------------------------------------------------


def _confusion(predicted: Iterable[str], expected: Iterable[str]) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) treating ``regenerate`` (or ``block``) as
    the positive class and ``pass`` as the negative class.

    The Phase 4 plan describes the gates' job as "fire when the corpus
    needs regeneration", so the positive class is "fire". A true positive
    is "gate fired and the holdout said the row needed regeneration".
    """
    tp = fp = fn = tn = 0
    for p, e in zip(predicted, expected):
        p_pos = p in ("regenerate", "block")
        e_pos = e in ("regenerate", "block")
        if p_pos and e_pos:
            tp += 1
        elif p_pos and not e_pos:
            fp += 1
        elif (not p_pos) and e_pos:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return (precision, recall, f1)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def sweep_thresholds(
    evaluator: Callable[[float], List[str]],
    expected: List[str],
    sweep_from: float,
    sweep_to: float,
    steps: int,
) -> List[SweepRow]:
    """Evaluate ``evaluator`` at ``steps`` thresholds across ``[from, to]``.

    Linear spacing. Returns one ``SweepRow`` per threshold.
    """
    if steps < 2:
        thresholds = [sweep_from]
    else:
        step_size = (sweep_to - sweep_from) / (steps - 1)
        thresholds = [round(sweep_from + i * step_size, 4) for i in range(steps)]

    sweep: List[SweepRow] = []
    for t in thresholds:
        predicted = evaluator(t)
        tp, fp, fn, tn = _confusion(predicted, expected)
        precision, recall, f1 = _f1(tp, fp, fn)
        sweep.append(
            SweepRow(
                threshold=t,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return sweep


def best_row(sweep: List[SweepRow]) -> Optional[SweepRow]:
    """Return the sweep row with the highest F1 (ties broken by lower threshold)."""
    if not sweep:
        return None
    return sorted(sweep, key=lambda r: (-r.f1, r.threshold))[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4 calibration: sweep validator thresholds against a "
            "holdout corpus and persist the best-F1 threshold."
        ),
    )
    parser.add_argument(
        "--course-slug",
        required=True,
        help="LibV2 course slug; reads holdout from "
        "LibV2/courses/<slug>/eval/phase4_holdout.jsonl and writes "
        "calibrated_thresholds.yaml in the same dir.",
    )
    parser.add_argument(
        "--gate",
        required=True,
        choices=GATE_CHOICES,
        help="Which Phase 4 gate to calibrate.",
    )
    parser.add_argument(
        "--sweep-from",
        type=float,
        default=0.30,
        help="Lower bound of the threshold sweep (default 0.30).",
    )
    parser.add_argument(
        "--sweep-to",
        type=float,
        default=0.80,
        help="Upper bound of the threshold sweep (default 0.80).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=11,
        help="Number of sweep steps (default 11 → 0.30, 0.35, … 0.80).",
    )
    # Subtask 33 / 34 hooks (parsed but no-op until those subtasks land).
    parser.add_argument(
        "--calibrate-temperature",
        action="store_true",
        help="Subtask 33: tune per-member softmax temperature for the "
        "BERT ensemble. No-op for non-bert_ensemble gates.",
    )
    parser.add_argument(
        "--temperature-from",
        type=float,
        default=0.5,
        help="Subtask 33: lower bound of the temperature sweep (default 0.5).",
    )
    parser.add_argument(
        "--temperature-to",
        type=float,
        default=3.0,
        help="Subtask 33: upper bound of the temperature sweep (default 3.0).",
    )
    parser.add_argument(
        "--temperature-steps",
        type=int,
        default=11,
        help="Subtask 33: number of temperature sweep steps (default 11).",
    )
    parser.add_argument(
        "--calibrate-dispersion",
        action="store_true",
        help="Subtask 34: sweep dispersion threshold against the holdout "
        "labels (0.3 → 1.0). Implies --gate bert_ensemble.",
    )
    parser.add_argument(
        "--dispersion-from",
        type=float,
        default=0.30,
        help="Subtask 34: lower bound of the dispersion sweep (default 0.30).",
    )
    parser.add_argument(
        "--dispersion-to",
        type=float,
        default=1.00,
        help="Subtask 34: upper bound of the dispersion sweep (default 1.00).",
    )
    parser.add_argument(
        "--dispersion-steps",
        type=int,
        default=15,
        help="Subtask 34: number of dispersion sweep steps (default 15).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG level).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    holdout_path = _holdout_path(args.course_slug)
    rows = load_holdout(holdout_path)
    if not rows:
        logger.error(
            "No holdout rows loaded from %s; cannot calibrate. Provide a "
            "JSONL fixture per the script's docstring.",
            holdout_path,
        )
        return 2

    builder = GATE_BUILDERS[args.gate]
    evaluator = builder(rows)
    expected = [r.expected_action for r in rows]

    logger.info(
        "Calibrating gate=%s on %d holdout rows; threshold sweep = [%.4f, %.4f] in %d steps.",
        args.gate,
        len(rows),
        args.sweep_from,
        args.sweep_to,
        args.steps,
    )

    sweep = sweep_thresholds(
        evaluator=evaluator,
        expected=expected,
        sweep_from=args.sweep_from,
        sweep_to=args.sweep_to,
        steps=args.steps,
    )
    best = best_row(sweep)
    if best is None:
        logger.error("Sweep produced no rows; nothing to persist.")
        return 3

    payload: Dict[str, Any] = {
        f"gate_{args.gate}": {
            "best_threshold": best.threshold,
            "best_f1": best.f1,
            "best_precision": best.precision,
            "best_recall": best.recall,
            "sweep": [asdict(r) for r in sweep],
            "n_holdout_rows": len(rows),
        },
    }

    # Subtask 33: per-member temperature calibration (BERT ensemble only).
    if args.calibrate_temperature and args.gate == GATE_BERT_ENSEMBLE:
        temp_payload = calibrate_ensemble_temperatures(
            rows=rows,
            sweep_from=args.temperature_from,
            sweep_to=args.temperature_to,
            steps=args.temperature_steps,
        )
        payload["ensemble_temperatures"] = temp_payload

    # Subtask 34: dispersion-threshold sweep against holdout labels.
    if args.calibrate_dispersion or args.gate == GATE_BERT_ENSEMBLE:
        dispersion_payload = sweep_dispersion_threshold(
            rows=rows,
            sweep_from=args.dispersion_from,
            sweep_to=args.dispersion_to,
            steps=args.dispersion_steps,
        )
        payload["dispersion_calibration"] = dispersion_payload

    out_path = _output_path(args.course_slug)
    write_yaml(payload, out_path)
    logger.info(
        "Wrote calibration to %s; best %s threshold = %.4f (F1=%.4f).",
        out_path,
        args.gate,
        best.threshold,
        best.f1,
    )
    return 0


# ---------------------------------------------------------------------------
# Subtask 33 / 34 stubs — implemented in their own commits.
# Defined here so the Subtask 32 commit's CLI surface forward-references
# them without ImportError; the real bodies land in their respective
# subtask commits below.
# ---------------------------------------------------------------------------


def calibrate_ensemble_temperatures(
    rows: List[HoldoutRow],
    sweep_from: float,
    sweep_to: float,
    steps: int,
) -> Dict[str, Any]:
    """Subtask 33: per-member softmax-temperature calibration.

    Stub for Subtask 32. The Subtask 33 commit replaces the body with
    the real ECE-minimising sweep over each ensemble member.
    """
    return {
        "calibrated": False,
        "reason": "calibrate_ensemble_temperatures landed in Subtask 33",
        "sweep_range": [sweep_from, sweep_to],
        "steps": steps,
    }


def sweep_dispersion_threshold(
    rows: List[HoldoutRow],
    sweep_from: float,
    sweep_to: float,
    steps: int,
) -> Dict[str, Any]:
    """Subtask 34: dispersion-threshold F1 sweep against holdout labels.

    Stub for Subtask 32. The Subtask 34 commit replaces the body with
    the real per-threshold confusion-matrix sweep that picks the
    dispersion threshold maximising F1 on the holdout corpus.
    """
    return {
        "calibrated": False,
        "reason": "sweep_dispersion_threshold landed in Subtask 34",
        "sweep_range": [sweep_from, sweep_to],
        "steps": steps,
    }


if __name__ == "__main__":
    raise SystemExit(main())
