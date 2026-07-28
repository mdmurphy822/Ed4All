"""SFT-D S8 — checkpoint-selection-by-downstream-probes scaffolding.

The SFT program (scratchpad/sft_data_program.md §C) is explicit that
checkpoint selection must be driven by DOWNSTREAM probes — 110-q gold
key-point coverage + sympy-verified assessment-answer correctness — using
held-out pair loss only as an OVERFITTING TRIPWIRE, never the selector. Pair
perplexity is a poor proxy for generation quality.

This module is the deterministic, CPU-only wiring for that:

* per-epoch checkpoint enumeration over the trainer's ``checkpoint-<step>``
  dirs (paired with ``save_total_limit`` retention in the SFTConfig),
* a ``ProbeRunner`` hook interface that consumes a checkpoint dir and returns
  a metric dict (the production course-aware GPU runner lives in
  ``checkpoint_probe.py``; this deterministic selector never loads a model),
* deterministic ``select_best_checkpoint`` with an overfit-preferring
  tie-break (earliest epoch wins a tie), and
* ``run_checkpoint_selection`` that ties it together and persists a
  ``checkpoint_selection.json`` audit record.

Nothing here imports torch / trl / transformers. A caller supplies a
``ProbeRunner`` (a callable / object) that owns the GPU work, so the selector
never fabricates a score.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Metric direction: True == higher-is-better. The two real probes both
# maximize; pair_loss (tripwire only) minimizes.
_METRIC_HIGHER_IS_BETTER = {
    "gold_keypoint_coverage": True,
    "sympy_correctness": True,
    "composite": True,
    "pair_loss": False,
    "held_out_pair_loss": False,
}

# Composite weighting when checkpoint_selection_metric == "composite".
_COMPOSITE_WEIGHTS = {
    "gold_keypoint_coverage": 0.5,
    "sympy_correctness": 0.5,
}

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass
class CheckpointProbeResult:
    """One evaluated checkpoint."""

    checkpoint_dir: Path
    step: int
    metrics: Dict[str, float] = field(default_factory=dict)

    def score(self, metric: str) -> Optional[float]:
        if metric == "composite":
            return combine_probe_metrics(self.metrics)
        val = self.metrics.get(metric)
        return float(val) if val is not None else None


# A ProbeRunner is any callable ``(checkpoint_dir: Path) -> Dict[str, float]``.
ProbeRunner = Callable[[Path], Dict[str, float]]


def enumerate_epoch_checkpoints(output_dir: Path) -> List[Tuple[int, Path]]:
    """Return ``(step, dir)`` for every ``checkpoint-<step>`` dir, step-sorted.

    Matches HF/TRL's native ``save_strategy="epoch"`` checkpoint dirs. Empty
    when none exist (a fresh / crashed run). Best-effort: an unreadable
    ``output_dir`` yields ``[]``.
    """
    out: List[Tuple[int, Path]] = []
    try:
        entries = list(Path(output_dir).iterdir())
    except OSError:
        return out
    for p in entries:
        if not p.is_dir():
            continue
        m = _CHECKPOINT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def combine_probe_metrics(metrics: Dict[str, float]) -> Optional[float]:
    """Weighted composite of gold_keypoint_coverage + sympy_correctness.

    Returns ``None`` when neither component is present. When only one is
    present, the composite is that component's value (weights renormalize over
    the present components) so a course without any sympy-verifiable items
    still selects on gold coverage alone.
    """
    present = {
        k: float(metrics[k])
        for k in _COMPOSITE_WEIGHTS
        if metrics.get(k) is not None
    }
    if not present:
        return None
    total_w = sum(_COMPOSITE_WEIGHTS[k] for k in present)
    if total_w <= 0:
        return None
    return sum(present[k] * _COMPOSITE_WEIGHTS[k] for k in present) / total_w


def select_best_checkpoint(
    results: List[CheckpointProbeResult],
    *,
    metric: str,
    higher_is_better: Optional[bool] = None,
) -> Optional[CheckpointProbeResult]:
    """Deterministically pick the best checkpoint on ``metric``.

    Tie-break: the EARLIER checkpoint (smaller step) wins — a narrow,
    repetitive corpus overfits with more steps (2310.06714), so given equal
    downstream quality we prefer the less-trained adapter. Checkpoints
    missing the metric are skipped. Returns ``None`` when no checkpoint has a
    scorable value.
    """
    if higher_is_better is None:
        higher_is_better = _METRIC_HIGHER_IS_BETTER.get(metric, True)
    scored: List[Tuple[float, int, CheckpointProbeResult]] = []
    for r in results:
        val = r.score(metric)
        if val is None:
            continue
        scored.append((val, r.step, r))
    if not scored:
        return None
    # Sort by (metric desc/asc), then step ASC (earlier wins ties).
    scored.sort(key=lambda t: (-t[0] if higher_is_better else t[0], t[1]))
    return scored[0][2]


def run_checkpoint_selection(
    output_dir: Path,
    probe_runner: ProbeRunner,
    *,
    metric: str = "gold_keypoint_coverage",
    higher_is_better: Optional[bool] = None,
    write_report: bool = True,
    early_stopping_patience: Optional[int] = None,
    strict: bool = False,
) -> Optional[CheckpointProbeResult]:
    """Run ``probe_runner`` over every epoch checkpoint and select the best.

    ``probe_runner`` owns the GPU work (loads each checkpoint, runs the gold
    key-point-coverage + sympy-correctness probes, returns a metric dict). A
    In compatibility mode a runner exception is logged and skipped. Production
    passes ``strict=True`` so any missing/invalid metric or failed checkpoint
    probe fails the run instead of silently selecting another epoch.

    Persists ``<output_dir>/checkpoint_selection.json`` (audit trail: per
    checkpoint metrics + the selected step + the selection metric) when
    ``write_report`` is set. Returns the selected
    :class:`CheckpointProbeResult` (or ``None`` when nothing scored).
    """
    checkpoints = enumerate_epoch_checkpoints(output_dir)
    if not checkpoints:
        logger.warning(
            "checkpoint_selection: no checkpoint-* dirs under %s; nothing to "
            "select (was save_strategy='epoch' + save_total_limit set?).",
            output_dir,
        )
        return None

    if metric not in _METRIC_HIGHER_IS_BETTER:
        raise ValueError(
            f"checkpoint_selection: unsupported metric {metric!r}; expected "
            f"one of {sorted(_METRIC_HIGHER_IS_BETTER)}"
        )
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError(
            "checkpoint_selection: early_stopping_patience must be >= 1 "
            f"when configured, got {early_stopping_patience!r}"
        )

    results: List[CheckpointProbeResult] = []
    stopped_after_step: Optional[int] = None
    best_seen: Optional[float] = None
    stale_epochs = 0
    direction = (
        _METRIC_HIGHER_IS_BETTER[metric]
        if higher_is_better is None
        else bool(higher_is_better)
    )
    for step, ckpt_dir in checkpoints:
        try:
            metrics = probe_runner(ckpt_dir)
        except Exception as exc:  # noqa: BLE001 - one bad checkpoint
            if strict:
                raise RuntimeError(
                    "checkpoint_selection: downstream probe failed for "
                    f"{ckpt_dir}: {type(exc).__name__}: {exc}"
                ) from exc
            logger.warning(
                "checkpoint_selection: probe_runner failed on %s (%s); "
                "skipping.", ckpt_dir, exc,
            )
            continue
        if not isinstance(metrics, dict):
            if strict:
                raise RuntimeError(
                    "checkpoint_selection: downstream probe returned "
                    f"{type(metrics).__name__}, not a metric mapping, for "
                    f"{ckpt_dir}"
                )
            logger.warning(
                "checkpoint_selection: probe_runner returned %s (not a dict) "
                "for %s; skipping.", type(metrics).__name__, ckpt_dir,
            )
            continue
        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        score = (
            combine_probe_metrics(numeric_metrics)
            if metric == "composite"
            else numeric_metrics.get(metric)
        )
        if score is None or not math.isfinite(float(score)):
            if strict:
                raise RuntimeError(
                    f"checkpoint_selection: required metric {metric!r} is "
                    f"missing or non-finite for {ckpt_dir}; returned keys="
                    f"{sorted(metrics)}"
                )
            logger.warning(
                "checkpoint_selection: required metric %r missing/non-finite "
                "for %s; skipping.",
                metric,
                ckpt_dir,
            )
            continue
        results.append(
            CheckpointProbeResult(
                checkpoint_dir=ckpt_dir,
                step=step,
                metrics=numeric_metrics,
            )
        )
        if early_stopping_patience is not None:
            current = float(score)
            improved = best_seen is None or (
                current > best_seen if direction else current < best_seen
            )
            if improved:
                best_seen = current
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= early_stopping_patience:
                    stopped_after_step = step
                    break

    selected = select_best_checkpoint(
        results, metric=metric, higher_is_better=higher_is_better,
    )
    if strict and selected is None:
        raise RuntimeError(
            f"checkpoint_selection: no checkpoint produced a valid {metric!r} "
            f"score under {output_dir}"
        )

    if write_report:
        _write_selection_report(
            output_dir,
            results,
            selected,
            metric,
            early_stopping_patience=early_stopping_patience,
            stopped_after_step=stopped_after_step,
        )
    return selected


def _apply_early_stopping_window(
    results: List[CheckpointProbeResult],
    *,
    metric: str,
    higher_is_better: Optional[bool],
    patience: Optional[int],
) -> Tuple[List[CheckpointProbeResult], Optional[int]]:
    """Return the chronological prefix retained by downstream early stopping.

    Probes are evaluated after TRL has flushed its epoch checkpoints.  Applying
    patience to their chronological scores produces the same selected adapter
    as an online callback, while avoiding model mutation inside Trainer's
    callback stack.  It does not claim to save training compute.
    """
    ordered = sorted(results, key=lambda item: item.step)
    if patience is None:
        return ordered, None
    direction = (
        _METRIC_HIGHER_IS_BETTER[metric]
        if higher_is_better is None
        else bool(higher_is_better)
    )
    best: Optional[float] = None
    stale = 0
    retained: List[CheckpointProbeResult] = []
    stopped_after: Optional[int] = None
    for result in ordered:
        score = result.score(metric)
        if score is None:
            continue
        retained.append(result)
        improved = best is None or (
            float(score) > best if direction else float(score) < best
        )
        if improved:
            best = float(score)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                stopped_after = result.step
                break
    return retained, stopped_after


def _write_selection_report(
    output_dir: Path,
    results: List[CheckpointProbeResult],
    selected: Optional[CheckpointProbeResult],
    metric: str,
    *,
    early_stopping_patience: Optional[int] = None,
    stopped_after_step: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {
        "selection_metric": metric,
        "selected_step": selected.step if selected else None,
        "selected_checkpoint": (
            str(selected.checkpoint_dir) if selected else None
        ),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_triggered": stopped_after_step is not None,
        "stopped_after_step": stopped_after_step,
        "checkpoints": [
            {
                "step": r.step,
                "checkpoint_dir": str(r.checkpoint_dir),
                "metrics": r.metrics,
                "score": r.score(metric),
            }
            for r in sorted(results, key=lambda r: r.step)
        ],
    }
    try:
        path = Path(output_dir) / "checkpoint_selection.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp.replace(path)
    except OSError as exc:
        logger.warning(
            "checkpoint_selection: failed to write report under %s (%s).",
            output_dir, exc,
        )


def unwired_probe_runner(checkpoint_dir: Path) -> Dict[str, float]:
    """Default ProbeRunner placeholder — fails loud, never fabricates a score.

    The real gold-key-point-coverage + sympy-correctness probes are GPU work
    that runs OUT-OF-BAND. Until an operator wires a concrete runner, this
    default raises so a caller can't silently "select" on empty metrics.
    """
    raise NotImplementedError(
        "checkpoint_selection: no concrete ProbeRunner wired. The downstream "
        "probes (gold key-point coverage + sympy correctness) run out-of-band "
        f"on a GPU; supply a ProbeRunner that evaluates {checkpoint_dir} and "
        "returns {'gold_keypoint_coverage': float, 'sympy_correctness': float}."
    )


__all__ = [
    "CheckpointProbeResult",
    "ProbeRunner",
    "enumerate_epoch_checkpoints",
    "combine_probe_metrics",
    "select_best_checkpoint",
    "run_checkpoint_selection",
    "unwired_probe_runner",
]
