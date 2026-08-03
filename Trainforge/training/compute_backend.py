"""Compute-backend abstraction for ``Trainforge.training``.

The :class:`LocalBackend` runs the trainer in-process on the calling
  machine. Requires a CUDA-capable GPU; raises a clear
  :class:`RuntimeError` if no GPU is visible.

The runner depends on the abstract :class:`ComputeBackend`, preserving an
injection seam for tested alternative implementations without exposing an
unimplemented backend to operators.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from lib.generation.stop_control import GracefulStopRequested
from lib.utils import read_jsonl as _utils_read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class TrainingJobSpec:
    """Inputs the backend needs to dispatch one training job.

    Backends consume this; the runner constructs it from the
    LibV2-imported course state plus the resolved
    :class:`~Trainforge.training.configs.TrainingConfig`.
    """

    course_slug: str
    base_model: str
    instruction_pairs_path: Path
    preference_pairs_path: Path
    training_config: Dict[str, Any]
    output_dir: Path
    run_dpo: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJobResult:
    """What :meth:`ComputeBackend.run` returns.

    ``adapter_path`` is the on-disk weights file the runner hashes /
    points to from the model card. ``metrics`` is opaque per-backend
    diagnostics (training loss curves, GPU minutes, etc.).
    """

    adapter_path: Path
    metrics: Dict[str, Any] = field(default_factory=dict)


class ComputeBackend(abc.ABC):
    """Abstract dispatch surface for one training job.

    Concrete subclasses implement :meth:`run`. Backends are expected
    to be cheap to instantiate (the constructor must not phone home
    or import heavy ML deps) — instantiation happens in tests on
    CPU-only CI runners.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def run(self, spec: TrainingJobSpec) -> TrainingJobResult:
        """Dispatch the job described by ``spec`` and block until done.

        Returns a :class:`TrainingJobResult`. Implementations are
        responsible for surfacing actionable errors (no GPU, no auth,
        OOM, etc.) as ``RuntimeError`` rather than letting backend-
        specific exceptions leak out.
        """


# ---------------------------------------------------------------------- #
# LocalBackend                                                            #
# ---------------------------------------------------------------------- #


class LocalBackend(ComputeBackend):
    """Runs the trainer in-process via :class:`PEFTTrainer`.

    Surfaces a clear error when no CUDA device is visible — refusing
    rather than running on CPU is intentional: a 1B+ model fits CPU
    only at multi-day timescales, which we'd rather flag up-front
    than silently kick off.
    """

    name = "local"

    def __init__(self, *, allow_no_gpu: bool = False) -> None:
        """
        Args:
            allow_no_gpu: When True, skips the CUDA visibility check.
                Tests use this with ``dry_run=True`` to exercise the
                backend code path without requiring a GPU.
        """
        self._allow_no_gpu = allow_no_gpu

    def _assert_gpu_available(self) -> None:
        """Raise ``RuntimeError`` when no CUDA device is visible.

        Best-effort: when ``torch`` isn't installed (CPU-only dev box
        without the ``[training]`` extra) we treat that as "no GPU"
        and surface the same actionable error.
        """
        if self._allow_no_gpu:
            return
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LocalBackend.run requires the [training] extra "
                "(torch, trl, peft, transformers, bitsandbytes). "
                f"Install with: pip install 'ed4all[training]'. ({exc})"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LocalBackend.run requires a CUDA-capable GPU. "
                "torch.cuda.is_available() returned False. Run training "
                "on a CUDA-capable host."
            )

    def run(self, spec: TrainingJobSpec) -> TrainingJobResult:
        self._assert_gpu_available()
        # Heavy imports are deferred to the trainer module so importing
        # ``compute_backend`` stays cheap on CPU-only CI.
        from Trainforge.training.peft_trainer import PEFTTrainer

        trainer = PEFTTrainer(
            base_model=spec.base_model,
            training_config=spec.training_config,
            course_dir=spec.extra.get("course_dir"),
            decision_capture=spec.extra.get("decision_capture"),
        )
        # Create the run directory before TRL/PEFT emits checkpoints.
        # PEFTTrainer.fit_sft owns the adapter filename.
        spec.output_dir.mkdir(parents=True, exist_ok=True)

        sft_pairs = _read_jsonl(spec.instruction_pairs_path)
        sft_out = trainer.fit_sft(sft_pairs, spec.output_dir)
        adapter_out = sft_out
        metrics: Dict[str, Any] = {"backend": self.name}

        if spec.run_dpo and spec.preference_pairs_path.exists():
            pref_pairs = _read_jsonl(spec.preference_pairs_path)
            pref_pairs = _filter_dpo_pairs(
                pref_pairs,
                str(spec.training_config.get(
                    "dpo_preference_filter",
                    "editorial_or_misconception",
                )),
            )
            min_dpo_pairs = int(spec.training_config.get("min_dpo_pairs", 10))
            if len(pref_pairs) < min_dpo_pairs:
                raise RuntimeError(
                    "DPO requested but filtered preference-pair count "
                    f"{len(pref_pairs)} is below min_dpo_pairs={min_dpo_pairs}."
                )
            # Preserve the completed SFT adapter when the explicitly
            # configured DPO policy permits an SFT-only result.
            try:
                adapter_out = trainer.fit_dpo(
                    pref_pairs, sft_out, spec.output_dir,
                )
                metrics["dpo_completed"] = True
            except GracefulStopRequested:
                # A graceful stop during DPO is a PAUSE, not a DPO failure —
                # never route it into the SFT-fallback / dpo_fail_hard arm.
                # Propagate so the runner/executor mark the phase paused.
                raise
            except Exception as exc:  # noqa: BLE001
                if bool(spec.training_config.get("dpo_fail_hard", True)):
                    raise RuntimeError(
                        f"DPO chain failed after SFT; refusing SFT fallback "
                        f"because dpo_fail_hard=true ({type(exc).__name__}: {exc})."
                    ) from exc
                logger.warning(
                    "LocalBackend: DPO chain failed (%s: %s). Falling "
                    "back to SFT-only adapter at %s. Model card will "
                    "be emitted against the SFT weights.",
                    type(exc).__name__, exc, sft_out,
                )
                metrics["dpo_completed"] = False
                metrics["dpo_error"] = f"{type(exc).__name__}: {exc}"
                adapter_out = sft_out

        return TrainingJobResult(
            adapter_path=adapter_out,
            metrics=metrics,
        )

# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list:
    """Thin wrapper around :func:`lib.utils.read_jsonl` (W-D6).

    Preserves the historical signature so existing call sites in this
    module (``_filter_dpo_pairs`` etc.) continue to work unchanged.
    """
    return _utils_read_jsonl(path)


#: The ``source`` / ``rejected_source`` values the
#: ``editorial_or_misconception`` DPO filter admits.
#:
#: ``mined_rejection`` rows come from
#: ``Trainforge/synthesis_reject_mining.py::select_mined_pairs`` — an accepted
#: instruction completion as ``chosen`` against a near-miss REJECTED completion
#: for the same chunk / objectives / base prompt / citation contract / teacher.
#:
#: LOCKSTEP CONTRACT: this set and the copy consumed by
#: ``Trainforge/training/runner.py::_count_dpo_eligible_records`` must always
#: agree. ``runner`` imports this constant precisely so they cannot drift —
#: widening one alone gives the worst failure available here: the count says
#: "run DPO", the filter then drops everything, and ``dpo_fail_hard=True``
#: (``configs/__init__.py``) kills the run AFTER SFT already succeeded.
DPO_EDITORIAL_SOURCES = frozenset({
    "misconception",
    "misconception_editorial",
    "mined_rejection",
})


def is_dpo_editorial_record(rec: Any) -> bool:
    """Whether one preference record passes ``editorial_or_misconception``.

    The single predicate behind both the filter (here) and the pre-flight
    count (``runner._count_dpo_eligible_records``).
    """
    source = str(rec.get("source") or rec.get("rejected_source") or "")
    return bool(rec.get("misconception_id")) or source in DPO_EDITORIAL_SOURCES


def _filter_dpo_pairs(records: list, mode: str) -> list:
    """Filter DPO pairs down to the high-signal preference sources."""
    if mode in ("", "all", None):
        return records
    if mode != "editorial_or_misconception":
        raise ValueError(
            f"Unknown dpo_preference_filter={mode!r}; expected "
            "'all' or 'editorial_or_misconception'."
        )
    return [rec for rec in records if is_dpo_editorial_record(rec)]


__all__ = [
    "ComputeBackend",
    "LocalBackend",
    "TrainingJobSpec",
    "TrainingJobResult",
]
