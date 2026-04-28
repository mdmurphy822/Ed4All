"""Wave 90 — compute-backend abstraction for ``Trainforge.training``.

Two implementations:

* :class:`LocalBackend` — runs the trainer in-process on the calling
  machine. Requires a CUDA-capable GPU; raises a clear
  :class:`RuntimeError` if no GPU is visible.

* :class:`RunPodBackend` — STUBBED in Wave 90. The real implementation
  dispatches a job to the same RunPod account the NeMo captioning
  pipeline uses (single GPU billing surface, per the SLM training
  plan). Lands in a follow-up wave; calling :meth:`run` today raises
  :class:`NotImplementedError`.

The abstract :class:`ComputeBackend` is what the runner depends on, so
swapping backends is a one-line config change.
"""
from __future__ import annotations

import abc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


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
                "torch.cuda.is_available() returned False. Use "
                "RunPodBackend (Wave 90+ follow-up) or run on a GPU host."
            )

    def run(self, spec: TrainingJobSpec) -> TrainingJobResult:
        self._assert_gpu_available()
        # Heavy imports are deferred to the trainer module so importing
        # ``compute_backend`` stays cheap on CPU-only CI.
        from Trainforge.training.peft_trainer import PEFTTrainer

        trainer = PEFTTrainer(
            base_model=spec.base_model,
            training_config=spec.training_config,
        )
        # Wave 100: ensure the run-dir parent exists for TRL/PEFT
        # checkpoint emit. The actual adapter filename is owned by
        # PEFTTrainer.fit_sft (returns adapter_model.safetensors per
        # Bug 2 fix); we only need the directory here.
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
            # Wave 100 Bug 5 extension: DPO failure must not void the
            # SFT adapter. SFT has already saved adapter_model.safetensors
            # to disk; if DPO crashes (e.g. peft-DPO grad_fn drift,
            # OOM during the higher-memory DPO pass), the runner still
            # has a usable trained adapter to point the model card at.
            try:
                adapter_out = trainer.fit_dpo(
                    pref_pairs, sft_out, spec.output_dir,
                )
                metrics["dpo_completed"] = True
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
# RunPodBackend (stub)                                                    #
# ---------------------------------------------------------------------- #


class RunPodBackend(ComputeBackend):
    """STUB — full implementation lands in a follow-up wave.

    Wave 90 only ships the abstract class + LocalBackend. The RunPod
    integration reuses the same RunPod account / API key the NeMo
    captioning pipeline uses (env: ``RUNPOD_API_KEY``); see the
    SLM training plan for the contract.

    Calling :meth:`run` raises :class:`NotImplementedError` with a
    pointer to the follow-up wave so the failure is not mysterious.
    """

    name = "runpod"

    def __init__(self, api_key: Optional[str] = None) -> None:
        # Read the env var eagerly so a misconfigured runner fails at
        # construction time, not at job dispatch time.
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")

    def run(self, spec: TrainingJobSpec) -> TrainingJobResult:
        raise NotImplementedError(
            "RunPodBackend.run is stubbed in Wave 90. The full RunPod "
            "dispatch integration (reusing the NeMo captioning RunPod "
            "account) lands in a follow-up wave. Use LocalBackend on a "
            "CUDA host until then."
        )


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list:
    """Tiny JSONL reader — local to compute_backend so we don't pull
    a heavy dataset library in until the trainer actually fits."""
    import json
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _filter_dpo_pairs(records: list, mode: str) -> list:
    """Filter DPO pairs down to the high-signal preference sources."""
    if mode in ("", "all", None):
        return records
    if mode != "editorial_or_misconception":
        raise ValueError(
            f"Unknown dpo_preference_filter={mode!r}; expected "
            "'all' or 'editorial_or_misconception'."
        )
    out = []
    for rec in records:
        source = str(rec.get("source") or rec.get("rejected_source") or "")
        if rec.get("misconception_id") or source in {
            "misconception",
            "misconception_editorial",
        }:
            out.append(rec)
    return out


__all__ = [
    "ComputeBackend",
    "LocalBackend",
    "RunPodBackend",
    "TrainingJobSpec",
    "TrainingJobResult",
]
