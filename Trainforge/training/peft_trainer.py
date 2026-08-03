"""Wave 90 — thin PEFT/QLoRA wrapper for ``Trainforge.training``.

Wraps :class:`trl.SFTTrainer` and (optionally) :class:`trl.DPOTrainer`
with the QLoRA defaults specified by the per-base
:class:`~Trainforge.training.configs.TrainingConfig` and the per-base
:class:`~Trainforge.training.base_models.BaseModelSpec`.

Heavy ML deps (``trl``, ``peft``, ``transformers``, ``bitsandbytes``,
``torch``) are imported INSIDE the methods. A bare
``import Trainforge.training.peft_trainer`` stays cheap on CPU-only
environments; the dependencies are required only when a ``fit_*`` method is
actually called. Missing-deps surface a clear
``RuntimeError("install with: pip install 'ed4all[training]'")``.

This module deliberately does **not** define its own training loop —
TRL's ``SFTTrainer`` / ``DPOTrainer`` are the trusted surface. The
wrapper's job is to format the dataset (chat-template-aware), build
the QLoRA config, and route the result to the run dir.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from Trainforge.training.base_models import (
    BaseModelRegistry,
    BaseModelSpec,
    format_instruction,
)
from Trainforge.training.telemetry import (
    TrainingTelemetryWriter,
    build_training_telemetry_callback,
    tokenized_dataset_metrics,
)
from lib.generation.stop_control import (
    GracefulStopRequested,
    StopPoller,
    check_stop,
)


logger = logging.getLogger(__name__)

_TRAINER_CHECKPOINT_ROOT = ".trainer_checkpoints"
_TRAINER_STAGES = frozenset({"sft", "dpo"})


def _write_training_diagnostics(
    output_dir: Path,
    *,
    stage: str,
    trainer: Any,
    torch_module: Any,
    elapsed_s: float,
    pair_count: int,
    max_seq_length: int,
) -> None:
    """Persist measured trainer and memory telemetry without inventing values."""
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda and cuda.is_available())
    stage_data: Dict[str, Any] = {
        "elapsed_seconds": round(elapsed_s, 6),
        "pair_count": pair_count,
        "configured_max_seq_length": max_seq_length,
        "log_history": list(getattr(getattr(trainer, "state", None),
                                    "log_history", []) or []),
        "cuda_available": cuda_available,
    }
    if cuda_available:
        for name in ("max_memory_allocated", "max_memory_reserved"):
            metric = getattr(cuda, name, None)
            if callable(metric):
                stage_data[f"cuda_{name}_bytes"] = int(metric())
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        available = next(
            line for line in meminfo.splitlines()
            if line.startswith("MemAvailable:")
        )
        stage_data["host_mem_available_bytes_after"] = (
            int(available.split()[1]) * 1024
        )
    except (OSError, StopIteration, ValueError):
        stage_data["host_mem_available_bytes_after"] = None
    diagnostics_path = output_dir / "training_diagnostics.json"
    payload: Dict[str, Any] = {"schema_version": 1, "stages": {}}
    if diagnostics_path.exists():
        payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    payload.setdefault("stages", {})[stage] = stage_data
    temporary = diagnostics_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, diagnostics_path)


# --------------------------------------------------------------------------- #
# Graceful-stop seam (P4b — "checkpoint on command" for trainforge_train)      #
# --------------------------------------------------------------------------- #
#
# The trainer loop is TRL's SFTTrainer/DPOTrainer (HF ``transformers`` Trainer)
# — the framework exposes a per-step / per-epoch callback boundary via
# ``transformers.TrainerCallback``. The stop seam checks the SAME filesystem
# sentinel every other Ed4All stage checks (``lib.generation.stop_control``);
# when armed it asks the trainer to (1) flush its NATIVE ``checkpoint-<step>``
# resume dir and (2) stop cleanly, then surfaces the pause as
# ``GracefulStopRequested`` so the runner/executor mark the phase ``paused``
# (never ``completed``). Worst-case loss = the single in-flight step's gradient
# (< one optimizer update); resume replays from the native checkpoint.
#
# The reaction logic lives in ``_GracefulStopMixin`` (NO ``transformers``
# import) so it is unit-testable without accelerator dependencies; ``_build_stop_callback``
# mixes it in front of the real ``TrainerCallback`` at call time.


class _GracefulStopMixin:
    """Stop→checkpoint→stop reaction for the training ``TrainerCallback``.

    Split out from ``transformers.TrainerCallback`` so the decision is
    importable + testable without the heavy ML deps. The ``on_*`` overrides
    win the MRO when mixed in front of ``TrainerCallback`` (see
    :func:`_build_stop_callback`), so they replace the base no-op events.
    """

    def _init_stop(
        self,
        *,
        run_id: Optional[str] = None,
        min_interval_s: float = 5.0,
    ) -> None:
        # A tight per-step ``stat`` is cheap but wasteful; throttle it.
        self._stop_poller = StopPoller(min_interval_s=float(min_interval_s))
        self._stop_run_id = run_id
        self.stop_triggered = False
        self.stop_global_step = 0

    def _react_to_stop(self, args: Any, state: Any, control: Any) -> Any:
        # Track progress for the resume hint even on the no-stop path.
        self.stop_global_step = int(getattr(state, "global_step", 0) or 0)
        if self.stop_triggered:
            # Latch: once we've asked the trainer to stop, stay stopped —
            # never re-probe or clear the flags mid-flush.
            return control
        if self._stop_poller.should_stop(self._stop_run_id):
            # Flush the trainer-native checkpoint at THIS unit boundary, then
            # halt after the flush. Both flags are consumed by HF's Trainer.
            control.should_save = True
            control.should_training_stop = True
            self.stop_triggered = True
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._react_to_stop(args, state, control)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._react_to_stop(args, state, control)


def _build_stop_callback(
    *,
    run_id: Optional[str] = None,
    min_interval_s: float = 5.0,
) -> Any:
    """Construct a ``transformers.TrainerCallback`` wired to the stop sentinel.

    The ``transformers`` import is deferred here (never at module import) so a
    bare ``import Trainforge.training.peft_trainer`` stays CPU-cheap and the
    reaction logic in :class:`_GracefulStopMixin` remains testable without the
    ``[training]`` extra installed. Fail-soft: if ``TrainerCallback`` can't be
    resolved (a partial / stubbed ``transformers``), returns ``None`` and logs
    a warning — training then runs WITHOUT the stop seam rather than crashing.
    """
    try:
        from transformers import TrainerCallback  # type: ignore
    except ImportError:
        logger.warning(
            "PEFTTrainer: transformers.TrainerCallback unavailable; the "
            "graceful-stop seam is DISABLED for this run (training will not "
            "checkpoint-on-command)."
        )
        return None

    class _GracefulStopCallback(_GracefulStopMixin, TrainerCallback):  # type: ignore[misc]
        pass

    cb = _GracefulStopCallback()
    cb._init_stop(run_id=run_id, min_interval_s=min_interval_s)
    return cb


def _has_trainer_checkpoint(output_dir: Path) -> bool:
    """True iff a TRL/HF-native ``checkpoint-*`` dir already exists in ``output_dir``.

    Presence signals a prior graceful-stop (or crash) left a resumable native
    checkpoint. Best-effort: any filesystem error degrades to ``False`` (fresh).
    """
    try:
        return any(
            p.is_dir() and p.name.startswith("checkpoint-")
            for p in Path(output_dir).iterdir()
        )
    except OSError:
        return False


def _resume_arg(output_dir: Path) -> Optional[bool]:
    """``True`` (auto-detect latest ``checkpoint-*``) when a native trainer
    checkpoint from a prior graceful-stop exists, else ``None`` (fresh run).

    HF's ``Trainer.train(resume_from_checkpoint=True)`` auto-selects the
    highest-step ``checkpoint-*`` under ``output_dir``; ``None`` trains from
    scratch. Since the runner mints a provenance-keyed ``model_id`` (the same
    course + base + specs re-run into the SAME ``run_dir``), a resumed
    ``ed4all run trainforge_train`` lands here with the paused run's checkpoint
    already on disk and continues from it.
    """
    return True if _has_trainer_checkpoint(output_dir) else None


def _stage_checkpoint_dir(output_dir: Path, stage: str) -> Path:
    """Return the checkpoint namespace owned by one trainer stage.

    SFT and DPO use different trainer classes, optimizer states, and dataset
    shapes.  A native checkpoint from one is therefore not a valid resume
    source for the other.  Keeping both under the final adapter directory is
    useful for one-artifact backup/restore, but each stage gets an exclusive
    child directory.
    """
    normalized = str(stage).strip().lower()
    if normalized not in _TRAINER_STAGES:
        raise ValueError(
            f"unknown trainer stage {stage!r}; expected one of "
            f"{sorted(_TRAINER_STAGES)}"
        )
    return Path(output_dir) / _TRAINER_CHECKPOINT_ROOT / normalized


def _assert_no_legacy_shared_checkpoints(output_dir: Path) -> None:
    """Reject ambiguous pre-isolation checkpoints at the adapter root.

    Historical runs wrote both SFT and DPO ``checkpoint-*`` directories into
    ``output_dir``.  Their stage cannot be inferred safely from the directory
    name, so automatically moving or consuming them risks restoring the wrong
    trainer state.  Fail loudly and leave them untouched for an operator-led
    migration.
    """
    try:
        legacy = sorted(
            p.name
            for p in Path(output_dir).iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(
            f"could not inspect trainer output directory {output_dir} for "
            f"legacy shared checkpoints: {type(exc).__name__}: {exc}"
        ) from exc
    if legacy:
        raise RuntimeError(
            "ambiguous legacy trainer checkpoints found at the adapter root "
            f"{output_dir}: {legacy}. SFT and DPO checkpoints are now isolated "
            f"under {_TRAINER_CHECKPOINT_ROOT}/sft and "
            f"{_TRAINER_CHECKPOINT_ROOT}/dpo; do not auto-migrate these files "
            "because their originating stage is not encoded. Review or remove "
            "them before resuming."
        )


def _raise_if_stopped(stop_cb: Any, site_id: str) -> None:
    """Surface a graceful stop as ``GracefulStopRequested`` after ``train()``.

    Called immediately AFTER ``trainer.train()`` returns. By then the trainer
    has already flushed its native ``checkpoint-<step>`` (we set
    ``control.should_save`` at the stop boundary), so the run is resumable.
    Raising here — BEFORE the final ``save_model`` / model-card emit (risk R5)
    — means the phase is reported ``paused`` and never ``completed``.
    """
    if stop_cb is not None and getattr(stop_cb, "stop_triggered", False):
        raise GracefulStopRequested(
            site_id=site_id,
            units_completed=int(getattr(stop_cb, "stop_global_step", 0) or 0),
        )


def _promote_selected_checkpoint(selected_dir: Path, output_dir: Path) -> Path:
    """Copy the selected PEFT checkpoint's inference artifacts to run root."""
    selected_dir = Path(selected_dir)
    required = selected_dir / "adapter_model.safetensors"
    if not required.is_file():
        raise RuntimeError(
            "selected checkpoint is missing adapter_model.safetensors: "
            f"{selected_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    excluded = {
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
    }
    for source in selected_dir.iterdir():
        if not source.is_file() or source.name in excluded:
            continue
        destination = output_dir / source.name
        tmp = destination.with_suffix(destination.suffix + ".selected.tmp")
        shutil.copy2(source, tmp)
        tmp.replace(destination)
        copied += 1
    if copied == 0 or not (output_dir / required.name).is_file():
        raise RuntimeError(
            f"failed to promote selected checkpoint {selected_dir} to {output_dir}"
        )
    return output_dir / required.name


def _missing_mamba_kernel_packages() -> List[str]:
    """Names of the Mamba fast-path kernel packages that fail to import.

    ``modeling_nemotron_h.py`` gates its fused-kernel fast path on
    ``mamba_ssm`` + ``causal_conv1d`` (``is_fast_path_available``); when
    either is absent the model silently falls back to the eager
    ``torch_forward`` scan (~5x slower). We probe the SAME imports the
    modeling file probes so the preflight check matches the model's own
    gate. Monkeypatch seam for tests.
    """
    missing: List[str] = []
    try:
        from mamba_ssm.ops.triton.selective_state_update import (  # type: ignore # noqa: F401
            selective_state_update,
        )
        from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore # noqa: F401
            mamba_chunk_scan_combined,
            mamba_split_conv1d_scan_combined,
        )
        from mamba_ssm.ops.triton.layernorm_gated import (  # type: ignore # noqa: F401
            rmsnorm_fn,
        )
    except Exception as exc:  # noqa: BLE001 — binary/ABI failures matter too
        missing.append(f"mamba_ssm ({type(exc).__name__}: {exc})")
    try:
        from causal_conv1d import (  # type: ignore # noqa: F401
            causal_conv1d_fn,
            causal_conv1d_update,
        )
    except Exception as exc:  # noqa: BLE001 — binary/ABI failures matter too
        missing.append(f"causal_conv1d ({type(exc).__name__}: {exc})")
    return missing


def _require_training_deps() -> None:
    """Raise a single actionable error when any heavy dep is missing.

    We probe the imports rather than try/except per-method so the
    failure mode is consistent regardless of which dep happens to be
    missing first.

    ``bitsandbytes`` is only probed when 4-bit (QLoRA) loading is
    requested — the bf16 LoRA default path (``use_4bit=False``) never
    imports it, so a CUDA environment without a bitsandbytes wheel can still
    train the default recipe. Pass ``require_bnb=False`` to skip it.
    """
    return _require_training_deps_impl(require_bnb=True)


def _require_training_deps_impl(*, require_bnb: bool) -> None:
    modules = ["torch", "trl", "peft", "transformers"]
    if require_bnb:
        modules.append("bitsandbytes")
    missing: List[str] = []
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            f"PEFTTrainer requires the [training] extra. Missing: {missing}. "
            f"Install with: pip install 'ed4all[training]'."
        )


def _assert_supported_runtime() -> Dict[str, str]:
    """Validate the qualified dependency band before any weight allocation."""
    from Trainforge.training.runtime_preflight import (
        assert_supported_training_runtime,
    )

    return assert_supported_training_runtime()


# --------------------------------------------------------------------------- #
# trl API-version tolerance                                                    #
# --------------------------------------------------------------------------- #
#
# trl renamed two call-site surfaces between the 0.12 band this repo pins and
# the 1.x line:
#   * ``SFTTrainer(tokenizer=...)``   -> ``SFTTrainer(processing_class=...)``
#     (same for ``DPOTrainer``).
#   * ``SFTConfig(max_seq_length=...)`` -> ``SFTConfig(max_length=...)``.
# Rather than pin one spelling and break on the other, we introspect the
# INSTALLED class's signature at call time and pick the accepted name. The
# helpers degrade sanely for the CPU-only test doubles (a fake class whose
# ``__init__`` takes ``**kwargs`` accepts either name -> we return the modern
# default), and raise a clear, actionable error when NEITHER name matches a
# real signature that has no ``**kwargs`` escape hatch.


def _accepted_kwarg(
    target: Any,
    candidates: List[str],
    *,
    default: str,
    surface: str,
) -> str:
    """Return the first name in ``candidates`` the callable accepts.

    ``target`` is a class or callable; its ``__init__``/signature is
    introspected. First-match-wins so the OLD spelling is preferred when
    both somehow exist (keeps a mixed-version environment byte-stable). When the
    signature exposes ``**kwargs`` (VAR_KEYWORD) and no candidate matches
    explicitly (the test-double case), fall back to ``default``. When the
    signature is real, has no ``**kwargs``, and matches NONE of the
    candidates, raise ``RuntimeError`` naming the surface + the installed
    trl version so the failure is actionable.
    """
    import inspect

    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        # C-extension / builtin with no introspectable signature — trust
        # the modern default.
        return default
    params = sig.parameters
    for name in candidates:
        if name in params:
            return name
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_var_kw:
        return default
    trl_version = "unknown"
    try:  # pragma: no cover - optional trl dependency may be absent
        import trl  # type: ignore

        trl_version = getattr(trl, "__version__", "unknown")
    except ImportError:
        pass
    raise RuntimeError(
        f"Installed trl (version={trl_version}) exposes none of "
        f"{candidates!r} on {surface}. PEFTTrainer supports the trl 0.12 "
        f"('{candidates[-1] if candidates else '?'}') and trl 1.x "
        f"('{candidates[0] if candidates else '?'}') call surfaces; a "
        f"different major may need a new call-site adapter."
    )


def _filter_accepted(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys the callable's signature does not accept.

    Used for the optional SFTConfig knobs (``completion_only_loss``,
    ``save_total_limit``, ``load_best_model_at_end`` …) that only exist on
    newer trl/transformers. A ``**kwargs`` signature accepts everything, so
    nothing is dropped (the test double keeps working). Real classes drop
    the unknown keys so an old trl doesn't crash on a new knob.
    """
    import inspect

    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _assert_completion_only_masked(labels_rows: Any) -> None:
    """Loud pre-train assertion that the SFT collator masks the prompt.

    ``labels_rows`` is an iterable of per-example label sequences (ints);
    each token that must NOT contribute to the loss is the HF sentinel
    ``-100``. Completion-only masking means every training example has BOTH
    a masked region (the prompt tokens) AND at least one live token (the
    completion). This function fails LOUD (``RuntimeError``) when a sampled
    batch shows NO masked tokens at all — i.e. the model would train on
    prompt tokens, silently degrading a course-tutor adapter — or when a
    row is entirely masked (nothing to learn).

    Pure and dependency-free so it is unit-testable without an accelerator;
    caller extracts ``batch["labels"]`` from a real dataloader and hands the
    rows here.
    """
    rows = list(labels_rows)
    if not rows:
        raise RuntimeError(
            "completion-only loss-mask verification: sampled batch carried "
            "zero label rows — cannot confirm prompt masking before training."
        )
    any_masked = False
    for idx, row in enumerate(rows):
        seq = [int(t) for t in row]
        if not seq:
            raise RuntimeError(
                f"completion-only loss-mask verification: label row {idx} is "
                f"empty (no tokens)."
            )
        masked = sum(1 for t in seq if t == -100)
        live = len(seq) - masked
        if live == 0:
            raise RuntimeError(
                f"completion-only loss-mask verification: label row {idx} is "
                f"ENTIRELY masked (-100) — no completion tokens contribute to "
                f"the loss, so this example teaches nothing."
            )
        if masked > 0:
            any_masked = True
    if not any_masked:
        raise RuntimeError(
            "completion-only loss-mask verification FAILED: not a single "
            "prompt token is masked (-100) across the sampled batch. The "
            "trainer would compute loss over PROMPT tokens, which trains the "
            "adapter to parrot questions instead of answers. Confirm "
            "SFTConfig.completion_only_loss is honoured by the installed trl "
            "and that the chat template exposes an assistant-generation "
            "boundary."
        )


class PEFTTrainer:
    """QLoRA SFT (+ optional DPO) trainer for one base model.

    The trainer is constructed cheaply (no model load) and only
    actually loads weights / tokenizer when :meth:`fit_sft` or
    :meth:`fit_dpo` is called. This keeps unit tests that only care
    about API shape from needing a GPU.

    Attributes:
        base_model: Short name resolved against
            :class:`BaseModelRegistry`.
        spec: The :class:`BaseModelSpec` from the registry.
        training_config: dict view of
            :class:`~Trainforge.training.configs.TrainingConfig`.
    """

    def __init__(
        self,
        *,
        base_model: str,
        training_config: Dict[str, Any],
        course_dir: Optional[Path] = None,
        decision_capture: Optional[Any] = None,
        checkpoint_probe_factory: Optional[
            Callable[[str, Path], Callable[[Path], Dict[str, float]]]
        ] = None,
    ) -> None:
        self.base_model = base_model
        self.spec: BaseModelSpec = BaseModelRegistry.resolve(base_model)
        self.training_config = dict(training_config)
        self.course_dir = Path(course_dir) if course_dir is not None else None
        self.decision_capture = decision_capture
        self.checkpoint_probe_factory = checkpoint_probe_factory
        self._stage_selection_scores: Dict[str, float] = {}

    def _select_and_promote_checkpoint(
        self,
        *,
        stage: str,
        checkpoint_dir: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Select by configured downstream metric; ``None`` means legacy mode."""
        metric = str(
            self.training_config.get("checkpoint_selection_metric", "")
            or ""
        ).strip()
        if not metric:
            return None
        if self.checkpoint_probe_factory is not None:
            probe = self.checkpoint_probe_factory(stage, checkpoint_dir)
        else:
            if self.course_dir is None:
                raise RuntimeError(
                    "checkpoint selection is enabled but PEFTTrainer received "
                    "no course_dir for the held-out downstream probes"
                )
            from Trainforge.training.probes.checkpoint import CourseCheckpointProbe
            probe = CourseCheckpointProbe(
                course_dir=self.course_dir,
                stage=stage,
                state_root=output_dir / ".checkpoint_probe_state",
                spec=self.spec,
                training_config=self.training_config,
                metric=metric,
                capture=self.decision_capture,
            )
        from Trainforge.training.probes.checkpoint_selection import (
            _METRIC_HIGHER_IS_BETTER,
            run_checkpoint_selection,
        )
        selected = run_checkpoint_selection(
            checkpoint_dir,
            probe,
            metric=metric,
            early_stopping_patience=int(
                self.training_config.get("early_stopping_patience", 1)
            ),
            strict=True,
        )
        if selected is None:  # strict=True already makes this unreachable.
            raise RuntimeError(
                f"checkpoint selection produced no {stage} checkpoint"
            )
        selected_score = selected.score(metric)
        if selected_score is None:
            raise RuntimeError(
                f"selected {stage} checkpoint has no {metric!r} score"
            )
        if stage == "dpo":
            sft_score = self._stage_selection_scores.get("sft")
            if sft_score is None:
                raise RuntimeError(
                    "DPO checkpoint promotion requires the selected SFT "
                    "baseline score from the same run."
                )
            higher = _METRIC_HIGHER_IS_BETTER.get(metric, True)
            regressed = (
                selected_score < sft_score
                if higher else selected_score > sft_score
            )
            if regressed:
                raise RuntimeError(
                    "best DPO checkpoint regressed against the selected SFT "
                    f"baseline on {metric}: dpo={selected_score:.6f}, "
                    f"sft={sft_score:.6f}; refusing DPO promotion."
                )
        self._stage_selection_scores[stage] = float(selected_score)
        selection_report: Dict[str, Any] = {}
        selection_report_path = checkpoint_dir / "checkpoint_selection.json"
        try:
            selection_report = json.loads(
                selection_report_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            selection_report = {}
        sft_baseline = (
            self._stage_selection_scores.get("sft")
            if stage == "dpo" else None
        )
        TrainingTelemetryWriter(output_dir, stage=stage).emit(
            "selection",
            status="completed",
            global_step=int(selected.step),
            metrics={
                "selected_checkpoint": selected.checkpoint_dir.name,
                "probe_metric": metric,
                "probe_score": float(selected_score),
                "selected_sft_baseline": sft_baseline,
                "dpo_delta": (
                    float(selected_score) - float(sft_baseline)
                    if stage == "dpo" and sft_baseline is not None else None
                ),
                "early_stop_triggered": int(
                    selection_report.get("stopped_after_step") is not None
                ),
                "early_stop_after_step": selection_report.get(
                    "stopped_after_step"
                ),
            },
        )
        if self.decision_capture is not None:
            baseline = self._stage_selection_scores.get("sft")
            self.decision_capture.log_decision(
                decision_type="eval_run_decision",
                decision=(
                    f"Selected {stage} checkpoint {selected.checkpoint_dir.name} "
                    f"at {metric}={selected_score:.6f}."
                ),
                rationale=(
                    f"Downstream probe selected step={selected.step} for "
                    f"base={self.base_model}, stage={stage}, metric={metric}, "
                    f"score={selected_score:.6f}; SFT baseline="
                    f"{baseline if baseline is not None else 'not-applicable'}."
                ),
                alternatives_considered=[
                    {
                        "option": "Promote the final epoch without comparison",
                        "reason_rejected": (
                            f"The downstream {metric} probe selected step "
                            f"{selected.step} at {selected_score:.6f}; using the "
                            "final epoch would ignore the configured selection "
                            "evidence."
                        ),
                    },
                    {
                        "option": "Promote a DPO checkpoint below the SFT baseline",
                        "reason_rejected": (
                            f"Stage={stage} is evaluated against SFT baseline "
                            f"{baseline if baseline is not None else 'not-applicable'} "
                            f"on {metric}; promotion cannot bypass that comparison."
                        ),
                    },
                ],
            )
        return _promote_selected_checkpoint(selected.checkpoint_dir, output_dir)

    def _checkpoint_selection_enabled(self) -> bool:
        return bool(str(
            self.training_config.get("checkpoint_selection_metric", "") or ""
        ).strip())

    # ------------------------------------------------------------------ #
    # Hybrid-Mamba fast-path preflight                                    #
    # ------------------------------------------------------------------ #

    def _assert_mamba_fast_path(self) -> None:
        """Fail LOUD when a ``nemotron_h`` fit would run without fused kernels.

        The Nemotron-H hybrid (23/52 layers are Mamba-2 mixers) gates its
        fused scan on ``mamba_ssm`` + ``causal_conv1d`` being importable
        (``is_fast_path_available`` in the snapshot's modeling file). When
        they are absent the model does NOT error — it silently degrades to
        the eager ``torch_forward`` path, ~5x slower, which on a 30B multi-
        epoch fit is the difference between hours and days. Owner rule:
        loud failure beats silent 5x degrade.

        Gated so the other five registered bases are untouched: returns
        immediately unless the spec opts into ``trust_remote_code`` AND the
        resolved model config declares ``model_type == "nemotron_h"``. The
        config is resolved via ``AutoConfig`` (no weight load) so the check
        fires BEFORE the ~59 GB base is materialised. A config-resolution
        failure is logged and skipped — the model load right after will
        fail loudly on the same problem with a better traceback.
        """
        if not getattr(self.spec, "trust_remote_code", False):
            return
        try:
            from transformers import AutoConfig  # type: ignore

            model_config = AutoConfig.from_pretrained(
                self.spec.huggingface_repo,
                revision=self.spec.default_revision,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001 - preflight best-effort
            effective_cache = os.environ.get("HF_MODULES_CACHE", "")
            try:
                from transformers.dynamic_module_utils import (  # type: ignore
                    HF_MODULES_CACHE,
                )
                effective_cache = str(HF_MODULES_CACHE)
            except Exception:  # noqa: BLE001 — retain env-derived diagnostic
                effective_cache = effective_cache or "<transformers default>"
            raise RuntimeError(
                "refusing trust_remote_code model load because the pinned "
                f"AutoConfig preflight failed for repo="
                f"{self.spec.huggingface_repo!r}, "
                f"revision={self.spec.default_revision!r}, "
                f"effective_HF_MODULES_CACHE={effective_cache!r}: "
                f"{type(exc).__name__}: {exc}. Set HF_MODULES_CACHE to a "
                "writable directory before Python starts; the run environment "
                "template provides a portable default."
            ) from exc
        model_type = getattr(model_config, "model_type", None)
        if self.base_model == "nemotron3-nano-30b" and model_type != "nemotron_h":
            raise RuntimeError(
                "pinned Nano AutoConfig identity mismatch: expected "
                f"model_type='nemotron_h' for repo="
                f"{self.spec.huggingface_repo!r}, "
                f"revision={self.spec.default_revision!r}, got "
                f"{model_type!r}; refusing to load or train a differently "
                "resolved architecture."
            )
        if model_type != "nemotron_h":
            return
        missing = _missing_mamba_kernel_packages()
        if missing:
            raise RuntimeError(
                f"Refusing to fit base {self.base_model!r} "
                f"(model_type=nemotron_h): the Mamba fused-kernel packages "
                f"{missing} are not importable, so the model would silently "
                f"fall back to the eager torch scan (~5x slower) for the 23 "
                f"Mamba-2 layers. Install mamba_ssm + causal_conv1d first — "
                f"on aarch64 (e.g. DGX Spark) there are no prebuilt wheels; "
                f"both must be built from source against the installed "
                f"torch/CUDA. NB: environment changes are deferred while a "
                f"production pipeline is live — schedule the kernel build "
                f"with the environment upgrade window."
            )

    # ------------------------------------------------------------------ #
    # SFT                                                                 #
    # ------------------------------------------------------------------ #

    def fit_sft(
        self,
        instruction_pairs: List[Dict[str, Any]],
        output_dir: Path,
    ) -> Path:
        """Fit a QLoRA SFT adapter and return the saved adapter path.

        Args:
            instruction_pairs: List of pair dicts as emitted by
                :func:`Trainforge.generators.pairs.instruction.synthesize_instruction_pair`.
                Must carry ``prompt`` and ``completion`` keys at
                minimum.
            output_dir: The run dir that hosts both the adapter file
                and TRL's checkpoint dirs.

        Returns:
            Path to ``output_dir / "adapter_model.safetensors"`` (the
            consolidated adapter the runner will hash + record in
            ``model_card.json``). NOTE: TRL ``save_model()`` writes the
            file as ``adapter_model.safetensors`` (with underscore), not
            ``adapter.safetensors`` — the Wave 100 fix renames the
            returned path to match.
        """
        _assert_no_legacy_shared_checkpoints(output_dir)
        checkpoint_dir = _stage_checkpoint_dir(output_dir, "sft")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Graceful-stop preflight: a sentinel armed BEFORE the (expensive)
        # weight load / trainer build stops the run here with zero GPU work.
        check_stop("trainforge_train.fit_sft.preflight", 0)

        # bf16 LoRA is now the DEFAULT recipe (use_4bit=False); QLoRA stays
        # reachable behind config. Only probe for bitsandbytes when the 4-bit
        # path is requested so bf16 CUDA training does not require it.
        use_4bit = bool(self.training_config.get("use_4bit", False))
        _require_training_deps_impl(require_bnb=use_4bit)
        _assert_supported_runtime()

        # Heavy imports — only reachable when deps are installed.
        import torch  # type: ignore
        from peft import LoraConfig, prepare_model_for_kbit_training  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore

        # TRL >= 0.7 expects a HuggingFace `Dataset`; we lazy-import
        # `datasets` only when actually fitting.
        from datasets import Dataset  # type: ignore

        # nemotron_h preflight: fail loud (not silent 5x degrade) when the
        # Mamba fused kernels are absent — BEFORE any weight is loaded.
        self._assert_mamba_fast_path()

        # trust_remote_code is spec-driven (True only for bases whose arch
        # ships as repo-local modeling code, e.g. nemotron_h). The executed
        # code is the pinned local snapshot's own modeling file — consistent
        # with the HF-offline posture (revision-pinned, nothing fetched).
        tokenizer = AutoTokenizer.from_pretrained(
            self.spec.huggingface_repo,
            revision=self.spec.default_revision,
            trust_remote_code=self.spec.trust_remote_code,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Completion-only masking: when supported by the installed trl AND
        # requested (default on), hand trl RAW prompt/completion columns so
        # its data collator masks the prompt tokens (-100). trl applies the
        # tokenizer's chat_template — which matches ``format_instruction`` for
        # every registered base — so the on-the-wire text is equivalent while
        # the loss now covers ONLY the assistant completion. When the config
        # knob or the trl version doesn't support it, fall back to the proven
        # pre-formatted single-"text" column (full-sequence loss).
        completion_only = bool(
            self.training_config.get("completion_only_loss", True)
        )
        sftconfig_supports_completion_only = (
            "completion_only_loss"
            in _filter_accepted(
                SFTConfig, {"completion_only_loss": True}
            )
        )
        use_completion_only = completion_only and sftconfig_supports_completion_only
        if use_completion_only:
            dataset = Dataset.from_dict({
                "prompt": [pair["prompt"] for pair in instruction_pairs],
                "completion": [pair["completion"] for pair in instruction_pairs],
            })
            # ``format_instruction`` matches each registered tokenizer's chat
            # template, so length/truncation metrics describe the actual
            # rendered training sequence rather than raw string concatenation.
            telemetry_sequences = [
                format_instruction(self.spec, pair)
                for pair in instruction_pairs
            ]
            telemetry_completions = [
                pair["completion"] for pair in instruction_pairs
            ]
        else:
            formatted_texts = [
                format_instruction(self.spec, pair) for pair in instruction_pairs
            ]
            dataset = Dataset.from_dict({"text": formatted_texts})
            telemetry_sequences = formatted_texts
            telemetry_completions = None
        telemetry_dataset_metrics = tokenized_dataset_metrics(
            tokenizer=tokenizer,
            sequences=telemetry_sequences,
            completion_sequences=telemetry_completions,
            max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
        )

        lora_config = LoraConfig(
            r=int(self.training_config.get("lora_rank", self.spec.recommended_lora_rank)),
            lora_alpha=int(self.training_config.get(
                "lora_alpha", self.spec.recommended_lora_alpha,
            )),
            lora_dropout=float(self.training_config.get("lora_dropout", 0.05)),
            target_modules=list(self.training_config.get("target_modules") or [
                "q_proj", "v_proj",
            ]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        cuda_ok = bool(torch.cuda.is_available())
        bf16_ok = bool(cuda_ok and torch.cuda.is_bf16_supported())
        model_kwargs: Dict[str, Any] = {
            "revision": self.spec.default_revision,
            # Spec-driven; executes the pinned snapshot's own modeling file
            # (HF-offline posture — see the tokenizer comment above).
            "trust_remote_code": self.spec.trust_remote_code,
        }
        if use_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
            model_kwargs.update({
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
            })
        elif cuda_ok:
            # bf16 LoRA default: load the base in half precision on GPU
            # (no quantization) so the LoRA adapters train in bf16/fp16.
            model_kwargs.update({
                "device_map": "auto",
                "torch_dtype": torch.bfloat16 if bf16_ok else torch.float16,
            })

        model = AutoModelForCausalLM.from_pretrained(
            self.spec.huggingface_repo,
            **model_kwargs,
        )
        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        # SFTConfig field names moved across the trl 0.12 -> 1.x boundary:
        # ``max_seq_length`` -> ``max_length``. Resolve the accepted name.
        seq_len_field = _accepted_kwarg(
            SFTConfig,
            ["max_seq_length", "max_length"],
            default="max_length",
            surface="trl.SFTConfig(max_seq_length=/max_length=)",
        )
        sft_config_kwargs: Dict[str, Any] = {
            "output_dir": str(checkpoint_dir),
            "num_train_epochs": int(self.training_config.get("epochs", 3)),
            "per_device_train_batch_size": int(
                self.training_config.get("batch_size", 4)
            ),
            "gradient_accumulation_steps": int(
                self.training_config.get("gradient_accumulation_steps", 1)
            ),
            "learning_rate": float(self.training_config.get("learning_rate", 2e-4)),
            "warmup_ratio": float(self.training_config.get("warmup_ratio", 0.0)),
            "weight_decay": float(self.training_config.get("weight_decay", 0.0)),
            "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
            "bf16": bf16_ok,
            "fp16": bool(cuda_ok and not bf16_ok),
            "seed": int(self.training_config.get("seed", 42)),
            seq_len_field: int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
            "save_strategy": "epoch",
            "logging_steps": 10,
        }
        if self.training_config.get("max_steps") is not None:
            sft_config_kwargs["max_steps"] = int(
                self.training_config["max_steps"]
            )
        # Per-epoch checkpoint retention (S8 checkpoint-selection scaffolding):
        # keep every epoch's checkpoint so the downstream-probe selector can
        # pick the best one out-of-band. Defaults to config.epochs.
        _save_total_limit = self.training_config.get("save_total_limit")
        if _save_total_limit is not None:
            sft_config_kwargs["save_total_limit"] = int(_save_total_limit)
        # Activation/gradient checkpointing (large-base memory bound, e.g.
        # the 30B nemotron bf16 fit). use_reentrant=False is the
        # non-deprecated torch.utils.checkpoint mode and the only one that
        # composes with PEFT-frozen params without requires_grad hacks.
        # Both keys ride the accepted-kwargs shim below, so a trl/
        # transformers band lacking either degrades gracefully (key dropped).
        if bool(self.training_config.get("gradient_checkpointing", False)):
            sft_config_kwargs["gradient_checkpointing"] = True
            sft_config_kwargs["gradient_checkpointing_kwargs"] = {
                "use_reentrant": False,
            }
        if use_completion_only:
            sft_config_kwargs["completion_only_loss"] = True
            # Train-time thinking-off: trl>=0.25's prompt-completion path
            # renders examples through the tokenizer's chat template; for
            # reasoning-capable templates (nemotron_h's chat_template.jinja
            # defaults enable_thinking=True) that would open a dangling
            # `<think>\n` block on the generation prompt, teaching the
            # adapter to emit reasoning tokens. Passing enable_thinking=False
            # renders the closed `<think></think>` sentinel instead.
            # Non-reasoning templates (qwen2.5 chatml etc.) ignore the kwarg.
            # Rides the accepted-kwargs shim: older trl without
            # SFTConfig.chat_template_kwargs drops the key (that band uses
            # the hand-rolled _format_chatml text path anyway — unchanged).
            sft_config_kwargs["chat_template_kwargs"] = {
                "enable_thinking": False,
            }
        # Drop any knob the installed trl's SFTConfig doesn't accept so an
        # older trl never crashes on a newer field.
        sft_config_kwargs = _filter_accepted(SFTConfig, sft_config_kwargs)
        sft_args = SFTConfig(**sft_config_kwargs)

        # ``tokenizer=`` -> ``processing_class=`` across the trl boundary.
        tok_field = _accepted_kwarg(
            SFTTrainer,
            ["processing_class", "tokenizer"],
            default="processing_class",
            surface="trl.SFTTrainer(tokenizer=/processing_class=)",
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=dataset,
            peft_config=lora_config,
            **{tok_field: tokenizer},
        )
        # Completion-only loss-mask verification (S8): sample a batch and
        # fail LOUD if the prompt tokens aren't masked before burning GPU
        # hours on a run that trains the adapter to parrot prompts.
        if use_completion_only and bool(
            self.training_config.get("verify_loss_mask", True)
        ):
            self._verify_completion_only_loss_mask(trainer)
        # Graceful-stop seam: check the sentinel at every step/epoch boundary;
        # on stop the callback flushes a native checkpoint + halts cleanly.
        stop_cb = _build_stop_callback()
        if stop_cb is not None and hasattr(trainer, "add_callback"):
            trainer.add_callback(stop_cb)
        # Resume from a prior graceful-stop's native checkpoint when present;
        # only pass the kwarg when actually resuming so the fresh-run call
        # stays signature-compatible with the trusted TRL default.
        _resume = _resume_arg(checkpoint_dir)
        telemetry_cb = build_training_telemetry_callback(
            run_dir=output_dir,
            stage="sft",
            pair_count=len(instruction_pairs),
            configured_max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
            torch_module=torch,
            resumed=bool(_resume),
            dataset_metrics=telemetry_dataset_metrics,
        )
        if not hasattr(trainer, "add_callback"):
            raise RuntimeError(
                "TRL SFTTrainer lacks add_callback; live training telemetry "
                "cannot be guaranteed"
            )
        trainer.add_callback(telemetry_cb)
        _train_started = time.monotonic()
        if _resume:
            trainer.train(resume_from_checkpoint=_resume)
        else:
            trainer.train()
        _write_training_diagnostics(
            output_dir,
            stage="sft",
            trainer=trainer,
            torch_module=torch,
            elapsed_s=time.monotonic() - _train_started,
            pair_count=len(instruction_pairs),
            max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
        )
        # Surface the pause BEFORE the final consolidated save_model (R5): the
        # native checkpoint is already on disk, so the run is resumable and the
        # runner must not emit a "completed" adapter / model card.
        _raise_if_stopped(stop_cb, "trainforge_train.fit_sft")
        if self._checkpoint_selection_enabled():
            # Release the training model before the downstream probe loads the
            # pinned base and a checkpoint, keeping peak model, activation, and
            # backend-workspace memory within the available accelerator budget.
            del trainer
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        selected_adapter = self._select_and_promote_checkpoint(
            stage="sft",
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )
        if selected_adapter is None:
            trainer.save_model(str(output_dir))

        # TRL's save_model() writes adapter_model.safetensors (with
        # underscore). Wave 99 worker found the old "adapter.safetensors"
        # return value tripped the runner's adapter-presence guard
        # because the file on disk was actually adapter_model.safetensors.
        adapter_path = selected_adapter or (
            output_dir / "adapter_model.safetensors"
        )
        return adapter_path

    # ------------------------------------------------------------------ #
    # Loss-mask verification                                              #
    # ------------------------------------------------------------------ #

    def _verify_completion_only_loss_mask(self, trainer: Any) -> None:
        """Decode a sampled train batch and assert the prompt is masked.

        Best-effort: pulls the first batch from ``trainer``'s real train
        dataloader and hands its ``labels`` to
        :func:`_assert_completion_only_masked`, which raises loud when no
        prompt token is masked. A missing/unavailable dataloader (a test
        double, or a trl that doesn't expose one) is logged and skipped —
        the verification never fabricates a pass, but it also never crashes
        a run over an introspection gap. A raised
        ``RuntimeError`` from the assertion itself PROPAGATES (that is the
        loud failure the step exists for).
        """
        if not hasattr(trainer, "get_train_dataloader"):
            logger.warning(
                "PEFTTrainer: trainer exposes no get_train_dataloader(); "
                "completion-only loss-mask verification SKIPPED (cannot "
                "sample a batch). Training proceeds."
            )
            return
        try:
            loader = trainer.get_train_dataloader()
            batch = next(iter(loader))
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - introspection best-effort
            logger.warning(
                "PEFTTrainer: could not sample a train batch for "
                "loss-mask verification (%s); SKIPPED. Training proceeds.",
                exc,
            )
            return
        labels = batch.get("labels") if isinstance(batch, dict) else None
        if labels is None:
            logger.warning(
                "PEFTTrainer: sampled batch carries no 'labels' key; "
                "completion-only loss-mask verification SKIPPED."
            )
            return
        try:
            rows = labels.tolist()  # torch.Tensor -> nested lists
        except AttributeError:
            rows = list(labels)
        _assert_completion_only_masked(rows)
        logger.info(
            "PEFTTrainer: completion-only loss-mask verification PASSED "
            "(sampled %d example(s); prompt tokens masked).",
            len(rows) if hasattr(rows, "__len__") else -1,
        )

    # ------------------------------------------------------------------ #
    # DPO                                                                 #
    # ------------------------------------------------------------------ #

    def fit_dpo(
        self,
        preference_pairs: List[Dict[str, Any]],
        sft_adapter_path: Path,
        output_dir: Path,
    ) -> Path:
        """Optional DPO chain on top of an existing SFT adapter.

        Args:
            preference_pairs: List of pair dicts from
                :func:`Trainforge.generators.pairs.preference.synthesize_preference_pair`
                / misconception-DPO emit. Must carry ``prompt``,
                ``chosen``, ``rejected`` keys.
            sft_adapter_path: Path returned by :meth:`fit_sft`. Wave 100
                accepts both the legacy file path
                (``output_dir/adapter_model.safetensors``) and a
                directory path; either way ``DPOTrainer`` is given the
                parent directory because TRL's ``DPOTrainer(model=...)``
                expects a model directory or HF repo ID, not a single
                weights file.
            output_dir: Run dir; the DPO adapter overwrites the SFT
                weights at ``output_dir / "adapter_model.safetensors"``.

        Returns:
            Path to the consolidated DPO+SFT adapter.
        """
        _assert_no_legacy_shared_checkpoints(output_dir)
        checkpoint_dir = _stage_checkpoint_dir(output_dir, "dpo")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Graceful-stop preflight (see fit_sft): stop before the DPO weight
        # load when a sentinel is already armed.
        check_stop("trainforge_train.fit_dpo.preflight", 0)

        use_4bit = bool(self.training_config.get("use_4bit", False))
        _require_training_deps_impl(require_bnb=use_4bit)
        _assert_supported_runtime()

        from datasets import Dataset  # type: ignore
        from peft import PeftModel  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import DPOConfig, DPOTrainer  # type: ignore

        # nemotron_h preflight (mirrors fit_sft): loud failure before any
        # weight load when the Mamba fused kernels are absent.
        self._assert_mamba_fast_path()

        rows = {
            "prompt": [pair["prompt"] for pair in preference_pairs],
            "chosen": [pair["chosen"] for pair in preference_pairs],
            "rejected": [pair["rejected"] for pair in preference_pairs],
        }
        dataset = Dataset.from_dict(rows)
        dpo_sequences: List[str] = []
        dpo_completions: List[str] = []
        for pair in preference_pairs:
            prompt = str(pair["prompt"])
            for key in ("chosen", "rejected"):
                completion = str(pair[key])
                dpo_sequences.append(f"{prompt}{completion}")
                dpo_completions.append(completion)

        # Hoisted ABOVE dpo_config_kwargs: these were previously defined
        # only after the tokenizer load (~40 lines below), so the dict
        # literal referenced them before assignment — a NameError on every
        # real fit_dpo invocation. Regression net:
        # Trainforge/tests/test_peft_trainer_nemotron.py::test_fit_dpo_*.
        cuda_ok = bool(torch.cuda.is_available())
        bf16_ok = bool(cuda_ok and torch.cuda.is_bf16_supported())
        dpo_learning_rate = self.training_config.get("dpo_learning_rate")
        if self.base_model == "nemotron3-nano-30b" and dpo_learning_rate is None:
            raise RuntimeError(
                "Nemotron Nano DPO requires an explicitly calibrated "
                "dpo_learning_rate. Run the documented short DPO canary and "
                "supply the selected value through --config-overrides; the "
                "SFT learning_rate is not reused implicitly."
            )
        if dpo_learning_rate is None:
            dpo_learning_rate = self.training_config.get(
                "learning_rate", 2e-4,
            )

        dpo_config_kwargs: Dict[str, Any] = {
            "output_dir": str(checkpoint_dir),
            "num_train_epochs": int(self.training_config.get("epochs", 3)),
            "per_device_train_batch_size": int(
                self.training_config.get("batch_size", 4)
            ),
            "gradient_accumulation_steps": int(
                self.training_config.get("gradient_accumulation_steps", 1)
            ),
            "learning_rate": float(dpo_learning_rate),
            "warmup_ratio": float(self.training_config.get("warmup_ratio", 0.0)),
            "weight_decay": float(self.training_config.get("weight_decay", 0.0)),
            "seed": int(self.training_config.get("seed", 42)),
            # bf16 LoRA default parity with fit_sft.
            "bf16": bf16_ok,
            "fp16": bool(cuda_ok and not bf16_ok),
            "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
            "save_strategy": "epoch",
        }
        if self.training_config.get("max_steps") is not None:
            dpo_config_kwargs["max_steps"] = int(
                self.training_config["max_steps"]
            )
        dpo_length_field = _accepted_kwarg(
            DPOConfig,
            ["max_length", "max_seq_length"],
            default="max_length",
            surface="trl.DPOConfig(max_length=/max_seq_length=)",
        )
        dpo_config_kwargs[dpo_length_field] = int(
            self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )
        )
        if bool(self.training_config.get("gradient_checkpointing", False)):
            dpo_config_kwargs["gradient_checkpointing"] = True
            dpo_config_kwargs["gradient_checkpointing_kwargs"] = {
                "use_reentrant": False,
            }
        dpo_config_kwargs["chat_template_kwargs"] = {
            "enable_thinking": False,
        }
        _dpo_save_total_limit = self.training_config.get("save_total_limit")
        if _dpo_save_total_limit is not None:
            dpo_config_kwargs["save_total_limit"] = int(_dpo_save_total_limit)
        dpo_args = DPOConfig(**_filter_accepted(DPOConfig, dpo_config_kwargs))

        # Wave 100: DPOTrainer expects a model directory or HF repo ID,
        # NOT a file path. Wave 90 mistakenly passed the
        # ``adapter_model.safetensors`` file string, which raised an
        # ``HFValidationError`` / ``OSError`` at DPOTrainer init time.
        # Resolve the parent directory so TRL can load the SFT-trained
        # adapter via the standard from_pretrained flow.
        sft_adapter_path = Path(sft_adapter_path)
        if sft_adapter_path.is_file():
            sft_model_dir = sft_adapter_path.parent
        else:
            sft_model_dir = sft_adapter_path

        # Wave 100: TRL 0.12+'s DPOTrainer requires `processing_class`
        # (the renamed tokenizer arg). The SFT save_model() path saves
        # the tokenizer alongside the adapter; base-model fallback
        # covers legacy SFT dirs that don't carry the tokenizer.
        # trust_remote_code is spec-driven (True only for repo-local-code
        # arches, e.g. nemotron_h); the executed code is the pinned local
        # snapshot's own modeling/tokenizer file (HF-offline posture).
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(sft_model_dir),
                trust_remote_code=self.spec.trust_remote_code,
            )
        except (OSError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(
                self.spec.huggingface_repo,
                revision=self.spec.default_revision,
                trust_remote_code=self.spec.trust_remote_code,
            )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        telemetry_dataset_metrics = tokenized_dataset_metrics(
            tokenizer=tokenizer,
            sequences=dpo_sequences,
            completion_sequences=dpo_completions,
            max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
        )

        # (cuda_ok / bf16_ok are computed once, above dpo_config_kwargs.)
        base_kwargs: Dict[str, Any] = {
            "revision": self.spec.default_revision,
            "trust_remote_code": self.spec.trust_remote_code,
        }
        if use_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
            base_kwargs.update({
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
            })
        elif cuda_ok:
            # bf16 LoRA default: half-precision base load on GPU, no quant.
            base_kwargs.update({
                "device_map": "auto",
                "torch_dtype": torch.bfloat16 if bf16_ok else torch.float16,
            })

        # Wave 100: stacking DPO on a saved PEFT-SFT adapter requires
        # loading the adapter via ``PeftModel.from_pretrained(...,
        # is_trainable=True)``. Passing the sft_model_dir as a string
        # to ``DPOTrainer(model=...)`` triggered
        # ``RuntimeError: element 0 of tensors does not require grad``
        # because TRL's auto-load path materialised a frozen merged
        # model rather than a trainable LoRA. Loading the adapter
        # explicitly + passing the live PeftModel object keeps the
        # LoRA layers trainable for the DPO update.
        base_model = AutoModelForCausalLM.from_pretrained(
            self.spec.huggingface_repo,
            **base_kwargs,
        )
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(sft_model_dir),
            is_trainable=True,
        )

        # ``tokenizer=`` -> ``processing_class=`` across the trl 0.12 -> 1.x
        # boundary (mirrors fit_sft). Resolve the accepted name rather than
        # pinning one spelling.
        dpo_tok_field = _accepted_kwarg(
            DPOTrainer,
            ["processing_class", "tokenizer"],
            default="processing_class",
            surface="trl.DPOTrainer(tokenizer=/processing_class=)",
        )
        trainer = DPOTrainer(
            model=peft_model,
            args=dpo_args,
            train_dataset=dataset,
            **{dpo_tok_field: tokenizer},
        )
        # Graceful-stop seam (same contract as fit_sft). DPO owns a distinct
        # checkpoint namespace, so it can never select an SFT optimizer/trainer
        # state merely because the SFT global step is numerically higher.
        stop_cb = _build_stop_callback()
        if stop_cb is not None and hasattr(trainer, "add_callback"):
            trainer.add_callback(stop_cb)
        _resume = _resume_arg(checkpoint_dir)
        telemetry_cb = build_training_telemetry_callback(
            run_dir=output_dir,
            stage="dpo",
            pair_count=len(preference_pairs),
            configured_max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
            torch_module=torch,
            resumed=bool(_resume),
            dataset_metrics=telemetry_dataset_metrics,
        )
        if not hasattr(trainer, "add_callback"):
            raise RuntimeError(
                "TRL DPOTrainer lacks add_callback; live training telemetry "
                "cannot be guaranteed"
            )
        trainer.add_callback(telemetry_cb)
        _train_started = time.monotonic()
        if _resume:
            trainer.train(resume_from_checkpoint=_resume)
        else:
            trainer.train()
        _write_training_diagnostics(
            output_dir,
            stage="dpo",
            trainer=trainer,
            torch_module=torch,
            elapsed_s=time.monotonic() - _train_started,
            pair_count=len(preference_pairs),
            max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
        )
        _raise_if_stopped(stop_cb, "trainforge_train.fit_dpo")
        if self._checkpoint_selection_enabled():
            del trainer
            del peft_model
            del base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        selected_adapter = self._select_and_promote_checkpoint(
            stage="dpo",
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )
        if selected_adapter is None:
            trainer.save_model(str(output_dir))
            selected_adapter = output_dir / "adapter_model.safetensors"
        return selected_adapter


__all__ = ["PEFTTrainer"]
