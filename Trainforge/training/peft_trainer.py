"""Wave 90 — thin PEFT/QLoRA wrapper for ``Trainforge.training``.

Wraps :class:`trl.SFTTrainer` and (optionally) :class:`trl.DPOTrainer`
with the QLoRA defaults specified by the per-base
:class:`~Trainforge.training.configs.TrainingConfig` and the per-base
:class:`~Trainforge.training.base_models.BaseModelSpec`.

Heavy ML deps (``trl``, ``peft``, ``transformers``, ``bitsandbytes``,
``torch``) are imported INSIDE the methods. A bare
``import Trainforge.training.peft_trainer`` stays cheap on CPU-only
boxes; the deps are only required when one of the ``fit_*`` methods is
actually called. Missing-deps surface a clear
``RuntimeError("install with: pip install 'ed4all[training]'")``.

This module deliberately does **not** define its own training loop —
TRL's ``SFTTrainer`` / ``DPOTrainer`` are the trusted surface. The
wrapper's job is to format the dataset (chat-template-aware), build
the QLoRA config, and route the result to the run dir.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from Trainforge.training.base_models import (
    BaseModelRegistry,
    BaseModelSpec,
    format_instruction,
)
from lib.generation.stop_control import (
    GracefulStopRequested,
    StopPoller,
    check_stop,
)


logger = logging.getLogger(__name__)


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
# import) so it is unit-testable on a CPU-only box; ``_build_stop_callback``
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


def _require_training_deps() -> None:
    """Raise a single actionable error when any heavy dep is missing.

    We probe the imports rather than try/except per-method so the
    failure mode is consistent regardless of which dep happens to be
    missing first.
    """
    missing: List[str] = []
    for module in ("torch", "trl", "peft", "transformers", "bitsandbytes"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            f"PEFTTrainer requires the [training] extra. Missing: {missing}. "
            f"Install with: pip install 'ed4all[training]'."
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
    ) -> None:
        self.base_model = base_model
        self.spec: BaseModelSpec = BaseModelRegistry.resolve(base_model)
        self.training_config = dict(training_config)

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
                :func:`Trainforge.generators.instruction_factory.synthesize_instruction_pair`.
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
        # Graceful-stop preflight: a sentinel armed BEFORE the (expensive)
        # weight load / trainer build stops the run here with zero GPU work.
        check_stop("trainforge_train.fit_sft.preflight", 0)

        _require_training_deps()

        # Heavy imports — only reachable when deps are installed.
        import torch  # type: ignore
        from peft import LoraConfig, prepare_model_for_kbit_training  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore

        formatted_texts = [
            format_instruction(self.spec, pair) for pair in instruction_pairs
        ]
        # TRL >= 0.7 expects a HuggingFace `Dataset`; we lazy-import
        # `datasets` only when actually fitting.
        from datasets import Dataset  # type: ignore
        dataset = Dataset.from_dict({"text": formatted_texts})

        tokenizer = AutoTokenizer.from_pretrained(
            self.spec.huggingface_repo,
            revision=self.spec.default_revision,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        model_kwargs: Dict[str, Any] = {
            "revision": self.spec.default_revision,
        }
        use_4bit = bool(self.training_config.get("use_4bit", True))
        if use_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            model_kwargs.update({
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
            })

        model = AutoModelForCausalLM.from_pretrained(
            self.spec.huggingface_repo,
            **model_kwargs,
        )
        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        sft_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=int(self.training_config.get("epochs", 3)),
            per_device_train_batch_size=int(self.training_config.get("batch_size", 4)),
            gradient_accumulation_steps=int(
                self.training_config.get("gradient_accumulation_steps", 1)
            ),
            learning_rate=float(self.training_config.get("learning_rate", 2e-4)),
            warmup_ratio=float(self.training_config.get("warmup_ratio", 0.0)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
            optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
            seed=int(self.training_config.get("seed", 42)),
            max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
            save_strategy="epoch",
            logging_steps=10,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
        )
        # Graceful-stop seam: check the sentinel at every step/epoch boundary;
        # on stop the callback flushes a native checkpoint + halts cleanly.
        stop_cb = _build_stop_callback()
        if stop_cb is not None and hasattr(trainer, "add_callback"):
            trainer.add_callback(stop_cb)
        # Resume from a prior graceful-stop's native checkpoint when present;
        # only pass the kwarg when actually resuming so the fresh-run call
        # stays signature-compatible with the trusted TRL default.
        _resume = _resume_arg(output_dir)
        if _resume:
            trainer.train(resume_from_checkpoint=_resume)
        else:
            trainer.train()
        # Surface the pause BEFORE the final consolidated save_model (R5): the
        # native checkpoint is already on disk, so the run is resumable and the
        # runner must not emit a "completed" adapter / model card.
        _raise_if_stopped(stop_cb, "trainforge_train.fit_sft")
        trainer.save_model(str(output_dir))

        # TRL's save_model() writes adapter_model.safetensors (with
        # underscore). Wave 99 worker found the old "adapter.safetensors"
        # return value tripped the runner's adapter-presence guard
        # because the file on disk was actually adapter_model.safetensors.
        adapter_path = output_dir / "adapter_model.safetensors"
        return adapter_path

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
                :func:`Trainforge.generators.preference_factory.synthesize_preference_pair`
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
        # Graceful-stop preflight (see fit_sft): stop before the DPO weight
        # load when a sentinel is already armed.
        check_stop("trainforge_train.fit_dpo.preflight", 0)

        _require_training_deps()

        from datasets import Dataset  # type: ignore
        from peft import PeftModel  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import DPOConfig, DPOTrainer  # type: ignore

        rows = {
            "prompt": [pair["prompt"] for pair in preference_pairs],
            "chosen": [pair["chosen"] for pair in preference_pairs],
            "rejected": [pair["rejected"] for pair in preference_pairs],
        }
        dataset = Dataset.from_dict(rows)

        dpo_args = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=int(self.training_config.get("epochs", 3)),
            per_device_train_batch_size=int(self.training_config.get("batch_size", 4)),
            gradient_accumulation_steps=int(
                self.training_config.get("gradient_accumulation_steps", 1)
            ),
            learning_rate=float(self.training_config.get("learning_rate", 2e-4)),
            warmup_ratio=float(self.training_config.get("warmup_ratio", 0.0)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
            seed=int(self.training_config.get("seed", 42)),
        )

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
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(sft_model_dir))
        except (OSError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(
                self.spec.huggingface_repo,
                revision=self.spec.default_revision,
            )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_kwargs: Dict[str, Any] = {
            "revision": self.spec.default_revision,
        }
        use_4bit = bool(self.training_config.get("use_4bit", True))
        if use_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            base_kwargs.update({
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
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

        trainer = DPOTrainer(
            model=peft_model,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        # Graceful-stop seam (same contract as fit_sft). NB: DPO shares
        # ``output_dir`` with the SFT phase, so ``checkpoint-*`` here can
        # co-exist with SFT's — resume_from_checkpoint auto-selects the
        # highest global_step, which is the DPO one once DPO has stepped.
        stop_cb = _build_stop_callback()
        if stop_cb is not None and hasattr(trainer, "add_callback"):
            trainer.add_callback(stop_cb)
        _resume = _resume_arg(output_dir)
        if _resume:
            trainer.train(resume_from_checkpoint=_resume)
        else:
            trainer.train()
        _raise_if_stopped(stop_cb, "trainforge_train.fit_dpo")
        trainer.save_model(str(output_dir))
        return output_dir / "adapter_model.safetensors"


__all__ = ["PEFTTrainer"]
