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


logger = logging.getLogger(__name__)


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
        _require_training_deps()

        # Heavy imports — only reachable when deps are installed.
        import torch  # type: ignore  # noqa: F401
        from peft import LoraConfig  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
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

        lora_config = LoraConfig(
            r=int(self.training_config.get("lora_rank", self.spec.recommended_lora_rank)),
            lora_alpha=int(self.training_config.get(
                "lora_alpha", self.spec.recommended_lora_alpha,
            )),
            bias="none",
            task_type="CAUSAL_LM",
        )

        sft_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=int(self.training_config.get("epochs", 3)),
            per_device_train_batch_size=int(self.training_config.get("batch_size", 4)),
            learning_rate=float(self.training_config.get("learning_rate", 2e-4)),
            seed=int(self.training_config.get("seed", 42)),
            max_seq_length=int(self.training_config.get(
                "max_seq_length", self.spec.recommended_max_seq_length,
            )),
            save_strategy="epoch",
            logging_steps=10,
        )

        trainer = SFTTrainer(
            model=self.spec.huggingface_repo,
            args=sft_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
        )
        trainer.train()
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
        _require_training_deps()

        from datasets import Dataset  # type: ignore
        from peft import PeftModel  # type: ignore
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
            learning_rate=float(self.training_config.get("learning_rate", 2e-4)),
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
            revision=self.spec.default_revision,
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
        trainer.train()
        trainer.save_model(str(output_dir))
        return output_dir / "adapter_model.safetensors"


__all__ = ["PEFTTrainer"]
