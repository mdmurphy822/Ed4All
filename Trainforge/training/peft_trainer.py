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

    ``bitsandbytes`` is only probed when 4-bit (QLoRA) loading is
    requested — the bf16 LoRA default path (``use_4bit=False``) never
    imports it, so a CUDA box without a bitsandbytes wheel can still
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
    both somehow exist (keeps a mixed-version box byte-stable). When the
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
    try:  # pragma: no cover - trl absent on CPU boxes
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

    Pure + dependency-free so it is unit-testable on a CPU-only box; the
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

        # bf16 LoRA is now the DEFAULT recipe (use_4bit=False); QLoRA stays
        # reachable behind config. Only probe for bitsandbytes when the 4-bit
        # path is actually requested so a plain bf16 CUDA box trains without it.
        use_4bit = bool(self.training_config.get("use_4bit", False))
        _require_training_deps_impl(require_bnb=use_4bit)

        # Heavy imports — only reachable when deps are installed.
        import torch  # type: ignore
        from peft import LoraConfig, prepare_model_for_kbit_training  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore

        # TRL >= 0.7 expects a HuggingFace `Dataset`; we lazy-import
        # `datasets` only when actually fitting.
        from datasets import Dataset  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            self.spec.huggingface_repo,
            revision=self.spec.default_revision,
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
        else:
            formatted_texts = [
                format_instruction(self.spec, pair) for pair in instruction_pairs
            ]
            dataset = Dataset.from_dict({"text": formatted_texts})

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
            "output_dir": str(output_dir),
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
        # Per-epoch checkpoint retention (S8 checkpoint-selection scaffolding):
        # keep every epoch's checkpoint so the downstream-probe selector can
        # pick the best one out-of-band. Defaults to config.epochs.
        _save_total_limit = self.training_config.get("save_total_limit")
        if _save_total_limit is not None:
            sft_config_kwargs["save_total_limit"] = int(_save_total_limit)
        if use_completion_only:
            sft_config_kwargs["completion_only_loss"] = True
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

        use_4bit = bool(self.training_config.get("use_4bit", False))
        _require_training_deps_impl(require_bnb=use_4bit)

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

        dpo_config_kwargs: Dict[str, Any] = {
            "output_dir": str(output_dir),
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
            "seed": int(self.training_config.get("seed", 42)),
            # bf16 LoRA default parity with fit_sft.
            "bf16": bf16_ok,
            "fp16": bool(cuda_ok and not bf16_ok),
            "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        }
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
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(sft_model_dir))
        except (OSError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(
                self.spec.huggingface_repo,
                revision=self.spec.default_revision,
            )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        cuda_ok = bool(torch.cuda.is_available())
        bf16_ok = bool(cuda_ok and torch.cuda.is_bf16_supported())
        base_kwargs: Dict[str, Any] = {
            "revision": self.spec.default_revision,
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
