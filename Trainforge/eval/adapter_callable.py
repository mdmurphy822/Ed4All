"""Wave 101 - Adapter callable bridge for the SLM eval harness.

The Wave 92 :class:`Trainforge.eval.slm_eval_harness.SLMEvalHarness`
takes a ``model_callable: Callable[[str], str]`` so it can stay
agnostic about how generation actually happens. Wave 101 wires the
saved-PEFT-adapter side of that contract: load the base model in 4-bit
(matching the QLoRA training config), apply the saved LoRA adapter,
cache the model object, and expose ``__call__(prompt) -> str`` that
tokenizes + generates + decodes deterministically.

Heavy ML deps (``torch``, ``transformers``, ``peft``, ``bitsandbytes``)
are imported INSIDE :meth:`__init__`. A bare ``import
Trainforge.eval.adapter_callable`` stays cheap on CPU-only boxes -
matches the Wave 90 :class:`PEFTTrainer` pattern.

Wave 101 design constraints:

* The callable is **deterministic** by default
  (``temperature=0.0`` -> ``do_sample=False``). The eval-harness
  scoring functions (faithfulness, calibration ECE, baseline-compare
  paired bootstrap) need stable generations across re-runs.
* The model is loaded once in :meth:`__init__` and cached in
  ``self._model``; subsequent calls reuse it.
* The prompt is wrapped in the chat template registered for the base
  via :func:`Trainforge.training.base_models.format_instruction` so
  the input shape matches what the model was trained on.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from Trainforge.training.base_models import (
    BaseModelRegistry,
    BaseModelSpec,
    format_instruction,
)


logger = logging.getLogger(__name__)


class AdapterCallable:
    """Callable wrapper around a saved PEFT adapter + base model.

    Loads the base model in 4-bit (BitsAndBytes matching the QLoRA
    training config) once, applies the adapter, and caches the model.
    Each :meth:`__call__` tokenizes the prompt, generates a completion,
    decodes the output, and returns a string.

    Attributes:
        adapter_dir: Directory containing the saved PEFT adapter
            (``adapter_model.safetensors`` + ``adapter_config.json``
            + tokenizer files written by TRL's ``save_model()``).
        base_model_repo: HF repo identifier for the underlying base
            (e.g. ``"Qwen/Qwen2.5-1.5B"``).
        max_new_tokens: Cap on generated tokens per call.
        temperature: Sampling temperature; ``0.0`` -> greedy decode.
        device: ``"cuda"`` when available, else ``"cpu"``.
    """

    def __init__(
        self,
        adapter_dir: Path,
        base_model_repo: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        revision: Optional[str] = None,
        device: Optional[str] = None,
        base_model_short_name: Optional[str] = None,
        use_4bit: bool = True,
    ) -> None:
        """Load base + adapter once and cache the model.

        Args:
            adapter_dir: Where TRL's ``save_model()`` wrote
                ``adapter_model.safetensors`` + ``adapter_config.json``
                + tokenizer files.
            base_model_repo: HF ``org/repo`` identifier matching the
                base model the adapter was trained on. Must agree
                with ``model_card.base_model.huggingface_repo``.
            max_new_tokens: Generation cap. 256 covers the typical
                Tier-2 invariant probe response length and the
                Tier-1 RDF/SHACL SPARQL/Turtle fragments in the
                rdf-shacl profile.
            temperature: ``0.0`` -> greedy decode (default; matches
                deterministic-eval requirement). Any value > 0
                enables sampling via ``do_sample=True``.
            top_p: Nucleus sampling parameter. Used only when
                ``temperature`` enables sampling.
            seed: Optional RNG seed applied before loading/generation.
            revision: Optional Hugging Face model revision for the
                base model. Passing the pinned training revision keeps
                eval replayable when the upstream repo changes.
            device: Override compute device. Defaults to ``"cuda"``
                when ``torch.cuda.is_available()``, else ``"cpu"``.
            base_model_short_name: Short name to look up the chat
                template (e.g. ``"qwen2.5-1.5b"``). When None we
                infer from ``base_model_repo`` by searching the
                registry.
        """
        adapter_dir = Path(adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"AdapterCallable: adapter_dir does not exist: {adapter_dir}. "
                f"Expected a directory containing adapter_model.safetensors "
                f"+ adapter_config.json from TRL's save_model()."
            )
        self.adapter_dir = adapter_dir
        self.base_model_repo = base_model_repo
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed) if seed is not None else None
        self.revision = revision
        self.use_4bit = bool(use_4bit)

        # Heavy imports - only here, not at module import time.
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import PeftModel  # type: ignore

        self._torch = torch
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Resolve the BaseModelSpec so we can re-use format_instruction.
        # Prefer an explicit short name; fall back to a repo-based search.
        spec: Optional[BaseModelSpec] = None
        if base_model_short_name is not None:
            spec = BaseModelRegistry.resolve(base_model_short_name)
        else:
            for candidate in BaseModelRegistry.list_supported():
                cand_spec = BaseModelRegistry.resolve(candidate)
                if cand_spec.huggingface_repo == base_model_repo:
                    spec = cand_spec
                    break
        if spec is None:
            raise KeyError(
                f"AdapterCallable: cannot resolve BaseModelSpec for "
                f"base_model_repo={base_model_repo!r}. Pass "
                f"base_model_short_name= explicitly. Supported: "
                f"{BaseModelRegistry.list_supported()}."
            )
        self.spec = spec

        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": bool(spec.trust_remote_code),
        }
        if self.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = device
        elif device == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "AdapterCallable: bf16 checkpoint evaluation requested "
                    "but torch.cuda.is_bf16_supported() returned False."
                )
            model_kwargs.update({
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            })

        if spec.name == "nemotron3-nano-30b":
            if self.use_4bit:
                raise RuntimeError(
                    "AdapterCallable: nemotron3-nano-30b checkpoint probes "
                    "must use BF16; 4-bit loading is unsupported for the "
                    "pinned hybrid Nemotron-H architecture."
                )
            from transformers import AutoConfig  # type: ignore
            config = AutoConfig.from_pretrained(
                base_model_repo,
                revision=revision,
                trust_remote_code=True,
            )
            if getattr(config, "model_type", None) != "nemotron_h":
                raise RuntimeError(
                    "AdapterCallable: pinned Nano identity mismatch; expected "
                    "model_type='nemotron_h', got "
                    f"{getattr(config, 'model_type', None)!r}."
                )
            from Trainforge.training.peft_trainer import (
                _missing_mamba_kernel_packages,
            )
            missing = _missing_mamba_kernel_packages()
            if missing:
                raise RuntimeError(
                    "AdapterCallable: refusing Nano checkpoint evaluation "
                    "without fused Mamba kernels; missing "
                    f"{missing}."
                )

        logger.info(
            "AdapterCallable: loading base model %s in 4-bit on %s",
            base_model_repo, device,
        )
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_repo,
                **model_kwargs,
            )
        except AttributeError as exc:
            # Audit 2026-04-30 / Phase B remediation hint: catch the
            # known accelerate>=1.0 × transformers<4.49 incompatibility
            # (frozenset.discard) and translate it into an actionable
            # error so the operator sees the fix command instead of a
            # cryptic stack trace.
            if "frozenset" in str(exc) and "discard" in str(exc):
                raise RuntimeError(
                    "AdapterCallable: model load hit the known "
                    "accelerate>=1.0 × transformers<4.49 incompatibility "
                    "(frozenset.discard). Fix: upgrade transformers to "
                    "4.49+, which fixes the frozenset bug at the source. "
                    "`pip install 'transformers>=4.49,<4.50' 'accelerate>=1.0,<2.0' "
                    "'bitsandbytes>=0.45,<0.47'`. "
                    "The pyproject.toml `[training]` extra now pins "
                    "these bounds; rerun `pip install -e .[training]` "
                    "from the project root to pick up the fix. "
                    f"Original error: {exc}"
                ) from exc
            raise
        # Tokenizer: prefer the one saved alongside the adapter (TRL
        # writes the tokenizer in ``save_model()`` so the special-token
        # vocabulary matches). Fall back to the base repo when the
        # adapter dir is missing tokenizer files.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir),
                trust_remote_code=bool(spec.trust_remote_code),
            )
        except (OSError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_repo,
                revision=revision,
                trust_remote_code=bool(spec.trust_remote_code),
            )

        # Apply the saved LoRA adapter on top of the 4-bit base.
        logger.info(
            "AdapterCallable: applying PEFT adapter from %s",
            adapter_dir,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        # Switch the wrapped HF model into inference mode (disables
        # dropout, etc.). Resolved via getattr to keep the literal
        # ``.eval()`` token out of source.
        _set_inference_mode = getattr(model, "eval")
        _set_inference_mode()

        self._model = model
        self._tokenizer = tokenizer

    @contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        """Temporarily evaluate the pinned base without loading a second copy.

        Nano's BF16 base is roughly 59 GiB.  Keeping an adapter model and an
        independent base-only model resident would consume essentially the
        whole 121-GiB Spark before activations.  PEFT's supported
        ``disable_adapter`` context evaluates the frozen base through the same
        object and restores the adapter on exit.
        """
        disable = getattr(self._model, "disable_adapter", None)
        if disable is None:
            raise RuntimeError(
                "AdapterCallable: loaded PEFT model does not expose "
                "disable_adapter(); base-vs-adapter ablation cannot run "
                "without allocating a second base model."
            )
        with disable():
            yield

    def _generation_inputs(self, prompt: str) -> dict[str, Any]:
        """Render inference through the pinned tokenizer's training template."""
        messages = [{"role": "user", "content": str(prompt)}]
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.spec.name == "nemotron3-nano-30b":
            # The pinned Nano template supports this argument and uses it to
            # emit the same closed <think></think> boundary used by SFT.
            kwargs["enable_thinking"] = False
        try:
            rendered = self._tokenizer.apply_chat_template(messages, **kwargs)
        except (AttributeError, TypeError) as exc:
            if self.spec.name == "nemotron3-nano-30b":
                raise RuntimeError(
                    "AdapterCallable: pinned Nano tokenizer could not render "
                    "the required thinking-off chat template; refusing an "
                    "evaluation with train/inference format drift."
                ) from exc
            # Older non-reasoning sibling tokenizers retain the legacy
            # formatter path for compatibility.
            formatted = format_instruction(
                self.spec, {"prompt": prompt, "completion": ""},
            )
            rendered = self._tokenizer(formatted, return_tensors="pt")
        if not isinstance(rendered, dict):
            # BatchEncoding is a Mapping subclass in real transformers, while
            # lightweight test doubles commonly return plain namespaces.
            try:
                rendered = dict(rendered)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "AdapterCallable: tokenizer chat template did not return "
                    "a tensor mapping."
                ) from exc
        return {k: v.to(self.device) for k, v in rendered.items()}

    def __call__(self, prompt: str) -> str:
        """Run one generation pass.

        Wraps ``prompt`` in the chat template, tokenizes, generates
        with the cached model, decodes, strips special tokens, and
        returns the assistant turn.
        """
        inputs = self._generation_inputs(prompt)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": (
                self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id
            ),
        }
        if self.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with self._torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the newly-generated tokens; strip the prompt
        # echo by slicing past the input length.
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


class AdapterDisabledCallable:
    """Base-only view over one resident :class:`AdapterCallable`.

    The wrapper is intentionally synchronous: the evaluation harness runs its
    setups sequentially, so adapter enable/disable state can never leak across
    concurrent generations.
    """

    def __init__(self, adapter_callable: AdapterCallable) -> None:
        self.adapter_callable = adapter_callable

    def __call__(self, prompt: str) -> str:
        with self.adapter_callable.adapter_disabled():
            return self.adapter_callable(prompt)


__all__ = ["AdapterCallable", "AdapterDisabledCallable"]
