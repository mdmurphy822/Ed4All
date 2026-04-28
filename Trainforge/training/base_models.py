"""Wave 90 — base model registry for ``Trainforge/training``.

Maps short-name keys (e.g. ``qwen2.5-1.5b``) to a :class:`BaseModelSpec`
carrying the Hugging Face repo, default revision, chat-template name,
and recommended hyperparameter defaults the runner uses to format
``training_specs/instruction_pairs.jsonl`` for the chosen base.

This is a pure-Python lookup table. No network IO. The runtime PEFT
trainer (``peft_trainer.py``) actually loads the tokenizer and weights;
this module only declares which weights to load.

Resolved chat templates for Wave 90:

* ``chatml`` — Qwen / SmolLM2 (``<|im_start|>role\\n…<|im_end|>``)
* ``llama3`` — Llama-3 family (``<|start_header_id|>role<|end_header_id|>…``)
* ``phi3`` — Phi-3.5 (``<|user|>\\n…<|end|>\\n<|assistant|>\\n…``)

Adding a new base = drop a row in :data:`_REGISTRY` plus, if needed,
extend :func:`format_instruction` with the new chat template.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BaseModelSpec:
    """Static metadata for one supported base model.

    Attributes:
        name: Short-name key (lowercase, dot/hyphen separators) the CLI
            and ``model_card.base_model.name`` use. e.g. ``qwen2.5-1.5b``.
        huggingface_repo: HF ``org/repo`` identifier.
        default_revision: HF revision (tag or commit sha) the runner
            pins by default. Card carries the same value in
            ``base_model.revision``.
        chat_template: One of ``"chatml"`` / ``"llama3"`` / ``"phi3"``.
            Drives :func:`format_instruction`.
        recommended_max_seq_length: Default ``training_config.max_seq_length``.
        recommended_lora_rank: Default LoRA ``r`` for QLoRA fits.
        recommended_lora_alpha: Default LoRA ``alpha``.
    """

    name: str
    huggingface_repo: str
    default_revision: str
    chat_template: str
    recommended_max_seq_length: int
    recommended_lora_rank: int
    recommended_lora_alpha: int


# ---------------------------------------------------------------------- #
# Registry                                                                #
# ---------------------------------------------------------------------- #
#
# Wave 90 ships 5 supported bases. The first entry (Qwen2.5-1.5B) is the
# default for textbook-to-course training because:
#   * Open weights (no HF gating, no HF_TOKEN required).
#   * Native ChatML template, plays nicely with TRL's SFTTrainer.
#   * 1.5B fits a single 24GB GPU with QLoRA at rank 16.
#
# Llama-3.2 + Phi-3.5 are gated; runner surfaces a clear error to set
# ``HF_TOKEN`` when those bases are selected (Wave 90 Risk register).

_REGISTRY: Dict[str, BaseModelSpec] = {
    "qwen2.5-1.5b": BaseModelSpec(
        name="qwen2.5-1.5b",
        huggingface_repo="Qwen/Qwen2.5-1.5B",
        default_revision="8faed761d45a263340a0528343f099c05c9a4323",
        chat_template="chatml",
        recommended_max_seq_length=2048,
        recommended_lora_rank=16,
        recommended_lora_alpha=32,
    ),
    "llama-3.2-1b": BaseModelSpec(
        name="llama-3.2-1b",
        huggingface_repo="meta-llama/Llama-3.2-1B",
        default_revision="main",
        chat_template="llama3",
        recommended_max_seq_length=2048,
        recommended_lora_rank=16,
        recommended_lora_alpha=32,
    ),
    "llama-3.2-3b": BaseModelSpec(
        name="llama-3.2-3b",
        huggingface_repo="meta-llama/Llama-3.2-3B",
        default_revision="main",
        chat_template="llama3",
        recommended_max_seq_length=2048,
        recommended_lora_rank=16,
        recommended_lora_alpha=32,
    ),
    "smollm2-1.7b": BaseModelSpec(
        name="smollm2-1.7b",
        huggingface_repo="HuggingFaceTB/SmolLM2-1.7B",
        default_revision="main",
        chat_template="chatml",
        recommended_max_seq_length=2048,
        recommended_lora_rank=16,
        recommended_lora_alpha=32,
    ),
    "phi-3.5-mini": BaseModelSpec(
        name="phi-3.5-mini",
        huggingface_repo="microsoft/Phi-3.5-mini-instruct",
        default_revision="main",
        chat_template="phi3",
        recommended_max_seq_length=4096,
        recommended_lora_rank=16,
        recommended_lora_alpha=32,
    ),
}


class BaseModelRegistry:
    """Read-only access to the supported-base table.

    Designed to be used as a class with classmethods (no instantiation
    needed). Mirrors the immutable-singleton pattern other Ed4All
    registries (``ChunkType``, ``BloomLevel``) use.
    """

    @classmethod
    def resolve(cls, name: str) -> BaseModelSpec:
        """Look up a base spec by short name.

        Raises:
            KeyError: if ``name`` is not in the registry. Lists all
                supported short names in the error message so the
                user sees which spelling to use.
        """
        spec = _REGISTRY.get(name)
        if spec is None:
            raise KeyError(
                f"Unknown base model {name!r}. Supported bases: "
                f"{cls.list_supported()}."
            )
        return spec

    @classmethod
    def list_supported(cls) -> List[str]:
        """Sorted list of supported short names."""
        return sorted(_REGISTRY.keys())

    @classmethod
    def has(cls, name: str) -> bool:
        return name in _REGISTRY


# ---------------------------------------------------------------------- #
# Chat-template formatters                                                #
# ---------------------------------------------------------------------- #


def _format_chatml(prompt: str, completion: str) -> str:
    """ChatML format — used by Qwen / SmolLM2.

    See https://github.com/openai/openai-python/blob/main/chatml.md
    """
    return (
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{completion}<|im_end|>"
    )


def _format_llama3(prompt: str, completion: str) -> str:
    """Llama-3 chat format — header_id wrappers + eot_id terminator."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{completion}<|eot_id|>"
    )


def _format_phi3(prompt: str, completion: str) -> str:
    """Phi-3 chat format — ``<|user|>`` / ``<|assistant|>`` / ``<|end|>``."""
    return (
        f"<|user|>\n{prompt}<|end|>\n"
        f"<|assistant|>\n{completion}<|end|>"
    )


_TEMPLATES = {
    "chatml": _format_chatml,
    "llama3": _format_llama3,
    "phi3": _format_phi3,
}


def format_instruction(spec: BaseModelSpec, pair: Dict[str, str]) -> str:
    """Render an instruction pair into the base model's chat template.

    Args:
        spec: The :class:`BaseModelSpec` whose ``chat_template`` selects
            the formatter.
        pair: A dict with at minimum ``prompt`` and ``completion`` keys
            (the shape ``Trainforge.generators.instruction_factory``
            emits). Extra keys (chunk_id, lo_refs, etc.) are ignored.

    Raises:
        KeyError: if ``pair`` is missing ``prompt`` or ``completion``.
        ValueError: if ``spec.chat_template`` is not a registered template.
    """
    if "prompt" not in pair or "completion" not in pair:
        raise KeyError(
            f"format_instruction requires pair with 'prompt' and 'completion'; "
            f"got keys {sorted(pair.keys())}"
        )
    formatter = _TEMPLATES.get(spec.chat_template)
    if formatter is None:
        raise ValueError(
            f"Unknown chat_template {spec.chat_template!r} for base "
            f"{spec.name!r}. Registered templates: {sorted(_TEMPLATES.keys())}"
        )
    return formatter(str(pair["prompt"]), str(pair["completion"]))
