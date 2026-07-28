"""Post-promotion serving export for PEFT adapters.

TensorRT-LLM dynamic-LoRA support is architecture/version dependent. Ed4All
therefore never assumes that a Nemotron-H adapter can be attached dynamically:
the supported TRT path is a merged Hugging Face checkpoint, produced only
after promotion. vLLM may consume the unmerged adapter when its installed
release reports support for the exact architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal


def serving_artifact_mode(
    backend: Literal["trtllm", "vllm"],
    *,
    dynamic_lora_supported: bool,
) -> str:
    if backend == "trtllm":
        if dynamic_lora_supported:
            return "dynamic_adapter"
        return "merged_checkpoint"
    if backend == "vllm":
        return "dynamic_adapter" if dynamic_lora_supported else "merged_checkpoint"
    raise ValueError(f"unsupported serving backend: {backend}")


def merge_promoted_adapter(
    *,
    base_model: str,
    revision: str,
    adapter_dir: Path,
    output_dir: Path,
    trust_remote_code: bool,
) -> Path:
    """Merge a promoted adapter into one BF16 HF checkpoint.

    This is intentionally explicit and heavyweight; callers must schedule it
    after promotion with enough RAM and disk. It never overwrites the adapter.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty merged export: {output_dir}"
        )
    from peft import PeftModel  # type: ignore
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


__all__ = ["merge_promoted_adapter", "serving_artifact_mode"]
