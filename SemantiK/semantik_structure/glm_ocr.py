"""Thin wrapper around zai-org/GLM-OCR for region-targeted OCR.

GLM-OCR is a 0.9B image-text-to-text model. The pipeline uses it only on
small regions where existing extractors are known to struggle:
  - Math blocks (pypdfium2 returns `(cid:NN)` garbage on CM math fonts)
  - Table cells (CID glyphs + weird column spacing confuse word grouping)

The model emits flat text with no bbox/font metadata, so it can't replace
the layout-aware extractors — it augments them by giving better text for
the region they already located.

License: MIT (model) + Apache 2.0 (optional PP-DocLayoutV3 pipeline). Both
commercial-OK. We don't use PP-DocLayoutV3 here — pdfplumber already
provides table layout.

Load once per process and reuse. Model is 0.9B params, ~1.8GB at fp16.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage  # noqa: F401


GLM_OCR_DEFAULT = "zai-org/GLM-OCR"

# The model card publishes these exact prompt strings; don't paraphrase —
# the prompt is part of the training contract.
PROMPT_TEXT = "Text Recognition:"
PROMPT_FORMULA = "Formula Recognition:"
PROMPT_TABLE = "Table Recognition:"


def load_glm_ocr(model_path: str = GLM_OCR_DEFAULT,
                 *, quantize_4bit: bool = False) -> dict:
    """Load GLM-OCR. Returns {'processor', 'model'}.

    `quantize_4bit` — the model is small enough (0.9B) that fp16 fits the
    3070 8GB easily. Enable only if co-loading with Qwen reasoner.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs: dict = {
        "torch_dtype": "auto",
        "device_map": {"": 0} if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True,
    }
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()
    return {"processor": processor, "model": model}


def unload_glm_ocr(glm: dict | None) -> None:
    import gc
    import torch
    if glm is None:
        return
    glm.pop("model", None)
    glm.pop("processor", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ocr_image(glm: dict, image, *, prompt: str = PROMPT_TEXT,
              max_new_tokens: int = 2048) -> str:
    """Run GLM-OCR on a single PIL image with the given prompt.

    `image` is a PIL.Image.Image. Scale the crop appropriately before
    calling — GLM-OCR is trained on document-page resolutions; very small
    crops (a single cell) may need upscaling.
    """
    import torch
    proc = glm["processor"]
    model = glm["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    # Older vision-LM processors return this; GLM-OCR's generate doesn't
    # consume it and some transformers versions error on the extra kwarg.
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)

    gen = out[0, inputs["input_ids"].shape[1]:]
    # Per the model card; leaving special tokens in lets us see any
    # truncation/EOS artifacts during prototyping. Strip later once we're
    # confident in output shape.
    text = proc.decode(gen, skip_special_tokens=True)
    return text.strip()
