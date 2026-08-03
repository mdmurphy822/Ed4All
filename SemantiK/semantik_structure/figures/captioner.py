"""Generate accessible figure descriptions with SmolVLM2-256M.

Stage 6b runs after specialist authoring and before the accessibility gates.
It processes figure regions whose payload carries ``image_png_bytes`` from
Stage 5c and attaches two descriptions:

  * ``alt_text``               : short caption — WCAG SC 1.1.1 target.
  * ``extended_description``   : longer caption for the extended description.

The assembler's figure emitter reads ``alt_text`` and uses an empty alt when
the field is absent.

SmolVLM2-256M-Video-Instruct is Apache-2.0, transformers-native, and supports
single-image prompts. Its compact footprint allows the caption stage to share
the SemantiK runtime with the region specialists.

A model-load or generation failure raises :class:`FigureCaptionError` so the
cascade surfaces the failure rather than producing empty alt downstream.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import replace
from functools import lru_cache
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)


class FigureCaptionError(RuntimeError):
    """SmolVLM2 captioning failed for a figure Region."""


# Remove generated numeric claims that the source caption does not support.
# The evaluator uses the same token pattern so acceptance and emission agree.
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Supply a number-free accessible name when both guarded alt text and the
# source caption are unavailable.
TYPE_LEVEL_ALT = "Figure."


def _numeric_tokens(text: str) -> set[str]:
    """Comma-normalized numeric tokens (``1,000`` == ``1000``)."""
    return {m.replace(",", "") for m in _NUM_RE.findall(text or "")}


def strip_numeric_hallucinations(text: str, ground_truth: str) -> str:
    """Drop sentences whose numbers are absent from ``ground_truth``.

    Returns the surviving sentences joined back; may be ``""`` if every
    sentence carried an invented number. Sentences with no numbers, or whose
    numbers all appear in ``ground_truth``, are kept verbatim.
    """
    if not text or not text.strip():
        return ""
    gt_nums = _numeric_tokens(ground_truth)
    kept = [
        s for s in _SENT_SPLIT_RE.split(text.strip())
        if s.strip() and not (_numeric_tokens(s) - gt_nums)
    ]
    return " ".join(s.strip() for s in kept).strip()


def guard_figure_alt(alt: str, caption: str) -> str:
    """Sanitize a model-generated figure alt against its caption.

    Strips invented-number sentences; if nothing survives, falls back to a
    type-level alt ONLY when there is no caption to carry the figure's
    accessible name (an ``<img alt="">`` inside a ``<figure>`` with a
    ``<figcaption>`` is already WCAG-conformant, so we leave it empty there).

    An ALREADY-empty alt (a decorative figure, or one Stage 6b never captioned)
    is returned empty — we never fabricate an alt that wasn't there.

    NOTE: this is the NO-CAPTION path. When a caption exists, the emitter uses
    :func:`alt_from_caption` instead (caption-first) — the caption is a more
    trustworthy accessible name than a 256M model alt, and the numeric guard
    only catches invented *numbers*, not invented objects/relationships.
    """
    if not (alt or "").strip():
        return ""
    cleaned = strip_numeric_hallucinations(alt, caption)
    if cleaned:
        return cleaned
    return "" if (caption or "").strip() else TYPE_LEVEL_ALT


# Keep caption-first accessible names consistent between the production emitter
# and the acceptance evaluator by sharing this label and sentence trim.
_FIG_LABEL_RE = re.compile(
    r"^\s*(?:figure|fig\.?)\s*\d+\s*[:.\)]\s*", re.IGNORECASE
)
_FIRST_SENTENCE_RE = re.compile(r"(.+?[.!?])(?:\s|$)")


def alt_from_caption(caption: str) -> str:
    """Derive a short, caption-first alt from a resolved ``<figcaption>``.

    Strips a leading ``Figure N:`` / ``Fig. N.`` label and returns the first
    sentence (falling back to the whole caption when there is no sentence
    terminator). The result is the accessible name a captioned figure ships —
    the model contributes only the longer ``extended_description``.
    """
    c = _FIG_LABEL_RE.sub("", caption or "")
    m = _FIRST_SENTENCE_RE.match(c)
    return (m.group(1) if m else c).strip()


SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
_ALT_PROMPT = (
    "Briefly describe this figure for accessible alt text (one sentence)."
)
_EXTENDED_PROMPT = (
    "Describe this figure in detail, including any visible data trends, "
    "labels, axes, and key elements. Suitable for an extended accessibility "
    "description."
)
_ALT_MAX_NEW_TOKENS = 64
_EXTENDED_MAX_NEW_TOKENS = 256


@lru_cache(maxsize=1)
def _smolvlm_model_and_processor():
    """Load SmolVLM2 once per process. Lazy (avoids cost in non-figure tests)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    proc = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        SMOLVLM_MODEL_ID, dtype=dtype,
    )
    if use_cuda:
        model = model.to("cuda")
    model.eval()
    return proc, model


def _run_smolvlm_caption(image_bytes: bytes, prompt: str,
                        *, max_new_tokens: int) -> str:
    """Run SmolVLM2 with one image + one prompt; return the decoded caption."""
    import torch
    from PIL import Image

    proc, model = _smolvlm_model_and_processor()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    # SmolVLM2: image is embedded inline in the message content (not a separate
    # `images=` kwarg — that double-passes and raises "multiple values for
    # keyword argument 'images'").
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    # Strip the prompt prefix: decode only the new tokens past the input length.
    n_in = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0][n_in:]
    text = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Some assistants prefix with "Assistant:" / role marker — strip if present.
    for prefix in ("Assistant:", "assistant:"):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    return text


# Emit one replayable decision per figure without making telemetry availability
# a prerequisite for caption generation.


def _figure_caption_course_code() -> str:
    """Course code for the figure-caption capture (best-effort context)."""
    raw = (
        os.environ.get("SEMANTIK_COURSE_CODE")
        or os.environ.get("ED4ALL_COURSE_CODE")
        or ""
    ).strip()
    return raw or "SEMANTIK"


def _build_caption_capture():
    """Construct a best-effort DecisionCapture for the VLM caption call site.

    Returns ``None`` when ``lib.decision_capture`` is unavailable or capture
    construction fails. Events use the ``semantik`` tool and
    ``semantik_conversion`` phase shared by conversion-stage decisions.
    """
    try:
        from lib.decision_capture import DecisionCapture

        return DecisionCapture(
            course_code=_figure_caption_course_code(),
            phase="semantik_conversion",
            tool="semantik",
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("figure-caption DecisionCapture unavailable (non-fatal): %s", exc)
        return None


def _log_caption_decision(
    capture,
    *,
    region_index: int,
    feature_block_indices: tuple,
    image_bytes: bytes,
    px_size,
    alt: str,
    extended: str,
    alt_max_new_tokens: int,
    extended_max_new_tokens: int,
) -> None:
    """Emit ONE ``alt_text_generation`` decision for a captioned figure.

    The dynamic rationale records the image hash, figure geometry, stable
    feature-block ID, model, token limits, and output lengths. Capture failures
    never propagate into the caption path.
    """
    if capture is None:
        return
    try:
        img_sha = hashlib.sha256(image_bytes or b"").hexdigest()[:16]
        min_fb = min(feature_block_indices) if feature_block_indices else None
        try:
            px = tuple(int(v) for v in (px_size or ()))
        except (TypeError, ValueError):
            px = ()
        rationale = (
            f"SmolVLM2 alt-text for figure region {region_index} "
            f"(min_fb={min_fb}, img_sha256={img_sha}, px_size={px}, "
            f"bytes={len(image_bytes or b'')}): model={SMOLVLM_MODEL_ID} "
            f"alt_max_new_tokens={alt_max_new_tokens} "
            f"extended_max_new_tokens={extended_max_new_tokens}; produced "
            f"alt_len={len(alt or '')} extended_len={len(extended or '')} chars "
            f"(numeric-hallucination guard applied downstream at emit)."
        )
        capture.log_decision(
            decision_type="alt_text_generation",
            decision=(
                f"caption figure region {region_index}: alt={(alt or '')[:80]!r}"
            ),
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "figure-caption DecisionCapture log failed (non-fatal) on region %s: %s",
            region_index, exc,
        )


def caption_figure_regions(regions: list[Any], *,
                           run_extended: bool = True) -> list[Any]:
    """Stage 6b — attach ``alt_text`` (+ optional ``extended_description``) to
    every figure Region whose payload carries ``image_png_bytes``.

    Non-figure Regions pass through unchanged. If no figure Regions are
    present, this is a no-op (no model load).

    Every figure reaching Stage 6b is captioned and guarded because the runtime
    has no decorative-vs-content classifier. A figure therefore ships with a
    caption-derived alt, a guarded model alt, or ``TYPE_LEVEL_ALT``. The
    evaluator retains ``is_decorative`` and ``route_figure`` so subtype-tagged
    inputs can measure decorative handling without changing production routing.

    Raises
    ------
    FigureCaptionError
        Any model-load or per-region generation failure. Per the
        no-silent-fallback discipline, surfaces rather than skipping.
    """
    fig_idxs = [i for i, r in enumerate(regions)
                if getattr(r, "kind", None) == "figure"]
    if not fig_idxs:
        return regions

    out = list(regions)
    # Reuse one capture sink while recording each captioned figure separately.
    capture = _build_caption_capture()
    for i in fig_idxs:
        region = out[i]
        payload = region.payload or {}
        img_bytes = payload.get("image_png_bytes")
        if not img_bytes:
            # Honor explicit render-skip markers while failing closed when a
            # figure unexpectedly lacks Stage-5c image data.
            skip_reason = payload.get("figure_render_skipped") or payload.get(
                "figure_render_degraded"
            )
            if skip_reason:
                logger.warning(
                    "figure region %d has no image_png_bytes (render "
                    "skipped: %s) — captioning skipped, type-level alt "
                    "ships",
                    i, skip_reason,
                )
                continue
            raise FigureCaptionError(
                f"figure region {i} has no payload['image_png_bytes'] — "
                f"did Stage 5c (image_extract) run?"
            )
        try:
            alt = _run_smolvlm_caption(
                img_bytes, _ALT_PROMPT, max_new_tokens=_ALT_MAX_NEW_TOKENS,
            )
            ext = (
                _run_smolvlm_caption(
                    img_bytes, _EXTENDED_PROMPT,
                    max_new_tokens=_EXTENDED_MAX_NEW_TOKENS,
                )
                if run_extended else ""
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't swallow
            raise FigureCaptionError(
                f"figure region {i}: {type(exc).__name__}: {exc}"
            ) from exc

        _log_caption_decision(
            capture,
            region_index=i,
            feature_block_indices=tuple(
                getattr(region, "feature_block_indices", ()) or ()
            ),
            image_bytes=img_bytes,
            px_size=payload.get("px_size"),
            alt=alt,
            extended=ext,
            alt_max_new_tokens=_ALT_MAX_NEW_TOKENS,
            extended_max_new_tokens=_EXTENDED_MAX_NEW_TOKENS,
        )

        new_payload = {
            **payload,
            "alt_text": alt,
            "alt_text_source": f"smolvlm2({SMOLVLM_MODEL_ID})",
        }
        if ext:
            new_payload["extended_description"] = ext
            new_payload["extended_description_source"] = (
                f"smolvlm2({SMOLVLM_MODEL_ID})"
            )
        out[i] = replace(region, payload=new_payload)
    return out
