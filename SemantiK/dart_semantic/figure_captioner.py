"""Stage 6b — figure captioner via SmolVLM2-256M (Apache-2.0, HF-maintained).

Plans/09 §2-4. After Stage 6 (``run_qwen_specialists``) and before Stage 7
(gates), iterate figure Regions whose payload carries ``image_png_bytes``
(from Stage 5c, ``image_extract.py``) and attach two captions to the payload:

  * ``alt_text``               : short caption — WCAG SC 1.1.1 target.
  * ``extended_description``   : longer caption for the extended description.

The assembler's figure HTML emitter reads ``alt_text`` (and falls back to an
empty alt when absent, preserving the prior behaviour).

Model choice rationale (Plans/09 §7 update, 2026-05-30):
  - User's first pick was **Qwen2.5-VL-3B**, but its license is the
    Qwen Research License (non-commercial only) — verified, rejected.
  - User's second pick was **Florence-2 + DePlot** — verified two
    transformers-5.x compatibility issues in Microsoft's checkpoint
    (config + processor), both confirmed and would require fragile
    monkey-patches.
  - **SmolVLM2-256M-Video-Instruct** picked (the §7 option C): HF-maintained
    (so transformers-native and tracked with our transformers 5.x), Apache-2.0,
    256M params (~512MB fp16, fits 8GB easily alongside the Qwen specialists),
    handles images (the "-Video-" variant also accepts single images cleanly).
  - DePlot chart-tier is still a planned follow-up once a chart-vs-photo
    router exists and we want to compare against the 19,357-pair dataset.

No silent fallback (``feedback_no_silent_fallbacks``): a model-load or
generation failure raises :class:`FigureCaptionError` so the cascade surfaces
the failure rather than producing empty alt downstream.
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


# ---------------------------------------------------------------------------
# No-hallucination numeric guard (Plans/09 §2.4)
# ---------------------------------------------------------------------------
# SmolVLM2-256M invents chart numbers absent from the caption ~57% of the time
# (R6 eval: no_hallucination 0.43 vs 0.95 gate — see project_figure_captioner_
# verdict). A wrong number is worse than none for a screen-reader user, so any
# generated sentence carrying a number NOT present in the caption (the only
# trustworthy ground truth at emission time) is dropped before the figure ships.
# The number regex mirrors ``scripts/eval_figure_captioner._NUM_RE`` so the
# guard and the acceptance metric measure the same thing.
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Type-level fallback alt for a figure whose model alt was fully stripped AND
# that has no caption to carry its accessible name. Non-empty (axe SC 1.1.1)
# and number-free. Intentionally generic — an honest "there is a figure here"
# beats a fabricated description.
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


# Caption-first label/sentence trim. ``Figure N:`` / ``Fig. N.`` prefix + first
# sentence. Shared verbatim by the production emitter
# (``assembler.fallbacks.fallback_figure``) and the acceptance eval
# (``scripts.eval_figure_captioner.route_figure``) so the shipped behaviour and
# the gate that scores it CANNOT drift — the whole point of wiring caption-first
# into production (Plans/09 §5 gate 3).
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


# ---------------------------------------------------------------------------
# Decision capture for the VLM call site (W7.6)
# ---------------------------------------------------------------------------
# Every LLM/VLM call site must wire up a DecisionCapture and emit at least one
# decision per call (per-figure here) with a DYNAMIC, replayable rationale (root
# /CLAUDE.md § LLM call-site instrumentation). Best-effort: a capture failure
# (lib not on path in the SemantiK subprocess venv, disk error, …) must NEVER
# break captioning — mirrors the "structure_review DecisionCapture failed
# (non-fatal)" posture in MCP/tools/pipeline_tools.py.


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

    Returns ``None`` (captioning proceeds unaffected) when ``lib.decision_capture``
    is unavailable or construction fails. Lands under the canonical ``dart`` tool
    / ``dart-conversion`` phase, matching the Stage-5d ``structure_review``
    capture in ``MCP/tools/pipeline_tools.py``.
    """
    try:
        from lib.decision_capture import DecisionCapture

        return DecisionCapture(
            course_code=_figure_caption_course_code(),
            phase="dart-conversion",
            tool="dart",
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

    Rationale is dynamic + replayable (image content hash, figure geometry /
    stable FB id, model + per-prompt max_tokens, produced caption lengths) so a
    post-hoc replay can attribute the alt/extended text to its exact input.
    Best-effort: never raises into the caption path.
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

    Decorative-figure scope (Plans/09 §5 gate 2 — DECISION 2026-06-09): every
    figure Region reaching Stage 6b is captioned + guarded; there is NO runtime
    decorative-vs-content routing. Rationale: (1) ``data/figure_alt_dataset`` has
    zero decorative rows (every figure carries a caption), so there is no
    train/eval signal; (2) distinguishing decorative needs a chart/photo/decor
    router we don't have (same blocker as the deferred DePlot chart-tier, N2);
    (3) the failure mode is benign — a decorative figure ships either a
    caption-derived alt, a guarded model alt, or the honest ``TYPE_LEVEL_ALT``,
    all non-empty and axe-conformant; the only cost is mild screen-reader
    verbosity, never a WCAG failure or a wrong claim. The ``is_decorative`` /
    ``route_figure`` logic stays in ``scripts.eval_figure_captioner`` as a
    forward-looking gate that lights up the moment a subtype-tagged build (or the
    N2 router) can flag decorative figures. Revisit with N2.

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
    # One capture per Stage-6b invocation; one decision per captioned figure.
    capture = _build_caption_capture()
    for i in fig_idxs:
        region = out[i]
        payload = region.payload or {}
        img_bytes = payload.get("image_png_bytes")
        if not img_bytes:
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
