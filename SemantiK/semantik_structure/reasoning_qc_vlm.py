"""VLM client for the Stage-9b reasoning-QC pass (``SEMANTIK_REASONING_QC``).

Self-contained multimodal client that judges a page's *reading order* and
*structure* from the re-rendered source raster + the ordered block texts. It is
the QC-side twin of :mod:`semantik_structure.vlm_extract`: same OpenAI-compatible
``image_url`` transport, same fail-loud seat gate, same bounded transient-retry —
but a DIFFERENT prompt (a JSON judgment, not a Markdown transcription) and
structured-output request fields (``response_format`` / ollama ``format:'json'``)
that the transcription client has no need for.

Self-containment (hard constraint): this module imports ONLY
``semantik_structure.vlm_extract`` and ``semantik_structure.extract_shared`` — it
never touches ``lib/`` or ``Trainforge`` — so it survives the cross-venv SemantiK
conversion subprocess. The orchestrator (:mod:`reasoning_qc`) owns the
DecisionCapture wiring (a best-effort lazy ``lib.decision_capture`` import that is
allowed to fail); this transport module stays pure.

Seat resolution reuses :func:`extract_shared.resolve_vlm_seat` (the
``SEMANTIK_VLM_*`` provider / base_url / api_key / model / timeout family) with an
OPTIONAL ``SEMANTIK_REASONING_QC_MODEL`` model-id override so QC can point at a
larger reasoning model (Nemotron-Omni via ``SEMANTIK_VLM_PROVIDER=nvidia`` +
base_url/key) while extraction keeps ``qwen2.5vl:7b``. :meth:`VLMSeat.require_ready`
is called BEFORE any POST (fail-loud on a non-loopback hosted seat without a key —
never a silent stub).

Fail-soft contract: a parse miss or a :class:`VlmExtractError` (after the bounded
retries) degrades to an EMPTY verdict ``{}`` — this module NEVER raises into the
cascade. A broken QC seat can only fail to *improve* a document, never corrupt one.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Sequence

# Self-containment boundary: SemantiK-local imports ONLY.
from . import extract_shared
from . import vlm_extract
from .extract_shared import VLMSeat, resolve_vlm_seat, resolve_vlm_timeout
from .vlm_extract import VlmExtractError

logger = logging.getLogger(__name__)

# The QC model-id override env (satellite of SEMANTIK_REASONING_QC). Default →
# the extraction VLM model (resolve_vlm_model). Selecting a DISTINCT model here
# is the only condition that adds a docs/LICENSING.md row (maintenance contract).
_QC_MODEL_ENV = "SEMANTIK_REASONING_QC_MODEL"

# Bounded transient-retry knobs — reuse the extraction client's constants so QC
# retry posture tracks extraction (fall back to literals if the names move).
_QC_MAX_RETRIES = getattr(vlm_extract, "_VLM_MAX_RETRIES", 2)
_QC_RETRY_BACKOFF_BASE_SECONDS = getattr(
    vlm_extract, "_VLM_RETRY_BACKOFF_BASE_SECONDS", 0.5
)
_QC_MAX_PX = getattr(vlm_extract, "_VLM_MAX_PX", 1200)


# ---------------------------------------------------------------------------
# System directive — a STRUCTURAL EDITOR judgment, never a rewrite.
# ---------------------------------------------------------------------------
_QC_SYSTEM_DIRECTIVE = (
    "You are a document-structure quality-control reviewer. You are given a page "
    "image and the ordered list of text blocks a converter extracted from it. "
    "Judge ONLY the reading order and the structural TYPE of each block against "
    "the page image. You NEVER rewrite, summarize, translate, or invent text — "
    "you emit ONLY a JSON judgment keyed by the given block indices. "
    "Respond with a single JSON object with these keys: "
    '"reading_order" (the block indices in the correct reading order, or [] if '
    'the given order is already correct); '
    '"phantom_headings" (indices of blocks typed as a heading that are NOT a '
    "real section heading on the page — a running header, page number, caption, "
    "or table-of-contents entry — each as "
    '{"index": int, "reason": str}); '
    '"apparatus_retype" (indices of blocks that are exercise/answer-key '
    "apparatus mistyped as a heading or body, each as "
    '{"index": int, "reason": str}); '
    '"misordered" (runs of indices whose emission order diverges from the page, '
    'each as {"run": [int, ...], "reason": str}); '
    '"confidence" (a float 0..1 for the overall judgment). '
    "Use ONLY the block indices you were given. If nothing is wrong, return every "
    "list empty. Output ONLY the JSON object, no prose, no code fence."
)


def resolve_reasoning_qc_model(default_model: str | None = None) -> str:
    """Return the QC seat's model id.

    ``SEMANTIK_REASONING_QC_MODEL`` (satellite override) wins; otherwise the
    supplied ``default_model`` (the resolved extraction VLM model) is used, and
    failing that :func:`extract_shared.resolve_vlm_model`. Read at CALL time.
    """
    raw = (os.environ.get(_QC_MODEL_ENV) or "").strip()
    if raw:
        return raw
    if default_model:
        return default_model
    return extract_shared.resolve_vlm_model()


def resolve_reasoning_qc_seat() -> VLMSeat:
    """Resolve + fail-loud-check the QC VLM seat.

    Reuses :func:`extract_shared.resolve_vlm_seat` (the ``SEMANTIK_VLM_*``
    family) and overlays the optional ``SEMANTIK_REASONING_QC_MODEL`` model id,
    then calls :meth:`VLMSeat.require_ready` FIRST — a non-local hosted seat
    with no credential raises :class:`extract_shared.VLMSeatError` here rather
    than fabricating a judgment.
    """
    base = resolve_vlm_seat()
    model = resolve_reasoning_qc_model(default_model=base.model)
    seat = VLMSeat(
        provider=base.provider,
        base_url=base.base_url,
        api_key=base.api_key,
        model=model,
    )
    return seat.require_ready()


# ---------------------------------------------------------------------------
# Rendering — re-render the specific page on demand (no persistence).
# ---------------------------------------------------------------------------
def render_page(pdf_path: str | Path, page_num: int, *, scale: float | None = None) -> str:
    """Render ONE 1-indexed physical page to a base64 JPEG data payload.

    Reuses the extraction render primitives verbatim
    (:func:`extract_shared._pypdfium2_render_page_to_image` +
    :func:`vlm_extract._compute_render_px` + :func:`vlm_extract._encode_jpeg_b64`)
    so the QC raster matches the extraction raster's downscale posture. Byte
    identity to the extraction render is NOT required — a legible raster suffices
    for a reasoning-order judgment, and page numbers are scale-independent so the
    page→block keying is unaffected. Returns the raw base64 string (no ``data:``
    prefix); the caller wraps it into the ``image_url`` data URI.
    """
    render_scale = scale if scale is not None else extract_shared.resolve_ocr_render_scale()
    image = extract_shared._pypdfium2_render_page_to_image(
        Path(pdf_path), int(page_num), scale=render_scale
    )
    render_px = vlm_extract._compute_render_px(image.width, image.height, _QC_MAX_PX)
    return vlm_extract._encode_jpeg_b64(image, render_px)


# ---------------------------------------------------------------------------
# The multimodal POST — mirrors vlm_extract._post_chat_completion, ADDS JSON mode.
# ---------------------------------------------------------------------------
def _build_qc_user_text(block_texts: Sequence[str]) -> str:
    """The user-turn text part: the ordered, index-labelled block list."""
    lines = ["Blocks extracted from this page, in the converter's current order:"]
    for i, txt in enumerate(block_texts):
        snippet = (txt or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        lines.append(f"[{i}] {snippet}")
    lines.append(
        "Judge the reading order and per-block structural type against the page "
        "image and return the JSON judgment described in the system instructions."
    )
    return "\n".join(lines)


def _post_qc_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    b64_jpeg: str,
    block_texts: Sequence[str],
    timeout: float,
    requests_module,
    directive: str = _QC_SYSTEM_DIRECTIVE,
) -> str:
    """One ``POST .../v1/chat/completions`` QC judgment call → raw JSON string.

    Mirrors :func:`vlm_extract._post_chat_completion`'s message shape (system
    directive + user ``[text, image_url]``, ``temperature=0``) and its
    HTTP-status → transient/permanent classification, but:

      * the user text is the ordered block list (not a "transcribe" prompt);
      * ``response_format={"type": "json_object"}`` + ollama ``format="json"``
        request structured output (the transcription client has neither);
      * ``chat_template_kwargs`` (thinking-off) is honored via
        :func:`vlm_extract.resolve_vlm_disable_thinking`.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": directive},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_qc_user_text(block_texts)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                    },
                ],
            },
        ],
        "temperature": 0,
        # Structured-output request — OpenAI dialect + ollama dialect. A server
        # that ignores one still sees the other; a server that honors neither is
        # handled by the tolerant JSON parse downstream.
        "response_format": {"type": "json_object"},
        "format": "json",
    }
    if vlm_extract.resolve_vlm_disable_thinking():
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = vlm_extract._chat_completions_url(base_url)

    try:
        resp = requests_module.post(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — timeout / conn → transient
        raise VlmExtractError(
            f"reasoning-QC request to endpoint failed ({type(exc).__name__}): {exc}",
            transient=True,
        ) from exc

    status = int(getattr(resp, "status_code", 200))
    if status != 200:
        body_head = (getattr(resp, "text", "") or "")[:500]
        transient = status >= 500 or status == 429
        raise VlmExtractError(
            f"reasoning-QC endpoint returned HTTP {status} (model={model}). "
            f"Body head: {body_head!r}",
            transient=transient,
        )
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise VlmExtractError(
            "reasoning-QC endpoint returned a non-JSON body", transient=False
        ) from exc
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        keys = list(data) if isinstance(data, dict) else type(data)
        raise VlmExtractError(
            f"reasoning-QC endpoint returned a malformed response "
            f"(no choices[0].message.content): keys={keys}",
            transient=False,
        ) from exc
    if text is None:
        raise VlmExtractError(
            "reasoning-QC endpoint returned a null message content", transient=False
        )
    return str(text)


def _post_with_retry(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    b64_jpeg: str,
    block_texts: Sequence[str],
    timeout: float,
    requests_module,
    directive: str = _QC_SYSTEM_DIRECTIVE,
) -> str:
    """Bounded transient-retry around :func:`_post_qc_completion`.

    Mirrors :func:`vlm_extract._post_with_retry`. Module-level so tests can
    monkeypatch it to return canned JSON (no torch / ollama / weights load).
    Re-raises the last :class:`VlmExtractError` on a permanent error or after
    the retry budget; the caller (:func:`run_qc_judgment`) turns that into the
    empty-verdict fail-soft.
    """
    import time  # noqa: WPS433 — local, mirrors the lazy import in vlm_extract

    attempt = 0
    while True:
        try:
            return _post_qc_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                b64_jpeg=b64_jpeg,
                block_texts=block_texts,
                timeout=timeout,
                requests_module=requests_module,
                directive=directive,
            )
        except VlmExtractError as exc:
            if not exc.transient or attempt >= _QC_MAX_RETRIES:
                raise
            attempt += 1
            backoff = _QC_RETRY_BACKOFF_BASE_SECONDS * attempt
            logger.warning(
                "reasoning-QC transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                _QC_MAX_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)


# ---------------------------------------------------------------------------
# Tolerant JSON parse.
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _first_balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, or ``None``."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_qc_json(text: str | None) -> dict[str, Any]:
    """Parse the model's judgment string tolerantly → dict (``{}`` on miss).

    Tolerates a bare object, a ```json``` fenced block, or an object embedded in
    surrounding prose. NEVER raises — a parse miss is the empty verdict.
    """
    if not text or not text.strip():
        return {}
    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)
    balanced = _first_balanced_object(text)
    if balanced:
        candidates.append(balanced)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return {}


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------
def run_qc_judgment(
    seat: VLMSeat,
    pdf_path: str | Path,
    page_num: int,
    block_texts: Sequence[str],
    *,
    directive: str = _QC_SYSTEM_DIRECTIVE,
    requests_module=None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Render ``page_num`` + POST the QC judgment → parsed verdict dict.

    Fail-soft: any render/transport/parse failure degrades to ``{}`` (an empty
    verdict = "nothing flagged"), so a broken seat never raises into the cascade
    and never corrupts a document. The caller supplies an already-``require_ready``
    seat (see :func:`resolve_reasoning_qc_seat`).
    """
    req = requests_module or vlm_extract._lazy_requests()
    tmo = timeout if timeout is not None else resolve_vlm_timeout()
    try:
        b64 = render_page(pdf_path, page_num)
    except Exception as exc:  # noqa: BLE001 — render is fail-soft
        logger.warning("reasoning-QC page render failed (page %s): %s", page_num, exc)
        return {}
    try:
        raw = _post_with_retry(
            base_url=seat.base_url,
            api_key=seat.api_key,
            model=seat.model,
            b64_jpeg=b64,
            block_texts=block_texts,
            timeout=tmo,
            requests_module=req,
            directive=directive,
        )
    except VlmExtractError as exc:
        logger.warning("reasoning-QC VLM call failed fail-soft (page %s): %s", page_num, exc)
        return {}
    return parse_qc_json(raw)


def unload_qc_model(seat: VLMSeat, *, requests_module=None) -> bool:
    """Best-effort ``keep_alive:0`` unload of the QC model. NEVER raises.

    Delegates to :func:`vlm_extract.unload_vlm_model` so the QC seat frees the
    card at the pass's ``finally`` (the 5.5 GB VLM cannot co-reside with a
    resident 7B specialist on an 8 GB box).
    """
    try:
        return vlm_extract.unload_vlm_model(
            seat.base_url, seat.model, requests_module=requests_module
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("reasoning-QC unload failed (non-fatal): %s", exc)
        return False
