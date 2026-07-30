"""Qwen3-VL alt-text / caption pass for GLM-OCR figure regions.

Flag ``SEMANTIK_ALTTEXT_PROVIDER`` (``off`` | ``qwen30``), default OFF inside
the lane until seat provisioning is wired. Harvested captions come FIRST (the
transform attaches a ``figure_title`` region as extracted ``caption_text``);
generation runs ONLY for caption-less figures + fills the ``figure_alt``
accessible name.

Contract (owner doctrine — never fabricate, never fail the conversion):
  * the VLM PROPOSES alt/long_desc/caption; the WCAG prompt (ported from
    ``runtime/scratchpad/alttext/prompt.py``) demands function-first, ≤125-char alt.
  * a seat that is unreachable / errors leaves the placeholder + records a
    loud escalation; the conversion never fails.
  * generated content is marked ``alt_source="generated"`` /
    ``caption_source="generated"`` (additive provenance keys).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from . import (
    resolve_alttext_api_key,
    resolve_alttext_base_url,
    resolve_alttext_concurrency,
    resolve_alttext_model,
    resolve_alttext_provider,
    resolve_alttext_timeout,
)

logger = logging.getLogger(__name__)

# ── WCAG prompt contract (ported verbatim intent from runtime/scratchpad/alttext). ──
SYSTEM = (
    "You are an accessibility (WCAG 2.2 AA) expert writing alternative text for "
    "figures in an educational textbook. You are given a cropped figure image plus "
    "the surrounding text context. Write alt text that conveys the figure's "
    "PEDAGOGICAL FUNCTION in the lesson, not a literal pixel description.\n\n"
    "Rules:\n"
    "1. `alt`: a single concise sentence, TARGET <=125 characters. Function-first. "
    "Do NOT begin with 'image of', 'picture of', 'a photo of', 'graphic of', or "
    "'this figure shows'. Convey what a sighted learner would take away. If the "
    'figure is purely decorative, return the empty string "" for alt.\n'
    "2. `long_desc`: include a longer structured description ONLY if the figure is "
    "COMPLEX (a chart, a data table, a multi-panel/step diagram, or a graph a "
    "learner must read values from). For a simple figure return null.\n"
    "3. `caption_suggestion`: a one-line figure caption ONLY IF no existing caption "
    "is supplied in the context; otherwise return null.\n\n"
    "Respond with STRICT JSON and nothing else: "
    '{"alt": "...", "long_desc": "..."|null, "caption_suggestion": "..."|null}'
)
TEMPERATURE = 0.2
MAX_TOKENS = 400


def build_user_text(existing_caption: Optional[str]) -> str:
    lines = ["Write alt text for the attached cropped figure from a textbook page."]
    if existing_caption:
        lines.append(
            f'Existing caption on the page: "{existing_caption}"  '
            "(so caption_suggestion should be null)"
        )
    else:
        lines.append("The page has NO caption for this figure (so DO supply "
                     "caption_suggestion).")
    lines.append("Return only the JSON object with keys alt, long_desc, "
                 "caption_suggestion.")
    return "\n".join(lines)


class AltTextClient(Protocol):
    """Injectable alt-text client — real seat or a stub."""

    def describe(self, image_b64: str, existing_caption: Optional[str]) -> Dict[str, Any]:
        ...


class OpenAICompatibleAltTextClient:
    """Real Qwen3-VL alt-text client over an OpenAI-compatible ``/chat``."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or resolve_alttext_base_url()).rstrip("/")
        self.model = model or resolve_alttext_model()
        self.api_key = api_key if api_key is not None else resolve_alttext_api_key()
        self.timeout = timeout or resolve_alttext_timeout()

    def describe(self, image_b64: str, existing_caption: Optional[str]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": build_user_text(existing_caption)},
                ]},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(), headers=headers,
        )
        _t0 = time.monotonic()
        resp = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        _duration_ms = (time.monotonic() - _t0) * 1000.0
        # OP2 usage tap — best-effort per-POST metering row (provider
        # ``semantik-alttext``, the per-surface labeling convention). Token
        # counts come ONLY from the server-reported ``usage``; any failure is
        # swallowed inside the meter (never perturbs the alt-text call).
        try:
            from .. import llm_usage_meter

            _finish = None
            try:
                _finish = resp["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError):
                pass
            llm_usage_meter.record_provider_usage(
                provider="semantik-alttext",
                model=self.model,
                usage=resp.get("usage") if isinstance(resp, dict) else None,
                duration_ms=_duration_ms,
                finish_reason=_finish,
                phase="semantik_conversion",
            )
        except Exception:  # noqa: BLE001 — metering must never crash a call
            pass
        content = resp["choices"][0]["message"]["content"]
        return _parse_alt_json(content)


def _parse_alt_json(content: str) -> Dict[str, Any]:
    """Tolerant strict-JSON parse of the model's alt response."""
    text = (content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    alt = str(data.get("alt", "") or "").strip()
    if len(alt) > 125:
        alt = alt[:122].rstrip() + "…"
    return {
        "alt": alt,
        "long_desc": (data.get("long_desc") or None),
        "caption_suggestion": (data.get("caption_suggestion") or None),
    }


def _crop_b64(image_path: str, bbox: Sequence[float]) -> Optional[str]:
    """Crop the figure bbox (normalised 0-1000) from the page render → b64 PNG."""
    try:
        import base64
        import io

        from PIL import Image
    except Exception:  # noqa: BLE001 — Pillow optional
        return None
    try:
        img = Image.open(image_path)
        w, h = img.size
        x0, y0, x1, y1 = bbox
        box = (int(x0 / 1000 * w), int(y0 / 1000 * h),
               int(x1 / 1000 * w), int(y1 / 1000 * h))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        crop = img.crop(box).convert("RGB")
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:  # noqa: BLE001
        logger.warning("alt-text crop failed for %s: %s", image_path, exc)
        return None


def apply_alt_text(
    region_provenance: List[Dict[str, Any]],
    page_images: Dict[int, str],
    *,
    client: Optional[AltTextClient] = None,
    escalations: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Fill caption-less figure regions with generated alt/caption in place.

    Returns the count of figures the pass GENERATED for. A no-op when the
    provider is ``off`` or no client resolves. Every unreachable / failed figure
    leaves its placeholder and records an escalation (never fabricates).
    """
    if resolve_alttext_provider() == "off":
        return 0
    if client is None:
        client = OpenAICompatibleAltTextClient()
    escalations = escalations if escalations is not None else []

    targets = [
        p for p in region_provenance
        if p.get("region_kind") == "figure" and not p.get("figure_alt")
        and not p.get("caption_text")
    ]
    generated = 0

    # Dead-seat circuit breaker: a WEDGED seat (accepts connections, never
    # answers — seen live 2026-07-17) makes every figure eat the full HTTP
    # timeout; at 250 figures that is ~an hour per document of pure timeouts.
    # After _BREAKER_THRESHOLD consecutive describe() failures the remaining
    # figures short-circuit to an escalation without touching the socket. A
    # success resets the strike count.
    _BREAKER_THRESHOLD = 5
    _breaker_lock = threading.Lock()
    _strikes = 0
    _breaker_open = False

    def _one(p: Dict[str, Any]) -> None:
        nonlocal generated, _strikes, _breaker_open
        page = int(p.get("source_page") or (p.get("pages") or [0])[0] or 0)
        image_path = page_images.get(page)
        bbox = p.get("bbox")
        if not image_path or not bbox:
            escalations.append({
                "region_index": p.get("first_raw_block_index"),
                "source_page": page, "native_label": p.get("native_label", "image"),
                "reason": "alt_text_no_crop",
                "detail": "no page render or bbox for figure crop",
            })
            return
        b64 = _crop_b64(image_path, bbox)
        if b64 is None:
            escalations.append({
                "region_index": p.get("first_raw_block_index"),
                "source_page": page, "native_label": p.get("native_label", "image"),
                "reason": "alt_text_no_crop", "detail": "crop failed",
            })
            return
        with _breaker_lock:
            if _breaker_open:
                escalations.append({
                    "region_index": p.get("first_raw_block_index"),
                    "source_page": page,
                    "native_label": p.get("native_label", "image"),
                    "reason": "alt_text_generation_failed",
                    "detail": "seat breaker open (consecutive failures)",
                })
                return
        try:
            result = client.describe(b64, p.get("caption_text"))
        except Exception as exc:  # noqa: BLE001 — never fail the conversion
            with _breaker_lock:
                _strikes += 1
                if _strikes >= _BREAKER_THRESHOLD and not _breaker_open:
                    _breaker_open = True
                    logger.error(
                        "alt-text seat breaker OPEN after %d consecutive "
                        "failures — remaining figures escalate without a "
                        "seat call (last error: %s)",
                        _strikes, str(exc)[:120],
                    )
            escalations.append({
                "region_index": p.get("first_raw_block_index"),
                "source_page": page, "native_label": p.get("native_label", "image"),
                "reason": "alt_text_generation_failed", "detail": str(exc)[:200],
            })
            return
        with _breaker_lock:
            _strikes = 0
        alt = result.get("alt") or ""
        if alt:
            p["figure_alt"] = alt
            p["alt_source"] = "generated"
        if result.get("long_desc"):
            p["long_desc"] = result["long_desc"]
        if result.get("caption_suggestion") and not p.get("caption_text"):
            p["caption_text"] = result["caption_suggestion"]
            p["caption_source"] = "generated"
        generated += 1

    workers = max(1, resolve_alttext_concurrency())
    if workers <= 1 or len(targets) <= 1:
        for p in targets:
            _one(p)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(targets))) as ex:
            list(ex.map(_one, targets))
    return generated


__all__ = [
    "AltTextClient",
    "OpenAICompatibleAltTextClient",
    "SYSTEM",
    "apply_alt_text",
    "build_user_text",
]
