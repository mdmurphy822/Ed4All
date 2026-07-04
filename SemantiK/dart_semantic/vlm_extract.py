"""VLM page-per-call extraction client — the P0 dispatch layer.

The PROVIDER-AGNOSTIC seat CONFIG (flag + provider / base_url / api_key / model
resolvers, the ``VLMSeat`` object, and the P2 ``mint_vlm_hint`` hint channel)
lives in :mod:`dart_semantic.extract_shared`. THIS module owns the P0 pieces
that consume a resolved seat:

    * a single OpenAI-compatible ``/chat/completions`` image-chat client,
    * a per-page Markdown disk cache (re-runs are free),
    * the per-document live-POST session (lifecycle-unload gating),
    * a best-effort local-model unload helper (VRAM hand-off).

It is kept SEPARATE from ``extract_shared`` so the heavy image-encode /
``requests`` path stays out of the Stage-1 orchestration module, and so the
client is unit-testable behind injected render / requests boundaries (no live
model calls in tests). See the settled Phase-4 architecture in
``plans/scan-conversion-improvements-2026-07.md``.

P0 scope: the VLM source is recorded on ``page["vlm"]`` but NEVER enters
``_merge_page`` — the merged stream the BERTs consume stays tesseract-only
until the P1 deterministic DP fusion lands, so flag-on output through the
council is byte-identical and the structural anti-hallucination tripwire (P1)
is never bypassed.

Self-contained on purpose
-------------------------
The vendored SemantiK tree runs inside SemantiK's own venv. We do NOT import
``lib/llm/vram_reclaim.py`` (outside the self-containment boundary) — its two
idioms (``/v1``-strip to the native ollama root + ``keep_alive:0`` unload) are
REPLICATED here with a small ``requests`` POST. The seat resolvers in
``extract_shared`` are imported LAZILY inside the render helper only (avoiding a
circular import: ``extract_shared`` imports this module at load time).

Fail-loud / fail-soft posture
-----------------------------
    * seat misconfig (a non-local hosted seat with no credential) →
      ``VLMSeatError`` (from ``seat.require_ready()``) BEFORE any render/POST.
    * per-page TRANSIENT endpoint failure (timeout / conn / 5xx / 429, after
      bounded retries) → ``VlmExtractError(transient=True)`` so the caller
      degrades that page's source alone (tesseract stays authoritative).
    * per-page PERMANENT failure (400/401/403 / malformed response) →
      ``VlmExtractError(transient=False)`` propagates (fail-loud).
"""

from __future__ import annotations

import base64
import datetime
import functools
import hashlib
import io
import json
import logging
import threading
from pathlib import Path

from . import paths as _semantik_paths

logger = logging.getLogger(__name__)

# Bumped when the transcription prompt changes (the per-page cache keys on this
# int rather than hashing the whole directive). Mirrored by the whole-doc
# extract-cache salt (``extract_shared_cached``) via the ``vlm_key`` term.
PROMPT_VERSION = 1

# Validated tier-0 render geometry — a page renders at the OCR scale then
# PIL-downscales to this max width before JPEG encode (a second render per page
# is ~ms vs 12 s inference, so re-rendering on a cache hit is cheap).
_VLM_MAX_PX = 1200

# Bounded retry on TRANSIENT endpoint errors (mirrors endpoint_runtime's
# linear-backoff idiom). Kept module constants (no new flag) so the VLM seat
# stays at its documented env-knob count.
_VLM_MAX_RETRIES = 2
_VLM_RETRY_BACKOFF_BASE_SECONDS = 0.5

# Best-effort unload timeout — a fire-and-return hand-off, not a generation
# call (mirrors ``vram_reclaim._RECLAIM_TIMEOUT_SECONDS``).
_UNLOAD_TIMEOUT_SECONDS = 30.0


class VlmExtractError(RuntimeError):
    """Raised on a per-page VLM HTTP failure.

    Carries a ``transient`` flag so the per-page arm can tell a retryable error
    (timeout / conn / 5xx / 429) from a permanent one (400/401/403, malformed
    response). Permanent errors propagate (fail-loud); transient errors degrade
    that page's ``vlm`` source (fail-soft), tesseract staying authoritative.
    A misconfigured SEAT raises ``extract_shared.VLMSeatError`` (permanent),
    not this — see ``seat.require_ready()``.
    """

    def __init__(self, message: str, *, transient: bool = False) -> None:
        self.transient = transient
        super().__init__(message)


# --- Per-document session (lifecycle unload gating) ------------------------
#
# The lifecycle unload fires ONCE per document, only when >=1 LIVE POST was
# made (an all-cache-hit run loads nothing → no unload). ``extract_shared``
# calls :func:`begin_document_session` before the page loop and reads
# :func:`document_had_live_post` in a ``finally``. Thread-local so concurrent
# in-process documents never cross-count; the offline data build runs one
# document per process, so this is naturally isolated there.

_session = threading.local()


def begin_document_session() -> None:
    """Reset the per-document live-POST counter (call before the page loop)."""
    _session.live_posts = 0


def _note_live_post() -> None:
    _session.live_posts = getattr(_session, "live_posts", 0) + 1


def document_had_live_post() -> bool:
    """Whether >=1 live VLM POST fired since the last :func:`begin_document_session`."""
    return getattr(_session, "live_posts", 0) > 0


# --- Per-page Markdown disk cache (glm_ocr_cache.py pattern) ----------------


def _cache_root(cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    # CWD-independent cache root (mirrors glm_ocr_cache.DEFAULT_CACHE_DIR).
    return _semantik_paths.resolve_cache("vlm_extract_cache")


def _cache_key(
    pdf_sha: str, page_num: int, model: str, prompt_version: int, render_px: list[int]
) -> str:
    raw = (
        f"{pdf_sha}|{page_num}|{model}|{prompt_version}|"
        f"{tuple(int(v) for v in render_px)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def _cache_get(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — corrupt cache file → treat as miss
        return None


def _cache_put(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(path)


@functools.lru_cache(maxsize=32)
def _document_sha_cached(path_str: str, size: int, mtime: int) -> str:
    # Memoized on (path, size, mtime) so a 400-page document hashes the PDF
    # bytes once, not per page. Mirrors glm_ocr_cache.sha256_file.
    from .glm_ocr_cache import sha256_file

    return sha256_file(Path(path_str))


def _document_sha(pdf_path: Path) -> str:
    p = Path(pdf_path)
    st = p.stat()
    return _document_sha_cached(str(p.resolve()), st.st_size, int(st.st_mtime))


# --- Image render + encode -------------------------------------------------


def _default_render_fn(pdf_path: Path, page_num: int, scale: float):
    # Lazy import to avoid a circular import at module load (extract_shared
    # imports this module). The leak-fixed helper closes bitmap+page handles.
    from .extract_shared import _pypdfium2_render_page_to_image

    return _pypdfium2_render_page_to_image(pdf_path, page_num, scale=scale)


def _compute_render_px(width: int, height: int, max_px: int) -> list[int]:
    """Downscaled ``[w, h]`` — width capped at ``max_px``, aspect preserved."""
    w = int(width)
    h = int(height)
    if w > max_px:
        ratio = max_px / float(w)
        return [max_px, max(1, round(h * ratio))]
    return [w, h]


def _encode_jpeg_b64(image, render_px: list[int]) -> str:
    w, h = int(render_px[0]), int(render_px[1])
    if [int(image.width), int(image.height)] != [w, h]:
        image = image.resize((w, h))
    rgb = image if getattr(image, "mode", "RGB") == "RGB" else image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- OpenAI-compatible image-chat client -----------------------------------

_SYSTEM_DIRECTIVE = (
    "You are a faithful document-transcription engine. Transcribe the page "
    "image into clean Markdown, preserving reading order, headings, lists, "
    "and tables. Render EVERY mathematical expression as LaTeX (inline $...$ "
    "or display $$...$$). Reproduce prose VERBATIM. Do NOT summarize, explain, "
    "translate, or add commentary — output ONLY the page's Markdown "
    "transcription."
)
_PAGE_PROMPT = (
    "Transcribe this page image to Markdown, following the system "
    "instructions exactly."
)


def _lazy_requests():
    import requests  # noqa: WPS433 — lazy, mirrors endpoint_runtime

    return requests


def _strip_trailing_v1(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def _chat_completions_url(base_url: str) -> str:
    """``{base}/v1/chat/completions``, tolerant of a base with OR without /v1.

    The seat's default base is the ollama native root (``http://localhost:11434``,
    no ``/v1``) but an operator may set a base that already carries ``/v1``.
    Strip a trailing ``/v1`` then re-add it so both forms resolve to the one
    OpenAI-compatible chat path (idempotent for a hosted ``.../v1`` base too).
    """
    return f"{_strip_trailing_v1(base_url)}/v1/chat/completions"


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    b64_jpeg: str,
    timeout: float,
    requests_module,
) -> str:
    """One ``POST .../v1/chat/completions`` image-chat call → Markdown.

    The request dialect is fixed (OpenAI-compatible with an ``image_url``
    data-URI part) so the seat serves ANY OpenAI-compatible endpoint (local
    ollama >=0.31, a hosted seat, Spark). The provider value selects only
    credential semantics, never the dialect.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_DIRECTIVE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PAGE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                    },
                ],
            },
        ],
        # Greedy → deterministic → cache-safe (the glm_ocr_cache rationale).
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = _chat_completions_url(base_url)

    try:
        resp = requests_module.post(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — timeout / conn → transient
        raise VlmExtractError(
            f"VLM request to endpoint failed ({type(exc).__name__}): {exc}",
            transient=True,
        ) from exc

    status = int(getattr(resp, "status_code", 200))
    if status != 200:
        body_head = (getattr(resp, "text", "") or "")[:500]
        # 5xx / 429 are transient (retryable); other 4xx is a permanent
        # config/client error (retrying just burns the timeout budget).
        transient = status >= 500 or status == 429
        raise VlmExtractError(
            f"VLM endpoint returned HTTP {status} (model={model}). "
            f"Body head: {body_head!r}",
            transient=transient,
        )
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise VlmExtractError(
            "VLM endpoint returned a non-JSON body", transient=False
        ) from exc
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        keys = list(data) if isinstance(data, dict) else type(data)
        raise VlmExtractError(
            f"VLM endpoint returned a malformed response (no "
            f"choices[0].message.content): keys={keys}",
            transient=False,
        ) from exc
    if text is None:
        raise VlmExtractError(
            "VLM endpoint returned a null message content", transient=False
        )
    return str(text)


def _post_with_retry(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    b64_jpeg: str,
    timeout: float,
    requests_module,
) -> str:
    import time  # noqa: WPS433 — local, mirrors the lazy requests import

    attempt = 0
    while True:
        try:
            return _post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                b64_jpeg=b64_jpeg,
                timeout=timeout,
                requests_module=requests_module,
            )
        except VlmExtractError as exc:
            if not exc.transient or attempt >= _VLM_MAX_RETRIES:
                raise
            attempt += 1
            backoff = _VLM_RETRY_BACKOFF_BASE_SECONDS * attempt
            logger.warning(
                "VLM transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                _VLM_MAX_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)


def _markdown_to_text_blocks(markdown: str) -> list[dict]:
    """One text block per non-empty Markdown line, ``bbox=None``.

    ``bbox=None`` is deliberate: bounding boxes arrive only via the P1 DP
    fusion (which places VLM TEXT on the Tesseract BBOX). P0 records the lines
    without geometry.
    """
    blocks: list[dict] = []
    for line in (markdown or "").splitlines():
        if not line.strip():
            continue
        blocks.append(
            {
                "bbox": None,
                "text": line,
                "font_size": None,
                "font_name": None,
                "is_bold": None,
                "is_italic": None,
                "confidence": None,
            }
        )
    return blocks


def _build_source(
    markdown: str, model: str, provider: str, render_px: list[int]
) -> dict:
    return {
        "markdown": markdown,
        "text_blocks": _markdown_to_text_blocks(markdown),
        "model": model,
        "provider": provider,
        "prompt_version": PROMPT_VERSION,
        "render_px": [int(render_px[0]), int(render_px[1])],
    }


def extract_page_markdown(
    pdf_path: Path,
    page_num: int,
    *,
    seat,
    render_scale: float,
    timeout: float,
    cache_dir: Path | str | None = None,
    render_fn=None,
    requests_module=None,
    pdf_sha_override: str | None = None,
) -> dict:
    """Transcribe one page image to a ``vlm`` source dict (cache-first).

    ``seat`` is a resolved ``extract_shared.VLMSeat`` (provider / base_url /
    api_key / model + ``require_ready()``). This calls ``seat.require_ready()``
    FIRST so a misconfigured hosted seat fails loud (``VLMSeatError``) before
    any render or POST.

    Renders the page (at ``render_scale``, then PIL-downscaled to
    ``_VLM_MAX_PX`` width), keys the per-page cache on
    ``sha256(pdf_sha | page | model | prompt_version | render_px)``, and on a
    MISS fires one OpenAI-compatible image-chat POST + writes the Markdown to
    disk. On a HIT no POST is made (re-runs are free) and the live-POST counter
    is NOT incremented. A transient endpoint failure (after bounded retries)
    raises ``VlmExtractError(transient=True)`` so the caller degrades that page
    alone.

    ``render_fn`` / ``requests_module`` / ``pdf_sha_override`` are injection
    seams for tests (mock the render + endpoint boundary; no live model calls).
    """
    # Fail-loud on a misconfigured hosted seat BEFORE we spend a render/POST.
    seat.require_ready()
    model = seat.model
    provider = seat.provider

    render = render_fn or _default_render_fn
    image = render(pdf_path, page_num, render_scale)
    render_px = _compute_render_px(image.width, image.height, _VLM_MAX_PX)

    pdf_sha = pdf_sha_override or _document_sha(pdf_path)
    root = _cache_root(cache_dir)
    key = _cache_key(pdf_sha, page_num, model, PROMPT_VERSION, render_px)
    path = _cache_path(key, root)

    cached = _cache_get(path)
    if cached is not None and "markdown" in cached:
        # Cache hit — no POST, no live-post note. The cached model id is
        # authoritative for the transcription; provider/render_px come from the
        # current resolve (they are part of the key, so they match by
        # construction).
        return _build_source(
            cached["markdown"], cached.get("model", model), provider, render_px
        )

    b64_jpeg = _encode_jpeg_b64(image, render_px)
    req = requests_module or _lazy_requests()
    markdown = _post_with_retry(
        base_url=seat.base_url,
        api_key=seat.api_key,
        model=model,
        b64_jpeg=b64_jpeg,
        timeout=timeout,
        requests_module=req,
    )
    _note_live_post()
    _cache_put(
        path,
        {
            "markdown": markdown,
            "model": model,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    return _build_source(markdown, model, provider, render_px)


# --- Local-model unload (VRAM hand-off) ------------------------------------
#
# SEAM: replace with lib/gpu_lifecycle.py when the sibling lane lands (that
# module does not exist yet — verified — so do NOT depend on it).

# Warn-once dedup so a 400-page doc logs an unload failure once (mirrors the
# ``_USER_WORDS_WARNED`` set pattern landed in extract_shared this batch).
_VLM_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _VLM_WARNED:
        _VLM_WARNED.add(key)
        logger.warning(msg, *args)


def unload_vlm_model(
    base_url: str, model: str, *, requests_module=None, timeout: float = _UNLOAD_TIMEOUT_SECONDS
) -> bool:
    """Best-effort ``keep_alive:0`` unload of a local VLM model. NEVER raises.

    ollama's OpenAI-compat endpoint does not honor a per-request
    ``keep_alive``, so an explicit end-of-batch POST to the native
    ``/api/generate`` (``keep_alive:0``) is the mechanism that frees the card.
    Replicates ``vram_reclaim`` LOCALLY (the self-containment boundary forbids
    importing that module). Returns ``True`` on a successful unload request,
    ``False`` on any failure (warn-once-deduped) — the VLM (5.5 GB) and the 7B
    text seat (6.3 GB) cannot co-reside on an 8 GB card, so this hands the card
    back before Stage-6 authoring.
    """
    try:
        req = requests_module or _lazy_requests()
        root = _strip_trailing_v1(base_url)
        resp = req.post(
            f"{root}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=timeout,
        )
        status = int(getattr(resp, "status_code", 200))
        if status != 200:
            _warn_once(
                f"vlm-unload-http:{model}",
                "VLM unload of %r returned HTTP %d (best-effort; ignored).",
                model,
                status,
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        _warn_once(
            f"vlm-unload:{model}",
            "VLM unload of %r failed (best-effort; ignored): %s",
            model,
            exc,
        )
        return False


__all__ = [
    "PROMPT_VERSION",
    "VlmExtractError",
    "begin_document_session",
    "document_had_live_post",
    "extract_page_markdown",
    "unload_vlm_model",
]
