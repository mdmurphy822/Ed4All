"""VLM page-per-call extraction client — the P0 dispatch layer.

The PROVIDER-AGNOSTIC seat CONFIG (flag + provider / base_url / api_key / model
resolvers, the ``VLMSeat`` object, and the P2 ``mint_vlm_hint`` hint channel)
lives in :mod:`semantik_structure.extract_shared`. THIS module owns the P0 pieces
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
import dataclasses
import datetime
import functools
import hashlib
import io
import json
import logging
import os
import threading
from pathlib import Path

from . import paths as _semantik_paths

logger = logging.getLogger(__name__)

# Bumped when the transcription prompt changes (the per-page cache keys on this
# int rather than hashing the whole directive). Mirrored by the whole-doc
# extract-cache salt (``extract_shared_cached``) via the ``vlm_key`` term.
PROMPT_VERSION = 3

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

# Sentinel distinguishing "capture not yet built for this document" from a
# built-but-unavailable (``None``) capture, so a failed build is memoized once
# per document rather than re-attempted per page.
_CAPTURE_UNSET = object()


def begin_document_session() -> None:
    """Reset the per-document live-POST counter (call before the page loop)."""
    _session.live_posts = 0
    # Reset the per-document DecisionCapture so a new document rebuilds it
    # lazily on its first live POST (see ``_document_capture``).
    _session.capture = _CAPTURE_UNSET


def _note_live_post() -> None:
    _session.live_posts = getattr(_session, "live_posts", 0) + 1


def document_had_live_post() -> bool:
    """Whether >=1 live VLM POST fired since the last :func:`begin_document_session`."""
    return getattr(_session, "live_posts", 0) > 0


# --- Per-page Markdown disk cache (GLM-OCR region cache pattern) ------------


def _cache_root(cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    # CWD-independent cache root (mirrors region_enrichment.cache.DEFAULT_CACHE_DIR).
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
    # bytes once, not per page. Mirrors region_enrichment.cache.sha256_file.
    from .glmocr.region_enrichment.cache import sha256_file

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
    "transcription. Do NOT emit Markdown image syntax ![...](...) and do NOT "
    "emit raw HTML image tags <img ...> (in any form), and do NOT invent image "
    "URLs or file paths. For a figure or diagram that cannot be "
    "transcribed as text, write a plain-text line beginning 'Figure:' "
    "describing it, with no link."
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


_VLM_DISABLE_THINKING_ENV = "SEMANTIK_VLM_DISABLE_THINKING"
_VLM_MAX_TOKENS_ENV = "SEMANTIK_VLM_MAX_TOKENS"
_VLM_THINKING_TRUTHY = frozenset({"1", "true", "yes", "on"})

_VLM_CONCURRENCY_ENV = "SEMANTIK_VLM_CONCURRENCY"
_VLM_DEFAULT_CONCURRENCY = 8


def resolve_vlm_concurrency() -> int:
    """Parse-with-fallback ``SEMANTIK_VLM_CONCURRENCY`` (default 8, min 1).

    Bounds the :class:`~concurrent.futures.ThreadPoolExecutor` that fans the
    per-page VLM transcription POSTs out (see :func:`dispatch_prepared_pages`),
    mirroring :func:`reasoning_qc.resolve_reasoning_qc_concurrency` /
    :func:`endpoint_runtime.resolve_specialist_concurrency`. The per-page
    transcription POST is ~24 s warm on a thinking-off reasoning seat, so a
    serial page loop leaves the vLLM continuous-batching window (``--max-num-seqs``)
    at 1; a fan-out saturates it (a 450-page scan → ~3 h serial vs ~15-25 min at
    width 8-16). ONLY the HTTP POST parallelizes — the page RENDER (pypdfium2,
    not thread-safe on one document handle) runs sequentially in ``prepare``, and
    the ``DecisionCapture`` / live-post accounting / cache write run on the
    dispatching thread in ``finalize``. Garbage / blank / non-positive → the
    default 8; a valid positive int → that value. Read at CALL time (never
    cached at import)."""
    raw = os.environ.get(_VLM_CONCURRENCY_ENV)
    if not raw:
        return _VLM_DEFAULT_CONCURRENCY
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _VLM_DEFAULT_CONCURRENCY
    if val < 1:
        return _VLM_DEFAULT_CONCURRENCY
    return val


def resolve_vlm_disable_thinking() -> bool:
    """Parse-with-fallback ``SEMANTIK_VLM_DISABLE_THINKING`` (default off).

    When truthy, the transcription POST carries
    ``chat_template_kwargs={"thinking": False, "enable_thinking": False}`` so a
    REASONING VLM (e.g. Nemotron-Omni, whose vLLM ``nemotron_v3`` parser honors
    ``enable_thinking is False``) skips the ``<think>`` block. Faithful page
    transcription is a copy task that gains nothing from chain-of-thought, and
    on a reasoning seat the ``<think>`` budget balloons the per-page latency by
    ~10× (measured: 23.7s off vs >4min on) while risking a null-content
    truncation. Default OFF → byte-identical for the non-reasoning default seat
    (``qwen2.5vl:7b``), which ignores the field anyway."""
    return (os.environ.get(_VLM_DISABLE_THINKING_ENV) or "").strip().lower() in _VLM_THINKING_TRUTHY


def resolve_vlm_max_tokens() -> int:
    """Parse-with-fallback ``SEMANTIK_VLM_MAX_TOKENS`` (default 0 → unset).

    A completion cap on the transcription POST. Default 0 sends NO ``max_tokens``
    (byte-identical to the historic call — the server generates to EOS). A
    positive value bounds runaway generation on a reasoning seat. Non-positive /
    garbage → 0 (unset)."""
    raw = os.environ.get(_VLM_MAX_TOKENS_ENV)
    if not raw:
        return 0
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 0
    return val if val > 0 else 0


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
        # Greedy → deterministic → cache-safe (the GLM-OCR cache rationale).
        "temperature": 0,
    }
    _mt = resolve_vlm_max_tokens()
    if _mt > 0:
        body["max_tokens"] = _mt
    if resolve_vlm_disable_thinking():
        # Reasoning-VLM CoT suppression (Nemotron etc.); harmless no-op field
        # for a non-reasoning seat. See resolve_vlm_disable_thinking.
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
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
    # Observability (step 6): a ``finish_reason == "length"`` means the
    # transcription was TRUNCATED at the completion-token ceiling — a real
    # silent content-loss channel (this audit did not hit it, but a dense page
    # can). Warn loudly; do NOT change the return signature (additive log only).
    try:
        finish_reason = data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        finish_reason = None
    if finish_reason == "length":
        logger.warning(
            "VLM transcription TRUNCATED (finish_reason=length, model=%s, "
            "max_tokens=%s, chars=%d) — page content may be silently lost; "
            "raise SEMANTIK_VLM_MAX_TOKENS or split the page.",
            model,
            body.get("max_tokens", "unset"),
            len(str(text)),
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


# --- Decision capture for the P0 VLM page-transcription call site ----------
#
# Every LLM/VLM call site must wire up a DecisionCapture and emit at least one
# decision per LLM call (per LIVE page transcription here) with a DYNAMIC,
# replayable rationale (root /CLAUDE.md § LLM call-site instrumentation).
# Best-effort: a capture failure (``lib`` not importable in the SemantiK
# subprocess venv, disk error, …) must NEVER break extraction — mirrors the
# Stage-6b figure-captioner's ``_build_caption_capture`` posture and the MCP
# bridge's ``structure_review`` capture.
#
# ONE capture per DOCUMENT (reset in :func:`begin_document_session`, built
# lazily on the first live POST), ONE decision per LIVE page transcription — a
# cache HIT makes no POST, so no LLM call fired and no decision is logged
# (mirrors the ``_note_live_post`` live-only accounting).


def _vlm_extract_course_code() -> str:
    """Course code for the VLM-extract capture (best-effort context)."""
    raw = (
        os.environ.get("SEMANTIK_COURSE_CODE")
        or os.environ.get("ED4ALL_COURSE_CODE")
        or ""
    ).strip()
    return raw or "SEMANTIK"


def _build_vlm_extract_capture():
    """Construct a best-effort DecisionCapture for the VLM-extract call site.

    Returns ``None`` (extraction proceeds unaffected) when
    ``lib.decision_capture`` is unavailable (the cross-venv bridge venv may not
    carry Ed4All's ``lib/``) or construction fails. Lands under the ratified
    ``semantik`` tool / ``semantik_conversion`` phase (task #19), matching the
    Stage-6b ``alt_text_generation`` capture and the Stage-5d
    ``structure_review`` capture at the MCP seam.
    """
    try:
        from lib.decision_capture import DecisionCapture

        return DecisionCapture(
            course_code=_vlm_extract_course_code(),
            phase="semantik_conversion",
            tool="semantik",
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("vlm-extract DecisionCapture unavailable (non-fatal): %s", exc)
        return None


def _document_capture():
    """Lazily build + memoize the per-document VLM-extract capture.

    Keyed off the per-document thread-local session so a 400-page document
    emits ONE capture carrying N per-page decisions (not N sessions/files). A
    failed build is memoized as ``None`` and never re-attempted for the same
    document. Safe to call when :func:`begin_document_session` was never called
    (e.g. a unit test that drives one page directly) — the sentinel default
    triggers a lazy build.
    """
    cap = getattr(_session, "capture", _CAPTURE_UNSET)
    if cap is _CAPTURE_UNSET:
        cap = _build_vlm_extract_capture()
        _session.capture = cap
    return cap


def _log_page_transcription_decision(
    capture,
    *,
    pdf_sha: str,
    page_num: int,
    model: str,
    provider: str,
    render_px: list[int],
    markdown: str,
    disable_thinking: bool,
    max_tokens: int,
) -> None:
    """Emit ONE decision for a LIVE VLM page transcription.

    Rationale is dynamic + replayable (pdf sha / page number, model + provider,
    render pixels, prompt version, thinking-off flag, ``max_tokens``, and the
    produced Markdown length + non-empty line count) so a post-hoc replay can
    attribute the transcription to its exact input. Best-effort: never raises
    into the extraction path.
    """
    if capture is None:
        return
    try:
        md = markdown or ""
        line_count = sum(1 for ln in md.splitlines() if ln.strip())
        try:
            px = tuple(int(v) for v in (render_px or ()))
        except (TypeError, ValueError):
            px = ()
        rationale = (
            f"Qwen2.5-VL page transcription pdf_sha256={(pdf_sha or '')[:16]} "
            f"page={page_num} (model={model} provider={provider} render_px={px} "
            f"prompt_version={PROMPT_VERSION} disable_thinking={disable_thinking} "
            f"max_tokens={max_tokens}): produced markdown_len={len(md)} chars "
            f"non_empty_lines={line_count} (P0 fusion-free — VLM source stamped "
            f"on page['vlm'], never entered _merge_page)."
        )
        capture.log_decision(
            decision_type="structure_detection",
            decision=(
                f"transcribe page {page_num} image to Markdown "
                f"({len(md)} chars, {line_count} non-empty lines)"
            ),
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "vlm-extract DecisionCapture log failed (non-fatal) on page %s: %s",
            page_num, exc,
        )


# --- Prepare / dispatch / finalize split (concurrency seam) ----------------
#
# ``extract_page_markdown`` is factored into three phases so the per-page POST
# can be fanned OUT across pages while the pdfium render stays sequential:
#
#   * ``prepare_page_request`` — ``require_ready()`` + RENDER (pypdfium2, NOT
#     thread-safe on one document handle) + cache lookup. Runs on the
#     dispatching thread, sequentially. Returns a ``PreparedVlmPage`` that is a
#     cache HIT (finalize inline, no POST) or a MISS (carries the encoded JPEG).
#   * ``dispatch_prepared_request`` — the one HTTP POST + parse. PURE (no
#     thread-local, no DecisionCapture, no cache write) → safe to run in a
#     worker thread. :func:`dispatch_prepared_pages` fans a batch of these out.
#   * ``finalize_prepared_request`` — ``_note_live_post`` (the thread-local
#     live-POST accounting) + one ``DecisionCapture`` + the disk-cache write +
#     ``_build_source``. Runs on the DISPATCHING thread in the caller's ordered
#     consumption loop (never in a worker) so the thread-local session
#     (``begin_document_session`` / ``document_had_live_post``) and the capture
#     helper are only ever touched single-threaded.
#
# ``extract_page_markdown`` itself is the byte-identical SERIAL composition of
# the three (used by the single-page direct-call path + every existing test).


@dataclasses.dataclass
class PreparedVlmPage:
    """A rendered, cache-resolved per-page VLM request awaiting its POST.

    A cache HIT carries ``cached_markdown`` (``b64_jpeg is None`` → no POST); a
    MISS carries ``b64_jpeg`` (``cached_markdown is None`` → dispatch fires the
    POST). ``model`` / ``provider`` / ``render_px`` / ``cache_path`` are the
    render-time resolve, threaded through to ``finalize`` so the fanned-out POST
    result is built + cached identically to the serial path.
    """

    pdf_sha: str
    page_num: int
    model: str
    provider: str
    render_px: list[int]
    cache_path: Path
    b64_jpeg: str | None
    cached_markdown: str | None
    cached_model: str | None

    @property
    def is_cache_hit(self) -> bool:
        return self.cached_markdown is not None

    def cached_source(self) -> dict:
        """The ``vlm`` source dict for a cache HIT (no POST, no live-post note).

        The cached model id is authoritative for the transcription;
        provider/render_px come from the current resolve (they are part of the
        cache key, so they match by construction) — byte-identical to the
        historic cache-hit return of ``extract_page_markdown``.
        """
        return _build_source(
            self.cached_markdown, self.cached_model or self.model, self.provider,
            self.render_px,
        )


def prepare_page_request(
    pdf_path: Path,
    page_num: int,
    *,
    seat,
    render_scale: float,
    cache_dir: Path | str | None = None,
    render_fn=None,
    pdf_sha_override: str | None = None,
) -> PreparedVlmPage:
    """Render + cache-resolve one page (SEQUENTIAL — pdfium is not thread-safe).

    Calls ``seat.require_ready()`` FIRST (a misconfigured hosted seat fails loud
    with ``VLMSeatError`` before any render), renders the page (at
    ``render_scale``, PIL-downscaled to ``_VLM_MAX_PX`` width), keys the per-page
    cache on ``sha256(pdf_sha | page | model | prompt_version | render_px)``, and
    returns a :class:`PreparedVlmPage`. A cache HIT carries the cached markdown
    (dispatch is skipped); a MISS carries the encoded JPEG for the POST. MUST run
    on the dispatching thread — it performs the pypdfium2 render, which is not
    safe under concurrent access to one document handle.
    """
    # Fail-loud on a misconfigured hosted seat BEFORE we spend a render.
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
        return PreparedVlmPage(
            pdf_sha=pdf_sha, page_num=page_num, model=model, provider=provider,
            render_px=render_px, cache_path=path, b64_jpeg=None,
            cached_markdown=cached["markdown"], cached_model=cached.get("model", model),
        )

    b64_jpeg = _encode_jpeg_b64(image, render_px)
    return PreparedVlmPage(
        pdf_sha=pdf_sha, page_num=page_num, model=model, provider=provider,
        render_px=render_px, cache_path=path, b64_jpeg=b64_jpeg,
        cached_markdown=None, cached_model=None,
    )


def dispatch_prepared_request(
    prepared: PreparedVlmPage,
    *,
    seat,
    timeout: float,
    requests_module=None,
) -> str:
    """Fire the one image-chat POST for a cache-MISS prepared request → Markdown.

    PURE with respect to process state — no thread-local mutation, no
    DecisionCapture, no cache write — so it is safe to run in a worker thread
    (that is what :func:`dispatch_prepared_pages` fans out). A transient endpoint
    failure (after bounded retries) raises ``VlmExtractError(transient=True)``; a
    permanent one raises ``VlmExtractError(transient=False)`` — the caller applies
    the per-page degrade/fail-loud contract. A cache HIT short-circuits to the
    cached markdown (defensive; the caller normally skips dispatch on a hit).
    """
    if prepared.is_cache_hit:
        return prepared.cached_markdown  # type: ignore[return-value]
    req = requests_module or _lazy_requests()
    return _post_with_retry(
        base_url=seat.base_url,
        api_key=seat.api_key,
        model=prepared.model,
        b64_jpeg=prepared.b64_jpeg,
        timeout=timeout,
        requests_module=req,
    )


def finalize_prepared_request(
    prepared: PreparedVlmPage,
    markdown: str,
    *,
    capture=None,
) -> dict:
    """Record the LIVE POST result: live-post note + capture + cache + source.

    Runs on the DISPATCHING thread (the caller's ordered consumption loop) — it
    touches the thread-local live-POST counter (``_note_live_post``) and the
    per-document ``DecisionCapture``, neither of which is safe to fire from a
    worker thread concurrently. ``capture`` is the per-document capture (built
    once on the dispatching thread via :func:`_document_capture`); one decision
    per LIVE page transcription is emitted (best-effort, non-fatal). Byte-
    identical to the miss tail of the historic ``extract_page_markdown``.
    """
    _note_live_post()
    _log_page_transcription_decision(
        capture,
        pdf_sha=prepared.pdf_sha,
        page_num=prepared.page_num,
        model=prepared.model,
        provider=prepared.provider,
        render_px=prepared.render_px,
        markdown=markdown,
        disable_thinking=resolve_vlm_disable_thinking(),
        max_tokens=resolve_vlm_max_tokens(),
    )
    _cache_put(
        prepared.cache_path,
        {
            "markdown": markdown,
            "model": prepared.model,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    return _build_source(markdown, prepared.model, prepared.provider, prepared.render_px)


def dispatch_prepared_pages(
    items,
    *,
    seat,
    timeout: float,
    requests_module=None,
    concurrency: int | None = None,
) -> dict:
    """Fan the POST for a batch of cache-MISS prepared pages out concurrently.

    ``items`` is a sequence of ``(key, PreparedVlmPage)`` — the caller's opaque
    ``key`` (e.g. page index) identifies each slot. Returns ``{key: outcome}``
    where ``outcome`` is the Markdown string on success OR the raised exception
    object (``VlmExtractError`` transient/permanent, or any other) — the caller
    consumes the slots in its own ORIGINAL order and applies the per-page
    degrade/fail-loud contract + ``finalize`` on the dispatching thread.

    Mirrors :meth:`endpoint_runtime.OpenAICompatibleRuntime.generate_batch` /
    :func:`reasoning_qc._fan_out_page_verifies`: a pre-sized result map keyed by
    the caller's key, each future writing its own slot, so completion order can
    never scramble downstream ordering. Width is
    :func:`resolve_vlm_concurrency` (``SEMANTIK_VLM_CONCURRENCY``, default 8)
    unless overridden, floored at 1 and capped at ``len(items)``. ONLY the POST
    parallelizes — render/cache already ran sequentially (``prepare``) and the
    caller finalizes (live-post note + capture + cache write) single-threaded.
    """
    items = list(items)
    results: dict = {}
    if not items:
        return results

    width = concurrency if concurrency is not None else resolve_vlm_concurrency()
    max_workers = max(1, min(int(width), len(items)))

    def _work(prep: PreparedVlmPage) -> str:
        return dispatch_prepared_request(
            prep, seat=seat, timeout=timeout, requests_module=requests_module
        )

    from concurrent.futures import ThreadPoolExecutor  # noqa: WPS433

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_key = {pool.submit(_work, prep): key for key, prep in items}
        for future in future_to_key:
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except BaseException as exc:  # noqa: BLE001 — capture per-slot; the
                # caller re-applies the transient/permanent + fail-loud contract
                # in ORIGINAL order (a permanent / non-VlmExtractError re-raises
                # there, exactly as the serial path would).
                results[key] = exc
    return results


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
    """Transcribe one page image to a ``vlm`` source dict (cache-first, SERIAL).

    The byte-identical serial composition of :func:`prepare_page_request` →
    :func:`dispatch_prepared_request` → :func:`finalize_prepared_request` (used
    by the single-page direct-call path; the document loop uses the three
    primitives directly to fan the POST out). ``seat.require_ready()`` fires
    FIRST (misconfigured hosted seat → ``VLMSeatError`` before any render/POST);
    a cache HIT makes no POST + no live-post note; a MISS fires one POST and
    writes the Markdown to disk. A transient endpoint failure (after bounded
    retries) raises ``VlmExtractError(transient=True)`` so the caller degrades
    that page alone.

    ``render_fn`` / ``requests_module`` / ``pdf_sha_override`` are injection
    seams for tests (mock the render + endpoint boundary; no live model calls).
    """
    prepared = prepare_page_request(
        pdf_path,
        page_num,
        seat=seat,
        render_scale=render_scale,
        cache_dir=cache_dir,
        render_fn=render_fn,
        pdf_sha_override=pdf_sha_override,
    )
    if prepared.is_cache_hit:
        return prepared.cached_source()
    markdown = dispatch_prepared_request(
        prepared, seat=seat, timeout=timeout, requests_module=requests_module
    )
    return finalize_prepared_request(prepared, markdown, capture=_document_capture())


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
    "PreparedVlmPage",
    "VlmExtractError",
    "begin_document_session",
    "dispatch_prepared_pages",
    "dispatch_prepared_request",
    "document_had_live_post",
    "extract_page_markdown",
    "finalize_prepared_request",
    "prepare_page_request",
    "resolve_vlm_concurrency",
    "unload_vlm_model",
]
