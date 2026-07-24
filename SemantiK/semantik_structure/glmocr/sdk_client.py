"""GLM-OCR SDK client + PDF page rendering for the extraction lane.

Wraps the ``glmocr`` 0.1.5 self-hosted pipeline (PageLoader → PP-DocLayoutV3
layout → per-region OCR on the OpenAI-compatible vLLM seat → ResultFormatter),
pointed at the seat resolved from ``SEMANTIK_GLMOCR_*`` env, and normalises its
per-page region JSON into the :class:`~.transform.GlmPage` shape the
deterministic transform consumes.

Two seams the transform / lane rely on:

  * ``render_pdf_to_pngs`` — pdftoppm 300-DPI page rasters (poppler).
  * ``GlmOcrClient`` protocol — an injectable OCR client, so the lane + tests
    run against a STUB with NO GPU / seat. ``SdkGlmOcrClient`` is the real
    implementation; it overrides the SDK ``label_task_mapping`` so ``aside_text``
    and ``reference`` regions are KEPT (the default SDK config abandons them),
    and drops the layout model onto ``SEMANTIK_GLMOCR_LAYOUT_DEVICE`` (CPU).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from . import (
    resolve_glmocr_base_url,
    resolve_glmocr_layout_device,
    resolve_glmocr_model,
    resolve_glmocr_render_dpi,
    resolve_glmocr_workers,
)
from .transform import GlmPage

logger = logging.getLogger(__name__)

# Labels the DEFAULT SDK config abandons that the lane KEEPS as content
# (``aside_text`` + ``reference`` carry instructional content and ``footnote``
# is a genuine note; only running header / folio / decoration are furniture, so
# dropping these three would be silent content loss).
_KEEP_AS_TEXT = ("aside_text", "reference", "footnote")

# transformers/accelerate model loading is NOT thread-safe: init_empty_weights
# patches nn.Module registration process-wide, so two threads constructing a
# GlmOcr parser concurrently race one another onto meta tensors
# ("Cannot copy out of meta tensor"). Construction is serialized; only the
# per-chunk parse() fan-out runs concurrently.
_PARSER_CONSTRUCT_LOCK = threading.Lock()


def render_pdf_to_pngs(
    pdf_path: Path, out_dir: Path, *, dpi: Optional[int] = None
) -> List[Path]:
    """Render every PDF page to a PNG at ``dpi`` via pdftoppm (poppler).

    Returns the page PNG paths in 1-based page order. Raises
    ``FileNotFoundError`` when ``pdftoppm`` is absent (fail-loud — no silent
    degraded lane).
    """
    dpi = dpi or resolve_glmocr_render_dpi()
    if shutil.which("pdftoppm") is None:
        raise FileNotFoundError(
            "pdftoppm (poppler-utils) is required for the GLM-OCR lane render "
            "step; install poppler-utils"
        )
    # Render into a PER-DOCUMENT subdir and clear any prior page renders
    # first. The post-render glob collects EVERY page-*.png in the dir, so a
    # shared render dir poisons the page list with leftovers from previously
    # rendered documents — pdftoppm pads page ordinals to the document's own
    # digit count, so a shorter document's stale "page-9.png" survives a
    # longer document's fresh "page-009.png" render and interleaves into the
    # numeric sort (seen live: ch2-opener pages inside the ch01 conversion).
    out_dir = out_dir / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("page*.png"):
        stale.unlink()
    prefix = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
        check=True, capture_output=True,
    )
    pages = sorted(out_dir.glob("page-*.png"), key=lambda p: _page_ord(p))
    if not pages:  # some poppler builds use page-1 vs page-01 — glob covers both
        pages = sorted(out_dir.glob("page*.png"), key=lambda p: _page_ord(p))
    return pages


def _page_ord(p: Path) -> int:
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return int(digits) if digits else 0


def _normalise_regions(json_result: Any) -> List[Dict[str, Any]]:
    """Coerce a PipelineResult ``json_result`` into a flat region-dict list.

    Handles both a flat per-page region list and a nested ``[[region,...]]``
    shape; each region carries ``index`` / ``native_label`` (or ``label``) /
    ``bbox_2d`` / ``content``.
    """
    if not json_result:
        return []
    items = json_result
    # nested [[...]] → flatten (single page per PipelineResult)
    if isinstance(items, list) and items and isinstance(items[0], list):
        flat: List[Any] = []
        for page in items:
            flat.extend(page or [])
        items = flat
    out: List[Dict[str, Any]] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        out.append({
            "index": r.get("index", len(out)),
            "native_label": r.get("native_label") or r.get("label") or "text",
            "bbox_2d": r.get("bbox_2d") or r.get("bbox"),
            "content": r.get("content", ""),
        })
    return out


class GlmOcrClient(Protocol):
    """Injectable OCR client — real SDK or a stub."""

    def parse_pages(self, image_paths: Sequence[Path]) -> List[GlmPage]:
        ...


class SdkGlmOcrClient:
    """Real ``glmocr`` self-hosted client pointed at the vLLM seat."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        workers: Optional[int] = None,
        layout_device: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or resolve_glmocr_base_url()
        self.model = model or resolve_glmocr_model()
        self.workers = max(1, workers or resolve_glmocr_workers())
        self.layout_device = layout_device or resolve_glmocr_layout_device()
        host, port = self._split_base_url(self.base_url)
        self._host, self._port = host, port

    @staticmethod
    def _split_base_url(base_url: str) -> tuple[str, int]:
        # http://host:port/v1 → (host, port)
        rest = base_url.split("://", 1)[-1]
        hostport = rest.split("/", 1)[0]
        if ":" in hostport:
            host, port = hostport.rsplit(":", 1)
            try:
                return host, int(port)
            except ValueError:
                return host, 8002
        return hostport, 8002

    def _make_parser(self):
        from glmocr.api import GlmOcr  # heavy import, deferred to call time

        with _PARSER_CONSTRUCT_LOCK:
            parser = GlmOcr(
                mode="selfhosted",
                ocr_api_host=self._host,
                ocr_api_port=self._port,
                model=self.model,
                layout_device=self.layout_device,
            )
        self._apply_label_override(parser)
        return parser

    def _apply_label_override(self, parser: Any) -> None:
        """Move aside_text/reference/footnote from ``abandon`` → ``text`` so the
        lane keeps them (best-effort; a failure keeps the SDK default)."""
        try:
            detector = parser._pipeline.layout_detector  # type: ignore[attr-defined]
            mapping = detector.label_task_mapping
            if not isinstance(mapping, dict):
                return
            mapping = {k: list(v) if isinstance(v, list) else v
                       for k, v in mapping.items()}
            abandon = list(mapping.get("abandon", []))
            text = list(mapping.get("text", []))
            for label in _KEEP_AS_TEXT:
                if label in abandon:
                    abandon.remove(label)
                if label not in text:
                    text.append(label)
            mapping["abandon"] = abandon
            mapping["text"] = text
            detector.label_task_mapping = mapping
        except Exception as exc:  # noqa: BLE001 — never fail the lane on override
            logger.warning("GLM-OCR label_task_mapping override failed: %s", exc)

    def _parse_with_parser(
        self, parser: Any, image_paths: Sequence[Path], start_page: int
    ) -> List[GlmPage]:
        """Parse ``image_paths`` on an ALREADY-CONSTRUCTED ``parser`` (meter +
        normalise). Does NOT construct or close the parser — the caller owns its
        lifecycle, so a rolling worker can reuse ONE parser across the pages it
        pulls (the parser is never rebuilt per page). Byte-identical per-page
        ``GlmPage`` regardless of how the pages were batched onto the parser
        (the SDK OCRs each page independently)."""
        _t0 = time.monotonic()
        results = parser.parse(
            [str(p) for p in image_paths], save_layout_visualization=False
        )
        # OP2 usage tap — best-effort per-call metering row (provider
        # ``semantik-glmocr``, the per-surface labeling convention). The SDK
        # fans its own per-region POSTs internally and surfaces NO server
        # ``usage``, so token counts are honest zeros per the tap contract;
        # the additive ``pages`` field carries this call's page count (the
        # single-worker chunk, or 1 on the rolling per-page dispatch).
        try:
            from .. import llm_usage_meter

            llm_usage_meter.record_provider_usage(
                provider="semantik-glmocr",
                model=self.model,
                usage=None,
                duration_ms=(time.monotonic() - _t0) * 1000.0,
                phase="semantik_conversion",
                extra={"pages": len(image_paths)},
            )
        except Exception:  # noqa: BLE001 — metering must never crash a call
            pass
        if not isinstance(results, list):
            results = [results]
        pages: List[GlmPage] = []
        for i, res in enumerate(results):
            err = getattr(res, "_error", None)
            regions = _normalise_regions(getattr(res, "json_result", None))
            pages.append(GlmPage(
                page_no=start_page + i, regions=regions,
                image_path=str(image_paths[i]) if i < len(image_paths) else None,
                error=str(err) if err else None,
            ))
        return pages

    def _parse_chunk(self, image_paths: Sequence[Path], start_page: int) -> List[GlmPage]:
        """Single-parser path: construct → parse the whole chunk → close.

        The byte-stable single-worker (``n <= 1``) code path; also the
        construction/close bookend the rolling path replaces with per-worker
        parser reuse."""
        parser = self._make_parser()
        try:
            return self._parse_with_parser(parser, image_paths, start_page)
        finally:
            try:
                parser.close()
            except Exception:  # noqa: BLE001
                pass

    def parse_pages(self, image_paths: Sequence[Path]) -> List[GlmPage]:
        paths = list(image_paths)
        if not paths:
            return []
        # Keep the VRAM/width contract EXACTLY as before: n resident GlmOcr
        # model instances, one per worker (never raised — more workers = more
        # resident models = more VRAM). Rolling at the SAME width is the win.
        n = min(self.workers, len(paths))
        if n <= 1:
            return self._parse_chunk(paths, 1)
        return self._parse_pages_rolling(paths, n)

    def _parse_pages_rolling(
        self, paths: List[Path], n: int
    ) -> List[GlmPage]:
        """ROLLING per-page dispatch: hand the pool the FLAT page list and let
        every one of the ``n`` workers pull the next page the instant it frees
        up — so a worker that finishes cheap pages immediately takes the next
        instead of idling while the worker with expensive pages runs (the
        static-partition drain-to-1 tail this replaces). Mirrors the alt-text
        lane's ``ex.map(_one, <flat list>)`` template.

        Parser reuse (contract): each WORKER THREAD constructs ONE GlmOcr
        parser lazily on its first pulled page (thread-local; construction
        already serialized under ``_PARSER_CONSTRUCT_LOCK``) and REUSES it for
        every subsequent page it pulls — the parser is NEVER rebuilt per page,
        so the model is not thrashed. Exactly ``n`` parsers are built (one per
        resident worker), matching the old one-parser-per-chunk residency.

        Page order (contract): ``ex.map`` yields in submission order, and the
        results are additionally sorted by ``page_no`` — so output is in page
        order regardless of completion order (as today). Error handling is
        unchanged: an SDK ``_error`` becomes ``GlmPage.error`` and a raising
        parse propagates, exactly as the chunk path."""
        thread_local = threading.local()
        built_parsers: List[Any] = []
        built_lock = threading.Lock()

        def _worker_parser() -> Any:
            parser = getattr(thread_local, "parser", None)
            if parser is None:
                parser = self._make_parser()  # construction is lock-guarded
                thread_local.parser = parser
                with built_lock:
                    built_parsers.append(parser)
            return parser

        def _one(item: tuple[int, Path]) -> GlmPage:
            idx, path = item
            parser = _worker_parser()
            # One page per work item → the pool pulls rolling; the reused
            # thread-local parser keeps construction off the per-page path.
            return self._parse_with_parser(parser, [path], idx + 1)[0]

        try:
            with ThreadPoolExecutor(max_workers=n) as ex:
                out: List[GlmPage] = list(ex.map(_one, list(enumerate(paths))))
        finally:
            # Every parse has drained (the pool joined) — release the n resident
            # models, one per worker (the close bookend the chunk path did).
            for parser in built_parsers:
                try:
                    parser.close()
                except Exception:  # noqa: BLE001
                    pass
        out.sort(key=lambda p: p.page_no)
        return out


__all__ = ["GlmOcrClient", "SdkGlmOcrClient", "render_pdf_to_pngs"]
