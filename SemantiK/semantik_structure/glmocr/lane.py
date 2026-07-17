"""GLM-OCR extraction lane orchestrator (master flag ``SEMANTIK_GLMOCR_LANE``).

Ties the stages together for a whole-document conversion:

    PDF --render_pdf_to_pngs--> page rasters
        --GlmOcrClient.parse_pages--> per-page GLM region layout
        --transform_document--> region_provenance + heading_tree + escalations
        --apply_alt_text (optional, SEMANTIK_ALTTEXT_PROVIDER)--> figure alt
        --> sidecars ({stem}.glmocr_layout.json + {stem}.glmocr_escalations.jsonl)
        --> (optional) accessible HTML via the lib/semantik adapter

The lane's PRIMARY product is the SemantiK wire contract (``region_provenance``
+ ``heading_tree``), which the existing Ed4All conversion seam renders to the
accessible HTML + ``data-semantik-*`` provenance contract UNCHANGED. The lane
can ALSO render the HTML itself (``render_html=True``) when ``lib/semantik`` is
importable (in-process install) — used by the standalone smoke.

Injectable clients (``sdk_client`` / ``alttext_client``) let the lane + its
tests run with NO GPU / seat. Default clients hit the real seats.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import resolve_alttext_provider
from .alttext import AltTextClient, apply_alt_text
from .escalation import write_escalations_sidecar
from .sdk_client import GlmOcrClient, SdkGlmOcrClient, render_pdf_to_pngs
from .transform import GlmPage, TransformResult, transform_document

logger = logging.getLogger(__name__)


@dataclass
class GlmOcrLaneResult:
    """The GLM-OCR lane output — the wire contract + sidecar paths."""

    pdf: str
    region_provenance: List[Dict[str, Any]]
    heading_tree: List[Any]
    escalations: List[Dict[str, Any]]
    pages: List[GlmPage] = field(default_factory=list)
    layout_sidecar: Optional[str] = None
    escalations_sidecar: Optional[str] = None
    html: Optional[str] = None
    html_path: Optional[str] = None
    alt_text_generated: int = 0


def _write_layout_sidecar(path: Path, pages: List[GlmPage]) -> Path:
    """Persist the per-page GLM region layout (the provenance backbone)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "glmocr-layout/1.0",
        "pages": [
            {
                "page_no": p.page_no,
                "error": p.error,
                "regions": [
                    {
                        "index": r.get("index"),
                        "native_label": r.get("native_label"),
                        "bbox_2d": r.get("bbox_2d"),
                        "content": r.get("content"),
                    }
                    for r in p.regions
                ],
            }
            for p in pages
        ],
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def run_glmocr_lane(
    pdf_path: Path | str,
    output_dir: Path | str,
    *,
    sdk_client: Optional[GlmOcrClient] = None,
    alttext_client: Optional[AltTextClient] = None,
    render_dir: Optional[Path | str] = None,
    render_html: bool = False,
    pdf_stem: Optional[str] = None,
) -> GlmOcrLaneResult:
    """Run the GLM-OCR lane on one PDF; write sidecars into ``output_dir``.

    Parameters
    ----------
    pdf_path : the source PDF.
    output_dir : where sidecars (+ HTML when ``render_html``) are written.
    sdk_client : injected OCR client (defaults to the real ``SdkGlmOcrClient``).
    alttext_client : injected alt-text client (defaults to the real seat when
        ``SEMANTIK_ALTTEXT_PROVIDER`` is on).
    render_dir : where page PNGs are rendered (default ``<output_dir>/_glmocr_render``).
    render_html : also render + write ``{stem}_accessible.html`` via the adapter.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    stem = pdf_stem or pdf_path.stem
    render_dir = Path(render_dir) if render_dir else output_dir / "_glmocr_render"

    # 1. render pages
    page_pngs = render_pdf_to_pngs(pdf_path, render_dir)

    # 2. GLM-OCR SDK
    client = sdk_client if sdk_client is not None else SdkGlmOcrClient()
    pages = client.parse_pages(page_pngs)
    # attach render dims/paths to pages for the alt-text crop step
    page_images: Dict[int, str] = {}
    for i, p in enumerate(pages):
        if p.image_path is None and i < len(page_pngs):
            p.image_path = str(page_pngs[i])
        if p.image_path:
            page_images[p.page_no] = p.image_path

    # 3. deterministic transform → wire contract + escalations
    tr: TransformResult = transform_document(pages)

    # 4. optional alt-text/caption generation
    generated = 0
    if resolve_alttext_provider() != "off":
        generated = apply_alt_text(
            tr.region_provenance, page_images,
            client=alttext_client, escalations=tr.escalations,
        )

    # 5. sidecars
    layout_path = output_dir / f"{stem}.glmocr_layout.json"
    escal_path = output_dir / f"{stem}.glmocr_escalations.jsonl"
    _write_layout_sidecar(layout_path, pages)
    write_escalations_sidecar(escal_path, tr.escalations)

    result = GlmOcrLaneResult(
        pdf=str(pdf_path),
        region_provenance=tr.region_provenance,
        heading_tree=tr.heading_tree,
        escalations=tr.escalations,
        pages=pages,
        layout_sidecar=str(layout_path),
        escalations_sidecar=str(escal_path),
        alt_text_generated=generated,
    )

    # 6. optional accessible-HTML render via the lib/semantik adapter
    if render_html:
        html = render_accessible_html(result, pdf_stem=stem)
        html_path = output_dir / f"{stem}_accessible.html"
        html_path.write_text(html, encoding="utf-8")
        result.html = html
        result.html_path = str(html_path)

    return result


def build_cascade_result_dict(
    result: GlmOcrLaneResult, *, pdf_path: str, return_html: bool = False
) -> Dict[str, Any]:
    """Shape a ``run_full_cascade``-compatible result dict from the lane output.

    Consumed by the ``run_pipeline_v2`` GLM-OCR branch so the Ed4All seam
    renders the accessible HTML from ``region_provenance`` exactly as it does
    for the omni cascade.
    """
    out: Dict[str, Any] = {
        "pdf": pdf_path,
        "region_provenance": result.region_provenance,
        "heading_tree": [list(t) for t in result.heading_tree],
        "wcag_status_under_mock": "not_evaluated",
        "lane_used": "glmocr",
        "theta": {"action": "ship_with_flag", "theta_score": None, "flags": []},
        "glmocr_escalations": result.escalations,
        "html_length": 0,
    }
    if return_html:
        out["html"] = ""  # rendered by the Ed4All seam from region_provenance
    return out


def render_accessible_html(
    result: GlmOcrLaneResult, *, pdf_stem: str
) -> str:
    """Render the accessible HTML via the lib/semantik adapter (in-process).

    Mirrors the Ed4All seam recipe (``build_chapters_ir`` → attach ``chapters``
    → ``normalize_cascade_to_ed4all``). Raises ``ImportError`` when
    ``lib/semantik`` is not on the path (pure SemantiK venv) — the caller then
    relies on the Ed4All seam to render.
    """
    from lib.semantik.adapter import normalize_cascade_to_ed4all
    from lib.semantik.cascade_ir import build_chapters_ir

    cascade_dict: Dict[str, Any] = {
        "region_provenance": result.region_provenance,
        "heading_tree": [list(t) for t in result.heading_tree],
        "wcag_status": "not_evaluated",
        "exit_action": "ship_with_flag",
        "theta_score": None,
        "flags": [],
        "lane_used": "glmocr",
    }
    cascade_dict["chapters"] = build_chapters_ir(cascade_dict)
    adapter_out = normalize_cascade_to_ed4all(cascade_dict, pdf_stem=pdf_stem)
    return adapter_out["html"]


__all__ = [
    "GlmOcrLaneResult",
    "build_cascade_result_dict",
    "render_accessible_html",
    "run_glmocr_lane",
]
