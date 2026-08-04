"""Standalone Super heading-level JUDGE runner (validation surface).

Judge the pending headings of an ALREADY-EMITTED GLM-OCR layout sidecar
(``{stem}.glmocr_layout.json`` + ``{stem}.glmocr_escalations.jsonl``) WITHOUT
re-running OCR — the layout pages are re-transformed via
:func:`transform.transform_document` (deterministic, no seat), the skeleton is
built + judged on the heading-judge seat, and the clamp rules are applied on a
DEEP COPY. It NEVER modifies its inputs.

CLI
---
``python3 -m semantik_structure.glmocr.heading_judge_standalone <layout.json> \
    [--out DIR] [--apply] [--source-html PRIOR_HTML]``

Report-only (default) writes ``<out>/<stem>.heading_judgments.json`` (the
skeleton digest, pending count, per-heading judgments, tallies). ``--apply``
ALSO writes ``<out>/<stem>.corrected_layout.json`` (the corrected
region_provenance + heading_tree), rewrites a corrected escalations sidecar
``<out>/<stem>.glmocr_escalations.jsonl`` (heading_level_judged rows), and — when
``lib/semantik`` imports — re-renders ``<out>/<stem>_accessible.html`` via the
lane's render path. Both invocations leave the SOURCE files untouched.

``python3 -m semantik_structure.glmocr.heading_judge <layout.json>`` reaches the
same ``main`` (the module delegates), so either entry point works.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .transform import GlmPage, transform_document

logger = logging.getLogger(__name__)


def _load_layout_pages(layout_path: Path) -> List[GlmPage]:
    """Rebuild the ordered ``GlmPage`` list from a layout sidecar."""
    doc = json.loads(layout_path.read_text(encoding="utf-8"))
    pages: List[GlmPage] = []
    for p in doc.get("pages", []):
        pages.append(GlmPage(
            page_no=int(p.get("page_no", 0)),
            regions=list(p.get("regions") or []),
            error=p.get("error"),
        ))
    return pages


def run_standalone(
    layout_path: Path,
    *,
    out_dir: Path,
    apply: bool = False,
    seat=None,
    post_fn=None,
    use_cache: Optional[bool] = None,
    source_html: Optional[Path] = None,
) -> Dict[str, Any]:
    """Judge one layout sidecar; return the report dict. Never touches inputs.

    ``source_html`` (``--source-html``): path to the PRIOR rendered
    ``{stem}_accessible.html`` (the enriched conversion output). The layout
    sidecar carries the RAW GLM pages only — the VLM alt-text enrichment
    (``SEMANTIK_ALTTEXT_PROVIDER``) lives solely in the prior render — so the
    ``--apply`` re-render would otherwise DEGRADE every VLM-captioned figure
    to the ``sr-only "Figure."`` placeholder. When given, the re-rendered
    HTML has its degraded figure captions/alt restored from the prior render
    via ``lib.semantik.figure_enrich_merge`` (ADD-only; heading levels
    untouched). Best-effort: a merge failure keeps the unmerged re-render.
    """
    from .heading_judge import (
        apply_judged_levels,
        build_heading_skeleton,
        judge_heading_levels,
        resolve_heading_judge_model,
        run_final_review,
    )

    layout_path = Path(layout_path)
    stem = layout_path.name
    for suffix in (".glmocr_layout.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = _load_layout_pages(layout_path)
    tr = transform_document(pages)

    # Deep-copy the transform output so the judge NEVER mutates a shared object
    # that the report render also reads (and so a caller re-using `tr` is safe).
    prov = copy.deepcopy(tr.region_provenance)
    heading_tree = copy.deepcopy([list(t) for t in tr.heading_tree])
    escalations = copy.deepcopy(list(tr.escalations))

    plan = build_heading_skeleton(prov)
    verdict_map: Dict[int, int] = {}
    meta: Dict[str, Any] = {}
    if plan.pending_ids:
        # ``use_cache=None`` defers to ``resolve_heading_judge_checkpoint()``
        # (default ON) so a killed / stopped standalone pass resumes its
        # already-judged windows from the content-addressed sidecar cache.
        verdict_map, meta = judge_heading_levels(
            plan, seat=seat, post_fn=post_fn, use_cache=use_cache)
    result = apply_judged_levels(prov, heading_tree, escalations, verdict_map)

    # Phase B (SEMANTIK_HEADING_JUDGE_FINAL_REVIEW, default OFF): whole-document
    # consistency reconciliation over the just-judged tree, applied on the same
    # DEEP COPY. Returns None (byte-identical) when the gate is off, so the
    # standalone / pipeline `--apply` path is unchanged by default.
    final_review = run_final_review(
        prov, heading_tree, escalations,
        seat=seat, post_fn=post_fn, use_cache=use_cache)

    report = {
        "final_review": final_review,
        "layout": str(layout_path),
        "stem": stem,
        "model": (seat.model if seat else resolve_heading_judge_model()),
        "n_headings": len(plan.entries),
        "n_pending": len(plan.pending_ids),
        # ``applied`` counts VERDICTS RECORDED, not headings re-levelled — a
        # verdict that re-confirms the level already on the region is applied
        # but render-invisible. ``changed``/``agreed``/``transitions`` are the
        # Additive split that keeps the two outcomes distinct:
        # ``applied == changed + agreed``, and only ``changed`` can move an
        # emitted <hN>. An idempotent re-judge of an already-judged conversion
        # legitimately reports applied>0 with changed==0 against the prior
        # render.
        "applied": result.applied,
        "changed": getattr(result, "changed", 0),
        "agreed": getattr(result, "agreed", 0),
        "transitions": dict(getattr(result, "transitions", {}) or {}),
        "clamped": result.clamped,
        "dropped": result.dropped,
        "kept": result.kept,
        # Fix-2 accounting distinction: kept == kept_judged + unjudged, where
        # ``unjudged`` names the truncation hole (no verdict ever covered the
        # heading) that the old report silently folded into ``kept``.
        "kept_judged": getattr(result, "kept_judged", 0),
        "unjudged": getattr(result, "unjudged", 0),
        # Failure mechanism: a chapter that ends
        # unjudged must carry WHY at the TOP level of the report, so the
        # pipeline seam can name it in its operator warning instead of the
        # anonymous "truncation hole" phrasing that hid a killed seat.
        "failure_modes": list(meta.get("failure_modes") or []),
        "unjudged_reason": meta.get("unjudged_reason"),
        "windows": len(plan.windows),
        "judgments": {str(k): {"from": v[0], "to": v[1], "clamped": v[2]}
                      for k, v in result.corrections.items()},
        "skeleton_digest": plan.digest,
        "meta": meta,
    }
    report_path = out_dir / f"{stem}.heading_judgments.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    report["report_path"] = str(report_path)

    if apply:
        corrected_path = out_dir / f"{stem}.corrected_layout.json"
        corrected_path.write_text(
            json.dumps({"region_provenance": prov, "heading_tree": heading_tree},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
        report["corrected_layout_path"] = str(corrected_path)

        from .escalation import write_escalations_sidecar

        escal_path = out_dir / f"{stem}.glmocr_escalations.jsonl"
        write_escalations_sidecar(escal_path, escalations)
        report["escalations_path"] = str(escal_path)

        # Optional re-render via the lane's adapter path (in-process lib/semantik).
        try:
            from .lane import GlmOcrLaneResult, render_accessible_html

            lane_result = GlmOcrLaneResult(
                pdf=str(layout_path), region_provenance=prov,
                heading_tree=[tuple(t) for t in heading_tree],
                escalations=escalations)
            html = render_accessible_html(lane_result, pdf_stem=stem)
            # Figure-enrichment preservation: the layout sidecar has no VLM
            # alt-text enrichment, so restore degraded figure captions/alt
            # from the prior enriched render (ADD-only; headings untouched).
            if source_html is not None and Path(source_html).is_file():
                try:
                    from lib.semantik.figure_enrich_merge import (
                        merge_figure_enrichment,
                    )

                    prior = Path(source_html).read_text(encoding="utf-8")
                    html, restored = merge_figure_enrichment(prior, html)
                    report["figure_enrichment_restored"] = restored
                    report["figure_enrichment_source"] = str(source_html)
                except Exception as exc:  # noqa: BLE001 — never lose the render
                    logger.warning(
                        "figure-enrichment merge failed (unmerged re-render "
                        "kept) for %s: %s", stem, exc)
            html_path = out_dir / f"{stem}_accessible.html"
            html_path.write_text(html, encoding="utf-8")
            report["html_path"] = str(html_path)
        except Exception as exc:  # noqa: BLE001 — render is optional (no lib/)
            logger.info("standalone re-render skipped (lib/semantik absent?): %s", exc)

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Super heading-level judge over a GLM-OCR layout sidecar "
                    "(report-only by default; --apply to write corrected output).")
    parser.add_argument("layout", help="path to a {stem}.glmocr_layout.json sidecar")
    parser.add_argument("--out", default="heading_judge_out",
                        help="output directory (default ./heading_judge_out)")
    parser.add_argument("--apply", action="store_true",
                        help="also write corrected layout + escalations + HTML")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the per-window resume sidecar cache "
                             "(default: cache ON per "
                             "SEMANTIK_HEADING_JUDGE_CHECKPOINT)")
    parser.add_argument("--source-html", default=None,
                        help="path to the PRIOR rendered {stem}_accessible."
                             "html (the enriched conversion output); with "
                             "--apply, degraded figure captions/alt in the "
                             "re-render are restored from it (the layout "
                             "sidecar carries no VLM alt-text enrichment)")
    args = parser.parse_args(argv)

    layout = Path(args.layout)
    if not layout.is_file():
        print(f"error: layout sidecar not found: {layout}", file=sys.stderr)
        return 2
    report = run_standalone(layout, out_dir=Path(args.out), apply=args.apply,
                            use_cache=False if args.no_cache else None,
                            source_html=(Path(args.source_html)
                                         if args.source_html else None))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "skeleton_digest"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
