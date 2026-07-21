#!/usr/bin/env python3
"""Cross-venv cascade runner — runs the SemantiK v2 cascade and writes a
JSON-serializable bridge file (migration plan §3.3a item 2 / §5 / Section 8 M4).

This script runs INSIDE the SemantiK RUNTIME venv (the one with
axe-playwright-python / llama-cpp-CUDA / torch + the 11GB GGUF/council
weights), NOT inside Ed4All's slim venv. It is the SemantiK-side half of the
subprocess bridge that ``MCP/tools/pipeline_tools.py::_run_semantik_v2_conversion``
shells out to when the in-process ``run_pipeline_v2`` import is unavailable
(the heavy deps are absent from Ed4All's venv by design).

The cascade (``semantik_structure.cascade.run_pipeline_v2``) resolves its models
relative to the current working directory (council/theta/qwen/figure-router
caches). Therefore this script MUST be invoked with ``cwd`` set to the
SemantiK runtime repo root (the Ed4All seam passes
``cwd=SEMANTIK_RUNTIME_DIR``), OR the operator must export the model-dir env
the cascade reads. The script itself is cwd-independent in its own imports
(it imports ``semantik_structure`` which the runtime venv has installed/on-path),
but the cascade's relative model resolution is NOT — hence the cwd contract.

Output contract (the JSON file written to ``--out-json``):

On SUCCESS — all JSON-serializable fields the P3a/P3b seam consumes::

    {
      "pdf":                <str>,        # the input PDF path
      "html":               <str>,        # assembled accessible HTML
      "region_provenance":  [ {region_index, region_kind, role, confidence,
                               wcag_status, first_raw_block_index, pages,
                               heading_text, level, figure_alt, raw_text}, ... ],
      "heading_tree":       [ [level, text], ... ],
      "exit_action":        <str>,        # Stage-13 ConfidenceAction value
      "theta_score":        <float|null>,
      "wcag_status":        <str>,        # "passed" | "failed"
      "flags":              [<str>, ...], # ThetaFlag values
      "lane_used":          <str>,
      "runtime_mode":       <str>,        # "real" | "mock" (R4 mock-trap input)
      "structure_review":   [ {block_id, verdict, kind_before, kind_after,
                               level_before, level_after, review_note,
                               reverted_for_invariant}, ... ] | null,
                                          # Stage-5d reviewer verdicts; null
                                          # when SEMANTIK_STRUCTURE_REVIEW is off
      "ocr_repair":         { ...stats } | null,   # SEMANTIK_OCR_CONFUSABLE_REPAIR
      "block_resegment":    [ {op, source_ids, ...}, ... ] | null,
                                          # Stage-5e SEMANTIK_BLOCK_RESEGMENT ops;
                                          # null when the re-segmenter is off
      "second_pass_verify": { adopted, rounds:[...] } | null,
                                          # Stage-9 SEMANTIK_SECOND_PASS per-round
                                          # audit; null when the loop is off
      "geom_order":         { mode, applied, doc_divergence, pages:[...],
                               bboxless_regions, warnings } | null
                                          # ITEM5 Stage-5 SEMANTIK_REGION_ORDER
                                          # divergence audit; null when off
      "unit_coverage":      { gate, enabled, min_coverage, page_floor,
                               document_passed, warned_pages, failed_pages,
                               pages:[...] } | null
                                          # task #43 SEMANTIK_UNIT_COVERAGE_GATE
                                          # content-loss report; null when off
      "page_arranger":      { ...audit } | null
                                          # task #43 SEMANTIK_PAGE_ARRANGER
                                          # scan-lane audit; null when off
    }

On FAILURE — a clear error object (the seam fails closed on this, NEVER a
silent fallback)::

    {"error": "<message>", "runtime_mode": <str|null>, "pdf": <str>}

The error arm catches import failures + cascade exceptions so the seam reads
a structured ``{"error": ...}`` rather than parsing a bare Python traceback
off a non-zero exit. (The seam ALSO treats a non-zero exit / unreadable JSON
as fail-closed, so either signal is safe.)
"""

from __future__ import annotations

import os

# R7 VRAM-OOM mitigation. The PRIMARY mitigation is theta->CPU
# (``SEMANTIK_THETA_DEVICE=cpu``, the default the seam applies) — that is the
# real OOM fix on the shared 8 GB card.
#
# ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` was a SECONDARY measure
# that BACKFIRED: it can trigger a CUDACachingAllocator INTERNAL ASSERT
# (``!handles_.at(i)`` in CUDACachingAllocator.cpp) inside a council BERT peft
# lora forward under some driver/torch combos. It is therefore now OPT-IN via
# ``SEMANTIK_EXPANDABLE_SEGMENTS`` and OFF by default — by default this script
# does NOT touch ``PYTORCH_CUDA_ALLOC_CONF`` at all. When opted in, ``setdefault``
# still means an operator's explicit ``PYTORCH_CUDA_ALLOC_CONF`` is never
# clobbered. This MUST run before any torch import; the cascade import (which
# pulls torch) is lazy inside ``main()``, so module-top placement is well ahead.
_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(name: str) -> bool:
    """True iff env var ``name`` holds a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _maybe_set_alloc_conf() -> None:
    """Gated opt-in: set the expandable-segments allocator conf ONLY when
    ``SEMANTIK_EXPANDABLE_SEGMENTS`` is truthy. ``setdefault`` so an operator's
    explicit ``PYTORCH_CUDA_ALLOC_CONF`` is never clobbered."""
    if _truthy("SEMANTIK_EXPANDABLE_SEGMENTS"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_maybe_set_alloc_conf()

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Prioritize the VENDORED SemantiK package (this script's repo) over any
# ``semantik_structure`` an interpreter's venv may have editable-installed (e.g. the
# Semantic source repo whose venv we borrow for the heavy runtime deps). Only
# the vendored copy carries the relocatable ``semantik_structure.paths`` resolver
# (honors SEMANTIK_MODEL_DIR) + the region_provenance surfacing + the R7 theta
# device knob; the installed copy uses CWD-relative model paths and would break.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _coerce_region_provenance(result: Any) -> List[Dict[str, Any]]:
    """Pull the JSON-safe ``region_provenance`` list off the cascade result.

    P3a promoted ``region_provenance`` to a top-level ``PipelineV2Result``
    field (``list[dict]``); fall back to the telemetry ``cascade`` dict for a
    runtime that pre-dates the promotion. Every element is already a plain
    dict of JSON scalars (see ``cascade._build_region_provenance``).
    """
    prov = getattr(result, "region_provenance", None)
    if not prov:
        cascade = getattr(result, "cascade", None)
        if isinstance(cascade, dict):
            prov = cascade.get("region_provenance")
    out: List[Dict[str, Any]] = []
    for region in prov or []:
        if isinstance(region, dict):
            out.append(dict(region))
    return out


def _coerce_heading_tree(result: Any) -> List[List[Any]]:
    """``heading_tree`` as a JSON list of ``[level, text]`` pairs."""
    ht = getattr(result, "heading_tree", None)
    if ht is None:
        cascade = getattr(result, "cascade", None)
        if isinstance(cascade, dict):
            ht = cascade.get("heading_tree")
    return [list(entry) for entry in (ht or [])]


def _resolve_runtime_mode(result: Any) -> Optional[str]:
    """``runtime_mode`` lives on the telemetry ``cascade`` dict, not the
    top-level ``PipelineV2Result``; promote it so the bridge JSON carries it
    at top level for the seam's R4 mock-trap (it never ships mock as real)."""
    mode = getattr(result, "runtime_mode", None)
    if mode is None:
        cascade = getattr(result, "cascade", None)
        if isinstance(cascade, dict):
            mode = cascade.get("runtime_mode")
    return str(mode) if mode is not None else None


def _resolve_structure_review(result: Any) -> Optional[List[Dict[str, Any]]]:
    """Pull the doc-level Stage-5d ``structure_review`` audit off the cascade
    result and promote it to a top-level bridge key (Phase 3 item 7).

    The audit lives at ``result.cascade["conformance_audit"]["structure_review"]``
    (``SemantiK/semantik_structure/cascade.py`` builds it as a list of
    ``ReviewVerdict``-as-dicts when the Stage-5d reviewer ran, ``None`` when
    the reviewer was off). The per-region ``review`` keys already ride through
    ``_coerce_region_provenance`` verbatim (it copies each region dict); this
    surfaces the DOCUMENT-level verdict list so the Ed4All seam can emit one
    ``structure_review`` DecisionCapture event per converted doc.

    The ``None`` (reviewer-did-not-run) vs ``[]`` (ran, found nothing)
    distinction is preserved — the seam skips the capture cleanly on ``None``.
    A runtime that pre-dates the audit (no ``conformance_audit`` /
    ``structure_review`` key) resolves to ``None`` (treated as not-run).
    """
    cascade = getattr(result, "cascade", None)
    audit: Any = None
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict):
            audit = conformance.get("structure_review")
    if audit is None:
        return None
    out: List[Dict[str, Any]] = []
    for verdict in audit if isinstance(audit, (list, tuple)) else []:
        if isinstance(verdict, dict):
            out.append(dict(verdict))
    return out


def _resolve_ocr_repair(result: Any) -> Optional[Dict[str, Any]]:
    """Pull the doc-level SEMANTIK_OCR_CONFUSABLE_REPAIR audit stats off the
    cascade result and promote it to a top-level bridge key (Phase 5).

    The stats live at ``result.cascade["conformance_audit"]["ocr_repair"]`` (or
    the ``result.cascade["ocr_repair"]`` top-level arm), a DICT when the pass
    ran and ``None`` when it was off. The seam reads this to emit ONE
    ``structure_review`` DecisionCapture (``ocr_confusable_repair`` discriminator)
    per converted doc; ``None`` (pass off) is preserved.
    """
    cascade = getattr(result, "cascade", None)
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict):
            arm = conformance.get("ocr_repair")
            if isinstance(arm, dict):
                return arm
        arm = cascade.get("ocr_repair")
        if isinstance(arm, dict):
            return arm
    return None


def _resolve_block_resegment(result: Any) -> Optional[List[Dict[str, Any]]]:
    """Pull the doc-level SEMANTIK_BLOCK_RESEGMENT op list off the cascade
    result and promote it to a top-level bridge key (Phase 9).

    ``run_full_cascade`` emits the audit at the result-dict TOP LEVEL
    (``result.cascade["block_resegment"]`` — ``cascade.py`` ~L2217), a LIST when
    the re-segmenter ran (``[]`` when it ran and made no edits) and ``None``
    (key absent) when the pass was off; a runtime that nested it under
    ``conformance_audit`` is also accepted (checked first). The seam reads this
    to emit ONE ``block_resegment`` DecisionCapture per converted doc; the
    ``None`` (pass off) vs ``[]`` (ran, no edits) distinction is preserved.
    """
    cascade = getattr(result, "cascade", None)
    audit: Any = None
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict) and conformance.get("block_resegment") is not None:
            audit = conformance.get("block_resegment")
        elif cascade.get("block_resegment") is not None:
            audit = cascade.get("block_resegment")
    if audit is None:
        return None
    out: List[Dict[str, Any]] = []
    for op in audit if isinstance(audit, (list, tuple)) else []:
        if isinstance(op, dict):
            out.append(dict(op))
    return out


def _resolve_second_pass_verify(result: Any) -> Optional[Dict[str, Any]]:
    """Pull the Stage-9 Pass-2 verify-refine per-round audit arm off the cascade
    result and promote it to a top-level bridge key (Phase 6/7).

    Unlike ``structure_review`` / ``block_resegment`` (lists), this arm is a
    DICT (``{"adopted": ..., "rounds": [...]}``); ``run_full_cascade`` emits it
    at the result-dict top level (``result.cascade["second_pass_verify"]`` —
    ``cascade.py`` ~L2226) and ``None`` (key absent) when SEMANTIK_SECOND_PASS
    was off / no round ran. A conformance-nested runtime is also accepted
    (checked first). The seam reads this to emit ONE ``structure_review``
    DecisionCapture (``second_pass_verify`` discriminator) PER verify round.
    """
    cascade = getattr(result, "cascade", None)
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict):
            arm = conformance.get("second_pass_verify")
            if isinstance(arm, dict):
                return dict(arm)
        arm = cascade.get("second_pass_verify")
        if isinstance(arm, dict):
            return dict(arm)
    return None


def _resolve_geom_order(result: Any) -> Optional[Dict[str, Any]]:
    """Pull the ITEM5 Stage-5 geometric region-order divergence audit DICT off
    the cascade result and promote it to a top-level bridge key.

    Like ``second_pass_verify`` this is a DICT (``{"mode", "applied",
    "doc_divergence", "pages", "bboxless_regions", "warnings"}``);
    ``run_full_cascade`` emits it at the result-dict top level
    (``result.cascade["geom_order"]``) and ``None`` when SEMANTIK_REGION_ORDER
    resolved ``off`` (the pass never ran). A conformance-nested runtime is also
    accepted (checked first). Operator/tooling-facing only (no DecisionCapture —
    a deterministic pass).
    """
    cascade = getattr(result, "cascade", None)
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict):
            arm = conformance.get("geom_order")
            if isinstance(arm, dict):
                return dict(arm)
        arm = cascade.get("geom_order")
        if isinstance(arm, dict):
            return dict(arm)
    return None


def _resolve_reasoning_qc(result: Any) -> Optional[Dict[str, Any]]:
    """Pull the Stage-9b reasoning-QC audit DICT off the cascade result and
    promote it to a top-level bridge key.

    Like ``second_pass_verify`` / ``geom_order`` this is a DICT (``{"schema",
    "mode", "ran", "model", "n_windows", "toc", "windows", "flagged",
    "retype", "merge_move", ...}``); ``run_full_cascade`` emits it at the
    result-dict top level (``result.cascade["reasoning_qc"]``) and ``None``
    (key absent) when SEMANTIK_REASONING_QC resolved ``off`` (the pass never
    ran). A conformance-nested runtime is also accepted (checked first). The
    Ed4All seam reads this to emit ONE ``structure_detection`` DecisionCapture
    per converted doc off the doc-level QC judgment tally.
    """
    cascade = getattr(result, "cascade", None)
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict):
            arm = conformance.get("reasoning_qc")
            if isinstance(arm, dict):
                return dict(arm)
        arm = cascade.get("reasoning_qc")
        if isinstance(arm, dict):
            return dict(arm)
    return None


def _resolve_unit_coverage(result: Any) -> Optional[Dict[str, Any]]:
    """Pull the SEMANTIK_UNIT_COVERAGE_GATE fidelity report DICT off the cascade
    result and promote it to a top-level bridge key (task #43).

    Like ``second_pass_verify`` / ``geom_order`` / ``reasoning_qc`` this is a
    DICT (``{gate, enabled, min_coverage, page_floor, document_passed,
    warned_pages, failed_pages, pages:[...]}``); ``run_full_cascade`` emits it at
    the result-dict top level (``result.cascade["unit_coverage"]``) and the key
    is ABSENT when SEMANTIK_UNIT_COVERAGE_GATE resolved off (the gate never ran).
    A conformance-nested runtime is also accepted (checked first). Deterministic
    content-loss gate — operator/tooling-facing only (no DecisionCapture).
    """
    cascade = getattr(result, "cascade", None)
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict):
            arm = conformance.get("unit_coverage")
            if isinstance(arm, dict):
                return dict(arm)
        arm = cascade.get("unit_coverage")
        if isinstance(arm, dict):
            return dict(arm)
    return None


def _resolve_page_arranger(result: Any) -> Optional[Dict[str, Any]]:
    """Pull the SEMANTIK_PAGE_ARRANGER audit DICT off the cascade result and
    promote it to a top-level bridge key (task #43).

    Like ``unit_coverage`` this is a DICT (arranger per-page/typing audit);
    ``run_full_cascade`` emits it at the result-dict top level
    (``result.cascade["page_arranger"]``) and the key is ABSENT when the
    page-arranger did not own structure (SEMANTIK_PAGE_ARRANGER off /
    born-digital lane). A conformance-nested runtime is also accepted (checked
    first). Operator/tooling-facing audit — the scan-lane structure-authority
    DecisionCapture rides the in-process arm; this forwards the audit dict so the
    cross-venv IR emit surfaces it too.
    """
    cascade = getattr(result, "cascade", None)
    if isinstance(cascade, dict):
        conformance = cascade.get("conformance_audit")
        if isinstance(conformance, dict):
            arm = conformance.get("page_arranger")
            if isinstance(arm, dict):
                return dict(arm)
        arm = cascade.get("page_arranger")
        if isinstance(arm, dict):
            return dict(arm)
    return None


def _build_bridge_dict(result: Any, pdf: str) -> Dict[str, Any]:
    """Assemble the JSON-serializable bridge dict from a cascade result."""
    theta = getattr(result, "theta_score", None)
    return {
        "pdf": pdf,
        "html": str(getattr(result, "html", "") or ""),
        "region_provenance": _coerce_region_provenance(result),
        "heading_tree": _coerce_heading_tree(result),
        "exit_action": getattr(result, "exit_action", None),
        "theta_score": (float(theta) if theta is not None else None),
        "wcag_status": getattr(result, "wcag_status", None),
        "flags": list(getattr(result, "flags", None) or []),
        "lane_used": getattr(result, "lane_used", None),
        "runtime_mode": _resolve_runtime_mode(result),
        # Phase 3 item 7 — the doc-level Stage-5d structure-review verdict list
        # (None when the reviewer was off). The per-region ``review`` keys
        # already ride through ``region_provenance`` verbatim.
        "structure_review": _resolve_structure_review(result),
        # Phase 5 — the doc-level OCR-confusable repair audit stats (None when
        # the pass was off). The per-region ``repaired_text`` / ``ocr_repair``
        # keys already ride through ``region_provenance`` verbatim.
        "ocr_repair": _resolve_ocr_repair(result),
        # Phase 9 — the doc-level SEMANTIK_BLOCK_RESEGMENT op list (None when the
        # re-segmenter was off). The Ed4All seam emits one ``block_resegment``
        # DecisionCapture per doc off this arm (the bridge arm was previously
        # unforwarded — the capture only fired on the in-process path).
        "block_resegment": _resolve_block_resegment(result),
        # Phase 6/7 — the Stage-9 Pass-2 verify-refine per-round audit DICT
        # (None when SEMANTIK_SECOND_PASS was off / no round ran). The seam
        # emits one ``structure_review`` DecisionCapture per verify round off it.
        "second_pass_verify": _resolve_second_pass_verify(result),
        # ITEM5 — the Stage-5 geometric region-order divergence audit DICT
        # (None when SEMANTIK_REGION_ORDER resolved off). Operator/tooling-facing
        # only; the region_provenance ORDER already reflects the applied order.
        "geom_order": _resolve_geom_order(result),
        # Stage 9b — the reasoning-QC audit DICT (None when SEMANTIK_REASONING_QC
        # resolved off). The Ed4All seam emits one ``structure_detection``
        # DecisionCapture per doc off this arm; the per-region re-type/reorder
        # effects already ride through ``region_provenance`` verbatim.
        "reasoning_qc": _resolve_reasoning_qc(result),
        # task #43 — the SEMANTIK_UNIT_COVERAGE_GATE deterministic content-loss
        # fidelity report DICT (None when the gate was off). The Ed4All seam
        # threads it into the emitted cascade_ir.json + surfaces below-floor
        # pages as quality-sidecar issues (previously dropped on the bridge).
        "unit_coverage": _resolve_unit_coverage(result),
        # task #43 — the SEMANTIK_PAGE_ARRANGER scan-lane structure-authority
        # audit DICT (None when the arranger did not own structure). Forwarded so
        # the emitted cascade_ir.json carries it on the cross-venv path too.
        "page_arranger": _resolve_page_arranger(result),
    }


def _write_json(out_json: Path, payload: Dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SemantiK v2 cascade and write a JSON bridge file the "
            "Ed4All seam consumes (cross-venv subprocess boundary)."
        )
    )
    parser.add_argument("--pdf", required=True, help="Input PDF path.")
    parser.add_argument(
        "--out-json", required=True, help="Destination JSON bridge file."
    )
    parser.add_argument(
        "--runtime",
        choices=("auto", "real", "mock"),
        default="auto",
        help=(
            "Force runtime mode. 'auto' (default) lets the cascade derive it "
            "from the loaded config (real iff the Qwen specialists are present)."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Candidates per region (Stage 6)."
    )
    parser.add_argument(
        "--max-regions-per-kind",
        type=int,
        default=None,
        help="Region cap per kind (0 = no cap). Smoke harnesses pass small ints.",
    )
    args = parser.parse_args(argv)

    out_json = Path(args.out_json)
    pdf = args.pdf

    # Lazy/guarded import — a missing runtime dep (axe_playwright_python /
    # llama_cpp / torch / semantik_structure not on path) becomes a structured
    # {"error": ...} JSON rather than a bare traceback the seam can't parse.
    try:
        from semantik_structure.cascade import run_pipeline_v2
    except Exception as exc:  # noqa: BLE001 — any import failure → clear error
        _write_json(
            out_json,
            {
                "error": (
                    "SemantiK runtime import failed (run this script from the "
                    f"SemantiK runtime venv, cwd=the SemantiK repo root): {exc}"
                ),
                "runtime_mode": None,
                "pdf": pdf,
                "traceback": traceback.format_exc(),
            },
        )
        return 2

    runtime_mode = None if args.runtime == "auto" else args.runtime

    kwargs: Dict[str, Any] = {}
    if runtime_mode is not None:
        kwargs["runtime_mode"] = runtime_mode
    if args.k is not None:
        kwargs["k"] = args.k
    if args.max_regions_per_kind is not None:
        kwargs["max_regions_per_kind"] = args.max_regions_per_kind

    try:
        result = run_pipeline_v2(pdf, **kwargs)
    except Exception as exc:  # noqa: BLE001 — cascade error → clear error JSON
        _write_json(
            out_json,
            {
                "error": f"SemantiK cascade failed: {exc}",
                "runtime_mode": runtime_mode,
                "pdf": pdf,
                "traceback": traceback.format_exc(),
            },
        )
        return 3

    try:
        payload = _build_bridge_dict(result, pdf)
        _write_json(out_json, payload)
    except Exception as exc:  # noqa: BLE001 — serialization failure → error JSON
        _write_json(
            out_json,
            {
                "error": f"bridge serialization failed: {exc}",
                "runtime_mode": _resolve_runtime_mode(result),
                "pdf": pdf,
                "traceback": traceback.format_exc(),
            },
        )
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
