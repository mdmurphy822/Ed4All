#!/usr/bin/env python3
"""Cross-venv cascade runner — runs the SemantiK v2 cascade and writes a
JSON-serializable bridge file (migration plan §3.3a item 2 / §5 / Section 8 M4).

This script runs INSIDE the SemantiK RUNTIME venv (the one with
axe-playwright-python / llama-cpp-CUDA / torch + the 11GB GGUF/council
weights), NOT inside Ed4All's slim venv. It is the SemantiK-side half of the
subprocess bridge that ``MCP/tools/pipeline_tools.py::_run_semantik_v2_conversion``
shells out to when the in-process ``run_pipeline_v2`` import is unavailable
(the heavy deps are absent from Ed4All's venv by design).

The cascade (``dart_semantic.cascade.run_pipeline_v2``) resolves its models
relative to the current working directory (council/theta/qwen/figure-router
caches). Therefore this script MUST be invoked with ``cwd`` set to the
SemantiK runtime repo root (the Ed4All seam passes
``cwd=SEMANTIK_RUNTIME_DIR``), OR the operator must export the model-dir env
the cascade reads. The script itself is cwd-independent in its own imports
(it imports ``dart_semantic`` which the runtime venv has installed/on-path),
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
      "runtime_mode":       <str>         # "real" | "mock" (R4 mock-trap input)
    }

On FAILURE — a clear error object (the seam fails closed on this, NEVER a
silent DART fallback)::

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
# ``dart_semantic`` an interpreter's venv may have editable-installed (e.g. the
# Semantic source repo whose venv we borrow for the heavy runtime deps). Only
# the vendored copy carries the relocatable ``dart_semantic.paths`` resolver
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
    # llama_cpp / torch / dart_semantic not on path) becomes a structured
    # {"error": ...} JSON rather than a bare traceback the seam can't parse.
    try:
        from dart_semantic.cascade import run_pipeline_v2
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
