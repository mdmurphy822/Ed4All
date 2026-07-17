"""Super-escalation contract for the GLM-OCR lane.

The lane does NOT rebuild SemantiK's reasoning passes (resegment / reviewer /
reasoning-QC). It emits their INPUT: a JSONL sidecar ``{stem}.glmocr_escalations.jsonl``
of flagged spans a downstream Super pass should reason over. This module owns
the sidecar SCHEMA + writer; the flagged records themselves are produced by the
deterministic transform (:mod:`.transform`) and the alt-text pass
(:mod:`.alttext`).

FOLLOW-UP WIRING (documented extension point — intentionally NOT built here):
the existing Super passes (``super_resegment`` / the Stage-5d reviewer /
``reasoning_qc``) consume this sidecar in a later task — each record carries a
``region_index`` + ``source_page`` + ``native_label`` + ``reason`` so the Super
pass can locate the span and re-reason over it. Nothing in this module
dispatches an LLM.

Escalation ``reason`` vocabulary (each a deterministic detector output):
  * ``sdk_error``            — a page whose SDK OCR failed
  * ``unpaired_ordinal``     — an exercise number with no paired formula
  * ``unpaired_formula``     — a display_formula with no paired ordinal
  * ``heading_level_pending``— a non-numbered title; level defaulted, ambiguous
  * ``caption_less_figure``  — a figure with no extracted caption (needs alt)
  * ``alt_text_generation_failed`` / ``alt_text_no_crop`` — the seat/crop failed
  * ``cross_page_stitch``    — a below-confidence continuation merge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

ESCALATION_SCHEMA_VERSION = "glmocr-escalation/1.0"

_REQUIRED_KEYS = ("reason",)


def normalise_escalation(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a schema-complete escalation record (fills optional keys)."""
    out = {
        "schema": ESCALATION_SCHEMA_VERSION,
        "region_index": record.get("region_index"),
        "source_page": record.get("source_page"),
        "native_label": record.get("native_label", ""),
        "reason": record.get("reason", "unknown"),
        "detail": record.get("detail", ""),
    }
    if "text" in record:
        out["text"] = record["text"]
    return out


def write_escalations_sidecar(
    path: Path, escalations: Sequence[Dict[str, Any]]
) -> Path:
    """Write the escalations JSONL sidecar (one normalised record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in escalations:
            fh.write(json.dumps(normalise_escalation(rec), ensure_ascii=False))
            fh.write("\n")
    return path


__all__ = [
    "ESCALATION_SCHEMA_VERSION",
    "normalise_escalation",
    "write_escalations_sidecar",
]
