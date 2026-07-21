"""Stage-5e block JOIN/SPLIT prompt builder (boundary-only, no-rewrite).

The OPTIONAL Stage-3 LLM layer for the block-resegment pass
(``block_resegment.py``). Mirrors ``reviewer_prompt.py``'s posture: a
self-contained ``SYSTEM:`` line carrying a STRUCTURE-only, NO-REWRITE,
JSON-only mandate + the region digest as the ``USER:`` turn.

The model may ONLY propose JOIN (merge adjacent same-kind regions) / SPLIT
(cut a region at a FeatureBlock boundary) operations — it never sees or emits
text it could rewrite. Every proposed op is re-validated by the R-PART
conservation gates in ``block_resegment.resegment_blocks`` (the JSON contract
is NOT the safety net — the gates are), so a hallucinated / unsafe op is
dropped without touching the deterministic result.
"""

from __future__ import annotations

import json
from typing import Any

from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock

# How much of each region's text to show the model (boundaries, not content).
_REGION_SNIPPET_CHARS = 240


_SYSTEM_RESEGMENT = (
    "You are a SemantiK document-STRUCTURE BOUNDARY editor. You are NOT a content "
    "author or generator. Your single job is to fix LOGICAL PRESENTATION "
    "defects by JOINING regions that were wrongly cut apart or SPLITTING a "
    "region that fused two logical units — by MOVING region boundaries only.\n"
    "\n"
    "HARD CONSTRAINT — VERBATIM TEXT, BOUNDARIES ONLY. You MUST NEVER rewrite, "
    "paraphrase, summarize, add, or remove any source TEXT. The atom is the "
    "FeatureBlock (FB): a MERGE concatenates whole FBs; a SPLIT cuts ONLY at "
    "an FB boundary, never inside an FB. You do NOT emit HTML or any text — "
    "you emit ONLY a list of boundary operations.\n"
    "\n"
    "POSTURE — CONSERVATIVE. Your DEFAULT is to propose NOTHING (an empty ops "
    "list). Propose an op ONLY when the region digest makes a defect clear: a "
    "list/paragraph shattered across a page break (MERGE the adjacent "
    "same-kind regions), or one region that fused two units — e.g. a new "
    "'EXAMPLE 2' / 'Solution' / 'Step 3' unit starting mid-region (SPLIT "
    "before the FB that starts it). NEVER merge a heading with body text "
    "(merge SAME-KIND adjacent regions only). NEVER touch a table or math "
    "region.\n"
    "\n"
    "OUTPUT — JSON ONLY. Return EXACTLY ONE JSON object and nothing else. NO "
    "Markdown, NO code fences, NO commentary. The object has one key, \"ops\", "
    "an array. Each MERGE op is {\"op\":\"merge\",\"region_indices\":[i,i+1,...]} "
    "(a CONTIGUOUS ascending run of >=2 region indices). Each SPLIT op is "
    "{\"op\":\"split\",\"region_indices\":[i],\"split_at\":[fb,...]} where each "
    "fb is an INTERIOR FeatureBlock index of region i that should START a new "
    "child. When nothing needs changing, return {\"ops\":[]}."
)


def _region_snippet(region: Region, feature_blocks: list[FeatureBlock]) -> str:
    """A short text snippet for the region (collapsed, truncated)."""
    text = str((region.payload or {}).get("text") or "")
    if not text:
        # Fall back to the joined FB text without importing reviewer (avoid a
        # cycle) — cheap inline join.
        parts: list[str] = []
        for idx in region.feature_block_indices or ():
            try:
                fb = feature_blocks[idx]
            except (IndexError, TypeError):
                continue
            parts.append((getattr(getattr(fb, "raw", None), "text", "") or "").strip())
        text = " ".join(p for p in parts if p)
    flat = " ".join(text.split())
    return flat[:_REGION_SNIPPET_CHARS]


def _region_obj(
    region: Region, region_index: int, feature_blocks: list[FeatureBlock]
) -> dict[str, Any]:
    """The per-region digest entry the model reasons over."""
    fb_indices = list(region.feature_block_indices or ())
    pages: list[int] = []
    for idx in fb_indices:
        try:
            fb = feature_blocks[idx]
        except (IndexError, TypeError):
            continue
        page = getattr(getattr(fb, "raw", None), "page", None)
        if page is not None and int(page) not in pages:
            pages.append(int(page))
    obj = {
        "region_index": region_index,
        "kind": region.kind,
        "feature_block_indices": fb_indices,
        "pages": sorted(pages),
        "is_passthrough": getattr(region, "source_region_id", None) is not None,
        "text_snippet": _region_snippet(region, feature_blocks),
    }
    # ITEM6 — additive council role-distribution evidence (stamped on
    # region.provenance by stamp_role_distributions before Stage-5e runs).
    # Absent for unstamped/mock regions -> byte-stable digest.
    role_top_k = (getattr(region, "provenance", {}) or {}).get("role_top_k")
    if role_top_k:
        obj["role_top_k"] = role_top_k
    return obj


def build_resegment_request(
    regions: list[Region], feature_blocks: list[FeatureBlock]
) -> str:
    """Emit a ``SYSTEM:\\nUSER:`` block-resegment prompt for the region list.

    The SYSTEM turn carries the boundary-only / no-rewrite / JSON-only
    mandate; the USER turn is the per-region digest (region_index, kind, FB
    indices, pages, passthrough flag, text snippet) as one JSON line.
    """
    digest = [
        _region_obj(region, idx, feature_blocks)
        for idx, region in enumerate(regions)
    ]
    user_json = json.dumps(
        {"regions": digest}, ensure_ascii=False, separators=(",", ":")
    )
    return f"SYSTEM: {_SYSTEM_RESEGMENT}\nUSER: {user_json}"


__all__ = ["build_resegment_request"]
