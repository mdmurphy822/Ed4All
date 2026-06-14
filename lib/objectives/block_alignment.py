"""W2 §6 — deterministic post-outline block↔objective alignment.

This is the BLOCK-side alignment (every two-pass content Block ties to its
delivering objective(s)), distinct from the CO→TO backlink (§4.3, objective↔
objective). The existing
``lib/validators/block_objective_delivery.py::BlockObjectiveDeliveryValidator`` is
the GATE; this pass populates the field the gate checks. It runs post-outline,
pre-inter-tier, as a small deterministic embedding classifier — no LLM, reusing
the embed client already loaded for Pass D.

Contract:

- A block aligns to every objective with cosine ≥ ``threshold`` AND always to any
  objective it ALREADY cites (``objective_refs[]`` / ``objective_ids`` — the
  model's explicit ref is authoritative; embeddings only ADD missed ones).
- The block's ``objective_refs`` is set to the union (existing ∪ embedding-matched),
  capped at top-K=3 by cosine to avoid over-tagging.
- A block with ZERO aligned objective is LEFT with empty refs + a
  ``no_objective_alignment`` ``structural_warnings`` marker — the gate surfaces it.
  This pass NEVER fabricates a ref to dodge the gate.
- Graceful degrade: embeddings absent → pass-through (refs unchanged; the model's
  outline-tier refs stand).

``objective_refs`` lives inside the outline block's ``content`` dict (the JSON the
outline tier emits); the Block dataclass also carries ``objective_ids`` (the
HTML-stamping list). We read/write the content-dict ``objective_refs`` (the field
``BlockObjectiveDeliveryValidator`` reads) and never touch the frozen Block's own
fields here — the caller re-stamps ``objective_ids`` downstream if needed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from lib.embedding._math import cosine_similarity

#: Default cosine floor for an embedding-added alignment.
_DEFAULT_ALIGN_THRESHOLD = 0.55

#: Cap on objective refs per block (avoid over-tagging).
_TOP_K = 3

#: Structural warning stamped on a block with no aligned objective.
_NO_ALIGNMENT_MARKER = "no_objective_alignment"


def _block_content(block: Any) -> Optional[Dict[str, Any]]:
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        return content
    return None


def _existing_objective_refs(block: Any, content: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for r in content.get("objective_refs") or []:
        if r:
            refs.append(str(r))
    # The Block dataclass's own objective_ids are authoritative too.
    for oid in getattr(block, "objective_ids", None) or ():
        if oid and str(oid) not in refs:
            refs.append(str(oid))
    return refs


def _block_text(content: Dict[str, Any]) -> str:
    """Gather a block's text surface: key_claims statements + section_skeleton."""
    parts: List[str] = []
    for c in content.get("key_claims") or []:
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, dict):
            claim = c.get("claim")
            if isinstance(claim, str):
                parts.append(claim)
    for sec in content.get("section_skeleton") or []:
        if isinstance(sec, str):
            parts.append(sec)
        elif isinstance(sec, dict):
            for key in ("heading", "title", "summary"):
                val = sec.get(key)
                if isinstance(val, str):
                    parts.append(val)
    return "\n".join(p for p in parts if p.strip())


def _objective_id(obj: Dict[str, Any]) -> str:
    return str(obj.get("id") or obj.get("objective_id") or "")


def _objective_statement(obj: Dict[str, Any]) -> str:
    return str(obj.get("statement") or obj.get("text") or "").strip()


def align_blocks_to_objectives(
    blocks: List[Any],
    objectives: List[Dict[str, Any]],
    *,
    embed: Optional[Any] = None,
    threshold: float = _DEFAULT_ALIGN_THRESHOLD,
    capture: Optional[Any] = None,
) -> List[Any]:
    """Populate each block's ``content['objective_refs']`` via embedding alignment.

    Returns the (possibly-replaced) block list. Blocks whose ``content`` is not a
    dict (rewrite-tier HTML, chrome) are passed through untouched. On embeddings
    absent the whole pass is a no-op pass-through.
    """
    import dataclasses as _dc

    if not blocks or not objectives:
        return blocks

    # Build the objective id/statement index.
    obj_ids: List[str] = []
    obj_statements: List[str] = []
    for obj in objectives:
        oid = _objective_id(obj)
        if not oid:
            continue
        obj_ids.append(oid)
        obj_statements.append(_objective_statement(obj) or " ")
    if not obj_ids:
        return blocks

    # Collect dict-content blocks + their text surfaces.
    indexed: List[Tuple[int, Dict[str, Any], str]] = []
    for idx, block in enumerate(blocks):
        content = _block_content(block)
        if content is None:
            continue
        indexed.append((idx, content, _block_text(content)))
    if not indexed:
        return blocks

    # Try to embed. On any failure, pass-through (model refs stand).
    obj_vecs = None
    block_vecs = None
    if embed is not None:
        try:
            obj_vecs = embed.encode_batch(obj_statements)
            block_texts = [t or " " for (_i, _c, t) in indexed]
            block_vecs = embed.encode_batch(block_texts)
        except Exception:  # noqa: BLE001 — degrade to pass-through
            obj_vecs = None
            block_vecs = None
    if obj_vecs is None or block_vecs is None:
        # Pass-through: outline-tier refs stand; no fabricated alignment.
        return blocks

    out_blocks = list(blocks)
    for row, (block_idx, content, _text) in enumerate(indexed):
        block = out_blocks[block_idx]
        existing = _existing_objective_refs(block, content)

        # Score every objective; embedding-add those ≥ threshold.
        scored: List[Tuple[str, float]] = []
        for o_idx, oid in enumerate(obj_ids):
            cos = cosine_similarity(block_vecs[row], obj_vecs[o_idx])
            scored.append((oid, cos))

        added = [oid for (oid, cos) in scored if cos >= threshold]
        # Union existing (authoritative) with embedding-matched; cap top-K by
        # cosine. Existing refs are always kept (never capped out).
        cosine_by_id = {oid: cos for (oid, cos) in scored}
        union_ids = list(existing)
        for oid in added:
            if oid not in union_ids:
                union_ids.append(oid)
        # Cap: keep all existing + the highest-cosine added up to _TOP_K total.
        if len(union_ids) > _TOP_K:
            extras = [oid for oid in union_ids if oid not in existing]
            extras.sort(key=lambda x: cosine_by_id.get(x, 0.0), reverse=True)
            keep_extra = extras[: max(0, _TOP_K - len(existing))]
            union_ids = list(existing) + [
                oid for oid in union_ids
                if oid in keep_extra and oid not in existing
            ]

        new_content = dict(content)
        added_count = len([o for o in union_ids if o not in existing])
        if union_ids:
            new_content["objective_refs"] = union_ids
        else:
            # No alignment at all — leave empty refs + stamp the warning. NEVER
            # fabricate a ref to dodge the gate.
            new_content["objective_refs"] = []
            warnings = list(new_content.get("structural_warnings") or [])
            if _NO_ALIGNMENT_MARKER not in warnings:
                warnings.append(_NO_ALIGNMENT_MARKER)
            new_content["structural_warnings"] = warnings

        out_blocks[block_idx] = _dc.replace(block, content=new_content)

        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="block_objective_alignment",
                    decision=(
                        f"block_objective_alignment:"
                        f"{getattr(block, 'block_id', '')}:"
                        f"{'aligned' if union_ids else 'unaligned'}"
                    ),
                    rationale=(
                        f"block_id={getattr(block, 'block_id', '')}; "
                        f"block_type={getattr(block, 'block_type', '')}; "
                        f"existing_refs={len(existing)}; "
                        f"added_refs={added_count}; "
                        f"final_refs={len(union_ids)}; "
                        f"threshold={threshold}; "
                        f"objective_pool={len(obj_ids)}; "
                        f"unaligned={1 if not union_ids else 0}"
                    ),
                    ml_features={
                        "block_id": getattr(block, "block_id", ""),
                        "existing_refs": len(existing),
                        "added_refs": added_count,
                        "final_refs": len(union_ids),
                        "unaligned": 0 if union_ids else 1,
                    },
                )
            except Exception:  # noqa: BLE001
                pass

    return out_blocks


__all__ = [
    "align_blocks_to_objectives",
    "_DEFAULT_ALIGN_THRESHOLD",
    "_NO_ALIGNMENT_MARKER",
]
