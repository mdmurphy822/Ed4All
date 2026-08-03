"""Pure, deterministic per-page ARRANGE contract (v2) for the scan-lane path.

This module is the unit-testable HALF of the page-arranger scan-lane structure
path (``SEMANTIK_PAGE_ARRANGER``, task #43). It holds ONLY pure functions — the
type enum, the alias-coercion table, the arrange system prompt, the validation +
deterministic-repair passes, the retry-ladder violation-message builder, the
tolerant JSON parse, and the schema_version-2 training-record relation shapes.

It reads NO environment, imports NO ``requests`` and touches NO filesystem — so
the whole ARRANGE contract (the part that decides whether a model's arrangement
is acceptable) is testable without a live seat or a PDF. The I/O half (seat
resolution, whole-doc gather, the HTTP call, the fan-out, region assembly, the
label-factory writes) lives in :mod:`semantik_structure.page_arranger`.

Ported VERBATIM (behavior-preserving) from the validated scratchpad prototype
``page_arranger_proto.py`` (contract v2: furniture bucket, alias coercion incl.
four observed failure-taxonomy extensions, 3-rung retry ladder, duplicate
auto-repair, and schema-version-2 training records.

Contract summary (per page):
  1. the model receives a page IMAGE + an id'd list of extracted text UNITS and
     returns ONLY JSON assigning EVERY unit id to a reading-ordered, typed block
     (it emits ids, never text — the verbatim-text contract holds);
  2. deterministic pre-validation repairs run BEFORE validation: duplicate-id
     auto-repair (keep-first) THEN alias-coercion (type-only synonyms → the
     enum, list family context-sensitive);
  3. validation demands every id covered exactly once + a legal type;
  4. a 3-rung retry ladder (violations named, then the full legal-id set at
     temp 0.3) recovers most misfires; an unrecoverable page is
     ``arrangement_failed`` (loud, never a dropped page — the I/O half emits a
     per-unit paragraph fallback).
"""
from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "CONTRACT_VERSION",
    "TYPE_ENUM",
    "LIST_FAMILY",
    "TYPE_ALIASES",
    "ARRANGE_SYSTEM",
    "validate_arrangement",
    "normalize_block_types",
    "repair_duplicate_ids",
    "repair_mechanical_ids",
    "build_violation_msg",
    "extract_json",
    "RELATIONS_SCHEMA",
    "derive_relations_v2",
    "page_dims_from_bboxes",
]

# CONTRACT_VERSION rides the per-unit resume-cache fingerprint + the train
# records (schema_version). BUMP whenever ARRANGE_SYSTEM, the enum, the alias
# table, or a validation/repair rule changes in a way that would change the
# accepted arrangement — so a stale sidecar written under the old contract is
# never served for the new one.
CONTRACT_VERSION = 2

# The 9 legal block types (the furniture bucket is critical — it is the label
# supply that keeps a furniture unit LISTED even though it is dropped at render).
TYPE_ENUM = frozenset(
    {
        "heading",
        "paragraph",
        "table",
        "figure_caption",
        "example",
        "solution",
        "exercise_list",
        "definition_box",
        "furniture",
    }
)

# The CONTEXT-SENSITIVE list family: a member text that looks like exercises
# (numbered / lettered / bulleted items) → ``exercise_list``, else ``paragraph``.
LIST_FAMILY = frozenset(
    {
        "list-item",
        "list_item",
        "list",
        "bullet_list",
        "ordered_list",
        "unordered_list",
        "numbered_list",
        "ol",
        "ul",
        "items",
    }
)

# Deterministic ALIAS-COERCION (pre-validation): lowercase/strip the emitted
# type, map obvious synonyms onto the enum. TYPE-only — never touches unit text.
TYPE_ALIASES = {
    "text": "paragraph",
    "prose": "paragraph",
    "body": "paragraph",
    "body_text": "paragraph",
    "caption": "figure_caption",
    "figure": "figure_caption",
    "title": "heading",
    "subheading": "heading",
    "subtitle": "heading",
    "definition": "definition_box",
    "def": "definition_box",
    "box": "definition_box",
    "callout": "definition_box",
    "note": "definition_box",
    "worked_example": "example",
    "problem": "example",
    "answer": "exercise_list",
    "answers": "exercise_list",
    "header": "furniture",
    "footer": "furniture",
    "page_number": "furniture",
    "running_head": "furniture",
    "running_header": "furniture",
    "watermark": "furniture",
    # Failure-taxonomy extensions: out-of-enum types observed at document scale.
    "section": "heading",
    "section-header": "heading",
    "section_header": "heading",
    "exercise": "exercise_list",
    "table_caption": "figure_caption",
}

ARRANGE_SYSTEM = (
    "You are a page-layout arranger. Input: a page image and text units already "
    "extracted from it, each with an id. Output ONLY one JSON object, no prose:\n"
    '{"blocks":[{"ids":["<unit-id>",...],"type":"heading|paragraph|table|'
    'figure_caption|example|solution|exercise_list|definition_box|furniture",'
    '"level":<int|null>,"continues_prev_page":<bool>}],"confidence":<0..1>}\n'
    "Rules: every input id appears exactly once across all blocks. Merge units "
    "that form one logical block by listing several ids in one block. Block order "
    "= correct reading order. NEVER rewrite text — use ids only. level = heading "
    "level for headings else null. continues_prev_page only on the first block "
    "(true if it continues text from the previous page).\n"
    "FURNITURE BUCKET (critical): NEVER omit a unit id. Page furniture — running "
    "headers, running footers, page numbers, watermarks — is NOT discarded here: "
    "assign every such unit to a block with type \"furniture\" (it is dropped at "
    "render time but MUST still be listed so no id goes missing). If you are "
    "unsure where a unit belongs, still place it in SOME block rather than "
    "dropping it. Use ONLY the exact ids given to you — never invent an id."
)

# Render/list-shape regexes (also used by the I/O half's region builder).
_PIPE_ROW = re.compile(r"^\s*\|?.*\|.*\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\(?\d{1,3}[.)]|[a-z]\))\s+")


# ---------------------------------------------------------------------------
# Tolerant JSON parse.
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """Fence-strip + outermost-brace tolerant JSON parse.

    Raises (``json.JSONDecodeError`` / ``ValueError``) on an unparseable body —
    the caller's ladder catches it and re-asks. Mirrors the prototype's
    ``_extract_json``.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def validate_arrangement(arr: dict, units: list[dict]) -> list[str]:
    """Return a list of violation strings (empty = valid).

    Checks: a non-empty ``blocks`` array, every block an object with a legal
    ``type`` and a non-empty ``ids`` list, every id known, no id covered twice,
    no id missing. The coverage invariant (every unit id exactly once) is the
    load-bearing one — it is what lets the I/O half assert no content is lost.
    """
    problems: list[str] = []
    blocks = arr.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return ["no 'blocks' array in output"]
    unit_ids = {u["id"] for u in units}
    seen: dict[str, int] = {}
    for bi, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            problems.append(f"block {bi} is not an object")
            continue
        btype = blk.get("type")
        if btype not in TYPE_ENUM:
            problems.append(f"block {bi} has invalid type {btype!r}")
        ids = blk.get("ids")
        if not isinstance(ids, list) or not ids:
            problems.append(f"block {bi} has no ids")
            continue
        for uid in ids:
            if uid not in unit_ids:
                problems.append(f"block {bi} references unknown id {uid!r}")
            seen[uid] = seen.get(uid, 0) + 1
    dupes = [u for u, c in seen.items() if c > 1]
    if dupes:
        problems.append(f"ids appearing more than once: {sorted(dupes)}")
    missing = sorted(unit_ids - set(seen))
    if missing:
        problems.append(f"ids missing from output: {missing}")
    return problems


# ---------------------------------------------------------------------------
# Deterministic pre-validation repairs.
# ---------------------------------------------------------------------------
def _block_looks_like_exercises(blk: dict, unit_by_id: dict) -> bool:
    """True when the block's unit texts look like exercise/list items
    (numbered / lettered / bulleted lines)."""
    hits = total = 0
    for uid in blk.get("ids", []):
        u = unit_by_id.get(uid)
        if not u:
            continue
        for line in (u.get("text") or "").splitlines():
            line = line.strip()
            if not line:
                continue
            total += 1
            if _LIST_ITEM.match(line):
                hits += 1
    return total > 0 and (hits / total) >= 0.4


def normalize_block_types(arr: dict, unit_by_id: dict) -> list[dict]:
    """Deterministic pre-validation ALIAS-COERCION (TYPE only, never text).

    Lowercase/strip the emitted type; map synonyms onto the enum. The list
    family is context-sensitive (exercise-looking text → ``exercise_list``, else
    ``paragraph``). Returns the coercion log: ``[{from, to, block_idx}]``.
    """
    coercions: list[dict] = []
    for bi, blk in enumerate((arr or {}).get("blocks") or []):
        if not isinstance(blk, dict):
            continue
        t = blk.get("type")
        if not isinstance(t, str):
            continue
        norm = t.strip().lower()
        if norm in TYPE_ENUM:
            if norm != t:
                blk["type"] = norm
                coercions.append({"from": t, "to": norm, "block_idx": bi})
            continue
        if norm in LIST_FAMILY:
            canon = (
                "exercise_list"
                if _block_looks_like_exercises(blk, unit_by_id)
                else "paragraph"
            )
        else:
            canon = TYPE_ALIASES.get(norm)
        if canon:
            blk["type"] = canon
            coercions.append({"from": t, "to": canon, "block_idx": bi})
    return coercions


def repair_duplicate_ids(arr: dict) -> list[dict]:
    """Deterministic DUPLICATE-ID auto-repair (CONTRACT V2 item 3).

    When the same unit id appears in more than one block, KEEP the first
    occurrence (reading order) and DROP every later one — it is unambiguous, so
    we repair instead of failing the page. A block whose ids all get dropped is
    removed. Returns the repair log:
    ``[{id, kept_block_idx, dropped_from_block_idx}]``.
    """
    repairs: list[dict] = []
    blocks = (arr or {}).get("blocks")
    if not isinstance(blocks, list):
        return repairs
    seen_first: dict[str, int] = {}
    for bi, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            continue
        ids = blk.get("ids")
        if not isinstance(ids, list):
            continue
        kept: list = []
        for uid in ids:
            if uid in seen_first:
                repairs.append(
                    {
                        "id": uid,
                        "kept_block_idx": seen_first[uid],
                        "dropped_from_block_idx": bi,
                    }
                )
                continue
            seen_first[uid] = bi
            kept.append(uid)
        blk["ids"] = kept
    # Drop any block left with no ids by the repair (avoids a spurious
    # "block has no ids" violation for a block that was pure duplicates).
    arr["blocks"] = [
        b for b in blocks if not (isinstance(b, dict) and b.get("ids") == [])
    ]
    return repairs


def repair_mechanical_ids(arr: dict, units: list[dict]) -> list[dict]:
    """Deterministic repair of the three MECHANICAL id-assignment failure classes
    on a returned arrangement (CONTRACT V2 item — ``SEMANTIK_ARRANGER_ID_REPAIR``).

    The arrange model's block set can fail the coverage invariant in exactly
    three id-bookkeeping ways; each is repairable WITHOUT re-asking the model and
    WITHOUT ever altering a correctly-assigned id (repairs are additive/subtractive
    bookkeeping only, never a structural re-typing of the model's real choices):

    * **(a) duplicated ids** → :func:`repair_duplicate_ids` (keep the first
      reading-order occurrence, drop the rest).
    * **(c) unknown / hallucinated ids** → DROPPED (they reference no unit, so they
      are pure noise — never content).
    * **(b) missing ids** → APPENDED, each as its OWN ``paragraph`` block (the
      honest content-preserving default the whole-page failure fallback uses —
      NOT ``furniture``, which is dropped at render), inserted ADJACENT to its
      source-order neighbor: right AFTER the block holding its nearest assigned
      predecessor, else right BEFORE its nearest assigned successor, else at the
      tail. Missing ids are processed in source order so a run of consecutive
      missing ids chains adjacently.

    Mutates ``arr['blocks']`` in place. Returns the repair log — a flat list of
    ``{op, id, ...}`` records (``op`` ∈ ``dup_drop`` / ``unknown_drop`` /
    ``missing_insert``) — so the caller can re-validate and decision-capture the
    repaired id lists. Text is NEVER touched (ids only).
    """
    log: list[dict] = []
    blocks = (arr or {}).get("blocks")
    if not isinstance(blocks, list):
        return log
    unit_ids = {u["id"] for u in units}
    unit_order = {u["id"]: i for i, u in enumerate(units)}

    # (a) duplicated ids — reuse the keep-first repair, tag its records.
    for rec in repair_duplicate_ids(arr):
        log.append({"op": "dup_drop", **rec})
    blocks = arr["blocks"]  # repair_duplicate_ids may have rebuilt the list

    # (c) unknown / hallucinated ids — drop ids referencing no unit.
    for bi, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            continue
        ids = blk.get("ids")
        if not isinstance(ids, list):
            continue
        kept: list = []
        for uid in ids:
            if uid in unit_ids:
                kept.append(uid)
            else:
                log.append({"op": "unknown_drop", "id": uid, "block_idx": bi})
        blk["ids"] = kept
    arr["blocks"] = [
        b for b in blocks if not (isinstance(b, dict) and b.get("ids") == [])
    ]
    blocks = arr["blocks"]

    # (b) missing ids — insert adjacent to the nearest assigned source-order
    # neighbor, one at a time (in source order) so consecutive misses chain.
    def _assigned_ids() -> set:
        return {
            uid
            for blk in blocks
            if isinstance(blk, dict)
            for uid in (blk.get("ids") or [])
        }

    def _block_index_of(uid) -> int | None:
        for bi, blk in enumerate(blocks):
            if isinstance(blk, dict) and uid in (blk.get("ids") or []):
                return bi
        return None

    seen = _assigned_ids()
    missing = sorted(unit_ids - seen, key=lambda u: unit_order.get(u, len(units)))
    for uid in missing:
        assigned = _assigned_ids()
        p = unit_order.get(uid, len(units))
        pred = max(
            (u for u in assigned if unit_order.get(u, -1) < p),
            key=lambda u: unit_order.get(u, -1),
            default=None,
        )
        new_block = {"type": "paragraph", "ids": [uid]}
        if pred is not None:
            at = (_block_index_of(pred) or 0) + 1
        else:
            succ = min(
                (u for u in assigned if unit_order.get(u, len(units)) > p),
                key=lambda u: unit_order.get(u, len(units)),
                default=None,
            )
            at = _block_index_of(succ) if succ is not None else len(blocks)
            if at is None:
                at = len(blocks)
        blocks.insert(at, new_block)
        log.append({"op": "missing_insert", "id": uid, "at_block_idx": at, "type": "paragraph"})
    return log


def build_violation_msg(problems: list[str], legal_ids: list[str] | None) -> str:
    """Retry directive. Always names the violations + the furniture-bucket rule;
    the third rung (``legal_ids`` set) also restates the COMPLETE legal id set."""
    msg = (
        "Your previous output was INVALID: "
        + "; ".join(problems[:8])
        + ". Re-emit ONLY one JSON object. The ONLY allowed type values are: "
        + ", ".join(sorted(TYPE_ENUM))
        + '. Every unit id must appear exactly once; put discardable furniture '
        '(running headers, page numbers, footers, watermarks) into a "furniture" '
        "block — NEVER omit a unit. Use ONLY the exact ids given; do not invent "
        "ids. Do not rewrite any text."
    )
    if any("valid JSON" in p or "JSON" in p for p in problems):
        msg += (
            " Emit strictly valid JSON: no trailing text, escape any quotes "
            "inside string values, no raw newlines inside strings."
        )
    if legal_ids:
        msg += (
            "\nThe COMPLETE set of legal unit ids (use ONLY these, each "
            "exactly once) is:\n" + ", ".join(legal_ids)
        )
    return msg


# ---------------------------------------------------------------------------
# schema_version-2 training-record relation shapes.
# ---------------------------------------------------------------------------
RELATIONS_SCHEMA = {
    "schema_version": 2,
    "description": (
        "Relation-type enum + exact field shapes for arranger train records "
        "(schema_version 2). Deterministically derived from a VALIDATED "
        "arrangement; no model output beyond the arrangement itself."
    ),
    "relation_types": {
        "same_unit": {
            "fields": {"block_idx": "int", "unit_ids": "list[str]"},
            "meaning": "the listed units form ONE logical block (merge signal)",
        },
        "caption_of": {
            "fields": {"caption_block_idx": "int", "target_block_idx": "int"},
            "meaning": (
                "figure_caption block -> nearest non-furniture "
                "neighbor block (prev preferred). Candidate edge."
            ),
        },
        "solution_of": {
            "fields": {"solution_block_idx": "int", "target_block_idx": "int"},
            "meaning": "solution block immediately following an example block",
        },
        "continues": {
            "fields": {"from": "str(p<N-1>_bNN)|null", "to": "str(p<N>_bNN)"},
            "meaning": (
                "cross-page text continuation. to = first unit of the "
                "first block of page N (whose continues_prev_page was "
                "true); from = LAST unit of the last NON-FURNITURE "
                "block of page N-1's record, null when page N-1 has "
                "no valid record in this set."
            ),
        },
    },
}


def page_dims_from_bboxes(units: list[dict]) -> tuple:
    """Infer page (width, height, dims_source) from the max bbox extents."""
    xs, ys = [], []
    for u in units:
        bb = u.get("bbox")
        if bb and len(bb) >= 4:
            try:
                xs.append(float(bb[2]))
                ys.append(float(bb[3]))
            except (TypeError, ValueError):
                pass
    if not xs or not ys:
        return None, None, "unavailable"
    return max(xs), max(ys), "bbox_inferred"


def derive_relations_v2(blocks: list[dict]) -> list[dict]:
    """schema-v2 within-page relation shapes; the cross-page ``continues``
    edge is materialized separately (the I/O half's ``finalize_continues_over_dir``)."""
    rels: list[dict] = []
    for bi, blk in enumerate(blocks):
        ids = blk.get("ids") or []
        if len(ids) > 1:
            rels.append({"type": "same_unit", "block_idx": bi, "unit_ids": ids})
    for bi in range(1, len(blocks)):
        if (
            blocks[bi - 1].get("type") == "example"
            and blocks[bi].get("type") == "solution"
        ):
            rels.append(
                {
                    "type": "solution_of",
                    "solution_block_idx": bi,
                    "target_block_idx": bi - 1,
                }
            )
    for bi, blk in enumerate(blocks):
        if blk.get("type") != "figure_caption":
            continue
        neighbor = None
        for j in (bi - 1, bi + 1):
            if 0 <= j < len(blocks) and blocks[j].get("type") != "furniture":
                neighbor = j
                break
        if neighbor is not None:
            rels.append(
                {
                    "type": "caption_of",
                    "caption_block_idx": bi,
                    "target_block_idx": neighbor,
                }
            )
    return rels


def last_content_unit_id(rec_blocks: list[dict], unit_ids_present: set) -> str | None:
    """Last unit id of the last NON-FURNITURE block (fallback: last block)."""
    for blk in reversed(rec_blocks):
        if blk.get("type") == "furniture":
            continue
        ids = [i for i in (blk.get("ids") or []) if i in unit_ids_present]
        if ids:
            return ids[-1]
    for blk in reversed(rec_blocks):
        ids = [i for i in (blk.get("ids") or []) if i in unit_ids_present]
        if ids:
            return ids[-1]
    return None
