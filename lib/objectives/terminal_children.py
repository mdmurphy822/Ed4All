"""Terminal-objective child back-annotation (one-owner lift).

``_annotate_terminals_with_children`` + its ``_co_parent_terminal_id`` helper
were historically defined inline in ``MCP/tools/pipeline_tools.py``. Defect F
(``ed4all objectives restructure``, ``lib/objectives/restructure.py``) needs the
SAME back-annotation deterministically, offline, with no pipeline_tools import —
so the pair is lifted here (byte-identical behaviour) and ``pipeline_tools``
re-exports both names (``from lib.objectives.terminal_children import ...``). This
keeps ONE owner for the TO→CO→chunk join logic; both the live course_planning
path and the offline restructure CLI compose the same function.

Pure: no env reads, no embeddings, no LLM. Anti-fabrication is preserved — the
TO ``source_refs`` chunk ids are a strict union of ids the child COs already
cite; a childless TO keeps its original (empty) ``source_refs``.
"""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["_annotate_terminals_with_children", "_co_parent_terminal_id"]


def _co_parent_terminal_id(co: Dict[str, Any]) -> str:
    """Resolve a CO's parent-terminal id under any back-pointer key.

    Mirrors ``lib/ontology/terminal_coverage._PARENT_TERMINAL_KEYS`` (the same
    keys ``backlink_cos_to_tos`` / the objective-review remap write) so this
    reads whatever back-pointer the CO actually carries. Returns the raw id
    string (casing preserved) or ``""`` when none resolves.
    """
    try:
        from lib.ontology.terminal_coverage import (  # noqa: PLC0415
            _PARENT_TERMINAL_KEYS,
        )
    except Exception:  # noqa: BLE001 — keep serialization resilient
        _PARENT_TERMINAL_KEYS = (
            "parent_to",
            "parent_terminal",
            "parent_terminal_id",
            "terminal_id",
        )
    for key in _PARENT_TERMINAL_KEYS:
        val = co.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _annotate_terminals_with_children(
    terminals: List[Dict[str, Any]],
    chapter_objectives: List[Dict[str, Any]],
) -> Dict[str, int]:
    """P2b — back-annotate each TO with its child COs + their chunk provenance.

    The CO→TO backlink (``backlink_cos_to_tos`` / bottom-up clustering) sets a
    ``terminal_id`` on every CO, but the persisted ``terminal_objectives[]``
    carried EMPTY ``source_refs`` and no child pointer — so there was no
    machine-readable TO→CO→chunk edge and an objective-coverage audit had to
    reconstruct the join by statement matching (the observability gap that made
    a covered TO look orphaned). This pass populates, per TO, in place:

      * ``child_co_ids`` — the ids of every CO whose parent-terminal resolves to
        this TO (course order, deduped).
      * ``source_refs`` — the UNION of those child COs' ``source_chunk_ids``
        (deduped, course order), shaped ``[{"ref": <to_id>, "chunk_ids": [...]}]``
        to match the existing CO ``source_refs`` shape. ANTI-FABRICATION: the
        chunk ids are a strict union of ids the child COs already cite — no
        chunk is invented, and a TO with no resolvable children keeps its
        original (empty) ``source_refs``.

    Matching is case-insensitive on the TO id (a backlink / remap may
    lower-case the written pointer). Returns a small counters dict for logging /
    decision-capture (``tos_annotated`` / ``cos_mapped`` / ``orphan_tos``).
    """
    by_lower: Dict[str, Dict[str, Any]] = {}
    for to in terminals:
        if not isinstance(to, dict):
            continue
        tid = to.get("id")
        if isinstance(tid, str) and tid.strip():
            by_lower[tid.strip().lower()] = to

    child_ids: Dict[str, List[str]] = {k: [] for k in by_lower}
    child_chunks: Dict[str, List[str]] = {k: [] for k in by_lower}
    cos_mapped = 0
    for co in chapter_objectives:
        if not isinstance(co, dict):
            continue
        parent = _co_parent_terminal_id(co).lower()
        if not parent or parent not in by_lower:
            continue
        cos_mapped += 1
        cid = co.get("id")
        if isinstance(cid, str) and cid.strip() and cid not in child_ids[parent]:
            child_ids[parent].append(cid)
        raw_chunks = co.get("source_chunk_ids")
        if isinstance(raw_chunks, list):
            for ch in raw_chunks:
                ch_s = str(ch)
                if ch_s and ch_s not in child_chunks[parent]:
                    child_chunks[parent].append(ch_s)

    tos_annotated = 0
    orphan_tos = 0
    for key, to in by_lower.items():
        cids = child_ids.get(key) or []
        chunks = child_chunks.get(key) or []
        to["child_co_ids"] = list(cids)
        if chunks:
            tos_annotated += 1
            # FIX A — do NOT mint ``ref=<this TO's own id>``. A self-referential
            # ref (e.g. TO-01 citing ref="TO-01") can never resolve against the
            # chapter/section source universe the ``objective_source_refs`` gate
            # audits, so it fired OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE on
            # every terminal objective deterministically. The TO IS validly
            # grounded — its ``chunk_ids`` (the union of its child COs' cited
            # chunks) all resolve against the DART chunkset — so we emit a
            # structured ``{chunk_ids}`` entry with NO ``ref`` key. The
            # validator's structured-shape arm only resolves a ``ref`` when it
            # is a non-empty string, so omitting it skips the impossible
            # textbook-universe check while the chunk_ids grounding still
            # passes. Every downstream consumer of TO ``source_refs`` reads only
            # ``ref["chunk_ids"]`` (never ``ref["ref"]``), so this is
            # behaviour-preserving for the pacing / chunk-index builders.
            # Anti-fabrication: no chapter id is invented — the chunk_ids are
            # exactly the child COs' real cited chunks.
            to["source_refs"] = [{"chunk_ids": list(chunks)}]
        else:
            orphan_tos += 1
    return {
        "tos_annotated": tos_annotated,
        "cos_mapped": cos_mapped,
        "orphan_tos": orphan_tos,
    }
