"""Defect-A — chapter-anchored TO derivation grouping core (PURE).

Terminal objectives are definitionally chapter-grained: a real textbook chapter
is exactly one terminal competency. The legacy bottom-up derivation clustered CO
*statement vectors*, which scattered a chapter's COs across theme clusters and
produced week-pair prerequisite inversions (Defect A: 38% of week pairs in the
sample-scan review). This module recovers the chapter signal that already exists in
the DART chunk metadata — every chunk carries a ``source.module_id`` that is 1:1
with a staged HTML file (one chapter) — and assigns each CO to the module its
cited chunks predominantly come from. One module → one terminal objective.

The core is deliberately PURE: no pipeline imports, no env reads, no embedding
model, no LLM. It resolves everything it can DETERMINISTICALLY (plurality vote +
two tie-breaks) and reports the residue (zero-resolve COs, defaulted to the last
module so nothing is ever dropped) so the pipeline wrapper can (optionally)
re-home the residue via statement cosine when an embedding client is available.

Ground truth (see ``plans/objective-synthesis-fixes-2026-07.md`` §A):

* ``chunk["source"]["module_id"]`` is generic + chunker-materialized (one staged
  HTML = one module_id); ``module_title`` rides beside it.
* Book order of modules = first-occurrence order in ``all_chunks`` (the loader
  preserves file order).

Public surface: :func:`assign_cos_to_modules` → :class:`AnchorResult`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["AnchorResult", "assign_cos_to_modules"]


@dataclass
class AnchorResult:
    """Deterministic CO→module assignment + book-ordered module grouping.

    Every CO index in ``cos`` appears in exactly one group (partition invariant):
    a zero-resolve CO is defaulted to the LAST module (recorded in
    ``unresolved_co_indices``) so a wrapper can override it via cosine without
    ever losing a CO.
    """

    #: Module ids in book (first-occurrence) order.
    module_order: List[str] = field(default_factory=list)
    #: module_id → human title (``source.module_title``; falls back to the id).
    module_title_by_id: Dict[str, str] = field(default_factory=dict)
    #: CO list-index → assigned module_id (EVERY CO; never missing).
    assignment: Dict[int, str] = field(default_factory=dict)
    #: One (module_id, [co_index, ...]) pair per module that received ≥1 CO,
    #: ordered by ``module_order``. Member index lists preserve CO input order.
    groups: List[Tuple[str, List[int]]] = field(default_factory=list)
    #: min ``source.position_in_module`` over a CO's cited chunks (for the
    #: reorder tie-break); large sentinel when no cited chunk resolves a position.
    min_position_by_co: Dict[int, int] = field(default_factory=dict)
    #: Count of COs whose module plurality was a tie (broken deterministically).
    ties: int = 0
    #: Count of COs whose cited chunks spanned >1 module.
    multi_module_cos: int = 0
    #: CO indices that resolved ZERO module from their cited chunks (defaulted
    #: to the last module; the wrapper may re-home these by cosine).
    unresolved_co_indices: List[int] = field(default_factory=list)
    #: Fraction of COs that resolved ≥1 module from their own cited chunks.
    co_coverage: float = 0.0

    @property
    def n_modules(self) -> int:
        return len(self.module_order)


_POSITION_SENTINEL = 1_000_000_000


def _chunk_module_id(chunk: Any) -> str:
    """Read ``source.module_id`` off a v4 chunk record (empty when absent)."""
    if not isinstance(chunk, dict):
        return ""
    src = chunk.get("source")
    if isinstance(src, dict):
        return str(src.get("module_id") or "").strip()
    # Tolerate a flattened test/legacy shape (module_id at top level).
    return str(chunk.get("module_id") or "").strip()


def _chunk_module_title(chunk: Any) -> str:
    if not isinstance(chunk, dict):
        return ""
    src = chunk.get("source")
    if isinstance(src, dict):
        return str(src.get("module_title") or "").strip()
    return str(chunk.get("module_title") or "").strip()


def _chunk_position(chunk: Any) -> Optional[int]:
    if not isinstance(chunk, dict):
        return None
    src = chunk.get("source") if isinstance(chunk.get("source"), dict) else chunk
    raw = src.get("position_in_module")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _co_source_chunk_ids(co: Any) -> List[str]:
    if not isinstance(co, dict):
        return []
    raw = co.get("source_chunk_ids")
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def assign_cos_to_modules(
    cos: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
) -> AnchorResult:
    """Assign every CO to one book-ordered module via cited-chunk plurality.

    Algorithm (deterministic; §A steps 1 + 3 + 4):

    1. Build the module map from ``all_chunks``: ``module_order`` = first
       occurrence of each non-empty ``source.module_id``; ``module_title_by_id``
       from ``source.module_title`` (first non-empty wins).
    2. Per CO, plurality-vote ``module_id`` over its ``source_chunk_ids``
       (resolved through ``chunks_by_id``). Tie-break, in order:
         (i)  the module of the CO's ``entailing_chunk_id`` if it is among the
              tied modules (the NLI-anchored chunk is the strongest single vote);
         (ii) the lowest ``module_order`` index among the tied modules.
    3. A zero-resolve CO (no cited chunk resolves a module) is defaulted to the
       LAST module and recorded in ``unresolved_co_indices`` (never dropped).
    4. Build one group per module that received ≥1 CO, ordered by
       ``module_order``.

    Pure: no env, no embeddings, no LLM. Returns an :class:`AnchorResult`.
    """
    result = AnchorResult()
    if not cos:
        return result

    # --- Step 1: module map from all_chunks (book order). -------------------
    order_index: Dict[str, int] = {}
    for chunk in all_chunks or []:
        mid = _chunk_module_id(chunk)
        if not mid:
            continue
        if mid not in order_index:
            order_index[mid] = len(result.module_order)
            result.module_order.append(mid)
            result.module_title_by_id[mid] = _chunk_module_title(chunk) or mid
        elif not result.module_title_by_id.get(mid):
            title = _chunk_module_title(chunk)
            if title:
                result.module_title_by_id[mid] = title

    last_module = result.module_order[-1] if result.module_order else ""

    # --- Steps 2-3: per-CO plurality + tie-break + zero-resolve default. -----
    resolved_count = 0
    for idx, co in enumerate(cos):
        chunk_ids = _co_source_chunk_ids(co)
        votes: Counter = Counter()
        min_pos = _POSITION_SENTINEL
        for cid in chunk_ids:
            chunk = chunks_by_id.get(cid)
            mid = _chunk_module_id(chunk)
            if not mid or mid not in order_index:
                continue
            votes[mid] += 1
            pos = _chunk_position(chunk)
            if pos is not None and pos < min_pos:
                min_pos = pos
        result.min_position_by_co[idx] = min_pos

        if not votes:
            # Zero-resolve → default to last module (wrapper may override).
            result.assignment[idx] = last_module
            result.unresolved_co_indices.append(idx)
            continue

        resolved_count += 1
        if len(votes) > 1:
            result.multi_module_cos += 1
        top_n = votes.most_common()
        best_count = top_n[0][1]
        tied = [m for m, n in top_n if n == best_count]
        if len(tied) > 1:
            result.ties += 1
            chosen = _break_tie(co, tied, chunks_by_id, order_index)
        else:
            chosen = tied[0]
        result.assignment[idx] = chosen

    result.co_coverage = resolved_count / len(cos)
    result.groups = _build_groups(result.assignment, result.module_order)
    return result


def _break_tie(
    co: Dict[str, Any],
    tied: List[str],
    chunks_by_id: Dict[str, Any],
    order_index: Dict[str, int],
) -> str:
    """Deterministic plurality tie-break: entailing-chunk module, else lowest order."""
    # (i) module of the CO's entailing (NLI-anchored) chunk if it is tied.
    ent = str(co.get("entailing_chunk_id") or "").strip()
    if ent:
        ent_mid = _chunk_module_id(chunks_by_id.get(ent))
        if ent_mid in tied:
            return ent_mid
    # (ii) lowest module_order index among the tied modules.
    return min(tied, key=lambda m: order_index.get(m, _POSITION_SENTINEL))


def _build_groups(
    assignment: Dict[int, str], module_order: List[str]
) -> List[Tuple[str, List[int]]]:
    """One (module_id, [co_index...]) per module with ≥1 CO, in book order."""
    members: Dict[str, List[int]] = {}
    for idx in sorted(assignment):
        members.setdefault(assignment[idx], []).append(idx)
    groups: List[Tuple[str, List[int]]] = []
    seen = set()
    for mid in module_order:
        if mid in members:
            groups.append((mid, members[mid]))
            seen.add(mid)
    # A module id that isn't in module_order (only possible via a wrapper
    # override to an unknown id) still gets a stable, trailing group.
    for mid in members:
        if mid not in seen:
            groups.append((mid, members[mid]))
    return groups
