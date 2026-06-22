"""Stage 4: Hierarchy resolution. Algorithmic. No model.

Contract:
    resolve_hierarchy(classified: list[ClassifiedBlock]) -> list[ResolvedBlock]

Given classified blocks in document order, compute:

  HEADING depth (1-6) via font-size stack
    - Walk headings in order, tracking a stack of (font_size, depth).
    - For each new heading:
        font == top  -> same depth as top
        font >  top  -> pop until top >= font; depth = new top's depth
        font <  top  -> push; depth = top's depth + 1
    - First heading resets the stack and is depth 1.
    - TITLE always maps to depth 1 (h1) regardless of the stack.

  LIST_ITEM nesting via left-edge quantization
    - Group contiguous list_item blocks as one "list" group_id.
    - Within a group, collect distinct indent buckets, sort ascending.
    - depth = position in sorted list (0 = shallowest).

  TABLE_ROW / TABLE_*_CELL structure via bbox overlap
    - Cluster cells by vertical overlap to get row_idx.
    - Cluster cells by horizontal overlap to get col_idx.
    - depth = row_idx for TABLE_ROW; col_idx for cells.

TITLE/PARAGRAPH/FIGURE/other non-structured roles get depth=0.

This module is deterministic and pure-algorithm. When output is wrong we
fix the algorithm here, not retrain. See docs/refactor_plan.md §4.
"""
from __future__ import annotations

from .classify import Role
from .types import ClassifiedBlock, ResolvedBlock


def resolve_hierarchy(classified: list[ClassifiedBlock]) -> list[ResolvedBlock]:
    resolved: list[ResolvedBlock] = []

    # First pass: headings.
    heading_depths = _resolve_heading_depths(classified)
    # Second pass: list groups + nesting.
    list_group_ids, list_depths = _resolve_list_groups(classified)
    # Third pass: table groups + row/col indices.
    table_group_ids, row_indices = _resolve_table_groups(classified)

    for i, cb in enumerate(classified):
        role = cb.role
        depth = 0
        group_id = -1
        if role == Role.TITLE.value:
            depth = 1
        elif role == Role.HEADING.value:
            depth = heading_depths[i]
        elif role == Role.LIST_ITEM.value:
            depth = list_depths[i]
            group_id = list_group_ids[i]
        elif role in (Role.TABLE_ROW.value,
                      Role.TABLE_HEADER_CELL.value,
                      Role.TABLE_DATA_CELL.value,
                      Role.TABLE_CAPTION.value,
                      Role.TABLE.value):
            depth = row_indices[i]
            group_id = table_group_ids[i]
        resolved.append(ResolvedBlock(
            classified=cb, depth=depth, group_id=group_id, enrichment=None,
        ))
    return resolved


# ---------- heading depth via font-size stack ----------

def _resolve_heading_depths(blocks: list[ClassifiedBlock]) -> list[int]:
    """Return a depth for every block; only HEADING entries are meaningful."""
    depths = [0] * len(blocks)
    stack: list[tuple[float, int]] = []  # (font_size, depth)

    for i, cb in enumerate(blocks):
        if cb.role != Role.HEADING.value:
            continue
        size = cb.features.raw.font_size or cb.features.relative_font_ratio
        if not stack:
            depths[i] = 1
            stack.append((size, 1))
            continue
        top_size, top_depth = stack[-1]
        EPS = 0.5  # font sizes within half a point are "same"
        if abs(size - top_size) <= EPS:
            depths[i] = top_depth
        elif size > top_size + EPS:
            while stack and stack[-1][0] + EPS < size:
                stack.pop()
            if stack:
                _, d = stack[-1]
                depths[i] = d
            else:
                depths[i] = 1
                stack.append((size, 1))
        else:  # size < top_size
            new_depth = min(6, top_depth + 1)
            depths[i] = new_depth
            stack.append((size, new_depth))
    return depths


# ---------- list groups + nesting via indent bucket ----------

def _resolve_list_groups(blocks: list[ClassifiedBlock]) -> tuple[list[int], list[int]]:
    """Return (group_ids, depths) aligned to `blocks`.

    Contiguous runs of LIST_ITEM blocks share a group_id. Within a group,
    depth is the rank of the block's indent_bucket among the group's distinct
    indent_buckets (0 = shallowest).
    """
    group_ids = [-1] * len(blocks)
    depths = [0] * len(blocks)
    current_group: list[int] = []
    next_gid = 0

    def flush(run_indices: list[int], gid: int):
        # Unique indent buckets in sorted order.
        unique = sorted({blocks[i].features.indent_bucket for i in run_indices})
        rank = {b: r for r, b in enumerate(unique)}
        for i in run_indices:
            group_ids[i] = gid
            depths[i] = rank[blocks[i].features.indent_bucket]

    for i, cb in enumerate(blocks):
        if cb.role == Role.LIST_ITEM.value:
            current_group.append(i)
        else:
            if current_group:
                flush(current_group, next_gid)
                next_gid += 1
                current_group = []
    if current_group:
        flush(current_group, next_gid)
    return group_ids, depths


# ---------- table groups + row/col clustering ----------

def _resolve_table_groups(blocks: list[ClassifiedBlock]) -> tuple[list[int], list[int]]:
    """Simple grouping: contiguous TABLE_* blocks form one table.

    Returns (group_ids, row_indices). row_index is 0-based within the group.
    Full row/col clustering (bbox-based) is in _cluster_cells_to_grid, which
    upper layers (stage 5) will use when assembling <table><tr><td>.
    """
    group_ids = [-1] * len(blocks)
    row_indices = [0] * len(blocks)
    current_group: list[int] = []
    next_gid = 0

    TABLE_ROLES = {
        Role.TABLE.value, Role.TABLE_ROW.value,
        Role.TABLE_HEADER_CELL.value, Role.TABLE_DATA_CELL.value,
        Role.TABLE_CAPTION.value,
    }

    def flush(run_indices: list[int], gid: int):
        # Cluster rows by y-center overlap.
        centers = []
        for i in run_indices:
            b = blocks[i].features.raw.bbox
            centers.append(((b[1] + b[3]) / 2, i))
        centers.sort()
        row_id = -1
        last_y = None
        row_tol = 5.0  # points
        for y, idx in centers:
            if last_y is None or abs(y - last_y) > row_tol:
                row_id += 1
                last_y = y
            group_ids[idx] = gid
            row_indices[idx] = row_id

    for i, cb in enumerate(blocks):
        if cb.role in TABLE_ROLES:
            current_group.append(i)
        else:
            if current_group:
                flush(current_group, next_gid)
                next_gid += 1
                current_group = []
    if current_group:
        flush(current_group, next_gid)
    return group_ids, row_indices


def cluster_cells_to_grid(cells: list[ResolvedBlock]) -> list[list[ResolvedBlock]]:
    """Used by stage 5 when assembling real tables. Returns rows of cells."""
    if not cells:
        return []
    # Group by row_index (which came from depth in the ResolvedBlock).
    rows: dict[int, list[ResolvedBlock]] = {}
    for c in cells:
        rows.setdefault(c.depth, []).append(c)
    ordered_rows = []
    for row_idx in sorted(rows.keys()):
        row = sorted(rows[row_idx], key=lambda b: b.classified.features.raw.bbox[0])
        ordered_rows.append(row)
    return ordered_rows
