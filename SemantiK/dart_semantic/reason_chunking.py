"""Page-based chunking for stage 3b (Qwen reasoner).

Per project decision (2026-04-21), Qwen sees the document in chunks of 3-5
pages with 1-page overlap. Overlap pages appear in two chunks; the label
from the chunk where that page is *more centered* wins. Divergent labels
on overlap pages are logged for diagnostic review — they usually indicate
Qwen uncertainty or a bad boundary split.

This replaces the earlier block-count chunking (MAX_BLOCKS_PER_CHUNK=120)
which could split a table or a list mid-structure.
"""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PAGES_PER_CHUNK = 4
DEFAULT_OVERLAP = 1
# Must match the training-data sub_size (scripts/split_chunks.py: 30).
# The training pipeline first builds 60-block chunks (build_qwen_data ->
# densify) and then split_chunks halves every example into 30-block
# sub-chunks so input+target fit the 2048-token trainer budget. Feeding
# 60-block chunks at inference is OOD: the adapter has to emit ~2x the
# JSON entries it was trained to emit, and the output truncates near
# ~5200 chars (max_new_tokens=2048 in reason._generate). Truncated
# chunks parse-fail and every block in them falls through to the
# DistilBERT hint, masquerading as `model:qwen_miss`.
DEFAULT_MAX_BLOCKS_PER_CHUNK = 30


@dataclass
class PageChunk:
    """One Qwen-visible window."""
    index: int                       # 0-based chunk index
    page_start: int                  # inclusive
    page_end: int                    # inclusive
    block_ids: list[int]             # indices into the full ClassifiedBlock list

    @property
    def midpoint(self) -> float:
        return (self.page_start + self.page_end) / 2.0


def build_page_chunks(
    page_for_block: list[int],
    *,
    pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
    overlap: int = DEFAULT_OVERLAP,
    max_blocks_per_chunk: int | None = DEFAULT_MAX_BLOCKS_PER_CHUNK,
) -> list[PageChunk]:
    """Return [PageChunk(...), ...] covering the given block stream.

    `page_for_block[i]` is the 1-based page number for the i-th block in
    the document's flattened block stream.

    The chunker walks page ranges in stride (pages_per_chunk - overlap).
    Each chunk holds every block whose page is inside [page_start, page_end].
    If the document has fewer than pages_per_chunk pages, a single chunk
    spans the whole document.

    When a resulting chunk holds more than `max_blocks_per_chunk` blocks,
    it is split into contiguous sub-chunks of that size. Sub-chunks do
    NOT overlap each other (page overlap still applies across page-chunk
    boundaries). Pass `None` to disable the cap.
    """
    if not page_for_block:
        return []
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be > 0")
    if overlap < 0 or overlap >= pages_per_chunk:
        raise ValueError("overlap must satisfy 0 <= overlap < pages_per_chunk")
    if max_blocks_per_chunk is not None and max_blocks_per_chunk <= 0:
        raise ValueError("max_blocks_per_chunk must be > 0 or None")

    first_page = min(page_for_block)
    last_page = max(page_for_block)
    stride = pages_per_chunk - overlap

    # Pre-bucket blocks by page so each chunk-assembly pass is O(n).
    blocks_by_page: dict[int, list[int]] = {}
    for i, p in enumerate(page_for_block):
        blocks_by_page.setdefault(p, []).append(i)

    chunks: list[PageChunk] = []
    page = first_page
    idx = 0
    while page <= last_page:
        ps = page
        pe = min(last_page, page + pages_per_chunk - 1)
        block_ids: list[int] = []
        for p in range(ps, pe + 1):
            block_ids.extend(blocks_by_page.get(p, []))
        # Skip empty chunks (pages with no blocks) rather than emit them.
        if block_ids:
            for sub in _split_by_block_count(block_ids, max_blocks_per_chunk):
                sub_ps = _page_of(page_for_block, sub[0])
                sub_pe = _page_of(page_for_block, sub[-1])
                chunks.append(PageChunk(index=idx, page_start=sub_ps,
                                        page_end=sub_pe, block_ids=sub))
                idx += 1
        if pe >= last_page:
            break
        page += stride
    return chunks


def _split_by_block_count(block_ids: list[int],
                          max_blocks: int | None) -> list[list[int]]:
    """Split a flat block_ids list into contiguous windows of `max_blocks`."""
    if max_blocks is None or len(block_ids) <= max_blocks:
        return [block_ids]
    return [block_ids[i:i + max_blocks]
            for i in range(0, len(block_ids), max_blocks)]


def _page_of(page_for_block: list[int], bid: int) -> int:
    return page_for_block[bid]


def reconcile_overlap(
    chunks: list[PageChunk],
    labels_by_chunk: list[dict[int, str]],
) -> tuple[dict[int, str], list[dict]]:
    """Merge per-chunk labels; overlap goes to whichever chunk has the
    page more centered. Returns (final_labels, conflicts).

    `labels_by_chunk[i]` is a dict mapping block_id -> role (as emitted by
    Qwen for chunk i). Block IDs NOT appearing in the dict keep the
    DistilBERT hint (handled upstream).

    `conflicts` is a list of diagnostic records for every block that
    appears in multiple chunks with disagreeing labels:

        {"block_id": int, "chunk_labels": {chunk_idx: role, ...},
         "winner_chunk": int, "winner_role": str}

    A high conflict count signals the overlap is too narrow or Qwen is
    uncertain at chunk boundaries.
    """
    if len(chunks) != len(labels_by_chunk):
        raise ValueError("chunks and labels_by_chunk must align")

    # For each block, find every (chunk_idx, role) pair that covers it.
    per_block: dict[int, list[tuple[int, str]]] = {}
    for ci, ch in enumerate(chunks):
        chunk_labels = labels_by_chunk[ci]
        for bid in ch.block_ids:
            if bid in chunk_labels:
                per_block.setdefault(bid, []).append((ci, chunk_labels[bid]))

    final: dict[int, str] = {}
    conflicts: list[dict] = []

    for bid, candidates in per_block.items():
        if len(candidates) == 1:
            final[bid] = candidates[0][1]
            continue

        # Multiple chunks labeled this block. Pick the chunk where the
        # block's page is most central.
        # (We don't have the block's page here directly; fall back to
        # using the chunk midpoint vs. which chunks cover the block.
        # For overlap=1 this collapses to "later chunk wins when the
        # block is on the first page of the earlier chunk's tail.")
        best_ci, best_role = max(
            candidates,
            key=lambda cr: _centrality_score(chunks[cr[0]], bid),
        )
        final[bid] = best_role

        distinct_roles = {r for _, r in candidates}
        if len(distinct_roles) > 1:
            conflicts.append({
                "block_id": bid,
                "chunk_labels": {ci: r for ci, r in candidates},
                "winner_chunk": best_ci,
                "winner_role": best_role,
            })

    return final, conflicts


def _centrality_score(chunk: PageChunk, block_id: int) -> float:
    """Higher = more central. We use block's position in the chunk's
    block_ids list — middle blocks score higher than edge blocks. This
    is a proxy for page centrality that doesn't require knowing the
    block's actual page number here (callers already filtered to chunks
    covering this block)."""
    try:
        pos = chunk.block_ids.index(block_id)
    except ValueError:
        return float("-inf")
    n = len(chunk.block_ids)
    if n <= 1:
        return 1.0
    # Normalize to [0, 1]; middle (pos == (n-1)/2) scores 1.0; edges score 0.
    mid = (n - 1) / 2
    return 1.0 - abs(pos - mid) / mid
