"""Shared text-normalization + containment helpers for the retrieval layer.

Deterministic, stdlib-only. Two consumers share this module:

  * ``lib.retrieval.citation_anchor`` — resolves a chunk's provenance back to
    its archived source page and verifies text / span.
  * ``lib.retrieval.gold_set`` (Executor B) — verifies each gold-set
    ``text_quote`` is contained in its cited chunk.

Keeping the normalization in one place guarantees the citation anchor, the
gold-set loader, and the chunk-sweep scorer all measure containment the same
way. The whitespace-collapse contract mirrors
``Trainforge/tests/test_provenance.py::_normalize_ws`` and the audit-trail
normalization documented at ``docs/reference/audit-trail.md``.
"""

from __future__ import annotations

from typing import List


def normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip ends.

    Mirrors ``HTMLTextExtractor.get_text`` collapse + the provenance suite's
    ``_normalize_ws``. ``" ".join(s.split())`` folds newlines, tabs, and
    repeated spaces uniformly so a chunk text and a page-text slice that
    differ only in whitespace compare equal.
    """
    return " ".join((text or "").split()).strip()


def shingles(tokens: List[str], size: int) -> List[tuple]:
    """Return the contiguous ``size``-token shingles of ``tokens``.

    When ``len(tokens) < size`` the whole token list is returned as a single
    shingle so short chunks still produce one comparable unit (rather than an
    empty set that would make containment trivially 1.0 or 0.0).
    """
    if size <= 0:
        raise ValueError("shingle size must be a positive integer")
    if not tokens:
        return []
    if len(tokens) <= size:
        return [tuple(tokens)]
    return [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def shingle_containment(needle: str, haystack: str, *, shingle_size: int = 8) -> float:
    """Fraction of ``needle``'s token-shingles present in ``haystack``.

    Both inputs are whitespace-normalized and lowercased before shingling.
    Returns a value in ``[0.0, 1.0]``:

      * ``1.0`` when every shingle of the needle appears in the haystack
        (full containment — the chunk text survives in the page even after
        boilerplate / feedback stripping reorders or drops surrounding spans).
      * ``0.0`` when the needle is empty (no evidence either way → treated as
        not-contained so an empty chunk can't masquerade as resolved).

    Deterministic; set membership only, no ordering requirement (so a chunk
    assembled from two non-adjacent page spans still scores high).
    """
    n_tokens = normalize_ws(needle).lower().split()
    h_tokens = normalize_ws(haystack).lower().split()
    if not n_tokens:
        return 0.0
    needle_shingles = shingles(n_tokens, shingle_size)
    if not needle_shingles:
        return 0.0
    haystack_shingles = set(shingles(h_tokens, shingle_size))
    hits = sum(1 for sh in needle_shingles if sh in haystack_shingles)
    return hits / len(needle_shingles)


__all__ = ["normalize_ws", "shingles", "shingle_containment"]
