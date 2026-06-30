"""Reading-order quality metric for SemantiK extraction.

WHY THIS EXISTS
---------------
SemantiK's structure head scores blocks with 20 PER-BLOCK INTRINSIC layout
features (``fs_norm``, ``bold``, ``x0_norm``, ``y0_norm``, ``text_len_log``,
... — see ``data/build_structure_data.LAYOUT_FEATURE_NAMES``). NONE of those
features encode a block's neighbours, ordinal position, or inter-block gaps,
so role macro-F1 is provably INVARIANT to the order in which blocks are fed:
permuting the block sequence permutes the per-block rows but changes no
per-block feature vector and therefore no per-block prediction.

The committed column-order fix (``SEMANTIK_COLUMN_ORDER``,
``dart_semantic/reading_order.py``) re-sorts a two-column page column-major
instead of raster (``y0, x0``) so a screen reader reads down one column then
the next instead of line-interleaving across the gutter. Because role-F1 is
order-invariant, that fix CANNOT show up in role-F1 — its real value is
correct OUTPUT READING ORDER, a WCAG / accessibility property. This module
gives that property a number.

THE METRIC
----------
Inputs: an ordered sequence of PDF blocks AS FED to the pipeline (i.e. already
sorted by whichever reading-order key is under test), each annotated with its
GOLD reading-order rank. The gold HTML is in DOM order, so a gold element's
LIST INDEX *is* its reading-order rank; a PDF block's gold rank is the MIN of
the gold indices it aligned to (a block can align to several gold elements via
a SPLIT — its earliest gold position is where a correct reading order would
place it). A block that aligns to NOTHING (a PDF_GAP — extractor furniture,
running heads, de-hyphenation noise) carries no gold rank and is DROPPED from
the metric; the count of dropped blocks is reported (it is not silently
ignored).

Given the kept blocks in fed order with gold ranks ``g[0..n-1]`` (fed order is
the strictly increasing index ``0,1,...,n-1`` by construction), we compute:

* ``kendall_tau_b`` — Kendall's tau-b between the fed order ``[0..n-1]`` and
  the gold-rank sequence. Because the fed order is strictly increasing with no
  ties, tau-b reduces to the rank correlation of the gold-rank sequence with
  the identity, and tau-b (not tau-a) is used so TIED gold ranks (several PDF
  blocks aligning to the SAME gold element — a MERGE the aligner expressed as
  repeated gold indices, or two fragments of one paragraph) do not penalise the
  score. +1 = fed order perfectly matches gold reading order; 0 = no better
  than random; -1 = exactly reversed. ``None`` when n < 2 (undefined).

* ``pairwise_accuracy`` — fraction of UNORDERED block pairs ``(i, j)`` whose
  relative order in the fed sequence agrees with their relative gold rank.
  Pairs with EQUAL gold rank are excluded from the denominator (their relative
  order is not determined by gold). This is the directly interpretable
  "what share of 'block A should come before block B' judgements does the fed
  order get right" number. ``None`` when there are no rank-distinct pairs.

* ``inversions`` / ``inversions_normalized`` — the count of fed-order pairs
  ``(i < j)`` with ``g[i] > g[j]`` (a strict reading-order inversion: a block
  fed earlier that gold says belongs later), and that count normalised by the
  number of rank-distinct pairs (so ``inversions_normalized == 1 -
  pairwise_accuracy``). Lower is better; 0 = no inversions.

* ``cross_column_inversions`` / ``cross_column_inversions_normalized`` — an
  OPTIONAL view available when a per-block ``column_index`` is supplied: of all
  inversions, the share that are CROSS-COLUMN (the two blocks sit in different
  columns). This isolates the exact failure the column-order fix targets — a
  raster sort interleaves columns, manufacturing cross-column inversions that a
  column-major sort eliminates. Normalised by the number of cross-column
  rank-distinct pairs. ``None`` when no column info is supplied or there are no
  cross-column rank-distinct pairs.

Pure stdlib + scipy (``scipy.stats.kendalltau`` for tau-b). No model loads, no
IO, deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy.stats import kendalltau

__all__ = [
    "ReadingOrderResult",
    "reading_order_metric",
    "aggregate_results",
]


@dataclass
class ReadingOrderResult:
    """Reading-order quality for one ordered block sequence vs gold.

    ``n_total`` is the number of blocks fed; ``n_dropped`` aligned to no gold
    element (no rank) and were excluded; ``n_scored = n_total - n_dropped`` is
    the number actually scored. The ``*_normalized`` fields are in ``[0, 1]``;
    ``kendall_tau_b`` is in ``[-1, 1]``. Metrics that are undefined for the
    given input (too few blocks / no rank-distinct pairs / no column info) are
    ``None``."""

    n_total: int
    n_dropped: int
    n_scored: int
    kendall_tau_b: float | None
    pairwise_accuracy: float | None
    rank_distinct_pairs: int
    inversions: int
    inversions_normalized: float | None
    cross_column_inversions: int | None
    cross_column_rank_distinct_pairs: int | None
    cross_column_inversions_normalized: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def reading_order_metric(
    gold_ranks: list[int | None],
    column_ids: list[int | None] | None = None,
) -> ReadingOrderResult:
    """Score one fed block sequence against gold reading order.

    Parameters
    ----------
    gold_ranks
        One entry per PDF block IN FED ORDER. The value is the block's gold
        reading-order rank (``min`` of its aligned gold indices) or ``None`` if
        the block aligned to nothing (dropped from the metric).
    column_ids
        Optional one-entry-per-block column index (0 = leftmost), aligned to
        ``gold_ranks`` positionally. Enables the cross-column inversion view.
        ``None`` (or a per-block ``None``) disables it for that block.
    """
    n_total = len(gold_ranks)
    if column_ids is not None and len(column_ids) != n_total:
        raise ValueError(
            f"column_ids length {len(column_ids)} != gold_ranks length {n_total}"
        )

    # Drop unaligned blocks, preserving fed order. The kept indices' POSITIONS
    # (0..n_scored-1) are the fed-order ranks (strictly increasing, no ties).
    kept_ranks: list[int] = []
    kept_cols: list[int | None] = []
    for i, r in enumerate(gold_ranks):
        if r is None:
            continue
        kept_ranks.append(int(r))
        kept_cols.append(column_ids[i] if column_ids is not None else None)

    n_scored = len(kept_ranks)
    n_dropped = n_total - n_scored

    # Kendall tau-b: fed order is identity 0..n_scored-1, so tau-b(identity,
    # kept_ranks) == tau-b of kept_ranks against its own position. tau-b
    # handles tied gold ranks (MERGE / fragmented gold) gracefully.
    tau: float | None = None
    if n_scored >= 2:
        fed = list(range(n_scored))
        res = kendalltau(fed, kept_ranks)  # variant='b' is the default
        stat = res.statistic if hasattr(res, "statistic") else res[0]
        # scipy returns nan when one input is constant (all gold ranks tied) —
        # an undefined correlation, surfaced as None rather than a nan leak.
        tau = None if stat != stat else float(stat)

    # Pairwise accuracy / inversions over rank-DISTINCT pairs (fed i < j).
    rank_distinct = 0
    inversions = 0
    cross_distinct = 0
    cross_inversions = 0
    have_cols = column_ids is not None
    for i in range(n_scored):
        gi = kept_ranks[i]
        ci = kept_cols[i]
        for j in range(i + 1, n_scored):
            gj = kept_ranks[j]
            if gi == gj:
                continue  # equal gold rank -> order not determined by gold
            rank_distinct += 1
            inverted = gi > gj  # fed earlier but gold says later
            if inverted:
                inversions += 1
            if have_cols and ci is not None and kept_cols[j] is not None and ci != kept_cols[j]:
                cross_distinct += 1
                if inverted:
                    cross_inversions += 1

    pairwise_accuracy = (
        (rank_distinct - inversions) / rank_distinct if rank_distinct else None
    )
    inversions_normalized = (
        inversions / rank_distinct if rank_distinct else None
    )
    cross_norm = (
        cross_inversions / cross_distinct if (have_cols and cross_distinct) else None
    )

    return ReadingOrderResult(
        n_total=n_total,
        n_dropped=n_dropped,
        n_scored=n_scored,
        kendall_tau_b=tau,
        pairwise_accuracy=pairwise_accuracy,
        rank_distinct_pairs=rank_distinct,
        inversions=inversions,
        inversions_normalized=inversions_normalized,
        cross_column_inversions=cross_inversions if have_cols else None,
        cross_column_rank_distinct_pairs=cross_distinct if have_cols else None,
        cross_column_inversions_normalized=cross_norm,
    )


def aggregate_results(results: list[ReadingOrderResult]) -> dict:
    """Aggregate per-doc :class:`ReadingOrderResult`s into summary stats.

    Reports both a MACRO mean (unweighted per-doc average — every doc counts
    equally, which is what we want when comparing reading-order quality across
    differently-sized docs) and pooled pair-level totals (so a global pairwise
    accuracy / inversion rate is also available). ``None`` per-doc values are
    excluded from their macro mean (and the count of contributing docs is
    reported as ``n_docs_with_tau`` / ``n_docs_with_pairwise``)."""

    def _macro(vals: list[float | None]) -> tuple[float | None, int]:
        present = [v for v in vals if v is not None]
        if not present:
            return None, 0
        return sum(present) / len(present), len(present)

    tau_mean, n_tau = _macro([r.kendall_tau_b for r in results])
    pa_mean, n_pa = _macro([r.pairwise_accuracy for r in results])
    inv_mean, n_inv = _macro([r.inversions_normalized for r in results])
    cc_mean, n_cc = _macro([r.cross_column_inversions_normalized for r in results])

    pooled_distinct = sum(r.rank_distinct_pairs for r in results)
    pooled_inv = sum(r.inversions for r in results)
    pooled_cross_distinct = sum(
        r.cross_column_rank_distinct_pairs or 0 for r in results
    )
    pooled_cross_inv = sum(r.cross_column_inversions or 0 for r in results)

    return {
        "n_docs": len(results),
        "n_blocks_total": sum(r.n_total for r in results),
        "n_blocks_dropped": sum(r.n_dropped for r in results),
        "n_blocks_scored": sum(r.n_scored for r in results),
        "macro_kendall_tau_b": tau_mean,
        "n_docs_with_tau": n_tau,
        "macro_pairwise_accuracy": pa_mean,
        "n_docs_with_pairwise": n_pa,
        "macro_inversions_normalized": inv_mean,
        "n_docs_with_inversions": n_inv,
        "macro_cross_column_inversions_normalized": cc_mean,
        "n_docs_with_cross_column": n_cc,
        "pooled_rank_distinct_pairs": pooled_distinct,
        "pooled_inversions": pooled_inv,
        "pooled_pairwise_accuracy": (
            (pooled_distinct - pooled_inv) / pooled_distinct if pooled_distinct else None
        ),
        "pooled_cross_column_rank_distinct_pairs": pooled_cross_distinct,
        "pooled_cross_column_inversions": pooled_cross_inv,
        "pooled_cross_column_inversions_normalized": (
            pooled_cross_inv / pooled_cross_distinct if pooled_cross_distinct else None
        ),
    }
