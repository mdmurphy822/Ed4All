"""Lossless, split/merge-aware global alignment of PDF blocks to gold HTML.

This is the FOUNDATION (Phase 0) of the structure-pipeline re-architecture.
It replaces the greedy monotonic Jaccard loop in
``data/build_structure_data.py`` (~:568-650) — a 0.30-threshold sliding-window
match that SILENTLY DROPS ~25.6% of the gold supervision and can express
neither a 1-PDF-block → many-gold SPLIT nor a many-PDF-block → 1-gold MERGE.

Design goals (in priority order):

1. **Lossless.** Every PDF block AND every gold element is accounted for in
   exactly one structural bucket (MATCH / SPLIT / MERGE / GOLD_GAP / PDF_GAP).
   ``AlignLedger.assert_complete`` fails closed if not — the same
   token-conservation / R-PART partition invariant the rest of SemantiK
   enforces (see ``qwen_specialists/block_resegment.py``).
2. **Split/merge-aware.** A global DP over the two sequences expresses the
   two moves the greedy loop cannot.
3. **Input-agnostic.** The aligner consumes normalized *views*
   (:class:`FBView`, :class:`GoldView`), never raw extractor dicts, so the
   SAME aligner serves today's rendered-HTML-PDF pairs AND a future
   real-PDF builder track. A real-PDF builder produces ``FBView`` instances;
   nothing else in this module changes.
4. **Full role space at align time; project at write time.**
   :func:`align_blocks` aligns over the FULL gold role space;
   :func:`project_rows` maps to the active head vocabulary at WRITE time.
   Adding a role later is a config change, not an aligner change.

Pure module: stdlib + numpy only. NO torch, NO model loads, no IO. The
ONLY first-party import is ``normalize_for_match`` from
``semantik_structure.text_utils`` — reused (not re-implemented) so this aligner's
tokenization can never drift from the rest of the pipeline's match
normalization.

The integration into ``build_structure_data.py`` is a DELIBERATE follow-up
after the contract (FBView/GoldView fields + ``structure_dataset/3`` row
schema + the real-PDF pair-JSON fields) is reviewed. This module touches no
other tracked file.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from semantik_structure.text_utils import normalize_for_match

__all__ = [
    "SCHEMA_VERSION",
    "FBView",
    "GoldView",
    "AlignRow",
    "AlignLedger",
    "AlignResult",
    "LedgerIncompleteError",
    "sim",
    "align_blocks",
    "project_rows",
]

# The additive row schema version. The v3 row stays backward-readable by
# ``training/train_structure.py::load_split`` (it reads a fixed key set and
# tolerates extra keys); the new fields are purely additive.
SCHEMA_VERSION = "structure_dataset/3"

# Structural move kinds. The first five partition the two sequences (every
# index lands in exactly one); ``low_sim`` / ``role_filtered`` are OVERLAY
# tallies (flavors of a MATCH/SPLIT/MERGE or a projection decision), not
# exclusive buckets.
MOVE_MATCH = "matched"
MOVE_SPLIT = "split"
MOVE_MERGE = "merge"
MOVE_GOLD_GAP = "gold_gap"
MOVE_PDF_GAP = "pdf_gap"

_STRUCTURAL_BUCKETS = (
    MOVE_MATCH,
    MOVE_SPLIT,
    MOVE_MERGE,
    MOVE_GOLD_GAP,
    MOVE_PDF_GAP,
)
_OVERLAY_REASONS = ("role_filtered", "low_sim")

# Per-extra-element op penalty: biases the DP toward fewer ops so a SPLIT or
# MERGE only wins when it is genuinely cheaper than the 1:1 + gap alternative
# (prevents spurious over-splitting when a plain MATCH is comparable).
_OP_PENALTY = 0.05

# Composite-sim weights: Jaccard is the backbone; directional containment
# lifts subset cases (the SPLIT/MERGE signal); char-shingle is the tiebreak.
_W_JACCARD = 0.70
_W_CONTAIN = 0.20
_W_SHINGLE = 0.10

_SHINGLE_N = 3


# ---------------------------------------------------------------------------
# Input-agnostic views
# ---------------------------------------------------------------------------


@dataclass
class FBView:
    """A normalized PDF / source block view — the aligner's PDF-side atom.

    This is the contract a real-PDF builder must also produce (it is
    deliberately decoupled from the rendered-HTML-PDF extractor's merged
    ``text_block`` dict). ``layout`` is the optional 20-dim layout
    side-channel vector (``build_structure_data.LAYOUT_FEATURE_NAMES``);
    ``None`` when a source supplies no geometry (a future text-only track).
    """

    text: str
    bbox: tuple[float, float, float, float]
    page: int
    order_index: int
    layout: list[float] | None = None

    @classmethod
    def from_merged_text_block(
        cls,
        block: dict,
        *,
        page: int,
        order_index: int,
        layout: list[float] | None = None,
    ) -> "FBView":
        """Construct from one ``page['merged']['text_blocks'][k]`` dict.

        The merged text-block shape is ``{text, font_size, bbox, is_bold,
        is_italic}`` (see ``build_structure_data.process_pair``); only
        ``text`` + ``bbox`` are load-bearing for alignment. ``layout`` is the
        caller-computed 20-dim vector (kept out of this helper so the layout
        builder stays the single source of truth in the data builder).
        """
        raw_bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        bbox = tuple(float(v) for v in (list(raw_bbox) + [0.0, 0.0, 0.0, 0.0])[:4])
        return cls(
            text=(block.get("text") or "").strip(),
            bbox=bbox,  # type: ignore[arg-type]
            page=int(page),
            order_index=int(order_index),
            layout=layout,
        )


@dataclass
class GoldView:
    """A normalized gold (HTML ground-truth) element view.

    ``role`` is over the FULL gold role space (``semantik_structure.classify.Role``
    values, as strings) — projection to the active head vocabulary happens at
    write time via :func:`project_rows`, NOT here.
    """

    text: str
    tag: str
    role: str
    gold_index: int

    @classmethod
    def from_html_block(cls, block: dict, gold_index: int) -> "GoldView":
        """Construct from one ``extract_html_blocks`` record.

        The record shape is ``{text, tag, role, list_nesting, in_table_html,
        in_image_block_html}``; ``role`` may be a ``Role`` enum (a
        ``str``-subclass enum) or a bare string — coerced to its string value
        so the view never leaks an enum into serialized rows.
        """
        role = block.get("role")
        role_str = getattr(role, "value", role)
        return cls(
            text=(block.get("text") or "").strip(),
            tag=str(block.get("tag") or ""),
            role=str(role_str) if role_str is not None else "",
            gold_index=int(gold_index),
        )


# ---------------------------------------------------------------------------
# Composite similarity
# ---------------------------------------------------------------------------


def _tokens(s: str) -> frozenset[str]:
    return frozenset(normalize_for_match(s).split())


def _char_shingles(s: str, n: int = _SHINGLE_N) -> frozenset[str]:
    norm = "".join(normalize_for_match(s).split())
    if len(norm) < n:
        return frozenset((norm,)) if norm else frozenset()
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


@lru_cache(maxsize=None)
def _sim_cached(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    union = len(ta | tb)
    jaccard = inter / union
    # Directional containment: max coverage of the smaller side by the larger
    # — the SPLIT/MERGE signal (a gold member fully contained in a fused PDF
    # block scores ~1.0 on containment but only ~0.5 on Jaccard).
    contain = inter / min(len(ta), len(tb))
    sa, sb = _char_shingles(a), _char_shingles(b)
    shingle = (len(sa & sb) / len(sa | sb)) if (sa and sb) else 0.0
    composite = _W_JACCARD * jaccard + _W_CONTAIN * contain + _W_SHINGLE * shingle
    # Numerical safety — weights sum to 1.0 and every term is in [0, 1].
    if composite < 0.0:
        return 0.0
    if composite > 1.0:
        return 1.0
    return composite


def sim(a: str, b: str) -> float:
    """Composite block-text similarity in ``[0, 1]``.

    ``token-Jaccard`` (backbone) + ``directional containment`` (the
    ``a ⊆ b`` / ``b ⊆ a`` signal that makes SPLIT/MERGE expressible) + a
    ``char-shingle`` tiebreak. Symmetric, deterministic, memoized. Empty /
    disjoint-token inputs → ``0.0`` (caller treats as "no support").
    """
    return _sim_cached(a or "", b or "")


# ---------------------------------------------------------------------------
# Row + ledger + result
# ---------------------------------------------------------------------------


@dataclass
class AlignRow:
    """One ``structure_dataset/3`` training row.

    Carries the fields a training row already needs (``text``, ``layout`` 20-d,
    ``labels`` / ``role``, ``html_tag``, ``source``, ``pair``) PLUS additive
    ``provenance`` + ``align`` blocks. Serializes via :meth:`to_row` to the
    SAME JSONL shape ``training/train_structure.py::load_split`` reads — the
    new keys are additive, ``labels.structural_role`` (filled at projection)
    keeps it backward-readable.
    """

    text: str
    layout: list[float]
    role: str  # FULL gold role space; projected to a head id at write time
    html_tag: str
    source: str
    pair: str
    labels: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    align: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def confidence(self) -> float:
        return float(self.align.get("confidence", 0.0))

    @property
    def kind(self) -> str:
        return str(self.align.get("kind", ""))

    def to_row(self) -> dict:
        """Serialize to the JSONL dict the trainer ingests (additive v3)."""
        return {
            "text": self.text,
            "layout": list(self.layout),
            "labels": dict(self.labels),
            "html_tag": self.html_tag,
            "source": self.source,
            "pair": self.pair,
            "schema_version": self.schema_version,
            "provenance": dict(self.provenance),
            "align": dict(self.align),
        }


class LedgerIncompleteError(RuntimeError):
    """Raised by :meth:`AlignLedger.assert_complete` when the structural
    buckets do not cover 100% of both sequences (fail-closed)."""


@dataclass
class AlignLedger:
    """Full alignment accounting — the never-drop audit trail.

    ``counts`` tallies every reason (the 5 structural buckets + the 2 overlay
    flavors). ``per_role`` / ``per_source`` break emitted rows down.
    ``pdf_buckets`` / ``gold_buckets`` map each sequence POSITION to the
    single structural bucket it landed in; double-recording a position fails
    loud (a partition violation), and :meth:`assert_complete` fails closed if
    any position is unaccounted.
    """

    counts: Counter = field(default_factory=Counter)
    per_role: Counter = field(default_factory=Counter)
    per_source: Counter = field(default_factory=Counter)
    projected_roles: Counter = field(default_factory=Counter)
    excluded_roles: Counter = field(default_factory=Counter)
    # position -> bucket (the exclusive structural partition)
    pdf_buckets: dict = field(default_factory=dict)
    gold_buckets: dict = field(default_factory=dict)
    # bucket -> sorted list of positions (audit detail)
    pdf_bucket_indices: dict = field(default_factory=lambda: defaultdict(list))
    gold_bucket_indices: dict = field(default_factory=lambda: defaultdict(list))

    def _claim_pdf(self, pos: int, bucket: str) -> None:
        if pos in self.pdf_buckets:
            raise LedgerIncompleteError(
                f"pdf position {pos} double-recorded "
                f"({self.pdf_buckets[pos]} -> {bucket})"
            )
        self.pdf_buckets[pos] = bucket
        self.pdf_bucket_indices[bucket].append(pos)

    def _claim_gold(self, pos: int, bucket: str) -> None:
        if pos in self.gold_buckets:
            raise LedgerIncompleteError(
                f"gold position {pos} double-recorded "
                f"({self.gold_buckets[pos]} -> {bucket})"
            )
        self.gold_buckets[pos] = bucket
        self.gold_bucket_indices[bucket].append(pos)

    def record_move(
        self,
        bucket: str,
        *,
        pdf_positions: list[int] | None = None,
        gold_positions: list[int] | None = None,
        low_sim: bool = False,
    ) -> None:
        """Record one structural move and claim the positions it consumed."""
        self.counts[bucket] += 1
        for p in pdf_positions or []:
            self._claim_pdf(p, bucket)
        for g in gold_positions or []:
            self._claim_gold(g, bucket)
        if low_sim:
            self.counts["low_sim"] += 1

    def record_row(self, row: AlignRow) -> None:
        """Tally an emitted row by role + source (overlay, not a claim)."""
        self.per_role[row.role] += 1
        self.per_source[row.source] += 1

    def record_role_filtered(self, role: str) -> None:
        self.counts["role_filtered"] += 1
        self.excluded_roles[role] += 1

    def record_role_projected(self, role: str) -> None:
        self.projected_roles[role] += 1

    def assert_complete(self, n_pdf: int, n_gold: int) -> None:
        """Fail closed unless EVERY pdf position ``[0, n_pdf)`` and EVERY gold
        position ``[0, n_gold)`` landed in exactly one structural bucket."""
        missing_pdf = [p for p in range(n_pdf) if p not in self.pdf_buckets]
        extra_pdf = [p for p in self.pdf_buckets if p < 0 or p >= n_pdf]
        missing_gold = [g for g in range(n_gold) if g not in self.gold_buckets]
        extra_gold = [g for g in self.gold_buckets if g < 0 or g >= n_gold]
        if missing_pdf or extra_pdf or missing_gold or extra_gold:
            raise LedgerIncompleteError(
                "alignment is not lossless: "
                f"pdf missing={missing_pdf} extra={extra_pdf} "
                f"gold missing={missing_gold} extra={extra_gold} "
                f"(expected n_pdf={n_pdf}, n_gold={n_gold})"
            )

    def summary(self) -> dict:
        """A plain-dict snapshot for a coverage report."""
        return {
            "counts": dict(self.counts),
            "per_role": dict(self.per_role),
            "per_source": dict(self.per_source),
            "projected_roles": dict(self.projected_roles),
            "excluded_roles": dict(self.excluded_roles),
        }


@dataclass
class AlignResult:
    rows: list[AlignRow]
    ledger: AlignLedger


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
#
# Segmentation rule (documented contract):
#
#   The DP is segmented on PDF *page* boundaries. A single PDF block lives on
#   exactly one physical page, so a true SPLIT (1 PDF block -> many gold) can
#   never straddle a page boundary; a true MERGE (many PDF blocks -> 1 gold)
#   of one semantic unit likewise stays within a page. We therefore cut the
#   PDF sequence at every page change and cut the GOLD sequence at the first
#   CONFIDENT 1:1 anchor (sim to the page's first PDF block >= anchor_floor).
#   A page change with NO confident anchor is NOT cut — those pages fold into
#   the adjacent segment, so the DP is never forced to place a boundary inside
#   a low-confidence region where a split/merge might live. Boundaries land
#   ONLY at confident 1:1 points, which by construction are not splits/merges.
#
# This bounds the O(n*m*max_run) DP to per-segment sizes without ever cutting
# through a real split/merge.


def _resolve_anchor_floor(soft_floor: float) -> float:
    return max(0.5, soft_floor + 0.2)


def _segment(
    pdf_blocks: list[FBView],
    gold_blocks: list[GoldView],
    *,
    anchor_floor: float,
) -> list[tuple[int, int, int, int]]:
    """Return contiguous ``(p0, p1, g0, g1)`` index ranges covering the full
    PDF + gold sequences with no overlaps and no gaps."""
    n, m = len(pdf_blocks), len(gold_blocks)
    if n == 0 or m == 0:
        return [(0, n, 0, m)]

    page_starts = [0]
    for i in range(1, n):
        if pdf_blocks[i].page != pdf_blocks[i - 1].page:
            page_starts.append(i)
    if len(page_starts) <= 1:
        return [(0, n, 0, m)]

    segments: list[tuple[int, int, int, int]] = []
    prev_p, prev_g = 0, 0
    for ps in page_starts[1:]:
        anchor_text = pdf_blocks[ps].text
        best_g, best_s = None, anchor_floor
        for g in range(prev_g, m):
            s = sim(anchor_text, gold_blocks[g].text)
            if s > best_s:
                best_s, best_g = s, g
        if best_g is None or best_g <= prev_g:
            # No confident anchor for this page start -> do not cut; the page
            # folds into the running segment (never cut a low-confidence run).
            continue
        segments.append((prev_p, ps, prev_g, best_g))
        prev_p, prev_g = ps, best_g
    segments.append((prev_p, n, prev_g, m))
    return segments


# ---------------------------------------------------------------------------
# Per-segment DP
# ---------------------------------------------------------------------------


def _align_segment(
    pdf: list[FBView],
    gold: list[GoldView],
    *,
    max_run: int,
    gap_penalty: float,
) -> list[tuple[str, int, int, int]]:
    """5-move global DP over one segment.

    Returns a list of ``(move, i, j, k)`` in document order where ``(i, j)``
    are segment-local start positions and ``k`` is the run length consumed on
    the active side. Moves: ``matched`` / ``split`` (1 pdf -> k gold) /
    ``merge`` (k pdf -> 1 gold) / ``gold_gap`` / ``pdf_gap``.

    cost[i][j] = minimal cost to align ``pdf[i:]`` with ``gold[j:]``.
    """
    n, m = len(pdf), len(gold)
    INF = math.inf
    cost = np.full((n + 1, m + 1), INF, dtype=np.float64)
    # back[i][j] = (move, k)
    back: list[list[tuple[str, int] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    cost[n][m] = 0.0
    for j in range(m - 1, -1, -1):
        cost[n][j] = cost[n][j + 1] + gap_penalty
        back[n][j] = (MOVE_GOLD_GAP, 1)
    for i in range(n - 1, -1, -1):
        cost[i][m] = cost[i + 1][m] + gap_penalty
        back[i][m] = (MOVE_PDF_GAP, 1)

    for i in range(n - 1, -1, -1):
        ptext_i = pdf[i].text
        for j in range(m - 1, -1, -1):
            best = INF
            bmove: tuple[str, int] | None = None

            # MATCH (1:1)
            c = (1.0 - sim(ptext_i, gold[j].text)) + cost[i + 1][j + 1]
            if c < best:
                best, bmove = c, (MOVE_MATCH, 1)

            # SPLIT (1 pdf -> k gold)
            kmax_g = min(max_run, m - j)
            for k in range(2, kmax_g + 1):
                gtext = " ".join(g.text for g in gold[j : j + k])
                c = (
                    (1.0 - sim(ptext_i, gtext))
                    + _OP_PENALTY * (k - 1)
                    + cost[i + 1][j + k]
                )
                if c < best:
                    best, bmove = c, (MOVE_SPLIT, k)

            # MERGE (k pdf -> 1 gold)
            kmax_p = min(max_run, n - i)
            for k in range(2, kmax_p + 1):
                ptext = " ".join(p.text for p in pdf[i : i + k])
                c = (
                    (1.0 - sim(ptext, gold[j].text))
                    + _OP_PENALTY * (k - 1)
                    + cost[i + k][j + 1]
                )
                if c < best:
                    best, bmove = c, (MOVE_MERGE, k)

            # GOLD_GAP
            c = gap_penalty + cost[i][j + 1]
            if c < best:
                best, bmove = c, (MOVE_GOLD_GAP, 1)

            # PDF_GAP
            c = gap_penalty + cost[i + 1][j]
            if c < best:
                best, bmove = c, (MOVE_PDF_GAP, 1)

            cost[i][j] = best
            back[i][j] = bmove

    moves: list[tuple[str, int, int, int]] = []
    i = j = 0
    while i < n or j < m:
        move, k = back[i][j]  # type: ignore[misc]
        if move == MOVE_MATCH:
            moves.append((MOVE_MATCH, i, j, 1))
            i, j = i + 1, j + 1
        elif move == MOVE_SPLIT:
            moves.append((MOVE_SPLIT, i, j, k))
            i, j = i + 1, j + k
        elif move == MOVE_MERGE:
            moves.append((MOVE_MERGE, i, j, k))
            i, j = i + k, j + 1
        elif move == MOVE_GOLD_GAP:
            moves.append((MOVE_GOLD_GAP, i, j, 1))
            j = j + 1
        else:  # MOVE_PDF_GAP
            moves.append((MOVE_PDF_GAP, i, j, 1))
            i = i + 1
    return moves


# ---------------------------------------------------------------------------
# Public aligner
# ---------------------------------------------------------------------------


def _layout_of(fb: FBView) -> list[float]:
    return list(fb.layout) if fb.layout is not None else []


def align_blocks(
    pdf_blocks: list[FBView],
    gold_blocks: list[GoldView],
    *,
    max_run: int = 6,
    soft_floor: float = 0.30,
    source: str = "unknown",
    pair: str = "",
) -> AlignResult:
    """Globally align ``pdf_blocks`` to ``gold_blocks``, lossless + split/merge
    aware.

    Emits one :class:`AlignRow` per (pdf-block, gold-role) unit a model can
    learn from:

    * **MATCH** -> 1 row (pdf text, gold role).
    * **SPLIT** (1 pdf -> k gold) -> k rows, one per recovered gold member
      (gold member text + its role; the shared pdf-block layout/provenance).
      This is the supervision the greedy loop drops.
    * **MERGE** (k pdf -> 1 gold) -> 1 row (concatenated pdf text, the single
      gold role; provenance spans all k pdf blocks).
    * **GOLD_GAP** / **PDF_GAP** -> NO training row (no learnable
      pdf-text↔gold-role pair), but the position is RECORDED in the ledger —
      nothing vanishes.

    A sub-``soft_floor`` association is recorded as a gap (the DP prefers two
    gaps over a below-floor match by construction: the gap penalty crossover
    is set at ``soft_floor``); a below-floor match that the global structure
    still selects is emitted and flagged ``low_sim`` in the ledger.

    The result's ledger ``assert_complete`` is invoked before return, so a
    returned :class:`AlignResult` is lossless by construction (fail-closed).
    """
    max_run = max(1, int(max_run))
    soft_floor = min(max(float(soft_floor), 0.0), 1.0)
    # Gap-penalty crossover: MATCH (cost 1-sim) beats GOLD_GAP+PDF_GAP
    # (cost 2*gap_penalty) iff sim > soft_floor.
    gap_penalty = (1.0 - soft_floor) / 2.0
    anchor_floor = _resolve_anchor_floor(soft_floor)

    ledger = AlignLedger()
    rows: list[AlignRow] = []

    for p0, p1, g0, g1 in _segment(pdf_blocks, gold_blocks, anchor_floor=anchor_floor):
        pdf_seg = pdf_blocks[p0:p1]
        gold_seg = gold_blocks[g0:g1]
        if not pdf_seg and not gold_seg:
            continue
        moves = _align_segment(
            pdf_seg, gold_seg, max_run=max_run, gap_penalty=gap_penalty
        )
        for move, li, lj, k in moves:
            i = p0 + li  # absolute pdf position
            j = g0 + lj  # absolute gold position

            if move == MOVE_MATCH:
                fb = pdf_blocks[i]
                gv = gold_blocks[j]
                conf = sim(fb.text, gv.text)
                low = conf < soft_floor
                ledger.record_move(
                    MOVE_MATCH, pdf_positions=[i], gold_positions=[j], low_sim=low
                )
                row = _build_row(
                    text=fb.text,
                    layout=_layout_of(fb),
                    role=gv.role,
                    html_tag=gv.tag,
                    source=source,
                    pair=pair,
                    fbs=[fb],
                    gold_indices=[gold_blocks[j].gold_index],
                    kind=MOVE_MATCH,
                    confidence=conf,
                    low_sim=low,
                )
                rows.append(row)
                ledger.record_row(row)

            elif move == MOVE_SPLIT:
                fb = pdf_blocks[i]
                members = [gold_blocks[j + t] for t in range(k)]
                gtext = " ".join(g.text for g in members)
                conf = sim(fb.text, gtext)
                low = conf < soft_floor
                ledger.record_move(
                    MOVE_SPLIT,
                    pdf_positions=[i],
                    gold_positions=[j + t for t in range(k)],
                    low_sim=low,
                )
                member_ids = [g.gold_index for g in members]
                for g in members:
                    row = _build_row(
                        text=g.text,  # the recovered single-role gold unit
                        layout=_layout_of(fb),  # shared pdf-block geometry
                        role=g.role,
                        html_tag=g.tag,
                        source=source,
                        pair=pair,
                        fbs=[fb],
                        gold_indices=member_ids,
                        kind=MOVE_SPLIT,
                        confidence=conf,
                        low_sim=low,
                    )
                    rows.append(row)
                    ledger.record_row(row)

            elif move == MOVE_MERGE:
                fbs = [pdf_blocks[i + t] for t in range(k)]
                gv = gold_blocks[j]
                ptext = " ".join(p.text for p in fbs)
                conf = sim(ptext, gv.text)
                low = conf < soft_floor
                ledger.record_move(
                    MOVE_MERGE,
                    pdf_positions=[i + t for t in range(k)],
                    gold_positions=[j],
                    low_sim=low,
                )
                row = _build_row(
                    text=ptext,
                    layout=_layout_of(fbs[0]),  # anchor block geometry
                    role=gv.role,
                    html_tag=gv.tag,
                    source=source,
                    pair=pair,
                    fbs=fbs,
                    gold_indices=[gv.gold_index],
                    kind=MOVE_MERGE,
                    confidence=conf,
                    low_sim=low,
                )
                rows.append(row)
                ledger.record_row(row)

            elif move == MOVE_GOLD_GAP:
                ledger.record_move(MOVE_GOLD_GAP, gold_positions=[j])

            else:  # MOVE_PDF_GAP
                ledger.record_move(MOVE_PDF_GAP, pdf_positions=[i])

    ledger.assert_complete(len(pdf_blocks), len(gold_blocks))
    return AlignResult(rows=rows, ledger=ledger)


def _build_row(
    *,
    text: str,
    layout: list[float],
    role: str,
    html_tag: str,
    source: str,
    pair: str,
    fbs: list[FBView],
    gold_indices: list[int],
    kind: str,
    confidence: float,
    low_sim: bool,
) -> AlignRow:
    anchor = fbs[0]
    return AlignRow(
        text=text,
        layout=layout,
        role=role,
        html_tag=html_tag,
        source=source,
        pair=pair,
        labels={},  # filled at projection (head-specific structural_role id)
        provenance={
            "page": anchor.page,
            "bbox": list(anchor.bbox),
            "pdf_order_index": anchor.order_index,
            "pdf_block_indices": [fb.order_index for fb in fbs],
        },
        align={
            "confidence": round(float(confidence), 6),
            "kind": kind,
            "gold_indices": list(gold_indices),
            "low_sim": bool(low_sim),
        },
    )


# ---------------------------------------------------------------------------
# Role-space projection (full role space -> active head vocabulary)
# ---------------------------------------------------------------------------


def project_rows(
    rows: list[AlignRow],
    active_vocab,
    *,
    ledger: AlignLedger | None = None,
) -> list[AlignRow]:
    """Project full-role-space rows onto the active head vocabulary at WRITE
    time.

    ``active_vocab`` is an ORDERED sequence of role-name strings (e.g.
    ``build_structure_data.ROLE_NAMES``); a row's ``role`` is mapped to its
    index. Rows whose role is not in ``active_vocab`` are EXCLUDED (recorded
    as ``role_filtered`` in the ledger). Kept rows get
    ``labels.structural_role`` (+ ``align.projected = True``) and are recorded
    as projected. Adding a role later is purely a change to ``active_vocab``.

    Returns the kept rows (the others are accounted in the ledger, not
    dropped silently).
    """
    role_to_id = {r: i for i, r in enumerate(active_vocab)}
    kept: list[AlignRow] = []
    for row in rows:
        if row.role in role_to_id:
            row.labels = dict(row.labels)
            row.labels["structural_role"] = role_to_id[row.role]
            row.align = dict(row.align)
            row.align["projected"] = True
            kept.append(row)
            if ledger is not None:
                ledger.record_role_projected(row.role)
        elif ledger is not None:
            ledger.record_role_filtered(row.role)
    return kept
