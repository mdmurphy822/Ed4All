"""Stage-5e block JOIN / SPLIT pass — FB-boundary re-partitioning only.

A dedicated Stage-5e pass that fixes *logical presentation* defects by
RE-PARTITIONING which FeatureBlocks each Region owns — it MERGES adjacent
same-kind regions that were wrongly cut apart (a list shattered across a
page break, a paragraph continued across a page) and SPLITS one region that
fused two logical units (an over-merged paragraph run, a pedagogical label
that starts a new unit mid-region).

Why a SEPARATE pass (not a reviewer / deterministic-cleaner extension)
----------------------------------------------------------------------
The Stage-5d 70B reviewer (``reviewer.py::_admits``) and the deterministic
cleaner (``deterministic_structure.py::_retag``) BOTH hold the invariant that
``feature_block_indices`` NEVER change — they only re-tag ``kind`` / level /
doc-role of a region whose FB partition is fixed. JOIN/SPLIT is exactly the
operation of MOVING those boundaries, so it cannot live inside either of
those passes; it needs its own invariant harness (R-PART below).

The atom is the FeatureBlock — text conservation by construction
----------------------------------------------------------------
The model (and the deterministic detectors) may move boundaries but never
touch text:

* MERGE concatenates whole FBs (the merged region's text =
  ``reviewer._joined_source_text`` of the union; FB indices = doc-order
  concat of the inputs).
* SPLIT cuts ONLY at FB boundaries (never inside an FB); each child keeps a
  contiguous slice of the parent's FB indices + the derived text slice.

So no word is ever rewritten — only the boundaries move.

R-PART invariant (fail-closed)
------------------------------
Let ``P_in`` = the multiset of FB indices across every input region's
``feature_block_indices`` and ``P_out`` = the same across the outputs.

  1. ``P_out == P_in`` as a multiset — every FB owned exactly once; none
     added / dropped / duplicated.
  2. Document order preserved — no region is re-ordered.
  3. After apply, ``reviewer.assert_token_conservation`` STILL passes.

ANY violation of (1)/(2)/(3) -> FAIL CLOSED: return the INPUT region list
unchanged. (The LLM layer re-validates each proposed op individually, so one
bad op is dropped without poisoning the deterministic result.)

DETERMINISTIC-FIRST, LLM-OPTIONAL
---------------------------------
Mirrors the front-matter-detector-is-load-bearing / 70B-is-secondary
posture. The deterministic detectors (reading the MergeOrSplit council head
off ``CouncilState`` + the pedagogical-label regex + the cross-page
continuation cue) are the load-bearing layer. The optional Stage-3 LLM layer
(``SEMANTIK_BLOCK_RESEGMENT_LLM``) proposes EXTRA ops, each re-validated by
the same gates; an endpoint-down LLM keeps the deterministic result.

NO MODEL / GPU is loaded here — the runtime is injected.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dart_semantic.structure_graph import (
    Region,
    _get_signal,
    _top1,
    resolve_reading_order_fix,
)
from dart_semantic.types import FeatureBlock

from .reviewer import (
    TokenConservationError,
    _extract_verdict_obj,
    _fb_text,
    _joined_source_text,
    assert_token_conservation,
    compute_cluster_signals,  # re-exported for callers; not used internally
)

logger = logging.getLogger(__name__)

_ = compute_cluster_signals  # keep the re-export reachable (parallel to reviewer)


# ---------------------------------------------------------------------------
# Mode resolvers — parse-with-fallback (copied posture from reviewer.py).
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_block_resegment_mode() -> bool:
    """Return True when the Stage-5e block-resegment pass is enabled.

    Reads ``SEMANTIK_BLOCK_RESEGMENT``. Default OFF (unset / blank / falsey /
    garbage) -> byte-identical to today (mirrors
    ``reviewer.resolve_structure_review_mode``). A truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive) enables it.
    """
    raw = (os.environ.get("SEMANTIK_BLOCK_RESEGMENT") or "").strip().lower()
    return raw in _TRUTHY


def resolve_block_resegment_llm_mode() -> bool:
    """Return True when the OPTIONAL Stage-5e LLM op-proposal layer is enabled.

    Reads ``SEMANTIK_BLOCK_RESEGMENT_LLM``. Default OFF. Only meaningful when
    ``resolve_block_resegment_mode`` is also on (the deterministic pass is the
    load-bearing layer; the LLM layer proposes extra ops re-validated by the
    same gates). Parse-with-fallback identical to
    :func:`resolve_block_resegment_mode`.
    """
    raw = (os.environ.get("SEMANTIK_BLOCK_RESEGMENT_LLM") or "").strip().lower()
    return raw in _TRUTHY


def resolve_split_fused_section_titles_mode() -> bool:
    """Return True when the Stage-5e fused-section-title SPLIT arm is enabled.

    Reads ``SEMANTIK_SPLIT_FUSED_SECTION_TITLES``. Default OFF (unset / blank /
    falsey / garbage) -> byte-identical to today (no fused-title op emitted; the
    detector never runs). A truthy value (``1``/``true``/``yes``/``on``,
    case-insensitive) enables it. Copied posture from
    :func:`resolve_block_resegment_mode`; the detector is a THIRD orthogonal
    Stage-5e arm gated per-flag INSIDE :func:`resegment_blocks`, so a new-flag-on
    / others-off run emits ONLY fused-title ops (and vice versa).
    """
    raw = (
        os.environ.get("SEMANTIK_SPLIT_FUSED_SECTION_TITLES") or ""
    ).strip().lower()
    return raw in _TRUTHY


# ``resolve_unit_regroup_mode`` (the ``SEMANTIK_UNIT_REGROUP`` gate) is co-homed
# in the neutral ``dart_semantic.pedagogical_units`` module (Phase 1) so the
# assembler ``shell.py`` can import it cycle-free; re-exported here for
# back-compat (Phase 0 lists it in this module's ``__all__``). The Phase-2
# unit-merge detector also reuses the boundary SoT (``is_unit_boundary`` + the
# two frozensets + ``ABSORB_MAX_RUN``) from that same neutral module.
from dart_semantic.pedagogical_units import (  # noqa: E402
    ABSORB_MAX_RUN,
    BODY_BEARING_COMPONENT_CLASSES,
    UNIT_START_PEDAGOGY_CLASSES,
    is_passthrough_region,
    is_unit_boundary,
    resolve_unit_regroup_mode,
    resolve_unit_regroup_table_mode,
)


# ---------------------------------------------------------------------------
# Detector thresholds.
# ---------------------------------------------------------------------------

# MergeOrSplit "same"/"not_same" confidence floor for a CONFIDENT verdict
# (mirrors the structure_graph default same_logical_block_threshold of 0.5).
_MERGE_SIGNAL_THRESHOLD = 0.5

# Terminal punctuation that closes a sentence/paragraph (the cross-page
# continuation cue uses its ABSENCE on the prev region's last FB).
_TERMINAL_PUNCT = (".", "!", "?", ":", ";", '"', "”", ")", "]")


# ---------------------------------------------------------------------------
# Typed op + audit result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResegmentOp:
    """One join/split operation (the audit + apply unit).

    ``op`` is ``"merge"`` or ``"split"``.

    For a MERGE: ``region_indices`` is the contiguous run of >=2 input-region
    indices to fuse (in document order); ``split_at`` is unused (``()``).

    For a SPLIT: ``region_indices`` is the single region index to cut;
    ``split_at`` is the tuple of FB indices that each START a new child (a cut
    BEFORE that FB). Every value MUST be an interior FB boundary of the
    region (not its first FB).

    ``source_ids`` is the list of ``first_raw_block_index`` (min FB index) of
    the input region(s) the op consumes — the provenance breadcrumb.
    ``origin`` is ``"deterministic"`` or ``"llm"``.
    """

    op: str
    region_indices: tuple[int, ...]
    split_at: tuple[int, ...] = ()
    source_ids: tuple[int, ...] = ()
    origin: str = "deterministic"
    # Discriminator for a cross-kind pedagogical-unit regroup vs a same-kind
    # merge/split. Default ``"merge"`` keeps EVERY existing op (and its
    # hash/serialization/audit row) byte-identical; only the Phase-2
    # ``_detect_unit_merges`` detector sets ``subtype="regroup"``. The
    # ``apply_resegment`` dispatch still keys on ``op`` alone (no new op type),
    # so this field drives only the breadcrumb / audit, never the apply path.
    subtype: str = "merge"
    # Phase 9 audit enrichment: the merged unit's gold component
    # ``semantic_class`` (the anchor's stamped class, else the deterministic
    # css_class->component FLOOR — see ``_resolve_unit_semantic_class``). Set
    # ONLY on a ``subtype='regroup'`` op (``_make_regroup_op`` /
    # ``_build_regroup_op``); default ``None`` keeps every same-kind op + its
    # audit row byte-identical (the audit row reads this only when
    # ``subtype=='regroup'``). It does NOT enter the apply path / R-PART gates.
    semantic_class: str | None = None


class PartitionConservationError(RuntimeError):
    """Raised when the R-PART partition invariant fails (multiset / order).

    :func:`resegment_blocks` catches this and FAILS CLOSED — reverting to the
    input region list (never shipping a content-moving op set that drops /
    duplicates / re-orders FeatureBlocks)."""


# ---------------------------------------------------------------------------
# R-PART partition-conservation assertion.
# ---------------------------------------------------------------------------


def _partition_multiset(regions: list[Region]) -> Counter[int]:
    """Multiset of FB indices owned across all regions (R-PART (1) unit)."""
    counter: Counter[int] = Counter()
    for region in regions:
        counter.update(region.feature_block_indices or ())
    return counter


def _ordered_fb_sequence(regions: list[Region]) -> list[int]:
    """Document-order concatenation of every region's FB indices (R-PART (2)).

    A pure re-partition (merge whole regions / split at FB boundaries) leaves
    this sequence byte-identical — only the region boundaries inside it move.
    """
    seq: list[int] = []
    for region in regions:
        seq.extend(region.feature_block_indices or ())
    return seq


def assert_partition_conservation(
    in_regions: list[Region], out_regions: list[Region]
) -> None:
    """R-PART (1)+(2) — the FB-partition multiset + document-order assertion.

    Asserts:
      * ``multiset(out FB indices) == multiset(in FB indices)`` — every FB
        owned exactly once; none added / dropped / duplicated.
      * the flattened document-order FB sequence is byte-identical — no FB
        was re-ordered (a merge/split only moves region boundaries WITHIN the
        sequence, never permutes it).

    Raises :class:`PartitionConservationError` on any mismatch (fail-closed).
    """
    in_ms = _partition_multiset(in_regions)
    out_ms = _partition_multiset(out_regions)
    if in_ms != out_ms:
        missing = in_ms - out_ms
        extra = out_ms - in_ms
        raise PartitionConservationError(
            "block-resegment partition conservation FAILED: FB multiset "
            f"changed (dropped={dict(missing)!r}, added/duplicated={dict(extra)!r})"
        )
    in_seq = _ordered_fb_sequence(in_regions)
    out_seq = _ordered_fb_sequence(out_regions)
    if in_seq != out_seq:
        raise PartitionConservationError(
            "block-resegment partition conservation FAILED: document order "
            f"changed (in={in_seq!r}, out={out_seq!r})"
        )


# ---------------------------------------------------------------------------
# Deterministic detectors.
# ---------------------------------------------------------------------------


def _region_first_raw(region: Region) -> int:
    """The region's stable id = min(feature_block_indices) (cascade.py:434)."""
    fb = region.feature_block_indices or ()
    return min(fb) if fb else -1


def _region_last_fb_text(region: Region, feature_blocks: list[FeatureBlock]) -> str:
    """Raw text of the region's LAST (highest-index) FeatureBlock."""
    fb = region.feature_block_indices or ()
    if not fb:
        return ""
    return _fb_text(feature_blocks, max(fb))


def _region_first_fb_text(region: Region, feature_blocks: list[FeatureBlock]) -> str:
    """Raw text of the region's FIRST (lowest-index) FeatureBlock."""
    fb = region.feature_block_indices or ()
    if not fb:
        return ""
    return _fb_text(feature_blocks, min(fb))


def _region_pages(region: Region, feature_blocks: list[FeatureBlock]) -> set[int]:
    """Set of physical PDF pages the region's FBs span."""
    pages: set[int] = set()
    for idx in region.feature_block_indices or ():
        try:
            fb = feature_blocks[idx]
        except (IndexError, TypeError):
            continue
        page = getattr(getattr(fb, "raw", None), "page", None)
        if page is not None:
            pages.add(int(page))
    return pages


# Single-sourced passthrough predicate — aliased to the neutral
# ``pedagogical_units.is_passthrough_region`` (table-v2 Phase 1) so the Stage-5e
# resegment and the assembler-side narrow absorb share ONE definition (a future
# passthrough-kind change updates one place). The cross-package import is
# cycle-free (the neutral module is dependency-light). A region is a table/math/
# figure Stage-4 passthrough iff it carries a non-null ``source_region_id``;
# merging/splitting one would orphan the Stage-4 link, so Stage-5e skips them.
_is_passthrough = is_passthrough_region


def _merge_says_not_same_at(
    state: Any, prev_idx: int, threshold: float = _MERGE_SIGNAL_THRESHOLD
) -> bool:
    """MergeOrSplit confidently says 'not_same' for the FB pair (prev, prev+1).

    Mirrors ``structure_graph._merge_present_not_same`` but local to Stage-5e
    so the same ``_get_signal`` + ``_top1`` plumbing is reused. A missing /
    low-confidence signal returns False."""
    if prev_idx < 0:
        return False
    sig = _get_signal(state, "merge_or_split", "same_logical_block", prev_idx)
    if sig is None:
        return False
    label, conf = _top1(sig)
    return label == "not_same" and conf >= threshold


def _merge_says_same_at(
    state: Any, prev_idx: int, threshold: float = _MERGE_SIGNAL_THRESHOLD
) -> bool:
    """MergeOrSplit confidently says 'same' for the FB pair (prev, prev+1).

    A PRESENT 'same' signal at/above threshold (mirrors
    ``structure_graph._merge_present_same``). Missing -> False (the merge must
    be evidence-driven OR ride the deterministic continuation cue)."""
    if prev_idx < 0:
        return False
    sig = _get_signal(state, "merge_or_split", "same_logical_block", prev_idx)
    if sig is None:
        return False
    label, conf = _top1(sig)
    return label == "same" and conf >= threshold


# The pedagogical-label resolver lives in deterministic_structure; import it
# lazily inside the detector to avoid a module-load cycle and keep the SoT.
def _pedagogical_label_at(text: str) -> bool:
    """Whether ``text`` opens with an OpenStax pedagogical label (EXAMPLE 2 /
    Solution / Step 3 / …) — a new logical unit starting mid-region.

    Reuses ``deterministic_structure._pedagogical_class_for`` (the SoT regex)
    so the split cue never drifts from the demotion regex."""
    from .deterministic_structure import _pedagogical_class_for

    return _pedagogical_class_for(text or "") is not None


def _ends_without_terminal_punct(text: str) -> bool:
    """Whether ``text`` ends WITHOUT terminal punctuation (continuation cue)."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    return not stripped.endswith(_TERMINAL_PUNCT)


def _starts_lowercase(text: str) -> bool:
    """Whether ``text``'s first alphabetic character is lowercase (the
    cross-page paragraph-continuation cue's second half)."""
    for ch in (text or "").lstrip():
        if ch.isalpha():
            return ch.islower()
        if ch.isspace():
            continue
        # A leading non-alpha non-space (digit / bullet / quote) is NOT a
        # continuation cue — bail conservatively.
        return False
    return False


def _detect_merges(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    state: Any,
) -> list[ResegmentOp]:
    """Deterministic MERGE detection over the formed region list.

    Fires on a maximal run of >=2 adjacent SAME-KIND, non-passthrough
    regions where each consecutive pair is a continuation:

      * ``list`` regions: MergeOrSplit 'same' at the boundary FB pair OR the
        regions are on different pages (a shattered list across a page break)
        AND no confident 'not_same' signal opposes it.
      * ``paragraph`` regions: a confident 'same' signal, OR — when there is
        NO within-page merge signal — the deterministic cross-page cue (prev
        region's last FB ends without terminal punctuation AND next region's
        first FB starts lowercase), with no confident 'not_same' opposing.

    v1 scope cut: SAME-KIND only (never heading+body); passthrough regions
    (table/math) are skipped. Returns non-overlapping merge ops in document
    order.
    """
    ops: list[ResegmentOp] = []
    n = len(regions)
    i = 0
    while i < n:
        region = regions[i]
        kind = str(region.kind)
        if kind not in ("list", "paragraph") or _is_passthrough(region):
            i += 1
            continue
        run = [i]
        j = i + 1
        while j < n:
            prev = regions[run[-1]]
            cur = regions[j]
            if str(cur.kind) != kind or _is_passthrough(cur):
                break
            if not _pair_is_continuation(prev, cur, feature_blocks, state, kind):
                break
            run.append(j)
            j += 1
        if len(run) >= 2:
            source_ids = tuple(_region_first_raw(regions[k]) for k in run)
            ops.append(
                ResegmentOp(
                    op="merge",
                    region_indices=tuple(run),
                    source_ids=source_ids,
                    origin="deterministic",
                )
            )
            i = run[-1] + 1
        else:
            i += 1
    return ops


def _pair_is_continuation(
    prev: Region,
    cur: Region,
    feature_blocks: list[FeatureBlock],
    state: Any,
    kind: str,
) -> bool:
    """Whether ``cur`` continues ``prev`` (the per-pair merge predicate).

    The boundary FB pair is (prev's last FB, cur's first FB). MergeOrSplit's
    ``same_logical_block`` head is keyed on the LEFT FB index of an adjacent
    pair; the boundary here is the consecutive pair (last-of-prev,
    first-of-cur) which is consecutive only when ``cur`` begins at
    ``prev_last + 1`` (the common formed-region case)."""
    prev_last = max(prev.feature_block_indices or (-1,))
    cur_first = min(cur.feature_block_indices or (-1,))
    boundary_idx = prev_last if (cur_first == prev_last + 1) else -1

    # A confident 'not_same' boundary signal always vetoes a merge.
    if _merge_says_not_same_at(state, boundary_idx):
        return False

    if _merge_says_same_at(state, boundary_idx):
        return True

    prev_pages = _region_pages(prev, feature_blocks)
    cur_pages = _region_pages(cur, feature_blocks)
    cross_page = bool(prev_pages and cur_pages and prev_pages.isdisjoint(cur_pages))

    if kind == "list":
        # A list shattered across a page break: adjacent list regions on
        # different pages with no opposing not_same signal.
        return cross_page

    # paragraph: the deterministic cross-page continuation cue (no within-page
    # signal needed) — prev ends mid-sentence AND cur starts lowercase.
    if not cross_page:
        return False
    prev_tail = _region_last_fb_text(prev, feature_blocks)
    cur_head = _region_first_fb_text(cur, feature_blocks)
    return _ends_without_terminal_punct(prev_tail) and _starts_lowercase(cur_head)


def _detect_splits(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    state: Any,
) -> list[ResegmentOp]:
    """Deterministic SPLIT detection over the formed region list.

    For each non-passthrough region with >=2 FBs, find interior FB boundaries
    to cut at:

      * a boundary where MergeOrSplit confidently says 'not_same' (the region
        was over-merged by the defensive missing-signal formation merge);
      * a boundary whose RIGHT FB's leading text matches the pedagogical-label
        regex ("EXAMPLE 2" / "Solution" / "Step 3") — a new logical unit
        started mid-region; split BEFORE it.

    Cuts are taken only at FB boundaries (never inside an FB). Returns one op
    per region that needs cutting, carrying every interior cut point.
    """
    ops: list[ResegmentOp] = []
    for region_index, region in enumerate(regions):
        if _is_passthrough(region):
            continue
        fb_indices = list(region.feature_block_indices or ())
        if len(fb_indices) < 2:
            continue
        cut_at: list[int] = []
        # Interior boundaries: between fb_indices[k-1] and fb_indices[k].
        for k in range(1, len(fb_indices)):
            left_fb = fb_indices[k - 1]
            right_fb = fb_indices[k]
            confident_not_same = (
                right_fb == left_fb + 1
                and _merge_says_not_same_at(state, left_fb)
            )
            pedagogical = _pedagogical_label_at(_fb_text(feature_blocks, right_fb))
            if confident_not_same or pedagogical:
                cut_at.append(right_fb)
        if cut_at:
            ops.append(
                ResegmentOp(
                    op="split",
                    region_indices=(region_index,),
                    split_at=tuple(cut_at),
                    source_ids=(_region_first_raw(region),),
                    origin="deterministic",
                )
            )
    return ops


# ---------------------------------------------------------------------------
# Fused-section-title SPLIT detection (lane B — SEMANTIK_SPLIT_FUSED_SECTION_TITLES).
#
# A scan often fuses a section's title into its first body paragraph
# ("[y| Introduction Suppose a stone falls ..." / "4.2 Solve Linear Systems A
# system of two linear equations ...") so no standalone heading candidate
# survives. This detector SPLITS the region at the FeatureBlock boundary between
# the fused title and the prose body; the post-Stage-5e promotion hook
# (``deterministic_structure.promote_fused_title_children``, gated on
# SEMANTIK_PROMOTE_SECTION_HEADINGS) then promotes the split-off title child to a
# section heading. The FB is the atom: a title fused with prose INSIDE ONE
# FeatureBlock is structurally unsplittable (R-PART forbids intra-FB cuts) and is
# recorded as a detect-only ``fused_title_intra_fb`` diagnostic.
# ---------------------------------------------------------------------------

# Title may wrap over up to this many leading OCR-line FeatureBlocks.
_FUSED_TITLE_MAX_PREFIX_FBS = 3
# Prose remainder floor (">= 1 sentence of prose").
_FUSED_TITLE_MIN_REMAINDER_WORDS = 8
# N.M section-title word ceiling / section-index (M) ceiling.
_FUSED_TITLE_SECTION_MAX_WORDS = 10
_FUSED_TITLE_SECTION_M_MAX = 40
# Sentence-terminal punctuation the prose remainder must carry.
_SENTENCE_TERMINALS = (".", "!", "?")


def _has_sentence_terminal(text: str) -> bool:
    """Whether ``text`` carries a sentence-terminal ``.``/``!``/``?``."""
    return any(p in (text or "") for p in _SENTENCE_TERMINALS)


def _is_prose_remainder(text: str) -> bool:
    """Whether ``text`` is ">= 1 sentence of prose" (the fused-title body cue).

    Requires >= ``_FUSED_TITLE_MIN_REMAINDER_WORDS`` whitespace tokens AND a
    sentence-terminal — so a title split off a genuine prose paragraph, never a
    second short title / label.
    """
    if not text:
        return False
    if len(text.split()) < _FUSED_TITLE_MIN_REMAINDER_WORDS:
        return False
    return _has_sentence_terminal(text)


def _fused_title_prefix_kind(prefix: str, dominant_ordinal: int | None) -> str | None:
    """Classify ``prefix`` as a WHOLE fused-title shape: 'section' / 'apparatus'.

    Returns ``"section"`` when ``prefix`` is WHOLLY a chapter-consistent
    ``"N.M Title-Case"`` (N == ``dominant_ordinal``, M in 1..40, <= 10 words),
    ``"apparatus"`` when it is a whole-match apparatus opener, else ``None``. The
    N.M arm is disabled when ``dominant_ordinal`` is None (ambiguous doc). Reuses
    the deterministic_structure SoT (``_STANDALONE_SECTION_RE`` +
    ``_matches_apparatus_opener``) via a lazy import (mirrors the
    ``_pedagogical_label_at`` idiom) so the split shape never drifts from the
    promotion re-verify.
    """
    from .deterministic_structure import (
        _STANDALONE_SECTION_RE,
        _matches_apparatus_opener,
    )

    text = (prefix or "").strip()
    if not text:
        return None
    if dominant_ordinal is not None:
        m = _STANDALONE_SECTION_RE.match(text)
        if m and m.group(0).strip() == text:
            try:
                n, mm = int(m.group(1)), int(m.group(2))
            except (TypeError, ValueError):
                n = mm = -1
            if (
                n == dominant_ordinal
                and 1 <= mm <= _FUSED_TITLE_SECTION_M_MAX
                and len(text.split()) <= _FUSED_TITLE_SECTION_MAX_WORDS
            ):
                return "section"
    if _matches_apparatus_opener(text):
        return "apparatus"
    return None


def _region_intra_fb_fusion(
    region: Region,
    feature_blocks: list[FeatureBlock],
    dominant_ordinal: int | None,
) -> bool:
    """Whether the region's FIRST FB fuses a title shape WITH in-FB prose.

    The unsplittable case: the title and its following prose live inside ONE
    FeatureBlock, so no FB boundary separates them (R-PART forbids intra-FB
    cuts). Fires when FB0 OPENS with an apparatus opener followed by >= the prose
    floor of trailing words + a sentence-terminal, OR opens with a
    chapter-consistent ``N.M`` ordinal, carries more than a title's worth of
    words, and has a sentence-terminal. Detect-only (the caller counts it).
    """
    fb_indices = list(region.feature_block_indices or ())
    if not fb_indices:
        return False
    first = _fb_text(feature_blocks, fb_indices[0])
    if not first:
        return False
    from .deterministic_structure import (
        _ANY_SECTION_ORDINAL_RE,
        _apparatus_opener_match_len,
    )

    n = _apparatus_opener_match_len(first)
    if n is not None:
        tail = first[n:]
        if (
            len(tail.split()) >= _FUSED_TITLE_MIN_REMAINDER_WORDS
            and _has_sentence_terminal(tail)
        ):
            return True
    if dominant_ordinal is not None:
        m = _ANY_SECTION_ORDINAL_RE.match(first)
        if m:
            try:
                same = int(m.group(1)) == dominant_ordinal
            except (TypeError, ValueError):
                same = False
            if (
                same
                and len(first.split()) > _FUSED_TITLE_SECTION_MAX_WORDS
                and _has_sentence_terminal(first)
            ):
                return True
    return False


def _detect_fused_title_splits(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
) -> tuple[list[ResegmentOp], int]:
    """Deterministic fused-section-title SPLIT detection (lane B).

    For each NON-heading, NON-passthrough region with >= 2 FBs, try prefix
    lengths k=1..3 (the title may wrap over 2-3 OCR line FBs): the prefix is the
    whitespace-joined text of FBs[0..k-1], the remainder is FBs[k..]. Emit a
    single split at the FB boundary ``fb_indices[k]`` (subtype ``fused_title``)
    when the prefix is WHOLLY a title shape (:func:`_fused_title_prefix_kind`)
    AND the remainder is >= 1 sentence of prose (:func:`_is_prose_remainder`).
    First matching k wins; at most one cut per region (the title is at region
    START by contract). A region whose title is fused with prose INSIDE ONE FB
    (no k aligns) increments the returned ``intra_fb`` counter (detect-only —
    the FB is the atom).

    Returns ``(ops, intra_fb_count)``. NO model / GPU is loaded.
    """
    ops: list[ResegmentOp] = []
    intra_fb = 0
    from .deterministic_structure import _dominant_chapter_ordinal

    dominant = _dominant_chapter_ordinal(regions)
    for region_index, region in enumerate(regions):
        if _is_passthrough(region) or str(region.kind) == "heading":
            continue
        fb_indices = list(region.feature_block_indices or ())
        if not fb_indices:
            continue
        matched = False
        if len(fb_indices) >= 2:
            max_k = min(_FUSED_TITLE_MAX_PREFIX_FBS, len(fb_indices) - 1)
            for k in range(1, max_k + 1):
                prefix = " ".join(
                    t
                    for t in (
                        _fb_text(feature_blocks, fb_indices[j]) for j in range(k)
                    )
                    if t
                )
                if _fused_title_prefix_kind(prefix, dominant) is None:
                    continue
                remainder = " ".join(
                    t
                    for t in (
                        _fb_text(feature_blocks, fb_indices[j])
                        for j in range(k, len(fb_indices))
                    )
                    if t
                )
                if not _is_prose_remainder(remainder):
                    continue
                ops.append(
                    ResegmentOp(
                        op="split",
                        region_indices=(region_index,),
                        split_at=(fb_indices[k],),
                        source_ids=(_region_first_raw(region),),
                        origin="deterministic",
                        subtype="fused_title",
                    )
                )
                matched = True
                break
        if not matched and _region_intra_fb_fusion(
            region, feature_blocks, dominant
        ):
            intra_fb += 1
    return ops, intra_fb


def _fused_title_op_valid(op: ResegmentOp, regions: list[Region]) -> bool:
    """Structural validity of a fused-title SPLIT op (interior FB-boundary cuts).

    A wrong split never poisons the pass: this belt (paired with the per-op
    conservation re-validation in the driver) drops an op whose ``region_indices``
    is out of range, whose ``split_at`` is empty / out-of-range, or whose cut is
    NON-INTERIOR (the region's first FB — never a valid boundary). Every cut must
    be an interior FB boundary of the single target region.
    """
    if op.op != "split" or len(op.region_indices) != 1:
        return False
    idx = op.region_indices[0]
    if not (0 <= idx < len(regions)):
        return False
    fb_indices = list(regions[idx].feature_block_indices or ())
    if len(fb_indices) < 2 or not op.split_at:
        return False
    interior = set(fb_indices[1:])  # every FB except the first starts a child
    return all(cut in interior for cut in op.split_at)


# ---------------------------------------------------------------------------
# Cross-kind pedagogical-unit MERGE detection (Phase 2 — uncalled).
# ---------------------------------------------------------------------------

# Stage-6-serving-window budget (whitespace-token count) bounding a single
# merged unit. Unlike the assembler-side absorb (which concatenates already-
# generated ``region_html`` fragments POST-Stage-6), the regroup FUSES regions
# at Stage-5e, so a merged unit becomes ONE Stage-6 specialist generation unit.
# ``ABSORB_MAX_RUN`` caps region COUNT, not tokens — a long worked example could
# exceed the local-7B serving window (the documented authoring head-truncation
# failure mode). This token cap bounds the summed raw-FB text of the run; a unit
# longer than the budget caps the run early (parity with the count cap, strictly
# safer than orphaning the body). Mirrors the reviewer's
# ``_ESCALATION_SOURCE_TOKEN_CAP`` (2048) "fit-n_ctx" rationale.
_UNIT_REGROUP_TOKEN_BUDGET = 2048


def _is_unit_anchor(region: Region) -> bool:
    """Whether ``region`` OPENS a pedagogical unit — the regroup ANCHOR.

    True when its always-on ``payload['css_class']`` is a unit-start pedagogy
    hint (``pedagogy-example`` / ``-try-it`` / ``-how-to`` / ``-be-prepared`` /
    ``-practice``, stamped by ``deterministic_structure._retag``) OR its
    ``payload['semantic_class']`` is a body-bearing component
    (``worked_example`` / ``definition_region`` / ``exercise``, stamped by the
    Stage-5d reviewer). The css_class anchor is primary (always present); the
    semantic_class anchor lights up only when the reviewer ran.
    """
    payload = getattr(region, "payload", None) or {}
    return (
        payload.get("css_class") in UNIT_START_PEDAGOGY_CLASSES
        or payload.get("semantic_class") in BODY_BEARING_COMPONENT_CLASSES
    )


def _region_token_count(region: Region, feature_blocks: list[FeatureBlock]) -> int:
    """Whitespace-token count of the region's summed raw-FB text (the budget
    metric — re-derived from raw FBs, never ``payload['text']``)."""
    total = 0
    for idx in region.feature_block_indices or ():
        total += len(_fb_text(feature_blocks, idx).split())
    return total


def _resolve_unit_semantic_class(anchor: Region) -> str | None:
    """The merged unit's gold component ``semantic_class``.

    Single SoT for the merged region's stamp (:func:`_merged_region`) AND the
    Phase-9 audit row (:func:`_make_regroup_op` / :func:`_build_regroup_op`):
    the anchor's already-stamped ``semantic_class`` (Stage-5d reviewer) wins;
    otherwise the always-on ``css_class`` is mapped through the deterministic
    catalog FLOOR (``component_for_pedagogy_class``, catalog-validated so it
    NEVER invents a component — anti-fabrication, mirrors the Stage-5d FLOOR).
    Returns ``None`` when neither resolves. Lazy import keeps the
    qwen_specialists->assembler edge off the module-top graph (mirrors
    ``reviewer.py``'s lazy ``semantic_catalog`` imports)."""
    payload = getattr(anchor, "payload", None) or {}
    existing = payload.get("semantic_class")
    if existing:
        return existing
    css_class = payload.get("css_class")
    if not css_class:
        return None
    from dart_semantic.assembler.semantic_catalog import (
        component_for_pedagogy_class,
        valid_components,
    )

    component = component_for_pedagogy_class(css_class)
    if component and component in valid_components():
        return component
    return None


def _build_regroup_op(
    run: list[int], regions: list[Region], *, origin: str = "deterministic"
) -> ResegmentOp:
    """Construct a ``subtype='regroup'`` merge op from a region-index run.

    The LOW-LEVEL run->op factory shared by the deterministic detector
    (:func:`_make_regroup_op`, which adds the anchor-first construction assert)
    AND the Phase-10 verifier-drivable seam (:func:`apply_proposed_regroups`,
    which supplies an EXTERNALLY-proposed run and re-validates it through the
    same gates rather than asserting). ``run[0]`` is treated as the anchor by
    payload inheritance; ``semantic_class`` is resolved off it for the audit.
    No anchor-pedagogy assert here so a verifier-proposed run is dropped by the
    R-PART/token gates (graceful) rather than raising.

    **Provenance blast radius (Phase 8):** the merge REDUCES region count
    (N->1), so the FB partition *multiset* stays conserved (R-PART) but the
    ``source_ids`` / ``block_id`` *SET shrinks* — every BODY region's id folds
    into the LABEL's ``min(feature_block_indices)`` (the unit's stable id =
    ``source_ids[0]``). This is NOT a stable multiset like the reading-order
    reorder fix; it matches the same-kind ``_detect_merges`` precedent."""
    source_ids = tuple(_region_first_raw(regions[k]) for k in run)
    return ResegmentOp(
        op="merge",
        subtype="regroup",
        region_indices=tuple(run),
        source_ids=source_ids,
        origin=origin,
        semantic_class=_resolve_unit_semantic_class(regions[run[0]]),
    )


def _make_regroup_op(run: list[int], regions: list[Region]) -> ResegmentOp:
    """Build a ``subtype='regroup'`` merge op from a contiguous run, asserting
    the ANCHOR-FIRST invariant at CONSTRUCTION (Phase 3).

    ``region_indices[0]`` MUST be the unit anchor (a unit-start ``css_class`` or
    a body-bearing ``semantic_class``) — the merged region inherits ``run[0]``'s
    payload (so the unit's ``semantic_class`` / ``css_class`` carry onto the
    merged region for free), so a body-first run would silently lose the unit's
    semantic class. The assert lives HERE, where the anchor is known — NOT in the
    shared :func:`_merged_region`, which same-kind merges also call with a
    NON-anchor ``run[0]`` (an unconditional assert there would wrongly raise). A
    future refactor that builds a body-first regroup run fails loudly here
    instead of silently mis-stamping. Delegates the op construction to
    :func:`_build_regroup_op` (the shared low-level factory).
    """
    assert _is_unit_anchor(
        regions[run[0]]
    ), "regroup anchor-first invariant: region_indices[0] must be the unit anchor"
    return _build_regroup_op(run, regions, origin="deterministic")


def _detect_unit_merges(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    state: Any,
) -> list[ResegmentOp]:
    """Deterministic CROSS-KIND pedagogical-unit MERGE detection (Phase 2).

    A SIBLING of :func:`_detect_merges` (which is SAME-KIND only) — this one
    fuses a unit's LABEL region with its following BODY regions across kinds.
    Reuses the neutral boundary SoT (``is_unit_boundary`` + the two frozensets +
    ``ABSORB_MAX_RUN``).

    For each non-passthrough ANCHOR region (``_is_unit_anchor`` — a unit-start
    ``css_class`` or a body-bearing ``semantic_class``), walk forward over the
    contiguous following regions absorbing each (pedagogy-solution/-step, plain
    paragraph, list) until the FIRST of:

      * ``is_unit_boundary(region, html=None)`` — the next unit-start label, a
        heading region, the next body-bearing component, or an empty region;
      * a passthrough table/math region (``_is_passthrough`` — v1 stops here so
        the Stage-4 ``source_region_id`` link is never orphaned; Phase 4 pins
        this);
      * ``ABSORB_MAX_RUN`` regions absorbed (the conservative count cap);
      * the ``_UNIT_REGROUP_TOKEN_BUDGET`` token cap (the Stage-6 generation
        window).

    FB-adjacency (``cur_first == prev_last + 1``, the same test
    :func:`_pair_is_continuation` uses) is REQUIRED at each step so a
    non-reordered / kind-segregated list never produces a spurious cross-document
    run (Phase 6 separately guards on the reading-order fix). Emits one
    ``ResegmentOp(op='merge', subtype='regroup', …)`` per run of length >= 2
    (mirrors ``compute_absorption_runs``' ``end > i+1`` guard); the ANCHOR is
    ALWAYS ``region_indices[0]``.

    UNCALLED in Phase 2 — no driver appends these ops yet (Phase 6 wires it),
    so this function is pure / GPU-free / zero behavior change.
    """
    # Self-gate (Phase 3): off -> no op emitted (Phase 6 ALSO gates inside the
    # driver; this is the defensive belt so a stray direct call is inert when
    # the flag is off).
    if not resolve_unit_regroup_mode():
        return []
    ops: list[ResegmentOp] = []
    n = len(regions)
    consumed: set[int] = set()
    i = 0
    while i < n:
        if i in consumed:
            i += 1
            continue
        anchor = regions[i]
        if _is_passthrough(anchor) or not _is_unit_anchor(anchor):
            i += 1
            continue
        run = [i]
        acc_tokens = _region_token_count(anchor, feature_blocks)
        j = i + 1
        while j < n and (j - i) < ABSORB_MAX_RUN:
            prev = regions[run[-1]]
            cur = regions[j]
            # FB-adjacency: cur must begin right after prev ends (the
            # reading-order-fix-makes-this-contiguous assumption is NOT baked in
            # — a non-adjacent follower ends the run).
            prev_last = max(prev.feature_block_indices or (-1,))
            cur_first = min(cur.feature_block_indices or (-1,))
            if cur_first != prev_last + 1:
                break
            # v1 passthrough stop (Phase 4): never fuse a Stage-4 table/math
            # region — ``_merged_region`` cannot preserve a child's
            # ``source_region_id``, so absorbing one would ORPHAN the Stage-4
            # table/math link (the example box ends just before an embedded
            # table; the table renders as its own adjacent region). The
            # justification is source_region_id ORPHANING, NOT token
            # conservation (which is kind-agnostic and re-derives from raw FBs).
            # v2: a future split-then-regroup that RETAINS the absorbed
            # passthrough child's ``source_region_id`` would let the table live
            # INSIDE the box.
            if _is_passthrough(cur):
                break
            if is_unit_boundary(cur, html=None):
                break
            cur_tokens = _region_token_count(cur, feature_blocks)
            if acc_tokens + cur_tokens > _UNIT_REGROUP_TOKEN_BUDGET:
                # The merged unit would overflow the Stage-6 serving window —
                # cap the run here (the rest stays as separate regions; no FB is
                # orphaned, R-PART holds).
                break
            run.append(j)
            acc_tokens += cur_tokens
            j += 1
        if len(run) >= 2:
            # Anchor-first invariant asserted at CONSTRUCTION inside
            # _make_regroup_op (region_indices[0] is the unit anchor, so the
            # merged region inherits the LABEL's semantic_class/css_class in
            # Phase 3).
            ops.append(_make_regroup_op(run, regions))
            consumed.update(run)
            i = run[-1] + 1
        else:
            i += 1
    return ops


# ---------------------------------------------------------------------------
# Op application.
# ---------------------------------------------------------------------------


def _merged_region(
    run_regions: list[Region],
    feature_blocks: list[FeatureBlock],
    *,
    subtype: str = "merge",
) -> Region:
    """Build the merged region from a doc-order run of regions.

    ``feature_block_indices`` = doc-order concat; ``payload['text']`` =
    ``_joined_source_text`` of the union; ``pages`` (if the payload carries
    one) = union; kind = the shared kind; keeps the lowest source id by
    construction (min FB index is preserved). A ``resegment`` breadcrumb is
    stamped on the payload (mirrors ``structure_clean``).

    ``subtype`` (Phase 3) discriminates a cross-kind pedagogical-unit REGROUP
    (``'regroup'``) from a same-kind merge (``'merge'``, the default). The
    default keeps every EXISTING same-kind merge byte-identical. For a regroup
    the merged region carries the unit's ``semantic_class`` (inherited from the
    anchor ``run[0]`` for free, or DETERMINISTICALLY derived from the always-on
    ``css_class`` when the reviewer was off) so the per-region gold wrap boxes
    the WHOLE unit, and the breadcrumb records ``op='regroup'``."""
    first = run_regions[0]
    fb_union: list[int] = []
    for r in run_regions:
        fb_union.extend(r.feature_block_indices or ())
    merged_fb = tuple(fb_union)

    payload = dict(first.payload or {})
    candidate = dataclasses.replace(first, feature_block_indices=merged_fb)
    payload["text"] = _joined_source_text(candidate, feature_blocks)

    # Union the per-region page set when the payload tracks one.
    pages: set[int] = set()
    for r in run_regions:
        existing = (r.payload or {}).get("pages")
        if existing:
            pages.update(int(p) for p in existing)
    if pages:
        payload["pages"] = sorted(pages)

    # Regroup-subtype path ONLY (same-kind merges stay byte-identical):
    breadcrumb_op = "merge"
    if subtype == "regroup":
        breadcrumb_op = "regroup"
        # The anchor's pay['semantic_class'] (stamped by the Stage-5d reviewer)
        # already carries onto `payload` via the dict copy above. When the
        # reviewer was OFF but the always-on css_class is present, DERIVE the
        # gold component deterministically so the wrap still fires (the shared
        # _resolve_unit_semantic_class SoT — catalog-validated, never invents;
        # the SAME resolver feeds the Phase-9 audit row so the merged region's
        # class and the op's audited class can never disagree).
        if not payload.get("semantic_class"):
            derived = _resolve_unit_semantic_class(first)
            if derived:
                payload["semantic_class"] = derived

    payload["resegment"] = {
        "op": breadcrumb_op,
        "source_ids": [_region_first_raw(r) for r in run_regions],
        "conservation_verified": True,
    }
    return dataclasses.replace(
        first, feature_block_indices=merged_fb, payload=payload
    )


def _split_children(
    region: Region,
    split_at: tuple[int, ...],
    feature_blocks: list[FeatureBlock],
    *,
    subtype: str = "merge",
) -> list[Region]:
    """Cut one region into K children at the FB boundaries in ``split_at``.

    Each child keeps a contiguous slice of the parent's FB indices + the
    derived text slice (``_joined_source_text`` of the slice). Child 0 keeps
    the parent's id (lowest FB index preserved); later children get a
    deterministic id from their slice min (cascade.py:434 recomputes it).
    Each child carries a ``resegment`` breadcrumb.

    ``subtype`` (lane B) discriminates a fused-title split (``'fused_title'``)
    from a same-kind split (``'merge'``, the default). It is stamped on the
    breadcrumb ONLY when non-default, so every EXISTING same-kind split
    breadcrumb (and its tests) stays byte-identical; the post-Stage-5e promotion
    hook keys on ``subtype=='fused_title'`` + ``child_index==0``."""
    fb_indices = list(region.feature_block_indices or ())
    cut_set = set(split_at)
    slices: list[list[int]] = []
    current: list[int] = []
    for idx in fb_indices:
        if idx in cut_set and current:
            slices.append(current)
            current = [idx]
        else:
            current.append(idx)
    if current:
        slices.append(current)

    if len(slices) <= 1:
        # No effective cut (every cut point fell outside the FB list) -> no-op.
        return [region]

    children: list[Region] = []
    parent_id = _region_first_raw(region)
    for slice_idx, fb_slice in enumerate(slices):
        child_fb = tuple(fb_slice)
        payload = dict(region.payload or {})
        candidate = dataclasses.replace(region, feature_block_indices=child_fb)
        payload["text"] = _joined_source_text(candidate, feature_blocks)
        # Restrict a payload page list to this slice's pages (if tracked).
        child_pages = _region_pages(candidate, feature_blocks)
        if (region.payload or {}).get("pages") is not None and child_pages:
            payload["pages"] = sorted(child_pages)
        breadcrumb = {
            "op": "split",
            "source_ids": [parent_id],
            "child_index": slice_idx,
            "conservation_verified": True,
        }
        if subtype != "merge":
            breadcrumb["subtype"] = subtype
        payload["resegment"] = breadcrumb
        children.append(
            dataclasses.replace(
                region, feature_block_indices=child_fb, payload=payload
            )
        )
    return children


def apply_resegment(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    ops: list[ResegmentOp],
) -> list[Region]:
    """Apply a NON-OVERLAPPING op list to the region list, document order.

    MERGE replaces its contiguous run of regions with one merged region;
    SPLIT replaces its single region with K children. Ops are keyed on the
    INPUT region index; an op touching a region another op already consumed
    is skipped (defensive — the detectors emit non-overlapping ops, the LLM
    layer is re-validated per op). Returns a NEW region list (input untouched).
    """
    n = len(regions)
    # Map each input region index -> the op (if any) that owns it.
    merge_by_start: dict[int, ResegmentOp] = {}
    consumed_by_merge: set[int] = set()
    split_by_index: dict[int, ResegmentOp] = {}
    for op in ops:
        if op.op == "merge":
            run = op.region_indices
            if len(run) < 2 or any(r in consumed_by_merge for r in run):
                continue
            if any(r in split_by_index for r in run):
                continue
            merge_by_start[run[0]] = op
            consumed_by_merge.update(run)
        elif op.op == "split":
            idx = op.region_indices[0] if op.region_indices else -1
            if idx in consumed_by_merge or idx in split_by_index:
                continue
            split_by_index[idx] = op

    out: list[Region] = []
    i = 0
    while i < n:
        if i in merge_by_start:
            op = merge_by_start[i]
            run_regions = [regions[k] for k in op.region_indices]
            out.append(_merged_region(run_regions, feature_blocks, subtype=op.subtype))
            i = op.region_indices[-1] + 1
            continue
        if i in split_by_index:
            op = split_by_index[i]
            out.extend(
                _split_children(
                    regions[i], op.split_at, feature_blocks, subtype=op.subtype
                )
            )
            i += 1
            continue
        out.append(regions[i])
        i += 1
    return out


# ---------------------------------------------------------------------------
# Phase 9 — decision-capture audit-row build (rides the block_resegment enum).
# ---------------------------------------------------------------------------


def build_resegment_audit_rows(ops: list[ResegmentOp]) -> list[dict[str, Any]]:
    """Build the ``result['block_resegment']`` audit rows from the applied ops.

    Single SoT for the cascade's audit section (``cascade.py`` consumes this).
    A cross-kind pedagogical-unit op (``subtype=='regroup'``) records
    ``op='regroup'`` + the merged unit's ``semantic_class`` + ``regions_folded``
    (= len(region_indices) - 1) so the (bridge-owned, still-unwired)
    ``block_resegment`` DecisionCapture can interpolate regroup counts/classes
    into a dynamic, replayable rationale — with NO ``decision_event`` schema
    change (``block_resegment`` is already in the enum). A same-kind op (subtype
    default ``'merge'``) keeps ``op=op.op`` and the original 4-key row VERBATIM,
    so a regroup-flag-off run (no regroup op exists) is BYTE-IDENTICAL to the
    pre-Phase-9 audit. **Honest gap:** no DecisionCapture EVENT fires for a
    regroup until the bridge TODO (``cascade.py``) is wired — the regroup is a
    DETERMINISTIC transform (mirrors the chunker / reading-order
    deterministic-transform posture, which also emit no capture), not an LLM
    call site; the row only enriches the audit the bridge will consume."""
    rows: list[dict[str, Any]] = []
    for op in ops:
        is_regroup = op.subtype == "regroup"
        is_fused_title = op.subtype == "fused_title"
        row = {
            "op": "regroup" if is_regroup else op.op,
            "source_ids": list(op.source_ids),
            "origin": op.origin,
            "conservation_verified": True,
        }
        if is_regroup:
            row["semantic_class"] = op.semantic_class
            row["regions_folded"] = max(0, len(op.region_indices) - 1)
        elif is_fused_title:
            # A fused-title split records op='split' + subtype so the (still
            # unwired) block_resegment DecisionCapture can interpolate a
            # fused-title recovery count; mirrors the regroup-only enrichment,
            # so a flag-off run (no fused-title op) is BYTE-IDENTICAL.
            row["subtype"] = "fused_title"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Phase 10 — verifier-drivable regroup seam (second-pass section_no_body fix).
# ---------------------------------------------------------------------------


def apply_proposed_regroups(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    proposed_runs: list[Sequence[int]],
) -> tuple[list[Region], list[ResegmentOp]]:
    """Apply EXTERNALLY-proposed pedagogical-unit regroup runs through the SAME
    ``apply_resegment`` + R-PART/token-conservation fail-closed gates the
    deterministic detector rides (Phase 10 seam).

    The second-pass verify-refine plan's ``section_no_body`` /
    ``example_misordered_from_body`` modes were DETECT-ONLY for lack of a
    MERGE/regroup op in the per-block vocabulary; this is that op channel. A
    proposer (the verifier) passes explicit ``region_indices`` runs (a flagged
    unit = heading/label + its real downstream body) and they are re-validated
    HERE — a hallucinated / over-broad / non-contiguous run is DROPPED PER-OP by
    the conservation gates (e.g. a non-contiguous run makes ``apply_resegment``
    skip the in-between regions -> the FB multiset shrinks -> R-PART raises ->
    that op is dropped), mirroring the ``SEMANTIK_BLOCK_RESEGMENT_LLM`` per-op
    re-validation: a bad op is dropped individually, the good ops + the input
    survive. Does NOT wire the verifier itself (that is the second-pass plan's
    build); inert without a driver — an empty ``proposed_runs`` returns the
    input unchanged + an EMPTY op list (byte-stable). NEVER raises.

    Returns ``(out_regions, applied_ops)``; ``applied_ops`` is the subset that
    survived the gates (``[]`` when none did or none were proposed)."""
    candidate_ops: list[ResegmentOp] = []
    for raw_run in proposed_runs or ():
        run = [int(x) for x in raw_run]
        if len(run) < 2:
            continue  # a merge needs >=2 regions
        if any(not (0 <= r < len(regions)) for r in run):
            continue  # out-of-range proposal -> drop
        # An external proposer owns run[0]=anchor ordering; build via the shared
        # low-level factory (no anchor-pedagogy assert — graceful drop, not a
        # raise) and tag origin='llm' (a non-deterministic proposer).
        candidate_ops.append(_build_regroup_op(run, regions, origin="llm"))
    if not candidate_ops:
        return regions, []

    # Per-op re-validation: keep only ops that survive BOTH conservation gates
    # individually (a hallucinated run cannot ship; the good ones still apply).
    valid_ops: list[ResegmentOp] = []
    for op in candidate_ops:
        try:
            out = apply_resegment(regions, feature_blocks, [op])
            assert_partition_conservation(regions, out)
            assert_token_conservation(regions, out, feature_blocks)
        except (PartitionConservationError, TokenConservationError) as exc:
            logger.warning(
                "verifier-proposed regroup op rejected (%s) -> dropped", exc
            )
            continue
        valid_ops.append(op)
    if not valid_ops:
        return regions, []

    # Apply the surviving set as a whole and re-gate (defensive: overlapping
    # proposed runs collapse via apply_resegment's first-registered-wins guard).
    try:
        out = apply_resegment(regions, feature_blocks, valid_ops)
        assert_partition_conservation(regions, out)
        assert_token_conservation(regions, out, feature_blocks)
    except (PartitionConservationError, TokenConservationError) as exc:
        logger.warning(
            "verifier-proposed regroup set rejected (%s) -> input preserved",
            exc,
        )
        return regions, []
    return out, list(valid_ops)


# ---------------------------------------------------------------------------
# Optional Stage-3 LLM op-proposal layer.
# ---------------------------------------------------------------------------


def _llm_propose_ops(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    runtime: Any,
) -> list[ResegmentOp]:
    """Propose EXTRA resegment ops via the injected endpoint runtime.

    Best-effort: builds one prompt (the region digest), fans it through
    ``runtime.generate_batch(..., fail_soft=True)``, tolerantly parses each
    completion (reusing ``reviewer._extract_verdict_obj``), and returns the
    proposed ops tagged ``origin='llm'``. Every proposed op is re-validated by
    the SAME gates downstream (``resegment_blocks`` re-asserts conservation
    per-op). An endpoint-down runtime / parse miss returns [] (the
    deterministic result stands). NEVER raises."""
    from .block_resegment_prompt import build_resegment_request

    try:
        prompt = build_resegment_request(regions, feature_blocks)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("block-resegment LLM prompt build failed: %s", exc)
        return []

    try:
        try:
            completions = runtime.generate_batch(
                [prompt], max_tokens=1024, fail_soft=True
            )
        except TypeError:
            completions = runtime.generate_batch([prompt], max_tokens=1024)
    except Exception as exc:
        logger.warning(
            "block-resegment LLM endpoint failed (%s) -> keeping deterministic "
            "result",
            exc,
        )
        return []

    completions = list(completions or [])
    if not completions or completions[0] is None:
        return []
    obj = _extract_verdict_obj(completions[0] if isinstance(completions[0], str) else "")
    if not obj:
        return []
    return _parse_llm_ops(obj, len(regions))


def _parse_llm_ops(obj: dict[str, Any], n_regions: int) -> list[ResegmentOp]:
    """Parse the LLM verdict object's ``ops`` array into ResegmentOps.

    Tolerant: drops any malformed / out-of-range op (it is re-validated by the
    conservation gates anyway). Returns [] when ``ops`` is absent/empty."""
    raw_ops = obj.get("ops")
    if not isinstance(raw_ops, list):
        return []
    out: list[ResegmentOp] = []
    for raw in raw_ops:
        if not isinstance(raw, dict):
            continue
        op_type = str(raw.get("op") or "").strip().lower()
        if op_type == "merge":
            indices = raw.get("region_indices")
            if not isinstance(indices, list) or len(indices) < 2:
                continue
            try:
                run = tuple(int(x) for x in indices)
            except (TypeError, ValueError):
                continue
            if any(not (0 <= r < n_regions) for r in run):
                continue
            if list(run) != sorted(run) or any(
                run[k] != run[0] + k for k in range(len(run))
            ):
                continue  # must be a contiguous ascending run
            out.append(
                ResegmentOp(op="merge", region_indices=run, origin="llm")
            )
        elif op_type == "split":
            indices = raw.get("region_indices")
            split_at = raw.get("split_at")
            if not isinstance(indices, list) or len(indices) != 1:
                continue
            if not isinstance(split_at, list) or not split_at:
                continue
            try:
                idx = int(indices[0])
                cuts = tuple(int(x) for x in split_at)
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < n_regions):
                continue
            out.append(
                ResegmentOp(
                    op="split",
                    region_indices=(idx,),
                    split_at=cuts,
                    origin="llm",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------


def resegment_blocks(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    state: Any,
    *,
    runtime: Any = None,
) -> tuple[list[Region], list[ResegmentOp]]:
    """Run the Stage-5e block JOIN/SPLIT pass over the formed region list.

    Flow:

      1. Deterministic detectors (``_detect_merges`` + ``_detect_splits``)
         read the MergeOrSplit council head off ``state`` + the deterministic
         text cues (pedagogical-label regex, cross-page continuation).
      2. Optional LLM layer (``resolve_block_resegment_llm_mode`` + a runtime)
         proposes EXTRA ops; each is appended to the deterministic ops.
      3. ``apply_resegment`` re-partitions the region list.
      4. R-PART gates: ``assert_partition_conservation`` (multiset + order) +
         ``reviewer.assert_token_conservation`` (text). ANY raise -> FAIL
         CLOSED: return the INPUT regions unchanged + an EMPTY op list.

    Returns ``(out_regions, applied_ops)``. ``applied_ops`` is [] on a
    fail-closed revert. NO model/GPU is loaded; ``runtime`` is injected (only
    consulted when the LLM layer is on)."""
    if not regions:
        return regions, []

    # Per-flag gating INSIDE the driver (Phase 6): each flag owns ONLY its own
    # detector(s). The cascade gate is widened to fire Stage-5e on EITHER flag,
    # so gating here is what keeps the flags ORTHOGONAL — a regroup-on /
    # block-resegment-off run must emit ZERO same-kind merge/split ops, and a
    # block-resegment-on / regroup-off run must emit no regroup op.
    det_ops: list[ResegmentOp] = []

    # Regroup ops are emitted FIRST (regroup-FIRST precedence). apply_resegment's
    # overlap guard is first-registered-wins + whole-op-drop, so prepending the
    # regroup makes an overlapping same-kind merge OR an interior-label split the
    # op that is DROPPED — never the regroup (which would leave the label-only
    # box unfixed, the exact defect this feature exists to fix). Guarded on the
    # reading-order fix: only then is a unit's label -> body contiguous for a
    # clean forward-adjacency merge; with it OFF this is a guarded no-op so the
    # regroup can never fuse a wrong (non-adjacent) body. The detector ALSO
    # self-gates on resolve_unit_regroup_mode() (defense in depth).
    if resolve_unit_regroup_mode() and resolve_reading_order_fix():
        det_ops += _detect_unit_merges(regions, feature_blocks, state)

    # Fused-section-title SPLIT arm (lane B). Precedence: regroup > fused-title >
    # same-kind. Appended AFTER the regroup detector (so a regroup consuming the
    # region wins via apply_resegment's first-registered overlap guard) but
    # BEFORE the same-kind detectors (so a fused-title split beats a colliding
    # same-kind split/merge on the same region — the label-only box this feature
    # exists to fix). NOTE: this refines the design's literal
    # ``det_ops = fused + det_ops`` prepend, which would place fused-title BEFORE
    # regroup; the design's own §4 precedence statement (regroup > fused-title >
    # same-kind) is authoritative, and append-in-precedence-order realises it.
    # Each op is re-validated INDIVIDUALLY (structural belt + the same R-PART +
    # token-conservation gates apply_proposed_regroups rides) before it joins
    # det_ops, so a wrong split is dropped alone and never poisons the set.
    if resolve_split_fused_section_titles_mode():
        fused_ops, fused_intra = _detect_fused_title_splits(regions, feature_blocks)
        if fused_intra:
            logger.info(
                "block-resegment fused-title: %d intra-FB fusion(s) unsplittable "
                "(title+prose inside one FeatureBlock; FB is the atom) -> "
                "detect-only miss",
                fused_intra,
            )
        for op in fused_ops:
            if not _fused_title_op_valid(op, regions):
                logger.warning(
                    "fused-title split op malformed (%r) -> dropped", op
                )
                continue
            try:
                probe = apply_resegment(regions, feature_blocks, [op])
                assert_partition_conservation(regions, probe)
                assert_token_conservation(regions, probe, feature_blocks)
            except (PartitionConservationError, TokenConservationError) as exc:
                logger.warning(
                    "fused-title split op rejected (%s) -> dropped", exc
                )
                continue
            det_ops.append(op)

    if resolve_block_resegment_mode():
        det_ops += _detect_merges(regions, feature_blocks, state)
        det_ops += _detect_splits(regions, feature_blocks, state)

    # Optional LLM op-proposal layer. Each proposed op is re-validated below by
    # the SAME conservation gates (we validate the deterministic+LLM op set as
    # a whole; if the union fails, fall back to deterministic-only, then to the
    # input). This keeps an endpoint-down LLM from regressing the result.
    llm_ops: list[ResegmentOp] = []
    if resolve_block_resegment_llm_mode() and runtime is not None:
        llm_ops = _llm_propose_ops(regions, feature_blocks, runtime)

    # Try the full op set, then deterministic-only, then no-op. Each candidate
    # is gated; the first that passes both conservation gates ships.
    for candidate_ops in ([*det_ops, *llm_ops], det_ops):
        if not candidate_ops:
            continue
        try:
            out = apply_resegment(regions, feature_blocks, candidate_ops)
            assert_partition_conservation(regions, out)
            assert_token_conservation(regions, out, feature_blocks)
        except (PartitionConservationError, TokenConservationError) as exc:
            logger.warning(
                "block-resegment op set rejected (%s) -> trying a narrower set",
                exc,
            )
            continue
        return out, list(candidate_ops)

    # No ops, or every candidate set failed a gate -> byte-stable pass-through.
    return regions, []


__all__ = [
    "PartitionConservationError",
    "ResegmentOp",
    "apply_resegment",
    "apply_proposed_regroups",
    "assert_partition_conservation",
    "build_resegment_audit_rows",
    "is_passthrough_region",
    "resegment_blocks",
    "resolve_block_resegment_llm_mode",
    "resolve_block_resegment_mode",
    "resolve_split_fused_section_titles_mode",
    "resolve_unit_regroup_mode",
    "resolve_unit_regroup_table_mode",
]
