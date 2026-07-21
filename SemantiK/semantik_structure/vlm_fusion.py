"""Deterministic P1 fusion of a VLM markdown source onto Tesseract line blocks.

Phase 4 of the scan-conversion plan (``plans/scan-conversion-improvements-2026-07.md``)
adds a VLM page transcription as a FOURTH extraction source and fuses it with
Tesseract *before the BERTs* — every measured OCR loss (inline-radical garble,
wrapped-line interleave, fused titles) is extraction-level, so the repair
belongs at the source, not post-hoc. This module is the load-bearing fusion
engine: **it is deterministic code, not a model.**

Contract (settled 2026-07-03, owner design discussion):

* The Tesseract line blocks are the PDF side; the VLM markdown lines are the
  gold side. We reuse the ``data.structure_align`` DP aligner's PUBLIC API
  (:func:`align_blocks` + :func:`sim`) — the same 5-move (MATCH / SPLIT /
  MERGE / GOLD_GAP / PDF_GAP) lossless global alignment that the structure
  training-data builder uses — rather than a lighter greedy line matcher,
  because a VLM logical line routinely MERGEs *k* wrapped OCR print-lines into
  one markdown line, which a monotonic greedy matcher cannot express. The
  aligner is deliberately input-agnostic: a Tesseract line maps to
  :class:`~data.structure_align.FBView` (it carries a real bbox), a VLM line to
  :class:`~data.structure_align.GoldView` with empty ``tag``/``role``.
* **Aligned (MATCH / SPLIT / MERGE)** → the VLM text lands on the Tesseract
  bbox union of the consumed Tesseract lines (bbox, font, confidence, and the
  20-dim-layout-feeding geometry all survive for the council), stamped
  ``fusion='vlm+tesseract'`` + ``fusion_sim=<DP confidence>``.
* **GOLD_GAP (VLM-only)** → a flagged low-confidence insert: bbox interpolated
  from aligned neighbors IN TESSERACT IMAGE-PIXEL SPACE (x = union of prev/next
  neighbor x-ranges; y strictly between ``prev.bottom`` and ``next.top`` so
  ``_merge_page``'s ``(y0, x0)`` sort lands it at the intended position),
  ``confidence=0.30`` sentinel, ``fusion='vlm-only-flagged'`` — the
  anti-hallucination tripwire is STRUCTURAL.
* **PDF_GAP (tesseract-only)** → kept byte-verbatim, ``fusion='tesseract-only'``
  (VLM-drop protection).

Two deterministic mitigations for the highest-value lines (dense math), where
Tesseract garble (``Vab = Va-Vb``) and the VLM's LaTeX
(``\\sqrt{ab}=\\sqrt{a}\\cdot\\sqrt{b}``) score near-zero on raw token
similarity and would otherwise land as adjacent gap pairs:

1. A fusion-side sim PRE-NORMALIZATION applied ONLY for alignment scoring —
   strip LaTeX control sequences / braces / ``$`` from the VLM line and apply
   the known OCR confusable folds (``V``→√-strip, ``l``/``1``, ``O``/``0``) to
   both sides before the aligner scores them. This is fusion-local by design
   and MUST NOT migrate into ``data.structure_align`` (that would silently
   change the training-data aligner). The fused block's OUTPUT text is a
   separate transform (:func:`_strip_markdown_structure`) that keeps LaTeX.
2. A gap-pairing RESCUE post-pass — an adjacent (PDF_GAP-run, GOLD_GAP-run)
   pair bounded by aligned anchors on BOTH sides is fused positionally (the
   *k* Tesseract lines' union bbox gets the VLM run's text). Pages where the
   rescue still leaves large gap runs surface in the per-page divergence score
   — the tripwire.

Per-block fusion provenance rides the NEW ``fusion`` key, NOT the
``provenance`` field: ``_merge_page`` forces ``provenance=['tesseract']`` on
the tesseract branch, so ``features.py``'s ``prov=tesseract`` council-input
marker is unchanged and the ``fusion`` key never reaches the classifier input.

Math text-form contract: the VLM's inline LaTeX is kept VERBATIM as text
characters in the fused block (``$``-delimiters preserved); only markdown
STRUCTURAL syntax (``#`` heading prefixes, list markers, ``**bold**`` wrappers)
is stripped. Downstream treats LaTeX as prose text — chunk and gold text WILL
carry LaTeX where OCR had garble. This is an accepted, flag-gated
representation change (flag off ⇒ byte-identical, zero LaTeX anywhere).

Pure module: stdlib only at import; ``numpy`` / the aligner are imported
lazily inside :func:`fuse_page` so ``resolve_vlm_fusion_mode`` (the flag
resolver + cache-salt path) stays a cheap env read.
"""

from __future__ import annotations

import os
import re

__all__ = [
    "SEMANTIK_VLM_FUSION_ENV",
    "SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV",
    "SEMANTIK_VLM_ORDER_DIVERGENCE_ENV",
    "SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV",
    "SEMANTIK_VLM_COLLAPSE_REPETITION_ENV",
    "SEMANTIK_FUSION_KEEP_ORDERED_MARKERS_ENV",
    "SEMANTIK_FUSION_LINE_UNITS_ENV",
    "resolve_vlm_fusion_mode",
    "resolve_vlm_order_authoritative_mode",
    "resolve_vlm_order_divergence_floor",
    "resolve_drop_garbage_tails_mode",
    "resolve_collapse_repetition_mode",
    "resolve_keep_ordered_markers_mode",
    "resolve_fusion_line_units_mode",
    "fuse_page",
    "PAGE_VLM_CONTRACT",
]

# ---------------------------------------------------------------------------
# Flag resolver (SEMANTIK_VLM_FUSION) — parse-with-fallback, default OFF.
# ---------------------------------------------------------------------------

SEMANTIK_VLM_FUSION_ENV = "SEMANTIK_VLM_FUSION"
_TRUTHY = {"1", "true", "yes", "on"}
# Explicit falsey set for the default-ON garbage-tail drop (mirrors
# ``vlm_furniture._FALSEY``): only these disable an otherwise-on switch.
_FALSEY = {"0", "false", "no", "off"}

# ---------------------------------------------------------------------------
# VLM-authoritative reading-order (dense multi-column repair).
# ---------------------------------------------------------------------------
#
# On a clean single-column page the Tesseract line geometry is a faithful
# reading order, so the fused blocks ride their real bboxes and the downstream
# ``_merge_page`` ``(y0, x0)`` sort reconstructs document order. On a DENSE
# MULTI-COLUMN page (answer grids, 3-column exercise banks) Tesseract's line
# segmentation fractures across the gutter and the DP alignment degrades into
# many gap lines — the fused list carries poorly-interpolated VLM-only inserts
# and orphaned tesseract-only tails, and the ``_merge_page`` bbox re-sort then
# interleaves them across columns (the clean VLM order is LOST: numbers drop /
# orphan, garbage tails append). The per-page ``divergence`` already measures
# exactly this (gap lines / total) — when it is high, Tesseract geometry is
# unreliable, so the VLM's known-good LINEAR order is made authoritative:
# emit the VLM-bearing blocks in VLM (gold) order, DROP the unmatched
# tesseract-only garbage, and overwrite each block's bbox with a synthetic
# MONOTONIC single-column band so the downstream ``(y0, x0)`` re-sort (and the
# region-level ``min(feature_block_indices)`` sort) preserve VLM order instead
# of re-scrambling it. Low-divergence (clean) pages fall below the threshold →
# byte-identical to the legacy path.
SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV = "SEMANTIK_VLM_ORDER_AUTHORITATIVE"
SEMANTIK_VLM_ORDER_DIVERGENCE_ENV = "SEMANTIK_VLM_ORDER_DIVERGENCE"
# Default-ON-within-fusion garbage-tail drop (see resolve_drop_garbage_tails_mode).
SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV = "SEMANTIK_VLM_DROP_GARBAGE_TAILS"
# Default-ON-within-fusion degenerate-repetition collapse (see
# resolve_collapse_repetition_mode).
SEMANTIK_VLM_COLLAPSE_REPETITION_ENV = "SEMANTIK_VLM_COLLAPSE_REPETITION"
# Default-ON-within-fusion exercise/answer ORDERED-MARKER preservation (Fix A —
# a line-leading "N."/"N)" in a textbook is a CONTENT exercise/answer number,
# not a markdown list marker; see resolve_keep_ordered_markers_mode).
SEMANTIK_FUSION_KEEP_ORDERED_MARKERS_ENV = "SEMANTIK_FUSION_KEEP_ORDERED_MARKERS"
# Default-ON-within-fusion per-VLM-line block granularity (Fix B — a SPLIT
# unit / gap-rescue run / no-tesseract page emits ONE fused block PER VLM line
# instead of one space-joined mega-block; see resolve_fusion_line_units_mode).
SEMANTIK_FUSION_LINE_UNITS_ENV = "SEMANTIK_FUSION_LINE_UNITS"
# Divergence at/above which Tesseract order is treated as unreliable and the
# VLM linear order is made authoritative. A clean single-column page aligns
# almost every line (divergence well under 0.1); a fractured multi-column grid
# runs far higher. 0.25 sits comfortably between the two regimes.
_DEFAULT_ORDER_DIVERGENCE = 0.25
# Synthetic single-column band height when the page height is unknown.
_SYNTH_LINE_H = 12.0


def resolve_vlm_order_authoritative_mode() -> bool:
    """Whether the VLM-authoritative reading-order repair may fire.

    **Default OFF** (opt-in, mirroring the rest of the ``SEMANTIK_VLM_*``
    lane — the whole VLM fusion path is opt-in): parse-with-fallback truthy
    set (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive); unset /
    blank / falsey / garbage → off. When ON it is STILL gated per-page by
    :func:`resolve_vlm_order_divergence_floor`, so it is a NO-OP on clean
    low-divergence pages even when enabled — only a genuinely fractured
    multi-column page (high divergence) is re-ordered. The whole path is inert
    unless ``SEMANTIK_VLM_FUSION`` is on (no ``fuse_page`` call otherwise).
    Read at CALL time so tests / Docker can flip it without a re-import.
    """
    return os.environ.get(
        SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, ""
    ).strip().lower() in _TRUTHY


def resolve_vlm_order_divergence_floor() -> float:
    """Per-page divergence at/above which VLM order becomes authoritative.

    Parse-with-fallback float in ``[0.0, 1.0]``; blank / non-float / NaN /
    Inf / out-of-range → :data:`_DEFAULT_ORDER_DIVERGENCE` (0.25). Read at
    CALL time so tests / Docker can tune it without a re-import.
    """
    raw = os.environ.get(SEMANTIK_VLM_ORDER_DIVERGENCE_ENV, "").strip()
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_ORDER_DIVERGENCE
    if not _finite(val) or not (0.0 <= val <= 1.0):
        return _DEFAULT_ORDER_DIVERGENCE
    return val


def resolve_vlm_fusion_mode() -> bool:
    """Whether deterministic VLM↔Tesseract fusion runs (default OFF).

    Parse-with-fallback, truthy-set (``1`` / ``true`` / ``yes`` / ``on``,
    case-insensitive); unset / blank / falsey / garbage → off. Read at CALL
    time (mirrors ``extract_shared.resolve_ocr_render_scale``) so tests /
    Docker can flip it without a re-import.

    The flag is a NO-OP without a populated ``page['vlm']`` source (P0 =
    ``SEMANTIK_VLM_EXTRACT``): fusion runs at the extract seam only when this
    is truthy AND both a tesseract source and a P0 ``page['vlm']`` dict exist
    (mirrors the ``SEMANTIK_SEMANTIC_CLASS`` requires-``SEMANTIK_BLOCK_REVIEW``
    pattern). Separate from P0's flag so VLM extraction can run in measure-only
    mode (source cached, merged output unchanged).
    """
    return os.environ.get(SEMANTIK_VLM_FUSION_ENV, "").strip().lower() in _TRUTHY


def resolve_drop_garbage_tails_mode() -> bool:
    """Whether unrescued tesseract-only OCR-garbage blocks are dropped (default ON).

    Reads ``SEMANTIK_VLM_DROP_GARBAGE_TAILS`` with the **default-ON**
    parse posture of :func:`vlm_furniture.resolve_strip_furniture_mode`: only an
    explicit falsey value (``0`` / ``false`` / ``no`` / ``off``,
    case-insensitive) disables it; unset / truthy / garbage keeps it on.

    It is a **NO-OP without VLM fusion** (the only caller,
    :func:`_reconstruct`'s tesseract-only loop, runs on the fusion path — itself
    off by default), so the flag-off pipeline is byte-identical; ``=0`` is the
    operator escape hatch (a run where an unrescued tesseract-only block should
    stay byte-verbatim even when it looks like OCR noise). Read at CALL time
    (mirrors :func:`resolve_vlm_fusion_mode`) so tests / Docker can flip it
    without a re-import.
    """
    raw = os.environ.get(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


def resolve_collapse_repetition_mode() -> bool:
    """Whether a pathological repeated-token/n-gram run is collapsed (default ON).

    Reads ``SEMANTIK_VLM_COLLAPSE_REPETITION`` with the **default-ON** parse
    posture of :func:`resolve_drop_garbage_tails_mode`: only an explicit falsey
    value (``0`` / ``false`` / ``no`` / ``off``, case-insensitive) disables it;
    unset / truthy / garbage keeps it on.

    It is a **NO-OP without VLM fusion** (the only caller is
    :func:`fuse_page`'s post-reconstruction pass, itself on the fusion path —
    off by default) AND a no-op on non-pathological text (a fused block whose
    text has no degenerate run is byte-identical), so the flag-off / no-fusion /
    clean-input pipeline is byte-identical; ``=0`` is the operator escape hatch
    (keep a fused block byte-verbatim even when it carries a degenerate run).
    Read at CALL time (mirrors :func:`resolve_drop_garbage_tails_mode`) so
    tests / Docker can flip it without a re-import.
    """
    raw = os.environ.get(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


def resolve_keep_ordered_markers_mode() -> bool:
    """Whether a line-leading ordered marker ("N." / "N)") is KEPT as content (default ON).

    Fix A for the silent extraction-drop of textbook exercise/answer numbers.
    Reads ``SEMANTIK_FUSION_KEEP_ORDERED_MARKERS`` with the **default-ON** parse
    posture of :func:`resolve_drop_garbage_tails_mode`: only an explicit falsey
    value (``0`` / ``false`` / ``no`` / ``off``, case-insensitive) disables it;
    unset / truthy / garbage keeps it on.

    In a textbook exercise/answer bank the VLM emits one exercise per line
    ("425. $\\frac{6}{13}+\\frac{5}{13}$"), and the legacy single-marker strip
    branch in :func:`_strip_markdown_structure` deleted the leading "425." as a
    markdown list marker — silently losing the exercise IDENTIFIER (38 numbers
    lost on the audited scanned-algebra-textbook ch01). When this mode is ON the
    single-marker strip is SKIPPED, so the number survives as content; genuine
    markdown-list semantics still reach P2 via the untouched raw ``vlm_md`` hint
    channel. **NO-OP without VLM fusion** (:func:`_strip_markdown_structure` is
    only reached from :func:`fuse_page`), so the flag-off pipeline is
    byte-identical; ``=0`` restores the legacy strip exactly. Read at CALL time.
    """
    raw = os.environ.get(SEMANTIK_FUSION_KEEP_ORDERED_MARKERS_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


def resolve_fusion_line_units_mode() -> bool:
    """Whether a multi-VLM-line fused unit emits ONE block PER line (default ON).

    Fix B for the unit-boundary destruction that glued a worked example's label,
    stem, "Solution", table, and TRY IT lines into single mega-blocks (audited
    scanned-algebra-textbook ch01 run-on exercise blobs). Reads
    ``SEMANTIK_FUSION_LINE_UNITS`` with the **default-ON** parse posture of
    :func:`resolve_collapse_repetition_mode`: only an explicit falsey value
    (``0`` / ``false`` / ``no`` / ``off``) disables it; unset / truthy / garbage
    keeps it on.

    When ON: a SPLIT aligned unit (``len(vlm) > 1``), an unbounded gap-pairing
    rescue run, and a no-tesseract page each emit one fused block PER VLM line
    over even y-bands of the tesseract union (instead of one ``_join_vlm``
    mega-block), with ``vlm_coverage='whole_block'`` (now genuinely correct —
    each block IS one whole VLM line). **NO-OP without VLM fusion** (the callers
    are all inside :func:`_reconstruct` / :func:`fuse_page`), so the flag-off /
    no-fusion pipeline is byte-identical; ``=0`` restores the legacy join/drop
    exactly. Read at CALL time.
    """
    raw = os.environ.get(SEMANTIK_FUSION_LINE_UNITS_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


# ---------------------------------------------------------------------------
# Domain-agnostic degenerate-repetition tripwire.
# ---------------------------------------------------------------------------
#
# The VLM (or the VLM×Tesseract fusion) occasionally transcribes a visual
# MANIPULATIVE / diagram figure — a row of algebra tiles, a tally grid, a ruler
# scale — as an endless literal repetition of a tiny token cycle
# ("x + x + x + …", "0 0 0 0 …", "- - - - …"). Because the fused text ships
# verbatim as prose (`raw_text` → the accessible HTML `<p>` AND the retrieval
# chunk), one such figure produced an 8k-word mega-chunk of pure retrieval
# poison (2026-07 scanned algebra textbook ch01 scan audit — a single region
# carried ~4,047 repetitions of the `x +` cycle).
#
# This tripwire is DOMAIN-AGNOSTIC (it keys on the STRUCTURE — a repeated UNIT,
# never on any specific token like `x`/`+` or any specific sentence) and
# CONSERVATIVE. It has TWO arms, both scanned in ONE left-to-right pass over the
# whitespace-token stream (they compose — a document carrying both a token loop
# and a sentence loop collapses both):
#
#   * SHORT-TOKEN arm — an unambiguous glyph run: a unit of
#     ≤`_REPEAT_MAX_TOKEN_CHARS`-char tokens, period ≤`_REPEAT_MAX_PERIOD`,
#     repeating STRICTLY MORE than `_REPEAT_MIN_CYCLES` (30) consecutive times.
#     This is the original arm ("x + x + …", "0 0 0 …").
#   * SENTENCE / MULTI-WORD arm — an unambiguous SENTENCE loop: a
#     clause/sentence-shaped unit of `_REPEAT_SENTENCE_MIN_WORDS`..
#     `_REPEAT_SENTENCE_MAX_WORDS` (~25) tokens repeating at least
#     `_REPEAT_MIN_SENTENCE_CYCLES` (6) consecutive times. This closes the
#     sentence-granularity blind spot the token arm misses (the VLM looping a
#     whole SENTENCE ~33× into a ~15 KB blob that the chunker then windowed into
#     byte-identical chunks — 2026-07 scanned algebra textbook ch01 re-audit).
#
# Both arms err toward KEEPING. A repeated SENTENCE is a far stronger degeneracy
# signal than a repeated word, so its cycle floor (6) is small — but to stay
# false-positive-proof the sentence unit must be genuinely clause-shaped: it ends
# in terminal punctuation (`.`/`!`/`?`) AND NO interior token ends in terminal
# punctuation. That interior rule is the load-bearing guard — it rejects a real
# numbered list (`1. 2. 3. 4.`), a table of repeated SHORT labels
# (`Pending. Pending. …`), and abbreviation-dense refrains, because each of
# those repeats a unit whose interior tokens themselves end in `.`. A real
# short repeated list, a 2×-incidental phrase repeat (well under the 6-cycle
# floor), a poem/refrain (separated by verses → not CONSECUTIVE), a table row,
# and normal prose are all left byte-identical.

# Max token length (chars) for a token to count as part of a short repeating
# unit — a genuinely degenerate glyph run is tiny tokens ("x", "+", "-", "."),
# never long words. Long-token repetition is left untouched.
_REPEAT_MAX_TOKEN_CHARS = 4
# Max period (in whitespace tokens) of the SHORT-TOKEN repeating unit scanned.
# Covers a single glyph (period 1: "0 0 0 …") through a short n-gram cycle
# (period 2: "x + x + …") up to a small bound. Kept small so a genuinely
# repetitive but meaningful clause structure is never mistaken for a glyph run.
_REPEAT_MAX_PERIOD = 4
# Consecutive-cycle count STRICTLY ABOVE which a SHORT-TOKEN run is unambiguously
# degenerate. Generous (30) so a real short repeated list never trips it.
_REPEAT_MIN_CYCLES = 30
# Cycles of the unit kept as a bounded representative before the count marker
# (shared by both arms).
_REPEAT_KEEP_CYCLES = 3

# --- SENTENCE / multi-word arm calibration ---------------------------------
# Min tokens for a unit to be considered a clause/sentence (below this a repeat
# is a short label/refrain, kept). Chosen conservatively — a genuine degenerate
# sentence loop is a substantial multi-word clause, and requiring ≥4 words keeps
# repeated 1–3 word table/column labels byte-identical.  # calibration
_REPEAT_SENTENCE_MIN_WORDS = 4
# Max tokens (~words) of a sentence unit scanned (K). A sentence longer than
# this is not a "loop" we collapse; keeps the per-position scan bounded.  # calibration
_REPEAT_SENTENCE_MAX_WORDS = 25
# Minimum CONSECUTIVE identical sentence repeats (M) to collapse. Small (6)
# because a verbatim ≥4-word clause repeated 6× back-to-back is already
# pathological — legitimate prose / refrains essentially never do this
# consecutively — yet far above an incidental 2× restatement.  # calibration
_REPEAT_MIN_SENTENCE_CYCLES = 6
# Terminal sentence punctuation + trailing wrappers stripped before the check.
_SENTENCE_TERMINALS = (".", "!", "?")
_SENTENCE_TRAILING = "\"'”’)]}»"
# Fast-path guard: the smallest token count either arm can possibly collapse —
# the SHORT-TOKEN arm needs > _REPEAT_MIN_CYCLES tokens (period 1), the SENTENCE
# arm needs _REPEAT_MIN_SENTENCE_CYCLES × _REPEAT_SENTENCE_MIN_WORDS tokens.
# Below the smaller of these (minus one, in spaces) no run can exist, so skip
# the tokenization entirely (byte-identical).
_REPEAT_MIN_SCAN_SPACES = min(
    _REPEAT_MIN_CYCLES,
    _REPEAT_MIN_SENTENCE_CYCLES * _REPEAT_SENTENCE_MIN_WORDS - 1,
)


def _ends_sentence(tok: str) -> bool:
    """True if ``tok`` ends a clause/sentence (terminal punct, wrappers stripped)."""
    return tok.rstrip(_SENTENCE_TRAILING).endswith(_SENTENCE_TERMINALS)


def _is_sentence_shaped(unit: list[str]) -> bool:
    """True for a clause/sentence-shaped multi-word unit (see the block comment).

    Requires ``>= _REPEAT_SENTENCE_MIN_WORDS`` tokens, the LAST token to end in
    terminal punctuation, and NO interior token to end in terminal punctuation
    (the load-bearing guard against repeated labels / numbered lists / `N.`
    grids, each of whose repeated unit carries interior terminal punctuation).
    Pure / deterministic.
    """
    if len(unit) < _REPEAT_SENTENCE_MIN_WORDS:
        return False
    if not _ends_sentence(unit[-1]):
        return False
    return not any(_ends_sentence(t) for t in unit[:-1])


def _count_cycles(toks: list[str], i: int, p: int, n: int, unit: list[str]) -> int:
    """Consecutive repeats of ``unit`` (== ``toks[i:i+p]``) starting at ``i``."""
    cycles = 1
    j = i + p
    while j + p <= n and toks[j : j + p] == unit:
        cycles += 1
        j += p
    return cycles


def _collapse_degenerate_repetition(text: str) -> str:
    """Collapse an unambiguous degenerate repeated UNIT to a bounded
    representative + a count marker.

    Domain-agnostic + conservative (see the block comment). Scans the
    whitespace-token stream left-to-right; at each position it considers BOTH
    arms and takes the highest-coverage degenerate run:

    * SHORT-TOKEN — a unit of ``<= _REPEAT_MAX_TOKEN_CHARS``-char tokens (period
      ``1.._REPEAT_MAX_PERIOD``) repeating STRICTLY MORE than
      ``_REPEAT_MIN_CYCLES`` times;
    * SENTENCE — a clause-shaped unit (``_is_sentence_shaped``, period
      ``_REPEAT_SENTENCE_MIN_WORDS.._REPEAT_SENTENCE_MAX_WORDS``) repeating at
      least ``_REPEAT_MIN_SENTENCE_CYCLES`` times.

    Each collapsed run becomes ``_REPEAT_KEEP_CYCLES`` copies of the unit
    followed by a ``[repeated N times]`` marker (N = the collapsed cycle count).
    Surrounding tokens are preserved; both arms compose in one pass.

    Returns the input string UNCHANGED (the same object) when NO degenerate run
    is present — so a clean fused block is byte-identical (the collapse only
    rebuilds pathological text, where whitespace normalization is immaterial).
    Pure / deterministic.
    """
    s = text or ""
    # Too short to hold a run under either arm → skip tokenization (byte-identical).
    if s.count(" ") < _REPEAT_MIN_SCAN_SPACES:
        return text
    toks = s.split()
    n = len(toks)
    out: list[str] = []
    i = 0
    changed = False
    while i < n:
        best: tuple[int, int] | None = None  # (period, cycles) — maximize p*cycles
        # SHORT-TOKEN arm.
        for p in range(1, _REPEAT_MAX_PERIOD + 1):
            if i + p > n:
                break
            unit = toks[i : i + p]
            if any(len(t) > _REPEAT_MAX_TOKEN_CHARS for t in unit):
                continue
            cycles = _count_cycles(toks, i, p, n, unit)
            if cycles > _REPEAT_MIN_CYCLES:
                # Prefer the run covering the most tokens (fundamental period
                # wins ties deterministically via the strict >).
                if best is None or cycles * p > best[1] * best[0]:
                    best = (p, cycles)
        # SENTENCE / multi-word arm (composes with the token arm; a longer
        # clause loop the token arm cannot express is caught here).
        for p in range(_REPEAT_SENTENCE_MIN_WORDS, _REPEAT_SENTENCE_MAX_WORDS + 1):
            if i + p > n:
                break
            unit = toks[i : i + p]
            if not _is_sentence_shaped(unit):
                continue
            cycles = _count_cycles(toks, i, p, n, unit)
            if cycles >= _REPEAT_MIN_SENTENCE_CYCLES:
                if best is None or cycles * p > best[1] * best[0]:
                    best = (p, cycles)
        if best is not None:
            p, cycles = best
            unit = toks[i : i + p]
            out.extend(unit * _REPEAT_KEEP_CYCLES)
            out.append(f"[repeated {cycles} times]")
            i += p * cycles
            changed = True
        else:
            out.append(toks[i])
            i += 1
    if not changed:
        return text
    return " ".join(out)


# ---------------------------------------------------------------------------
# Conservative OCR-garbage predicate for unrescued tesseract-only blocks.
# ---------------------------------------------------------------------------
#
# On a cleanly-VLM-transcribed scan a tesseract-only block (the DP aligner
# matched it to NO vlm line AND the gap-pairing rescue skipped it) is almost
# always OCR noise the VLM correctly omitted — BUT it can occasionally be REAL
# content the VLM dropped (the documented "VLM-drop protection" rationale). So
# this predicate is deliberately CONSERVATIVE: it drops only SHORT lines that
# are unambiguously junk-glyph fragments, and errs toward KEEPING (a
# false-negative — a bit of surviving garbage) over dropping (a false-positive
# — deleted real content). Real content lands on the VLM-aligned / rescued path,
# never here; and this predicate never even looks at a line carrying a math span.
#
# The four live-fire evidence lines it MUST catch (unrescued tesseract-only
# blocks that shipped verbatim into a scanned algebra textbook ch01 conversion):
#   "TRY Tiss ® ©"      (junk-symbol-dominated fragment)
#   "©) obi Dom -19"    (leading junk glyph + word soup)
#   "© Solution No."    (leading junk glyph + fragment)
#   "2,791 © 2,795"     (embedded junk glyph, no real word — a numeric misread)
# and must NOT catch a normal short sentence, a real math line, or a legit short
# label ("Solution").

# The "junk" glyph set (the task-specified isolated symbols almost never a
# legitimate LINE-LEADING content character): copyright/trademark/registered,
# pipe, tilde, caret, backtick, logical-not, degree. Brackets/parens are
# DELIBERATELY excluded so a real "(a) first option" / "See section 3(a)" line
# is never flagged.
_GARBAGE_JUNK_CHARS = frozenset("®™©|~^`¬°")
# A run of >= 3 consecutive ASCII letters — a "plausible word". Its ABSENCE (on
# a short junk-bearing line) is a strong garbage signal.
_PLAUSIBLE_WORD_RE = re.compile(r"[A-Za-z]{3,}")
# A math span (real math arrives via the VLM-aligned path, never tesseract-only,
# but be defensive): $...$, \(...\), \[...\].
_MATH_SPAN_RE = re.compile(r"\$[^$]+\$|\\\([^)]*\\\)|\\\[[^]]*\\]")
# Junk-glyph share of non-space characters at/above which a short line is judged
# junk-dominated even when it carries a plausible word ("TRY Tiss ® ©").
_GARBAGE_JUNK_RATIO = 0.20
# Only SHORT lines are drop-eligible (BOTH ceilings must hold → conservative).
_GARBAGE_MAX_CHARS = 40
_GARBAGE_MAX_TOKENS = 6

# ---------------------------------------------------------------------------
# ®/© re-OCR duplicate-tail recognizer (ch01 exercises 207/208).
# ---------------------------------------------------------------------------
#
# A clean VLM-transcribed exercise block sometimes ships with a Tesseract
# RE-OCR of the same exercises appended as a garbage tesseract-only tail block,
# using the circled-marker confusables ``®``/``©``/``⓪`` as list-enumeration
# mis-reads INTERLEAVED with junk-glyph pipe-runs (``-Iq|`` / ``—|b|``):
#   "…b $-|b|$ when $b = -12$ ® -Iq| when g = -33 © —|b| when b = —12 Add Integers"
# The base ``_looks_like_ocr_garbage`` missed it — it is long (> the char cap),
# carries a ``$…$`` math span (the defensive math exemption), and its OCR
# fragments look like plausible words. This recognizer fires on the
# ®/©-marker + pipe-junk COMBINATION specifically, which is the signal — a
# clean LaTeX span alone (``$-|b|$``, ``$b = -12$``) is NOT (those are stripped
# out before the marker/junk scan, so clean math can never be the trigger).
# Conservative: requires ≥2 circled markers AND a pipe-junk run in the
# OUT-OF-MATH remainder before it will drop.
_REOCR_CIRCLED_MARKERS = frozenset("®©⓪①②③④⑤⑥⑦⑧⑨")
_REOCR_MIN_MARKERS = 2
# A pipe-bearing OCR-garble run outside a math span: a hyphen/dash/pipe run
# butted against a short letter run and a pipe (``-Iq|``, ``—|b|``, ``|b|``).
_REOCR_PIPE_JUNK_RE = re.compile(r"[-—–|][A-Za-z]{0,3}\||\|[A-Za-z]{0,3}[-—–|]")


def _looks_like_reocr_marker_garbage(s: str) -> bool:
    """True for a ®/©-enumerated re-OCR duplicate tail (see the block comment).

    Strips ``$…$`` / ``\\(…\\)`` / ``\\[…\\]`` math spans FIRST so a legitimate
    ``$-|b|$`` can never be the trigger, then requires BOTH a clear run of
    circled-marker confusables (``>= _REOCR_MIN_MARKERS``) AND a pipe-junk run
    in the out-of-math remainder. Domain-agnostic + conservative — a normal
    sentence carries neither signal. Pure / deterministic.
    """
    remainder = _MATH_SPAN_RE.sub(" ", s)
    markers = sum(1 for c in remainder if c in _REOCR_CIRCLED_MARKERS)
    if markers < _REOCR_MIN_MARKERS:
        return False
    if "|" not in remainder:
        return False
    return _REOCR_PIPE_JUNK_RE.search(remainder) is not None


def _looks_like_ocr_garbage(text: str) -> bool:
    """Conservative, domain-agnostic OCR-garbage test for a tesseract-only line.

    Returns True ONLY for a SHORT line (``<= _GARBAGE_MAX_CHARS`` chars AND
    ``<= _GARBAGE_MAX_TOKENS`` whitespace tokens) that CONTAINS at least one
    junk glyph (:data:`_GARBAGE_JUNK_CHARS`) AND meets one of three
    corroborating conditions:

    * no plausible word at all (no ``[A-Za-z]{3,}`` run) — e.g. a numeric
      misread ``"2,791 © 2,795"``;
    * the first non-space character is itself a junk glyph — e.g.
      ``"© Solution No."`` / ``"©) obi Dom -19"`` (a real content line
      practically never opens with ©/®/™/|);
    * the junk glyphs dominate (``>= _GARBAGE_JUNK_RATIO`` of non-space chars)
      — e.g. ``"TRY Tiss ® ©"``.

    A line with NO junk glyph is NEVER garbage here (a clean short sentence, a
    legit ``"Solution"`` label, a clean numeric answer line ``"37. 84"`` all
    have zero junk glyphs → kept). A line carrying a ``$...$`` / ``\\(...\\)``
    math span is never garbage (defensive). Long lines are kept (the target is
    short junk fragments; err toward keeping). Pure / deterministic.
    """
    s = (text or "").strip()
    if not s:
        return False
    # ®/© re-OCR duplicate-tail: fires REGARDLESS of length / a clean math span
    # (both of which the base short-junk path deliberately exempts) because the
    # ®/©-marker + pipe-junk COMBINATION is the signal, not the LaTeX.
    if _looks_like_reocr_marker_garbage(s):
        return True
    if len(s) > _GARBAGE_MAX_CHARS:
        return False
    if len(s.split()) > _GARBAGE_MAX_TOKENS:
        return False
    if _MATH_SPAN_RE.search(s):
        return False
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return False
    junk = sum(1 for c in non_space if c in _GARBAGE_JUNK_CHARS)
    if junk == 0:
        # No junk glyph → not our (unambiguous) target; keep.
        return False
    if _PLAUSIBLE_WORD_RE.search(s) is None:
        # Short + junk + no real word (a numeric / symbol misread).
        return True
    if non_space[0] in _GARBAGE_JUNK_CHARS:
        # Short + leads with a junk glyph (real content rarely opens with one).
        return True
    if junk / len(non_space) >= _GARBAGE_JUNK_RATIO:
        # Short + junk-symbol-dominated even though a word survives.
        return True
    return False


# ---------------------------------------------------------------------------
# P0 interface contract (pinned; P0 = SEMANTIK_VLM_EXTRACT, not yet in-tree).
# ---------------------------------------------------------------------------
#
# P1 consumes ONLY this shape off the page dict — it never calls the VLM. The
# P0 lane populates it (per-page image → VLM markdown transcription, cached
# like the tesseract source). P1 is inert without it.
PAGE_VLM_CONTRACT = {
    "markdown": "str — the raw VLM page transcription (LaTeX math preserved)",
    "lines": "list[str] — deterministic line split; structural markdown "
    "markers preserved for P2 (which reads page['vlm'], never the fused "
    "blocks). P1 strips those markers from the fused TEXT only.",
}

# A page whose Tesseract×VLM product exceeds this is skipped deterministically
# (tesseract kept verbatim, divergence=1.0) — the O(n*m*max_run) DP must never
# become a wall-clock cliff. ~40-100 lines/page is the norm; 250×250 is a wide
# fail-open ceiling.
_PATHOLOGICAL_CEILING = 250 * 250

# Low-confidence sentinel stamped on a VLM-only insert (the structural
# anti-hallucination tripwire — nothing downstream trusts it yet).
_VLM_ONLY_CONFIDENCE = 0.30

# Aligner knobs. ``max_run`` is bumped past the aligner default (6) because a
# VLM logical line can absorb many wrapped OCR print-lines (MERGE run length).
_ALIGN_MAX_RUN = 8
_ALIGN_SOFT_FLOOR = 0.30

# Fallback line height (image px) when a VLM-only insert has no bounding
# neighbor to interpolate a band from.
_FALLBACK_LINE_H = 12.0


# ---------------------------------------------------------------------------
# Scoring-only text normalization (fusion-local; NEVER migrate into the
# training-data aligner's normalize_for_match).
# ---------------------------------------------------------------------------

_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_LATEX_PUNCT_RE = re.compile(r"[{}\\^_&$]")


# ---------------------------------------------------------------------------
# Nemotron-Omni grounding/detection special-token strip (ch03 p.101 defect).
# ---------------------------------------------------------------------------
#
# The VLM (Nemotron-Omni) occasionally leaks its internal grounding/detection
# special tokens verbatim into the page transcription, e.g.
#   ``<|ref|>Question<|/ref|><|det|>[[155, 100, 760, 120]]<|/det|>``
# which shipped straight into the HTML. The ``<|ref|>…<|/ref|>`` pair wraps the
# REAL human-readable text ("Question" → KEEP the inner text); the
# ``<|det|>…<|/det|>`` pair wraps a machine ``[[x1,y1,x2,y2]]`` bbox payload
# (DROP it whole). This is a token STRIP, not a block DROP — do NOT fold it
# into ``_looks_like_ocr_garbage`` (which drops whole blocks). Conservative:
# only the Omni ``<|ref|>``/``<|det|>``-family tokens + the det bbox payload are
# touched, and a token-free line is byte-identical (the ``"<|"`` guard).
_VLM_DET_PAIR_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
_VLM_REF_PAIR_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
# Any residual UNPAIRED Omni ref/det-family special token.
_VLM_STRAY_SPECIAL_RE = re.compile(r"<\|/?(?:ref|det)\|>")
# A stray numeric det bbox payload ``[[x1, y1, x2, y2]]`` left by an unpaired
# det token (bounded to the digits/space/.,;-shape so real ``[[text]]`` is safe).
_VLM_DET_BBOX_RE = re.compile(r"\[\[[\s\d.,;+-]+\]\]")


def _strip_vlm_special_tokens(text: str) -> str:
    """Strip Nemotron-Omni ``<|ref|>``/``<|det|>`` grounding tokens from VLM text.

    ``<|ref|>Question<|/ref|>`` → ``Question`` (keep the inner human text);
    ``<|det|>[[…]]<|/det|>`` → dropped whole (machine bbox coords); any stray
    unpaired ref/det token or bare det bbox payload is removed. Conservative:
    a line WITHOUT the ``<|`` special-token sigil is returned byte-identical, so
    the flag-off / no-leak path is unchanged. Whitespace the removals leave
    behind is collapsed. Pure / deterministic.
    """
    if not text or "<|" not in text:
        return text
    s = _VLM_DET_PAIR_RE.sub("", text)  # det pair (+ its bbox payload) → gone
    s = _VLM_REF_PAIR_RE.sub(lambda m: m.group(1), s)  # ref pair → inner text
    s = _VLM_STRAY_SPECIAL_RE.sub("", s)  # stray unpaired ref/det tokens
    s = _VLM_DET_BBOX_RE.sub("", s)  # stray det bbox payload
    s = _WS_RUN_RE.sub(" ", s)
    return s.strip()


def _strip_latex(s: str) -> str:
    """Drop LaTeX control sequences / braces / ``$`` — SCORING ONLY.

    Turns ``\\sqrt{ab}=\\sqrt{a}\\cdot\\sqrt{b}`` into ``  ab =  a  b`` so its
    alphanumeric OPERANDS survive to match the OCR garble's operands. Harmless
    on plain OCR text (no backslashes / braces). Never touches output text.
    """
    s = _LATEX_CMD_RE.sub(" ", s)
    s = _LATEX_PUNCT_RE.sub(" ", s)
    return s


def _fold_confusables(s: str) -> str:
    """Apply the known OCR confusable folds — SCORING ONLY, symmetric.

    ``√`` OCRs as a leading ``V``/``v`` (so ``Vab`` ↔ ``\\sqrt{ab}``→``ab``);
    ``l``↔``1`` and ``O``↔``0`` are the classic digit/letter confusions.
    Applied to BOTH sides so prose (``have`` → ``hae`` on both) still matches
    while the radical operands align. Fusion-local: it changes only the
    similarity the aligner sees, never the fused block text.
    """
    s = s.lower()
    s = s.replace("v", "")  # √ read as V — strip so 'vab' == 'ab'
    s = s.replace("l", "1").replace("o", "0")
    # © (U+00A9) is the VLM's rendering of the list marker "(c)" (Defect 6): the
    # OCR side reads the literal "(c)", the VLM the glyph. Fold to "(c)" on BOTH
    # sides so an "(a) (b) (c)" option run still aligns. Scoring-only (the fused
    # block's OUTPUT text is untouched by this function).
    s = s.replace("©", "(c)")
    return s


def _score_norm(text: str) -> str:
    """The composed scoring-only normalization fed to the aligner views."""
    return _fold_confusables(_strip_latex(text or ""))


# ---------------------------------------------------------------------------
# Output text normalization (keeps LaTeX; strips markdown structure only).
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_MD_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s+")
_MD_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+")
_MD_ORDERED_RE = re.compile(r"^\s{0,3}\d+[.)]\s+")
# Ordered-marker ANYWHERE on the line (line-start OR after whitespace). Used to
# tell a genuine single-item markdown ordered-list line ("1. First point…") from
# an ANSWER-GRID / exercise row where several "N." pairs are jammed onto one line
# ("37. 84  38. 9,696  39. 75") — in a grid row each "N." is a CONTENT exercise
# label, NOT a list marker, so the line-leading number must be PRESERVED (else
# every row-leading exercise number is silently stripped: "37. 84" -> "84").
_MD_ORDERED_GLOBAL_RE = re.compile(r"(?:^|\s)\d+[.)]\s")
# Defect 2 — a WHOLE-LINE markdown code fence: ``` optionally with a language
# tag (```markdown / ```math). The VLM wraps a whole page in ```markdown … ```
# and the fence line leaks into visible text (worst: an <h2> reading
# "```markdown Chapter 9 …"). A fence line carries no content → drop it.
_MD_FENCE_LINE_RE = re.compile(r"^\s{0,3}`{3,}[A-Za-z0-9_+-]*\s*$")
# A STRAY inline fence marker (whole-page wrap collapsed onto one line, or a
# mid-line fence): ``` plus an optional immediately-adjacent language token.
_MD_FENCE_INLINE_RE = re.compile(r"`{3,}[A-Za-z0-9_+-]*")
# Defect B (coordinator follow-up, ch09 live validation) — a WHOLE-LINE LaTeX
# sectioning wrapper the VLM sometimes emits instead of a markdown heading
# (``\section*{REVIEW EXERCISES}``). The wrapper is STRUCTURAL syntax (same
# class as a ``#`` heading prefix), not content — unwrap to the inner text so
# downstream heading typing / apparatus promotion sees clean text. Whole-line
# only (anchored), starred or not, section/subsection/subsubsection. An INLINE
# ``\section`` mention inside prose is never rewritten, and inline ``$…$`` math
# LaTeX is untouched (this fires only on the anchored sectioning-command shape).
_LATEX_SECTION_WRAP_RE = re.compile(
    r"^\\(?:sub){0,2}section\*?\{(?P<inner>[^{}]*)\}\s*$"
)
# ch02 audit (2026-07-04) — the VLM transcribes blank table spacing as LITERAL
# ``&nbsp;`` HTML-entity TEXT in its markdown (runs of ``&nbsp; &nbsp; …``,
# sometimes double-escaped ``&amp;nbsp;``); fusion previously carried them into
# block text verbatim (520 visible "nbsp" occurrences in the audited ch02).
# Entities are markup syntax, not content → decode the common ones to their
# character (nbsp→space, amp→&, lt→<, gt→>) and DROP any other literal
# ``&[a-z]+;`` entity to a space. ``(?:amp;)*`` unwraps the double-escaped form
# in one pass (``&amp;nbsp;`` → the ``nbsp`` decode). Lowercase-only, ``;``
# required — prose like "R&D" or a bare word "nbsp" never matches.
_HTML_ENTITY_RE = re.compile(r"&(?:amp;)*([a-z]+);")
_ENTITY_DECODE = {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">"}
_WS_RUN_RE = re.compile(r"\s{2,}")


def _strip_markdown_structure(line: str) -> str:
    """Strip ONLY markdown STRUCTURAL syntax; keep inline LaTeX verbatim.

    Structure hints (``#`` heading prefix, ``>`` blockquote, ``-``/``*``/``+``
    and ``1.`` list markers, ``**bold**`` / ``__bold__`` wrappers, ```` ``` ````
    code-fence wrappers — Defect 2 — and whole-line ``\\section*{…}`` LaTeX
    sectioning wrappers — Defect B) belong to P2's non-authoritative
    side-channel, not to text. ``$``-delimited LaTeX is preserved
    character-for-character (the documented representation change). A whole-line
    fence (```` ```markdown ````) returns ``""`` (dropped by the caller's
    non-empty filter); a stray inline fence is scrubbed in place.
    """
    s = (line or "").strip()
    if _MD_FENCE_LINE_RE.match(s):
        return ""
    s = _MD_FENCE_INLINE_RE.sub("", s).strip()
    m = _LATEX_SECTION_WRAP_RE.match(s)
    if m:
        s = m.group("inner").strip()
    s = _MD_HEADING_RE.sub("", s)
    s = _MD_BLOCKQUOTE_RE.sub("", s)
    s = _MD_BULLET_RE.sub("", s)
    # Strip the leading ordered-list marker ONLY for a genuine single-item list
    # line. A line carrying MULTIPLE "N." markers is an answer-grid / exercise
    # row where each number is a CONTENT exercise label — preserve it verbatim.
    # Fix A (SEMANTIK_FUSION_KEEP_ORDERED_MARKERS, default ON within fusion):
    # even a SINGLE-marker "N." line is a CONTENT exercise/answer number in a
    # textbook (one exercise per line), so the strip is SKIPPED entirely; the
    # number survives. ``=0`` restores the legacy single-marker strip exactly.
    if (
        not resolve_keep_ordered_markers_mode()
        and len(_MD_ORDERED_GLOBAL_RE.findall(s)) <= 1
    ):
        s = _MD_ORDERED_RE.sub("", s)
    # Bold wrappers only (single * / _ can be multiplication / subscript).
    s = s.replace("**", "").replace("__", "")
    # Literal HTML entities (ch02 nbsp defect): decode the common ones, drop
    # the rest, then collapse the whitespace runs the decode leaves behind.
    # Guarded so entity-free lines stay byte-identical.
    if "&" in s and _HTML_ENTITY_RE.search(s):
        s = _HTML_ENTITY_RE.sub(
            lambda m: _ENTITY_DECODE.get(m.group(1), " "), s
        )
        s = _WS_RUN_RE.sub(" ", s)
    return s.strip()


def _join_vlm(lines: list[str], idxs: list[int]) -> str:
    """Markdown-strip + space-join the VLM lines at ``idxs`` (order preserved)."""
    parts = [_strip_markdown_structure(lines[i]) for i in sorted(idxs)]
    return " ".join(p for p in parts if p)


def _vlm_md_and_coverage(lines: list[str], idxs: list[int]) -> tuple[str, str]:
    """RAW (markers-preserved) markdown + coverage for the P2 hint channel.

    ``extract_shared._attach_vlm_struct_hints`` mints a NON-AUTHORITATIVE
    ``vlm_hint`` from ``vlm_md`` (the RAW markdown, so ``#``/list markers
    survive) + ``vlm_coverage``. ``coverage`` is ``whole_block`` iff a SINGLE
    VLM line fills the fused block (MATCH / MERGE / a VLM-only insert): the
    minted head-line hint then covers the whole block and may CORROBORATE a
    heading. When MULTIPLE VLM lines were joined (a SPLIT — one Tesseract line
    unfolded into several VLM lines, e.g. a title fused with body prose), the
    head hint covers only a ``prefix`` — the load-bearing distinction that
    stops a fused-title alignment from ever re-typing a prose block into an
    ``<h2>`` (the hint consumers require an explicit ``whole_block``)."""
    ordered = [lines[i] for i in sorted(idxs)]
    md = "\n".join(ordered)
    coverage = "whole_block" if len(ordered) <= 1 else "prefix"
    return md, coverage


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------


def _union_bbox(bboxes: list[list[float]]) -> list[float]:
    xs0 = [float(b[0]) for b in bboxes]
    ys0 = [float(b[1]) for b in bboxes]
    xs1 = [float(b[2]) for b in bboxes]
    ys1 = [float(b[3]) for b in bboxes]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def _even_y_bands(bbox: list[float], n: int) -> list[list[float]]:
    """Subdivide ``bbox`` into ``n`` even, monotone, non-overlapping y-bands.

    Fix B geometry: when a multi-VLM-line unit is emitted one-block-per-line the
    per-line bboxes are synthetic even y-bands of the tesseract union (x extent
    preserved), so the ``(y0, x0)`` reading-order merge and the region-level
    ``min(feature_block_indices)`` sort keep the VLM line order. Every band stays
    inside the union → no inverted / out-of-page box is possible (the caller
    still passes each band through :func:`_sanitize_bbox`). ``n <= 1`` returns
    the whole box.
    """
    x0, y0, x1, y1 = (float(v) for v in list(bbox)[:4])
    if n <= 1:
        return [[x0, y0, x1, y1]]
    h = (y1 - y0) / float(n)
    bands: list[list[float]] = []
    for i in range(n):
        by0 = y0 + i * h
        by1 = y1 if i == n - 1 else y0 + (i + 1) * h
        bands.append([x0, by0, x1, by1])
    return bands


def _median(xs: list[float]) -> float:
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return 0.0
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _sanitize_bbox(
    bbox: list[float], page_w: float, page_h: float
) -> list[float]:
    """Return a bbox valid in the tesseract IMAGE-PIXEL space the sibling
    tesseract blocks live in: four finite floats, ``x0 < x1`` and ``y0 < y1``,
    clamped to ``[0, page_w] × [0, page_h]`` (when a positive page dim is
    known). Every bbox the fusion writes — an aligned/rescue union OR a
    neighbor-interpolated VLM-only insert — passes through here so no fused
    block ever ships an inverted or out-of-page box.

    A downstream consumer that crops the page by a fused bbox (the Stage-5c
    figure-region renderer) inverts an out-of-range box after its own
    page-clamp (``y0 > y1`` → an empty crop that aborts the chapter). The
    real fault paths this guards: a VLM-only insert's fallback band
    overshooting ``page_h`` (no bounding ``next`` neighbor at the page foot),
    a missing tesseract page dim (``page_w``/``page_h`` == 0), or a malformed
    tesseract union. Order-swap fixes an inverted input; the clamp fixes an
    out-of-range one; a residual degenerate box (a clamp that collapsed the
    extent) is nudged to a ≥1px band rather than propagated.
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in list(bbox)[:4])
    except (TypeError, ValueError, IndexError):
        x0 = y0 = x1 = y1 = 0.0
    if not all(_finite(v) for v in (x0, y0, x1, y1)):
        x0 = y0 = x1 = y1 = 0.0
    # Correct an inverted box (a neighbor-interpolation or malformed-union
    # artifact) before clamping, so the clamp preserves the real extent.
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    # Clamp to the page dims of this space (only when a positive dim is known;
    # a 0/missing dim must not collapse the box to the origin).
    if page_w and page_w > 0:
        x0 = max(0.0, min(x0, page_w))
        x1 = max(0.0, min(x1, page_w))
    else:
        x0, x1 = max(0.0, x0), max(0.0, x1)
    if page_h and page_h > 0:
        y0 = max(0.0, min(y0, page_h))
        y1 = max(0.0, min(y1, page_h))
    else:
        y0, y1 = max(0.0, y0), max(0.0, y1)
    # Guarantee a strictly-positive extent (a clamp can collapse x0==x1 at a
    # page edge). Grow inward from the page bound if we're pinned against it.
    if x1 <= x0:
        if page_w and page_w > 0 and x0 + 1.0 > page_w:
            x0 = max(0.0, page_w - 1.0)
            x1 = page_w
        else:
            x1 = x0 + 1.0
    if y1 <= y0:
        if page_h and page_h > 0 and y0 + 1.0 > page_h:
            y0 = max(0.0, page_h - 1.0)
            y1 = page_h
        else:
            y1 = y0 + 1.0
    return [x0, y0, x1, y1]


# ---------------------------------------------------------------------------
# Public fusion entry point.
# ---------------------------------------------------------------------------


def fuse_page(
    tesseract_blocks: list[dict],
    vlm_lines: list[str],
    page_dims: tuple[float, float],
) -> tuple[list[dict], dict]:
    """Fuse ``vlm_lines`` onto ``tesseract_blocks`` deterministically.

    Returns ``(fused_blocks, stats)`` where ``fused_blocks`` REPLACES
    ``page['tesseract']['text_blocks']`` (consumed transparently by
    ``_merge_page`` / features / the council) and ``stats`` is a plain dict:
    ``{matched, split, merge, tesseract_only, vlm_only, rescued, divergence}``
    (``tesseract_only`` / ``vlm_only`` are the RAW ledger gap counts, so they
    always equal the aligner's ``pdf_gap`` / ``gold_gap`` bucket sizes;
    ``divergence = (tesseract_only + vlm_only) / (n_tess + n_vlm)``). On the
    pathological-page guard the stats gain ``skipped_too_large: True`` and the
    input blocks are returned unchanged.

    The bboxes are in Tesseract IMAGE-PIXEL space (``page_dims`` too) — never
    mixed with PDF points, or downstream fraction-of-page normalization skews.
    Deterministic: the aligner + this reconstruction are pure; the unbounded
    ``_sim_cached`` lru_cache on the aligner ``sim()`` is cleared per call.
    """
    page_w, page_h = float(page_dims[0] or 0.0), float(page_dims[1] or 0.0)
    tess = list(tesseract_blocks or [])
    # Strip Nemotron-Omni grounding/detection special tokens as the VLM text
    # ENTERS fusion, so a leaked ``<|ref|>…<|/ref|><|det|>[[…]]<|/det|>`` is
    # cleaned regardless of whether the line later aligns or lands VLM-only
    # (both scoring + output flow from this list). Token-free lines are
    # byte-identical (the ``"<|"`` guard).
    vlm = [_strip_vlm_special_tokens(ln) for ln in (vlm_lines or [])]
    n_tess, n_vlm = len(tess), len(vlm)

    # Fix B (c): a page with VLM lines but NO tesseract blocks (image-only /
    # OCR-empty page) SILENTLY DISCARDED every VLM line in the legacy path
    # (the early return below emits only the — empty — tesseract list). Under
    # SEMANTIK_FUSION_LINE_UNITS (default ON within fusion) emit each VLM line
    # as a flagged vlm-only insert over synthetic even bands instead of losing
    # the whole page. Flag-off → the legacy silent-drop is byte-identical.
    if n_tess == 0 and n_vlm > 0 and resolve_fusion_line_units_mode():
        fused_blocks = _emit_vlm_only_page(vlm, page_w, page_h)
        stats = {
            "matched": 0,
            "split": 0,
            "merge": 0,
            "tesseract_only": 0,
            "vlm_only": n_vlm,
            "rescued": 0,
            "divergence": 1.0,
            "vlm_only_page": True,
        }
        _apply_collapse(fused_blocks, stats)
        return fused_blocks, stats

    if n_tess == 0 or n_vlm == 0:
        # Nothing to fuse — keep tesseract verbatim (byte-identical input),
        # zero-divergence stats.
        return (
            [{**b, "fusion": "tesseract-only"} for b in tess],
            {
                "matched": 0,
                "split": 0,
                "merge": 0,
                "tesseract_only": n_tess,
                "vlm_only": n_vlm,
                "rescued": 0,
                "divergence": 0.0,
            },
        )

    if n_tess * n_vlm > _PATHOLOGICAL_CEILING:
        # Fail-open: never a wall-clock cliff. Tesseract kept verbatim.
        return (
            list(tess),
            {
                "matched": 0,
                "split": 0,
                "merge": 0,
                "tesseract_only": n_tess,
                "vlm_only": n_vlm,
                "rescued": 0,
                "divergence": 1.0,
                "skipped_too_large": True,
            },
        )

    # Lazy import — keeps numpy / the aligner out of the flag-off / cache path.
    from data import structure_align as _sa

    try:
        fb_views = [
            _sa.FBView(
                text=_score_norm(b.get("text", "")),
                bbox=tuple(
                    float(v) for v in (list(b.get("bbox") or []) + [0.0, 0.0, 0.0, 0.0])[:4]
                ),
                page=0,
                order_index=i,
            )
            for i, b in enumerate(tess)
        ]
        gold_views = [
            _sa.GoldView(text=_score_norm(ln), tag="", role="", gold_index=i)
            for i, ln in enumerate(vlm)
        ]

        result = _sa.align_blocks(
            fb_views,
            gold_views,
            max_run=_ALIGN_MAX_RUN,
            soft_floor=_ALIGN_SOFT_FLOOR,
            source="tesseract",
        )
        fused_blocks, stats = _reconstruct(
            result, tess, vlm, page_w, page_h, n_tess, n_vlm
        )
    finally:
        # The aligner's unbounded @lru_cache(maxsize=None) sim() has ~zero
        # cross-page reuse and ratchets ~750 B/entry — clear per page (the
        # SEMANTIK_WORKER_MAX_TASKS per-pair-clear precedent).
        _sa._sim_cached.cache_clear()

    # Degenerate-repetition tripwire (SEMANTIK_VLM_COLLAPSE_REPETITION, default
    # ON within fusion): a manipulative/diagram figure the VLM transcribed as an
    # endless "x + x + x …" token cycle — OR a whole SENTENCE the VLM looped ~33×
    # into a ~15 KB blob — would otherwise ship verbatim as a mega-chunk of
    # retrieval poison. Collapse each fused block's TEXT to a bounded
    # representative + count marker (both the token arm and the sentence arm).
    # Conservative (only an unambiguous degenerate run is touched) →
    # byte-identical on non-pathological input; the per-block collapse count is
    # tallied additively in ``stats``.
    _apply_collapse(fused_blocks, stats)

    return fused_blocks, stats


def _emit_vlm_only_page(
    vlm: list[str], page_w: float, page_h: float
) -> list[dict]:
    """Emit each VLM line of a no-tesseract page as a flagged vlm-only insert.

    Fix B (c): a page with VLM lines but zero tesseract blocks previously lost
    every VLM line at fusion (the legacy early return emits only the empty
    tesseract list). Each surviving line (empty-after-strip fences skipped) is a
    ``vlm-only-flagged`` block over a synthetic even y-band spanning the page, so
    the page's content reaches the council/assembler instead of vanishing. Pure;
    no tesseract geometry to key off, so the bands are synthetic (mirrors the
    ``_interp_vlm_only`` fallback posture)."""
    n = len(vlm)
    span_h = page_h if page_h and page_h > 0 else float(max(n, 1)) * _SYNTH_LINE_H
    span_w = page_w if page_w and page_w > 0 else 1.0
    bands = _even_y_bands([0.0, 0.0, span_w, span_h], n)
    out: list[dict] = []
    for band, ln in zip(bands, vlm):
        text = _strip_markdown_structure(ln)
        if not text:
            continue
        out.append(
            {
                "bbox": _sanitize_bbox(band, span_w, span_h),
                "text": text,
                "font_size": None,
                "font_name": None,
                "is_bold": None,
                "is_italic": None,
                "confidence": _VLM_ONLY_CONFIDENCE,
                "fusion": "vlm-only-flagged",
                "fusion_sim": 0.0,
                "vlm_md": ln,
                "vlm_coverage": "whole_block",
            }
        )
    return out


def _apply_collapse(fused_blocks: list[dict], stats: dict) -> None:
    """Degenerate-repetition tripwire pass (SEMANTIK_VLM_COLLAPSE_REPETITION).

    Collapses each fused block's degenerate repeated-token / looped-sentence run
    to a bounded representative + count marker; conservative → byte-identical on
    non-pathological input. The per-block collapse count is tallied additively
    in ``stats['repetition_collapsed']``. No-op when the mode is off (no key
    written), so a flag-off run's stats are byte-identical."""
    if not resolve_collapse_repetition_mode():
        return
    collapsed = 0
    for b in fused_blocks:
        txt = b.get("text")
        if not isinstance(txt, str):
            continue
        new = _collapse_degenerate_repetition(txt)
        if new is not txt:
            b["text"] = new
            collapsed += 1
    stats["repetition_collapsed"] = collapsed


def _reconstruct(
    result,
    tess: list[dict],
    vlm: list[str],
    page_w: float,
    page_h: float,
    n_tess: int,
    n_vlm: int,
) -> tuple[list[dict], dict]:
    ledger = result.ledger

    # --- aligned units (MATCH / SPLIT / MERGE), deduped from rows -----------
    # A MATCH / MERGE emits one row; a SPLIT emits k identical-provenance rows
    # (one per recovered gold member) — dedup by the (pdf-indices, gold-indices)
    # partition key so each aligned unit appears once.
    units: list[dict] = []
    seen: set = set()
    for row in result.rows:
        pdf_idx = list(row.provenance.get("pdf_block_indices", []))
        gold_idx = list(row.align.get("gold_indices", []))
        key = (tuple(pdf_idx), tuple(gold_idx))
        if key in seen:
            continue
        seen.add(key)
        units.append(
            {
                "kind": row.align.get("kind", ""),
                "tess": pdf_idx,
                "vlm": gold_idx,
                "conf": float(row.align.get("confidence", 0.0)),
            }
        )
    units.sort(key=lambda u: min(u["tess"]))

    # --- raw gaps from the ledger (gaps emit no rows) -----------------------
    tess_only_all = sorted(ledger.pdf_bucket_indices.get("pdf_gap", []))
    vlm_only_all = sorted(ledger.gold_bucket_indices.get("gold_gap", []))
    tess_gap_set = set(tess_only_all)
    vlm_gap_set = set(vlm_only_all)

    fused: list[dict] = []
    # Parallel to ``fused``: the representative VLM (gold) index each block
    # covers, or ``None`` for a tesseract-only block (no VLM content). Used
    # ONLY by the VLM-authoritative reorder; discarded on the clean path, so
    # building it unconditionally is byte-neutral.
    sort_keys: list[int | None] = []
    counts = {
        "matched": 0,
        "split": 0,
        "merge": 0,
        "tesseract_only": len(tess_only_all),
        "vlm_only": len(vlm_only_all),
        "rescued": 0,
        # Unrescued tesseract-only blocks dropped as unambiguous OCR garbage by
        # SEMANTIK_VLM_DROP_GARBAGE_TAILS (auditable; mirrors vlm_furniture's
        # per-page ``vlm_furniture_dropped``). ``tesseract_only`` stays the RAW
        # ledger gap count (divergence invariant), so this is ADDITIVE.
        "garbage_tail_dropped": 0,
    }

    # Anchors (aligned units + rescue fusions) used to interpolate VLM-only
    # insert positions. Populated as we build aligned + rescue blocks.
    anchors: list[dict] = []

    line_units = resolve_fusion_line_units_mode()

    def _emit_aligned(u: dict) -> None:
        bbox = _sanitize_bbox(
            _union_bbox([tess[t]["bbox"] for t in u["tess"]]), page_w, page_h
        )
        anchor = tess[min(u["tess"])]
        vlm_idxs = sorted(u["vlm"])
        # Anchor (for VLM-only interpolation) + kind tally stay ONE per unit,
        # independent of how many fused blocks the unit emits.
        anchors.append(
            {
                "vlm_lo": vlm_idxs[0],
                "vlm_hi": vlm_idxs[-1],
                "bbox": bbox,
                "font_size": anchor.get("font_size"),
            }
        )
        kind = u["kind"]
        if kind == "matched":
            counts["matched"] += 1
        elif kind == "split":
            counts["split"] += 1
        elif kind == "merge":
            counts["merge"] += 1

        def _base(text: str, vlm_md: str, coverage: str) -> dict:
            return {
                "text": text,
                "font_size": anchor.get("font_size"),
                "font_name": anchor.get("font_name"),
                "is_bold": anchor.get("is_bold"),
                "is_italic": anchor.get("is_italic"),
                "confidence": float(anchor.get("confidence", 1.0)),
                "fusion": "vlm+tesseract",
                "fusion_sim": round(float(u["conf"]), 6),
                "vlm_md": vlm_md,
                "vlm_coverage": coverage,
            }

        # Fix B: a SPLIT unit (>1 VLM line) emits ONE block PER VLM line over
        # even y-bands of the tesseract union, so the worked-example label /
        # stem / table / TRY-IT lines survive as DISTINCT units instead of one
        # space-joined mega-block. Flag-off / single-line unit → legacy join.
        if line_units and len(vlm_idxs) > 1:
            bands = _even_y_bands(bbox, len(vlm_idxs))
            emitted = 0
            for band, g in zip(bands, vlm_idxs):
                text = _strip_markdown_structure(vlm[g])
                if not text:  # skip an empty-after-strip line (a code fence)
                    continue
                blk = _base(text, vlm[g], "whole_block")
                blk["bbox"] = _sanitize_bbox(band, page_w, page_h)
                fused.append(blk)
                sort_keys.append(g)
                emitted += 1
            if emitted:
                return
            # Every line stripped to empty (all fences) → fall through to the
            # legacy single-block emit so the unit is never silently vanished.

        text = _join_vlm(vlm, vlm_idxs)
        vlm_md, coverage = _vlm_md_and_coverage(vlm, vlm_idxs)
        blk = _base(text, vlm_md, coverage)
        blk["bbox"] = bbox
        fused.append(blk)
        sort_keys.append(vlm_idxs[0])

    for u in units:
        _emit_aligned(u)

    # --- gap-pairing rescue -------------------------------------------------
    # Between two aligned anchors, an adjacent (tesseract-only-run,
    # vlm-only-run) pair is fused positionally: the k tesseract lines' union
    # bbox gets the VLM run's text. Bounded by aligned anchors on BOTH sides —
    # a leading/trailing gap run (anchor on only one side) is NOT rescued.
    rescued_tess: set = set()
    rescued_vlm: set = set()
    for a in range(len(units) - 1):
        left, right = units[a], units[a + 1]
        t_lo, t_hi = max(left["tess"]), min(right["tess"])
        v_lo, v_hi = max(left["vlm"]), min(right["vlm"])
        region_tess = [t for t in tess_only_all if t_lo < t < t_hi]
        region_vlm = [v for v in vlm_only_all if v_lo < v < v_hi]
        if not region_tess or not region_vlm:
            continue
        bbox = _sanitize_bbox(
            _union_bbox([tess[t]["bbox"] for t in region_tess]), page_w, page_h
        )
        anchor = tess[min(region_tess)]
        region_vlm_sorted = sorted(region_vlm)
        # Recorded sim is low by construction — a positional guess, not a
        # token match (surfaces in the divergence tripwire).
        fusion_sim = _rescue_sim(tess, vlm, region_tess, region_vlm)

        def _rescue_base(text: str, vlm_md: str, coverage: str) -> dict:
            return {
                "text": text,
                "font_size": anchor.get("font_size"),
                "font_name": anchor.get("font_name"),
                "is_bold": anchor.get("is_bold"),
                "is_italic": anchor.get("is_italic"),
                "confidence": float(anchor.get("confidence", 1.0)),
                "fusion": "vlm+tesseract",
                "fusion_sim": round(float(fusion_sim), 6),
                "vlm_md": vlm_md,
                "vlm_coverage": coverage,
            }

        # Fix B: an unbounded rescued VLM run (the p127 mega-block source) emits
        # ONE block PER VLM line over even y-bands of the rescued tesseract
        # union. Flag-off / single-line run → legacy space-joined block.
        emitted_rescue = 0
        if line_units and len(region_vlm_sorted) > 1:
            bands = _even_y_bands(bbox, len(region_vlm_sorted))
            for band, g in zip(bands, region_vlm_sorted):
                text = _strip_markdown_structure(vlm[g])
                if not text:
                    continue
                blk = _rescue_base(text, vlm[g], "whole_block")
                blk["bbox"] = _sanitize_bbox(band, page_w, page_h)
                fused.append(blk)
                sort_keys.append(g)
                emitted_rescue += 1
        if not emitted_rescue:
            text = _join_vlm(vlm, region_vlm)
            vlm_md, coverage = _vlm_md_and_coverage(vlm, region_vlm)
            blk = _rescue_base(text, vlm_md, coverage)
            blk["bbox"] = bbox
            fused.append(blk)
            sort_keys.append(region_vlm_sorted[0])
        anchors.append(
            {
                "vlm_lo": region_vlm_sorted[0],
                "vlm_hi": region_vlm_sorted[-1],
                "bbox": bbox,
                "font_size": anchor.get("font_size"),
            }
        )
        rescued_tess.update(region_tess)
        rescued_vlm.update(region_vlm)
        counts["rescued"] += 1

    anchors.sort(key=lambda x: x["vlm_lo"])

    # --- tesseract-only (unrescued PDF gaps) → byte-verbatim ----------------
    # An unrescued tesseract-only block is one the DP aligner matched to NO vlm
    # line AND the gap-pairing rescue skipped; on a cleanly-VLM-transcribed scan
    # it is almost always OCR noise the VLM correctly omitted. The default-ON
    # (within-fusion) SEMANTIK_VLM_DROP_GARBAGE_TAILS pass DROPS the
    # unambiguously-garbage ones (isolated junk-glyph fragments — "© Solution
    # No.", "2,791 © 2,795") via the conservative :func:`_looks_like_ocr_garbage`
    # predicate; everything else is kept byte-verbatim (VLM-drop protection).
    # ``tesseract_only`` in ``counts`` is the RAW ledger gap count (the
    # divergence invariant) and is NOT decremented — the drop is tallied
    # separately in ``garbage_tail_dropped``. ``=0`` reverts to keep-verbatim.
    drop_garbage_tails = resolve_drop_garbage_tails_mode()
    for t in tess_only_all:
        if t in rescued_tess:
            continue
        if drop_garbage_tails and _looks_like_ocr_garbage(tess[t].get("text", "")):
            counts["garbage_tail_dropped"] += 1
            continue
        fused.append({**tess[t], "fusion": "tesseract-only"})
        sort_keys.append(None)  # no VLM content → dropped by the reorder

    # --- vlm-only (unrescued GOLD gaps) → flagged low-confidence inserts -----
    global_font = _median([b.get("font_size") for b in tess])
    unresc_vlm = [v for v in vlm_only_all if v not in rescued_vlm]
    for bbox, font_size, v in _interp_vlm_only(
        unresc_vlm, anchors, page_w, page_h, global_font
    ):
        text = _strip_markdown_structure(vlm[v])
        fused.append(
            {
                "bbox": bbox,
                "text": text,
                "font_size": font_size,
                "font_name": None,
                "is_bold": None,
                "is_italic": None,
                "confidence": _VLM_ONLY_CONFIDENCE,
                "fusion": "vlm-only-flagged",
                "fusion_sim": 0.0,
                # A standalone new block = a single VLM line = whole_block.
                "vlm_md": vlm[v],
                "vlm_coverage": "whole_block",
            }
        )
        sort_keys.append(v)

    total = n_tess + n_vlm
    divergence = (
        (counts["tesseract_only"] + counts["vlm_only"]) / total if total else 0.0
    )
    counts["divergence"] = divergence

    # VLM-authoritative reading-order repair on high-divergence pages (dense
    # multi-column): Tesseract geometry is unreliable, so re-emit the
    # VLM-bearing blocks in VLM order over synthetic monotonic single-column
    # bboxes and DROP the tesseract-only garbage. Low-divergence (clean) pages
    # fall below the floor → the fused list + counts are byte-identical.
    if (
        divergence >= resolve_vlm_order_divergence_floor()
        and resolve_vlm_order_authoritative_mode()
    ):
        fused = _apply_vlm_order(fused, sort_keys, page_w, page_h)
        counts["vlm_authoritative"] = True
    else:
        counts["vlm_authoritative"] = False
    return fused, counts


def _apply_vlm_order(
    fused: list[dict],
    sort_keys: list[int | None],
    page_w: float,
    page_h: float,
) -> list[dict]:
    """Re-emit VLM-bearing blocks in VLM order with synthetic monotonic bboxes.

    Keeps only blocks that carry VLM content (``sort_keys[i] is not None`` —
    aligned / rescued / VLM-only inserts), orders them by their representative
    VLM (gold) index (a total order — every VLM line belongs to exactly one
    block), and overwrites each block's bbox with a single-column band whose
    ``y0`` is strictly increasing in VLM order. Because all bands share
    ``x0 == 0`` (one column) and ascend in ``y0``, BOTH the downstream
    ``_merge_page`` ``(y0, x0)`` sort AND the region-level
    ``min(feature_block_indices)`` sort reproduce VLM order instead of the
    scrambled Tesseract geometry. Unmatched tesseract-only garbage lines carry
    ``None`` → they are dropped. Fail-open: an empty VLM-bearing set returns
    ``fused`` unchanged (nothing to reorder)."""
    indexed = [
        (k, b) for k, b in zip(sort_keys, fused) if k is not None
    ]
    if not indexed:
        return fused
    indexed.sort(key=lambda kb: kb[0])
    n = len(indexed)
    line_h = (page_h / n) if (page_h and page_h > 0) else _SYNTH_LINE_H
    width = page_w if (page_w and page_w > 0) else 1.0
    out: list[dict] = []
    for rank, (_k, block) in enumerate(indexed):
        y0 = rank * line_h
        y1 = y0 + line_h
        nb = dict(block)
        nb["bbox"] = _sanitize_bbox([0.0, y0, width, y1], page_w, page_h)
        out.append(nb)
    return out


def _rescue_sim(
    tess: list[dict], vlm: list[str], region_tess: list[int], region_vlm: list[int]
) -> float:
    """Scoring-normalized sim between the rescued runs (low by construction)."""
    from data import structure_align as _sa

    t = _score_norm(" ".join(tess[i].get("text", "") for i in sorted(region_tess)))
    v = _score_norm(" ".join(vlm[i] for i in sorted(region_vlm)))
    return _sa.sim(t, v)


def _interp_vlm_only(
    vlm_idxs: list[int],
    anchors: list[dict],
    page_w: float,
    page_h: float,
    global_font: float,
):
    """Yield ``(bbox, font_size, vlm_index)`` for each VLM-only insert.

    bbox is interpolated from the nearest aligned neighbors in VLM order:
    x = union of prev/next x-ranges, y strictly between ``prev.bottom`` and
    ``next.top``. Consecutive inserts sharing the same (prev, next) neighbor
    pair are distributed evenly through the band so their ``(y0, x0)`` sort
    keys stay strictly increasing (no two inserts collide).
    """
    if not vlm_idxs:
        return

    def _neighbors(v: int) -> tuple[dict | None, dict | None]:
        prev = None
        nxt = None
        for a in anchors:
            if a["vlm_hi"] < v:
                if prev is None or a["vlm_hi"] > prev["vlm_hi"]:
                    prev = a
            elif a["vlm_lo"] > v:
                if nxt is None or a["vlm_lo"] < nxt["vlm_lo"]:
                    nxt = a
        return prev, nxt

    # Group consecutive inserts by identical (prev, next) neighbor identity.
    groups: list[list[tuple[int, dict | None, dict | None]]] = []
    for v in sorted(vlm_idxs):
        prev, nxt = _neighbors(v)
        pid = id(prev) if prev is not None else None
        nid = id(nxt) if nxt is not None else None
        if groups and groups[-1][0][3] == pid and groups[-1][0][4] == nid:
            groups[-1].append((v, prev, nxt, pid, nid))  # type: ignore[arg-type]
        else:
            groups.append([(v, prev, nxt, pid, nid)])  # type: ignore[list-item]

    for group in groups:
        g = len(group)
        for rank, (v, prev, nxt, _pid, _nid) in enumerate(group):
            y_lo = float(prev["bbox"][3]) if prev is not None else 0.0
            if nxt is not None:
                y_hi = float(nxt["bbox"][1])
            else:
                y_hi = page_h if page_h > y_lo else (y_lo + g * _FALLBACK_LINE_H)
            if y_hi <= y_lo:
                y_hi = y_lo + g * _FALLBACK_LINE_H
            band = (y_hi - y_lo) / g
            slot_lo = y_lo + rank * band
            y0 = slot_lo + 0.25 * band
            y1 = slot_lo + 0.75 * band
            xs: list[float] = []
            if prev is not None:
                xs += [float(prev["bbox"][0]), float(prev["bbox"][2])]
            if nxt is not None:
                xs += [float(nxt["bbox"][0]), float(nxt["bbox"][2])]
            if xs:
                x0, x1 = min(xs), max(xs)
            else:
                x0, x1 = 0.0, (page_w if page_w > 0 else 1.0)
            fonts = [
                a["font_size"]
                for a in (prev, nxt)
                if a is not None and a.get("font_size") is not None
            ]
            font_size = _median(fonts) if fonts else global_font
            # Sanitize in tesseract IMAGE-PIXEL space: a trailing insert at the
            # page foot (no bounding ``next`` neighbor) takes the fallback band
            # ``y_lo + g*_FALLBACK_LINE_H``, which OVERSHOOTS ``page_h``; a
            # missing tesseract page dim leaves x/y unbounded. Clamp so the
            # insert never ships an out-of-page / inverted box to the council
            # or the Stage-5c figure renderer.
            yield _sanitize_bbox([x0, y0, x1, y1], page_w, page_h), font_size, v
