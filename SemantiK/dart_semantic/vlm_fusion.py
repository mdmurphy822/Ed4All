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
    "resolve_vlm_fusion_mode",
    "fuse_page",
    "PAGE_VLM_CONTRACT",
]

# ---------------------------------------------------------------------------
# Flag resolver (SEMANTIK_VLM_FUSION) — parse-with-fallback, default OFF.
# ---------------------------------------------------------------------------

SEMANTIK_VLM_FUSION_ENV = "SEMANTIK_VLM_FUSION"
_TRUTHY = {"1", "true", "yes", "on"}


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
    vlm = [ln for ln in (vlm_lines or [])]
    n_tess, n_vlm = len(tess), len(vlm)

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

    return fused_blocks, stats


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
    counts = {
        "matched": 0,
        "split": 0,
        "merge": 0,
        "tesseract_only": len(tess_only_all),
        "vlm_only": len(vlm_only_all),
        "rescued": 0,
    }

    # Anchors (aligned units + rescue fusions) used to interpolate VLM-only
    # insert positions. Populated as we build aligned + rescue blocks.
    anchors: list[dict] = []

    def _emit_aligned(u: dict) -> None:
        bbox = _sanitize_bbox(
            _union_bbox([tess[t]["bbox"] for t in u["tess"]]), page_w, page_h
        )
        anchor = tess[min(u["tess"])]
        text = _join_vlm(vlm, u["vlm"])
        vlm_md, coverage = _vlm_md_and_coverage(vlm, u["vlm"])
        fused.append(
            {
                "bbox": bbox,
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
        )
        anchors.append(
            {
                "vlm_lo": min(u["vlm"]),
                "vlm_hi": max(u["vlm"]),
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
        text = _join_vlm(vlm, region_vlm)
        vlm_md, coverage = _vlm_md_and_coverage(vlm, region_vlm)
        # Recorded sim is low by construction — a positional guess, not a
        # token match (surfaces in the divergence tripwire).
        fusion_sim = _rescue_sim(tess, vlm, region_tess, region_vlm)
        fused.append(
            {
                "bbox": bbox,
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
        )
        anchors.append(
            {
                "vlm_lo": min(region_vlm),
                "vlm_hi": max(region_vlm),
                "bbox": bbox,
                "font_size": anchor.get("font_size"),
            }
        )
        rescued_tess.update(region_tess)
        rescued_vlm.update(region_vlm)
        counts["rescued"] += 1

    anchors.sort(key=lambda x: x["vlm_lo"])

    # --- tesseract-only (unrescued PDF gaps) → byte-verbatim ----------------
    for t in tess_only_all:
        if t in rescued_tess:
            continue
        fused.append({**tess[t], "fusion": "tesseract-only"})

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

    total = n_tess + n_vlm
    counts["divergence"] = (
        (counts["tesseract_only"] + counts["vlm_only"]) / total if total else 0.0
    )
    return fused, counts


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
