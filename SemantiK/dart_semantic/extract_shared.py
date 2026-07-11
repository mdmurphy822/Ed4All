"""Stage 1 core: run every applicable extractor and emit a standardized
per-page JSON document.

Rather than routing to a single best extractor per document, we run all
licence-compatible extractors that apply and expose their outputs side-by-
side in one JSON shape. Downstream stages (features, classify) read the
MERGED view but can also inspect each extractor's raw output for
disagreement signals or training augmentation.

Licensing inventory (all must be commercial-safe):
    pikepdf       MPL-2.0     metadata, encryption, tagged-PDF detection, widgets
    pypdfium2     Apache-2    primary text extraction w/ bboxes + rendering
    pdfplumber    MIT         table detection, char-level layout
    pdfminer.six  MIT         (transitive via pdfplumber; not called directly)
    Tesseract     Apache-2    OCR fallback when text layer missing/corrupt

Every library above is permissively licensed and commercial-safe.

Output JSON per document:
{
  "pdf_path": "...",
  "metadata": { producer, creator, title, pdf_version, is_encrypted,
                is_tagged, page_count },
  "pages": [
    {
      "page_num": 1,                 // 1-indexed
      "width": 612.0, "height": 792.0,
      "sources_used": ["pypdfium2", "pdfplumber"],   // or "tesseract", "tagged_pdf"
      "pikepdf": { "widgets": [...], "structure_nodes": [...] },  // present if applicable
      "pypdfium2": { "text_blocks": [...] },
      "pdfplumber": { "text_blocks": [...], "tables": [...] },
      "tesseract": { "text_blocks": [...] },          // only when we ran OCR
      "merged": {
        "text_blocks": [...],        // reconciled stream of {bbox, text, font_size, font_name, is_bold, is_italic, confidence, provenance}
        "tables": [...],             // from pdfplumber
        "widgets": [...]             // from pikepdf when fillable
      }
    }, ...
  ]
}

Where `provenance` is a list of source-names that contributed that block,
e.g. ["pypdfium2", "pdfplumber"] if both agreed on the line, or
["tesseract"] if only OCR produced it.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import paths as _semantik_paths
from . import vlm_extract as _vlm_extract
from .vlm_fusion import (
    resolve_drop_garbage_tails_mode as _resolve_drop_garbage_tails_mode,
    resolve_vlm_fusion_mode,
)
from .vlm_furniture import (
    resolve_strip_furniture_mode as _resolve_strip_furniture_mode,
)
from .reading_order import (
    _DEFAULT_PAGE_WIDTH,
    _GUTTER_GAP_FRAC,
    column_edges_from_lines,
    column_ids_for_bboxes,
    column_ids_for_x0s,
    resolve_column_extract_mode,
    resolve_column_order_mode,
    resolve_deploy_profile,
)
from .region_detection import CID_RE

logger = logging.getLogger(__name__)


# ---------- figure detection gate (Part F) ----------

# Default OFF — byte-stable. When on, every extracted page gains an
# ``images`` list (PDF IMAGE page-objects, top-left bbox + px_size) that
# ``region_detection.detect_image_region_candidates`` consumes. The full
# resolver of record is ``cascade.resolve_detect_figures``; this is the
# extract-side mirror (extract runs inside the council orchestrator, which
# does not get the flag threaded). Parse-with-fallback: truthy
# (``1``/``true``/``yes``/``on``) → on; unset / falsey / garbage → off.
_DETECT_FIGURES_ENV = "SEMANTIK_DETECT_FIGURES"
_FIG_TRUTHY = {"1", "true", "yes", "on"}


def _detect_figures_enabled() -> bool:
    """Whether per-page IMAGE page-object extraction runs (default OFF).

    Reads ``SEMANTIK_DETECT_FIGURES`` OR the bundled ``SEMANTIK_DEPLOY_PROFILE``
    (W7.1), mirroring ``cascade.resolve_detect_figures``.
    """
    if os.environ.get(_DETECT_FIGURES_ENV, "").strip().lower() in _FIG_TRUTHY:
        return True
    return resolve_deploy_profile()


# NOTE: the VLM extraction-source gate + provider-agnostic seat resolvers
# (``resolve_vlm_extract_mode`` / ``resolve_vlm_seat`` / ``VLMSeat`` + the P2
# ``mint_vlm_hint`` hint channel) live further down this module (see the
# "P2 — VLM extraction source" section). The P0 per-page HTTP client, disk
# cache, and unload lifecycle live in the sibling ``vlm_extract.py`` and consume
# those resolvers — kept there so the heavy image-encode / requests path is
# isolated from this Stage-1 orchestration module.


# Minimum text-layer chars below which we assume the page is scanned.
TEXT_LAYER_MIN_CHARS = 10
# If >10% of text-layer chars are Unicode replacement, treat as corrupt.
REPLACEMENT_CHAR = "�"
REPLACEMENT_MAX_FRAC = 0.10
# pdfplumber word-break gap threshold as a fraction of font size. 0.15 (≈1.5pt
# at 10pt) recovers spaces on tight-kerned LaTeX PDFs without over-splitting
# large-font titles. See _pdfplumber_page / Plans/11 Stage 3.
_X_TOLERANCE_RATIO = 0.15

# Max horizontal gap (pt) under which a single-character leading word is
# treated as a *styled first letter that was split off its word body* (an
# textbook colored / bold drop-cap: the Bold "S" of "Subtraction" becomes its
# own pdfplumber word, ABUTTING "ubtraction"). Measured on a real textbook
# order-of-operations chart page: every such split sits at a ~0pt gap
# (-0.01..0.01) AND carries a font-name change (Bold→Regular), while every
# genuine single-letter word ("I remember", "a way", articles "A"/"a"/"I"
# across body pages) has a real word-space gap >= ~1.99pt. 0.5 is a safe
# midpoint with a wide margin on both sides. See _merge_split_first_letters.
_FIRST_LETTER_REJOIN_MAX_GAP = 0.5


def _merge_split_first_letters(line: list[dict]) -> list[dict]:
    """Rejoin a single-character leading word that a font/style change split
    off from its word body.

    pdfplumber's ``extract_words`` starts a new word at a font/style change,
    so a styled colored / bold drop-cap first letter (the Bold "S" of
    "Subtraction") is emitted as its OWN word ABUTTING the rest
    ("ubtraction"). The downstream space-join then fabricates "S ubtraction".

    Conservative gate (BOTH required, calibrated on measured geometry —
    extraction is load-bearing):

    * the leading word is exactly ONE letter, AND
    * it ABUTS the following word: x-gap <= ``_FIRST_LETTER_REJOIN_MAX_GAP``
      (~0pt). A real inter-word space is >= ~2pt, so genuine single-letter
      words ("I remember", "a way") are never merged, AND
    * the two words carry DIFFERENT font names — the styling that caused the
      split. Same-font adjacencies (which pdfplumber would not have split in
      the first place) are left untouched.

    Merge is text-only (``leader + follower``, no space); the merged word
    keeps the FOLLOWER's font attributes (the word body, so heading / prose
    font features stay stable) with a union bbox. Words ``line`` must already
    be x-sorted. Returns a new list (input unmutated). A single left-to-right
    pass handles mnemonic chains ("M y D ear" → "My" then "Dear") because a
    merged word is multi-character and so can no longer act as a leader.
    """
    if len(line) < 2:
        return line
    out: list[dict] = []
    i = 0
    n = len(line)
    while i < n:
        w = line[i]
        text = w.get("text", "")
        if i + 1 < n and len(text) == 1 and text.isalpha():
            nxt = line[i + 1]
            try:
                gap = float(nxt.get("x0", 0)) - float(w.get("x1", 0))
            except (TypeError, ValueError):
                gap = None
            fn_w = w.get("fontname")
            fn_n = nxt.get("fontname")
            font_changed = bool(fn_w) and bool(fn_n) and fn_w != fn_n
            if (
                gap is not None
                and gap <= _FIRST_LETTER_REJOIN_MAX_GAP
                and font_changed
            ):
                merged = dict(nxt)  # keep the body word's font/size attrs
                merged["text"] = text + nxt.get("text", "")
                try:
                    merged["x0"] = min(float(w.get("x0", 0)), float(nxt.get("x0", 0)))
                    merged["x1"] = max(float(w.get("x1", 0)), float(nxt.get("x1", 0)))
                    merged["top"] = min(float(w.get("top", 0)), float(nxt.get("top", 0)))
                    merged["bottom"] = max(
                        float(w.get("bottom", 0)), float(nxt.get("bottom", 0))
                    )
                except (TypeError, ValueError):
                    pass
                out.append(merged)
                i += 2
                continue
        out.append(w)
        i += 1
    return out


# ---------- entry point ----------


def extract_shared(pdf_path: Path) -> dict:
    """Run all applicable extractors and return the shared JSON."""
    pdf_path = Path(pdf_path)

    meta = _pikepdf_inspect(pdf_path)
    if meta["is_encrypted"]:
        raise EncryptedPDFError(pdf_path)

    # Per-page extraction. When the VLM source is on, open a per-document
    # session so the end-of-document unload fires exactly once (and only when
    # >=1 LIVE POST was actually made).
    vlm_on = resolve_vlm_extract_mode()
    # Defect 1 — when the VLM DP fusion is on we DEFER per-page fusion + merge
    # until AFTER the whole document is extracted, so a document-level repeated
    # page-furniture strip (``vlm_furniture``) can run over every page's VLM
    # lines BEFORE any page is fused (furniture detection is inherently
    # cross-page). Flag-off / no-fusion → ``defer`` is False → the historic
    # single-loop fuse-and-merge-inline path (byte-identical).
    defer_finalize = vlm_on and resolve_vlm_fusion_mode()
    if vlm_on:
        _vlm_extract.begin_document_session()
    pages: list[dict] = []
    try:
        for page_num in range(1, meta["page_count"] + 1):
            pages.append(
                _extract_page(
                    pdf_path, page_num, meta, defer_finalize=defer_finalize
                )
            )
    finally:
        # Lifecycle unload — hand the card back to the 7B text seat. Gated on
        # provider==local (a hosted seat has no ollama /api root) AND >=1 live
        # POST this document (an all-cache-hit run loaded nothing). Best-effort
        # (never raises); finally-guaranteed so a mid-document raise still frees
        # the card. SEAM: replace with lib/gpu_lifecycle.py when it lands.
        if (
            vlm_on
            and resolve_vlm_provider() == "local"
            and _vlm_extract.document_had_live_post()
        ):
            seat = resolve_vlm_seat()
            _vlm_extract.unload_vlm_model(seat.base_url, seat.model)

    # Deferred finalize (VLM fusion on): strip cross-page repeated furniture
    # from every page's VLM lines (Defect 1), THEN fuse + merge each page. The
    # per-page ``_apply_vlm_fusion`` + ``_merge_page`` were held back inside
    # ``_extract_page`` so the furniture detector could see the whole document.
    if defer_finalize:
        _strip_document_furniture(pages)
        for page in pages:
            _finalize_page(page)

    return {
        "pdf_path": str(pdf_path),
        "metadata": meta,
        "pages": pages,
    }


def _tesseract_lines_reading_order(page: dict) -> list[str]:
    """A page's tesseract block texts sorted by bbox ``(y0, x0)`` reading order.

    The furniture detector position-gates on FIRST/LAST list positions, so the
    list must be in top-to-bottom order — the raw tesseract dict-insertion
    order is only approximately so. Blocks lacking a bbox sort to the top with
    a stable secondary order (harmless: they can only vote as page-top, and a
    recurring signature still needs the cross-page threshold).
    """
    tb = page.get("tesseract", {}).get("text_blocks") or []

    def _key(b: dict) -> tuple[float, float]:
        bbox = b.get("bbox") or []
        try:
            return (float(bbox[1]), float(bbox[0]))
        except (TypeError, ValueError, IndexError):
            return (0.0, 0.0)

    return [b.get("text", "") for b in sorted(tb, key=_key)]


def _strip_document_furniture(pages: list[dict]) -> None:
    """Detect + strip repeated page-furniture across the document's lines.

    Defect 1: the VLM emits a per-page running footer / header as its own
    markdown line at the page top / bottom; the P1 fusion then glues it into
    body prose. Defect A (coordinator follow-up, ch09 live validation):
    tesseract ALSO reads the footer cleanly on ~36 pages, where it survives as
    a standalone LAST-position tesseract block regardless of the VLM-markdown
    strip — and the DP fusion then aligns the VLM footer line onto it
    (``fusion='vlm+tesseract'``) instead of dropping it. So signatures are
    detected from BOTH sources — the VLM lines AND the (y-sorted) tesseract
    blocks — and the UNION is stripped from both sides. The tesseract-side
    detection also closes the VLM-side position-gate gap: a footer that is not
    literally the last VLM markdown line on some pages still gets its
    signature voted in by the tesseract side (footer = last tesseract block),
    and ``strip_furniture_lines``-style whole-line matching then removes it
    from the VLM lines ANYWHERE on the page. No-op when the strip is disabled
    or no furniture recurs. Fail-soft — never aborts extraction.
    """
    if not _resolve_strip_furniture_mode():
        return
    try:
        from .vlm_furniture import (
            detect_furniture_signatures,
            normalize_furniture_sig,
        )

        vlm_pages_lines: list[list[str]] = []
        tess_pages_lines: list[list[str]] = []
        for page in pages:
            tb = (page.get("vlm") or {}).get("text_blocks") or []
            vlm_pages_lines.append([b.get("text", "") for b in tb])
            tess_pages_lines.append(_tesseract_lines_reading_order(page))
        sigs = detect_furniture_signatures(vlm_pages_lines)
        sigs |= detect_furniture_signatures(tess_pages_lines)
        if not sigs:
            return
        for page in pages:
            vlm = page.get("vlm")
            if vlm and vlm.get("text_blocks"):
                kept = [
                    b
                    for b in vlm["text_blocks"]
                    if normalize_furniture_sig(b.get("text", "")) not in sigs
                ]
                page["vlm_furniture_dropped"] = len(vlm["text_blocks"]) - len(kept)
                vlm["text_blocks"] = kept
            # Strip the tesseract side too (Defect A): on many pages tesseract
            # reads the footer CLEANLY, so it survives as a standalone block
            # (and the DP fusion aligns the VLM footer line onto it instead of
            # dropping it). Only garbled furniture fails to normalize to the
            # signature — that residue stays and is harmless (it never aligned
            # anyway).
            tess_blocks = page.get("tesseract", {}).get("text_blocks")
            if tess_blocks:
                kept_tess = [
                    b
                    for b in tess_blocks
                    if normalize_furniture_sig(b.get("text", "")) not in sigs
                ]
                n_dropped = len(tess_blocks) - len(kept_tess)
                if n_dropped:
                    page["tesseract_furniture_dropped"] = n_dropped
                page["tesseract"]["text_blocks"] = kept_tess
    except Exception as exc:  # noqa: BLE001 — furniture strip never aborts extract
        logger.debug(f"vlm furniture strip failed: {exc}")


def _finalize_page(page: dict) -> None:
    """Run the deferred VLM fusion + merge for one page (Defect 1 second loop).

    Consumes the ``_is_text_ok`` flag ``_extract_page`` stashed when finalize
    was deferred, then applies fusion + builds ``merged`` — the exact tail
    ``_extract_page`` runs inline in the non-deferred (byte-identical) path.
    """
    is_text_ok = bool(page.pop("_is_text_ok", True))
    _apply_vlm_fusion(page)
    page["merged"] = _merge_page(page, is_text_ok)


# Bump whenever the extraction *output* changes for the same input PDF
# (new extractor settings, parser fixes, schema changes). Entries written
# under an older salt become unreachable, so stale extractions can never
# be served after a behavior change. v2: the 2026-06-08 pdfplumber
# x_tolerance_ratio=0.15 fix (Plans/11 Stage 3) — pre-fix caches glued
# sub-3pt LaTeX inter-word gaps into single words. v3: the 2026-06-17
# pypdfium2 line-segment re-extraction fix — pre-fix caches garbled
# letter-spaced small-caps ("M ARKWAYNE", "Before S YKES L EE").
# v4: the styled-drop-cap first-letter rejoin in _pdfplumber_page
# (_merge_split_first_letters) — pre-fix caches split a bold/colored first
# letter off its word body ("S ubtraction", "P arentheses", "m ultiplication").
EXTRACT_CACHE_VERSION = 4


def _compute_extract_cache_key(pdf_path: Path) -> str:
    """The 24-hex whole-document extract-cache key (raises OSError on stat fail).

    Salted by every flag that CHANGES THE EXTRACTED SHAPE:

    * ``fig_key`` — SEMANTIK_DETECT_FIGURES (symmetric fig0/fig1; the Part-F
      precedent).
    * ``vlm_key`` — SEMANTIK_VLM_EXTRACT (asymmetric, append-only-when-on): the
      P0 VLM source adds a per-page ``vlm`` key, so flipping the flag ON must
      never serve a stale non-VLM extraction — but the flag-OFF key MUST stay
      byte-identical to the historic ``v{VER}|{fig_key}|...`` key (no ``vlm0``
      salt), so every existing extract cache survives untouched when the
      feature is off (the house byte-identical-when-unset rule). Mirrors the
      sibling ``fuse_key``'s append-only shape.
    * ``fuse_key`` — SEMANTIK_VLM_FUSION (asymmetric, append-only-when-on): the
      P1 fusion rewrites the merged tesseract text; append-only keeps the
      flag-off key byte-identical to the P0 key. ``vlmfuse2``: the fusion-side
      bbox sanitation (live-fire ch09 fix) changed what ``fuse_page`` can emit
      — a cache written under ``vlmfuse1`` may carry unsanitized
      (out-of-page / inverted) interpolated-insert bboxes. ``vlmfuse3``: the
      Defect-1 document-level repeated-furniture strip (``vlm_furniture``) +
      Defect-2 markdown code-fence strip change what the fused blocks carry, so
      a ``vlmfuse2`` cache is invalid. ``vlmfuse4``: the coordinator-follow-up
      Defect-A tesseract-side furniture detection (signatures now also voted by
      the y-sorted tesseract blocks and stripped from the tesseract source) +
      Defect-B ``\\section*{…}`` LaTeX sectioning-wrapper normalization change
      the fused/merged output again — a ``vlmfuse3`` cache still carries the 36
      clean-OCR footer blocks and the wrapped apparatus headings. ``vlmfuse5``:
      the ch02 literal-HTML-entity scrub (``vlm_fusion._strip_markdown_structure``
      now decodes/drops ``&nbsp;`` / ``&amp;…;``-style entity text from VLM
      lines before fusion) changes the fused block text — a ``vlmfuse4`` cache
      may carry literal entity text (corrected downstream by the adapter-seam
      scrub in ``lib/semantik/adapter.py``, so the CURRENT corpus deliberately
      stays on its ``vlmfuse4`` caches; the salt protects future fresh
      conversions). The
      ``SEMANTIK_VLM_STRIP_FURNITURE`` escape hatch is deliberately kept OUT of
      the key (the render-scale kept-out-of-key precedent) — flipping the strip
      off after a warm cache needs a manual extract-cache clear. The P0
      per-page VLM markdown cache (``vlm_extract_cache``) is keyed
      independently and survives; only tesseract OCR + fusion re-run.
    * ``gtail_key`` — SEMANTIK_VLM_DROP_GARBAGE_TAILS (asymmetric,
      append-only-when-fusion-on-AND-mode-on): the default-ON garbage-tail drop
      changes what ``fuse_page`` emits (unrescued tesseract-only junk-glyph
      fragments are dropped, so the merged block list — and thus which FBs /
      sourceIds exist — differs from a warm ``vlmfuse5`` cache). A NEW
      append-only ``|gtail1`` salt (rather than a ``vlmfuse6`` bump) is used
      DELIBERATELY: because the mode has a ``=0`` escape hatch whose output is
      byte-identical to the pre-change ``vlmfuse5`` behavior, salting on the
      MODE (fusion-on AND drop-on) means a ``=0`` run keys back to the historic
      ``vlmfuse5`` (its warm cache stays valid) while the default (drop-on) run
      re-extracts once. The flag-OFF (no-fusion) key is byte-identical (no
      ``fuse_key`` → no ``gtail_key``). Matches the ``vlm_furniture`` escape-hatch
      posture but, unlike that flag, the shape change IS keyed (it is default-ON
      and reader-visible, not a furniture-only refinement folded into a salt bump).
    * ``hints_key`` — SEMANTIK_VLM_STRUCT_HINTS (asymmetric, append-only-when-on):
      the P2 hint mint (`_attach_vlm_struct_hints`) runs INSIDE `_merge_page`, so
      the `vlm_hint` payload is baked into the cached merged blocks. Without this
      salt a cache warmed with hints OFF is served (hint-less) after the operator
      flips hints ON — the exact "ship fusion text-only, flip hints later" flow —
      and the reverse serves hint-carrying blocks with the flag off. Append-only
      keeps the flag-off key byte-identical to the fusion key.
    * ``col_key`` — SEMANTIK_COLUMN_EXTRACT (asymmetric, append-only-when-on):
      the column-aware ``_pdfplumber_page`` AND ``_tesseract_page_blocks`` line
      assembly re-segments each page into column bands BEFORE the space-join, so
      the extracted text-block SHAPE changes (columns no longer fused). Bumped
      ``|col1`` → ``|col2`` when the Tesseract path became column-aware (prong 1
      changed the extracted shape for OCR pages under the same flag, so a
      ``|col1``-warmed cache must NOT be served). Append-only ``|col2`` when on
      keeps the flag-OFF key byte-identical to the historic key (no ``col0``
      salt), so every existing extract cache survives untouched when off.
    * ``ord_key`` — SEMANTIK_COLUMN_ORDER (asymmetric, append-only-when-on AND
      extract-off): the column-major merged-block re-sort in ``_merge_page``
      changes the cached ``merged`` order, but SEMANTIK_COLUMN_ORDER was absent
      from this key (a verified pre-existing hole). Append-only ``|ord1`` when
      column-order is on AND column-extract is OFF (extract-on already salts via
      ``col_key`` and forces order on) keeps every flag-off key byte-identical.

    NB the render-scale / tesseract-config / user-words flags are deliberately
    kept OUT of the key (they do not change the extracted SHAPE and keeping them
    out avoids invalidating every existing cache — see those flags' rows).
    """
    import hashlib

    st = pdf_path.stat()
    fig_key = "fig1" if _detect_figures_enabled() else "fig0"
    vlm_key = "|vlm1" if resolve_vlm_extract_mode() else ""
    fuse_key = "|vlmfuse5" if resolve_vlm_fusion_mode() else ""
    # gtail_key: the default-ON garbage-tail drop is a fusion-only, mode-gated
    # shape change (see the docstring). Keyed only when fusion AND the drop are
    # both on, so a ``=0`` run keeps its warm ``vlmfuse5`` cache and the
    # no-fusion key stays byte-identical.
    gtail_key = (
        "|gtail1"
        if resolve_vlm_fusion_mode() and _resolve_drop_garbage_tails_mode()
        else ""
    )
    hints_key = "|vlmhints1" if resolve_vlm_struct_hints_mode() else ""
    col_key = "|col2" if resolve_column_extract_mode() else ""
    # ord_key closes the pre-existing hole where SEMANTIK_COLUMN_ORDER (the
    # column-major merged-block re-sort) changed the cached `merged` order but
    # was ABSENT from the cache key. Append-only-when-on (and only when
    # column-extract is OFF, since extract-on already implies + salts the order
    # via col_key) keeps every flag-off key byte-identical.
    ord_key = (
        "|ord1"
        if resolve_column_order_mode() and not resolve_column_extract_mode()
        else ""
    )
    key_raw = (
        f"v{EXTRACT_CACHE_VERSION}|{fig_key}{vlm_key}{fuse_key}{gtail_key}{hints_key}{col_key}{ord_key}|"
        f"{pdf_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    )
    return hashlib.sha256(key_raw.encode()).hexdigest()[:24]


def extract_shared_cached(pdf_path: Path, cache_dir: Path | str | None = None) -> dict:
    """extract_shared with a disk cache keyed on (version, path, size, mtime).

    Useful for batch workflows where the same PDF is consulted many times
    (e.g., arXiv section pairs — one arXiv PDF produces 4-15 section pairs,
    each of which would otherwise re-run pypdfium2 + pdfplumber + pikepdf
    from scratch).

    Cache hits are ~instant; cache misses run the full extractor and write
    the result to disk. The cache key includes mtime so a modified PDF
    auto-invalidates, and ``EXTRACT_CACHE_VERSION`` so an extractor
    behavior change auto-invalidates too. Safe across processes — each
    worker reads/writes its own cache files independently; last writer
    wins on collision.
    """
    import json as _json

    pdf_path = Path(pdf_path)
    # Default cache_dir resolves under the SemantiK cache root (CWD-independent)
    # instead of the legacy CWD-relative ``data/extract_cache`` literal; an
    # explicit caller-supplied dir is honored unchanged. Byte-stable to the
    # historic ``<root>/data/extract_cache`` layout when no SEMANTIK_* env is set.
    cache_dir = (
        _semantik_paths.resolve_cache("extract_cache")
        if cache_dir is None
        else Path(cache_dir)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        key = _compute_extract_cache_key(pdf_path)
    except OSError:
        # If we can't stat the file, skip the cache — let extract_shared
        # raise its normal error.
        return extract_shared(pdf_path)

    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        try:
            return _json.loads(cache_path.read_text())
        except Exception:
            # Corrupt cache file — fall through to fresh extraction.
            pass

    shared = extract_shared(pdf_path)
    try:
        cache_path.write_text(_json.dumps(shared, default=str))
    except Exception:
        # Failing to cache is non-fatal.
        pass
    return shared


class EncryptedPDFError(Exception):
    pass


# ---------- pikepdf inspection ----------


def _pikepdf_inspect(pdf_path: Path) -> dict:
    """Metadata + tagged-PDF check + encryption check via pikepdf."""
    import pikepdf  # type: ignore

    info = {
        "producer": None,
        "creator": None,
        "title": None,
        "pdf_version": None,
        "is_encrypted": False,
        "is_tagged": False,
        "page_count": 0,
    }
    try:
        # pikepdf refuses encrypted PDFs by default; we catch the specific error.
        pdf = pikepdf.open(str(pdf_path))
    except pikepdf.PasswordError:
        info["is_encrypted"] = True
        return info
    except Exception as exc:
        raise ExtractionError(f"pikepdf failed to open {pdf_path}: {exc}") from exc

    try:
        dm = pdf.docinfo
        for dm_key, out_key in (
            ("/Producer", "producer"),
            ("/Creator", "creator"),
            ("/Title", "title"),
        ):
            val = dm.get(dm_key) if dm is not None else None
            if val is not None:
                info[out_key] = str(val)
        info["pdf_version"] = str(pdf.pdf_version)
        info["page_count"] = len(pdf.pages)

        # Tagged PDF detection via StructTreeRoot.
        root = pdf.Root
        if "/StructTreeRoot" in root.keys():
            info["is_tagged"] = True
            info["structtree_ref"] = True
    finally:
        pdf.close()

    return info


def _pikepdf_page_widgets(pdf_path: Path, page_num: int) -> list[dict]:
    """Read AcroForm widget annotations on one page via pikepdf.

    Widget /Rect is in PDF default user space (bottom-left origin, y up).
    We normalize to top-left origin so bboxes are comparable with
    pypdfium2 text-block bboxes throughout the rest of the pipeline.
    """
    import pikepdf  # type: ignore

    out: list[dict] = []
    try:
        pdf = pikepdf.open(str(pdf_path))
    except Exception:
        return out
    try:
        if page_num < 1 or page_num > len(pdf.pages):
            return out
        page = pdf.pages[page_num - 1]

        # Page height for y-flip (bottom-left -> top-left origin).
        mediabox = page.get("/MediaBox")
        if mediabox is not None and len(mediabox) >= 4:
            page_h = float(mediabox[3]) - float(mediabox[1])
        else:
            page_h = 792.0  # Letter default

        annots = page.get("/Annots")
        if annots is None:
            return out
        for a in annots:
            try:
                subtype = str(a.get("/Subtype", ""))
                if subtype != "/Widget":
                    continue
                rect = a.get("/Rect")
                if rect is not None and len(rect) >= 4:
                    x0, y0_bl, x1, y1_bl = (
                        float(rect[0]),
                        float(rect[1]),
                        float(rect[2]),
                        float(rect[3]),
                    )
                    # Flip to top-left origin so downstream bbox-in-bbox
                    # checks work against pypdfium2 / pdfplumber blocks.
                    bbox = [x0, page_h - y1_bl, x1, page_h - y0_bl]
                else:
                    bbox = None
                field_type = str(a.get("/FT", "")).lstrip("/") or None
                name = str(a.get("/T", "")) or None
                tooltip = str(a.get("/TU", "")) or None
                flags = int(a.get("/Ff", 0) or 0)
                out.append(
                    {
                        "bbox": bbox,
                        "field_type": field_type,
                        "name": name,
                        "tooltip": tooltip,
                        "flags": flags,
                    }
                )
            except Exception as exc:
                logger.debug(f"widget parse error page {page_num}: {exc}")
                continue
    finally:
        pdf.close()
    return out


def _pikepdf_page_structure(pdf_path: Path, page_num: int) -> list[dict]:
    """Extract structure-tree nodes for a tagged page. Best-effort."""
    # Walking StructTreeRoot properly is complex. For v1 we just flag the
    # document as tagged in metadata and defer full StructTree parsing to
    # a later pass — the per-block classifier can still benefit from the
    # flag even without the tree details.
    return []


# ---------- pypdfium2 text extraction ----------


def _pypdfium2_page_blocks(pdf_path: Path, page_num: int) -> tuple[dict, float, float]:
    """Return ({text_blocks, chars_hint}, page_width, page_height) from pypdfium2."""
    import pypdfium2 as pdfium  # type: ignore

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_num - 1]
        page_w, page_h = float(page.get_size()[0]), float(page.get_size()[1])
        textpage = page.get_textpage()

        blocks: list[dict] = []
        try:
            n_rects = textpage.count_rects()
        except Exception:
            n_rects = 0

        # `count_rects()` usually returns one rect per visual line, but on
        # LETTER-SPACED small-caps (common in legal captions/headings —
        # "MARKWAYNE MULLIN", "Before SYKES, LEE, …") it over-splits: the leading
        # full-size capital becomes its own rect AND the rect reading order
        # scrambles ("M" … "ARKWAYNE" emitted apart). Emitting one block per rect
        # and space-joining them downstream yields garbled "M ARKWAYNE M ULLIN".
        # Instead: collect the rects, cluster them into line-segments (same line
        # by y-overlap; split on a big horizontal gap so columns / inline math
        # stay separate), then re-extract each segment's text with
        # ``get_text_bounded(union_bbox)`` — which pypdfium2 returns in correct
        # reading order ("MARKWAYNE MULLIN"). Single-rect lines are unchanged
        # (the union is the rect itself).
        raw: list[tuple[float, float, float, float]] = []
        for i in range(n_rects):
            try:
                left, bottom, right, top = textpage.get_rect(i)  # (l, b, r, t)
            except Exception:
                continue
            raw.append((float(left), float(bottom), float(right), float(top)))

        # Cluster into lines by vertical-midpoint proximity (native coords: y
        # increases upward, so read high-y first).
        raw.sort(key=lambda r: (-(r[1] + r[3]) / 2.0, r[0]))
        lines: list[list[tuple[float, float, float, float]]] = []
        for r in raw:
            mid = (r[1] + r[3]) / 2.0
            h = max(1.0, r[3] - r[1])
            if lines:
                prev = lines[-1]
                pmid = sum((p[1] + p[3]) / 2.0 for p in prev) / len(prev)
                ph = max(1.0, max(p[3] for p in prev) - min(p[1] for p in prev))
                if abs(mid - pmid) <= 0.6 * min(h, ph):
                    prev.append(r)
                    continue
            lines.append([r])

        # Within each line, split into segments on a big horizontal gap (a column
        # gutter or an inline-math gap), then one block per contiguous segment.
        for line in lines:
            line.sort(key=lambda r: r[0])
            segments: list[list[tuple[float, float, float, float]]] = [[line[0]]]
            for r in line[1:]:
                prev_r = segments[-1][-1]
                gap = r[0] - prev_r[2]
                h = max(1.0, r[3] - r[1])
                if gap > 3.0 * h:
                    segments.append([r])
                else:
                    segments[-1].append(r)
            for seg in segments:
                L = min(s[0] for s in seg)
                B = min(s[1] for s in seg)
                R = max(s[2] for s in seg)
                T = max(s[3] for s in seg)
                try:
                    text = " ".join(textpage.get_text_bounded(L, B, R, T).split())
                except Exception:
                    text = ""
                if not text:
                    continue
                # Convert to (x0, y0, x1, y1) with y0=top, y1=bottom (top-left).
                bbox = (L, float(page_h - T), R, float(page_h - B))
                blocks.append(
                    {
                        "bbox": list(bbox),
                        "text": text,
                        "font_size": None,  # pypdfium2 doesn't expose size per rect cheaply
                        "font_name": None,
                        "is_bold": None,
                        "is_italic": None,
                        "confidence": 1.0,
                    }
                )
        return {"text_blocks": blocks}, page_w, page_h
    finally:
        pdf.close()


def _image_bbox_top_left(
    bounds_bl: tuple[float, float, float, float],
    page_height: float,
) -> list[float]:
    """Convert pypdfium2 ``PdfImage.get_bounds()`` PDF bottom-left
    ``(L, B, R, T)`` to the cascade's top-left ``[x0, y0, x1, y1]``.

    Y-flip: ``y0 = H - T`` (top edge), ``y1 = H - B`` (bottom edge),
    matching ``_pypdfium2_page_blocks`` (``page_h - T`` / ``page_h - B``)
    and what ``image_extract.py`` consumes. Pure (no PDF IO) so the Y-flip
    is unit-testable without pypdfium2.
    """
    left, bottom, right, top = (float(x) for x in bounds_bl)
    return [left, page_height - top, right, page_height - bottom]


def _pypdfium2_page_images(pdf_path: Path, page_num: int) -> list[dict]:
    """Return IMAGE page-objects on this page as
    ``[{bbox:[x0,y0,x1,y1] top-left, px_size:[w,h], index:int}, ...]``.

    Part F Stage A — mirrors :func:`_pypdfium2_page_blocks`. Enumerates
    every IMAGE page-object via ``page.get_objects`` filtered to
    ``raw.FPDF_PAGEOBJ_IMAGE`` and reads each one's bounds. pypdfium2's
    ``PdfImage.get_bounds()`` returns PDF bottom-left ``(L, B, R, T)``; we
    Y-FLIP to the top-left convention the rest of the cascade uses (and
    that :func:`_pypdfium2_page_blocks` produces) as
    ``(L, H - T, R, H - B)``. ``get_px_size()`` is captured so the
    spacer/rule filter can reject hairline page furniture.

    Fail-soft: any per-object error is skipped (one bad image must not
    abort a 39-image page); a page-level failure returns ``[]``.
    """
    import pypdfium2 as pdfium  # type: ignore
    import pypdfium2.raw as pdfium_raw  # type: ignore

    out: list[dict] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_num - 1]
        page_h = float(page.get_size()[1])
        try:
            objs = page.get_objects(filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"pypdfium2 get_objects page {page_num}: {exc}")
            return out
        for idx, obj in enumerate(objs):
            try:
                # PdfImage.get_bounds() → (left, bottom, right, top) in PDF
                # points with a bottom-left origin. Y-flip to top-left.
                bbox = _image_bbox_top_left(obj.get_bounds(), page_h)
                try:
                    px_w, px_h = obj.get_px_size()
                    px_size = [int(px_w), int(px_h)]
                except Exception:
                    px_size = [0, 0]
                out.append(
                    {
                        "bbox": bbox,
                        "px_size": px_size,
                        "index": idx,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — skip one bad object
                logger.debug(
                    f"pypdfium2 image obj {idx} page {page_num}: {exc}"
                )
                continue
        return out
    finally:
        pdf.close()


def _pypdfium2_render_page_to_image(pdf_path: Path, page_num: int, scale: float = 2.0):
    """Render a page to a PIL image for Tesseract."""
    import pypdfium2 as pdfium  # type: ignore

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_num - 1]
        bm = page.render(scale=scale)
        img = bm.to_pil()
        # Close the native bitmap + page handles explicitly — leaving them to GC
        # leaks pypdfium2 C-side buffers on the OCR render path (only the
        # PdfDocument was being closed).
        bm.close()
        page.close()
        return img
    finally:
        pdf.close()


# ---------- pdfplumber (tables + text with font info) ----------


def _word_in_table(w: dict, table_bboxes: list) -> bool:
    """True when a pdfplumber word's centroid falls inside any table bbox.

    Used to keep table cell words from SEEDING the column-band gutter detection
    (a wide multi-column data table would otherwise spawn a spurious gutter);
    the whole-line table skip in :func:`_finalize_line_block` still drops any
    line that lands inside a table, so nothing about the emitted blocks changes.
    """
    if not table_bboxes:
        return False
    mx = (float(w.get("x0", 0)) + float(w.get("x1", 0))) / 2.0
    my = (float(w.get("top", 0)) + float(w.get("bottom", 0))) / 2.0
    return any(
        bx0 <= mx <= bx1 and by0 <= my <= by1 for (bx0, by0, bx1, by1) in table_bboxes
    )


def _finalize_line_block(line: list, table_bboxes: list) -> dict | None:
    """Turn one bucketed physical line into a text-block dict (or None to skip).

    Byte-identical to the historic per-line body of :func:`_pdfplumber_page`:
    x-sort, styled-drop-cap first-letter rejoin, bbox/text/font computation, and
    the in-table line skip. Factored out so the single-width (legacy) and
    column-aware assembly paths share one emitter.
    """
    if not line:
        return None
    line.sort(key=lambda w: float(w.get("x0", 0)))
    # Rejoin styled drop-cap first letters that a font change split off their
    # word body ("S" + "ubtraction" -> "Subtraction").
    line = _merge_split_first_letters(line)
    x0 = min(float(w.get("x0", 0)) for w in line)
    x1 = max(float(w.get("x1", 0)) for w in line)
    top = min(float(w.get("top", 0)) for w in line)
    bot = max(float(w.get("bottom", 0)) for w in line)
    text = " ".join(w.get("text", "") for w in line).strip()
    if not text:
        return None
    # Skip lines that fall entirely inside a detected table bbox.
    mid_x = (x0 + x1) / 2
    mid_y = (top + bot) / 2
    if any(
        bx0 <= mid_x <= bx1 and by0 <= mid_y <= by1 for (bx0, by0, bx1, by1) in table_bboxes
    ):
        return None
    # Dominant font size (median).
    sizes = [float(w.get("size") or 0) for w in line if w.get("size")]
    font_name = line[0].get("fontname")
    size = sorted(sizes)[len(sizes) // 2] if sizes else None
    is_bold = bool(font_name and "Bold" in font_name)
    is_italic = bool(font_name and any(t in font_name for t in ("Italic", "Oblique")))
    return {
        "bbox": [x0, top, x1, bot],
        "text": text,
        "font_size": size,
        "font_name": font_name,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "confidence": 1.0,
    }


def _bucket_lines_by_top(words: list) -> list[list[dict]]:
    """Bucket words (already sorted by ``(round(top), x0)``) into physical lines.

    A word joins the current line when its ``top`` is within 3pt of the line's
    first word; otherwise it starts a new line. Byte-identical to the historic
    inline bucketing in :func:`_pdfplumber_page`.
    """
    lines: list[list[dict]] = []
    for w in words:
        top = float(w.get("top", 0))
        if not lines or abs(top - float(lines[-1][0].get("top", 0))) > 3:
            lines.append([w])
        else:
            lines[-1].append(w)
    return lines


def _assemble_text_blocks_singlewidth(words: list, table_bboxes: list) -> list[dict]:
    """Legacy full-page-width line assembly (SEMANTIK_COLUMN_EXTRACT off).

    Buckets every word into a full-width physical line then x-sorts + space-joins
    — so multi-column words sharing a baseline fuse into one string. Kept
    byte-identical to the pre-column-extract behaviour.
    """
    blocks: list[dict] = []
    for line in _bucket_lines_by_top(words):
        blk = _finalize_line_block(line, table_bboxes)
        if blk is not None:
            blocks.append(blk)
    return blocks


def _assemble_text_blocks_columnaware(
    words: list, table_bboxes: list, page_w: float
) -> list[dict]:
    """Column-band line assembly (SEMANTIK_COLUMN_EXTRACT on).

    Segments the page into column bands BEFORE line assembly so each column's
    text stays contiguous (fixes the Fed-Register/CFR/W-9 cross-column fusion
    salad where a full-width baseline sort glues ``col1 col2 col3`` into one
    string). Steps:

    1. **Detect columns from COLUMN-START seeds** via the REUSED
       :func:`reading_order.column_ids_for_x0s` single-linkage gutter detector
       (``single_column_guard=True``). A real multi-column page's word
       left-edges densely fill the whole page width (interior words start at
       every x), so seeding the detector on EVERY word's x0 finds no gutter;
       instead only *column-start* words seed — the first word of each physical
       line and any word opening a new column (preceded by a gutter-sized
       horizontal gap). Those x0s cluster tightly at each column's left margin,
       which is what single-linkage needs. Table-cell words never seed (a wide
       data table would otherwise spawn a spurious gutter).
    2. When the guard collapses to ONE column (genuine single-column page), fall
       straight through to the byte-identical single-width path (second safety
       net atop the default-off flag + append-only cache salt).
    3. **Assign every word to a band by its left-edge** — the column whose left
       margin is the greatest ``<= word.x0``. This is deliberately NOT the
       detector's left-edge *midpoint* split (which bisects each column's own
       text width and misassigns wide interior words); a word belongs to the
       column it starts within.
    4. Bucket words into full-width physical lines, then per line: a line whose
       words all fall in one band joins it; a multi-column line is either a
       **bridging full-width header** (a word STRADDLES an interior column
       left-edge — i.e. sits in the gutter x-range) kept as ONE block leading
       column 0, or a genuine cross-gutter baseline SPLIT into per-column
       sub-lines.
    5. Emit column 0 (top->bottom), then column 1, ... .
    """
    if not words:
        return []
    if page_w is None or page_w <= 0:
        page_w = _DEFAULT_PAGE_WIDTH

    phys_lines = _bucket_lines_by_top(words)
    in_table = (
        {id(w): _word_in_table(w, table_bboxes) for w in words}
        if table_bboxes
        else None
    )

    # --- 1. column-start seeds -> detect columns via the SHARED gutter engine
    # (reading_order.column_edges_from_lines — one engine, two callers; D2). The
    # seed_masks arm excludes table-cell words from SEEDING while the helper
    # still advances the previous-word x1 chain through them, keeping the
    # gap-to-previous seeding byte-identical to the historic inline loop. ---
    lines_xs: list[list[tuple[float, float]]] = []
    seed_masks: list[list[bool]] = []
    for line in phys_lines:
        ln_sorted = sorted(line, key=lambda w: float(w.get("x0", 0)))
        lines_xs.append(
            [(float(w.get("x0", 0)), float(w.get("x1", 0))) for w in ln_sorted]
        )
        seed_masks.append(
            [not (in_table and in_table.get(id(w))) for w in ln_sorted]
        )
    edges = column_edges_from_lines(lines_xs, page_w, seed_masks=seed_masks)
    if edges is None:
        # No seeds / genuine single-column page / defensive seedless band ->
        # identical to the full-width path.
        return _assemble_text_blocks_singlewidth(words, table_bboxes)
    ncols = len(edges)

    def _col_of(x0: float) -> int:
        c = 0
        for j in range(1, ncols):
            if x0 >= edges[j]:
                c = j
            else:
                break
        return c

    # --- 2. assign each physical line (or its split sub-lines) to a band ---
    col_lines: dict[int, list[list[dict]]] = {i: [] for i in range(ncols)}
    for line in phys_lines:
        ln_sorted = sorted(line, key=lambda w: float(w.get("x0", 0)))
        cols_in_line = {_col_of(float(w.get("x0", 0))) for w in ln_sorted}
        if len(cols_in_line) == 1:
            col_lines[next(iter(cols_in_line))].append(ln_sorted)
            continue
        # Multi-column physical line: full-width header vs genuine split. A
        # header word STRADDLES an interior column left-edge (x0 < Lj < x1 — it
        # sits in the gutter x-range); a genuine multi-column baseline has no
        # word crossing the gutter (col-c text ends before Lj, col-(c+1) starts
        # at Lj).
        bridges = any(
            float(w.get("x0", 0)) < edges[j] < float(w.get("x1", 0))
            for w in ln_sorted
            for j in range(1, ncols)
        )
        if bridges:
            col_lines[0].append(ln_sorted)  # keep ONE block, leading column 0
        else:
            by_col: dict[int, list[dict]] = {}
            for w in ln_sorted:
                by_col.setdefault(_col_of(float(w.get("x0", 0))), []).append(w)
            for c, ws in by_col.items():
                col_lines[c].append(ws)

    # --- 3. emit column 0 top->bottom, then column 1, ... ---
    blocks: list[dict] = []
    for c in range(ncols):
        lines_c = col_lines[c]
        lines_c.sort(key=lambda ln: min(float(w.get("top", 0)) for w in ln))
        for line in lines_c:
            blk = _finalize_line_block(line, table_bboxes)
            if blk is not None:
                blocks.append(blk)
    return blocks


def _split_tesseract_lines_by_column(lines: dict, page_w: float) -> dict:
    """Column-aware re-key of Tesseract's ``(block, par, line)`` word groups.

    The OCR analog of :func:`_assemble_text_blocks_columnaware`, sharing the
    same gutter engine (D2/D3). Each Tesseract line group carries parallel
    ``words/heights/confs/lefts/rights/tops/bottoms`` arrays. Bboxes here are
    in image-PIXEL space, so ``page_w`` is the image width (the same space —
    this is what makes the seeding correct where ``_merge_page`` was not).

    Behavior:
      * Build per-group ``(left, right)`` word tuples sorted by left and seed
        the SHARED :func:`reading_order.column_edges_from_lines`; ``None`` (the
        single-column guard collapsed the split, or no seeds) → return ``lines``
        UNCHANGED (byte-identical — a single-column scan is never re-keyed).
      * Otherwise, per group: a group with a word STRADDLING an interior column
        edge (``left < edge < right``) is a full-width header → whole group
        keyed ``(0, *old_key)``; else words are split by ``_col_of(left)`` into
        per-column sub-groups keyed ``(col, *old_key)`` (parallel arrays rebuilt
        in original within-line order so text join + bbox stay faithful; word
        multiset conserved).
      * Deterministic output-dict insertion order: sorted by
        ``(col, min(tops), min(lefts))``.

    Deterministic, CPU-only. No table detection on the OCR path (no seed mask).
    """
    lines_xs: list[list[tuple[float, float]]] = []
    group_items = list(lines.items())
    for _key, ln in group_items:
        order = sorted(range(len(ln["lefts"])), key=lambda i: float(ln["lefts"][i]))
        lines_xs.append(
            [(float(ln["lefts"][i]), float(ln["rights"][i])) for i in order]
        )
    edges = column_edges_from_lines(lines_xs, page_w)
    if edges is None:
        return lines
    ncols = len(edges)

    def _col_of(x0: float) -> int:
        c = 0
        for j in range(1, ncols):
            if x0 >= edges[j]:
                c = j
            else:
                break
        return c

    _ARRAY_KEYS = ("words", "heights", "confs", "lefts", "rights", "tops", "bottoms")

    def _empty_group() -> dict:
        return {k: [] for k in _ARRAY_KEYS}

    def _append_word(dst: dict, src: dict, i: int) -> None:
        for k in _ARRAY_KEYS:
            dst[k].append(src[k][i])

    # (new_key, group, col, min_top, min_left) so we can order deterministically.
    built: list[tuple[tuple, dict, int, float, float]] = []
    for key, ln in group_items:
        n = len(ln["lefts"])
        # Full-width header: any word straddles an interior column edge.
        straddles = any(
            float(ln["lefts"][i]) < edges[j] < float(ln["rights"][i])
            for i in range(n)
            for j in range(1, ncols)
        )
        if straddles:
            new_key = (0, *key)
            built.append(
                (
                    new_key,
                    ln,
                    0,
                    min((float(t) for t in ln["tops"]), default=0.0),
                    min((float(x) for x in ln["lefts"]), default=0.0),
                )
            )
            continue
        buckets: dict[int, dict] = {}
        for i in range(n):
            c = _col_of(float(ln["lefts"][i]))
            _append_word(buckets.setdefault(c, _empty_group()), ln, i)
        for c, grp in buckets.items():
            new_key = (c, *key)
            built.append(
                (
                    new_key,
                    grp,
                    c,
                    min((float(t) for t in grp["tops"]), default=0.0),
                    min((float(x) for x in grp["lefts"]), default=0.0),
                )
            )

    built.sort(key=lambda t: (t[2], t[3], t[4]))
    out: dict = {}
    for new_key, grp, _c, _mt, _ml in built:
        out[new_key] = grp
    return out


def _pdfplumber_page(pdf_path: Path, page_num: int) -> dict:
    """Extract text blocks + tables via pdfplumber. Slower than pypdfium2
    but adds table detection and actual font metadata."""
    import pdfplumber  # type: ignore

    out = {"text_blocks": [], "tables": []}
    try:
        pdf = pdfplumber.open(str(pdf_path))
    except Exception as exc:
        logger.debug(f"pdfplumber open failed for {pdf_path}: {exc}")
        return out
    try:
        if page_num < 1 or page_num > len(pdf.pages):
            return out
        page = pdf.pages[page_num - 1]

        # Tables first so we can avoid double-counting their text.
        try:
            found_tables = page.find_tables()
        except Exception as exc:
            logger.debug(f"pdfplumber find_tables page {page_num}: {exc}")
            found_tables = []

        table_bboxes: list[tuple] = []
        for t in found_tables:
            try:
                rows = t.extract()
            except Exception:
                rows = None
            if not rows:
                continue
            bbox = t.bbox  # (x0, top, x1, bottom)
            table_bboxes.append(bbox)
            out["tables"].append(
                {
                    "bbox": [float(x) for x in bbox],
                    "rows": rows,
                }
            )

        # Extract line-level text via extract_words + group by line top.
        #
        # ``x_tolerance_ratio`` scales the word-break gap threshold by font
        # size so spaces are recovered on PDFs whose inter-word gaps are
        # smaller than pdfplumber's fixed 3pt default — common in LaTeX
        # output, which kerns words apart with NO space glyph. Without it,
        # ``extract_words`` returns whole lines as one token
        # ("frameworkforderivingvariousEPIs…"), which then poisons every
        # text feature downstream (heading detection, prose, tables). The
        # ratio (vs a fixed x_tolerance) avoids over-splitting large-font
        # titles. See Plans/11 Stage 3.
        words = []
        _word_kwargs = dict(
            keep_blank_chars=False,
            use_text_flow=True,
            x_tolerance_ratio=_X_TOLERANCE_RATIO,
            extra_attrs=["fontname", "size"],
        )
        try:
            words = page.extract_words(**_word_kwargs)
        except TypeError:
            # Older pdfplumber without x_tolerance_ratio — retry without it.
            _word_kwargs.pop("x_tolerance_ratio", None)
            try:
                words = page.extract_words(**_word_kwargs)
            except Exception as exc:
                logger.debug(f"pdfplumber extract_words page {page_num}: {exc}")
                words = []
        except Exception as exc:
            logger.debug(f"pdfplumber extract_words page {page_num}: {exc}")
            words = []

        # Bucket into lines by top-coordinate (within 3pt), then space-join.
        # Words are sorted by (round(top), x0) first so the bucketing is stable.
        words.sort(key=lambda w: (round(float(w.get("top", 0)), 0), float(w.get("x0", 0))))
        # SEMANTIK_COLUMN_EXTRACT (default OFF): when on, segment the page into
        # column bands BEFORE line assembly so a multi-column page keeps each
        # column's text contiguous instead of fusing columns into one string
        # (the Fed-Register/CFR/W-9 salad). OFF -> the legacy full-page-width
        # loop, byte-identical.
        if resolve_column_extract_mode():
            page_w = float(getattr(page, "width", 0) or _DEFAULT_PAGE_WIDTH)
            out["text_blocks"].extend(
                _assemble_text_blocks_columnaware(words, table_bboxes, page_w)
            )
        else:
            out["text_blocks"].extend(
                _assemble_text_blocks_singlewidth(words, table_bboxes)
            )
    finally:
        pdf.close()
    return out


# ---------- tesseract OCR ----------

# pypdfium2 render scale for the Tesseract OCR raster. A pypdfium2 page renders
# at 72 DPI * scale, so the default 2.0 is ~144 DPI on a 612x792pt page —
# adequate for typical body text but MARGINAL for small (~10pt) print on a
# rasterized/image-only scan, where OCR wants ~300 DPI (scale ~4.0). The env
# knob lets an operator raise the raster DPI for such a corpus without a code
# change. NB: the scale is NOT part of the extract disk-cache key
# (``EXTRACT_CACHE_VERSION`` / ``fig_key``), so changing it does NOT invalidate
# a prior OCR extraction — clear the extract cache (or use fresh inputs) after
# a flip.
_OCR_RENDER_SCALE_ENV = "SEMANTIK_OCR_RENDER_SCALE"
_DEFAULT_OCR_RENDER_SCALE = 2.0


def resolve_ocr_render_scale() -> float:
    """Parse-with-fallback ``SEMANTIK_OCR_RENDER_SCALE`` (default ``2.0``).

    Returns the pypdfium2 render scale used to raster a page for Tesseract OCR
    (72 DPI * scale). Read at call time (not import time) so tests / Docker can
    flip it without a re-import, mirroring ``_detect_figures_enabled`` /
    ``resolve_structure_review_temperature``. Garbage / non-float / non-finite /
    non-positive values fall back to ``2.0`` (a run never crashes on a malformed
    flag; a <= 0 scale would render an empty raster)."""
    raw = os.environ.get(_OCR_RENDER_SCALE_ENV)
    if not raw:
        return _DEFAULT_OCR_RENDER_SCALE
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_OCR_RENDER_SCALE
    if val <= 0 or val != val or val in (float("inf"), float("-inf")):
        return _DEFAULT_OCR_RENDER_SCALE
    return val


# Freeform Tesseract CLI config fragment threaded into ``image_to_data`` (e.g.
# ``--oem 1 -c preserve_interword_spaces=1``). Like ``SEMANTIK_OCR_RENDER_SCALE``
# this is NOT part of the extract disk-cache key (``EXTRACT_CACHE_VERSION`` /
# ``fig_key``), so flipping it does NOT invalidate a prior OCR extraction —
# clear the extract cache (or use fresh inputs) after a flip. Deliberately kept
# out of the cache key (mirrors the render-scale precedent) to avoid
# invalidating every existing cache; the A/B script isolates caches itself.
_TESSERACT_CONFIG_ENV = "SEMANTIK_TESSERACT_CONFIG"
_TESSERACT_USER_WORDS_ENV = "SEMANTIK_TESSERACT_USER_WORDS"

# Module-level dedup for the user-words missing-file warning so a 400-page doc
# does not log the same warning 400 times (the resolver is called per page).
_USER_WORDS_WARNED: set[str] = set()


def resolve_tesseract_config() -> str:
    """Parse-with-fallback ``SEMANTIK_TESSERACT_CONFIG`` (default ``""``).

    Returns a Tesseract CLI config fragment passed to
    ``pytesseract.image_to_data(config=…)``. Read at CALL time (mirrors
    ``resolve_ocr_render_scale``) so tests / Docker can flip it without a
    re-import. Unset / blank → ``""`` (byte-identical default: the call site
    omits the ``config=`` kwarg entirely). Otherwise the value is stripped and
    any embedded newlines are collapsed to spaces (it is a single CLI fragment,
    not a multi-line string). A freeform string, so parse-with-fallback is
    trivially total — a run never crashes on a malformed flag."""
    raw = os.environ.get(_TESSERACT_CONFIG_ENV)
    if not raw or not raw.strip():
        return ""
    return " ".join(raw.split())


def resolve_tesseract_user_words() -> Path | None:
    """Parse-with-fallback ``SEMANTIK_TESSERACT_USER_WORDS`` (default ``None``).

    Returns the path to a Tesseract ``--user-words`` file (a domain-vocabulary
    lexicon, one word per line) or ``None`` when unset. Read at CALL time.
    Unset / blank → ``None``. A set-but-missing file logs a de-duplicated
    warning (once per distinct path) and returns ``None`` — a bad path degrades
    to the byte-identical no-user-words behaviour rather than crashing the run
    (parse-with-fallback; never raises)."""
    raw = os.environ.get(_TESSERACT_USER_WORDS_ENV)
    if not raw or not raw.strip():
        return None
    p = Path(raw.strip()).expanduser()
    if not p.is_file():
        key = str(p)
        if key not in _USER_WORDS_WARNED:
            _USER_WORDS_WARNED.add(key)
            logger.warning(
                "SEMANTIK_TESSERACT_USER_WORDS points at a non-existent file "
                "(%s); ignoring (OCR runs without a user-words lexicon).",
                key,
            )
        return None
    return p


def _tesseract_page_blocks(pdf_path: Path, page_num: int) -> dict:
    """OCR one page via Tesseract. Rendered via pypdfium2 (Apache-2).

    The raster DPI is operator-tunable via ``SEMANTIK_OCR_RENDER_SCALE``
    (default 2.0 ≈ 144 DPI); raise it for small-print image-only scans.
    """
    import pytesseract

    scale = resolve_ocr_render_scale()
    image = _pypdfium2_render_page_to_image(pdf_path, page_num, scale=scale)

    # Optional operator-tunable Tesseract config (SEMANTIK_TESSERACT_CONFIG) +
    # a domain-vocabulary user-words file (SEMANTIK_TESSERACT_USER_WORDS). When
    # BOTH are unset the call shape is byte-identical to the historic
    # ``image_to_data(image, output_type=…)`` — no ``config=`` kwarg is passed.
    config = resolve_tesseract_config()
    user_words = resolve_tesseract_user_words()
    if user_words is not None:
        # pytesseract shlex.split()s the config string (POSIX), so the quoting
        # survives a path containing spaces; composes with SEMANTIK_TESSERACT_CONFIG.
        config = (config + f" --user-words {shlex.quote(str(user_words))}").strip()
    kwargs: dict[str, Any] = {"output_type": pytesseract.Output.DICT}
    if config:
        kwargs["config"] = config
    data = pytesseract.image_to_data(image, **kwargs)
    page_w = float(image.width)
    page_h = float(image.height)

    # Group words -> lines using (block_num, par_num, line_num).
    lines: dict[tuple, dict] = {}
    for j, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][j], data["par_num"][j], data["line_num"][j])
        ln = lines.setdefault(
            key,
            {
                "words": [],
                "heights": [],
                "confs": [],
                "lefts": [],
                "rights": [],
                "tops": [],
                "bottoms": [],
            },
        )
        ln["words"].append(word)
        ln["heights"].append(data["height"][j])
        conf = data.get("conf", [80])
        ln["confs"].append(float(conf[j] if j < len(conf) else 80))
        left = data["left"][j]
        top = data["top"][j]
        w = data["width"][j]
        h = data["height"][j]
        ln["lefts"].append(left)
        ln["rights"].append(left + w)
        ln["tops"].append(top)
        ln["bottoms"].append(top + h)

    # Column-aware re-key when SEMANTIK_COLUMN_EXTRACT is on (prong 1): split a
    # Tesseract line that fused two columns into per-column line groups so the
    # merged stream is not glued across the gutter. page_w is the IMAGE width
    # (same pixel space as the word coordinates). Single-column scans collapse
    # via the guard → byte-identical grouping (see helper).
    if resolve_column_extract_mode() and lines:
        lines = _split_tesseract_lines_by_column(lines, page_w)

    blocks: list[dict] = []
    for ln in lines.values():
        heights = sorted(ln["heights"])
        est_size = heights[len(heights) // 2] if heights else 0.0
        conf = sum(ln["confs"]) / len(ln["confs"]) if ln["confs"] else 0.0
        blocks.append(
            {
                "bbox": [
                    float(min(ln["lefts"])),
                    float(min(ln["tops"])),
                    float(max(ln["rights"])),
                    float(max(ln["bottoms"])),
                ],
                "text": " ".join(ln["words"]),
                "font_size": float(est_size),
                "font_name": None,
                "is_bold": None,
                "is_italic": None,
                "confidence": conf / 100.0,
            }
        )
    # Tesseract's bboxes are in image-pixel space, not PDF-point space —
    # we expose page_w/page_h at the image scale too, so downstream
    # normalization (fractions of page) still works.
    return {"text_blocks": blocks, "page_width": page_w, "page_height": page_h}


# ---------- per-page orchestration ----------


def _extract_page(
    pdf_path: Path, page_num: int, meta: dict, *, defer_finalize: bool = False
) -> dict:
    page: dict[str, Any] = {"page_num": page_num, "sources_used": []}

    # pypdfium2 first (primary text extractor)
    try:
        pfx, page_w, page_h = _pypdfium2_page_blocks(pdf_path, page_num)
        page["pypdfium2"] = pfx
        page["width"] = page_w
        page["height"] = page_h
        page["sources_used"].append("pypdfium2")
    except Exception as exc:
        page["pypdfium2"] = {"text_blocks": [], "error": str(exc)}
        page["width"] = 612.0
        page["height"] = 792.0

    pfx_blocks = page.get("pypdfium2", {}).get("text_blocks", [])
    pypdfium2_text = " ".join(b["text"] for b in pfx_blocks)
    is_text_ok = _text_layer_quality_ok(pypdfium2_text)

    # pdfplumber (tables + alternative text — always runs on text-layer pages)
    if is_text_ok:
        try:
            ppx = _pdfplumber_page(pdf_path, page_num)
            page["pdfplumber"] = ppx
            page["sources_used"].append("pdfplumber")
        except Exception as exc:
            page["pdfplumber"] = {"text_blocks": [], "tables": [], "error": str(exc)}

    # Tesseract: only if we don't have a clean text layer.
    if not is_text_ok:
        try:
            tes = _tesseract_page_blocks(pdf_path, page_num)
            page["tesseract"] = tes
            # Tesseract image pixel dims supersede the PDF point dims for
            # bbox normalization — store them separately.
            page["tesseract_width"] = tes.pop("page_width", None)
            page["tesseract_height"] = tes.pop("page_height", None)
            page["sources_used"].append("tesseract")
        except Exception as exc:
            page["tesseract"] = {"text_blocks": [], "error": str(exc)}

    # VLM extraction source (P0) — a FOURTH source on the OCR-routed (scanned)
    # pages only, ADDITIVE to tesseract (never replacing it). Runs immediately
    # after the tesseract block, gated on SEMANTIK_VLM_EXTRACT AND ``not
    # is_text_ok`` so it lands exactly on the scanned pages the OCR arm serves.
    # The P0 fusion-free invariant: this stamps ``page["vlm"]`` but NEVER enters
    # ``_merge_page`` (the merged stream the BERTs consume stays tesseract-only
    # until the P1 DP fusion lands) — so flag-on output through the council is
    # byte-identical and the structural anti-hallucination tripwire is never
    # bypassed. Flag-off → no ``vlm`` key anywhere (byte-identical).
    if resolve_vlm_extract_mode() and not is_text_ok:
        try:
            # ``extract_page_markdown`` calls ``seat.require_ready()`` first, so
            # a misconfigured hosted seat fails loud (VLMSeatError) before any
            # render or POST.
            page["vlm"] = _vlm_extract.extract_page_markdown(
                pdf_path,
                page_num,
                seat=resolve_vlm_seat(),
                render_scale=resolve_ocr_render_scale(),
                timeout=resolve_vlm_timeout(),
            )
            page["sources_used"].append("vlm")
        except _vlm_extract.VlmExtractError as exc:
            # TRANSIENT endpoint failure (timeout / conn / 5xx / 429, after
            # bounded retries) → degrade THIS page's vlm source alone; tesseract
            # stays authoritative (the additive contract). PERMANENT failures
            # (400/401/403 / malformed response) propagate (fail-loud).
            if not exc.transient:
                raise
            page["vlm"] = {"markdown": "", "text_blocks": [], "error": str(exc)}

    # Image page-objects (Part F) — only when SEMANTIK_DETECT_FIGURES is on.
    # Fail-soft: a per-page extraction failure leaves an empty list so a
    # figure-bearing document never aborts conversion. Absent / empty when
    # the flag is off → byte-stable (no ``images`` key downstream consumers
    # read).
    if _detect_figures_enabled():
        try:
            page["images"] = _pypdfium2_page_images(pdf_path, page_num)
        except Exception as exc:  # noqa: BLE001 — surface nothing, degrade
            logger.debug(f"image extraction page {page_num}: {exc}")
            page["images"] = []

    # pikepdf: widgets + structure nodes (always check for widgets; cheap).
    try:
        widgets = _pikepdf_page_widgets(pdf_path, page_num)
    except Exception:
        widgets = []
    pike: dict[str, Any] = {"widgets": widgets}
    if meta.get("is_tagged"):
        pike["structure_nodes"] = _pikepdf_page_structure(pdf_path, page_num)
    page["pikepdf"] = pike
    if widgets or meta.get("is_tagged"):
        page["sources_used"].append("pikepdf")

    # VLM fusion (P1) — deterministically fuse the P0 ``page["vlm"]`` markdown
    # source onto the tesseract line blocks BEFORE the merge, so features /
    # FBs / council consume the fused text transparently. No-op (byte-identical)
    # unless SEMANTIK_VLM_FUSION is on AND both a tesseract source and a P0
    # ``page["vlm"]`` source exist (P0 = SEMANTIK_VLM_EXTRACT).
    # Defer fusion + merge to the document-level second loop when the caller
    # asked (Defect 1 — the cross-page furniture strip must run over every
    # page's VLM lines BEFORE any page is fused). Stash the merge's is_text_ok
    # for ``_finalize_page``. Non-deferred (default) → the historic inline
    # fuse-and-merge, byte-identical.
    if defer_finalize:
        page["_is_text_ok"] = is_text_ok
        return page

    _apply_vlm_fusion(page)

    # Merge step.
    page["merged"] = _merge_page(page, is_text_ok)
    return page


def _apply_vlm_fusion(page: dict) -> None:
    """Fuse the P0 ``page["vlm"]`` markdown source onto the tesseract blocks.

    Rewrites ``page["tesseract"]["text_blocks"]`` IN PLACE with the fused
    blocks (VLM text on the tesseract bbox union + a ``fusion`` provenance key
    + ``fusion_sim`` + the ``vlm_md`` / ``vlm_coverage`` the P2 hint channel
    reads), and records the per-page divergence tripwire
    (``page["vlm_divergence"]`` + ``page["vlm_fusion_stats"]``).

    No-op (byte-identical) unless :func:`resolve_vlm_fusion_mode` is on AND
    both a tesseract source WITH blocks and a P0 ``page["vlm"]`` source WITH
    lines exist — so a run with P0 off (no ``vlm`` key) or the flag off is
    unchanged. Fusion must NEVER abort extraction: any failure degrades to the
    unfused tesseract blocks (fail-soft, mirroring the P0 additive contract).
    """
    if not resolve_vlm_fusion_mode():
        return
    tess_blocks = page.get("tesseract", {}).get("text_blocks")
    vlm = page.get("vlm")
    if not tess_blocks or not vlm:
        return
    vlm_lines = [
        b.get("text", "")
        for b in vlm.get("text_blocks", [])
        if (b.get("text") or "").strip()
    ]
    if not vlm_lines:
        return
    try:
        from .vlm_fusion import fuse_page

        page_w = float(page.get("tesseract_width") or page.get("width") or 0.0)
        page_h = float(page.get("tesseract_height") or page.get("height") or 0.0)
        fused, stats = fuse_page(tess_blocks, vlm_lines, (page_w, page_h))
        page["tesseract"]["text_blocks"] = fused
        page["vlm_fusion_stats"] = stats
        page["vlm_divergence"] = stats.get("divergence", 0.0)
    except Exception as exc:  # noqa: BLE001 — fusion never aborts extraction
        logger.debug(f"vlm fusion failed page {page.get('page_num')}: {exc}")


# ---------------------------------------------------------------------------
# P2 — VLM extraction source + NON-AUTHORITATIVE structural hints.
#
# Phase-4 (plan scan-conversion-improvements-2026-07.md) fuses a Qwen2.5-VL
# per-page markdown transcription as a FOURTH extraction source (P0/P1 — the
# DP aligner in ``data/structure_align`` binds VLM lines onto Tesseract
# bboxes). THIS module owns the P2 hint side-channel: the merged block dict
# gains a ``vlm_hint`` — a NON-AUTHORITATIVE per-block recommendation minted
# from the aligned VLM markdown shape — consumed by ``structure_graph`` /
# ``deterministic_structure`` exactly like a council recommendation (heading
# CORROBORATION only; never a kind authority). All gated + byte-identical
# when the flags are off.
#
# Seat is PROVIDER-AGNOSTIC (OpenAI-compatible), mirroring the
# ``SEMANTIK_SPECIALIST_*`` env-resolution chain in ``endpoint_runtime.py`` —
# a new provider is env config, never a subclass. Default model
# ``qwen2.5vl:7b`` on the local ollama endpoint (Apache-2.0; see
# ``docs/LICENSING.md``).
# ---------------------------------------------------------------------------

_VLM_TRUTHY = frozenset({"1", "true", "yes", "on"})
_VLM_EXTRACT_ENV = "SEMANTIK_VLM_EXTRACT"
_VLM_STRUCT_HINTS_ENV = "SEMANTIK_VLM_STRUCT_HINTS"
_VLM_PROVIDER_ENV = "SEMANTIK_VLM_PROVIDER"
_VLM_BASE_URL_ENV = "SEMANTIK_VLM_BASE_URL"
_VLM_API_KEY_ENV = "SEMANTIK_VLM_API_KEY"
_VLM_MODEL_ENV = "SEMANTIK_VLM_MODEL"
_VLM_TIMEOUT_ENV = "SEMANTIK_VLM_TIMEOUT_SECONDS"

# Provider values that explicitly mean "stay local" (mirrors
# ``runtime._LOCAL_PROVIDER_VALUES``; a local seat needs no credential).
_VLM_LOCAL_PROVIDER_VALUES = frozenset({"", "local", "gguf", "ollama"})
# Sane literal defaults — the local ollama Qwen2.5-VL-7B seat (Apache-2.0).
_DEFAULT_VLM_MODEL = "qwen2.5vl:7b"
_DEFAULT_VLM_BASE_URL = "http://localhost:11434"
# Cold-load headroom: 64 s cold + 12 s/page warm (tier-0 probe, ollama 0.31.1).
_DEFAULT_VLM_TIMEOUT_SECONDS = 180.0


def resolve_vlm_extract_mode() -> bool:
    """Whether the VLM extraction source (P0) is enabled (default OFF).

    Reads ``SEMANTIK_VLM_EXTRACT``. Truthy (``1``/``true``/``yes``/``on``,
    case-insensitive) → the VLM markdown transcription is fused as a fourth
    extraction source and the P2 hint channel is live. Unset / blank / falsey
    / garbage → off (byte-identical: no VLM source, no hints). Read at CALL
    time (mirrors ``_detect_figures_enabled`` / ``resolve_ocr_render_scale``)
    so tests / Docker can flip it without a re-import. Parse-with-fallback."""
    return (os.environ.get(_VLM_EXTRACT_ENV) or "").strip().lower() in _VLM_TRUTHY


def resolve_vlm_struct_hints_mode() -> bool:
    """Whether VLM structural hints are ATTACHED + CONSUMED (default OFF).

    Reads ``SEMANTIK_VLM_STRUCT_HINTS`` and is **subordinate to**
    ``SEMANTIK_VLM_EXTRACT`` — returns True only when BOTH are truthy (there
    is no VLM markdown to mint a hint from unless the extract source ran), so
    the fusion can ship text-only with hints independently revertible (mirrors
    the SPLIT_FUSED_SECTION_TITLES / PROMOTE_SECTION_HEADINGS dual-gate
    precedent). Unset / blank / falsey / garbage → off (byte-identical).
    Parse-with-fallback. Read at CALL time."""
    hints = (os.environ.get(_VLM_STRUCT_HINTS_ENV) or "").strip().lower() in _VLM_TRUTHY
    return hints and resolve_vlm_extract_mode()


def resolve_vlm_provider() -> str:
    """Normalized VLM seat provider (default ``"local"``).

    Reads ``SEMANTIK_VLM_PROVIDER``. Blank / ``local`` / GGUF / ``ollama``
    aliases → ``"local"``; any other non-empty value is the lower-cased,
    stripped provider name (a new provider is env config, never a subclass —
    the seat serves any OpenAI-compatible endpoint incl. Spark later).
    Parse-with-fallback (garbage → treated as a provider name; a non-local
    provider without a credential fails loud at dispatch, not here)."""
    v = (os.environ.get(_VLM_PROVIDER_ENV) or "").strip().lower()
    return "local" if v in _VLM_LOCAL_PROVIDER_VALUES else v


def resolve_vlm_base_url() -> str:
    """VLM seat base URL (default ``http://localhost:11434``).

    Resolution: ``SEMANTIK_VLM_BASE_URL`` (explicit) > the local ollama
    default. Mirrors ``endpoint_runtime._resolve_base_url`` but with a machine
    default (the local endpoint) because the seat's default IS local."""
    return (os.environ.get(_VLM_BASE_URL_ENV) or "").strip() or _DEFAULT_VLM_BASE_URL


def resolve_vlm_api_key() -> str | None:
    """VLM seat bearer token (default ``None`` — the local seat needs none).

    Resolution: ``SEMANTIK_VLM_API_KEY`` (explicit) > ``None``. A non-local
    hosted provider without a key fails loud at dispatch (see
    ``resolve_vlm_seat``), never silently."""
    raw = (os.environ.get(_VLM_API_KEY_ENV) or "").strip()
    return raw or None


def resolve_vlm_model() -> str:
    """VLM seat model id (default ``qwen2.5vl:7b`` — Apache-2.0).

    Resolution: ``SEMANTIK_VLM_MODEL`` (explicit) > the local Qwen2.5-VL-7B
    default. Selects an LLM model → carries a ``docs/LICENSING.md`` row."""
    return (os.environ.get(_VLM_MODEL_ENV) or "").strip() or _DEFAULT_VLM_MODEL


def resolve_vlm_timeout() -> float:
    """Parse-with-fallback ``SEMANTIK_VLM_TIMEOUT_SECONDS`` (default 180.0).

    Per-request HTTP timeout for the P0 VLM image-chat POST. The tier-0 probe
    measured 64 s cold-load + 12 s/page warm on ollama 0.31.1, so this defaults
    HIGHER than the specialist seat's 120 s (a cold load on the FIRST scanned
    page eats load+inference in one request). Read at CALL time. Non-float /
    non-finite / non-positive / garbage → 180.0 (a run never crashes on a
    malformed flag; a <=0 timeout would abort every request)."""
    raw = os.environ.get(_VLM_TIMEOUT_ENV)
    if not raw:
        return _DEFAULT_VLM_TIMEOUT_SECONDS
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_VLM_TIMEOUT_SECONDS
    if val <= 0 or val != val or val in (float("inf"), float("-inf")):
        return _DEFAULT_VLM_TIMEOUT_SECONDS
    return val


class VLMSeatError(RuntimeError):
    """Raised when the VLM seat config is unusable (a non-local hosted
    provider without a credential). Fail-loud — no silent mock fallback,
    mirroring ``endpoint_runtime.EndpointRuntimeError``."""


def _is_loopback_base_url(base_url: str) -> bool:
    """True iff the URL's HOST is loopback / localhost (no credential required).

    Parses the URL and compares the resolved hostname — NOT a substring scan,
    which would match ``https://localhost.evil.example`` or a query string
    containing ``0.0.0.0`` and let a non-local provider skip the fail-loud
    credential gate. Mirrors ``lib/retrieval/answer_backend._is_loopback_host``.
    ``0.0.0.0`` is treated as loopback for local-bind parity with the historic
    check (it is not routable off-box)."""
    try:
        parsed = urlparse(base_url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # A bare hostname that isn't "localhost" → treat as non-loopback.
        return False
    return addr.is_loopback or addr == ipaddress.ip_address("0.0.0.0")


class VLMSeat:
    """Resolved VLM extraction seat (provider-agnostic OpenAI-compatible).

    Pure config object — holds provider / base_url / api_key / model and a
    fail-loud credential check. The actual per-page HTTP transcription (P0)
    calls :meth:`require_ready` before dispatch, so a misconfigured hosted
    seat raises here rather than fabricating content."""

    __slots__ = ("provider", "base_url", "api_key", "model")

    def __init__(self, provider: str, base_url: str, api_key: str | None, model: str) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    @property
    def is_local(self) -> bool:
        return self.provider == "local"

    def require_ready(self) -> "VLMSeat":
        """Fail loud when a non-local hosted (non-loopback) seat has no key.

        A local seat (``provider == "local"``) or a loopback base_url needs no
        credential. A non-local provider pointed at a remote host without an
        api_key raises :class:`VLMSeatError` — the P0 dispatch calls this
        before any POST (never a live model call in tests)."""
        if self.is_local:
            return self
        if not self.api_key and not _is_loopback_base_url(self.base_url):
            raise VLMSeatError(
                f"VLM provider {self.provider!r} at {self.base_url!r} requires a "
                "credential; set SEMANTIK_VLM_API_KEY (fail-closed, no silent stub)."
            )
        return self


def resolve_vlm_seat() -> VLMSeat:
    """Resolve the provider-agnostic VLM seat from the ``SEMANTIK_VLM_*`` env
    family (parse-with-fallback throughout)."""
    return VLMSeat(
        provider=resolve_vlm_provider(),
        base_url=resolve_vlm_base_url(),
        api_key=resolve_vlm_api_key(),
        model=resolve_vlm_model(),
    )


# Ordered list-marker shapes the VLM markdown emits (``-``/``*``/``+`` bullets,
# ``1.``/``1)`` ordered markers). Deliberately conservative — a hint is minted
# only for an UNAMBIGUOUS structural shape; plain prose yields None (no hint).
_VLM_ORDERED_MARKER_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+")
_VLM_BULLET_MARKER_RE = re.compile(r"^\s*([-*+])\s+")
_VLM_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+\S")
_VLM_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}")
_VLM_COVERAGE_VALUES = frozenset({"whole_block", "prefix"})


def mint_vlm_hint(markdown: str | None, coverage: str = "whole_block") -> dict | None:
    """Mint a NON-AUTHORITATIVE structural hint from a VLM markdown line.

    Returns ``{"kind", "level", "marker", "coverage"}`` for a recognised
    structural shape, else ``None`` (plain prose → no hint). ``kind`` is one
    of ``heading`` / ``list_item`` / ``table``; ``level`` is the ``#``-count
    for headings (else None); ``marker`` is the list marker string (else
    None); ``coverage`` records whether the VLM line covered the ``whole_block``
    or only a head-anchored ``prefix`` of the fused Tesseract block — the
    load-bearing distinction that keeps a partial (fused-title) alignment from
    ever re-typing a prose block into an ``<h2>``.

    Parse-with-fallback: ``None`` / blank input → ``None``; a coverage value
    outside the known set collapses to the CONSERVATIVE ``"prefix"`` (the
    NON-corroborating reading). Only an EXPLICIT ``"whole_block"`` makes a hint
    heading-corroboration-eligible, so an unknown / missing coverage can never
    fail OPEN into re-typing a prose block into an ``<h2>`` — the exact
    whole_block/prefix conflation the schema must prevent."""
    if not markdown or not markdown.strip():
        return None
    cov = coverage if coverage in _VLM_COVERAGE_VALUES else "prefix"
    first = markdown.lstrip().splitlines()[0] if markdown.strip() else ""

    m = _VLM_HEADING_RE.match(first)
    if m:
        return {"kind": "heading", "level": len(m.group(1)), "marker": None, "coverage": cov}

    # A markdown table: a separator row (``|---|---|``) anywhere, or two+ pipe
    # cells on the first line. Coverage is irrelevant to table hints (v1 does
    # not consume them for re-typing) but recorded for audit symmetry.
    for line in markdown.splitlines():
        if _VLM_TABLE_SEP_RE.match(line) and "|" in line:
            return {"kind": "table", "level": None, "marker": None, "coverage": cov}
    if first.count("|") >= 2:
        return {"kind": "table", "level": None, "marker": None, "coverage": cov}

    mo = _VLM_ORDERED_MARKER_RE.match(first)
    if mo:
        return {"kind": "list_item", "level": None, "marker": mo.group(1), "coverage": cov}
    mb = _VLM_BULLET_MARKER_RE.match(first)
    if mb:
        return {"kind": "list_item", "level": None, "marker": mb.group(1), "coverage": cov}

    return None


def _attach_vlm_struct_hints(merged_blocks: list[dict]) -> None:
    """Stamp ``vlm_hint`` on each merged block from its aligned VLM markdown.

    The P1 fusion stamps ``vlm_md`` (the aligned VLM markdown line(s) for the
    block) and ``vlm_coverage`` (``whole_block`` / ``prefix``) onto merged
    block dicts. This mints the hint from that shape IN PLACE. No-op (byte-
    identical) when a block carries no ``vlm_md`` — which is every block until
    the P0/P1 VLM source lands, and every block when the flags are off. Gated
    by the caller under :func:`resolve_vlm_struct_hints_mode`."""
    for b in merged_blocks:
        md = b.get("vlm_md")
        if not md:
            continue
        # Default a MISSING vlm_coverage to the conservative NON-corroborating
        # "prefix" (never fail-open to whole_block) — a future stamper that
        # omits/typos the key can never make every fused block
        # heading-corroboration-eligible.
        hint = mint_vlm_hint(md, b.get("vlm_coverage", "prefix"))
        if hint is not None:
            b["vlm_hint"] = hint


# ---------- merge / reconcile ----------


def _merge_page(page: dict, text_layer_ok: bool) -> dict:
    """Produce the unified block list downstream stages consume.

    Strategy:
      - If text layer is clean: prefer pdfplumber line blocks (richer
        font info). For each pdfplumber block, suppress every pypdfium2
        block whose bbox is mostly contained — pypdfium2 enumerates math
        glyphs character-by-character and would otherwise flood the
        merged stream with single-char fragments.
      - Orphan pypdfium2 blocks (math equations pdfplumber missed) are
        clustered into per-line groups before emission.
      - If text layer is corrupt/missing: use tesseract blocks.
      - Attach pdfplumber tables + pikepdf widgets separately.
    """
    merged_blocks: list[dict] = []

    if text_layer_ok:
        ppx = page.get("pdfplumber", {}).get("text_blocks", [])
        pfx = page.get("pypdfium2", {}).get("text_blocks", [])
        used_pfx = [False] * len(pfx)
        for pp in ppx:
            covered = _all_contained_indices(pp["bbox"], pfx, used_pfx, frac=0.6)
            provenance = ["pdfplumber"]
            if covered:
                provenance.append("pypdfium2")
                for i in covered:
                    used_pfx[i] = True
            merged_blocks.append({**pp, "provenance": provenance})

        orphans = [pfx[i] for i, u in enumerate(used_pfx) if not u]
        for cluster in _cluster_blocks_by_line(orphans):
            merged_blocks.append({**cluster, "provenance": ["pypdfium2"]})
    else:
        for b in page.get("tesseract", {}).get("text_blocks", []):
            merged_blocks.append({**b, "provenance": ["tesseract"]})

    # Sort the page's blocks. Column-major (Fix A) when SEMANTIK_COLUMN_ORDER
    # is on — cluster into columns by left-edge gutter and order
    # (column, y0, x0), with pdfplumber tables as float bboxes excluded from
    # seeding the gutter detection so a table's internal cells don't spawn a
    # spurious column. Single-column -> one column -> byte-identical to the
    # raster (y0, x0) key. Default OFF -> the legacy raster sort.
    if resolve_column_order_mode() and merged_blocks:
        if not text_layer_ok and page.get("tesseract_width"):
            # OCR-path bboxes are image pixels (see _tesseract_page_blocks); the
            # gutter fraction must scale in the SAME space or the threshold is
            # wrong by the render scale (verified: 0.06*612pt vs a 1224px page).
            page_w = float(page["tesseract_width"])
        else:
            page_w = float(page.get("width") or _DEFAULT_PAGE_WIDTH)
        table_bboxes = [
            t.get("bbox")
            for t in page.get("pdfplumber", {}).get("tables", [])
            if t.get("bbox")
        ]
        col = column_ids_for_bboxes(
            [b["bbox"] for b in merged_blocks],
            page_w,
            float_bboxes=table_bboxes or None,
            # W7.2 — only reorder for a GENUINE gutter; a single-column page
            # with indents / floats must stay in raster order (no invented
            # column). Reorder path only (never the council BERT feature).
            single_column_guard=True,
        )
        order = sorted(
            range(len(merged_blocks)),
            key=lambda k: (col[k], merged_blocks[k]["bbox"][1], merged_blocks[k]["bbox"][0]),
        )
        merged_blocks = [merged_blocks[k] for k in order]
    else:
        # Sort top-to-bottom, left-to-right.
        merged_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    # P2 — mint NON-AUTHORITATIVE VLM structural hints onto the merged blocks
    # from the aligned VLM markdown the P1 fusion stamped (``vlm_md`` /
    # ``vlm_coverage``). Gated + subordinate to SEMANTIK_VLM_EXTRACT; a no-op
    # (byte-identical) whenever no block carries ``vlm_md`` or the flag is off.
    if resolve_vlm_struct_hints_mode():
        _attach_vlm_struct_hints(merged_blocks)

    return {
        "text_blocks": merged_blocks,
        "tables": page.get("pdfplumber", {}).get("tables", []),
        "widgets": page.get("pikepdf", {}).get("widgets", []),
    }


def _all_contained_indices(outer_bbox, candidates, used, *, frac: float) -> list[int]:
    """Indices of unused candidates whose bbox is at least `frac`-contained
    in `outer_bbox`. Used to dedupe pypdfium2's per-glyph rects against
    pdfplumber's line-level rects."""
    ax0, ay0, ax1, ay1 = outer_bbox
    out: list[int] = []
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        bx0, by0, bx1, by1 = c["bbox"]
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        area = max(1e-6, (bx1 - bx0) * (by1 - by0))
        if inter / area >= frac:
            out.append(i)
    return out


def _cluster_blocks_by_line(
    blocks: list[dict], y_tol_ratio: float = 0.6, x_gap_ratio: float = 3.0
) -> list[dict]:
    """Cluster character-level blocks into line-level groups.

    Two blocks are on the same line if their y-midpoints differ by at
    most `y_tol_ratio * min_height`. Within a line, blocks are sorted by
    x and merged into a single block whose text concatenates parts with
    a space; bbox is the union; font metadata comes from the first part
    that exposed it.
    """
    if not blocks:
        return []
    items = []
    for b in blocks:
        bx = b.get("bbox")
        if not bx or len(bx) != 4:
            continue
        x0, y0, x1, y1 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
        items.append((x0, y0, x1, y1, b))
    items.sort(key=lambda it: ((it[1] + it[3]) / 2.0, it[0]))

    lines: list[list[tuple]] = []
    for it in items:
        x0, y0, x1, y1, _ = it
        my = (y0 + y1) / 2.0
        h = max(1.0, y1 - y0)
        if lines:
            prev = lines[-1]
            py0, py1 = min(p[1] for p in prev), max(p[3] for p in prev)
            pmy = (py0 + py1) / 2.0
            ph = max(1.0, py1 - py0)
            if abs(my - pmy) <= y_tol_ratio * min(h, ph):
                prev.append(it)
                continue
        lines.append([it])

    out: list[dict] = []
    for line in lines:
        line.sort(key=lambda it: it[0])
        # Within-line: split on big horizontal gaps so inline math equations
        # next to body text stay separate from the body text.
        groups: list[list[tuple]] = [[line[0]]]
        for it in line[1:]:
            prev = groups[-1][-1]
            gap = it[0] - prev[2]
            h = max(1.0, it[3] - it[1])
            if gap > x_gap_ratio * h:
                groups.append([it])
            else:
                groups[-1].append(it)
        for g in groups:
            xs0 = min(it[0] for it in g)
            ys0 = min(it[1] for it in g)
            xs1 = max(it[2] for it in g)
            ys1 = max(it[3] for it in g)
            text = " ".join(
                ((it[4].get("text") or "").strip()) for it in g if (it[4].get("text") or "").strip()
            )
            if not text:
                continue
            ref = g[0][4]
            font_size = next(
                (it[4].get("font_size") for it in g if it[4].get("font_size") is not None), None
            )
            font_name = next((it[4].get("font_name") for it in g if it[4].get("font_name")), None)
            out.append(
                {
                    "bbox": [xs0, ys0, xs1, ys1],
                    "text": text,
                    "font_size": font_size,
                    "font_name": font_name,
                    "is_bold": ref.get("is_bold"),
                    "is_italic": ref.get("is_italic"),
                    "confidence": ref.get("confidence", 1.0),
                }
            )
    return out


def _best_overlap_index(bbox_a, candidates, used) -> int | None:
    """Return index of candidate whose bbox overlaps bbox_a most, or None."""
    ax0, ay0, ax1, ay1 = bbox_a
    best_i = None
    best_iou = 0.0
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        bx0, by0, bx1, by1 = c["bbox"]
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_i = i
    return best_i if best_iou > 0.3 else None


# ---------- text-layer quality check ----------

_WORD_CHAR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def _text_layer_quality_ok(text: str) -> bool:
    """Heuristic: does the text layer look like real human-readable content?"""
    if len(text) < TEXT_LAYER_MIN_CHARS:
        return False
    # Corruption fraction: Unicode replacement chars AND undecoded `(cid:N)`
    # glyph tokens (a font with no ToUnicode CMap surfaces every glyph as
    # `(cid:NN)` rather than `�`, so a CID-blind check let CID-corrupt text
    # layers pass). Count the CHARS spanned by cid tokens so numerator and
    # denominator share a unit. Conservative 0.10 threshold unchanged.
    if REPLACEMENT_CHAR in text or "(cid:" in text:
        n_repl = text.count(REPLACEMENT_CHAR)
        n_cid_chars = sum(len(m) for m in CID_RE.findall(text))
        if (n_repl + n_cid_chars) / max(1, len(text)) > REPLACEMENT_MAX_FRAC:
            return False
    # At least 30% of chars should be letters.
    word_chars = len(_WORD_CHAR_RE.findall(text))
    if word_chars / max(1, len(text)) < 0.30:
        return False
    return True


class ExtractionError(Exception):
    pass
