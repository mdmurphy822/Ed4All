"""SEMANTIK_PAGE_ARRANGER — the scan-lane, model-driven page-ARRANGER structure path.

Task #43 productionization of the validated scratchpad prototype
(``page_arranger_proto.py``). This is the I/O half of the arranger (the pure
ARRANGE contract — enum, alias-coercion, validation, retry-ladder messages,
train-record shapes — lives in :mod:`page_arranger_contract`).

Owner directive: *"use extractors and vlm to gather the text per page, then hand
the full page to the model to read and the model figures out best way to arrange
the text."* On the OCR/scan lane the textbook-tuned council BERTs COLLAPSE
(task-#25 oracle), so the arranger REPLACES the council + cross-reranker +
structure-graph stages with one multimodal ARRANGE call per page: the page image
+ the extracted, id'd text units → JSON assigning every unit id to a
reading-ordered, typed block (ids only, never text — the verbatim contract). The
result is assembled into the SAME accessible-HTML block contract the cascade
emits (``Region`` objects with ``data-semantik-*`` provenance and
``typing_authority='vlm-arranger'``), so the downstream chunker / QC / adapter
consume it unchanged.

Design invariants (mirroring the reasoning-QC pass and the scan-lane de-BERT gate):

* **Flag-gated, default OFF, byte-identical off.** ``resolve_page_arranger_route``
  returns ``None`` IMMEDIATELY when the flag is off — ZERO extraction, zero
  imports of the heavy path, byte-identical to today. A born-digital PDF is a
  natural no-op even flag-on (the arranger route fires only on the OCR/scan lane).
* **Whole-document gather (cache-correct).** Units are minted from the ONE
  whole-doc ``extract_shared_cached`` extraction indexed by page — never per-page
  PDF splits (a single-page split re-derives a fresh pdf sha and MISSES the
  whole-doc-keyed VLM extract cache — the prototype's load-bearing lesson).
* **Deterministic pre-labels STOLEN from the cascade.** The always-on
  pedagogical / apparatus / section regexes (``deterministic_structure``) +
  running-header detection stamp non-binding HINT fields into the arrange prompt
  — the cascade's best deterministic layer, reused verbatim.
* **Per-unit resume sidecar + in-fan-out stop poll.** One PAGE is one unit; the
  fan-out mirrors ``reasoning_qc._fan_out_page_verifies`` (submit-one-consume-one,
  stop-sentinel probe before each submission, worst-case loss ≤ concurrency).
* **Coverage invariant.** Every non-image FeatureBlock lands in exactly one
  emitted Region (asserted) — the anti-content-loss contract (the class of defect
  the whole task targets).
* **BERT-hint seam is a STUB.** ``arrange_page`` takes an optional
  ``hints_provider`` callable (default ``None``); it is the rung-1 integration
  point for the future BERT-v2 heads — no real heads are wired here.

Bypassed on the arranger route (documented in the cascade seam): council heads
(Stage 3), cross-reranker (Stage 4), structure_graph (Stage 5), the Stage-5d-det
clean_structure / Stage-5d reviewer / Stage-5e resegment / Stage-5d-router
passes. Retained unchanged: Stage 5b/5c/6/6b/7-13 (gates, assembler,
reasoning-QC, theta, exit) + the whole ``region_provenance`` / adapter contract.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# --- Env var names ---------------------------------------------------------
SEMANTIK_PAGE_ARRANGER_ENV = "SEMANTIK_PAGE_ARRANGER"
_CONCURRENCY_ENV = "SEMANTIK_PAGE_ARRANGER_CONCURRENCY"
_CHECKPOINT_ENV = "SEMANTIK_PAGE_ARRANGER_CHECKPOINT"
_CHECKPOINT_FAMILY_ENV = "ED4ALL_GENERATION_CHECKPOINT"
_TRAIN_RECORDS_DIR_ENV = "SEMANTIK_ARRANGER_TRAIN_RECORDS_DIR"
_TIMEOUT_ENV = "SEMANTIK_PAGE_ARRANGER_TIMEOUT"
# Deterministic HEADING-SANITY post-pass (default ON WITHIN the arranger; the
# arranger itself is default-off, so no global default-behavior change).
_HEADING_SANITY_ENV = "SEMANTIK_ARRANGER_HEADING_SANITY"
_HEADING_MAX_CHARS_ENV = "SEMANTIK_ARRANGER_HEADING_MAX_CHARS"
# Deterministic VLM-marker HEADING-PROMOTION arm (default ON WITHIN the arranger;
# the arranger itself is default-off, so no global default-behavior change). A
# non-heading unit the VLM marked '#'/'##' (surfaced as fb.vlm_hint kind=heading,
# whole_block) AND that passes the anti-furniture SoT is re-typed to heading.
_HEADING_PROMOTE_ENV = "SEMANTIK_ARRANGER_MD_HEADING_PROMOTE"
# Tier-1 own arranger seat (mirrors the reasoning-QC seat rows).
_ARRANGER_BASE_URL_ENV = "SEMANTIK_PAGE_ARRANGER_BASE_URL"
_ARRANGER_API_KEY_ENV = "SEMANTIK_PAGE_ARRANGER_API_KEY"
_ARRANGER_MODEL_ENV = "SEMANTIK_PAGE_ARRANGER_MODEL"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})

_DEFAULT_CONCURRENCY = 4
_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_HEADING_MAX_CHARS = 120
# A plausible tail TITLE is short + terminal-period-free; this caps its length.
_TAIL_TITLE_MAX_CHARS = 60
_ARRANGE_MAX_TOKENS = 6144
_RENDER_SCALE = 2.3
_RENDER_MAX_PX = 1500
_CACHE_BASENAME = "page_arranger_cache"

_RUNNING_HEADER_MIN_SHARE = 0.35  # TODO(calibration) — task #43 pending live re-validation

__all__ = [
    "resolve_page_arranger_mode",
    "resolve_arranger_concurrency",
    "resolve_arranger_checkpoint",
    "resolve_arranger_train_records_dir",
    "resolve_arranger_timeout",
    "resolve_arranger_heading_sanity",
    "resolve_arranger_heading_max_chars",
    "resolve_arranger_md_heading_promote",
    "resolve_arranger_seat",
    "resolve_page_arranger_route",
    "ArrangerRoute",
    "arrange_page",
    "apply_heading_sanity",
    "apply_md_heading_promote",
    "build_title_lexicon",
    "apply_leading_title_split",
    "mint_units_by_page",
    "build_page_hints",
    "finalize_continues_over_dir",
]


# ---------------------------------------------------------------------------
# Part A: resolvers + route probe.
# ---------------------------------------------------------------------------
def resolve_page_arranger_mode() -> bool:
    """Whether the scan-lane page-arranger path may fire (default OFF).

    Truthy-set parse-with-fallback (``1``/``true``/``yes``/``on``,
    case-insensitive); unset / blank / falsey / garbage → off (byte-identical
    pass-through — the route probe returns ``None`` before any extraction). Read
    at CALL time. Mirrors ``scan_lane.resolve_scan_lane_debert_mode``.
    """
    return (os.environ.get(SEMANTIK_PAGE_ARRANGER_ENV) or "").strip().lower() in _TRUTHY


def resolve_arranger_concurrency() -> int:
    """Per-page arrange-POST fan-out width (default 4, floored at 1).

    Parse-with-fallback: blank / non-int / non-positive / garbage →
    :data:`_DEFAULT_CONCURRENCY`.
    """
    raw = (os.environ.get(_CONCURRENCY_ENV) or "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CONCURRENCY
    return max(1, val) if val > 0 else _DEFAULT_CONCURRENCY


def resolve_arranger_checkpoint() -> bool:
    """Per-page resume-cache gate (default ON). Read at CALL time.

    Precedence mirrors ``reasoning_qc.resolve_reasoning_qc_checkpoint``: the site
    env ``SEMANTIK_PAGE_ARRANGER_CHECKPOINT``, WHEN SET (non-blank), wins (falsey
    token → off, else on); otherwise the family ``ED4ALL_GENERATION_CHECKPOINT``
    decides (falsey token → off, unset / garbage → on). Off → byte-identical: the
    fan-out neither reads nor writes the sidecar (the cache dir is never created).
    """
    site = os.environ.get(_CHECKPOINT_ENV, "")
    if site.strip():
        return site.strip().lower() not in _FALSEY
    return os.environ.get(_CHECKPOINT_FAMILY_ENV, "").strip().lower() not in _FALSEY


def resolve_arranger_train_records_dir() -> Optional[Path]:
    """Run-scoped label-factory dir (default None = OFF → zero writes).

    A set, non-blank ``SEMANTIK_ARRANGER_TRAIN_RECORDS_DIR`` → ``Path``; unset /
    blank → ``None`` (the label factory writes nothing, creates no dir).
    """
    raw = (os.environ.get(_TRAIN_RECORDS_DIR_ENV) or "").strip()
    return Path(raw) if raw else None


def resolve_arranger_timeout() -> float:
    """Per-request HTTP timeout (float s) for the arrange POST (default 600.0).

    Parse-with-fallback: blank / non-float / non-finite / non-positive / garbage
    → :data:`_DEFAULT_TIMEOUT_SECONDS`.
    """
    raw = (os.environ.get(_TIMEOUT_ENV) or "").strip()
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS
    if val != val or val in (float("inf"), float("-inf")) or val <= 0:
        return _DEFAULT_TIMEOUT_SECONDS
    return val


def resolve_arranger_heading_sanity() -> bool:
    """Deterministic HEADING-SANITY post-pass gate (default ON). Read at CALL time.

    Default-ON parse-with-fallback (mirrors ``resolve_arranger_checkpoint``): only
    the explicit falsey tokens (``0``/``false``/``no``/``off``, case-insensitive)
    disable it; unset / blank / truthy / garbage → on. The pass runs ONLY inside
    the arranger (itself default-off), so on-by-default here changes NO global
    default behaviour. Off → the arranger's arrangement is byte-identical (no
    demotion / no synthetic tail-title heading).
    """
    return (os.environ.get(_HEADING_SANITY_ENV) or "").strip().lower() not in _FALSEY


def resolve_arranger_heading_max_chars() -> int:
    """Char ceiling above which a heading-typed block is DEMOTED (default 120).

    Parse-with-fallback: blank / non-int / non-positive / garbage →
    :data:`_DEFAULT_HEADING_MAX_CHARS`.
    """
    raw = (os.environ.get(_HEADING_MAX_CHARS_ENV) or "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_HEADING_MAX_CHARS
    return val if val > 0 else _DEFAULT_HEADING_MAX_CHARS


def resolve_arranger_md_heading_promote() -> bool:
    """Deterministic VLM-marker heading-PROMOTION arm gate (default ON). Read at
    CALL time.

    Default-ON parse-with-fallback (mirrors ``resolve_arranger_heading_sanity``):
    only the explicit falsey tokens (``0``/``false``/``no``/``off``,
    case-insensitive) disable it; unset / blank / truthy / garbage → on. The pass
    runs ONLY inside the arranger (itself default-off), so on-by-default here
    changes NO global default behaviour. Off → the arrangement is byte-identical
    (no promotion). NATURAL no-op without the VLM struct-hint channel
    (``SEMANTIK_VLM_STRUCT_HINTS`` — no ``vlm_heading_level`` on any unit → nothing
    to promote), so a non-VLM run is byte-identical even flag-on.
    """
    return (os.environ.get(_HEADING_PROMOTE_ENV) or "").strip().lower() not in _FALSEY


def resolve_arranger_seat():
    """Resolve + fail-loud-check the multimodal ARRANGE seat.

    Three-tier chain, FIRST HIT WINS (mirrors
    ``reasoning_qc_vlm.resolve_reasoning_qc_seat``, but the FALLBACK is the VLM
    seat — the arranger is a MULTIMODAL task, so the seat that GATHERED the pages
    is the natural default, not the text specialist):

      1. **Explicit arranger seat.** ``SEMANTIK_PAGE_ARRANGER_BASE_URL`` (+
         ``_API_KEY``, + the ``_MODEL`` override or the VLM model).
      2. **VLM extraction seat.** ``extract_shared.resolve_vlm_seat`` — the
         multimodal-appropriate seat that gathered the pages (the omni/VL model).
      3. **Specialist TEXT seat.** ``SEMANTIK_SPECIALIST_BASE_URL`` (+
         ``SEMANTIK_SPECIALIST_API_KEY`` + review-model chain) — a last-resort
         text-only fallback (the arrange call degrades to text-only when the seat
         has no vision tower, via ``arrange_page(..., include_image=False)``).

    ``VLMSeat.require_ready`` is called on the winner (fail-loud — a non-local
    hosted seat with no key raises rather than fabricating an arrangement).
    """
    from .extract_shared import VLMSeat, resolve_vlm_seat

    model_override = (os.environ.get(_ARRANGER_MODEL_ENV) or "").strip()

    # (1) Explicit arranger seat.
    base = (os.environ.get(_ARRANGER_BASE_URL_ENV) or "").strip()
    if base:
        api_key = (os.environ.get(_ARRANGER_API_KEY_ENV) or "").strip() or None
        model = model_override or resolve_vlm_seat().model
        return VLMSeat(
            provider="page-arranger", base_url=base, api_key=api_key, model=model
        ).require_ready()

    # (2) VLM extraction seat (multimodal — the seat that gathered).
    vlm = resolve_vlm_seat()
    if model_override:
        vlm = VLMSeat(
            provider=vlm.provider,
            base_url=vlm.base_url,
            api_key=vlm.api_key,
            model=model_override,
        )
    # A local/loopback VLM seat is always ready; a hosted one w/o key fails loud.
    if not vlm.base_url:
        # (3) Specialist TEXT seat fallback.
        spec_base = (os.environ.get("SEMANTIK_SPECIALIST_BASE_URL") or "").strip()
        if spec_base:
            api_key = (os.environ.get("SEMANTIK_SPECIALIST_API_KEY") or "").strip() or None
            from .reasoning_qc_vlm import _resolve_specialist_review_model

            model = model_override or _resolve_specialist_review_model()
            return VLMSeat(
                provider="specialist", base_url=spec_base, api_key=api_key, model=model
            ).require_ready()
    return vlm.require_ready()


def _text_layer_ok_scan_lane(feature_blocks: Any) -> bool:
    """Decoupled fallback lane predicate (documented alternative to
    ``scan_lane.is_ocr_scan_lane``).

    Used only if the in-flux ``scan_lane`` module cannot be imported (agent (a)'s
    file churns). Same semantics: the OCR/scan lane iff ≥50% of the TEXT
    FeatureBlocks carry ``tesseract`` provenance.
    """
    ocr = total = 0
    for fb in feature_blocks or ():
        if getattr(fb, "is_image", False):
            continue
        total += 1
        src = getattr(fb, "provenance", None) or getattr(
            getattr(fb, "raw", None), "source", None
        )
        if "tesseract" in str(src or "").lower():
            ocr += 1
    return total > 0 and (ocr / total) >= 0.5


def resolve_page_arranger_route(pdf_path: Any) -> "Optional[ArrangerRoute]":
    """Return an :class:`ArrangerRoute` when the arranger should own structure,
    else ``None``.

    Returns ``None`` IMMEDIATELY when the flag is off — ZERO extraction,
    byte-identical to today. When on, runs the WHOLE-DOCUMENT
    ``extract_shared_cached`` + ``featurize_with_regions`` (the cache-correct
    pattern — whole-doc sha keys the extract/VLM caches), then a lane check:
    a born-digital PDF (not the OCR/scan lane) → ``None`` (the council path owns
    it). Fail-soft: any extraction/lane error logs a warning and returns ``None``
    (the cascade falls back to the council path — never a crash).
    """
    if not resolve_page_arranger_mode():
        return None
    try:
        from .extract_shared import extract_shared_cached
        from .features import featurize_with_regions

        shared = extract_shared_cached(Path(pdf_path))
        feature_set = featurize_with_regions(shared)
    except Exception as exc:  # noqa: BLE001 — never crash the cascade on the probe
        logger.warning(
            "page-arranger route: extraction failed (non-fatal, falling back to "
            "the council path): %s", exc
        )
        return None

    feature_blocks = feature_set.feature_blocks
    if not _is_ocr_scan_lane(feature_blocks):
        return None
    return ArrangerRoute(pdf_path=Path(pdf_path), shared=shared, feature_set=feature_set)


def _is_ocr_scan_lane(feature_blocks: Any) -> bool:
    """Lane predicate — reuse ``scan_lane.is_ocr_scan_lane`` (read-only), with a
    documented local fallback if agent (a)'s module cannot be imported."""
    try:
        from .scan_lane import is_ocr_scan_lane

        return bool(is_ocr_scan_lane(feature_blocks))
    except Exception as exc:  # noqa: BLE001
        logger.debug("page-arranger: scan_lane import failed, using local lane predicate: %s", exc)
        return _text_layer_ok_scan_lane(feature_blocks)


# ---------------------------------------------------------------------------
# Part B: GATHER (whole-doc, cache-correct) — units minted from FeatureBlocks.
# ---------------------------------------------------------------------------
def mint_units_by_page(feature_blocks: list) -> "dict[int, list[dict]]":
    """Mint per-page id'd text units from the whole-doc FeatureBlock stream.

    Units come from the FeatureBlocks (NOT raw merged blocks) so each unit's
    ``fb_index`` is the EXACT position in ``feature_blocks`` — the coverage
    invariant + the Region ``feature_block_indices`` are then exact. Non-image
    FBs only (synthetic image FBs are region-less on the arranger route, v1). id
    = ``p{page}_b{ordinal:02d}``, ordinal = per-page 0-based stream position of
    the kept units. Text is VERBATIM.
    """
    by_page: dict[int, list[dict]] = {}
    ordinal: dict[int, int] = {}
    for fb_index, fb in enumerate(feature_blocks):
        if getattr(fb, "is_image", False):
            continue
        raw = getattr(fb, "raw", None)
        text = (getattr(raw, "text", "") or "").strip()
        if not text:
            continue
        page = int(getattr(raw, "page", 0) or 0)
        ord_i = ordinal.get(page, 0)
        ordinal[page] = ord_i + 1
        unit = {
            "id": f"p{page}_b{ord_i:02d}",
            "text": text,
            "fb_index": fb_index,
            "bbox": list(getattr(raw, "bbox", None) or ()) or None,
        }
        # ADDITIVE, prompt-inert signal: the VLM's own '#'/'##' heading marker on
        # this unit's line, surfaced through the EXISTING vlm_hint channel
        # (fb.vlm_hint, mirrored from raw.vlm_hint by features.py; populated only
        # under SEMANTIK_VLM_STRUCT_HINTS). ONLY a whole_block heading hint counts
        # (a head-anchored 'prefix' is a fused-title fragment — never a standalone
        # heading). Absent when no hint → _unit_listing / the arrange prompt are
        # UNCHANGED (byte-identical prompt + cache key); consumed only by the
        # deterministic heading-PROMOTE post-pass.
        hint = getattr(fb, "vlm_hint", None)
        if (
            isinstance(hint, dict)
            and hint.get("kind") == "heading"
            and hint.get("coverage") == "whole_block"
            and isinstance(hint.get("level"), int)
        ):
            unit["vlm_heading_level"] = hint["level"]
        by_page.setdefault(page, []).append(unit)
    return by_page


def _render_page_image_b64(pdf_path: Path, page_num: int) -> str:
    """Render one page (1-indexed) to a JPEG b64 data payload (port of the
    prototype's ``render_arrange_image_b64``). Sequential-only (pypdfium2 is not
    thread-safe on one handle)."""
    from .extract_shared import _pypdfium2_render_page_to_image

    img = _pypdfium2_render_page_to_image(pdf_path, page_num, scale=_RENDER_SCALE)
    if img.width > _RENDER_MAX_PX:
        ratio = _RENDER_MAX_PX / float(img.width)
        img = img.resize((_RENDER_MAX_PX, max(1, round(img.height * ratio))))
    if getattr(img, "mode", "RGB") != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Part C: deterministic pre-labels (STOLEN from cascade — its best layer).
# ---------------------------------------------------------------------------
def _detect_running_header_ids(units_by_page: "dict[int, list[dict]]") -> set:
    """Unit ids whose text is a doc-wide RUNNING HEADER/FOOTER.

    A number-masked, whitespace-normalized signature recurring at the FIRST or
    LAST unit position of a page on ≥ :data:`_RUNNING_HEADER_MIN_SHARE` of the
    document's pages (mirrors ``vlm_furniture``'s signature approach; reuses its
    ``normalize_furniture_sig`` when importable).
    """
    try:
        from .vlm_furniture import normalize_furniture_sig as _sig
    except Exception:  # noqa: BLE001 — ~local impl if vlm_furniture churns
        import re as _re

        def _sig(line: str) -> str:  # type: ignore[misc]
            t = _re.sub(r"\d+", "#", (line or "").strip())
            return " ".join(t.split()).lower()

    n_pages = len(units_by_page) or 1
    counts: dict[str, int] = {}
    id_by_sig: dict[str, list] = {}
    for page, units in units_by_page.items():
        if not units:
            continue
        edge = [units[0]]
        if len(units) > 1:
            edge.append(units[-1])
        seen_sigs = set()
        for u in edge:
            sig = _sig(u["text"])
            if not sig or len(sig) < 3:
                continue
            id_by_sig.setdefault(sig, []).append(u["id"])
            if sig not in seen_sigs:
                counts[sig] = counts.get(sig, 0) + 1
                seen_sigs.add(sig)
    hdr_ids: set = set()
    for sig, c in counts.items():
        if c / n_pages >= _RUNNING_HEADER_MIN_SHARE:
            hdr_ids.update(id_by_sig.get(sig, ()))
    return hdr_ids


def build_page_hints(
    units: list[dict], running_header_ids: set
) -> "dict[str, str]":
    """Deterministic per-unit hint dict (STOLEN from ``deterministic_structure``).

    Reuses the pedagogical/apparatus/section SoT regexes (read-only, the
    structure-router precedent): ``_pedagogical_class_for`` →
    ``pedagogical_label:<css>``; ``_matches_apparatus_opener`` →
    ``apparatus_opener``; ``_STANDALONE_SECTION_RE`` → ``section_title``; the
    running-header set → ``likely_running_header``. First matching hint per unit
    wins (running-header takes precedence — furniture beats a pedagogical read).
    """
    try:
        from .qwen_specialists.deterministic_structure import (
            _STANDALONE_SECTION_RE,
            _matches_apparatus_opener,
            _pedagogical_class_for,
        )
    except Exception as exc:  # noqa: BLE001 — hints are advisory; degrade to none
        logger.debug("page-arranger: deterministic_structure import failed: %s", exc)
        return {}

    hints: dict[str, str] = {}
    for u in units:
        uid = u["id"]
        text = u["text"]
        if uid in running_header_ids:
            hints[uid] = "likely_running_header"
            continue
        css = _pedagogical_class_for(text)
        if css:
            hints[uid] = f"pedagogical_label:{css}"
            continue
        if _matches_apparatus_opener(text):
            hints[uid] = "apparatus_opener"
            continue
        if _STANDALONE_SECTION_RE.match(text.strip()):
            hints[uid] = "section_title"
    return hints


def _render_hints_section(units: list[dict], hints: "dict[str, str]") -> str:
    """Advisory hint block appended to the arrange user prompt (non-binding)."""
    lines = [f"{u['id']}: {hints[u['id']]}" for u in units if u["id"] in hints]
    if not lines:
        return ""
    return (
        "Deterministic hints (non-binding — a running_header/apparatus_opener "
        "unit is usually furniture; a section_title/pedagogical_label is a "
        "heading or a labelled block):\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Part C2: deterministic HEADING-SANITY post-pass (glued-mega-heading repair).
#
# The arrange model can only type WHOLE extraction units, so when extraction
# fuses a page-top band into ONE unit (running-header + "TRY IT :: 1.93 … TRY
# IT :: 1.94 … Divide Integers") and the model types it a heading, the real
# section title is trapped as a >200-char mega-heading. This deterministic
# post-pass (applied AFTER arrangement validation) DEMOTES such a block and,
# when a plausible title survives at the unit tail, emits a SYNTHETIC heading
# carrying the VERBATIM tail text (offsets recorded; never invented text).
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402 — module-local alias for the sanity regexes

# A bare standalone page number ("83") — page furniture.
_PAGE_NUMBER_RE = _re.compile(r"^\s*\d{1,4}\s*$")
# Sentence-terminator split (fallback tail-title boundary when a unit has no
# newline). Split on a terminal .?! followed by whitespace.
_SENTENCE_BOUNDARY_RE = _re.compile(r"[.!?]\s+")


def _sanity_regexes():
    """Lazy-load the co-homed deterministic pedagogical/apparatus/running-header
    SoT (the SAME regexes the pre-label hints steal). Degrades to no-markers when
    ``deterministic_structure`` cannot be imported (advisory — never breaks the
    pass)."""
    try:
        from .qwen_specialists.deterministic_structure import (
            _RUNNING_HEADER_LEADING_PAGE_RE,
            _RUNNING_HEADER_TEXT_RE,
            _apparatus_opener_match_len,
            _matches_apparatus_opener,
            _pedagogical_class_for,
        )
    except Exception as exc:  # noqa: BLE001 — markers advisory; length still gates
        logger.debug("page-arranger heading-sanity: deterministic_structure import failed: %s", exc)
        return None
    return {
        "running_text": _RUNNING_HEADER_TEXT_RE,
        "running_lead": _RUNNING_HEADER_LEADING_PAGE_RE,
        "apparatus_len": _apparatus_opener_match_len,
        "apparatus_whole": _matches_apparatus_opener,
        "pedagogical": _pedagogical_class_for,
    }


def _heading_furniture_marker(text: str, rx) -> "Optional[str]":
    """The furniture/apparatus marker class of a heading ``text`` (or None).

    Reuses the deterministic SoT: a running-header (leading/trailing page-number)
    shape, a leading pedagogical label (EXAMPLE/TRY IT/Solution/Step/…), or a
    leading apparatus opener (Introduction/Summary/Key Terms/…). First hit wins.
    """
    s = (text or "").strip()
    if not s or rx is None:
        return None
    if rx["running_text"].match(s) or rx["running_lead"].match(s):
        return "running_header"
    if rx["pedagogical"](s):
        return "pedagogical_label"
    if rx["apparatus_len"](s) is not None:
        return "apparatus_opener"
    return None


def _is_wholly_furniture(text: str, rx) -> bool:
    """Whether the ENTIRE ``text`` is discardable furniture (→ furniture demote).

    A bare page number, a whole running-header ("Chapter 1 Foundations 83"), or a
    whole apparatus opener. Conservative: a heading with a real title tail is NOT
    wholly furniture (it demotes to paragraph so the tail-title recovery can run).
    """
    s = (text or "").strip()
    if not s:
        return False
    if _PAGE_NUMBER_RE.match(s):
        return True
    if rx is None:
        return False
    if rx["running_text"].match(s) or rx["running_lead"].match(s):
        return True
    if rx["apparatus_whole"](s):
        return True
    return False


def _is_plausible_tail_title(text: str, rx, *, max_chars: int) -> bool:
    """Whether a unit-tail segment is a plausible SECTION TITLE.

    Title-ish shape: non-empty, ≤ ``max_chars``, no terminal period, carries
    alphabetic content, opens with a capital letter or a section ordinal digit,
    and is NOT itself an apparatus/furniture/pedagogical marker.
    """
    s = (text or "").strip()
    if not s or len(s) > max_chars:
        return False
    if s.endswith("."):
        return False
    if not _re.search(r"[A-Za-z]", s):
        return False
    if not (s[0].isupper() or s[0].isdigit()):
        return False
    if _heading_furniture_marker(s, rx) is not None:
        return False
    if rx is not None and rx["apparatus_whole"](s):
        return False
    return True


def _extract_tail_title(text: str, rx, *, max_chars: int) -> "Optional[tuple[str, int, int]]":
    """Extract a plausible tail TITLE from a unit's text → ``(title, start, end)``.

    Splits on the FINAL newline first; else on the last sentence boundary. The
    returned ``(start, end)`` are exact character offsets into ``text`` such that
    ``text[start:end] == title`` VERBATIM (never invented text). ``None`` when no
    plausible title survives.
    """
    if not text or not text.strip():
        return None
    candidate: "Optional[tuple[str, int, int]]" = None
    nl = text.rfind("\n")
    if nl != -1:
        seg = text[nl + 1 :]
        stripped = seg.strip()
        if stripped:
            lead = len(seg) - len(seg.lstrip())
            start = nl + 1 + lead
            candidate = (stripped, start, start + len(stripped))
    if candidate is None:
        last_end = None
        for m in _SENTENCE_BOUNDARY_RE.finditer(text):
            last_end = m.end()
        if last_end is not None and last_end < len(text):
            seg = text[last_end:]
            stripped = seg.strip()
            if stripped:
                lead = len(seg) - len(seg.lstrip())
                start = last_end + lead
                candidate = (stripped, start, start + len(stripped))
    if candidate is None:
        return None
    title, start, end = candidate
    if text[start:end] != title:  # defensive: never emit a mismatched offset
        return None
    if not _is_plausible_tail_title(title, rx, max_chars=max_chars):
        return None
    return title, start, end


def apply_heading_sanity(
    arr: dict,
    unit_by_id: "dict[str, dict]",
    *,
    max_chars: int,
    enabled: bool,
) -> list:
    """Deterministic HEADING-SANITY post-pass over a VALIDATED arrangement.

    Mutates ``arr['blocks']`` in place and returns a list of demotion/extraction
    records ``[{op, block_idx, to, reason, length, [tail_title, synthesized_from,
    tail_offsets]}]`` (the label factory learns from these). A heading-typed block
    whose joined member text exceeds ``max_chars`` OR matches an
    apparatus/furniture marker is DEMOTED: to ``furniture`` when the ENTIRE text
    is furniture, else to ``paragraph`` (keeping the full original text). On a
    paragraph demote a plausible TAIL TITLE (final line, else last sentence) is
    emitted as a SYNTHETIC heading block inserted immediately AFTER the demoted
    block — VERBATIM text from the unit tail, ``duplicate_of_tail`` marked so the
    coverage gate never double-counts (the synthetic block carries NO unit ids →
    its Region claims zero FeatureBlocks). ``enabled=False`` → no-op (byte-identical).
    """
    records: list = []
    if not enabled:
        return records
    blocks = (arr or {}).get("blocks")
    if not isinstance(blocks, list):
        return records
    rx = _sanity_regexes()
    new_blocks: list = []
    for bi, blk in enumerate(blocks):
        new_blocks.append(blk)
        if not isinstance(blk, dict) or blk.get("type") != "heading":
            continue
        if blk.get("synthetic_heading"):  # never re-process our own synthetics
            continue
        ids = [u for u in (blk.get("ids") or []) if u in unit_by_id]
        if not ids:
            continue
        member_texts = [unit_by_id[u]["text"] for u in ids]
        joined = " ".join(t.replace("\n", " ").strip() for t in member_texts).strip()
        if not joined:
            continue
        too_long = len(joined) > max_chars
        marker = _heading_furniture_marker(joined, rx)
        if not (too_long or marker):
            continue
        wholly_furniture = _is_wholly_furniture(joined, rx)
        new_type = "furniture" if wholly_furniture else "paragraph"
        blk["type"] = new_type
        reason = "too_long" if too_long else marker
        blk["heading_sanity"] = {
            "demoted_from": "heading", "reason": reason, "length": len(joined),
        }
        rec: dict = {
            "op": "heading_demote", "block_idx": bi, "to": new_type,
            "reason": reason, "length": len(joined),
        }
        if new_type == "paragraph":
            tail_uid = ids[-1]
            extracted = _extract_tail_title(
                unit_by_id[tail_uid]["text"], rx, max_chars=_TAIL_TITLE_MAX_CHARS
            )
            if extracted is not None:
                title, start, end = extracted
                level = blk.get("level")
                new_blocks.append({
                    "type": "heading",
                    "level": level if isinstance(level, int) else 2,
                    "ids": [],
                    "synthetic_heading": True,
                    "sanity_pass": "heading_sanity_tail_title",
                    "text": title,
                    "duplicate_of_tail": True,
                    "synthesized_from": tail_uid,
                    "tail_offsets": [start, end],
                })
                rec["tail_title"] = title
                rec["synthesized_from"] = tail_uid
                rec["tail_offsets"] = [start, end]
        records.append(rec)
    arr["blocks"] = new_blocks
    return records


def apply_md_heading_promote(
    arr: dict,
    unit_by_id: "dict[str, dict]",
    *,
    max_chars: int,
    enabled: bool,
) -> list:
    """Deterministic VLM-MARKER heading PROMOTION over a VALIDATED arrangement.

    Root cause of the de-poisoned scan lane's heading-recall REGRESSION (0.474 vs
    the old 0.583): the VLM emits ``#``/``##`` heading markers on the page's own
    section titles (``## Divide Integers``, ``## 1.1 Introduction to Whole
    Numbers``), but the arrange model — typing WHOLE units off the marker-STRIPPED
    unit text + the page image — inconsistently leaves many standalone section
    titles as ``paragraph`` (measured: 6 of the 10 metric-missed vendor headings
    are exactly this MISTYPED-standalone mode; the same title is even typed
    heading on one page and paragraph on another). The VLM's marker survives as
    ``fb.vlm_hint`` (mirrored onto the unit as ``vlm_heading_level`` by
    :func:`mint_units_by_page`), so this pass re-types such a block to ``heading``.

    Anti-re-poisoning: the VLM ALSO heading-marks apparatus / pedagogical labels
    ("EXAMPLE 1.1", "MANIPULATIVE MATHEMATICS", "HOW TO ::", "1.1 EXERCISES") — so
    a promotion fires ONLY when the joined block text passes the SAME deterministic
    furniture/apparatus/pedagogical SoT guard the demotion arm uses
    (:func:`_is_plausible_tail_title`): short (≤ ``max_chars``), no terminal
    period, opens capital/digit, alphabetic content, and NOT a furniture /
    apparatus-opener / pedagogical-label marker. A real prose paragraph (ends in a
    period / runs long) and every apparatus/example label are rejected, so
    precision is preserved while the genuinely-mis-typed section titles are
    recovered.

    Coverage-safe: this is a pure RE-TYPE — the block KEEPS its unit ids (unlike
    the synthetic tail/leading headings), so the coverage invariant is untouched
    (no new/duplicate FeatureBlock claims). Mutates ``arr['blocks']`` in place;
    returns the promotion records ``[{op, block_idx, from, to, level, reason,
    length}]`` (the label factory learns from them). ``enabled=False`` OR no unit
    carrying ``vlm_heading_level`` → no-op (byte-identical).
    """
    records: list = []
    if not enabled:
        return records
    blocks = (arr or {}).get("blocks")
    if not isinstance(blocks, list):
        return records
    rx = _sanity_regexes()
    for bi, blk in enumerate(blocks):
        if not isinstance(blk, dict) or blk.get("type") != "paragraph":
            continue
        if blk.get("synthetic_heading"):
            continue
        ids = [u for u in (blk.get("ids") or []) if u in unit_by_id]
        if not ids:
            continue
        # The VLM must have marked the LEAD unit a heading (whole_block); a
        # trailing-only hint would be a mid-block fragment, not a block heading.
        lead_level = unit_by_id[ids[0]].get("vlm_heading_level")
        if not isinstance(lead_level, int):
            continue
        member_texts = [unit_by_id[u]["text"] for u in ids]
        joined = " ".join(t.replace("\n", " ").strip() for t in member_texts).strip()
        # Title-shape + anti-furniture SoT (rejects prose, apparatus, pedagogical
        # labels, running headers). This is the whole precision guardrail.
        if not _is_plausible_tail_title(joined, rx, max_chars=max_chars):
            continue
        level = min(4, max(2, lead_level))
        blk["type"] = "heading"
        blk["level"] = level
        blk["heading_sanity"] = {
            "promoted_from": "paragraph", "reason": "vlm_marker",
            "vlm_heading_level": lead_level, "length": len(joined),
        }
        records.append({
            "op": "heading_promote", "block_idx": bi, "from": "paragraph",
            "to": "heading", "level": level, "reason": "vlm_marker",
            "length": len(joined),
        })
    return records


# ---------------------------------------------------------------------------
# Part C3: symmetric LEADING-TITLE split (outline-anchored — gate-1 follow-up).
#
# The second glue variant: the real section title fuses at the HEAD of the
# FOLLOWING paragraph unit ("Divide Integers What about division? …") — typed
# paragraph, so heading recall misses it. The split is OUTLINE-ANCHORED to
# kill false positives: a paragraph's leading words are only carved into a
# synthetic heading when they match a title in the DOCUMENT-level lexicon
# (chapter-outline/ToC "N.N Title" entries + already-heading-typed units).
# Folds under the SAME SEMANTIK_ARRANGER_HEADING_SANITY gate (same defect
# family). Runs document-level in ``arrange_regions`` AFTER the fan-out (the
# lexicon needs every page), so per-page sidecars stay valid and the split
# re-applies deterministically on cached results.
# ---------------------------------------------------------------------------
# A chapter-outline / ToC marker line ("Chapter Outline", "Table of Contents").
_OUTLINE_MARKER_RE = _re.compile(
    r"chapter\s+outline|table\s+of\s+contents", _re.IGNORECASE
)
# Split an outline line at each "N.N " section-number token; segments AFTER the
# first are the entry titles.
_OUTLINE_SPLIT_RE = _re.compile(r"\b\d+\.\d+\s+")
# Leading "N.N " / "N." / "N) " numbering stripped by title normalization.
_LEADING_NUMBERING_RE = _re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")
# Trailing ToC page number (+ dot leaders) stripped from an outline entry title.
_TRAILING_TOC_PAGE_RE = _re.compile(r"[\s.…]*\d{1,4}\s*$")


def _normalize_title(s: str) -> str:
    """Casefolded, numbering-stripped, whitespace-collapsed title key."""
    s = (s or "").strip()
    s = _LEADING_NUMBERING_RE.sub("", s)
    s = s.rstrip(" .:;,")
    return " ".join(s.split()).casefold()


def _parse_outline_titles(text: str) -> list:
    """Parse the ``N.N Title`` entry titles out of a chapter-outline/ToC unit.

    Line-oriented: each line is split at its ``N.N `` tokens; every segment
    AFTER the first is one entry title (a fused single-line outline run yields
    them all). A trailing ToC page number (+ dot leaders) is stripped.
    """
    titles: list = []
    for line in (text or "").splitlines():
        segs = _OUTLINE_SPLIT_RE.split(line)
        for seg in segs[1:]:
            t = _TRAILING_TOC_PAGE_RE.sub("", seg.strip()).strip()
            if t and _re.search(r"[A-Za-z]", t):
                titles.append(t)
    return titles


def build_title_lexicon(
    units_by_page: "dict[int, list[dict]]", results: "dict[int, dict]"
) -> set:
    """Build the page-independent, document-level TITLE lexicon (normalized).

    Two sources: (a) any unit matching a chapter-outline/ToC marker — its
    ``N.N Title`` entries are parsed out; (b) blocks already TYPED heading in a
    valid arrangement (short, clean ones only — plausible-title shape, ≤60
    chars; includes prior heading-sanity synthetics). With no outline present,
    (b) alone feeds the lexicon. Entries are normalized via
    :func:`_normalize_title` (casefold, numbering stripped).
    """
    rx = _sanity_regexes()
    lexicon: set = set()
    unit_text: dict = {}
    for units in (units_by_page or {}).values():
        for u in units:
            unit_text[u["id"]] = u.get("text") or ""
            text = u.get("text") or ""
            if _OUTLINE_MARKER_RE.search(text):
                for t in _parse_outline_titles(text):
                    if len(t) <= _TAIL_TITLE_MAX_CHARS and not _is_wholly_furniture(t, rx):
                        norm = _normalize_title(t)
                        if norm:
                            lexicon.add(norm)
    for res in (results or {}).values():
        if not isinstance(res, dict) or res.get("status") != "ok":
            continue
        for blk in ((res.get("arrangement") or {}).get("blocks") or []):
            if not isinstance(blk, dict) or blk.get("type") != "heading":
                continue
            if blk.get("synthetic_heading"):
                text = (blk.get("text") or "").strip()
            else:
                texts = [unit_text.get(uid, "") for uid in (blk.get("ids") or [])]
                text = " ".join(t.replace("\n", " ").strip() for t in texts).strip()
            if text and _is_plausible_tail_title(text, rx, max_chars=_TAIL_TITLE_MAX_CHARS):
                norm = _normalize_title(text)
                if norm:
                    lexicon.add(norm)
    return lexicon


def _match_leading_lexicon_title(
    text: str, lexicon: set
) -> "Optional[tuple[str, int, int]]":
    """Match a lexicon title at char offset 0 of a unit's text (modulo leading
    whitespace) → ``(verbatim_prefix, start, end)`` or ``None``.

    Word-boundary prefixes are grown up to :data:`_TAIL_TITLE_MAX_CHARS`; a
    prefix whose normalization is IN the lexicon AND whose remainder opens with
    a sentence-start (capital letter or digit) is a match — LONGEST wins. The
    returned offsets satisfy ``text[start:end] == verbatim_prefix``.
    """
    if not text or not lexicon:
        return None
    lead = len(text) - len(text.lstrip())
    body = text[lead:]
    best: "Optional[tuple[str, int, int]]" = None
    for m in _re.finditer(r"\S+", body):
        end = m.end()
        prefix = body[:end]
        if len(prefix) > _TAIL_TITLE_MAX_CHARS:
            break
        if _normalize_title(prefix) in lexicon:
            remainder = body[end:].lstrip()
            if remainder and (remainder[0].isupper() or remainder[0].isdigit()):
                best = (prefix, lead, lead + end)
    return best


def apply_leading_title_split(
    arr: dict,
    unit_by_id: "dict[str, dict]",
    lexicon: set,
    *,
    enabled: bool,
) -> list:
    """Symmetric LEADING-TITLE split over a VALIDATED arrangement (outline-anchored).

    For each paragraph-typed block whose FIRST unit's text BEGINS with a
    normalized lexicon title (offset 0, ≤60 chars, followed by a sentence
    start), a SYNTHETIC heading block carrying the VERBATIM leading substring is
    inserted BEFORE the paragraph (offsets recorded; provenance pass
    ``heading_sanity_leading_title``; ``duplicate_of_head`` marked so coverage
    never double-counts — the synthetic carries NO unit ids). The paragraph's
    full text stays intact (same conservation approach as the tail titles).
    Returns the split records; ``enabled=False`` / empty lexicon → no-op.
    """
    records: list = []
    if not enabled or not lexicon:
        return records
    blocks = (arr or {}).get("blocks")
    if not isinstance(blocks, list):
        return records
    new_blocks: list = []
    for bi, blk in enumerate(blocks):
        split_rec = None
        if (
            isinstance(blk, dict)
            and blk.get("type") == "paragraph"
            and not blk.get("synthetic_heading")
        ):
            ids = [u for u in (blk.get("ids") or []) if u in unit_by_id]
            if ids:
                head_uid = ids[0]
                # Idempotence guard: never re-split a paragraph whose immediate
                # predecessor is already OUR synthetic leading-title for the
                # same head unit (safe on any re-entry over a split arrangement).
                prev = blocks[bi - 1] if bi > 0 else None
                already_split = (
                    isinstance(prev, dict)
                    and prev.get("synthetic_heading")
                    and prev.get("sanity_pass") == "heading_sanity_leading_title"
                    and prev.get("synthesized_from") == head_uid
                )
                found = None
                if not already_split:
                    found = _match_leading_lexicon_title(
                        unit_by_id[head_uid]["text"], lexicon
                    )
                if found is not None:
                    title, start, end = found
                    new_blocks.append({
                        "type": "heading",
                        "level": 2,
                        "ids": [],
                        "synthetic_heading": True,
                        "sanity_pass": "heading_sanity_leading_title",
                        "text": title,
                        "duplicate_of_head": True,
                        "synthesized_from": head_uid,
                        "head_offsets": [start, end],
                    })
                    split_rec = {
                        "op": "leading_title_split",
                        "block_idx": bi,
                        "reason": "lexicon_match",
                        "leading_title": title,
                        "synthesized_from": head_uid,
                        "head_offsets": [start, end],
                    }
        new_blocks.append(blk)
        if split_rec is not None:
            records.append(split_rec)
    if records:
        arr["blocks"] = new_blocks
    return records


# ---------------------------------------------------------------------------
# Part D: ARRANGE call + 3-rung ladder.
# ---------------------------------------------------------------------------
def _unit_listing(units: list[dict]) -> str:
    return "\n".join(f'{u["id"]}: {u["text"]}' for u in units)


def _arrange_call(
    seat,
    image_b64: str,
    units: list[dict],
    page_label: int,
    *,
    hints_text: str,
    extra_user: str,
    temperature: float,
    include_image: bool,
    timeout: float,
    requests_module=None,
) -> "tuple[Optional[dict], Optional[str], dict, float]":
    """One ARRANGE POST. Returns ``(parsed | None, parse_err | None, usage, wall_s)``.

    Tolerant parse (a malformed body → ``(None, err, ...)`` so the ladder can
    recover it, never raise out of the worker). thinking-off, ``max_tokens``
    6144, ``response_format`` json_object. Text-only when ``include_image`` is
    False or no image bytes (a seat with no vision tower).
    """
    from . import page_arranger_contract as contract
    from .vlm_extract import _chat_completions_url

    requests = requests_module
    if requests is None:
        import requests as requests  # noqa: PLC0415

    user_text = (
        f"Page {page_label}. Arrange these {len(units)} text units into "
        f"reading-ordered typed blocks. Units:\n{_unit_listing(units)}"
    )
    if hints_text:
        user_text += f"\n\n{hints_text}"
    if extra_user:
        user_text += f"\n\n{extra_user}"

    if not include_image or not image_b64:
        user_content: Any = user_text
    else:
        user_content = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
    body = {
        "model": seat.model,
        "messages": [
            {"role": "system", "content": contract.ARRANGE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": _ARRANGE_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if getattr(seat, "api_key", None):
        headers["Authorization"] = f"Bearer {seat.api_key}"
    url = _chat_completions_url(seat.base_url)

    t0 = time.time()
    resp = requests.post(url, json=body, headers=headers, timeout=timeout)
    wall = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {}) or {}
    content = data["choices"][0]["message"]["content"]
    try:
        return contract.extract_json(content), None, usage, wall
    except Exception as exc:  # noqa: BLE001 — tolerant parse
        return None, str(exc), usage, wall


def arrange_page(
    seat,
    image_b64: str,
    units: list[dict],
    page_label: int,
    *,
    hints: "Optional[dict[str, str]]" = None,
    include_image: bool = True,
    timeout: Optional[float] = None,
    hints_provider: "Optional[Callable[[list[dict]], dict[str, str]]]" = None,
    requests_module=None,
) -> dict:
    """Arrange one page with deterministic repair + the 3-rung retry ladder.

      rung 1  temp 0.0  first pass (deterministic hints + hints_provider stub)
      rung 2  temp 0.0  violations named + furniture-bucket directive
      rung 3  temp 0.3  full LEGAL id list restated

    ``hints`` = the deterministic per-unit hints (rendered in ALL rungs).
    ``hints_provider`` (BERT-hint STUB, item 7) — when provided, its per-unit
    outputs MERGE into the deterministic hints for RUNG 1 ONLY (the future
    BERT-v2 integration point; default ``None`` → no real heads wired).

    Returns ``{status, arrangement, attempts, coercions, repairs, problems,
    usages, arrange_s}`` — ``status`` ∈ ``{"ok", "arrangement_failed"}``.
    """
    from . import page_arranger_contract as contract

    hints = dict(hints or {})
    timeout = resolve_arranger_timeout() if timeout is None else timeout
    unit_by_id = {u["id"]: u for u in units}
    legal_ids = [u["id"] for u in units]

    det_hints_text = _render_hints_section(units, hints)
    # Rung-1 hints: deterministic hints MERGED with the (stub) provider outputs.
    rung1_hints = dict(hints)
    if hints_provider is not None:
        try:
            provided = hints_provider(units) or {}
            for uid, label in provided.items():
                rung1_hints.setdefault(uid, label)
        except Exception as exc:  # noqa: BLE001 — a hints provider never breaks arrange
            logger.debug("page-arranger: hints_provider raised (non-fatal): %s", exc)
    rung1_hints_text = _render_hints_section(units, rung1_hints)

    total_wall = 0.0
    usages: list[dict] = []
    coercions: list[dict] = []
    repairs: list[dict] = []
    heading_sanity: list = []

    sanity_on = resolve_arranger_heading_sanity()
    sanity_max = resolve_arranger_heading_max_chars()
    promote_on = resolve_arranger_md_heading_promote()

    def _finish_ok(arr, attempts):
        # Deterministic heading-sanity post-pass runs AFTER validation succeeds,
        # on the final valid arrangement (byte-identical when the flag is off).
        heading_sanity.extend(
            apply_heading_sanity(arr, unit_by_id, max_chars=sanity_max, enabled=sanity_on)
        )
        # VLM-marker heading PROMOTION runs AFTER demotion (demotion owns the
        # model-typed headings; promotion then recovers section titles the model
        # left as paragraph). Reuses the same anti-furniture SoT, so a promoted
        # block is never one demotion would have dropped (byte-identical off).
        heading_sanity.extend(
            apply_md_heading_promote(arr, unit_by_id, max_chars=sanity_max, enabled=promote_on)
        )
        return _ok(arr, attempts, total_wall, usages, coercions, repairs, heading_sanity)

    def _attempt(hints_text: str, extra_user: str, temperature: float):
        nonlocal total_wall
        parsed, parse_err, usage, wall = _arrange_call(
            seat,
            image_b64,
            units,
            page_label,
            hints_text=hints_text,
            extra_user=extra_user,
            temperature=temperature,
            include_image=include_image,
            timeout=timeout,
            requests_module=requests_module,
        )
        total_wall += wall
        usages.append(usage)
        if parsed is None:
            return {"blocks": []}, [f"output was not valid JSON: {parse_err}"]
        repairs.extend(contract.repair_duplicate_ids(parsed))
        coercions.extend(contract.normalize_block_types(parsed, unit_by_id))
        return parsed, contract.validate_arrangement(parsed, units)

    # rung 1
    arr, problems = _attempt(rung1_hints_text, "", 0.0)
    if not problems:
        return _finish_ok(arr, 1)
    # rung 2 — violations named + furniture directive
    arr, problems = _attempt(
        det_hints_text, contract.build_violation_msg(problems, legal_ids=None), 0.0
    )
    if not problems:
        return _finish_ok(arr, 2)
    # rung 3 — full legal id list + slight diversification
    arr, problems = _attempt(
        det_hints_text, contract.build_violation_msg(problems, legal_ids=legal_ids), 0.3
    )
    if not problems:
        return _finish_ok(arr, 3)
    return {
        "status": "arrangement_failed",
        "arrangement": arr,
        "attempts": 3,
        "arrange_s": total_wall,
        "usages": usages,
        "problems": problems,
        "coercions": coercions,
        "repairs": repairs,
        "heading_sanity": heading_sanity,
    }


def _ok(arr, attempts, wall, usages, coercions, repairs, heading_sanity=None) -> dict:
    return {
        "status": "ok",
        "arrangement": arr,
        "attempts": attempts,
        "arrange_s": wall,
        "usages": usages,
        "problems": [],
        "coercions": coercions,
        "repairs": repairs,
        "heading_sanity": heading_sanity or [],
    }


# ---------------------------------------------------------------------------
# Part E: per-page sidecar + stop (mirror reasoning_qc._fan_out_page_verifies).
# ---------------------------------------------------------------------------
def _cache_root() -> Path:
    from . import paths as _paths

    return _paths.resolve_cache(_CACHE_BASENAME)


def _cache_path(key: str, root: Path) -> Path:
    return root / key[:2] / f"{key}.json"


def _unit_fingerprint(units: list[dict], hints_text: str, *, model: str, include_image: bool) -> str:
    """Content-address one PAGE arrange unit → a sha256 hex key.

    Hashes the unit listing text, the deterministic-hints text, CONTRACT_VERSION,
    the ARRANGE_SYSTEM sha, the model id, the image-included bool, max_tokens, and
    the ladder identity — any change moves the key so a stale sidecar is never
    served for a changed input.
    """
    from . import page_arranger_contract as contract

    listing = _unit_listing(units)
    sys_sha = hashlib.sha256(contract.ARRANGE_SYSTEM.encode("utf-8")).hexdigest()[:16]
    # The heading-sanity flag + ceiling change the ACCEPTED arrangement, so they
    # salt the key — a flag flip never serves a stale sidecar written under the
    # other setting.
    hs = int(resolve_arranger_heading_sanity())
    hs_max = resolve_arranger_heading_max_chars()
    # The VLM-marker heading-PROMOTE flag also changes the accepted arrangement,
    # so it salts the key too — a flip never serves a stale (pre-promotion)
    # sidecar. The vlm_heading_level signal itself is unit-derived + deterministic
    # for a given PDF+flags, so the flag bit is a sufficient salt.
    hp = int(resolve_arranger_md_heading_promote())
    raw = (
        f"{listing}␟{hints_text}␟{contract.CONTRACT_VERSION}␟{sys_sha}"
        f"␟{model}␟{int(bool(include_image))}␟{_ARRANGE_MAX_TOKENS}"
        f"␟ladder:0.0/0.0/0.3␟hs:{hs}/{hs_max}␟hp:{hp}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(path: Path) -> "Optional[dict]":
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else None
    except Exception as exc:  # noqa: BLE001 — corrupt/unreadable sidecar → miss
        logger.debug("page-arranger cache read failed (non-fatal) %s: %s", path, exc)
        return None


def _cache_put(path: Path, verdict: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(verdict, ensure_ascii=False))
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 — a cache write must never break arrange
        logger.debug("page-arranger cache write failed (non-fatal) %s: %s", path, exc)


def _stop_requested() -> bool:
    try:
        from .stop_seam import stop_sentinel_present

        return bool(stop_sentinel_present())
    except Exception:  # noqa: BLE001 — a probe error is never a stop
        return False


def _check_stop(site_id: str) -> None:
    from .stop_seam import check_cascade_stop

    check_cascade_stop(site_id)


@dataclass
class _PageInput:
    page_label: int
    units: list[dict]
    hints: "dict[str, str]"
    hints_text: str
    image_b64: str


def _fan_out_arrange(
    seat,
    page_inputs: "list[_PageInput]",
    *,
    include_image: bool,
    log: Callable[[str], None],
    hints_provider: "Optional[Callable[[list[dict]], dict[str, str]]]" = None,
    stop_site_id: str = "cascade:page-arranger-page",
    requests_module=None,
) -> "dict[int, dict]":
    """Fan the per-PAGE arrange calls out concurrently, resume-cached + stop-aware.

    Mirrors ``reasoning_qc._fan_out_page_verifies``: submit-one-consume-one, the
    stop sentinel probed (non-raising) before EACH new submission; on a stop the
    fan-out stops submitting, drains in-flight (each persists to its sidecar),
    then RAISES via :func:`_check_stop` — worst-case loss ≤ concurrency. Only a
    ``status=='ok'`` result is cached (a failed page re-runs). Returns
    ``{page_label: result}`` (result is the ``arrange_page`` dict).
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    if not page_inputs:
        return {}
    use_cache = resolve_arranger_checkpoint()
    max_workers = max(1, min(resolve_arranger_concurrency(), len(page_inputs)))
    model = str(getattr(seat, "model", "") or "")

    def _work(pin: _PageInput) -> dict:
        cache_path: Optional[Path] = None
        if use_cache:
            try:
                key = _unit_fingerprint(
                    pin.units, pin.hints_text, model=model, include_image=include_image
                )
                cache_path = _cache_path(key, _cache_root())
                cached = _cache_get(cache_path)
                if cached is not None:
                    return cached
            except Exception as exc:  # noqa: BLE001 — cache infra never breaks arrange
                logger.debug("page-arranger cache lookup failed (non-fatal): %s", exc)
                cache_path = None
        try:
            res = arrange_page(
                seat,
                pin.image_b64,
                pin.units,
                pin.page_label,
                hints=pin.hints,
                include_image=include_image,
                hints_provider=hints_provider,
                requests_module=requests_module,
            )
        except Exception as exc:  # noqa: BLE001 — per-page fail-soft
            log(f"[cascade] page-arranger page {pin.page_label} arrange raised fail-soft: {exc}")
            return {"status": "arrangement_failed", "arrangement": {"blocks": []},
                    "attempts": 0, "problems": [f"exception: {exc}"],
                    "coercions": [], "repairs": [], "usages": [], "arrange_s": 0.0,
                    "heading_sanity": []}
        if use_cache and cache_path is not None and res.get("status") == "ok":
            _cache_put(cache_path, res)
        return res

    results: dict[int, dict] = {}
    by_future: dict[Any, _PageInput] = {}
    next_idx = 0
    n = len(page_inputs)
    stop_requested = False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        in_flight: dict[Any, _PageInput] = {}
        while next_idx < n and len(in_flight) < max_workers:
            if _stop_requested():
                stop_requested = True
                break
            pin = page_inputs[next_idx]
            next_idx += 1
            in_flight[pool.submit(_work, pin)] = pin
        while in_flight:
            done, _pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                pin = in_flight.pop(future)
                try:
                    results[pin.page_label] = future.result()
                except BaseException as exc:  # noqa: BLE001 — belt: never propagate
                    log(f"[cascade] page-arranger future raised fail-soft (page {pin.page_label}): {exc}")
                    results[pin.page_label] = {
                        "status": "arrangement_failed", "arrangement": {"blocks": []},
                        "attempts": 0, "problems": [f"future exception: {exc}"],
                        "coercions": [], "repairs": [], "usages": [], "arrange_s": 0.0,
                        "heading_sanity": [],
                    }
            while not stop_requested and next_idx < n and len(in_flight) < max_workers:
                if _stop_requested():
                    stop_requested = True
                    break
                pin = page_inputs[next_idx]
                next_idx += 1
                in_flight[pool.submit(_work, pin)] = pin

    if stop_requested:
        log(
            "[cascade] page-arranger: stop sentinel observed mid-fan-out — "
            "in-flight pages drained + checkpointed; pausing."
        )
        _check_stop(stop_site_id)
    return results


# ---------------------------------------------------------------------------
# Part F: Region build — the production 'render' (emit Regions == the same
# accessible-HTML block contract the cascade emits).
# ---------------------------------------------------------------------------
# Map an arrange block type -> (Region.kind, optional css_class, optional
# semantic_class). Downstream renders from region_provenance raw_text, so
# emitting Regions IS the accessible-HTML block contract.
_TYPE_TO_KIND: "dict[str, tuple[str, Optional[str], Optional[str]]]" = {
    "heading": ("heading", None, None),
    "paragraph": ("paragraph", None, None),
    "table": ("table", None, None),
    "figure_caption": ("paragraph", "figure-caption", None),
    "example": ("paragraph", "pedagogy-example", "worked_example"),
    "solution": ("paragraph", "pedagogy-solution", None),
    "definition_box": ("paragraph", "definition-box", "definition_region"),
    "exercise_list": ("list", None, None),
    "furniture": ("metadata_drop", None, None),
}


def _clamp_level(level: Any) -> int:
    try:
        lvl = int(level)
    except (TypeError, ValueError):
        return 2
    return max(1, min(6, lvl))


def build_regions_for_page(
    page_label: int,
    units: list[dict],
    result: dict,
    feature_blocks: list,
) -> list:
    """Map one page's arrangement (or fallback) into ``Region`` objects.

    A VALID page maps each arrange block to a Region in reading order; a FAILED
    page (post-ladder) emits ONE paragraph Region PER UNIT in extraction order (a
    loud, content-preserving fallback — a page is NEVER dropped). Every emitted
    Region carries ``feature_block_indices`` (sorted member fb indices),
    ``payload['typing_authority']='vlm-arranger'``, ``payload['confidence']``, and
    a ``provenance`` breadcrumb. Furniture → an EMPTY-payload ``metadata_drop``
    Region that is still a region row (the dropped-at-render-but-listed contract).
    """
    from . import page_arranger_contract as contract
    from .structure_graph import Region, compute_region_page_bboxes

    unit_by_id = {u["id"]: u for u in units}
    status = result.get("status")
    arrangement = result.get("arrangement") or {}
    blocks = arrangement.get("blocks") or []
    confidence = arrangement.get("confidence")
    attempts = result.get("attempts", 0)

    regions: list = []

    def _mk(kind, fb_idx_list, payload, css_class=None, semantic_class=None,
            provenance_extra=None):
        payload = dict(payload)
        payload["typing_authority"] = "vlm-arranger"
        if confidence is not None:
            payload.setdefault("confidence", confidence)
        if css_class:
            payload["css_class"] = css_class
        if semantic_class:
            payload["semantic_class"] = semantic_class
        fb_tuple = tuple(sorted(int(i) for i in fb_idx_list))
        prov = {
            "pass": "page_arranger",
            "contract_version": contract.CONTRACT_VERSION,
            "attempts": attempts,
            "arrangement_status": status,
        }
        if provenance_extra:
            prov.update(provenance_extra)
        return Region(
            kind=kind,
            feature_block_indices=fb_tuple,
            payload=payload,
            provenance=prov,
            page_bboxes=compute_region_page_bboxes(fb_tuple, feature_blocks),
        )

    if status == "ok" and blocks:
        for blk in blocks:
            # Synthetic heading (heading-sanity post-pass — tail-title OR
            # leading-title): carries the VERBATIM extracted text but NO unit
            # ids, so its Region claims ZERO FeatureBlocks (the source block
            # still claims them all) — the coverage gate never double-counts.
            # `duplicate_of_tail` / `duplicate_of_head` mark that the text is a
            # verbatim copy of the source unit's tail/head slice.
            if isinstance(blk, dict) and blk.get("synthetic_heading"):
                title = (blk.get("text") or "").strip()
                if not title:
                    continue
                sanity_pass = blk.get("sanity_pass") or "heading_sanity_tail_title"
                payload = {
                    "arrange_type": "heading",
                    "text": title,
                    "level_hint": _clamp_level(blk.get("level")),
                    "synthesized_from": blk.get("synthesized_from"),
                }
                if blk.get("duplicate_of_head"):
                    payload["duplicate_of_head"] = True
                    payload["head_offsets"] = blk.get("head_offsets")
                else:
                    payload["duplicate_of_tail"] = True
                    payload["tail_offsets"] = blk.get("tail_offsets")
                regions.append(_mk(
                    "heading", [], payload,
                    provenance_extra={
                        "pass": sanity_pass,
                        "synthesized_from": blk.get("synthesized_from"),
                    },
                ))
                continue
            ids = [uid for uid in (blk.get("ids") or []) if uid in unit_by_id]
            if not ids:
                continue
            member_units = [unit_by_id[uid] for uid in ids]
            fb_idxs = [u["fb_index"] for u in member_units]
            texts = [u["text"] for u in member_units]
            btype = blk.get("type", "paragraph")
            kind, css_class, semantic_class = _TYPE_TO_KIND.get(
                btype, ("paragraph", None, None)
            )
            payload: dict[str, Any] = {"arrange_type": btype}
            joined = " ".join(t.replace("\n", " ").strip() for t in texts).strip()
            if kind == "heading":
                payload["text"] = joined
                payload["level_hint"] = _clamp_level(blk.get("level"))
            elif kind == "list":
                items = _split_list_items(texts)
                payload["items"] = items or [joined]
                payload["text"] = joined
            elif kind == "metadata_drop":
                # dropped-at-render-but-listed: still a region row, EMPTY body.
                payload["text"] = ""
                payload["furniture_text"] = joined
            else:
                payload["text"] = joined
            regions.append(_mk(kind, fb_idxs, payload, css_class, semantic_class))
    else:
        # FAILED page (or empty arrangement): one paragraph Region per unit,
        # content preserved, loud audit flag on each region.
        for u in units:
            regions.append(
                _mk(
                    "paragraph",
                    [u["fb_index"]],
                    {"text": u["text"], "arrangement_failed": True},
                )
            )
    return regions


def _split_list_items(texts: list[str]) -> list[str]:
    """Split joined exercise/list unit texts into `items` (matches the prototype
    render's list splitter)."""
    from .page_arranger_contract import _LIST_ITEM
    import re as _re

    items: list[str] = []
    for t in texts:
        for line in t.splitlines():
            line = line.strip()
            if not line:
                continue
            if _LIST_ITEM.match(line):
                items.append(_re.sub(_LIST_ITEM, "", line).strip())
            elif items:
                items[-1] = (items[-1] + " " + line).strip()
            else:
                items.append(line)
    return items


# ---------------------------------------------------------------------------
# ArrangerRoute — the cascade-facing façade.
# ---------------------------------------------------------------------------
@dataclass
class ArrangerRoute:
    """The cascade seam's handle on the arranger path.

    Holds the whole-doc ``shared`` extraction + ``feature_set`` (so the cascade
    reuses the SAME extraction — no re-extract), and exposes
    :meth:`council_substitute` (the de-BERT step — council heads NEVER load) +
    :meth:`arrange_regions` (the fan-out + Region assembly, the production
    'render' step).
    """

    pdf_path: Path
    shared: dict
    feature_set: Any

    @property
    def feature_blocks(self) -> list:
        return self.feature_set.feature_blocks

    def council_substitute(self):
        """Return ``(CouncilState(outputs={}), council_regions=[], feature_blocks)``.

        The de-BERT step: the cascade uses this INSTEAD of ``run_council`` on the
        arranger route, so the council BERTs never load. Downstream consumers of
        an empty ``CouncilState`` (``signal_coverage``, the cross-reranker
        table/image confirm, Stage-6 authoring) all no-op cleanly on empty
        outputs / empty council_regions.
        """
        from .council.types import CouncilState

        return CouncilState(outputs={}), [], self.feature_blocks

    def arrange_regions(
        self,
        feature_blocks: list,
        *,
        log: Optional[Callable[[str], None]] = None,
        hints_provider: "Optional[Callable[[list[dict]], dict[str, str]]]" = None,
        requests_module=None,
    ) -> "tuple[list, dict]":
        """Run the fan-out + assemble Regions. Returns ``(structure_regions, audit)``.

        This is where the per-page arrange fan-out runs (STEP: the fan-out runs
        HERE, not in ``resolve_page_arranger_route``). Enforces the coverage
        invariant (every non-image FB in exactly one Region). ``audit`` is the
        additive ``result['page_arranger']`` arm.
        """
        _log = log or (lambda _m: None)
        seat = resolve_arranger_seat()
        # The arrange call always attempts to include the page image; a seat
        # without a vision tower degrades to text-only at HTTP time.
        include_image = True

        units_by_page = mint_units_by_page(feature_blocks)
        running_hdr_ids = _detect_running_header_ids(units_by_page)

        # Sequential pre-pass: render page images + build hints (pypdfium2 not
        # thread-safe on one handle).
        page_inputs: list[_PageInput] = []
        render_failures: list[dict] = []
        for page in sorted(units_by_page.keys()):
            units = units_by_page[page]
            if not units:
                continue
            hints = build_page_hints(units, running_hdr_ids)
            hints_text = _render_hints_section(units, hints)
            img_b64 = ""
            try:
                img_b64 = _render_page_image_b64(self.pdf_path, page)
            except Exception as exc:  # noqa: BLE001 — text-only fallback for this page
                _log(f"[cascade] page-arranger page {page} render failed (text-only): {exc}")
                render_failures.append({"page": page, "error": str(exc)})
            page_inputs.append(
                _PageInput(page_label=page, units=units, hints=hints,
                           hints_text=hints_text, image_b64=img_b64)
            )

        results = _fan_out_arrange(
            seat,
            page_inputs,
            include_image=include_image,
            log=_log,
            hints_provider=hints_provider,
            requests_module=requests_module,
        )

        # Document-level LEADING-TITLE split (heading-sanity Part C3): the
        # outline-anchored lexicon needs the WHOLE document (the chapter-outline
        # unit + headings typed on any page), so this symmetric arm runs here —
        # after the fan-out, before assembly. In-memory only: per-page sidecars
        # were persisted pre-split, and the split re-applies deterministically
        # on cached results (idempotent — synthetics are skipped on re-entry).
        if resolve_arranger_heading_sanity():
            lexicon = build_title_lexicon(units_by_page, results)
            if lexicon:
                for pin in page_inputs:
                    res = results.get(pin.page_label)
                    if not isinstance(res, dict) or res.get("status") != "ok":
                        continue
                    recs = apply_leading_title_split(
                        res.get("arrangement") or {},
                        {u["id"]: u for u in pin.units},
                        lexicon,
                        enabled=True,
                    )
                    if recs:
                        res.setdefault("heading_sanity", []).extend(recs)

        # Assemble Regions in page order; label-factory + coverage.
        train_dir = resolve_arranger_train_records_dir()
        run_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
        structure_regions: list = []
        page_rows: list[dict] = []
        n_valid = n_failed = 0
        for pin in page_inputs:
            result = results.get(pin.page_label) or {
                "status": "arrangement_failed", "arrangement": {"blocks": []},
                "attempts": 0, "problems": ["no result"], "coercions": [],
                "repairs": [], "usages": [], "arrange_s": 0.0, "heading_sanity": [],
            }
            regions = build_regions_for_page(
                pin.page_label, pin.units, result, feature_blocks
            )
            structure_regions.extend(regions)
            valid = result.get("status") == "ok"
            n_valid += int(valid)
            n_failed += int(not valid)
            hs_rows = result.get("heading_sanity", []) or []
            page_rows.append({
                "page": pin.page_label,
                "units": len(pin.units),
                "status": result.get("status"),
                "attempts": result.get("attempts", 0),
                "coercions": len(result.get("coercions", []) or []),
                "repairs": len(result.get("repairs", []) or []),
                "heading_sanity": len(hs_rows),
                "heading_sanity_tail_titles": sum(1 for r in hs_rows if r.get("tail_title")),
                "heading_sanity_leading_titles": sum(1 for r in hs_rows if r.get("leading_title")),
                "problems": result.get("problems", []) or [],
            })
            if valid and train_dir is not None:
                try:
                    _emit_training_record(train_dir, pin, result, run_ts)
                except Exception as exc:  # noqa: BLE001 — label factory never breaks
                    _log(f"[cascade] page-arranger train-record write failed page {pin.page_label}: {exc}")

        if train_dir is not None:
            try:
                finalize_continues_over_dir(train_dir)
            except Exception as exc:  # noqa: BLE001
                _log(f"[cascade] page-arranger continues finalize failed: {exc}")

        # Coverage invariant: every non-image FB in exactly one region.
        _assert_coverage(feature_blocks, structure_regions)

        _emit_arranger_capture(page_rows, n_valid, n_failed, str(getattr(seat, "model", "")))

        audit = {
            "pass": "page_arranger",
            "contract_version": 2,
            "seat_model": str(getattr(seat, "model", "")),
            "pages": len(page_inputs),
            "pages_valid": n_valid,
            "pages_failed": n_failed,
            "render_failures": render_failures,
            "page_rows": page_rows,
        }
        return structure_regions, audit


def _assert_coverage(feature_blocks: list, regions: list) -> None:
    """Assert every non-image FB index is claimed by exactly one Region."""
    expected = {
        i for i, fb in enumerate(feature_blocks) if not getattr(fb, "is_image", False)
        and (getattr(getattr(fb, "raw", None), "text", "") or "").strip()
    }
    claimed: dict[int, int] = {}
    for r in regions:
        for idx in getattr(r, "feature_block_indices", ()) or ():
            claimed[idx] = claimed.get(idx, 0) + 1
    dupes = {i for i, c in claimed.items() if c > 1}
    missing = expected - set(claimed)
    extra = set(claimed) - expected
    if dupes or missing or extra:
        raise AssertionError(
            f"page-arranger coverage invariant violated: "
            f"missing={sorted(missing)[:20]} dupes={sorted(dupes)[:20]} "
            f"extra={sorted(extra)[:20]}"
        )


# ---------------------------------------------------------------------------
# Part G: label factory (schema_version-2 train records) + DecisionCapture.
# ---------------------------------------------------------------------------
def _emit_training_record(train_dir: Path, pin: "_PageInput", result: dict, timestamp: str) -> Path:
    """Write one schema_version-2 train record per VALID page."""
    from . import page_arranger_contract as contract

    train_dir.mkdir(parents=True, exist_ok=True)
    arrangement = result.get("arrangement") or {}
    blocks = list(arrangement.get("blocks") or [])
    for blk in blocks:  # null level on non-heading (v2)
        if isinstance(blk, dict) and blk.get("type") != "heading":
            blk["level"] = None
    pw, ph, dims_src = contract.page_dims_from_bboxes(pin.units)
    rec = {
        "schema_version": 2,
        "source_page": pin.page_label,
        "seat": None,  # filled by the caller model if desired; kept schema-stable
        "page_width": pw,
        "page_height": ph,
        "dims_source": dims_src,
        "units": [
            {"id": u["id"], "text": u["text"], "bbox": u.get("bbox"), "order_in": i}
            for i, u in enumerate(pin.units)
        ],
        "arrangement": {
            "blocks": blocks,
            "confidence": arrangement.get("confidence"),
            "coercions": result.get("coercions", []),
            "repairs": result.get("repairs", []),
            "heading_sanity": result.get("heading_sanity", []),
        },
        "relations": contract.derive_relations_v2(blocks),
        "relations_schema": "relations_schema.json",
        "timestamp_from_summary": timestamp,
    }
    p = train_dir / f"train_p{pin.page_label}.json"
    p.write_text(json.dumps(rec, indent=2))
    return p


def finalize_continues_over_dir(results_dir: Path) -> dict:
    """Dir-wide cross-page ``continues`` materialization + ``relations_schema.json``.

    Idempotent: every ``train_p*.json`` in the dir has its ``continues`` edges
    stripped then re-added iff its first block carries ``continues_prev_page`` and
    a ``to`` unit resolves; ``from`` = last content unit of page N-1's record
    (null when N-1 is absent). Ported from the prototype.
    """
    import re as _re
    from . import page_arranger_contract as contract

    recs: dict[int, dict] = {}
    for tp in results_dir.glob("train_p*.json"):
        m = _re.match(r"train_p(\d+)\.json$", tp.name)
        if not m:
            continue
        try:
            recs[int(m.group(1))] = json.loads(tp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    n_edges = n_resolved = 0
    for page_label, rec in recs.items():
        blocks = (rec.get("arrangement") or {}).get("blocks") or []
        unit_ids = {u["id"] for u in rec.get("units") or []}
        rels = [r for r in (rec.get("relations") or []) if r.get("type") != "continues"]
        if blocks and blocks[0].get("continues_prev_page"):
            to_ids = [i for i in (blocks[0].get("ids") or []) if i in unit_ids]
            to_id = to_ids[0] if to_ids else None
            if to_id is not None:
                prev = recs.get(page_label - 1)
                frm = None
                if prev is not None:
                    prev_blocks = (prev.get("arrangement") or {}).get("blocks") or []
                    prev_ids = {u["id"] for u in prev.get("units") or []}
                    frm = contract.last_content_unit_id(prev_blocks, prev_ids)
                rels.append({"type": "continues", "from": frm, "to": to_id})
                n_edges += 1
                if frm:
                    n_resolved += 1
        rec["relations"] = rels
        (results_dir / f"train_p{page_label}.json").write_text(json.dumps(rec, indent=2))
    (results_dir / "relations_schema.json").write_text(
        json.dumps(contract.RELATIONS_SCHEMA, indent=2)
    )
    return {"records": len(recs), "continues_edges": n_edges, "continues_resolved_from": n_resolved}


def _emit_arranger_capture(page_rows: list, n_valid: int, n_failed: int, model: str) -> None:
    """Best-effort per-document DecisionCapture (mirrors vlm_extract's posture).

    Rides the EXISTING ``structure_detection`` decision_type enum (no schema
    change). Dynamic rationale: pages, valid/failed, coercion/repair tallies,
    model, attempts histogram. Never breaks conversion.
    """
    try:
        from lib.decision_capture import DecisionCapture

        cap = DecisionCapture(course_code="_semantik", phase="dart-conversion", tool="dart")
    except Exception as exc:  # noqa: BLE001 — capture best-effort (bridge venv may lack lib/)
        logger.debug("page-arranger DecisionCapture unavailable (non-fatal): %s", exc)
        return
    try:
        total_coerce = sum(r.get("coercions", 0) for r in page_rows)
        total_repair = sum(r.get("repairs", 0) for r in page_rows)
        total_hsanity = sum(r.get("heading_sanity", 0) for r in page_rows)
        total_tail = sum(r.get("heading_sanity_tail_titles", 0) for r in page_rows)
        total_lead = sum(r.get("heading_sanity_leading_titles", 0) for r in page_rows)
        attempts_hist: dict[int, int] = {}
        for r in page_rows:
            a = int(r.get("attempts", 0) or 0)
            attempts_hist[a] = attempts_hist.get(a, 0) + 1
        rationale = (
            f"page-arranger structure: model={model} pages={len(page_rows)} "
            f"valid={n_valid} failed={n_failed} coercions={total_coerce} "
            f"repairs={total_repair} heading_sanity_ops={total_hsanity} "
            f"tail_titles={total_tail} leading_titles={total_lead} "
            f"attempts_hist={attempts_hist} "
            f"(contract_version=2, typing_authority=vlm-arranger)."
        )
        cap.log_decision(
            decision_type="structure_detection",
            decision=f"arrange {len(page_rows)} pages ({n_valid} valid, {n_failed} failed)",
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("page-arranger DecisionCapture log failed (non-fatal): %s", exc)
