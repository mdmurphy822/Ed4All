"""Document shell template (DOCTYPE, <html lang>, <head>, <main>).

Plans/04 §1.5 — fixed shell:
  * <!DOCTYPE html>
  * <meta charset="utf-8">
  * <title>...</title>
  * <a class="skip-link" href="#main-content">Skip to main content</a>
  * <main id="main-content">

The ``<!-- DART_TITLE_SLOT -->`` comment immediately inside ``<main>`` is
the splice target Stage 9c uses when a ``missing_title`` GapSlot is
filled (the gap-fill output may carry a fresh ``<h1>`` that needs to
land before any other body content).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from .gold_shell_markup import (
    GOLD_ACCESSIBILITY_JSONLD,
    GOLD_DOC_CLOSE,
    GOLD_DOC_OPEN,
    GOLD_JSONLD_SLOT,
    GOLD_STYLE_SLOT,
)
from .wcag22_css import WCAG22_CSS

logger = logging.getLogger(__name__)

# Truthy tokens for the parse-with-fallback gate below. Replicated inline
# (the assembler has no shared frozenset) to match the reviewer's
# ``reviewer._TRUTHY`` set exactly — keep the two in sync.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# One-shot de-duplicated deprecation warning latch for SEMANTIK_GOLD_ABSORB
# (mirrors ``extract_shared.resolve_tesseract_user_words``'s once-per-run
# warning). ITEM1: the absorb is superseded by the default-on Stage-5e regroup;
# it can now engage only under an explicit legacy revert, and warns when it does.
_GOLD_ABSORB_DEPRECATION_WARNED = False


def resolve_gold_shell_mode() -> bool:
    """Return True when the assembler emits the gold ARIA markup + shell.

    Reads ``SEMANTIK_GOLD_SHELL``. Default OFF (unset / blank / falsey /
    garbage) -> byte-identical to today (mirrors the reviewer's
    ``resolve_block_review_mode`` parse-with-fallback posture). A truthy
    value (``1``/``true``/``yes``/``on``, case-insensitive) gates the gold
    per-region ARIA wrap + gold document shell/CSS. A pure read until Phase
    7+ consumes it.
    """
    raw = (os.environ.get("SEMANTIK_GOLD_SHELL") or "").strip().lower()
    return raw in _TRUTHY


def resolve_gold_absorb_mode() -> bool:
    """**DEPRECATED (ITEM1)** — superseded by the default-on Stage-5e regroup
    (``SEMANTIK_UNIT_REGROUP``); scheduled for removal with the containment-tree
    wave (ITEM4). It can now engage ONLY under an explicit legacy revert
    (``SEMANTIK_UNIT_REGROUP=0`` or ``SEMANTIK_READING_ORDER_FIX=0``) and emits a
    one-shot ``logger.warning`` when it does.

    Return True when a body-bearing gold component box ABSORBS the
    following sibling body regions into its container.

    Stopgap for the split EXAMPLE-label/body defect: the council emits an
    example's LABEL ("EXAMPLE 1.3") and its BODY ("Solution" + steps) as
    SEPARATE regions, so the Phase-7 per-region wrap boxes only the label.
    When this is on, the body-bearing component region pulls the immediately
    following non-boundary regions into the same box (see
    ``gold_shell_markup.compute_absorption_runs``).

    Reads ``SEMANTIK_GOLD_ABSORB``. Default OFF (unset / blank / falsey /
    garbage), parse-with-fallback (mirrors ``resolve_gold_shell_mode``).
    ADDITIONALLY requires ``resolve_gold_shell_mode()`` on — absorption only
    matters when components are wrapped, so absorb-off-with-shell-on stays
    byte-identical to the v3 gold render.

    **Phase 7 precedence — SUPERSEDED when the Stage-5e regroup is FIRING.**
    When the cross-kind pedagogical-unit REGROUP is ACTUALLY firing — i.e.
    ``resolve_unit_regroup_mode()`` AND ``resolve_reading_order_fix()`` (the
    SAME composite condition the regroup fires under in
    ``block_resegment.resegment_blocks``) — each unit's label+body has already
    been fused into ONE region carrying the unit's ``semantic_class``, so the
    per-region gold wrap boxes the WHOLE unit and the assembler-side absorb
    would DOUBLE-box. This short-circuits the absorb OFF so the regroup is the
    canonical, mutually-exclusive grouping. Gating on ``unit_regroup_mode``
    ALONE would, in the reading-order-OFF case, make the regroup inert (its
    Phase-6 driver guard) AND kill the absorb -> box NEITHER (a regression vs
    absorb-alone); requiring BOTH keeps the absorb as the FALLBACK whenever the
    regroup is inert. (Defense-in-depth only: after the merge the anchor has no
    absorbable follower left, so ``compute_absorption_runs`` is a natural no-op
    anyway.) Imports are cycle-free — ``resolve_unit_regroup_mode`` from the
    neutral ``pedagogical_units`` and ``resolve_reading_order_fix`` from
    ``structure_graph`` (both dependency-light, no assembler edge); kept
    function-local as belt-and-braces against any future import cycle.
    """
    if not resolve_gold_shell_mode():
        return False
    raw = (os.environ.get("SEMANTIK_GOLD_ABSORB") or "").strip().lower()
    if raw not in _TRUTHY:
        return False
    # Phase 7: regroup-firing (regroup-on AND reading-order-on) supersedes the
    # absorb stopgap (mutual exclusion — never double-group). Regroup-inert
    # (reading-order off) leaves the absorb as the fallback (no regression).
    from ..pedagogical_units import resolve_unit_regroup_mode
    from ..structure_graph import resolve_reading_order_fix

    if resolve_unit_regroup_mode() and resolve_reading_order_fix():
        return False
    # ITEM1: reaching here means the absorb ACTUALLY engages — i.e. an operator
    # explicitly reverted the default-on regroup (SEMANTIK_UNIT_REGROUP=0) or the
    # reading-order fix (SEMANTIK_READING_ORDER_FIX=0). Warn once per run that the
    # stopgap is deprecated and scheduled for removal with the containment tree.
    global _GOLD_ABSORB_DEPRECATION_WARNED
    if not _GOLD_ABSORB_DEPRECATION_WARNED:
        _GOLD_ABSORB_DEPRECATION_WARNED = True
        logger.warning(
            "SEMANTIK_GOLD_ABSORB is DEPRECATED (superseded by the default-on "
            "SEMANTIK_UNIT_REGROUP regroup); scheduled for removal with the "
            "containment-tree wave"
        )
    return True


def resolve_narrow_table_absorb_mode() -> bool:
    """Return True when the table-v2 NARROW passthrough-only absorb should run
    at the ``pass_9a`` Sub-task-1b call site.

    Composite gate (table-v2 Phase 2): the sub-flag
    ``SEMANTIK_UNIT_REGROUP_TABLE`` AND the regroup-firing condition
    (``resolve_unit_regroup_mode()`` AND ``resolve_reading_order_fix()`` — the
    SAME composite ``resolve_gold_absorb_mode`` short-circuits the full absorb
    under at ``:98``). Single-sourcing the composite here keeps the narrow gate
    in LOCKSTEP with the absorb-off short-circuit: if that composite ever gains
    a third guard, both gates move together. ``SEMANTIK_GOLD_SHELL`` is NOT in
    the composite because the call site already lives inside
    ``if resolve_gold_shell_mode():`` (``pass_9a.py``). Reuses the EXACT
    cycle-safe function-local imports of ``resolve_gold_absorb_mode``.

    Effect: when the regroup is firing AND the sub-flag is on, the merged-prose
    box ENCLOSES its trailing FB-adjacent passthrough table (the v1 passthrough
    gap). Default off / regroup inert -> ``False`` -> byte-identical to v1
    (table renders adjacent). Parse-with-fallback throughout.
    """
    from ..pedagogical_units import (
        resolve_unit_regroup_mode,
        resolve_unit_regroup_table_mode,
    )
    from ..structure_graph import resolve_reading_order_fix

    return (
        resolve_unit_regroup_table_mode()
        and resolve_unit_regroup_mode()
        and resolve_reading_order_fix()
    )


DOC_OPEN = (
    '<!DOCTYPE html>\n'
    '<html lang="{lang}">\n'
    '<head>\n'
    '<meta charset="utf-8">\n'
    '<title>{title}</title>\n'
    '</head>\n'
    '<body>\n'
    '<a class="skip-link" href="#main-content">Skip to main content</a>\n'
    '<main id="main-content">\n'
    '<!-- DART_TITLE_SLOT -->\n'
)
DOC_CLOSE = '</main>\n</body>\n</html>\n'

TITLE_SLOT_SENTINEL = '<!-- DART_TITLE_SLOT -->\n'


def build_shell(
    *,
    lang: str,
    title: str,
    fabricated_title: bool = False,
) -> tuple[str, str]:
    """Return ``(doc_open, doc_close)`` for the document shell.

    When ``fabricated_title=True``, inject a
    ``<meta name="dart-title-source" content="fallback">`` immediately
    after the ``<title>`` tag so any consumer of ``assembled.html`` can
    see at a glance that the title was synthesised by the assembler
    (no doc_role:title region, no input h1, no Qwen gap-fill survivor)
    rather than extracted from the source document. Default ``False``
    keeps backward compatibility with existing callers.
    """
    if resolve_gold_shell_mode():
        # Gold accessible shell branch — returns the NEW gold constants
        # (authored in ``gold_shell_markup``), NEVER mutating the existing
        # minimal ``DOC_OPEN``/``DOC_CLOSE`` constants in place. ``GOLD_DOC_OPEN``
        # keeps the EXACT ``{lang}``/``{title}`` ``.format()`` contract, so the
        # call below is byte-for-byte the same shape as the minimal branch.
        doc_open = GOLD_DOC_OPEN.format(
            lang=_escape_attr(lang), title=_escape(title),
        )
        # Splice the WCAG CSS + schema.org JSON-LD at the brace-free sentinels
        # via ``str.replace`` (NOT ``.format`` — both payloads carry literal
        # ``{}`` that would break ``str.format``).
        doc_open = doc_open.replace(
            GOLD_STYLE_SLOT, f"<style>{WCAG22_CSS}</style>", 1,
        )
        doc_open = doc_open.replace(
            GOLD_JSONLD_SLOT, GOLD_ACCESSIBILITY_JSONLD, 1,
        )
        doc_close = GOLD_DOC_CLOSE
    else:
        doc_open = DOC_OPEN.format(
            lang=_escape_attr(lang), title=_escape(title),
        )
        doc_close = DOC_CLOSE

    if fabricated_title:
        doc_open = doc_open.replace(
            f"<title>{_escape(title)}</title>\n",
            (
                f"<title>{_escape(title)}</title>\n"
                '<meta name="dart-title-source" content="fallback">\n'
            ),
            1,
        )
    return (doc_open, doc_close)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _escape_attr(s: str) -> str:
    return _escape(s)


_LANG_ALLOWLIST = frozenset({"en", "es", "fr", "de", "zh", "ja"})


def detect_language(feature_blocks: Sequence[Any]) -> str:
    """Use lingua to detect doc language; return BCP-47 tag. Default 'en'.

    Mirrors the cascade in ``dart_semantic/enrich.py:69-94`` (a single
    cached detector, ``with_low_accuracy_mode`` so no model download
    blocks the smoke harness). Returns ``"en"`` on any failure mode —
    language wrong is a low-cost mistake (Plans/04 §1.5 rationale).
    """
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        return "en"
    text_parts: list[str] = []
    for fb in feature_blocks[:200]:
        raw = getattr(fb, "raw", None)
        text = getattr(raw, "text", None) if raw is not None else None
        if text:
            text_parts.append(text)
    text = " ".join(text_parts)[:8000]
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 200:
        return "en"
    try:
        detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_low_accuracy_mode()
            .build()
        )
        lang = detector.detect_language_of(text)
    except Exception:
        return "en"
    if lang is None:
        return "en"
    try:
        code = lang.iso_code_639_1.name.lower()
    except Exception:
        return "en"
    return code if code in _LANG_ALLOWLIST else "en"


__all__ = [
    "DOC_CLOSE",
    "DOC_OPEN",
    "TITLE_SLOT_SENTINEL",
    "build_shell",
    "detect_language",
    "resolve_gold_absorb_mode",
    "resolve_gold_shell_mode",
]
