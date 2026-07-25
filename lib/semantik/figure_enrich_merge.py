"""Figure-enrichment preservation merge for heading-judge re-renders.

Root defect (whole-book single-PDF reference corpus, 2026-07-22): the VLM alt-text
pass (``SEMANTIK_ALTTEXT_PROVIDER=qwen30``) enriches caption-less figure
regions IN MEMORY (``figure_alt`` / ``caption_text`` on
``region_provenance``) *after* ``transform_document`` and *before* the HTML
render — but the ``{stem}.glmocr_layout.json`` sidecar is written from the
RAW GLM pages, so the enrichment is never persisted anywhere except the
rendered ``{stem}_accessible.html``. The standalone heading judge
(``semantik_structure.glmocr.heading_judge_standalone``) re-renders that HTML
from a fresh ``transform_document`` over the layout sidecar, which therefore
has NO enrichment: every figure whose caption was VLM-generated degrades to
the caption-less placeholder
``<figcaption><span class="sr-only">Figure.</span></figcaption>``, and the
pipeline copy-back then ships the degraded bytes over the enriched
conversion output — an accessibility REGRESSION (alt text is a core
conversion promise).

This module is the deterministic repair: given the PRIOR (enriched) HTML and
the JUDGED (re-rendered) HTML, restore each judged figure's caption / alt
from its prior counterpart **only where the judged figure degraded to the
placeholder and the prior figure carries real enrichment**. ADD-only by
construction:

* a judged figure with a real caption (e.g. an extracted "Figure 1.1: …") is
  NEVER touched — the judge's own render wins;
* a prior figure that is itself degraded contributes nothing (the merge
  never invents a caption);
* replacements happen strictly INSIDE ``<figure>…</figure>`` spans, so the
  judge's heading-level corrections — the whole point of the re-render —
  are untouched byte-for-byte.

Figures are paired by the nearest preceding ``data-semantik-block-id``
(dual-read: the legacy pre-SemantiK attribute spelling is admitted too),
falling back to ordinal pairing when ids are absent on either side and the
figure counts agree. Unpairable figures are skipped (never guessed).

Unconditional correctness fix (no behavior flag): a document with no
degraded figures round-trips byte-identically, so legacy corpora are
unaffected by construction. Pure deterministic string transform — NO LLM
call site, NO DecisionCapture obligation.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

# Byte-frozen mirror of ``lib/semantik/cascade_ir._TYPE_LEVEL_ALT`` /
# ``figure_captioner.TYPE_LEVEL_ALT`` — the honest type-level accessible name
# a caption-less figure ships with.
TYPE_LEVEL_ALT = "Figure."

_FIGURE_RE = re.compile(r"<figure\b[^>]*>.*?</figure>", re.IGNORECASE | re.DOTALL)
_FIGCAP_RE = re.compile(
    r"<figcaption\b[^>]*>.*?</figcaption>", re.IGNORECASE | re.DOTALL
)
# Dual-read block-id attribute (the chunker's harvest posture): the emitted
# ``data-semantik-block-id`` plus the legacy pre-SemantiK spelling.
_BLOCK_ID_RE = re.compile(
    r"data-(?:semantik|dart)-block-id=\"([^\"]+)\"", re.IGNORECASE
)
_IMG_ALT_RE = re.compile(r"(<img\b[^>]*?\balt=\")([^\"]*)(\")", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _caption_text(fig_html: str) -> Optional[str]:
    """Visible text content of the figure's ``<figcaption>`` (tags stripped,
    whitespace collapsed); ``None`` when the figure has no figcaption."""
    m = _FIGCAP_RE.search(fig_html)
    if m is None:
        return None
    return " ".join(_TAG_RE.sub(" ", m.group(0)).split())


def _caption_degraded(fig_html: str) -> bool:
    """True when the figure carries no real caption — no figcaption at all,
    an empty one, or the type-level ``"Figure."`` placeholder (both the
    sr-only-span shape and the bare visible shape)."""
    text = _caption_text(fig_html)
    return text is None or text == "" or text == TYPE_LEVEL_ALT


def _figures_with_ids(html: str) -> List[Tuple[int, int, str, Optional[str]]]:
    """Ordered ``(start, end, figure_html, block_id)`` for every ``<figure>``
    span; ``block_id`` is the nearest PRECEDING block-id attribute (the
    enclosing provenance-stamped ``<section>``'s id), or ``None``."""
    id_positions: List[int] = []
    id_values: List[str] = []
    for m in _BLOCK_ID_RE.finditer(html):
        id_positions.append(m.start())
        id_values.append(m.group(1))
    out: List[Tuple[int, int, str, Optional[str]]] = []
    for m in _FIGURE_RE.finditer(html):
        idx = bisect_right(id_positions, m.start()) - 1
        bid = id_values[idx] if idx >= 0 else None
        out.append((m.start(), m.end(), m.group(0), bid))
    return out


def _keyed(
    figs: List[Tuple[int, int, str, Optional[str]]],
) -> List[Tuple[Optional[Tuple[str, int]], Tuple[int, int, str, Optional[str]]]]:
    """Attach a ``(block_id, occurrence)`` key to each figure (``None`` key
    when the figure has no resolvable block id)."""
    counts: Dict[str, int] = {}
    out = []
    for fig in figs:
        bid = fig[3]
        if bid is None:
            out.append((None, fig))
            continue
        occ = counts.get(bid, 0)
        counts[bid] = occ + 1
        out.append(((bid, occ), fig))
    return out


def _restore_one(judged_fig: str, prior_fig: str) -> Tuple[str, bool]:
    """Restore caption / img-alt enrichment from ``prior_fig`` into
    ``judged_fig`` where the judged figure degraded. Returns
    ``(new_fig_html, changed)``."""
    changed = False
    out = judged_fig

    if _caption_degraded(out) and not _caption_degraded(prior_fig):
        prior_cap_m = _FIGCAP_RE.search(prior_fig)
        if prior_cap_m is not None:
            prior_cap = prior_cap_m.group(0)
            judged_cap_m = _FIGCAP_RE.search(out)
            if judged_cap_m is not None:
                out = (
                    out[: judged_cap_m.start()]
                    + prior_cap
                    + out[judged_cap_m.end():]
                )
            else:
                close = out.rfind("</figure>")
                if close < 0:  # malformed span — never guess
                    return judged_fig, False
                out = out[:close] + prior_cap + out[close:]
            changed = True

    # <img alt="…"> — restore a degraded/empty alt from the prior figure.
    jm = _IMG_ALT_RE.search(out)
    if jm is not None and jm.group(2).strip() in ("", TYPE_LEVEL_ALT):
        pm = _IMG_ALT_RE.search(prior_fig)
        if pm is not None and pm.group(2).strip() not in ("", TYPE_LEVEL_ALT):
            out = out[: jm.start(2)] + pm.group(2) + out[jm.end(2):]
            changed = True

    return out, changed


def merge_figure_enrichment(
    prior_html: str, judged_html: str
) -> Tuple[str, int]:
    """Merge figure enrichment from ``prior_html`` into ``judged_html``.

    Returns ``(merged_html, restored_count)``. ``restored_count == 0`` means
    ``merged_html`` is byte-identical to ``judged_html`` (the no-op
    contract). Never raises on well-formed input; a figure that cannot be
    paired or repaired safely is left exactly as the judge rendered it.
    """
    judged = _figures_with_ids(judged_html)
    prior = _figures_with_ids(prior_html)
    if not judged or not prior:
        return judged_html, 0

    prior_keyed = _keyed(prior)
    prior_map: Dict[Tuple[str, int], Tuple[int, int, str, Optional[str]]] = {
        key: fig for key, fig in prior_keyed if key is not None
    }
    counts_equal = len(judged) == len(prior)

    replacements: List[Tuple[int, int, str]] = []
    restored = 0
    for i, (key, fig) in enumerate(_keyed(judged)):
        counterpart: Optional[Tuple[int, int, str, Optional[str]]] = None
        if key is not None:
            counterpart = prior_map.get(key)
        if counterpart is None and counts_equal:
            counterpart = prior[i]
        if counterpart is None:
            continue
        new_fig, changed = _restore_one(fig[2], counterpart[2])
        if changed:
            replacements.append((fig[0], fig[1], new_fig))
            restored += 1

    if not replacements:
        return judged_html, 0

    parts: List[str] = []
    cursor = 0
    for start, end, new_fig in replacements:
        parts.append(judged_html[cursor:start])
        parts.append(new_fig)
        cursor = end
    parts.append(judged_html[cursor:])
    return "".join(parts), restored


__all__ = ["TYPE_LEVEL_ALT", "merge_figure_enrichment"]
