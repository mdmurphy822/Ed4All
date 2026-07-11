"""Stage 9c — Deterministic merge-back.

For each gap with at least one survivor (per-region gate already
passed), apply a rule-based per-gap scorer and splice the argmax
candidate into the pre-9a HTML. For gaps with zero survivors, splice
:attr:`GapSlot.fallback_html` instead.

Per the v1 cuts (user spec): the heading-tree normalization is NOT
re-run after splice. The splice operations are carefully scoped so
the heading-tree stays valid.

Splice contracts (per Plans/04 §4.1):

  * ``missing_title``: extract ``<title>...</title>`` and ``<h1>...</h1>``
    from the candidate; replace the doc shell ``<title>`` and inject
    the ``<h1>`` at the ``<!-- DART_TITLE_SLOT -->`` sentinel.
  * ``author_block``: locate the original region's HTML in the body
    and replace it with the ``<address>...</address>`` candidate.
  * ``citation_unresolved``: replace the matched plain-text span (e.g.
    "Section 4.2") with the candidate's chosen ``<a href="#...">`` IFF
    the anchor target is one of the gap's candidate_targets; otherwise
    (plain-text confirmation or a hallucinated anchor) splice nothing.
  * ``copyright_block`` / ``legal_disclaimer``: locate the original
    region's HTML (the ``region_html`` key 9a stashed in the gap
    context) and replace it with the candidate's ``<aside>``/``<footer>``
    fragment. The splice refuses candidates with no aside/footer root
    (and, for copyright, candidates that dropped the ©/copyright
    keyword); a refused splice falls through to the deterministic
    ``fallback_html`` aside wrap (Plans/04 §3.4 / §4.1 — body-scope
    ``<footer>`` collection is owned by the page template per
    architecture.md, so the splice is strictly in-place).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..qwen_specialists.prompts import COPYRIGHT_SUBFLAG_RE
from ..qwen_specialists.types import Candidate
from ..structure_graph import Region
from ..types import FeatureBlock
from .heading_tree import slugify
from .types import AssembledDoc, GapKind, GapSlot

if TYPE_CHECKING:
    from .api import AssemblerConfig


_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TITLE_OPEN_RE = re.compile(r"<title\b([^>]*)>", re.IGNORECASE)
_H1_RE = re.compile(r"<h1\b[^>]*>.*?</h1>", re.IGNORECASE | re.DOTALL)
_H1_OPEN_RE = re.compile(r"<h1\b([^>]*)>", re.IGNORECASE)
_H1_INNER_RE = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_ID_ATTR_RE = re.compile(r'\bid\s*=\s*"[^"]*"', re.IGNORECASE)
_DOC_IDS_RE = re.compile(r'\bid\s*=\s*"([^"]+)"', re.IGNORECASE)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_ADDRESS_RE = re.compile(
    r"<address\b[^>]*>.*?</address>",
    re.IGNORECASE | re.DOTALL,
)
# First <a href="#anchor">...</a> in a gap-fill citation candidate.
_ANCHOR_RE = re.compile(
    r"<a\b[^>]*\bhref\s*=\s*\"#([^\"]+)\"[^>]*>.*?</a>",
    re.IGNORECASE | re.DOTALL,
)
# First <aside>...</aside> or <footer>...</footer> in a gap-fill
# copyright/legal candidate (Plans/04 §3.2 accepts either root; the
# canonical dataset/fallback shape is <aside class="copyright|legal">).
_ASIDE_FOOTER_RE = re.compile(
    r"<(aside|footer)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _ensure_h1_id(h1_html: str, doc_html: str) -> str:
    """Add an ``id="..."`` to the gap-fill ``<h1>`` if missing, deduped
    against IDs already present in ``doc_html``."""
    open_m = _H1_OPEN_RE.search(h1_html)
    if open_m is None:
        return h1_html
    attrs = open_m.group(1) or ""
    if _ID_ATTR_RE.search(attrs):
        return h1_html
    inner_m = _H1_INNER_RE.search(h1_html)
    text = _TAG_STRIP_RE.sub("", inner_m.group(1)).strip() if inner_m else ""
    base = slugify(text) or "section-1"
    existing = set(_DOC_IDS_RE.findall(doc_html))
    ident = base
    n = 2
    while ident in existing:
        ident = f"{base}-{n}"
        n += 1
    safe = ident.replace('"', "&quot;")
    new_open = f'<h1 id="{safe}"{attrs}>'
    return h1_html[: open_m.start()] + new_open + h1_html[open_m.end() :]


def _expected_root_tag(kind: GapKind) -> str | None:
    if kind is GapKind.MISSING_TITLE:
        return "title"  # candidate carries <title> + <h1>; check title
    if kind is GapKind.AUTHOR_BLOCK:
        return "address"
    return None


def _expected_length(kind: GapKind) -> tuple[int, int]:
    """Return (min, max) char length expectations per Plans/04 §3.2."""
    if kind is GapKind.MISSING_TITLE:
        return (5, 200)
    if kind is GapKind.AUTHOR_BLOCK:
        return (10, 400)
    if kind is GapKind.CITATION_UNRESOLVED:
        # A single restored anchor (or the plain-text passthrough) — short.
        return (1, 300)
    if kind is GapKind.COPYRIGHT_BLOCK:
        # Plans/04 §3.2 band is 30-250 text chars; widened to absorb the
        # <aside class="copyright"><p>…</p></aside> wrapper overhead and
        # the longest real-corpus statements (PMC max ≈ 310 chars).
        return (30, 450)
    if kind is GapKind.LEGAL_DISCLAIMER:
        # License/terms paragraphs run long (CC license-p ≈ 1050 chars).
        return (40, 1150)
    return (1, 1000)


def _citation_fit(candidate_text: str, gap: GapSlot) -> float:
    """Per Plans/04 §3.2: 1.0 iff the chosen anchor target exists in the
    doc index (i.e. equals one of the gap's candidate_targets anchor ids).

    The plain-text passthrough (no ``<a>``) scores 0.5 — a legitimate,
    non-hallucinating outcome that's still preferred over an anchor
    pointing outside the candidate set (which scores 0.0).
    """
    m = _ANCHOR_RE.search(candidate_text)
    if m is None:
        # No anchor → plain-text passthrough. Acceptable; mid score.
        return 0.5
    if not _TAG_STRIP_RE.sub("", m.group(0)).strip():
        # Empty/whitespace-only anchor text → would ship a link with no
        # discernible name (axe link-name, WCAG 2.4.4). Score it BELOW the
        # plain-text passthrough so the passthrough wins instead.
        return 0.0
    chosen = m.group(1)
    valid_ids = {
        str(t.get("anchor_id")) for t in (gap.context or {}).get("candidate_targets", []) or []
    }
    return 1.0 if chosen in valid_ids else 0.0


def _kind_fit(candidate_text: str, kind: GapKind) -> float:
    if kind is GapKind.CITATION_UNRESOLVED:
        # Citation fit is scored against the gap context (anchor-exists),
        # not a fixed root tag — handled in _score_candidate.
        return 0.5
    if kind in (GapKind.COPYRIGHT_BLOCK, GapKind.LEGAL_DISCLAIMER):
        # Plans/04 §3.2: 1.0 iff root is <aside>/<footer> (and, for
        # copyright, the ©/copyright keyword survived in the TEXT —
        # the wrapper's class attribute doesn't count); lower otherwise.
        m = _ASIDE_FOOTER_RE.search(candidate_text)
        if m is None:
            return 0.0
        frag_text = _TAG_STRIP_RE.sub("", m.group(0))
        if kind is GapKind.COPYRIGHT_BLOCK and not COPYRIGHT_SUBFLAG_RE.search(frag_text):
            return 0.5
        return 1.0
    expected = _expected_root_tag(kind)
    if expected is None:
        return 0.5
    pattern = re.compile(rf"<\s*{expected}\b", re.IGNORECASE)
    return 1.0 if pattern.search(candidate_text) else 0.0


def _length_fit(candidate_text: str, kind: GapKind) -> float:
    lo, hi = _expected_length(kind)
    n = len(candidate_text)
    if lo <= n <= hi:
        return 1.0
    # Sigmoid decay outside band.
    if n < lo:
        return 1.0 / (1.0 + math.exp((lo - n) / max(1.0, lo / 2)))
    return 1.0 / (1.0 + math.exp((n - hi) / max(1.0, hi)))


def _diff_from_context(candidate_text: str, gap: GapSlot) -> float:
    """Small penalty if candidate has zero overlap with the gap context.

    Cheap word-level intersection against the most-relevant context
    string per gap kind. Returns 0.0 (no penalty) when context is empty.
    """
    ctx = ""
    if gap.kind is GapKind.MISSING_TITLE:
        ctx = (gap.context or {}).get("first_paragraph_text", "") or ""
    elif gap.kind is GapKind.AUTHOR_BLOCK:
        ctx = (gap.context or {}).get("raw_text", "") or ""
    elif gap.kind is GapKind.CITATION_UNRESOLVED:
        # The candidate must re-use the original match text verbatim; the
        # surrounding window is the relevant context.
        ctx = (gap.context or {}).get("surrounding_50_chars", "") or ""
    elif gap.kind in (GapKind.COPYRIGHT_BLOCK, GapKind.LEGAL_DISCLAIMER):
        # The candidate must carry the region text verbatim — zero
        # overlap means a hallucinated rights/legal claim.
        ctx = (gap.context or {}).get("raw_text", "") or ""
    if not ctx:
        return 0.0
    cand_words = set(re.findall(r"\w+", candidate_text.lower()))
    ctx_words = set(re.findall(r"\w+", ctx.lower()))
    if not cand_words or not ctx_words:
        return 0.1
    overlap = len(cand_words & ctx_words) / max(1, len(ctx_words))
    # No overlap → small penalty (0.2); strong overlap → no penalty.
    return max(0.0, 0.2 - overlap * 0.2)


def _score_candidate(candidate: Candidate, gap: GapSlot) -> float:
    pass_gate = 1.0  # already filtered upstream
    if gap.kind is GapKind.CITATION_UNRESOLVED:
        kind_fit = _citation_fit(candidate.text, gap)
    else:
        kind_fit = _kind_fit(candidate.text, gap.kind)
    length_fit = _length_fit(candidate.text, gap.kind)
    diff = _diff_from_context(candidate.text, gap)
    return pass_gate * (1.0 + kind_fit + length_fit) - diff


def _splice_missing_title(html: str, candidate_text: str) -> str:
    """Replace doc <title> with the candidate's <title>, and inject <h1>
    at the <!-- DART_TITLE_SLOT --> sentinel.

    If the candidate's ``<title>`` opening tag carries
    ``data-dart-fabricated="title"`` (i.e. came from the deterministic
    fallback in :func:`_fallback_for`), preserve that attribute on the
    spliced ``<title>`` so the marker survives into the final HTML.
    """
    new_title_m = _TITLE_RE.search(candidate_text)
    new_title_text = new_title_m.group(1).strip() if new_title_m else ""
    if new_title_text:
        # Re-escape minimal — mirror shell._escape behaviour for safety.
        safe_title = (
            new_title_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        # Detect fabrication marker on the candidate's <title> open tag.
        open_m = _TITLE_OPEN_RE.search(candidate_text)
        attrs = (open_m.group(1) or "").strip() if open_m else ""
        title_open = "<title>"
        # DART->semantik purge Stage 3: emit the semantik marker, but dual-READ
        # so a legacy dart-fabricated title (re-run over prior output) is honored.
        if (
            'data-semantik-fabricated="title"' in attrs
            or 'data-dart-fabricated="title"' in attrs
        ):
            title_open = '<title data-semantik-fabricated="title">'
        html = _TITLE_RE.sub(
            f"{title_open}{safe_title}</title>",
            html,
            count=1,
        )

    new_h1_m = _H1_RE.search(candidate_text)
    if new_h1_m:
        h1_html = _ensure_h1_id(new_h1_m.group(0), html)
        html = html.replace(
            "<!-- DART_TITLE_SLOT -->\n",
            f"{h1_html}\n",
            1,
        )
    return html


def _splice_author_block(
    html: str,
    candidate_text: str,
    original_region_html: str,
) -> str:
    """Replace the original prose region's HTML with the candidate's <address>."""
    addr_m = _ADDRESS_RE.search(candidate_text)
    if addr_m is None:
        return html
    address_html = addr_m.group(0)
    if original_region_html and original_region_html in html:
        html = html.replace(original_region_html, address_html, 1)
    return html


def _splice_citation_unresolved(
    html: str,
    candidate_text: str,
    gap: GapSlot,
) -> str:
    """In-place replace the matched plain-text span with the candidate <a>.

    Per Plans/04 §4.1: the splice replaces the FIRST occurrence of the
    matched reference text (e.g. "Section 4.2") with the gap-fill's
    chosen ``<a href="#...">match_text</a>``. The surrounding text is
    untouched.

    No-op cases (splice nothing — "leave as plain text", Plans/04 §3.4 +
    §1.4 do-NOT list):

      * candidate carries no ``<a href="#...">`` (the explicit plain-text
        confirmation), OR
      * the chosen anchor id is NOT one of the gap's candidate_targets
        (no-silent-fallbacks: never splice a hallucinated anchor), OR
      * the match_text span can't be located in the body HTML.
    """
    m = _ANCHOR_RE.search(candidate_text)
    if m is None:
        return html  # plain-text passthrough — leave unchanged.
    anchor_html = m.group(0)
    chosen = m.group(1)
    if not _TAG_STRIP_RE.sub("", anchor_html).strip():
        # Empty/whitespace-only anchor text — splicing it would both ship a
        # link with no discernible name (axe link-name, WCAG 2.4.4) AND
        # delete the visible reference text. Refuse; leave plain text
        # (no-silent-fallbacks: never trade content for a broken anchor).
        return html
    valid_ids = {
        str(t.get("anchor_id")) for t in (gap.context or {}).get("candidate_targets", []) or []
    }
    if chosen not in valid_ids:
        # Hallucinated anchor target — refuse to splice (leave plain text).
        return html
    match_text = (gap.context or {}).get("match_text") or ""
    if not match_text or match_text not in html:
        return html
    return html.replace(match_text, anchor_html, 1)


def _splice_legal_aside(
    html: str,
    candidate_text: str,
    gap: GapSlot,
    *,
    require_copyright_keyword: bool,
) -> str:
    """Shared core for the copyright/legal splice (Plans/04 §4.1).

    Replaces the FIRST occurrence of the original region's HTML (the
    ``region_html`` key 9a stashed in the gap context) with the
    candidate's ``<aside>``/``<footer>`` fragment.

    No-op cases (splice nothing; ``run_pass_9c`` then applies the
    deterministic ``fallback_html`` wrap instead):

      * the candidate carries no ``<aside>``/``<footer>`` root, OR
      * (copyright only) the fragment dropped every ©/copyright/rights
        keyword — content loss on a rights claim, refuse it, OR
      * the original region HTML can't be located in the body.
    """
    m = _ASIDE_FOOTER_RE.search(candidate_text)
    if m is None:
        return html
    frag = m.group(0)
    # Keyword check runs on the fragment's TEXT content — the wrapper's
    # own class="copyright" attribute must not satisfy it.
    frag_text = _TAG_STRIP_RE.sub("", frag)
    if require_copyright_keyword and not COPYRIGHT_SUBFLAG_RE.search(frag_text):
        return html
    region_html = (gap.context or {}).get("region_html") or ""
    if not region_html or region_html not in html:
        return html
    return html.replace(region_html, frag, 1)


def _splice_copyright_block(html: str, candidate_text: str, gap: GapSlot) -> str:
    """In-place replace the region with the candidate's © ``<aside>``."""
    return _splice_legal_aside(
        html,
        candidate_text,
        gap,
        require_copyright_keyword=True,
    )


def _splice_legal_disclaimer(html: str, candidate_text: str, gap: GapSlot) -> str:
    """In-place replace the region with the candidate's legal ``<aside>``."""
    return _splice_legal_aside(
        html,
        candidate_text,
        gap,
        require_copyright_keyword=False,
    )


def _fallback_for(gap: GapSlot) -> str:
    """Default fallback HTML for a zero-survivor gap (Plans/04 §3.4).

    For ``MISSING_TITLE`` we stamp ``data-dart-fabricated="title"`` on
    BOTH the ``<title>`` and ``<h1>`` so the splice (which uses regex
    on this exact string) carries the marker through into the final
    HTML — making the fabrication self-identifying per the
    no-silent-fallbacks policy.
    """
    if gap.fallback_html:
        return gap.fallback_html
    if gap.kind is GapKind.MISSING_TITLE:
        return (
            '<title data-semantik-fabricated="title">Untitled document</title>'
            '<h1 data-semantik-fabricated="title">Untitled document</h1>'
        )
    return ""


def _apply_fallback(html: str, gap: GapSlot) -> str:
    """Splice the gap's deterministic fallback into the HTML.

    ``missing_title``: splice the fabrication-marked default title.
    ``copyright_block`` / ``legal_disclaimer``: replace the original
    region HTML with the deterministic
    ``<aside class="copyright|legal">`` wrap 9a prepared (Plans/04
    §3.4) — real text, deterministic structure, recorded by the caller
    in ``gaps_fallback`` (no-silent-fallbacks).
    ``author_block`` fallback is "leave the region as-is" and
    ``citation_unresolved`` fallback is "leave the plain text
    unchanged" (Plans/04 §3.4), both of which are already the state of
    the HTML at this point (no-op).
    """
    if gap.kind is GapKind.MISSING_TITLE:
        fb = _fallback_for(gap)
        return _splice_missing_title(html, fb)
    if gap.kind in (GapKind.COPYRIGHT_BLOCK, GapKind.LEGAL_DISCLAIMER):
        fb = _fallback_for(gap)
        region_html = (gap.context or {}).get("region_html") or ""
        if fb and region_html and region_html in html:
            return html.replace(region_html, fb, 1)
        return html
    return html


def run_pass_9c(
    pre_doc: AssembledDoc,
    gaps_found: Sequence[GapSlot],
    candidates_per_gap: dict[int, list[Candidate]],
    regions: Sequence[Region],
    feature_blocks: Sequence[FeatureBlock],
    *,
    config: "AssemblerConfig | None" = None,
) -> AssembledDoc:
    """Score, splice, and return the updated AssembledDoc.

    * If a gap has at least one survivor: pick argmax candidate by
      :func:`_score_candidate`, splice into ``pre_doc.html``, and add
      the gap to ``gaps_resolved``.
    * If a gap has zero survivors: splice the deterministic fallback
      and add the gap to ``gaps_fallback``.
    """
    if config is None:
        from .api import AssemblerConfig

        config = AssemblerConfig()

    html = pre_doc.html
    gaps_resolved: list[GapSlot] = []
    gaps_fallback: list[GapSlot] = []
    splice_log: list[str] = []

    for gi, gap in enumerate(gaps_found):
        survivors = candidates_per_gap.get(gi, [])
        if survivors:
            scored = [(c, _score_candidate(c, gap)) for c in survivors]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            best = scored[0][0]
            if gap.kind is GapKind.MISSING_TITLE:
                html = _splice_missing_title(html, best.text)
            elif gap.kind is GapKind.AUTHOR_BLOCK:
                original_html = ""
                if 0 <= gap.region_index < len(pre_doc.region_provenance):
                    # Best-effort: use the candidate's fallback_html as the
                    # search target (set by 9a to the rendered region HTML).
                    original_html = gap.fallback_html or ""
                html = _splice_author_block(html, best.text, original_html)
            elif gap.kind is GapKind.CITATION_UNRESOLVED:
                before = html
                html = _splice_citation_unresolved(html, best.text, gap)
                if html == before:
                    # Splice was a no-op — the best candidate confirmed
                    # "leave as plain text" (or refused a bad anchor). Per
                    # Plans/04 §3.4 the gap is treated as a fallback (no
                    # markup change), not a resolution.
                    gaps_fallback.append(gap)
                    splice_log.append(f"{gap.kind.value}:plain_text")
                    continue
            elif gap.kind in (
                GapKind.COPYRIGHT_BLOCK,
                GapKind.LEGAL_DISCLAIMER,
            ):
                before = html
                if gap.kind is GapKind.COPYRIGHT_BLOCK:
                    html = _splice_copyright_block(html, best.text, gap)
                else:
                    html = _splice_legal_disclaimer(html, best.text, gap)
                if html == before:
                    # Splice refused the candidate (no <aside>/<footer>
                    # root, dropped © keyword, or region not located).
                    # Apply the honest deterministic aside wrap instead
                    # and record it as a fallback, never a resolution
                    # (no-silent-fallbacks).
                    html = _apply_fallback(html, gap)
                    gaps_fallback.append(gap)
                    splice_log.append(f"{gap.kind.value}:fallback")
                    continue
            # Token-level provenance for theta's hallucinated-structure
            # anchoring (Plan 12 A3): record exactly what was spliced.
            gap.resolved_html = best.text
            gaps_resolved.append(gap)
            splice_log.append(f"{gap.kind.value}:resolved")
        else:
            html = _apply_fallback(html, gap)
            gaps_fallback.append(gap)
            splice_log.append(f"{gap.kind.value}:fallback")

    new_log = dict(pre_doc.sub_task_log or {})
    new_log["splice"] = ", ".join(splice_log) if splice_log else "no-op"

    return AssembledDoc(
        html=html,
        gaps_found=list(gaps_found),
        gaps_resolved=gaps_resolved,
        gaps_fallback=gaps_fallback,
        heading_tree=list(pre_doc.heading_tree),
        landmarks=dict(pre_doc.landmarks),
        anchors=dict(pre_doc.anchors),
        region_provenance=list(pre_doc.region_provenance),
        sub_task_log=new_log,
    )


__all__ = ["run_pass_9c"]
