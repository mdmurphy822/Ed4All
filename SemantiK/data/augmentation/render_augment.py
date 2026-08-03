"""Realistic-rendering PRESENTATION augmenter for the structure-data build.

SemantiK's structure head trains on PDFs rendered from gold HTML via Playwright
— a clean, single-column browser layout. It collapses on real PDFs (in-dist
role-F1 ~0.87 -> projected-wild ~0.35). A deploy-gap audit found the dominant
failure mode is **2-column reading order**, plus ``(cid:N)`` glyph-encoding.

KEY INSIGHT: the HTML tags ARE the gold labels, and anything we RENDER keeps a
clean element->block alignment. So if we render the SAME structured HTML
*realistically* (2-column, ligature/CID-prone fonts, dense furniture-bearing
layout), we get deploy-LIKE training+eval data WITH FREE, correctly-aligned
labels and NO hand-labeling.

This module changes **PRESENTATION ONLY** — it injects a ``<style>`` block (and
nothing else; furniture like running headers/footers is realized via CSS
``::before``/``::after`` generated content, never new DOM tags). It NEVER alters
the HTML tag structure, so the downstream ``structural_role`` / ``is_heading`` /
``list`` / ``caption`` / ``table`` gold labels (which are read from the ORIGINAL
``output_html``, not the augmented render) stay correct by construction.

Gated behind ``SEMANTIK_RENDER_AUGMENT`` (default OFF). When off, every profile
resolves to ``clean`` and :func:`augment_html` is the identity -> byte-identical
to today's render. Deterministic + seeded throughout (seed by doc index): same
corpus in -> same augmented renders out.

Pure stdlib (``os`` + ``random`` + ``re``); no SemantiK internals, no model,
no network. License-clean: the only font it names beyond CSS generic families
is **DejaVu Serif** (a permissively-licensed, freely-redistributable font that
ships on the box), used to exercise ligature extraction.

Mirrors the parse-with-fallback posture of
:func:`semantik_structure.reading_order.resolve_column_order_mode`.
"""
from __future__ import annotations

import os
import random
import re

__all__ = [
    "RENDER_PROFILES",
    "resolve_render_augment_mode",
    "resolve_render_profile_for",
    "augment_html",
]

# ---------------------------------------------------------------------------
# Profile vocabulary
# ---------------------------------------------------------------------------
#
# A profile is a presentation recipe. ``clean`` is the control (no-op render,
# byte-identical to today). The other three target the audit's measured gaps;
# they compose with ``+`` (e.g. ``two_column+serif_embed``).
RENDER_PROFILES = (
    "clean",  # control: single column, no injected style (== today)
    "two_column",  # academic multi-column layout (the dominant real-PDF gap)
    "serif_embed",  # ligature-prone serif body font (CID / ligature extraction)
    "dense_layout",  # tight leading + running header/footer furniture bands
)

# A composite the resolver may hand to augment_html (and the mixed
# distribution may assign): two_column rendered in a ligature serif.
_COMPOSITE_PROFILES = ("two_column+serif_embed",)

# Modes the env / resolver understand. ``mixed`` is the domain-randomization
# distribution; a bare profile name pins that profile corpus-wide.
_KNOWN_MODES = frozenset(
    {"off", "mixed", *RENDER_PROFILES, *_COMPOSITE_PROFILES}
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Weighted domain-randomization distribution for ``mixed``. two_column is the
# highest-value profile (the audit's dominant gap), so it dominates; clean is
# kept as an in-distribution control anchor. Weights need not sum to 1 — they
# are normalized at draw time.
_MIXED_DISTRIBUTION: tuple[tuple[str, float], ...] = (
    ("clean", 0.15),
    ("two_column", 0.40),
    ("serif_embed", 0.15),
    ("dense_layout", 0.15),
    ("two_column+serif_embed", 0.15),
)


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def resolve_render_augment_mode() -> str:
    """Resolve the render-augmentation mode from ``SEMANTIK_RENDER_AUGMENT``.

    **Default OFF.** Resolution (parse-with-fallback, mirroring
    :func:`semantik_structure.reading_order.resolve_column_order_mode`):

    * unset / blank / falsey / garbage -> ``"off"`` (every doc renders
      ``clean`` -> byte-identical to today).
    * a truthy token (``1`` / ``true`` / ``yes`` / ``on``) -> ``"mixed"``
      (the weighted domain-randomization distribution).
    * an explicit known mode / profile name (``two_column`` /
      ``serif_embed`` / ``dense_layout`` / ``two_column+serif_embed`` /
      ``clean`` / ``mixed``) -> that mode, pinned corpus-wide.
    """
    raw = (os.environ.get("SEMANTIK_RENDER_AUGMENT") or "").strip().lower()
    if not raw:
        return "off"
    if raw in _TRUTHY:
        return "mixed"
    if raw in _KNOWN_MODES:
        return raw
    return "off"  # unknown / garbage -> off (parse-with-fallback)


def resolve_render_profile_for(doc_index: int, mode: str | None = None) -> tuple[str, int]:
    """Assign a ``(profile, seed)`` to one document for the given mode.

    Deterministic by ``doc_index`` — the same corpus position always draws the
    same profile + seed, so an augmented build is reproducible. The returned
    ``seed`` is consumed by :func:`augment_html` to pick minor presentation
    variation (margins / gutter / font-size), so two docs that draw the same
    profile still vary slightly (domain randomization).

    * ``mode is None`` -> resolved from the env via
      :func:`resolve_render_augment_mode`.
    * ``"off"`` -> always ``("clean", 0)`` (byte-identical render).
    * a specific profile / composite name -> that profile, pinned, with a
      doc-derived seed.
    * ``"mixed"`` -> a weighted deterministic draw from
      :data:`_MIXED_DISTRIBUTION`.
    """
    if mode is None:
        mode = resolve_render_augment_mode()
    try:
        di = int(doc_index)
    except (TypeError, ValueError):
        di = 0
    seed = di & 0x7FFFFFFF

    if mode == "off":
        return "clean", 0
    if mode in RENDER_PROFILES or mode in _COMPOSITE_PROFILES:
        return mode, seed
    if mode == "mixed":
        total = sum(w for _, w in _MIXED_DISTRIBUTION)
        # Deterministic draw seeded by the doc index — reproducible per corpus
        # position, independent of iteration order.
        draw = random.Random(f"render_profile:{di}").random() * total
        acc = 0.0
        for profile, w in _MIXED_DISTRIBUTION:
            acc += w
            if draw <= acc:
                return profile, seed
        return _MIXED_DISTRIBUTION[-1][0], seed
    # Unknown mode (defensive) -> clean.
    return "clean", 0


# ---------------------------------------------------------------------------
# CSS segment builders (deterministic, seeded minor variation)
# ---------------------------------------------------------------------------


def _two_column_css(rng: random.Random) -> str:
    """Academic two-column layout. Overrides the gold pair's single-column
    ``max-width`` / ``margin`` (``!important``) so the body genuinely lays out
    in two columns with a realistic gutter and justified text. The page title
    (``h1``) spans both columns (real arXiv-style banner); section headings
    (``h2``+) stay within a column so the body keeps a strong left-edge (x0)
    bimodality — exactly the reading-order signal the head must learn."""
    gutter = rng.choice([18, 24, 28, 34])  # px
    margin = rng.choice([0.5, 0.6, 0.7, 0.75])  # in
    fs = rng.choice([9.5, 10.0, 10.5, 11.0])  # pt
    return (
        "html, body { max-width: none !important; width: auto !important; }\n"
        "body {\n"
        f"  margin: {margin}in !important;\n"
        f"  column-count: 2 !important;\n"
        f"  column-gap: {gutter}px !important;\n"
        "  column-fill: balance !important;\n"
        "  text-align: justify !important;\n"
        f"  font-size: {fs}pt !important;\n"
        "  line-height: 1.35 !important;\n"
        "}\n"
        "/* Title banner spans both columns; section headings stay in-column. */\n"
        "h1 { column-span: all !important; }\n"
        "p, li, blockquote, dd, dt { break-inside: avoid-column; }\n"
    )


def _serif_embed_css(rng: random.Random) -> str:
    """Ligature-prone serif body font.

    HONEST LIMITATION: producing genuine ``(cid:N)`` glyph codes at extraction
    time is a property of the embedded font *file* (a missing / non-injective
    ToUnicode CMap), which CSS alone cannot force — the browser embeds a
    subsetted font with a valid ToUnicode in the common case. So this profile
    cannot *guarantee* CID-soup locally. What it DOES do honestly is switch the
    body to a real serif (DejaVu Serif — permissively licensed, on the box)
    with discretionary ligatures + kerning enabled, which exercises the
    ligature-extraction path (``fi`` / ``ffi`` / ``fl`` glyph clusters) that
    feeds the same downstream brittleness. Documented as a partial profile."""
    fs = rng.choice([10.5, 11.0, 11.5, 12.0])  # pt
    return (
        "body {\n"
        '  font-family: "DejaVu Serif", Georgia, "Times New Roman", serif !important;\n'
        f"  font-size: {fs}pt !important;\n"
        "  text-rendering: optimizeLegibility !important;\n"
        '  font-feature-settings: "liga" 1, "dlig" 1, "kern" 1 !important;\n'
        "  font-variant-ligatures: common-ligatures discretionary-ligatures !important;\n"
        "}\n"
    )


def _dense_layout_css(rng: random.Random) -> str:
    """Tight leading, varied margins, and running header/footer furniture.

    The header/footer bands are realized with ``body::before`` / ``body::after``
    fixed-position GENERATED CONTENT (no new DOM tag), so they repeat on every
    printed page and land in the top/bottom 5% furniture zones — giving the
    head extractable running-header/footer artifacts to learn (``top_5pct`` /
    ``bottom_5pct`` / ``is_artifact``). Pure CSS; the content tags are
    untouched. NB: CSS-only paged-media page COUNTERS do not render via the
    unmodified ``page.pdf()`` path (no ``displayHeaderFooter`` access), so the
    footer carries a static running band, not an incrementing page number —
    documented honestly."""
    top = rng.choice([0.4, 0.5, 0.6])  # in
    side = rng.choice([0.6, 0.75, 0.9])  # in
    lh = rng.choice([1.05, 1.1, 1.15])
    return (
        "body {\n"
        f"  margin: {top}in {side}in {top}in {side}in !important;\n"
        f"  line-height: {lh} !important;\n"
        "}\n"
        "p { margin: 0.3em 0 !important; }\n"
        "/* Running furniture bands (generated content, no DOM tag). */\n"
        "body::before {\n"
        '  content: "Preprint \\2014 working draft";\n'
        "  position: fixed; top: 0; left: 0; right: 0;\n"
        "  font-size: 8pt; color: #555; text-align: center;\n"
        "  padding: 2px 0; border-bottom: 0.5px solid #bbb;\n"
        "}\n"
        "body::after {\n"
        '  content: "Section draft \\2014 not for distribution";\n'
        "  position: fixed; bottom: 0; left: 0; right: 0;\n"
        "  font-size: 8pt; color: #555; text-align: center;\n"
        "  padding: 2px 0; border-top: 0.5px solid #bbb;\n"
        "}\n"
    )


_CSS_BUILDERS = {
    "two_column": _two_column_css,
    "serif_embed": _serif_embed_css,
    "dense_layout": _dense_layout_css,
}


def _build_css_for_profile(profile: str, seed: int) -> str:
    """Compose the CSS for a (possibly composite ``a+b``) profile. Each segment
    gets its OWN seeded RNG keyed by the segment name so composition order does
    not perturb a segment's draws (deterministic + reproducible)."""
    segments: list[str] = []
    for part in profile.split("+"):
        part = part.strip()
        builder = _CSS_BUILDERS.get(part)
        if builder is None:
            continue
        rng = random.Random(f"render_augment:{part}:{int(seed)}")
        segments.append(f"/* render-augment:{part} */\n{builder(rng)}")
    return "\n".join(segments)


# ---------------------------------------------------------------------------
# Public augmenter
# ---------------------------------------------------------------------------

_HEAD_CLOSE_RE = re.compile(r"</head\s*>", re.IGNORECASE)
_BODY_OPEN_RE = re.compile(r"<body\b", re.IGNORECASE)


def augment_html(html: str, profile: str, seed: int = 0) -> str:
    """Return ``html`` with a presentation ``<style>`` block injected to realize
    ``profile``. PRESENTATION ONLY — never restructures or removes any tag, so
    the gold tag-structure (and thus every downstream label) is preserved.

    * ``profile == "clean"`` (or unknown / empty) -> the input is returned
      UNCHANGED (byte-identical render to today — the control / off path).
    * any other profile -> a ``<style data-render-augment="...">`` block is
      appended at the END of ``<head>`` (so it cascades AFTER the gold pair's
      own ``<style>`` and wins on equal specificity / ``!important``). With no
      ``<head>``, it is inserted just before ``<body>``; with neither, it is
      prepended.

    ``seed`` selects deterministic minor variation (margins / gutter /
    font-size) so reruns are reproducible and same-profile docs still vary.
    """
    if not html or profile == "clean" or profile not in _KNOWN_MODES:
        return html
    if profile in ("off", "mixed"):
        # Not a renderable profile (a mode leaked in) — no-op rather than guess.
        return html

    css = _build_css_for_profile(profile, seed)
    if not css.strip():
        return html  # composite resolved to no known segment -> no-op

    style_block = (
        f'<style data-render-augment="{profile}" data-render-seed="{int(seed)}">\n'
        f"{css}</style>"
    )

    m = _HEAD_CLOSE_RE.search(html)
    if m:
        return html[: m.start()] + style_block + "\n" + html[m.start() :]
    m = _BODY_OPEN_RE.search(html)
    if m:
        return html[: m.start()] + style_block + "\n" + html[m.start() :]
    return style_block + "\n" + html
