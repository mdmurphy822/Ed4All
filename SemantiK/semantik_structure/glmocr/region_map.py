"""GLM-OCR 25-class region → SemantiK region-kind classification.

Deterministic, data-driven, NO LLM. Maps a PP-DocLayoutV3 ``native_label``
(the 25-class layout taxonomy, id2label 0..24) onto the SemantiK
``region_kind`` vocabulary the downstream adapter renders
(``heading`` / ``paragraph`` / ``figure`` / ``table`` / ``math`` / ``list`` /
``metadata_drop``), plus the pedagogical / apparatus / furniture demotions
validated in ``scratchpad/glmocr/build_html.py``.

Publisher apparatus vocabulary is DATA (wide-net owner doctrine): the marker
lexicon is loaded from ``schemas/taxonomies/openstax_lexicon.json`` +
``exercise_apparatus_lexicon.json`` when reachable, with a frozen fallback so
the cross-venv SemantiK runtime never hard-depends on the Ed4All schema tree.
A new publisher/corpus is onboarded by adding a marker row to the lexicon,
never by editing this module.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── 25-class native_label buckets (PP-DocLayoutV3 id2label 0..24). ──────────
# Furniture the transform DROPS to ``metadata_drop`` (running header/footer,
# folio, decorative seal, header/footer images).
FURNITURE_LABELS = frozenset(
    {"header", "footer", "number", "page_number", "footer_image", "header_image", "seal"}
)
# Text-bearing prose labels routed through the paragraph/box-title/apparatus rules.
TEXTISH_LABELS = frozenset(
    {"text", "content", "abstract", "reference_content", "vertical_text", "algorithm"}
)
# Footnote-family labels kept as role-annotated paragraphs.
FOOTNOTE_LABELS = frozenset({"footnote", "vision_footnote"})
# The two labels the default SDK config ABANDONS that the lane explicitly KEEPS
# (``aside_text`` and ``reference`` carry instructional content — dropping them
# is silent content loss, not furniture removal).
KEPT_ASIDE_LABELS = frozenset({"aside_text", "reference"})
FORMULA_LABELS = frozenset({"display_formula", "inline_formula"})
FIGURE_LABELS = frozenset({"image", "chart"})

# ── Pedagogical-apparatus marker → css pedagogy class (page-arranger parity). ─
# Frozen fallback (openstax + generic textbook shapes) used when the schema
# lexicon files are unreachable in the SemantiK venv.
_FALLBACK_APPARATUS: Tuple[Tuple[str, str], ...] = (
    ("manipulative mathematics", "pedagogy-example"),
    ("be prepared", "pedagogy-be-prepared"),
    ("try it", "pedagogy-try-it"),
    ("self check", "pedagogy-try-it"),
    ("how to", "pedagogy-how-to"),
    ("everyday math", "pedagogy-example"),
    ("writing exercises", "pedagogy-try-it"),
    ("example", "pedagogy-example"),
    ("solution", "pedagogy-solution"),
    ("step", "pedagogy-step"),
)

_ROLE_TO_CSS: Dict[str, str] = {
    "guided_practice": "pedagogy-try-it",
    "worked_example": "pedagogy-how-to",
    "exercise_item": "pedagogy-try-it",
}

# ── D3: caption-label + end-matter-apparatus furniture constraints. ─────────
# A numbered caption/pedagogical label ("Figure 1.2" / "Example 1.3" /
# "Table 1.4" / "Try It 1.5") the layout model over-promoted to a heading. It
# must NEVER become a heading at section level ≤3 — Figure/Table route to the
# caption path, Example/Try It to the apparatus path.
_CAPTION_LABEL_RE = re.compile(
    r"^(figure|example|table|try\s+it)\s+\d+\.\d+", re.IGNORECASE
)

# Genuine end-matter apparatus section headings the layout model failed to
# recognize (KEY TERMS / KEY CONCEPTS / CHAPTER REVIEW / REVIEW EXERCISES /
# PRACTICE TEST). Frozen fallback; the live set is DATA-DRIVEN from the schema
# lexicon profiles' ``apparatus_sections`` display list when reachable.
_FALLBACK_ENDMATTER: Tuple[str, ...] = (
    "key terms",
    "key concepts",
    "chapter review",
    "review exercises",
    "practice test",
)
# Leading / trailing chapter-ordinal junk ("1.5 Key Terms" / "Key Terms 415")
# stripped conservatively before the whole-text marker match.
_CH_ORDINAL_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s+")
_CH_ORDINAL_SUFFIX_RE = re.compile(r"\s+\d+(?:\.\d+)?\s*$")


def _schema_dir() -> Optional[Path]:
    """Best-effort locate ``schemas/taxonomies`` walking up from this file.

    Returns ``None`` when the Ed4All schema tree is not reachable (a bare
    SemantiK checkout) — the caller then uses the frozen fallback.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "schemas" / "taxonomies"
        if cand.is_dir():
            return cand
    return None


@lru_cache(maxsize=1)
def _apparatus_markers() -> Tuple[Tuple[str, str], ...]:
    """(marker_casefold, pedagogy_css_class) pairs, longest-first.

    Data-driven from the schema lexicons when present; frozen fallback else.
    Longest-first so a specific marker ("be prepared") wins over a generic
    prefix, and the leading-text prefix match is unambiguous.
    """
    markers: Dict[str, str] = {}
    schema = _schema_dir()
    if schema is not None:
        ox = schema / "openstax_lexicon.json"
        try:
            data = json.loads(ox.read_text(encoding="utf-8"))
            for row in data.get("x-markers", []):
                marker = str(row.get("marker", "")).strip().lower()
                role = str(row.get("role", "")).strip()
                if marker:
                    markers[marker] = _ROLE_TO_CSS.get(role, "pedagogy-example")
        except (OSError, ValueError, TypeError):
            pass
    # Always fold the frozen fallback in (union), so Example/Solution/Step —
    # generic shapes not in the openstax publisher lexicon — are covered.
    for marker, css in _FALLBACK_APPARATUS:
        markers.setdefault(marker, css)
    return tuple(sorted(markers.items(), key=lambda kv: (-len(kv[0]), kv[0])))


# ── Deterministic text-shape predicates (ported from build_html.py). ────────
_SECTION_NUM_RE = re.compile(r"^\s*\d+\.\d+(?:\.\d+)*\s+\S")
_TABLE_CAPTION_RE = re.compile(r"^\s*table\s+\d", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.)\s+\d", re.IGNORECASE)
_MD_HEAD_RE = re.compile(r"^#{1,6}\s*")


def strip_md_heading(text: str) -> str:
    """Drop a leading markdown ``#`` run (GLM emits ATX headings for titles)."""
    return _MD_HEAD_RE.sub("", (text or "").strip()).strip()


def apparatus_pedagogy_class(text: str) -> Optional[str]:
    """Return the pedagogy css class if ``text`` OPENS with an apparatus marker.

    Case-insensitive leading-text prefix match against the loaded lexicon.
    Returns ``None`` for ordinary prose.
    """
    t = strip_md_heading(text).lstrip("*_ \t").lower()
    if not t:
        return None
    for marker, css in _apparatus_markers():
        if t.startswith(marker):
            # Guard: "stepping" must not match "step" — require a word boundary.
            tail = t[len(marker):]
            if not tail or not tail[0].isalpha():
                return css
    return None


def is_table_caption(text: str) -> bool:
    return bool(_TABLE_CAPTION_RE.match(strip_md_heading(text)))


def is_figure_caption_text(text: str) -> bool:
    return bool(_FIGURE_CAPTION_RE.match(strip_md_heading(text)))


def caption_label_kind(text: str) -> Optional[str]:
    """Classify a numbered caption/pedagogical LABEL (D3, item 1).

    Returns ``"caption"`` for ``Figure N.M`` / ``Table N.M`` (route to the
    caption paragraph path), ``"apparatus"`` for ``Example N.M`` / ``Try It
    N.M`` (route to the apparatus paragraph path), or ``None`` for anything
    else. Matched after ``strip_md_heading``, case-insensitive.
    """
    m = _CAPTION_LABEL_RE.match(strip_md_heading(text))
    if not m:
        return None
    head = re.sub(r"\s+", "", m.group(1).lower())  # "try it" → "tryit"
    return "caption" if head in {"figure", "table"} else "apparatus"


@lru_cache(maxsize=1)
def _endmatter_markers() -> Tuple[str, ...]:
    """End-matter apparatus SECTION-heading markers (casefold), unioned with the
    frozen fallback (so a bare SemantiK checkout still recognizes the canonical
    five).

    Data source is each lexicon profile's ``apparatus_sections`` display list —
    DELIBERATELY not the broader ``apparatus_whitelist``. The whitelist folds in
    worked-example BODY labels (``solution``/``self check``) and the scholarly /
    admonition overlay CALLOUT labels (``theorem``/``lemma``/``note``/
    ``warning``/…); promoting any of those BARE labels to a level-2 SECTION
    heading is exactly the over-segmentation the D3 furniture constraint fights.
    ``apparatus_sections`` is the genuine end-matter-section list (generic-
    academic → the canonical five, matching the fallback), so a new corpus
    onboards an end-matter section by adding an ``apparatus_sections`` row.
    """
    markers = set(_FALLBACK_ENDMATTER)
    schema = _schema_dir()
    if schema is not None:
        lex = schema / "semantik_lexicon.json"
        try:
            data = json.loads(lex.read_text(encoding="utf-8"))
            for prof in (data.get("profiles") or {}).values():
                for row in prof.get("apparatus_sections", []) or []:
                    disp = str((row or {}).get("display", "")).strip()
                    if disp:
                        markers.add(disp.lower())
        except (OSError, ValueError, TypeError):
            pass
    return tuple(sorted(markers))


def is_endmatter_apparatus(text: str) -> bool:
    """True when ``text`` is a BARE end-matter apparatus section heading (D3,
    item 2).

    Only a short (≤6-word) heading-shaped occurrence whose whole text IS the
    marker — optionally wrapped by a leading/trailing chapter ordinal — matches;
    a prose sentence merely mentioning "key terms" does not.
    """
    t = strip_md_heading(text).strip()
    if not t or len(t.split()) > 6:
        return False
    core = _CH_ORDINAL_PREFIX_RE.sub("", t)
    core = _CH_ORDINAL_SUFFIX_RE.sub("", core).strip()
    return core.lower() in _endmatter_markers()


def heading_level_for_title(text: str) -> Tuple[int, bool]:
    """Deterministic heading level for a ``paragraph_title`` region.

    Returns ``(level, ambiguous)``. An ``N.M`` numbered title is level 2
    (unambiguous — the section-number spine). Everything else defaults to
    level 3 but is flagged AMBIGUOUS so the Super reasoning pass can adjudicate
    (``data-heading-level="pending"``).
    """
    t = strip_md_heading(text)
    if _SECTION_NUM_RE.match(t):
        return 2, False
    return 3, True


def is_box_title(text: str, next_text: str, next_kind: str) -> bool:
    """A short title-case line preceding a prose body → promote to heading.

    Ported from ``build_html.is_box_title`` (defect 4): ≤6 words, title-case
    lead, no terminal punctuation, single line, followed by a real prose body
    region (≥4 words). Guards against pairing two titles.
    """
    t = (text or "").strip()
    if not t or "\n" in t:
        return False
    if not re.match(r"^[A-Z]", t):
        return False
    if t[-1] in ".:;,!?":
        return False
    if len(t.split()) > 6:
        return False
    if next_kind not in {"paragraph"}:
        return False
    if len((next_text or "").split()) < 4:
        return False
    return True


def classify_native_label(native_label: str) -> str:
    """Coarse ``native_label`` → base ``region_kind`` (before text-shape rules).

    The transform refines this with apparatus / box-title / caption / ordinal
    rules; this is the first-pass bucket.
    """
    nl = (native_label or "text").strip()
    if nl in FURNITURE_LABELS:
        return "metadata_drop"
    if nl == "doc_title":
        return "heading"
    if nl == "paragraph_title":
        return "heading"
    if nl in FORMULA_LABELS:
        return "math"
    if nl == "table":
        return "table"
    if nl in FIGURE_LABELS:
        return "figure"
    if nl == "figure_title":
        return "caption"
    if nl in FOOTNOTE_LABELS:
        return "footnote"
    if nl in KEPT_ASIDE_LABELS:
        return "aside"
    if nl == "formula_number":
        return "metadata_drop"
    # text / content / abstract / reference_content / vertical_text / algorithm
    return "paragraph"


__all__ = [
    "FURNITURE_LABELS",
    "TEXTISH_LABELS",
    "FOOTNOTE_LABELS",
    "KEPT_ASIDE_LABELS",
    "FORMULA_LABELS",
    "FIGURE_LABELS",
    "apparatus_pedagogy_class",
    "classify_native_label",
    "heading_level_for_title",
    "is_box_title",
    "is_table_caption",
    "is_figure_caption_text",
    "caption_label_kind",
    "is_endmatter_apparatus",
    "strip_md_heading",
]
