"""Deterministic OCR heading-candidate classifier (adapter-seam, model-free).

Forensic audit (2026-07-03) of 8 OCR-converted textbook chapters
(``sample-algebra-2e-scan-chNN``) surfaced three heading-classification
defects the upstream council + ``deterministic_structure.clean_structure``
pass let through on scanned corpora:

  (a) RUN-IN "Solution" apparatus labels typed as headings, OCR-decorated with
      leading gutter glyphs — ``) Solution`` / ``™ Solution`` /
      ``“ Solution`` (90 across 8 chapters). As headings they mint spurious
      section boundaries in the heading-segmenting chunker (and duplicate the
      ``id="solution"`` HTML id).
  (b) TOP-of-page RUNNING HEADERS promoted to headings, in BOTH forms —
      leading page number (``188 Chapter 1 Foundations``,
      ``686 Chapter 6 Polynomials``) and a bare repeated ``Chapter N Title``
      (``Chapter 4 Graphs`` ×34 as phantom chapter titles). The repeat-count
      distinguishes a real chapter opener (the FIRST occurrence) from
      per-page furniture (all the rest).
  (c) SCANNER-WATERMARK tokens promoted to headings — ``Lexar 2`` /
      ``Lexar 117 rr`` / ``Ivo`` / ``Niw`` / ``AIS`` (short, high-OCR-garbage,
      non-dictionary-shaped).

These are pure text predicates over a heading candidate's text, applied at the
Ed4All adapter seam (:mod:`lib.semantik.adapter`) so BOTH the live-conversion
render path AND the ``scripts/ops/semantik_rerender.py`` re-render path (which
reconstructs heading blocks from already-emitted HTML/IR and never re-runs the
Region-level ``clean_structure`` pass) benefit without a cascade re-run. The
conversion path additionally mirrors (a)/(b) at the Region level in
``SemantiK/semantik_structure/qwen_specialists/deterministic_structure.py`` so a
FRESH conversion never mints the phantom chapters in the first place.

Conservative by design: a genuine section title (``9.3 Add and Subtract Square
Roots``), an apparatus heading (``Key Concepts`` / ``Practice Test``) and a
long real chapter word (``Polynomials`` / ``Foundations``) must survive; only
unambiguous OCR furniture / garbage is demoted.
"""

from __future__ import annotations

import re

# Wave #22 — the apparatus-heading vocabulary + short-heading whitelist are
# sourced from the profile-organized pedagogical lexicon
# (``schemas/taxonomies/semantik_lexicon.json``) via the canonical loader, so a
# new corpus is onboarded by a lexicon entry, not a code edit. Behavior-
# preserving: the default ``generic-academic+open-textbook`` profile reproduces the
# historical hardcoded names / whitelist exactly.
from lib.ontology.taxonomy import (
    get_lexicon_apparatus_names,
    get_lexicon_apparatus_whitelist,
    get_lexicon_interior_apparatus_names,
    get_lexicon_numbered_apparatus_names,
    strip_leading_ordinal,
)

# ---------------------------------------------------------------------------
# CSS class hint carried onto a demoted "Solution" run-in label (parity with
# ``deterministic_structure._PEDAGOGICAL_LABEL_CLASSES`` / the assembler's
# pedagogy-* styling), so the worked-example solution semantics survive the
# heading→paragraph demotion.
# ---------------------------------------------------------------------------
SOLUTION_CSS_CLASS = "pedagogy-solution"


# ---------------------------------------------------------------------------
# (a) Decorated run-in "Solution" label.
# ---------------------------------------------------------------------------
# Up to 3 leading OCR gutter glyphs (``)`` / ``™`` / ``“`` / a stray rule) then
# a STANDALONE "Solution" (anchored ``$``) — so "Find Solutions to a Linear
# Equation" (a real section heading, does not start with the label) and
# "Solution set of the inequality" (trailing words) never match.
_SOLUTION_RUNIN_RE = re.compile(
    r"^\s*[^A-Za-z0-9]{0,3}\s*Solution\s*$", re.IGNORECASE
)


def is_decorated_solution_label(text: str) -> bool:
    """Whether ``text`` is a run-in "Solution" apparatus label (defect a)."""
    return bool(_SOLUTION_RUNIN_RE.match(text or ""))


# ---------------------------------------------------------------------------
# (b) Running-header furniture.
# ---------------------------------------------------------------------------
# Trailing page number: "Chapter 9 Roots and Radicals 1039".
_RUNNING_HEADER_TRAILING_RE = re.compile(
    r"^\s*chapter\s+\d+\b.*\s\d{2,4}\s*$", re.IGNORECASE
)
# Leading page number: "188 Chapter 1 Foundations" / "686 Chapter 6 Polynomials".
_RUNNING_HEADER_LEADING_RE = re.compile(
    r"^\s*\d{2,4}\s+chapter\s+\d+\b", re.IGNORECASE
)
# Bare "Chapter N Title" shape (no page number) — a candidate for the
# repeat-count rule, NOT dropped on its own (the first occurrence is the real
# chapter opener).
_CHAPTER_SHAPE_RE = re.compile(r"^\s*chapter\s+\d+\b", re.IGNORECASE)


def is_running_header(text: str) -> bool:
    """Whether ``text`` is a page-numbered running header (defect b).

    Matches BOTH the leading-page ("188 Chapter 1 Foundations") and
    trailing-page ("Chapter 9 Roots and Radicals 1039") forms. A bare
    "Chapter N Title" (no page number) is NOT a running header here — a single
    such heading is a legitimate chapter opener; only its REPEATS are furniture
    (see :func:`repeated_running_header_indices`).
    """
    t = text or ""
    return bool(
        _RUNNING_HEADER_TRAILING_RE.match(t)
        or _RUNNING_HEADER_LEADING_RE.match(t)
    )


def normalize_heading_key(text: str) -> str:
    """Whitespace-collapsed, lowercased key for repeat-count comparison."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# The repeat threshold: a normalized bare "Chapter N Title" appearing MORE than
# this many times in one document's candidates is per-page furniture (all but
# the first occurrence). A real book never repeats an identical chapter opener
# 4+ times; the audit's running header recurred 34×.
_REPEAT_FURNITURE_THRESHOLD = 3


def repeated_running_header_indices(texts: list[str]) -> set[int]:
    """Indices of REPEATED bare "Chapter N Title" candidates that are furniture.

    Only bare chapter-shaped candidates participate. For each normalized text
    appearing ``> _REPEAT_FURNITURE_THRESHOLD`` times, every occurrence EXCEPT
    the first is marked furniture (the first is the real chapter opener). A
    candidate that already carries a page number is caught by
    :func:`is_running_header` regardless of count and need not repeat.
    """
    order: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        if not _CHAPTER_SHAPE_RE.match((t or "").strip()):
            continue
        order.setdefault(normalize_heading_key(t), []).append(i)
    furniture: set[int] = set()
    for idxs in order.values():
        if len(idxs) > _REPEAT_FURNITURE_THRESHOLD:
            furniture.update(idxs[1:])
    return furniture


# ---------------------------------------------------------------------------
# (c) Scanner-watermark garbage.
# ---------------------------------------------------------------------------
# A real "N.M" section-number token anywhere → a genuine section heading.
_SECTION_NUM_RE = re.compile(r"\b\d+\.\d+\b")

# Known apparatus / structural headings that are short but legitimate — never
# demoted even though they are <=3 tokens. Sourced from the active lexicon
# profile (default reproduces the historical whitelist).
_APPARATUS_WHITELIST = get_lexicon_apparatus_whitelist()

# Common short (4-6 char) English / textbook words that can legitimately be the
# SOLE token of a genuine short heading (a chapter title like "Graphs", a
# section word like "Roots"). A token of length >= 7 is accepted structurally
# without a lookup (OCR garbage is short), so this set only needs the 4-6 char
# real words worth protecting. Conservative: err toward INCLUSION so a real
# short heading is never demoted.
_COMMON_WORDS = frozenset(
    {
        # textbook chapter / section vocabulary (4-6 char)
        "graph", "graphs", "roots", "root", "ratio", "ratios", "slope",
        "slopes", "angle", "angles", "power", "powers", "prime", "whole",
        "cube", "cubes", "line", "lines", "area", "areas", "unit", "units",
        "term", "terms", "rule", "rules", "sign", "signs", "order", "solve",
        "value", "digit", "tenth", "point", "plane", "model", "table",
        "chart", "plot", "data", "mean", "mode", "range", "shape", "sum",
        "add", "money", "words", "steps", "notes", "facts", "index",
        # general common (4-6 char)
        "about", "above", "after", "again", "other", "which", "their",
        "there", "these", "those", "would", "could", "world", "water",
        "place", "right", "small", "large", "great", "first", "three",
        "under", "while", "being", "every", "group", "might", "still",
        "thing", "think", "using", "write", "years", "level", "learn",
        "study", "basic", "begin", "check", "class", "count", "equal",
        "final", "focus", "given", "guide", "ideas", "logic", "match",
        "parts", "proof", "ready", "review", "short", "topic",
        "total", "types", "usage", "works", "apply",
    }
)

_VOWELS = frozenset("aeiouyAEIOUY")


def _has_vowel(token: str) -> bool:
    return any(c in _VOWELS for c in token)


def _is_content_word(token: str) -> bool:
    """Whether ``token`` reads as a genuine content word (not OCR garbage).

    A content word is alphabetic, carries a vowel, and is EITHER long
    (``len >= 7`` — OCR watermark tokens are short) OR a known common short
    word. So ``Polynomials`` / ``Foundations`` pass on length, ``Roots`` /
    ``Graphs`` on the common-word set, while ``Lexar`` / ``Nolo`` / ``Niw`` /
    ``AIS`` (short, uncommon) do not.
    """
    core = token.strip("().,;:[]{}“”’\"'`*|-–—")
    if not core.isalpha() or len(core) < 4 or not _has_vowel(core):
        return False
    return len(core) >= 7 or core.lower() in _COMMON_WORDS


def is_watermark_garbage_heading(text: str) -> bool:
    """Whether ``text`` is a scanner-watermark / OCR-garbage heading (defect c).

    Demote iff ALL hold: <= 3 tokens, no "N.M" section number, not a
    "Chapter N" opener, not a whitelisted apparatus heading, and NO token is a
    content word. Conservative: any single genuine content word (or a section
    number, or the whitelist) keeps the heading.
    """
    t = (text or "").strip()
    if not t:
        return False
    if normalize_heading_key(t) in _APPARATUS_WHITELIST:
        return False
    if _SECTION_NUM_RE.search(t):
        return False
    if _CHAPTER_SHAPE_RE.match(t):
        # A "Chapter N …" opener is handled by the running-header / repeat-count
        # rules, never by the watermark gate.
        return False
    tokens = t.split()
    if not tokens or len(tokens) > 3:
        return False
    return not any(_is_content_word(tok) for tok in tokens)


# ---------------------------------------------------------------------------
# (d) Defect 4 — end-of-chapter apparatus headings mis-typed as body <p>.
# ---------------------------------------------------------------------------
# The generic academic-textbook end-of-chapter apparatus set (a NAMED constant,
# not scattered literals). These are real section headings that the council /
# scan pipeline sometimes demotes to a paragraph ("PRACTICE TEST" emitted as
# <p>PRACTICE TEST</p>) or files under a stray heading. When one appears as a
# STANDALONE short line it is promoted back to a heading.
APPARATUS_HEADING_NAMES: tuple[str, ...] = get_lexicon_apparatus_names()
# Standalone match: up to 3 leading OCR gutter glyphs, then exactly one of the
# apparatus names, then end (all-caps "PRACTICE TEST" or title-case both pass —
# the compare is case-insensitive). A "Practice Test your knowledge" sentence
# (trailing words) never matches (anchored ``$``).
_APPARATUS_HEADING_RE = re.compile(
    r"^\s*[^A-Za-z0-9]{0,3}\s*(?:"
    + "|".join(re.escape(name).replace(r"\ ", r"\s+") for name in APPARATUS_HEADING_NAMES)
    + r")\s*$",
    re.IGNORECASE,
)


def is_apparatus_heading(text: str) -> bool:
    """Whether ``text`` is a STANDALONE end-of-chapter apparatus heading (d).

    Matches one of :data:`APPARATUS_HEADING_NAMES` as the whole line (bar up to
    3 leading OCR gutter glyphs). Used to PROMOTE a mis-typed
    ``<p>PRACTICE TEST</p>`` back to a heading.
    """
    return bool(_APPARATUS_HEADING_RE.match(text or ""))


# ---------------------------------------------------------------------------
# (e) Numbered per-section apparatus BANNER — demote.
# ---------------------------------------------------------------------------
# A printed per-section drill banner carries a leading section ordinal
# ("1.4 Exercises", "10.3 Review Exercises", "9.2 Practice Test"). The
# standalone :func:`is_apparatus_heading` match is anchored on the bare name
# (leading chars may only be OCR gutter glyphs, not a numeric ordinal), so the
# numbered form slips through as a REAL section heading and mints a spurious
# per-section boundary in the heading-segmenting chunker. When the ordinal is
# stripped and the remainder is exactly an apparatus display name, the heading
# is a numbered apparatus banner and is DEMOTED to prose. Vocabulary is the
# same data-driven lexicon set (``get_lexicon_apparatus_names``) — no code
# vocabulary. Conservative: a real numbered section title ("1.3 Add and
# Subtract Integers") strips to a non-apparatus name and is never demoted.
_APPARATUS_NAME_LOWER: frozenset[str] = frozenset(
    n.strip().lower() for n in APPARATUS_HEADING_NAMES
)
# BARE apparatus words that are apparatus ONLY when ordinal-prefixed — a printed
# per-section drill banner "N.M Exercises" is furniture, but a STANDALONE
# "Exercises" is a real end-of-chapter section heading. The lexicon apparatus
# display names resolve to ['Chapter Review', 'Key Concepts', 'Key Terms',
# 'Practice Test', 'Review Exercises'] — no bare "Exercises" — so the numbered
# banner slipped through as a real heading (66 "N.M Exercises" leaked into 62
# chunk section_headings on the real corpus). This SEPARATE data-driven key
# (schemas/taxonomies/semantik_lexicon.json::numbered_apparatus_names) is
# consumed ONLY here (the numbered variant), NEVER by is_apparatus_heading — so
# the standalone/leading-split behavior of bare "Exercises" is unchanged.
_NUMBERED_ONLY_APPARATUS_LOWER: frozenset[str] = frozenset(
    n.strip().lower() for n in get_lexicon_numbered_apparatus_names()
)
# The union the numbered-banner match tests the ordinal-stripped remainder
# against: every standalone apparatus display name PLUS the numbered-only words.
_NUMBERED_APPARATUS_NAME_LOWER: frozenset[str] = (
    _APPARATUS_NAME_LOWER | _NUMBERED_ONLY_APPARATUS_LOWER
)


def is_numbered_apparatus_heading(text: str) -> bool:
    """Whether ``text`` is an ordinal-prefixed apparatus banner (defect e).

    True iff ``text`` begins with an ``N`` / ``N.M`` / ``N.M.K`` ordinal AND
    the ordinal-stripped remainder is either:

    * exactly one of :data:`APPARATUS_HEADING_NAMES` OR a numbered-only
      apparatus word (:data:`_NUMBERED_ONLY_APPARATUS_LOWER` — e.g. bare
      ``"Exercises"``), matched case-insensitively; OR
    * COMPOSITIONAL — it STARTS WITH a numbered-only apparatus word
      (word-boundary) whose residual tail is itself an apparatus name/phrase
      from the lexicon union. This catches the OCR-fused banner
      ``"2.7 EXERCISES Practice Makes Perfect"`` (OCR welded the section-exercise
      banner onto its "Practice Makes Perfect" subhead), whose stripped
      remainder ``"EXERCISES Practice Makes Perfect"`` is not an EXACT lexicon
      entry. All vocabulary stays in ``schemas/taxonomies/semantik_lexicon.json``
      (the numbered-only + apparatus display names) — only the composition shape
      is code.

    A heading with no leading ordinal (a bare ``"Exercises"``), or whose stripped
    remainder is not an apparatus name and whose apparatus-word prefix has a
    NON-apparatus tail (``"3.4 Exercises in Measure Theory"`` — a real titled
    section), returns ``False`` — so ``is_apparatus_heading('Exercises')`` is
    unaffected and genuine "Exercises in <topic>" titles are never demoted.
    """
    t = (text or "").strip()
    if not t:
        return False
    stripped = strip_leading_ordinal(t).strip()
    if not stripped or stripped == t:
        return False  # no leading ordinal -> not the numbered-banner case
    low = stripped.lower()
    # Exact ordinal-stripped remainder is an apparatus display / numbered word.
    if low in _NUMBERED_APPARATUS_NAME_LOWER:
        return True
    # Compositional: the remainder starts with a numbered-only apparatus word
    # ("Exercises") and its residual tail is itself apparatus vocabulary
    # ("Practice Makes Perfect") from the lexicon union. A tail that is NOT
    # apparatus vocabulary (a real titled section) does not demote.
    for name in _NUMBERED_ONLY_APPARATUS_LOWER:
        prefix = name + " "
        if low.startswith(prefix):
            tail = low[len(prefix):].strip()
            if not tail or tail in _NUMBERED_APPARATUS_NAME_LOWER:
                return True
    return False


# ---------------------------------------------------------------------------
# (f) Leading-apparatus heading FUSED into the start of a paragraph.
# ---------------------------------------------------------------------------
# The scan pipeline sometimes fuses a section heading into the first line of the
# paragraph that follows it — e.g. "Key Terms index $\sqrt[n]{a}$ … is called the
# index of the radical." — so the section matcher reports the "Key Terms" section
# missing even though its content is present. When a paragraph STARTS WITH an
# apparatus name (in a heading-like Title-Case / ALL-CAPS form) followed by
# non-trivial prose, the adapter SPLITS it into an apparatus heading + a remainder
# paragraph. Distinct from :func:`is_apparatus_heading` (the STANDALONE case, no
# trailing prose), which promotes the whole block.
#
# The name matches at the VERY START (anchored ``^``), captured with its actual
# casing so :func:`_is_heading_like_cased` can reject a lowercase mid-sentence
# "key terms are listed …". A single space in a name is matched as ``\s+``.
_LEADING_APPARATUS_RE = re.compile(
    r"^\s*("
    + "|".join(
        r"\s+".join(re.escape(w) for w in name.split())
        for name in APPARATUS_HEADING_NAMES
    )
    + r")\s+(\S.*)$",
    re.IGNORECASE | re.DOTALL,
)

# The remainder must carry at least this many word tokens to count as "real"
# trailing prose; fewer and it is the STANDALONE apparatus case (promoted whole
# by :func:`is_apparatus_heading`), not a fused heading to split off.
_LEADING_APPARATUS_MIN_REMAINDER_WORDS = 3


def _is_heading_like_cased(text: str) -> bool:
    """Whether ``text`` is cased like a heading: ALL CAPS or Title Case.

    Guards the leading-apparatus split against a lowercase mid-sentence usage
    ("the key terms are listed below"): "Key Terms" / "KEY TERMS" pass; "key
    terms" does not.
    """
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not words:
        return False
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True  # ALL CAPS
    for w in words:
        first_alpha = next((c for c in w if c.isalpha()), "")
        if not first_alpha or not first_alpha.isupper():
            return False
    return True  # Title Case (every word starts uppercase)


def split_leading_apparatus_heading(text: str) -> tuple[str, str] | None:
    """Split a paragraph that STARTS WITH an apparatus name (defect f).

    Returns ``(apparatus_name, remainder_prose)`` when ``text`` begins with one
    of :data:`APPARATUS_HEADING_NAMES` — in a heading-like Title-Case / ALL-CAPS
    form — followed by ``>= _LEADING_APPARATUS_MIN_REMAINDER_WORDS`` words of
    trailing prose. Returns ``None`` otherwise (no leading apparatus name, wrong
    casing, or a trivial/absent remainder = the standalone case handled by
    :func:`is_apparatus_heading`). The returned name carries its verbatim casing.
    """
    m = _LEADING_APPARATUS_RE.match(text or "")
    if not m:
        return None
    name, remainder = m.group(1), m.group(2).strip()
    if not _is_heading_like_cased(name):
        return None
    if len(remainder.split()) < _LEADING_APPARATUS_MIN_REMAINDER_WORDS:
        return None
    return name, remainder


# ---------------------------------------------------------------------------
# (g) RULE A — interior ALL-CAPS apparatus token fused MID-block.
# ---------------------------------------------------------------------------
# rerender_v5 corpus interrogation (2026-07-04): 4 of the 5 residual apparatus
# misses are INTERIOR fusions — the page-top apparatus banner swallowed into
# the middle of a paragraph ("…whole numbers are the numbers 0, 1, 2, 3, …
# KEY CONCEPTS 1.1 Introduction to Whole Numbers", "…$\\frac{10}{7}$ PRACTICE
# TEST Write as a whole number…"). Only the exact ALL-CAPS token splits (a
# prose mention "the key concepts of this chapter" is lowercase and never
# matches); the token must be followed by >= 3 words; first occurrence per
# block only. A LEADING token stays owned by the leading-split rule (f).
_INTERIOR_APPARATUS_NAMES: tuple[str, ...] = (
    get_lexicon_interior_apparatus_names()
)
# Case-SENSITIVE (no re.IGNORECASE) — ALL-CAPS exactly, per the guard above.
_INTERIOR_APPARATUS_RE = re.compile(
    r"\b("
    + "|".join(n.replace(" ", r"\s+") for n in _INTERIOR_APPARATUS_NAMES)
    + r")\b"
)
_INTERIOR_APPARATUS_MIN_REMAINDER_WORDS = 3


def split_interior_apparatus_heading(
    text: str,
) -> tuple[str, str, str] | None:
    """Split a block whose text fused an INTERIOR apparatus banner (Rule A).

    Returns ``(preceding_text, apparatus_name, remainder)`` when ``text``
    contains an exact ALL-CAPS member of :data:`_INTERIOR_APPARATUS_NAMES`
    with non-empty text before it and >= 3 words after it. Returns ``None``
    otherwise — including the LEADING-token case (empty preceding text), which
    :func:`split_leading_apparatus_heading` owns. First occurrence only.
    """
    t = text or ""
    m = _INTERIOR_APPARATUS_RE.search(t)
    if not m:
        return None
    pre = t[: m.start()].strip()
    if not pre:
        return None  # leading case → rule (f)
    remainder = t[m.end():].strip()
    if len(remainder.split()) < _INTERIOR_APPARATUS_MIN_REMAINDER_WORDS:
        return None
    return pre, m.group(1), remainder


# ---------------------------------------------------------------------------
# (h) RULE B — standalone "Introduction" paragraph typed as a heading.
# ---------------------------------------------------------------------------
# ch04's page-1 opener emitted "<p>Introduction</p>" while other chapters got a
# natural h2/h3 "Introduction". The exact-match STANDALONE arm promotes it via
# the same apparatus-promotion path. Deliberately NOT added to
# :data:`APPARATUS_HEADING_NAMES` — the leading/interior SPLIT arms must never
# fire on "Introduction to Whole Numbers …" leading prose. Title-case exact
# ("Introduction", optional OCR gutter glyphs) only.
_STANDALONE_INTRODUCTION_RE = re.compile(
    r"^\s*[^A-Za-z0-9]{0,3}\s*Introduction\s*$"
)


def is_standalone_apparatus_heading(text: str) -> bool:
    """Standalone-promotion predicate: apparatus names PLUS exact "Introduction".

    Superset of :func:`is_apparatus_heading` used ONLY by the standalone
    paragraph→heading promotion arm (Rule B) — never by the leading/interior
    split arms.
    """
    return bool(
        is_apparatus_heading(text)
        or _STANDALONE_INTRODUCTION_RE.match(text or "")
    )


# ---------------------------------------------------------------------------
# (i) RULE C — phantom running-header headings (folio / chapter-title shapes).
# ---------------------------------------------------------------------------
# rerender_v5 heading pollution: running headers survived as REAL h2/h3s in
# shapes the existing predicates missed —
#   * "8 Chapter 1 Foundations"  (1-digit folio; `is_running_header` needs 2-4)
#   * "Chapter 4 Graphs" ×2      (bare chapter shape, below the ×3 repeat rule)
#   * "130 The Real Numbers"     (folio + a REAL section title — the title is
#     genuine, only the folio is furniture → STRIP, don't demote)
#   * "Chapter 6 Polynomials Solution" (running header fused ONTO a real
#     "Solution" label → strip the chapter-title prefix, keep the label)
# The bare-chapter and prefix arms are gated on the KNOWN chapter title (the
# render-time --title/--title-map knowledge) so a legitimate "Chapter N …"
# section heading is never demoted by shape alone.

# Folio-prefixed chapter running header: ANY-length folio + "Chapter N" + a
# short tail (a real heading never starts with a stray page number before
# "Chapter N").
_FOLIO_CHAPTER_HEADER_RE = re.compile(
    r"^\s*\d{1,4}\s+chapter\s+\d+\b.{0,40}$", re.IGNORECASE
)
# Bare "Chapter N <tail>" (tail bounded to a title-ish 40 chars).
_BARE_CHAPTER_HEADER_RE = re.compile(
    r"^\s*chapter\s+\d+\s+(.{1,40})$", re.IGNORECASE
)
# Folio-prefixed section heading: 2-4 digit folio, then an uppercase-led
# remainder ("130 The Real Numbers"). A numbered section "1.2 Use the …" has a
# dot after the first digit(s) and never matches.
_FOLIO_SECTION_RE = re.compile(r"^\s*\d{2,4}\s+([A-Z].*)$")
_FOLIO_STRIP_MIN_WORDS = 2
_FOLIO_STRIP_MAX_WORDS = 8


def _normalize_title_key(title: str) -> str:
    """Chapter-title compare key: optional "Chapter N " prefix dropped."""
    t = re.sub(r"^\s*chapter\s+\d+\s+", "", title or "", flags=re.IGNORECASE)
    return normalize_heading_key(t)


def is_chapter_running_header(
    text: str, chapter_title: str | None = None
) -> bool:
    """Whether ``text`` is a running-header-shaped chapter heading (Rule C).

    True for a folio-prefixed "D Chapter N …" (any folio length — always
    furniture), or a bare "Chapter N <tail>" whose tail EQUALS the known
    ``chapter_title`` (case/whitespace-insensitive). Without title knowledge
    the bare shape is never demoted (the first "Chapter N Title" may be the
    real chapter opener).
    """
    t = (text or "").strip()
    if _FOLIO_CHAPTER_HEADER_RE.match(t):
        return True
    if chapter_title:
        m = _BARE_CHAPTER_HEADER_RE.match(t)
        if m and normalize_heading_key(m.group(1)) == _normalize_title_key(
            chapter_title
        ):
            return True
    return False


def strip_folio_prefix(text: str) -> str | None:
    """Strip a leading folio from a REAL section heading (Rule C strip arm).

    ``"130 The Real Numbers"`` → ``"The Real Numbers"``. Returns the stripped
    remainder when ``text`` is a 2-4 digit folio + a plausible title (2-8
    words, uppercase-led, no sentence terminal, not itself chapter-shaped —
    the chapter shape belongs to the demote arm), else ``None``.
    """
    m = _FOLIO_SECTION_RE.match(text or "")
    if not m:
        return None
    remainder = m.group(1).strip()
    if _CHAPTER_SHAPE_RE.match(remainder):
        return None  # "188 Chapter 1 …" → the running-header DEMOTE arm
    words = remainder.split()
    if not (_FOLIO_STRIP_MIN_WORDS <= len(words) <= _FOLIO_STRIP_MAX_WORDS):
        return None
    if remainder[-1] in ".!?":
        return None
    return remainder


# ---------------------------------------------------------------------------
# (j) Defect 2 — mid-body folio (printed page-number) leakage.
# ---------------------------------------------------------------------------
# The OCR scan leaks standalone printed folios into BODY blocks: a whole
# paragraph that is just "233" (a page number that landed between two content
# blocks), or a running-header + folio fused onto the TAIL of a body paragraph
# ("… = 2(7m - 11)$ Chapter 2 Solving Linear Equations and Inequalities 251").
# ``strip_folio_prefix`` (Rule C) only covered a folio at the START of a
# HEADING; these two body shapes were untouched.
#
# CONSERVATIVE by design: the standalone arm fires ONLY on a whole-block bare
# 2-4 digit number (never an exercise number followed by content). The trailing
# arm fires ONLY on a Title-Case running-header phrase ("Chapter N <Title …>
# <folio>") at the very end of a block — a shape a real math answer / exercise
# number NEVER takes. A bare trailing folio with no running-header anchor is
# deliberately NOT stripped: the physical-PDF ``pages`` metadata cannot be
# mapped to the printed folio (physical != printed, per §3.5) so a plausibility
# check is not computable, and a naive "digit after math" strip corrupts real
# answers ("… is 107.", "… \\quad 27").
_STANDALONE_FOLIO_RE = re.compile(r"^\s*\d{2,4}\s*$")

# A trailing running-header + folio: whitespace, "Chapter N", then a Title-Case
# first word (defeats a lowercase mid-sentence "Chapter 2 provides …"), 0-7 more
# title words (letters / apostrophe / hyphen only — no digits or sentence
# punctuation), then the 2-4 digit folio, an optional trailing period, and any
# closing HTML tags (so it strips cleanly from an html ``<p>…</p>`` body too).
_TRAILING_RUNNING_HEADER_RE = re.compile(
    r"\s+chapter\s+\d+\s+[A-Z][A-Za-z'\-]*(?:\s+[A-Za-z'\-]+){0,7}\s+\d{2,4}"
    r"\s*\.?(?P<tail>(?:\s*</[a-zA-Z0-9]+>)*\s*)$",
    re.IGNORECASE,
)


def is_standalone_folio(text: str) -> bool:
    """Whether ``text`` is a whole-block bare printed folio (Defect 2, arm a).

    True iff the ENTIRE text is a bare 2-4 digit number (a page number that
    leaked into a standalone body block). An exercise number followed by content
    ("243. Simplify …") is NOT a bare number and never matches.
    """
    return bool(_STANDALONE_FOLIO_RE.match(text or ""))


def strip_trailing_running_header(text: str) -> str | None:
    """Strip a trailing running-header + folio fused into a body block (arm b).

    ``"… = 2(7m - 11)$ Chapter 2 Solving Linear Equations and Inequalities 251"``
    -> ``"… = 2(7m - 11)$"``. Returns the stripped text (preserving any closing
    HTML tags) when a Title-Case ``"Chapter N <Title …> <folio>"`` running header
    ends the block, else ``None``. Conservative: the Title-Case first word + the
    letters-only title tail + the 2-4 digit folio make this a running-header
    shape a real answer / cross-reference never takes.
    """
    m = _TRAILING_RUNNING_HEADER_RE.search(text or "")
    if not m:
        return None
    return (text[: m.start()].rstrip() + m.group("tail")) or None


def strip_chapter_title_prefix(
    text: str, chapter_title: str | None
) -> str | None:
    """Strip a fused "Chapter N <chapter_title> " prefix (Rule C prefix arm).

    ``("Chapter 6 Polynomials Solution", "Polynomials")`` → ``"Solution"`` —
    the running header fused onto a real label; the remainder re-enters the
    normal heading predicates (so a bare "Solution" still takes the decorated-
    solution demotion). Conservative: requires the KNOWN chapter title to
    follow "Chapter N" verbatim (case-insensitive) with a non-empty remainder;
    returns ``None`` otherwise.
    """
    if not chapter_title:
        return None
    key = _normalize_title_key(chapter_title)
    if not key:
        return None
    m = _BARE_CHAPTER_HEADER_RE.match((text or "").strip())
    if not m:
        return None
    tail = m.group(1).strip()
    tail_key = normalize_heading_key(tail)
    if not tail_key.startswith(key + " "):
        return None
    remainder = " ".join(tail.split()[len(key.split()):]).strip()
    return remainder or None


# ---------------------------------------------------------------------------
# (e) Defect 3 — obviously-FUSED heading text (not a real heading).
# ---------------------------------------------------------------------------
# A heading is "obviously fused" when its text swallowed body content: a
# sentence-length prose run or several inline math runs (the page-top mega-
# heading the audit found: a footer + two exercise items + a word problem fused
# into one <h2>). Such text must NOT be kept as a heading — it is demoted to
# prose. Conservative thresholds so a real heading ("9.3 Add and Subtract
# Square Roots", "Chapter 9 Roots and Radicals") never trips.
_FUSED_HEADING_MIN_PROSE_WORDS = 12
_INLINE_MATH_RUN_RE = re.compile(r"\$[^$]+\$")
_SENTENCE_TERMINAL_RE = re.compile(r"[.!?]")


# ---------------------------------------------------------------------------
# (k) Wave #22 — a heading that is WHOLLY a ``\textbf{…}`` / ``\textit{…}``
# inline-emphasis label mis-typed as a section heading.
# ---------------------------------------------------------------------------
# The scan types a bold in-prose DEFINITION label (``\textbf{Square of a
# Number}``, ``\textbf{Square Root Notation}``) as an ``<h3>``, polluting the
# heading stream + TOC with entries that are really emphasis callouts, not
# sections. A genuine section title is NEVER wholly wrapped in ``\textbf{}`` /
# ``\textit{}`` (the cascade emits section headings as plain text), so a heading
# whose ENTIRE text (bar up to 3 leading OCR gutter glyphs) is one such wrapper
# is unambiguously a mis-typed label -> demoted to a bold paragraph.
_EMPHASIS_LABEL_HEADING_RE = re.compile(
    r"^\s*[^A-Za-z0-9\\]{0,3}\s*\\text(?:bf|it)\s*\{[^{}]+\}\s*$"
)


def is_emphasis_label_heading(text: str) -> bool:
    r"""Whether ``text`` is wholly a ``\textbf{…}`` / ``\textit{…}`` label (k).

    Used to DEMOTE a mis-typed inline-emphasis label out of the heading stream
    (a real section title is never a lone ``\textbf{}`` wrapper).
    """
    return bool(_EMPHASIS_LABEL_HEADING_RE.match(text or ""))


def is_fused_heading(text: str) -> bool:
    """Whether ``text`` is an obviously-fused blob mis-typed as a heading (e).

    Fires when the heading text carries EITHER (a) >= 2 inline ``$…$`` math
    runs, OR (b) a sentence-length prose run — >= ``_FUSED_HEADING_MIN_PROSE_WORDS``
    whitespace tokens AND a sentence-terminal ``.``/``!``/``?``. A genuine short
    numbered / title-case heading has neither.
    """
    t = (text or "").strip()
    if not t:
        return False
    if len(_INLINE_MATH_RUN_RE.findall(t)) >= 2:
        return True
    if (
        len(t.split()) >= _FUSED_HEADING_MIN_PROSE_WORDS
        and _SENTENCE_TERMINAL_RE.search(t)
    ):
        return True
    return False


__all__ = [
    "SOLUTION_CSS_CLASS",
    "APPARATUS_HEADING_NAMES",
    "is_apparatus_heading",
    "is_chapter_running_header",
    "is_decorated_solution_label",
    "is_emphasis_label_heading",
    "is_fused_heading",
    "is_numbered_apparatus_heading",
    "is_running_header",
    "is_standalone_apparatus_heading",
    "is_standalone_folio",
    "is_watermark_garbage_heading",
    "normalize_heading_key",
    "repeated_running_header_indices",
    "split_interior_apparatus_heading",
    "split_leading_apparatus_heading",
    "strip_chapter_title_prefix",
    "strip_folio_prefix",
    "strip_trailing_running_header",
]
