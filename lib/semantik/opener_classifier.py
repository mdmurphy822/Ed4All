r"""Deterministic pedagogical-opener classifier (A7, adapter-seam, model-free).

End-user-HTML audit (2026-07-04, ``plans/enduser-html-improvements-2026-07.md``
§A7): scanned-textbook chapters emit the pedagogical *openers* — ``Learning
Objectives`` / ``Be Prepared`` / ``Try It`` / ``Example`` / ``How To`` /
``Solution`` — as flat inline ``<p>`` prose (or fused into the front of the
following paragraph), never as headings. Because the chunker only breaks a
section on a real ``<hN>`` boundary (see ``Trainforge/chunker/chunker.py``), a
single ``<p>`` routinely fuses a worked example, its solution, two TRY-IT
exercises and the next example.

This module is a pure text predicate over an opener candidate's text, applied at
the Ed4All adapter seam (:mod:`lib.semantik.adapter`) so BOTH the live-conversion
render path AND the ``scripts/ops/semantik_rerender.py`` re-render path promote the
opener labels to real ``<h4>`` headings carrying a machine-readable
``data-semantik-opener`` role — no cascade re-run.

Conservative by design (mirrors :mod:`lib.semantik.heading_classifier`): the
STANDALONE arm matches only a whole-line opener label (up to 3 leading OCR
gutter glyphs + an optional ``N`` / ``N.N`` number); the LEADING-SPLIT arm
additionally requires heading-like casing on the matched label and a
non-trivial trailing prose remainder, and requires the number for the
numbered openers (``EXAMPLE 9.1`` splits; a lowercase ``example shows …``
never does). ``Key Terms`` is deliberately NOT handled here — it is owned by
the apparatus path in :mod:`lib.semantik.heading_classifier`.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Wave #22 — the opener vocabulary is sourced from the profile-organized
# pedagogical lexicon (``schemas/taxonomies/semantik_lexicon.json``) via the
# canonical loader, so a new corpus is onboarded by a lexicon entry, not a code
# edit. Behavior-preserving: the default ``generic-academic+open-textbook`` profile
# reproduces the historical hardcoded ``_OPENERS`` tuple (role / pattern /
# display / numbered / interior-split) exactly (byte-identical classifier
# behavior; the compiled-regex STRING may reorder harmlessly since the label
# patterns are disjoint). The role slugs below are re-exported constants so
# downstream imports (``ROLE_TRY_IT`` etc.) are unchanged.
from lib.ontology.taxonomy import get_lexicon_openers

# ---------------------------------------------------------------------------
# Canonical opener role slugs (the ``data-semantik-opener`` / ``data-semantik-block-role``
# value stamped on the promoted heading block). Mirror the lexicon ``role`` keys.
# ---------------------------------------------------------------------------
ROLE_OBJECTIVES = "objectives"
ROLE_READINESS = "readiness_check"
ROLE_TRY_IT = "try_it"
ROLE_WORKED_EXAMPLE = "worked_example"
ROLE_HOW_TO = "how_to"
ROLE_SOLUTION = "solution"

# Module-level cached lexicon view (resolved at import from
# ``SEMANTIK_LEXICON_PROFILE``, default ``generic-academic+open-textbook``).
_LEXICON_OPENERS = get_lexicon_openers()

#: Every opener role this module can emit — consumed by the adapter emit-filter
#: bypass (``adapter._is_opener_promoted``) so a promoted opener heading is
#: never dropped by the ``_is_noncontent_heading`` filter. Derived from the
#: active lexicon profile.
OPENER_ROLES = frozenset(o["role"] for o in _LEXICON_OPENERS)

#: Abstract composite-unit ROLE for each concrete opener role (Wave #22 Tier-2):
#: e.g. ``worked_example`` -> ``example``, ``try_it`` -> ``practice``,
#: ``how_to`` -> ``procedure``. Consumed by the adapter's composite-unit pass so
#: association rules are expressed over the abstract vocabulary, not slugs.
OPENER_ASSOCIATION_ROLE: Dict[str, str] = {
    o["role"]: o.get("association_role", o["role"]) for o in _LEXICON_OPENERS
}

# Opener spec: (label-body regex, role, canonical Title-Case display, numbered?).
# ``numbered`` openers carry a textbook exercise number (``9.1``); the number
# is folded into the display ("Try It 9.1"). Built from the lexicon.
_OPENERS = tuple(
    (o["pattern"], o["role"], o["display"], bool(o["numbered"]))
    for o in _LEXICON_OPENERS
)

# Up to 3 leading OCR gutter glyphs (``)`` / ``™`` / a stray rule), then label.
_GUTTER = r"[^A-Za-z0-9]{0,3}\s*"
# Optional trailing exercise number (``9`` / ``9.1``), captured.
_NUM = r"(?:\s+(\d+(?:\.\d+)?))?"

# Standalone matcher per opener — the WHOLE line is just the label (+ number),
# with an optional trailing colon / period.
_STANDALONE = tuple(
    (
        re.compile(rf"^\s*{_GUTTER}(?:{body}){_NUM}\s*[:.]?\s*$", re.IGNORECASE),
        role,
        display,
        numbered,
    )
    for body, role, display, numbered in _OPENERS
)

# Leading-split matcher per opener — label (+ number) at the very start, then a
# separator, then trailing prose (captured). ``$`` is not anchored.
_LEADING = tuple(
    (
        re.compile(
            rf"^\s*({_GUTTER}(?:{body}){_NUM})\s*[:.]?\s+(\S.*)$",
            re.IGNORECASE | re.DOTALL,
        ),
        role,
        display,
        numbered,
    )
    for body, role, display, numbered in _OPENERS
)

#: The leading remainder must carry at least this many word tokens to count as
#: real trailing prose (else it is the standalone case).
_MIN_REMAINDER_WORDS = 2


def _is_heading_like_cased(text: str) -> bool:
    """Whether ``text`` is cased like a heading: ALL CAPS or Title Case.

    Guards the leading split against a lowercase mid-sentence usage
    ("try it again later"): "Try It" / "TRY IT" pass; "try it" does not.
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


def classify_opener_label(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """Classify a STANDALONE opener label.

    Returns ``(display, role)`` when ``text`` is a whole-line opener label
    (``EXAMPLE 9.1`` → ``("Example 9.1", "worked_example")``,
    ``Learning Objectives`` → ``("Learning Objectives", "objectives")``), else
    ``None``. The display carries canonical Title Case + the folded number.
    """
    if not text:
        return None
    for rx, role, display, _numbered in _STANDALONE:
        m = rx.match(text)
        if m:
            num = m.group(1)
            return (f"{display} {num}" if num else display, role)
    return None


# ---------------------------------------------------------------------------
# ITEM 4 (round-5 audit) — LABEL-ONLY stacked-opener block.
# ---------------------------------------------------------------------------
# The scan sometimes drops the BODY of a worked example upstream, leaving a bare
# paragraph that is EXACTLY the stacked opener labels — ch02 s74 "EXAMPLE 2.3
# Solution" (the example number label + a bare "Solution" whose steps were lost).
# Emitting these as the stacked opener HEADINGS (empty groups) is an honest
# marker of where the example sits (and lets the chunker see the boundary), where
# today it ships as undifferentiated prose. STRICT guard: the whole block text
# must decompose into >= 2 opener labels and NOTHING else — a single label is the
# standalone case (:func:`classify_opener_label`); any trailing prose is the
# leading-split case (:func:`split_leading_opener`); neither is owned here.
_LABEL_ONLY_OPENERS = tuple(
    (
        re.compile(
            rf"^\s*{_GUTTER}(?:{body}){_NUM}\s*[:.]?\s*", re.IGNORECASE
        ),
        role,
        display,
        numbered,
    )
    for body, role, display, numbered in _OPENERS
)


def split_label_only_openers(
    text: Optional[str],
) -> Optional[list]:
    """Decompose a LABEL-ONLY block into its stacked opener labels (ITEM 4).

    Returns ``[(display, role), …]`` when ``text`` consists EXACTLY of a run of
    >= 2 opener labels (each in a heading-like Title-Case / ALL-CAPS form, with
    the required number for the numbered openers) and NOTHING else — e.g.
    ``"EXAMPLE 2.3 Solution"`` → ``[("Example 2.3", "worked_example"),
    ("Solution", "solution")]``. Returns ``None`` for a single label (the
    standalone case), any leftover non-label content (real prose → the
    leading-split case owns it), or a lowercase mid-sentence usage. The strict
    "nothing else remains" contract is the anti-fabrication guard — it never
    invents a heading where the block carries real body text.
    """
    if not text:
        return None
    rest = text.strip()
    labels: list = []
    while rest:
        matched = False
        for rx, role, display, numbered in _LABEL_ONLY_OPENERS:
            m = rx.match(rest)
            if not m:
                continue
            num = m.group(1)
            if numbered and not num:
                # A numbered opener with no number is a prose word — refuse.
                continue
            if not _is_heading_like_cased(m.group(0)):
                continue
            labels.append((f"{display} {num}" if num else display, role))
            rest = rest[m.end():].strip()
            matched = True
            break
        if not matched:
            return None  # leftover non-label content → not a label-only block
    if len(labels) < 2:
        return None  # a single label is the standalone case
    return labels


def split_leading_opener(
    text: Optional[str],
) -> Optional[Tuple[str, str, str]]:
    """Split a paragraph that STARTS WITH an opener label + trailing prose.

    Returns ``(display, role, remainder)`` when ``text`` begins with an opener
    label — in a heading-like Title-Case / ALL-CAPS form, with the number
    present for the numbered openers (``Be Prepared`` / ``Try It`` / ``Example``)
    — followed by ``>= _MIN_REMAINDER_WORDS`` words of prose. Returns ``None``
    otherwise (no leading opener, wrong casing, missing required number, or a
    trivial remainder = the standalone case). The display carries canonical
    Title Case + the folded number.
    """
    if not text:
        return None
    for rx, role, display, numbered in _LEADING:
        m = rx.match(text)
        if not m:
            continue
        label_raw = m.group(1)
        num = m.group(2)
        remainder = m.group(3).strip()
        if numbered and not num:
            # A numbered opener with no number is almost certainly a prose word
            # ("Example shows …") — refuse the split.
            continue
        if not _is_heading_like_cased(label_raw):
            continue
        if len(remainder.split()) < _MIN_REMAINDER_WORDS:
            continue
        return (f"{display} {num}" if num else display, role, remainder)
    return None


# ---------------------------------------------------------------------------
# ITEM 4 (round-2 audit) — INTERIOR numbered-opener split.
# ---------------------------------------------------------------------------
# A textbook scan fuses successive worked units into one run-on <p>:
# "… Combine like radicals. TRY IT 9.201 Simplify: … TRY IT 9.202 Simplify: …
# EXAMPLE 9.102 Simplify: …". Each interior marker is a STRONG shape — a
# numbered opener label (``TRY IT`` / ``EXAMPLE`` / ``BE PREPARED``) + a decimal
# section number (``9.201``) — the same pattern class as the interior apparatus
# banner split (``heading_classifier.split_interior_apparatus_heading``), so
# splitting at every such marker is safe and de-fuses the block into one unit per
# opener. ONLY the numbered openers participate (the bare ``Solution`` / ``How
# To`` / ``Learning Objectives`` are NOT interior split points — too weak a
# shape). The required decimal number + heading-like casing defeats a lowercase
# mid-sentence "for example 9.2" usage.
# Interior-split-capable openers = the lexicon's ``interior_split``-flagged
# (numbered) openers. Reproduces the historical (try_it / example / be_prepared)
# set; the ALL-ORDER is harmless (disjoint alternation).
_INTERIOR_OPENERS = tuple(
    (o["pattern"], o["role"], o["display"])
    for o in _LEXICON_OPENERS
    if o.get("interior_split")
)
# A numbered interior marker: a label body + a REQUIRED decimal section number.
_INTERIOR_OPENER_RE = re.compile(
    r"(?<![A-Za-z])("
    + "|".join(body for body, _r, _d in _INTERIOR_OPENERS)
    + r")\s+(\d+\.\d+)\b",
    re.IGNORECASE,
)
# Delimited math is masked before the marker scan so a marker-shaped token inside
# a ``$…$`` run is never a split point (math-aware, per the §A3 caveat).
_INTERIOR_MATH_MASK_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$[^$]*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)
# Round-3 (Defect 3) — a markdown pipe-row RUN is masked too, so a marker fused
# INTO a pipe-table CELL (``| TRY IT :: 2.35 | …``) is NEVER a split point: only
# a marker OUTSIDE the pipe rows (the trailing ``… | TRY IT 2.36 …`` the cascade
# spilled past the last pipe) splits off. This lets a ``table``-declared block's
# trailing fused openers surface as separate blocks while the pipe run stays
# intact for the downstream ``parse_table`` reconstruction. Harmless on prose (a
# lone ``|`` never matches; a bare absolute-value ``|x|`` carries no opener).
_INTERIOR_PIPE_MASK_RE = re.compile(r"(?:\|[^|\n]*)+\|")
#: Min word count for the text preceding the FIRST interior marker (a genuine
#: interior fusion has real content before the marker) and each remainder.
_INTERIOR_MIN_WORDS = 2


def _role_display_for_body(label_raw: str) -> Optional[Tuple[str, str]]:
    """Map a matched interior label to its ``(role, canonical display)``."""
    key = " ".join(label_raw.lower().split())
    for body, role, display in _INTERIOR_OPENERS:
        if re.fullmatch(body, key):
            return role, display
    return None


def split_interior_openers(
    text: Optional[str],
) -> Optional[list]:
    r"""Split a block that FUSED interior numbered openers into ordered parts.

    Returns an ordered list of parts — ``("text", content)`` for a prose span and
    ``("opener", display, role)`` for a promoted opener heading — when ``text``
    carries at least one INTERIOR numbered opener (a ``TRY IT`` / ``EXAMPLE`` /
    ``BE PREPARED`` + ``N.NNN`` marker with non-empty preceding text and
    heading-like casing). Returns ``None`` when there is no such interior marker
    (a leading-only opener is owned by :func:`split_leading_opener`; a block with
    no marker is untouched). Math runs AND markdown pipe-table cell runs are
    masked so a marker inside ``$…$`` or inside a ``| … |`` cell is never a split
    point — only a marker OUTSIDE the pipe rows (a trailing opener the cascade
    spilled past the last pipe) splits off (Defect 3). First-and-every
    occurrence; the display carries canonical Title Case + the section number.
    """
    if not text:
        return None
    masked = _INTERIOR_MATH_MASK_RE.sub(
        lambda m: " " * len(m.group(0)), text
    )
    # Mask pipe-table cell runs so a marker fused inside a cell is not a split
    # point (Defect 3) — only markers OUTSIDE the pipe rows split off.
    masked = _INTERIOR_PIPE_MASK_RE.sub(lambda m: " " * len(m.group(0)), masked)
    marks = []
    for m in _INTERIOR_OPENER_RE.finditer(masked):
        label_raw = m.group(1)
        # Heading-like casing guard (ALL CAPS / Title Case) — defeats a lowercase
        # "for example 9.2" mid-sentence usage.
        if not _is_heading_like_cased(label_raw):
            continue
        rd = _role_display_for_body(label_raw)
        if rd is None:
            continue
        marks.append((m.start(), m.end(), m.group(2), rd))
    if not marks:
        return None
    lead = text[: marks[0][0]].strip()
    leading_marker = len(lead.split()) < _INTERIOR_MIN_WORDS
    # Fire on a genuine fusion: EITHER >= 2 fused openers (split them all, even
    # when the first is at the block start), OR a single opener with non-trivial
    # prose BEFORE it. A lone LEADING opener (1 marker, no preceding prose) is a
    # single-unit block owned by :func:`split_leading_opener` — return None.
    if len(marks) < 2 and leading_marker:
        return None
    parts: list = []
    if not leading_marker and lead:
        parts.append(("text", lead))
    for i, (_s, e, num, (role, display)) in enumerate(marks):
        parts.append(("opener", f"{display} {num}", role))
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        remainder = text[e:end].strip()
        if remainder:
            parts.append(("text", remainder))
    return parts


__all__ = [
    "OPENER_ROLES",
    "OPENER_ASSOCIATION_ROLE",
    "ROLE_OBJECTIVES",
    "ROLE_READINESS",
    "ROLE_TRY_IT",
    "ROLE_WORKED_EXAMPLE",
    "ROLE_HOW_TO",
    "ROLE_SOLUTION",
    "classify_opener_label",
    "split_label_only_openers",
    "split_leading_opener",
    "split_interior_openers",
]
