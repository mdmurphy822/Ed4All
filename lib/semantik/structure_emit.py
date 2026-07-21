r"""Deterministic, shape-driven structural-body emission (adapter-seam, model-free).

Exemplar-parity wave (2026-07-04, ``plans/enduser-html-improvements-2026-07.md``
§A1/§A4/§A5): the SemantiK cascade ships every block as a flat ``<p>`` body even
when it declares ``data-semantik-block-role="list"`` / ``"table"`` /
``"definition_list"`` — so the emitted HTML never *delivers* the structure it
*declares* (the structure-scorecard "deliver" dimension reads ~0). This module
reconstructs the real semantic element from the block's deterministic text when —
and ONLY when — the SHAPE is high-confidence:

* :func:`parse_table` — a run of markdown pipe rows (``| a | b | | c | d |``,
  with an optional ``| --- | --- |`` header separator) → ``<table>`` with a
  ``<thead>`` of ``<th scope="col">`` cells when a separator is present.
* :func:`parse_definition_list` — a ``term: definition`` run (>= 2 pairs, short
  non-sentence term prefixes, >= 70% text coverage) → ``<dl><dt>…<dd>…``.
* :func:`parse_list` — a run of >= 2 CONSECUTIVE list markers — bullets
  (``• - *``), sequential ``(a)(b)(c)`` alpha parts, or sequential ``1. 2. 3.``
  enumeration — covering >= 80% of the text mass → ``<ul>`` / ``<ol>``.

The OWNER CONTRACT (this file is a *wide net* for any human-readable content):
every rule keys on SHAPE (pipes, item markers, term-def punctuation), NEVER on
subject-matter vocabulary, and NEVER fabricates a boundary — a low-confidence
shape is left as prose (returns ``None``) so the adapter can honestly demote the
mis-declared role to ``paragraph`` rather than ship an invented structure. Inline
math (``$…$`` / ``\(…\)`` / ``\[…\]``) is masked out before shape-scanning and
restored verbatim inside each cell / item, so an item's math is preserved and a
mid-math pipe / paren / period never trips the detector.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

__all__ = [
    "emit_structure",
    "parse_table",
    "parse_definition_list",
    "parse_list",
    "STRUCTURAL_ROLES",
]

#: The declared roles held to the "declare → deliver" contract. A block carrying
#: one of these whose shape is not high-confidence is demoted to ``paragraph`` by
#: the adapter (declaration/delivery reconciliation).
STRUCTURAL_ROLES = frozenset({"list", "table", "definition_list"})


def _esc(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Delimited-math spans are masked out before shape-scanning. ``(?<!\\)`` so an
# escaped ``\$`` (currency) is not treated as a math delimiter.
_MATH_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$[^$]*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)
_PLACEHOLDER_RE = re.compile("\x00(\\d+)\x00")


def _mask_math(text: str) -> Tuple[str, List[str]]:
    """Replace each delimited-math span with an opaque placeholder token."""
    spans: List[str] = []

    def _sub(m: "re.Match[str]") -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    return _MATH_SPAN_RE.sub(_sub, text), spans


def _unmask(text: str, spans: List[str]) -> str:
    return _PLACEHOLDER_RE.sub(lambda m: spans[int(m.group(1))], text)


def _cell(text: str, spans: List[str]) -> str:
    """Escape a cell/item's non-math text and restore its verbatim math spans."""
    return _unmask(_esc(text), spans)


def _orig_len(masked: str, spans: List[str]) -> int:
    return len(_unmask(masked, spans))


# ---------------------------------------------------------------------------
# TABLE (§A4)
# ---------------------------------------------------------------------------
# A contiguous pipe region: one or more ``| cell`` groups closed by a final
# ``|``. Cells never contain a raw pipe (math pipes are masked out first).
_PIPE_REGION_RE = re.compile(r"(?:\|[^|\n]*)+\|")
_SEP_CELL_RE = re.compile(r":?-{2,}:?")
# A fragment that is ONLY pipe/gutter debris (pipes, ``>``/``&gt;`` glyphs,
# whitespace). Such a pre/post/lead fragment around a parsed structure is OCR
# cell-boundary residue, not content — it must never ship as a ``<p>``.
_PIPE_ONLY_FRAGMENT_RE = re.compile(r"^(?:\||>|&gt;|\s)+$")


def _fragment(text: str) -> str:
    """Normalize a pre/post/lead fragment: pipe-only debris collapses to ''."""
    text = text.strip()
    if text and _PIPE_ONLY_FRAGMENT_RE.fullmatch(text):
        return ""
    return text


def parse_table(text: str) -> Optional[str]:
    r"""Reconstruct a ``<table>`` from a markdown pipe-row run inside ``text``.

    Finds the longest contiguous pipe region, splits it into rows on the
    ``| |`` row boundary that markdown-row concatenation produces, and emits a
    ``<table>``. When a ``| --- | --- |`` separator row is present the row above
    it becomes a ``<thead>`` of ``<th scope="col">`` cells. Text before / after
    the pipe region is preserved as ``<p>`` siblings (no fabrication). Returns
    ``None`` when there is no >= 2-row, >= 2-column pipe region.
    """
    masked, spans = _mask_math(text)
    best: Optional[re.Match[str]] = None
    for m in _PIPE_REGION_RE.finditer(masked):
        if best is None or (m.end() - m.start()) > (best.end() - best.start()):
            best = m
    if best is None:
        return None
    region = masked[best.start() : best.end()]
    pre = _fragment(masked[: best.start()])
    post = _fragment(masked[best.end() :])

    rows: List[List[str]] = []
    for raw_row in re.split(r"\|\s*\|", region):
        cells = [c.strip() for c in raw_row.split("|")]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append(cells)
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return None

    sep_idx: Optional[int] = None
    for i, row in enumerate(rows):
        if all(_SEP_CELL_RE.fullmatch(c) for c in row):
            sep_idx = i
            break

    header: Optional[List[str]] = None
    body = rows
    if sep_idx is not None:
        if sep_idx > 0:
            header = rows[sep_idx - 1]
            body = rows[:sep_idx - 1] + rows[sep_idx + 1 :]
        else:
            body = rows[sep_idx + 1 :]
    if not body:
        return None

    parts: List[str] = ["<table>"]
    if header:
        parts.append("<thead><tr>")
        parts.extend(f'<th scope="col">{_cell(c, spans)}</th>' for c in header)
        parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{_cell(c, spans)}</td>" for c in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")

    out = ""
    if pre:
        out += f"<p>{_cell(pre, spans)}</p>"
    out += "".join(parts)
    if post:
        out += f"<p>{_cell(post, spans)}</p>"
    return out


# ---------------------------------------------------------------------------
# DEFINITION LIST (§A5)
# ---------------------------------------------------------------------------
# A ``term: `` anchor — a short (<= 60 char) prefix at a sentence/segment
# boundary, followed by a colon + space + a Capitalized / math definition start.
_DL_ANCHOR_RE = re.compile(
    r"(?:^|(?<=[.\s]))\s*([A-Za-z][A-Za-z0-9 '\-]{0,60}?):\s+(?=[A-Z$\\])"
)
_DL_MIN_PAIRS = 2
_DL_MAX_TERM_WORDS = 6
_DL_MAX_LEAD_FRACTION = 0.30  # >= 70% coverage


def parse_definition_list(text: str) -> Optional[str]:
    r"""Reconstruct a ``<dl>`` from a ``term: definition`` run inside ``text``.

    Requires >= 2 ``term: definition`` pairs where each term is a short
    (<= 6-word) non-sentence prefix and the pairs cover >= 70% of the text mass.
    Returns ``None`` otherwise (the anti-fabrication guard — a lone ``word:`` in
    prose, or a single pair, is left as prose).
    """
    masked, spans = _mask_math(text)
    marks = list(_DL_ANCHOR_RE.finditer(masked))
    if len(marks) < _DL_MIN_PAIRS:
        return None
    pairs: List[Tuple[str, str]] = []
    for i, m in enumerate(marks):
        term = m.group(1).strip()
        if not term or len(term.split()) > _DL_MAX_TERM_WORDS:
            return None
        # An ALL-CAPS term is a pedagogical opener label ("TRY IT ::", "BE
        # PREPARED"), not a glossary term (which is lower / Title case). Shape
        # guard (casing), not a vocabulary check.
        letters = [c for c in term if c.isalpha()]
        if letters and all(c.isupper() for c in letters):
            return None
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(masked)
        defn = masked[start:end].strip()
        if not defn:
            return None
        pairs.append((term, defn))
    total = _orig_len(masked, spans)
    lead = _unmask(masked[: marks[0].start()], spans)
    if total == 0 or (len(lead) / total) > _DL_MAX_LEAD_FRACTION:
        return None
    body = "".join(
        f"<dt>{_cell(t, spans)}</dt><dd>{_cell(d, spans)}</dd>" for t, d in pairs
    )
    return f"<dl>{body}</dl>"


# ---------------------------------------------------------------------------
# LIST (§A1)
# ---------------------------------------------------------------------------
_BULLET_RE = re.compile(r"[•‣◦⁃]\s*")
_ALPHA_MARK_RE = re.compile(r"\(([a-zA-Z])\)\s*")
_NUM_MARK_RE = re.compile(r"(?<![\d.])(\d{1,3})[.)]\s+")
_SENTENCE_BREAK_RE = re.compile(r"\.\s+[A-Z]")
_LIST_MAX_LEAD_FRACTION = 0.20  # >= 80% coverage
_LIST_MAX_ITEM_CHARS = 120

# ITEM 2 (round-2 audit) — bare single-letter alpha part markers (``a`` … ``b``
# … ``c`` with NO parentheses), whitespace/math-placeholder bounded so a letter
# glued inside a prose word never matches. Restricted to ``a``-``e`` (the OpenStax
# exercise-part alphabet) to keep the false-positive surface tiny; the math /
# short-content + cycle-structure guards below defeat the prose-article ``a``.
_BARE_ALPHA_MARK_RE = re.compile(r"(?:(?<=[\s\x00])|^)([a-e])(?=[ \t]*\x00|\s)")
_ALPHA_CYCLE_MAX_LEAD_FRACTION = 0.30  # >= 70% coverage (ITEM 2 spec)


def _num_sequential(labels: List[str]) -> bool:
    nums = [int(x) for x in labels]
    return all(nums[i] - nums[0] == i for i in range(len(nums)))


def _split_marked_items(
    regex: re.Pattern[str],
    seq_ok,
    masked: str,
    spans: List[str],
) -> Optional[Tuple[int, List[str]]]:
    """Split an enumerated run into ``(lead_start, items)`` or ``None``.

    Guards: >= 2 markers, the labels are strictly SEQUENTIAL (``a,b,c`` /
    ``1,2,3`` — never an arbitrary scatter of parenthesised letters), the run
    covers >= 80% of the text mass, and every item is SHORT (no internal
    sentence break, <= 120 chars) so fused worked-example prose ("(a) Since … .
    Now (b) …") is refused.
    """
    marks = list(regex.finditer(masked))
    if len(marks) < 2:
        return None
    if not seq_ok([m.group(1) for m in marks]):
        return None
    lead = masked[: marks[0].start()]
    total = _orig_len(masked, spans)
    if total == 0 or (len(_unmask(lead, spans)) / total) > _LIST_MAX_LEAD_FRACTION:
        return None
    items: List[str] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(masked)
        seg = masked[start:end].strip(" .;,")
        if not seg:
            return None
        restored = _unmask(seg, spans)
        if _SENTENCE_BREAK_RE.search(restored) or len(restored) > _LIST_MAX_ITEM_CHARS:
            return None
        items.append(seg)
    return marks[0].start(), items


def _decompose_alpha_cycles(labels: List[str]) -> Optional[List[List[int]]]:
    """Decompose alpha part labels into sequential-from-``a`` cycles or ``None``.

    An exercise block repeats its part alphabet per numbered item — ``a b c a b
    c`` — so the labels must decompose cleanly into cycles that each START at
    ``a`` and increment by one (``a`` → ``b`` → ``c``). Returns a list of cycles
    (each a list of indices into ``labels``) when the whole sequence is such a
    clean repetition, else ``None`` (a scrambled / non-``a``-anchored run is left
    as prose — the anti-fabrication guard). Requires the first label to be ``a``
    and at least one cycle of length >= 2 (a degenerate all-``a`` run — a prose
    article repeated — is refused).
    """
    if not labels or labels[0].lower() != "a":
        return None
    cycles: List[List[int]] = []
    cur: List[int] = [0]
    for i in range(1, len(labels)):
        prev = labels[i - 1].lower()
        curr = labels[i].lower()
        if ord(curr) - ord(prev) == 1:
            cur.append(i)
        elif curr == "a":
            cycles.append(cur)
            cur = [i]
        else:
            return None  # broken (non-sequential, non-reset) → refuse
    cycles.append(cur)
    if max(len(c) for c in cycles) < 2:
        return None
    return cycles


def _parse_alpha_marked(
    regex: "re.Pattern[str]", masked: str, spans: List[str], *, bare: bool
) -> Optional[str]:
    """Reconstruct an ``<ol type="a">`` (or nested exercise list) from alpha marks.

    ITEM 2 — handles BOTH the single-run case (``(a)…(b)…(c)`` → one
    ``<ol type="a">``) and the REPEATING case (``(a)…(b)…(a)…(b)…`` → a nested
    ``<ol class="semantik-exercise-list">`` of per-cycle ``<ol type="a">`` lists, one
    numbered exercise per cycle). Guards (anti-fabrication): >= 3 total markers,
    a clean sequential-from-``a`` cycle decomposition, >= 70% text coverage, and
    every item is math-bearing OR short (no sentence break, <= 120 chars). For
    ``bare`` (no-paren) markers a stronger guard also requires >= half the items
    to carry a math run, so a prose ``a … b …`` never triggers. Returns ``None``
    when any guard fails (the block stays prose).
    """
    marks = list(regex.finditer(masked))
    # Parenthesized ``(a)`` markers are unambiguous → a 2-part ``(a)(b)`` list is
    # allowed (preserves prior behavior). Bare ``a``/``b`` markers are ambiguous
    # with prose, so the spec's >= 3-alternation floor applies to them only.
    min_marks = 3 if bare else 2
    if len(marks) < min_marks:
        return None
    labels = [m.group(1) for m in marks]
    cycles = _decompose_alpha_cycles(labels)
    if cycles is None:
        return None
    lead = masked[: marks[0].start()]
    total = _orig_len(masked, spans)
    if total == 0 or (
        len(_unmask(lead, spans)) / total
    ) > _ALPHA_CYCLE_MAX_LEAD_FRACTION:
        return None
    items: List[str] = []
    math_bearing = 0
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(masked)
        seg = masked[start:end].strip(" .;,")
        if not seg:
            return None
        restored = _unmask(seg, spans)
        # A fused worked-example part ("(a) Since … . Now (b) …") or an
        # over-long item is refused UNCONDITIONALLY — a math run inside the item
        # does not exempt it from the short-non-sentence contract.
        if _SENTENCE_BREAK_RE.search(restored) or len(restored) > _LIST_MAX_ITEM_CHARS:
            return None
        if "\x00" in seg:
            math_bearing += 1
        items.append(seg)
    if bare and math_bearing * 2 < len(items):
        # A bare (no-paren) run that is mostly NON-math is prose, not an
        # exercise list — refuse (defeats "a dog and b cat …").
        return None
    # ITEM 6 (round-5 audit) — a bare exercise-number lead-in ("430." before the
    # first ``(a)``) is the numbered exercise's own number, not prose. For the
    # nested exercise-list (multi-cycle) shape it is folded into the list's
    # ``start`` attribute — the outer ``<ol>`` numbers its first exercise ``430``
    # — instead of shipping a bare ``<p>430</p>`` debris paragraph. STRICT guard:
    # only when the lead is EXACTLY a 1-4 digit number AND the list IS a numbered
    # exercise list (>= 2 cycles → ``<ol class="semantik-exercise-list">``); a
    # single-cycle ``<ol type="a">`` (alpha parts, no numeric wrapper) keeps its
    # lead as prose (a numeric ``start`` on an alpha list is nonsensical).
    lead_stripped = lead.strip(" :.")
    exercise_start: Optional[str] = None
    if len(cycles) > 1 and re.fullmatch(r"\d{1,4}", lead_stripped):
        exercise_start = lead_stripped
        lead_stripped = ""
    if len(cycles) == 1:
        lis = "".join(f"<li>{_cell(items[i], spans)}</li>" for i in cycles[0])
        body = f'<ol type="a">{lis}</ol>'
    else:
        start_attr = f' start="{exercise_start}"' if exercise_start else ""
        parts: List[str] = [f'<ol class="semantik-exercise-list"{start_attr}>']
        for cyc in cycles:
            parts.append('<li><ol type="a">')
            parts.extend(f"<li>{_cell(items[i], spans)}</li>" for i in cyc)
            parts.append("</ol></li>")
        parts.append("</ol>")
        body = "".join(parts)
    if lead_stripped.strip():
        body = f"<p>{_cell(lead_stripped, spans)}</p>" + body
    return body


def parse_list(text: str) -> Optional[str]:
    r"""Reconstruct a ``<ul>`` / ``<ol>`` from a marker run inside ``text``.

    Bullet glyphs (``• ‣ ◦ ⁃``) → ``<ul>``; sequential ``(a)(b)(c)`` → an
    ``<ol type="a">`` (exercise parts); a REPEATING ``(a)(b)(c)(a)(b)(c)`` (or
    the bare ``a … b … c …`` form) → a nested exercise list (ITEM 2); sequential
    ``1. 2. 3.`` → ``<ol>``. A short lead-in before the first marker
    ("Simplify:") is preserved as a ``<p>``. Returns ``None`` when no
    high-confidence marker run is present.
    """
    masked, spans = _mask_math(text)

    if len(_BULLET_RE.findall(masked)) >= 2:
        parts = [p.strip() for p in _BULLET_RE.split(masked) if p.strip()]
        if len(parts) >= 2:
            lis = "".join(f"<li>{_cell(p, spans)}</li>" for p in parts)
            return f"<ul>{lis}</ul>"

    # Alpha part markers (ITEM 2): prefer the unambiguous parenthesized ``(a)``
    # form; fall back to the bare ``a`` form (guarded harder for prose). Both are
    # cycle-aware (repeating exercise alphabets → a nested list).
    if len(_ALPHA_MARK_RE.findall(masked)) >= 2:
        alpha = _parse_alpha_marked(_ALPHA_MARK_RE, masked, spans, bare=False)
        if alpha is not None:
            return alpha
    elif len(_BARE_ALPHA_MARK_RE.findall(masked)) >= 3:
        alpha = _parse_alpha_marked(
            _BARE_ALPHA_MARK_RE, masked, spans, bare=True
        )
        if alpha is not None:
            return alpha

    split = _split_marked_items(_NUM_MARK_RE, _num_sequential, masked, spans)
    if split is not None:
        start, items = split
        lead = _fragment(masked[:start].strip(" :."))
        lis = "".join(f"<li>{_cell(it, spans)}</li>" for it in items)
        body = f"<ol>{lis}</ol>"
        if lead:
            body = f"<p>{_cell(lead, spans)}</p>" + body
        return body
    return None


def emit_structure(text: str) -> Optional[Tuple[str, str]]:
    """Return ``(role, html)`` for the highest-confidence structure in ``text``.

    Tries table → definition list → list (most-specific / lowest-false-positive
    first). Returns ``None`` when no shape is high-confidence, so the caller
    leaves the block as prose (and reconciles a mis-declared structural role to
    ``paragraph``).
    """
    if not text or not text.strip():
        return None
    table = parse_table(text)
    if table is not None:
        return ("table", table)
    dl = parse_definition_list(text)
    if dl is not None:
        return ("definition_list", dl)
    lst = parse_list(text)
    if lst is not None:
        return ("list", lst)
    return None
