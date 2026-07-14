r"""Deterministic table CELL-TOPOLOGY reconstruction (``SEMANTIK_TABLE_STRUCTURE``).

Why this lane exists
--------------------
The scan-lane VLM page-arranger transcribes a textbook table as a **markdown pipe
grid** — and it emits the ``| --- | --- |`` HEADER SEPARATOR row along with the
data rows, inside ``region_provenance[].raw_text``. That separator is the only
piece of evidence in the whole pipeline that says *"row 0 is a header, and this
table has exactly N columns"*.

Today that evidence is DESTROYED before anything can read it:
:func:`lib.semantik.math_fold.sanitize_body_latex` drops the separator row as
"markdown garbage" (``_MD_SEP_ROW_RE.sub(" ", s)``) while sanitizing body text —
which runs BEFORE ``adapter._emit_structured_bodies`` calls
``structure_emit.parse_table``. So ``parse_table``'s ``<thead>`` /
``<th scope="col">`` branch is **unreachable dead code on the live path** (measured:
``parse_table`` is called 1421× on a real chapter render and the separator is
present in ZERO of them), and production ships **0 ``<th>`` elements** — every
table is a headerless ``<td>`` soup, which a screen reader cannot navigate
(WCAG 2.2 SC 1.3.1).

The second defect is structural. ``parse_table`` splits rows on
``re.split(r"\|\s*\|", region)`` and then drops empty cells
(``[c for c in cells if c != ""]``) — but ``| |`` is EXACTLY what an EMPTY CELL
looks like, so an empty cell both SHREDS the row split and gets DELETED. A real
self-assessment checklist ("I can… | Confidently | With some help | No—I don't
get it!" with blank response cells) renders today as one 4-cell row followed by a
run of 1-cell rows: ragged, column-misaligned, and semantically wrong.

The insight
-----------
**The separator row is a TOPOLOGY DECLARATION, not garbage.** Its dash-cell count
declares ``n_cols``, and once ``n_cols`` is known the whole ``| |`` ambiguity
collapses: a run of ``R`` rows tokenizes to *exactly* ``R*n_cols + (R-1)`` tokens
(``n_cols`` cell tokens per row, one "join artifact" token where row *k*'s closing
pipe meets row *k+1*'s opening pipe). Positional arithmetic — not a guess — then
tells an empty CELL apart from a row JOIN.

So this module keeps the separator (via the additive ``keep_md_sep=True`` kwarg on
``sanitize_body_latex``), reads the topology off it, and reconstructs a real
``<table>`` with ``<thead>`` / ``<th scope="col">``.

Anti-fabrication contract (self-verifying, fail-closed)
------------------------------------------------------
* **No separator → no table.** Without the declaration there is no header
  evidence, so :func:`parse_table_topology` returns ``None`` — it NEVER assumes
  row 0 is a header. The caller falls back to today's ``parse_table`` output.
* **Arithmetic must fit exactly.** If the token count does not fit
  ``R*n_cols + (R-1)``, or any row does not come out to exactly ``n_cols`` cells,
  the parse returns ``None``. A guessed grid is never shipped.
* **Text conservation.** :func:`verify_text_conservation` asserts the multiset of
  non-empty cell texts in the reconstructed topology equals the multiset
  tokenized from the source pipe region. Any drift → ``None``. It is called
  INSIDE :func:`parse_table_topology`, so a topology that loses or duplicates a
  cell can never escape the module.
* **Real accept gate, reused not cloned.** The emitted markup must pass the
  cascade's OWN WCAG H43 table gate
  (``SemantiK/semantik_structure/gates/table_h43.py``: ``verify_h43`` +
  ``analyze_table``). If that module cannot be loaded the gate is treated as
  UNAVAILABLE and the emit is REFUSED (fail closed → today's output), never
  soft-passed.
* **Spans are never fabricated.** A markdown pipe grid cannot express
  ``colspan`` / ``rowspan``; there is nothing to derive, so this module NEVER
  emits a span attribute. Honest under-expression beats an invented merge.
* **Row headers need real evidence.** ``<th scope="row">`` is emitted ONLY on the
  classic stub/matrix shape — the header row's FIRST cell is EMPTY (the corner
  cell), there are ≥2 body rows, and every body row's first cell is non-empty.
  No other heuristic. Otherwise first-column cells stay ``<td>``.

Decision-capture posture
------------------------
This pass is purely **DETERMINISTIC** — it adds no LLM call site, so per house
convention (root ``CLAUDE.md`` § LLM call-site instrumentation) it carries **NO
DecisionCapture obligation**. The VLM's markdown separator row is the topology
PROPOSAL; this module and the reused H43 gate DECIDE.

Gated by ``SEMANTIK_TABLE_STRUCTURE`` at the adapter seam
(``lib/semantik/adapter.py::_resolve_table_structure``); flag OFF is
**byte-identical** — the separator keeps being stripped, ``table_src`` is never
stashed, and this module is not even imported.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# The separator-row regex is the SAME object the sanitizer uses to strip it —
# reused, never re-derived, so the "what counts as a separator" contract cannot
# drift between the stripper and this reconstructor.
from lib.semantik.math_fold import _MD_SEP_ROW_RE, _UNICODE_RE

# Masking / escaping / region-finding primitives are the cascade's own
# (structure_emit) — reused verbatim so a mid-math pipe (``$a|b$``) is masked out
# before tokenization exactly as ``parse_table`` masks it, and a cell's math runs
# are restored verbatim inside the emitted markup.
from lib.semantik.structure_emit import (
    _MATH_SPAN_RE,
    _PIPE_REGION_RE,
    _SEP_CELL_RE,
    _cell,
    _fragment,
    _mask_math,
    _unmask,
)

__all__ = [
    "TableTopology",
    "emit_structured_table",
    "has_separator_row",
    "header_row_is_label_shaped",
    "parse_table_topology",
    "render_topology_html",
    "verify_text_conservation",
]


# ---------------------------------------------------------------------------
# The H43 accept gate — the cascade's own module, loaded BY PATH.
# ---------------------------------------------------------------------------
# ``SemantiK/semantik_structure/gates/table_h43.py`` is PURE stdlib (dataclasses +
# typing), but ``gates/__init__.py`` eagerly pulls the converter-venv-only
# playwright/axe stack — so a normal package import would explode inside a plain
# Ed4All venv. Load the single source file directly under a private module name
# (the ``latex_mathml.load_validate_mathml`` precedent). The module MUST be
# registered in ``sys.modules`` BEFORE ``exec_module``, or the ``@dataclass``
# decorators in its body cannot resolve their own defining module.
_GATES_DIR = (
    Path(__file__).resolve().parents[2] / "SemantiK" / "semantik_structure" / "gates"
)
_H43_MODULE_NAME = "_semantik_table_h43_lite"

_h43_module: Optional[Any] = None
_h43_load_attempted = False


def _load_h43() -> Optional[Any]:
    """Return the cascade's ``table_h43`` module, or ``None`` if unavailable.

    Cached (including the failure) at module level: the gate is consulted once
    per reconstructed table, and a missing SemantiK tree / lxml is a stable
    property of the box, not a per-call condition.
    """
    global _h43_module, _h43_load_attempted
    if _h43_load_attempted:
        return _h43_module
    _h43_load_attempted = True
    try:
        path = _GATES_DIR / "table_h43.py"
        spec = importlib.util.spec_from_file_location(_H43_MODULE_NAME, path)
        if spec is None or spec.loader is None:  # pragma: no cover — defensive
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[_H43_MODULE_NAME] = module  # BEFORE exec (see note above)
        spec.loader.exec_module(module)
        _h43_module = module
    except Exception:  # noqa: BLE001 — gate UNAVAILABLE ⇒ fail closed, never raise
        _h43_module = None
    return _h43_module


_TAG_RE = re.compile(r"<[^>]+>")


def _tag_stripped_text(html: str) -> str:
    """``html`` with every TAG collapsed to a boundary sentinel — its text only.

    The independent witness that :func:`_repair_h43` added ATTRIBUTES and nothing
    else. Every tag (with its attributes) becomes ``\\x00``, so two fragments
    compare equal iff their text nodes AND their cell boundaries are identical —
    an ``id=`` / ``headers=`` addition is invisible, while any drift in cell TEXT
    (or a lost/added cell) is not.
    """
    return _TAG_RE.sub("\x00", html)


def _repair_h43(html: str) -> Optional[str]:
    """H43-enrich a COMPLEX table via the gate module's OWN ``apply_h43``, or ``None``.

    A reconstructed stub/matrix table is DUAL-AXIS (a ``<thead>`` of
    ``<th scope="col">`` PLUS a ``<th scope="row">`` stub column), which
    ``analyze_table`` correctly classifies as COMPLEX — and a complex table needs
    explicit ``<th id>`` / ``<td headers>`` association (WCAG technique H43),
    because ``scope`` alone is ambiguous on two axes. Without this step
    :func:`_h43_accept` would (rightly) reject every such table and the whole
    ``row_headers`` arm would be unreachable.

    The enrichment is the gate module's own deterministic grid arithmetic —
    reused, never cloned — so the authority that JUDGES the association is the
    authority that BUILDS it. ``apply_h43`` is a documented no-op (byte-identical
    input) on a simple table and on a non-table fragment, so the col-headers-only
    path is untouched.

    Returns ``None`` when the gate module is unavailable or ``apply_h43`` raises:
    the repair must never become a NEW failure mode, so the caller falls back to
    the UNrepaired html and still runs the accept gate on it.
    """
    module = _load_h43()
    if module is None:
        return None
    try:
        repaired = module.apply_h43(html)
    except Exception:  # noqa: BLE001 — a failed repair degrades, never raises
        return None
    if not isinstance(repaired, str) or not repaired:
        return None
    return repaired


def _h43_accept(html: str) -> bool:
    """Whether ``html`` passes the cascade's WCAG H43 table gate.

    Two conditions, both from the gate module itself (no cloned logic):
    ``verify_h43(html)[0] is True`` (a COMPLEX table must carry resolvable
    ``id``/``headers`` association; a simple table passes trivially) AND
    ``analyze_table(html).n_header_rows >= 1`` (the whole point of this lane is a
    real header row — a reconstruction that produced none is not worth shipping).

    Fail closed: an unloadable gate module, or ANY exception out of it, returns
    ``False`` so the caller degrades to today's output rather than shipping
    unvalidated markup.
    """
    module = _load_h43()
    if module is None:
        return False
    try:
        passed = module.verify_h43(html)[0]
        if passed is not True:
            return False
        return module.analyze_table(html).n_header_rows >= 1
    except Exception:  # noqa: BLE001 — a gate that cannot judge is a gate that says no
        return False


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableTopology:
    """The cell topology a markdown pipe grid actually declares.

    ``header_rows`` / ``body_rows`` are tuples of per-row cell tuples, each of
    exactly ``n_cols`` cells — EMPTY cells preserved as ``""`` (they are real
    grid positions, not garbage). ``row_headers`` records the evidence-based
    stub-column verdict (see :func:`parse_table_topology`). ``pre`` / ``post``
    are the prose fragments before / after the pipe region, mirroring
    ``structure_emit.parse_table``'s existing pre/post handling so no text is
    dropped when a block carries a table plus surrounding prose.
    """

    header_rows: Tuple[Tuple[str, ...], ...]
    body_rows: Tuple[Tuple[str, ...], ...]
    n_cols: int
    row_headers: bool
    pre: str
    post: str


def has_separator_row(text: str) -> bool:
    """Whether ``text`` carries a markdown table header-separator row.

    The adapter's cheap pre-check: only a block whose raw text declares a
    topology is worth stashing a separator-preserving ``table_src`` for.
    """
    return bool(text) and bool(_MD_SEP_ROW_RE.search(text))


# ---------------------------------------------------------------------------
# Header-label admissibility — the VLM PROPOSES, we DECIDE.
# ---------------------------------------------------------------------------
# Measured on a real chapter (110 tables): the separator row is a FORMATTING TIC,
# not a topology judgment. On worked-example step tables the VLM drops its
# ``|---|`` after the FIRST STEP ROW, so accepting the declaration uncritically
# promotes a DATA row to a column header — a ~65% false-positive rate on the
# ``scope="col"`` arm. A ``<th>`` in the wrong place actively MISLEADS a screen-
# reader user (it is announced as the column's label for every cell below it), so
# a wrong header is strictly worse than no header.
#
# So the separator is necessary evidence, not sufficient. A column header is a
# LABEL, and a label is not content. These rules key on SHAPE ONLY — never on
# subject-matter vocabulary (owner doctrine: gates stay domain-agnostic, wide-net)
# — and they REJECT the whole reconstruction rather than fabricate a different
# header or ship a headerless grid.

# L2 — a SINGLE trailing period ends a sentence. An ellipsis ("I can...", "The
# result is...") is a real label idiom, and ``!`` / ``?`` / ``:`` are all fine
# ("No—I don't get it!", "Say:"), so only a lone final ``.`` is disqualifying.
_TERMINAL_PERIOD_RE = re.compile(r"(?<!\.)\.\Z")


def _is_numeric_label(cell: str) -> bool:
    """Whether ``cell`` is purely numeric / a numeric list (L4).

    Shape-only: carries at least one digit and NO letter at all, so ``"1, 2, 3,
    4,..."`` and ``"42"`` are numeric while ``"Greg's age"`` / ``"Length"`` are
    not. A label is a word, not a number list.
    """
    return (
        bool(cell)
        and any(ch.isdigit() for ch in cell)
        and not any(ch.isalpha() for ch in cell)
    )


def header_row_is_label_shaped(
    cells: Sequence[str], *, corner_exempt: bool
) -> bool:
    """Whether the declared header row is shaped like a row of column LABELS.

    The admissibility gate that stops us promoting a worked-example STEP ROW to a
    column header just because the VLM's separator tic landed after it.

    **Applied to the WEAK-EVIDENCE arm only.** Evidence strength decides how much
    corroboration we demand: the ``scope="row"`` stub/matrix arm rests on the
    EMPTY CORNER CELL — strong, structural, self-standing evidence that row 0
    labels the columns — and :func:`parse_table_topology` therefore does NOT run
    this gate on it (its header row is legitimately math-bearing / numeric: a
    worked example's problem statement, a percent axis). The ``scope="col"`` arm
    rests on nothing but the separator tic, which is unreliable, so it MUST be
    corroborated by label shape. ``corner_exempt`` keeps the L3 exemption for a
    caller that wants the gate WITH the corner allowance; the production wiring
    passes ``False`` (the stub arm bypasses the gate entirely).

    Rejects the row (returns ``False``) if ANY cell trips one of four shape rules:

    * **L1 math** — the cell carries math in EITHER representation: a delimited
      LaTeX span (``$…$`` / ``\\(…\\)`` / ``\\[…\\]`` / ``$$…$$``, via
      ``structure_emit._MATH_SPAN_RE``) **or** a unicode math symbol (``·`` ``×``
      ``÷`` ``√`` ``≠`` ``≥`` ``±`` ``∑`` ``π``, super/subscript digits …, via
      ``math_fold._UNICODE_RE`` — the codebase's canonical unicode-math map,
      REUSED, never re-derived). A column label is never an equation, so
      ``"Substitute $-8$ for $x$" | "$-x$"`` is a step row, not a header.

      Both representations are required because the scan lane emits BOTH: the
      measured ``"9 weeks" | "9 wk · 7 days · 24 hr · 60 min"`` step table
      (body: "Write 1 as…", "Divide out the common units.", "Multiply.") slipped
      through a LaTeX-only L1 and shipped a false ``<th>`` on its PROBLEM
      STATEMENT. A step table is a step table whether its math is ``$…$`` or
      ``·`` — the representation is not the point, the math is.
    * **L2 sentence** — the cell ends with a SINGLE terminal period. A label is a
      noun phrase; ``"Simplify inside the parentheses."`` / ``"Translate."`` /
      ``"Step 1. Read the problem."`` are instructions. An ellipsis (``"I
      can..."``, ``"The result is..."``) and ``!`` / ``?`` / ``:`` are NOT
      terminal — those are real labels in this corpus.
    * **L3 empty** — the cell is empty / whitespace. EXCEPT the corner cell
      (index 0) when ``corner_exempt`` — an empty corner is the stub/matrix arm's
      OWN evidence, not a defect.
    * **L4 numeric** — the cell is purely numeric or a numeric list
      (:func:`_is_numeric_label`).

    NO subject-matter vocabulary is consulted, by doctrine. That leaves one known
    residual: a mnemonic table whose first data row is two ordinary nouns
    (``"Parentheses" | "Please"``) is shape-indistinguishable from a legitimate
    two-label header (``"Expression" | "Words"``) and is NOT caught here.
    """
    if not cells:
        return False
    for idx, raw in enumerate(cells):
        cell = (raw or "").strip()
        if not cell:
            if idx == 0 and corner_exempt:
                continue  # the stub/matrix corner cell — its own evidence
            return False  # L3
        if _MATH_SPAN_RE.search(cell) or _UNICODE_RE.search(cell):
            return False  # L1
        if _TERMINAL_PERIOD_RE.search(cell):
            return False  # L2
        if _is_numeric_label(cell):
            return False  # L4
    return True


def _split_rows(
    tokens: List[str], n_cols: int
) -> Optional[List[Tuple[str, ...]]]:
    """Group ``tokens`` into rows of EXACTLY ``n_cols`` cells, or ``None``.

    The load-bearing arithmetic. A run of ``R`` rows tokenizes to exactly
    ``R*n_cols + (R-1)`` tokens: ``n_cols`` cell tokens per row, plus one JOIN
    ARTIFACT token per row boundary (the empty/whitespace token produced where
    row ``k``'s closing ``|`` meets row ``k+1``'s opening ``|``). Solving for
    ``R`` gives ``R = (n + 1) / (n_cols + 1)``, so a token count that does not
    divide evenly cannot be a clean grid → ``None`` (fail closed).

    This is what disambiguates an EMPTY CELL from a ROW JOIN — both are
    whitespace-only tokens, and only POSITION (not content) can tell them apart.
    Each inter-row token is additionally required to be empty/whitespace; a
    non-empty token where a join must be is a shape we do not model → ``None``.
    """
    n = len(tokens)
    if n_cols < 1 or n < n_cols:
        return None
    if (n + 1) % (n_cols + 1) != 0:
        return None
    n_rows = (n + 1) // (n_cols + 1)
    if n_rows < 1:
        return None
    rows: List[Tuple[str, ...]] = []
    i = 0
    for r in range(n_rows):
        rows.append(tuple(t.strip() for t in tokens[i : i + n_cols]))
        i += n_cols
        if r < n_rows - 1:
            if tokens[i].strip() != "":
                return None  # a join artifact is empty by construction
            i += 1
    return rows


def _region_tokens(masked_region: str) -> Optional[List[str]]:
    """Split a pipe region into cell/join tokens, dropping the OUTERMOST edges.

    ``_PIPE_REGION_RE`` both opens and closes on a ``|``, so ``split("|")``
    yields an empty artifact at each end — those two are the outermost pipes'
    edges, not cells, and are the ONLY tokens dropped here. Everything else
    (including whitespace-only tokens) survives for the arithmetic to classify.
    """
    raw = masked_region.split("|")
    if len(raw) < 3 or raw[0] != "" or raw[-1] != "":
        return None
    return raw[1:-1]


def _source_cells(text: str) -> Optional[Counter]:
    """Multiset of the NON-EMPTY, NON-SEPARATOR cell texts in ``text``'s grid.

    The conservation ORACLE — re-derived straight from the source string (not
    from the topology), so it is an independent witness of what the grid
    contained. Separator cells (``---``) are apparatus, not content; whitespace-
    only tokens are empty cells / row joins and carry no text to conserve.
    """
    if not text or "|" not in text:
        return None
    masked, spans = _mask_math(text)
    best = None
    for m in _PIPE_REGION_RE.finditer(masked):
        if best is None or (m.end() - m.start()) > (best.end() - best.start()):
            best = m
    if best is None:
        return None
    tokens = _region_tokens(masked[best.start() : best.end()])
    if tokens is None:
        return None
    cells: List[str] = []
    for tok in tokens:
        stripped = tok.strip()
        if not stripped or _SEP_CELL_RE.fullmatch(stripped):
            continue
        cells.append(_unmask(stripped, spans))
    return Counter(cells)


def verify_text_conservation(topo: TableTopology, source_text: str) -> bool:
    """Whether ``topo`` conserves every non-empty cell text of ``source_text``.

    HARD REQUIREMENT of this lane: a reconstruction may re-shape a grid, never
    lose or duplicate a cell's text. Compares the multiset of non-empty cell
    texts in the topology against :func:`_source_cells`' independent tokenization
    of the source pipe region. Any drift (a dropped empty-cell-shredded row, a
    duplicated header, a mis-sliced token) → ``False``, and
    :func:`parse_table_topology` refuses the topology.
    """
    src = _source_cells(source_text)
    if src is None:
        return False
    got = Counter(
        cell
        for row in (tuple(topo.header_rows) + tuple(topo.body_rows))
        for cell in row
        if cell
    )
    return got == src


def parse_table_topology(text: str) -> Optional[TableTopology]:
    r"""Reconstruct the declared cell topology of a markdown pipe grid in ``text``.

    The algorithm (deterministic, self-verifying — no LLM):

    1. Mask delimited math spans first (``structure_emit._mask_math``, EXACTLY as
       ``parse_table`` does) so a mid-math pipe (``$a|b$``) never trips the
       tokenizer, then take the LONGEST contiguous pipe region.
    2. Find the ``| --- | --- |`` separator row. Its dash-cell count DECLARES
       ``n_cols``. **No separator → ``None``** (no header evidence; we never
       assume row 0 is a header).
    3. Tokenize the region on ``|``, dropping only the outermost pipes' empty
       edge artifacts.
    4. Split the tokens BEFORE the separator into rows of ``n_cols`` (the single
       row there IS the header row) and those AFTER it into body rows of
       ``n_cols`` — resolving the ``| |`` empty-cell-vs-row-join ambiguity by the
       ``R*n_cols + (R-1)`` arithmetic in :func:`_split_rows`. A whitespace token
       immediately adjacent to the separator on either side is the join artifact
       between the separator row and its neighbour, and is dropped ONLY when the
       arithmetic does not otherwise fit (so a genuinely EMPTY edge cell wins
       when it can).
    5. **Fail closed** on anything that does not fit exactly, and finally on a
       text-conservation failure (:func:`verify_text_conservation`).

    ``row_headers`` is set ONLY on real evidence: the header row's FIRST cell is
    EMPTY (the classic corner cell of a matrix / stub table) AND there are ≥2
    body rows AND every body row's first cell is non-empty. Nothing else.
    """
    if not text or "|" not in text:
        return None
    masked, spans = _mask_math(text)

    best = None
    for m in _PIPE_REGION_RE.finditer(masked):
        if best is None or (m.end() - m.start()) > (best.end() - best.start()):
            best = m
    if best is None:
        return None
    region = masked[best.start() : best.end()]
    pre = _unmask(_fragment(masked[: best.start()]), spans)
    post = _unmask(_fragment(masked[best.end() :]), spans)

    if not _MD_SEP_ROW_RE.search(region):
        return None  # no topology declaration → no header evidence → decline

    tokens = _region_tokens(region)
    if tokens is None:
        return None

    sep_positions = [
        i for i, tok in enumerate(tokens) if _SEP_CELL_RE.fullmatch(tok.strip())
    ]
    if not sep_positions:
        return None
    start = sep_positions[0]
    end = start
    while end + 1 < len(tokens) and _SEP_CELL_RE.fullmatch(tokens[end + 1].strip()):
        end += 1
    # A SECOND separator run is a shape this lane does not model (markdown has
    # exactly one) → fail closed rather than guess which one declares the grid.
    if any(pos > end for pos in sep_positions):
        return None
    n_cols = end - start + 1
    if n_cols < 2:
        return None

    head = tokens[:start]
    tail = tokens[end + 1 :]

    head_rows = _split_rows(head, n_cols)
    if head_rows is None and head and head[-1].strip() == "":
        # The trailing token is the join between the header row's closing pipe
        # and the separator's opening pipe ("… get it! | |----|"), not a cell.
        head_rows = _split_rows(head[:-1], n_cols)
    if head_rows is None or len(head_rows) != 1:
        return None  # markdown declares EXACTLY one header row above the separator

    body_rows = _split_rows(tail, n_cols)
    if body_rows is None and tail and tail[0].strip() == "":
        # Mirror image: the leading token is the join between the separator's
        # closing pipe and body row 1's opening pipe.
        body_rows = _split_rows(tail[1:], n_cols)
    if not body_rows:
        return None

    header = tuple(tuple(_unmask(c, spans) for c in row) for row in head_rows)
    body = tuple(tuple(_unmask(c, spans) for c in row) for row in body_rows)

    # EVIDENCE STRENGTH DECIDES HOW MUCH CORROBORATION WE DEMAND. The two arms
    # are INDEPENDENT and rest on different evidence:
    #
    #   * scope="row" (stub/matrix arm) — the evidence is the EMPTY CORNER CELL,
    #     with >= 2 body rows and a fully-populated stub column. That is strong,
    #     STRUCTURAL evidence — a grid only has an empty corner because row 0
    #     labels the columns and column 0 labels the rows. It stands ALONE and
    #     needs no label-shape corroboration. (Measured 11/11 correct.) Crucially,
    #     these header rows are often the worked example's PROBLEM STATEMENT
    #     ("" | "the product of $-2$ and $14$") or a percent axis ("" | "6%"),
    #     which are math-bearing / numeric BY NATURE — running L1/L4 on them would
    #     destroy the arm while telling us nothing about the (already proven)
    #     header-ness of row 0.
    #
    #   * scope="col" (plain col-header arm) — the ONLY evidence is the VLM's
    #     separator tic, which we measured to be unreliable (it lands after the
    #     first STEP ROW of a worked example ~65% of the time). Weak evidence must
    #     be CORROBORATED, so the declared header row must also be label-shaped.
    row_headers = (
        header[0][0] == ""
        and len(body) >= 2
        and all(row[0] != "" for row in body)
    )

    # ADMISSIBILITY (weak-evidence arm only): the separator merely PROPOSES a
    # header row; the shape of that row DECIDES. A declared header that is really
    # a data/step row is refused OUTRIGHT — we neither fabricate a different
    # header nor demote it to a body row and ship a headerless grid. Returning
    # None means ``adapter._emit_structured_bodies`` falls through to today's
    # output byte-identically (fail closed = fall back, never ship broken).
    if not row_headers and not header_row_is_label_shaped(
        header[0], corner_exempt=False
    ):
        return None

    topo = TableTopology(
        header_rows=header,
        body_rows=body,
        n_cols=n_cols,
        row_headers=row_headers,
        pre=pre,
        post=post,
    )
    if not verify_text_conservation(topo, text):
        return None
    return topo


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _render_cell(text: str) -> str:
    """Escape a cell's non-math text and restore its math spans VERBATIM.

    Re-masks then delegates to ``structure_emit._cell`` (the same escape+restore
    the cascade's own table/list emitters use), so ``$a<b$`` keeps its math
    intact while surrounding prose is HTML-escaped.
    """
    masked, spans = _mask_math(text)
    return _cell(masked, spans)


def render_topology_html(
    topo: TableTopology, *, caption: Optional[str] = None
) -> str:
    """Render a :class:`TableTopology` as accessible ``<table>`` markup.

    Emits an optional ``<caption>``, a ``<thead>`` of ``<th scope="col">`` cells,
    and a ``<tbody>`` whose first column is ``<th scope="row">`` when — and only
    when — ``topo.row_headers`` says the stub-column evidence is there (else
    ``<td>``). EMPTY cells are preserved as empty ``<td>`` (a blank response cell
    in a checklist IS the grid). The ``pre`` / ``post`` prose fragments render as
    ``<p>`` siblings, mirroring ``structure_emit.parse_table``, so wrapping a
    block's body in this markup never drops text.

    **No spans, ever.** A markdown pipe grid cannot express ``colspan`` /
    ``rowspan``; we derive none, so we never fabricate a span attribute.
    """
    parts: List[str] = []
    if topo.pre:
        parts.append(f"<p>{_render_cell(topo.pre)}</p>")
    parts.append("<table>")
    if caption:
        parts.append(f"<caption>{_render_cell(caption)}</caption>")
    if topo.header_rows:
        parts.append("<thead>")
        for row in topo.header_rows:
            parts.append("<tr>")
            parts.extend(f'<th scope="col">{_render_cell(c)}</th>' for c in row)
            parts.append("</tr>")
        parts.append("</thead>")
    parts.append("<tbody>")
    for row in topo.body_rows:
        parts.append("<tr>")
        for idx, cell in enumerate(row):
            if idx == 0 and topo.row_headers:
                parts.append(f'<th scope="row">{_render_cell(cell)}</th>')
            else:
                parts.append(f"<td>{_render_cell(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if topo.post:
        parts.append(f"<p>{_render_cell(topo.post)}</p>")
    return "".join(parts)


def emit_structured_table(
    text: str, *, caption: Optional[str] = None
) -> Optional[str]:
    """Public entry: separator-bearing pipe text → accessible ``<table>`` HTML.

    The full chain:

    1. :func:`parse_table_topology` — the declared topology (this itself enforces
       the ``R*n_cols + (R-1)`` arithmetic AND :func:`verify_text_conservation`).
    2. :func:`render_topology_html` — the markup.
    3. :func:`verify_text_conservation` again, against the FINAL topology — the
       hard contract, re-asserted at the emit seam.
    4. :func:`_repair_h43` — the gate module's OWN ``apply_h43`` grid arithmetic
       adds ``id`` / ``headers`` association to a COMPLEX (dual-axis stub/matrix)
       table so it can actually ship its ``<th scope="row">``. A no-op on the
       simple col-headers-only table (byte-identical). If the repair is
       unavailable or raises, we fall back to the UNrepaired html and still run
       the gate — the repair must never become a new failure mode. But if the
       repair ALTERED any cell TEXT (it must only add attributes), that breaks
       the conservation contract → **fail closed**, return ``None``.
    5. :func:`_h43_accept` — the reused accept gate has the last word.

    Returns ``None`` at ANY refusal, so the caller
    (``adapter._emit_structured_bodies``) falls through to today's
    ``structure_emit.emit_structure`` path with byte-identical behaviour.
    """
    topo = parse_table_topology(text)
    if topo is None:
        return None
    if not verify_text_conservation(topo, text):  # belt-and-braces at the seam
        return None
    html = render_topology_html(topo, caption=caption)

    repaired = _repair_h43(html)
    if repaired is not None:
        if _tag_stripped_text(repaired) != _tag_stripped_text(html):
            # apply_h43 may add id=/headers= ATTRIBUTES only. Any drift in cell
            # TEXT (or a lost/added cell) breaks text conservation — the hard
            # contract of this lane — so refuse rather than ship the repair.
            return None
        html = repaired

    if not _h43_accept(html):
        return None
    return html
