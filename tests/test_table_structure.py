r"""Tests for the ``SEMANTIK_TABLE_STRUCTURE`` cell-topology lane.

Covers the contract stated in ``lib/semantik/table_structure.py``:

* the REAL scan-lane checklist grid (empty response cells) parses to
  4 cols x 1 header row + 3 body rows, with EMPTY cells PRESERVED
* ``<th scope="col">`` is emitted for the header row (the thing production
  ships zero of today)
* ``<th scope="row">`` fires ONLY on the empty-corner-cell stub/matrix shape
* no separator row -> ``None`` (row 0 is NEVER assumed to be a header)
* a token count that does not fit ``R*n_cols + (R-1)`` -> ``None`` (fail closed)
* text conservation rejects a topology whose cell text drifted
* the H43 accept gate is genuinely consulted (monkeypatched reject -> ``None``)
* flag OFF is byte-identical (``sanitize_body_latex`` unchanged by the additive
  ``keep_md_sep`` kwarg; the adapter stashes nothing and emits nothing)
* a math span carrying a pipe (``$a|b$``) never trips the tokenizer
"""
from __future__ import annotations

import pytest

from lib.semantik import table_structure as ts
from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _emit_structured_bodies,
    _resolve_table_structure,
    _sanitize_block_body_latex,
)
from lib.semantik.math_fold import sanitize_body_latex, wrap_bare_math
from lib.semantik.table_structure import (
    TableTopology,
    emit_structured_table,
    has_separator_row,
    header_row_is_label_shaped,
    parse_table_topology,
    render_topology_html,
    verify_text_conservation,
)

# The real defect source text (a self-assessment checklist whose response cells
# are BLANK): today ``parse_table`` renders this as one 4-cell row followed by a
# run of 1-cell rows, because ``| |`` — an EMPTY CELL — both shreds its
# ``\|\s*\|`` row split and is then dropped by its ``c != ""`` filter.
CHECKLIST = (
    "| I can... | Confidently | With some help | No-I don't get it! | "
    "|----------|-------------|----------------|-------------------| "
    "| use place value with whole numbers. | | | | "
    "| identify multiples and apply divisibility tests. | | | | "
    "| find prime factorizations. | | | |"
)

# A stub / matrix table: the header row's FIRST cell is the EMPTY corner cell.
MATRIX = "|  | Mon | Tue | |---|---|---| | Alice | 3 | 4 | | Bob | 5 | 6 |"


def _sanitized(raw: str) -> str:
    """The exact ``table_src`` the adapter stashes for ``raw``."""
    return wrap_bare_math(
        sanitize_body_latex(raw, html=False, keep_md_sep=True), html=False
    )


# ---------------------------------------------------------------------------
# (a) The real checklist example: 4 cols, 1 header row, 3 body rows, empties kept.
# ---------------------------------------------------------------------------


def test_checklist_parses_to_declared_topology_with_empty_cells_preserved():
    topo = parse_table_topology(_sanitized(CHECKLIST))

    assert topo is not None
    assert topo.n_cols == 4
    assert topo.header_rows == (
        ("I can...", "Confidently", "With some help", "No-I don't get it!"),
    )
    assert topo.body_rows == (
        ("use place value with whole numbers.", "", "", ""),
        ("identify multiples and apply divisibility tests.", "", "", ""),
        ("find prime factorizations.", "", "", ""),
    )
    # Every body row is exactly n_cols wide — the ragged 4-then-1-cell shredding
    # is gone, and the blank response cells survive as real grid positions.
    assert all(len(row) == topo.n_cols for row in topo.body_rows)


def test_checklist_empty_cells_render_as_empty_td():
    html = render_topology_html(parse_table_topology(_sanitized(CHECKLIST)))

    assert html.count("<tr>") == 4  # 1 header + 3 body
    assert html.count("<td></td>") == 9  # 3 body rows x 3 blank response cells


# ---------------------------------------------------------------------------
# (b) <th scope="col"> for the header row — the whole point of the lane.
# ---------------------------------------------------------------------------


def test_header_row_emits_th_scope_col():
    html = render_topology_html(parse_table_topology(_sanitized(CHECKLIST)))

    assert html.count('<th scope="col">') == 4
    assert "<thead><tr>" in html
    assert '<th scope="col">Confidently</th>' in html


def test_emit_structured_table_ships_a_thead_through_the_gate():
    html = emit_structured_table(_sanitized(CHECKLIST))

    assert html is not None
    assert '<th scope="col">I can...</th>' in html
    assert html.startswith("<table><thead>")


def test_no_span_attributes_are_ever_fabricated():
    # A markdown pipe grid cannot express colspan/rowspan; we derive none, so we
    # must never emit one.
    html = render_topology_html(parse_table_topology(_sanitized(CHECKLIST)))

    assert "colspan" not in html
    assert "rowspan" not in html


# ---------------------------------------------------------------------------
# (c) Row headers — evidence-based ONLY (empty corner cell + >=2 body rows).
# ---------------------------------------------------------------------------


def test_row_header_rule_fires_on_empty_corner_cell_matrix():
    topo = parse_table_topology(MATRIX)

    assert topo is not None
    assert topo.row_headers is True

    html = render_topology_html(topo)
    assert '<th scope="row">Alice</th>' in html
    assert '<th scope="row">Bob</th>' in html
    assert html.count('<th scope="row">') == 2


def test_matrix_table_round_trips_through_emit_with_h43_association():
    # A dual-axis stub/matrix table (<thead> of <th scope="col"> PLUS a
    # <th scope="row"> stub column) is COMPLEX by H43's definition, so `scope`
    # alone is ambiguous and the gate demands explicit id/headers association.
    # apply_h43 (the gate module's OWN grid arithmetic) supplies it, so the table
    # can actually ship instead of being rejected by its own gate.
    html = emit_structured_table(MATRIX)

    assert html is not None
    assert '<th scope="row"' in html
    assert '<th scope="col"' in html

    module = ts._load_h43()
    info = module.analyze_table(html)
    assert info.is_complex is True
    passed, message, _details = module.verify_h43(html)
    assert passed is True, message

    # Every <th> carries an id, and every governed <td> resolves to those ids.
    assert 'id="t-r1c0"' in html  # the "Alice" row header
    assert 'headers="t-r0c1 t-r1c0"' in html  # Mon x Alice
    assert ts._h43_accept(html) is True


def test_apply_h43_preserves_cell_text_exactly():
    # apply_h43 may add id=/headers= ATTRIBUTES and nothing else — the text
    # conservation contract must survive the repair.
    topo = parse_table_topology(MATRIX)
    plain = render_topology_html(topo)
    repaired = ts._repair_h43(plain)

    assert repaired is not None
    assert repaired != plain  # attributes WERE added
    assert ts._tag_stripped_text(repaired) == ts._tag_stripped_text(plain)
    assert verify_text_conservation(topo, MATRIX) is True

    # Every source cell text still reaches the emitted markup.
    for cell in ("Mon", "Tue", "Alice", "Bob", "3", "4", "5", "6"):
        assert f">{cell}<" in emit_structured_table(MATRIX)


def test_apply_h43_is_a_no_op_on_the_simple_col_headers_only_table(monkeypatch):
    # The simple (non-complex) path must be byte-identical with and without the
    # repair step — apply_h43 documents itself as a no-op there.
    src = _sanitized(CHECKLIST)
    plain = render_topology_html(parse_table_topology(src))

    assert ts._repair_h43(plain) == plain  # byte-identical
    with_repair = emit_structured_table(src)

    monkeypatch.setattr(ts, "_repair_h43", lambda html: None)  # skip the repair
    without_repair = emit_structured_table(src)

    assert with_repair == without_repair == plain
    assert "headers=" not in with_repair
    assert "id=" not in with_repair


def test_emit_fails_closed_when_the_repair_alters_cell_text(monkeypatch):
    # If apply_h43 ever touched cell TEXT it would break the hard conservation
    # contract — refuse the emit rather than ship the drift.
    monkeypatch.setattr(
        ts, "_repair_h43", lambda html: html.replace("Confidently", "CONFIDENTLY")
    )

    assert emit_structured_table(_sanitized(CHECKLIST)) is None


def test_emit_falls_back_to_unrepaired_html_when_the_repair_is_unavailable(monkeypatch):
    # A repair that is unavailable / raises must never become a NEW failure mode:
    # fall back to the unrepaired html and still run the accept gate on it.
    monkeypatch.setattr(ts, "_repair_h43", lambda html: None)
    src = _sanitized(CHECKLIST)

    html = emit_structured_table(src)
    assert html is not None  # the simple table still passes the gate
    assert html == render_topology_html(parse_table_topology(src))


def test_row_header_rule_does_not_fire_without_an_empty_corner_cell():
    # The checklist's corner cell is "I can..." — NOT empty — so the first column
    # is data, not a stub. No <th scope="row"> may be invented.
    topo = parse_table_topology(_sanitized(CHECKLIST))

    assert topo.row_headers is False
    assert 'scope="row"' not in render_topology_html(topo)


def test_empty_corner_without_stub_evidence_is_inadmissible():
    # Empty corner cell but only ONE body row: too little evidence for a stub
    # column (the rule requires >= 2 body rows). Without the corner exemption the
    # empty header cell trips L3, so the whole reconstruction is refused — we do
    # NOT ship a header row with a blank <th>.
    assert parse_table_topology("|  | Mon | Tue | |---|---|---| | Alice | 3 | 4 |") is None


def test_empty_corner_with_a_stubless_body_row_is_inadmissible():
    # Empty corner cell, 2 body rows, but row 2's first cell is EMPTY — a stub
    # column with a missing label is not a stub column, so the corner exemption
    # does not apply and L3 refuses the row.
    assert (
        parse_table_topology(
            "|  | Mon | Tue | |---|---|---| | Alice | 3 | 4 | | | 5 | 6 |"
        )
        is None
    )


# ---------------------------------------------------------------------------
# Header-label admissibility (L1-L4) — the VLM PROPOSES, we DECIDE.
# ---------------------------------------------------------------------------
# The separator row is a FORMATTING TIC, not a topology judgment: on worked-
# example step tables the VLM drops its |---| after the FIRST STEP ROW, so
# accepting it uncritically promotes a DATA row to a column header. A <th> in the
# wrong place actively misleads a screen-reader user — worse than no <th>.


@pytest.mark.parametrize(
    "cells,rule",
    [
        # L1 — a column label is never an equation.
        (["Simplify inside the parentheses.", "$-5 + 3(-2 + 7)$"], "L1+L2"),
        (["Translate.", "$13 - (-21)$"], "L1+L2"),
        (["Substitute $-8$ for $x$", "$-x$"], "L1"),
        (["Result", r"\(x + 1\)"], "L1"),
        (["Result", r"\[x + 1\]"], "L1"),
        (["Result", "$$x + 1$$"], "L1"),
        # L2 — a label is a noun phrase, not an instruction/sentence.
        (["Simplify inside the parentheses.", "Expression"], "L2"),
        (["Step 1. Read the problem.", "Notes"], "L2"),
        # L3 — an empty header cell with no stub evidence.
        (["Expression", ""], "L3"),
        (["", "Expression"], "L3"),
        # L4 — a label is a word, not a number list.
        (["Term", "1, 2, 3, 4,..."], "L4"),
        (["Term", "42"], "L4"),
    ],
)
def test_inadmissible_header_rows_are_rejected(cells, rule):
    assert header_row_is_label_shaped(cells, corner_exempt=False) is False, rule


@pytest.mark.parametrize(
    "cells",
    [
        # The REAL true headers from the corpus — these must all survive.
        ["I can...", "Confidently", "With some help", "No—I don't get it!"],
        ["Operation", "Notation", "Say:", "The result is..."],
        ["Expression", "Words", "English Phrase"],
        ["Length", "Mass", "Capacity"],
        ["Greg's age", "Alex's age"],
    ],
)
def test_real_true_headers_survive_the_admissibility_gate(cells):
    assert header_row_is_label_shaped(cells, corner_exempt=False) is True


def test_ellipsis_and_bang_and_colon_are_not_terminal_periods():
    # L2 matches a SINGLE trailing '.' only.
    assert header_row_is_label_shaped(["I can...", "Say:"], corner_exempt=False) is True
    assert (
        header_row_is_label_shaped(["No—I don't get it!", "Why?"], corner_exempt=False)
        is True
    )
    assert header_row_is_label_shaped(["I can.", "Say"], corner_exempt=False) is False


def test_corner_exempt_only_spares_index_zero():
    # The stub/matrix arm's own evidence: an empty CORNER cell is fine, an empty
    # cell anywhere else is not.
    assert header_row_is_label_shaped(["", "Mon", "Tue"], corner_exempt=True) is True
    assert header_row_is_label_shaped(["", "Mon", ""], corner_exempt=True) is False
    assert header_row_is_label_shaped(["", "Mon", "Tue"], corner_exempt=False) is False


def test_corner_exempt_covers_l3_only_at_the_function_level():
    # The function's own contract: corner_exempt waives L3 on the corner cell ONLY,
    # not L1/L2/L4 on the other cells. NOTE the production wiring never relies on
    # this — the stub/matrix arm BYPASSES the gate entirely (see
    # test_stub_matrix_arm_bypasses_the_label_gate_on_a_math_header), because its
    # empty-corner evidence is structural and self-standing.
    assert header_row_is_label_shaped(["", "$x + 1$"], corner_exempt=True) is False
    assert header_row_is_label_shaped(["", "Translate."], corner_exempt=True) is False
    assert header_row_is_label_shaped(["", "1, 2, 3"], corner_exempt=True) is False


# ---------------------------------------------------------------------------
# The two arms are INDEPENDENT and rest on different evidence.
# ---------------------------------------------------------------------------
# scope="row" — evidence is the EMPTY CORNER CELL: strong, structural, stands
#   alone. Its header row is legitimately math-bearing (a worked example's problem
#   statement) or numeric (a percent axis), so the label gate must NOT run on it.
# scope="col" — evidence is ONLY the unreliable separator tic, so it must be
#   corroborated by label shape.


def test_stub_matrix_arm_bypasses_the_label_gate_on_a_math_header():
    # The exact regression: the stub/matrix header row IS the worked example's
    # problem statement, so it is math-bearing. L1 must not be allowed to kill it —
    # the empty corner cell already proved row 0 is a header.
    src = (
        "|  | the product of $-2$ and $14$ | "
        "|---|---| "
        "| Translate. | $(-2)(14)$ | "
        "| Simplify. | $-28$ |"
    )
    topo = parse_table_topology(src)

    assert topo is not None
    assert topo.row_headers is True
    # The header row would FAIL the label gate on its own (L1) — and that is fine.
    assert header_row_is_label_shaped(topo.header_rows[0], corner_exempt=True) is False

    html = emit_structured_table(src)
    assert html is not None
    assert '<th scope="row"' in html
    assert ">Translate.</th>" in html
    assert ">Simplify.</th>" in html
    assert ">the product of $-2$ and $14$</th>" in html  # math preserved verbatim

    module = ts._load_h43()
    passed, message, _details = module.verify_h43(html)
    assert passed is True, message


def test_percent_matrix_emits_both_scope_col_and_scope_row():
    # A numeric header axis (L4 would reject it) — but the empty corner cell is
    # structural evidence, so the stub arm stands and BOTH axes get their <th>.
    src = (
        "|  | 6% | 78% | 135% | "
        "|---|---|---|---| "
        "| Decimal | 0.06 | 0.78 | 1.35 | "
        "| Fraction | 3/50 | 39/50 | 27/20 |"
    )
    topo = parse_table_topology(src)

    assert topo is not None
    assert topo.row_headers is True

    html = emit_structured_table(src)
    assert html is not None
    assert '<th scope="col"' in html
    assert '<th scope="row"' in html
    assert ">6%</th>" in html
    assert ">Decimal</th>" in html

    module = ts._load_h43()
    passed, message, _details = module.verify_h43(html)
    assert passed is True, message


def test_l1_catches_unicode_math_not_just_latex():
    # The scan lane emits math in BOTH representations. A LaTeX-only L1 let this
    # REAL step table through and shipped a false <th> on its problem statement:
    # the header row's math is unicode (·, /), not $…$.
    assert (
        header_row_is_label_shaped(
            ["9 weeks", "9 wk · 7 days · 24 hr · 60 min"], corner_exempt=False
        )
        is False
    )
    # …and the other unicode math symbols the map covers.
    for cell in ["a × b", "a ÷ b", "√2", "a ≠ b", "x ≥ 1", "±5", "∑ x", "2π", "x²"]:
        assert header_row_is_label_shaped(["Label", cell], corner_exempt=False) is False, cell


def test_unicode_math_l1_does_not_bite_the_real_true_headers():
    # The measured-correct col-arm headers carry no unicode math — an em dash,
    # a curly apostrophe and a percent sign are NOT math symbols.
    for cells in (
        ["I can...", "Confidently", "With some help", "No—I don't get it!"],
        ["Greg’s age", "Alex’s age"],
        ["Operation", "Notation", "Say:", "The result is..."],
        ["Expression", "Words", "English Phrase"],
        ["Length", "Mass", "Capacity"],
        ["Equation", "English Sentence"],
        ["Operation", "Phrase", "Expression"],
    ):
        assert header_row_is_label_shaped(cells, corner_exempt=False) is True, cells


def test_real_unicode_math_step_table_falls_back_end_to_end():
    # The exact production source text (a unit-conversion worked example whose
    # separator landed after the PROBLEM row). Must fall back, not ship a <th>.
    src = (
        "| 9 weeks | 9 wk · 7 days · 24 hr · 60 min | "
        "| --- | --- | "
        "| Write 1 as 7 days / 1 week, 24 hours / 1 day, and 60 minutes / 1 hour. "
        "| 9 wk · 7 days · 24 hr · 60 min / 1 · 1 wk · 1 day · 1 hr | "
        "| Divide out the common units. | 9 wk · 7 days · 24 hr / 1 · 1 wk · 1 day | "
        "| Multiply. | 90,720 min |"
    )

    assert parse_table_topology(src) is None
    assert emit_structured_table(src) is None


def test_step_table_without_an_empty_corner_stays_rejected():
    # The false-positive class must stay DEAD: no empty corner => no structural
    # evidence => the weak separator tic must be corroborated by label shape, and
    # a step row is not label-shaped.
    src = (
        "| Simplify inside the parentheses. | $-5 + 3(-2 + 7)$ | "
        "|---|---| "
        "| Multiply. | $-5 + 3(5)$ | "
        "| Add. | $-5 + 15$ |"
    )

    assert parse_table_topology(src) is None
    assert emit_structured_table(src) is None


def test_step_table_whose_separator_landed_after_the_first_step_falls_back():
    # The measured production defect, end to end: the VLM dropped its |---| after
    # step 1, so the declared "header" is really a step row. We must NOT emit a
    # <th> for it — and we must NOT ship a headerless <table> either. Fall back.
    src = (
        "| Simplify inside the parentheses. | $-5 + 3(-2 + 7)$ | "
        "|---|---| "
        "| Multiply. | $-5 + 3(5)$ | "
        "| Add. | $-5 + 15$ |"
    )

    assert parse_table_topology(src) is None
    assert emit_structured_table(src) is None


def test_a_rejected_header_never_ships_a_headerless_table():
    src = "| Translate. | $13 - (-21)$ | |---|---| | Simplify. | $34$ |"
    html = emit_structured_table(src)

    assert html is None  # fall back to today's output, never a <table> with no <th>


def test_stub_matrix_step_table_still_emits_th_scope_row_and_passes_h43():
    # The 11 measured-correct stub/matrix tables: empty corner, step labels down
    # the stub column, math in the data column. The corner exemption keeps them
    # admissible, and they still round-trip through the H43 gate.
    src = (
        "|  | Expression | "
        "|---|---| "
        "| Step 1 | $-5 + 3(-2 + 7)$ | "
        "| Step 2 | $-5 + 3(5)$ | "
        "| Step 3 | $-5 + 15$ |"
    )
    topo = parse_table_topology(src)

    assert topo is not None
    assert topo.row_headers is True

    # The emitted <th> carries its H43 id=, so match on the scope + text, not on
    # an exact tag string.
    html = emit_structured_table(src)
    assert html is not None
    assert '<th scope="row"' in html
    assert ">Step 1</th>" in html
    assert ">$-5 + 3(-2 + 7)$</td>" in html  # math preserved verbatim in the cell

    module = ts._load_h43()
    passed, message, _details = module.verify_h43(html)
    assert passed is True, message


# ---------------------------------------------------------------------------
# (d) No separator row -> None. Row 0 is NEVER assumed to be a header.
# ---------------------------------------------------------------------------


def test_no_separator_row_returns_none():
    # A perfectly well-formed 2x2 pipe grid — but with NO topology declaration,
    # there is no evidence that row 0 is a header, so the lane declines.
    assert parse_table_topology("| a | b | | c | d |") is None
    assert emit_structured_table("| a | b | | c | d |") is None


def test_prose_with_no_pipes_returns_none():
    assert parse_table_topology("Simplify the expression and check your answer.") is None


def test_has_separator_row_is_the_adapter_precheck():
    assert has_separator_row(CHECKLIST) is True
    assert has_separator_row("| a | b | | c | d |") is False
    assert has_separator_row("") is False


# ---------------------------------------------------------------------------
# (e) Token-count mismatch -> None (fail closed; never ship a guessed grid).
# ---------------------------------------------------------------------------


def test_token_count_mismatch_fails_closed():
    # The separator declares 3 columns, but the body carries 4 cell tokens with
    # no join in sight — the R*n_cols + (R-1) fit fails, so we decline rather
    # than guess which token is a cell and which is a row boundary.
    assert parse_table_topology("| a | b | c | |---|---|---| | 1 | 2 | 3 | 4 |") is None


def test_ragged_body_row_fails_closed():
    # Row 2 has only 2 cells where 3 are declared.
    assert (
        parse_table_topology("| a | b | c | |---|---|---| | 1 | 2 | 3 | | 4 | 5 |")
        is None
    )


def test_single_column_separator_is_declined():
    # A one-column "grid" is not a table shape worth reconstructing.
    assert parse_table_topology("| a | |---| | 1 |") is None


def test_no_body_rows_fails_closed():
    assert parse_table_topology("| a | b | |---|---|") is None


# ---------------------------------------------------------------------------
# (f) Text conservation — a topology whose cell text drifted is rejected.
# ---------------------------------------------------------------------------


def test_verify_text_conservation_accepts_the_faithful_topology():
    src = _sanitized(CHECKLIST)

    assert verify_text_conservation(parse_table_topology(src), src) is True


def test_verify_text_conservation_rejects_drifted_cell_text():
    src = _sanitized(CHECKLIST)
    faithful = parse_table_topology(src)

    # Mutate ONE cell's text: the multiset no longer matches the source grid.
    drifted_rows = list(faithful.body_rows)
    drifted_rows[0] = ("use place value with WHOLE numbers.", "", "", "")
    drifted = TableTopology(
        header_rows=faithful.header_rows,
        body_rows=tuple(drifted_rows),
        n_cols=faithful.n_cols,
        row_headers=faithful.row_headers,
        pre=faithful.pre,
        post=faithful.post,
    )

    assert verify_text_conservation(drifted, src) is False


def test_verify_text_conservation_rejects_a_dropped_cell():
    src = _sanitized(CHECKLIST)
    faithful = parse_table_topology(src)

    lossy = TableTopology(
        header_rows=faithful.header_rows,
        body_rows=faithful.body_rows[:-1],  # a whole row silently lost
        n_cols=faithful.n_cols,
        row_headers=faithful.row_headers,
        pre=faithful.pre,
        post=faithful.post,
    )

    assert verify_text_conservation(lossy, src) is False


# ---------------------------------------------------------------------------
# (g) The H43 accept gate is genuinely consulted.
# ---------------------------------------------------------------------------


def test_emit_refuses_when_the_h43_gate_rejects(monkeypatch):
    src = _sanitized(CHECKLIST)
    assert emit_structured_table(src) is not None  # baseline: the gate accepts

    monkeypatch.setattr(ts, "_h43_accept", lambda html: False)

    assert emit_structured_table(src) is None


def test_emit_refuses_when_the_h43_gate_module_is_unavailable(monkeypatch):
    # An unloadable gate module means NO accept gate — fail closed (fall back to
    # today's output) rather than ship unvalidated markup.
    monkeypatch.setattr(ts, "_load_h43", lambda: None)

    assert ts._h43_accept("<table><thead><tr><th>a</th></tr></thead></table>") is False


def test_the_real_gate_accepts_the_emitted_simple_table():
    html = emit_structured_table(_sanitized(CHECKLIST))

    assert ts._h43_accept(html) is True


# ---------------------------------------------------------------------------
# (h) Flag OFF -> byte-identical.
# ---------------------------------------------------------------------------


def test_resolve_table_structure_defaults_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_TABLE_STRUCTURE", raising=False)
    assert _resolve_table_structure() is False

    for falsey in ("", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("SEMANTIK_TABLE_STRUCTURE", falsey)
        assert _resolve_table_structure() is False

    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv("SEMANTIK_TABLE_STRUCTURE", truthy)
        assert _resolve_table_structure() is True


@pytest.mark.parametrize(
    "text",
    [
        CHECKLIST,
        r"\textbf{Bold} and \textit{italic} with $\frac{1}{2}$ math",
        r"A \begin{tabular}{cc} a & b \end{tabular} block \checkmark",
        "| a | b | |---|---| | 1 | 2 |",
        "plain prose with no markup at all",
        "",
    ],
)
@pytest.mark.parametrize("html", [True, False])
def test_sanitize_body_latex_default_kwarg_is_byte_identical(text, html):
    # The additive kwarg must not perturb the default path in EITHER mode.
    assert sanitize_body_latex(text, html=html) == sanitize_body_latex(
        text, html=html, keep_md_sep=False
    )


def test_keep_md_sep_preserves_only_the_separator_row():
    stripped = sanitize_body_latex(CHECKLIST, html=False)
    kept = sanitize_body_latex(CHECKLIST, html=False, keep_md_sep=True)

    assert "---" not in stripped  # today's behaviour: the declaration is destroyed
    assert "---" in kept
    assert has_separator_row(kept) is True
    assert has_separator_row(stripped) is False


def _one_table_chapter() -> list:
    return [
        _AdapterChapter(
            title="Ch 1",
            blocks=[
                _AdapterBlock(
                    html="<p>x</p>",
                    region_kind="paragraph",
                    raw_block_index=0,
                    raw_text=CHECKLIST,
                )
            ],
        )
    ]


def test_adapter_flag_off_stashes_nothing_and_emits_no_thead(monkeypatch):
    monkeypatch.delenv("SEMANTIK_TABLE_STRUCTURE", raising=False)
    chapters = _one_table_chapter()

    _sanitize_block_body_latex(chapters)
    _emit_structured_bodies(chapters)
    block = chapters[0].blocks[0]

    assert block.table_src is None
    assert "<th" not in (block.html or "")  # today's headerless <td> soup


def test_adapter_flag_on_stashes_table_src_and_emits_a_real_thead(monkeypatch):
    monkeypatch.setenv("SEMANTIK_TABLE_STRUCTURE", "1")
    chapters = _one_table_chapter()

    _sanitize_block_body_latex(chapters)
    block = chapters[0].blocks[0]
    assert block.table_src is not None
    assert has_separator_row(block.table_src) is True
    # raw_text keeps today's EXACT (separator-stripped) treatment — the chunker /
    # sidecar text, and the content-hash sourceIds derived from it, are untouched.
    assert has_separator_row(block.raw_text) is False
    stashed = block.table_src

    _emit_structured_bodies(chapters)
    assert block.block_role == "table"
    assert block.region_kind == "table"
    assert '<th scope="col">Confidently</th>' in block.html
    assert block.html.count("<td></td>") == 9

    # The SECOND _sanitize_block_body_latex call (the post-structure self-balance
    # sweep) must not clobber the stash with post-split text.
    _sanitize_block_body_latex(chapters)
    assert block.table_src == stashed


def test_adapter_flag_on_falls_through_when_the_lane_declines(monkeypatch):
    # A separator-bearing block whose grid does not fit the arithmetic must fall
    # back to today's emit_structure path, not lose its body.
    monkeypatch.setenv("SEMANTIK_TABLE_STRUCTURE", "1")
    chapters = _one_table_chapter()
    chapters[0].blocks[0].raw_text = "| a | b | c | |---|---|---| | 1 | 2 | 3 | 4 |"

    _sanitize_block_body_latex(chapters)
    _emit_structured_bodies(chapters)
    block = chapters[0].blocks[0]

    assert block.table_src is not None  # stashed (it HAS a separator row)
    assert emit_structured_table(block.table_src) is None  # but the lane declines
    assert '<th scope="col">' not in (block.html or "")


# ---------------------------------------------------------------------------
# (i) A math span carrying a pipe never trips the tokenizer.
# ---------------------------------------------------------------------------


def test_math_span_with_a_pipe_does_not_trip_the_tokenizer():
    src = "| set | size | |---|---| | $a|b$ | 2 | | $\\{x | x>0\\}$ | 3 |"
    topo = parse_table_topology(src)

    assert topo is not None
    assert topo.n_cols == 2
    assert topo.body_rows == (("$a|b$", "2"), ("$\\{x | x>0\\}$", "3"))

    html = render_topology_html(topo)
    # The math run is restored VERBATIM inside its cell (never escaped, never
    # split across cells).
    assert "<td>$a|b$</td>" in html
    assert "<td>$\\{x | x>0\\}$</td>" in html


def test_prose_around_the_grid_is_preserved_as_p_siblings():
    src = "Compare the results. | a | b | |---|---| | 1 | 2 | See the notes."
    topo = parse_table_topology(src)

    assert topo is not None
    assert topo.pre == "Compare the results."
    assert topo.post == "See the notes."

    html = render_topology_html(topo)
    assert html.startswith("<p>Compare the results.</p><table>")
    assert html.endswith("</tbody></table><p>See the notes.</p>")


def test_cell_text_is_html_escaped():
    topo = parse_table_topology("| a < b | tag | |---|---| | x & y | <b> | | 1 | 2 |")

    html = render_topology_html(topo)
    assert '<th scope="col">a &lt; b</th>' in html
    assert "<td>x &amp; y</td>" in html
    assert "<td>&lt;b&gt;</td>" in html


def test_caption_is_emitted_when_supplied():
    topo = parse_table_topology(_sanitized(CHECKLIST))

    html = render_topology_html(topo, caption="Self-assessment")
    assert "<table><caption>Self-assessment</caption><thead>" in html
