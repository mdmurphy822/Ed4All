"""Shape-driven structural-body emission (A1/A4/A5) + the declare→deliver
reconciliation through the adapter, plus the wrap_bare_math balancer (B3+) and
the SEMANTIK_EMIT_TOC nav (§4). Synthetic fixtures only — no corpus files.
"""
from __future__ import annotations

import re

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)
from lib.semantik.math_fold import (
    wrap_bare_math,
    _balance_math_delimiters,
    separate_adjacent_math_spans,
)
from lib.semantik.adapter import _scrub_marker_artifacts
from lib.semantik.structure_emit import (
    emit_structure,
    parse_definition_list,
    parse_list,
    parse_table,
)


class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "fast"
        self.lang = "en"


def _render(chapters, **kw):
    return normalize_cascade_to_ed4all(
        _Result(chapters), pdf_stem="synthetic_ch", **kw
    )["html"]


def _block(raw, idx, *, role, kind=None):
    return _AdapterBlock(
        html=f"<p>{raw}</p>",
        region_kind=kind or role,
        raw_block_index=idx,
        raw_text=raw,
        heading_text=None,
        block_role=role,
    )


# ---------------------------------------------------------------------------
# parse_table
# ---------------------------------------------------------------------------
def test_table_with_separator_header():
    src = "Intro. | Number | Root | | --- | --- | | 4 | $\\sqrt{4}$ | | 9 | $\\sqrt{9}$ | tail"
    html = parse_table(src)
    assert html is not None
    assert "<table>" in html and "<thead>" in html
    assert '<th scope="col">Number</th>' in html
    assert "<td>4</td>" in html and "$\\sqrt{4}$" in html
    assert "<p>Intro.</p>" in html and "<p>tail</p>" in html  # prose preserved


def test_table_double_pipe_rows_no_separator():
    src = "| a | b | | c | d |"
    html = parse_table(src)
    assert html is not None and "<tbody>" in html and "<thead>" not in html
    assert html.count("<tr>") == 2


def test_table_rejects_single_pipe_prose():
    assert parse_table("conditional P(x|y) is not a table") is None


# ---------------------------------------------------------------------------
# parse_definition_list
# ---------------------------------------------------------------------------
def test_dl_colon_form():
    src = (
        "contradiction: An equation false for all values. "
        "identity: An equation true for any value."
    )
    html = parse_definition_list(src)
    assert html is not None
    assert "<dt>contradiction</dt>" in html
    assert "<dd>An equation false for all values.</dd>" in html
    assert "<dt>identity</dt>" in html


def test_dl_rejects_all_caps_opener_terms():
    # "TRY IT ::" is an opener label, not a glossary term (all-caps guard).
    src = "TRY IT :: 9.1 Evaluate this. BE PREPARED :: 9.2 Review that."
    assert parse_definition_list(src) is None


def test_dl_rejects_single_pair():
    assert parse_definition_list("term: only one definition here.") is None


# ---------------------------------------------------------------------------
# parse_list
# ---------------------------------------------------------------------------
def test_list_alpha_exercise_parts():
    src = "Simplify: (a) $\\sqrt{36}$ (b) $\\sqrt{196}$ (c) $-\\sqrt{81}$"
    html = parse_list(src)
    assert html is not None
    assert '<ol type="a">' in html
    assert "<li>$\\sqrt{36}$</li>" in html
    assert "<p>Simplify</p>" in html  # lead kept


def test_list_bullets():
    html = parse_list("• Simplify roots • Estimate roots • Add radicals")
    assert html is not None and "<ul>" in html and html.count("<li>") == 3


def test_list_rejects_nonsequential_alpha():
    assert parse_list("value (a) here and also (c) there") is None


def test_list_rejects_fused_sentence_prose():
    # (a)/(b) present but items are full sentences — not exercise parts.
    src = "(a) Since $6^2=36$ we get 6. Now consider (b) the value equals 14."
    assert parse_list(src) is None


# --- ITEM 2 (round-2 audit): repeating-alternation exercise sublists ----------
def test_list_alpha_repeating_cycles_nested():
    # (a)(b)(c) repeated per numbered exercise → a NESTED exercise list, one
    # per-cycle <ol type="a"> per numbered item.
    src = (
        "Simplify: (a) $\\sqrt[3]{216}$ (b) $\\sqrt[4]{256}$ (c) $\\sqrt[5]{32}$ "
        "(a) $\\sqrt[3]{27}$ (b) $\\sqrt[4]{16}$ (c) $\\sqrt[5]{243}$"
    )
    html = parse_list(src)
    assert html is not None
    assert 'class="semantik-exercise-list"' in html
    # Two cycles → two nested <ol type="a"> lists, six leaf <li> items.
    assert html.count('<ol type="a">') == 2
    assert html.count("<li>$\\sqrt") == 6
    assert "<p>Simplify</p>" in html


def test_list_bare_letter_alternation_with_math():
    # Bare (no-paren) a/b/c markers, each followed by a math run → <ol type="a">.
    src = "a $\\sqrt{2}$ b $\\sqrt{3}$ c $\\sqrt{5}$"
    html = parse_list(src)
    assert html is not None
    assert '<ol type="a">' in html
    assert html.count("<li>") == 3


def test_list_bare_letter_prose_article_refused():
    # Anti-fabrication: "a"/"b" as prose words (no math, < 3 alternation) → None.
    assert parse_list("a dog and b cat ran home") is None


def test_list_bare_letter_capital_article_refused():
    # "A car travels a distance …": capital A + prose 'a' → never a list.
    assert parse_list("A car travels a distance of five miles today") is None


def test_list_bare_letter_mostly_prose_refused():
    # >= 3 bare markers but mostly NON-math items → prose, refused.
    assert parse_list("a apple b banana c cherry d date") is None


def test_emit_structure_priority_table_first():
    kind, _ = emit_structure("| a | b | | c | d |")
    assert kind == "table"


# ---------------------------------------------------------------------------
# Declare -> deliver reconciliation through the adapter.
# ---------------------------------------------------------------------------
def test_declared_list_with_shape_delivers_ul_keeps_role():
    ch = _AdapterChapter(
        title="Chapter 1 Whole Numbers",
        blocks=[_block("• one • two • three", 0, role="list")],
    )
    html = _render([ch])
    assert "<ul>" in html and 'data-semantik-block-role="list"' in html


def test_declared_list_without_shape_demoted_to_paragraph(monkeypatch):
    # A mis-typed "list" whose body is flat prose loses the false declaration.
    monkeypatch.setenv("SEMANTIK_EMIT_TOC", "0")  # TOC has its own <ol>
    ch = _AdapterChapter(
        title="Chapter 1 Whole Numbers",
        blocks=[_block("9.2 Simplify Square Roots 9.3 Add Square Roots", 0, role="list")],
    )
    html = _render([ch])
    assert 'data-semantik-block-role="list"' not in html
    assert "<ul>" not in html and "<ol>" not in html
    assert "<p>9.2 Simplify Square Roots 9.3 Add Square Roots</p>" in html


def test_declared_table_without_pipes_demoted():
    ch = _AdapterChapter(
        title="Chapter 9 Roots",
        blocks=[_block("This is just explanatory prose, no pipes at all.", 0, role="table")],
    )
    html = _render([ch])
    assert 'data-semantik-block-role="table"' not in html
    assert "<table>" not in html
    # Blind-spot guard — the honest demotion is stamped so the scorecard can
    # count it (the declaration is otherwise invisible post-reconciliation).
    assert 'data-semantik-demoted-role="table"' in html


# ---------------------------------------------------------------------------
# TABLE-DELIVERY-AWARE marker scrub (wave-19 table-regression fix, 2026-07-04).
# The marker/gutter scrub runs BEFORE structure emission; it must NOT eat the
# pipe rows of a block that parse_table can deliver as a <table> (else 47 tables
# -> 0, rows ship as debris) — but MUST still scrub pipes from prose/list debris
# that does NOT deliver a table (else <p>|</p> / trailing-pipe <li> debris).
# ---------------------------------------------------------------------------
def test_scrub_preserves_pipe_table_rows():
    src = "| a | b | | c | d |"
    # The pipe rows survive the scrub verbatim (previously eaten by the gutter
    # arm as whitespace-bounded stray pipes).
    assert _scrub_marker_artifacts(src, html=False) == src
    # ... and still parse to a real table afterwards.
    out = parse_table(_scrub_marker_artifacts(src, html=False))
    assert out is not None and "<table>" in out and out.count("<td>") == 4


def test_scrub_still_strips_isolated_prose_pipe():
    # Round-1 behaviour intact: a single stray gutter '|' in prose is scrubbed.
    got = _scrub_marker_artifacts("TRY IT :: 9.129 | done", html=False)
    assert "|" not in got
    assert got == "TRY IT 9.129 done"


def test_scrub_strips_pipes_from_prose_debris_not_a_table():
    # ch02 s35 shape: a paragraph with pipe-ish OCR debris that does NOT deliver
    # a table (parse_table -> None) still gets its gutter pipes scrubbed. A pipe-
    # DENSITY heuristic over-protected this and shipped "&gt;| TRYIT" debris.
    assert parse_table("TRYIT | TRYIT") is None  # not a >=2-row table
    got = _scrub_marker_artifacts("&gt;| TRYIT &gt;| TRYIT ", html=False)
    assert "|" not in got and "&gt;" not in got


def test_scrub_strips_trailing_pipes_from_list_debris():
    # ch02 s783 shape: a list whose items carry a trailing '|' — NOT a table, so
    # the trailing pipes are scrubbed (leaving the list body clean for parse_list).
    src = "first item | second item | third item |"
    assert parse_table(src) is None
    got = _scrub_marker_artifacts(src, html=False)
    assert "|" not in got


def test_scrub_previews_sanitize_sep_strip():
    # ch02 s783 exact shape: a '| --- |' separator + ONE data row parses as a
    # table on the RAW text, but sanitize_body_latex (which runs between the
    # scrub and structure emission) strips the separator row, dropping it below
    # parse_table's 2-row floor at emit time. The scrub must gate on the
    # SANITIZED preview and scrub the pipes (else the block falls to parse_list
    # with '<p>|</p>' lead + trailing-pipe <li> debris).
    src = (
        "| --- | --- | --- | | 289. $0.2(p - 6) = 0.4(p + 14)$ "
        "| 290. $0.2(30n + 50) = 28$ | 291. $0.5(16m + 34) = -15$ |"
    )
    assert parse_table(src) is not None  # raw text WOULD parse ...
    got = _scrub_marker_artifacts(src, html=False)
    assert "|" not in got  # ... but emit-time won't, so pipes are scrubbed


def test_parse_table_drops_pipe_only_pre_post_fragments():
    # A leading/trailing pipe-debris fragment around the pipe region must not
    # ship as a '<p>|</p>' sibling of the emitted table.
    out = parse_table("> | a | b | | c | d |")
    assert out is not None
    assert "<p>" not in out


def test_parse_list_drops_pipe_only_lead():
    out = parse_list("| 1. first item here 2. second item here 3. third one")
    assert out is not None
    assert "<p>" not in out and "<ol>" in out


def test_scrub_table_block_still_folds_colon_markers():
    # On a table-shaped block the ':: ' marker fold still fires; only the pipe
    # strip is suppressed.
    src = "TRY IT :: 2.1 | a | b | | c | d |"
    got = _scrub_marker_artifacts(src, html=False)
    assert "::" not in got
    assert "| a | b |" in got  # table pipes preserved


def test_pipe_table_survives_scrub_end_to_end():
    # The wave-19 regression, reproduced through the full adapter: a declared
    # 'table' block whose body is a pipe run must EMIT a <table>, not ship the
    # pipe rows as scrubbed prose debris.
    ch = _AdapterChapter(
        title="Chapter 2 Solving Equations",
        blocks=[_block("| Step | Action | | 1 | Simplify | | 2 | Solve |", 0, role="table")],
    )
    html = _render([ch])
    assert "<table>" in html
    assert 'data-semantik-block-role="table"' in html
    assert "data-semantik-demoted-role" not in html  # delivered, not demoted


def test_ch02_debris_shape_with_fused_marker_parses_to_table():
    # ch02-shape debris: a pipe table with a trailing fused 'TRY IT' marker
    # (round-3's interior-opener pass pre-splits the outside-pipe marker). The
    # scrub keeps the pipes; the table delivers AND the opener heading promotes.
    ch = _AdapterChapter(
        title="Chapter 2 Solving Equations",
        blocks=[
            _block(
                "| Step | Action | | 1 | Simplify | | 2 | Solve | "
                "TRY IT :: 2.15 Solve: x + 3 = 9.",
                0,
                role="table",
            )
        ],
    )
    html = _render([ch])
    assert "<table>" in html
    assert "::" not in html
    assert 'data-semantik-opener' in html  # the split-off marker promoted


# ---------------------------------------------------------------------------
# wrap_bare_math + balancer (B3+).
# ---------------------------------------------------------------------------
def test_wrap_bare_math_wraps_undelimited_run():
    got = wrap_bare_math(r"answer \sqrt{16n^2} = 4n here", html=False)
    assert "$" in got and "\\sqrt{16n^2} = 4n" in got
    # the run is now delimited
    assert re.search(r"\$[^$]*\\sqrt\{16n\^2\} = 4n[^$]*\$", got)


def test_wrap_bare_math_noop_on_prose():
    assert wrap_bare_math("plain english with no math", html=False) == (
        "plain english with no math"
    )


def test_wrap_bare_math_preserves_prose_words_adjacent_to_math():
    got = wrap_bare_math(r"\sqrt{x} and then \sqrt{y} done", html=False)
    assert "and then" in got and "done" in got


def test_wrap_bare_math_strips_orphan_tabular_scaffolding():
    got = wrap_bare_math(r"work \hline \end{tabular} then Figure 9.2", html=False)
    assert "tabular" not in got and "hline" not in got
    assert "Figure 9.2" in got


def test_balancer_drops_stray_dollar():
    # An unclosed $$ display split from its close → DROPPED (Round-7: never
    # shipped as a visible literal, never escaped to \$).
    out = _balance_math_delimiters("open $$ x = y and more prose")
    assert out.count("$$") == 0  # no unbalanced display delimiter survives
    assert "\\$" not in out      # not escaped to a literal backslash-dollar either


def test_balancer_keeps_balanced_math():
    src = "value $a$ and $$b = c$$ end"
    assert _balance_math_delimiters(src) == src


# ---------------------------------------------------------------------------
# Round-11 (true-final) — nested-placeholder restore leak (ch03/ch05 coin tbl).
# ---------------------------------------------------------------------------
def test_balancer_no_sentinel_leak_across_currency_table():
    r"""Currency ``$`` around a real ``\(d\)`` cell must not leak a U+0001 sentinel.

    An OCR coin table (``| Dimes | \(d\) | $0.10 |``) has currency ``$`` on either
    side of an already-paired ``\(d\)`` span. The balancer used to (a) pair the
    currency ``$`` into a bogus span whose CONTENT held the ``\(d\)`` placeholder,
    then (b) restore it single-pass, stranding the inner ``\x01`` sentinel — which
    reached MathJax as "Math input error". The refusal guard + iterative restore
    keep the ``\(d\)`` independent and emit ZERO control chars.
    """
    src = r"| Dimes | \(d\) | $0.10 | Nickels | \(d + 9\) | $0.05 |"
    out = _balance_math_delimiters(src)
    assert "\x01" not in out  # no leaked stash sentinel
    assert r"\(d\)" in out and r"\(d + 9\)" in out  # real cells preserved


# ---------------------------------------------------------------------------
# Round-11 (true-final) — adjacent inline-math separation (ch06 scorecard).
# ---------------------------------------------------------------------------
def test_separate_adjacent_inline_math_spans():
    # ``$a$$b$`` — the ``$$`` junction is a FALSE display delimiter → space it.
    assert (
        separate_adjacent_math_spans(r"$10^4$$17^1$$\left(\frac{1}{2}\right)^2$")
        == r"$10^4$ $17^1$ $\left(\frac{1}{2}\right)^2$"
    )


def test_separate_preserves_genuine_display():
    # A real ``$$…$$`` display span (matched close) is passed through verbatim.
    src = r"$$10^2 - 2^2$$ Simplify."
    assert separate_adjacent_math_spans(src) == src


def test_separate_idempotent_and_noop_on_separated():
    src = r"$x$ and $y$"
    assert separate_adjacent_math_spans(src) == src
    once = separate_adjacent_math_spans(r"$a$$b$")
    assert separate_adjacent_math_spans(once) == once  # idempotent


def test_balancer_neutralizes_orphan_display_across_split():
    # A block ending with an unclosed $$ (its close split to the next block) is
    # neutralized so it can't desync a whole-document $$-pairing pass.
    out = _balance_math_delimiters("prior text $$ \\sqrt{x} = y trailing")
    assert out.count("$$") == 0  # no dangling display delimiter
    # the fenced content survives as text (re-wrappable downstream)
    assert "\\sqrt{x} = y" in out


# ---------------------------------------------------------------------------
# Round-3 Defect 1 — orphan-$$ phantom-math prose swallow.
# ---------------------------------------------------------------------------
def test_balancer_orphan_dd_does_not_swallow_following_prose():
    # An orphan $$ before an opener + real math (the ch02 s507 shape): the stray
    # $ must NOT pair with the $3p opener and drag "TRY IT … Solve:" into math.
    out = _balance_math_delimiters(
        "$$ TRY IT 2.59 Solve: $3p - 14 = 5p$. Solve: $8m + 9 = 5m$."
    )
    assert out.count("$$") == 0  # orphan display gone
    # prose stays OUTSIDE any $-span
    assert "TRY IT 2.59 Solve:" in out
    # the real inline math stays delimited (a single $ each side, paired)
    assert "$3p - 14 = 5p$" in out
    assert "$8m + 9 = 5m$" in out


def test_balancer_legit_display_math_untouched():
    # A well-formed $$display$$ span is preserved verbatim (no false neutralize).
    src = "before $$x^2 + 1 = 0$$ after"
    assert _balance_math_delimiters(src) == src


def test_balancer_orphan_dd_before_math_like_still_pairs():
    # Orphan $$ then a real inline math span whose content is math-like: the
    # orphan is dropped and the math span still pairs.
    out = _balance_math_delimiters("$$ $3p - 14 = 5p$")
    assert out.count("$$") == 0
    assert "$3p - 14 = 5p$" in out


def test_balancer_prose_opener_inline_span_refused():
    # A single-$ orphan gluing an opener onto the next real opener is refused
    # (the prose/opener word-ratio guard), math preserved.
    out = _balance_math_delimiters("$ Simplify: $x^2$ done")
    assert "Simplify:" in out  # prose not italicized
    assert "$x^2$" in out       # real math preserved


def test_balancer_text_heavy_inline_math_not_wrecked():
    # A legitimately-delimited inline span with a \text{} annotation (few prose
    # words, LaTeX-command tokens) is NOT mistaken for swallowed prose.
    src = r"holds when $\text{if } x > 0$ always"
    out = _balance_math_delimiters(src)
    assert r"$\text{if } x > 0$" in out


# ---------------------------------------------------------------------------
# Round-7 — closer / mid-block delimiter literals + stray escapes.
# ---------------------------------------------------------------------------
def test_balancer_round7_exact_defect_no_literals():
    # The exact ch02 Example 2.38 leak: orphan closer \) + literal $$ + stray \y.
    # All three fold out; no visible delimiter / escape literal survives.
    out = _balance_math_delimiters(r"Let \y = -17\). $$ - (y + 9 = 8")
    for lit in (r"\)", r"\]", r"\(", r"\[", "$$", r"\y"):
        assert lit not in out, f"{lit!r} leaked: {out!r}"
    assert "$" not in out          # no lone dollar remnant
    assert "y = -17" in out        # \y folded to bare y, content preserved
    assert "(y + 9 = 8" in out     # plain-ASCII paren prose untouched


def test_balancer_round7_orphan_closer_dropped():
    # Orphan CLOSERS \) and \] (split from their open) drop to space.
    out = _balance_math_delimiters(r"result \) and \] leak here")
    assert "\\)" not in out and "\\]" not in out
    assert "result" in out and "leak here" in out


def test_balancer_round7_orphan_dd_glued_to_real_display_pair():
    # An orphan $$ opener that would GLUE onto a LATER well-formed $$…$$ display
    # pair: the orphan is dropped, the real display span is preserved verbatim
    # (was: orphan $$ kept + the real span's closing $$ eaten).
    out = _balance_math_delimiters(r"intro $$ orphan prose here $$x^2 = 4$$ tail")
    assert "orphan prose here" in out          # prose stays OUT of any span
    assert "$$x^2 = 4$$" in out                # the real display pair survives
    assert out.count("$$") == 2                # exactly the ONE real pair


def test_balancer_round7_stray_escape_no_delimiter():
    # Stray single-letter escapes fold even when the block has NO math delimiter.
    out = _balance_math_delimiters(r"we get \y and \q here")
    assert out == "we get y and q here"


def test_balancer_round7_currency_dollar_preserved():
    # A lone unpaired $ before a digit is CURRENCY — preserved as $ (unchanged).
    assert _balance_math_delimiters("costs $5 to enter") == "costs $5 to enter"


def test_balancer_round7_lone_dollar_not_currency_dropped():
    # A lone $ NOT before a digit drops to space (never a visible literal, never \$).
    out = _balance_math_delimiters("a $ b lone dollar")
    assert "$" not in out and "\\$" not in out
    assert "lone dollar" in out


def test_balancer_round7_row_separators_and_real_commands_untouched():
    # \\ array row separators (inside stashed display math) and multi-letter
    # commands survive the stray-single-letter-escape fold.
    src = r"$$\begin{array}{r} a \\ b \end{array}$$ and \sqrt{2} bare"
    out = _balance_math_delimiters(src)
    assert r"$$\begin{array}{r} a \\ b \end{array}$$" in out  # row sep intact
    assert r"\sqrt{2}" in out                                 # \sqrt not folded


# ---------------------------------------------------------------------------
# SEMANTIK_EMIT_TOC nav (§4).
# ---------------------------------------------------------------------------
def _toc_chapters():
    return [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _AdapterBlock(
                    html="",
                    region_kind="heading",
                    raw_block_index=0,
                    raw_text="9.1 Simplify Square Roots",
                    heading_text="9.1 Simplify Square Roots",
                ),
                _AdapterBlock(
                    html="<p>body</p>",
                    region_kind="paragraph",
                    raw_block_index=1,
                    raw_text="body",
                    heading_text=None,
                ),
            ],
        )
    ]


def test_toc_emitted_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_EMIT_TOC", raising=False)
    html = _render(_toc_chapters())
    assert '<nav class="toc" aria-label="Contents">' in html
    assert 'href="#chap-1"' in html
    # section anchor present (h3 section id)
    assert re.search(r'<li><a href="#[^"]+">9\.1 Simplify Square Roots</a></li>', html)


def test_toc_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("SEMANTIK_EMIT_TOC", "0")
    html = _render(_toc_chapters())
    assert '<nav class="toc"' not in html


def test_toc_anchors_reference_existing_ids_no_duplicates(monkeypatch):
    monkeypatch.setenv("SEMANTIK_EMIT_TOC", "1")
    html = _render(_toc_chapters())
    # the chap-1 id the TOC links to exists on the article
    assert 'id="chap-1"' in html
