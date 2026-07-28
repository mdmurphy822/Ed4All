"""QTI nested-HTML projection must not consume literal math inequalities."""
from __future__ import annotations

from MCP.tools.pipeline_tools import _harvest_qti_assessment_chunks


def _qti(stem: str, choices: list[str]) -> str:
    import html

    labels = "".join(
        "<response_label ident='choice-{index}'>"
        "<material><mattext texttype='text/html'>{choice}</mattext></material>"
        "</response_label>".format(
            index=index, choice=html.escape(choice),
        )
        for index, choice in enumerate(choices)
    )
    return (
        "<questestinterop><assessment title='Inequality Quiz'><section>"
        "<item ident='item-1' title='CO-01'><presentation>"
        "<material><mattext texttype='text/html'>"
        f"{html.escape(stem)}</mattext></material>"
        f"<response_lid>{labels}</response_lid>"
        "</presentation></item></section></assessment></questestinterop>"
    )


def _harvest(stem: str, choices: list[str]) -> dict:
    calls = []

    def create_chunk(**kwargs):
        calls.append(kwargs)
        return {
            "id": kwargs["chunk_id"],
            "text": kwargs["text"],
            "html": kwargs["html"],
            "chunk_type": kwargs["chunk_type"],
            "learning_outcome_refs": kwargs["item"]["objective_refs"],
        }

    chunks = _harvest_qti_assessment_chunks(
        [{"path": "06_assessments/inequality.xml",
          "content": _qti(stem, choices)}],
        create_chunk=create_chunk,
        existing_chunks=[],
        course_code="generic",
    )
    assert len(chunks) == 1
    assert len(calls) == 1
    return chunks[0]


def test_harvest_preserves_all_latex_comparison_operators() -> None:
    chunk = _harvest(
        "<p>Classify $Ax+By<C$, $x>y$, $m<=n$, and $p>=q$.</p>",
        [
            "$Ax+By<C$",
            "$x>y$",
            "$m<=n$",
            "$p>=q$",
        ],
    )
    assert chunk["text"] == (
        "Classify $Ax+By<C$, $x>y$, $m<=n$, and $p>=q$.\n"
        "$Ax+By<C$\n$x>y$\n$m<=n$\n$p>=q$"
    )


def test_harvest_preserves_paren_and_display_math_comparators() -> None:
    chunk = _harvest(
        r"<p>Compare \(a<b\) with \[c>=d\].</p>",
        [r"\(a<=b\)", r"\[c>d\]"],
    )
    assert r"\(a<b\)" in chunk["text"]
    assert r"\[c>=d\]" in chunk["text"]
    assert r"\(a<=b\)" in chunk["text"]
    assert r"\[c>d\]" in chunk["text"]


def test_real_nested_markup_is_still_projected_not_escaped() -> None:
    chunk = _harvest(
        (
            "<p>Choose <strong>one</strong> expression and ignore "
            "<script>unsafe()</script><style>.bad{}</style> chrome.</p>"
        ),
        ["<em>$x<y$</em>", "<span>ordinary choice</span>"],
    )
    assert "<strong>" not in chunk["text"]
    assert "<em>" not in chunk["text"]
    assert "<span>" not in chunk["text"]
    assert "unsafe" not in chunk["text"]
    assert ".bad" not in chunk["text"]
    assert "Choose one expression" in chunk["text"]
    assert "$x<y$" in chunk["text"]
    assert "ordinary choice" in chunk["text"]


def test_actual_markup_inside_math_span_remains_markup() -> None:
    chunk = _harvest(
        "<p>Read $<strong>x</strong><y$ safely.</p>",
        ["$<em>a</em>>b$"],
    )
    assert "<strong>" not in chunk["text"]
    assert "<em>" not in chunk["text"]
    # The canonical HTML projection may insert token-boundary spaces where a
    # nested element was removed; the comparison operators and operands must
    # nevertheless survive.
    assert "x <y$" in chunk["text"]
    assert "a >b$" in chunk["text"]
