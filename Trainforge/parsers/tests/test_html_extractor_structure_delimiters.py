"""HTMLTextExtractor structural-delimiter emission (A-item pairing).

SemantiK now emits real <ul>/<ol>/<table>/<dl> bodies; the extractor must keep
that structure signal (newline per item/row, " | " between cells, ": " after a
term) instead of collapsing everything to a run-on string. Synthetic HTML only.
"""
from __future__ import annotations

from Trainforge.parsers.html_content_parser import HTMLTextExtractor


def _extract(html: str) -> str:
    ex = HTMLTextExtractor()
    ex.feed(html)
    return ex.get_text()


def test_list_items_newline_delimited():
    text = _extract("<ul><li>Simplify roots</li><li>Estimate roots</li></ul>")
    # items are separated (a newline token lands between them), not run-on.
    assert "\n" in text
    assert "Simplify roots" in text and "Estimate roots" in text
    # the two items are not directly space-joined into one token run
    assert "Simplify roots Estimate roots" not in text


def test_table_cells_pipe_delimited_rows_newline():
    text = _extract(
        "<table><tr><td>4</td><td>two</td></tr>"
        "<tr><td>9</td><td>three</td></tr></table>"
    )
    assert "4 | two" in text
    assert "9 | three" in text
    assert "\n" in text  # row separator


def test_table_header_cells_delimited():
    text = _extract(
        "<table><thead><tr><th>Number</th><th>Root</th></tr></thead>"
        "<tbody><tr><td>4</td><td>2</td></tr></tbody></table>"
    )
    assert "Number | Root" in text
    assert "4 | 2" in text


def test_definition_list_term_colon_definition():
    text = _extract(
        "<dl><dt>index</dt><dd>the index of the radical.</dd>"
        "<dt>radical</dt><dd>a root expression.</dd></dl>"
    )
    assert "index" in text and "the index of the radical." in text
    assert ":" in text  # dt terminated with a colon delimiter
    assert "\n" in text  # dd terminated with a newline


def test_sr_only_labels_do_not_leak_delimiters_or_text():
    # A screen-reader-only label subtree is skipped entirely — its text and any
    # structural delimiters inside it stay out of the chunk text.
    text = _extract(
        '<section><p class="sr-only" hidden>List block</p>'
        "<ul><li>real item</li></ul></section>"
    )
    assert "List block" not in text
    assert "real item" in text


def test_plain_paragraph_unchanged():
    text = _extract("<p>The order of operations matters here.</p>")
    assert text == "The order of operations matters here."
