"""Regression coverage for generic Markdown and MDX sanitization."""

from __future__ import annotations

from lib.importers._markdown import _strip_mdx, md_to_html, scan_leaks

_PRESENTATIONAL_HEADER = (
    "<br>\n"
    '<a href="https://docs.example.org/guide/">\n'
    '    <div style="width: 55%; background-color: white;">\n'
    '    <img src="https://assets.example.org/header-mark.png"\n'
    '         width="400" style="width: 300px"/></div>\n'
    "</a>\n"
    '<h1 style="line-height: 1.4;"><font color="#2255aa"><b>Reliable Data Pipelines</b></font></h1>\n'
    "<h2><b>Lesson 3: </b>Routing Content Safely</h2>\n"
    "<br>"
)


def test_presentational_header_does_not_leak_escaped_markup():
    html = md_to_html(_PRESENTATIONAL_HEADER)

    assert scan_leaks(html) == []
    for marker in ("&lt;br", "&lt;div", "&lt;font", "&lt;img", "header-mark.png"):
        assert marker not in html, f"leaked {marker!r}: {html!r}"


def test_presentational_header_preserves_heading_content():
    html = md_to_html(_PRESENTATIONAL_HEADER)

    assert "Reliable Data Pipelines" in html
    assert "Routing Content Safely" in html
    assert "Lesson 3" not in html


def test_presentational_tags_are_removed_from_markdown_heading():
    html = md_to_html('#### <font color="2255aa">Validation Complete</font>')

    assert "&lt;font" not in html
    assert "Validation Complete" in html


_QUESTION_MDX = """\
### Q1: What does the validation stage protect?
Which statement best describes its purpose?

<Question
choices={[
  {
    text: "It rejects malformed content before publication",
    explain: "Validation enforces the output contract.",
    correct: true
  },
  {
    text: "It changes the source material without review",
    explain: "Validation should not silently rewrite source meaning."
  }
]}
/>

---
"""


def test_multiline_question_component_is_removed_but_prose_survives():
    html = md_to_html(_strip_mdx(_QUESTION_MDX))

    for marker in ("<Question", "&lt;Question", "choices={", "explain:", "correct:"):
        assert marker not in html, f"leaked {marker!r}"
    assert "What does the validation stage protect?" in html


def test_paired_and_self_closing_components_are_removed():
    paired = _strip_mdx("Before\n<Callout>\nhidden component body\n</Callout>\nAfter")
    assert "Callout" not in paired
    assert "hidden component body" not in paired
    assert "Before" in paired and "After" in paired

    self_closing = _strip_mdx("Lead\n<Check id={5} />\nTrail")
    assert "Check" not in self_closing
    assert "Lead" in self_closing and "Trail" in self_closing


def test_html_comments_and_embedded_images_are_dropped():
    markdown = (
        '<!-- > <img src="images/diagram.png" /> -->\n'
        '> <img src="https://assets.example.org/diagram.png" width=400px/>\n'
        '<!-- another <img/> -->'
    )

    html = md_to_html(markdown)

    assert "&lt;img" not in html
    assert "diagram.png" not in html


def test_standard_markdown_rendering_is_preserved():
    html = md_to_html(
        "## Core Concepts\n\n"
        "A normal paragraph with `code` and **bold**.\n\n"
        "- item one\n"
        "- item two"
    )

    assert "<h3>Core Concepts</h3>" in html
    assert "<code>code</code>" in html
    assert "<strong>bold</strong>" in html
    assert "<li>item one</li>" in html
