"""Regression guards for nvidia_corpus_to_dart raw-HTML / MDX sanitization.

Bug #2: embedded raw HTML (the NVIDIA logo header) and multi-line JSX
(``<Question choices={[ ... ]}/>``) leaked verbatim into the emitted
``*_accessible.html`` and surfaced as escaped ``&lt;...&gt;`` entities in
chunk text. These tests pin the sanitization so the leak cannot regress.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "nvidia_corpus_to_dart.py"
_spec = importlib.util.spec_from_file_location("nvidia_corpus_to_dart", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


# --- 1. NVIDIA logo header (embedded raw HTML in a markdown cell) ----------
_LOGO_HEADER = (
    '<br>\n'
    '<a href="https://www.nvidia.com/en-us/training/">\n'
    '    <div style="width: 55%; background-color: white;">\n'
    '    <img src="https://dli-lms.s3.amazonaws.com/assets/general/nvidia-logo.png"\n'
    '         width="400" style="width: 300px"/></div>\n'
    '</a>\n'
    '<h1 style="line-height: 1.4;"><font color="#76b900"><b>Building RAG Agents with LLMs</b></font></h1>\n'
    '<h2><b>Notebook 3: </b>LangChain Expression Language</h2>\n'
    '<br>'
)


def test_logo_header_no_escaped_markup_in_paragraphs():
    html = mod.md_to_html(_LOGO_HEADER)
    # No presentational markup leaks as escaped entities inside <p> bodies.
    assert mod._scan_leaks(html) == []
    # And nowhere in the page either, for these chrome markers.
    for marker in ("&lt;br", "&lt;div", "&lt;font", "&lt;img", "nvidia-logo.png"):
        assert marker not in html, f"leaked {marker!r}: {html!r}"


def test_logo_header_preserves_real_heading_text():
    html = mod.md_to_html(_LOGO_HEADER)
    assert "LangChain Expression Language" in html
    assert "Building RAG Agents with LLMs" in html
    # "Notebook 3:" scaffolding prefix is stripped from the heading.
    assert "Notebook 3" not in html


# --- 2. Heading text carrying inline presentational tags -------------------
def test_font_in_markdown_heading_is_stripped():
    html = mod.md_to_html('#### <font color="76b900">Great Job!</font>')
    assert "&lt;font" not in html
    assert "Great Job!" in html


# --- 3. Multi-line JSX <Question> block (MDX) ------------------------------
_QUIZ_MDX = """\
### Q1: What is the primary purpose of LangGraph?
Which statement best describes what LangGraph is designed for?

<Question
choices={[
  {
    text: "A framework to build control flows for applications containing LLMs",
    explain: "LangGraph is designed to help manage control flow.",
    correct: true
  },
  {
    text: "A library that provides interfaces to interact with different LLM models",
    explain: "This better describes LangChain's role.",
  }
]}
/>

---
"""


def test_question_jsx_block_stripped_but_prose_survives():
    stripped = mod._strip_mdx(_QUIZ_MDX)
    html = mod.md_to_html(stripped)
    # The JSX block + its JS object literal are gone.
    for marker in ("<Question", "&lt;Question", "choices={", "explain:", "correct:"):
        assert marker not in html, f"leaked {marker!r}"
    # The real quiz question heading prose survives.
    assert "primary purpose of LangGraph" in html


def test_paired_and_self_closing_jsx_removed():
    paired = mod._strip_mdx("Before\n<Tip>\nhidden tip body\n</Tip>\nAfter")
    assert "Tip" not in paired and "hidden tip body" not in paired
    assert "Before" in paired and "After" in paired

    self_closing = mod._strip_mdx("Lead\n<Quiz id={5} />\nTrail")
    assert "Quiz" not in self_closing
    assert "Lead" in self_closing and "Trail" in self_closing


# --- 4. HTML comments holding alternate <img> markup ----------------------
def test_html_comment_with_img_is_dropped():
    md = (
        '<!-- > <img src="imgs/diagram.png" /> -->\n'
        '> <img src="https://example.com/diagram.png" width=400px/>\n'
        '<!-- another <img/> -->'
    )
    html = mod.md_to_html(md)
    assert "&lt;img" not in html
    assert "diagram.png" not in html


# --- 5. Genuine Markdown is untouched -------------------------------------
def test_plain_markdown_unaffected():
    html = mod.md_to_html(
        "## Real Section\n\nA normal paragraph with `code` and **bold**.\n\n- item one\n- item two"
    )
    assert "<h3>Real Section</h3>" in html
    assert "<code>code</code>" in html and "<strong>bold</strong>" in html
    assert "<li>item one</li>" in html
