"""Unit tests — whole-document heading-contiguity normalization.

Closes the gap where a Stage-6 specialist (e.g. the hosted 70B prose
seat) emits ``<hN>`` tags inside a non-heading region body, bypassing
the per-region heading-tree normalization and producing a level skip the
Stage-10 ``heading_tree`` gate rejects.
"""

from __future__ import annotations

import re

from dart_semantic.assembler.heading_contiguity import (
    normalize_document_heading_levels,
)
from dart_semantic.gates.hard_document import _check_heading_hierarchy


def _levels(html: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"<h([1-6])\b", html, re.I)]


def _texts(html: str) -> list[str]:
    return re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", html, re.I | re.S)


def test_repairs_embedded_h2_h6_skip():
    # The exact shape the 70B prose seat produced on the mini slice.
    html = (
        "<h1>Identify Multiples</h1>"
        "<h2>12 Chapter 1 Foundations</h2>"
        "<h6>EXAMPLE 1.6</h6>"
        "<p>body</p>"
    )
    out = normalize_document_heading_levels(html)
    assert _levels(out) == [1, 2, 3]
    assert _check_heading_hierarchy(out).passed is True


def test_preserves_heading_text_and_attrs():
    html = (
        '<h1 id="a">Alpha</h1>'
        '<h4 class="x" data-y="z">Beta</h4>'
    )
    out = normalize_document_heading_levels(html)
    # h1 -> h1, h4 -> h2 (demote-forward never-skip).
    assert _levels(out) == [1, 2]
    assert _texts(out) == ["Alpha", "Beta"]
    # Attributes preserved on the demoted heading.
    assert 'class="x"' in out and 'data-y="z"' in out
    assert 'id="a"' in out


def test_idempotent_on_contiguous_doc():
    html = "<h1>A</h1><h2>B</h2><h3>C</h3><h2>D</h2>"
    out = normalize_document_heading_levels(html)
    assert out == html  # byte-identical


def test_promote_first_heading_to_h1():
    html = "<h3>First</h3><h4>Second</h4>"
    out = normalize_document_heading_levels(html)
    assert _levels(out) == [1, 2]


def test_no_headings_unchanged():
    html = "<p>no headings here</p>"
    assert normalize_document_heading_levels(html) == html


def test_empty_unchanged():
    assert normalize_document_heading_levels("") == ""


def test_close_tag_level_rewritten():
    # The matching close tag must be rewritten too, else html5 balance
    # would see <h6>...</h6> mismatch after the open became <h3>.
    html = "<h1>A</h1><h2>B</h2><h6>C</h6>"
    out = normalize_document_heading_levels(html)
    assert "<h3>C</h3>" in out
    assert "<h6>" not in out and "</h6>" not in out


def test_multiple_h1_with_skip_only_fixes_skip():
    # Multiple h1 is allowed by the gate (no skip); a trailing skip is
    # the only thing that needs repair.
    html = "<h1>A</h1><h1>B</h1><h1>C</h1><h2>D</h2><h6>E</h6>"
    out = normalize_document_heading_levels(html)
    assert _levels(out) == [1, 1, 1, 2, 3]
    assert _check_heading_hierarchy(out).passed is True


# ---------------------------------------------------------------------------
# assemble_document wiring (_finalize_heading_contiguity)
# ---------------------------------------------------------------------------


def _make_doc(html: str):
    from dart_semantic.assembler.types import AssembledDoc

    return AssembledDoc(
        html=html,
        gaps_found=[],
        gaps_resolved=[],
        gaps_fallback=[],
        heading_tree=[(2, "12 Chapter 1 Foundations"), (6, "EXAMPLE 1.6")],
        landmarks={"main": 1},
        anchors={},
        region_provenance=[],
        sub_task_log={},
    )


def test_finalize_updates_html_and_tree():
    from dart_semantic.assembler.api import _finalize_heading_contiguity

    doc = _make_doc(
        "<h1>Identify Multiples</h1>"
        "<h2>12 Chapter 1 Foundations</h2>"
        "<h6>EXAMPLE 1.6</h6>"
    )
    out = _finalize_heading_contiguity(doc)
    assert _levels(out.html) == [1, 2, 3]
    # heading_tree re-derived to match the emitted levels (all three).
    assert out.heading_tree == [
        (1, "Identify Multiples"),
        (2, "12 Chapter 1 Foundations"),
        (3, "EXAMPLE 1.6"),
    ]
    assert out.sub_task_log.get("heading_contiguity")


def test_finalize_idempotent_byte_stable():
    from dart_semantic.assembler.api import _finalize_heading_contiguity

    clean = "<h1>A</h1><h2>B</h2><h3>C</h3>"
    doc = _make_doc(clean)
    original_tree = list(doc.heading_tree)
    out = _finalize_heading_contiguity(doc)
    # No rewrite -> html byte-identical AND the (stale) heading_tree is
    # left untouched (the pass only re-derives when it changed html).
    assert out.html == clean
    assert out.heading_tree == original_tree
    assert "heading_contiguity" not in out.sub_task_log
