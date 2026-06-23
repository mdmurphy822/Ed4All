"""WCAGValidator catches for converter-misclassified headings.

The SemantiK ``clean_structure`` pass demotes fused-TOC + bullet/list-item
headings upstream; these validation-time catches are the SECOND LINE that flag
the mis-classification even when the deterministic demote misses an edge case.
Covers ``WCAGValidator._check_misclassified_headings`` (SC 1.3.1 / 2.4.6).
"""

from __future__ import annotations

from lib.validators.wcag import WCAGValidator


def _doc(body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "  <head><title>Test</title></head>\n"
        f"  <body><main role=\"main\"><h1>Top</h1>{body}</main></body>\n"
        "</html>\n"
    )


def _messages(html: str) -> list[str]:
    report = WCAGValidator().validate(html)
    return [i.message for i in report.issues]


_FUSED = (
    "1.1 Introduction to Whole Numbers 1.2 Use the Language of Algebra "
    "1.3 Add and Subtract Integers"
)
_BULLET = "◦ Yes–add 1 to the digit in the given place value. ◦ No–do not change"


def test_fused_toc_heading_flagged():
    msgs = _messages(_doc(f"<h3>{_FUSED}</h3>"))
    assert any("fused table-of-contents" in m for m in msgs), msgs


def test_bullet_heading_flagged():
    msgs = _messages(_doc(f"<h3>{_BULLET}</h3>"))
    assert any("list-item bullet marker" in m for m in msgs), msgs


def test_dash_bullet_heading_flagged():
    msgs = _messages(_doc("<h3>– an en-dash bullet list item here</h3>"))
    assert any("list-item bullet marker" in m for m in msgs), msgs


def test_clean_single_section_heading_not_flagged():
    # Exactly ONE "N.M" -> legitimate section heading, never flagged.
    msgs = _messages(_doc("<h3>1.1 Introduction to Whole Numbers</h3>"))
    assert not any("fused table-of-contents" in m for m in msgs), msgs
    assert not any("list-item bullet marker" in m for m in msgs), msgs


def test_hyphen_hugging_operand_not_flagged_as_bullet():
    # "-40 degrees" is a minus hugging its operand, not a list bullet.
    msgs = _messages(_doc("<h3>-40 degrees and falling</h3>"))
    assert not any("list-item bullet marker" in m for m in msgs), msgs


def test_plain_heading_not_flagged():
    msgs = _messages(_doc("<h2>Chapter 2: Solving Equations</h2>"))
    assert not any("fused table-of-contents" in m for m in msgs), msgs
    assert not any("list-item bullet marker" in m for m in msgs), msgs
