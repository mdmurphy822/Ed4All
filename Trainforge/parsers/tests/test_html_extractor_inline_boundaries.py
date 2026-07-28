"""HTMLTextExtractor whitespace contract (EXTRACTION_TEXT_CONTRACT_VERSION 2).

Under contract 1 the extractor ``.strip()``-ed every HTML text node and
re-joined the pieces with a single space, so EVERY inline tag boundary
fabricated a space: ``<p>The result is <strong>79</strong>.</p>`` extracted as
``"The result is 79 ."``. That artifact is not merely cosmetic — it is exactly
the signal the training-synthesis content gate reads as auto-generated
slot-filler residue (``synthesis_eligibility._STEM_SLOT_GAP_RE``), so real
chunks were being excluded from synthesis by a defect in our own extractor.

Contract 2 assembles text whitespace-correctly instead: source whitespace is
preserved (collapsed to one space), an INLINE element boundary contributes no
separator of its own, a BLOCK element boundary does.

Both failure directions are pinned here. The fabricated space is the bug being
fixed; a fabricated JOIN ("theresult") would be strictly worse, so the
block-boundary and no-false-join batteries below are the load-bearing half.

Synthetic HTML only — no corpus paths, no course slugs.
"""
from __future__ import annotations

import pytest

from Trainforge.chunker import EXTRACTION_TEXT_CONTRACT_VERSION
from Trainforge.parsers.html_content_parser import HTMLTextExtractor
from Trainforge.parsers.xpath_walker import find_body_xpath, resolve_xpath
from Trainforge.synthesis_eligibility import content_gate_eligibility


def _extract(html: str) -> str:
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ---------------------------------------------------------------------------
# Direction 1: inline boundaries must not fabricate whitespace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "html,expected",
    [
        # The canonical defect: emphasis immediately before terminal
        # punctuation.
        ("<p>The result is <strong>79</strong>.</p>", "The result is 79."),
        ("<p>Use the <em>distributive</em> property.</p>",
         "Use the distributive property."),
        ("<p>Call <code>solve()</code>, then check.</p>",
         "Call solve(), then check."),
        # Exponents / subscripts fuse with their base, as they render.
        ("<p>x<sup>2</sup> + 1</p>", "x2 + 1"),
        ("<p>H<sub>2</sub>O</p>", "H2O"),
        # A link mid-sentence keeps exactly the author's spacing.
        ("<p>See <a href='#'>the docs</a> for more.</p>",
         "See the docs for more."),
        # Adjacent inline spans with no whitespace between them are one token.
        ("<p><span>un</span><span>breakable</span></p>", "unbreakable"),
        # Nested inline markup collapses the same way.
        ("<p>a<strong><em>b</em></strong>c</p>", "abc"),
    ],
)
def test_inline_boundary_emits_no_fabricated_space(html, expected):
    assert _extract(html) == expected


@pytest.mark.parametrize(
    "html,expected",
    [
        # Whitespace the author DID write around an inline element survives.
        ("<p>the <strong>bold</strong> word</p>", "the bold word"),
        ("<p>the <strong>bold</strong>\n    word</p>", "the bold word"),
        # Whitespace INSIDE the inline element counts too.
        ("<p>the<strong> bold </strong>word</p>", "the bold word"),
        # A whitespace-only text node between two inline elements is the
        # separator, and only one space comes out of it.
        ("<p><em>a</em>   <em>b</em></p>", "a b"),
    ],
)
def test_source_whitespace_at_inline_boundary_is_preserved(html, expected):
    assert _extract(html) == expected


def test_no_spaced_terminal_punctuation_from_markup():
    """The gate-visible artifact is gone for every inline wrapper."""
    for tag in ("strong", "em", "b", "i", "span", "code", "a", "sup", "sub"):
        text = _extract(f"<p>The value is <{tag}>42</{tag}>.</p>")
        assert " ." not in text, tag
        assert text.endswith("42."), tag


# ---------------------------------------------------------------------------
# Direction 2 (the worse failure): block boundaries must still separate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "html,expected",
    [
        ("<p>the</p><p>result</p>", "the result"),
        ("<div>the</div><div>result</div>", "the result"),
        ("<h2>the</h2><p>result</p>", "the result"),
        ("<section><p>the</p></section><section><p>result</p></section>",
         "the result"),
        ("<blockquote>the</blockquote><p>result</p>", "the result"),
        ("<p>the<br/>result</p>", "the result"),
        ("<p>the</p><figure><figcaption>result</figcaption></figure>",
         "the result"),
        # Unknown / custom elements are NOT assumed inline — they separate.
        ("<the-widget>the</the-widget><p>result</p>", "the result"),
    ],
)
def test_block_boundary_still_separates(html, expected):
    text = _extract(html)
    assert text == expected
    assert "theresult" not in text


def test_no_false_join_across_structural_delimiters():
    """List / table / definition delimiters keep their separating role."""
    assert _extract("<ul><li>the</li><li>result</li></ul>") == "the\nresult\n"
    assert _extract("<table><tr><td>the</td><td>result</td></tr></table>") == (
        "the | result"
    )
    assert _extract("<dl><dt>the</dt><dd>result</dd></dl>") == "the: result\n"
    for rendered in (
        _extract("<ul><li>the</li><li>result</li></ul>"),
        _extract("<table><tr><td>the</td><td>result</td></tr></table>"),
        _extract("<dl><dt>the</dt><dd>result</dd></dl>"),
    ):
        assert "theresult" not in rendered


def test_elided_subtree_still_separates_its_neighbours():
    """A skipped INLINE wrapper must not fuse the text on either side of it.

    The force-injected ``data-cf-curie`` span and SemantiK's screen-reader-only
    labels are dropped from chunk text; dropping them must not silently join
    the surrounding words into one token.
    """
    curie = _extract(
        '<p>the<span hidden data-cf-curie="ex:thing">ex:thing</span>result</p>'
    )
    assert curie == "the result"
    sr_only = _extract('<p>the<span class="sr-only">List block</span>result</p>')
    assert sr_only == "the result"
    assert "List block" not in sr_only


def test_no_leading_or_doubled_whitespace_is_emitted():
    """Separators materialize only between real text, never around it."""
    text = _extract(
        "<html><body>\n  <main>\n    <h2>Title</h2>\n"
        "    <p>  Body   text.  </p>\n  </main>\n</body></html>"
    )
    assert text == "Title Body text."


# ---------------------------------------------------------------------------
# Downstream consequences
# ---------------------------------------------------------------------------

def test_content_gate_admits_an_item_the_artifact_used_to_exclude():
    """The production gate — not the leaf predicate — flips to eligible."""
    html = (
        "<div><p>Identify the property that justifies rewriting the sum as "
        "the <strong>commutative</strong>.</p></div>"
    )
    text = _extract(html)
    assert text.endswith("commutative.")
    chunk = {"id": "c1", "text": text, "concept_tags": ["properties"]}
    assert content_gate_eligibility(chunk).eligible

    # The contract-1 projection of the SAME markup is what the gate rejected:
    # the ``</strong>`` boundary fabricated a space before the period.
    legacy_text = text.replace("commutative.", "commutative .")
    legacy_chunk = dict(chunk, text=legacy_text)
    legacy = content_gate_eligibility(legacy_chunk)
    assert not legacy.eligible
    assert legacy.reason == "degenerate_source_stem"


def test_walker_container_text_matches_extractor_for_prose():
    """``chunker._locate`` str.find()s chunk text inside the walker's text.

    A divergence between the two whitespace models silently degrades every
    ``char_span`` to the approximated fallback, so parity is pinned.
    """
    html = (
        "<html><body><main>"
        "<h2>Roots</h2>"
        "<p>The result is <strong>79</strong>.</p>"
        "<p>Next para <em>here</em>, then more.</p>"
        "</main></body></html>"
    )
    extracted = _extract(html)
    container = resolve_xpath(html, find_body_xpath(html))
    assert container == extracted
    assert container.find(extracted) == 0


def test_extraction_contract_version_is_two():
    """Guard: an extraction-text change must bump the contract version.

    The contract has no opt-out flag by design, so the version stamp is the
    only signal downstream provenance gates get that a corpus predates the
    current text semantics.
    """
    assert EXTRACTION_TEXT_CONTRACT_VERSION == 2
