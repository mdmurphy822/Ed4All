"""figure_enrich_merge — regression net for the heading-judge caption loss.

Root defect (whole-book single-PDF reference corpus, 2026-07-22): the heading judge's
--apply re-render rebuilds the accessible HTML from the layout sidecar, which
never carried the VLM alt-text enrichment — so every VLM-captioned figure
degraded from a rich ``<figcaption>`` to the sr-only ``"Figure."``
placeholder, and the pipeline copy-back shipped the degraded bytes.

These tests pin the merge contract: degraded judged figures are restored from
the prior enriched render (ADD-only), extracted captions and judged heading
levels are untouched byte-for-byte, unpairable figures are never guessed, and
a clean document round-trips byte-identically.

No GPU, no network, no course data — synthetic fixtures only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.semantik.figure_enrich_merge import (  # noqa: E402
    TYPE_LEVEL_ALT,
    merge_figure_enrichment,
)

# The exact shapes the live corpus showed (SemantiK/output byte evidence):
PLACEHOLDER_FIG = (
    '<figure><figcaption><span class="sr-only">Figure.</span>'
    "</figcaption></figure>"
)
# The SEMANTIK_GLMOCR_MATH_NORMALIZE=0 legacy shape (visible placeholder).
BARE_PLACEHOLDER_FIG = "<figure><figcaption>Figure.</figcaption></figure>"
RICH_FIG = (
    "<figure><figcaption>A network diagram showing interconnected nodes "
    "and links, illustrating a complex system structure."
    "</figcaption></figure>"
)
EXTRACTED_FIG = (
    "<figure><figcaption>Figure 1.1: The system throughput on the y axis."
    "</figcaption></figure>"
)


def _section(bid: str, inner: str) -> str:
    return (
        f'<section class="semantik-section" id="{bid}" '
        f'data-semantik-block-id="{bid}" data-semantik-source="synthesized" '
        f'data-semantik-block-role="figure">{inner}</section>'
    )


def test_placeholder_sr_only_figcaption_restored_by_block_id():
    prior = "<h2>Old level</h2>" + _section("s2", RICH_FIG)
    judged = "<h3>New level</h3>" + _section("s2", PLACEHOLDER_FIG)
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 1
    assert "A network diagram showing interconnected nodes" in merged
    assert 'sr-only">Figure.' not in merged
    # the judged heading level (the point of the re-render) is untouched
    assert "<h3>New level</h3>" in merged
    assert "<h2>Old level</h2>" not in merged


def test_bare_visible_placeholder_also_restored():
    prior = _section("s2", RICH_FIG)
    judged = _section("s2", BARE_PLACEHOLDER_FIG)
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 1
    assert "A network diagram" in merged
    assert ">Figure.</figcaption>" not in merged


def test_extracted_caption_never_touched():
    """A judged figure with a REAL caption (extracted from source text)
    keeps the judge's own render — even when the prior differs."""
    prior = _section("s5", RICH_FIG)
    judged = _section("s5", EXTRACTED_FIG)
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 0
    assert merged == judged


def test_prior_also_degraded_never_invents_a_caption():
    prior = _section("s2", PLACEHOLDER_FIG)
    judged = _section("s2", PLACEHOLDER_FIG)
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 0
    assert merged == judged


def test_no_degraded_figures_is_byte_identical_no_op():
    doc = _section("s1", EXTRACTED_FIG) + _section("s2", RICH_FIG)
    merged, restored = merge_figure_enrichment(doc, doc)
    assert restored == 0
    assert merged == doc


def test_missing_figcaption_gets_prior_caption_inserted():
    prior = _section("s2", RICH_FIG)
    judged = _section("s2", "<figure></figure>")
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 1
    assert (
        "<figure><figcaption>A network diagram showing interconnected "
        "nodes" in merged
    )
    assert merged.count("</figure>") == 1


def test_img_alt_placeholder_restored_from_prior():
    prior = _section(
        "s2",
        '<figure><img src="fig-1.png" alt="A red rectangle with an arrow">'
        "<figcaption>A red rectangle with an arrow</figcaption></figure>",
    )
    judged = _section(
        "s2",
        f'<figure><img src="fig-1.png" alt="{TYPE_LEVEL_ALT}">'
        '<figcaption><span class="sr-only">Figure.</span></figcaption>'
        "</figure>",
    )
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 1
    assert 'alt="A red rectangle with an arrow"' in merged
    assert f'alt="{TYPE_LEVEL_ALT}"' not in merged
    assert "<figcaption>A red rectangle with an arrow</figcaption>" in merged


def test_block_id_pairing_survives_count_mismatch():
    """An extra judged figure (no prior counterpart) is skipped; the
    id-matched degraded one is still restored."""
    prior = _section("s2", RICH_FIG)
    judged = (
        _section("s2", PLACEHOLDER_FIG)
        + _section("s9", PLACEHOLDER_FIG)  # no prior counterpart
    )
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 1
    assert "A network diagram" in merged
    # the unmatched figure keeps its honest placeholder (never guessed)
    assert merged.count('sr-only">Figure.') == 1


def test_ordinal_fallback_when_no_block_ids_and_counts_equal():
    prior = f"<div>{RICH_FIG}</div>"
    judged = f"<div>{PLACEHOLDER_FIG}</div>"
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 1
    assert "A network diagram" in merged


def test_no_ids_and_count_mismatch_returns_judged_unchanged():
    prior = RICH_FIG
    judged = PLACEHOLDER_FIG + PLACEHOLDER_FIG
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 0
    assert merged == judged


def test_duplicate_block_ids_pair_by_occurrence():
    """Two figures under the SAME block id pair by occurrence order —
    each degraded judged figure gets ITS OWN prior counterpart."""
    rich_second = (
        "<figure><figcaption>Timeline showing events at x=1 and x=3."
        "</figcaption></figure>"
    )
    prior = _section("s7", RICH_FIG + rich_second)
    judged = _section("s7", PLACEHOLDER_FIG + PLACEHOLDER_FIG)
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 2
    assert "A network diagram" in merged
    assert "Timeline showing events" in merged
    # order preserved: the network diagram caption comes first
    assert merged.index("A network diagram") < merged.index(
        "Timeline showing events"
    )


def test_legacy_block_id_attribute_spelling_admitted_on_read():
    prior = (
        '<section data-dart-block-id="s3">' + RICH_FIG + "</section>"
    )
    judged = (
        '<section data-dart-block-id="s3">' + PLACEHOLDER_FIG + "</section>"
    )
    merged, restored = merge_figure_enrichment(prior, judged)
    assert restored == 1
    assert "A network diagram" in merged


def test_empty_documents_are_safe():
    assert merge_figure_enrichment("", "") == ("", 0)
    assert merge_figure_enrichment(RICH_FIG, "") == ("", 0)
    assert merge_figure_enrichment("", PLACEHOLDER_FIG) == (
        PLACEHOLDER_FIG,
        0,
    )
