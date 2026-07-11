"""End-to-end CPU regression — a figure AND a structurally-strong table
survive the WHOLE deterministic assembly path into the Ed4All adapter HTML.

This is the regression the per-stage unit tests missed: the structure graph
formed ``figure`` / ``table`` regions and the Stage-9 assembler emitted real
``<figure><img>`` / ``<table>`` HTML, but the cascade_ir adapter — which
consumes ONLY the distilled ``region_provenance`` list, never the assembler's
rich HTML — dropped the structured data on the floor. A figure provenance
entry carried no caption (a synthetic image FB has empty ``raw_text``) so the
adapter emitted an empty ``<figure></figure>``; a table region had no
``cell_grid`` carried so it flattened to a ``<p>``. The net effect: the final
Ed4All HTML contained paragraph/heading/list block-roles only — ZERO
figure/table — despite the cascade detecting both.

The fix carries the figure/table caption text + the deterministic table grid
onto the provenance (``cascade.py::_build_region_provenance``) and renders them
in the adapter (``lib/semantik/cascade_ir.py``).

Path exercised (the REAL functions, no GPU / no models / no PDF IO):

    arbitrate
      → build_structure_graph (Pass-3 table + Pass-3a figure)
      → assemble_document (pass_9a/9b/9c)
      → cascade._build_region_provenance
      → cascade_ir.build_chapters_ir
      → adapter.normalize_cascade_to_ed4all

All CPU, synthetic CouncilState + candidates only.
"""

from __future__ import annotations

import re

import semantik_structure.cascade as cascade
from semantik_structure.assembler.api import AssemblerConfig, assemble_document
from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.types import CouncilState, ImageCandidate, TableCandidate
from semantik_structure.structure_graph import build_structure_graph
from semantik_structure.types import FeatureBlock, RawBlock

from lib.semantik.adapter import normalize_cascade_to_ed4all
from lib.semantik.cascade_ir import build_chapters_ir


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _fb(text: str, *, is_image: bool = False, ratio: float = 1.0, size: str = "md") -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
        page_width=200.0,
        page_height=300.0,
        font_size=11.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket=size,
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=ratio,
        is_image=is_image,
    )


def _run_cascade_to_html(*, stamp_image_src: bool) -> tuple[str, dict]:
    """Drive the real arbitrate → graph → assemble → provenance → adapter path.

    Returns (final_adapter_html, normalize_output_dict). ``stamp_image_src``
    simulates the Stage-F sidecar-write pass stamping the figure region's
    ``payload['image_src']`` so the figure renders a real ``<img>`` (the
    sidecar-written case) vs the deferred-captioning text-only ``<figure>``.
    """
    # An interleaved FB stream: an image (synthetic, empty text), its caption,
    # a "Table 1:" caption, and the structurally-strong table's cells.
    feature_blocks = [
        _fb("", is_image=True),                       # 0 synthetic image FB
        _fb("Figure 1: A number line from zero."),    # 1 figure caption
        _fb("Table 1: Quantities by item."),          # 2 table caption
        _fb("Number Quantity"),                       # 3 table cell text region
    ]

    image_cand = ImageCandidate(
        kind="figure",
        bbox=(50.0, 50.0, 250.0, 250.0),
        pages=[1],
        evidence_features={},
        member_block_indices=[0],
        px_size=(400, 400),
        image_index=0,
    )
    table_cand = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 100.0),
        pages=[1],
        evidence_features={"n_rows": 4, "n_cols": 2, "bordered": False},
        member_block_indices=[3],
        cell_grid=[["Number", "Quantity"], ["1", "10"], ["2", "20"], ["3", "30"]],
        header_row_indices=[0],
        bordered=False,
    )

    state = CouncilState(outputs={})
    candidates = [image_cand, table_cand]
    decisions = arbitrate(state, candidates)
    assert [d.route for d in decisions] == ["figure", "table"], (
        "deterministic figure/table routing precondition failed"
    )

    regions = build_structure_graph(state, feature_blocks, candidates, decisions)
    assert "figure" in {r.kind for r in regions}, "no figure region formed"
    assert "table" in {r.kind for r in regions}, "no table region formed"

    if stamp_image_src:
        for region in regions:
            if region.kind == "figure":
                region.payload["image_src"] = "./t_figures/fig-0.png"

    # No Stage-6 candidate text → every region takes its deterministic
    # fallback (the exact production case for figure/table — they are never
    # generated by Stage-6, only captioned/structured deterministically).
    top_per_region = {i: None for i in range(len(regions))}
    assembled = assemble_document(
        top_per_region,
        regions,
        feature_blocks,
        council_state=state,
        runtime_mode="mock",
        config=AssemblerConfig(skip_gap_fill=True),
    )

    provenance = cascade._build_region_provenance(
        list(assembled.region_provenance),
        regions,
        feature_blocks,
        stage7_results={},
        review_verdicts=None,
    )

    chapters = build_chapters_ir(
        {"region_provenance": provenance, "heading_tree": []}
    )
    out = normalize_cascade_to_ed4all(
        {
            "chapters": chapters,
            "exit_action": "ship_with_confidence",
            "wcag_status": "passed",
        },
        pdf_stem="figtable",
    )
    return out["html"], out


def _block_roles(html: str) -> set[str]:
    return set(re.findall(r'data-dart-block-role="([^"]+)"', html))


# ---------------------------------------------------------------------------
# The regression.
# ---------------------------------------------------------------------------


def test_figure_and_table_reach_adapter_html_with_img_src():
    """The sidecar-written case: a real ``<figure><img>`` + ``<table>`` reach
    the final Ed4All adapter HTML with the correct block-roles."""
    html, _out = _run_cascade_to_html(stamp_image_src=True)

    # Figure: a real <img> with the resolved sidecar src and a non-empty alt.
    assert "<figure>" in html, "figure region dropped before the adapter HTML"
    assert "<img" in html, "figure rendered with no <img> despite a resolved src"
    assert "./t_figures/fig-0.png" in html, "sidecar src not threaded into <img>"
    # Caption-first alt: the figure's accessible name comes from its caption.
    assert "A number line" in html

    # Table: a real <table> reconstructed from the deterministic cell grid.
    assert "<table>" in html, "table region flattened to <p> before the adapter HTML"
    assert "<th scope=" in html, "table header cells not emitted with scope"
    assert "<caption>Table 1: Quantities by item.</caption>" in html
    assert "Quantity" in html and "30" in html, "table cell content lost"

    # The correct block-roles are stamped on the <section> wrappers.
    roles = _block_roles(html)
    assert "figure" in roles, f"no figure block-role (saw {sorted(roles)})"
    assert "table" in roles, f"no table block-role (saw {sorted(roles)})"


def test_figure_never_empty_when_caption_deferred():
    """The deferred-captioning case (no sidecar src): the figure ships a
    text-only ``<figure>`` carrying the caption — NEVER an empty
    ``<figure></figure>`` (the silent-drop the fix closes) — and the table
    still reconstructs from the grid."""
    html, _out = _run_cascade_to_html(stamp_image_src=False)

    assert "<figure>" in html
    # No broken <img src="">; a text-only figure with the caption as figcaption.
    assert '<img src=""' not in html
    assert "<figure></figure>" not in html, (
        "figure degraded to an EMPTY element — the drop this regression guards"
    )
    assert "<figcaption>" in html
    assert "A number line" in html

    # The table reconstructs regardless of the figure's caption mode.
    assert "<table>" in html
    assert "<caption>Table 1: Quantities by item.</caption>" in html

    roles = _block_roles(html)
    assert {"figure", "table"} <= roles, f"saw {sorted(roles)}"
