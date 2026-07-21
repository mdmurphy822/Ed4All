"""Regression: per-chunk source provenance is SECTION-scoped, not whole-document.

Root cause (pre-fix): ``chunk_text_block`` built its block-probe index over the
whole item / chapter HTML, so the one-way containment match attributed a block
to a chunk whenever the block's leading ≤8-word probe appeared ANYWHERE in the
chunk's text. Generic apparatus probes (``"Solution ..."``) and repeated
instructional stems recur verbatim across a chapter, so a Section-1.1 chunk
smeared refs across every distant section that shared the phrasing (median ~54
refs/chunk on the real corpus).

The fix (THRESHOLD-FREE): the probe index is built over the section's OWN
source-block range (``_section_block_scope_html`` — heading → next-heading
boundary), so a chunk can only ever be attributed blocks that belong to its own
synthesized section. No locality distance threshold: the scope is the
structural section boundary.

Contract under test:
  - An out-of-section ``"Solution"`` apparatus block AND a distant repeated
    instructional stem block (in OTHER sections) do NOT attach to an unrelated
    chunk.
  - A short ``"Solution"`` block WITHIN the chunk's own section DOES attach and
    carries its role/WCAG enrichment (the block-enrichment contract is
    preserved — only OUT-of-section blocks are dropped).

Driven through ``Trainforge.chunker.chunk_content`` with a recording callback
and a synthetic multi-section SemantiK page (no real course identifiers). The
provenance harvest dual-reads both ``data-semantik-*`` (current emit) and the
legacy ``data-dart-*`` block-attribute spellings; the main fixture uses the
current spelling and one legacy-compat test pins the ``data-dart-*`` read.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.chunker import ChunkerContext, chunk_content  # noqa: E402
from Trainforge.chunker.helpers import (  # noqa: E402
    build_semantik_block_offset_index,
    resolve_dart_refs_for_chunk,
)

# Generic phrasings that recur verbatim across sections — exactly the smear
# vectors the whole-document probe index over-matched on.
_STEM = "Follow these listed numbered steps to solve the stated problem carefully now"
_SOLUTION = "Solution the worked answer equals forty two exactly here every time"


def _block(
    bid: str,
    pages: str,
    heading_html: str,
    body: str,
    *,
    role: str = "paragraph",
    attr: str = "semantik",
) -> str:
    return (
        f'<section class="{attr}-section" data-{attr}-block-id="{bid}" '
        f'data-{attr}-pages="{pages}" data-{attr}-block-role="{role}" '
        f'data-{attr}-wcag="passed">{heading_html}<p>{body}</p></section>'
    )


def _multi_section_item(attr: str = "semantik") -> Dict[str, Any]:
    """Two well-separated headed sections, each with a short 'Solution' block.

    Each headed content block clears the ~100-word merge floor and any two
    together exceed the ~800-word ceiling, so the two sections stay two separate
    chunks. Both sections carry a SHORT (headingless) 'Solution' apparatus block
    and a repeated instructional stem block sharing IDENTICAL leading probes —
    so a whole-document index would attribute BOTH sections' apparatus to EACH
    chunk (the smear); a section-scoped index attributes only the in-section
    apparatus.
    """
    alpha_body = "Mitochondria " + " ".join(f"alpha{n}" for n in range(440)) + " " + _STEM
    beta_body = "Chloroplasts " + " ".join(f"beta{n}" for n in range(440)) + " " + _STEM
    raw_html = (
        "<html><body><main>"
        + _block("blk_alpha", "2", "<h2>Cellular Respiration</h2>", alpha_body, attr=attr)
        + _block("sol_alpha", "2", "", _SOLUTION, role="worked_solution", attr=attr)
        + _block("stem_alpha", "3", "", _STEM, attr=attr)
        + _block("blk_beta", "40", "<h2>Photosynthesis Overview</h2>", beta_body, attr=attr)
        + _block("sol_beta", "41", "", _SOLUTION, role="worked_solution", attr=attr)
        + _block("stem_beta", "41", "", _STEM, attr=attr)
        + "</main></body></html>"
    )

    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parsed = HTMLContentParser().parse(raw_html)
    return {
        "module_id": "mod1",
        "module_title": "Mod 1",
        "item_id": "synthetic-multi-section",
        "title": parsed.title or "Synthetic Page",
        "resource_type": "page",
        "item_path": "synthetic-multi-section.html",
        "raw_html": raw_html,
        "sections": parsed.sections,
        "learning_objectives": [],
    }


def _recording_ctx(records: List[Dict[str, Any]]) -> ChunkerContext:
    def create_chunk(*, chunk_id, text, html, item, section_heading,
                     chunk_type, follows_chunk_id=None, position_in_module=0,
                     html_xpath=None, char_span=None, section_source_ids=None,
                     merged_headings=None, **kwargs):
        refs = kwargs.get("dart_source_refs") or []
        records.append({
            "chunk_id": chunk_id,
            "heading": section_heading,
            "block_ids": [r["block_id"] for r in refs],
            "roles": {r["block_id"]: r.get("block_role") for r in refs},
        })
        return {"id": chunk_id, "text": text, "source": {}}

    return ChunkerContext(create_chunk=create_chunk)


def test_out_of_section_apparatus_does_not_smear_into_unrelated_chunk() -> None:
    records: List[Dict[str, Any]] = []
    chunk_content(
        [_multi_section_item()], "TEST_101", ctx=_recording_ctx(records)
    )
    assert len(records) == 2, f"expected two section chunks, got {records}"

    by_heading = {r["heading"]: r for r in records}
    alpha = by_heading["Cellular Respiration"]
    beta = by_heading["Photosynthesis Overview"]

    # (1) In-section 'Solution' block IS attributed AND carries enrichment.
    assert "sol_alpha" in alpha["block_ids"], (
        f"in-section Solution block dropped: {alpha['block_ids']}"
    )
    assert alpha["roles"].get("sol_alpha") == "worked_solution", (
        "in-section Solution block lost its role enrichment: "
        f"{alpha['roles']}"
    )

    # (2) The OTHER section's identical-probe apparatus does NOT smear in.
    for foreign in ("sol_beta", "stem_beta", "blk_beta"):
        assert foreign not in alpha["block_ids"], (
            f"cross-section smear: {foreign} attached to the Alpha chunk "
            f"({alpha['block_ids']})"
        )
    for foreign in ("sol_alpha", "stem_alpha", "blk_alpha"):
        assert foreign not in beta["block_ids"], (
            f"cross-section smear: {foreign} attached to the Beta chunk "
            f"({beta['block_ids']})"
        )

    # (3) Every attributed block genuinely belongs to that section's block range
    #     (block_id namespaced by its section).
    assert all(bid.endswith("_alpha") for bid in alpha["block_ids"]), (
        alpha["block_ids"]
    )
    assert all(bid.endswith("_beta") for bid in beta["block_ids"]), (
        beta["block_ids"]
    )


def test_legacy_data_dart_attrs_section_scoped() -> None:
    """Legacy-compat: the same section-scoping contract holds when the source
    blocks carry the legacy ``data-dart-*`` attribute spelling (dual-read)."""
    records: List[Dict[str, Any]] = []
    chunk_content(
        [_multi_section_item(attr="dart")], "TEST_101", ctx=_recording_ctx(records)
    )
    assert len(records) == 2, f"expected two section chunks, got {records}"
    by_heading = {r["heading"]: r for r in records}
    alpha = by_heading["Cellular Respiration"]
    assert "sol_alpha" in alpha["block_ids"]
    assert "sol_beta" not in alpha["block_ids"]


def test_whole_document_index_would_have_smeared() -> None:
    """Pin the bug: over the WHOLE-document index the identical 'Solution' probe
    resolves to BOTH sections' apparatus, proving the section scope is what
    eliminates the smear (not the probe logic)."""
    item = _multi_section_item()
    whole_doc_index = build_semantik_block_offset_index(item["raw_html"])
    # Alpha's chunk text contains the shared 'Solution ...' phrase, so over the
    # whole-document index it matches sol_alpha AND the distant sol_beta.
    alpha_text = "Mitochondria alpha0 alpha1 " + _SOLUTION + " " + _STEM
    smeared = {
        r["block_id"] for r in resolve_dart_refs_for_chunk(whole_doc_index, alpha_text)
    }
    assert {"sol_alpha", "sol_beta"} <= smeared, (
        "expected the whole-document index to over-match both Solution blocks "
        f"(the bug being fixed): {smeared}"
    )
