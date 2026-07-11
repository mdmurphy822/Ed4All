"""Phase 1+2 — pure verifier digest builder + spot-HTML slicer (GPU-free).

Pure, unconsumed helpers over the assembled doc: ``build_verifier_digest``
(region_provenance-VALUE-keyed) + the ``slice_section_html`` /
``slice_region_html`` / ``enclosing_section_heading_id`` slicers. The section
slice is the PRIMARY (re-level-immune) heading/section path; everything here is
a pure read so ``assemble_document`` stays byte-identical.
"""

from __future__ import annotations

from semantik_structure.assembler.api import assemble_document, AssemblerConfig
from semantik_structure.assembler.types import AssembledDoc
from semantik_structure.assembler.verify_digest import (
    _estimate_tokens,
    _split_into_sections,
    build_verifier_digest,
    enclosing_section_heading_id,
    slice_region_html,
    slice_section_html,
    window_verifier_digest,
)
from semantik_structure.qwen_specialists.reviewer_prompt import (
    build_second_pass_verify_request,
)
from semantik_structure.soft_reranker.types import RankedCandidate
from semantik_structure.qwen_specialists.types import Candidate
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(text=text, page=page, bbox=(0.0, 0.0, 10.0, 10.0),
                   page_width=100.0, page_height=100.0)
    return FeatureBlock(raw=raw, size_bucket="md", gap_above=None,
                        is_top_of_page=False, is_centered=False, caps=None,
                        indent_bucket=0, relative_font_ratio=1.0)


def _stage6(text: str) -> RankedCandidate:
    return RankedCandidate(candidate=Candidate(adapter="prose", request_id="r", text=text), score=1.0)


def _two_heading_doc(monkeypatch=None):
    """Real assembled doc — two section headings + a worked_example whose
    embedded ``<h4>`` is re-leveled to ``<h3>`` by the final contiguity pass.
    Built under the gold shell so the ``<section aria-labelledby>`` grouping
    fires (the slicer's primary path)."""
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "Chapter One"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "Body para one."}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 3, "text": "Sub Section"}),
        Region(kind="paragraph", feature_block_indices=(3,),
               payload={"text": "EXAMPLE 1.3 solve x.", "semantic_class": "worked_example"}),
    ]
    fbs = [_fb("Chapter One"), _fb("Body para one."), _fb("Sub Section"), _fb("EXAMPLE 1.3 solve x.")]
    top = {
        0: _stage6("<h2>Chapter One</h2>"),
        1: _stage6("<p>Body para one.</p>"),
        2: _stage6("<h4>Sub Section</h4>"),
        3: _stage6("<p>EXAMPLE 1.3 solve x.</p>"),
    }
    return regions, fbs, top


def _clean_doc():
    """Two plain headings + plain paragraphs, no embedded component heading —
    so ``heading_tree`` count == ``anchors`` count (the heading-outline test)."""
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "Alpha"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "Body alpha."}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "Beta"}),
        Region(kind="paragraph", feature_block_indices=(3,), payload={"text": "Body beta."}),
    ]
    fbs = [_fb("Alpha"), _fb("Body alpha."), _fb("Beta"), _fb("Body beta.")]
    top = {
        0: _stage6("<h2>Alpha</h2>"),
        1: _stage6("<p>Body alpha.</p>"),
        2: _stage6("<h2>Beta</h2>"),
        3: _stage6("<p>Body beta.</p>"),
    }
    return regions, fbs, top


# ---------------------------------------------------------------------------
# Phase 1 — digest builder.
# ---------------------------------------------------------------------------


def test_digest_one_entry_per_emitted_region():
    regions, fbs, top = _clean_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    digest = build_verifier_digest(doc, regions, fbs, edge_tokens=6)
    # One entry per emitted (non-empty) region in region_provenance order.
    assert len(digest["regions"]) == len(doc.region_provenance)
    for entry in digest["regions"]:
        assert 0 <= entry["region_index"] < len(regions)


def test_digest_round_trips_region_index():
    regions, fbs, top = _clean_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    digest = build_verifier_digest(doc, regions, fbs, edge_tokens=6)
    for entry in digest["regions"]:
        assert regions[entry["region_index"]].kind == entry["kind"]


def test_digest_round_trips_non_identity_provenance():
    # Synthetic non-identity region_provenance: position p -> VALUE; the entry's
    # region_index is the stored VALUE, resolving to the correct region.
    regions = [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "r0"}),
        Region(kind="heading", feature_block_indices=(1,), payload={"text": "r1", "level_hint": 2}),
        Region(kind="list", feature_block_indices=(2,), payload={"text": "r2"}),
    ]
    fbs = [_fb("r0"), _fb("r1"), _fb("r2")]
    doc = AssembledDoc(
        html="<p>r2</p><p>r0</p><p>r1</p>",
        gaps_found=[], gaps_resolved=[], gaps_fallback=[],
        heading_tree=[], landmarks={}, anchors={},
        region_provenance=[2, 0, 1],
        sub_task_log={"region_html": ["<p>r2</p>", "<p>r0</p>", "<p>r1</p>"]},
    )
    digest = build_verifier_digest(doc, regions, fbs, edge_tokens=6)
    # Position 0 -> value 2 -> regions[2] (list); pos 1 -> 0 (paragraph); pos 2 -> 1 (heading).
    assert [e["region_index"] for e in digest["regions"]] == [2, 0, 1]
    assert [e["kind"] for e in digest["regions"]] == ["list", "paragraph", "heading"]
    for entry in digest["regions"]:
        assert regions[entry["region_index"]].kind == entry["kind"]


def test_runtime_shape_contract():
    regions, fbs, top = _clean_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    # anchors values are int (the stale dict[str,str] annotation is fixed).
    assert doc.anchors, "fixture should produce at least one anchored heading"
    for v in doc.anchors.values():
        assert isinstance(v, int)
    # sub_task_log['region_html'] is a list index-aligned with region_provenance.
    rh = doc.sub_task_log["region_html"]
    assert isinstance(rh, list)
    assert len(rh) == len(doc.region_provenance)


def test_digest_carries_semantic_class_and_aria_wrapper():
    regions, fbs, top = _two_heading_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    digest = build_verifier_digest(doc, regions, fbs, edge_tokens=6)
    by_idx = {e["region_index"]: e for e in digest["regions"]}
    # The worked_example region carries its semantic_class + a non-None wrapper.
    we = by_idx[3]
    assert we["semantic_class"] == "worked_example"
    assert we["aria_wrapper"] is not None
    # A bare paragraph (no semantic_class) -> aria_wrapper None.
    para = by_idx[1]
    assert para["semantic_class"] is None
    assert para["aria_wrapper"] is None


def test_heading_outline_from_heading_tree():
    regions, fbs, top = _clean_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    digest = build_verifier_digest(doc, regions, fbs, edge_tokens=6)
    outline = digest["heading_outline"]
    assert len(outline) == len(doc.heading_tree)
    for node in outline:
        assert node["heading_id"] in doc.anchors
        assert isinstance(node["region_index"], int)


def test_metadata_drop_empty_marked():
    # A metadata_drop region emits an empty fragment -> SKIPPED, not a phantom.
    regions = [
        Region(kind="metadata_drop", feature_block_indices=(0,), payload={"text": "running header"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "Real body."}),
    ]
    fbs = [_fb("running header"), _fb("Real body.")]
    top = {0: _stage6(""), 1: _stage6("<p>Real body.</p>")}
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    digest = build_verifier_digest(doc, regions, fbs, edge_tokens=6)
    # The empty metadata_drop region (idx 0) is absent; only the paragraph remains.
    idxs = [e["region_index"] for e in digest["regions"]]
    assert 0 not in idxs
    assert 1 in idxs


def test_assembler_output_unchanged():
    # Building the digest does not perturb assemble_document (no consumer).
    regions, fbs, top = _clean_doc()
    html_a = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True)).html
    regions, fbs, top = _clean_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    _ = build_verifier_digest(doc, regions, fbs, edge_tokens=6)
    regions, fbs, top = _clean_doc()
    html_b = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True)).html
    assert html_a == html_b


# ---------------------------------------------------------------------------
# Phase 2 — spot-HTML slicer.
# ---------------------------------------------------------------------------


def test_section_slice_spans_heading_to_close(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    regions, fbs, top = _two_heading_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    sec = slice_section_html(doc, "sub-section")
    assert sec.startswith('<section aria-labelledby="sub-section">')
    assert sec.rstrip().endswith("</section>")
    assert 'id="sub-section"' in sec  # the heading
    assert "EXAMPLE 1.3 solve x." in sec  # its following block


def test_section_slice_survives_heading_relevel(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    regions, fbs, top = _two_heading_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    # The worked_example's persisted region_html carries the PRE-relevel <h4>;
    # the final HTML re-leveled it to <h3>, so a direct substring search MISSES.
    region_html = doc.sub_task_log["region_html"]
    we_stash = region_html[3]
    assert "<h4 " in we_stash
    assert we_stash not in doc.html  # documented substring miss after relevel
    # The level-agnostic section slice still returns the enclosing section.
    sec = slice_section_html(doc, "sub-section")
    assert sec
    assert "EXAMPLE 1.3 solve x." in sec


def test_enclosing_section_resolves_content_block(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    regions, fbs, top = _two_heading_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    # The worked_example content region (idx 3) -> its nearest preceding heading.
    assert enclosing_section_heading_id(doc, 3) == "sub-section"
    # The first body paragraph (idx 1) sits under "chapter-one".
    assert enclosing_section_heading_id(doc, 1) == "chapter-one"


def test_enclosing_section_pre_first_heading_is_none():
    # A region with no preceding heading -> None (front matter / pre-first-heading).
    doc = AssembledDoc(
        html="<p>r0</p>", gaps_found=[], gaps_resolved=[], gaps_fallback=[],
        heading_tree=[], landmarks={},
        anchors={"sec-1": 1},  # the heading is AFTER region 0
        region_provenance=[0, 1],
        sub_task_log={"region_html": ["<p>r0</p>", "<h2 id='sec-1'>H</h2>"]},
    )
    assert enclosing_section_heading_id(doc, 0) is None


def test_nonheading_region_slice_is_substring_of_final_html():
    regions, fbs, top = _clean_doc()
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    # Every non-heading body region's slice is a verbatim substring of final HTML.
    for region_index in (1, 3):
        sliced = slice_region_html(doc, region_index)
        assert sliced
        assert sliced in doc.html


def test_gap_region_pulls_from_final_html():
    # A region whose stash IS present in the final html returns that located
    # slice (the post-wrap bytes), demonstrating the find-based resolution.
    regions = [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "Body."}),
    ]
    fbs = [_fb("Body.")]
    top = {0: _stage6("<p>Body.</p>")}
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    sliced = slice_region_html(doc, 0)
    assert sliced
    assert sliced in doc.html  # pulled from final html, not a phantom


def test_empty_region_returns_empty():
    doc = AssembledDoc(
        html="<p>kept</p>", gaps_found=[], gaps_resolved=[], gaps_fallback=[],
        heading_tree=[], landmarks={}, anchors={},
        region_provenance=[0, 1],
        sub_task_log={"region_html": ["", "<p>kept</p>"]},
    )
    assert slice_region_html(doc, 0) == ""  # metadata_drop / empty -> '' (no raise)


# ---------------------------------------------------------------------------
# Section-aligned digest windowing (window_verifier_digest).
# ---------------------------------------------------------------------------


def _entry(region_index, kind, head="h", tail="t", semantic_class=None):
    return {
        "region_index": region_index,
        "kind": kind,
        "semantic_class": semantic_class,
        "aria_wrapper": None,
        "head": head,
        "tail": tail,
    }


def _section_digest(n_sections=3):
    """A digest of ``n_sections`` heading+paragraph sections (uniform size)."""
    regions = []
    outline = []
    for s in range(n_sections):
        h_idx, p_idx = 2 * s, 2 * s + 1
        regions.append(_entry(h_idx, "heading", head=f"Head{s}", tail=f"Head{s}"))
        regions.append(_entry(p_idx, "paragraph", head=f"Body{s}", tail=f"Body{s}"))
        outline.append({"region_index": h_idx, "level": 2, "text": f"Head{s}",
                        "heading_id": f"head-{s}"})
    return {"regions": regions, "heading_outline": outline}


def _one_section_budget(digest):
    """The token budget that fits exactly ONE section (so each section becomes
    its own window): the largest section's token estimate."""
    return max(_estimate_tokens(sec) for sec in _split_into_sections(digest["regions"]))


def _window_of(windows):
    """region_index -> window position map across a windows list."""
    out = {}
    for wpos, win in enumerate(windows):
        for e in win["regions"]:
            out[e["region_index"]] = wpos
    return out


def test_window_packs_whole_sections():
    digest = _section_digest(3)
    budget = _one_section_budget(digest)
    windows = window_verifier_digest(digest, window_tokens=budget)
    # One window per whole section; a heading and its paragraph never split.
    assert len(windows) == 3
    for win in windows:
        ids = [e["region_index"] for e in win["regions"]]
        # Each window is exactly one heading (even idx) + its paragraph (odd).
        assert ids == [ids[0], ids[0] + 1]
        assert win["regions"][0]["kind"] == "heading"


def test_window_never_splits_section():
    digest = _section_digest(4)
    budget = _one_section_budget(digest) * 2 + 1  # ~2 sections / window
    windows = window_verifier_digest(digest, window_tokens=budget)
    pos = _window_of(windows)
    # Every section's heading + content land in the SAME window.
    for sec in _split_into_sections(digest["regions"]):
        win_positions = {pos[e["region_index"]] for e in sec}
        assert len(win_positions) == 1, "a section was split across windows"


def test_window_front_matter_initial():
    # Two pre-first-heading entries form an initial front-matter pseudo-section.
    regions = [
        _entry(0, "paragraph", head="Front0"),
        _entry(1, "paragraph", head="Front1"),
        _entry(2, "heading", head="Head0"),
        _entry(3, "paragraph", head="Body0"),
    ]
    digest = {"regions": regions, "heading_outline": [
        {"region_index": 2, "level": 2, "text": "Head0", "heading_id": "head-0"}]}
    budget = _one_section_budget(digest)
    windows = window_verifier_digest(digest, window_tokens=budget)
    # The first window is the front-matter section (no heading entry in it).
    first_ids = [e["region_index"] for e in windows[0]["regions"]]
    assert first_ids == [0, 1]
    assert all(e["kind"] != "heading" for e in windows[0]["regions"])


def test_window_over_budget_section_subsplits():
    # One giant section (heading + 5 content) that alone exceeds the budget ->
    # entry-range sub-split, no window over budget, no entries dropped.
    regions = [_entry(0, "heading", head="H", tail="H")]
    for i in range(1, 6):
        regions.append(_entry(i, "paragraph", head=f"Body{i}", tail=f"Tail{i}"))
    digest = {"regions": regions, "heading_outline": [
        {"region_index": 0, "level": 2, "text": "H", "heading_id": "h"}]}
    per_entry = max(_estimate_tokens(e) for e in regions)
    budget = per_entry * 2 + 1  # ~2 entries per slice, section (6) far exceeds it
    windows = window_verifier_digest(digest, window_tokens=budget)
    assert len(windows) > 1  # the section was sub-split
    seen = []
    for win in windows:
        # No window exceeds the budget (every entry here is < budget).
        assert _estimate_tokens(win["regions"]) <= budget
        seen.extend(e["region_index"] for e in win["regions"])
    # No entry dropped: the union is every region.
    assert sorted(seen) == [0, 1, 2, 3, 4, 5]


def test_window_outline_capped():
    # A huge full-doc outline (referencing out-of-window headings) -> capped at
    # outline_token_cap with a truncation marker.
    regions = [_entry(0, "heading", head="H"), _entry(1, "paragraph", head="B")]
    outline = [{"region_index": 1000 + i, "level": 2, "text": f"Heading number {i}",
                "heading_id": f"head-{i}"} for i in range(200)]
    digest = {"regions": regions, "heading_outline": outline}
    windows = window_verifier_digest(digest, window_tokens=100000, outline_token_cap=40)
    assert len(windows) == 1
    capped = windows[0]["heading_outline"]
    assert _estimate_tokens(capped) <= 40
    assert capped[-1].get("outline_truncated") is True
    assert capped[-1]["omitted_headings"] == 200


def test_window_allowed_ids_are_window_local():
    digest = _section_digest(3)
    budget = _one_section_budget(digest)
    windows = window_verifier_digest(digest, window_tokens=budget)
    assert len(windows) == 3
    for win in windows:
        entry_ids = sorted(e["region_index"] for e in win["regions"])
        assert win["allowed_region_indices"] == entry_ids


def test_single_window_equiv_to_full_digest(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    digest = _section_digest(2)
    windows = window_verifier_digest(digest, window_tokens=1_000_000)
    assert len(windows) == 1
    # The single window's prompt is byte-identical to the non-windowed digest.
    assert build_second_pass_verify_request(windows[0]) == \
        build_second_pass_verify_request(digest)


def test_window_empty_digest_returns_empty():
    assert window_verifier_digest({"regions": [], "heading_outline": []},
                                  window_tokens=8000) == []
