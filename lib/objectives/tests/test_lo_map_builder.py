"""Tests for the 3-level Learning Objectives Map builder + renderer.

Covers (per the task gate):
  * the 3-level blueprint is assembled (TO -> CO -> concept -> blocks);
  * instruction blocks are listed under their CO carrying a topic +
    content-type and a well-formed deep link (#anchor) into the source HTML;
  * the deep-link form mirrors the Studio source route;
  * the reusable ``render_objective_list`` helper (finding 18) lists a TO +
    each of its COs with statements;
  * the standalone renderer re-renders an existing export with no re-gen;
  * a CO with no parent TO lands in the honest "unmapped" grouping;
  * sub_objectives, when present (finding 17), drive the level-3 labels.

All fixtures are SYNTHETIC and built inline — no real course slug or export
path is pinned anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.objectives.lo_map_builder import (
    InstructionBlock,
    build_lo_map,
    deep_link_for_ref,
    extract_blocks_from_html,
    parse_dart_ref,
    render_objective_list,
    summarize_terminals,
)

# Import the standalone renderer (Courseforge/scripts is a script dir, not a
# package — load it by path).
import importlib.util

_RENDER_PATH = (
    Path(__file__).resolve().parents[3]
    / "Courseforge"
    / "scripts"
    / "rendering"
    / "render_learning_objectives_page.py"
)
_spec = importlib.util.spec_from_file_location(
    "render_learning_objectives_page", _RENDER_PATH
)
render_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_mod)


# --------------------------------------------------------------------------- #
# Synthetic fixtures (inline — never pin the real course).
# --------------------------------------------------------------------------- #

SLUG = "synthetic-101"
SRC = "synthetic-source_accessible"


def _objectives_doc() -> dict:
    """A tiny synthesized_objectives.json: 1 TO, 2 COs (one with sub-objs)."""
    return {
        "course_name": "Synthetic Algebra 101",
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Apply core reasoning to solve problems.",
                "bloom_level": "apply",
            }
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {
                        "id": "CO-01",
                        "statement": "Apply divisibility tests.",
                        "bloom_level": "apply",
                        "terminal_id": "TO-01",
                        "sub_objectives": [
                            "Divisibility by 2 and 5",
                            "Divisibility by 3",
                        ],
                    },
                    {
                        "id": "CO-02",
                        "statement": "Identify prime factors.",
                        "bloom_level": "understand",
                        "terminal_id": "TO-01",
                        "sub_objectives": [],
                    },
                ],
            }
        ],
    }


def _page_html() -> str:
    """A fixture content page: 3 blocks bound to TO-01 with source ids."""
    ref1 = f"semantik:{SRC}#anchorAAA"
    ref2 = f"semantik:{SRC}#anchorBBB"
    return f"""<!DOCTYPE html><html><body>
<section data-cf-block-id="week_01_content_01#concept_divisibility_0"
         data-cf-source-ids="{ref1},{ref2}"
         data-cf-objective-id="TO-01">
  <section data-cf-source-ids="{ref1},{ref2}"
           data-cf-block-id="week_01_content_01#concept_divisibility_0"
           data-cf-content-type="explanation" data-cf-objective-id="TO-01">
    <h2 data-cf-content-type="explanation">Understanding Divisibility</h2>
    <p>A number is divisible when...</p>
  </section>
</section>
<section data-cf-block-id="week_01_content_01#example_divisibility_1"
         data-cf-source-ids="{ref1}"
         data-cf-objective-id="TO-01">
  <section data-cf-source-ids="{ref1}"
           data-cf-block-id="week_01_content_01#example_divisibility_1"
           data-cf-content-type="example" data-cf-objective-id="TO-01">
    <h3 data-cf-content-type="example">Worked Example: Divisible by 3</h3>
    <p>Is 5625 divisible by 3?</p>
  </section>
</section>
<section data-cf-block-id="week_01_content_01#concept_primes_2"
         data-cf-source-ids="{ref2}"
         data-cf-objective-id="CO-02">
  <section data-cf-source-ids="{ref2}"
           data-cf-block-id="week_01_content_01#concept_primes_2"
           data-cf-content-type="concept" data-cf-objective-id="CO-02">
    <h2 data-cf-content-type="concept">Prime Factorization</h2>
    <p>Every integer factors into primes.</p>
  </section>
</section>
</body></html>"""


@pytest.fixture()
def export_dir(tmp_path: Path) -> Path:
    """A synthetic export root: objectives json + a content-dev page."""
    obj_dir = tmp_path / "01_learning_objectives"
    obj_dir.mkdir()
    (obj_dir / "synthesized_objectives.json").write_text(
        json.dumps(_objectives_doc()), encoding="utf-8"
    )
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    (content_dir / "week_01_content_01.html").write_text(
        _page_html(), encoding="utf-8"
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# Deep-link construction.
# --------------------------------------------------------------------------- #


def test_parse_source_ref():
    # Main path: SemantiK provenance refs parse into (src, anchor).
    assert parse_dart_ref(f"semantik:{SRC}#anchorAAA") == (SRC, "anchorAAA")
    # Legacy read-compat: the pre-SemantiK prefix is still accepted on read.
    assert parse_dart_ref(f"dart:{SRC}#anchorAAA") == (SRC, "anchorAAA")
    assert parse_dart_ref("not-a-source-ref") is None
    assert parse_dart_ref("semantik:onlysrc") is None
    assert parse_dart_ref("") is None


def test_deep_link_form_mirrors_studio_route():
    link = deep_link_for_ref(f"semantik:{SRC}#anchorAAA", SLUG)
    assert link == (
        f"/api/learn/source/{SLUG}?item_path={SRC}.html#anchorAAA"
    )
    # well-formed #anchor present
    assert link.endswith("#anchorAAA")
    assert link.startswith(f"/api/learn/source/{SLUG}?item_path=")


def test_deep_link_none_for_bad_ref():
    assert deep_link_for_ref("garbage", SLUG) is None


# --------------------------------------------------------------------------- #
# Block extraction.
# --------------------------------------------------------------------------- #


def test_extract_blocks_topic_and_deeplink():
    blocks = extract_blocks_from_html(_page_html(), "week_01_content_01", SLUG)
    # 3 distinct blocks, each bound to an objective.
    ids = {b.block_id for b in blocks}
    assert len(ids) == 3
    by_id = {b.block_id: b for b in blocks}
    concept = by_id["week_01_content_01#concept_divisibility_0"]
    # topic comes from the heading text, not the filename.
    assert concept.topic == "Understanding Divisibility"
    assert concept.content_type == "explanation"
    assert concept.objective_id == "TO-01"
    # deep link is well-formed with the source anchor.
    assert concept.deep_link.endswith("#anchorAAA")
    assert "/api/learn/source/" in concept.deep_link
    # the CO-02-bound block carries its own objective.
    primes = by_id["week_01_content_01#concept_primes_2"]
    assert primes.objective_id == "CO-02"


# --------------------------------------------------------------------------- #
# 3-level blueprint assembly.
# --------------------------------------------------------------------------- #


def test_build_lo_map_three_levels(export_dir: Path):
    obj = json.loads(
        (export_dir / "01_learning_objectives" / "synthesized_objectives.json")
        .read_text()
    )
    terminals, unmapped = build_lo_map(
        obj, export_dir / "03_content_development", slug=SLUG
    )
    assert len(terminals) == 1  # level 1: TO
    to = terminals[0]
    assert to.id == "TO-01"
    assert len(to.children) == 2  # level 2: COs
    assert not unmapped

    co1 = next(c for c in to.children if c.id == "CO-01")
    co2 = next(c for c in to.children if c.id == "CO-02")

    # level 3: concept groups. CO-01 declares sub_objectives → those are labels.
    labels1 = [c.label for c in co1.concepts]
    assert "Divisibility by 2 and 5" in labels1
    assert "Divisibility by 3" in labels1

    # CO-01 gets the TO-stamped blocks (fan-out) under its first concept.
    co1_block_ids = {
        b.block_id for c in co1.concepts for b in c.blocks
    }
    assert "week_01_content_01#concept_divisibility_0" in co1_block_ids
    assert "week_01_content_01#example_divisibility_1" in co1_block_ids

    # CO-02 binds its own CO-02 block PLUS the TO-stamped ones (fan-out);
    # it has no declared sub_objectives → fallback concept grouping by heading.
    co2_block_ids = {
        b.block_id for c in co2.concepts for b in c.blocks
    }
    assert "week_01_content_01#concept_primes_2" in co2_block_ids
    # fallback grouping produced at least one concept label.
    assert co2.concepts


def test_blocks_listed_under_co_with_topics(export_dir: Path):
    obj = json.loads(
        (export_dir / "01_learning_objectives" / "synthesized_objectives.json")
        .read_text()
    )
    terminals, _ = build_lo_map(
        obj, export_dir / "03_content_development", slug=SLUG
    )
    co1 = next(c for c in terminals[0].children if c.id == "CO-01")
    all_blocks = [b for c in co1.concepts for b in c.blocks]
    # every block carries a non-empty topic + a clean label + deep link.
    for b in all_blocks:
        assert b.topic
        # Label is "{topic} — {block_type}" with NO duplication / filler.
        label = b.label()
        assert b.topic in label
        assert "—" in label
        assert "that develops" not in label
        assert b.deep_link and "#" in b.deep_link


# --------------------------------------------------------------------------- #
# Unmapped-objective honesty.
# --------------------------------------------------------------------------- #


def test_unmapped_co_grouping(export_dir: Path):
    obj = json.loads(
        (export_dir / "01_learning_objectives" / "synthesized_objectives.json")
        .read_text()
    )
    # Orphan a CO by removing its terminal_id.
    obj["chapter_objectives"][0]["objectives"][1].pop("terminal_id")
    terminals, unmapped = build_lo_map(
        obj, export_dir / "03_content_development", slug=SLUG
    )
    assert len(unmapped) == 1
    assert unmapped[0].id == "CO-02"
    assert unmapped[0].terminal_id is None


# --------------------------------------------------------------------------- #
# Reusable objective-list renderer (finding 18).
# --------------------------------------------------------------------------- #


def test_render_objective_list_lists_to_and_cos():
    doc = _objectives_doc()
    tos = doc["terminal_objectives"]
    cos = doc["chapter_objectives"][0]["objectives"]
    html = render_objective_list(tos, cos)
    # TO present with statement.
    assert "TO-01" in html
    assert "Apply core reasoning to solve problems." in html
    # each CO present with its statement as a brief explanation.
    assert "CO-01" in html and "Apply divisibility tests." in html
    assert "CO-02" in html and "Identify prime factors." in html
    # nested: COs appear under the TO's terminal-objective wrapper.
    assert 'data-cf-objective-id="TO-01"' in html
    assert 'data-cf-objective-id="CO-01"' in html


# --------------------------------------------------------------------------- #
# Standalone renderer — re-render an existing export with no re-gen.
# --------------------------------------------------------------------------- #


def test_blueprint_page_renders_three_levels(export_dir: Path):
    out = export_dir / "learning_objectives.html"
    written = render_mod.write_learning_objectives_page(
        export_dir / "01_learning_objectives" / "synthesized_objectives.json",
        out,
        course_code="SYNTHETIC_101",
        content_dir=export_dir / "03_content_development",
        slug=SLUG,
    )
    html = written.read_text(encoding="utf-8")
    # 3 levels present in the markup.
    assert "<h3>TO-01" in html  # level 1
    assert "<h4>CO-01" in html  # level 2
    assert "<h5>" in html  # level 3 concept grouping
    # deep links into the accessible source HTML, well-formed (#anchor).
    assert f'href="/api/learn/source/{SLUG}?item_path=' in html
    assert "#anchorAAA" in html
    # block topics surface as link text.
    assert "Understanding Divisibility" in html
    # the page is a complete, single-h1 document.
    assert html.count("<h1>") == 1
    assert html.strip().endswith("</html>")


def test_blueprint_page_renders_summary_table_above_outline(export_dir: Path):
    """The blueprint page heads with a per-TO summary table, outline below it."""
    out = export_dir / "lo_with_table.html"
    render_mod.write_learning_objectives_page(
        export_dir / "01_learning_objectives" / "synthesized_objectives.json",
        out,
        course_code="SYNTHETIC_101",
        content_dir=export_dir / "03_content_development",
        slug=SLUG,
    )
    html = out.read_text(encoding="utf-8")
    # A real semantic table with a header row.
    assert "<table" in html and "<thead>" in html and "<caption>" in html
    assert "Terminal Objective Summary" in html
    for col in ("# COs", "# Instruction Blocks", "# Source Links"):
        assert col in html
    # Exactly one summary body row per TO (one TO here).
    import re as _re

    tbody = _re.search(
        r'<h2 id="summary-heading">.*?<tbody>(.*?)</tbody>', html, _re.DOTALL
    ).group(1)
    assert tbody.count("<tr>") == 1
    # The row's TO id links to the outline section's in-page anchor.
    assert '<a href="#lo-map-to-01">TO-01</a>' in html
    assert 'id="lo-map-to-01"' in html
    # The counts cells carry the deduped aggregation (3 blocks, 2 links).
    assert '<td class="text-right">2</td>' in tbody  # # COs
    assert '<td class="text-right">3</td>' in tbody  # # instruction blocks
    # The summary table appears ABOVE the detailed outline section.
    assert html.index('id="summary-heading"') < html.index('id="objectives-heading"')
    # The detailed 3-level outline is still present below.
    assert "<h3>TO-01" in html
    assert "<h4>CO-01" in html
    # Single-h1 document; the summary <h2> does not skip a level.
    assert html.count("<h1>") == 1


def test_blueprint_to_only_course_renders_zero_co_cell(export_dir: Path):
    """A TO-only course renders the table with a 0 in the # COs column."""
    doc = {
        "course_name": "TO Only",
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Master the core skill.",
             "bloom_level": "apply"}
        ],
    }
    obj_path = export_dir / "01_learning_objectives" / "to_only.json"
    obj_path.write_text(json.dumps(doc), encoding="utf-8")
    out = export_dir / "lo_to_only.html"
    render_mod.write_learning_objectives_page(
        obj_path,
        out,
        course_code="SYNTHETIC_101",
        content_dir=export_dir / "03_content_development",
        slug=SLUG,
    )
    html = out.read_text(encoding="utf-8")
    import re as _re

    tbody = _re.search(
        r'<h2 id="summary-heading">.*?<tbody>(.*?)</tbody>', html, _re.DOTALL
    ).group(1)
    assert tbody.count("<tr>") == 1
    # # COs column is 0 for a TO-only course (the proxy child is excluded).
    assert '<td class="text-right">0</td>' in tbody


def test_statements_only_fallback_when_no_content_dir(export_dir: Path):
    """No content dir → statements-only page (no deep links, still valid)."""
    out = export_dir / "lo_statements.html"
    render_mod.write_learning_objectives_page(
        export_dir / "01_learning_objectives" / "synthesized_objectives.json",
        out,
        course_code="SYNTHETIC_101",
    )
    html = out.read_text(encoding="utf-8")
    assert "<h1>Learning Objectives Map</h1>" in html
    assert "TO-01" in html and "CO-01" in html
    # no instruction-block deep links in the statements-only fallback.
    assert "/api/learn/source/" not in html


def test_blueprint_re_renders_existing_export_idempotent(export_dir: Path):
    """Re-rendering the same inputs is deterministic (no re-gen needed)."""
    out = export_dir / "lo.html"
    render_mod.write_learning_objectives_page(
        export_dir / "01_learning_objectives" / "synthesized_objectives.json",
        out,
        course_code="SYNTHETIC_101",
        content_dir=export_dir / "03_content_development",
        slug=SLUG,
    )
    first = out.read_text(encoding="utf-8")
    render_mod.write_learning_objectives_page(
        export_dir / "01_learning_objectives" / "synthesized_objectives.json",
        out,
        course_code="SYNTHETIC_101",
        content_dir=export_dir / "03_content_development",
        slug=SLUG,
    )
    assert out.read_text(encoding="utf-8") == first


# --------------------------------------------------------------------------- #
# Topic-quality hardening: skip JSON-LD / front-matter / number-only noise;
# topic comes from the heading; clean "{topic} — {type}" labels.
# --------------------------------------------------------------------------- #


def _noisy_page_html() -> str:
    """A page mixing one good heading block with three noise blocks.

    Noise = a JSON-LD ``<script>`` block, a funder/front-matter block with no
    heading, and a number-only block with no heading. Only the heading block
    should survive extraction, carrying its heading as the topic.
    """
    ref = f"semantik:{SRC}#anchorAAA"
    return f"""<!DOCTYPE html><html><body>
<section data-cf-block-id="week_01_content_01#concept_real-topic_0"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <h2 data-cf-content-type="concept">Adding Fractions</h2>
  <p>To add fractions you first find a common denominator.</p>
</section>
<section data-cf-block-id="week_01_content_01#objective_jsonld_1"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <script type="application/ld+json">
  {{"@context": "https://ed4all.dev/ns/courseforge/v1", "@type": "CoursePage"}}
  </script>
</section>
<section data-cf-block-id="week_01_content_01#objective_charles-koch-foundation_2"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <p>charles koch foundation the stuart family foundation provided support.</p>
</section>
<section data-cf-block-id="week_01_content_01#objective_numbers_3"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <p>37889005 16 62008465 84 90 69 55</p>
</section>
</body></html>"""


def test_jsonld_and_frontmatter_blocks_dropped():
    blocks = extract_blocks_from_html(
        _noisy_page_html(), "week_01_content_01", SLUG
    )
    # Only the heading block survives; the JSON-LD, funder, and number-only
    # blocks are all dropped as non-instructional noise.
    assert len(blocks) == 1
    blk = blocks[0]
    assert blk.block_id == "week_01_content_01#concept_real-topic_0"
    # Topic is the HEADING text, never the raw JSON-LD / funder / numbers.
    assert blk.topic == "Adding Fractions"
    # No noise leaked into the surviving topic.
    for noise in ("@context", "CoursePage", "koch", "foundation", "37889005"):
        assert noise not in blk.topic
    # Clean label: "{topic} — {block_type}", single type, no filler.
    assert blk.label() == "Adding Fractions — concept"


def test_topic_falls_back_to_clean_summary_when_no_heading():
    """A no-heading block with usable prose gets a short clean summary topic."""
    ref = f"semantik:{SRC}#anchorAAA"
    html = f"""<html><body>
<section data-cf-block-id="week_01_content_01#concept_nohead_0"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <li>Identify and simplify radical expressions using prime factorization.</li>
</section>
</body></html>"""
    blocks = extract_blocks_from_html(html, "week_01_content_01", SLUG)
    assert len(blocks) == 1
    # Summary is short, clean, and drawn from the prose (not the block-id slug).
    assert blocks[0].topic
    assert "nohead" not in blocks[0].topic.lower()
    assert blocks[0].topic.lower().startswith("identify")


def test_to_only_course_proxy_is_honest_not_fake_co():
    """A TO-only course renders an honest TO-direct grouping, not a fake CO."""
    doc = {
        "course_name": "TO Only",
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Master the core skill.",
             "bloom_level": "apply"}
        ],
        # No chapter_objectives at all (the Sonnet TO-only shape).
    }
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        content = Path(td) / "03_content_development"
        content.mkdir()
        ref = f"semantik:{SRC}#anchorAAA"
        (content / "week_01_content_01.html").write_text(
            f"""<html><body>
<section data-cf-block-id="week_01_content_01#concept_x_0"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <h2 data-cf-content-type="concept">Solving Equations</h2>
  <p>Move terms across the equals sign.</p>
</section>
</body></html>""",
            encoding="utf-8",
        )
        terminals, _ = build_lo_map(doc, content, slug=SLUG)
    assert terminals and terminals[0].children
    proxy = terminals[0].children[0]
    # The proxy has NO fake CO id and does NOT duplicate the TO statement.
    assert proxy.id == ""
    assert proxy.statement == "Instruction blocks for this objective"
    assert proxy.statement != terminals[0].statement
    assert proxy.block_count() == 1


# --------------------------------------------------------------------------- #
# Per-TO summary aggregation (summarize_terminals).
# --------------------------------------------------------------------------- #


def test_summarize_terminals_one_row_per_to_with_counts(export_dir: Path):
    """One summary row per TO, with deduped block + distinct-source-link counts."""
    obj = json.loads(
        (export_dir / "01_learning_objectives" / "synthesized_objectives.json")
        .read_text()
    )
    terminals, _ = build_lo_map(
        obj, export_dir / "03_content_development", slug=SLUG
    )
    rows = summarize_terminals(terminals)
    assert len(rows) == 1  # one TO -> one row
    row = rows[0]
    assert row.terminal_id == "TO-01"
    assert row.statement == "Apply core reasoning to solve problems."
    assert row.bloom_level == "apply"
    # Two real COs (CO-01, CO-02) — the proxy is absent here.
    assert row.num_cos == 2
    # Three distinct instruction blocks across the fixture page, deduped even
    # though the two TO-stamped blocks fan out across both COs.
    assert row.num_blocks == 3
    # The fixture's blocks resolve to two distinct deep links (anchorAAA /
    # anchorBBB), counted once each per TO.
    assert row.num_source_links == 2


def test_summarize_terminals_to_only_course_shows_zero_cos():
    """A TO-only course (proxy child, empty id) renders 0 COs without error."""
    doc = {
        "course_name": "TO Only",
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Master the core skill.",
             "bloom_level": "apply"}
        ],
        # No chapter_objectives at all (the Sonnet TO-only shape).
    }
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        content = Path(td) / "03_content_development"
        content.mkdir()
        ref = f"semantik:{SRC}#anchorAAA"
        (content / "week_01_content_01.html").write_text(
            f"""<html><body>
<section data-cf-block-id="week_01_content_01#concept_x_0"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <h2 data-cf-content-type="concept">Solving Equations</h2>
  <p>Move terms across the equals sign.</p>
</section>
</body></html>""",
            encoding="utf-8",
        )
        terminals, _ = build_lo_map(doc, content, slug=SLUG)
    rows = summarize_terminals(terminals)
    assert len(rows) == 1
    row = rows[0]
    # The TO-direct proxy child must NOT inflate the CO count.
    assert row.num_cos == 0
    # Its instruction blocks + source links are still counted.
    assert row.num_blocks == 1
    assert row.num_source_links == 1


def test_summarize_terminals_empty_when_no_terminals():
    """No TOs -> no summary rows (the renderer then omits the table)."""
    assert summarize_terminals([]) == []


# --------------------------------------------------------------------------- #
# Non-additive CO binding + concept-label-from-heading hardening.
# --------------------------------------------------------------------------- #


def _two_co_objectives_doc() -> dict:
    """1 TO, 2 COs (NO declared sub_objectives → fallback concept grouping)."""
    return {
        "course_name": "Demo 101",
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Reason about numbers.",
             "bloom_level": "apply"}
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "statement": "Test divisibility.",
                     "bloom_level": "apply", "terminal_id": "TO-01"},
                    {"id": "CO-02", "statement": "Find prime factors.",
                     "bloom_level": "understand", "terminal_id": "TO-01"},
                ],
            }
        ],
    }


def _per_co_page_html() -> str:
    """Two blocks, each stamped a DISTINCT CO id (block A -> CO-01, B -> CO-02)."""
    ref = f"semantik:{SRC}#anchorAAA"
    return f"""<!DOCTYPE html><html><body>
<section data-cf-block-id="week_01_content_01#concept_div_0"
         data-cf-source-ids="{ref}" data-cf-objective-id="CO-01">
  <h2 data-cf-content-type="concept">Divisibility Rules</h2>
  <p>A number is divisible by 3 when its digit sum is.</p>
</section>
<section data-cf-block-id="week_01_content_01#concept_primes_1"
         data-cf-source-ids="{ref}" data-cf-objective-id="CO-02">
  <h2 data-cf-content-type="concept">Prime Factorization</h2>
  <p>Every integer factors into primes.</p>
</section>
</body></html>"""


def test_distinct_co_stamps_no_cross_contamination(tmp_path: Path):
    """Block A stamped CO-01 and block B stamped CO-02 do NOT fan out.

    A appears only under CO-01, B only under CO-02 — no additive duplication.
    """
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    (content_dir / "week_01_content_01.html").write_text(
        _per_co_page_html(), encoding="utf-8"
    )
    terminals, unmapped = build_lo_map(
        _two_co_objectives_doc(), content_dir, slug="demo_101"
    )
    assert not unmapped
    co1 = next(c for c in terminals[0].children if c.id == "CO-01")
    co2 = next(c for c in terminals[0].children if c.id == "CO-02")

    co1_ids = {b.block_id for c in co1.concepts for b in c.blocks}
    co2_ids = {b.block_id for c in co2.concepts for b in c.blocks}

    # CO-01 has ONLY its own block; CO-02 has ONLY its own block.
    assert co1_ids == {"week_01_content_01#concept_div_0"}
    assert co2_ids == {"week_01_content_01#concept_primes_1"}
    # No cross-contamination either direction.
    assert "week_01_content_01#concept_primes_1" not in co1_ids
    assert "week_01_content_01#concept_div_0" not in co2_ids


def test_co_without_direct_blocks_falls_back_to_to_blocks(tmp_path: Path):
    """A CO with no direct blocks shows the parent TO's directly-stamped blocks.

    CO-01 has its own block; CO-02 has none of its own but the parent TO has a
    TO-stamped block → that TO block surfaces under CO-02 (fallback works).
    """
    ref = f"semantik:{SRC}#anchorAAA"
    html = f"""<!DOCTYPE html><html><body>
<section data-cf-block-id="week_01_content_01#concept_div_0"
         data-cf-source-ids="{ref}" data-cf-objective-id="CO-01">
  <h2 data-cf-content-type="concept">Divisibility Rules</h2>
  <p>Digit-sum test.</p>
</section>
<section data-cf-block-id="week_01_content_01#concept_general_1"
         data-cf-source-ids="{ref}" data-cf-objective-id="TO-01">
  <h2 data-cf-content-type="concept">Number Sense Overview</h2>
  <p>Numbers have structure.</p>
</section>
</body></html>"""
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    (content_dir / "week_01_content_01.html").write_text(html, encoding="utf-8")

    terminals, _ = build_lo_map(
        _two_co_objectives_doc(), content_dir, slug="demo_101"
    )
    co1 = next(c for c in terminals[0].children if c.id == "CO-01")
    co2 = next(c for c in terminals[0].children if c.id == "CO-02")

    co1_ids = {b.block_id for c in co1.concepts for b in c.blocks}
    co2_ids = {b.block_id for c in co2.concepts for b in c.blocks}

    # CO-01 keeps ONLY its own block (the TO block is NOT added on top).
    assert co1_ids == {"week_01_content_01#concept_div_0"}
    # CO-02 has no direct block → falls back to the TO-stamped block.
    assert co2_ids == {"week_01_content_01#concept_general_1"}


def test_concept_label_uses_heading_not_block_id_slug(tmp_path: Path):
    """The concept-group label is the block's heading text, not an id slug.

    The block-id slug would truncate to "what you will"; the cleaned heading
    "What You Will Learn Today" must win as the concept-group label.
    """
    ref = f"semantik:{SRC}#anchorAAA"
    html = f"""<!DOCTYPE html><html><body>
<section data-cf-block-id="week_01_content_01#concept_what-you-will_0"
         data-cf-source-ids="{ref}" data-cf-objective-id="CO-01">
  <h2 data-cf-content-type="concept">What You Will Learn Today</h2>
  <p>An overview of the week's plan.</p>
</section>
</body></html>"""
    doc = {
        "course_name": "Demo 101",
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Reason about numbers.",
             "bloom_level": "apply"}
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "statement": "Plan the week.",
                     "bloom_level": "understand", "terminal_id": "TO-01"},
                ],
            }
        ],
    }
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    (content_dir / "week_01_content_01.html").write_text(html, encoding="utf-8")

    terminals, _ = build_lo_map(doc, content_dir, slug="demo_101")
    co1 = terminals[0].children[0]
    labels = [c.label for c in co1.concepts]
    # The real heading text is the concept-group label.
    assert "What You Will Learn Today" in labels
    # The truncated block-id slug ("what you will") is NOT used as a label.
    assert "what you will" not in [lbl.lower() for lbl in labels]


# --------------------------------------------------------------------------- #
# Page-number deep-links (Phase 1) — thread data-dart-pages onto the LO map.
# --------------------------------------------------------------------------- #


def test_deep_link_appends_page_when_present():
    """A positive page appends ``&page=N`` BEFORE the ``#anchor`` fragment."""
    link = deep_link_for_ref(f"semantik:{SRC}#anchorAAA", SLUG, page=47)
    assert link == (
        f"/api/learn/source/{SLUG}?item_path={SRC}.html&page=47#anchorAAA"
    )
    # &page= sits before the fragment so the URL stays well-formed.
    assert "&page=47#anchorAAA" in link


def test_deep_link_page_less_byte_identical():
    """No page (None / 0 / negative) → byte-identical to the page-less URL."""
    base = deep_link_for_ref(f"semantik:{SRC}#anchorAAA", SLUG)
    assert deep_link_for_ref(f"semantik:{SRC}#anchorAAA", SLUG, page=None) == base
    assert deep_link_for_ref(f"semantik:{SRC}#anchorAAA", SLUG, page=0) == base
    assert deep_link_for_ref(f"semantik:{SRC}#anchorAAA", SLUG, page=-3) == base
    assert "&page=" not in base


def _page_html_with_pages(pages_attr: str, kind_attr: str | None = None) -> str:
    """A single grounded block carrying ``data-dart-pages`` (+ optional kind)."""
    ref = f"semantik:{SRC}#anchorAAA"
    kind_html = (
        f' data-dart-page-kind="{kind_attr}"' if kind_attr is not None else ""
    )
    return f"""<!DOCTYPE html><html><body>
<section data-cf-block-id="week_01_content_01#concept_x_0"
         data-cf-source-ids="{ref}" data-dart-pages="{pages_attr}"{kind_html}
         data-cf-content-type="explanation" data-cf-objective-id="TO-01">
  <h2 data-cf-content-type="explanation">Topic With Pages</h2>
  <p>Grounded prose.</p>
</section>
</body></html>"""


def test_block_pages_parsed_and_first_page_deep_linked():
    """``data-dart-pages="47"`` lands [47] on the block + ``&page=47`` in href."""
    blocks = extract_blocks_from_html(
        _page_html_with_pages("47"), "week_01_content_01", SLUG
    )
    blk = blocks[0]
    assert blk.pages == [47]
    assert blk.deep_link.endswith("#anchorAAA")
    assert "&page=47#anchorAAA" in blk.deep_link


def test_block_pages_range_targets_first_page():
    """RISK-C: ``data-dart-pages="3-5"`` parses to [3,4,5]; deep link targets 3."""
    blocks = extract_blocks_from_html(
        _page_html_with_pages("3-5"), "week_01_content_01", SLUG
    )
    blk = blocks[0]
    assert blk.pages == [3, 4, 5]
    assert "&page=3#anchorAAA" in blk.deep_link
    assert "&page=4" not in blk.deep_link


def test_block_without_pages_deep_link_page_less():
    """No ``data-dart-pages`` → empty pages + page-less (byte-identical) href."""
    ref = f"semantik:{SRC}#anchorAAA"
    html = f"""<!DOCTYPE html><html><body>
<section data-cf-block-id="week_01_content_01#concept_x_0"
         data-cf-source-ids="{ref}"
         data-cf-content-type="explanation" data-cf-objective-id="TO-01">
  <h2 data-cf-content-type="explanation">Topic No Pages</h2>
  <p>Grounded prose.</p>
</section>
</body></html>"""
    blocks = extract_blocks_from_html(html, "week_01_content_01", SLUG)
    blk = blocks[0]
    assert blk.pages == []
    assert "&page=" not in blk.deep_link
    assert blk.deep_link.endswith("#anchorAAA")


# --------------------------------------------------------------------------- #
# Shared page-label helper (lib/page_label) — single source of truth (OQ-2).
# --------------------------------------------------------------------------- #


def test_shared_page_label_helpers():
    from lib.page_label import pdf_page_citation, pdf_pages_label

    # pdf_pages_label keeps the legacy answer-render wording byte-identical.
    assert pdf_pages_label([12]) == "page 12"
    assert pdf_pages_label([3, 7]) == "pages 3, 7"

    # pdf_page_citation is the honest "PDF p." surface (RISK-A) for new links.
    assert pdf_page_citation([47]) == "PDF p. 47"
    assert pdf_page_citation([3, 4, 5]) == "PDF pp. 3, 4, 5"
    # Empty / no-int → "" so callers omit the suffix entirely.
    assert pdf_page_citation([]) == ""


# --------------------------------------------------------------------------- #
# Kind-aware printed-vs-physical labelling (data-dart-page-kind).
# --------------------------------------------------------------------------- #


def test_block_pages_kind_absent_defaults_physical_byte_identical():
    """No ``data-dart-page-kind`` → "physical" + byte-identical "PDF p. N"."""
    blocks = extract_blocks_from_html(
        _page_html_with_pages("47"), "week_01_content_01", SLUG
    )
    blk = blocks[0]
    assert blk.pages == [47]
    assert blk.pages_kind == "physical"
    # The kind-aware citation matches the legacy honest PDF-page label.
    assert blk.page_citation() == "PDF p. 47"


def test_block_pages_kind_printed_renders_bare_p():
    """``data-dart-page-kind="printed"`` → honest book "p. N" label."""
    blocks = extract_blocks_from_html(
        _page_html_with_pages("47", "printed"), "week_01_content_01", SLUG
    )
    blk = blocks[0]
    assert blk.pages == [47]
    assert blk.pages_kind == "printed"
    assert blk.page_citation() == "p. 47"


def test_block_pages_kind_interpolated_renders_bare_p():
    """``interpolated`` is a derived BOOK page → bare "p. N"."""
    blocks = extract_blocks_from_html(
        _page_html_with_pages("3-5", "interpolated"), "week_01_content_01", SLUG
    )
    blk = blocks[0]
    assert blk.pages == [3, 4, 5]
    assert blk.pages_kind == "interpolated"
    assert blk.page_citation() == "pp. 3, 4, 5"


def test_block_pages_kind_physical_renders_pdf_p():
    """Explicit ``physical`` → "PDF p. N" (anti-fabrication: never upgraded)."""
    blocks = extract_blocks_from_html(
        _page_html_with_pages("47", "physical"), "week_01_content_01", SLUG
    )
    blk = blocks[0]
    assert blk.pages_kind == "physical"
    assert blk.page_citation() == "PDF p. 47"


def test_block_pages_kind_unknown_falls_back_to_physical():
    """A garbage kind normalizes to "physical" → honest "PDF p. N"."""
    blocks = extract_blocks_from_html(
        _page_html_with_pages("47", "bogus"), "week_01_content_01", SLUG
    )
    blk = blocks[0]
    assert blk.pages_kind == "physical"
    assert blk.page_citation() == "PDF p. 47"
