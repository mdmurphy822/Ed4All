"""Phase 7 — per-region gold-standard ARIA wrap keyed by ``payload['semantic_class']``.

The assembler wraps each region's selected HTML fragment in its gold ARIA
container (gated on ``SEMANTIK_GOLD_SHELL``, default off, flag-off
byte-identical). Each ``role=region`` / labelled-section / nav container mints
its OWN resolvable inner label heading id+text from the catalog, so a flag-on
render is valid ARIA even before Phase 9's outer-``section`` grouping. CPU-only,
no model load.
"""

from __future__ import annotations

import re

import pytest

from dart_semantic.assembler.gold_shell_markup import (
    _wrap_semantic_class,
    collect_doc_ids,
)
from dart_semantic.assembler.api import assemble_document, AssemblerConfig
from dart_semantic.assembler.semantic_catalog import label_for
from dart_semantic.soft_reranker.types import RankedCandidate
from dart_semantic.qwen_specialists.types import Candidate
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        page_width=100.0,
        page_height=100.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _region(idx: int, semantic_class: str | None = None, *, kind: str = "paragraph") -> Region:
    payload: dict = {"text": f"body {idx}"}
    if semantic_class is not None:
        payload["semantic_class"] = semantic_class
    return Region(kind=kind, feature_block_indices=(idx,), payload=payload)


def _stage6(text: str) -> RankedCandidate:
    """A stage6-lane top-1 candidate carrying the given HTML fragment."""
    return RankedCandidate(
        candidate=Candidate(adapter="prose", request_id="r", text=text),
        score=1.0,
    )


def _assemble(top_per_region, regions, feature_blocks):
    return assemble_document(
        top_per_region,
        regions,
        feature_blocks,
        config=AssemblerConfig(skip_gap_fill=True),
    )


# ---------------------------------------------------------------------------
# Unit tests on ``_wrap_semantic_class``
# ---------------------------------------------------------------------------


def test_definition_region_wraps_div_role_region():
    region = _region(0, "definition_region")
    out = _wrap_semantic_class("<p>A formal definition.</p>", region, doc_ids=set())
    assert '<div class="definition" role="region" aria-labelledby=' in out
    assert out.endswith("</div>")
    assert "<p>A formal definition.</p>" in out


def test_worked_example_wraps_algorithm_region():
    region = _region(2, "worked_example")
    out = _wrap_semantic_class("<p>EXAMPLE 1.3 solve x.</p>", region, doc_ids=set())
    # The EXAMPLE render — an .algorithm/.worked-example role=region container,
    # NOT an <h1> over-promotion.
    assert 'class="algorithm worked-example"' in out
    assert 'role="region"' in out
    assert "<h1" not in out
    # Catalog label, not the raw class token.
    assert ">Example</h4>" in out


def test_callout_warning_wraps_panel():
    region = _region(4, "callout_warning")
    out = _wrap_semantic_class("<p>Heed this.</p>", region, doc_ids=set())
    assert out.startswith('<div class="callout callout-warning">')
    assert "<h4>Warning</h4>" in out
    # Callouts carry an h4 label, no minted id / aria-labelledby.
    assert "aria-labelledby" not in out


def test_toc_objectives_exercise_abstract_wrap():
    # The four previously-inert classes each render their container — never a
    # bare <p>.
    cases = {
        "toc": ('<nav class="toc"', "</nav>"),
        "objectives": ('<section class="objectives"', "</section>"),
        "exercise": ('<div class="exercise"', "</div>"),
        "abstract": ('<section class="abstract"', "</section>"),
    }
    for i, (cls, (open_sig, close_tag)) in enumerate(cases.items()):
        region = _region(10 + i, cls)
        out = _wrap_semantic_class("<p>inert content</p>", region, doc_ids=set())
        assert out.startswith(open_sig), (cls, out)
        assert out.endswith(close_tag), (cls, out)
        # Never drops to a bare <p> as the outer element.
        assert not out.startswith("<p>"), cls


def test_container_aria_labelledby_resolves_inline():
    # Every minted aria-labelledby points at a non-empty-named id that EXISTS
    # in the wrapped fragment — no dangling reference (Phase-9-independent).
    for cls in ("definition_region", "worked_example", "references", "exercise",
                "abstract", "toc", "objectives"):
        out = _wrap_semantic_class("<p>x</p>", _region(0, cls), doc_ids=set())
        m = re.search(r'aria-labelledby="([^"]+)"', out)
        assert m is not None, cls
        target = m.group(1)
        assert target.strip(), cls  # non-empty-named
        # The id is present IN the wrapped fragment on a heading carrying text.
        hm = re.search(rf'<h\d id="{re.escape(target)}">([^<]+)</h\d>', out)
        assert hm is not None and hm.group(1).strip(), cls


def test_container_label_text_from_catalog():
    # The synthesized container heading text equals the catalog label.
    region = _region(0, "worked_example")
    out = _wrap_semantic_class("<p>x</p>", region, doc_ids=set())
    label = label_for("worked_example")
    assert label == "Example"
    assert f">{label}</h4>" in out


def test_figure_unchanged():
    # figure reuses fallback_figure (already emitted) — no double wrap.
    frag = '<figure><img src="" alt="A graph"></figure>'
    region = _region(0, "figure", kind="figure")
    out = _wrap_semantic_class(frag, region, doc_ids=set())
    assert out == frag


def test_none_and_empty_fragment_passthrough():
    # semantic_class None -> unchanged.
    assert _wrap_semantic_class("<p>x</p>", _region(0, None), doc_ids=set()) == "<p>x</p>"
    # Empty / whitespace fragment -> unchanged even with a class set.
    assert _wrap_semantic_class("", _region(0, "abstract"), doc_ids=set()) == ""
    assert _wrap_semantic_class("   ", _region(0, "abstract"), doc_ids=set()) == "   "


def test_code_region_wrap_and_idempotent():
    region = _region(0, "code_region")
    out = _wrap_semantic_class("<code>x = 1</code>", region, doc_ids=set())
    assert out == '<pre role="region"><code>x = 1</code></pre>'
    # Already a <pre> -> not re-wrapped.
    pre = "<pre><code>y = 2</code></pre>"
    assert _wrap_semantic_class(pre, region, doc_ids=set()) == pre


def test_container_label_ids_unique_across_regions():
    # Two definition_regions sharing a doc_ids set get DISTINCT label ids.
    doc_ids: set[str] = set()
    a = _wrap_semantic_class("<p>a</p>", _region(0, "definition_region"), doc_ids=doc_ids)
    b = _wrap_semantic_class("<p>b</p>", _region(0, "definition_region"), doc_ids=doc_ids)
    id_a = re.search(r'aria-labelledby="([^"]+)"', a).group(1)
    id_b = re.search(r'aria-labelledby="([^"]+)"', b).group(1)
    assert id_a != id_b
    assert id_b == f"{id_a}-2"


# ---------------------------------------------------------------------------
# Integration tests via ``assemble_document`` — BOTH lanes + flag gating
# ---------------------------------------------------------------------------


def test_both_lanes_wrap(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("h"), _fb("def body"), _fb("ex body")]
    regions = [
        _region(0, None, kind="heading"),
        _region(1, "definition_region"),
        _region(2, "worked_example"),
    ]
    top = {
        0: _stage6("<h2>Title</h2>"),
        # Stage6 lane: a re-typed worked_example returns via the prose seat.
        2: _stage6("<p>EXAMPLE worked.</p>"),
        # Fallback lane: top-1 is None -> emit_fallback runs.
        1: None,
    }
    doc = _assemble(top, regions, fbs)
    html = doc.html
    # Fallback-lane definition_region wrapped.
    assert '<div class="definition" role="region"' in html
    # Stage6-lane worked_example wrapped (covers the prose-seat lane).
    assert 'class="algorithm worked-example"' in html


def test_no_double_nesting_on_stage6_lane(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("def stage6"), _fb("def fallback")]
    regions = [
        _region(0, "definition_region"),
        _region(1, "definition_region"),
    ]
    # Region 0 (stage6 lane): the LLM already emitted the gold wrapper.
    prewrapped = (
        '<div class="definition" role="region" aria-labelledby="dart-definition_region-0">'
        '<h4 id="dart-definition_region-0">Definition</h4><p>already wrapped</p></div>'
    )
    top = {
        0: _stage6(prewrapped),
        1: None,  # fallback lane -> wrapped once by the assembler
    }
    doc = _assemble(top, regions, fbs)
    html = doc.html
    # Exactly TWO definition containers total (one per region), NOT nested:
    # the pre-wrapped one is not re-wrapped, the fallback one is wrapped once.
    assert html.count('class="definition"') == 2
    # No nested definition div inside a definition div.
    assert "<div class=\"definition\"><div class=\"definition\"" not in html.replace(" role", "ROLE")


def test_flag_off_byte_identical(monkeypatch):
    fbs = [_fb("h"), _fb("def body"), _fb("ex body")]
    regions = [
        _region(0, None, kind="heading"),
        _region(1, "definition_region"),
        _region(2, "worked_example"),
    ]
    top = {
        0: _stage6("<h2>Title</h2>"),
        2: _stage6("<p>EXAMPLE worked.</p>"),
        1: None,
    }

    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    absent = _assemble(top, regions, fbs).html

    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "off")
    off = _assemble(top, regions, fbs).html

    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "0")
    zero = _assemble(top, regions, fbs).html

    assert absent == off == zero
    # And the class-bearing regions did NOT get wrapped when off.
    assert 'class="definition"' not in absent
    assert "worked-example" not in absent


def test_collect_doc_ids_seeds_from_fragments():
    frags = ['<h2 id="intro">A</h2>', '<p>no id</p>', '<div id="box">x</div>']
    assert collect_doc_ids(frags) == {"intro", "box"}


# ---------------------------------------------------------------------------
# Phase 8 — gold-standard document shell + <style> + JSON-LD + landmarks
# ---------------------------------------------------------------------------

from dart_semantic.assembler.shell import (  # noqa: E402
    DOC_CLOSE,
    DOC_OPEN,
    TITLE_SLOT_SENTINEL,
    build_shell,
)
from dart_semantic.assembler.pass_9c import _splice_missing_title  # noqa: E402
from dart_semantic.gates.hard_document import _check_html5_wellformed  # noqa: E402


def test_gold_shell_injects_style_and_jsonld(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    doc_open, _ = build_shell(lang="en", title="Doc")
    # The injected <style> carries the real WCAG component classes.
    assert "<style>" in doc_open and "</style>" in doc_open
    assert ".definition" in doc_open
    assert ".callout-warning" in doc_open
    # The schema.org accessibility JSON-LD landed in the head.
    assert '<script type="application/ld+json">' in doc_open
    assert '"@context": "https://schema.org"' in doc_open
    # role=main present (the gold <main>).
    assert '<main id="main-content" role="main">' in doc_open


def test_gold_shell_emits_footer_contentinfo(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    _, doc_close = build_shell(lang="en", title="Doc")
    assert '<footer role="contentinfo">' in doc_close
    assert "</article>" in doc_close
    assert "</main>" in doc_close


def test_gold_doc_open_format_contract_unchanged(monkeypatch):
    # The same (lang=, title=) kwargs drive build_shell in BOTH branches.
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    off_open, off_close = build_shell(lang="es", title="Título & <x>")
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    on_open, on_close = build_shell(lang="es", title="Título & <x>")
    # Identical kwargs accepted; lang + escaped title rendered in both.
    for out in (off_open, on_open):
        assert '<html lang="es">' in out
        assert "Título &amp; &lt;x&gt;" in out
    # Branches differ (gold adds style/jsonld/footer), proving the gold branch
    # is selected rather than a no-op.
    assert on_open != off_open
    assert on_close != off_close


def test_title_slot_sentinel_preserved(monkeypatch):
    # The exact sentinel survives in BOTH branches, and a _splice_missing_title
    # round-trip still lands the <h1> in the gold shell.
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    off_open, _ = build_shell(lang="en", title="Doc")
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    on_open, _ = build_shell(lang="en", title="Doc")
    assert TITLE_SLOT_SENTINEL in off_open
    assert TITLE_SLOT_SENTINEL in on_open
    # The sentinel string itself is the literal pass_9c splice key.
    assert TITLE_SLOT_SENTINEL == "<!-- DART_TITLE_SLOT -->\n"
    # Round-trip the title splice on the gold shell.
    candidate = '<title>Real Title</title><h1>Real Title</h1>'
    spliced = _splice_missing_title(on_open, candidate)
    assert "<!-- DART_TITLE_SLOT -->" not in spliced
    assert "<h1" in spliced and "Real Title</h1>" in spliced


def test_landmarks_dict_reports_gold_footer(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("h"), _fb("body")]
    regions = [_region(0, None, kind="heading"), _region(1, None)]
    top = {0: _stage6("<h2>Title</h2>"), 1: _stage6("<p>body</p>")}
    doc = _assemble(top, regions, fbs)
    assert doc.landmarks["footer"] == 1
    assert doc.landmarks["main"] == 1


def test_minimal_shell_byte_identical_flag_off(monkeypatch):
    # build_shell output byte-identical with SEMANTIK_GOLD_SHELL absent vs off.
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    absent_open, absent_close = build_shell(lang="en", title="Doc")
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "off")
    off_open, off_close = build_shell(lang="en", title="Doc")
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "0")
    zero_open, zero_close = build_shell(lang="en", title="Doc")
    assert absent_open == off_open == zero_open
    assert absent_close == off_close == zero_close
    # And byte-identical to the untouched minimal constants.
    assert absent_open == DOC_OPEN.format(lang="en", title="Doc")
    assert absent_close == DOC_CLOSE


def test_landmarks_dict_byte_identical_flag_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    fbs = [_fb("h"), _fb("body")]
    regions = [_region(0, None, kind="heading"), _region(1, None)]
    top = {0: _stage6("<h2>Title</h2>"), 1: _stage6("<p>body</p>")}
    doc = _assemble(top, regions, fbs)
    assert doc.landmarks["footer"] == 0


def test_html5_wellformed_and_single_main(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    doc_open, doc_close = build_shell(lang="en", title="Doc")
    html = doc_open + "<p>body</p>" + doc_close
    outcome = _check_html5_wellformed(html)
    assert outcome.passed, outcome.message
    # Exactly one <main>.
    assert html.count("<main") == 1
    assert html.count("</main>") == 1


# ---------------------------------------------------------------------------
# Phase 9 — outer ``<section aria-labelledby>`` document-outline grouping over
# the flat region list.
# ---------------------------------------------------------------------------

from dart_semantic.assembler.pass_9a import run_pass_9a  # noqa: E402
from dart_semantic.gates.hard_document import _check_heading_hierarchy  # noqa: E402


def _heading_region(idx: int, *, level: int = 1) -> Region:
    """A structural heading region pinned to ``level`` via the level_hint."""
    return Region(
        kind="heading",
        feature_block_indices=(idx,),
        payload={"text": f"body {idx}", "level_hint": level},
    )


def _levels9(html: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"<h([1-6])\b", html, re.I)]


def test_section_wraps_heading_and_following_blocks(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("a"), _fb("p1"), _fb("p2"), _fb("b")]
    regions = [
        _heading_region(0, level=1),
        _region(1, None),
        _region(2, None),
        _heading_region(3, level=1),
    ]
    top = {
        0: _stage6("<h2>Alpha</h2>"),
        1: _stage6("<p>para one</p>"),
        2: _stage6("<p>para two</p>"),
        3: _stage6("<h2>Beta</h2>"),
    }
    doc = _assemble(top, regions, fbs)
    html = doc.html
    # One section labelled by the first heading id encloses the heading + both
    # paragraphs, and is CLOSED before the next same-level heading.
    m = re.search(r'<section aria-labelledby="alpha">(.*?)</section>', html, re.S)
    assert m is not None, html
    seg = m.group(1)
    assert "Alpha</h1>" in seg
    assert "para one" in seg and "para two" in seg
    assert "Beta" not in seg  # closed at the next same-level heading
    # The second heading opened its own section.
    assert '<section aria-labelledby="beta">' in html
    # Sections are balanced (open count == close count).
    assert html.count("<section aria-labelledby=") == html.count("</section>") == 2


def test_aria_labelledby_resolves_to_real_id(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("h"), _fb("d"), _fb("w")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "definition_region"),
        _region(2, "worked_example"),
    ]
    top = {
        0: _stage6("<h2>Chapter One</h2>"),
        1: _stage6("<p>a definition</p>"),
        2: _stage6("<p>EXAMPLE worked</p>"),
    }
    doc = _assemble(top, regions, fbs)
    html = doc.html
    # Every aria-labelledby (outer section + Phase-7 containers) points at an id
    # that EXISTS in the doc — INCLUDING after the contiguity re-level demoted
    # the container <h4>s.
    targets = re.findall(r'aria-labelledby="([^"]+)"', html)
    assert len(targets) >= 3  # outer section + definition + worked_example
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    for t in targets:
        assert t in ids, (t, sorted(ids))
    # The container headings were re-leveled (h1 structural, h4 -> h3) and the
    # outer section still labels the structural heading.
    assert '<section aria-labelledby="chapter-one">' in html


def test_container_label_id_unique(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("d1"), _fb("d2")]
    regions = [_region(0, "definition_region"), _region(1, "definition_region")]
    top = {0: _stage6("<p>def one</p>"), 1: _stage6("<p>def two</p>")}
    doc = _assemble(top, regions, fbs)
    ids = re.findall(r'aria-labelledby="(dart-definition_region[^"]*)"', doc.html)
    assert len(ids) == 2
    assert len(set(ids)) == 2  # distinct


def test_splice_keys_survive_section_wrap(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("h"), _fb("legal"), _fb("cite")]
    regions = [
        _heading_region(0, level=1),
        Region(
            kind="paragraph",
            feature_block_indices=(1,),
            payload={
                "text": "Copyright 2026 Ed4All. All rights reserved.",
                "doc_role": "legal",
            },
        ),
        Region(
            kind="paragraph",
            feature_block_indices=(2,),
            payload={"text": "See Section 4.2 for the proof of this claim."},
        ),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>Copyright 2026 Ed4All. All rights reserved.</p>"),
        2: _stage6("<p>See Section 4.2 for the proof of this claim.</p>"),
    }
    pre, gaps, _ = run_pass_9a(
        top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True),
    )
    body = pre.html
    # Grouping happened.
    assert "<section aria-labelledby=" in body
    # Every per-region fragment (the pass_9c ``region_html[idx]`` splice key)
    # is still a contiguous substring of the assembled body after the wrap.
    for frag in pre.sub_task_log["region_html"]:
        if frag:
            assert frag in body, frag
    # The legal gap's stashed ``region_html`` context + any match_text survive.
    for g in gaps:
        rh = (g.context or {}).get("region_html")
        if rh:
            assert rh in body, g.kind
        mt = (g.context or {}).get("match_text")
        if mt:
            assert mt in body, g.kind


def test_example_renders_as_region_not_h1(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    fbs = [_fb("h"), _fb("ex")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),  # kind="paragraph" -> NOT a heading
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE 1.3 solve x</p>"),
    }
    doc = _assemble(top, regions, fbs)
    html = doc.html
    # The EXAMPLE renders inside a worked_example role=region container.
    assert 'class="algorithm worked-example"' in html
    assert 'role="region"' in html
    # It is NOT promoted to an <h1> (the indexing bug): only the structural
    # heading is an <h1>, and only that heading opens an outer section.
    assert html.count("<h1") == 1
    assert ">Intro</h1>" in html
    assert html.count("<section aria-labelledby=") == 1
    # The example body is not a structural heading entry in the outline.
    assert (1, "EXAMPLE 1.3 solve x") not in doc.heading_tree
    assert all("EXAMPLE 1.3 solve x" != text for _lvl, text in doc.heading_tree)


def test_flag_off_flat_concat_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    fbs = [_fb("a"), _fb("p1"), _fb("b")]
    regions = [
        _heading_region(0, level=1),
        _region(1, None),
        _heading_region(2, level=1),
    ]
    top = {
        0: _stage6("<h2>Alpha</h2>"),
        1: _stage6("<p>para</p>"),
        2: _stage6("<h2>Beta</h2>"),
    }
    doc = _assemble(top, regions, fbs)
    # No outer-section grouping when off.
    assert "<section aria-labelledby=" not in doc.html
    # Body is the plain flat concat of the index-aligned fragments.
    flat = "".join(f for f in doc.sub_task_log["region_html"] if f)
    assert flat in doc.html


# ---------------------------------------------------------------------------
# Stopgap — SEMANTIK_GOLD_ABSORB body-region absorption.
#
# A body-bearing component label region (worked_example / definition_region /
# exercise) ABSORBS the immediately-following sibling body regions (Solution /
# steps / prose / tables) into its container box, stopping at the FIRST boundary
# (heading / unit-opening pedagogical label / next body-bearing component /
# empty), bounded by a conservative cap. Flag-off (incl. shell-on/absorb-off) is
# byte-identical to the pre-absorption gold wrap.
# ---------------------------------------------------------------------------

from dart_semantic.assembler.gold_shell_markup import (  # noqa: E402
    _ABSORB_MAX_RUN,
    compute_absorption_runs,
)
from dart_semantic.assembler.shell import resolve_gold_absorb_mode  # noqa: E402

_WE_DIV_RE = re.compile(
    r'<div class="algorithm worked-example"[^>]*>(.*?)</div>', re.S,
)


def _css_region(idx: int, css_class: str, *, text: str | None = None) -> Region:
    """A plain paragraph region carrying a deterministic ``css_class`` hint."""
    return Region(
        kind="paragraph",
        feature_block_indices=(idx,),
        payload={"text": text or f"body {idx}", "css_class": css_class},
    )


def test_gold_absorb_mode_flag(monkeypatch):
    # Default OFF (shell on so the resolver's shell-gate isn't what zeroes it).
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    # ITEM1: pin the regroup OFF so the (now default-ON) regroup short-circuit
    # doesn't zero the absorb — this test isolates the absorb's own resolver.
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")
    monkeypatch.delenv("SEMANTIK_GOLD_ABSORB", raising=False)
    assert resolve_gold_absorb_mode() is False
    for v in ("", "  ", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", v)
        assert resolve_gold_absorb_mode() is False
    # Truthy tokens -> True, but ONLY while GOLD_SHELL is on.
    for v in ("1", "true", "YES", "on", "  On  "):
        monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", v)
        assert resolve_gold_absorb_mode() is True
    # Requires SEMANTIK_GOLD_SHELL — a truthy absorb with shell off is False.
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "0")
    assert resolve_gold_absorb_mode() is False


def test_worked_example_absorbs_solution_and_body(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: isolate the absorb (regroup default-ON)
    fbs = [_fb("h"), _fb("ex"), _fb("sol"), _fb("p1"), _fb("p2"), _fb("h2")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),
        _css_region(2, "pedagogy-solution"),
        _region(3, None),
        _region(4, None),
        _heading_region(5, level=1),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE 1.3 solve x</p>"),
        2: _stage6("<p>Solution: factor the expression</p>"),
        3: _stage6("<p>step a value</p>"),
        4: _stage6("<p>step b value</p>"),
        5: _stage6("<h2>Next Section</h2>"),
    }
    html = _assemble(top, regions, fbs).html
    m = _WE_DIV_RE.search(html)
    assert m is not None, html
    box = m.group(1)
    # The box encloses the label + Solution + both body paragraphs.
    assert "EXAMPLE 1.3 solve x" in box
    assert "Solution: factor the expression" in box
    assert "step a value" in box
    assert "step b value" in box
    # The following heading is OUTSIDE the box (a boundary).
    assert "Next Section" not in box
    assert "Next Section" in html
    # Absorbed fragments appear EXACTLY ONCE in the whole document.
    for frag in ("Solution: factor the expression", "step a value", "step b value"):
        assert html.count(frag) == 1, frag


def test_absorption_stops_at_next_example(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: isolate the absorb (regroup default-ON)
    fbs = [_fb("h"), _fb("ex1"), _fb("sol1"), _fb("ex2"), _fb("sol2")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),
        _css_region(2, "pedagogy-solution"),
        # The SECOND example: a unit-opening pedagogy-example label AND its own
        # body-bearing component -> a boundary that starts its own unit.
        Region(
            kind="paragraph",
            feature_block_indices=(3,),
            payload={
                "text": "EXAMPLE 2",
                "css_class": "pedagogy-example",
                "semantic_class": "worked_example",
            },
        ),
        _css_region(4, "pedagogy-solution"),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE 1 first</p>"),
        2: _stage6("<p>Solution one body</p>"),
        3: _stage6("<p>EXAMPLE 2 second</p>"),
        4: _stage6("<p>Solution two body</p>"),
    }
    html = _assemble(top, regions, fbs).html
    boxes = _WE_DIV_RE.findall(html)
    # Two separate worked-example boxes (the second example started its own).
    assert len(boxes) == 2, html
    first = boxes[0]
    assert "EXAMPLE 1 first" in first
    assert "Solution one body" in first
    # The first box STOPS before the second example.
    assert "EXAMPLE 2 second" not in first
    # The second box owns the second example + its own solution.
    second = boxes[1]
    assert "EXAMPLE 2 second" in second
    assert "Solution two body" in second


def test_absorption_stops_at_heading(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: isolate the absorb (regroup default-ON)
    fbs = [_fb("h"), _fb("ex"), _fb("sol"), _fb("h2"), _fb("p")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),
        _css_region(2, "pedagogy-solution"),
        _heading_region(3, level=1),  # boundary
        _region(4, None),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE here</p>"),
        2: _stage6("<p>Solution body here</p>"),
        3: _stage6("<h2>Another Heading</h2>"),
        4: _stage6("<p>after the heading</p>"),
    }
    html = _assemble(top, regions, fbs).html
    m = _WE_DIV_RE.search(html)
    assert m is not None, html
    box = m.group(1)
    assert "Solution body here" in box
    # The heading and everything after it is OUTSIDE the box.
    assert "Another Heading" not in box
    assert "after the heading" not in box
    assert "Another Heading" in html and "after the heading" in html


def test_absorption_cap():
    # A long run with NO boundary -> absorption stops at the conservative cap.
    n_paras = 12
    regions = [_region(0, "worked_example")] + [
        _region(i, None) for i in range(1, 1 + n_paras)
    ]
    region_html = ["<p>EXAMPLE label</p>"] + [
        f"<p>pp{i}</p>" for i in range(1, 1 + n_paras)
    ]
    runs = compute_absorption_runs(regions, region_html)
    # Anchor at 0 absorbs at most _ABSORB_MAX_RUN following regions:
    # end = 1 + _ABSORB_MAX_RUN.
    assert runs == {0: 1 + _ABSORB_MAX_RUN}
    # The (cap)-th follower is absorbed; the (cap+1)-th is left outside.
    absorbed = set(range(1, runs[0]))
    assert _ABSORB_MAX_RUN in absorbed
    assert (_ABSORB_MAX_RUN + 1) not in absorbed


def test_splice_keys_survive_absorption(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: isolate the absorb (regroup default-ON)
    fbs = [_fb("h"), _fb("ex"), _fb("sol"), _fb("p1"), _fb("h2")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),
        _css_region(2, "pedagogy-solution"),
        _region(3, None),
        _heading_region(4, level=1),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE 9 here</p>"),
        2: _stage6("<p>Solution splice body</p>"),
        3: _stage6("<p>more body text</p>"),
        4: _stage6("<h2>Tail</h2>"),
    }
    pre, gaps, _ = run_pass_9a(
        top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True),
    )
    body = pre.html
    # Absorption happened (a worked-example box exists).
    assert 'class="algorithm worked-example"' in body
    # Every per-region fragment recorded for pass_9c (incl. the ABSORBED ones,
    # which the test scope demands) is still a contiguous substring of the body.
    for frag in pre.sub_task_log["region_html"]:
        if frag:
            assert frag in body, frag
    # The absorbed Solution / body fragments specifically survive verbatim.
    assert "<p>Solution splice body</p>" in body
    assert "<p>more body text</p>" in body
    # Any gap-stashed splice key also survives.
    for g in gaps:
        rh = (g.context or {}).get("region_html")
        if rh:
            assert rh in body, g.kind
        mt = (g.context or {}).get("match_text")
        if mt:
            assert mt in body, g.kind


def _absorb_corpus():
    fbs = [_fb("h"), _fb("ex"), _fb("sol"), _fb("p1"), _fb("h2")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),
        _css_region(2, "pedagogy-solution"),
        _region(3, None),
        _heading_region(4, level=1),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE body</p>"),
        2: _stage6("<p>Solution body</p>"),
        3: _stage6("<p>tail body</p>"),
        4: _stage6("<h2>End</h2>"),
    }
    return top, regions, fbs


def test_absorb_flag_off_byte_identical(monkeypatch):
    # GOLD_SHELL ON throughout; only SEMANTIK_GOLD_ABSORB toggles. Absorb-off
    # (absent / off / 0) must be byte-identical to the pre-absorption gold wrap.
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: isolate the absorb (regroup default-ON)

    monkeypatch.delenv("SEMANTIK_GOLD_ABSORB", raising=False)
    absent = _assemble(*_absorb_corpus()).html

    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "off")
    off = _assemble(*_absorb_corpus()).html

    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "0")
    zero = _assemble(*_absorb_corpus()).html

    assert absent == off == zero
    # The pre-absorption gold render does NOT pull the Solution into the box:
    # with absorb OFF the worked_example box wraps only its own label fragment.
    m = _WE_DIV_RE.search(absent)
    assert m is not None
    assert "Solution body" not in m.group(1)

    # And with absorb ON the body IS different (proves the flag actually fires).
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    on = _assemble(*_absorb_corpus()).html
    assert on != absent
    m_on = _WE_DIV_RE.search(on)
    assert m_on is not None and "Solution body" in m_on.group(1)


def test_absorb_output_byte_identical_after_hoist():
    """compute_absorption_runs is byte-identical after the SoT hoist to the
    neutral pedagogical_units module (covers the positional-call signature of
    the re-exported _is_absorption_boundary).

    The expected runs dict is the pre-hoist behavior pinned by value: a
    worked_example anchor absorbs its solution + a plain-paragraph body, stops
    at the next pedagogy-example (a unit-start boundary) AND at a heading.
    """
    regions = [
        _heading_region(0, level=1),          # 0: heading (anchor scan skips)
        _region(1, "worked_example"),         # 1: anchor
        _css_region(2, "pedagogy-solution"),  # 2: body (absorbed)
        _region(3, None),                     # 3: body (absorbed)
        Region(                               # 4: next unit-start -> boundary
            kind="paragraph",
            feature_block_indices=(4,),
            payload={
                "text": "EXAMPLE 2",
                "css_class": "pedagogy-example",
                "semantic_class": "worked_example",
            },
        ),
        _css_region(5, "pedagogy-solution"),  # 5: body of the 2nd unit
        _heading_region(6, level=1),          # 6: heading -> boundary
    ]
    region_html = [
        "<h2>Intro</h2>",
        "<p>EXAMPLE 1</p>",
        "<p>Solution body</p>",
        "<p>step</p>",
        "<p>EXAMPLE 2</p>",
        "<p>Solution two</p>",
        "<h2>Next</h2>",
    ]
    runs = compute_absorption_runs(regions, region_html)
    # Anchor 1 absorbs indices 2,3 -> end 4 (stops at the unit-start at 4).
    # Anchor 4 absorbs index 5 -> end 6 (stops at the heading at 6).
    assert runs == {1: 4, 4: 6}


# ---------------------------------------------------------------------------
# Phase 7 — SEMANTIK_UNIT_REGROUP supersedes SEMANTIK_GOLD_ABSORB when FIRING.
#
# When the Stage-5e cross-kind regroup is ACTUALLY firing (regroup-on AND
# reading-order-on — the SAME composite condition the regroup fires under),
# each unit's label+body is already ONE region carrying the unit's
# semantic_class, so the per-region gold wrap boxes the WHOLE unit; the
# assembler-side absorb is forced OFF (mutual exclusion — never double-box).
# Regroup-INERT (reading-order off) leaves the absorb as the fallback.
# ---------------------------------------------------------------------------


def test_absorb_forced_off_when_regroup_firing(monkeypatch):
    # Regroup ON + reading-order ON -> regroup is firing -> absorb forced off
    # regardless of SEMANTIK_GOLD_ABSORB.
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", v)
        assert resolve_gold_absorb_mode() is False
    # Reading-order DEFAULT-ON (unset) also counts as firing.
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    assert resolve_gold_absorb_mode() is False


def test_absorb_stays_on_when_regroup_inert(monkeypatch):
    # Regroup ON but reading-order OFF -> the regroup is INERT (its Phase-6
    # driver guard), so the absorb MUST stay the fallback (no regression vs
    # absorb-alone). Gating the short-circuit on unit_regroup alone would box
    # NEITHER here.
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "off")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    assert resolve_gold_absorb_mode() is True


def test_absorb_short_circuit_engages_by_default(monkeypatch):
    # ITEM1: with the regroup default-ON, the absorb short-circuit is now the
    # DEFAULT. GOLD_SHELL + GOLD_ABSORB on, NO SEMANTIK_UNIT_REGROUP set,
    # reading-order unset (both default-ON) -> regroup is firing -> absorb forced
    # off. Explicitly reverting the regroup (=0) re-arms the absorb fallback.
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP", raising=False)
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)
    assert resolve_gold_absorb_mode() is False  # supersession is the default

    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")
    assert resolve_gold_absorb_mode() is True  # revert lever re-arms the fallback


def test_no_double_box_regroup_plus_absorb(monkeypatch):
    # The regroup is firing AND SEMANTIK_GOLD_ABSORB is on. The merged unit
    # arrives as ONE worked_example region (Stage-5e already fused it); the
    # per-region wrap boxes it EXACTLY ONCE and the absorb does NOT add a
    # second box on top (it was forced off).
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    assert resolve_gold_absorb_mode() is False
    fbs = [_fb("h"), _fb("unit"), _fb("h2")]
    regions = [
        _heading_region(0, level=1),
        # The merged unit: ONE worked_example region whose Stage-6 HTML carries
        # the label + Solution + body (Stage-5e fused them upstream).
        _region(1, "worked_example"),
        _heading_region(2, level=1),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6(
            "<p>EXAMPLE 1.3 solve x</p><p>Solution: factor</p><p>step a value</p>"
        ),
        2: _stage6("<h2>Next Section</h2>"),
    }
    html = _assemble(top, regions, fbs).html
    # Exactly ONE worked-example box.
    boxes = _WE_DIV_RE.findall(html)
    assert len(boxes) == 1, html
    box = boxes[0]
    # The single box holds the WHOLE unit.
    assert "EXAMPLE 1.3 solve x" in box
    assert "Solution: factor" in box
    assert "step a value" in box
    # Each fragment appears EXACTLY ONCE in the whole document (no absorb-on-top
    # double emit).
    for frag in ("EXAMPLE 1.3 solve x", "Solution: factor", "step a value"):
        assert html.count(frag) == 1, frag


def test_absorb_unchanged_when_regroup_off(monkeypatch):
    # Regroup OFF -> resolve_gold_absorb_mode is byte-identical to its baseline:
    # absorb-on renders the absorbed body, absorb-off does not (the Phase-7
    # short-circuit never engages when SEMANTIK_UNIT_REGROUP is off).
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")

    monkeypatch.delenv("SEMANTIK_GOLD_ABSORB", raising=False)
    absent = _assemble(*_absorb_corpus()).html
    assert resolve_gold_absorb_mode() is False

    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    on = _assemble(*_absorb_corpus()).html
    assert resolve_gold_absorb_mode() is True
    assert on != absent
    m_on = _WE_DIV_RE.search(on)
    assert m_on is not None and "Solution body" in m_on.group(1)
    m_off = _WE_DIV_RE.search(absent)
    assert m_off is not None and "Solution body" not in m_off.group(1)


# ===========================================================================
# table-v2 (SEMANTIK_UNIT_REGROUP_TABLE) — narrow passthrough-only absorb.
# ===========================================================================

from dart_semantic.pedagogical_units import is_passthrough_region  # noqa: E402
from dart_semantic.assembler.shell import (  # noqa: E402
    resolve_narrow_table_absorb_mode,
)
from dart_semantic.qwen_specialists.block_resegment import (  # noqa: E402
    _is_passthrough,
)

# A <table> fragment with a <th scope> header (mirrors the Stage-6 table
# specialist's GLM-OCR-enriched output).
_TABLE_HTML = (
    '<table><thead><tr><th scope="col">x</th><th scope="col">y</th></tr>'
    "</thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
)


def _pt_region(idx: int, *, kind: str = "table", fb: tuple[int, ...] | None = None) -> Region:
    """A Stage-4 passthrough region (non-null ``source_region_id``) at FB ``idx``."""
    return Region(
        kind=kind,
        feature_block_indices=fb if fb is not None else (idx,),
        payload={"text": f"passthrough {idx}"},
        source_region_id=100 + idx,
    )


# ---------------------------------------------------------------------------
# Phase 1 — compute_absorption_runs(passthrough_only=...) unit tests.
# ---------------------------------------------------------------------------


def test_is_passthrough_region_predicate():
    none_region = _region(0, "worked_example")
    assert is_passthrough_region(none_region) is False
    assert is_passthrough_region(_pt_region(1)) is True
    # block_resegment._is_passthrough aliases the same single source.
    assert _is_passthrough is is_passthrough_region


def test_compute_absorption_runs_passthrough_only_default_byte_identical():
    """passthrough_only=False is byte-identical to the no-kwarg call."""
    regions = [
        _region(0, "worked_example"),
        _css_region(1, "pedagogy-solution"),
        _region(2, None),
        _heading_region(3, level=1),
    ]
    html = ["<p>anchor</p>", "<p>sol</p>", "<p>tail</p>", "<h2>h</h2>"]
    assert compute_absorption_runs(regions, html) == compute_absorption_runs(
        regions, html, passthrough_only=False
    )


def test_passthrough_only_absorbs_only_trailing_table():
    """anchor + FB-adjacent table + plain paragraph -> only the table is absorbed."""
    regions = [
        _region(0, "worked_example"),
        _pt_region(1),
        _region(2, None),
    ]
    html = ["<p>anchor</p>", _TABLE_HTML, "<p>plain</p>"]
    runs = compute_absorption_runs(regions, html, passthrough_only=True)
    # Run is {anchor: anchor+2} — the table only; the plain paragraph is a stop.
    assert runs == {0: 2}


def test_passthrough_only_stops_at_first_non_passthrough():
    """anchor + plain-paragraph(non-passthrough) -> no run (immediate hard stop)."""
    regions = [_region(0, "worked_example"), _region(1, None)]
    html = ["<p>anchor</p>", "<p>plain</p>"]
    assert compute_absorption_runs(regions, html, passthrough_only=True) == {}


def test_passthrough_only_fb_adjacency_bounds_two_tables():
    """anchor + FB-adjacent solution table + NON-FB-adjacent standalone table ->
    only the FIRST (FB-contiguous) table is captured (the over-capture guard)."""
    regions = [
        _region(0, "worked_example"),          # FB 0
        _pt_region(1, fb=(1,)),                 # FB 1 — adjacent to 0
        _pt_region(5, fb=(5,)),                 # FB 5 — GAP after FB 1
    ]
    html = ["<p>anchor</p>", _TABLE_HTML, _TABLE_HTML]
    runs = compute_absorption_runs(regions, html, passthrough_only=True)
    assert runs == {0: 2}  # only the first table; the gapped one is not pulled in


def test_passthrough_only_captures_math_and_figure():
    """anchor + FB-adjacent math passthrough + FB-adjacent figure passthrough ->
    both absorbed (source_region_id-keyed parity)."""
    regions = [
        _region(0, "worked_example"),          # FB 0
        _pt_region(1, kind="math", fb=(1,)),   # FB 1
        _pt_region(2, kind="figure", fb=(2,)), # FB 2
    ]
    html = ["<p>anchor</p>", "<math></math>", "<figure></figure>"]
    runs = compute_absorption_runs(regions, html, passthrough_only=True)
    assert runs == {0: 3}


# ---------------------------------------------------------------------------
# Phase 2 — wire the narrow absorb at the pass_9a Sub-task-1b call site.
# ---------------------------------------------------------------------------


def _narrow_env(monkeypatch, *, sub_flag="on", gold_absorb=None):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")
    if sub_flag is None:
        monkeypatch.delenv("SEMANTIK_UNIT_REGROUP_TABLE", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", sub_flag)
    if gold_absorb is None:
        monkeypatch.delenv("SEMANTIK_GOLD_ABSORB", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", gold_absorb)


def _we_table_corpus(anchor_class="worked_example"):
    """merged-prose body-bearing anchor + trailing FB-adjacent passthrough table."""
    fbs = [_fb("h"), _fb("unit"), _fb("table"), _fb("h2")]
    regions = [
        _heading_region(0, level=1),
        _region(1, anchor_class),                # FB 1
        _pt_region(2, fb=(2,)),                  # FB 2 — adjacent passthrough table
        _heading_region(3, level=1),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE 1.3 solve x</p><p>Solution: factor</p>"),
        2: _stage6(_TABLE_HTML),
        3: _stage6("<h2>Next Section</h2>"),
    }
    return top, regions, fbs


def test_narrow_absorb_wraps_solution_table_inside_box(monkeypatch):
    _narrow_env(monkeypatch)
    html = _assemble(*_we_table_corpus()).html
    m = _WE_DIV_RE.search(html)
    assert m is not None, html
    box = m.group(1)
    # The box ENCLOSES the <table> fragment.
    assert "<table>" in box
    assert '<th scope="col">x</th>' in box
    # And it is NOT a separate sibling region after the box.
    after_box = html[m.end():]
    assert "<table>" not in after_box


def test_narrow_absorb_table_rendered_once(monkeypatch):
    _narrow_env(monkeypatch)
    html = _assemble(*_we_table_corpus()).html
    # The rendered table body appears EXACTLY ONCE (bare "<table>" also occurs in
    # the gold-style CSS comment, so count a unique rendered-cell substring).
    assert html.count("<td>1</td>") == 1
    # The table index landed in absorbed_indices (guards the hoist regression):
    # it does not render as a standalone <section>/sibling outside the box.
    m = _WE_DIV_RE.search(html)
    assert "<table>" in m.group(1)


def test_narrow_absorb_fires_with_gold_absorb_unset(monkeypatch):
    """The REAL default deployment: regroup firing + sub-flag on + GOLD_ABSORB
    UNSET -> the narrow path still boxes the table."""
    _narrow_env(monkeypatch, gold_absorb=None)
    assert resolve_narrow_table_absorb_mode() is True
    html = _assemble(*_we_table_corpus()).html
    m = _WE_DIV_RE.search(html)
    assert m is not None and "<table>" in m.group(1)


def test_narrow_absorb_non_merged_definition_anchor(monkeypatch):
    """A non-merged definition_region anchor + trailing FB-adjacent table -> boxed
    too (anchor scope is ALL body-bearing components)."""
    _narrow_env(monkeypatch)
    fbs = [_fb("h"), _fb("def"), _fb("table")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "definition_region"),         # FB 1
        _pt_region(2, fb=(2,)),                   # FB 2
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>A monomial is ...</p>"),
        2: _stage6(_TABLE_HTML),
    }
    html = _assemble(top, regions, fbs).html
    # The definition box encloses the table.
    m = re.search(r'<div class="definition[^"]*"[^>]*>(.*?)</div>', html, re.S)
    assert m is not None, html
    assert "<table>" in m.group(1)
    assert html.count("<td>1</td>") == 1


def test_narrow_absorb_off_table_adjacent_byte_identical(monkeypatch):
    """sub-flag OFF (regroup on) -> assembled HTML byte-identical to v1 (table
    renders adjacent, OUTSIDE the box)."""
    # sub-flag ON render.
    _narrow_env(monkeypatch, sub_flag="on")
    on_html = _assemble(*_we_table_corpus()).html
    # sub-flag OFF render (the committed v1 baseline).
    _narrow_env(monkeypatch, sub_flag=None)
    off_html = _assemble(*_we_table_corpus()).html
    # The two DIFFER (the feature does something) and the OFF render keeps the
    # table OUTSIDE the worked-example box (the v1 table-adjacent shape).
    assert on_html != off_html
    m_off = _WE_DIV_RE.search(off_html)
    assert m_off is not None and "<table>" not in m_off.group(1)
    assert "<table>" in off_html  # still rendered, just adjacent


def test_narrow_absorb_does_not_grab_trailing_prose(monkeypatch):
    """merged-prose + table + closing-paragraph -> the box encloses prose+table
    ONLY; the closing paragraph stays a sibling (passthrough-only stop)."""
    _narrow_env(monkeypatch)
    fbs = [_fb("h"), _fb("unit"), _fb("table"), _fb("concl"), _fb("h2")]
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),            # FB 1
        _pt_region(2, fb=(2,)),                  # FB 2 — adjacent table
        _region(3, None),                        # FB 3 — closing prose (non-passthrough)
        _heading_region(4, level=1),
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE solve x</p>"),
        2: _stage6(_TABLE_HTML),
        3: _stage6("<p>conclusion prose</p>"),
        4: _stage6("<h2>Next</h2>"),
    }
    html = _assemble(top, regions, fbs).html
    box = _WE_DIV_RE.search(html).group(1)
    assert "<table>" in box
    assert "conclusion prose" not in box  # documented scope limit
    assert "conclusion prose" in html      # still rendered, as a sibling


# ---------------------------------------------------------------------------
# Phase 3 — compose with the Phase-7 absorb-supersession.
# ---------------------------------------------------------------------------


def test_no_double_box_narrow_table_absorb(monkeypatch):
    """regroup firing + sub-flag on + GOLD_ABSORB on -> the gold box appears
    EXACTLY ONCE around the anchor+table span (no full-absorb-on-top)."""
    _narrow_env(monkeypatch, gold_absorb="1")
    html = _assemble(*_we_table_corpus()).html
    boxes = _WE_DIV_RE.findall(html)
    assert len(boxes) == 1, html
    assert "<table>" in boxes[0]
    assert html.count("<td>1</td>") == 1


def test_full_absorb_stays_off_under_narrow(monkeypatch):
    """regroup firing + sub-flag on -> resolve_gold_absorb_mode still False (the
    full absorb never runs; only the narrow elif)."""
    _narrow_env(monkeypatch, gold_absorb="1")
    assert resolve_gold_absorb_mode() is False
    assert resolve_narrow_table_absorb_mode() is True


def test_narrow_inert_when_parent_regroup_off(monkeypatch):
    """sub-flag on + SEMANTIK_UNIT_REGROUP off -> narrow branch dead; with
    GOLD_ABSORB on the FULL absorb fires as the v1 fallback."""
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "on")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    assert resolve_narrow_table_absorb_mode() is False
    # The full absorb is live (regroup not firing).
    assert resolve_gold_absorb_mode() is True


def test_narrow_inert_when_reading_order_off(monkeypatch):
    """sub-flag on + reading-order OFF + regroup on -> narrow dead, full absorb
    fires (no regression)."""
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "off")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "on")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    assert resolve_narrow_table_absorb_mode() is False
    assert resolve_gold_absorb_mode() is True


# ---------------------------------------------------------------------------
# Phase 4 — assembler-level survival coverage (relocation, not mutation).
# ---------------------------------------------------------------------------


def test_in_box_table_enriched_bytes_byte_identical(monkeypatch):
    """The in-box table's enriched <table>/<th scope> bytes are byte-identical
    pre/post absorb — the absorb RELOCATES rendered bytes, never mutates them."""
    # Baseline: the table fragment as the Stage-6 specialist emitted it.
    _narrow_env(monkeypatch)
    html = _assemble(*_we_table_corpus()).html
    # The exact enriched table substring survives intact inside the box.
    assert _TABLE_HTML in html
    box = _WE_DIV_RE.search(html).group(1)
    assert _TABLE_HTML in box


def test_post_9a_passes_position_independent(monkeypatch):
    """The byte-identical table fragment is relocated, not mutated: it is a
    contiguous substring whether standalone (sub-flag off) or nested in the box
    (sub-flag on). No post-9a pass keys on the table's top-level position."""
    _narrow_env(monkeypatch, sub_flag="on")
    on_html = _assemble(*_we_table_corpus()).html
    _narrow_env(monkeypatch, sub_flag=None)
    off_html = _assemble(*_we_table_corpus()).html
    # The enriched fragment is a contiguous substring in BOTH renders.
    assert _TABLE_HTML in on_html
    assert _TABLE_HTML in off_html
    # The document is well-formed + single-<main> in both (post-9a passes ran).
    for h in (on_html, off_html):
        assert h.count("<main") == 1


def test_in_box_math_figure_source_region_id_parity(monkeypatch):
    """A math/figure passthrough span-wrapped into a box keeps its source_region_id
    parity; a figure additionally renders its <img src> + sidecar inside the box."""
    _narrow_env(monkeypatch)
    fbs = [_fb("h"), _fb("unit"), _fb("fig")]
    fig_region = Region(
        kind="figure",
        feature_block_indices=(2,),
        payload={"text": "figure", "image_src": "figs/fig-2.png"},
        source_region_id=205,
    )
    regions = [
        _heading_region(0, level=1),
        _region(1, "worked_example"),            # FB 1
        fig_region,                              # FB 2 — adjacent figure passthrough
    ]
    top = {
        0: _stage6("<h2>Intro</h2>"),
        1: _stage6("<p>EXAMPLE solve x</p>"),
        2: _stage6('<figure><img src="figs/fig-2.png" alt="Figure."></figure>'),
    }
    html = _assemble(top, regions, fbs).html
    # The figure region kept its source_region_id (never folded).
    assert fig_region.source_region_id == 205
    box = _WE_DIV_RE.search(html).group(1)
    # The figure's <img src> + the sidecar reference live INSIDE the box.
    assert '<img src="figs/fig-2.png"' in box
    assert html.count('<img src="figs/fig-2.png"') == 1


def test_metadata_drop_dropped_even_with_stage6_candidate():
    """A metadata_drop region routes to PROSE (routing.py) so Stage-6 may author
    it, but it must ALWAYS be dropped from the render — the clean pass already
    decided it is a running header / page number / phantom-TOC line, not content.
    Regression: the real-Stage-6 path was emitting the authored "Chapter 1
    Foundations 27" running header as a <p>."""
    regions = [
        Region(kind="metadata_drop", feature_block_indices=(0,),
               payload={"text": "Chapter 1 Foundations 27"}),
        Region(kind="paragraph", feature_block_indices=(1,),
               payload={"text": "real body"}),
    ]
    fbs = [_fb("Chapter 1 Foundations 27"), _fb("real body")]
    top = {0: _stage6("<p>Chapter 1 Foundations 27</p>"),
           1: _stage6("<p>real body sentence.</p>")}
    html = _assemble(top, regions, fbs).html
    assert "Chapter 1 Foundations 27" not in html  # authored running header dropped
    assert "real body sentence." in html            # real content survives
