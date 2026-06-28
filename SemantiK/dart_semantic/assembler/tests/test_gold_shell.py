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
