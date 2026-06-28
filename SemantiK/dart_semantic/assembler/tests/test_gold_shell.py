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
