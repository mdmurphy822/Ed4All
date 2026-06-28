"""Phase 1 — vendored gold-standard CSS + shell-markup self-containment.

These tests pin the data-only vendoring contract: the two new SemantiK-owned
modules import, the CSS is the UNION of both gold vocabularies plus the two
net-new rules, ``GOLD_DOC_OPEN`` keeps ``shell.DOC_OPEN``'s ``.format()``
slot contract, neither module references the retired-DART worktree, and —
because nothing consumes them yet — ``assemble_document`` output is
unchanged. CPU-only, no model load.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from dart_semantic.assembler.api import AssemblerConfig, assemble_document
from dart_semantic.assembler.shell import DOC_OPEN
from dart_semantic.assembler.wcag22_css import WCAG22_CSS
from dart_semantic.assembler import gold_shell_markup
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=1,
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


def _para(text: str, idx: int) -> Region:
    return Region(
        kind="paragraph",
        feature_block_indices=(idx,),
        payload={"text": text},
        source_region_id=idx,
    )


def _assemble_fixture():
    regions = [_para("First paragraph.", 0), _para("Second paragraph.", 1)]
    fbs = [_fb("First paragraph."), _fb("Second paragraph.")]
    top_per_region = {i: None for i in range(len(regions))}
    return assemble_document(
        top_per_region,
        regions,
        fbs,
        runtime_mode="mock",
        config=AssemblerConfig(skip_gap_fill=True),
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_wcag22_css_importable_and_nonempty():
    assert isinstance(WCAG22_CSS, str)
    assert WCAG22_CSS.strip(), "WCAG22_CSS is empty"


def test_css_union_covers_both_vocabularies():
    # gold inline vocabulary
    for sel in (".definition", ".algorithm", "caption"):
        assert sel in WCAG22_CSS, f"missing gold selector {sel!r}"
    # WCAG-22 base vocabulary
    for sel in (".callout-warning", ".pullquote", ".sr-only", "pre[role=\"region\"]"):
        assert sel in WCAG22_CSS, f"missing wcag22 selector {sel!r}"
    # net-new Phase-1 rules
    for sel in (".exercise", ".objectives"):
        assert sel in WCAG22_CSS, f"missing net-new selector {sel!r}"


def test_table_role_reconciled_to_single_convention():
    # ONE bare-`table` convention; no conflicting `table[role="grid"]` rule.
    assert "table[role=\"grid\"]" not in WCAG22_CSS
    assert "\n  table {" in WCAG22_CSS or " table {" in WCAG22_CSS


def test_gold_shell_markup_importable():
    assert isinstance(gold_shell_markup.GOLD_DOC_OPEN, str)
    assert isinstance(gold_shell_markup.GOLD_DOC_CLOSE, str)
    assert isinstance(gold_shell_markup.GOLD_ACCESSIBILITY_JSONLD, str)
    # The JSON-LD mirrors the gold accessibility feature set.
    jsonld = gold_shell_markup.GOLD_ACCESSIBILITY_JSONLD
    for token in (
        "alternativeText",
        "readingOrder",
        "structuralNavigation",
        "tableOfContents",
        "ARIA",
        '"accessibilityHazard": "none"',
        "accessMode",
        "accessibilitySummary",
    ):
        assert token in jsonld, f"JSON-LD missing {token!r}"


def test_gold_doc_open_matches_doc_open_format_contract():
    # Same slot contract as shell.DOC_OPEN: format(lang=..., title=...) works.
    rendered = gold_shell_markup.GOLD_DOC_OPEN.format(lang="en", title="X")
    assert '<html lang="en">' in rendered
    assert "<title>X</title>" in rendered
    # The reference shell accepts the identical kwargs.
    assert DOC_OPEN.format(lang="en", title="X")
    # No stray unescaped braces survive the format (would have raised already);
    # belt-and-suspenders: the only field names are lang/title.
    field_names = {
        fname
        for _, fname, _, _ in __import__("string").Formatter().parse(
            gold_shell_markup.GOLD_DOC_OPEN
        )
        if fname
    }
    assert field_names == {"lang", "title"}, f"unexpected format fields {field_names}"


def test_no_dart_worktree_import():
    # Self-containment: neither vendored module references the retired-DART
    # worktree or a cross-tree DART.templates import.
    sources = {
        "wcag22_css.py": Path(
            inspect.getsourcefile(__import__("dart_semantic.assembler.wcag22_css", fromlist=["x"]))
        ).read_text(encoding="utf-8"),
        "gold_shell_markup.py": Path(
            inspect.getsourcefile(gold_shell_markup)
        ).read_text(encoding="utf-8"),
    }
    for name, src in sources.items():
        assert ".claude/worktrees" not in src, f"{name} references the DART worktree"
        assert "DART.templates" not in src, f"{name} imports DART.templates"


def test_assembler_output_unchanged():
    # The DEFAULT (flag-off SEMANTIK_GOLD_SHELL) assembly is deterministic AND
    # free of any gold-shell / vendored-CSS leakage — byte-stable to pre-Phase-7
    # output. (Phase 7 wires gold_shell_markup into pass_9a, but it is gated, so
    # the flag-off path is unchanged.)
    doc_a = _assemble_fixture()
    doc_b = _assemble_fixture()
    assert doc_a.html == doc_b.html, "assemble_document is non-deterministic"

    html = doc_a.html
    # The vendored gold shell / CSS must NOT have leaked into the flag-off output.
    for marker in (
        "WCAG22_CSS",
        ".exercise",
        "role=\"contentinfo\"",
        "<article>",
        "application/ld+json",
    ):
        assert marker not in html, f"vendored marker {marker!r} leaked into assembler output"

    import dart_semantic.assembler.api as api_mod
    import dart_semantic.assembler.shell as shell_mod
    import dart_semantic.assembler.fallbacks as fallbacks_mod
    import dart_semantic.assembler.pass_9a as pass_9a_mod

    # The vendored WCAG CSS (Phase 8) is still consumed by NObody.
    for mod in (api_mod, shell_mod, fallbacks_mod, pass_9a_mod):
        src = Path(inspect.getsourcefile(mod)).read_text(encoding="utf-8")
        assert "wcag22_css" not in src, f"{mod.__name__} already imports wcag22_css"

    # Phase 7 wires gold_shell_markup into pass_9a (the per-region gold ARIA
    # wrap), gated behind SEMANTIK_GOLD_SHELL. api/shell/fallbacks still must not
    # import it.
    for mod in (api_mod, shell_mod, fallbacks_mod):
        src = Path(inspect.getsourcefile(mod)).read_text(encoding="utf-8")
        assert "gold_shell_markup" not in src, (
            f"{mod.__name__} already imports gold_shell_markup"
        )
