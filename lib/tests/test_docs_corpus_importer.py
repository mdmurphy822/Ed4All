"""Tests for the docs-corpus importer (``lib.importers.docs_corpus``).

Deterministic, LLM-free. Synthetic Markdown fixtures only — no real corpus
names, slugs, or paths. Covers Markdown→HTML conversion, mkdocs-nav ordering,
orphan sweep, frontmatter/admonition handling, and manifest emission.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.importers import import_docs_corpus
from lib.importers.docs_corpus import _degrade_admonitions, _strip_frontmatter


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Conversion basics
# ---------------------------------------------------------------------------
def test_converts_markdown_to_accessible_html(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(
        src,
        "guide.md",
        "# Getting Started\n\n"
        "This is intro prose about widgets.\n\n"
        "## Configuration\n\n"
        "Set the `timeout` value.\n\n"
        "```python\nx = 1\n```\n",
    )

    result = import_docs_corpus(src, out)

    assert result.doc_count == 1
    doc = result.documents[0]
    html = (out / doc["html_file"]).read_text(encoding="utf-8")
    # Synthetic page <h1> is the chapter; markdown headings shift down one
    # level (`#`->h2, `##`->h3), so the sole page <h1> is the title.
    assert "<h1>Getting Started</h1>" in html
    assert "<h3>Configuration</h3>" in html
    assert "<code>timeout</code>" in html
    # Fenced code preserved.
    assert '<pre><code class="language-python">' in html
    # Plain .html (NOT _accessible.html) so the vendor-ingest seam picks it up.
    assert doc["html_file"].endswith(".html")
    assert not doc["html_file"].endswith("_accessible.html")


def test_title_from_frontmatter_beats_heading(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(
        src,
        "page.md",
        "---\ntitle: Canonical Title\n---\n\n# A Different Heading\n\nBody.\n",
    )

    result = import_docs_corpus(src, out)
    doc = result.documents[0]
    html = (out / doc["html_file"]).read_text(encoding="utf-8")
    assert doc["title"] == "Canonical Title"
    assert "<h1>Canonical Title</h1>" in html
    # Frontmatter block must not leak into the body.
    assert "title: Canonical Title" not in html


def test_strip_frontmatter_helper():
    body, fields = _strip_frontmatter("---\ntitle: X\nweight: 3\n---\nHello\n")
    assert body.strip() == "Hello"
    assert fields["title"] == "X"
    # No frontmatter → unchanged, empty fields.
    body2, fields2 = _strip_frontmatter("No frontmatter here\n")
    assert body2 == "No frontmatter here\n"
    assert fields2 == {}


# ---------------------------------------------------------------------------
# Admonition degrade
# ---------------------------------------------------------------------------
def test_admonition_degrades_to_heading_and_body():
    md = (
        "Intro line.\n\n"
        '!!! warning "Be Careful"\n'
        "    Do not delete this.\n"
        "    Second body line.\n\n"
        "After the block.\n"
    )
    degraded = _degrade_admonitions(md)
    assert "### Be Careful" in degraded
    assert "Do not delete this." in degraded
    # Body is dedented back to column zero.
    assert "\nDo not delete this." in degraded
    assert "After the block." in degraded
    # No admonition marker survives.
    assert "!!!" not in degraded


def test_admonition_without_title_uses_type():
    md = '!!! note\n    A tip.\n'
    degraded = _degrade_admonitions(md)
    assert "### Note" in degraded
    assert "A tip." in degraded


def test_admonition_rendered_in_page(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(
        src,
        "adm.md",
        "# Doc\n\n"
        '!!! danger "Critical Step"\n'
        "    Back up your data first.\n",
    )
    result = import_docs_corpus(src, out)
    html = (out / result.documents[0]["html_file"]).read_text(encoding="utf-8")
    assert "Critical Step" in html
    assert "Back up your data first." in html
    assert "!!!" not in html


# ---------------------------------------------------------------------------
# mkdocs nav ordering + orphan sweep
# ---------------------------------------------------------------------------
def test_mkdocs_nav_ordering(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(src, "docs/intro.md", "# Intro\n\nStart here.\n")
    _write(src, "docs/deep/advanced.md", "# Advanced\n\nHard stuff.\n")
    _write(src, "docs/middle.md", "# Middle\n\nMiddle stuff.\n")
    _write(
        src,
        "mkdocs.yml",
        "site_name: Test\n"
        "nav:\n"
        "  - Intro: intro.md\n"
        "  - Group:\n"
        "    - Advanced: deep/advanced.md\n"
        "    - Middle: middle.md\n",
    )

    result = import_docs_corpus(src, out)
    assert result.ordering_source == "mkdocs-nav"
    order = [d["title"] for d in result.documents]
    assert order == ["Intro", "Advanced", "Middle"]
    assert result.orphan_count == 0


def test_orphan_sweep_appends_unlisted_docs(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(src, "docs/a.md", "# A\n\nA body.\n")
    _write(src, "docs/z_orphan.md", "# Zeta Orphan\n\nNot in nav.\n")
    _write(
        src,
        "mkdocs.yml",
        "nav:\n  - A: a.md\n",
    )

    result = import_docs_corpus(src, out)
    assert result.ordering_source == "mkdocs-nav"
    assert result.orphan_count == 1
    titles = [d["title"] for d in result.documents]
    # Nav doc first, orphan swept in after.
    assert titles[0] == "A"
    assert "Zeta Orphan" in titles
    orphan_flags = {d["title"]: d["orphan"] for d in result.documents}
    assert orphan_flags["Zeta Orphan"] is True
    assert orphan_flags["A"] is False


def test_no_mkdocs_degrades_to_sorted_rglob(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(src, "b.md", "# Bravo\n\nB.\n")
    _write(src, "a.md", "# Alpha\n\nA.\n")

    result = import_docs_corpus(src, out)
    assert result.ordering_source == "rglob"
    order = [d["source_path"] for d in result.documents]
    assert order == sorted(order)


# ---------------------------------------------------------------------------
# Manifest emission
# ---------------------------------------------------------------------------
def test_manifest_emitted_with_provenance(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(src, "one.md", "# One\n\nBody one.\n")
    _write(src, "two.md", "# Two\n\nBody two.\n")

    result = import_docs_corpus(
        src,
        out,
        source_name="Acme Handbook",
        license_note="CC BY 4.0",
        provenance_tag="acme-docs",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_name"] == "Acme Handbook"
    assert manifest["license_note"] == "CC BY 4.0"
    assert manifest["provenance_tag"] == "acme-docs"
    assert manifest["document_count"] == 2
    assert {d["html_file"] for d in manifest["documents"]} == {
        p.name for p in out.glob("*.html")
    }
    # Section provenance carries the neutral docs-import tag (brand-neutral).
    # Every emitted page renders and has ≥1 section.
    for d in manifest["documents"]:
        assert d["sections"] >= 1
        assert (out / d["html_file"]).exists()


def test_unique_slugs_for_same_stem_in_different_dirs(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(src, "alpha/index.md", "# Alpha Section\n\nA.\n")
    _write(src, "beta/index.md", "# Beta Section\n\nB.\n")

    result = import_docs_corpus(src, out)
    slugs = [d["slug"] for d in result.documents]
    assert len(slugs) == len(set(slugs))  # no collisions
    files = {d["html_file"] for d in result.documents}
    assert len(files) == 2


def test_mdx_jsx_stripped(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(
        src,
        "lesson.mdx",
        "---\ntitle: Lesson\n---\n\n"
        "import Foo from './foo'\n\n"
        "# Lesson\n\n"
        "Real prose here.\n\n"
        '<Question choices={[{a: 1}]} />\n',
    )
    result = import_docs_corpus(src, out)
    html = (out / result.documents[0]["html_file"]).read_text(encoding="utf-8")
    assert "Real prose here." in html
    assert "import Foo" not in html
    assert "<Question" not in html
    assert "&lt;Question" not in html


def test_empty_dir_returns_zero_docs(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    result = import_docs_corpus(src, out)
    assert result.doc_count == 0
    assert result.manifest_path.exists()


def test_src_must_be_directory(tmp_path: Path):
    f = tmp_path / "file.md"
    f.write_text("# X\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        import_docs_corpus(f, tmp_path / "out")
