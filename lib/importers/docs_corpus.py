"""Docs-site → clean accessible HTML importer (deterministic, no LLM).

Walks a Markdown / MDX documentation tree and emits one clean semantic
``{slug}.html`` page per source document into an output directory, plus a small
``import_manifest.json`` provenance record. The user then passes that directory
as ``--corpus`` to ``ed4all run textbook-to-course``: the ``semantik_conversion``
phase classifies a directory of plain ``*.html`` as a vendor corpus and
re-normalizes it into the canonical ``{stem}_accessible.html`` contract itself
(``_run_vendor_ingest_conversion``). No PDF/OCR step, no forged provenance
sidecars, no LLM.

Ordering: if an ``mkdocs.yml`` is present its ``nav`` gives the reading order;
documents not referenced by the nav (orphans) are swept in afterwards in sorted
path order. With no ``mkdocs.yml`` (or no PyYAML) the importer degrades to a
sorted ``rglob`` of every Markdown document.

Generic Markdown constructs only — YAML frontmatter is stripped, fenced code is
preserved, and admonition blocks (``!!! type "Title"`` / ``??? type``) degrade
to a heading + dedented body. No documentation-tool branding lives in the
emitted HTML.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._markdown import (
    _strip_mdx,
    build_sections,
    md_to_html,
    scan_leaks,
    slugify,
)

_MD_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})

# YAML frontmatter delimited by leading `---` fences.
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# A single ATX `# Title` line (first-level heading), used for title derivation.
_FIRST_H1 = re.compile(r"^\s*#\s+(.+?)\s*#*\s*$", re.MULTILINE)
# Admonition opener: `!!! note "Title"` or `???+ warning` (mkdocs / PyMarkdown
# extension syntax, but the shape is generic — degrade to heading + body).
_ADMONITION = re.compile(
    r'^(?P<indent>\s*)(?:!!!|\?\?\?\+?)\s+(?P<type>[A-Za-z0-9_-]+)'
    r'(?:\s+"(?P<title>[^"]*)")?\s*$'
)


@dataclass
class DocsImportResult:
    """Outcome of an :func:`import_docs_corpus` run."""

    output_dir: Path
    manifest_path: Path
    documents: List[Dict[str, Any]] = field(default_factory=list)
    ordering_source: str = "rglob"
    orphan_count: int = 0

    @property
    def doc_count(self) -> int:
        return len(self.documents)


def _strip_frontmatter(md: str) -> Tuple[str, Dict[str, str]]:
    """Strip a leading YAML frontmatter block; return (body, parsed_fields).

    Only the flat top-level scalar fields are surfaced (enough for a title);
    parsing degrades to an empty dict if PyYAML is unavailable or the block is
    not a mapping.
    """
    m = _FRONTMATTER.match(md)
    if not m:
        return md, {}
    body = md[m.end():]
    fields: Dict[str, str] = {}
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(m.group(1))
        if isinstance(parsed, dict):
            fields = {str(k): v for k, v in parsed.items()}
    except Exception:  # noqa: BLE001 — frontmatter is best-effort metadata
        fields = {}
    return body, fields


def _degrade_admonitions(md: str) -> str:
    """Degrade admonition blocks to a heading + dedented body.

    ``!!! note "Something important"`` followed by an indented body becomes a
    ``### Something important`` heading (or the title-cased type when no title
    is given) and the body lines dedented back to column zero, so the standard
    Markdown block renderer picks them up as a real section + prose. Nested
    admonitions collapse to the same flat treatment.
    """
    lines = md.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        m = _ADMONITION.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        base_indent = len(m.group("indent"))
        title = m.group("title")
        if title is None or not title.strip():
            title = m.group("type").replace("_", " ").replace("-", " ").title()
        # Collect the indented body (blank lines allowed inside the block).
        i += 1
        body: List[str] = []
        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                body.append("")
                i += 1
                continue
            indent = len(ln) - len(ln.lstrip())
            if indent <= base_indent:
                break
            body.append(ln[base_indent + 4:] if len(ln) > base_indent + 4
                        else ln.lstrip())
            i += 1
        # Trim trailing blanks accrued from the block.
        while body and not body[-1].strip():
            body.pop()
        out.append("")
        out.append(f"### {title.strip()}")
        out.extend(body)
        out.append("")
    return "\n".join(out)


def _derive_title(body: str, frontmatter: Dict[str, Any], rel: Path) -> str:
    """Title = frontmatter ``title`` > first ``# `` heading > humanized stem."""
    ft = frontmatter.get("title")
    if isinstance(ft, str) and ft.strip():
        return ft.strip()
    h1 = _FIRST_H1.search(body)
    if h1:
        cleaned = re.sub(r"[*_`]", "", h1.group(1)).strip()
        if cleaned:
            return cleaned
    stem = rel.stem
    if stem.lower() in ("index", "readme"):
        # Use the parent directory name for an index page when possible.
        stem = rel.parent.name or stem
    return stem.replace("_", " ").replace("-", " ").strip().title() or "Document"


def _render_document(path: Path) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Render one Markdown/MDX file → (title, page_html, sections)."""
    raw = path.read_text(encoding="utf-8")
    body, frontmatter = _strip_frontmatter(raw)
    if path.suffix.lower() == ".mdx":
        body = _strip_mdx(body)
    body = _degrade_admonitions(body)
    title = _derive_title(body, frontmatter, path)
    inner = md_to_html(body)
    import html as _html

    page_html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\"/>\n"
        f"<title>{_html.escape(title)}</title>\n</head>\n<body>\n"
        f"<h1>{_html.escape(title)}</h1>\n{inner}\n</body>\n</html>\n"
    )
    sections = build_sections(inner, provenance_source="docs-import")
    return title, page_html, sections


def _nav_files(mkdocs_yml: Path) -> Tuple[List[Path], Path]:
    """Return (ordered doc paths, docs_dir) from an ``mkdocs.yml`` nav.

    Degrades to ``([], docs_dir)`` when PyYAML is missing or the file has no
    usable nav. ``docs_dir`` honors the mkdocs ``docs_dir`` key (default
    ``docs``), resolved relative to the mkdocs.yml location.
    """
    docs_dir = mkdocs_yml.parent / "docs"
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(mkdocs_yml.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — degrade to rglob on any parse failure
        return [], docs_dir
    if not isinstance(cfg, dict):
        return [], docs_dir
    dd = cfg.get("docs_dir")
    if isinstance(dd, str) and dd.strip():
        docs_dir = mkdocs_yml.parent / dd

    ordered: List[Path] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if Path(node).suffix.lower() in _MD_SUFFIXES:
                ordered.append(docs_dir / node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(cfg.get("nav"))
    return ordered, docs_dir


def import_docs_corpus(
    src_dir: str | Path,
    output_dir: str | Path,
    *,
    source_name: Optional[str] = None,
    license_note: Optional[str] = None,
    provenance_tag: str = "docs-import",
) -> DocsImportResult:
    """Import a docs tree at ``src_dir`` → clean HTML pages in ``output_dir``.

    Emits one ``{slug}.html`` per source document (slug derived from the
    relative source path, so pages never collide) and an ``import_manifest.json``
    recording the source name, license note, and provenance tag alongside a
    per-document inventory. Returns a :class:`DocsImportResult`.

    Ordering honors an ``mkdocs.yml`` nav when present; unreferenced documents
    (orphans) are appended in sorted path order. Deterministic and LLM-free.
    """
    src = Path(src_dir)
    if not src.is_dir():
        raise NotADirectoryError(f"docs source is not a directory: {src}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Resolve reading order via mkdocs nav (if any), else sorted rglob.
    ordering_source = "rglob"
    ordered: List[Path] = []
    scan_root = src
    mkdocs_yml = None
    for candidate in (src / "mkdocs.yml", src / "mkdocs.yaml"):
        if candidate.is_file():
            mkdocs_yml = candidate
            break
    if mkdocs_yml is not None:
        nav_files, docs_dir = _nav_files(mkdocs_yml)
        if docs_dir.is_dir():
            scan_root = docs_dir
        # Keep only nav entries that actually exist on disk (deterministic).
        ordered = [p for p in nav_files if p.is_file()]
        if ordered:
            ordering_source = "mkdocs-nav"

    # 2. Orphan sweep — every Markdown doc under the scan root not already in
    #    the nav order, appended in sorted path order.
    seen = {p.resolve() for p in ordered}
    all_docs = sorted(
        p for p in scan_root.rglob("*")
        if p.is_file() and p.suffix.lower() in _MD_SUFFIXES
    )
    orphans = [p for p in all_docs if p.resolve() not in seen]
    documents_in_order = ordered + orphans
    orphan_count = len(orphans) if ordering_source == "mkdocs-nav" else 0

    # 3. Render each document to a clean HTML page + collect manifest rows.
    used_slugs: set[str] = set()
    manifest_docs: List[Dict[str, Any]] = []
    for order_idx, path in enumerate(documents_in_order):
        try:
            rel = path.relative_to(scan_root)
        except ValueError:
            rel = Path(path.name)
        # Slug from the relative path (without suffix) to guarantee uniqueness.
        slug = slugify(str(rel.with_suffix("")))
        base_slug = slug
        dedup = 1
        while slug in used_slugs:
            dedup += 1
            slug = f"{base_slug}_{dedup}"
        used_slugs.add(slug)

        title, page_html, sections = _render_document(path)
        html_name = f"{slug}.html"
        (out / html_name).write_text(page_html, encoding="utf-8")

        manifest_docs.append({
            "order": order_idx,
            "slug": slug,
            "title": title,
            "source_path": str(rel),
            "html_file": html_name,
            "sections": len(sections),
            "orphan": path in orphans and ordering_source == "mkdocs-nav",
            "leaks": scan_leaks(page_html),
        })

    # 4. Emit the import manifest (provenance record, NOT a pipeline sidecar).
    manifest = {
        "importer": "ed4all.import-docs",
        "importer_version": 1,
        "source_name": source_name or src.name,
        "source_dir": str(src),
        "license_note": license_note or "",
        "provenance_tag": provenance_tag,
        "ordering_source": ordering_source,
        "orphan_count": orphan_count,
        "document_count": len(manifest_docs),
        "documents": manifest_docs,
    }
    manifest_path = out / "import_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return DocsImportResult(
        output_dir=out,
        manifest_path=manifest_path,
        documents=manifest_docs,
        ordering_source=ordering_source,
        orphan_count=orphan_count,
    )
