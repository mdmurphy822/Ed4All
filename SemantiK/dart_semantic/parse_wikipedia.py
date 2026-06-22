"""Wikipedia REST HTML → IR document parser.

Takes the output of https://en.wikipedia.org/api/rest_v1/page/html/<title> and
produces ir.Document instances, one per top-level (h2) section of the article.
The article's lead content (before the first h2) is its own document.

Strategy:
    1. Strip Wikipedia chrome (edit links, navboxes, infoboxes, reference
       sections, citation superscripts, etc.).
    2. Unwrap MediaWiki's <section> elements — they're layout hooks, not
       semantic structure, and block-level elements live inside them.
    3. Partition the remaining block stream at h2 boundaries. Each partition
       becomes one ir.Document with the article title as <h1> followed by the
       section's content.
    4. Translate each block into IR. Tables without header rows, unlabeled
       figures, and other un-emitable shapes cause IRError, which the caller
       is expected to handle by dropping that section.

The emitter-level validator (emit_html + axe) is the final quality gate —
this parser tries to produce emit-able IR but doesn't attempt its own
WCAG checks.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from . import ir

logger = logging.getLogger(__name__)

WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"

# Classes / selectors we consider pure chrome. Stripped before anything else.
STRIP_SELECTORS = [
    "style", "script", "link",
    ".mw-editsection", ".reference", ".mw-references-wrap",
    ".navbox", ".infobox", ".hatnote", ".thumb", ".metadata",
    "sup.reference", "[role='note']",
    "#References", "#External_links", "#See_also", "#Further_reading",
    "#Notes", "#Bibliography", "#Footnotes",
    ".mw-empty-elt",
]

# Block-level tags we translate into IR. Anything else becomes either inline
# content (when nested in a block) or is dropped.
BLOCK_TAGS = {
    "p", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "dl", "table", "figure", "blockquote", "pre",
}


class WikipediaParseError(Exception):
    pass


# ---------- public API ----------

def fetch_article_html(title: str, *, timeout: int = 30) -> str:
    url = WIKI_REST.format(title=quote(title.replace(" ", "_")))
    r = requests.get(url, headers={"User-Agent": "dart-semantic/0.0.1"},
                     timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_article(raw_html: str, article_title: str,
                  article_url: str | None = None) -> list[ir.Document]:
    """Parse one Wikipedia REST HTML document into a list of section-level
    IR documents. Sections that can't be translated cleanly are skipped with a
    warning log; they don't fail the whole article."""
    soup = BeautifulSoup(raw_html, "lxml")

    # Section wrappers are layout-only; flatten.
    for section in soup.find_all("section"):
        section.unwrap()
    for sel in STRIP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    for sup in soup.find_all("sup"):
        sup.decompose()

    body = soup.body or soup
    top_blocks = _collect_top_level_blocks(body)
    sections = _split_at_h2(top_blocks)

    docs: list[ir.Document] = []
    for section_title, blocks in sections:
        # Sections that are just a heading with no body add no structural signal.
        if all(b.name == "h2" for b in blocks):
            continue
        try:
            doc = _build_section_document(article_title, section_title, blocks,
                                          article_url)
        except ir.IRError as exc:
            logger.info(f"drop section {article_title!r} / {section_title!r}: {exc}")
            continue
        docs.append(doc)
    return docs


# ---------- internal: block collection & section split ----------

def _collect_top_level_blocks(body: Tag) -> list[Tag]:
    blocks: list[Tag] = []
    for el in body.find_all(BLOCK_TAGS, recursive=True):
        if any(p.name in BLOCK_TAGS for p in el.parents
               if p is not body and p.name is not None):
            continue
        blocks.append(el)
    return blocks


def _split_at_h2(blocks: list[Tag]) -> list[tuple[str, list[Tag]]]:
    sections: list[tuple[str, list[Tag]]] = []
    current_title = "Lead"
    current_blocks: list[Tag] = []
    for b in blocks:
        if b.name == "h2":
            if current_blocks:
                sections.append((current_title, current_blocks))
            current_title = b.get_text().strip() or "Section"
            current_blocks = [b]
        else:
            current_blocks.append(b)
    if current_blocks:
        sections.append((current_title, current_blocks))
    return sections


def _build_section_document(article_title: str, section_title: str,
                            blocks: list[Tag],
                            article_url: str | None) -> ir.Document:
    ir_blocks: list[ir.Block] = [
        ir.Heading(level=1, runs=[ir.Run(article_title)])
    ]
    for b in blocks:
        converted = _convert_top_block(b)
        if converted is None:
            continue
        if isinstance(converted, list):
            ir_blocks.extend(converted)
        else:
            ir_blocks.append(converted)

    if len(ir_blocks) < 2:
        raise ir.IRError("section has no convertible content")

    return ir.Document(
        title=f"{article_title} — {section_title}"
            if section_title != "Lead" else article_title,
        language="en",
        source="wikipedia",
        source_id=article_title,
        source_url=article_url,
        blocks=ir_blocks,
    )


# ---------- internal: block translation ----------

def _convert_top_block(el: Tag) -> ir.Block | list[ir.Block] | None:
    if el.name == "p":
        runs = _tag_to_runs(el)
        return ir.Paragraph(runs=runs) if _has_text(runs) else None

    if el.name in {"h2", "h3", "h4", "h5", "h6"}:
        # Wikipedia starts at h2 under the article title; map level directly.
        level = int(el.name[1])
        return ir.Heading(level=level, runs=_tag_to_runs(el))

    if el.name in {"ul", "ol"}:
        return _convert_list(el)

    if el.name == "dl":
        return _convert_definition_list(el)

    if el.name == "table":
        return _convert_table(el)

    if el.name == "figure":
        return _convert_figure(el)

    if el.name == "blockquote":
        inner = []
        for child in el.find_all(True, recursive=False):
            converted = _convert_top_block(child)
            if converted is None:
                continue
            if isinstance(converted, list):
                inner.extend(converted)
            else:
                inner.append(converted)
        if not inner:
            runs = _tag_to_runs(el)
            if _has_text(runs):
                inner = [ir.Paragraph(runs=runs)]
        if not inner:
            return None
        return ir.Blockquote(children=inner)

    if el.name == "pre":
        text = el.get_text("", strip=False)
        if not text.strip():
            return None
        return ir.CodeBlock(text=text)

    return None


def _convert_list(el: Tag) -> ir.BulletList | None:
    kind: ir.ListKind = "ordered" if el.name == "ol" else "unordered"
    items: list[ir.ListItem] = []
    for li in el.find_all("li", recursive=False):
        children: list[ir.Block] = []
        # Flatten inline runs at the top of the <li> into one paragraph.
        runs = _tag_to_runs(li, skip_child_block_tags=True)
        if _has_text(runs):
            children.append(ir.Paragraph(runs=runs))
        # Nested block content inside the <li>.
        for child in li.find_all(BLOCK_TAGS, recursive=False):
            converted = _convert_top_block(child)
            if converted is None:
                continue
            if isinstance(converted, list):
                children.extend(converted)
            else:
                children.append(converted)
        if children:
            items.append(ir.ListItem(children=children))
    return ir.BulletList(kind=kind, items=items) if items else None


def _convert_definition_list(el: Tag) -> ir.DefinitionList | None:
    items: list[ir.DefinitionItem] = []
    pending_term: list[ir.Run] | None = None
    pending_defs: list[ir.Block] = []

    def flush():
        if pending_term and pending_defs:
            items.append(ir.DefinitionItem(term=pending_term, definition=pending_defs))

    for child in el.find_all(["dt", "dd"], recursive=False):
        if child.name == "dt":
            flush()
            pending_term = _tag_to_runs(child)
            pending_defs = []
        else:  # dd
            runs = _tag_to_runs(child)
            if _has_text(runs):
                pending_defs.append(ir.Paragraph(runs=runs))
    flush()
    return ir.DefinitionList(items=items) if items else None


def _convert_table(el: Tag) -> ir.Table | None:
    rows = el.find_all("tr")
    if not rows:
        return None

    # Caption
    caption_el = el.find("caption")
    caption = _tag_to_runs(caption_el) if caption_el else None

    # Build cell matrix; decide header rows.
    all_rows: list[list[ir.TableCell]] = []
    for tr in rows:
        cells: list[ir.TableCell] = []
        for cell_el in tr.find_all(["td", "th"], recursive=False):
            cells.append(ir.TableCell(
                runs=_tag_to_runs(cell_el),
                colspan=int(cell_el.get("colspan", 1) or 1),
                rowspan=int(cell_el.get("rowspan", 1) or 1),
                is_header=(cell_el.name == "th"),
            ))
        if cells:
            all_rows.append(cells)
    if not all_rows:
        return None

    # Heuristic: the header rows are the contiguous leading rows whose cells
    # are ALL <th>. If none, we can't emit accessibly — raise IRError.
    header_rows: list[list[ir.TableCell]] = []
    for row in all_rows:
        if all(c.is_header for c in row):
            header_rows.append(row)
        else:
            break
    if not header_rows:
        raise ir.IRError("Wikipedia table has no all-<th> header row — not emitable")

    body_rows = all_rows[len(header_rows):]
    return ir.Table(caption=caption, header_rows=header_rows, body_rows=body_rows)


def _convert_figure(el: Tag) -> ir.Figure | None:
    img = el.find("img")
    if not img:
        return None
    src = img.get("src") or ""
    alt = img.get("alt") or ""
    caption_el = el.find("figcaption")
    caption = _tag_to_runs(caption_el) if caption_el else None

    # Wikipedia images rarely have meaningful alt. Use figcaption text as the
    # alt when alt is empty — screen readers then read the caption.
    if not alt.strip() and caption:
        alt = " ".join(r.text for r in caption)
    if not alt.strip():
        raise ir.IRError("Wikipedia figure has no usable alt or caption")
    return ir.Figure(src=src, alt=alt, caption=caption)


# ---------- internal: inline runs ----------

_INLINE_IGNORE_TAGS = {"abbr", "small", "sub", "sup", "time"}


def _tag_to_runs(el: Tag, *, skip_child_block_tags: bool = False) -> list[ir.Run]:
    runs: list[ir.Run] = []
    _walk_inline(el, runs, context={}, skip_block=skip_child_block_tags)
    return _merge_runs(runs)


def _walk_inline(node, runs: list[ir.Run], context: dict, skip_block: bool = False):
    if isinstance(node, NavigableString):
        text = str(node)
        if not text:
            return
        runs.append(ir.Run(
            text=text,
            bold=context.get("bold", False),
            italic=context.get("italic", False),
            code=context.get("code", False),
            href=context.get("href"),
        ))
        return
    if not isinstance(node, Tag):
        return

    name = node.name
    if skip_block and name in BLOCK_TAGS:
        return

    new_ctx = dict(context)
    if name in {"b", "strong"}:
        new_ctx["bold"] = True
    elif name in {"i", "em"}:
        new_ctx["italic"] = True
    elif name in {"code", "tt"}:
        new_ctx["code"] = True
    elif name == "a":
        href = node.get("href")
        if href:
            # Convert Wikipedia's "./Article_Name" hrefs to absolute URLs so
            # the emitter's output is self-contained.
            if href.startswith("./"):
                href = f"https://en.wikipedia.org/wiki/{href[2:]}"
            new_ctx["href"] = href
    elif name in _INLINE_IGNORE_TAGS:
        # Traverse the text but drop the tag itself.
        pass
    elif name == "br":
        runs.append(ir.Run(text="\n"))
        return

    for child in node.children:
        _walk_inline(child, runs, new_ctx, skip_block=skip_block)


def _merge_runs(runs: list[ir.Run]) -> list[ir.Run]:
    """Coalesce adjacent runs with identical styling."""
    out: list[ir.Run] = []
    for r in runs:
        if out and _same_style(out[-1], r):
            out[-1] = ir.Run(
                text=out[-1].text + r.text,
                bold=r.bold, italic=r.italic, code=r.code, href=r.href,
            )
        else:
            out.append(r)
    # Collapse runs of whitespace.
    for r in out:
        r.text = re.sub(r"[ \t\r\f\v]+", " ", r.text)
    # Drop empty trailing/leading whitespace-only runs.
    while out and not out[0].text.strip():
        out.pop(0)
    while out and not out[-1].text.strip():
        out.pop()
    return out


def _same_style(a: ir.Run, b: ir.Run) -> bool:
    return (a.bold == b.bold and a.italic == b.italic
            and a.code == b.code and a.href == b.href)


def _has_text(runs: list[ir.Run]) -> bool:
    return any(r.text.strip() for r in runs)
