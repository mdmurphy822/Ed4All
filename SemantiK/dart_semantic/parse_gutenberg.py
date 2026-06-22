"""Project Gutenberg HTML → IR, one document per chapter.

Gutenberg publishes each book as well-structured HTML at
    https://www.gutenberg.org/cache/epub/<id>/pg<id>-images.html
with consistent conventions: the book title in <h1>, chapters in <h2>,
prose in <p>, and section breaks marked by blank lines / <br>.

This parser strips Gutenberg chrome (front-matter license text, transcriber
notes, table-of-contents auto-linking), splits on <h2> chapter boundaries,
and returns one IR Document per chapter. That keeps each training pair
close to our 2048-token budget.

Gutenberg texts are either public-domain (pre-1928 US works) or
explicitly released under permissive terms — commercial-OK.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from . import ir


# Gutenberg's standard front-matter / chrome we strip before walking:
FRONT_SENTINELS = (
    "*** START OF THIS PROJECT GUTENBERG",
    "*** START OF THE PROJECT GUTENBERG",
)
BACK_SENTINELS = (
    "*** END OF THIS PROJECT GUTENBERG",
    "*** END OF THE PROJECT GUTENBERG",
)


def parse_gutenberg(html: str, *, book_id: int,
                    source_url: str | None = None) -> list[ir.Document]:
    """Return one IR Document per chapter. Drops the front/back matter."""
    soup = BeautifulSoup(html, "lxml")

    body = soup.body or soup
    _strip_gutenberg_chrome(body)

    title = _find_title(body)
    blocks = _collect_top_blocks(body)
    chapters = _split_at_h2(blocks)

    docs: list[ir.Document] = []
    for idx, (chapter_title, chapter_blocks) in enumerate(chapters):
        if all(b.name == "h2" for b in chapter_blocks):
            continue
        try:
            doc = _build_chapter_document(
                book_title=title, book_id=book_id,
                chapter_title=chapter_title, chapter_idx=idx,
                blocks=chapter_blocks, source_url=source_url,
            )
        except ir.IRError:
            continue
        docs.append(doc)
    return docs


# ---------- preprocessing ----------

def _strip_gutenberg_chrome(body: Tag) -> None:
    """Remove Gutenberg's front/back matter + transcriber notes."""
    # Gutenberg often wraps front/back matter in divs with classes like
    # "chapter" or explicit text anchors.
    for div in body.find_all("div"):
        cls = div.get("class") or []
        if any("gutenberg" in c.lower() for c in cls):
            div.decompose()

    # Remove everything before the start-of-text sentinel if present.
    text_content = body.get_text()
    for sentinel in FRONT_SENTINELS:
        if sentinel in text_content:
            # Find the parent element containing the sentinel and drop it + preceding siblings.
            for el in body.find_all(string=re.compile(re.escape(sentinel[:40]))):
                parent = el.parent
                if parent:
                    for sib in list(parent.previous_siblings):
                        if isinstance(sib, Tag):
                            sib.decompose()
                    parent.decompose()
            break

    for sentinel in BACK_SENTINELS:
        for el in body.find_all(string=re.compile(re.escape(sentinel[:40]))):
            parent = el.parent
            if parent:
                for sib in list(parent.next_siblings):
                    if isinstance(sib, Tag):
                        sib.decompose()
                parent.decompose()
            break

    # Tables in Gutenberg are usually ToCs we don't need in chapter-sized docs.
    for tbl in body.find_all("table"):
        tbl.decompose()


BLOCK_TAGS = {
    "p", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "dl", "blockquote", "pre",
}


def _collect_top_blocks(body: Tag) -> list[Tag]:
    blocks: list[Tag] = []
    for el in body.find_all(BLOCK_TAGS, recursive=True):
        if any(p.name in BLOCK_TAGS for p in el.parents
               if p is not body and p.name is not None):
            continue
        blocks.append(el)
    return blocks


def _find_title(body: Tag) -> str:
    h1 = body.find("h1")
    if h1:
        text = h1.get_text(" ", strip=True)
        if text:
            return text
    return "Untitled"


# ---------- split ----------

def _split_at_h2(blocks: list[Tag]) -> list[tuple[str, list[Tag]]]:
    """Group blocks by h2 boundary. Front matter before the first h2 is
    discarded (it's license text, transcriber notes, etc.).
    """
    chapters: list[tuple[str, list[Tag]]] = []
    current_title: str | None = None
    current_blocks: list[Tag] = []
    for b in blocks:
        if b.name == "h2":
            if current_title is not None and current_blocks:
                chapters.append((current_title, current_blocks))
            current_title = b.get_text(" ", strip=True) or "Chapter"
            current_blocks = [b]
        elif current_title is not None:
            current_blocks.append(b)
        # Blocks before the first h2 are discarded (front matter).
    if current_title is not None and current_blocks:
        chapters.append((current_title, current_blocks))
    return chapters


# ---------- per-chapter IR construction ----------

def _build_chapter_document(*, book_title: str, book_id: int,
                            chapter_title: str, chapter_idx: int,
                            blocks: list[Tag],
                            source_url: str | None) -> ir.Document:
    ir_blocks: list[ir.Block] = [
        ir.Heading(level=1, runs=[ir.Run(book_title)]),
    ]
    for b in blocks:
        converted = _convert_block(b)
        if converted is None:
            continue
        if isinstance(converted, list):
            ir_blocks.extend(converted)
        else:
            ir_blocks.append(converted)

    if len(ir_blocks) < 2:
        raise ir.IRError(f"Gutenberg book {book_id} chapter {chapter_idx} empty")

    doc = ir.Document(
        title=f"{book_title} — {chapter_title}",
        language="en",
        source="gutenberg",
        source_id=f"{book_id}__{chapter_idx:03d}",
        source_url=source_url,
        blocks=ir_blocks,
    )
    return ir.normalize_heading_levels(doc)


def _convert_block(el: Tag) -> ir.Block | list[ir.Block] | None:
    if el.name == "p":
        runs = _runs(el)
        return ir.Paragraph(runs=runs) if runs else None
    if el.name in {"h2", "h3", "h4", "h5", "h6"}:
        runs = _runs(el)
        if not runs:
            return None
        return ir.Heading(level=int(el.name[1]), runs=runs)
    if el.name in {"ul", "ol"}:
        return _convert_list(el)
    if el.name == "blockquote":
        children: list[ir.Block] = []
        for sub in el.find_all(BLOCK_TAGS, recursive=False):
            c = _convert_block(sub)
            if c is None:
                continue
            if isinstance(c, list):
                children.extend(c)
            else:
                children.append(c)
        if not children:
            runs = _runs(el)
            if runs:
                children = [ir.Paragraph(runs=runs)]
        return ir.Blockquote(children=children) if children else None
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
        runs = _runs(li)
        if runs:
            items.append(ir.ListItem(children=[ir.Paragraph(runs=runs)]))
    return ir.BulletList(kind=kind, items=items) if items else None


def _runs(el: Tag) -> list[ir.Run]:
    """Collapse an element's inline content to runs with bold/italic preserved."""
    out: list[ir.Run] = []
    _walk(el, out, bold=False, italic=False)
    # Merge adjacent same-style runs.
    merged: list[ir.Run] = []
    for r in out:
        if (merged and merged[-1].bold == r.bold
                and merged[-1].italic == r.italic
                and merged[-1].code == r.code):
            merged[-1] = ir.Run(
                text=merged[-1].text + r.text,
                bold=r.bold, italic=r.italic, code=r.code,
            )
        else:
            merged.append(r)
    # Collapse whitespace.
    for r in merged:
        r.text = re.sub(r"\s+", " ", r.text)
    while merged and not merged[0].text.strip():
        merged.pop(0)
    while merged and not merged[-1].text.strip():
        merged.pop()
    return merged


def _walk(node, out: list[ir.Run], *, bold: bool, italic: bool) -> None:
    from bs4 import NavigableString
    if isinstance(node, NavigableString):
        text = str(node)
        if text:
            out.append(ir.Run(text=text, bold=bold, italic=italic))
        return
    if not isinstance(node, Tag):
        return
    name = node.name.lower() if node.name else ""
    b = bold
    i = italic
    if name in {"b", "strong"}:
        b = True
    elif name in {"i", "em"}:
        i = True
    elif name == "br":
        out.append(ir.Run(text=" "))
        return
    for child in node.children:
        _walk(child, out, bold=b, italic=i)
