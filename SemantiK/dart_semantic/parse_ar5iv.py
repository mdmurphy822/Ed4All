"""ar5iv (LaTeXML) HTML → IR document parser.

Handles the HTML produced by LaTeXML for arXiv papers, served from
https://ar5iv.labs.arxiv.org/html/<arxiv_id>. LaTeXML emits a well-defined
class vocabulary (`ltx_*` classes everywhere); this parser walks it into IR.

Key decisions, per project data strategy memory:
    * Strip all `ltx_*` classes — they're styling hooks, not semantic.
    * Preserve <math> elements verbatim. Modern screen readers handle MathML
      well; keeping it is a genuine accessibility win DART gets "for free".
    * Drop figures without <figcaption> — we have no alt-text ground truth.
    * Strip bibliography sections (`section.ltx_bibliography`). Reference
      lists are repetitive, low structural signal, and bloat training tokens.
    * Map heading levels by LaTeXML convention:
        h1 (ltx_title_document) -> IR h1 (article title)
        h2 (ltx_title_section)  -> IR h2
        h3 (ltx_title_subsection) -> IR h3
        etc.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from . import ir

logger = logging.getLogger(__name__)

AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

STRIP_SELECTORS = [
    "style", "script", "link", "meta",
    ".ltx_page_header", ".ltx_page_footer",
    ".ltx_bibliography",         # References list
    ".ltx_role_footnote",        # Inline footnote markers
    ".ltx_note_content",         # Footnote bodies
    ".ltx_tag",                  # Section-number tags we'll reconstruct from text
    ".ltx_pagination",
    ".ltx_dates",
    "button",
    "noscript",
]

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "dl", "table", "figure", "blockquote", "pre",
}
# <section> and some <div> wrappers are structural containers we traverse
# INTO (via find_all recursive=True) but don't emit. They must not be in
# BLOCK_TAGS, or their children get mis-flagged as nested-inside-a-block.
CONTAINER_TAGS = {"section", "article"}


class Ar5ivParseError(Exception):
    pass


# ---------- public API ----------

def fetch_ar5iv_html(arxiv_id: str, *, timeout: int = 30) -> str:
    """Fetch one paper's LaTeXML-rendered HTML from ar5iv.

    Raises Ar5ivParseError on 404 (paper not yet converted) or network error.
    """
    url = AR5IV_URL.format(arxiv_id=arxiv_id)
    try:
        r = requests.get(url, headers={"User-Agent": "dart-semantic/0.0.1"},
                         timeout=timeout)
    except requests.RequestException as exc:
        raise Ar5ivParseError(f"fetch failed: {exc}") from exc
    if r.status_code == 404:
        raise Ar5ivParseError(f"ar5iv has no HTML for {arxiv_id}")
    r.raise_for_status()
    return r.text


def parse_paper(raw_html: str, arxiv_id: str, *,
                paper_url: str | None = None) -> ir.Document:
    """Parse one ar5iv HTML document into a single IR document.

    Unlike the Wikipedia parser, we do NOT section-split here — the caller
    decides whether to train on whole papers or split by section. The emitter's
    strictness (table-header requirements, non-empty alt, etc.) still applies.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    for sel in STRIP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    # Find the article container. LaTeXML wraps everything in ltx_page_main /
    # ltx_document; fall back to body if neither is present.
    article = soup.select_one(".ltx_document") or soup.select_one("article") or soup.body
    if article is None:
        raise Ar5ivParseError("no article body found")

    title = _extract_title(article) or f"arXiv:{arxiv_id}"
    language = "en"

    ir_blocks: list[ir.Block] = [ir.Heading(level=1, runs=[ir.Run(title)])]
    _walk_article(article, ir_blocks)

    if len(ir_blocks) < 2:
        raise ir.IRError(f"arXiv paper {arxiv_id} produced no body content")

    doc = ir.Document(
        title=title,
        language=language,
        source="arxiv",
        source_id=arxiv_id,
        source_url=paper_url or AR5IV_URL.format(arxiv_id=arxiv_id),
        blocks=ir_blocks,
    )
    # Normalize heading skips (h2 -> h4 is a frequent LaTeXML artifact).
    return ir.normalize_heading_levels(doc)


# ---------- title extraction ----------

def _extract_title(article: Tag) -> str | None:
    # LaTeXML title: <h1 class="ltx_title ltx_title_document">
    t = article.select_one("h1.ltx_title_document, h1.ltx_title")
    if t:
        text = t.get_text(" ", strip=True)
        if text:
            return text
    return None


# ---------- article walking ----------

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONVERT_TAGS = BLOCK_TAGS | HEADING_TAGS


def _walk_article(container: Tag, out: list[ir.Block]) -> None:
    """Traverse the article depth-first, emitting IR blocks in document order.

    LaTeXML wraps paragraphs in <div class="ltx_para"><p class="ltx_p">...</p></div>
    and sections in <section class="ltx_section">; a naive walk of container.children
    only sees those wrapper divs/sections and misses the real blocks. We descend
    into structural wrappers and pick up each block at its innermost tag, skipping
    blocks that are nested inside another block we've already chosen to emit
    (e.g., a <p> inside a <figcaption> inside a <figure>).
    """
    # Find every convertible tag anywhere in the container, in document order.
    candidates = container.find_all(CONVERT_TAGS, recursive=True)

    # For each candidate, decide whether to emit it. We skip a candidate when
    # one of its ancestors (up to but not including `container`) is also a
    # candidate — that ancestor will have handled its inner content via
    # _convert_block.
    for el in candidates:
        nested = False
        for parent in el.parents:
            if parent is container or not isinstance(parent, Tag):
                break
            if parent.name in CONVERT_TAGS:
                nested = True
                break
        if nested:
            continue

        if el.name in HEADING_TAGS:
            classes = el.get("class") or []
            if "ltx_title_document" in classes:
                continue
            if "ltx_title_abstract" in classes:
                # We synthesize the abstract heading ourselves below.
                continue
            runs = _tag_to_runs(el)
            if not _has_text(runs):
                continue
            out.append(ir.Heading(level=int(el.name[1]), runs=runs))
            continue

        # Abstract wrapper: emit a synthesized "Abstract" heading in front of
        # the abstract's paragraphs (LaTeXML uses <h6 class="ltx_title_abstract">
        # which is the wrong level).
        if (el.name == "div"
                and "ltx_abstract" in (el.get("class") or [])):
            out.append(ir.Heading(level=2, runs=[ir.Run("Abstract")]))
            # Recurse into the abstract to emit its paragraphs.
            _walk_article(el, out)
            continue

        try:
            converted = _convert_block(el)
        except ir.IRError as exc:
            logger.debug(f"drop block {el.name}: {exc}")
            continue
        if converted is None:
            continue
        if isinstance(converted, list):
            out.extend(converted)
        else:
            out.append(converted)


def _convert_block(el: Tag) -> ir.Block | list[ir.Block] | None:
    if el.name == "p":
        runs = _tag_to_runs(el)
        return ir.Paragraph(runs=runs) if _has_text(runs) else None

    if el.name in {"ul", "ol"}:
        return _convert_list(el)

    if el.name == "dl":
        return _convert_definition_list(el)

    if el.name == "table":
        return _convert_table(el)

    if el.name == "figure":
        return _convert_figure(el)

    if el.name == "blockquote":
        children: list[ir.Block] = []
        for sub in el.find_all(True, recursive=False):
            converted = _convert_block(sub)
            if converted is None:
                continue
            if isinstance(converted, list):
                children.extend(converted)
            else:
                children.append(converted)
        if not children:
            runs = _tag_to_runs(el)
            if _has_text(runs):
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
        children: list[ir.Block] = []
        runs = _tag_to_runs(li, skip_child_block_tags=True)
        if _has_text(runs):
            children.append(ir.Paragraph(runs=runs))
        for sub in li.find_all(BLOCK_TAGS, recursive=False):
            converted = _convert_block(sub)
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
    term: list[ir.Run] | None = None
    defs: list[ir.Block] = []

    def flush():
        if term and defs:
            items.append(ir.DefinitionItem(term=term, definition=defs))

    for ch in el.find_all(["dt", "dd"], recursive=False):
        if ch.name == "dt":
            flush()
            term = _tag_to_runs(ch)
            defs = []
        else:
            runs = _tag_to_runs(ch)
            if _has_text(runs):
                defs.append(ir.Paragraph(runs=runs))
    flush()
    return ir.DefinitionList(items=items) if items else None


def _convert_table(el: Tag) -> ir.Table | None:
    rows = el.find_all("tr")
    if not rows:
        return None

    caption_el = el.find("caption") or el.find_previous_sibling("figcaption")
    caption = _tag_to_runs(caption_el) if caption_el else None

    all_rows: list[list[ir.TableCell]] = []
    for tr in rows:
        cells: list[ir.TableCell] = []
        for c in tr.find_all(["td", "th"], recursive=False):
            cells.append(ir.TableCell(
                runs=_tag_to_runs(c),
                colspan=int(c.get("colspan", 1) or 1),
                rowspan=int(c.get("rowspan", 1) or 1),
                is_header=(c.name == "th"),
            ))
        if cells:
            all_rows.append(cells)
    if not all_rows:
        return None

    header_rows: list[list[ir.TableCell]] = []
    for row in all_rows:
        if all(c.is_header for c in row):
            header_rows.append(row)
        else:
            break
    if not header_rows:
        # LaTeXML usually emits the first row as all-<td> even for header rows
        # (no <thead> guidance in LaTeX source). If the first row looks header-
        # shaped — short text, no colspan — promote it. Otherwise drop.
        first = all_rows[0]
        if all(len(" ".join(r.text for r in c.runs)) < 60 and c.colspan == 1 for c in first):
            for c in first:
                c.is_header = True
            header_rows = [first]
            all_rows = all_rows[1:]
        else:
            raise ir.IRError("arxiv table has no header row — not emitable")
    body_rows = all_rows[len(header_rows):]
    return ir.Table(caption=caption, header_rows=header_rows, body_rows=body_rows)


def _convert_figure(el: Tag) -> ir.Figure | None:
    # LaTeXML sometimes wraps tables in <figure class="ltx_figure ltx_table">.
    # Detect and route to table if there's a <table> inside.
    if el.find("table"):
        return _convert_table(el.find("table"))

    img = el.find("img")
    caption_el = el.find("figcaption")
    caption = _tag_to_runs(caption_el) if caption_el else None

    if not img:
        # Caption-only figure; emit as a Figure with no image? No — requires src.
        return None

    src = img.get("src") or ""
    alt = img.get("alt") or ""
    if not alt.strip() and caption:
        alt = " ".join(r.text for r in caption)[:300]
    if not alt.strip():
        raise ir.IRError("arxiv figure has no usable alt or caption")
    return ir.Figure(src=src, alt=alt, caption=caption)


# ---------- inline runs (with MathML preservation) ----------

_INLINE_IGNORE = {"abbr", "small", "sub", "sup", "time", "span"}


def _tag_to_runs(el: Tag, *, skip_child_block_tags: bool = False) -> list[ir.Run]:
    runs: list[ir.Run] = []
    _walk_inline(el, runs, {}, skip_child_block_tags)
    return _merge_runs(runs)


def _walk_inline(node, runs: list[ir.Run], ctx: dict, skip_block: bool):
    if isinstance(node, NavigableString):
        text = str(node)
        if not text:
            return
        runs.append(ir.Run(
            text=text,
            bold=ctx.get("bold", False),
            italic=ctx.get("italic", False),
            code=ctx.get("code", False),
            href=ctx.get("href"),
        ))
        return
    if not isinstance(node, Tag):
        return

    name = node.name
    if skip_block and name in BLOCK_TAGS:
        return

    # MathML: emit a compact <math> tag carrying the LaTeX alttext rather than
    # the full LaTeXML subtree. ar5iv's <math> is ~100x larger than its LaTeX
    # source because of xref/semantics bookkeeping the model doesn't need.
    # Downstream HTML generators can re-render from alttext via MathJax/KaTeX;
    # screen readers read the alttext directly.
    if name == "math":
        import html as _html
        alttext = (node.get("alttext") or "").strip()
        if alttext:
            display = node.get("display", "inline")
            compact = (f'<math alttext="{_html.escape(alttext, quote=True)}" '
                       f'display="{_html.escape(display, quote=True)}">'
                       f'{_html.escape(alttext, quote=False)}</math>')
            runs.append(ir.Run(text=compact, bold=ctx.get("bold", False)))
        # If there's no alttext, skip entirely — preferring lost math to
        # bloating the training pair past the context budget.
        return

    new_ctx = dict(ctx)
    classes = node.get("class") or []
    if name in {"b", "strong"} or "ltx_text_bold" in classes:
        new_ctx["bold"] = True
    elif name in {"i", "em"} or "ltx_text_italic" in classes:
        new_ctx["italic"] = True
    elif name in {"code", "tt"} or "ltx_verbatim" in classes:
        new_ctx["code"] = True
    elif name == "a":
        href = node.get("href")
        if href:
            # Resolve ar5iv-relative hrefs to absolute ones.
            if href.startswith("#") or href.startswith("/"):
                pass  # let consumers resolve if needed
            new_ctx["href"] = href
    elif name == "br":
        runs.append(ir.Run(text="\n"))
        return

    for child in node.children:
        _walk_inline(child, runs, new_ctx, skip_block)


def _merge_runs(runs: list[ir.Run]) -> list[ir.Run]:
    out: list[ir.Run] = []
    for r in runs:
        # Don't coalesce runs that contain MathML — they need to stay intact.
        if out and _same_style(out[-1], r) and not _has_math(out[-1]) and not _has_math(r):
            out[-1] = ir.Run(
                text=out[-1].text + r.text,
                bold=r.bold, italic=r.italic, code=r.code, href=r.href,
            )
        else:
            out.append(r)
    for r in out:
        if not _has_math(r):
            r.text = re.sub(r"[ \t\r\f\v]+", " ", r.text)
    while out and not out[0].text.strip():
        out.pop(0)
    while out and not out[-1].text.strip():
        out.pop()
    return out


def _has_math(r: ir.Run) -> bool:
    return "<math" in r.text


def _same_style(a: ir.Run, b: ir.Run) -> bool:
    return (a.bold == b.bold and a.italic == b.italic
            and a.code == b.code and a.href == b.href)


def _has_text(runs: list[ir.Run]) -> bool:
    return any(r.text.strip() for r in runs)
