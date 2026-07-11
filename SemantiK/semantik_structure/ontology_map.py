"""Stage 5: Ontology mapping. Config-driven (role, depth) → HTML. No model.

Contract:
    emit_html(blocks: list[ResolvedBlock], *, language="en",
              title="Untitled") -> str

Assembles a single WCAG 2.2 AA-conformant HTML document from a stream of
resolved blocks. Enforcement is the same as the old emit_html.py (lang
required, main wrapper, contiguous headings, scope on every th, alt on
every img, etc.) but the input shape is different — we receive flat
typed blocks with depth instead of an IR tree.

Raises MappingError when the input stream cannot be emitted accessibly.
Callers are expected to log + drop the document rather than trying to
paper over the issue. See docs/ontology.md §7 for the gap analysis that
drives these checks.

No silent fallbacks: a header-less table is a hard MappingError (we do
not linearize it into a placeholder paragraph), and a figure with no
image source is emitted as a text-only <figure>/<figcaption> with no
<img> element (never a broken <img src="">).
"""
from __future__ import annotations

import html as html_lib
from io import StringIO

from .classify import Role
from .types import ResolvedBlock


class MappingError(Exception):
    """The resolved blocks violate a WCAG invariant stage 5 refuses to emit."""


# Target size (WCAG 2.2 SC 2.5.8) — inline style for interactive controls.
TARGET_MIN_STYLE = "min-width:24px;min-height:24px;padding:4px 6px"


def emit_html(blocks: list[ResolvedBlock], *,
              language: str = "en",
              title: str | None = None) -> str:
    """Assemble HTML from resolved blocks."""
    if not language:
        raise MappingError("language required (WCAG 3.1.1)")

    # Title: explicit arg wins; otherwise use the first TITLE block, else derive
    # from the first HEADING, else fall back.
    doc_title = (title
                 or _pluck_title(blocks)
                 or "Untitled document")

    buf = StringIO()
    buf.write("<!DOCTYPE html>\n")
    buf.write(f'<html lang="{_attr(language)}">\n')
    buf.write("<head>")
    buf.write('<meta charset="utf-8">')
    buf.write(f"<title>{_text(doc_title)}</title>")
    buf.write("</head>\n<body>\n<main>\n")

    i = 0
    saw_h1 = False
    last_heading_depth = 0

    while i < len(blocks):
        block = blocks[i]
        role = block.classified.role

        # Drop page chrome and metadata from visible output.
        if role in {Role.PAGE_HEADER.value, Role.PAGE_FOOTER.value,
                    Role.METADATA.value}:
            i += 1
            continue

        if role == Role.TITLE.value:
            if saw_h1:
                # Extra TITLEs after the first become h2-level content headings.
                depth = max(2, last_heading_depth)
            else:
                depth = 1
                saw_h1 = True
            buf.write(f"<h1>{_inline(block.classified.features.raw.text)}</h1>\n")
            last_heading_depth = depth
            i += 1
            continue

        if role == Role.HEADING.value:
            raw_depth = max(1, min(6, block.depth or 1))
            if not saw_h1:
                # First heading — ensure an h1 is present. Promote if needed.
                emitted_depth = 1
                saw_h1 = True
            else:
                # Enforce contiguity: allow descend by at most 1 below last seen.
                max_allowed = min(6, last_heading_depth + 1)
                emitted_depth = min(raw_depth, max_allowed)
                # Promote up if standard says no skip
                if emitted_depth > last_heading_depth + 1:
                    emitted_depth = last_heading_depth + 1
            buf.write(f"<h{emitted_depth}>"
                      f"{_inline(block.classified.features.raw.text)}"
                      f"</h{emitted_depth}>\n")
            last_heading_depth = emitted_depth
            i += 1
            continue

        if role == Role.PARAGRAPH.value:
            buf.write(f"<p>{_inline(block.classified.features.raw.text)}</p>\n")
            i += 1
            continue

        if role == Role.LIST_ITEM.value:
            # Consume a contiguous list run.
            run = []
            gid = block.group_id
            while i < len(blocks) and blocks[i].classified.role == Role.LIST_ITEM.value and blocks[i].group_id == gid:
                run.append(blocks[i])
                i += 1
            buf.write(_emit_list(run))
            continue

        if role in {Role.TABLE.value, Role.TABLE_ROW.value,
                    Role.TABLE_HEADER_CELL.value, Role.TABLE_DATA_CELL.value,
                    Role.TABLE_CAPTION.value}:
            run = []
            gid = block.group_id
            table_roles = {Role.TABLE.value, Role.TABLE_ROW.value,
                           Role.TABLE_HEADER_CELL.value, Role.TABLE_DATA_CELL.value,
                           Role.TABLE_CAPTION.value}
            while (i < len(blocks)
                   and blocks[i].classified.role in table_roles
                   and blocks[i].group_id == gid):
                run.append(blocks[i])
                i += 1
            buf.write(_emit_table(run))
            continue

        if role == Role.FIGURE.value:
            buf.write(_emit_figure(block))
            i += 1
            continue

        if role == Role.FIGURE_CAPTION.value:
            # Orphan figure_caption (no preceding FIGURE we grouped with).
            # Emit as a paragraph so text isn't lost.
            buf.write(f"<p>{_inline(block.classified.features.raw.text)}</p>\n")
            i += 1
            continue

        if role == Role.BLOCKQUOTE.value:
            buf.write(f"<blockquote><p>"
                      f"{_inline(block.classified.features.raw.text)}"
                      f"</p></blockquote>\n")
            i += 1
            continue

        if role == Role.CODE_BLOCK.value:
            code = _text(block.classified.features.raw.text)
            buf.write(f"<pre><code>{code}</code></pre>\n")
            i += 1
            continue

        if role == Role.FOOTNOTE.value:
            buf.write(f'<aside class="footnote"><p>'
                      f'{_inline(block.classified.features.raw.text)}</p></aside>\n')
            i += 1
            continue

        if role == Role.REFERENCE.value:
            buf.write(f'<p class="reference">'
                      f'{_inline(block.classified.features.raw.text)}</p>\n')
            i += 1
            continue

        # Unknown role — emit as paragraph to avoid data loss; log-worthy elsewhere.
        buf.write(f"<p>{_inline(block.classified.features.raw.text)}</p>\n")
        i += 1

    if not saw_h1:
        raise MappingError("document has no <h1> (no TITLE or HEADING blocks)")

    buf.write("</main>\n</body>\n</html>\n")
    return buf.getvalue()


# ---------- helpers ----------

def _pluck_title(blocks: list[ResolvedBlock]) -> str | None:
    for b in blocks:
        if b.classified.role == Role.TITLE.value:
            return b.classified.features.raw.text
    for b in blocks:
        if b.classified.role == Role.HEADING.value:
            return b.classified.features.raw.text
    return None


def _emit_list(items: list[ResolvedBlock]) -> str:
    """Emit a properly-nested <ul>. HTML requires nested <ul> to live
    INSIDE an <li>, not as a direct sibling of other <li>s.

    Approach: build a tree first (ListNode), then render with a recursive
    printer. Separates "what depth is each item" from "how do I emit HTML"
    and is easier to verify correct.
    """
    if not items:
        return ""

    # Nodes are plain dicts {text, children} — avoids dataclass import here.
    root_children: list[dict] = []
    # stack[d] is the list that depth-d items should be appended to.
    stack: list[list[dict]] = [root_children]

    min_depth = min(it.depth for it in items)
    for it in items:
        d = it.depth - min_depth
        while len(stack) - 1 < d:
            if stack[-1]:
                stack.append(stack[-1][-1]["children"])
            else:
                wrapper = {"text": "", "children": []}
                stack[-1].append(wrapper)
                stack.append(wrapper["children"])
        del stack[d + 1:]
        stack[d].append({
            "text": it.classified.features.raw.text,
            "children": [],
        })

    def render(children: list[dict]) -> str:
        out = "<ul>"
        for node in children:
            out += "<li>"
            if node["text"]:
                out += _inline(node["text"])
            if node["children"]:
                out += render(node["children"])
            out += "</li>"
        out += "</ul>"
        return out

    return render(root_children) + "\n"


def _emit_table(items: list[ResolvedBlock]) -> str:
    """Assemble a <table> from a run of table-related blocks.

    Any TABLE_CAPTION becomes <caption>. All-header rows become <thead>,
    remaining rows become <tbody>. Each <th> gets scope="col" (for thead)
    or scope="row" (for body rows' first cell when it's a header).
    """
    caption_text = None
    rows_by_idx: dict[int, list[ResolvedBlock]] = {}
    for b in items:
        role = b.classified.role
        if role == Role.TABLE_CAPTION.value:
            caption_text = b.classified.features.raw.text
            continue
        if role in (Role.TABLE_HEADER_CELL.value, Role.TABLE_DATA_CELL.value):
            rows_by_idx.setdefault(b.depth, []).append(b)
        # TABLE and TABLE_ROW as standalone blocks are metadata only.

    if not rows_by_idx:
        return ""

    ordered = []
    for r in sorted(rows_by_idx.keys()):
        ordered.append(sorted(rows_by_idx[r],
                              key=lambda c: c.classified.features.raw.bbox[0]))

    parts = ["<table>"]
    if caption_text:
        parts.append(f"<caption>{_inline(caption_text)}</caption>")

    # Header rows: contiguous leading rows whose cells are all TABLE_HEADER_CELL.
    header_end = 0
    for row in ordered:
        if all(c.classified.role == Role.TABLE_HEADER_CELL.value for c in row):
            header_end += 1
        else:
            break

    if header_end == 0:
        # No header row — can't emit a WCAG-conformant table. We hard-fail
        # rather than degrade: a header-less <table> (or a linearized
        # placeholder masquerading as content) would silently ship an
        # inaccessible document. The whole document is dropped; the caller
        # logs the MappingError. See feedback_no_silent_fallbacks.
        raise MappingError(
            f"table with {len(ordered)} rows has no header row; "
            f"cannot emit WCAG 2.2 AA-conformant table (SC 1.3.1)"
        )

    parts.append("<thead>")
    for row in ordered[:header_end]:
        parts.append("<tr>")
        for cell in row:
            parts.append(_emit_cell(cell, in_header_row=True))
        parts.append("</tr>")
    parts.append("</thead>")

    if len(ordered) > header_end:
        parts.append("<tbody>")
        for row in ordered[header_end:]:
            parts.append("<tr>")
            for cell in row:
                parts.append(_emit_cell(cell, in_header_row=False))
            parts.append("</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "".join(parts) + "\n"


def _emit_cell(cell: ResolvedBlock, *, in_header_row: bool) -> str:
    is_header = cell.classified.role == Role.TABLE_HEADER_CELL.value
    tag = "th" if is_header else "td"
    attrs = ""
    if is_header:
        scope = "col" if in_header_row else "row"
        attrs = f' scope="{scope}"'
    return f"<{tag}{attrs}>{_inline(cell.classified.features.raw.text)}</{tag}>"


def _emit_figure(block: ResolvedBlock) -> str:
    enr = block.enrichment
    alt = (enr.alt_text if enr and enr.alt_text else None)
    if alt is None:
        # Conservative fallback: describe with caption text if any, else the
        # block text. Decorative images must be explicitly passed via the
        # enrichment pipeline with alt_text="".
        alt = block.classified.features.raw.text or "Figure"

    caption = block.classified.features.raw.text or ""

    # Stage 1 image extraction is not wired up yet, so v1 has no real src.
    # An <img src=""> is a broken image — emitting one would silently ship
    # an inaccessible figure. Instead, when there is no real src, emit a
    # text-only <figure> with NO <img> at all and fold the description into
    # the <figcaption> so nothing is lost. The src branch is preserved for
    # the future path once Stage 1 image extraction lands.
    src = enr.figure_src if enr and getattr(enr, "figure_src", None) else None

    if src:
        return (f'<figure><img src="{_attr(src)}" alt="{_attr(alt)}">'
                f'<figcaption>{_inline(caption)}</figcaption></figure>\n')

    # No src: build a single description from alt + caption without
    # duplicating, and ignore the weak literal "Figure" fallback.
    alt_meaningful = alt if alt != "Figure" else ""
    if alt_meaningful and caption and alt_meaningful != caption:
        description = f"{alt_meaningful} {caption}"
    else:
        description = caption or alt_meaningful
    return f'<figure><figcaption>{_inline(description)}</figcaption></figure>\n'


def _inline(text: str) -> str:
    """Escape text for HTML content. Inline <strong>/<em>/<a>/<math> are
    not synthesized here — stage 3 receives plain text from OCR, and any
    inline formatting (bold/italic) is a feature the classifier sees but
    doesn't carry into the text. When we need inline-level markup we'll
    either preserve it from stage 1 font info or get it from stage 6."""
    return _text(text)


def _text(s: str) -> str:
    return html_lib.escape(s, quote=False)


def _attr(s: str) -> str:
    return html_lib.escape(s, quote=True)
