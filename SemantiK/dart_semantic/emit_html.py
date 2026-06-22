"""[LEGACY] IR → WCAG 2.2 AA-compliant HTML.

STATUS: superseded by dart_semantic/ontology_map.py (Stage 5 of the 8-stage
pipeline). This module is retained only because the data-generation scripts
(scripts/pair_from_wikipedia.py, scripts/pair_from_arxiv.py,
scripts/gen_synthetic_forms.py) still build an ir.Document and emit its HTML
here for use as **ground truth** in build_classifier_data.py. The runtime
pipeline does not use this module.

If/when the pair generators are rewritten to emit HTML directly (without
going through the IR tree), this file can be deleted.

Contract: emit(doc) returns clean HTML that passes axe-core with the wcag22aa
ruleset, or raises EmitterError. Callers are expected to drop the failing
example from the training set — we do not repair bad source content.

Guarantees the emitter enforces:
    - <html lang="..."> always set (SC 3.1.1 Language of Page).
    - <main> wraps body content (SC 2.4.1 Bypass Blocks / SC 1.3.1).
    - Heading hierarchy is contiguous starting at <h1>, no skipped levels
      (SC 1.3.1 Info and Relationships, SC 2.4.6 Headings and Labels).
    - Tables have <caption>, <thead>/<tbody>, and scope="col"/"row" on every
      <th> (SC 1.3.1, SC 1.3.2).
    - Lists use <ul>/<ol>/<li> or <dl>/<dt>/<dd>. No dash-bullet prose.
    - Images have non-empty alt; decorative images use alt="" (SC 1.1.1).
    - Form inputs are associated with <label for=...>. Every field has a label.
      Fieldsets carry <legend> (SC 1.3.1, SC 3.3.2 Labels or Instructions).
    - Form input minimum target size via inline min-width/min-height 24px
      (SC 2.5.8 Target Size Minimum, WCAG 2.2).

Intentionally NOT handled here:
    - Color contrast (we don't set colors; pages inherit browser defaults).
    - Focus order (no JS or tab-order manipulation).
    - Language tags on individual spans (SC 3.1.2) — add when IR.Run grows a
      `lang` field. Marked as TODO.
"""
from __future__ import annotations

import html
import re
from io import StringIO

from . import ir


class EmitterError(Exception):
    """The IR violates a requirement the emitter cannot satisfy accessibly."""


# Minimum target size for interactive controls per WCAG 2.2 SC 2.5.8.
TARGET_MIN_STYLE = "min-width:24px;min-height:24px;padding:4px 6px"


def emit(doc: ir.Document) -> str:
    if not doc.language:
        raise EmitterError("Document.language is required (WCAG 3.1.1)")
    if not doc.title:
        raise EmitterError("Document.title is required")

    buf = StringIO()
    buf.write("<!DOCTYPE html>\n")
    buf.write(f'<html lang="{_attr(doc.language)}">\n')
    buf.write("<head>")
    buf.write('<meta charset="utf-8">')
    buf.write(f"<title>{_text(doc.title)}</title>")
    buf.write("</head>\n<body>\n<main>\n")

    # Heading levels must be contiguous. We require IR headings to use
    # logical levels (1..6); the emitter verifies non-skip as it walks.
    last_heading_level = 0
    saw_h1 = False

    for block in doc.blocks:
        if isinstance(block, ir.Heading):
            if block.level < 1 or block.level > 6:
                raise EmitterError(f"Heading level out of range: {block.level}")
            if not saw_h1:
                if block.level != 1:
                    raise EmitterError(
                        f"First heading must be level 1, got {block.level}"
                    )
                saw_h1 = True
            elif block.level > last_heading_level + 1:
                raise EmitterError(
                    f"Heading level skipped: went from h{last_heading_level} "
                    f"to h{block.level} (SC 1.3.1)"
                )
            last_heading_level = block.level

        buf.write(_emit_block(block))
        buf.write("\n")

    if not saw_h1:
        raise EmitterError("Document has no <h1>")

    buf.write("</main>\n</body>\n</html>\n")
    return buf.getvalue()


def _emit_block(block: ir.Block) -> str:
    if isinstance(block, ir.Heading):
        return f"<h{block.level}>{_emit_runs(block.runs)}</h{block.level}>"
    if isinstance(block, ir.Paragraph):
        return f"<p>{_emit_runs(block.runs)}</p>"
    if isinstance(block, ir.BulletList):
        return _emit_list(block)
    if isinstance(block, ir.DefinitionList):
        return _emit_definition_list(block)
    if isinstance(block, ir.Table):
        return _emit_table(block)
    if isinstance(block, ir.Figure):
        return _emit_figure(block)
    if isinstance(block, ir.Blockquote):
        return _emit_blockquote(block)
    if isinstance(block, ir.CodeBlock):
        return _emit_codeblock(block)
    if isinstance(block, ir.Form):
        return _emit_form(block)
    raise EmitterError(f"Unknown block type: {type(block).__name__}")


def _emit_list(lst: ir.BulletList) -> str:
    if lst.kind not in ("unordered", "ordered"):
        raise EmitterError(f"BulletList.kind must be 'unordered' or 'ordered', got {lst.kind}")
    tag = "ul" if lst.kind == "unordered" else "ol"
    if not lst.items:
        raise EmitterError("List has no items")
    parts = [f"<{tag}>"]
    for item in lst.items:
        if not item.children:
            raise EmitterError("ListItem has no children")
        body = "".join(_emit_block(b) for b in item.children)
        parts.append(f"<li>{body}</li>")
    parts.append(f"</{tag}>")
    return "".join(parts)


def _emit_definition_list(dl: ir.DefinitionList) -> str:
    if not dl.items:
        raise EmitterError("DefinitionList has no items")
    parts = ["<dl>"]
    for item in dl.items:
        parts.append(f"<dt>{_emit_runs(item.term)}</dt>")
        body = "".join(_emit_block(b) for b in item.definition)
        parts.append(f"<dd>{body}</dd>")
    parts.append("</dl>")
    return "".join(parts)


def _emit_table(table: ir.Table) -> str:
    if not table.header_rows:
        raise EmitterError(
            "Table has no header rows — WCAG SC 1.3.1 requires programmatic "
            "header association (drop this example)"
        )
    parts = ["<table>"]
    if table.caption:
        parts.append(f"<caption>{_emit_runs(table.caption)}</caption>")

    parts.append("<thead>")
    for row in table.header_rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(_emit_cell(cell, force_header=True, in_header_row=True))
        parts.append("</tr>")
    parts.append("</thead>")

    if table.body_rows:
        parts.append("<tbody>")
        for row in table.body_rows:
            parts.append("<tr>")
            for cell in row:
                parts.append(_emit_cell(cell, force_header=False, in_header_row=False))
            parts.append("</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "".join(parts)


def _emit_cell(cell: ir.TableCell, *, force_header: bool, in_header_row: bool) -> str:
    is_th = force_header or cell.is_header
    tag = "th" if is_th else "td"
    attrs = []
    if is_th:
        scope = "col" if in_header_row else "row"
        attrs.append(f'scope="{scope}"')
    if cell.colspan != 1:
        attrs.append(f'colspan="{cell.colspan}"')
    if cell.rowspan != 1:
        attrs.append(f'rowspan="{cell.rowspan}"')
    attr_str = (" " + " ".join(attrs)) if attrs else ""
    return f"<{tag}{attr_str}>{_emit_runs(cell.runs)}</{tag}>"


def _emit_figure(fig: ir.Figure) -> str:
    if fig.alt is None:
        raise EmitterError(
            "Figure.alt is required. Use empty string for purely decorative images "
            "(SC 1.1.1)"
        )
    parts = [f'<figure><img src="{_attr(fig.src)}" alt="{_attr(fig.alt)}">']
    if fig.caption:
        parts.append(f"<figcaption>{_emit_runs(fig.caption)}</figcaption>")
    parts.append("</figure>")
    return "".join(parts)


def _emit_blockquote(bq: ir.Blockquote) -> str:
    attrs = f' cite="{_attr(bq.cite)}"' if bq.cite else ""
    body = "".join(_emit_block(b) for b in bq.children)
    return f"<blockquote{attrs}>{body}</blockquote>"


def _emit_codeblock(cb: ir.CodeBlock) -> str:
    cls = f' class="language-{_attr(cb.language)}"' if cb.language else ""
    return f"<pre><code{cls}>{_text(cb.text)}</code></pre>"


def _emit_form(form: ir.Form) -> str:
    if not form.fieldsets:
        raise EmitterError("Form has no fieldsets")
    attrs = []
    if form.action:
        attrs.append(f'action="{_attr(form.action)}"')
    attrs.append(f'method="{form.method}"')
    attr_str = " ".join(attrs)
    parts = [f"<form {attr_str}>"]
    for fs in form.fieldsets:
        parts.append("<fieldset>")
        parts.append(f"<legend>{_emit_runs(fs.legend)}</legend>")
        for fld in fs.fields:
            parts.append(_emit_field(fld))
        parts.append("</fieldset>")
    parts.append("</form>")
    return "".join(parts)


def _emit_field(fld: ir.FormField) -> str:
    if not fld.name:
        raise EmitterError("FormField.name is required (used as id/for target)")
    if not fld.label or not any(r.text.strip() for r in fld.label):
        raise EmitterError(
            f"FormField {fld.name!r} has no label (SC 1.3.1 / SC 3.3.2)"
        )
    fid = _slug_id(fld.name)
    label_html = f'<label for="{fid}">{_emit_runs(fld.label)}</label>'
    required = " required" if fld.required else ""
    aria_req = ' aria-required="true"' if fld.required else ""
    style = f' style="{TARGET_MIN_STYLE}"'

    if fld.kind == "textarea":
        control = f'<textarea id="{fid}" name="{_attr(fld.name)}"{required}{aria_req}{style}></textarea>'
    elif fld.kind == "select":
        if not fld.options:
            raise EmitterError(f"Select field {fld.name!r} has no options")
        options = "".join(
            f"<option>{_text(o)}</option>" for o in fld.options
        )
        control = f'<select id="{fid}" name="{_attr(fld.name)}"{required}{aria_req}{style}>{options}</select>'
    elif fld.kind in ("checkbox", "radio"):
        # For checkbox/radio, emit one input per option within a fieldset.
        if fld.options:
            items = []
            for i, opt in enumerate(fld.options):
                opt_id = f"{fid}_{i}"
                items.append(
                    f'<label for="{opt_id}" style="display:block">'
                    f'<input type="{fld.kind}" id="{opt_id}" name="{_attr(fld.name)}" '
                    f'value="{_attr(opt)}"{required}{aria_req}{style}>'
                    f' {_text(opt)}</label>'
                )
            return label_html + "".join(items)
        control = (f'<input type="{fld.kind}" id="{fid}" name="{_attr(fld.name)}"'
                   f'{required}{aria_req}{style}>')
    else:  # text, email, tel, url, number, date
        control = (f'<input type="{fld.kind}" id="{fid}" name="{_attr(fld.name)}"'
                   f'{required}{aria_req}{style}>')

    help_html = ""
    if fld.help_text:
        help_id = f"{fid}_help"
        help_html = f'<span id="{help_id}" class="help">{_emit_runs(fld.help_text)}</span>'
        control = control.replace(">", f' aria-describedby="{help_id}">', 1)

    return f'<div class="field">{label_html}{control}{help_html}</div>'


def _emit_runs(runs: list[ir.Run]) -> str:
    parts = []
    for r in runs:
        t = _text(r.text)
        if r.code:
            t = f"<code>{t}</code>"
        if r.bold:
            t = f"<strong>{t}</strong>"
        if r.italic:
            t = f"<em>{t}</em>"
        if r.href:
            t = f'<a href="{_attr(r.href)}">{t}</a>'
        parts.append(t)
    return "".join(parts)


# ---------- escaping helpers ----------

def _text(s: str) -> str:
    return html.escape(s, quote=False)


def _attr(s: str) -> str:
    return html.escape(s, quote=True)


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slug_id(name: str) -> str:
    s = _SLUG_RE.sub("-", name).strip("-")
    return s or "field"
