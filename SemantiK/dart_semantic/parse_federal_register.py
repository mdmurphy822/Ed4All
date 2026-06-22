"""Federal Register XML → legacy IR (Document tree).

The Federal Register publishes every document in a structured XML schema
at https://www.federalregister.gov/documents/full_text/xml/<YYYY>/<MM>/<DD>/<id>.xml.
That XML is the canonical source of truth for document structure — the
rendered HTML page is a client-side shell with almost no content.

This parser maps FR's custom tag vocabulary onto our legacy IR:
  <RULE>/<PRORULE>/<NOTICE>/<PRESDOCU> root            -> Document
  <SUBJECT> or <TITLE> in PREAMB                         -> Document.title / H1
  <HD SOURCE="HED">AGENCY:</HD>                          -> H2 (preamble section)
  <HD SOURCE="HD1"|"HD2"|"HD3">                          -> H2 / H3 / H4
  <P>                                                    -> Paragraph
  <LI>                                                   -> ListItem (batched into BulletList)
  <GPH>                                                  -> skipped (no image data in XML)

Unknown tags are walked-into so inline text is preserved.

Public domain — FR content is a federal government publication.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from . import ir


# Tag → heading level mapping for HD elements based on SOURCE attribute.
HD_LEVEL = {
    "HED": 2,
    "HD1": 2,
    "HD2": 3,
    "HD3": 4,
    "HD4": 5,
    "HD5": 6,
}


def parse_fr_xml(xml_text: str,
                 *,
                 document_number: str,
                 title: str,
                 source_url: str | None = None) -> ir.Document:
    """Parse one Federal Register XML document into an IR Document."""
    root = ET.fromstring(xml_text)

    blocks: list[ir.Block] = [ir.Heading(level=1, runs=[ir.Run(title)])]

    # The root element is RULE / PRORULE / NOTICE / PRESDOCU etc.
    # Walk all children in order, converting each element we recognize.
    _walk(root, blocks)

    # Coalesce consecutive ListItem blocks into BulletLists for clean emission.
    blocks = _group_list_items(blocks)

    if len(blocks) < 2:
        raise ir.IRError(f"FR document {document_number} had no body content")

    doc = ir.Document(
        title=title,
        language="en",
        source="federal_register",
        source_id=document_number,
        source_url=source_url,
        blocks=blocks,
    )
    return ir.normalize_heading_levels(doc)


def _walk(element: ET.Element, blocks: list[ir.Block]) -> None:
    """Depth-first walk that emits IR blocks as we encounter them."""
    for child in element:
        tag = child.tag.upper()

        if tag == "HD":
            source = child.get("SOURCE", "HED")
            level = HD_LEVEL.get(source, 3)
            text = _text(child)
            if text:
                blocks.append(ir.Heading(level=level, runs=[ir.Run(text)]))
            continue

        if tag == "P":
            runs = _inline_runs(child)
            if runs:
                blocks.append(ir.Paragraph(runs=runs))
            continue

        if tag == "LI":
            runs = _inline_runs(child)
            if runs:
                # Wrap as a single-item list for now; _group_list_items coalesces runs.
                blocks.append(ir.ListItem(children=[ir.Paragraph(runs=runs)]))
            continue

        if tag in ("PREAMB", "SUPLINF", "AGY", "ACT", "SUM", "DATES",
                   "ADD", "FURINF", "LSTSUB", "SECTNO", "REGTEXT",
                   "PART", "SUBPART", "SECTION", "APPENDIX"):
            # Container; walk into it.
            _walk(child, blocks)
            continue

        if tag in ("AGENCY", "SUBAGY", "CFR", "DEPDOC", "SUBJECT"):
            # Preamble metadata. Emit as bold-label paragraphs so the text is
            # preserved. We deliberately don't promote these to headings —
            # they're field labels like "AGENCY: ...", not section headings.
            text = _text(child)
            if text:
                blocks.append(ir.Paragraph(runs=[
                    ir.Run(text=f"{tag}: ", bold=True),
                    ir.Run(text=text),
                ]))
            continue

        if tag in ("FTNT", "NOTE"):
            # Footnotes — emit as blockquotes so they stay distinguishable.
            inner: list[ir.Block] = []
            _walk(child, inner)
            if inner:
                blocks.append(ir.Blockquote(children=inner))
            continue

        if tag in ("GPH", "MATH", "CHEM"):
            # Graphics / math / chemistry — no image bytes in XML, skip.
            continue

        if tag == "TABLE":
            # FR tables are complex (CALS format). Skip for v1 — text content
            # inside will be walked by the fallback.
            _walk(child, blocks)
            continue

        # Unknown tag: walk into it, text will come from nested elements.
        _walk(child, blocks)


def _group_list_items(blocks: list[ir.Block]) -> list[ir.Block]:
    """Merge consecutive ListItem blocks into BulletLists."""
    out: list[ir.Block] = []
    current_run: list[ir.ListItem] = []

    def flush():
        if current_run:
            out.append(ir.BulletList(kind="unordered", items=list(current_run)))
            current_run.clear()

    for b in blocks:
        if isinstance(b, ir.ListItem):
            current_run.append(b)
        else:
            flush()
            out.append(b)
    flush()
    return out


# ---------- text/runs extraction ----------

def _text(el: ET.Element) -> str:
    """Return collapsed whitespace-stripped text content of an element."""
    s = "".join(el.itertext())
    return re.sub(r"\s+", " ", s).strip()


def _inline_runs(el: ET.Element) -> list[ir.Run]:
    """Return a flat list of Runs for element's inline content, preserving
    bold/italic/code where FR uses <E T="..."> emphasis markers.

    FR's emphasis tag is <E> with a T attribute: T="01" bold, T="02" italic,
    T="03" bold italic, T="04" typewriter (code), T="52" smaller text, etc.
    """
    runs: list[ir.Run] = []
    _walk_inline(el, runs, bold=False, italic=False, code=False)
    # Drop empty/whitespace-only leading/trailing runs.
    while runs and not runs[0].text.strip():
        runs.pop(0)
    while runs and not runs[-1].text.strip():
        runs.pop()
    return runs


def _walk_inline(el: ET.Element, runs: list[ir.Run],
                 *, bold: bool, italic: bool, code: bool) -> None:
    # Text before the first child.
    if el.text:
        text = re.sub(r"\s+", " ", el.text)
        runs.append(ir.Run(text=text, bold=bold, italic=italic, code=code))

    for child in el:
        child_bold = bold
        child_italic = italic
        child_code = code
        if child.tag.upper() == "E":
            t = child.get("T", "")
            if t == "01":
                child_bold = True
            elif t == "02":
                child_italic = True
            elif t == "03":
                child_bold = True
                child_italic = True
            elif t == "04":
                child_code = True
        _walk_inline(child, runs,
                     bold=child_bold, italic=child_italic, code=child_code)

        # Tail text (between child and next sibling) takes PARENT styling.
        if child.tail:
            tail = re.sub(r"\s+", " ", child.tail)
            runs.append(ir.Run(text=tail, bold=bold, italic=italic, code=code))
