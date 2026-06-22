"""govinfo.gov CFR XML → legacy IR (Document tree).

The Code of Federal Regulations is published by the U.S. Government
Publishing Office at https://www.govinfo.gov/bulkdata/CFR/<year>/title-<n>/
as one XML volume per (title, volume). Public domain (17 U.S.C. 105),
no per-document copyright filter required.

CFR XML uses the same vocabulary as Federal Register (HD/SOURCE, P,
FTNT, AUTH, ...) but adds structural wrappers — PART, SUBPART, SECTION
with SECTNO/SUBJECT — that we map to headings + metadata blocks.

The volume files are too large to render to a single PDF (some are
>10MB of text). The bulk fetcher splits each volume into per-PART
sub-documents BEFORE handing them to this parser; this module exposes:

    parse_cfr_part(part_element, ...) -> ir.Document

so the fetcher can iterate at PART granularity.

Mapping (designed to populate the rare doc_role classes):

  PART/HD SOURCE="HED"   "PART 1—DEFINITIONS"   -> H1 (title)
  AUTH                   "Authority: 44 USC 1506" -> Paragraph (metadata via extractor)
  SOURCE                 "Source: 37 FR 23603..."  -> Paragraph (metadata)
  EAR                    "Pt. 1"                   -> Paragraph (metadata)
  SECTNO                 "§ 1.1"                  -> H2 (with SUBJECT as title text)
  SUBJECT (in SECTION)   "Definitions."            -> appended to H2
  P                      regulation body text      -> Paragraph (body)
  FTNT                   footnote                  -> Blockquote (footer)
  CITA / CITEP / EDNOTE  citation / edition note   -> Paragraph (footer/metadata)
  HD SOURCE="HD1"|"HD2"|...                        -> H3/H4/...

Inline emphasis follows the FR convention (<E T="01"|"02"|"03"|"04">).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from . import ir


HD_LEVEL = {
    "HED": 2,
    "HD1": 2,
    "HD2": 3,
    "HD3": 4,
    "HD4": 5,
    "HD5": 6,
}


def parse_cfr_part(part_el: ET.Element,
                   *,
                   title_num: str,
                   volume: str,
                   year: str,
                   source_url: str | None = None,
                   front_matter_el: ET.Element | None = None) -> ir.Document:
    """Parse one CFR PART element into an IR Document.

    The PART becomes the document; its leading HD SOURCE="HED" text is
    used as the H1 title. Subsequent SECTION children are emitted as
    H2 headings + body paragraphs.

    If `front_matter_el` is provided (typically the volume's <FMTR>
    block), its OENOTICE / GPO / SUDOCS / BTITLE notice paragraphs are
    prepended to the document before the PART body. This is how we get
    the rendered PDF to carry the volume's legal-class boilerplate so
    the alignment step can pair it back to the extractor's `legal`
    labels.

    Raises ir.IRError if the PART has no recognizable body content.
    """
    title_text = _part_title(part_el) or f"CFR Title {title_num} Part"
    blocks: list[ir.Block] = [ir.Heading(level=1, runs=[ir.Run(title_text)])]

    # Front-matter line — the citation header (legal/metadata signal).
    blocks.append(ir.Paragraph(runs=[
        ir.Run(text=f"Title {title_num}, Code of Federal Regulations"),
    ]))
    blocks.append(ir.Paragraph(runs=[
        ir.Run(text=f"Revised as of January 1, {year}", italic=True),
    ]))

    if front_matter_el is not None:
        # Emit the volume's front-matter notice (OENOTICE / GPO / SUDOCS)
        # as paragraphs so they appear in the rendered PDF and can be
        # matched back to the extractor's `legal` labels via Jaccard.
        # Limit to text-bearing notices to keep the PDF size reasonable.
        for tag in ("OENOTICE", "GPO", "SUDOCS"):
            for el in front_matter_el.iter(tag):
                for p in el.iter("P"):
                    runs = _inline_runs(p)
                    if runs:
                        blocks.append(ir.Paragraph(runs=runs))

    _walk(part_el, blocks, skip_first_hed=True)

    blocks = _group_list_items(blocks)

    # Need at least the H1 + 1 body paragraph for a meaningful pair.
    if sum(1 for b in blocks if isinstance(b, ir.Paragraph)) < 2:
        raise ir.IRError(f"CFR PART {title_text} had no body paragraphs")

    doc = ir.Document(
        title=title_text,
        language="en",
        source="cfr",
        source_id=f"title{title_num}_vol{volume}_{_slug_part(title_text)}",
        source_url=source_url,
        blocks=blocks,
    )
    return ir.normalize_heading_levels(doc)


def _part_title(part_el: ET.Element) -> str | None:
    """First <HD SOURCE="HED"> direct child becomes the part title."""
    for child in part_el:
        if child.tag.upper() == "HD" and child.get("SOURCE", "").upper() == "HED":
            text = _text(child)
            if text:
                return text
        # Don't descend; the part title is at the top level.
    return None


def _walk(element: ET.Element,
          blocks: list[ir.Block],
          *,
          skip_first_hed: bool = False) -> None:
    """Depth-first walk that emits IR blocks."""
    seen_first_hed = not skip_first_hed
    for child in element:
        tag = child.tag.upper()

        if tag == "HD":
            if not seen_first_hed:
                seen_first_hed = True
                # The first HED is consumed as the part title (already emitted).
                continue
            source = child.get("SOURCE", "HED").upper()
            level = HD_LEVEL.get(source, 3)
            text = _text(child)
            if text:
                blocks.append(ir.Heading(level=level, runs=[ir.Run(text)]))
            continue

        if tag == "SECTION":
            _emit_section(child, blocks)
            continue

        if tag in ("AUTH", "SOURCE", "EFFDATE"):
            # Authority / Source citation lines — two-paragraph pattern:
            # a leading bold HD and a trailing P. Emit as a single
            # paragraph so the extractor can label both pieces metadata.
            label = ""
            tail_runs: list[ir.Run] = []
            for sub in child:
                stag = sub.tag.upper()
                if stag == "HD":
                    label = _text(sub)
                elif stag == "P":
                    tail_runs.extend(_inline_runs(sub))
            if label or tail_runs:
                runs: list[ir.Run] = []
                if label:
                    runs.append(ir.Run(text=f"{label} ", bold=True))
                runs.extend(tail_runs)
                if runs:
                    blocks.append(ir.Paragraph(runs=runs))
            continue

        if tag == "EAR":
            text = _text(child)
            if text:
                blocks.append(ir.Paragraph(
                    runs=[ir.Run(text=text, italic=True)]))
            continue

        if tag == "SECTNO":
            text = _text(child)
            if text:
                blocks.append(ir.Heading(level=2, runs=[ir.Run(text)]))
            continue

        if tag == "SUBJECT":
            text = _text(child)
            if text:
                blocks.append(ir.Paragraph(runs=[ir.Run(text=text, bold=True)]))
            continue

        if tag in ("CITA", "CITEP", "EDNOTE"):
            text = _text(child)
            if text:
                blocks.append(ir.Paragraph(runs=[ir.Run(text=text)]))
            continue

        if tag == "P":
            runs = _inline_runs(child)
            if runs:
                blocks.append(ir.Paragraph(runs=runs))
            continue

        if tag == "FP":
            # Flush paragraph (often a continuation paragraph).
            runs = _inline_runs(child)
            if runs:
                blocks.append(ir.Paragraph(runs=runs))
            continue

        if tag == "LI":
            runs = _inline_runs(child)
            if runs:
                blocks.append(ir.ListItem(
                    children=[ir.Paragraph(runs=runs)]))
            continue

        if tag in ("FTNT", "NOTE"):
            inner: list[ir.Block] = []
            _walk(child, inner)
            if inner:
                blocks.append(ir.Blockquote(children=inner))
            continue

        if tag in ("PRTPAGE", "GPH", "MATH", "CHEM", "GID",
                   "BOXHD", "GPOTABLE", "TABLE",
                   "FRDOC", "FILED", "SIG"):
            # Pagination markers / graphics / tables — skip for v1 (text
            # inside tables would distort the alignment scoring).
            continue

        if tag in ("SUBPART", "SUBJGRP", "EXTRACT", "EXAMPLE"):
            _walk(child, blocks)
            continue

        # Unknown tag — descend so any nested text isn't lost.
        _walk(child, blocks)


def _emit_section(section_el: ET.Element, blocks: list[ir.Block]) -> None:
    """A SECTION has SECTNO + SUBJECT + body P/FP/LI children."""
    sectno = ""
    subject = ""
    for sub in section_el:
        stag = sub.tag.upper()
        if stag == "SECTNO":
            sectno = _text(sub)
        elif stag == "SUBJECT":
            subject = _text(sub)
        if sectno and subject:
            break

    if sectno:
        # Combine "§ 1.1" + "Definitions." into one H2.
        heading_text = sectno
        if subject:
            heading_text = f"{sectno} {subject}"
        blocks.append(ir.Heading(level=2, runs=[ir.Run(heading_text)]))

    # Then walk body content, but skip SECTNO / SUBJECT (already emitted).
    for sub in section_el:
        stag = sub.tag.upper()
        if stag in ("SECTNO", "SUBJECT"):
            continue
        # Re-use the main walker on a single-element synthesizing approach.
        _walk_one(sub, blocks)


def _walk_one(node: ET.Element, blocks: list[ir.Block]) -> None:
    """Walk a single element through the same dispatch as _walk."""
    # Wrap in a synthetic parent so _walk sees it as a child.
    parent = ET.Element("WRAP")
    parent.append(node)
    _walk(parent, blocks)


def _group_list_items(blocks: list[ir.Block]) -> list[ir.Block]:
    out: list[ir.Block] = []
    current: list[ir.ListItem] = []

    def flush():
        if current:
            out.append(ir.BulletList(kind="unordered", items=list(current)))
            current.clear()

    for b in blocks:
        if isinstance(b, ir.ListItem):
            current.append(b)
        else:
            flush()
            out.append(b)
    flush()
    return out


def _slug_part(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:maxlen] or "part"


# ---------- text/runs extraction (mirrors parse_federal_register) ----------

def _text(el: ET.Element) -> str:
    s = "".join(el.itertext())
    return re.sub(r"\s+", " ", s).strip()


def _inline_runs(el: ET.Element) -> list[ir.Run]:
    runs: list[ir.Run] = []
    _walk_inline(el, runs, bold=False, italic=False, code=False)
    while runs and not runs[0].text.strip():
        runs.pop(0)
    while runs and not runs[-1].text.strip():
        runs.pop()
    return runs


def _walk_inline(el: ET.Element, runs: list[ir.Run],
                 *, bold: bool, italic: bool, code: bool) -> None:
    if el.text:
        text = re.sub(r"\s+", " ", el.text)
        runs.append(ir.Run(text=text, bold=bold, italic=italic, code=code))

    for child in el:
        cb, ci, cc = bold, italic, code
        if child.tag.upper() == "E":
            t = child.get("T", "")
            if t == "01":
                cb = True
            elif t == "02":
                ci = True
            elif t == "03":
                cb = True
                ci = True
            elif t == "04":
                cc = True
        _walk_inline(child, runs, bold=cb, italic=ci, code=cc)
        if child.tail:
            tail = re.sub(r"\s+", " ", child.tail)
            runs.append(ir.Run(text=tail, bold=bold, italic=italic, code=code))
