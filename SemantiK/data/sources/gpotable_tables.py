"""GPO ``<GPOTABLE>`` XML → accessible HTML5 ``<table>`` normalizer.

CFR and the Federal Register store tables as GPO Locator Code XML
(``<GPOTABLE>``), not HTML. This module converts each to the SAME accessible
HTML5 ``<table>`` contract the ar5iv / OpenStax / PMC tables satisfy, so the
multi-source builders ingest it unchanged. It adds the **regulatory**
``table_type`` (nested headers, US-gov public-domain) — the diversity axis the
table set otherwise has zero of (Plans/06 §1b, §6).

GPOTABLE shape (verified against on-disk CFR pairs):

  * ``<GPOTABLE COLS="n">`` wraps the table; ``COLS`` is the body column count.
  * ``<BOXHD>`` holds column headers as ``<CHED H="level">`` — *hierarchical by
    the ``H`` attribute*, not by XML nesting: ``H="1"`` is the top grouping
    level, ``H="2"`` the sub-columns under it, etc. We emit one header ``<tr>``
    of ``<th scope="col">`` per distinct ``H`` level, in level order. Spans
    between levels are not reconstructed (the grouping cells simply land in an
    earlier header row); downstream ``_expand_grid`` pads rows rectangular, so
    the result is well-formed and the header cells are correctly ``<th>``.
  * ``<ROW>`` holds body cells as ``<ENT>``. A leading ``<ENT I="...">`` marks a
    stub (row-label) column; we keep all body cells as ``<td>`` (GPO stub
    semantics are inconsistent, and a wrong ``scope="row"`` is worse than none —
    matches how PMC bodies are handled).
  * ``<TTITLE>`` (when present) becomes the ``<caption>``.
  * Inline GPO markup (``<E>`` emphasis etc.) is flattened to text.

CPU-only, no network, no ML weights.
"""
from __future__ import annotations

import re

from lxml import etree

# Common GPO/CFR XML entity references that aren't predefined in XML. Without
# a DTD these would break the parser; substitute to Unicode before parsing.
_GPO_ENTITIES = {
    "&sect;": "§", "&para;": "¶", "&prime;": "′", "&Prime;": "″",
    "&times;": "×", "&divide;": "÷", "&plusmn;": "±", "&deg;": "°",
    "&micro;": "µ", "&middot;": "·", "&bull;": "•", "&dagger;": "†",
    "&Dagger;": "‡", "&minus;": "−", "&ndash;": "–", "&mdash;": "—",
    "&hellip;": "…", "&rsquo;": "’", "&lsquo;": "‘", "&ldquo;": "“",
    "&rdquo;": "”", "&trade;": "™", "&reg;": "®", "&copy;": "©",
    "&frac12;": "½", "&frac14;": "¼", "&frac34;": "¾", "&nbsp;": " ",
}
_AMP_FIXUP = re.compile(r"&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)")


def _local(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[-1].lower()


def _text_of(elem: etree._Element) -> str:
    return " ".join("".join(elem.itertext()).split())


def _sanitize(xml: str) -> str:
    """Replace known GPO entity refs, then escape any remaining bare ``&`` so
    the recover parser doesn't choke on undefined entities."""
    for ref, ch in _GPO_ENTITIES.items():
        if ref in xml:
            xml = xml.replace(ref, ch)
    return _AMP_FIXUP.sub("&amp;", xml)


def _convert_one(gpotable: etree._Element) -> str | None:
    """Convert a single ``<GPOTABLE>`` element to an HTML5 ``<table>`` string."""
    out = etree.Element("table")

    # Caption from <TTITLE> if present (child or preceding sibling).
    ttitle = gpotable.xpath(
        './/*[local-name()="TTITLE" or local-name()="ttitle"]'
    )
    if ttitle:
        cap = _text_of(ttitle[0])
        if cap:
            etree.SubElement(out, "caption").text = cap

    # Header: <BOXHD> CHEDs grouped by H level → one <tr> of <th> per level.
    boxhd = gpotable.xpath('./*[local-name()="BOXHD" or local-name()="boxhd"]')
    thead = None
    if boxhd:
        cheds = boxhd[0].xpath('.//*[local-name()="CHED" or local-name()="ched"]')
        by_level: dict[str, list[etree._Element]] = {}
        order: list[str] = []
        for ched in cheds:
            lvl = ched.get("H") or ched.get("h") or "1"
            if lvl not in by_level:
                by_level[lvl] = []
                order.append(lvl)
            by_level[lvl].append(ched)
        rows_with_cells = [lvl for lvl in order if by_level[lvl]]
        if rows_with_cells:
            thead = etree.SubElement(out, "thead")
            for lvl in sorted(rows_with_cells):
                tr = etree.SubElement(thead, "tr")
                for ched in by_level[lvl]:
                    th = etree.SubElement(tr, "th")
                    th.set("scope", "col")
                    th.text = _text_of(ched)

    # Body: <ROW> → <tr>, <ENT> → <td>.
    tbody = etree.SubElement(out, "tbody")
    for row in gpotable.xpath('./*[local-name()="ROW" or local-name()="row"]'):
        tr = etree.SubElement(tbody, "tr")
        for ent in row.xpath('./*[local-name()="ENT" or local-name()="ent"]'):
            td = etree.SubElement(tr, "td")
            td.text = _text_of(ent)

    if len(tbody) == 0:
        out.remove(tbody)
    if thead is not None and len(thead) == 0:
        out.remove(thead)
    if len(out.xpath('.//*[local-name()="tr"]')) == 0:
        return None
    return etree.tostring(out, encoding="unicode", with_tail=False)


def gpotable_to_html5_tables(raw_source_xml: str) -> list[tuple[str, str | None]]:
    """Convert every ``<GPOTABLE>`` in a CFR/FedReg doc to HTML5.

    Returns ``(html5_table_string, caption_or_None)`` per table. Never raises.
    """
    if not raw_source_xml or "gpotable" not in raw_source_xml.lower():
        return []
    try:
        root = etree.fromstring(
            _sanitize(raw_source_xml).encode("utf-8"),
            etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True),
        )
    except Exception:
        return []
    if root is None:
        return []

    out: list[tuple[str, str | None]] = []
    for gt in root.xpath('//*[local-name()="GPOTABLE" or local-name()="gpotable"]'):
        html = _convert_one(gt)
        if html:
            caps = gt.xpath('.//*[local-name()="TTITLE" or local-name()="ttitle"]')
            cap = _text_of(caps[0]) if caps else None
            out.append((html, cap or None))
    return out
