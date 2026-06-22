"""PMC JATS ``<table-wrap>`` → clean accessible HTML5 ``<table>`` normalizer.

The shared keystone for the multi-source table datasets (Plans/06, plan b).
PMC stores tables as JATS XML in a pair's ``raw_source_xml``. This module
converts each ``<table-wrap>`` into the SAME accessible HTML5 ``<table>``
contract the ar5iv/OpenStax tables already satisfy, so that BOTH downstream
consumers can ingest PMC unchanged:

  * the Qwen table adapter builder (``data/build_table_qwen_multi.py``) — parses
    the HTML5 ``<table>`` via lxml and reuses the ar5iv grid/role/target code.
  * the BERT cell-role builder (``data/build_table_specialist_data.py``) — feeds
    the HTML5 string to its ``parse_html_tables`` HTMLParser.

JATS quirks this normalizer fixes (verified against on-disk PMC pairs):

  * **Graphic-only tables.** Many ``<table-wrap>`` hold only a ``<graphic>``
    (table-as-image), no markup. Those are unusable (would need OCR) and are
    skipped — ~10% of PMC table-wraps on disk.
  * **Headers as ``<td>`` inside ``<thead>``.** JATS routinely puts column
    headers in ``<td>`` (not ``<th>``) cells inside ``<thead>``. Left alone,
    every PMC header would read as a data cell. We promote any ``<thead>``
    cell to ``<th scope="col">``.
  * **Stub / row-header cells.** A leading ``<th>`` in a body row (or JATS
    ``@scope``) becomes ``<th scope="row">``.
  * **Presentational attributes.** ``frame`` / ``rules`` / ``align`` /
    ``valign`` / ``char`` / ``style`` / ``width`` carry no semantics and fail
    the WCAG-clean target contract; stripped. ``rowspan`` / ``colspan`` kept.
  * **Caption.** Hoisted from ``<table-wrap><caption>`` (its ``<title>`` and/or
    ``<p>`` text), not from a child ``<caption>`` — matching where JATS puts it.
  * **Inline content.** ``<sup>`` / ``<sub>`` / ``<italic>`` / ``<bold>`` are
    mapped to HTML5 (``sup``/``sub``/``em``/``strong``) and kept; numeric
    character references decode via lxml. Cross-refs / footnote ``<xref>`` are
    unwrapped to their text (the link target is document-position, dropped).

CPU-only, no network, no ML weights. Safe to run alongside a GPU job.
"""
from __future__ import annotations

from lxml import etree

# JATS inline tags → HTML5 inline tags. Anything not here is unwrapped to its
# text content (e.g. <xref>, <named-content>) so no foreign tags leak into the
# target.
_INLINE_MAP = {
    "sup": "sup",
    "sub": "sub",
    "italic": "em",
    "bold": "strong",
    "monospace": "code",
    "sc": "span",  # small-caps — no HTML5 element, keep text in a span
}

# Attributes stripped from every emitted element: presentational or
# JATS-specific, none WCAG-load-bearing. rowspan/colspan/scope are preserved
# explicitly where they matter (handled in code, not here).
_DROP_ATTRS = frozenset(
    {
        "frame", "rules", "align", "valign", "char", "charoff", "style",
        "width", "height", "border", "cellpadding", "cellspacing", "class",
        "id", "xml:id", "content-type", "specific-use",
    }
)


def _local(tag: object) -> str:
    """Local name of an lxml tag (namespace-agnostic; '' for comments/PIs)."""
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[-1].lower()


def _text_of(elem: etree._Element) -> str:
    """Flattened, whitespace-normalized text of a JATS subtree."""
    return " ".join("".join(elem.itertext()).split())


def _convert_inline(src: etree._Element, dst: etree._Element) -> None:
    """Copy ``src``'s mixed inline content into HTML5 element ``dst``.

    Recognized inline tags are remapped (``_INLINE_MAP``); unrecognized tags
    are unwrapped (their text + tail are spliced in) so the HTML5 output never
    contains a JATS-only element.
    """
    dst.text = (dst.text or "") + (src.text or "")

    def _append_text(target: etree._Element, last_child, text: str) -> None:
        if not text:
            return
        if last_child is None:
            target.text = (target.text or "") + text
        else:
            last_child.tail = (last_child.tail or "") + text

    for child in src:
        lname = _local(child.tag)
        html_tag = _INLINE_MAP.get(lname)
        if html_tag is not None:
            sub = etree.SubElement(dst, html_tag)
            _convert_inline(child, sub)
            sub.tail = child.tail or ""
        else:
            # Unwrap unknown inline (xref, named-content, …): splice text+tail.
            last = dst[-1] if len(dst) else None
            _append_text(dst, last, _text_of(child))
            last = dst[-1] if len(dst) else None
            _append_text(dst, last, child.tail or "")


def _emit_cell(tr: etree._Element, src_cell: etree._Element, *, force_header: bool,
               first_col: bool) -> None:
    """Append one normalized ``<th>``/``<td>`` to row ``tr`` from a JATS cell."""
    lname = _local(src_cell.tag)  # 'th' or 'td'
    src_scope = (src_cell.get("scope") or "").lower()
    is_header = force_header or lname == "th" or src_scope in ("col", "row",
                                                               "colgroup", "rowgroup")
    tag = "th" if is_header else "td"
    cell = etree.SubElement(tr, tag)

    if is_header:
        if src_scope in ("row", "rowgroup"):
            scope = "row"
        elif src_scope in ("col", "colgroup"):
            scope = "col"
        elif force_header:
            scope = "col"          # promoted thead cell labels its column
        else:
            scope = "row" if first_col else "col"
        cell.set("scope", scope)

    for span in ("rowspan", "colspan"):
        v = src_cell.get(span)
        if v and v.strip().isdigit() and int(v) > 1:
            cell.set(span, str(int(v)))

    _convert_inline(src_cell, cell)
    # Collapse whitespace-only text to empty so downstream "all-empty" filters
    # see a clean signal.
    if cell.text and not cell.text.strip() and len(cell) == 0:
        cell.text = ""


def _caption_text(table_wrap: etree._Element) -> str | None:
    """Hoist the table-wrap caption (``<caption>`` title + paragraphs)."""
    caps = table_wrap.xpath('./*[local-name()="caption"]')
    if not caps:
        # Some JATS put a <label> ("Table 1") but no caption — skip label-only.
        return None
    text = _text_of(caps[0])
    return text or None


def _convert_table(src_table: etree._Element, caption: str | None) -> str | None:
    """Convert one JATS ``<table>`` to an HTML5 ``<table>`` string.

    Returns None if the table has no usable rows.
    """
    out = etree.Element("table")
    if caption:
        etree.SubElement(out, "caption").text = caption

    # Collect rows grouped by section so we know which rows are headers.
    # JATS: <thead>/<tbody>/<tfoot> may be present, or bare <tr> under <table>.
    def _rows_in(parent: etree._Element):
        return parent.xpath('./*[local-name()="tr"]')

    thead_rows: list[etree._Element] = []
    body_rows: list[etree._Element] = []
    for sec in src_table:
        sname = _local(sec.tag)
        if sname == "thead":
            thead_rows.extend(_rows_in(sec))
        elif sname in ("tbody", "tfoot"):
            body_rows.extend(_rows_in(sec))
        elif sname == "tr":
            body_rows.append(sec)

    if not thead_rows and not body_rows:
        return None

    html_thead = etree.SubElement(out, "thead") if thead_rows else None
    html_tbody = etree.SubElement(out, "tbody")

    for tr in thead_rows:
        out_tr = etree.SubElement(html_thead, "tr")
        for cell in tr:
            if _local(cell.tag) in ("th", "td"):
                _emit_cell(out_tr, cell, force_header=True, first_col=False)

    for tr in body_rows:
        out_tr = etree.SubElement(html_tbody, "tr")
        col = 0
        for cell in tr:
            if _local(cell.tag) not in ("th", "td"):
                continue
            _emit_cell(out_tr, cell, force_header=False, first_col=(col == 0))
            col += 1

    # Drop an empty tbody if everything was header (rare).
    if len(html_tbody) == 0:
        out.remove(html_tbody)
    if html_thead is not None and len(html_thead) == 0:
        out.remove(html_thead)
    if len(out.xpath('.//*[local-name()="tr"]')) == 0:
        return None

    return etree.tostring(out, encoding="unicode", with_tail=False)


def jats_to_html5_tables(raw_source_xml: str) -> list[tuple[str, str | None]]:
    """Convert every usable ``<table-wrap>`` in a PMC JATS doc to HTML5.

    Returns a list of ``(html5_table_string, caption_text_or_None)``. Skips
    graphic-only table-wraps (table-as-image) and any table with no rows.
    Never raises on malformed input — returns what it can parse.
    """
    if not raw_source_xml or "<table-wrap" not in raw_source_xml:
        return []
    try:
        root = etree.fromstring(
            raw_source_xml.encode("utf-8"),
            etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True),
        )
    except Exception:
        return []
    if root is None:
        return []

    out: list[tuple[str, str | None]] = []
    for wrap in root.xpath('//*[local-name()="table-wrap"]'):
        tables = wrap.xpath('.//*[local-name()="table"]')
        if not tables:
            continue  # graphic-only — unusable
        caption = _caption_text(wrap)
        for tbl in tables:
            html = _convert_table(tbl, caption)
            if html:
                out.append((html, caption))
            # only the first <table> per wrap carries the caption; nested
            # tables (rare) share it harmlessly.
    return out
