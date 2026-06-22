"""Qwen reasoner input schema — locked short-form serialization.

Stage 3b's Qwen sees one chunk of a document at a time. Each block becomes
one line in Qwen's input. Token budget is tight: 4096 ctx, ~120 blocks per
chunk, so ~34 tokens/block including Qwen's own overhead.

The format is deliberately stable. Any change here invalidates existing
Qwen training data and requires retraining. Keep the schema frozen across
iterations unless the label set itself changes.

## Block line

    [FLAGS] TEXT

FLAGS are space-separated tokens in canonical order (omit defaults):

    s=<xl|lg|sm>   size bucket           md is default, omit
    b              bold                  absent = not bold
    c              centered              absent = not centered
    t              top-of-page (<15%)    absent = not top
    g=lg          large gap above        (no other gap values in use)
    cap=<a|t>      caps all / title      absent = normal case
    tbl            in_table              (per pdfplumber bbox)
    hdr            in_header_row         only emitted when tbl
    wid            in_widget             (per pikepdf widget rect)
    w=<t|b|s|sg>   widget kind           text/button/select/signature; only with wid
    p=<prov>       provenance            only when non-default
    h=<code>       DistilBERT hint       always emitted — see LABEL_TO_CODE
    hc=.XX         hint confidence       omit when >= .9 (high-conf default)

TEXT is the block's raw text, single-line. Newlines inside the block
(rare) are escaped as backslash-n.

## Chunk header (first line of each chunk)

    [doc=<type> dconf=.XX chunk=N/K pages=P1-P2]

`doc=` is the document-type hint (academic_paper, form, textbook, ...) —
empty string when unknown. `dconf=` may be omitted when unknown.

## Page separator (between pages inside a chunk)

    --- page <N> ---

## Qwen output

One JSON array per chunk, one object per block in chunk order:

    [{"id": 0, "role": "heading", "depth": 2}, ...]

- `role` uses the long-form label name (same vocabulary as LABEL_TO_CODE keys).
- `depth` is emitted only for roles where depth is meaningful (heading,
  list_item). For everything else it is omitted.
- `id` is the block's zero-based index within the chunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Label vocabulary → 2-3 char code for the `h=` flag.
LABEL_TO_CODE = {
    "title":             "ti",
    "heading":           "hd",
    "paragraph":         "pa",
    "list":              "ls",
    "list_item":         "li",
    "table":             "tb",
    "table_row":         "tr",
    "table_header_cell": "thc",
    "table_data_cell":   "tdc",
    "table_caption":     "tbc",
    "figure":            "fg",
    "figure_caption":    "fgc",
    "blockquote":        "bq",
    "code_block":        "cb",
    "form_field":        "ff",
    "form_label":        "fl",
    "reference":         "rf",
    "footnote":          "fn",
    "definition":        "df",
    "metadata":          "mt",
    "page_header":       "ph",
    "page_footer":       "pf",
}

CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}

# Widget-kind short codes.
WIDGET_CODE = {
    "text":      "t",
    "button":    "b",
    "select":    "s",
    "signature": "sg",
}

# Provenance values that count as the default (boring born-digital text).
# Anything else gets emitted as `p=<prov>`.
DEFAULT_PROVENANCE = {
    None, "",
    "pypdfium2",
    "pypdfium2+pdfplumber",
    "pdfplumber+pypdfium2",
}

# Provenance abbreviations — applied to the rare non-default strings.
PROV_ABBREV = {
    "tesseract": "tess",
    "pikepdf":   "pk",
    "pdfplumber": "pl",
    "pypdfium2": "pp",
}

# Confidence threshold above which we omit the `hc=` flag.
# Chosen so "confident enough to be default" saves tokens on the majority
# of blocks; the threshold is above DistilBERT's typical correct-case
# confidence but below its typical uncertain-case confidence.
HINT_CONF_DEFAULT_THRESHOLD = 0.9


@dataclass
class SerializedBlock:
    """What one block looks like in Qwen's input. Kept as a record so
    downstream code can emit it either as one line (training/inference)
    or as a debug dump (evaluation)."""
    id: int                                # within-chunk 0-based index
    text: str
    flags: list[str]                       # canonical-order short-form tokens

    def to_line(self) -> str:
        prefix = f"[{' '.join(self.flags)}] " if self.flags else ""
        return f"{prefix}{_escape_text(self.text)}"


def _escape_text(t: str) -> str:
    # Keep block lines single-line; rare internal newlines become \n.
    return t.replace("\n", "\\n")


def _fmt_conf(c: float) -> str:
    """`.88` for 0.88. Two decimals, no leading zero."""
    # Clip to [0, 1]
    c = max(0.0, min(1.0, float(c)))
    # Drop leading 0: 0.88 -> ".88"
    s = f"{c:.2f}"
    return s.lstrip("0") if s.startswith("0") else s


def _abbrev_provenance(prov: str) -> str:
    """Collapse a '+'-joined provenance string through PROV_ABBREV."""
    parts = prov.split("+")
    return "+".join(PROV_ABBREV.get(p, p) for p in parts)


def serialize_block(
    *, block_id: int,
    text: str,
    size_bucket: str = "md",
    is_bold: bool = False,
    is_centered: bool = False,
    is_top_of_page: bool = False,
    gap_above: str | None = None,
    caps: str | None = None,
    in_table: bool = False,
    in_header_row: bool = False,
    in_widget: bool = False,
    widget_kind: str | None = None,
    provenance: str | None = None,
    hint_label: str = "paragraph",
    hint_confidence: float = 1.0,
) -> SerializedBlock:
    """Emit a block's short-form line per the locked schema.

    Canonical flag order (matches the spec above): s, b, c, t, g, cap,
    tbl/hdr, wid/w, p, h, hc.
    """
    flags: list[str] = []

    # Typography / layout
    if size_bucket in ("xl", "lg", "sm"):
        flags.append(f"s={size_bucket}")
    if is_bold:
        flags.append("b")
    if is_centered:
        flags.append("c")
    if is_top_of_page:
        flags.append("t")
    if gap_above == "lg":
        flags.append("g=lg")
    if caps == "all":
        flags.append("cap=a")
    elif caps == "title":
        flags.append("cap=t")

    # Table + widget structural signals
    if in_table:
        flags.append("tbl")
        if in_header_row:
            flags.append("hdr")
    if in_widget:
        flags.append("wid")
        if widget_kind and widget_kind in WIDGET_CODE:
            flags.append(f"w={WIDGET_CODE[widget_kind]}")

    # Extractor provenance (omit when default)
    if provenance not in DEFAULT_PROVENANCE:
        flags.append(f"p={_abbrev_provenance(provenance)}")

    # DistilBERT hint — always emitted. Confidence omitted when high.
    hint_code = LABEL_TO_CODE.get(hint_label)
    if hint_code is None:
        # Unknown label — fall back to paragraph so Qwen never sees a blank.
        hint_code = LABEL_TO_CODE["paragraph"]
    flags.append(f"h={hint_code}")
    if hint_confidence < HINT_CONF_DEFAULT_THRESHOLD:
        flags.append(f"hc={_fmt_conf(hint_confidence)}")

    return SerializedBlock(id=block_id, text=text, flags=flags)


def chunk_header(*, chunk_index: int, chunk_count: int,
                 page_start: int, page_end: int,
                 doc_type: str | None = None,
                 doc_conf: float | None = None) -> str:
    """One-line chunk header. First line of every chunk."""
    parts: list[str] = []
    if doc_type:
        parts.append(f"doc={doc_type}")
        if doc_conf is not None:
            parts.append(f"dconf={_fmt_conf(doc_conf)}")
    parts.append(f"chunk={chunk_index + 1}/{chunk_count}")
    parts.append(f"pages={page_start}-{page_end}")
    return f"[{' '.join(parts)}]"


def page_separator(page_number: int) -> str:
    return f"--- page {page_number} ---"


# ---------- output parsing ----------

def parse_qwen_output(raw: str) -> list[dict[str, Any]]:
    """Parse Qwen's JSON-array output for a single chunk.

    Returns a list of {id, role, depth?} dicts. Raises ValueError on
    malformed output — callers decide whether to fall back to the
    DistilBERT hint in that case.
    """
    import json
    raw = raw.strip()
    # Strip common wrappers: ```json ... ``` fences.
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("Qwen output is not a JSON array")
    out: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            raise ValueError(f"Qwen output entry is not an object: {entry!r}")
        if "id" not in entry or "role" not in entry:
            raise ValueError(f"Qwen output entry missing id/role: {entry!r}")
        out.append(entry)
    return out
