"""[LEGACY] Document intermediate representation (tree shape).

STATUS: used by legacy emit_html.py and by the data-generation parsers
(parse_wikipedia, parse_ar5iv, gen_synthetic_forms) that build a Document
tree and hand it to emit_html for ground-truth HTML output. The runtime
pipeline (dart_semantic/pipeline.py) uses the flat block representation
in dart_semantic/types.py instead.

If/when the data-gen parsers are rewritten to emit HTML directly, this
module can be deleted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

ListKind = Literal["unordered", "ordered", "definition"]
FormFieldKind = Literal[
    "text", "email", "tel", "url", "number", "date",
    "checkbox", "radio", "select", "textarea",
]


class IRError(Exception):
    """Raised when source content can't be faithfully represented in IR."""


# ---------- inline content ----------

@dataclass
class Run:
    """A span of inline text with optional styling.

    Styling is limited to what's semantically meaningful — bold, italic, code,
    link. Visual-only styling (color, size) is deliberately absent; the emitter
    has no way to represent it accessibly in generic HTML.
    """
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    href: str | None = None


# ---------- block content ----------

@dataclass
class Heading:
    level: int  # Logical level 1..6. Emitter assigns the actual h<n> tag.
    runs: list[Run]


@dataclass
class Paragraph:
    runs: list[Run]


@dataclass
class ListItem:
    # List items can contain arbitrary block content — nested lists, paragraphs, etc.
    children: list["Block"]


@dataclass
class DefinitionItem:
    term: list[Run]
    definition: list["Block"]


@dataclass
class BulletList:
    kind: ListKind  # "unordered" or "ordered" — "definition" uses DefinitionList below
    items: list[ListItem]


@dataclass
class DefinitionList:
    items: list[DefinitionItem]


@dataclass
class TableCell:
    runs: list[Run]
    colspan: int = 1
    rowspan: int = 1
    is_header: bool = False


@dataclass
class Table:
    """A table with an explicit distinction between header and body rows.

    The emitter fails if header_rows is empty — WCAG 2.2 AA cannot be met for a
    table without programmatic header association.
    """
    caption: list[Run] | None
    header_rows: list[list[TableCell]]
    body_rows: list[list[TableCell]]


@dataclass
class Figure:
    """Image + caption. alt is non-optional here — the emitter enforces it."""
    src: str
    alt: str
    caption: list[Run] | None = None


@dataclass
class Blockquote:
    children: list["Block"]
    cite: str | None = None


@dataclass
class CodeBlock:
    text: str
    language: str | None = None


@dataclass
class FormField:
    kind: FormFieldKind
    name: str
    label: list[Run]  # Required — the emitter rejects unlabeled fields.
    required: bool = False
    options: list[str] = field(default_factory=list)  # For select/radio/checkbox groups
    help_text: list[Run] | None = None


@dataclass
class Fieldset:
    legend: list[Run]
    fields: list[FormField]


@dataclass
class Form:
    action: str | None
    method: Literal["get", "post"]
    fieldsets: list[Fieldset]


Block = Union[
    Heading, Paragraph, BulletList, DefinitionList, Table, Figure,
    Blockquote, CodeBlock, Form,
]


# ---------- root ----------

@dataclass
class Document:
    """Top-level document node.

    language is a BCP-47 tag ("en", "en-US", "es", ...). Required — the emitter
    fails if omitted.
    """
    title: str
    language: str
    blocks: list[Block]

    # Optional metadata carried through for logging / filtering but not emitted.
    source: str | None = None           # e.g., "arxiv", "wikipedia"
    source_id: str | None = None        # e.g., arxiv ID or wiki page title
    source_url: str | None = None


def split_at_level(doc: "Document", split_level: int = 2) -> list["Document"]:
    """Return one sub-document per top-level section.

    Walks the block sequence; each heading at `split_level` starts a new
    section. Content before the first such heading (the "lead") becomes
    its own sub-document. Every sub-document inherits the parent's title,
    language, and source metadata, and gets the original <h1> title
    prepended before the section's own content so each stands alone.

    Returns [] if `doc` has no blocks or no <h1>. Empty sections (just the
    heading, no content) are skipped.
    """
    if not doc.blocks:
        return []

    root_title = doc.title
    h1_block: Heading | None = None
    body: list[Block] = []
    for b in doc.blocks:
        if isinstance(b, Heading) and b.level == 1 and h1_block is None:
            h1_block = b
            continue
        body.append(b)
    if h1_block is None:
        return []

    sections: list[tuple[str, list[Block]]] = []
    current_title = "Lead"
    current_blocks: list[Block] = []

    def flush() -> None:
        if current_blocks and any(not isinstance(b, Heading) for b in current_blocks):
            sections.append((current_title, list(current_blocks)))

    for b in body:
        if isinstance(b, Heading) and b.level == split_level:
            flush()
            current_title = "".join(r.text for r in b.runs).strip() or "Section"
            current_blocks = [b]
        else:
            current_blocks.append(b)
    flush()

    out_docs: list[Document] = []
    for sec_title, sec_blocks in sections:
        # Each section doc starts with the article's h1 so the model sees
        # the full document title at training time — consistent with
        # Wikipedia section pairs.
        new_blocks: list[Block] = [
            Heading(level=1, runs=list(h1_block.runs)),
            *sec_blocks,
        ]
        doc_title = (f"{root_title} — {sec_title}"
                     if sec_title != "Lead" else root_title)
        out_docs.append(Document(
            title=doc_title,
            language=doc.language,
            blocks=new_blocks,
            source=doc.source,
            source_id=doc.source_id,
            source_url=doc.source_url,
        ))
    return out_docs


def normalize_heading_levels(doc: "Document") -> "Document":
    """Return a copy of doc with heading levels renumbered to be contiguous.

    Source documents (especially LaTeX-converted HTML) frequently skip levels:
    a section is <h2> but the first subsection is <h4> because the LaTeX had
    no \\subsection but did have a \\subsubsection. WCAG SC 1.3.1 / 2.4.6
    require contiguous heading levels, so the emitter rejects these docs.

    This helper walks the heading sequence and demotes any level that skips.
    Example: [1, 2, 4, 4, 3] -> [1, 2, 3, 3, 3] (the 4s become 3s because the
    surrounding depth was 2+1; the trailing 3 stays because it's at-or-before
    the previous depth).
    """
    new_blocks: list[Block] = []
    last_level = 0
    for b in doc.blocks:
        if isinstance(b, Heading):
            if last_level == 0:
                # First heading — clamp to 1.
                new_level = 1
            else:
                new_level = min(b.level, last_level + 1)
            new_blocks.append(Heading(level=new_level, runs=list(b.runs)))
            last_level = new_level
        else:
            new_blocks.append(b)
    return Document(
        title=doc.title, language=doc.language, blocks=new_blocks,
        source=doc.source, source_id=doc.source_id, source_url=doc.source_url,
    )
