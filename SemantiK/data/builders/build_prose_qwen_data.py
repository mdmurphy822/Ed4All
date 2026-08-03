r"""Build the Qwen-prose LoRA adapter training dataset (Phase 1.3).

One row per prose-shaped HTML region in an ar5iv pair's
``raw_source_html``. The source side is the EXACT JSON envelope that
``semantik_structure.qwen_specialists.prompts.build_prose_request`` emits at
inference time, given a ``Region`` derived from the same element. The
target side is the literal HTML5 fragment (lxml.etree.tostring of the
element subtree) so the adapter learns to reproduce the structural
wrapper byte-for-byte.

What "prose" covers
-------------------

Per ``semantik_structure.qwen_specialists.routing.REGION_KIND_TO_ADAPTER``,
the prose adapter handles every Stage-5 ``Region.kind`` except
``table`` and ``math``. Concretely:

    paragraph, heading, list, definition_list, code_block,
    blockquote, figure, form, metadata_drop

This builder walks the ar5iv body looking for HTML elements that map
to those kinds:

    <p>                                  -> paragraph
    <h1>..<h6>                           -> heading        (level_hint = 1..6)
    <ul>, <ol>                           -> list           (ordered = (tag == "ol"))
    <dl>                                 -> definition_list
    <blockquote>                         -> blockquote
    <pre>                                -> code_block
    <figure>                             -> figure
    <address>                            -> metadata_drop  (close ar5iv mapping)

ar5iv conventions we adapt to
-----------------------------

* ar5iv wraps every paragraph in ``<p class="ltx_p">`` inside a
  ``<div class="ltx_para">``. The ``<p>`` is what we extract; the
  wrapping div is the FB-level container Stage 5 sees.
* ar5iv headings carry ``ltx_title_document|section|subsection|...``
  classes; we map those into ``doc_role`` ("title" for the document
  title, otherwise None — heading regions don't carry doc_role per
  ``prompts.py``'s prose envelope, but we record it in
  ``extraction_metadata`` for future filter / analysis).
* ar5iv abstracts are inside a ``<div class="ltx_abstract">``; the
  contained ``<p>`` rows get ``doc_role="abstract"``.
* Authors / affiliations live inside spans like
  ``<span class="ltx_creator ltx_role_author">``; we don't currently
  surface those as regions (Stage 5 emits them as paragraphs with
  ``doc_role="author"`` from the trained Semantic head, not from class
  inspection — but we mirror the heuristic for ar5iv to keep the
  doc_role distribution somewhat realistic).

Source-format binding
---------------------

The live ``build_prose_request`` reads:

    region.kind                          -> "kind"
    region.payload.level_hint            -> "level_hint"   (heading)
    region.payload.doc_role              -> "doc_role"     (paragraph)
    region.payload.ordered               -> "ordered"      (list)
    region.payload.items                 -> "items"        (list)
    region.aria_hints                    -> "aria_hints"
    region.payload.text  OR
        join(items[*].text)              -> "text"

Every dataset row stores its ``request_payload`` as
``build_prose_request`` produced — re-running the live builder against
the same input must reproduce the row's prompt byte-for-byte modulo
whitespace. Test: ``tests/test_qwen_prose_dataset_contract.py``.

Target-HTML stripping policy (Gate 4)
-------------------------------------

The target_html is ``etree.tostring(elem, encoding="unicode")`` over
the element subtree, with these *conservative* attribute strips:

* ``id`` is dropped UNLESS it looks like a stable anchor (we keep it
  if it doesn't start with ``S\d`` / ``Sx`` — those are LaTeXML
  section auto-IDs that change every build; everything else (e.g.
  named-anchor IDs the author wrote) is preserved).
* ``class`` attribute values are filtered: ``ltx_*`` tokens are
  dropped, but any non-ltx tokens are kept. If the resulting class
  list is empty, the attribute is removed entirely. Rationale: the
  trained adapter will never emit ``ltx_*`` classes (we'd never
  validate them), so if the target has them the cross-entropy
  encourages the adapter to memorize ar5iv-specific tokens it
  shouldn't.
* All other attributes (href, src, scope, role, aria-*, alt,
  rowspan, colspan, etc.) are preserved verbatim.
* Children are preserved verbatim (recursive — we do the strip on
  every descendant). This is intentional: list items, blockquote
  inner paragraphs, etc., need the same cleanup.

CPU-ONLY. Loads no ML weights, no tokenizers — token counts are
whitespace approximated. Safe to run while training is on the GPU.

Usage::

    python data/builders/build_prose_qwen_data.py                    # full build
    python data/builders/build_prose_qwen_data.py --max-pairs 30 \\
        --out-dir /tmp/qwen_prose_smoke                     # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from lxml import etree, html as lxml_html

# Live prompt builder — the contract surface this dataset is bound to.
# Any drift in prompts.py surfaces as failure in
# tests/test_qwen_prose_dataset_contract.py.
from semantik_structure.qwen_specialists.prompts import build_prose_request
# Single source of truth for the chat-template wrap; the length filter
# below must measure exactly what the trainer/inference path tokenizes.
from semantik_structure.qwen_specialists.chat_format import wrap_for_qwen

from data.common._splits import stable_split_for_id


# Default tokenizer for the wrapped-length filter. Hardcoded to the
# trainer's base model — a build-time/train-time tokenizer mismatch is
# exactly what let inline-MathML targets blow past max_len undetected.
DEFAULT_FILTER_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


def _load_filter_tokenizer(name: str) -> Any:
    """Lazy-load the HuggingFace tokenizer for the length filter.

    Imported inside the function so ``import build_prose_qwen_data``
    stays free of transformers (the CPU-only contract test pins this).
    Raises loudly if transformers is missing — we never silently fall
    back to whitespace-token counts, since that blind spot is what
    shipped the broken v1 prose dataset.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "transformers is required for tokenizer-aware length filtering "
            "(--max-token-len). Install transformers or run from the venv "
            f"that hosts the trainer. Original error: {exc}"
        ) from exc
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_PAIRS_DIR = Path("data/pairs/arxiv")
DEFAULT_OUT_DIR = Path("data/qwen_prose_dataset")

# Length filter (whitespace tokens). Matches semantic_preservation's
# range so cross-builder length distributions are comparable.
MIN_REGION_TOKENS = 20
MAX_REGION_TOKENS = 450

# Hard guard on the literal target_html — anything shorter than this is
# almost certainly a degenerate strip (empty p, lone marker, etc).
MIN_TARGET_CHARS = 30

WS_RE = re.compile(r"\s+")

# Histogram bucket edges (right-exclusive) — exposes the long tail
# without 100s of empty bins.
TOKEN_HIST_EDGES = [0, 20, 40, 80, 120, 160, 200, 250, 300, 350, 400, 450]

# ar5iv heading-class -> doc_role hint. Used for the
# ``extraction_metadata.doc_role_hint`` field; live prose envelope only
# carries doc_role for paragraph regions, but we record the heading-
# level role too so future filters can stratify by it.
LTX_HEADING_DOC_ROLE: dict[str, str] = {
    "ltx_title_document": "title",
    "ltx_title_section": "body",
    "ltx_title_subsection": "body",
    "ltx_title_subsubsection": "body",
    "ltx_title_paragraph": "body",
    "ltx_title_subparagraph": "body",
    "ltx_title_abstract": "body",
    "ltx_title_appendix": "body",
    "ltx_title_chapter": "body",
    "ltx_title_part": "body",
    "ltx_title_acknowledgement": "body",
    "ltx_title_bibliography": "body",
}

# heading tag -> level (h1=1, h6=6). The ar5iv heading classes also
# encode level (ltx_title_document is h1-ish, etc.) but using the actual
# tag is more robust.
HEADING_TAG_LEVEL: dict[str, int] = {
    "h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6,
}

# Tag-level kind mapping. Order in this dict is the dispatch order in
# the walker (we never re-emit an element under two kinds).
TAG_TO_KIND: dict[str, str] = {
    "p": "paragraph",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "ul": "list",
    "ol": "list",
    "dl": "definition_list",
    "blockquote": "blockquote",
    "pre": "code_block",
    "figure": "figure",
    "address": "metadata_drop",
}

# ar5iv class tokens we strip from the target HTML's class= values.
# Anything that *starts with* one of these prefixes is dropped from
# the class attribute.
LTX_CLASS_DROP_PREFIXES = ("ltx_",)

# ar5iv id= values we drop. LaTeXML-generated section IDs look like
# "S1.SS2.p3" / "S1.E1" / "Sx1" / "id1" — they're not stable across
# builds and the adapter has no way to learn them. We drop:
#   - anything that starts with "S\d" or "Sx"
#   - anything that starts with "p\d", "id\d", or "E\d"
# and KEEP anything else (the author may have written explicit anchors).
LTX_ID_DROP_RE = re.compile(r"^(?:S\d|Sx|p\d|id\d|E\d|cite\.|bib\.|fn\d|footnote)")


# --------------------------------------------------------------------------- #
# Skip / counter bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class SkipCounters:
    too_short: int = 0
    too_long: int = 0
    target_too_short: int = 0
    empty_text: int = 0
    serialize_error: int = 0
    parse_error: int = 0
    nested_skip: int = 0          # element is descendant of an already-emitted region
    inside_table: int = 0         # element lives inside a <table>; table adapter handles
    inside_math: int = 0          # element lives inside a <math>; math adapter handles
    figure_with_table: int = 0    # ar5iv "ltx_table" figures (table caption wrapper)
    over_max_tokens: int = 0      # wrapped (prompt+target) token length > max_token_len


@dataclass
class BuildStats:
    n_pairs_seen: int = 0
    n_pairs_with_html: int = 0
    n_pairs_with_arxiv_id: int = 0
    n_unique_arxiv_ids_seen: int = 0
    n_elements_inspected: int = 0
    n_rows_emitted: int = 0
    rows_per_split: Counter = field(default_factory=Counter)
    rows_per_kind: Counter = field(default_factory=Counter)
    rows_per_doc_role: Counter = field(default_factory=Counter)
    rows_per_arxiv_id: Counter = field(default_factory=Counter)
    src_token_hist: Counter = field(default_factory=Counter)
    tgt_token_hist: Counter = field(default_factory=Counter)
    skip: SkipCounters = field(default_factory=SkipCounters)


def _bucket_for(n: int, edges: list[int]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= n < edges[i + 1]:
            return f"[{edges[i]},{edges[i+1]})"
    return f"[{edges[-1]},inf)"


def _ws_token_count(s: str) -> int:
    return len([t for t in WS_RE.split(s.strip()) if t])


# --------------------------------------------------------------------------- #
# Helpers — class / id / text
# --------------------------------------------------------------------------- #


def _classes(elem: etree._Element) -> list[str]:
    return (elem.get("class") or "").split()


def _has_ltx_class(elem: etree._Element, name: str) -> bool:
    return name in _classes(elem)


def _flat_text(elem: etree._Element) -> str:
    """Whitespace-collapsed text of an element subtree.

    Math elements are replaced by their alttext when present, falling
    back to their flat itertext. Without this, math-bearing prose would
    extract gibberish (lxml's text_content() emits the recursive
    Unicode-rendered LaTeXML soup, which is correct for ar5iv but
    irrelevant to the prose adapter). The target_html applies the same
    alttext substitution structurally via :func:`_stub_inline_math`, so
    prompt text and target stay consistent and neither carries
    presentation MathML.
    """
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if isinstance(child.tag, str):
            tag_local = child.tag.split("}", 1)[-1]
            if tag_local == "math":
                alt = child.get("alttext") or ""
                if alt:
                    parts.append(alt)
                else:
                    parts.append(_flat_text(child))
            else:
                parts.append(_flat_text(child))
        if child.tail:
            parts.append(child.tail)
    return WS_RE.sub(" ", "".join(parts)).strip()


def _is_inside_tag(elem: etree._Element, tag: str) -> bool:
    """True if ``elem`` has any ancestor with the given local-name tag."""
    for anc in elem.iterancestors():
        if not isinstance(anc.tag, str):
            continue
        local = anc.tag.split("}", 1)[-1]
        if local == tag:
            return True
    return False


# --------------------------------------------------------------------------- #
# Per-region derivations
# --------------------------------------------------------------------------- #


def _heading_level(elem: etree._Element) -> int:
    return HEADING_TAG_LEVEL.get(elem.tag, 6)


def _paragraph_doc_role(elem: etree._Element) -> str | None:
    """Best-effort doc_role for a paragraph element.

    Mirrors ``structure_graph._PARAGRAPH_DOC_ROLE_OVERRIDES`` so the
    distribution of doc_role values seen at training time is consistent
    with what Stage 5 emits at inference time. Heuristics:

    * Inside ``<div class="ltx_abstract">``     -> "abstract"
    * Inside ``<div class="ltx_acknowledgement">`` -> "metadata"
    * Inside ``<div class="ltx_bibliography">`` -> "citation"
    * Inside any ``ltx_authors`` / ``ltx_creator`` ancestor -> "author"
    * Inside footer-shaped ``ltx_keyword``       -> "metadata"
    * else None (the live Semantic head's "body" default)

    None vs "body": ``prompts.py`` writes the doc_role through verbatim
    (None becomes JSON null), so we leave None as the "no override"
    sentinel — same way Stage 5 does (see
    structure_graph.py:_doc_role_override).
    """
    for anc in elem.iterancestors():
        if not isinstance(anc.tag, str):
            continue
        cls = _classes(anc)
        if "ltx_abstract" in cls:
            return "abstract"
        if "ltx_acknowledgement" in cls:
            return "metadata"
        if "ltx_bibliography" in cls:
            return "citation"
        if "ltx_authors" in cls or "ltx_creator" in cls:
            return "author"
        if "ltx_keywords" in cls:
            return "metadata"
        if "ltx_dates" in cls:
            return "metadata"
        if "ltx_classification" in cls:
            return "metadata"
    return None


def _list_items(elem: etree._Element) -> list[dict[str, Any]]:
    """Return per-item plain-text payloads for a <ul>/<ol>/<dl>.

    Mirrors the shape ``structure_graph.py`` emits for list regions:
    ``[{"fb_index": int, "text": str, "depth": int}, ...]``. We don't
    have FB indices at dataset-build time (ar5iv items don't map 1:1
    to PDF FBs), so we use the item's position within the list as the
    fb_index proxy. Depth is derived from list nesting.
    """
    items: list[dict[str, Any]] = []
    if elem.tag == "dl":
        # Definition list: pair each <dt> with its following <dd>(s).
        idx = 0
        for child in elem:
            if not isinstance(child.tag, str):
                continue
            if child.tag in ("dt", "dd"):
                items.append({
                    "fb_index": idx,
                    "text": _flat_text(child),
                    "depth": _list_nesting_depth(elem),
                    "tag": child.tag,
                })
                idx += 1
    else:
        for idx, li in enumerate(elem.xpath("./li")):
            items.append({
                "fb_index": idx,
                "text": _flat_text(li),
                "depth": _list_nesting_depth(li),
            })
    return items


def _list_nesting_depth(elem: etree._Element) -> int:
    """Count enclosing <ul>/<ol>/<dl> ancestors, capped at 3 to mirror
    Structure's ``depth_3plus`` bucket.
    """
    depth = 0
    for anc in elem.iterancestors():
        if not isinstance(anc.tag, str):
            continue
        local = anc.tag.split("}", 1)[-1]
        if local in ("ul", "ol", "dl"):
            depth += 1
            if depth >= 3:
                return 3
    return depth


def _aria_hints(elem: etree._Element) -> list[str]:
    """Collect aria-* attributes from the element itself.

    Stage 5 builds aria_hints from a mix of doc-role + ARIA-attr
    extraction; ar5iv pages don't carry aria-* attributes typically
    (LaTeXML doesn't emit them), so this is mostly empty in practice.
    We surface anything that is there to keep the envelope schema-faithful.
    """
    hints: list[str] = []
    for k, v in elem.attrib.items():
        if k.startswith("aria-"):
            hints.append(f"{k}={v}")
    return hints


# --------------------------------------------------------------------------- #
# Target HTML — strip + serialize
# --------------------------------------------------------------------------- #


def _strip_attrs_recursive(elem: etree._Element) -> None:
    """Apply the documented strip policy to ``elem`` and every descendant.

    Mutates in place — caller should clone the element first if it
    cares about preserving the source tree (we do, see
    ``_serialize_target_html``).
    """
    for el in elem.iter():
        if not isinstance(el.tag, str):
            continue

        # 1) class= filter: keep non-ltx_* tokens, drop the rest.
        cls = el.get("class")
        if cls:
            kept = [
                c for c in cls.split()
                if not any(c.startswith(p) for p in LTX_CLASS_DROP_PREFIXES)
            ]
            if kept:
                el.set("class", " ".join(kept))
            else:
                # Empty class — remove the attribute entirely.
                if "class" in el.attrib:
                    del el.attrib["class"]

        # 2) id= filter: drop LaTeXML-style auto IDs, keep author IDs.
        eid = el.get("id")
        if eid and LTX_ID_DROP_RE.match(eid):
            del el.attrib["id"]


# Encoding attr for the TeX annotation we keep when stubbing inline math.
_TEX_ENCODING = "application/x-tex"


def _stub_inline_math(root: etree._Element) -> int:
    """Replace every ``<math>`` subtree with a minimal TeX-annotation stub.

    The prose adapter must NOT emit presentation MathML — that is the
    math specialist's job (see routing: ``math`` regions -> MATH). ar5iv
    paragraphs embed full ``<semantics>`` MathML inline, which (a) blows
    each target past the 1024-token trainer cap (~74% of prose targets
    carry inline math; the presentation soup is ~4.8K chars per equation)
    and (b) trains prose to reproduce markup it shouldn't own.

    We collapse each ``<math>`` to::

        <math><annotation encoding="application/x-tex">{alttext}</annotation></math>

    preserving the element's document position and tail text so the
    surrounding prose is byte-identical. The TeX annotation (from the
    ar5iv ``alttext`` attribute, falling back to flattened itertext)
    keeps an accessibility hook a later deterministic LaTeX->MathML pass
    can expand. Returns the number of ``<math>`` elements stubbed.
    """
    # Top-level math only: a <math> nested inside another is discarded
    # wholesale when its ancestor is replaced.
    maths = [
        m for m in root.iter()
        if isinstance(m.tag, str)
        and m.tag.split("}", 1)[-1] == "math"
        and not _is_inside_tag(m, "math")
    ]
    for m in maths:
        parent = m.getparent()
        if parent is None:
            continue  # math is the root element itself — leave untouched.
        alt = m.get("alttext") or WS_RE.sub(" ", "".join(m.itertext())).strip()
        stub = etree.Element("math")
        ann = etree.SubElement(stub, "annotation")
        ann.set("encoding", _TEX_ENCODING)
        ann.text = alt
        stub.tail = m.tail
        parent.replace(m, stub)
    return len(maths)


def _serialize_target_html(elem: etree._Element) -> str:
    """Clone, strip, stub inline math, serialize the element subtree.

    Returns the HTML5 fragment as a string (no XML declaration, no
    namespace prefix re-mapping). Inline ``<math>`` is collapsed to a
    TeX-annotation stub (:func:`_stub_inline_math`) so the prose target
    never carries presentation MathML.
    """
    # Deep-copy so we don't mutate the source tree (we walk the same
    # tree more than once in the per-pair loop).
    clone = etree.fromstring(etree.tostring(elem))
    _strip_attrs_recursive(clone)
    _stub_inline_math(clone)
    return etree.tostring(clone, encoding="unicode", with_tail=False)


# --------------------------------------------------------------------------- #
# Region payload construction (matches Stage 5 region.payload shape)
# --------------------------------------------------------------------------- #


def _build_region_payload_and_text(
    *,
    elem: etree._Element,
    kind: str,
) -> tuple[dict[str, Any], str, str | None, int | None, bool | None,
           list[dict[str, Any]] | None]:
    """Construct the Region.payload dict + return derived fields.

    Returns
    -------
    (payload, text, doc_role, level_hint, ordered, items)

    where the right-side fields are denormalized copies of the
    payload sub-fields, returned for the row-level metadata columns.
    """
    text = _flat_text(elem)

    payload: dict[str, Any] = {}
    doc_role: str | None = None
    level_hint: int | None = None
    ordered: bool | None = None
    items: list[dict[str, Any]] | None = None

    if kind == "heading":
        level_hint = _heading_level(elem)
        payload["level_hint"] = level_hint
        payload["text"] = text
    elif kind == "paragraph":
        doc_role = _paragraph_doc_role(elem)
        payload["doc_role"] = doc_role
        payload["text"] = text
    elif kind == "list":
        ordered = (elem.tag == "ol")
        items = _list_items(elem)
        payload["ordered"] = ordered
        payload["items"] = items
        # text is _region_text's join-of-items fallback (matches
        # prompts._region_text branch).
        text = "\n".join((it.get("text") or "") for it in items)
        payload["text"] = None
    elif kind == "definition_list":
        items = _list_items(elem)
        payload["items"] = items
        text = "\n".join((it.get("text") or "") for it in items)
        payload["text"] = None
    elif kind == "code_block":
        payload["text"] = text
    elif kind == "blockquote":
        payload["text"] = text
    elif kind == "figure":
        payload["text"] = text
    elif kind == "metadata_drop":
        payload["text"] = text
    elif kind == "form":
        # ar5iv has no native form regions; we never emit form here,
        # but the schema is filled in for completeness.
        payload["text"] = text
    else:
        payload["text"] = text

    return payload, text, doc_role, level_hint, ordered, items


def _make_region_namespace(
    *, kind: str, payload: dict[str, Any], aria_hints: tuple[str, ...]
) -> SimpleNamespace:
    """Stand-in for ``structure_graph.Region``.

    ``build_prose_request`` reads ``region.kind``,
    ``region.payload``, ``region.aria_hints``,
    ``region.feature_block_indices`` — so a SimpleNamespace with those
    four attributes is enough; using one avoids importing the full
    Region dataclass.
    """
    return SimpleNamespace(
        payload=payload,
        feature_block_indices=(),
        aria_hints=aria_hints,
        kind=kind,
    )


# --------------------------------------------------------------------------- #
# Per-element row construction
# --------------------------------------------------------------------------- #


def build_row_for_element(
    *,
    elem: etree._Element,
    kind: str,
    arxiv_id: str,
    source_pair: str,
    region_idx: int,
    skip: SkipCounters,
    tok: Any = None,
    max_token_len: int = 0,
) -> dict[str, Any] | None:
    """Return a JSONL-serializable row, or ``None`` to skip.

    When ``tok`` is provided and ``max_token_len > 0``, rows whose
    chat-template-wrapped (prompt+target) token length exceeds the cap
    are dropped (``skip.over_max_tokens``). The whitespace filter above
    is necessary but not sufficient — inline-math stubs and dense markup
    can still push a 450-word region past the trainer's BPE max_len.
    """

    # Skip if inside a <table> or <math>.
    if _is_inside_tag(elem, "table"):
        skip.inside_table += 1
        return None
    if _is_inside_tag(elem, "math"):
        skip.inside_math += 1
        return None

    # ar5iv "ltx_table" figures wrap data tables (caption + table). The
    # table builder handles those; we'd over-extract their figcaptions.
    if kind == "figure" and _has_ltx_class(elem, "ltx_table"):
        skip.figure_with_table += 1
        return None

    payload, text, doc_role, level_hint, ordered, items = (
        _build_region_payload_and_text(elem=elem, kind=kind)
    )
    aria_hints = _aria_hints(elem)

    # Length filter (whitespace tokens).
    n_text_tokens = _ws_token_count(text)
    if n_text_tokens < MIN_REGION_TOKENS:
        skip.too_short += 1
        return None
    if n_text_tokens > MAX_REGION_TOKENS:
        skip.too_long += 1
        return None
    if not text.strip():
        skip.empty_text += 1
        return None

    try:
        target_html = _serialize_target_html(elem)
    except Exception:
        skip.serialize_error += 1
        return None
    if len(target_html.strip()) < MIN_TARGET_CHARS:
        skip.target_too_short += 1
        return None

    region = _make_region_namespace(
        kind=kind, payload=payload, aria_hints=tuple(aria_hints),
    )
    request = build_prose_request(region, feature_blocks=())
    request_payload = {
        "adapter": request.adapter.value,
        "payload": request.payload,
        "request_id": request.request_id,
        "sampling_overrides": request.sampling_overrides,
    }

    n_target_tokens = _ws_token_count(target_html)

    # Tokenizer-aware length filter: measure exactly what the trainer
    # tokenizes (wrap_for_qwen over prompt+target) and drop rows past
    # the cap so we never silently truncate the completion-only target.
    n_wrap_tokens: int | None = None
    if tok is not None and max_token_len > 0:
        prompt = request_payload["payload"].get("prompt")
        if isinstance(prompt, str) and prompt:
            wrapped = wrap_for_qwen(tok, prompt, target=target_html)
            n_wrap_tokens = len(tok(wrapped, add_special_tokens=False)["input_ids"])
            if n_wrap_tokens > max_token_len:
                skip.over_max_tokens += 1
                return None

    # Heading-level doc_role hint (informational only — the prose
    # envelope only carries doc_role for paragraph; for heading we
    # record into extraction_metadata).
    heading_doc_role_hint: str | None = None
    if kind == "heading":
        cls = _classes(elem)
        for c in cls:
            if c in LTX_HEADING_DOC_ROLE:
                heading_doc_role_hint = LTX_HEADING_DOC_ROLE[c]
                break

    return {
        "source_pair": source_pair,
        "arxiv_id": arxiv_id,
        "region_idx": region_idx,
        "kind": kind,
        "level_hint": level_hint,
        "doc_role": doc_role,
        "ordered": ordered,
        "items": items,
        "aria_hints": aria_hints,
        "text": text,
        "request_payload": request_payload,
        "target_html": target_html,
        "n_text_tokens": n_text_tokens,
        "n_target_tokens": n_target_tokens,
        "n_wrap_tokens": n_wrap_tokens,
        "extraction_metadata": {
            "tag": elem.tag,
            "ltx_class": " ".join(_classes(elem)) or None,
            "heading_doc_role_hint": heading_doc_role_hint,
            "n_aria_hints": len(aria_hints),
        },
    }


# --------------------------------------------------------------------------- #
# Per-pair iteration
# --------------------------------------------------------------------------- #


def _is_emitted_under_ancestor(
    elem: etree._Element,
    emitted_ids: set[int],
) -> bool:
    """True if any ancestor of ``elem`` is in ``emitted_ids``.

    Used so we don't emit a <p> inside a <blockquote> when we already
    emitted the <blockquote> as one region. ``emitted_ids`` is the set
    of ``id(ancestor)`` values for already-emitted region elements
    within the same pair.
    """
    for anc in elem.iterancestors():
        if id(anc) in emitted_ids:
            return True
    return False


def _iter_prose_elements(html_str: str) -> Iterator[tuple[etree._Element, str]]:
    """Yield ``(elem, kind)`` for every prose-shaped element in document order.

    We walk the body looking for elements whose tag is in TAG_TO_KIND.
    Skipping descendants of an already-emitted region happens at row
    build time (so figures' inner <p>s don't get re-extracted as their
    own paragraph).
    """
    try:
        root = lxml_html.fromstring(html_str)
    except Exception:
        return iter(())
    body = root.find(".//body")
    if body is None:
        body = root
    out: list[tuple[etree._Element, str]] = []
    for el in body.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag
        # Drop namespaced tags (we don't want SVG/MathML noise).
        if "}" in tag:
            continue
        kind = TAG_TO_KIND.get(tag)
        if kind is None:
            continue
        out.append((el, kind))
    return iter(out)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def iter_pair_files(pairs_dir: Path) -> Iterator[Path]:
    yield from sorted(pairs_dir.glob("*.json"))


def _open_split_files(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train": (out_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "test.jsonl").open("w", encoding="utf-8"),
    }


def _assert_no_gpu() -> None:
    """Defensive: refuse to run if CUDA looks engaged.

    Phase 4b training is on the GPU concurrently; we must not contend.
    The simplest signal is the ``CUDA_VISIBLE_DEVICES`` env var — if
    explicitly empty, we're CPU-locked. Otherwise we just warn and
    continue (we never import torch anyway, but this is a belt-and-
    braces check).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd != "":
        # Not explicitly empty — print a warning but don't refuse.
        # The builder doesn't import torch, so it can't actually use
        # the GPU; the warning is documentation for the operator.
        sys.stderr.write(
            f"[note] CUDA_VISIBLE_DEVICES={cvd!r}; this script is CPU-only "
            "but invoke with CUDA_VISIBLE_DEVICES= to be defensive\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Stop after this many pair files (0 = all). "
                         "Useful for smoke tests.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=200,
                    help="Print a progress line every N pair files.")
    ap.add_argument("--max-token-len", type=int, default=1024,
                    help="Drop rows whose chat-template-wrapped token "
                         "length exceeds this. Should match the prose "
                         "adapter's max_len in train_config.py. 0 disables "
                         "the tokenizer-aware filter (whitespace only).")
    ap.add_argument("--tokenizer-name", type=str,
                    default=DEFAULT_FILTER_TOKENIZER,
                    help="HuggingFace tokenizer used for the length "
                         "filter; must match the trainer's base model.")
    args = ap.parse_args()

    if not args.pairs_dir.exists():
        sys.exit(f"[fatal] pairs dir not found: {args.pairs_dir}")

    _assert_no_gpu()

    rng = random.Random(args.seed)  # reserved for any future shuffling
    _ = rng                         # silence the "unused" linter
    stats = BuildStats()
    handles = _open_split_files(args.out_dir)

    filter_tok = None
    if args.max_token_len > 0:
        print(f"[setup] loading filter tokenizer: {args.tokenizer_name}",
              flush=True)
        filter_tok = _load_filter_tokenizer(args.tokenizer_name)
        print(f"[setup] tokenizer loaded; max_token_len={args.max_token_len}",
              flush=True)

    seen_arxiv_ids: dict[str, str] = {}            # id -> split

    try:
        for i, pair_path in enumerate(iter_pair_files(args.pairs_dir)):
            if args.max_pairs and i >= args.max_pairs:
                break
            stats.n_pairs_seen += 1
            try:
                d = json.loads(pair_path.read_text(encoding="utf-8"))
            except Exception as exc:
                sys.stderr.write(f"[parse-error] {pair_path}: {exc}\n")
                stats.skip.parse_error += 1
                continue
            arxiv_id = d.get("arxiv_id")
            html_str = d.get("raw_source_html")
            if not html_str:
                continue
            stats.n_pairs_with_html += 1
            if not arxiv_id:
                continue
            stats.n_pairs_with_arxiv_id += 1

            # Per-arxiv_id dedup: pair files are sectioned (~5x duplication).
            if arxiv_id in seen_arxiv_ids:
                continue
            split = stable_split_for_id(arxiv_id, 0.80, 0.10)
            seen_arxiv_ids[arxiv_id] = split
            stats.n_unique_arxiv_ids_seen += 1

            source_pair = pair_path.stem

            # Per-pair: track ancestors of already-emitted regions so we
            # don't double-extract (e.g. a <p> inside a <figure>).
            emitted_ancestor_ids: set[int] = set()

            region_idx = 0
            for elem, kind in _iter_prose_elements(html_str):
                stats.n_elements_inspected += 1
                if _is_emitted_under_ancestor(elem, emitted_ancestor_ids):
                    stats.skip.nested_skip += 1
                    continue

                row = build_row_for_element(
                    elem=elem,
                    kind=kind,
                    arxiv_id=arxiv_id,
                    source_pair=source_pair,
                    region_idx=region_idx,
                    skip=stats.skip,
                    tok=filter_tok,
                    max_token_len=args.max_token_len,
                )
                if row is None:
                    continue

                # Once a region is emitted, mark its element so any of
                # its descendants are skipped by the nested-skip check.
                emitted_ancestor_ids.add(id(elem))

                row["split"] = split
                handles[split].write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                handles[split].flush()
                stats.n_rows_emitted += 1
                stats.rows_per_split[split] += 1
                stats.rows_per_kind[kind] += 1
                stats.rows_per_doc_role[
                    row.get("doc_role")
                    or row["extraction_metadata"].get("heading_doc_role_hint")
                    or "_none"
                ] += 1
                stats.rows_per_arxiv_id[arxiv_id] += 1
                stats.src_token_hist[
                    _bucket_for(row["n_text_tokens"], TOKEN_HIST_EDGES)
                ] += 1
                stats.tgt_token_hist[
                    _bucket_for(row["n_target_tokens"], TOKEN_HIST_EDGES)
                ] += 1
                region_idx += 1

            if (i + 1) % args.progress_every == 0:
                total_skips = (
                    stats.skip.too_short
                    + stats.skip.too_long
                    + stats.skip.target_too_short
                    + stats.skip.empty_text
                    + stats.skip.nested_skip
                    + stats.skip.inside_table
                    + stats.skip.inside_math
                    + stats.skip.figure_with_table
                )
                print(
                    f"[progress] pairs={i+1} "
                    f"unique_arxiv={stats.n_unique_arxiv_ids_seen} "
                    f"elems_seen={stats.n_elements_inspected} "
                    f"rows={stats.n_rows_emitted} "
                    f"skips={total_skips}",
                    flush=True,
                )
    finally:
        for h in handles.values():
            h.close()

    # Coverage report.
    coverage = {
        "schema_version": 2,
        "seed": args.seed,
        "pairs_dir": str(args.pairs_dir),
        "out_dir": str(args.out_dir),
        "max_token_len_filter": args.max_token_len,
        "filter_tokenizer": args.tokenizer_name if args.max_token_len > 0 else None,
        "n_pairs_seen": stats.n_pairs_seen,
        "n_pairs_with_html": stats.n_pairs_with_html,
        "n_pairs_with_arxiv_id": stats.n_pairs_with_arxiv_id,
        "n_unique_arxiv_ids_seen": stats.n_unique_arxiv_ids_seen,
        "n_elements_inspected": stats.n_elements_inspected,
        "n_rows_emitted": stats.n_rows_emitted,
        "rows_per_split": dict(stats.rows_per_split),
        "rows_per_kind": dict(stats.rows_per_kind),
        "rows_per_doc_role": dict(stats.rows_per_doc_role),
        "n_unique_arxiv_ids_in_dataset": len(stats.rows_per_arxiv_id),
        "rows_per_arxiv_id_top20": dict(
            stats.rows_per_arxiv_id.most_common(20)
        ),
        "src_token_hist": dict(sorted(stats.src_token_hist.items())),
        "tgt_token_hist": dict(sorted(stats.tgt_token_hist.items())),
        "skip": {
            "too_short": stats.skip.too_short,
            "too_long": stats.skip.too_long,
            "target_too_short": stats.skip.target_too_short,
            "empty_text": stats.skip.empty_text,
            "serialize_error": stats.skip.serialize_error,
            "parse_error": stats.skip.parse_error,
            "nested_skip": stats.skip.nested_skip,
            "inside_table": stats.skip.inside_table,
            "inside_math": stats.skip.inside_math,
            "figure_with_table": stats.skip.figure_with_table,
            "over_max_tokens": stats.skip.over_max_tokens,
        },
        "notes": {
            "split_function": (
                "data.common._splits.stable_split_for_id (re-implementation of "
                "scripts.build_semantic_preservation_dataset's). "
                "train_frac=0.80, val_frac=0.10. Same arxiv_id -> "
                "same split across all qwen_* and "
                "semantic_preservation builders."
            ),
            "kinds_covered": (
                "Per semantik_structure.qwen_specialists.routing, the prose "
                "adapter handles every Stage-5 region kind except table "
                "and math: paragraph, heading, list, definition_list, "
                "code_block, blockquote, figure, form, metadata_drop. "
                "ar5iv exposes paragraph/heading/list/code_block/"
                "blockquote/figure/definition_list/metadata_drop "
                "natively; form regions are not present in ar5iv ground-"
                "truth (LaTeX papers don't have AcroForms) so this "
                "builder emits zero form rows."
            ),
            "target_format": (
                "lxml.etree.tostring(<elem>) of the cloned subtree, "
                "with conservative attribute strips: ltx_* class "
                "tokens dropped (full attribute removed if no other "
                "tokens remain), and LaTeXML auto-id values dropped "
                "(S\\d/Sx/p\\d/id\\d/E\\d/cite./bib./fn\\d patterns); "
                "all other attributes preserved verbatim. Inline <math> "
                "is collapsed to <math><annotation encoding="
                "'application/x-tex'>{alttext}</annotation></math> "
                "(presentation MathML is the math adapter's job, not "
                "prose's) — see _stub_inline_math."
            ),
            "doc_role_heuristic": (
                "Paragraph doc_role mirrors structure_graph.py's "
                "_PARAGRAPH_DOC_ROLE_OVERRIDES via ancestor-class "
                "inspection: ltx_abstract->abstract, ltx_authors / "
                "ltx_creator->author, ltx_bibliography->citation, "
                "ltx_acknowledgement / ltx_keywords / ltx_dates / "
                "ltx_classification->metadata. None otherwise (= "
                "'no override', not 'body')."
            ),
            "form_regions": (
                "ar5iv has no <form>/<input> elements, so no form-kind "
                "rows are produced. The form adapter slot is reserved "
                "for future training-data acquisition (see "
                "qwen_specialists/routing.py module docstring on "
                "form->PROSE)."
            ),
            "length_filter": (
                f"text token count must be in [{MIN_REGION_TOKENS}, "
                f"{MAX_REGION_TOKENS}] (matches "
                f"semantic_preservation's filter)."
            ),
        },
    }
    (args.out_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2)
    )

    print()
    print(f"[done] rows={stats.n_rows_emitted} "
          f"(train={stats.rows_per_split.get('train', 0)} "
          f"val={stats.rows_per_split.get('val', 0)} "
          f"test={stats.rows_per_split.get('test', 0)})")
    print(f"[done] elems_seen={stats.n_elements_inspected} skips="
          f"short:{stats.skip.too_short} "
          f"long:{stats.skip.too_long} "
          f"empty:{stats.skip.empty_text} "
          f"target_short:{stats.skip.target_too_short} "
          f"nested:{stats.skip.nested_skip} "
          f"in_table:{stats.skip.inside_table} "
          f"in_math:{stats.skip.inside_math}")
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
