"""Build the Qwen-gap-fill LoRA adapter training dataset (Phase 1.4).

Scope
-----

Per ``Plans/04_assembler_layer_investigation.md`` §1/§2 and the live
gap-detection logic in ``dart_semantic.assembler.pass_9a.run_pass_9a``,
the assembler flags these gap kinds (and this builder emits rows for
each):

    * ``missing_title`` — emitted when no Semantic.title region was
      stamped AND no <h1> survived heading-tree normalization. The
      gap-fill prompt receives ``first_paragraph_text`` + ``doc_lang``
      (see ``prompts.build_gap_fill_request`` ``missing_title`` branch).
    * ``author_block`` — emitted when Semantic top-1 == "author" with
      conf >= τ_gap on a paragraph/heading region not already wrapped
      in ``<address>``. The prompt receives ``raw_text`` +
      ``surrounding_context``.
    * ``citation_unresolved`` — emitted when an inline ref's target is
      missing but a near-miss target exists. Ground truth is the
      inline-cite -> reference link ar5iv (``ltx_ref`` href="#bib.bibN")
      and PMC JATS (``<xref ref-type="bibr">``) carry natively; we
      "make the gap" by stripping the wrapper. Each cite emits a
      POSITIVE row (candidate_targets includes the true referent,
      target = restored anchor) and, when a near-miss sibling exists, a
      NEGATIVE/no-op row (near-miss-only candidate_targets, target = the
      verbatim plain text) so the adapter learns NOT to hallucinate.

    * ``copyright_block`` / ``legal_disclaimer`` — emitted when
      BERT-Semantic flags a paragraph as ``legal`` (or Stage 5 stamped a
      legal/copyright ``doc_role``), split by the deterministic
      copyright sub-flag shared with the runtime
      (``prompts.COPYRIGHT_SUBFLAG_RE``). Ground truth is found
      MECHANICALLY (no LLM labels): PMC JATS ``<permissions>``
      (``<copyright-statement>`` → copyright, ``<license-p>`` → legal,
      ~3.5K/1.7K docs) plus boilerplate paragraphs in the PD/CC HTML
      corpora (gutenberg © pages, courtlistener/CFR/federal_register
      legal text) matched by a strict ©/year regex or anchored
      disclaimer phrases with a position prior (first/last
      ``_LEGAL_POSITION_PRIOR`` blocks). The target is the canonical
      ``<aside class="copyright|legal"><p>…</p></aside>`` wrap of the
      verbatim text. Highly-repeated boilerplate (e.g. the CC-BY
      license paragraph, identical across 1,000+ PMC articles) is
      capped at ``_LEGAL_DUP_CAP`` occurrences per distinct normalized
      text so template echo can't swamp the kind. License: PMC OA /
      Project Gutenberg / US-gov / CourtListener only — wikipedia
      (CC-BY-SA) and arXiv are excluded from these kinds
      (feedback_license_policy; Plans/08 do-NOT list).

Source side / contract
----------------------

The source side of every row is the EXACT JSON envelope that
``dart_semantic.qwen_specialists.prompts.build_gap_fill_request``
emits at inference time, given a ``GapSlot`` reconstructed from the
ar5iv ground-truth HTML. Because the prompt builder reads two
``GapSlot`` attributes (``kind`` and ``context``) plus a third
(``region_index`` for the metadata sidecar), we build a
``SimpleNamespace`` with those three fields plus an enum-shaped
``kind`` (the builder reads ``kind.value`` first, falling back to
``str(kind)``) and feed it to the live builder.

The contract test at
``tests/test_qwen_gap_fill_dataset_contract.py`` re-runs
``build_gap_fill_request`` against each stored row and pins the
prompt byte-for-byte (modulo whitespace normalisation).

K=8 diversity / paraphrastic targets
------------------------------------

The gap-fill adapter generates K=8 candidates at inference (per
``dart_semantic/qwen_specialists/config.yaml``). To teach diversity
without inventing facts, we emit 2-3 deterministic, rule-based
paraphrastic gold targets per slot where the source allows:

    missing_title:
        variant 0 — original h1 + <title> envelope
        variant 1 — title with leading "An Investigation of " /
                    "An Overview of " / "On the " / "A Study of "
                    stripped, OR (failing that) with everything after
                    a colon (subtitle) dropped
        variant 2 — title with everything after a colon kept and
                    the prefix before the colon dropped (the "subtitle
                    as title" reformulation), only emitted if the
                    title contains a colon and variant 1 produced a
                    different fragment
    author_block:
        variant 0 — full names exactly as ar5iv parsed them
        variant 1 — initialised first names ("J. Smith" instead of
                    "John Smith")

Variant emission is gated by source-text shape: if a transformation
yields the same string as variant 0, it is skipped (we don't emit
duplicates). This means most papers emit either 1 or 2 variants for
the title; author blocks always emit 1 or 2.

Per-kind balance cap (Plans/08 §3; feedback_balance_dominant_category)
----------------------------------------------------------------------

``citation_unresolved`` is ~6x more abundant than the other kinds at
full extraction volume (25,628 vs 4,338 ``missing_title`` in the
2026-05 build); training on the raw mix regresses the minor kinds (the
exact footgun that hit the structure_v2 role head and the earlier
gap_fill citation work). The builder therefore caps dominant kinds via
``--cap-per-kind``:

    --cap-per-kind -1   (default) AUTO — cap = round(1.5 x the row
                        count of the SECOND-largest kind), computed
                        from the extracted data; only kinds above the
                        cap (in practice the dominant one) are
                        downsampled.
    --cap-per-kind 0    disable capping entirely.
    --cap-per-kind N    explicit cap of N rows applied to every kind.

Downsampling is DETERMINISTIC (seeded by ``--seed``, default 42) and
operates on whole ``(arxiv_id, gap_kind)`` groups — never individual
variants — so ``variant_idx`` runs stay contiguous (the contract test
pins this), and it is stratified per split (proportional quotas) so
the train/val/test shares of the capped kind are preserved.

CPU-ONLY. Loads no ML weights, no tokenizers — token counts are
whitespace approximated. Safe to run while training is on the GPU.

Usage::

    python data/build_gap_fill_qwen_data.py                      # full
    python data/build_gap_fill_qwen_data.py --max-pairs 30 \
        --out-dir /tmp/qwen_gap_fill_smoke                       # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from lxml import etree, html as lxml_html

# Live inference-time prompt builder.
#
# We deliberately do NOT import ``dart_semantic.assembler.types.GapKind``
# even though that is the canonical source — importing the assembler
# package transitively triggers ``dart_semantic.gates.__init__`` which
# imports ``axe_playwright_python``, an inference-time-only dependency
# that is not always installed in the dataset-build environment. The
# live ``build_gap_fill_request`` only reads ``slot.kind.value`` (or
# falls back to ``str(slot.kind)``), so a tiny ``str`` Enum with
# matching values is contract-equivalent. The unit test
# ``test_local_gapkind_matches_canonical`` pins the values to the
# upstream definition.
from enum import Enum


class GapKind(str, Enum):
    MISSING_TITLE = "missing_title"
    AUTHOR_BLOCK = "author_block"
    CITATION_UNRESOLVED = "citation_unresolved"
    COPYRIGHT_BLOCK = "copyright_block"
    LEGAL_DISCLAIMER = "legal_disclaimer"


# The legal/copyright sub-flag heuristics are SHARED with the runtime
# detector (pass_9a imports the same names) so the copyright-vs-
# disclaimer split and the legal_subkind discriminator can never drift
# between training rows and inference GapSlots.
from dart_semantic.qwen_specialists.prompts import (
    COPYRIGHT_SUBFLAG_RE,
    COPYRIGHT_YEAR_RE,
    build_gap_fill_request,
    classify_legal_subkind,
)

from data._splits import stable_split_for_id


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_PAIRS_DIR = Path("data/pairs/arxiv")
DEFAULT_OUT_DIR = Path("data/qwen_gap_fill_dataset")

WS_RE = re.compile(r"\s+")

# Histogram bucket edges (right-exclusive).
TOKEN_HIST_EDGES = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280]

# Regex prefixes the missing_title variant 1 strips (case-insensitive,
# only at the start of the title). Order matters — longer prefixes
# first, so "An Investigation of " is preferred over "An ".
TITLE_PREFIX_STRIPS = [
    re.compile(r"^An?\s+Investigation\s+(?:of|into)\s+", re.IGNORECASE),
    re.compile(r"^An?\s+Overview\s+of\s+", re.IGNORECASE),
    re.compile(r"^An?\s+Introduction\s+to\s+", re.IGNORECASE),
    re.compile(r"^An?\s+Analysis\s+of\s+", re.IGNORECASE),
    re.compile(r"^An?\s+Empirical\s+Study\s+of\s+", re.IGNORECASE),
    re.compile(r"^A\s+Study\s+of\s+", re.IGNORECASE),
    re.compile(r"^A\s+Survey\s+of\s+", re.IGNORECASE),
    re.compile(r"^A\s+Note\s+on\s+", re.IGNORECASE),
    re.compile(r"^On\s+the\s+", re.IGNORECASE),
    re.compile(r"^Towards?\s+(?:a\s+|an\s+)?", re.IGNORECASE),
]


# --------------------------------------------------------------------------- #
# Skip / counter bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class SkipCounters:
    no_html: int = 0
    no_arxiv_id: int = 0
    no_title: int = 0
    no_author: int = 0
    parse_error: int = 0
    title_too_short: int = 0
    author_too_short: int = 0
    no_citation: int = 0
    no_legal_block: int = 0
    legal_dup_capped: int = 0


@dataclass
class BuildStats:
    n_pairs_seen: int = 0
    n_unique_arxiv_ids: int = 0
    n_rows_emitted: int = 0
    rows_per_split: Counter = field(default_factory=Counter)
    rows_per_gap_kind: Counter = field(default_factory=Counter)
    rows_per_variant: Counter = field(default_factory=Counter)
    tgt_token_hist: Counter = field(default_factory=Counter)
    rows_per_arxiv_id: Counter = field(default_factory=Counter)
    skip: SkipCounters = field(default_factory=SkipCounters)


def _bucket_for(n: int, edges: list[int]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= n < edges[i + 1]:
            return f"[{edges[i]},{edges[i + 1]})"
    return f"[{edges[-1]},inf)"


def _ws_token_count(s: str) -> int:
    return len([t for t in WS_RE.split(s.strip()) if t])


def _norm_text(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


# --------------------------------------------------------------------------- #
# ar5iv extractors — title, author block, first paragraph, lang
# --------------------------------------------------------------------------- #


def _extract_title_element(root: etree._Element) -> etree._Element | None:
    """Return the document <h1> (ltx_title_document) or None.

    ar5iv wraps the paper title in
    ``<h1 class="ltx_title ltx_title_document">``. We prefer that
    over the head ``<title>`` because (a) head <title> includes the
    arxiv id in brackets ("[0709.0551] ...") and (b) the assembler's
    title detection ladder targets the body H1, not the head title.
    """
    cands = root.xpath(".//h1[contains(@class, 'ltx_title_document')]")
    if cands:
        return cands[0]
    # Fallback: first <h1> anywhere.
    cands = root.xpath(".//h1")
    if cands:
        return cands[0]
    return None


def _extract_title_text(elem: etree._Element) -> str:
    """Flatten the title element to plain text, normalised whitespace."""
    return _norm_text(" ".join(elem.itertext()))


def _extract_author_block(root: etree._Element) -> etree._Element | None:
    """Return the ar5iv ``<div class="ltx_authors">`` element or None.

    ar5iv consistently uses this container; older / hand-rolled HTML
    might use ``<address>`` directly, so we fall back to that.
    """
    cands = root.xpath(".//div[contains(@class, 'ltx_authors')]")
    if cands:
        return cands[0]
    cands = root.xpath(".//address")
    if cands:
        return cands[0]
    return None


def _extract_author_names(authors_elem: etree._Element) -> list[str]:
    """Return one cleaned name per ``<span class="ltx_personname">``.

    ar5iv emits one personname span per author. Inside the span,
    affiliations / footnotes are nested in their own
    ``<span class="ltx_note">`` / ``<sup>`` children — we strip those
    by walking the span and collecting only direct text + descendant
    text from non-note children. This is a best-effort heuristic;
    the dataset records the raw element HTML as the author_block
    target so the trained adapter sees the canonical structure
    regardless of how we parsed the names for variant 1.
    """
    names: list[str] = []
    for span in authors_elem.xpath(".//span[contains(@class, 'ltx_personname')]"):
        # Walk the span, dropping any subtree that is a footnote/note
        # (ltx_note, ltx_role_footnote, ltx_author_notes) before
        # flattening to text.
        clone = etree.fromstring(etree.tostring(span))
        for bad in clone.xpath(
            ".//*[contains(@class, 'ltx_note') "
            "or contains(@class, 'ltx_role_footnote') "
            "or contains(@class, 'ltx_author_notes') "
            "or contains(@class, 'ltx_contact') "
            "or contains(@class, 'ltx_ERROR')"
            "or self::sup]"
        ):
            parent = bad.getparent()
            if parent is not None:
                # remove the bad node but preserve its tail text in parent
                tail = bad.tail or ""
                idx = parent.index(bad)
                parent.remove(bad)
                if tail:
                    if idx > 0:
                        prev = parent[idx - 1]
                        prev.tail = (prev.tail or "") + tail
                    else:
                        parent.text = (parent.text or "") + tail
        text = _norm_text(" ".join(clone.itertext()))
        # Split on " and " / commas conservatively if a single
        # personname span contains multiple comma-separated authors
        # (some ar5iv pages collapse them this way).
        for piece in re.split(r"\s+and\s+|,(?=\s*[A-Z])", text):
            piece = piece.strip().rstrip(",")
            if piece and not piece.isdigit():
                names.append(piece)
    return names


def _extract_first_paragraph(root: etree._Element) -> str:
    """Return up-to-500-char first paragraph text.

    Mirrors what ``pass_9a`` does: scan in document order looking for
    the first ``<p>`` (ar5iv wraps these in ``ltx_p``) and return up
    to 500 characters of flat text.
    """
    # ar5iv conventional class.
    paras = root.xpath(".//p[contains(@class, 'ltx_p')]")
    if not paras:
        paras = root.xpath(".//p")
    for p in paras:
        text = _norm_text(" ".join(p.itertext()))
        if text:
            return text[:500]
    return ""


def _extract_doc_lang(root: etree._Element) -> str:
    """Return the document's lang attribute (fall back to "en")."""
    lang = (root.get("lang") or "").strip()
    return lang or "en"


# --------------------------------------------------------------------------- #
# citation_unresolved extraction (Plans/04 §1.4 + §2; Plans/08 §3)
# --------------------------------------------------------------------------- #
#
# Ground-truth recipe: ar5iv and PMC carry the inline-cite -> reference link
# natively (ar5iv ``<a class="ltx_ref" href="#bib.bibN">N</a>``; PMC JATS
# ``<xref ref-type="bibr" rid="...">N</xref>``). We "make the gap" by
# stripping the wrapper and recording the restored anchor as the POSITIVE
# target. We also synthesise NEGATIVE rows (a near-miss candidate set that
# does NOT include the true referent) whose target is the plain-text span,
# teaching the adapter to refuse hallucinating an anchor.
#
# License: ar5iv (CC-cleared subset already paired) + PMC OA (JATS) only.

_LTX_REF_BIB_RE = "bib.bib"


@dataclass
class CiteSample:
    """One inline citation extracted from a CC-cleared source pair."""

    match_text: str  # the literal inline text, e.g. "12" or "[12]"
    anchor_id: str  # the true target anchor id, e.g. "ref-12"
    reference_kind: str  # "numbered_citation"|"section"|"figure"|"table"
    surrounding: str  # up to 50 chars either side of the match
    true_key: str  # the numeric/section key of the true target
    sibling_keys: list[str]  # other target keys present in the doc (for negatives)


def _slug_anchor(reference_kind: str, key: str) -> str:
    """Canonical DART anchor id for a (kind, key) pair (matches pass_9a)."""
    if reference_kind == "section":
        return f"sec-{key.replace('.', '-')}"
    prefix = {
        "figure": "fig",
        "table": "tab",
        "numbered_citation": "ref",
    }.get(reference_kind, "ref")
    return f"{prefix}-{key}"


def _ar5iv_citations(root: etree._Element) -> list[CiteSample]:
    """Pull inline numbered-citation refs from an ar5iv document.

    Restricts to bibliography refs (``href="#bib.bibN"``) — the cleanest,
    most abundant, shortest target. Section/figure/table ar5iv refs use
    nested ``<span>`` tags whose visible text is not a clean number, so
    we leave them out of the V1 citation rows (the assembler still
    DETECTS them at runtime; we just don't synthesise training rows for
    the noisier shapes).
    """
    samples: list[CiteSample] = []
    # All bib target ids defined in the doc (the resolvable universe).
    present_nums: list[str] = []
    for el in root.xpath(".//*[@id]"):
        ident = el.get("id") or ""
        m = re.match(r"bib\.bib(\d+)$", ident)
        if m:
            present_nums.append(m.group(1))
    present_set = sorted(set(present_nums), key=lambda s: int(s))

    for a in root.xpath('.//a[contains(@class, "ltx_ref")]'):
        href = (a.get("href") or "").strip()
        if not href.startswith("#" + _LTX_REF_BIB_RE):
            continue
        m = re.match(r"#bib\.bib(\d+)$", href)
        if m is None:
            continue
        num = m.group(1)
        visible = _norm_text(" ".join(a.itertext()))
        if not visible or not visible.isdigit():
            continue
        # Surrounding text: the parent's flattened text, windowed.
        parent = a.getparent()
        ptext = _norm_text(" ".join(parent.itertext())) if parent is not None else visible
        idx = ptext.find(visible)
        if idx < 0:
            surrounding = visible
        else:
            lo = max(0, idx - 50)
            hi = min(len(ptext), idx + len(visible) + 50)
            surrounding = ptext[lo:hi]
        samples.append(
            CiteSample(
                match_text=visible,
                anchor_id=_slug_anchor("numbered_citation", num),
                reference_kind="numbered_citation",
                surrounding=surrounding,
                true_key=num,
                sibling_keys=[k for k in present_set if k != num],
            )
        )
    return samples


def _pmc_citations(root: etree._Element) -> list[CiteSample]:
    """Pull inline numbered-citation refs from a PMC JATS document.

    Uses ``<xref ref-type="bibr" rid="...">N</xref>``. The true anchor
    is derived from the visible number (canonicalised to ``ref-N``);
    siblings come from all ``ref-type="bibr"`` rids in the doc.
    """
    samples: list[CiteSample] = []
    present_nums: list[str] = []
    for xref in root.xpath('.//xref[@ref-type="bibr"]'):
        visible = _norm_text(" ".join(xref.itertext()))
        if visible.isdigit():
            present_nums.append(visible)
    present_set = sorted(set(present_nums), key=lambda s: int(s))

    for xref in root.xpath('.//xref[@ref-type="bibr"]'):
        visible = _norm_text(" ".join(xref.itertext()))
        if not visible or not visible.isdigit():
            continue
        num = visible
        parent = xref.getparent()
        ptext = _norm_text(" ".join(parent.itertext())) if parent is not None else visible
        idx = ptext.find(visible)
        if idx < 0:
            surrounding = visible
        else:
            lo = max(0, idx - 50)
            hi = min(len(ptext), idx + len(visible) + 50)
            surrounding = ptext[lo:hi]
        samples.append(
            CiteSample(
                match_text=visible,
                anchor_id=_slug_anchor("numbered_citation", num),
                reference_kind="numbered_citation",
                surrounding=surrounding,
                true_key=num,
                sibling_keys=[k for k in present_set if k != num],
            )
        )
    return samples


def _citation_candidate_targets(
    sample: CiteSample,
    *,
    include_true: bool,
) -> list[dict[str, str]]:
    """Build the candidate_targets list for a citation slot.

    ``include_true=True`` (POSITIVE): the true referent's near-miss
    neighbours PLUS the true target itself — the adapter should pick the
    true one. ``include_true=False`` (NEGATIVE): only near-miss
    neighbours that are NOT the true referent — the adapter should
    refuse to splice (leave plain text), because none is correct.

    Near-miss = |ΔN| ≤ 1 on the numeric key (matches pass_9a's gate).
    """
    try:
        n = int(sample.true_key)
    except ValueError:
        n = None
    neighbours: list[str] = []
    if n is not None:
        sib = set(sample.sibling_keys)
        for cand in (str(n - 1), str(n + 1)):
            if cand in sib:
                neighbours.append(cand)
    keys: list[str] = list(neighbours)
    if include_true and sample.true_key not in keys:
        keys.append(sample.true_key)
    keys = sorted(set(keys), key=lambda s: int(s) if s.isdigit() else 0)
    return [
        {
            "key": k,
            "anchor_id": _slug_anchor(sample.reference_kind, k),
            "snippet": k,
        }
        for k in keys
    ]


def _citation_variants(
    sample: CiteSample,
) -> list[tuple[str, dict[str, Any], str]]:
    """Return (variant_label, context, target_html) tuples for a citation.

    Variant 0 (POSITIVE) — candidate_targets includes the true referent;
    target = the restored ``<a href="#anchor">match_text</a>``.

    Variant 1 (NEGATIVE / no-op) — candidate_targets are near-miss-only
    (true referent absent); target = the plain-text match_text. Emitted
    only when at least one near-miss sibling exists (otherwise the
    runtime gate wouldn't fire the gap at all, so a negative row would
    teach an off-distribution case). This is the "leave as plain text"
    discipline from Plans/08 §3 + the no-silent-fallbacks policy.
    """
    out: list[tuple[str, dict[str, Any], str]] = []

    pos_targets = _citation_candidate_targets(sample, include_true=True)
    pos_ctx = {
        "match_text": sample.match_text,
        "surrounding_50_chars": sample.surrounding[:120],
        "candidate_targets": pos_targets,
        "reference_kind": sample.reference_kind,
    }
    pos_target_html = f'<a href="#{sample.anchor_id}">{sample.match_text}</a>'
    out.append(("anchor_restore", pos_ctx, pos_target_html))

    neg_targets = _citation_candidate_targets(sample, include_true=False)
    if neg_targets:
        neg_ctx = {
            "match_text": sample.match_text,
            "surrounding_50_chars": sample.surrounding[:120],
            "candidate_targets": neg_targets,
            "reference_kind": sample.reference_kind,
        }
        # The "leave as plain text" target IS the verbatim match text.
        out.append(("leave_as_plain_text", neg_ctx, sample.match_text))

    return out


def _build_citation_slot(context: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        kind=GapKind.CITATION_UNRESOLVED,
        region_index=0,
        context=context,
    )


def build_citation_rows_for_paper(
    *,
    doc_id: str,
    source_pair: str,
    samples: list[CiteSample],
    skip: SkipCounters,
    max_cites_per_doc: int = 3,
) -> list[dict[str, Any]]:
    """Return citation_unresolved rows for one document.

    ``samples`` are the inline citations extracted from the (CC-cleared)
    source. We cap at ``max_cites_per_doc`` DISTINCT cited numbers per
    document so one heavily-cited paper can't swamp the kind (Plans/08
    §3 balance discipline) and so each doc contributes a small bounded
    row count (mirrors the per-arxiv_id dedup cap on the other kinds).
    """
    rows: list[dict[str, Any]] = []
    if not samples:
        skip.no_citation += 1
        return rows

    # Dedup by (match_text, anchor_id) and cap the distinct count.
    seen: set[tuple[str, str]] = set()
    chosen: list[CiteSample] = []
    for s in samples:
        key = (s.match_text, s.anchor_id)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(s)
        if len(chosen) >= max_cites_per_doc:
            break

    variant_idx = 0
    for s in chosen:
        for variant_label, context, target_html in _citation_variants(s):
            slot = _build_citation_slot(context)
            request_payload = _make_request_payload(slot)
            rows.append(
                {
                    "source_pair": source_pair,
                    "arxiv_id": doc_id,
                    "gap_kind": "citation_unresolved",
                    "variant_idx": variant_idx,
                    "request_payload": request_payload,
                    "target_html": target_html,
                    "n_target_tokens": _ws_token_count(target_html),
                    "extraction_metadata": {
                        "context_extracted_from": "inline_cite_surrounding",
                        "is_paraphrastic": False,
                        "is_negative": variant_label == "leave_as_plain_text",
                        "variant_label": variant_label,
                        "reference_kind": s.reference_kind,
                        "true_anchor_id": s.anchor_id,
                    },
                }
            )
            variant_idx += 1

    return rows


# --------------------------------------------------------------------------- #
# copyright_block / legal_disclaimer extraction (Plans/04 §2; Plans/08 §3)
# --------------------------------------------------------------------------- #
#
# Ground-truth recipe (mechanical, NO LLM labels):
#
#   * PMC JATS ``<permissions>``: ``<copyright-statement>`` (per-article
#     "Copyright: © 2005 X et al." lines) and ``<license>//<license-p>``
#     (CC-BY / public-domain license paragraphs).
#   * PD/CC HTML corpora (gutenberg, courtlistener, cfr,
#     federal_register): block-level text matching a STRICT copyright
#     regex (©, "(c) <year>", "copyright <year>", "all rights
#     reserved") anywhere, or anchored legal-disclaimer phrases within
#     the first/last ``_LEGAL_POSITION_PRIOR`` blocks (the position
#     prior kills mid-opinion body sentences that merely mention
#     "liability"/"license").
#
# The kind split mirrors the runtime detector exactly: a block whose
# text matches ``COPYRIGHT_SUBFLAG_RE`` becomes ``copyright_block``,
# anything else ``legal_disclaimer`` — even for blocks that came from a
# JATS <license> element (e.g. "…is therefore without copyright" is a
# copyright_block at inference time, so it must be one at training
# time too).
#
# License: PMC OA, Project Gutenberg (PD), CourtListener (PD),
# US-gov bulk data (PD). wikipedia (CC-BY-SA) and arXiv are EXCLUDED
# from these kinds (Plans/08 do-NOT list; feedback_license_policy).

# HTML corpora eligible for boilerplate extraction (pmc is handled via
# its JATS permissions path).
LEGAL_HTML_SOURCES = ("gutenberg", "courtlistener", "cfr", "federal_register")

# Position prior for the (weaker) disclaimer-phrase match: boilerplate
# lives in front/back matter, not mid-document.
_LEGAL_POSITION_PRIOR = 15

# Per-distinct-normalized-text occurrence cap. The CC-BY license
# paragraph is byte-identical across 1,000+ PMC articles; unbounded
# emission would turn legal_disclaimer into one template repeated 1K
# times (feedback_balance_dominant_category, intra-kind edition).
_LEGAL_DUP_CAP = 25

# Text-length bands (chars) per kind — keeps targets tiny (p99 well
# inside max_len=512 tokens, Plans/08 §4) and drops fragment noise.
_COPYRIGHT_TEXT_BAND = (15, 400)
_LEGAL_TEXT_BAND = (40, 900)

# STRICT extraction regex for copyright blocks in generic HTML. Tighter
# than the runtime sub-flag (which only SPLITS kinds after BERT said
# "legal"): extraction has no BERT gate, so a bare "copyright" word in
# body prose must not fire — require ©, "(c)"/"copyright" + a plausible
# year (1500-2099), or the full "all rights reserved" phrase.
_COPYRIGHT_EXTRACT_RE = re.compile(
    r"(©|\(c\)\s*(?:1[5-9]|20)\d{2}|\bcopyright\b[\s:]*(?:\(c\)\s*)?(?:1[5-9]|20)\d{2}"
    r"|\ball rights reserved\b)",
    re.IGNORECASE,
)

# Anchored disclaimer/license/terms phrases (position-prior gated).
_LEGAL_PHRASE_RE = re.compile(
    r"(it is prohibited to use"
    r"|without the express,? written permission"
    r"|this is the official u\.s\. government edition"
    r"|herein identified to certify its authenticity"
    r"|no warranty|without warranty"
    r"|disclaims? (?:any|all) (?:liability|warrant)"
    r"|terms of use|terms of service"
    r"|for the use of anyone anywhere"
    r"|public domain in the united states"
    r"|not legal advice"
    r"|no claim to original u\.s\. government works)",
    re.IGNORECASE,
)


def _escape_html_text(s: str) -> str:
    """Minimal HTML text-node escaping for the aside-wrap targets."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _legal_kind_for(text: str) -> GapKind:
    """Split a detected block into the two kinds — mirrors pass_9a."""
    if COPYRIGHT_SUBFLAG_RE.search(text):
        return GapKind.COPYRIGHT_BLOCK
    return GapKind.LEGAL_DISCLAIMER


def _pmc_permission_blocks(root: etree._Element) -> list[str]:
    """Pull <permissions> copyright-statement + license-p texts (JATS)."""
    texts: list[str] = []
    for el in root.findall(".//permissions/copyright-statement"):
        t = _norm_text(" ".join(el.itertext()))
        if t:
            texts.append(t)
    for el in root.findall(".//permissions/license//license-p"):
        t = _norm_text(" ".join(el.itertext()))
        if t:
            texts.append(t)
    return texts


def _html_boilerplate_blocks(root: etree._Element) -> list[str]:
    """Scan block-level elements of an HTML doc for ©/legal boilerplate.

    Strict-copyright matches fire at any position; disclaimer-phrase
    matches only within the first/last ``_LEGAL_POSITION_PRIOR``
    blocks (front/back-matter prior).
    """
    blocks = root.xpath(".//p | .//footer | .//aside | .//small")
    n = len(blocks)
    texts: list[str] = []
    for i, el in enumerate(blocks):
        t = _norm_text(" ".join(el.itertext()))
        if not t:
            continue
        in_boundary = i < _LEGAL_POSITION_PRIOR or i >= n - _LEGAL_POSITION_PRIOR
        if _COPYRIGHT_EXTRACT_RE.search(t):
            texts.append(t)
        elif in_boundary and _LEGAL_PHRASE_RE.search(t):
            texts.append(t)
    return texts


def _build_legal_slot(kind: GapKind, context: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(kind=kind, region_index=0, context=context)


def build_legal_rows_for_paper(
    *,
    doc_id: str,
    source_pair: str,
    texts: list[str],
    skip: SkipCounters,
    dup_counts: Counter,
    origin: str,
) -> list[dict[str, Any]]:
    """Return copyright_block / legal_disclaimer rows for one document.

    At most ONE row per kind per document (first eligible block wins) —
    keeps the per-doc row bound intact. ``dup_counts`` is the build-wide
    Counter of normalized block texts implementing ``_LEGAL_DUP_CAP``.
    """
    rows: list[dict[str, Any]] = []
    if not texts:
        skip.no_legal_block += 1
        return rows

    emitted_kinds: set[GapKind] = set()
    seen_in_doc: set[str] = set()
    for text in texts:
        kind = _legal_kind_for(text)
        if kind in emitted_kinds:
            continue
        lo, hi = _COPYRIGHT_TEXT_BAND if kind is GapKind.COPYRIGHT_BLOCK else _LEGAL_TEXT_BAND
        if not (lo <= len(text) <= hi):
            continue
        norm_key = _norm_text(text).lower()
        if norm_key in seen_in_doc:
            continue
        seen_in_doc.add(norm_key)
        if dup_counts[norm_key] >= _LEGAL_DUP_CAP:
            skip.legal_dup_capped += 1
            continue
        dup_counts[norm_key] += 1

        # Context — EXACTLY the keys the runtime detector (pass_9a)
        # ships and the live prompt builder consumes. The runtime also
        # stashes region_html/semantic_confidence in the GapSlot, but
        # build_gap_fill_request never reads those, so the prompt is
        # byte-identical either way (pinned by the contract test).
        context: dict[str, Any] = {
            "raw_text": text[:1000],
            "surrounding_context": "",
        }
        meta: dict[str, Any] = {
            "context_extracted_from": origin,
            "is_paraphrastic": False,
            "variant_label": "aside_wrap",
        }
        if kind is GapKind.COPYRIGHT_BLOCK:
            aside_class = "copyright"
            context["has_copyright_year_match"] = bool(COPYRIGHT_YEAR_RE.search(text))
            meta["has_copyright_year_match"] = context["has_copyright_year_match"]
        else:
            aside_class = "legal"
            context["legal_subkind"] = classify_legal_subkind(text)
            meta["legal_subkind"] = context["legal_subkind"]

        target_html = f'<aside class="{aside_class}"><p>{_escape_html_text(text)}</p></aside>'
        slot = _build_legal_slot(kind, context)
        rows.append(
            {
                "source_pair": source_pair,
                "arxiv_id": doc_id,
                "gap_kind": kind.value,
                "variant_idx": 0,
                "request_payload": _make_request_payload(slot),
                "target_html": target_html,
                "n_target_tokens": _ws_token_count(target_html),
                "extraction_metadata": meta,
            }
        )
        emitted_kinds.add(kind)
        if len(emitted_kinds) == 2:
            break

    if not rows:
        skip.no_legal_block += 1
    return rows


# --------------------------------------------------------------------------- #
# Variant generators — deterministic paraphrastic gold targets
# --------------------------------------------------------------------------- #


def _missing_title_variants(title_text: str) -> list[tuple[str, str]]:
    """Return list of (variant_label, target_html) pairs.

    Variant 0 is always emitted. Variants 1/2 are emitted only when
    the transformation yields a non-trivial change (not equal to
    variant 0 after whitespace normalisation, and at least one word).
    """
    variants: list[tuple[str, str]] = []

    # Variant 0 — original.
    v0 = f"<title>{title_text}</title><h1>{title_text}</h1>"
    variants.append(("original", v0))

    # Variant 1 — strip a leading boilerplate prefix, OR drop the
    # subtitle (everything after the first colon). We try prefixes
    # first because "An Investigation of: X" is rare; the colon
    # split is a fallback.
    v1_text: str | None = None
    for pat in TITLE_PREFIX_STRIPS:
        m = pat.match(title_text)
        if m:
            stripped = title_text[m.end() :].strip()
            # Capitalise the first letter to keep the title
            # well-formed after the prefix is removed.
            if stripped:
                stripped = stripped[0].upper() + stripped[1:]
                if stripped and stripped.lower() != title_text.lower():
                    v1_text = stripped
                    break
    if v1_text is None and ":" in title_text:
        head = title_text.split(":", 1)[0].strip()
        if head and head.lower() != title_text.lower():
            v1_text = head
    if v1_text is not None and v1_text != title_text:
        v1 = f"<title>{v1_text}</title><h1>{v1_text}</h1>"
        variants.append(("prefix_or_subtitle_strip", v1))

    # Variant 2 — "subtitle-as-title": when the title is "Foo: Bar",
    # emit the subtitle ("Bar") as the title. This teaches the
    # adapter to consider that an alternate canonical phrasing.
    if ":" in title_text:
        tail = title_text.split(":", 1)[1].strip()
        if tail:
            tail = tail[0].upper() + tail[1:]
            already = any(t == tail for _, t in [(_, _v) for _, _v in variants])
            # Compare with previously emitted text portions; we want
            # uniqueness on the inner text, not the full HTML.
            existing_inner = {_extract_inner_h1_text(v[1]) for v in variants}
            if tail not in existing_inner and tail.lower() != title_text.lower():
                v2 = f"<title>{tail}</title><h1>{tail}</h1>"
                variants.append(("subtitle_as_title", v2))

    return variants


def _extract_inner_h1_text(html_frag: str) -> str:
    """Pull the text between <h1>...</h1>; small helper for dedup."""
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html_frag, re.DOTALL | re.IGNORECASE)
    if m is None:
        return ""
    return _norm_text(re.sub(r"<[^>]+>", "", m.group(1)))


def _initialize_first_name(name: str) -> str:
    """ "John Smith" -> "J. Smith"; "J. Smith" -> "J. Smith";
    single-word names ("Plato") are returned unchanged.

    Initialisation is conservative: only the FIRST whitespace-
    separated token is shortened. Two-letter tokens like "Yi" become
    "Y." too — this is the same convention IEEE/Springer use.
    """
    parts = name.split()
    if len(parts) < 2:
        return name
    first = parts[0]
    # Already initialised? ("J." or "J.A." patterns).
    if re.fullmatch(r"[A-Z]\.(?:[A-Z]\.)*", first):
        return name
    # Strip trailing period if any, take first character + period.
    initial = first[0].upper() + "."
    return " ".join([initial] + parts[1:])


def _author_block_variants(
    authors_elem: etree._Element,
    author_names: list[str],
) -> list[tuple[str, str]]:
    """Return list of (variant_label, target_html) for the author block.

    Variant 0 — the literal ``<address>`` wrap of the ar5iv author
    block flattened to text (exactly the shape pass_9a emits when
    the assembler resolves the gap manually).

    Variant 1 — same, but with first names initialised. Skipped if
    the initialised list equals the variant-0 list (e.g. all authors
    are already initialised, or single-word names).
    """
    variants: list[tuple[str, str]] = []

    if not author_names:
        return variants

    # Variant 0 — full names, joined with ", " then " and " for the
    # last separator (the standard list-of-authors convention used
    # in academic papers).
    full = _join_authors(author_names)
    v0 = f"<address>{full}</address>"
    variants.append(("full_names", v0))

    # Variant 1 — initialised first names.
    init_names = [_initialize_first_name(n) for n in author_names]
    if init_names != author_names:
        init = _join_authors(init_names)
        if init != full:
            v1 = f"<address>{init}</address>"
            variants.append(("initialised_first_names", v1))

    return variants


def _join_authors(names: list[str]) -> str:
    """ "X", "Y", "Z" -> "X, Y, and Z"; "X", "Y" -> "X and Y"; "X" -> "X"."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


# --------------------------------------------------------------------------- #
# GapSlot reconstruction + request payload
# --------------------------------------------------------------------------- #


def _build_missing_title_slot(*, first_para: str, doc_lang: str) -> SimpleNamespace:
    """Construct a SimpleNamespace shaped like a GapSlot for the live
    builder.

    ``build_gap_fill_request`` reads ``kind`` (and prefers
    ``kind.value``), ``context`` (a dict), and ``region_index``.
    Other GapSlot fields aren't consumed by the prompt path.
    """
    return SimpleNamespace(
        kind=GapKind.MISSING_TITLE,
        region_index=-1,
        context={
            "first_paragraph_text": first_para[:500],
            "doc_lang": doc_lang,
        },
    )


def _build_author_block_slot(*, raw_text: str, surrounding_context: str) -> SimpleNamespace:
    return SimpleNamespace(
        kind=GapKind.AUTHOR_BLOCK,
        region_index=0,
        context={
            "raw_text": raw_text[:1000],
            "surrounding_context": surrounding_context[:300],
        },
    )


def _make_request_payload(slot: SimpleNamespace) -> dict[str, Any]:
    """Run the LIVE prompt builder; serialize the result as a dict."""
    req = build_gap_fill_request(slot)
    return {
        "adapter": req.adapter.value,
        "payload": req.payload,
        "request_id": req.request_id,
        "sampling_overrides": req.sampling_overrides,
    }


# --------------------------------------------------------------------------- #
# Non-arXiv faithful-title resolver
# --------------------------------------------------------------------------- #

# Project Gutenberg wraps the work title in header boilerplate, e.g.
# "PRIDE. and PREJUDICE — The Project Gutenberg eBook of Pride and
# Prejudice" -> "Pride and Prejudice".
_GUTENBERG_TITLE_RE = re.compile(r"[Tt]he Project Gutenberg eBook of\s+(.+?)\s*$")


def _resolve_nonarxiv_title(d: dict[str, Any]) -> str:
    """Return the faithful document title for a non-arXiv pair.

    These corpora don't keep the title in a usable body ``<h1>``:
    wikipedia stores it only in JSON metadata, gutenberg's ``<h1>`` is
    OCR-noisy, and openstax's JSON ``title`` is a URL slug while its
    ``<h1>`` is clean. We source the cleanest faithful title per source.

    Returns ``""`` to signal "no metadata title — let the caller fall
    back to the ``<h1>`` extraction path" (used for openstax, whose
    ``<h1>`` is the clean title).
    """
    src = (d.get("source") or "").strip()
    if src == "wikipedia":
        return _norm_text(d.get("article_title") or "")
    if src == "courtlistener":
        return _norm_text(d.get("case_name") or "")
    if src == "gutenberg":
        raw = d.get("title") or ""
        m = _GUTENBERG_TITLE_RE.search(raw)
        return _norm_text(m.group(1) if m else raw.split(" — ", 1)[0])
    # openstax (and any other source): defer to the <h1> path.
    return ""


# --------------------------------------------------------------------------- #
# Per-pair row construction
# --------------------------------------------------------------------------- #


def build_rows_for_paper(
    *,
    arxiv_id: str,
    source_pair: str,
    html_str: str,
    skip: SkipCounters,
    title_override: str | None = None,
    variant0_only: bool = False,
    emit_authors: bool = True,
    title_source_label: str = "ltx_title_document",
) -> list[dict[str, Any]]:
    """Return all (gap_kind, variant) rows for one paper.

    Returns an empty list if neither the title nor the author block
    can be extracted.

    Non-arXiv HTML corpora pass ``title_override`` (the faithful title
    from per-corpus metadata), ``variant0_only=True`` (skip the
    academic-tuned paraphrase variants), and ``emit_authors=False``
    (these sources carry no ar5iv-shaped author block). ``first_para``
    and ``doc_lang`` are still read from the HTML so the prompt envelope
    is byte-for-byte identical to the arXiv path.
    """
    try:
        root = lxml_html.fromstring(html_str)
    except Exception:
        skip.parse_error += 1
        return []

    rows: list[dict[str, Any]] = []
    doc_lang = _extract_doc_lang(root)
    first_para = _extract_first_paragraph(root)

    # ---- missing_title rows -------------------------------------------------
    if title_override is not None:
        title_text = _norm_text(title_override)
    else:
        title_elem = _extract_title_element(root)
        title_text = _extract_title_text(title_elem) if title_elem is not None else ""
    if not title_text:
        skip.no_title += 1
    elif len(title_text) < 4:
        # Single-character "h1" cells in malformed pages — refuse.
        skip.title_too_short += 1
    else:
        slot = _build_missing_title_slot(first_para=first_para, doc_lang=doc_lang)
        request_payload = _make_request_payload(slot)
        variants = _missing_title_variants(title_text)
        if variant0_only:
            variants = variants[:1]
        for variant_idx, (variant_label, target_html) in enumerate(variants):
            rows.append(
                {
                    "source_pair": source_pair,
                    "arxiv_id": arxiv_id,
                    "gap_kind": "missing_title",
                    "variant_idx": variant_idx,
                    "request_payload": request_payload,
                    "target_html": target_html,
                    "n_target_tokens": _ws_token_count(target_html),
                    "extraction_metadata": {
                        "context_extracted_from": "first_paragraph",
                        "is_paraphrastic": variant_idx > 0,
                        "variant_label": variant_label,
                        "doc_lang": doc_lang,
                        "title_source": title_source_label,
                    },
                }
            )

    # ---- author_block rows --------------------------------------------------
    authors_elem = _extract_author_block(root) if emit_authors else None
    if authors_elem is None:
        skip.no_author += 1
    else:
        author_names = _extract_author_names(authors_elem)
        if not author_names:
            skip.no_author += 1
        else:
            raw_text = _norm_text(" ".join(authors_elem.itertext()))
            if len(raw_text) < 3:
                skip.author_too_short += 1
            else:
                # surrounding_context: the title text + first paragraph
                # (gives the adapter enough hint to pick formatting that
                # matches the paper's style).
                surrounding = f"Title: {title_text}\n{first_para}" if title_text else first_para
                slot = _build_author_block_slot(
                    raw_text=raw_text,
                    surrounding_context=surrounding,
                )
                request_payload = _make_request_payload(slot)
                for variant_idx, (variant_label, target_html) in enumerate(
                    _author_block_variants(authors_elem, author_names)
                ):
                    rows.append(
                        {
                            "source_pair": source_pair,
                            "arxiv_id": arxiv_id,
                            "gap_kind": "author_block",
                            "variant_idx": variant_idx,
                            "request_payload": request_payload,
                            "target_html": target_html,
                            "n_target_tokens": _ws_token_count(target_html),
                            "extraction_metadata": {
                                "context_extracted_from": "title_plus_first_paragraph",
                                "is_paraphrastic": variant_idx > 0,
                                "variant_label": variant_label,
                                "n_authors": len(author_names),
                                "doc_lang": doc_lang,
                            },
                        }
                    )

    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def iter_pair_files(pairs_dir: Path) -> Iterator[Path]:
    yield from sorted(pairs_dir.glob("*.json"))


def _apply_per_kind_cap(
    rows: list[dict[str, Any]],
    *,
    cap_per_kind: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Downsample dominant gap kinds to a per-kind row cap.

    Cap selection (see module docstring "Per-kind balance cap"):
        cap_per_kind  < 0 — AUTO (default): cap = round(1.5 * row count
                            of the SECOND-largest kind); only kinds
                            exceeding the cap are downsampled.
        cap_per_kind == 0 — capping disabled; rows returned unchanged.
        cap_per_kind  > 0 — explicit cap applied to every kind above it.

    Sampling is deterministic (seeded) and drops whole
    ``(arxiv_id, gap_kind)`` groups — never individual variants — so
    ``variant_idx`` runs stay contiguous, and it is stratified per
    split with proportional quotas so the kind's train/val/test shares
    are preserved. Within a split, group ids are sorted then shuffled
    with ``random.Random(f"{seed}:{kind}:{split}")`` and taken greedily
    until the next group would overflow the split quota (undershoot is
    bounded by one group, i.e. a handful of rows).

    Returns ``(kept_rows, cap_info)`` where ``cap_info`` is the
    coverage-report block describing what was done. Row order is
    preserved.
    """
    uncapped = Counter(r["gap_kind"] for r in rows)
    info: dict[str, Any] = {
        "mode": ("disabled" if cap_per_kind == 0 else "explicit" if cap_per_kind > 0 else "auto"),
        "rule": (
            "auto (-1): cap = round(1.5 * next-largest kind's rows); "
            "0 disables; N>0 explicit. Deterministic seeded sampling of "
            "whole (arxiv_id, gap_kind) groups, stratified per split."
        ),
        "requested_cap_per_kind": cap_per_kind,
        "cap_value": None,
        "uncapped_rows_per_gap_kind": dict(uncapped),
        "rows_dropped_by_cap": {},
    }
    if cap_per_kind == 0 or len(uncapped) < 2:
        return rows, info

    if cap_per_kind > 0:
        cap = cap_per_kind
    else:
        cap = round(1.5 * uncapped.most_common(2)[1][1])
    info["cap_value"] = cap

    over_kinds = sorted(k for k, n in uncapped.items() if n > cap)
    if not over_kinds:
        return rows, info

    drop_keys: set[tuple[str, str]] = set()
    for kind in over_kinds:
        # split -> {arxiv_id -> n_rows} for this kind. A doc lives in
        # exactly one split (split is a hash of the doc id).
        group_sizes: dict[str, dict[str, int]] = {}
        for r in rows:
            if r["gap_kind"] == kind:
                g = group_sizes.setdefault(r["split"], {})
                g[r["arxiv_id"]] = g.get(r["arxiv_id"], 0) + 1
        split_totals = {s: sum(g.values()) for s, g in group_sizes.items()}
        kind_total = sum(split_totals.values())
        for split in sorted(group_sizes):
            quota = int(round(cap * split_totals[split] / kind_total))
            ids = sorted(group_sizes[split])
            random.Random(f"{seed}:{kind}:{split}").shuffle(ids)
            kept = 0
            cut = len(ids)
            for j, aid in enumerate(ids):
                size = group_sizes[split][aid]
                if kept + size > quota:
                    cut = j
                    break
                kept += size
            for aid in ids[cut:]:
                drop_keys.add((aid, kind))

    kept_rows = [r for r in rows if (r["arxiv_id"], r["gap_kind"]) not in drop_keys]
    capped = Counter(r["gap_kind"] for r in kept_rows)
    info["rows_dropped_by_cap"] = {k: uncapped[k] - capped.get(k, 0) for k in over_kinds}
    return kept_rows, info


def _open_split_files(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train": (out_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "test.jsonl").open("w", encoding="utf-8"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_PAIRS_DIR],
        help="One or more pair dirs. arXiv pairs (those with an "
        "arxiv_id) get the full variant+author treatment "
        "from ar5iv <h1>/ltx_authors. Non-arXiv HTML corpora "
        "(wikipedia/openstax/courtlistener/gutenberg) emit a "
        "single faithful missing_title row sourced from "
        "per-corpus metadata; no author rows, no paraphrases.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Stop after this many pair files (0 = all). Useful for smoke tests.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cap-per-kind",
        type=int,
        default=-1,
        help="Per-kind row cap (Plans/08 §3 balance discipline). "
        "-1 (default) = AUTO: cap at round(1.5 x the next-largest "
        "kind's row count), computed from the extracted data; "
        "0 = disable capping; N>0 = explicit cap. Deterministic "
        "seeded (arxiv_id, gap_kind) group-level downsampling, "
        "stratified per split (see module docstring).",
    )
    ap.add_argument(
        "--progress-every", type=int, default=500, help="Print a progress line every N pair files."
    )
    args = ap.parse_args()

    for pd in args.pairs_dir:
        if not pd.exists():
            sys.exit(f"[fatal] pairs dir not found: {pd}")

    stats = BuildStats()

    seen_arxiv_ids: dict[str, str] = {}  # id -> split
    all_rows: list[dict[str, Any]] = []
    # Build-wide per-distinct-text occurrence counter for the
    # copyright/legal boilerplate dedup cap (_LEGAL_DUP_CAP).
    legal_dup_counts: Counter = Counter()

    all_pairs = [p for pd in args.pairs_dir for p in iter_pair_files(pd)]
    for i, pair_path in enumerate(all_pairs):
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
        pmcid = d.get("pmcid")
        html_str = d.get("raw_source_html")
        xml_str = d.get("raw_source_xml")
        # PMC pairs carry JATS XML (raw_source_xml) + a pmcid instead
        # of HTML. They feed the citation_unresolved kind (JATS xref)
        # and the copyright/legal kinds (JATS <permissions>); they
        # carry no ar5iv <h1>/ltx_authors so no title/author rows.
        is_pmc = bool(pmcid) and not html_str
        src = (d.get("source") or "").strip()
        output_html = d.get("output_html")
        # Gov XML corpora (cfr / federal_register) carry no
        # raw_source_html — they are legal-boilerplate-only sources and
        # are scanned via their ground-truth output_html below.
        is_legal_only = (
            not html_str and not is_pmc and src in LEGAL_HTML_SOURCES and bool(output_html)
        )
        if not html_str and not (is_pmc and xml_str) and not is_legal_only:
            stats.skip.no_html += 1
            continue

        # arXiv pairs carry an arxiv_id and are deduped per paper (the
        # same paper appears once per section but with identical
        # raw_source_html). Non-arXiv HTML corpora have no arxiv_id; we
        # key dedup/split on the pair filename (each file is already one
        # document) and reuse the ``arxiv_id`` row field to carry that
        # id so the split-hash contract test stays valid.
        is_arxiv = bool(arxiv_id)
        if is_pmc:
            doc_id = pmcid
        elif is_arxiv:
            doc_id = arxiv_id
        else:
            doc_id = pair_path.stem

        if doc_id in seen_arxiv_ids:
            continue

        split = stable_split_for_id(doc_id, 0.80, 0.10)
        seen_arxiv_ids[doc_id] = split

        source_pair = pair_path.stem
        if is_pmc:
            # PMC: citation rows (JATS xref) + copyright/legal rows
            # (JATS <permissions>). No title/author.
            legal_texts: list[str] = []
            try:
                pmc_root = etree.fromstring(xml_str.encode("utf-8"))
                cite_samples = _pmc_citations(pmc_root)
                legal_texts = _pmc_permission_blocks(pmc_root)
            except Exception:
                stats.skip.parse_error += 1
                cite_samples = []
            rows = build_citation_rows_for_paper(
                doc_id=doc_id,
                source_pair=source_pair,
                samples=cite_samples,
                skip=stats.skip,
            )
            rows += build_legal_rows_for_paper(
                doc_id=doc_id,
                source_pair=source_pair,
                texts=legal_texts,
                skip=stats.skip,
                dup_counts=legal_dup_counts,
                origin="pmc_permissions",
            )
        elif is_arxiv:
            rows = build_rows_for_paper(
                arxiv_id=doc_id,
                source_pair=source_pair,
                html_str=html_str,
                skip=stats.skip,
            )
            # ar5iv citation_unresolved rows (CC-cleared subset).
            try:
                ar5iv_root = lxml_html.fromstring(html_str)
                cite_samples = _ar5iv_citations(ar5iv_root)
            except Exception:
                cite_samples = []
            rows += build_citation_rows_for_paper(
                doc_id=doc_id,
                source_pair=source_pair,
                samples=cite_samples,
                skip=stats.skip,
            )
        else:
            # Non-arXiv HTML: faithful title from metadata (empty ->
            # fall back to the <h1> path, used for openstax), variant-0
            # only, no author block, no citation rows (these corpora
            # carry no clean inline-cite->reference link structure).
            # Legal-only XML corpora (cfr / federal_register) skip the
            # title path entirely.
            rows = []
            if html_str:
                title_override = _resolve_nonarxiv_title(d)
                rows = build_rows_for_paper(
                    arxiv_id=doc_id,
                    source_pair=source_pair,
                    html_str=html_str,
                    skip=stats.skip,
                    title_override=(title_override or None),
                    variant0_only=True,
                    emit_authors=False,
                    title_source_label=(f"metadata:{src}" if title_override else "h1"),
                )
            # copyright_block / legal_disclaimer boilerplate from the
            # PD/CC allowlisted sources only (wikipedia is CC-BY-SA —
            # excluded; openstax matches were body-prose false
            # positives — excluded). Both the ground-truth output_html
            # and the raw_source_html are scanned (gutenberg keeps its
            # © pages only in the raw PG html); the per-doc dedup in
            # build_legal_rows_for_paper collapses the overlap.
            if src in LEGAL_HTML_SOURCES:
                legal_texts = []
                for html_field in (output_html, html_str):
                    if not html_field:
                        continue
                    try:
                        legal_root = lxml_html.fromstring(html_field)
                    except Exception:
                        continue
                    legal_texts += _html_boilerplate_blocks(legal_root)
                rows += build_legal_rows_for_paper(
                    doc_id=doc_id,
                    source_pair=source_pair,
                    texts=legal_texts,
                    skip=stats.skip,
                    dup_counts=legal_dup_counts,
                    origin="html_boilerplate",
                )
        for row in rows:
            row["split"] = split
            all_rows.append(row)

        if (i + 1) % args.progress_every == 0:
            print(
                f"[progress] pairs={i + 1} "
                f"unique_ids={len(seen_arxiv_ids)} "
                f"rows={len(all_rows)} "
                f"skips(no_title={stats.skip.no_title} "
                f"no_author={stats.skip.no_author})",
                flush=True,
            )

    stats.n_unique_arxiv_ids = len(seen_arxiv_ids)

    # ------------------------------------------------------------------
    # Per-kind balance cap (Plans/08 §3) — then write the splits.
    # ------------------------------------------------------------------
    n_uncapped = len(all_rows)
    all_rows, cap_info = _apply_per_kind_cap(
        all_rows, cap_per_kind=args.cap_per_kind, seed=args.seed
    )
    if cap_info["cap_value"] is not None:
        print(
            f"[cap] mode={cap_info['mode']} cap_value={cap_info['cap_value']} "
            f"dropped={cap_info['rows_dropped_by_cap']} "
            f"rows {n_uncapped} -> {len(all_rows)}",
            flush=True,
        )

    handles = _open_split_files(args.out_dir)
    try:
        for row in all_rows:
            split = row["split"]
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.n_rows_emitted += 1
            stats.rows_per_split[split] += 1
            stats.rows_per_gap_kind[row["gap_kind"]] += 1
            stats.rows_per_variant[row["variant_idx"]] += 1
            stats.rows_per_arxiv_id[row["arxiv_id"]] += 1
            stats.tgt_token_hist[_bucket_for(row["n_target_tokens"], TOKEN_HIST_EDGES)] += 1
    finally:
        for h in handles.values():
            h.close()

    # ------------------------------------------------------------------
    # Coverage report.
    # ------------------------------------------------------------------
    coverage = {
        "schema_version": 1,
        "seed": args.seed,
        "pairs_dir": [str(p) for p in args.pairs_dir],
        "out_dir": str(args.out_dir),
        "n_pairs_seen": stats.n_pairs_seen,
        "n_unique_arxiv_ids_processed": stats.n_unique_arxiv_ids,
        "n_unique_arxiv_ids_in_dataset": len(stats.rows_per_arxiv_id),
        "n_rows_emitted": stats.n_rows_emitted,
        "rows_per_split": dict(stats.rows_per_split),
        "rows_per_gap_kind": dict(stats.rows_per_gap_kind),
        "cap_per_kind": cap_info,
        "rows_per_variant": {str(k): v for k, v in sorted(stats.rows_per_variant.items())},
        "tgt_token_hist": dict(sorted(stats.tgt_token_hist.items())),
        "rows_per_arxiv_id_top20": dict(stats.rows_per_arxiv_id.most_common(20)),
        "skip": {
            "no_html": stats.skip.no_html,
            "no_arxiv_id": stats.skip.no_arxiv_id,
            "no_title": stats.skip.no_title,
            "no_author": stats.skip.no_author,
            "title_too_short": stats.skip.title_too_short,
            "author_too_short": stats.skip.author_too_short,
            "no_citation": stats.skip.no_citation,
            "no_legal_block": stats.skip.no_legal_block,
            "legal_dup_capped": stats.skip.legal_dup_capped,
            "parse_error": stats.skip.parse_error,
        },
        "notes": {
            "scope": (
                "Emits all 5 GapKinds: missing_title + author_block (from "
                "ar5iv), citation_unresolved (from ar5iv ltx_ref bib links "
                "+ PMC JATS bibr xref), and copyright_block + "
                "legal_disclaimer (from PMC JATS <permissions> + "
                "PD/CC-source HTML boilerplate: gutenberg, courtlistener, "
                "cfr, federal_register; wikipedia/arXiv excluded — "
                "CC-BY-SA / license policy). citation_unresolved rows "
                "come in two variants per cite: variant 'anchor_restore' "
                "(POSITIVE — candidate_targets includes the true referent, "
                "target = the restored <a href='#...'>); variant "
                "'leave_as_plain_text' (NEGATIVE — near-miss-only "
                "candidate_targets, target = the verbatim plain-text span) "
                "so the adapter learns not to hallucinate anchors. "
                "copyright/legal targets are the canonical <aside "
                "class='copyright|legal'><p>...</p></aside> wrap of the "
                "verbatim block text; max 1 row per kind per doc; "
                "build-wide per-distinct-text occurrence cap "
                f"({_LEGAL_DUP_CAP}) stops byte-identical boilerplate "
                "(the CC-BY license paragraph) from swamping the kind."
            ),
            "split_function": (
                "data._splits.stable_split_for_id, train_frac=0.80, "
                "val_frac=0.10. Same arxiv_id -> same split as the "
                "math/prose/table/semantic_preservation builders."
            ),
            "variant_strategy": (
                "K=8 diversity training. Variant 0 is always the "
                "original ground truth. Variant 1 is rule-based: "
                "missing_title strips a leading boilerplate prefix or "
                "drops the subtitle after a colon; author_block "
                "initialises first names. Variant 2 (missing_title "
                "only) flips to subtitle-as-title when the original "
                "had a colon. Variants are skipped when they produce "
                "no change vs. variant 0."
            ),
            "ar5iv_quirks": (
                "Author names are extracted from "
                "<span class='ltx_personname'> children of "
                "<div class='ltx_authors'>; nested ltx_note / "
                "ltx_role_footnote / ltx_author_notes / ltx_contact / "
                "ltx_ERROR / <sup> subtrees are stripped before name "
                "flattening. We then split on ' and ' / ', ' (when "
                "the next token starts with a capital) for the cases "
                "where a single personname span lists multiple authors."
            ),
            "target_html_format": (
                "missing_title targets are <title>X</title><h1>X</h1>. "
                "author_block targets are <address>...</address> with "
                "the joined name list as inner text."
            ),
            "cap_per_kind": (
                "Plans/08 §3 balance discipline / "
                "feedback_balance_dominant_category: dominant kinds are "
                "downsampled to --cap-per-kind rows (default auto = "
                "round(1.5 x next-largest kind)). Deterministic seeded "
                "sampling of whole (arxiv_id, gap_kind) groups, "
                "stratified per split; variant_idx runs stay contiguous. "
                "See the 'cap_per_kind' block above for what this build "
                "actually dropped."
            ),
        },
    }
    (args.out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))

    # Summary to stdout.
    print()
    print(
        f"[done] rows={stats.n_rows_emitted} "
        f"(train={stats.rows_per_split.get('train', 0)} "
        f"val={stats.rows_per_split.get('val', 0)} "
        f"test={stats.rows_per_split.get('test', 0)})"
    )
    print(
        f"[done] gap_kind={dict(stats.rows_per_gap_kind)} variants={dict(stats.rows_per_variant)}"
    )
    print(
        f"[done] unique_ids={stats.n_unique_arxiv_ids} "
        f"skip(no_title={stats.skip.no_title} "
        f"no_author={stats.skip.no_author} "
        f"parse_err={stats.skip.parse_error})"
    )
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
