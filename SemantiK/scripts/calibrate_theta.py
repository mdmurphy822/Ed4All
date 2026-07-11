#!/usr/bin/env python
"""Theta calibration harness (Plan 12 A3, Phase V2 item 8).

WHY this exists: theta's composite weights were a uniform-1/8
placeholder and the exit thresholds (TAU_THETA_CONFIDENCE /
TAU_THETA_RETRY) plus per-dimension floors were never tuned on held-out
data — so the score the pipeline ships was not yet meaningful as a
quality claim (Plan 12 blocker A3). This script builds a labeled
perturbation set from WCAG-clean pair HTML, scores every variant
through the REAL 8 theta dimensions (including the v8
semantic-preservation cross-encoder, forced onto CPU), fits
per-dimension weights that separate clean from degraded variants, picks
thresholds/floors from the fitted composite distribution, and writes a
provenance-carrying report. ``--write-config`` regenerates
``semantik_structure/theta/config.yaml`` (theta-config-2.0) from the fit.

Perturbation taxonomy (adapted from the document-level analogues of
``scripts/build_semantic_preservation_dataset.py`` — sentence
splitting, donor-pool hallucination injection, shuffle/drop machinery):

* ``shuffled_sections`` — section groups reordered while
  ``region_provenance`` keeps claiming the original order (models a
  stage-2 reading-order misread: provenance is trusted but wrong).
* ``dropped_headings`` — h2..h6 flattened to ``<p>`` (mild: ~40%,
  severe: all); heading tree, ToC resolution, section preservation
  degrade.
* ``broken_refs`` — nav/ToC hrefs pointed at missing ids (severe also
  strips ids from half the surviving headings).
* ``hallucinated_content`` — gap-fill provenance (``resolved_html``)
  filled with paragraphs drawn from a DIFFERENT source document
  (out-of-vocabulary w.r.t. this doc's anchor pool); severe also
  splices donor sentences into emitted paragraph regions.
* ``fragmentation`` — paragraphs split at sentence boundaries (mild:
  ~40% split in two, severe: every paragraph one-sentence-per-<p>).
* ``meaning_distortion`` — emitted paragraph text damaged while the
  structure stays intact: cross-document sentence swaps (the builder's
  ``perturb_semantic_drift``; mild = fewer paragraphs / fewer swapped
  sentences). This is the family only the cross-encoder can see —
  without it the fit is free to zero out ``semantic_preservation``.
  (Word-shuffle was probed and dropped: the cls-pooled v8 encoder is
  order-insensitive, see make_meaning_distortion.)

Hard requirements honored here:

* CPU-only — ``CUDA_VISIBLE_DEVICES=""`` and ``DART_THETA_DEVICE=cpu``
  are forced before any torch import (another project owns the GPU).
* No silent 7-dim calibration — ``DART_ALLOW_THETA_STUB`` is scrubbed
  from the environment, so a missing/broken v8 cross-encoder raises
  instead of calibrating against the 0.7 stub
  (feedback_no_silent_fallbacks).
* No new dependencies — fitting is a small class-balanced logistic
  regression in numpy (scikit-learn is not in uv.lock).

Usage:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/calibrate_theta.py \
        --n 80 --write-config
"""

from __future__ import annotations

import os

# CPU enforcement MUST precede any torch import (transitively via
# semantik_structure.theta). The GPU is off-limits during calibration runs
# (train CUDA-context guard) and the v8 cross-encoder runs fine on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DART_THETA_DEVICE"] = "cpu"
# Strict-by-default: never calibrate against the 0.7 stub. If the v8
# model can't load, evaluate() must raise, not degrade to 7 real dims.
os.environ.pop("DART_ALLOW_THETA_STUB", None)

import argparse
import html as html_mod
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from lxml import html as lxml_html

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from semantik_structure.assembler.types import AssembledDoc, GapKind, GapSlot  # noqa: E402
from semantik_structure.theta.evaluator import evaluate  # noqa: E402
from semantik_structure.theta.types import _WEIGHT_KEYS  # noqa: E402
from semantik_structure.types import FeatureBlock, RawBlock  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM_ORDER: tuple[str, ...] = tuple(_WEIGHT_KEYS)

#: Pair sources to stratify over (subdirectories of data/pairs plus the
#: top-level markdown__*.json wiki-markdown set). *_smoke dirs excluded.
PAIR_SOURCES: tuple[str, ...] = (
    "arxiv",
    "pmc",
    "wikipedia",
    "openstax",
    "courtlistener",
    "nces_digest",
    "cfr",
    "gutenberg",
    "synthetic_blockquote_code",
    "forms",
    "federal_register",
    "markdown",
)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“(])")
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "blockquote")

#: Minimum doc shape for a useful variant family: enough headings for
#: shuffle/drop/ToC signal and enough prose for the cross-encoder.
MIN_HEADINGS = 2
MIN_PARAGRAPHS = 4
MIN_DOC_WORDS = 200

PERTURBATIONS: tuple[str, ...] = (
    "shuffled_sections",
    "dropped_headings",
    "broken_refs",
    "hallucinated_content",
    "fragmentation",
    "meaning_distortion",
)
SEVERITIES: tuple[str, ...] = ("mild", "severe")

#: Architecture §6.2 v1 floors — retained when the clean distribution
#: for a dimension is degenerate (zero variance on the synthetic set, so
#: a percentile would be fake calibration, not measurement).
PRIOR_FLOORS: dict[str, float] = {
    "reference_integrity": 0.80,
    "hallucinated_structure_penalty": 0.85,
    "semantic_preservation": 0.65,
}

#: Threshold clamp bands. WHY: the synthetic clean fixtures are
#: assembled by construction (every id resolves, provenance is exact),
#: so their composites sit far above real pipeline outputs (R10
#: real-runtime theta averaged 0.721 under uniform weights). Shipping
#: the raw rule output (~0.9+) would route essentially every real doc
#: into the offline-retry lane — a retry storm, not calibration. The
#: clamps keep the thresholds inside the band the architecture's exit
#: table was designed around while the raw rule values stay recorded in
#: the report; revisit once Plan 12 C1 scales the real-corpus eval.
TAU_RETRY_CLAMP: tuple[float, float] = (0.55, 0.75)
TAU_CONF_CLAMP: tuple[float, float] = (0.65, 0.85)

#: Floor clamps — "clamped sensibly" per Plan 12 A3. The
#: semantic-preservation cap of 0.70 keeps the documented stub value
#: (0.7, dev/eval only) exactly non-breaching and matches the v8
#: calibrated-floor band (clean ≈ 0.77, see _module_state docstring);
#: the 0.70 lower bounds for the deterministic dims stop a fat-tailed
#: clean p2 from neutering the flag entirely.
FLOOR_CLAMPS: dict[str, tuple[float, float]] = {
    "reference_integrity": (0.70, 0.95),
    "hallucinated_structure_penalty": (0.70, 0.95),
    "semantic_preservation": (0.55, 0.70),
}


class CalibrationError(RuntimeError):
    """Loud failure for any condition that would corrupt the calibration."""


# ---------------------------------------------------------------------------
# Document loading (data/pairs/*.json -> block lists)
# ---------------------------------------------------------------------------


@dataclass
class CalBlock:
    """One emitted block of the synthetic 'assembled' document."""

    tag: str  # h1..h6 | p | ul | ol | blockquote
    text: str  # normalized visible text
    items: list[str] = field(default_factory=list)  # li texts for ul/ol

    @property
    def is_heading(self) -> bool:
        return self.tag in HEADING_TAGS

    @property
    def level(self) -> int:
        return int(self.tag[1]) if self.is_heading else 0


@dataclass
class CalDoc:
    doc_id: str
    source: str
    blocks: list[CalBlock]

    @property
    def n_headings(self) -> int:
        return sum(1 for b in self.blocks if b.is_heading)

    @property
    def n_paragraphs(self) -> int:
        return sum(1 for b in self.blocks if b.tag == "p")

    @property
    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def split_sentences(text: str) -> list[str]:
    """Sentence split — same heuristic family as
    build_semantic_preservation_dataset.split_sentences."""
    out = SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in out if s.strip()]


def extract_blocks(raw_html: str, max_blocks: int) -> list[CalBlock]:
    """Ordered top-level content blocks from a pair's ``output_html``.

    Mirrors build_semantic_preservation_dataset.extract_regions but
    keeps DOCUMENT-level ordering (calibration perturbs whole docs, not
    isolated regions). Nested matches (p inside blockquote/li) are
    skipped; tables are dropped — re-synthesizing faithful table HTML
    from text cells would add a confound none of the 8 dims needs.
    """
    try:
        tree = lxml_html.fromstring(raw_html)
    except Exception:  # noqa: BLE001 — caller skips unparseable docs
        return []
    for bad in tree.xpath("//script | //style | //noscript | //nav"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)

    selector = " | ".join(f"//{t}" for t in BLOCK_TAGS)
    blocks: list[CalBlock] = []
    for el in tree.xpath(selector):
        # Skip nested hits — the outermost block owns the text.
        anc = el.getparent()
        nested = False
        while anc is not None:
            if anc.tag in BLOCK_TAGS:
                nested = True
                break
            anc = anc.getparent()
        if nested:
            continue
        tag = str(el.tag)
        if tag in ("ul", "ol"):
            items = [_norm_text(li.text_content()) for li in el.xpath("./li")]
            items = [i for i in items if i]
            if not items:
                continue
            blocks.append(CalBlock(tag=tag, text=" ".join(items), items=items))
        else:
            text = _norm_text(el.text_content())
            if not text:
                continue
            blocks.append(CalBlock(tag=tag, text=text))
        if len(blocks) >= max_blocks:
            break
    return blocks


def _iter_pair_files(pairs_root: Path, source: str) -> list[Path]:
    if source == "markdown":
        return sorted(pairs_root.glob("markdown__*.json"))
    d = pairs_root / source
    if not d.is_dir():
        return []
    return sorted(d.rglob("*.json"))


def load_docs(pairs_root: Path, n: int, max_blocks: int, rng: random.Random) -> list[CalDoc]:
    """Stratified (round-robin across sources) load of N usable docs."""
    per_source: dict[str, list[Path]] = {}
    for src in PAIR_SOURCES:
        files = _iter_pair_files(pairs_root, src)
        rng.shuffle(files)
        if files:
            per_source[src] = files
    if not per_source:
        raise CalibrationError(f"no pair files found under {pairs_root}")

    docs: list[CalDoc] = []
    cursors = dict.fromkeys(per_source, 0)
    while len(docs) < n:
        progressed = False
        for src, files in per_source.items():
            if len(docs) >= n:
                break
            i = cursors[src]
            while i < len(files):
                path = files[i]
                i += 1
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                raw = payload.get("output_html")
                if not raw or len(raw) < 500:
                    continue
                blocks = extract_blocks(raw, max_blocks)
                doc = CalDoc(doc_id=f"{src}/{path.stem}", source=src, blocks=blocks)
                if (
                    doc.n_headings < MIN_HEADINGS
                    or doc.n_paragraphs < MIN_PARAGRAPHS
                    or doc.word_count < MIN_DOC_WORDS
                ):
                    continue
                docs.append(doc)
                progressed = True
                break
            cursors[src] = i
        if not progressed:
            break  # every source exhausted
    if len(docs) < min(n, max(8, n // 2)):
        raise CalibrationError(
            f"only {len(docs)} usable docs found (asked for {n}) — refusing to "
            f"calibrate on a corpus this thin; check --pairs-root"
        )
    return docs


# ---------------------------------------------------------------------------
# Variant synthesis
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    perturbation: str  # "clean" | one of PERTURBATIONS
    severity: str  # "none" | "mild" | "severe"
    block_htmls: list[str]  # per-region emitted HTML (index == provenance)
    heading_tree: list[tuple[int, str]]
    nav_html: str
    extra_html: str = ""  # appended inside <main> after the regions
    gaps_resolved: list[GapSlot] = field(default_factory=list)

    @property
    def label_clean(self) -> bool:
        return self.perturbation == "clean"


def _esc(s: str) -> str:
    return html_mod.escape(s, quote=False)


def _block_html(block: CalBlock, idx: int, *, with_id: bool = True) -> str:
    if block.is_heading:
        id_attr = f' id="sec-{idx}"' if with_id else ""
        return f"<{block.tag}{id_attr}>{_esc(block.text)}</{block.tag}>"
    if block.tag in ("ul", "ol"):
        lis = "".join(f"<li>{_esc(i)}</li>" for i in block.items)
        return f"<{block.tag}>{lis}</{block.tag}>"
    if block.tag == "blockquote":
        return f"<blockquote><p>{_esc(block.text)}</p></blockquote>"
    return f"<p>{_esc(block.text)}</p>"


def _nav_html(blocks: list[CalBlock], broken: set[int] | None = None) -> str:
    """ToC <nav> over heading ids. ``broken`` heading indices point at
    missing ids (the broken_refs perturbation)."""
    broken = broken or set()
    links = []
    for i, b in enumerate(blocks):
        if not b.is_heading:
            continue
        target = f"#missing-{i}" if i in broken else f"#sec-{i}"
        links.append(f'<li><a href="{target}">{_esc(b.text[:60])}</a></li>')
        if len(links) >= 10:
            break
    if not links:
        return ""
    return f'<nav aria-label="Contents"><ul>{"".join(links)}</ul></nav>'


def _heading_tree(blocks: list[CalBlock], order: list[int], dropped: set[int]) -> list:
    tree = []
    for i in order:
        b = blocks[i]
        if b.is_heading and i not in dropped:
            tree.append((b.level, b.text))
    return tree


def _sections(blocks: list[CalBlock]) -> list[list[int]]:
    """Group block indices into heading-anchored sections (preamble first)."""
    sections: list[list[int]] = []
    current: list[int] = []
    for i, b in enumerate(blocks):
        if b.is_heading and current:
            sections.append(current)
            current = []
        current.append(i)
    if current:
        sections.append(current)
    return sections


def make_clean(doc: CalDoc) -> Variant:
    n = len(doc.blocks)
    return Variant(
        perturbation="clean",
        severity="none",
        block_htmls=[_block_html(b, i) for i, b in enumerate(doc.blocks)],
        heading_tree=_heading_tree(doc.blocks, list(range(n)), set()),
        nav_html=_nav_html(doc.blocks),
    )


def make_shuffled_sections(doc: CalDoc, severity: str, rng: random.Random) -> Variant | None:
    sections = _sections(doc.blocks)
    if len(sections) < 2:
        return None
    order = list(range(len(sections)))
    if severity == "mild":
        i = rng.randrange(len(order) - 1)
        order[i], order[i + 1] = order[i + 1], order[i]
    else:
        for _ in range(10):
            rng.shuffle(order)
            if order != sorted(order):
                break
    flat = [bi for si in order for bi in sections[si]]
    # Provenance stays 0..n-1 (the pipeline TRUSTS its order) while the
    # emitted content at position i comes from the shuffled stream —
    # this is what a stage-2 reading-order misread looks like downstream.
    block_htmls = [_block_html(doc.blocks[flat[i]], flat[i]) for i in range(len(flat))]
    return Variant(
        perturbation="shuffled_sections",
        severity=severity,
        block_htmls=block_htmls,
        heading_tree=_heading_tree(doc.blocks, flat, set()),
        nav_html=_nav_html(doc.blocks),
    )


def make_dropped_headings(doc: CalDoc, severity: str, rng: random.Random) -> Variant | None:
    droppable = [i for i, b in enumerate(doc.blocks) if b.is_heading and b.level >= 2]
    if not droppable:
        return None
    if severity == "mild":
        k = max(1, round(0.4 * len(droppable)))
        dropped = set(rng.sample(droppable, k))
    else:
        dropped = set(droppable)
    block_htmls = []
    for i, b in enumerate(doc.blocks):
        if i in dropped:
            block_htmls.append(f"<p>{_esc(b.text)}</p>")  # flattened heading
        else:
            block_htmls.append(_block_html(b, i))
    return Variant(
        perturbation="dropped_headings",
        severity=severity,
        block_htmls=block_htmls,
        heading_tree=_heading_tree(doc.blocks, list(range(len(doc.blocks))), dropped),
        nav_html=_nav_html(doc.blocks),  # links to dropped ids now break
    )


def make_broken_refs(doc: CalDoc, severity: str, rng: random.Random) -> Variant | None:
    heading_idx = [i for i, b in enumerate(doc.blocks) if b.is_heading]
    if not heading_idx:
        return None
    frac = 0.4 if severity == "mild" else 0.9
    k = max(1, round(frac * len(heading_idx)))
    broken = set(rng.sample(heading_idx, k))
    strip_ids: set[int] = set()
    if severity == "severe":
        survivors = [i for i in heading_idx if i not in broken]
        strip_ids = set(rng.sample(heading_idx, max(1, len(heading_idx) // 2)))
        strip_ids |= set(survivors[: max(0, len(survivors) // 2)])
    block_htmls = [_block_html(b, i, with_id=i not in strip_ids) for i, b in enumerate(doc.blocks)]
    return Variant(
        perturbation="broken_refs",
        severity=severity,
        block_htmls=block_htmls,
        heading_tree=_heading_tree(doc.blocks, list(range(len(doc.blocks))), set()),
        nav_html=_nav_html(doc.blocks, broken=broken),
    )


def make_hallucinated(
    doc: CalDoc, severity: str, rng: random.Random, donor_pool: list[str]
) -> Variant | None:
    """Gap-fill provenance filled with out-of-vocabulary donor text.

    Document-level adaptation of
    build_semantic_preservation_dataset.perturb_hallucinate_fact: the
    donor pool holds paragraphs from OTHER source documents, so the
    injected tokens have no anchor in this doc's feature blocks.
    """
    donors = [d for d in donor_pool if d]
    if not donors:
        return None
    n_gaps = 1 if severity == "mild" else 3
    gaps: list[GapSlot] = []
    extra_parts: list[str] = []
    for _ in range(n_gaps):
        donor = rng.choice(donors)
        sents = split_sentences(donor)[:3]
        para = " ".join(sents) if sents else donor[:300]
        resolved = f"<p>{_esc(para)}</p>"
        gaps.append(
            GapSlot(
                kind=GapKind.CITATION_UNRESOLVED,
                region_index=-1,
                context={},
                detected_by="calibration:synthetic_hallucination",
                resolved_html=resolved,
            )
        )
        extra_parts.append(resolved)

    block_htmls = [_block_html(b, i) for i, b in enumerate(doc.blocks)]
    if severity == "severe":
        # Also splice donor sentences into ~30% of emitted paragraphs so
        # the cross-encoder sees the contamination, not just the gap dim.
        para_idx = [i for i, b in enumerate(doc.blocks) if b.tag == "p"]
        for i in rng.sample(para_idx, max(1, round(0.3 * len(para_idx)))):
            donor_sent = rng.choice(split_sentences(rng.choice(donors)) or ["lorem"])
            block_htmls[i] = f"<p>{_esc(doc.blocks[i].text)} {_esc(donor_sent)}</p>"
    return Variant(
        perturbation="hallucinated_content",
        severity=severity,
        block_htmls=block_htmls,
        heading_tree=_heading_tree(doc.blocks, list(range(len(doc.blocks))), set()),
        nav_html=_nav_html(doc.blocks),
        extra_html="".join(extra_parts),
        gaps_resolved=gaps,
    )


def make_fragmented(doc: CalDoc, severity: str, rng: random.Random) -> Variant | None:
    para_idx = [i for i, b in enumerate(doc.blocks) if b.tag == "p"]
    splittable = [i for i in para_idx if len(split_sentences(doc.blocks[i].text)) >= 2]
    if not splittable:
        return None
    if severity == "mild":
        chosen = set(rng.sample(splittable, max(1, round(0.4 * len(splittable)))))
        per_sentence = False
    else:
        chosen = set(splittable)
        per_sentence = True
    block_htmls = []
    for i, b in enumerate(doc.blocks):
        if i in chosen:
            sents = split_sentences(b.text)
            if per_sentence:
                parts = sents
            else:
                mid = max(1, len(sents) // 2)
                parts = [" ".join(sents[:mid]), " ".join(sents[mid:])]
            block_htmls.append("".join(f"<p>{_esc(p)}</p>" for p in parts if p))
        else:
            block_htmls.append(_block_html(b, i))
    return Variant(
        perturbation="fragmentation",
        severity=severity,
        block_htmls=block_htmls,
        heading_tree=_heading_tree(doc.blocks, list(range(len(doc.blocks))), set()),
        nav_html=_nav_html(doc.blocks),
    )


def _drift_sentences(text: str, rng: random.Random, donors: list[str], swap_frac: float) -> str:
    """Cross-document sentence swap — adapted from
    build_semantic_preservation_dataset.perturb_semantic_drift (swap,
    not insert: length stays comparable, meaning does not)."""
    sents = split_sentences(text)
    if not sents or not donors:
        return text
    n_swap = max(1, round(swap_frac * len(sents)))
    for i in rng.sample(range(len(sents)), min(n_swap, len(sents))):
        donor_sents = split_sentences(rng.choice(donors))
        if donor_sents:
            sents[i] = rng.choice(donor_sents)
    return " ".join(sents)


def make_meaning_distortion(
    doc: CalDoc, severity: str, rng: random.Random, donor_pool: list[str]
) -> Variant | None:
    """Structure-preserving meaning damage in the emitted text.

    Headings, ids, nav, paragraph counts all stay exactly clean — only
    the prose content is wrong. The seven deterministic dims are blind
    to this by design; it exists so the cross-encoder's contribution to
    clean/degraded separation is actually measured (otherwise the fit
    legitimately zeroes out semantic_preservation and theta goes blind
    to meaning damage in production).

    Both severities use semantic drift (cross-doc sentence swap), NOT
    the builder's word-shuffle: a direct probe showed the v8 cls-pooled
    cross-encoder scores a fully word-shuffled paragraph identically to
    the clean one (order-insensitive), so word-shuffle would only inject
    undetectable-by-construction label noise. That model limitation is a
    finding for the report, not a perturbation to calibrate on.
    """
    para_idx = [i for i, b in enumerate(doc.blocks) if b.tag == "p"]
    if not para_idx:
        return None
    frac, swap_frac = (0.4, 0.34) if severity == "mild" else (0.9, 0.55)
    chosen = set(rng.sample(para_idx, max(1, round(frac * len(para_idx)))))
    block_htmls = []
    for i, b in enumerate(doc.blocks):
        if i in chosen:
            damaged = _drift_sentences(b.text, rng, donor_pool, swap_frac)
            block_htmls.append(f"<p>{_esc(damaged)}</p>")
        else:
            block_htmls.append(_block_html(b, i))
    return Variant(
        perturbation="meaning_distortion",
        severity=severity,
        block_htmls=block_htmls,
        heading_tree=_heading_tree(doc.blocks, list(range(len(doc.blocks))), set()),
        nav_html=_nav_html(doc.blocks),
    )


def make_variants(doc: CalDoc, rng: random.Random, donor_pool: list[str]) -> list[Variant]:
    variants = [make_clean(doc)]
    for severity in SEVERITIES:
        for maker in (
            lambda: make_shuffled_sections(doc, severity, rng),
            lambda: make_dropped_headings(doc, severity, rng),
            lambda: make_broken_refs(doc, severity, rng),
            lambda: make_hallucinated(doc, severity, rng, donor_pool),
            lambda: make_fragmented(doc, severity, rng),
            lambda: make_meaning_distortion(doc, severity, rng, donor_pool),
        ):
            v = maker()
            if v is not None:
                variants.append(v)
    return variants


# ---------------------------------------------------------------------------
# AssembledDoc wrapping + theta scoring
# ---------------------------------------------------------------------------


@dataclass
class _CalRegion:
    """Minimal Region shim — only the attrs the theta scorers read."""

    kind: str
    feature_block_indices: list[int]


_KIND_MAP = {"p": "paragraph", "ul": "list", "ol": "list", "blockquote": "blockquote"}


def build_source_side(doc: CalDoc) -> tuple[list[FeatureBlock], list[_CalRegion]]:
    """Approximate feature_blocks + regions from the CLEAN doc text.

    One FeatureBlock per block (same construction as the
    tests/test_theta.py ``_fb`` helper); headings get ``size_bucket=lg``
    + ``gap_above=lg`` so ``_score_section_preservation`` counts one
    source section per heading, matching the clean output exactly.
    """
    fbs: list[FeatureBlock] = []
    regions: list[_CalRegion] = []
    y = 0.0
    for i, b in enumerate(doc.blocks):
        raw = RawBlock(
            text=b.text,
            page=1,
            bbox=(0.0, y, 100.0, y + 12.0),
            page_width=600.0,
            page_height=800.0,
        )
        y += 14.0
        fbs.append(
            FeatureBlock(
                raw=raw,
                size_bucket="lg" if b.is_heading else "md",
                gap_above="lg" if b.is_heading else None,
                is_top_of_page=False,
                is_centered=False,
                caps=None,
                indent_bucket=0,
                relative_font_ratio=1.4 if b.is_heading else 1.0,
            )
        )
        regions.append(
            _CalRegion(
                kind="heading" if b.is_heading else _KIND_MAP.get(b.tag, "paragraph"),
                feature_block_indices=[i],
            )
        )
    return fbs, regions


def assemble_variant(doc: CalDoc, variant: Variant) -> AssembledDoc:
    title = next((b.text for b in doc.blocks if b.is_heading), doc.doc_id)
    body = variant.nav_html + "".join(variant.block_htmls) + variant.extra_html
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{_esc(title)}</title></head><body><header><p>{_esc(title)}</p></header>"
        f"<main>{body}</main><footer><p>calibration fixture</p></footer></body></html>"
    )
    return AssembledDoc(
        html=html,
        gaps_found=list(variant.gaps_resolved),
        gaps_resolved=list(variant.gaps_resolved),
        gaps_fallback=[],
        heading_tree=list(variant.heading_tree),
        landmarks={"main": 1, "nav": 1 if variant.nav_html else 0, "header": 1, "footer": 1},
        anchors={},
        region_provenance=list(range(len(variant.block_htmls))),
        # The evaluator's semantic_preservation path reads the exact
        # per-region emitted HTML from here (index-aligned with
        # region_provenance) — same contract the cascade uses.
        sub_task_log={"region_html": list(variant.block_htmls)},
    )


def score_variant(
    doc: CalDoc,
    variant: Variant,
    fbs: list[FeatureBlock],
    regions: list[_CalRegion],
) -> dict[str, float]:
    asm = assemble_variant(doc, variant)
    report = evaluate(
        asm,
        feature_blocks=fbs,
        regions=regions,
        wcag_status="passed",
        lane="fast",
    )
    dims = report.dimensions
    sem_method = dims["semantic_preservation"]["breakdown"].get("method")
    if sem_method != "model_v1":
        raise CalibrationError(
            f"semantic_preservation scored via {sem_method!r}, not the real "
            f"cross-encoder — refusing to calibrate 8 weights on stub data "
            f"(no-silent-fallbacks). Check models/theta/semantic_preservation/v8."
        )
    return {name: float(dims[name]["score"]) for name in DIM_ORDER}


# ---------------------------------------------------------------------------
# Fitting (numpy logistic regression; sklearn is not in uv.lock)
# ---------------------------------------------------------------------------


def auc_clean_vs_degraded(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(clean > degraded) with tie correction."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise CalibrationError("AUC undefined: one class is empty")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    combined = np.concatenate([pos, neg])
    for val in np.unique(combined):
        mask = combined == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def fit_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    iters: int = 4000,
    lr: float = 0.5,
    l2_to_uniform: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Class-balanced logistic regression with non-negative weights.

    The composite is a convex combination of dim scores, so the fit is
    constrained to w >= 0 (projected gradient) and L2-regularized
    toward the uniform direction — WHY: with only ~hundreds of docs a
    free fit happily puts all mass on the single most separable
    dimension, which would make theta blind to every other failure
    mode. The prior keeps unused dims at small-but-nonzero weight
    unless the data actively argues otherwise. Returned weights are
    normalized to sum exactly 1.0 (AUC is invariant to the positive
    rescaling, so the reported fitted AUC is the shipped composite's).
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.full(d, 1.0 / d) + rng.normal(0, 1e-3, d)
    w = np.clip(w, 0.0, None)
    b = 0.0
    pos = y == 1
    neg = ~pos
    # Class balancing: each class contributes half the gradient.
    sw = np.where(pos, 0.5 / max(pos.sum(), 1), 0.5 / max(neg.sum(), 1))
    uniform = np.full(d, 1.0 / d)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * sw  # d(balanced logloss)/dz
        grad_w = X.T @ err + l2_to_uniform * (w - uniform)
        grad_b = err.sum()
        w = np.clip(w - lr * grad_w, 0.0, None)
        b -= lr * grad_b
    total = float(w.sum())
    if total <= 0.0:
        raise CalibrationError("logistic fit collapsed to all-zero weights")
    weights = w / total
    diag = {
        "iters": iters,
        "lr": lr,
        "l2_to_uniform": l2_to_uniform,
        "bias": float(b),
        "raw_weight_sum": total,
    }
    return weights, diag


def round_weights_sum_one(weights: np.ndarray, ndigits: int = 4) -> list[float]:
    """Round to ndigits while keeping an exact sum of 1.0 (largest
    remainder on the heaviest dim absorbs the residual)."""
    rounded = [round(float(v), ndigits) for v in weights]
    residual = round(1.0 - sum(rounded), ndigits)
    rounded[int(np.argmax(weights))] = round(rounded[int(np.argmax(weights))] + residual, ndigits)
    return rounded


def pick_thresholds(fitted: np.ndarray, labels: np.ndarray) -> tuple[float, float, dict]:
    """TAU_THETA_RETRY / TAU_THETA_CONFIDENCE from the fitted composite.

    * retry (raw rule): the highest score t such that the population
      below t is >= 90% degraded AND at most 2% of clean docs fall
      below it — "scores down here are almost always real damage, and
      we won't burn offline retries on healthy docs".
    * confidence (raw rule): the clean 10th percentile (90% of
      known-good docs clear it), at least retry + 0.05 so the
      borderline ship_with_flag band cannot vanish.
    * both are then clamped into TAU_RETRY_CLAMP / TAU_CONF_CLAMP —
      see the clamp constants for WHY (synthetic-clean vs real-pipeline
      distribution mismatch; raw values stay in the diagnostics).
    """
    clean = np.sort(fitted[labels == 1])
    clean_p02 = float(np.percentile(clean, 2))
    clean_p10 = float(np.percentile(clean, 10))

    candidates = np.arange(0.40, 0.96, 0.005)
    raw_retry = None
    for t in candidates:
        below = fitted < t
        if below.sum() == 0:
            continue
        purity = float((labels[below] == 0).mean())
        if purity >= 0.90 and t <= clean_p02:
            raw_retry = float(t)
    if raw_retry is None:
        # No grid point satisfies both — fall back (LOUDLY visible in
        # the report) to the clean 2nd percentile minus a small margin.
        raw_retry = max(0.40, clean_p02 - 0.02)
    raw_conf = max(clean_p10, raw_retry + 0.05)

    tau_retry = round(min(max(raw_retry, TAU_RETRY_CLAMP[0]), TAU_RETRY_CLAMP[1]), 3)
    tau_conf = round(min(max(raw_conf, TAU_CONF_CLAMP[0]), TAU_CONF_CLAMP[1]), 3)
    if tau_conf <= tau_retry:
        raise CalibrationError(
            f"threshold clamping produced tau_conf ({tau_conf}) <= tau_retry "
            f"({tau_retry}) — clamp bands need rethinking, not shipping"
        )

    below = fitted < tau_retry
    diag = {
        "clean_p02": round(clean_p02, 4),
        "clean_p10": round(clean_p10, 4),
        "retry_rule": "max t: P(degraded | score<t) >= 0.90 and t <= clean_p02",
        "confidence_rule": "clean 10th percentile, at least retry + 0.05",
        "raw_rule_retry": round(raw_retry, 4),
        "raw_rule_confidence": round(raw_conf, 4),
        "clamp_retry": list(TAU_RETRY_CLAMP),
        "clamp_confidence": list(TAU_CONF_CLAMP),
        "clamp_rationale": (
            "synthetic clean fixtures are assembled by construction and sit "
            "above the real-pipeline theta distribution (R10 avg 0.721 under "
            "uniform weights); unclamped rule values would trigger offline "
            "retries on virtually every real document. Revisit at C1 "
            "corpus scale-up."
        ),
        "n_below_shipped_retry": int(below.sum()),
        "purity_below_shipped_retry": (
            round(float((labels[below] == 0).mean()), 4) if below.sum() else None
        ),
    }
    return tau_conf, tau_retry, diag


def pick_floors(rows: list[dict]) -> dict[str, dict]:
    """Per-dimension floors = clean 2nd percentile, clamped (see
    FLOOR_CLAMPS). Only the three flag-mapped dims carry floors —
    evaluator.flag_map has no flag vocabulary for the other five.

    Degenerate guard: clean reference_integrity / hallucinated_structure
    are constant 1.0 on the synthetic set (every id resolves, no gaps
    are filled in clean fixtures), so a percentile there would be fake
    calibration. Those dims RETAIN the architecture-§6.2 prior floor,
    with the reason recorded.
    """
    out: dict[str, dict] = {}
    for dim, (lo, hi) in FLOOR_CLAMPS.items():
        clean_scores = np.array(
            [r["dims"][dim] for r in rows if r["label"] == "clean"], dtype=float
        )
        p2 = float(np.percentile(clean_scores, 2))
        std = float(np.std(clean_scores))
        if std < 0.01:
            out[dim] = {
                "floor": PRIOR_FLOORS[dim],
                "clean_p02": round(p2, 4),
                "clean_std": round(std, 5),
                "method": "retained_prior",
                "reason": (
                    "clean distribution degenerate on the synthetic set "
                    "(std < 0.01) — keeping the architecture §6.2 prior "
                    "instead of a fake percentile"
                ),
            }
            continue
        floor = round(min(max(p2, lo), hi), 2)
        out[dim] = {
            "floor": floor,
            "clean_p02": round(p2, 4),
            "clean_std": round(std, 5),
            "method": "clean_p02_clamped",
            "clamp": [lo, hi],
        }
    return out


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------

CONFIG_VERSION = "theta-config-2.0"


def write_config(
    config_path: Path,
    weights: dict[str, float],
    tau_conf: float,
    tau_retry: float,
    floors: dict[str, float],
    calibration: dict,
) -> None:
    """Regenerate theta/config.yaml from the fit.

    Carries over ``semantic_model_version``, ``delta_theta_improve``
    and the ``cognitive_load_calibration`` block from the existing
    config (they are owned by other workstreams), replaces weights /
    taus / floors, bumps the version, and appends the calibration
    provenance block.
    """
    import yaml

    existing = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lines = [
        "# Theta configuration — versioned weights + thresholds.",
        "# Locked at model-release time per Plans/03_theta_investigation.md §7.3.",
        "# theta-config-2.0: weights, taus and floors are CALIBRATED by",
        "# scripts/calibrate_theta.py (Plan 12 A3) — see the calibration block",
        "# below for dataset provenance. Do not hand-edit the numbers; re-run",
        "# the harness with --write-config instead.",
        "",
        f"version: {CONFIG_VERSION}",
        "",
        "weights:",
    ]
    for k in DIM_ORDER:
        lines.append(f"  {k}: {weights[k]}")
    lines += [
        "",
        f"tau_theta_ship: {tau_conf}",
        f"tau_theta_retry: {tau_retry}",
        f"delta_theta_improve: {existing.get('delta_theta_improve', 0.05)}",
        "",
        f"floor_reference_integrity: {floors['reference_integrity']}",
        f"floor_hallucinated_structure: {floors['hallucinated_structure_penalty']}",
        f"floor_semantic_preservation: {floors['semantic_preservation']}",
        "",
        f"semantic_model_version: {existing.get('semantic_model_version', 'theta-semantic-deberta-1.0')}",
        "",
    ]
    clc = existing.get("cognitive_load_calibration")
    if clc:
        lines.append("cognitive_load_calibration:")
        for k, v in clc.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    lines.append("# Provenance of the calibrated values above (Plan 12 A3).")
    lines.append("calibration:")
    for k, v in calibration.items():
        if isinstance(v, str):
            lines.append(f'  {k}: "{v}"')
        else:
            lines.append(f"  {k}: {v}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print(*a) -> None:
    """Progress prints flush immediately — long CPU runs are usually
    redirected to a log and block-buffered stdout hides liveness."""
    print(*a, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n", type=int, default=80, help="number of source docs")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--max-blocks", type=int, default=50)
    parser.add_argument("--pairs-root", type=Path, default=REPO_ROOT / "data/pairs")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data/eval_reports/theta_calibration_v1.json",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="regenerate semantik_structure/theta/config.yaml from the fit",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().date().isoformat(),
        help="provenance date stamp (default: today)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()

    # Load the REAL v8 cross-encoder up front — raises (strict-by-default,
    # DART_ALLOW_THETA_STUB scrubbed above) if it can't load, so we never
    # calibrate 8 weights against 7 real dims + a 0.7 constant.
    _print("[calibrate] loading semantic-preservation cross-encoder (CPU)...")
    from semantik_structure.theta._module_state import _get_model

    bundle = _get_model()
    if bundle is None:
        raise CalibrationError(
            "semantic-preservation model unavailable — refusing to calibrate "
            "on the stub (no-silent-fallbacks)."
        )
    if bundle.device != "cpu":
        raise CalibrationError(f"model loaded on {bundle.device!r}; GPU is off-limits")
    # The checkpoint ships bf16; x86 CPUs emulate bf16 matmul (~4x slower
    # than fp32) and the calibrated scores agree to ~3 decimals. Cast once.
    bundle.model.float()
    bundle.head.float()

    _print(f"[calibrate] loading up to {args.n} docs from {args.pairs_root} ...")
    docs = load_docs(args.pairs_root, args.n, args.max_blocks, rng)
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d.source] = by_source.get(d.source, 0) + 1
    _print(f"[calibrate] {len(docs)} docs: {by_source}")

    # Donor pool for hallucination injection: paragraphs from OTHER docs.
    donor_by_doc = {
        d.doc_id: [b.text for b in d.blocks if b.tag == "p" and len(b.text.split()) >= 15]
        for d in docs
    }

    rows: list[dict] = []
    perturbation_counts: dict[str, int] = {}
    for di, doc in enumerate(docs):
        fbs, regions = build_source_side(doc)
        donor_pool = [
            p
            for other_id, paras in donor_by_doc.items()
            if other_id != doc.doc_id
            for p in paras[:3]
        ]
        for variant in make_variants(doc, rng, donor_pool):
            dims = score_variant(doc, variant, fbs, regions)
            key = "clean" if variant.label_clean else f"{variant.perturbation}:{variant.severity}"
            perturbation_counts[key] = perturbation_counts.get(key, 0) + 1
            rows.append(
                {
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "label": "clean" if variant.label_clean else "degraded",
                    "perturbation": variant.perturbation,
                    "severity": variant.severity,
                    "dims": dims,
                }
            )
        if (di + 1) % 10 == 0 or di == len(docs) - 1:
            _print(
                f"[calibrate] scored {di + 1}/{len(docs)} docs "
                f"({len(rows)} variants, {time.time() - t0:.0f}s elapsed)"
            )

    # Checkpoint the raw scores before fitting — the scoring loop is the
    # expensive (potentially ~1h CPU) part; a bug in the cheap fitting /
    # threshold code below must not cost a re-score.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_ckpt = args.out.with_suffix(".rows.json")
    rows_ckpt.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    _print(f"[calibrate] raw variant scores checkpointed to {rows_ckpt}")

    X = np.array([[r["dims"][k] for k in DIM_ORDER] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "clean" else 0 for r in rows], dtype=int)

    uniform = np.full(len(DIM_ORDER), 1.0 / len(DIM_ORDER))
    auc_uniform = auc_clean_vs_degraded(X @ uniform, y)
    fitted_w, fit_diag = fit_weights(X, y, seed=args.seed)
    auc_fitted = auc_clean_vs_degraded(X @ fitted_w, y)
    _print(f"[calibrate] AUC uniform={auc_uniform:.4f} fitted={auc_fitted:.4f}")
    if auc_fitted < auc_uniform:
        raise CalibrationError(
            f"fitted weights AUC ({auc_fitted:.4f}) is below uniform "
            f"({auc_uniform:.4f}) — fit is worse than the placeholder; "
            f"refusing to ship it. Inspect per-dimension separation."
        )

    weight_list = round_weights_sum_one(fitted_w)
    weights = dict(zip(DIM_ORDER, weight_list))
    fitted_scores = X @ np.array([weights[k] for k in DIM_ORDER])
    uniform_scores = X @ uniform
    for r, fs, us in zip(rows, fitted_scores, uniform_scores):
        r["fitted_composite"] = round(float(fs), 4)
        r["uniform_composite"] = round(float(us), 4)

    tau_conf, tau_retry, threshold_diag = pick_thresholds(fitted_scores, y)
    floor_diag = pick_floors(rows)
    floors = {k: v["floor"] for k, v in floor_diag.items()}

    # Per-dimension single-dim AUC — which dims actually separate.
    per_dim_auc = {k: round(auc_clean_vs_degraded(X[:, i], y), 4) for i, k in enumerate(DIM_ORDER)}

    clean_mask = y == 1
    separation = {
        "auc_uniform": round(auc_uniform, 4),
        "auc_fitted": round(auc_fitted, 4),
        "per_dimension_auc": per_dim_auc,
        "fitted_mean_clean": round(float(fitted_scores[clean_mask].mean()), 4),
        "fitted_mean_degraded": round(float(fitted_scores[~clean_mask].mean()), 4),
        "uniform_mean_clean": round(float(uniform_scores[clean_mask].mean()), 4),
        "uniform_mean_degraded": round(float(uniform_scores[~clean_mask].mean()), 4),
    }

    try:
        report_rel = str(args.out.relative_to(REPO_ROOT))
    except ValueError:
        report_rel = str(args.out)
    calibration_block = {
        "report": report_rel,
        "date": args.date,
        "n_docs": len(docs),
        "n_variants": len(rows),
        "seed": args.seed,
        "git_sha": _git_sha() or "unknown",
        "auc_uniform": round(auc_uniform, 4),
        "auc_fitted": round(auc_fitted, 4),
    }

    report = {
        "schema_version": "theta-calibration/1.0",
        "config_version": CONFIG_VERSION,
        "provenance": {
            "date": args.date,
            "git_sha": _git_sha(),
            "seed": args.seed,
            "n_docs": len(docs),
            "docs_by_source": by_source,
            "n_variants": len(rows),
            "perturbation_counts": perturbation_counts,
            "max_blocks": args.max_blocks,
            "pairs_root": str(args.pairs_root),
            "semantic_model_dir": os.environ.get(
                "DART_SEMANTIC_MODEL_DIR", "models/theta/semantic_preservation/v8"
            ),
            "device": "cpu",
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "fit": {
            "method": "class-balanced logistic regression, w >= 0, "
            "L2 prior toward uniform, weights normalized to sum 1.0",
            **fit_diag,
        },
        "weights": weights,
        "thresholds": {
            "tau_theta_confidence": tau_conf,
            "tau_theta_retry": tau_retry,
            **threshold_diag,
        },
        "dim_floors": floor_diag,
        "separation": separation,
        "variants": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    _print(f"[calibrate] report written to {args.out}")

    if args.write_config:
        config_path = REPO_ROOT / "semantik_structure/theta/config.yaml"
        write_config(config_path, weights, tau_conf, tau_retry, floors, calibration_block)
        _print(f"[calibrate] config regenerated at {config_path} ({CONFIG_VERSION})")
        # Validate the regenerated config loads (loudly catches a bad write).
        from semantik_structure.theta.types import _reset_theta_config_cache, load_theta_config

        _reset_theta_config_cache()
        cfg = load_theta_config()
        assert cfg.version == CONFIG_VERSION, cfg.version
        _print(f"[calibrate] regenerated config validates: version={cfg.version}")

    _print(
        f"[calibrate] done in {time.time() - t0:.0f}s — "
        f"weights={weights} tau_conf={tau_conf} tau_retry={tau_retry} floors={floors}"
    )


if __name__ == "__main__":
    main()
