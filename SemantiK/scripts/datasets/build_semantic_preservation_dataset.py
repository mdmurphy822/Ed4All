"""Phase 4a — semantic_preservation cross-encoder training-data construction.

Walks the corpus-internal pair JSONs in ``data/pairs/<domain>/`` and produces a
JSONL training set for a cross-encoder that scores how well a candidate text
preserves the semantics of a clean reference region.

We rely on lxml only (NLTK / spaCy intentionally excluded — see the Phase 4a
plan in the user's prompt). Perturbations are regex / token / corpus-internal
operations.

Outputs:
    data/semantic_preservation_dataset/{train,val,test,
                                        probe_qwen_style,probe_cross_domain}.jsonl
    data/semantic_preservation_dataset/coverage_report.json

Determinism: a single ``--seed`` flag controls every random choice. The same
seed produces an identical dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator, Optional

from lxml import html as lxml_html


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REGION_TAGS = ("p", "li", "dt", "dd")  # plus table-flat (handled separately)

# Length filter (in whitespace tokens).
MIN_REGION_TOKENS = 20
MAX_REGION_TOKENS = 450

# Hard guard on candidate output (chars).
MIN_CANDIDATE_CHARS = 10

# Splits.
SPLIT_TRAIN_FRAC = 0.80
SPLIT_VAL_FRAC = 0.10  # test = remainder

# Perturbation label bands. Each row's label is sampled uniformly inside the
# band, then jittered ±0.05 (clamped to [0, 1]).
LABEL_BANDS: dict[str, tuple[float, float]] = {
    # name                          (low, high)
    "clean":               (0.95, 1.00),
    "paraphrase_shuffle":  (0.65, 0.85),
    "entity_swap":         (0.40, 0.65),
    "negation_flip":       (0.10, 0.30),
    "sentence_drop":       (0.55, 0.80),
    "sentence_shuffle":    (0.75, 0.90),
    "hallucinate_fact":    (0.25, 0.55),
    "semantic_drift":      (0.30, 0.55),
    # Length-matched negative (random different-arxiv region).
    "length_matched_neg":  (0.00, 0.15),
    # Qwen-style failure modes.
    "table_flatten":       (0.20, 0.45),
    "math_as_prose":       (0.25, 0.50),
    "tokenizer_artifact":  (0.30, 0.50),
}

LABEL_JITTER = 0.05
LABEL_CLAMP = (0.0, 1.0)

# Cross-domain probe cap per domain.
CROSS_DOMAIN_PROBE_CAP = 200

# Zero-width / replacement chars used by tokenizer_artifact.
ZERO_WIDTH_CHARS = ["​", "‌", "‍", "﻿", "�"]

# Sentence splitter — naive, regex-based (no NLTK).
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# Word split for paraphrase shuffle and counts.
WORD_SPLIT_RE = re.compile(r"\s+")

# Capitalized-noun-phrase regex for entity_swap pool. We require ≥1 word, all
# starting-uppercase, optionally connected by "of/the/and/-/&/' " etc.
ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'\-]+(?:\s+(?:of|the|and|de|du|von|van|le|la)\s+[A-Z][A-Za-z0-9'\-]+|\s+[A-Z][A-Za-z0-9'\-]+){0,3}\b"
)

# Negation flip rules. Each pair is (pattern -> replacement). Applied as
# whole-word substitutions; we pick at most a few per call.
NEGATION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bis not\b", re.I), "is"),
    (re.compile(r"\bare not\b", re.I), "are"),
    (re.compile(r"\bwas not\b", re.I), "was"),
    (re.compile(r"\bwere not\b", re.I), "were"),
    (re.compile(r"\bdo not\b", re.I), "do"),
    (re.compile(r"\bdoes not\b", re.I), "does"),
    (re.compile(r"\bdid not\b", re.I), "did"),
    (re.compile(r"\bcannot\b", re.I), "can"),
    (re.compile(r"\bcan not\b", re.I), "can"),
    (re.compile(r"\bshould not\b", re.I), "should"),
    (re.compile(r"\bwill not\b", re.I), "will"),
    (re.compile(r"\bwon't\b", re.I), "will"),
    (re.compile(r"\bdoesn't\b", re.I), "does"),
    (re.compile(r"\bdidn't\b", re.I), "did"),
    (re.compile(r"\bdon't\b", re.I), "do"),
    (re.compile(r"\bisn't\b", re.I), "is"),
    (re.compile(r"\baren't\b", re.I), "are"),
    (re.compile(r"\bwasn't\b", re.I), "was"),
    (re.compile(r"\bweren't\b", re.I), "were"),
    (re.compile(r"\bcan't\b", re.I), "can"),
    # Inverse direction — inject "not" after "is/are/was/were".
    (re.compile(r"\bis\b(?! not)", re.I), "is not"),
    (re.compile(r"\bare\b(?! not)", re.I), "are not"),
    (re.compile(r"\bwas\b(?! not)", re.I), "was not"),
    (re.compile(r"\bwere\b(?! not)", re.I), "were not"),
]

# Math/LaTeX-looking spans we strip in the math_as_prose perturbation.
MATH_REGEX = re.compile(
    r"(\$[^$\n]+\$|\\\([^)]+\\\)|\\\[[^]]+\\\])",
    re.MULTILINE,
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("phase4a")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class Region:
    text: str
    kind: str            # "p" / "li" / "dt" / "dd" / "table_flat"
    pair_id: str         # e.g. "0704_1551__00_quantum_zeno..."
    arxiv_id: str        # corpus-internal grouping key (used for splits)
    region_idx: int
    domain: str
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.n_tokens:
            self.n_tokens = ws_token_count(self.text)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def ws_token_count(s: str) -> int:
    return len([t for t in WORD_SPLIT_RE.split(s.strip()) if t])


def stable_split_for_id(arxiv_id: str, train_frac: float, val_frac: float) -> str:
    """Hash-based split. SHA1(arxiv_id) -> [0, 1) -> bucket."""
    h = hashlib.sha1(arxiv_id.encode("utf-8")).hexdigest()
    # Use first 8 hex chars -> int -> normalize to [0, 1)
    val = int(h[:8], 16) / 0xFFFFFFFF
    if val < train_frac:
        return "train"
    if val < train_frac + val_frac:
        return "val"
    return "test"


def jitter_label(rng: random.Random, low: float, high: float) -> float:
    base = rng.uniform(low, high)
    base += rng.uniform(-LABEL_JITTER, LABEL_JITTER)
    return max(LABEL_CLAMP[0], min(LABEL_CLAMP[1], base))


def split_sentences(text: str) -> list[str]:
    out = SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in out if s.strip()]


# --------------------------------------------------------------------------- #
# HTML region extraction
# --------------------------------------------------------------------------- #


def _normalize_text(s: str) -> str:
    # Collapse internal whitespace, drop NBSPs, keep paragraphs intact.
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    return s.strip()


def extract_regions(
    raw_html: str,
    pair_id: str,
    arxiv_id: str,
    domain: str,
    skip_counter: Counter,
) -> list[Region]:
    """Extract candidate regions from a clean reference HTML document.

    Filters by length (min 20 / max 450 whitespace tokens). Skipped regions
    are tallied in ``skip_counter``.
    """
    if not raw_html or len(raw_html) < 200:
        skip_counter["empty_html"] += 1
        return []

    try:
        tree = lxml_html.fromstring(raw_html)
    except Exception as e:  # noqa: BLE001
        skip_counter["html_parse_error"] += 1
        log.debug("html parse failure for %s: %s", pair_id, e)
        return []

    # Drop scripts/styles to avoid contaminating text.
    for bad in tree.xpath("//script | //style | //noscript"):
        bad.getparent().remove(bad) if bad.getparent() is not None else None

    regions: list[Region] = []
    idx = 0

    for tag in REGION_TAGS:
        for el in tree.xpath(f"//{tag}"):
            text = _normalize_text(el.text_content() or "")
            n = ws_token_count(text)
            if n < MIN_REGION_TOKENS:
                skip_counter[f"too_short_{tag}"] += 1
                continue
            if n > MAX_REGION_TOKENS:
                skip_counter[f"too_long_{tag}"] += 1
                continue
            regions.append(
                Region(
                    text=text,
                    kind=tag,
                    pair_id=pair_id,
                    arxiv_id=arxiv_id,
                    region_idx=idx,
                    domain=domain,
                    n_tokens=n,
                )
            )
            idx += 1

    # Table-flat region: per-table joined cell text.
    for tbl in tree.xpath("//table"):
        cells = []
        for cell in tbl.xpath(".//th | .//td"):
            t = _normalize_text(cell.text_content() or "")
            if t:
                cells.append(t)
        if not cells:
            continue
        text = " | ".join(cells)
        n = ws_token_count(text)
        if n < MIN_REGION_TOKENS:
            skip_counter["too_short_table_flat"] += 1
            continue
        if n > MAX_REGION_TOKENS:
            skip_counter["too_long_table_flat"] += 1
            continue
        regions.append(
            Region(
                text=text,
                kind="table_flat",
                pair_id=pair_id,
                arxiv_id=arxiv_id,
                region_idx=idx,
                domain=domain,
                n_tokens=n,
            )
        )
        idx += 1

    return regions


def get_pair_html(pair_json: dict) -> Optional[str]:
    """Pick the cleanest available reference HTML for THIS section.

    Prefer ``output_html`` because that field is the section-scoped WCAG-clean
    rendering — exactly what the OCR for this pair was supposed to produce.
    ``raw_source_html`` is the full-document reference (every pair for the
    same arxiv_id shares the same value), so using it would multiply-count
    every region by the section count and inflate the dataset by ~10x.
    """
    if pair_json.get("output_html"):
        return pair_json["output_html"]
    if pair_json.get("raw_source_html"):
        return pair_json["raw_source_html"]
    if pair_json.get("raw_source_xml"):
        return pair_json["raw_source_xml"]
    return None


def get_pair_arxiv_id(pair_json: dict, fallback_path: Path) -> str:
    """Return the corpus-internal grouping key.

    For arxiv: the ``arxiv_id`` field. For other domains, the most stable
    document-level identifier (article title hash, pmcid, cluster_id, etc.).
    """
    if pair_json.get("arxiv_id"):
        return str(pair_json["arxiv_id"])
    if pair_json.get("pmcid"):
        return f"pmc:{pair_json['pmcid']}"
    if pair_json.get("cluster_id"):
        return f"cluster:{pair_json['cluster_id']}"
    if pair_json.get("book_id"):
        return f"book:{pair_json['book_id']}"
    if pair_json.get("article_title"):
        return f"wiki:{pair_json['article_title']}"
    if pair_json.get("book"):
        return f"openstax:{pair_json['book']}"
    return f"file:{fallback_path.stem}"


def get_pair_id(path: Path) -> str:
    return path.stem


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #


def perturb_paraphrase_shuffle(
    rng: random.Random, region: Region
) -> Optional[tuple[str, dict]]:
    """Shuffle words within each sentence (preserves sentence boundaries)."""
    sents = split_sentences(region.text)
    if not sents:
        return None
    out_sents = []
    swaps = 0
    for sent in sents:
        words = sent.split()
        if len(words) < 4:
            out_sents.append(sent)
            continue
        # Randomized severity: shuffle a contiguous slice of size in [3, len/2]
        slice_len = rng.randint(3, max(3, len(words) // 2))
        start = rng.randint(0, len(words) - slice_len)
        sub = words[start:start + slice_len]
        rng.shuffle(sub)
        words[start:start + slice_len] = sub
        out_sents.append(" ".join(words))
        swaps += slice_len
    return " ".join(out_sents), {
        "shuffled_words": swaps,
        "n_sentences": len(sents),
    }


def perturb_entity_swap(
    rng: random.Random,
    region: Region,
    entity_pool: list[str],
) -> Optional[tuple[str, dict]]:
    """Replace ≥1 capitalized noun phrases with sampled-from-pool entities."""
    if not entity_pool:
        return None
    matches = list(ENTITY_RE.finditer(region.text))
    if not matches:
        return None
    # Randomized severity: swap k = randint(1, min(4, len(matches)))
    k = rng.randint(1, min(4, len(matches)))
    swap_targets = rng.sample(matches, k)
    # Apply right-to-left so spans stay valid.
    out = region.text
    swapped = []
    for m in sorted(swap_targets, key=lambda x: -x.start()):
        original = m.group(0)
        donor = rng.choice(entity_pool)
        # Avoid trivial self-swap.
        for _ in range(5):
            if donor != original:
                break
            donor = rng.choice(entity_pool)
        out = out[: m.start()] + donor + out[m.end():]
        swapped.append({"orig": original, "new": donor})
    return out, {"swaps": len(swapped), "examples": swapped[:3]}


def perturb_negation_flip(
    rng: random.Random, region: Region
) -> Optional[tuple[str, dict]]:
    """Flip 1–3 negation polarities at random matching sites."""
    text = region.text
    sites: list[tuple[int, int, re.Pattern, str]] = []
    for pat, repl in NEGATION_RULES:
        for m in pat.finditer(text):
            sites.append((m.start(), m.end(), pat, repl))
    if not sites:
        return None
    k = rng.randint(1, min(3, len(sites)))
    chosen = rng.sample(sites, k)
    chosen.sort(key=lambda s: -s[0])
    out = text
    for start, end, _pat, repl in chosen:
        out = out[:start] + repl + out[end:]
    return out, {"flips": k}


def perturb_sentence_drop(
    rng: random.Random, region: Region
) -> Optional[tuple[str, dict]]:
    sents = split_sentences(region.text)
    if len(sents) < 2:
        return None
    # Randomized severity: drop in [1, max(1, len/3)] sentences.
    max_drop = max(1, len(sents) // 3)
    n_drop = rng.randint(1, max_drop)
    drop_idxs = set(rng.sample(range(len(sents)), n_drop))
    kept = [s for i, s in enumerate(sents) if i not in drop_idxs]
    if not kept:
        return None
    return " ".join(kept), {
        "sentences_dropped": n_drop,
        "drop_rate": round(n_drop / len(sents), 3),
        "n_sentences": len(sents),
    }


def perturb_sentence_shuffle(
    rng: random.Random, region: Region
) -> Optional[tuple[str, dict]]:
    sents = split_sentences(region.text)
    if len(sents) < 3:
        return None
    shuffled = sents[:]
    # Force at least one swap.
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != sents:
            break
    if shuffled == sents:
        return None
    return " ".join(shuffled), {"n_sentences": len(sents)}


def perturb_hallucinate_fact(
    rng: random.Random,
    region: Region,
    donor_pool: list[tuple[str, str]],   # (sentence, donor_arxiv_id)
) -> Optional[tuple[str, dict]]:
    """Inject a sentence drawn from a different ``arxiv_id``."""
    eligible = [s for s, aid in donor_pool if aid != region.arxiv_id]
    if not eligible:
        return None
    n_inject = rng.randint(1, 2)
    sents = split_sentences(region.text)
    if not sents:
        return None
    # Insert at random positions.
    out = sents[:]
    injected = []
    for _ in range(n_inject):
        donor_sent = rng.choice(eligible)
        pos = rng.randint(0, len(out))
        out.insert(pos, donor_sent)
        injected.append(donor_sent[:80])
    return " ".join(out), {
        "n_injected": n_inject,
        "samples": injected,
    }


def perturb_semantic_drift(
    rng: random.Random,
    region: Region,
    donor_pool: list[tuple[str, str]],   # (sentence, arxiv_id) — ANY arxiv_id allowed except self
) -> Optional[tuple[str, dict]]:
    """Replace a clause in source with a clause from a different arxiv_id.

    Different from hallucinate_fact: we *swap*, not *insert*.
    """
    eligible = [s for s, aid in donor_pool if aid != region.arxiv_id]
    if not eligible:
        return None
    sents = split_sentences(region.text)
    if len(sents) < 2:
        return None
    n_swap = rng.randint(1, max(1, len(sents) // 3))
    swap_idxs = rng.sample(range(len(sents)), n_swap)
    out = sents[:]
    for i in swap_idxs:
        out[i] = rng.choice(eligible)
    return " ".join(out), {"n_swapped": n_swap, "n_sentences": len(sents)}


def perturb_table_flatten(
    rng: random.Random, region: Region
) -> Optional[tuple[str, dict]]:
    """Drop newlines + cell separators in table-derived prose."""
    if region.kind != "table_flat":
        return None
    out = region.text.replace(" | ", " ").replace("\n", " ")
    out = re.sub(r"\s+", " ", out).strip()
    if out == region.text or len(out) < MIN_CANDIDATE_CHARS:
        return None
    return out, {"removed": "cell_separators"}


def perturb_math_as_prose(
    rng: random.Random, region: Region
) -> Optional[tuple[str, dict]]:
    """Strip math-looking spans (very lightweight; ar5iv text is already
    LaTeXML-rendered Unicode, so we also remove obvious leftover ``$...$``
    fragments)."""
    text = region.text
    matches = list(MATH_REGEX.finditer(text))
    if not matches:
        # Fallback: drop runs of "ket|psi|rangle" style LaTeXML residue —
        # those are common ar5iv leftovers when math is present.
        residue = re.compile(r"\b(ket|bra|rangle|langle|psi|phi)\b", re.I)
        residues = list(residue.finditer(text))
        if not residues:
            return None
        out = residue.sub("", text)
        if out.strip() == text.strip():
            return None
        return out, {"stripped": "latexml_residue", "n_strips": len(residues)}
    out = MATH_REGEX.sub("", text)
    return out, {"stripped": "math_spans", "n_strips": len(matches)}


def perturb_tokenizer_artifact(
    rng: random.Random, region: Region
) -> Optional[tuple[str, dict]]:
    """Sprinkle zero-width / replacement chars at random positions."""
    text = region.text
    if len(text) < 20:
        return None
    n_inject = rng.randint(3, 12)
    chars = list(text)
    positions = sorted(rng.sample(range(len(chars)), min(n_inject, len(chars))), reverse=True)
    for p in positions:
        chars.insert(p, rng.choice(ZERO_WIDTH_CHARS))
    return "".join(chars), {"n_injected": n_inject}


# --------------------------------------------------------------------------- #
# Length-matched negative
# --------------------------------------------------------------------------- #


@dataclass
class CorpusIndex:
    """All regions, grouped for fast length-matched negative mining."""
    by_arxiv_id: dict[str, list[Region]] = field(default_factory=dict)
    all_regions: list[Region] = field(default_factory=list)
    # Donor sentences (text, arxiv_id) for hallucinate_fact / semantic_drift.
    donor_sentences: list[tuple[str, str]] = field(default_factory=list)
    # Entity pool (capitalized noun phrases) sampled across the corpus.
    entity_pool: list[str] = field(default_factory=list)

    def add(self, region: Region) -> None:
        self.by_arxiv_id.setdefault(region.arxiv_id, []).append(region)
        self.all_regions.append(region)

    def freeze(self, rng: random.Random, max_donors: int = 50_000,
               max_entities: int = 50_000) -> None:
        """Build the donor sentence pool + entity pool. Done once."""
        # Donors: sample sentences from any region.
        donors: list[tuple[str, str]] = []
        rng.shuffle(self.all_regions)  # in place; harmless after split
        for r in self.all_regions:
            for s in split_sentences(r.text):
                if 5 <= len(s.split()) <= 60:
                    donors.append((s, r.arxiv_id))
                if len(donors) >= max_donors:
                    break
            if len(donors) >= max_donors:
                break
        rng.shuffle(donors)
        self.donor_sentences = donors

        # Entities.
        ent_seen: set[str] = set()
        for r in self.all_regions:
            for m in ENTITY_RE.finditer(r.text):
                e = m.group(0)
                if 2 <= len(e) <= 60 and e not in ent_seen:
                    ent_seen.add(e)
                    self.entity_pool.append(e)
                    if len(self.entity_pool) >= max_entities:
                        break
            if len(self.entity_pool) >= max_entities:
                break
        rng.shuffle(self.entity_pool)


def mine_length_matched_negative(
    rng: random.Random,
    index: CorpusIndex,
    source: Region,
    pct: float = 0.15,
    max_tries: int = 50,
) -> Optional[Region]:
    """Random region from a different ``arxiv_id`` whose token count is
    within ±``pct`` of the source."""
    target = source.n_tokens
    low = target * (1 - pct)
    high = target * (1 + pct)
    pool = index.all_regions
    if not pool:
        return None
    for _ in range(max_tries):
        cand = rng.choice(pool)
        if cand.arxiv_id == source.arxiv_id:
            continue
        if low <= cand.n_tokens <= high:
            return cand
    return None


# --------------------------------------------------------------------------- #
# Row builder
# --------------------------------------------------------------------------- #


def make_row(
    source: Region,
    candidate_text: str,
    candidate_kind: str,
    perturbation: str,
    label: float,
    severity: dict,
) -> Optional[dict]:
    if not candidate_text or len(candidate_text.strip()) < MIN_CANDIDATE_CHARS:
        return None
    return {
        "source_text": source.text,
        "candidate_text": candidate_text,
        "label": round(float(label), 4),
        "perturbation_type": perturbation,
        "source_pair": source.pair_id,
        "source_domain": source.domain,
        "source_region_idx": source.region_idx,
        "source_region_kind": source.kind,
        "candidate_region_kind": candidate_kind,
        "n_source_tokens": source.n_tokens,
        "n_candidate_tokens": ws_token_count(candidate_text),
        "severity_metadata": severity,
    }


# --------------------------------------------------------------------------- #
# Build pipelines
# --------------------------------------------------------------------------- #


# Perturbations enabled per region; "table_flatten" only fires when kind == table_flat.
GENERIC_PERTURBATIONS = [
    "paraphrase_shuffle",
    "entity_swap",
    "negation_flip",
    "sentence_drop",
    "sentence_shuffle",
    "hallucinate_fact",
    "semantic_drift",
]


def build_main_rows(
    rng: random.Random,
    regions: list[Region],
    index: CorpusIndex,
    skip_counter: Counter,
) -> Iterator[dict]:
    """For each clean region, emit:
      - 1 clean (label ~1.0)
      - 1 row per generic perturbation (silently skipped if degenerate)
      - 1 length-matched negative
    """
    for r in regions:
        # Clean.
        lo, hi = LABEL_BANDS["clean"]
        row = make_row(r, r.text, r.kind, "clean", jitter_label(rng, lo, hi), {})
        if row:
            yield row
        else:
            skip_counter["empty_clean"] += 1

        # Generic perturbations.
        for pname in GENERIC_PERTURBATIONS:
            res: Optional[tuple[str, dict]] = None
            if pname == "paraphrase_shuffle":
                res = perturb_paraphrase_shuffle(rng, r)
            elif pname == "entity_swap":
                res = perturb_entity_swap(rng, r, index.entity_pool)
            elif pname == "negation_flip":
                res = perturb_negation_flip(rng, r)
            elif pname == "sentence_drop":
                res = perturb_sentence_drop(rng, r)
            elif pname == "sentence_shuffle":
                res = perturb_sentence_shuffle(rng, r)
            elif pname == "hallucinate_fact":
                res = perturb_hallucinate_fact(rng, r, index.donor_sentences)
            elif pname == "semantic_drift":
                res = perturb_semantic_drift(rng, r, index.donor_sentences)

            if res is None:
                skip_counter[f"degen_{pname}"] += 1
                continue
            cand, severity = res
            lo, hi = LABEL_BANDS[pname]
            row = make_row(r, cand, r.kind, pname, jitter_label(rng, lo, hi), severity)
            if row is None:
                skip_counter[f"empty_{pname}"] += 1
                continue
            yield row

        # Qwen-style failure modes — TRAIN on these (previously probe-only,
        # the root cause of v7's weak qwen_style ranking). Emitted before the
        # length-matched negative so a region with no length match still
        # contributes its qwen-style rows.
        yield from emit_qwen_style_rows(rng, r, skip_counter)

        # Length-matched negative.
        neg = mine_length_matched_negative(rng, index, r)
        if neg is None:
            skip_counter["no_length_match"] += 1
            continue
        lo, hi = LABEL_BANDS["length_matched_neg"]
        row = make_row(
            r, neg.text, neg.kind, "length_matched_neg",
            jitter_label(rng, lo, hi),
            {
                "neg_pair": neg.pair_id,
                "neg_region_idx": neg.region_idx,
                "neg_arxiv_id": neg.arxiv_id,
                "n_neg_tokens": neg.n_tokens,
            },
        )
        if row:
            yield row


def emit_qwen_style_rows(
    rng: random.Random,
    r: "Region",
    skip_counter: Counter,
) -> Iterator[dict]:
    """Yield the 3 Qwen-style failure-mode rows for ONE region:
    table_flatten (table_flat kind only), math_as_prose, tokenizer_artifact.

    Shared by build_main_rows (so train/val/test TRAIN on these — previously
    they were probe-only, which is why the v7 cross-encoder scored
    spearman~0.24 on the qwen_style distribution it never saw) and by the
    held-out probe. Bands come from LABEL_BANDS.
    """
    # table_flatten — only fires for table_flat kind.
    if r.kind == "table_flat":
        res = perturb_table_flatten(rng, r)
        if res is not None:
            cand, severity = res
            lo, hi = LABEL_BANDS["table_flatten"]
            row = make_row(r, cand, r.kind, "table_flatten",
                           jitter_label(rng, lo, hi), severity)
            if row:
                yield row
            else:
                skip_counter["empty_table_flatten"] += 1
        else:
            skip_counter["degen_table_flatten"] += 1

    # math_as_prose — try on any prose region (most ar5iv prose has math residue).
    res = perturb_math_as_prose(rng, r)
    if res is not None:
        cand, severity = res
        lo, hi = LABEL_BANDS["math_as_prose"]
        row = make_row(r, cand, r.kind, "math_as_prose",
                       jitter_label(rng, lo, hi), severity)
        if row:
            yield row
        else:
            skip_counter["empty_math_as_prose"] += 1
    # No 'else' counter here — most regions legitimately have no math.

    # tokenizer_artifact — works on any region.
    res = perturb_tokenizer_artifact(rng, r)
    if res is not None:
        cand, severity = res
        lo, hi = LABEL_BANDS["tokenizer_artifact"]
        row = make_row(r, cand, r.kind, "tokenizer_artifact",
                       jitter_label(rng, lo, hi), severity)
        if row:
            yield row
        else:
            skip_counter["empty_tokenizer_artifact"] += 1


def build_qwen_probe_rows(
    rng: random.Random,
    regions: list[Region],
    skip_counter: Counter,
) -> Iterator[dict]:
    """Held-out Qwen-style probe: 3 failure-mode buckets per region.

    Now drawn from VAL regions (see main), so it remains a true held-out
    OOD measure even though train also contains qwen-style rows.
    """
    for r in regions:
        yield from emit_qwen_style_rows(rng, r, skip_counter)


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #


def load_pair(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to load %s: %s", path, e)
        return None


def gather_arxiv_pairs(arxiv_dir: Path, max_pairs: Optional[int] = None) -> list[Path]:
    paths = sorted(arxiv_dir.glob("*.json"))
    if max_pairs is not None:
        paths = paths[:max_pairs]
    return paths


def gather_cross_domain_pairs(
    pairs_root: Path,
    domains: list[str],
) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for d in domains:
        ddir = pairs_root / d
        if not ddir.exists():
            log.info("cross-domain probe: %s missing — skipping", d)
            continue
        paths = sorted(ddir.glob("*.json"))
        if paths:
            out[d] = paths
    return out


def write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def summarize_jsonl(path: Path) -> dict:
    """Streaming pass to compute per-perturbation, per-region-kind, label
    distribution stats."""
    counts_by_perturb: Counter = Counter()
    counts_by_kind: Counter = Counter()
    counts_by_domain: Counter = Counter()
    label_sum_by_perturb: dict[str, list[float]] = defaultdict(list)
    length_hist: list[int] = []
    n = 0
    if not path.exists():
        return {"rows": 0}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n += 1
            counts_by_perturb[row["perturbation_type"]] += 1
            counts_by_kind[row["source_region_kind"]] += 1
            counts_by_domain[row.get("source_domain", "unknown")] += 1
            label_sum_by_perturb[row["perturbation_type"]].append(row["label"])
            length_hist.append(row["n_source_tokens"])
    label_stats = {}
    for k, vs in label_sum_by_perturb.items():
        m = sum(vs) / len(vs) if vs else 0.0
        var = sum((v - m) ** 2 for v in vs) / len(vs) if vs else 0.0
        label_stats[k] = {
            "n": len(vs),
            "mean": round(m, 4),
            "std": round(var ** 0.5, 4),
            "min": round(min(vs), 4) if vs else None,
            "max": round(max(vs), 4) if vs else None,
        }
    # Crude length histogram bins.
    bins = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 250), (250, 350), (350, 451)]
    hist = {}
    for lo, hi in bins:
        hist[f"{lo}-{hi-1}"] = sum(1 for x in length_hist if lo <= x < hi)
    return {
        "rows": n,
        "by_perturbation_type": dict(counts_by_perturb),
        "by_source_region_kind": dict(counts_by_kind),
        "by_source_domain": dict(counts_by_domain),
        "label_distribution_per_type": label_stats,
        "source_token_length_hist": hist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-root", default="data/pairs",
        help="Root directory containing per-domain pair JSON folders.")
    parser.add_argument(
        "--out-dir", default="data/semantic_preservation_dataset",
        help="Where to write {train,val,test,probe_*}.jsonl + coverage_report.json.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="If set, cap the number of arxiv pairs (smoke run).")
    parser.add_argument(
        "--cross-domain-cap", type=int, default=CROSS_DOMAIN_PROBE_CAP,
        help="Max pairs per cross-domain probe domain.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pairs_root = Path(args.pairs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_counter: Counter = Counter()
    domain_row_counts: Counter = Counter()

    # --------------------------------------------------------------------- #
    # Stage 1 — walk arxiv pairs, group regions by split.
    # --------------------------------------------------------------------- #
    arxiv_paths = gather_arxiv_pairs(pairs_root / "arxiv", max_pairs=args.max_pairs)
    log.info("arxiv pairs to process: %d", len(arxiv_paths))

    by_split_regions: dict[str, list[Region]] = {"train": [], "val": [], "test": []}
    arxiv_ids_per_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    n_pairs_loaded = 0

    for path in arxiv_paths:
        pj = load_pair(path)
        if pj is None:
            skip_counter["pair_load_fail"] += 1
            continue
        raw_html = get_pair_html(pj)
        if not raw_html:
            skip_counter["no_raw_html"] += 1
            continue
        arxiv_id = get_pair_arxiv_id(pj, path)
        pair_id = get_pair_id(path)
        split = stable_split_for_id(arxiv_id, SPLIT_TRAIN_FRAC, SPLIT_VAL_FRAC)
        regions = extract_regions(raw_html, pair_id, arxiv_id, "arxiv", skip_counter)
        if not regions:
            skip_counter["no_regions_after_filter"] += 1
            continue
        by_split_regions[split].extend(regions)
        arxiv_ids_per_split[split].add(arxiv_id)
        n_pairs_loaded += 1

    log.info(
        "loaded %d arxiv pairs; regions per split: train=%d val=%d test=%d",
        n_pairs_loaded,
        len(by_split_regions["train"]),
        len(by_split_regions["val"]),
        len(by_split_regions["test"]),
    )

    # Acceptance gate: arxiv_id non-overlap.
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = arxiv_ids_per_split[a] & arxiv_ids_per_split[b]
        if overlap:
            log.error("arxiv_id overlap between %s and %s: %d ids", a, b, len(overlap))
            sys.exit(1)

    # --------------------------------------------------------------------- #
    # Stage 2 — build per-split corpus index (donors + entity pool drawn from
    # train regions only, to avoid leaking val/test text into perturbations).
    # --------------------------------------------------------------------- #
    train_index = CorpusIndex()
    for r in by_split_regions["train"]:
        train_index.add(r)
    train_index.freeze(random.Random(args.seed + 1))

    # For val/test we still need donors and entity pool — use the train index
    # so val/test perturbations are not derived from val/test text.
    val_index = train_index
    test_index = train_index

    # --------------------------------------------------------------------- #
    # Stage 3 — write train/val/test jsonl files.
    # --------------------------------------------------------------------- #
    for split, idx_for_split in [
        ("train", train_index),
        ("val", val_index),
        ("test", test_index),
    ]:
        out_path = out_dir / f"{split}.jsonl"
        regions = by_split_regions[split]
        rng_split = random.Random(args.seed + {"train": 100, "val": 200, "test": 300}[split])
        n = write_jsonl(
            out_path,
            build_main_rows(rng_split, regions, idx_for_split, skip_counter),
        )
        domain_row_counts[f"arxiv:{split}"] = n
        log.info("wrote %s rows: %d", out_path, n)

    # --------------------------------------------------------------------- #
    # Stage 4 — Qwen-style probe (drawn from train regions; small).
    # --------------------------------------------------------------------- #
    qwen_probe_path = out_dir / "probe_qwen_style.jsonl"
    rng_qwen = random.Random(args.seed + 1000)
    # Sample up to 2000 VAL regions for the probe (stratify a little: take all
    # tables we have, then top up with prose). Drawn from VAL — not train —
    # because train now also emits qwen-style rows (build_main_rows); using
    # held-out val regions keeps this a true OOD generalization probe.
    probe_source_regions = by_split_regions["val"]
    table_regions = [r for r in probe_source_regions if r.kind == "table_flat"]
    prose_regions = [r for r in probe_source_regions if r.kind != "table_flat"]
    rng_qwen.shuffle(prose_regions)
    probe_regions = table_regions + prose_regions[: max(0, 2000 - len(table_regions))]
    n = write_jsonl(
        qwen_probe_path,
        build_qwen_probe_rows(rng_qwen, probe_regions, skip_counter),
    )
    domain_row_counts["probe_qwen_style"] = n
    log.info("wrote %s rows: %d", qwen_probe_path, n)

    # --------------------------------------------------------------------- #
    # Stage 5 — Cross-domain probe.
    # --------------------------------------------------------------------- #
    cross_domain_path = out_dir / "probe_cross_domain.jsonl"
    cross_domains = ["openstax", "pmc", "courtlistener", "gutenberg", "wikipedia"]
    cross_pairs = gather_cross_domain_pairs(pairs_root, cross_domains)
    rng_cd = random.Random(args.seed + 2000)
    cd_regions: list[Region] = []
    cd_per_domain: Counter = Counter()
    cap = args.cross_domain_cap
    for domain, paths in cross_pairs.items():
        # Region-level cap: stop ingesting when we hit the per-domain cap.
        domain_regs: list[Region] = []
        for p in paths:
            if len(domain_regs) >= cap:
                break
            pj = load_pair(p)
            if pj is None:
                continue
            raw = get_pair_html(pj)
            if not raw:
                continue
            arxiv_id = get_pair_arxiv_id(pj, p)
            pair_id = get_pair_id(p)
            regs = extract_regions(raw, pair_id, arxiv_id, domain, skip_counter)
            domain_regs.extend(regs)
        # Hard cap.
        if len(domain_regs) > cap:
            rng_cd.shuffle(domain_regs)
            domain_regs = domain_regs[:cap]
        cd_regions.extend(domain_regs)
        cd_per_domain[domain] = len(domain_regs)
        log.info("cross-domain %s regions: %d (cap=%d)", domain, len(domain_regs), cap)

    n = write_jsonl(
        cross_domain_path,
        build_main_rows(rng_cd, cd_regions, train_index, skip_counter),
    )
    domain_row_counts["probe_cross_domain"] = n
    log.info("wrote %s rows: %d", cross_domain_path, n)

    # --------------------------------------------------------------------- #
    # Stage 6 — Coverage report.
    # --------------------------------------------------------------------- #
    report = {
        "seed": args.seed,
        "max_pairs": args.max_pairs,
        "n_arxiv_pairs_loaded": n_pairs_loaded,
        "arxiv_ids_per_split": {k: len(v) for k, v in arxiv_ids_per_split.items()},
        "regions_per_split": {k: len(v) for k, v in by_split_regions.items()},
        "split_overlap_check": "PASS",
        "row_counts_per_output": dict(domain_row_counts),
        "cross_domain_region_counts": dict(cd_per_domain),
        "skip_counts": dict(skip_counter),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        report["splits"][split] = summarize_jsonl(out_dir / f"{split}.jsonl")
    report["probe_qwen_style"] = summarize_jsonl(out_dir / "probe_qwen_style.jsonl")
    report["probe_cross_domain"] = summarize_jsonl(out_dir / "probe_cross_domain.jsonl")

    report_path = out_dir / "coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote coverage report: %s", report_path)


if __name__ == "__main__":
    main()
