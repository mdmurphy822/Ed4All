"""Build Qwen 3 training pairs for the stage-3b reasoner (v2 format).

This rewrite drops the "corrections JSON" target of the earlier script
and emits full per-block labelings aligned with the new reason pipeline:

  - DistilBERT v3 produces a hint + confidence for every block (a feature,
    not a candidate to correct).
  - Qwen's target is a JSON array labeling every block in the chunk.
  - Input uses the locked short-form schema in reason_schema.py; chunking
    by pages (reason_chunking.py) with 1-page overlap.

Training-time contract:
    input  = (extractor feature stream + DistilBERT v3 hints)
    target = full block-level labeling as a JSON array

DistilBERT is FROZEN during Qwen's training. Its outputs feed the
training data, but its weights never see gradient from Qwen's loss.

## Stratified override oversampling

DistilBERT v3 has known systematic biases (notably the in_header_row
25%-top-of-table artifact that over-predicts `table_header_cell`).
Generic "duplicate every override-containing chunk" teaches Qwen to
override *any* hint, which causes the opposite failure mode of
over-correction.

The stratified oversampler (see `stratified_oversample()`) instead:

  1. **Tracks confusion pairs.** Each override is recorded as a
     (hint, actual) tuple. We compute the natural distribution of
     those pairs across the corpus.
  2. **Per-pair floor.** Only pairs that appear in >= `--pair-floor`
     chunks are eligible for oversampling. Rare confusions are noise
     and we don't invent signal for them.
  3. **Per-pair target.** For each eligible pair, bring the total chunk
     count to `--per-pair-target`. Frequent artifact patterns (th -> td)
     and rarer but systematic ones (hd -> pa) both get floor coverage.
  4. **Diversity cap.** No single chunk is duplicated more than
     `--max-dup-per-chunk` times. Forces the oversampling pattern to
     come from diverse source documents so Qwen doesn't memorize one
     chunk's idiosyncrasies.
  5. **Saturation logged.** When a pair's diverse-source pool is
     exhausted (e.g. 30 natural sources * 5 max dup = 150 total
     possible, below a 200 target), we stop and log it so the caller
     sees the limit rather than silently duplicating past the cap.

Diagnostics printed at the end show the pre- and post-oversample
confusion matrix and the per-hint override chunk counts, so you can
see what Qwen will actually be taught to correct.

Usage:
    python data/builders/build_qwen_data.py \\
        --classifier models/classifier_v3/final \\
        --out-dir data/qwen_dataset_v2 \\
        --pages-per-chunk 4 --overlap 1 \\
        --override-target-frac 0.20 \\
        --pair-floor 20 --per-pair-target 200 \\
        --max-dup-per-chunk 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from bs4 import BeautifulSoup

from semantik_structure.classify import classify_blocks, load_classifier
from semantik_structure.extract_shared import extract_shared_cached
from semantik_structure.features import featurize_from_shared
from semantik_structure.prerender_cache import cache_path_for
from semantik_structure.reason import SYSTEM_PROMPT
from semantik_structure.reason_chunking import build_page_chunks
from semantik_structure.reason_schema import (
    chunk_header,
    page_separator,
    serialize_block,
)
from semantik_structure.text_utils import jaccard_overlap


# HTML tag → role map. Same vocabulary v2 classifier data uses.
TAG_TO_ROLE = {
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "p": "paragraph", "li": "list_item",
    "th": "table_header_cell", "td": "table_data_cell",
    "caption": "table_caption", "figcaption": "figure_caption",
    "blockquote": "blockquote", "pre": "code_block", "code": "code_block",
    "dt": "definition", "dd": "definition",
    "legend": "form_label", "label": "form_label",
    "input": "form_field", "select": "form_field", "textarea": "form_field",
}
OUTER_TAG_ALLOW_NESTED = {"th", "td", "li", "caption", "figcaption", "legend", "label"}


def extract_html_blocks(html: str) -> list[tuple[str, str]]:
    """Return [(text, role), ...] in document order from output_html."""
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str]] = []
    for el in soup.find_all(list(TAG_TO_ROLE.keys())):
        if el.name not in OUTER_TAG_ALLOW_NESTED:
            has_outer = any(
                p.name in TAG_TO_ROLE and p is not el
                for p in el.parents if p.name is not None
            )
            if has_outer:
                continue
        text = " ".join((el.get_text() or "").split()).strip()
        if text:
            out.append((text, TAG_TO_ROLE[el.name]))
    return out


def align_to_ground_truth(features, html_blocks, *,
                          min_overlap_heading: float = 0.6,
                          min_overlap_default: float = 0.3,
                          heading_len_ratio_min: float = 0.5):
    """Return {global_block_idx: target_role} via sliding-window match.

    Accepts the feature block list directly (we only need `.raw.text`),
    so this can run in CPU workers before classification — keeping the
    O(N * window) jaccard cost off the main process.

    The v7 dataset was poisoned by three failure modes in the previous
    version of this function (uniform min_overlap=0.3, no hyphenation
    handling, no length sanity); v8 tightens those failure modes
    without collapsing paragraph coverage:

    1. Hyphenation tails. PDF extraction often leaves a hyphenated word
       split across two blocks: "...and func-" / "tional materials.".
       The tail's text shares only one token with anything in the HTML
       and the old code jaccard-matched it to whatever short heading
       shared that token. We detect tails (previous block ends with
       `-`, current starts lowercase) and copy the previous label —
       same as if the two blocks had been merged.

    2. Role-aware threshold. Heading match needs jaccard >= 0.6;
       everything else stays at 0.3. False-positive headings break
       reading-order in the rendered HTML; false-positive paragraphs
       still render as <p>, so we keep paragraph coverage broad.

    3. Heading length sanity. Real headings are short. If the best
       candidate is a heading but its text length is wildly different
       from the PDF block (ratio < 0.5), reject — a 100-char body
       fragment has no business being labeled a 20-char heading even
       when a noisy jaccard happens to cross the threshold.

    Together these dropped the v7-vs-v8 false-positive heading rate
    from ~51% of heading targets to <5% on a wikipedia+arxiv smoke
    set, with single-digit coverage loss on non-heading targets.
    """
    labels: dict[int, str] = {}
    cursor = 0
    window = 8
    for i, fb in enumerate(features):
        text = (fb.raw.text or "")

        # Hyphenation tail: if the previous block ended with `-` and we
        # start with lowercase, we are the second half of a wrapped
        # hyphenated word. Inherit the previous block's label and skip
        # an independent jaccard search.
        if i > 0 and (i - 1) in labels:
            prev_text = (features[i - 1].raw.text or "").rstrip()
            if prev_text.endswith("-") and text and text[0].islower():
                labels[i] = labels[i - 1]
                continue

        best_idx, best_score = -1, 0.0
        for j in range(max(0, cursor - 2),
                       min(len(html_blocks), cursor + window)):
            s = jaccard_overlap(text, html_blocks[j][0])
            if s > best_score:
                best_score = s
                best_idx = j

        if best_idx < 0:
            continue

        cand_text, cand_role = html_blocks[best_idx]
        threshold = (min_overlap_heading if cand_role == "heading"
                     else min_overlap_default)
        if best_score < threshold:
            continue

        # Heading-length sanity: real headings are short; a long PDF
        # block claiming a heading match is almost always a paragraph
        # fragment that sneaked one token past the threshold.
        if cand_role == "heading" and cand_text:
            len_pdf = len(text)
            len_html = len(cand_text)
            ratio = min(len_pdf, len_html) / max(len_pdf, len_html, 1)
            if ratio < heading_len_ratio_min:
                continue

        labels[i] = cand_role
        cursor = max(cursor, best_idx + 1)
    return labels


def build_chunk_example(classified, chunk, targets, chunk_count: int,
                        *,
                        glm_ocr_by_page: dict[int, list[dict]] | None = None,
                        ) -> dict | None:
    """Assemble one Qwen training example for a single page-chunk.

    The target is a JSON array covering every aligned block in the chunk.
    Blocks without a ground-truth label are omitted from the target
    (Qwen still sees them in the input and gets to predict, but they
    don't contribute to loss — we lack a target for them).

    When `glm_ocr_by_page` is provided, regions that overlap the chunk's
    page range get an extra section appended to the user prompt with
    `--- OCR regions ---` followed by one line per region:
        [ocr p=<pg> k=<math|table> h=<distilbert_role>] <glm_ocr_text>
    """
    covered = [bid for bid in chunk.block_ids if bid in targets]
    if not covered:
        return None

    # Build the input user-message: chunk header, page separators, block lines.
    lines: list[str] = [chunk_header(
        chunk_index=chunk.index, chunk_count=chunk_count,
        page_start=chunk.page_start, page_end=chunk.page_end,
    )]
    current_page = -1
    local_to_global: list[int] = []  # local_id index -> global bid
    for local_id, global_bid in enumerate(chunk.block_ids):
        cb = classified[global_bid]
        fb = cb.features
        if fb.raw.page != current_page:
            lines.append(page_separator(fb.raw.page))
            current_page = fb.raw.page
        sb = serialize_block(
            block_id=local_id,
            text=fb.raw.text,
            size_bucket=fb.size_bucket,
            is_bold=bool(fb.raw.is_bold),
            is_centered=fb.is_centered,
            is_top_of_page=fb.is_top_of_page,
            gap_above=fb.gap_above,
            caps=fb.caps,
            in_table=fb.in_table,
            in_header_row=fb.in_header_row,
            in_widget=fb.in_widget,
            widget_kind=fb.widget_kind,
            provenance=fb.provenance or fb.raw.source,
            hint_label=cb.role,
            hint_confidence=cb.confidence,
        )
        lines.append(sb.to_line())
        local_to_global.append(global_bid)

    if glm_ocr_by_page:
        pages_in_chunk = range(chunk.page_start, chunk.page_end + 1)
        ocr_lines: list[str] = []
        for p in pages_in_chunk:
            for region in glm_ocr_by_page.get(p, []):
                text = (region.get("text") or "").strip()
                if not text:
                    continue
                kind = region.get("kind") or "ocr"
                hint = region.get("distilbert_role") or "?"
                # Keep each region to a single line to preserve the
                # existing block-line-per-line structure Qwen was
                # trained on. Collapse any newlines the OCR emitted.
                text_one_line = " ".join(text.split())
                ocr_lines.append(f"[ocr p={p} k={kind} h={hint}] {text_one_line}")
        if ocr_lines:
            lines.append("--- OCR regions ---")
            lines.extend(ocr_lines)
    user_msg = "\n".join(lines)

    # Build target JSON array: one entry per chunk block we have ground truth for.
    target_entries = []
    override_count = 0
    # Track confusion pairs (hint, actual) on overrides — used by the
    # stratified oversampler so we can enforce a per-pair floor rather
    # than treating every override equally.
    override_pairs: list[tuple[str, str]] = []
    for local_id, global_bid in enumerate(local_to_global):
        if global_bid not in targets:
            continue
        gt = targets[global_bid]
        hint = classified[global_bid].role
        entry = {"id": local_id, "role": gt}
        target_entries.append(entry)
        if gt != hint:
            override_count += 1
            override_pairs.append((hint, gt))
    assistant_msg = json.dumps(target_entries, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "n_blocks": len(chunk.block_ids),
        "n_covered": len(covered),
        "n_overrides": override_count,
        "override_rate": override_count / len(covered) if covered else 0.0,
        "override_pairs": override_pairs,   # list[(hint, actual)]
    }


def process_pair_cpu(pair_path_str: str,
                     pages_per_chunk: int = 4,
                     overlap: int = 1,
                     max_blocks_per_chunk: int | None = 60,
                     use_glm_ocr: bool = False) -> dict:
    """CPU worker. Returns a bundle the main process can classify + assemble.

    Runs in a worker process. Does everything that doesn't need the GPU:
      - extract + featurize
      - parse HTML ground-truth blocks
      - align classifier-feature text to HTML blocks (jaccard, the biggest
        per-pair CPU cost on big arxiv papers)
      - compute page chunks from feature page numbers

    Does NOT invoke the classifier — DistilBERT runs batched in the main
    process where the GPU model is loaded once. This split keeps the parent
    from pegging a single core on align+chunk while workers idle.

    Assumes the PDF is either at pair["local_pdf"] (arXiv, form PDFs) or
    in the prerender cache (Wikipedia, etc.). Run scripts/datasets/prerender_pairs.py
    before this to warm the cache.

    Returns dict with either {features, html_blocks, targets, chunks} or
    {error: "..."}.
    """
    try:
        pair = json.loads(Path(pair_path_str).read_text())
    except Exception as exc:
        return {"pair_path": pair_path_str, "error": f"read: {exc}"}

    output_html = pair.get("output_html")
    if not output_html:
        return {"pair_path": pair_path_str, "error": "no_html"}

    local_pdf = pair.get("local_pdf")
    if local_pdf and Path(local_pdf).exists():
        pdf_path = Path(local_pdf)
    else:
        cache_pdf = cache_path_for(output_html)
        if not cache_pdf.exists():
            return {"pair_path": pair_path_str, "error": "no_pdf_no_cache"}
        pdf_path = cache_pdf

    try:
        shared = extract_shared_cached(pdf_path)
    except Exception as exc:
        return {"pair_path": pair_path_str, "error": f"extract: {exc}"}

    if use_glm_ocr:
        try:
            from semantik_structure.glmocr.region_enrichment.pipeline import enrich_from_cache
            enrich_from_cache(shared, pdf_path)
        except Exception as exc:
            return {"pair_path": pair_path_str, "error": f"glm_ocr_cache: {exc}"}

    features = featurize_from_shared(shared)
    if not features:
        return {"pair_path": pair_path_str, "error": "no_features"}

    # Guard against pathological inputs: corrupt PDFs can featurize into
    # hundreds of thousands of tiny blocks, and the main process's
    # per-block serialize loop stalls for minutes on them. Most real
    # papers are <20K features; 50K is generous and skips only outliers.
    MAX_FEATURES = 25_000
    if len(features) > MAX_FEATURES:
        return {"pair_path": pair_path_str,
                "error": f"too_many_features: {len(features)} > {MAX_FEATURES}"}

    html_blocks = extract_html_blocks(output_html)
    if not html_blocks:
        return {"pair_path": pair_path_str, "error": "no_html_blocks"}

    # Align features (not classifier output — we only use .raw.text for
    # jaccard matching) to HTML ground-truth blocks here, off the main
    # process. `align_to_ground_truth` walks each feature block with a
    # sliding window; cost grows with block count.
    targets = align_to_ground_truth(features, html_blocks)
    if not targets:
        return {"pair_path": pair_path_str, "error": "no_targets"}

    pages = [fb.raw.page for fb in features]
    chunks = build_page_chunks(
        pages, pages_per_chunk=pages_per_chunk, overlap=overlap,
        max_blocks_per_chunk=max_blocks_per_chunk,
    )

    # Collect GLM-OCR regions by page so the main process can classify
    # their cleaned text alongside the regular feature blocks. Only the
    # minimum fields (page/kind/bbox/text) travel — the raw page dicts
    # can be large.
    glm_ocr_by_page: dict[int, list[dict]] = {}
    if use_glm_ocr:
        for pg in shared.get("pages") or []:
            rows = pg.get("glm_ocr") or []
            if not rows:
                continue
            kept = [{"kind": r.get("kind"), "bbox": r.get("bbox"),
                     "text": r.get("text") or "",
                     "source_text": r.get("source_text") or ""}
                    for r in rows if (r.get("text") or "").strip()]
            if kept:
                glm_ocr_by_page[int(pg["page_num"])] = kept

    return {
        "pair_path": pair_path_str,
        "pdf_path": str(pdf_path),
        "features": features,
        "html_blocks": html_blocks,
        "targets": targets,
        "chunks": chunks,
        "glm_ocr_by_page": glm_ocr_by_page,
    }


def build_examples_for_pair(classifier, bundle: dict) -> list[dict]:
    """Main-process half of the pipeline: batched classify + assemble.

    The worker already did extract + featurize + align + chunk (see
    `process_pair_cpu`). All that's left here is the GPU classify pass
    and per-chunk example assembly.

    When `bundle["glm_ocr_by_page"]` is populated (i.e. --glm-ocr was
    enabled), the GLM-OCR region text for each math/table region is
    classified by DistilBERT here — that's the stage 3a "review" of
    stage-1 OCR output, so Qwen sees both the text and a structural
    hint for it.
    """
    features = bundle["features"]
    targets = bundle["targets"]
    chunks = bundle["chunks"]
    glm_ocr_by_page: dict[int, list[dict]] = bundle.get("glm_ocr_by_page") or {}

    classified = classify_blocks(features, model=classifier)

    if glm_ocr_by_page:
        _classify_glm_ocr_regions(classifier, features, glm_ocr_by_page)

    out: list[dict] = []
    for chunk in chunks:
        ex = build_chunk_example(classified, chunk, targets,
                                 chunk_count=len(chunks),
                                 glm_ocr_by_page=glm_ocr_by_page)
        if ex:
            out.append(ex)
    return out


def _classify_glm_ocr_regions(classifier,
                              features,
                              glm_ocr_by_page: dict[int, list[dict]]) -> None:
    """Run DistilBERT on each region's OCR text and stash the predicted
    role + confidence onto the region dict in-place."""
    from semantik_structure.types import FeatureBlock, RawBlock

    # Page metrics: we pass these to the synthetic RawBlocks so the
    # classifier's input-serializer sees plausible geometry.
    page_dims: dict[int, tuple[float, float]] = {}
    for fb in features:
        p = fb.raw.page
        if p not in page_dims:
            page_dims[p] = (fb.raw.page_width, fb.raw.page_height)

    flat_fbs: list[FeatureBlock] = []
    back_refs: list[dict] = []
    for page_num, regions in glm_ocr_by_page.items():
        pw, ph = page_dims.get(page_num, (612.0, 792.0))
        for r in regions:
            bbox = tuple(float(v) for v in r.get("bbox") or (0, 0, 0, 0))
            raw = RawBlock(text=r["text"], page=page_num, bbox=bbox,
                           page_width=pw, page_height=ph, source="glm_ocr")
            fb = FeatureBlock(
                raw=raw, size_bucket="md", gap_above=None,
                is_top_of_page=False, is_centered=False, caps=None,
                indent_bucket=0, relative_font_ratio=1.0,
                in_table=(r.get("kind") == "table"),
                in_header_row=False, in_widget=False,
                provenance="glm_ocr",
            )
            flat_fbs.append(fb)
            back_refs.append(r)

    if not flat_fbs:
        return
    classified_regions = classify_blocks(flat_fbs, model=classifier)
    for cb, ref in zip(classified_regions, back_refs):
        ref["distilbert_role"] = cb.role
        ref["distilbert_confidence"] = float(cb.confidence)


def _confusion_counts(examples: list[dict]) -> Counter:
    """Count (hint, actual) pairs across every override in the dataset."""
    c: Counter = Counter()
    for ex in examples:
        for hint, actual in ex.get("override_pairs", []):
            c[(hint, actual)] += 1
    return c


def stratified_oversample(
    examples: list[dict],
    *,
    target_frac: float,
    pair_floor: int,
    per_pair_target: int,
    max_dup_per_chunk: int,
    seed: int = 13,
) -> list[dict]:
    """Oversample override chunks with two safeguards:

    1. **Per-pair floor.** For each confusion pair (hint, actual) that
       already appears in >= `pair_floor` chunks, we bring the total
       count of chunks containing that pair up to at least
       `per_pair_target`. This prevents any single frequent confusion
       (e.g. the `table_header_cell → table_data_cell` pattern from the
       25% heuristic artifact) from being drowned out by other overrides.

    2. **Diversity cap.** No single chunk can be duplicated more than
       `max_dup_per_chunk` times, so we never teach Qwen to override
       based on one memorized chunk's pattern.

    After per-pair topping-up, if overall override-chunk frequency is
    still below `target_frac`, we uniform-sample remaining duplicates
    from the full override pool (still subject to max_dup).
    """
    if not examples:
        return examples

    rng = random.Random(seed)
    override_chunks = [ex for ex in examples if ex.get("n_overrides", 0) > 0]
    if not override_chunks:
        return examples

    confusion = _confusion_counts(examples)

    # Index chunks by each (hint, actual) pair they carry.
    chunks_by_pair: dict[tuple[str, str], list[int]] = {}
    for idx, ex in enumerate(examples):
        for pair in ex.get("override_pairs", []):
            chunks_by_pair.setdefault(pair, []).append(idx)

    dup_counts: Counter = Counter()  # per-example-index duplicate count
    duplicates: list[dict] = []

    def _can_dup(idx: int) -> bool:
        return dup_counts[idx] < max_dup_per_chunk

    # Pass 1: per-pair floor. Only topf pairs that already have enough
    # signal — we don't invent signal for tiny classes.
    for pair, count in confusion.items():
        if count < pair_floor:
            continue
        # How many chunks already contain this pair? (a chunk can contain
        # multiple of the same pair; we treat each chunk once for the target.)
        chunk_pool = list(set(chunks_by_pair.get(pair, [])))
        existing = len(chunk_pool)
        need = per_pair_target - existing
        if need <= 0:
            continue
        # Round-robin duplicate from the pool, respecting the per-chunk cap
        # to keep the source mix diverse.
        added = 0
        attempts = 0
        while added < need and attempts < need * 20:
            attempts += 1
            idx = chunk_pool[rng.randrange(len(chunk_pool))]
            if not _can_dup(idx):
                continue
            duplicates.append(examples[idx])
            dup_counts[idx] += 1
            added += 1
        if added < need:
            # Pool is saturated (every chunk at the cap). Log it.
            print(f"[stratified] pair {pair[0]}->{pair[1]} saturated at "
                  f"{existing + added} (target {per_pair_target})",
                  file=sys.stderr)

    # Pass 2: top up to target_frac from the full override pool, uniform.
    if target_frac > 0:
        total = len(examples) + len(duplicates)
        high = len(override_chunks) + len(duplicates)
        current_frac = high / total
        if current_frac < target_frac:
            # k more duplicates needed: (high+k)/(total+k) = target_frac
            #   => k = (target_frac * total - high) / (1 - target_frac)
            k = int((target_frac * total - high) / (1 - target_frac))
            added = 0
            attempts = 0
            # Build index pool restricted to chunks still under the cap.
            pool = [i for i, ex in enumerate(examples)
                    if ex.get("n_overrides", 0) > 0 and _can_dup(i)]
            while added < k and pool and attempts < k * 20:
                attempts += 1
                idx = pool[rng.randrange(len(pool))]
                if not _can_dup(idx):
                    pool = [i for i in pool if _can_dup(i)]
                    continue
                duplicates.append(examples[idx])
                dup_counts[idx] += 1
                added += 1

    return examples + duplicates


def _pair_pdf_path(pair_path_str: str) -> Path | None:
    """Resolve a pair's PDF path — same logic as process_pair_cpu."""
    try:
        pair = json.loads(Path(pair_path_str).read_text())
    except Exception:
        return None
    output_html = pair.get("output_html")
    if not output_html:
        return None
    local_pdf = pair.get("local_pdf")
    if local_pdf and Path(local_pdf).exists():
        return Path(local_pdf)
    cache_pdf = cache_path_for(output_html)
    if cache_pdf.exists():
        return cache_pdf
    return None


def _glm_ocr_prepass(pair_paths: list[Path], model_path: str) -> None:
    """Populate data/glm_ocr_cache/ for every pair's detected region.

    Two-phase:
      1. Scan pairs, detect regions, check cache. If every region is
         already cached, skip the GPU load entirely — critical for
         multi-shard runs where four shards would otherwise each eat
         ~1.8 GB VRAM for nothing.
      2. If any misses, load GLM-OCR and OCR the misses.

    Idempotent.
    """
    from semantik_structure.extract_shared import extract_shared_cached
    from semantik_structure.glmocr.region_enrichment.cache import (
        CacheKey,
        get as cache_get,
        sha256_file,
    )
    from semantik_structure.region_detection import (
        detect_low_conf_tesseract_regions,
        detect_math_regions,
        detect_table_regions,
    )
    from semantik_structure.glmocr.region_enrichment.model import (
        PROMPT_FORMULA,
        PROMPT_TABLE,
        PROMPT_TEXT,
    )

    def _prompt_for(kind: str) -> str:
        return {"math": PROMPT_FORMULA,
                "table": PROMPT_TABLE}.get(kind, PROMPT_TEXT)

    print(f"[glm-ocr] scanning {len(pair_paths)} pairs for cache misses",
          file=sys.stderr)
    t0 = time.time()

    # Phase 1: scan
    pending_pairs: list[Path] = []
    total_regions = 0
    hit_regions = 0
    for p in pair_paths:
        pdf_path = _pair_pdf_path(str(p))
        if pdf_path is None:
            continue
        try:
            shared = extract_shared_cached(pdf_path)
        except Exception:
            continue
        pdf_sha = sha256_file(pdf_path)
        pair_has_miss = False
        for pg in shared.get("pages") or []:
            regions = (
                detect_table_regions(pg)
                + detect_math_regions(pg)
                + detect_low_conf_tesseract_regions(pg)
            )
            for r in regions:
                total_regions += 1
                key = CacheKey(pdf_sha=pdf_sha, page_num=int(pg["page_num"]),
                               bbox=r.bbox, prompt=_prompt_for(r.kind))
                if cache_get(key) is None:
                    pair_has_miss = True
                else:
                    hit_regions += 1
        if pair_has_miss:
            pending_pairs.append(p)

    scan_s = time.time() - t0
    print(f"[glm-ocr] scan done in {scan_s:.1f}s  "
          f"regions_total={total_regions} cached={hit_regions}  "
          f"pairs_needing_ocr={len(pending_pairs)}", file=sys.stderr)

    if not pending_pairs:
        print(f"[glm-ocr] all regions cached — skipping model load",
              file=sys.stderr)
        return

    # Phase 2: OCR the misses
    from semantik_structure.glmocr.region_enrichment.model import load_glm_ocr, unload_glm_ocr
    from semantik_structure.glmocr.region_enrichment.pipeline import enrich_shared
    print(f"[glm-ocr] loading {model_path}", file=sys.stderr)
    glm = load_glm_ocr(model_path)
    try:
        t1 = time.time()
        done = 0
        for p in pending_pairs:
            pdf_path = _pair_pdf_path(str(p))
            if pdf_path is None:
                continue
            try:
                shared = extract_shared_cached(pdf_path)
                enrich_shared(shared, pdf_path, glm)
            except Exception as exc:
                print(f"[glm-ocr] skip {p.name}: {exc}", file=sys.stderr)
            done += 1
            if done % 50 == 0:
                elapsed = (time.time() - t1) / 60
                print(f"[glm-ocr] {done}/{len(pending_pairs)} pending  "
                      f"elapsed={elapsed:.1f}min", file=sys.stderr)
    finally:
        unload_glm_ocr(glm)
    print(f"[glm-ocr] pre-pass done in {(time.time()-t0)/60:.1f}min",
          file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dirs", type=Path, nargs="+",
                    default=[Path("data/pairs/wikipedia"),
                             Path("data/pairs/openstax"),
                             Path("data/pairs/federal_register"),
                             Path("data/pairs/gutenberg"),
                             Path("data/pairs/forms"),
                             Path("data/pairs/arxiv")])
    ap.add_argument("--classifier", type=Path,
                    default=Path("models/classifier_v3/final"),
                    help="Path to DistilBERT v3 (hint source)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/qwen_dataset_v2"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--pages-per-chunk", type=int, default=4)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--max-blocks-per-chunk", type=int, default=60,
                    help="Split page chunks into sub-chunks of at most this "
                         "many blocks. Caps tokens per example so the full "
                         "assistant response fits within the trainer's "
                         "max_length (default 2048). Pass -1 to disable.")
    ap.add_argument("--glm-ocr", action="store_true",
                    help="Run GLM-OCR on detected math/table regions during "
                         "extraction and include the OCR text + DistilBERT's "
                         "classification of it in each chunk's user prompt. "
                         "Requires an initial pre-pass that populates "
                         "data/glm_ocr_cache/ (main process, GPU); workers "
                         "then read cache only.")
    ap.add_argument("--glm-ocr-model", default="zai-org/GLM-OCR")
    ap.add_argument("--glm-ocr-skip-prepass", action="store_true",
                    help="Skip the main-process GLM-OCR pre-pass; workers "
                         "still read glm_ocr_cache. Use when the cache is "
                         "already populated (from a prior run or a "
                         "standalone scripts/datasets/glm_ocr_prepass.py).")
    ap.add_argument("--override-target-frac", type=float, default=0.20,
                    help="Target fraction of examples containing any override")
    ap.add_argument("--pair-floor", type=int, default=20,
                    help="A (hint,actual) confusion pair is eligible for per-pair "
                         "oversampling only when it naturally appears in at least "
                         "this many chunks")
    ap.add_argument("--per-pair-target", type=int, default=200,
                    help="Target chunk count per eligible confusion pair after oversample. "
                         "Prevents one dominant override pattern (e.g. the in_header_row "
                         "artifact) from drowning out rarer confusions.")
    ap.add_argument("--max-dup-per-chunk", type=int, default=5,
                    help="Hard cap on how many times any single chunk can be duplicated. "
                         "Enforces diversity across source documents.")
    default_workers = max(2, min(6, (os.cpu_count() or 4) - 1))
    ap.add_argument("--workers", type=int, default=default_workers,
                    help="CPU worker processes for extract+featurize+align. "
                         "Classifier runs batched in the main process.")
    ap.add_argument("--shard", type=int, default=0,
                    help="0-indexed shard id for parallel runs. Each shard "
                         "processes pairs where (sorted_idx %% num_shards == shard).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total number of parallel shards. When > 1, this "
                         "run only processes its shard and skips final "
                         "oversample+split — run again with --assemble-only "
                         "after all shards finish.")
    ap.add_argument("--assemble-only", action="store_true",
                    help="Skip pair processing; just load every staging "
                         "file, oversample, split, and write train/val/test. "
                         "Use this after all shards have completed.")
    args = ap.parse_args()

    print(f"[load] classifier {args.classifier}", file=sys.stderr)
    # Assemble-only doesn't need the classifier, but keep the load for a
    # clean error if the path is wrong.
    if not args.assemble_only:
        classifier = load_classifier(str(args.classifier))
    else:
        classifier = None

    pair_paths: list[Path] = []
    for d in args.pair_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("*.json"))
        if args.limit_per_source:
            files = files[:args.limit_per_source]
        pair_paths.extend(files)
    # All shards see the *same* pair_paths (sorted). Each shard only
    # processes its own slice, but the final assembly must load every
    # shard's staging output.
    full_pair_paths = pair_paths
    if args.num_shards > 1 and not args.assemble_only:
        pair_paths = [p for i, p in enumerate(full_pair_paths)
                      if i % args.num_shards == args.shard]
        print(f"[shard] shard {args.shard}/{args.num_shards} -> "
              f"{len(pair_paths)}/{len(full_pair_paths)} pairs",
              file=sys.stderr)
    print(f"[plan] {len(pair_paths)} pair files  cpu-workers={args.workers}",
          file=sys.stderr)

    # Per-pair checkpoint dir. Each completed pair's examples are flushed
    # to {out_dir}/_partial/<sha1(pair_path)>.jsonl so that a crash mid-run
    # (WSL reboot, OOM, Ctrl-C) only costs the in-flight pair — rerun
    # picks up from where it left off by skipping pairs whose staging
    # file already exists.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = args.out_dir / "_partial"
    staging_dir.mkdir(parents=True, exist_ok=True)

    def _staging_path(pair_path_str: str) -> Path:
        h = hashlib.sha1(pair_path_str.encode("utf-8")).hexdigest()[:16]
        return staging_dir / f"{h}.jsonl"

    pending_paths: list[Path] = []
    already_done = 0
    for p in pair_paths:
        if _staging_path(str(p)).exists():
            already_done += 1
        else:
            pending_paths.append(p)
    if already_done:
        print(f"[resume] {already_done}/{len(pair_paths)} pairs already "
              f"checkpointed — processing {len(pending_paths)} remaining",
              file=sys.stderr)

    # GLM-OCR pre-pass — populate data/glm_ocr_cache/ for every pending
    # pair's PDF so the worker processes can read cache without a GPU.
    if (args.glm_ocr and pending_paths and not args.assemble_only
            and not args.glm_ocr_skip_prepass):
        _glm_ocr_prepass(pending_paths, args.glm_ocr_model)

    stats: Counter = Counter()
    stage_time = Counter()  # rough per-stage wall-clock aggregation
    start = time.time()

    # Producer pool: CPU-heavy extract + featurize + html-block parse per pair.
    # Consumer (main process): batched GPU classify + align + chunk assembly.
    #
    # BACKPRESSURE: we used to submit every pending pair upfront, which let
    # worker results pile up in completed futures while the main process
    # was still chewing through earlier ones. Big arXiv papers can emit
    # feature lists of 30K+ block dicts, so a backlog of ~50 pending
    # bundles cost ~12 GB RSS and OOM-killed the parent. Cap in-flight
    # futures at `workers * 2` so the queue never grows unbounded.
    if args.assemble_only:
        print("[assemble-only] skipping pair processing", file=sys.stderr)
    elif pending_paths:
        in_flight_cap = max(2, args.workers * 2)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            path_iter = iter(pending_paths)
            in_flight: dict = {}

            def _submit_next() -> bool:
                try:
                    p = next(path_iter)
                except StopIteration:
                    return False
                _mbpc = args.max_blocks_per_chunk if args.max_blocks_per_chunk > 0 else None
                in_flight[ex.submit(
                    process_pair_cpu, str(p),
                    args.pages_per_chunk, args.overlap, _mbpc,
                    args.glm_ocr,
                )] = str(p)
                return True

            # Prime the pool with the initial batch.
            for _ in range(in_flight_cap):
                if not _submit_next():
                    break

            done = 0
            while in_flight:
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                fut = next(iter(finished))
                pair_path_str = in_flight.pop(fut)
                done += 1
                try:
                    bundle = fut.result()
                except Exception as exc:
                    stats["skip_worker_exc"] += 1
                    _staging_path(pair_path_str).write_text("")
                    _submit_next()
                    continue

                if "error" in bundle:
                    stats[f"skip_{bundle['error'].split(':')[0]}"] += 1
                    _staging_path(pair_path_str).write_text("")
                    _submit_next()
                    continue

                t0 = time.time()
                try:
                    pair_examples = build_examples_for_pair(classifier, bundle)
                except Exception as exc:
                    # Classify/assemble failed (e.g. CUDA corruption, OOM on
                    # one huge pair). Skip this pair and continue — cheaper
                    # than restarting the whole run.
                    print(f"[main-error] pair {Path(pair_path_str).name}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    stats["skip_main_exc"] += 1
                    _staging_path(pair_path_str).write_text("")
                    _submit_next()
                    continue
                stage_time["gpu_and_chunk"] += time.time() - t0
                # Drop the worker's feature list before submitting the next
                # pair so its memory gets reclaimed promptly.
                del bundle

                # Atomic per-pair flush: write to .tmp then rename.
                sp = _staging_path(pair_path_str)
                tmp = sp.with_suffix(".jsonl.tmp")
                with tmp.open("w") as f:
                    for pex in pair_examples:
                        f.write(json.dumps(pex, ensure_ascii=False) + "\n")
                tmp.replace(sp)
                n_examples = len(pair_examples)
                del pair_examples

                stats["examples"] += n_examples
                stats["pairs"] += 1
                _submit_next()

                if done % 50 == 0 or done == len(pending_paths):
                    rate = done / max(0.1, (time.time() - start) / 60)
                    print(f"[{done}/{len(pending_paths)}] examples={stats['examples']}  "
                          f"rate={rate:.0f} pairs/min  "
                          f"elapsed={(time.time()-start)/60:.1f}min  "
                          f"in_flight={len(in_flight)}",
                          file=sys.stderr)

    # In sharded mode, a single shard only processed its slice — the other
    # shards write the remaining staging files in parallel. Exit here and
    # let a later `--assemble-only` pass stitch everything together.
    if args.num_shards > 1 and not args.assemble_only:
        print(f"[shard-done] shard {args.shard}/{args.num_shards} finished; "
              f"run with --assemble-only to build train/val/test",
              file=sys.stderr)
        return

    # Load every staged example back for oversampling + split. Use the
    # full (unsharded) pair list so assemble-only sees every shard's output.
    print(f"[load-staging] reading {len(full_pair_paths)} checkpoint files",
          file=sys.stderr)
    examples: list[dict] = []
    for p in full_pair_paths:
        sp = _staging_path(str(p))
        if not sp.exists():
            continue
        for line in sp.read_text().splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            # JSON has no tuples — override_pairs round-trips as list[list].
            # Restore tuples so they can key dicts in stratified_oversample.
            if ex.get("override_pairs"):
                ex["override_pairs"] = [tuple(p) for p in ex["override_pairs"]]
            examples.append(ex)

    if not examples:
        raise SystemExit("no Qwen examples produced")

    # Pre-oversample diagnostic: show the (hint, actual) confusion matrix
    # so the caller can see what Qwen will be taught to correct before
    # we start duplicating.
    pre_confusion = _confusion_counts(examples)
    print(f"\n[confusion] natural distribution of override (hint -> actual):",
          file=sys.stderr)
    for (hint, actual), n in pre_confusion.most_common(20):
        tag = " *" if n >= args.pair_floor else ""
        print(f"  {hint:22} -> {actual:22}  {n}{tag}", file=sys.stderr)
    print(f"  (* = eligible for per-pair oversample "
          f"at floor={args.pair_floor})", file=sys.stderr)

    pre_override_count = sum(1 for ex in examples if ex.get("n_overrides", 0) > 0)
    print(f"\n[override] pre-oversample: {pre_override_count}/{len(examples)} "
          f"({pre_override_count/len(examples)*100:.1f}%) have overrides",
          file=sys.stderr)
    examples = stratified_oversample(
        examples,
        target_frac=args.override_target_frac,
        pair_floor=args.pair_floor,
        per_pair_target=args.per_pair_target,
        max_dup_per_chunk=args.max_dup_per_chunk,
        seed=args.seed,
    )
    post_override_count = sum(1 for ex in examples if ex.get("n_overrides", 0) > 0)
    post_confusion = _confusion_counts(examples)
    print(f"[override] post-oversample: {post_override_count}/{len(examples)} "
          f"({post_override_count/len(examples)*100:.1f}%) have overrides",
          file=sys.stderr)
    print(f"\n[confusion] post-oversample counts (hint -> actual):",
          file=sys.stderr)
    for (hint, actual), n in post_confusion.most_common(20):
        delta = n - pre_confusion.get((hint, actual), 0)
        print(f"  {hint:22} -> {actual:22}  {n}  (+{delta})", file=sys.stderr)

    # Confirm/override balance: how often is DistilBERT *right* for each
    # hint-label that appears in overrides? Surfaces cases where oversampling
    # may be making the override signal dominate the natural confirm signal.
    confirm_counts: Counter = Counter()   # hint -> count of confirmed chunks
    override_counts: Counter = Counter()  # hint -> count of override chunks
    for ex in examples:
        hints_seen: set[str] = set()
        for hint, actual in ex.get("override_pairs", []):
            override_counts[hint] += 1
            hints_seen.add(hint)
        # Can't recover the "hint was right" count without re-parsing the
        # user message; skip that side for now — the override_counts alone
        # tell us whether any one hint class is dominating.
    print(f"\n[balance] per-hint override chunk count (post-oversample):",
          file=sys.stderr)
    for hint, n in override_counts.most_common():
        print(f"  {hint:22} {n}", file=sys.stderr)

    random.Random(args.seed).shuffle(examples)
    n = len(examples)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val - n_test
    splits = {
        "train": examples[:n_train],
        "val":   examples[n_train:n_train + n_val],
        "test":  examples[n_train + n_val:],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps({"messages": row["messages"]},
                                    ensure_ascii=False) + "\n")
        print(f"[write] {path}  {len(rows)} rows", file=sys.stderr)

    print(f"\n[stats] {dict(stats)}", file=sys.stderr)

    # Diagnostic: override distribution across examples.
    rates = Counter()
    for ex in examples:
        # Bucket override rate to nearest 10%.
        b = int(ex.get("override_rate", 0.0) * 10) * 10
        rates[b] += 1
    print("\n[override-rate distribution]", file=sys.stderr)
    for b in sorted(rates.keys()):
        print(f"  ~{b:3}%: {rates[b]}", file=sys.stderr)


if __name__ == "__main__":
    main()
