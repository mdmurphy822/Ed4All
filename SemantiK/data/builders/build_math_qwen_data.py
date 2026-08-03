"""Build the Qwen-math LoRA adapter training dataset.

One row per ``<math>`` element in an ar5iv pair's ``raw_source_html``.
The source side is the EXACT JSON envelope that
``semantik_structure.qwen_specialists.prompts.build_math_request`` emits at
inference time, given a ``Region`` derived from the same math element.
The target side is the literal MathML subtree (so the adapter learns
to emit MathML tokens verbatim, including ``display`` / ``alttext``
attributes).

Splits use the same hash-bucket function as
``scripts/datasets/build_semantic_preservation_dataset.py`` so an arxiv_id that
trains the semantic_preservation head won't appear in this builder's
val/test (and vice versa). Cross-builder leakage would silently inflate
held-out scores once the two heads run together.

CPU-ONLY in the sense that it loads no GPU weights — but it DOES
load the Qwen tokenizer (CPU) so it can apply the same chat-template
wrap the trainer/inference path uses, then drop rows whose wrapped
token length exceeds ``--max-token-len`` (default 1024). Without
this filter, ~half the math rows blow past the trainer's
``max_len=1024`` and get silently truncated, killing loss signal on
exactly the long-tail rows we most need.

The tokenizer load happens inside ``main()``, not at import time, so
``import data.builders.build_math_qwen_data`` stays GPU-import-free (see
``test_no_gpu_imports_in_builder``).

Usage::

    python data/builders/build_math_qwen_data.py                  # full build
    python data/builders/build_math_qwen_data.py --max-pairs 30 \\
        --out-dir /tmp/qwen_math_smoke                   # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from lxml import etree, html as lxml_html

# Same package path the rest of the repo uses; ensures we import the
# LIVE prompt builder (any drift in prompts.py would surface as a test
# failure in tests/test_qwen_math_dataset_contract.py).
from semantik_structure.qwen_specialists.prompts import build_math_request
from semantik_structure.qwen_specialists.chat_format import wrap_for_qwen

from data.common._splits import stable_split_for_id


# Default tokenizer for length filtering. Hardcoded — the same model the
# trainer/inference path uses; making this a CLI flag invites silent
# drift between build-time and train-time tokenizers.
DEFAULT_FILTER_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


def _load_filter_tokenizer(name: str) -> Any:
    """Lazy-load the HuggingFace tokenizer for length filtering.

    Imported inside the function (not at module top-level) so
    ``import data.builders.build_math_qwen_data`` stays free of any GPU /
    transformers machinery — the CPU-only contract test pins this.
    Raises a clear error if transformers isn't installed; we do NOT
    silently fall back to whitespace-token counts because that
    masking-by-vibe is exactly what got us here.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"transformers is required for tokenizer-aware length filtering "
            f"(--max-token-len). Install transformers or run from the venv "
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
DEFAULT_OUT_DIR = Path("data/qwen_math_dataset")

# Skip thresholds — see Acceptance Gate 8.
MIN_TARGET_CHARS = 50
WS_RE = re.compile(r"\s+")
PUNCT_ONLY_RE = re.compile(r"^[\s\W_]*$")

# Histogram bucket edges (right-exclusive); chosen to expose the long
# tail of large MathML trees without 100s of empty bins.
TOKEN_HIST_EDGES = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]


# --------------------------------------------------------------------------- #
# Skip / counter bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class SkipCounters:
    no_alttext_no_text: int = 0
    target_too_short: int = 0
    alttext_whitespace_or_punct: int = 0
    serialize_error: int = 0
    parse_error: int = 0
    over_max_tokens: int = 0


@dataclass
class BuildStats:
    n_pairs_seen: int = 0
    n_pairs_with_html: int = 0
    n_pairs_with_arxiv_id: int = 0
    n_math_elements: int = 0
    n_rows_emitted: int = 0
    rows_per_split: Counter = field(default_factory=Counter)
    rows_per_display: Counter = field(default_factory=Counter)
    rows_per_arxiv_id: Counter = field(default_factory=Counter)
    src_token_hist: Counter = field(default_factory=Counter)
    tgt_token_hist: Counter = field(default_factory=Counter)
    wrap_token_hist: Counter = field(default_factory=Counter)
    skip: SkipCounters = field(default_factory=SkipCounters)


def _bucket_for(n: int, edges: list[int]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= n < edges[i + 1]:
            return f"[{edges[i]},{edges[i+1]})"
    return f"[{edges[-1]},inf)"


def _ws_token_count(s: str) -> int:
    return len([t for t in WS_RE.split(s.strip()) if t])


# --------------------------------------------------------------------------- #
# Math-element extraction
# --------------------------------------------------------------------------- #


def _norm_display(raw: str | None) -> str:
    """Normalise the ``display`` attr to {"inline","block"}.

    LaTeXML always emits the attribute, but be defensive: anything not
    "block" treats as inline. The ``payload.display`` boolean
    ``build_math_request`` consumes is ``True`` ↔ block.
    """
    v = (raw or "inline").strip().lower()
    return "block" if v == "block" else "inline"


def _ltx_alttext(elem: etree._Element) -> str:
    return (elem.get("alttext") or "").strip()


def _math_text(elem: etree._Element) -> str:
    """Best-effort flat text for a math element; alttext preferred."""
    alt = _ltx_alttext(elem)
    if alt:
        return alt
    txt = " ".join((elem.text_content() if hasattr(elem, "text_content")
                    else "".join(elem.itertext())).split())
    return txt.strip()


# Equation-number patterns (tail/lead) — same shape as the BERT
# extractor. Used only for the metadata field ``eq_num_assoc``; the
# Qwen prompt itself does not consume eq_num_assoc, but we record it
# for future filter / weighting passes.
EQ_NUM_TRAILING_RE = re.compile(r"\(\s*\d+(?:\.\d+)?\s*\)\s*$")
EQ_NUM_LEADING_RE = re.compile(r"^\s*\(\s*\d+(?:\.\d+)?\s*\)")

# LaTeXML class names that scope an equation block.
_EQ_CLASS_TARGETS = {
    "ltx_equation", "ltx_eqn_row", "ltx_eqn_table",
    "ltx_equationgroup", "ltx_para", "ltx_p", "ltx_caption",
}


def _has_class(elem: etree._Element, name: str) -> bool:
    cls = elem.get("class") or ""
    return name in cls.split()


def _ancestor_block(elem: etree._Element) -> etree._Element | None:
    cell_fallback: etree._Element | None = None
    for p in elem.iterancestors():
        cls = (p.get("class") or "").split()
        if any(c in _EQ_CLASS_TARGETS for c in cls):
            return p
        if p.tag in ("p", "li"):
            return p
        if p.tag in ("td", "th") and cell_fallback is None:
            cell_fallback = p
    return cell_fallback


def _eq_num_assoc(elem: etree._Element) -> str:
    """Approximate eq-number association.

    Best-effort only. We mirror the BERT extractor's heuristics so the
    metadata field is comparable across builders, but the Qwen prompt
    does not consume this — so we accept a coarser approximation here.
    """
    block = _ancestor_block(elem)
    if block is None:
        return "none"
    # ltx_tag_equation sibling presence.
    tag_in_block = None
    for d in block.iter():
        cls = (d.get("class") or "").split()
        if "ltx_tag_equation" in cls or "ltx_tag" in cls:
            tag_in_block = d
            break
    if tag_in_block is not None:
        # crude document-order comparison via sourceline + index
        try:
            order = list(block.iter())
            if order.index(tag_in_block) < order.index(elem):
                return "attached_left"
            return "attached_right"
        except ValueError:
            return "separate"
    # Fall back to (N) text patterns on adjacent text/siblings.
    prev = elem.getprevious()
    nxt = elem.getnext()
    prev_text = ""
    next_text = ""
    if prev is not None:
        prev_text = " ".join("".join(prev.itertext()).split())
    elif elem.getparent() is not None and elem.getparent().text:
        prev_text = elem.getparent().text.strip()
    if nxt is not None:
        next_text = " ".join("".join(nxt.itertext()).split())
    elif elem.tail:
        next_text = elem.tail.strip()
    if EQ_NUM_TRAILING_RE.search(next_text):
        return "attached_right"
    if EQ_NUM_LEADING_RE.search(prev_text):
        return "attached_left"
    block_text = " ".join("".join(block.itertext()).split())
    if EQ_NUM_TRAILING_RE.search(block_text) and _norm_display(elem.get("display")) == "block":
        return "separate"
    return "none"


def _math_type(elem: etree._Element, eq_assoc: str) -> str:
    """Reproduce the BERT extractor's math_type rule for the metadata."""
    display = _norm_display(elem.get("display"))
    mtables = elem.findall(".//{*}mtable")
    if not mtables:
        mtables = elem.findall(".//mtable")
    has_mtable = bool(mtables)
    is_matrix = False
    if has_mtable:
        for mt in mtables:
            parent = mt.getparent()
            if (parent is not None
                    and isinstance(parent.tag, str)
                    and parent.tag.endswith("mfenced")):
                is_matrix = True
                break
            prev = mt.getprevious()
            nxt = mt.getnext()
            prev_tag = prev.tag if (prev is not None and isinstance(prev.tag, str)) else ""
            nxt_tag = nxt.tag if (nxt is not None and isinstance(nxt.tag, str)) else ""
            if (prev_tag.endswith("mo")
                    and ("".join(prev.itertext()).strip() in ("(", "[", "|"))
                    and nxt_tag.endswith("mo")
                    and ("".join(nxt.itertext()).strip() in (")", "]", "|"))):
                is_matrix = True
                break
    if is_matrix:
        return "matrix"
    if has_mtable and any(
        len(mt.findall("./{*}mtr") + mt.findall("./mtr")) > 1
        for mt in mtables
    ):
        return "multiline"
    if display == "block" and eq_assoc != "none":
        return "numbered"
    if display == "block":
        return "display"
    return "inline"


def _surrounding_text(elem: etree._Element, *, max_chars: int = 200) -> tuple[str, str]:
    """Return (context_before, context_after) — text immediately around
    the math element within the same paragraph/block.

    We split the surrounding-text window at the math element so train
    rows record both halves separately; the prompt envelope itself
    only embeds ``src_text`` (the alttext), so the context is for
    metadata / future-filter use — but it lets us reconstruct the
    region for diagnostics.
    """
    block = _ancestor_block(elem)
    if block is None:
        block = elem.getparent()
    if block is None:
        return "", ""
    before_chunks: list[str] = []
    after_chunks: list[str] = []
    seen_target = False
    for descendant in block.iter():
        if descendant is elem:
            seen_target = True
            continue
        # Skip comments / processing-instructions — their .tag is a
        # callable, not a string. Filter those before any string ops.
        if not isinstance(descendant.tag, str):
            continue
        # Skip nested math — they get their own row.
        if descendant.tag == "math" or descendant.tag.endswith("}math"):
            continue
        # Pull only direct text from each node, not deep itertext (deep
        # itertext over the whole block would double-count children).
        t = (descendant.text or "")
        if t.strip():
            (after_chunks if seen_target else before_chunks).append(t)
        # Tail text comes after the descendant in document order.
        tail = (descendant.tail or "")
        if tail.strip():
            (after_chunks if seen_target else before_chunks).append(tail)
    before = WS_RE.sub(" ", " ".join(before_chunks)).strip()
    after = WS_RE.sub(" ", " ".join(after_chunks)).strip()
    if len(before) > max_chars:
        before = before[-max_chars:]
    if len(after) > max_chars:
        after = after[:max_chars]
    return before, after


def _serialize_math_subtree(elem: etree._Element) -> str:
    """Return the literal MathML subtree as a string.

    Strips namespace prefixes from the output by re-parsing with
    namespace-cleaning behaviour off — ar5iv emits ``<math
    xmlns="http://www.w3.org/1998/Math/MathML">`` which lxml expands;
    we want the namespace declaration preserved on the root for
    fidelity with the trained-time HTML.
    """
    return etree.tostring(elem, encoding="unicode", with_tail=False)


# --------------------------------------------------------------------------- #
# Per-pair iteration
# --------------------------------------------------------------------------- #


def _iter_math_elements(html_str: str) -> Iterator[etree._Element]:
    """Yield every ``<math>`` element in document order.

    We use ``lxml.html`` to parse (lenient HTML5 — ar5iv pages have
    plenty of tag-soup quirks) but then look up math elements by both
    the bare ``math`` tag and the namespaced form
    ``{http://www.w3.org/1998/Math/MathML}math``: depending on whether
    the page declares the MathML namespace at the root, lxml may
    expand the tag name.
    """
    try:
        root = lxml_html.fromstring(html_str)
    except Exception:
        return iter(())
    # Match by local-name() so namespaced and non-namespaced tags both
    # surface. xpath local-name() is well supported.
    return iter(root.xpath(".//*[local-name()='math']"))


def _build_region_payload(
    *, alttext: str, math_text: str, display: str
) -> dict[str, Any]:
    """Construct the ``Region.payload`` dict ``build_math_request``
    will read.

    Mapping decisions (locked at dataset-build time, contract-tested):
        - ``src_text`` ← alttext, falling back to flattened text. This
          mirrors the live Stage-5 region builder, which uses alttext
          when present.
        - ``display`` ← True for block, False for inline.
        - ``glyph_density_features`` ← {} — ar5iv pages do not carry
          the OCR glyph-density features (those are computed at
          inference time from PDF rasterisation). The dataset thus
          records the empty-dict default; this is documented in the
          coverage_report.json under ``notes.glyph_density_features``.
    """
    src_text = alttext if alttext else math_text
    return {
        "src_text": src_text,
        "display": display == "block",
        "glyph_density_features": {},
    }


def _make_region_namespace(payload: dict[str, Any]) -> SimpleNamespace:
    """Tiny stand-in for ``structure_graph.Region``.

    ``build_math_request`` reads only ``region.payload`` and
    ``region.feature_block_indices``; ``aria_hints`` is referenced via
    ``region.aria_hints or ()`` — None-safe — but math regions don't
    set it. ``region.kind`` is similarly recorded only in metadata. So
    a SimpleNamespace with three attributes is enough; using one
    avoids importing the full Region dataclass (which would drag the
    structure-graph module into the data-builder import graph).
    """
    return SimpleNamespace(
        payload=payload,
        feature_block_indices=(),
        aria_hints=(),
        kind="math",
    )


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def build_row_for_math_elem(
    *,
    elem: etree._Element,
    arxiv_id: str,
    source_pair: str,
    math_idx: int,
    skip: SkipCounters,
    tok: Any,
    max_token_len: int,
) -> dict[str, Any] | None:
    """Return a JSONL-serializable row, or ``None`` to skip.

    Skip rules (Gate 8):
        - empty alttext AND empty math_text
        - target subtree round-trips to <50 chars
        - alttext is whitespace / punctuation only
        - wrapped token length > ``max_token_len`` (the trainer would
          truncate these to silence anyway).

    ``tok`` is the HuggingFace tokenizer used to compute the wrapped
    token count; it must be the same family the trainer uses.
    """
    alttext = _ltx_alttext(elem)
    math_text = _math_text(elem)
    if not alttext and not math_text:
        skip.no_alttext_no_text += 1
        return None
    if alttext and PUNCT_ONLY_RE.match(alttext):
        skip.alttext_whitespace_or_punct += 1
        return None

    try:
        target_html = _serialize_math_subtree(elem)
    except Exception:
        skip.serialize_error += 1
        return None
    if len(target_html) < MIN_TARGET_CHARS:
        skip.target_too_short += 1
        return None

    display = _norm_display(elem.get("display"))
    eq_assoc = _eq_num_assoc(elem)
    math_type = _math_type(elem, eq_assoc)
    ctx_before, ctx_after = _surrounding_text(elem)

    payload = _build_region_payload(
        alttext=alttext, math_text=math_text, display=display,
    )
    region = _make_region_namespace(payload)
    request = build_math_request(region, feature_blocks=())
    request_payload = {
        "adapter": request.adapter.value,
        "payload": request.payload,
        "request_id": request.request_id,
        "sampling_overrides": request.sampling_overrides,
    }

    n_src_tokens = _ws_token_count(request.payload.get("prompt", ""))
    n_tgt_tokens = _ws_token_count(target_html)

    # Tokenizer-aware length filter. We wrap with the SAME function
    # (``wrap_for_qwen``) the trainer/inference path uses, so the
    # token count we measure here is exactly what the trainer will
    # see — no reimplementation, no drift.
    wrapped = wrap_for_qwen(tok, request.payload["prompt"], target=target_html)
    n_wrap_tokens = len(tok(wrapped, add_special_tokens=False)["input_ids"])
    if n_wrap_tokens > max_token_len:
        skip.over_max_tokens += 1
        return None

    return {
        "source_pair": source_pair,
        "arxiv_id": arxiv_id,
        "math_idx": math_idx,
        "display": display,
        "math_type": math_type,
        "eq_num_assoc": eq_assoc,
        "alttext": alttext,
        "context_before": ctx_before,
        "context_after": ctx_after,
        "request_payload": request_payload,
        "target_html": target_html,
        "n_source_tokens": n_src_tokens,
        "n_target_tokens": n_tgt_tokens,
        "n_wrap_tokens": n_wrap_tokens,
        "extraction_metadata": {
            "src_text_origin": "alttext" if alttext else "math_text",
            "has_alttext": bool(alttext),
            "glyph_density_features_origin": "empty_default",
        },
    }


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
                         "length exceeds this. Should match the math "
                         "adapter's max_len in train_config.py.")
    ap.add_argument("--tokenizer-name", type=str,
                    default=DEFAULT_FILTER_TOKENIZER,
                    help="HuggingFace tokenizer used for the length "
                         "filter; must match the trainer's base model.")
    args = ap.parse_args()

    if not args.pairs_dir.exists():
        sys.exit(f"[fatal] pairs dir not found: {args.pairs_dir}")

    rng = random.Random(args.seed)  # reserved for any future shuffling
    _ = rng  # silence the "unused" linter; kept so --seed is honoured
    stats = BuildStats()
    handles = _open_split_files(args.out_dir)

    print(f"[setup] loading filter tokenizer: {args.tokenizer_name}",
          flush=True)
    filter_tok = _load_filter_tokenizer(args.tokenizer_name)
    print(f"[setup] tokenizer loaded; max_token_len={args.max_token_len}",
          flush=True)

    seen_arxiv_ids: dict[str, str] = {}  # id -> split (for split-coherence assert)

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
                # No arxiv_id → cannot place this pair stably; skip (the
                # semantic_preservation builder rejects these the same way).
                continue
            stats.n_pairs_with_arxiv_id += 1

            # Per-arxiv_id dedup: ``data/pairs/arxiv/`` stores ONE pair
            # file per (paper, section) but every pair in the same paper
            # carries the SAME ``raw_source_html``. Without dedup we
            # would emit each <math> element N times where N = number of
            # sections in the paper — silent train-time duplication that
            # both inflates dataset size and over-weights long papers.
            # We process each arxiv_id exactly once.
            if arxiv_id in seen_arxiv_ids:
                continue

            split = stable_split_for_id(arxiv_id, 0.80, 0.10)
            seen_arxiv_ids[arxiv_id] = split

            source_pair = pair_path.stem
            for math_idx, elem in enumerate(_iter_math_elements(html_str)):
                stats.n_math_elements += 1
                row = build_row_for_math_elem(
                    elem=elem,
                    arxiv_id=arxiv_id,
                    source_pair=source_pair,
                    math_idx=math_idx,
                    skip=stats.skip,
                    tok=filter_tok,
                    max_token_len=args.max_token_len,
                )
                if row is None:
                    continue
                row["split"] = split
                handles[split].write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                stats.n_rows_emitted += 1
                stats.rows_per_split[split] += 1
                stats.rows_per_display[row["display"]] += 1
                stats.rows_per_arxiv_id[arxiv_id] += 1
                stats.src_token_hist[
                    _bucket_for(row["n_source_tokens"], TOKEN_HIST_EDGES)
                ] += 1
                stats.tgt_token_hist[
                    _bucket_for(row["n_target_tokens"], TOKEN_HIST_EDGES)
                ] += 1
                stats.wrap_token_hist[
                    _bucket_for(row["n_wrap_tokens"], TOKEN_HIST_EDGES)
                ] += 1

            if (i + 1) % args.progress_every == 0:
                total_skips = (
                    stats.skip.no_alttext_no_text
                    + stats.skip.target_too_short
                    + stats.skip.alttext_whitespace_or_punct
                    + stats.skip.over_max_tokens
                )
                print(
                    f"[progress] pairs={i+1} "
                    f"math_elems={stats.n_math_elements} "
                    f"rows={stats.n_rows_emitted} skips={total_skips} "
                    f"(over_max={stats.skip.over_max_tokens})",
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
        "filter_tokenizer": args.tokenizer_name,
        "n_pairs_seen": stats.n_pairs_seen,
        "n_pairs_with_html": stats.n_pairs_with_html,
        "n_pairs_with_arxiv_id": stats.n_pairs_with_arxiv_id,
        "n_math_elements": stats.n_math_elements,
        "n_rows_emitted": stats.n_rows_emitted,
        "rows_per_split": dict(stats.rows_per_split),
        "rows_per_display": dict(stats.rows_per_display),
        "n_unique_arxiv_ids_in_dataset": len(stats.rows_per_arxiv_id),
        "rows_per_arxiv_id_top20": dict(
            stats.rows_per_arxiv_id.most_common(20)
        ),
        "src_token_hist": dict(sorted(stats.src_token_hist.items())),
        "tgt_token_hist": dict(sorted(stats.tgt_token_hist.items())),
        "wrap_token_hist": dict(sorted(stats.wrap_token_hist.items())),
        "over_max_tokens": stats.skip.over_max_tokens,
        "skip": {
            "no_alttext_no_text": stats.skip.no_alttext_no_text,
            "target_too_short": stats.skip.target_too_short,
            "alttext_whitespace_or_punct": stats.skip.alttext_whitespace_or_punct,
            "serialize_error": stats.skip.serialize_error,
            "parse_error": stats.skip.parse_error,
            "over_max_tokens": stats.skip.over_max_tokens,
        },
        "notes": {
            "glyph_density_features": (
                "ar5iv ground-truth HTML carries no OCR glyph-density "
                "features (they are computed at inference time from PDF "
                "rasterisation). Every dataset row records "
                "glyph_density_features={} as the contract default; "
                "the live build_math_request copies whatever the "
                "Region.payload provides, which is also {} for math "
                "regions whose density features were not computed."
            ),
            "split_function": (
                "data.common._splits.stable_split_for_id (re-implementation of "
                "scripts.build_semantic_preservation_dataset's). "
                "train_frac=0.80, val_frac=0.10. Same arxiv_id -> "
                "same split across both builders."
            ),
            "target_format": (
                "lxml.etree.tostring(<math>) — preserves attributes and "
                "children byte-for-byte from the ar5iv source HTML."
            ),
        },
    }
    (args.out_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2)
    )

    # Summary to stdout.
    print()
    print(f"[done] rows={stats.n_rows_emitted} "
          f"(train={stats.rows_per_split.get('train', 0)} "
          f"val={stats.rows_per_split.get('val', 0)} "
          f"test={stats.rows_per_split.get('test', 0)})")
    print(f"[done] math_elements={stats.n_math_elements} skips="
          f"no_text:{stats.skip.no_alttext_no_text} "
          f"short:{stats.skip.target_too_short} "
          f"punct:{stats.skip.alttext_whitespace_or_punct} "
          f"err:{stats.skip.serialize_error} "
          f"over_max:{stats.skip.over_max_tokens}")
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
