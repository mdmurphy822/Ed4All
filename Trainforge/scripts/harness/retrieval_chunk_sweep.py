"""Chunk-size / overlap retrieval calibration sweep (read-only; defaults UNCHANGED).

Operator CLI that re-chunks an archived course corpus at a grid of
chunk-size / overlap parameter points, scores each variant's lexical
retrieval (BM25) against a course gold set, and emits a JSON comparison
report. It answers: *would a different chunk size or a sentence-overlap pass
improve retrieval recall/MRR on this corpus?* — without touching the canonical
on-disk chunkset or the chunker's defaults.

Key invariants:

* **Read-only w.r.t. canonical chunksets.** Every variant is chunked
  in-memory; the script NEVER writes ``semantik_chunks/`` / ``imscc_chunks/``.
  The only write is the ``--out`` report.
* **Chunker defaults untouched.** The sweep drives the chunker's existing
  per-call size knobs (``min_chunk_size`` / ``max_chunk_size`` /
  ``target_chunk_size``); it never mutates ``MIN/MAX/TARGET_CHUNK_SIZE`` module
  constants. The KG pipeline is a second consumer of the chunker — any
  proposal to change canonical defaults from sweep findings MUST re-run the KG
  eval (chunk-local-tag trap + graph fragmentation) and is a separate, gated
  wave (master-plan two-consumer gate). WS1 changes NO defaults.
* **Overlap is harness-local.** The canonical chunker has no overlap support.
  A post-pass (:func:`_apply_sentence_overlap`) prepends the last N sentences
  of chunk *i-1* to chunk *i* (within the same ``item_path`` only) as an
  experiment arm — not a contract change to the chunker.
* **Gold answers matched by text-quote containment, not chunk_id.** Variant
  chunk IDs differ from the canonical chunkset, so a retrieved chunk is
  "relevant" iff its normalized text contains a gold passage's normalized
  ``anchor.text_quote``. (This is exactly why the gold-set schema carries a
  text anchor.)
* **Scored through the real retriever.** Each variant is indexed with the
  canonical ``LibV2.tools.libv2.retriever.LazyBM25`` (imported directly, never
  shelled out) so grid points are read as deltas against the on-disk-chunkset
  baseline row.
* **Deterministic, no LLM, no network, no decision capture.** Two runs with
  the same inputs produce byte-identical reports modulo ``generated_at`` and
  timing fields.

Usage::

    python -m Trainforge.scripts.harness.retrieval_chunk_sweep \\
        --course-code <course-slug> \\
        --source-kind semantik \\
        --max-chunk-sizes 400,800,1200 \\
        --target-chunk-sizes 250,500 \\
        --overlap-sentences 0,1,2 \\
        --ks 1,3,5,10 \\
        [--gold-set PATH]   # default: <course>/retrieval_eval/gold_set.json
        [--out PATH]        # default: <course>/retrieval_eval/chunk_sweep_report.json

The default ``--out`` (course-local ``retrieval_eval/chunk_sweep_report.json``)
is operator calibration data, not code — don't commit generated reports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Shared text normalization / containment (Executor A's module — import-only).
from lib.retrieval._text import normalize_ws, shingle_containment

SCHEMA_VERSION = "1.0"

#: Two-consumer gate reminder surfaced in the report + module docstring.
_TWO_CONSUMER_NOTE = (
    "Read-only calibration sweep. The chunker is shared by the KG pipeline; "
    "any proposal to change canonical chunk-size defaults from these findings "
    "must re-run the KG eval (chunk-local-tag trap + graph fragmentation) and "
    "is a separate, gated wave. This sweep changes no chunker defaults."
)

#: Sentence splitter mirrors the canonical chunker's
#: ``split_by_sentences`` regex so the overlap pass segments text the same way
#: the chunker does.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Default search roots for archived source pages, per source kind. Mirrors the
# resolve order documented for the citation anchor.
_SEMANTIK_SOURCE_ROOTS = ("sources/textbooks", "source/html", "source/dart")


# --------------------------------------------------------------------------- #
# Parameter parsing
# --------------------------------------------------------------------------- #
def _parse_int_csv(raw: str, *, flag: str, positive: bool = True) -> List[int]:
    """Parse a comma-separated int list; validate positivity."""
    out: List[int] = []
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            val = int(tok)
        except ValueError as exc:
            raise ValueError(f"{flag}: {tok!r} is not an integer") from exc
        if positive and val <= 0:
            raise ValueError(f"{flag}: {val} must be a positive integer")
        if not positive and val < 0:
            raise ValueError(f"{flag}: {val} must be a non-negative integer")
        out.append(val)
    if not out:
        raise ValueError(f"{flag}: at least one value required")
    # Deterministic, de-duplicated, ascending.
    return sorted(set(out))


@dataclass(frozen=True)
class SweepParams:
    course_code: str
    course_slug: str
    source_kind: str
    max_chunk_sizes: Tuple[int, ...]
    target_chunk_sizes: Tuple[int, ...]
    overlap_sentences: Tuple[int, ...]
    ks: Tuple[int, ...]
    min_chunk_size: int


# --------------------------------------------------------------------------- #
# Source loading
# --------------------------------------------------------------------------- #
def _course_dir(course_slug: str) -> Path:
    """Resolve the LibV2 course dir via lib.paths (honors ED4ALL_LIBV2_ROOT)."""
    from lib.paths import libv2_path  # lazy: keeps import-time slim

    return libv2_path() / "courses" / course_slug


def _iter_source_html(course_dir: Path, source_kind: str) -> List[Tuple[str, str]]:
    """Return [(item_path, html_text), ...] for the archived source pages.

    For ``semantik`` (and the legacy ``dart`` read alias) source kind, walks
    the SemantiK HTML roots. For ``imscc`` / ``corpus``, reads HTML members
    from any archived ``*.imscc`` under ``source/imscc`` (and the pre-unpacked
    dir). Deterministic ordering.
    """
    pages: List[Tuple[str, str]] = []
    if source_kind in ("semantik", "dart"):
        for rel in _SEMANTIK_SOURCE_ROOTS:
            root = course_dir / rel
            if not root.exists():
                continue
            for html_path in sorted(root.rglob("*.html")):
                try:
                    text = html_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                pages.append((html_path.name, text))
            if pages:
                break
    else:  # imscc / corpus
        import zipfile

        imscc_root = course_dir / "source" / "imscc"
        archives = sorted(imscc_root.glob("*.imscc")) if imscc_root.exists() else []
        for arc in archives:
            try:
                with zipfile.ZipFile(arc) as zf:
                    for member in sorted(zf.namelist()):
                        if member.lower().endswith((".html", ".htm")):
                            try:
                                raw = zf.read(member).decode("utf-8", "replace")
                            except KeyError:
                                continue
                            pages.append((member, raw))
            except (zipfile.BadZipFile, OSError):
                continue
            if pages:
                break
        # Pre-unpacked fallback.
        if not pages:
            unpacked = imscc_root / "_wave81_unpacked"
            if unpacked.exists():
                for html_path in sorted(unpacked.rglob("*.html")):
                    try:
                        text = html_path.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    rel = str(html_path.relative_to(unpacked))
                    pages.append((rel, text))
    return pages


def _build_parsed_items(
    pages: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Parse each (item_path, html) into a chunker-ready parsed_item dict.

    Mirrors the minimal parsed_item shape ``_run_semantik_chunking`` threads into
    the chunker (item_id / item_path / module_id / sections / key_concepts /
    raw_html). Pipeline_tools is NOT imported (it pulls the whole MCP surface
    and is churn-fenced).
    """
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    parsed_items: List[Dict[str, Any]] = []
    for item_path, html in pages:
        try:
            parsed = parser.parse(html)
        except Exception:  # noqa: BLE001 — a bad page shouldn't abort the sweep
            continue
        slug = Path(item_path).stem.lower().replace(" ", "-")
        parsed_items.append(
            {
                "item_id": slug,
                "item_path": item_path,
                "title": parsed.title or slug,
                "resource_type": "page",
                "module_id": slug,
                "module_title": parsed.title or slug,
                "week_num": 0,
                "word_count": parsed.word_count,
                "sections": parsed.sections,
                "learning_objectives": parsed.learning_objectives,
                "key_concepts": parsed.key_concepts,
                "raw_html": html,
            }
        )
    return parsed_items


def _minimal_create_chunk(
    *,
    chunk_id: str,
    text: str,
    html: str,
    item: Dict[str, Any],
    section_heading: str,
    chunk_type: str,
    follows_chunk_id: Optional[str] = None,
    position_in_module: int = 0,
    html_xpath: Optional[str] = None,
    char_span: Optional[List[int]] = None,
    section_source_ids: Optional[List[str]] = None,
    merged_headings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Materialise a retrieval-scoring-minimal v4 chunk dict.

    Cribbed from ``MCP/tools/pipeline_tools.py::_run_semantik_chunking``'s
    ``_create_chunk`` — id / text / html / source / word_count only. Retrieval
    scoring (BM25 + gold text-quote containment) needs nothing else: no concept
    tags, bloom, LO refs, or per-page SHA. Keeping it minimal also keeps the
    sweep deterministic (no env-flag-dependent tag extraction).
    """
    word_count = len(text.split())
    source: Dict[str, Any] = {
        "module_id": item.get("module_id") or "",
        "module_title": item.get("module_title") or "",
        "lesson_id": item.get("item_id") or "",
        "lesson_title": item.get("title") or "",
        "section_heading": section_heading,
        "position_in_module": position_in_module,
    }
    if item.get("item_path"):
        source["item_path"] = item["item_path"]
    if html_xpath:
        source["html_xpath"] = html_xpath
    if char_span is not None:
        source["char_span"] = list(char_span)
    return {
        "id": chunk_id,
        "schema_version": "v4",
        "chunk_type": chunk_type,
        "text": text,
        "html": html,
        "follows_chunk": follows_chunk_id,
        "source": source,
        "word_count": word_count,
    }


def _chunk_variant(
    parsed_items: List[Dict[str, Any]],
    course_code: str,
    *,
    min_chunk_size: int,
    max_chunk_size: int,
    target_chunk_size: int,
) -> List[Dict[str, Any]]:
    """Chunk parsed_items at one (min,max,target) grid point. In-memory only."""
    from Trainforge.chunker import ChunkerContext, chunk_content

    result = chunk_content(
        parsed_items,
        course_code,
        ctx=ChunkerContext(create_chunk=_minimal_create_chunk),
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        target_chunk_size=target_chunk_size,
    )
    return list(result.chunks)


# --------------------------------------------------------------------------- #
# Harness-local overlap (NOT a chunker change)
# --------------------------------------------------------------------------- #
def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries (mirrors the chunker's regex)."""
    return [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _apply_sentence_overlap(
    chunks: List[Dict[str, Any]], n_sentences: int
) -> List[Dict[str, Any]]:
    """Prepend the last ``n_sentences`` of chunk i-1 to chunk i's text.

    Overlap is applied only WITHIN the same ``item_path`` (never across a page
    boundary). ``n_sentences == 0`` is the identity (returns deep-ish copies so
    the input chunk list is never mutated). New chunk dicts are returned; the
    ``word_count`` is recomputed for overlapped chunks.
    """
    if n_sentences < 0:
        raise ValueError("overlap n_sentences must be >= 0")
    out: List[Dict[str, Any]] = []
    prev_item_path: Optional[str] = None
    prev_tail: List[str] = []
    for chunk in chunks:
        new_chunk = dict(chunk)
        new_chunk["source"] = dict(chunk.get("source") or {})
        item_path = new_chunk["source"].get("item_path")
        text = chunk.get("text", "")
        if n_sentences > 0 and prev_item_path == item_path and prev_tail:
            prefix = " ".join(prev_tail)
            new_text = f"{prefix} {text}".strip() if text else prefix
            new_chunk["text"] = new_text
            new_chunk["word_count"] = len(new_text.split())
        out.append(new_chunk)
        # Compute this chunk's tail (from its ORIGINAL text) for the next one.
        if n_sentences > 0:
            sentences = _split_sentences(text)
            prev_tail = sentences[-n_sentences:] if sentences else []
            prev_item_path = item_path
    return out


# --------------------------------------------------------------------------- #
# Gold-set scoring
# --------------------------------------------------------------------------- #
def _load_gold_questions(gold_path: Path) -> List[Dict[str, Any]]:
    """Load the gold-set questions with their normalized text quotes.

    Returns [{question_text, quotes: [normalized_lower_quote, ...]}, ...].
    A question with zero usable quotes is dropped (can't score by containment).
    """
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for q in gold.get("questions", []):
        quotes: List[str] = []
        for passage in q.get("relevant_passages", []) or []:
            anchor = passage.get("anchor") or {}
            quote = anchor.get("text_quote")
            if quote:
                quotes.append(normalize_ws(quote).lower())
        if not quotes:
            continue
        out.append({"question_text": q.get("question_text", ""), "quotes": quotes})
    return out


def _chunk_is_relevant(chunk_text: str, quotes: List[str]) -> bool:
    """A retrieved chunk is relevant iff it contains any gold quote.

    Containment uses substring match after whitespace-normalization +
    lowercasing for the primary check, with a shingle-containment fallback at
    a high threshold so an overlap pass that lightly reflows whitespace around
    the quote still credits the chunk.
    """
    norm = normalize_ws(chunk_text).lower()
    for quote in quotes:
        if quote and quote in norm:
            return True
        # Fallback: most of the quote's shingles present (handles a quote that
        # straddles a chunk-internal whitespace/punctuation reflow).
        if quote and shingle_containment(quote, norm, shingle_size=4) >= 0.9:
            return True
    return False


def _score_variant(
    chunks: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    ks: Tuple[int, ...],
) -> Dict[str, Any]:
    """Index ``chunks`` with LazyBM25, score every question, compute recall@k + MRR."""
    from LibV2.tools.libv2.retriever import LazyBM25

    max_k = max(ks)
    t_index_start = time.perf_counter()
    bm25 = LazyBM25(chunks)
    index_seconds = time.perf_counter() - t_index_start

    recall_hits: Dict[int, int] = {k: 0 for k in ks}
    reciprocal_ranks: List[float] = []
    query_times: List[float] = []

    for q in questions:
        quotes = q["quotes"]
        t_q = time.perf_counter()
        results = bm25.search(q["question_text"], limit=max_k)
        query_times.append(time.perf_counter() - t_q)
        # Rank of the first relevant retrieved chunk (1-based), or 0 = none.
        first_relevant_rank = 0
        for rank, (chunk, _score) in enumerate(results, start=1):
            if _chunk_is_relevant(chunk.get("text", ""), quotes):
                first_relevant_rank = rank
                break
        if first_relevant_rank:
            reciprocal_ranks.append(1.0 / first_relevant_rank)
            for k in ks:
                if first_relevant_rank <= k:
                    recall_hits[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    n_q = len(questions) or 1
    recall_at = {str(k): round(recall_hits[k] / n_q, 6) for k in ks}
    mrr = round(sum(reciprocal_ranks) / n_q, 6)
    words = [c.get("word_count", len(str(c.get("text", "")).split())) for c in chunks]
    mean_words = round(sum(words) / len(words), 4) if words else 0.0
    query_times_sorted = sorted(query_times)
    p50 = (
        round(query_times_sorted[len(query_times_sorted) // 2], 6)
        if query_times_sorted
        else 0.0
    )
    return {
        "n_chunks": len(chunks),
        "mean_words": mean_words,
        "recall_at": recall_at,
        "mrr": mrr,
        "index_seconds": round(index_seconds, 6),
        "query_seconds_p50": p50,
    }


# --------------------------------------------------------------------------- #
# Sweep driver
# --------------------------------------------------------------------------- #
def _load_baseline_chunks(course_dir: Path, source_kind: str) -> Optional[List[Dict]]:
    """Load the canonical on-disk chunkset for the baseline row, if present."""
    from lib.libv2_storage import resolve_imscc_chunks_path

    if source_kind == "semantik":
        path = course_dir / "semantik_chunks" / "chunks.jsonl"
    elif source_kind == "dart":
        # Legacy read-compat: pre-migration corpora shipped dart_chunks/.
        path = course_dir / "dart_chunks" / "chunks.jsonl"
    else:
        # imscc/corpus: use the file-aware resolver (imscc -> dart -> corpus).
        path = resolve_imscc_chunks_path(course_dir)
    if not path.exists():
        return None
    chunks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return chunks or None


def run_sweep(params: SweepParams, gold_path: Path) -> Dict[str, Any]:
    """Execute the full sweep and return the report dict (does not write it)."""
    course_dir = _course_dir(params.course_slug)
    questions = _load_gold_questions(gold_path)
    if not questions:
        raise ValueError(
            f"gold set at {gold_path} has no questions with text_quote anchors; "
            "cannot score retrieval by containment"
        )

    pages = _iter_source_html(course_dir, params.source_kind)
    if not pages:
        raise ValueError(
            f"no archived source pages found for course {params.course_slug!r} "
            f"(kind={params.source_kind!r}) under {course_dir}"
        )
    parsed_items = _build_parsed_items(pages)
    if not parsed_items:
        raise ValueError(
            f"source pages for {params.course_slug!r} parsed to zero items"
        )

    results: List[Dict[str, Any]] = []
    # Deterministic grid order: max asc, target asc, overlap asc.
    for max_size in params.max_chunk_sizes:
        for target_size in params.target_chunk_sizes:
            base_chunks = _chunk_variant(
                parsed_items,
                params.course_code,
                min_chunk_size=params.min_chunk_size,
                max_chunk_size=max_size,
                target_chunk_size=target_size,
            )
            for overlap in params.overlap_sentences:
                variant_chunks = _apply_sentence_overlap(base_chunks, overlap)
                metrics = _score_variant(variant_chunks, questions, params.ks)
                results.append(
                    {
                        "params": {
                            "min_chunk_size": params.min_chunk_size,
                            "max_chunk_size": max_size,
                            "target_chunk_size": target_size,
                            "overlap_sentences": overlap,
                        },
                        **metrics,
                    }
                )

    # Baseline row from the canonical on-disk chunkset (if archived).
    baseline: Optional[Dict[str, Any]] = None
    baseline_chunks = _load_baseline_chunks(course_dir, params.source_kind)
    if baseline_chunks:
        baseline = {
            "params": "canonical-on-disk",
            **_score_variant(baseline_chunks, questions, params.ks),
        }

    grid = [
        {
            "min_chunk_size": params.min_chunk_size,
            "max_chunk_size": m,
            "target_chunk_size": t,
            "overlap_sentences": o,
        }
        for m in params.max_chunk_sizes
        for t in params.target_chunk_sizes
        for o in params.overlap_sentences
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "course_slug": params.course_slug,
        "source_kind": params.source_kind,
        "ks": list(params.ks),
        "n_questions": len(questions),
        "n_source_pages": len(pages),
        "note": _TWO_CONSUMER_NOTE,
        "grid": grid,
        "results": results,
        "baseline": baseline,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _slug_from_course_code(course_code: str) -> str:
    return course_code.lower().replace("_", "-").replace(" ", "-")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m Trainforge.scripts.harness.retrieval_chunk_sweep",
        description=(
            "Read-only chunk-size/overlap retrieval calibration sweep. "
            "Never writes the canonical chunkset; never changes chunker defaults."
        ),
    )
    p.add_argument("--course-code", required=True, help="Course code or slug.")
    p.add_argument(
        "--source-kind",
        choices=("semantik", "dart", "imscc", "corpus"),
        default="semantik",
        help=(
            "Which archived chunkset/source to re-chunk (default: semantik). "
            "'dart' is a legacy read-compat alias for pre-migration corpora."
        ),
    )
    p.add_argument("--max-chunk-sizes", default="400,800,1200")
    p.add_argument("--target-chunk-sizes", default="250,500")
    p.add_argument("--overlap-sentences", default="0,1,2")
    p.add_argument("--ks", default="1,3,5,10")
    p.add_argument(
        "--min-chunk-size",
        type=int,
        default=None,
        help="Min chunk size (default: chunker MIN_CHUNK_SIZE).",
    )
    p.add_argument("--gold-set", default=None, help="Gold set JSON path.")
    p.add_argument("--out", default=None, help="Report output path.")
    return p


def parse_params(args: argparse.Namespace) -> Tuple[SweepParams, Path, Path]:
    """Validate CLI args into SweepParams + resolved gold/out paths."""
    from Trainforge.chunker import MIN_CHUNK_SIZE

    course_code = args.course_code
    course_slug = _slug_from_course_code(course_code)
    course_dir = _course_dir(course_slug)

    min_chunk_size = (
        args.min_chunk_size if args.min_chunk_size is not None else MIN_CHUNK_SIZE
    )
    if min_chunk_size <= 0:
        raise ValueError("--min-chunk-size must be a positive integer")

    max_sizes = _parse_int_csv(args.max_chunk_sizes, flag="--max-chunk-sizes")
    target_sizes = _parse_int_csv(args.target_chunk_sizes, flag="--target-chunk-sizes")
    overlaps = _parse_int_csv(
        args.overlap_sentences, flag="--overlap-sentences", positive=False
    )
    ks = _parse_int_csv(args.ks, flag="--ks")

    # Sanity: target must not exceed max (the chunker only sentence-splits when
    # word_count > max; a target above max is meaningless for the sweep).
    for t in target_sizes:
        if t > max(max_sizes):
            raise ValueError(
                f"--target-chunk-sizes value {t} exceeds the largest "
                f"--max-chunk-sizes value {max(max_sizes)}"
            )

    params = SweepParams(
        course_code=course_code,
        course_slug=course_slug,
        source_kind=args.source_kind,
        max_chunk_sizes=tuple(max_sizes),
        target_chunk_sizes=tuple(target_sizes),
        overlap_sentences=tuple(overlaps),
        ks=tuple(ks),
        min_chunk_size=min_chunk_size,
    )

    gold_path = (
        Path(args.gold_set)
        if args.gold_set
        else course_dir / "retrieval_eval" / "gold_set.json"
    )
    out_path = (
        Path(args.out)
        if args.out
        else course_dir / "retrieval_eval" / "chunk_sweep_report.json"
    )
    return params, gold_path, out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        params, gold_path, out_path = parse_params(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not gold_path.exists():
        print(f"error: gold set not found at {gold_path}", file=sys.stderr)
        return 2
    try:
        report = run_sweep(params, gold_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {out_path} — {len(report['results'])} grid rows, "
        f"{report['n_questions']} questions, "
        f"baseline={'present' if report['baseline'] else 'absent'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
