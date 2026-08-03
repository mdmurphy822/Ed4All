#!/usr/bin/env python3
"""Local hybrid (BM25 + semantic) code-search index over the tracked tree.

Purpose: a fast "what code relates to X" query surface for agents and
operators working on this repo. It complements — never replaces — exact
search: absence-of-reference claims (sanitization, orphan detection) stay
`git grep` territory; this index answers fuzzy-concept questions
("where do we filter DPO admissibility?") that exact tokens miss.

Design constraints honored:
- OFFLINE: embeds with the already-local `all-MiniLM-L6-v2` via
  ``lib.embedding.sentence_embedder.SentenceEmbedder`` (the repo's
  standard wrapper); no network, no model download at run time.
- CPU by default (``--device cuda`` opts in when the card is free) —
  the GPU is frequently leased to vLLM seats or training.
- Index lives under ``runtime/state/code_index/`` (gitignored via the
  ``runtime/state/*/*`` defensive catch-all; nothing here may land in git).
- Hybrid retrieval: BM25 over code-aware tokens + cosine over MiniLM,
  fused with reciprocal-rank fusion — pure-semantic never beat lexical
  on this project's own retrieval evals, so neither side runs alone.

Usage:
    python scripts/harness/code_index.py build            # full (re)build, ~minutes on CPU
    python scripts/harness/code_index.py build --changed-only   # re-embed only changed files
    python scripts/harness/code_index.py query "where is DPO admissibility decided" -k 8
    python scripts/harness/code_index.py status           # index freshness vs git HEAD

Re-embedding reuse: unchanged chunks (by content hash) reuse their prior
vectors, so incremental rebuilds are cheap even in full ``build`` mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "runtime" / "state" / "code_index"

# Text/code files worth indexing. Everything else tracked (images, pdf
# fixtures, archives) carries no queryable text.
_INDEXED_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".sh", ".bash",
    ".ttl", ".txt", ".toml", ".cfg", ".ini", ".html", ".css", ".js",
    ".sql", ".xml", ".example", ".env",
}
_MAX_FILE_BYTES = 4 * 1024 * 1024  # only skip truly huge files; the one
# tracked file over the old 512KB cap was MCP/tools/pipeline_tools.py —
# the tool-registry hub, exactly what the index must cover
_CHUNK_LINES = 60
_CHUNK_STRIDE = 45  # 15-line overlap so a symbol split across a seam still hits
_RRF_K = 60

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(text: str) -> List[str]:
    """Code-aware tokens: identifiers, split on snake_case and CamelCase."""
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text):
        low = tok.lower()
        out.append(low)
        parts = [p for p in re.split(r"_+", tok) if p]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
        for part in parts or [tok]:
            camel = [c for c in _CAMEL_RE.split(part) if len(c) > 1]
            if len(camel) > 1:
                out.extend(c.lower() for c in camel)
    return out


def _git(args: List[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        check=True,
    ).stdout


def _tracked_files() -> List[str]:
    return [p for p in _git(["ls-files", "-z"]).split("\0") if p]


def _iter_chunks(rel: str, text: str) -> Iterable[Tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return
    if len(lines) <= _CHUNK_LINES:
        yield 1, len(lines), text
        return
    start = 0
    while start < len(lines):
        window = lines[start : start + _CHUNK_LINES]
        yield start + 1, start + len(window), "\n".join(window)
        if start + _CHUNK_LINES >= len(lines):
            break
        start += _CHUNK_STRIDE


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _load_prior_vectors(index_dir: Path) -> Dict[str, object]:
    """content-sha -> prior embedding row, for re-embed reuse."""
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"
    if not (chunks_path.exists() and emb_path.exists()):
        return {}
    import numpy as np

    try:
        mat = np.load(emb_path)
        prior: Dict[str, object] = {}
        with chunks_path.open() as fh:
            for i, line in enumerate(fh):
                rec = json.loads(line)
                if i < len(mat):
                    prior[rec["sha"]] = mat[i]
        return prior
    except Exception:
        return {}


def cmd_build(args: argparse.Namespace) -> int:
    if args.device == "cpu":
        # Force CPU BEFORE torch import — the card is usually leased elsewhere.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from lib.embedding.sentence_embedder import SentenceEmbedder  # heavy import late

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    prior = {} if args.no_reuse else _load_prior_vectors(index_dir)

    records: List[dict] = []
    texts: List[str] = []
    skipped = 0
    for rel in _tracked_files():
        path = PROJECT_ROOT / rel
        if path.suffix.lower() not in _INDEXED_SUFFIXES:
            skipped += 1
            continue
        try:
            data = path.read_bytes()
        except OSError:
            skipped += 1
            continue
        if len(data) > _MAX_FILE_BYTES or b"\x00" in data:
            skipped += 1
            continue
        text = data.decode("utf-8", errors="ignore")
        for start, end, chunk in _iter_chunks(rel, text):
            records.append(
                {"path": rel, "start": start, "end": end, "sha": _sha(chunk)}
            )
            texts.append(f"{rel}\n{chunk}")

    import numpy as np

    embedder = SentenceEmbedder(model_name=args.model)
    dim: Optional[int] = None
    vectors: List[object] = [None] * len(records)
    to_embed_idx = []
    for i, rec in enumerate(records):
        hit = prior.get(rec["sha"])
        if hit is not None:
            vectors[i] = hit
            dim = len(hit)
        else:
            to_embed_idx.append(i)
    print(
        f"[code-index] {len(records)} chunks from tracked tree "
        f"({len(to_embed_idx)} to embed, {len(records) - len(to_embed_idx)} reused, "
        f"{skipped} files skipped)"
    )
    B = 512
    for off in range(0, len(to_embed_idx), B):
        idx_batch = to_embed_idx[off : off + B]
        mat = embedder.encode_batch(
            [texts[i] for i in idx_batch], normalize=True,
            batch_size=args.batch_size, length_sort=True,
        )
        for j, i in enumerate(idx_batch):
            vectors[i] = mat[j]
        dim = mat.shape[1]
        done = min(off + B, len(to_embed_idx))
        print(f"[code-index] embedded {done}/{len(to_embed_idx)}", flush=True)

    if not records:
        print("[code-index] nothing to index", file=sys.stderr)
        return 1
    matrix = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])

    # BM25 side: inverted index term -> [(chunk_idx, tf)] + doc lengths.
    inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    doc_len = np.zeros(len(records), dtype=np.int32)
    for i, body in enumerate(texts):
        counts = Counter(_tokenize(body))
        doc_len[i] = sum(counts.values())
        for term, tf in counts.items():
            inverted[term].append((i, tf))

    np.save(index_dir / "embeddings.npy", matrix)
    with (index_dir / "chunks.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    with (index_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump(
            {"inverted": dict(inverted), "doc_len": doc_len,
             "avg_len": float(doc_len.mean())},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    (index_dir / "meta.json").write_text(json.dumps({
        "git_head": _git(["rev-parse", "HEAD"]).strip(),
        "model": args.model,
        "dim": dim,
        "chunks": len(records),
        "chunk_lines": _CHUNK_LINES,
        "stride": _CHUNK_STRIDE,
    }, indent=2))
    print(f"[code-index] built: {len(records)} chunks, dim={dim} -> {index_dir}")
    return 0


def _bm25_scores(query_terms: List[str], bm25: dict, n_docs: int) -> object:
    import numpy as np

    k1, b = 1.5, 0.75
    scores = np.zeros(n_docs, dtype=np.float32)
    doc_len = bm25["doc_len"]
    avg = bm25["avg_len"] or 1.0
    for term in set(query_terms):
        postings = bm25["inverted"].get(term)
        if not postings:
            continue
        idf = math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
        for i, tf in postings:
            denom = tf + k1 * (1 - b + b * doc_len[i] / avg)
            scores[i] += idf * tf * (k1 + 1) / denom
    return scores


def cmd_query(args: argparse.Namespace) -> int:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    index_dir = Path(args.index_dir)
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        print(
            "[code-index] no index found — run: python scripts/harness/code_index.py build",
            file=sys.stderr,
        )
        return 2
    import numpy as np

    meta = json.loads(meta_path.read_text())
    head = _git(["rev-parse", "HEAD"]).strip()
    if head != meta.get("git_head"):
        print(
            f"[code-index] NOTE: index built at {meta.get('git_head', '?')[:10]}, "
            f"HEAD is {head[:10]} — results may be stale "
            "(rebuild: python scripts/harness/code_index.py build)",
            file=sys.stderr,
        )
    matrix = np.load(index_dir / "embeddings.npy")
    records = [json.loads(l) for l in (index_dir / "chunks.jsonl").open()]
    with (index_dir / "bm25.pkl").open("rb") as fh:
        bm25 = pickle.load(fh)

    from lib.embedding.sentence_embedder import SentenceEmbedder

    q = SentenceEmbedder(model_name=meta["model"]).encode(args.query, normalize=True)
    sem = matrix @ np.asarray(q, dtype=np.float32)
    lex = _bm25_scores(_tokenize(args.query), bm25, len(records))

    fused = np.zeros(len(records), dtype=np.float32)
    for ranks in (np.argsort(-sem), np.argsort(-lex)):
        for r, i in enumerate(ranks[: args.k * 10]):
            fused[i] += 1.0 / (_RRF_K + r + 1)

    # Collapse overlapping windows: best chunk per file wins a slot first.
    order = np.argsort(-fused)
    shown: List[int] = []
    seen_files: Dict[str, int] = {}
    for i in order:
        if fused[i] <= 0:
            break
        rec = records[i]
        if seen_files.get(rec["path"], 0) >= args.per_file:
            continue
        seen_files[rec["path"]] = seen_files.get(rec["path"], 0) + 1
        shown.append(int(i))
        if len(shown) >= args.k:
            break

    for rank, i in enumerate(shown, start=1):
        rec = records[i]
        print(f"{rank}. {rec['path']}:{rec['start']}-{rec['end']}  "
              f"(sem={sem[i]:.3f} bm25={lex[i]:.1f})")
        if not args.paths_only:
            path = PROJECT_ROOT / rec["path"]
            try:
                lines = path.read_text(errors="ignore").splitlines()
                snippet = lines[rec["start"] - 1 : rec["start"] - 1 + args.snippet]
                for ln, text in enumerate(snippet, start=rec["start"]):
                    print(f"     {ln:>5} | {text[:160]}")
            except OSError:
                pass
            print()
    if not shown:
        print("[code-index] no hits")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    index_dir = Path(args.index_dir)
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        print("[code-index] no index built")
        return 1
    meta = json.loads(meta_path.read_text())
    head = _git(["rev-parse", "HEAD"]).strip()
    fresh = "FRESH" if head == meta.get("git_head") else "STALE"
    print(f"[code-index] {fresh}: {meta['chunks']} chunks, model={meta['model']}, "
          f"index@{meta.get('git_head', '?')[:10]} vs HEAD@{head[:10]}")
    return 0 if fresh == "FRESH" else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="(re)build the index")
    p_build.add_argument("--model", default="all-MiniLM-L6-v2")
    p_build.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--no-reuse", action="store_true",
                         help="ignore prior vectors; embed everything fresh")
    p_build.add_argument("--changed-only", action="store_true",
                         help="alias of the default hash-reuse behavior (kept "
                              "for discoverability)")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="hybrid search")
    p_query.add_argument("query")
    p_query.add_argument("-k", type=int, default=8)
    p_query.add_argument("--per-file", type=int, default=2)
    p_query.add_argument("--snippet", type=int, default=6)
    p_query.add_argument("--paths-only", action="store_true")
    p_query.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p_query.set_defaults(func=cmd_query)

    p_status = sub.add_parser("status", help="index freshness vs git HEAD")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    sys.path.insert(0, str(PROJECT_ROOT))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
