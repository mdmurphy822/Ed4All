---
name: code-search
description: Hybrid (BM25 + semantic) search over the tracked codebase via the local code index. Use for fuzzy-concept queries — "where is X decided", "code related to Y" — when the exact symbol or filename is unknown. NOT for absence-of-reference claims (sanitization, orphan detection): those must use git grep, which is exhaustive; the index is ranked and approximate.
---

# code-search

Query the local hybrid code index (`runtime/state/code_index/`, gitignored):

```bash
python scripts/code_index.py query "<natural-language or symbol-ish query>" -k 8
```

Useful flags: `--paths-only` (no snippets), `-k N` (result count),
`--per-file N` (max hits per file, default 2), `--snippet N` (snippet lines).

Interpreting output: each hit is `path:start-end` with a semantic score
(cosine) and a BM25 score. High BM25 + low sem = exact-token match; high sem +
low BM25 = conceptual neighbor. Follow up with Read/Grep on the hit — the
index locates, it does not prove.

Freshness: `python scripts/code_index.py status` (nonzero exit = stale vs git
HEAD). If stale after significant changes, rebuild — cheap because unchanged
chunks reuse their vectors:

```bash
python scripts/code_index.py build          # CPU by default; --device cuda only when the card is free
```

Constraints baked in: offline (local `all-MiniLM-L6-v2` only), CPU by
default (the GPU is usually leased), index dir never tracked by git.
