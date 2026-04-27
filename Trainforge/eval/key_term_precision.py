"""Wave 92 — Tier 3 key-term definition precision.

For each ``key_terms[].term`` value in the corpus, ask the model to
"define X" and score the response against the corpus's stored
definition.

Scoring uses a two-pass strategy:

1. **Embedding similarity** when ``sentence-transformers`` is
   installed (cosine similarity over a model like
   ``all-MiniLM-L6-v2``).
2. **Bag-of-words Jaccard** as a stdlib fallback so the harness works
   on a CPU-only dev box without heavy NLP dependencies.

Plus a "required-element" check that's data-driven (NOT hardcoded):
distinguishing tokens are extracted from the stored definition by
filtering stopwords + sorting by inverse document frequency across
the corpus's definition set, then taking the top-3. The model
response must include at least one of those tokens.

Schema correction baked in: ``key_terms`` is a list of ``{"term": "...",
"definition": "..."}`` dicts, NOT flat strings (per Wave 92 corpus
inspection).
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at",
    "by", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "from", "into", "but",
    "not", "no", "yes", "than", "then", "so", "if", "such", "each", "any",
    "all", "some", "may", "can", "will", "would", "should", "could",
    "have", "has", "had", "do", "does", "did", "more", "most", "other",
    "another", "which", "who", "whom", "what", "where", "when", "why", "how",
}
_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


def _tokenize(text: str) -> List[str]:
    return [
        t.lower() for t in _TOKEN_PATTERN.findall(text or "")
        if t.lower() not in _STOPWORDS and len(t) > 2
    ]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _try_load_embedder():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:  # noqa: BLE001 — broad to handle any install/load fail
        return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _extract_required_elements(
    target_definition: str,
    all_definitions: Sequence[str],
    top_k: int = 3,
) -> List[str]:
    """Pick the most distinguishing tokens of ``target_definition``
    relative to ``all_definitions`` via inverse document frequency.

    Heuristic: tokens that appear in the target but rarely elsewhere
    score high; tokens common across the corpus score low. We return
    the top-K. This is data-driven (no hand-curated keyword list)
    so the eval scales to any corpus.
    """
    target_tokens = _tokenize(target_definition)
    if not target_tokens:
        return []

    df: Counter = Counter()
    for d in all_definitions:
        df.update(set(_tokenize(d)))
    n_docs = max(1, len(all_definitions))

    scores: Dict[str, float] = {}
    for tok in set(target_tokens):
        idf = math.log((n_docs + 1) / (df.get(tok, 0) + 1)) + 1.0
        scores[tok] = idf

    return [tok for tok, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]]


def _load_chunks_jsonl(course_path: Path) -> List[Dict[str, Any]]:
    p = course_path / "corpus" / "chunks.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {p}")
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _harvest_key_terms(
    chunks: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Pull the {term, definition} dicts out of every chunk.

    Wave 92 schema correction: ``key_terms`` is always a list of
    dicts, never flat strings. Defensive coercion keeps legacy
    chunks (where someone sent a string) in the eval at reduced
    fidelity rather than blowing up.
    """
    out: List[Dict[str, str]] = []
    seen: set = set()
    for c in chunks:
        kts = c.get("key_terms") or []
        for kt in kts:
            if isinstance(kt, dict):
                term = (kt.get("term") or "").strip()
                definition = (kt.get("definition") or "").strip()
            else:
                term = str(kt).strip()
                definition = ""
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"term": term, "definition": definition})
    return out


class KeyTermPrecisionEvaluator:
    """Score a model on definitional precision over corpus key terms.

    Args:
        course_path: Path to ``LibV2/courses/<slug>/``.
        model_callable: ``Callable[[str], str]``.
        max_terms: Cap to bound eval cost; defaults to 50.
        embedder: Optional pre-loaded sentence-transformers model.
            When None we attempt to load one and fall back to Jaccard
            if unavailable.
    """

    def __init__(
        self,
        course_path: Path,
        model_callable: Callable[[str], str],
        max_terms: Optional[int] = 50,
        embedder: Optional[Any] = None,
    ) -> None:
        self.course_path = Path(course_path)
        chunks = _load_chunks_jsonl(self.course_path)
        self.terms = _harvest_key_terms(chunks)
        if max_terms is not None:
            self.terms = self.terms[:max_terms]
        self.model_callable = model_callable
        self.embedder = embedder if embedder is not None else _try_load_embedder()

    def evaluate(self) -> Dict[str, Any]:
        all_defs = [t["definition"] for t in self.terms if t.get("definition")]
        per_term: List[Dict[str, Any]] = []
        sims: List[float] = []
        elem_hits = 0
        elem_total = 0

        for entry in self.terms:
            term = entry["term"]
            definition = entry.get("definition", "")
            prompt = f"Define '{term}' precisely and concisely."
            try:
                response = str(self.model_callable(prompt))
            except Exception as exc:  # noqa: BLE001
                per_term.append({
                    "term": term, "outcome": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            similarity = self._score_similarity(definition, response)
            required = _extract_required_elements(definition, all_defs)
            if required:
                elem_total += 1
                response_tokens = set(_tokenize(response))
                if any(req.lower() in response_tokens for req in required):
                    elem_hits += 1
                    elem_hit = True
                else:
                    elem_hit = False
            else:
                elem_hit = None

            sims.append(similarity)
            per_term.append({
                "term": term,
                "expected_definition": definition,
                "response": response,
                "similarity": similarity,
                "required_elements": required,
                "required_element_hit": elem_hit,
            })

        avg_sim = sum(sims) / len(sims) if sims else 0.0
        elem_precision = elem_hits / elem_total if elem_total > 0 else 0.0
        return {
            "avg_similarity": avg_sim,
            "required_element_precision": elem_precision,
            "scoring_method": "embedding" if self.embedder is not None else "jaccard",
            "total": len(self.terms),
            "per_term": per_term,
        }

    def _score_similarity(self, expected: str, actual: str) -> float:
        if not expected.strip() or not actual.strip():
            return 0.0
        if self.embedder is not None:
            try:
                vecs = self.embedder.encode([expected, actual])
                return float(_cosine(vecs[0], vecs[1]))
            except Exception:  # noqa: BLE001
                pass
        return _jaccard(_tokenize(expected), _tokenize(actual))


__all__ = ["KeyTermPrecisionEvaluator"]
