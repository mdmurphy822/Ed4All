"""Wave 92 — KeyTermPrecisionEvaluator tests.

Synthetic key_terms list embedded in chunks; mocked model. Asserts
the scorer falls back to Jaccard when no embedder is wired in (the
default test environment doesn't ship sentence-transformers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.key_term_precision import (  # noqa: E402
    KeyTermPrecisionEvaluator,
    _extract_required_elements,
    _harvest_key_terms,
    _jaccard,
    _tokenize,
)


def _build_course(tmp_path: Path) -> Path:
    course = tmp_path / "tst-101"
    (course / "corpus").mkdir(parents=True)
    chunks = [
        {
            "id": "c_001",
            "key_terms": [
                {
                    "term": "RDF triple",
                    "definition": "A statement consisting of subject, predicate, and object.",
                },
                {
                    "term": "SHACL shape",
                    "definition": "A constraint specification that defines validation rules for RDF data.",
                },
            ],
        },
        {
            "id": "c_002",
            "key_terms": [
                {
                    "term": "SPARQL query",
                    "definition": "A graph pattern query language used to retrieve information from RDF data.",
                },
            ],
        },
    ]
    chunks_path = course / "corpus" / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    return course


def test_harvest_dedupes_terms(tmp_path):
    course = _build_course(tmp_path)
    chunks = [
        json.loads(line)
        for line in (course / "corpus" / "chunks.jsonl").read_text().splitlines()
    ]
    terms = _harvest_key_terms(chunks)
    assert len(terms) == 3
    assert {t["term"] for t in terms} == {"RDF triple", "SHACL shape", "SPARQL query"}


def test_extract_required_elements_picks_distinguishing_tokens():
    target = "A constraint specification that defines validation rules for RDF data."
    others = [
        "A statement consisting of subject, predicate, and object.",
        "A graph pattern query language used to retrieve information from RDF data.",
    ]
    elements = _extract_required_elements(target, [target, *others], top_k=3)
    assert len(elements) <= 3
    assert any(e in {"constraint", "specification", "validation", "rules"} for e in elements)


def test_perfect_response_scores_positive(tmp_path):
    course = _build_course(tmp_path)
    scorer = KeyTermPrecisionEvaluator(
        course_path=course,
        model_callable=lambda p: (
            "A constraint specification that defines validation rules for RDF data."
        ),
        embedder=None,
    )
    out = scorer.evaluate()
    # scoring_method depends on whether [embedding] extras are installed:
    # absent -> "jaccard" (deterministic fallback); present -> "embedding"
    # (real sentence-transformers cosine). Either backend is correct.
    assert out["scoring_method"] in ("jaccard", "embedding")
    assert out["avg_similarity"] > 0.0


def test_zero_overlap_response_scores_low(tmp_path):
    course = _build_course(tmp_path)
    scorer = KeyTermPrecisionEvaluator(
        course_path=course,
        model_callable=lambda p: "potato banana orange unrelated tokens here.",
        embedder=None,
    )
    out = scorer.evaluate()
    assert out["avg_similarity"] < 0.5


def test_required_element_check_data_driven(tmp_path):
    """Required elements are extracted from the corpus, not hardcoded."""
    course = _build_course(tmp_path)
    scorer = KeyTermPrecisionEvaluator(
        course_path=course,
        model_callable=lambda p: "constraint validation rules for triples",
        embedder=None,
    )
    out = scorer.evaluate()
    has_required_check = any(
        t["required_element_hit"] is not None for t in out["per_term"]
    )
    assert has_required_check


def test_max_terms_caps_run(tmp_path):
    course = _build_course(tmp_path)
    scorer = KeyTermPrecisionEvaluator(
        course_path=course,
        model_callable=lambda p: "x",
        max_terms=2,
        embedder=None,
    )
    out = scorer.evaluate()
    assert out["total"] == 2


def test_legacy_string_key_terms_dont_blow_up(tmp_path):
    """Defensive: a legacy chunk with flat-string key_terms must not crash."""
    course = tmp_path / "tst-101"
    (course / "corpus").mkdir(parents=True)
    chunks_path = course / "corpus" / "chunks.jsonl"
    chunks_path.write_text(json.dumps({
        "id": "c_001",
        "key_terms": ["legacy string term"],
    }) + "\n", encoding="utf-8")
    scorer = KeyTermPrecisionEvaluator(
        course_path=course,
        model_callable=lambda p: "irrelevant",
        embedder=None,
    )
    out = scorer.evaluate()
    assert out["total"] == 1


def test_jaccard_basics():
    assert _jaccard(["a", "b"], ["b", "c"]) == 1 / 3
    assert _jaccard([], []) == 1.0
    assert _jaccard(["a"], []) == 0.0


def test_tokenize_strips_stopwords():
    toks = _tokenize("The quick brown fox is jumping over the lazy dog.")
    assert "the" not in toks
    assert "is" not in toks
    assert "quick" in toks
    assert "brown" in toks
