"""Wave 102 - RAGCallable tests.

All tests use a mock CLI runner so no subprocess fires and no LibV2
state needs to exist. Asserts that:

* The retrieval prelude is correctly composed (numbered passages +
  citation instruction + chunk_id labels).
* The wrapped callable receives the augmented prompt, not the bare
  one.
* Latency is recorded per call and averaged.
* Invalid method / limit values raise.
* Empty retrieval gracefully degrades to the bare prompt.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_cli_runner(chunks: List[Dict[str, Any]]):
    """Return a runner that always yields the supplied chunks."""
    def _run(args):
        return {"retrieved_chunks": list(chunks)}
    return _run


def test_rag_callable_formats_prelude_with_numbered_chunks():
    from Trainforge.eval.rag_callable import RAGCallable

    seen_prompts: List[str] = []

    def base(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "stub answer"

    chunks = [
        {"rank": 1, "chunk_id": "chunk_aaa", "text": "RDF is a triple model."},
        {"rank": 2, "chunk_id": "chunk_bbb", "text": "SHACL validates RDF graphs."},
    ]
    rag = RAGCallable(
        base_callable=base,
        course_slug="rdf-shacl-551-2",
        method="bm25",
        limit=5,
        cli_runner=_build_cli_runner(chunks),
    )
    out = rag("What is SHACL?")

    assert out == "stub answer"
    assert len(seen_prompts) == 1
    augmented = seen_prompts[0]
    # Both chunk_ids are labelled in the prelude
    assert "[chunk_aaa]" in augmented
    assert "[chunk_bbb]" in augmented
    # Citation instruction is present
    assert "cite the chunk_id in [brackets]" in augmented
    # The original prompt survives
    assert "What is SHACL?" in augmented


def test_rag_callable_records_latency_per_call_and_mean():
    from Trainforge.eval.rag_callable import RAGCallable

    chunks = [{"rank": 1, "chunk_id": "chunk_a", "text": "x"}]
    rag = RAGCallable(
        base_callable=lambda p: "ok",
        course_slug="slug",
        method="bm25",
        limit=3,
        cli_runner=_build_cli_runner(chunks),
    )
    rag("p1")
    rag("p2")
    rag("p3")
    assert rag.last_latency_ms is not None
    assert rag.mean_latency_ms is not None
    # Three samples, mean is finite and non-negative
    assert rag.mean_latency_ms >= 0
    assert len(rag._latencies) == 3


def test_rag_callable_invalid_method_raises():
    from Trainforge.eval.rag_callable import RAGCallable

    with pytest.raises(ValueError) as exc:
        RAGCallable(
            base_callable=lambda p: "x",
            course_slug="slug",
            method="not-a-method",
        )
    assert "unknown method" in str(exc.value)


def test_rag_callable_invalid_limit_raises():
    from Trainforge.eval.rag_callable import RAGCallable

    with pytest.raises(ValueError):
        RAGCallable(
            base_callable=lambda p: "x",
            course_slug="slug",
            method="bm25",
            limit=0,
        )
    with pytest.raises(ValueError):
        RAGCallable(
            base_callable=lambda p: "x",
            course_slug="slug",
            method="bm25",
            limit=51,
        )


def test_rag_callable_empty_retrieval_falls_back_to_bare_prompt():
    from Trainforge.eval.rag_callable import RAGCallable

    seen: List[str] = []

    def base(prompt: str) -> str:
        seen.append(prompt)
        return ""

    rag = RAGCallable(
        base_callable=base,
        course_slug="slug",
        method="bm25",
        cli_runner=_build_cli_runner([]),  # empty
    )
    rag("ask")
    # No prelude; bare prompt fed straight through
    assert seen == ["ask"]


def test_rag_callable_passes_method_and_limit_to_cli():
    """The CLI invocation must carry --method and --limit so LibV2
    routes through the chosen retrieval preset."""
    from Trainforge.eval.rag_callable import RAGCallable

    captured_args: List[List[str]] = []

    def runner(args):
        captured_args.append(list(args))
        return {"retrieved_chunks": []}

    rag = RAGCallable(
        base_callable=lambda p: "x",
        course_slug="my-slug",
        method="bm25+intent",
        limit=7,
        cli_runner=runner,
    )
    rag("hello")
    args = captured_args[0]
    # Spot-check the canonical flag positions.
    assert "--method" in args
    assert args[args.index("--method") + 1] == "bm25+intent"
    assert "--limit" in args
    assert args[args.index("--limit") + 1] == "7"
    assert "--course" in args
    assert args[args.index("--course") + 1] == "my-slug"
    assert "-o" in args
    assert args[args.index("-o") + 1] == "json"
    # ask --force keeps the call deterministic w.r.t. cache state.
    assert "--force" in args
