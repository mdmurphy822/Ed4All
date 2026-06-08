"""Tests for ``gui.services.retrieval_service`` — real BM25 retrieval + typed errors.

Builds a tiny temp LibV2 course (manifest.json + chunks.jsonl) under a redirected
``ED4ALL_LIBV2_ROOT`` (via the ``libv2_course`` fixture) so the real ``LibV2/`` is
untouched. ``retrieve`` runs the REAL ``retrieve_chunks``; ``adapter_infer`` is
driven down the deps-missing path by forcing the lazy import to raise ImportError.
No fastapi needed.
"""

from __future__ import annotations

from gui.services import retrieval_service


def test_list_courses_returns_real_course(libv2_course):
    courses = retrieval_service.list_courses()
    by_slug = {c["slug"]: c for c in courses}
    assert libv2_course["slug"] in by_slug
    # chunk_count read from manifest content_profile.total_chunks.
    assert by_slug[libv2_course["slug"]]["chunk_count"] == 3


def test_retrieve_returns_real_hits(libv2_course):
    results = retrieval_service.retrieve(
        libv2_course["slug"], "velocity rate of change of position", top_k=2
    )
    assert results, "BM25 should return at least one hit for an on-topic query"
    # The velocity chunk should rank above the unrelated photosynthesis chunk.
    texts = " ".join(r["text"] for r in results)
    assert "velocity" in texts.lower() or "acceleration" in texts.lower()
    top = results[0]
    assert top["chunk_id"] in {"chunk_0001", "chunk_0002"}
    assert isinstance(top["score"], float)
    # include_rationale=True is passed, so rationale rides along.
    assert "rationale" in top


def test_retrieve_empty_query_returns_empty(libv2_course):
    assert retrieval_service.retrieve(libv2_course["slug"], "   ") == []


def test_retrieve_top_k_clamped(libv2_course):
    results = retrieval_service.retrieve(libv2_course["slug"], "kinematics", top_k=999)
    # Only 3 chunks exist; clamp never invents more.
    assert len(results) <= 3


def test_retrieve_filter_by_chunk_type(libv2_course):
    results = retrieval_service.retrieve(
        libv2_course["slug"],
        "acceleration velocity",
        top_k=5,
        filters={"chunk_type": "example"},
    )
    # Only the example chunk matches the filter.
    assert all(r.get("chunk_type") == "example" for r in results)
    assert any(r["chunk_id"] == "chunk_0002" for r in results)


def test_adapter_infer_training_deps_missing(libv2_course, monkeypatch):
    """When the heavy ML import raises ImportError, return the typed error."""
    slug = libv2_course["slug"]
    # Materialize a fake promoted adapter so the function reaches the lazy import.
    models_dir = libv2_course["course_dir"] / "models" / "adapter-v1"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "model_card.json").write_text(
        '{"model_id": "adapter-v1", "base_model": "Qwen/Qwen2.5-1.5B"}',
        encoding="utf-8",
    )

    # Force the lazy ``from Trainforge.eval.adapter_callable import AdapterCallable``
    # to raise ImportError, regardless of whether torch/peft are actually present.
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "Trainforge.eval.adapter_callable":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    result = retrieval_service.adapter_infer(slug, "adapter-v1", "explain velocity")
    assert result["error"] == "training_deps_missing"
    assert "torch" in result["detail"] or "training" in result["detail"].lower()
    assert "generation" not in result  # never fabricates an answer


def test_adapter_infer_adapter_not_found(libv2_course):
    result = retrieval_service.adapter_infer(
        libv2_course["slug"], "no-such-adapter", "hello"
    )
    assert result["error"] == "adapter_not_found"


def test_adapter_infer_empty_prompt(libv2_course):
    result = retrieval_service.adapter_infer(libv2_course["slug"], "any", "  ")
    assert result["error"] == "invalid_prompt"


def test_list_adapters_empty_when_none(libv2_course):
    assert retrieval_service.list_adapters(libv2_course["slug"]) == []
