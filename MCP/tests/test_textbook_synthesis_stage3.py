"""Stage-3 integration — the three-stage textbook-synthesis payoff.

Plan: ``plans/textbook-llm-synthesis-3stage-2026-05.md`` §4 + test-plan
item 4. Wave C.

This is the END-TO-END PAYOFF test. It regression-pins the OpenStax
0-``DomainConcept``-node failure (plan §0): a fixture DART chunkset whose
chunks carry EMPTY ``concept_tags`` plus a ``textbook_structure.json``
carrying per-chapter ``chapter_text``. With ``TEXTBOOK_SYNTHESIS_PROVIDER``
set and a stub provider, ``_run_concept_extraction`` synthesizes a
domain-concept vocabulary, re-tags the chunks IN-MEMORY, and emits
``concept_graph_semantic.json`` carrying ≥1 ``DomainConcept`` node.

Contract assertions:

1. **≥1 ``DomainConcept`` node** — the payoff. A chunkset that produced
   zero domain-concept nodes pre-Wave-C now produces real nodes once the
   vocabulary re-tags the chunks.
2. **``domain_concept_vocabulary.json`` persisted** as a sibling of
   ``concept_graph_semantic.json``.
3. **``dart_chunks_sha256`` byte-stable** — the on-disk ``chunks.jsonl``
   is NOT rewritten by the in-memory re-tag (plan §4.2 Risk-6a).
4. **Per-chapter failure isolation** — 1 of N chapter calls failing
   still succeeds the phase (plan §5.4).
5. **All-fail → empty-seed fallback** — every chapter call failing
   degrades to the status-quo graph (no domain concepts), phase still
   succeeds.
6. **Default-off** — ``TEXTBOOK_SYNTHESIS_PROVIDER`` unset → no
   vocabulary artifact, behaves exactly as today.

The LLM is mocked — no real API calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402
import Courseforge.generators._textbook_synthesis_provider as _tsp_mod  # noqa: E402
from Courseforge.generators._textbook_synthesis_provider import (  # noqa: E402
    TextbookSynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Fixture domain concepts — reuse the linear-algebra vocabulary proven to
# survive ``classify_concept`` (mirrors test_concept_extraction_semantic_graph).
# ---------------------------------------------------------------------------

_CONCEPT_TERMS = [
    "vector space", "linear map", "eigenvalue", "eigenvector",
    "matrix rank", "determinant", "basis set", "orthogonality",
    "inner product", "subspace", "kernel space", "image space",
]


def _empty_tag_chunkset() -> List[Dict[str, Any]]:
    """A DART chunkset with EMPTY ``concept_tags`` — the general-textbook
    shape that produced 0 domain-concept nodes pre-Wave-C.

    Each concept TERM appears verbatim in the ``text`` of ≥2 chunks so
    the per-course seed regexes match it in ≥2 chunks and it clears the
    co-occurrence builder's ``min_freq=2`` floor after the re-tag.
    """
    chunks: List[Dict[str, Any]] = []
    for i in range(24):
        window = [
            _CONCEPT_TERMS[(i + j) % len(_CONCEPT_TERMS)] for j in range(3)
        ]
        chunks.append({
            "id": f"linalg_chunk_{i:05d}",
            "text": (
                f"This passage explains the {window[0]} in relation to "
                f"the {window[1]} and the {window[2]} for a linear "
                f"algebra course chapter."
            ),
            "chunk_type": "explanation",
            # EMPTY concept_tags — the pre-Wave-C general-textbook shape.
            "concept_tags": [],
            "learning_outcome_refs": [],
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": f"week_{(i // 4) + 1:02d}",
                "item_path": f"week_{(i // 4) + 1:02d}/page_{i:03d}.html",
            },
        })
    return chunks


def _textbook_structure(n_chapters: int = 3) -> Dict[str, Any]:
    """A ``textbook_structure.json`` carrying per-chapter ``chapter_text``.

    Each chapter's text mentions a slice of the concept vocabulary; the
    stub provider parses these terms back out of the chapter text.
    """
    chapters: List[Dict[str, Any]] = []
    per = max(1, len(_CONCEPT_TERMS) // n_chapters)
    for ci in range(n_chapters):
        terms = _CONCEPT_TERMS[ci * per:(ci + 1) * per] or _CONCEPT_TERMS[:per]
        chapters.append({
            "id": f"ch{ci + 1}",
            "title": f"Chapter {ci + 1}",
            "chapter_text": (
                f"Chapter {ci + 1} introduces "
                + ", ".join(terms)
                + ". These concepts ground the chapter's worked examples."
            ),
            "sections": [],
        })
    return {"course_name": "LINALG_STAGE3", "chapters": chapters}


# ---------------------------------------------------------------------------
# Stub provider — extracts concept terms straight out of chapter_text. No
# LLM dispatch, no network. Mirrors the TextbookSynthesisProvider public
# surface the Stage-3 handler block consumes.
# ---------------------------------------------------------------------------


class _StubProvider:
    """Drop-in stub for ``TextbookSynthesisProvider`` (Stage 3 only).

    ``synthesize_concepts`` scans the chapter's ``chapter_text`` for any
    of ``_CONCEPT_TERMS`` and returns them as a concept list. Chapters in
    ``fail_chapter_ids`` raise ``TextbookSynthesisProviderError`` to
    exercise the §5.4 per-chapter failure-isolation path.
    """

    def __init__(self, *, fail_chapter_ids: List[str] | None = None,
                 **_kwargs: Any) -> None:
        self._provider = "stub"
        self._model = "stub-model-v1"
        self._fail = set(fail_chapter_ids or [])

    def synthesize_concepts(
        self, chapter: Dict[str, Any], *, course_name: str,
    ) -> Dict[str, Any]:
        chapter_id = str(chapter.get("id") or "")
        if chapter_id in self._fail:
            raise TextbookSynthesisProviderError(
                f"stub forced failure on chapter {chapter_id!r}",
                code="concepts_exhausted",
            )
        text = str(chapter.get("chapter_text") or "").lower()
        concepts: List[Dict[str, Any]] = []
        for term in _CONCEPT_TERMS:
            if term.lower() in text:
                concepts.append({
                    "canonical": term,
                    "aliases": [],
                    "definition_hint": f"a linear-algebra concept: {term}",
                    "chapter_ids": [chapter_id],
                })
        return {"chapter_id": chapter_id, "concepts": concepts}

    @staticmethod
    def batch_chapters(chapters: List[Any], batch_size: int = 10,
                       ) -> List[List[Any]]:
        size = max(1, int(batch_size))
        return [
            list(chapters[i:i + size])
            for i in range(0, len(chapters), size)
        ]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _write_chunkset(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


def _run(
    tmp_path: Path,
    *,
    chunks: List[Dict[str, Any]],
    structure: Dict[str, Any] | None,
    provider_factory: Any | None,
    provider_env: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Any]:
    """Invoke ``_run_concept_extraction`` and return the parsed envelope
    plus the on-disk graph + chunkset bytes.
    """
    chunks_path = tmp_path / "dart_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, chunks)
    chunks_sha_before = hashlib.sha256(chunks_path.read_bytes()).hexdigest()

    ts_path = ""
    if structure is not None:
        ts_file = tmp_path / "textbook_structure.json"
        ts_file.write_text(json.dumps(structure), encoding="utf-8")
        ts_path = str(ts_file)

    if provider_env is None:
        monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", provider_env)

    if provider_factory is not None:
        monkeypatch.setattr(
            _tsp_mod, "TextbookSynthesisProvider", provider_factory,
        )

    custom_libv2 = tmp_path / "libv2"
    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="LINALG_STAGE3",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            textbook_structure_path=ts_path,
            libv2_root=str(custom_libv2),
        )
    )
    payload = json.loads(result)
    graph = json.loads(
        Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
    )
    payload["_graph"] = graph
    payload["_chunks_sha_before"] = chunks_sha_before
    payload["_chunks_sha_after"] = hashlib.sha256(
        chunks_path.read_bytes()
    ).hexdigest()
    return payload


_CONCEPT_CLASSES = {
    "Concept", "DomainConcept", "concept", "domain_concept",
}


def _domain_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        n for n in (graph.get("nodes") or [])
        if n.get("class") in _CONCEPT_CLASSES
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage3_emits_domain_concept_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE PAYOFF — a chunkset with EMPTY concept_tags + a textbook
    structure with chapter_text yields ≥1 ``DomainConcept`` node.

    Regression-pins the OpenStax 0-node failure (plan §0).
    """
    payload = _run(
        tmp_path,
        chunks=_empty_tag_chunkset(),
        structure=_textbook_structure(),
        provider_factory=_StubProvider,
        provider_env="stub",
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True, f"payload={payload!r}"

    nodes = payload["_graph"].get("nodes") or []
    domain_nodes = _domain_nodes(payload["_graph"])
    assert domain_nodes, (
        "PAYOFF FAILURE: a chunkset with empty concept_tags + a "
        "textbook_structure carrying chapter_text must, after Stage-3 "
        "re-tagging, yield >=1 DomainConcept node. Node classes seen: "
        f"{sorted({n.get('class') for n in nodes})}."
    )
    # Plan §0 payoff: pre-Wave-C this exact chunkset produced ZERO
    # domain-concept nodes. Any non-zero count is the regression-pin.
    assert payload["node_count"] == len(nodes)


def test_stage3_persists_vocabulary_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``domain_concept_vocabulary.json`` is written as a sibling of
    ``concept_graph_semantic.json`` and is schema-shaped.
    """
    payload = _run(
        tmp_path,
        chunks=_empty_tag_chunkset(),
        structure=_textbook_structure(),
        provider_factory=_StubProvider,
        provider_env="stub",
        monkeypatch=monkeypatch,
    )
    vocab_path = payload.get("domain_concept_vocabulary_path")
    assert vocab_path, "Stage 3 must surface domain_concept_vocabulary_path"
    vp = Path(vocab_path)
    assert vp.exists(), f"vocabulary artifact not on disk: {vp}"
    # Sibling of concept_graph_semantic.json.
    assert vp.parent == Path(payload["concept_graph_path"]).parent

    vocab = json.loads(vp.read_text(encoding="utf-8"))
    assert vocab["schema_version"] == "v1"
    assert vocab["concept_count"] == len(vocab["concepts"])
    assert vocab["concept_count"] >= 8, (
        "the stub yields the full linear-algebra vocabulary; the "
        "concept_count should clear the validator floor"
    )
    for concept in vocab["concepts"]:
        assert concept["canonical"], "every concept needs a canonical"
        # canonical was run through canonical_slug → kebab-case.
        assert " " not in concept["canonical"]
    assert payload["domain_concept_count"] == vocab["concept_count"]
    assert payload["chunks_retagged"] >= 1


def test_stage3_dart_chunks_sha_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory re-tag must NOT rewrite ``chunks.jsonl`` on disk —
    ``dart_chunks_sha256`` stays byte-stable (plan §4.2 Risk-6a).
    """
    payload = _run(
        tmp_path,
        chunks=_empty_tag_chunkset(),
        structure=_textbook_structure(),
        provider_factory=_StubProvider,
        provider_env="stub",
        monkeypatch=monkeypatch,
    )
    assert payload["_chunks_sha_before"] == payload["_chunks_sha_after"], (
        "Stage-3 re-tag rewrote chunks.jsonl on disk — dart_chunks_sha256 "
        "would diverge and break the chunkset / LibV2 manifest hash "
        "triangle. The re-tag MUST be in-memory only."
    )


def test_stage3_per_chapter_failure_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1 of N per-chapter calls failing degrades — the phase still
    succeeds and the surviving chapters' concepts still seed the graph
    (plan §5.4).
    """
    def _factory(**kwargs: Any) -> _StubProvider:
        return _StubProvider(fail_chapter_ids=["ch2"], **kwargs)

    payload = _run(
        tmp_path,
        chunks=_empty_tag_chunkset(),
        structure=_textbook_structure(),
        provider_factory=_factory,
        provider_env="stub",
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True, (
        "one failed chapter call must NOT abort the phase (§5.4)"
    )
    vocab = json.loads(
        Path(payload["domain_concept_vocabulary_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert "ch2" in vocab["chapter_synthesis_failures"], (
        "the failed chapter id must be recorded in "
        "chapter_synthesis_failures"
    )
    # Surviving chapters still produced concepts → real domain nodes.
    assert _domain_nodes(payload["_graph"]), (
        "the surviving chapters' concepts must still seed DomainConcept "
        "nodes despite one chapter failing"
    )


def test_stage3_all_fail_empty_seed_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every per-chapter call failing → empty-seed fallback. The phase
    still succeeds; the graph degrades to the status-quo (no domain
    concepts) per the §5.4 fail-soft floor.
    """
    structure = _textbook_structure()
    all_ids = [c["id"] for c in structure["chapters"]]

    def _factory(**kwargs: Any) -> _StubProvider:
        return _StubProvider(fail_chapter_ids=all_ids, **kwargs)

    payload = _run(
        tmp_path,
        chunks=_empty_tag_chunkset(),
        structure=structure,
        provider_factory=_factory,
        provider_env="stub",
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True, (
        "an all-chapter-fail Stage-3 run must degrade, not abort"
    )
    # The vocabulary artifact is still written (degraded provenance),
    # but it carries zero concepts.
    vocab_path = payload.get("domain_concept_vocabulary_path")
    if vocab_path:
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        assert vocab["concept_count"] == 0
        assert len(vocab["chapter_synthesis_failures"]) == len(all_ids)
    # Chunks carried empty concept_tags + the re-tag had no seeds →
    # the graph has no domain-concept nodes (status quo).
    assert not _domain_nodes(payload["_graph"]), (
        "all-fail fallback must NOT seed domain concepts — the graph "
        "stays at the status-quo empty shape"
    )


def test_stage3_default_off_no_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TEXTBOOK_SYNTHESIS_PROVIDER`` unset → no Stage-3 run, no
    vocabulary artifact, no domain-concept nodes from empty-tag chunks.
    Behaves exactly as today.
    """
    payload = _run(
        tmp_path,
        chunks=_empty_tag_chunkset(),
        structure=_textbook_structure(),
        provider_factory=None,
        provider_env=None,
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True
    assert "domain_concept_vocabulary_path" not in payload, (
        "a default-off run must not emit the vocabulary artifact key"
    )
    graph_dir = Path(payload["concept_graph_path"]).parent
    assert not (graph_dir / "domain_concept_vocabulary.json").exists(), (
        "no domain_concept_vocabulary.json on a default-off run"
    )
    # Empty-tag chunks with no seeds → no domain-concept nodes (status quo).
    assert not _domain_nodes(payload["_graph"])
