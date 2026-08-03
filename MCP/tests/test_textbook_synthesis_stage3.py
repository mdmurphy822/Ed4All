"""Stage-3 integration — the three-stage textbook-synthesis payoff.

Plan: ``plans/textbook-llm-synthesis-3stage-2026-05.md`` §4 + test-plan
item 4. Wave C.

This is the END-TO-END PAYOFF test. It regression-pins the real-corpus
0-``DomainConcept``-node failure (plan §0): a fixture SemantiK chunkset whose
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
3. **``semantik_chunks_sha256`` byte-stable** — the on-disk ``chunks.jsonl``
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
    """A SemantiK chunkset with EMPTY ``concept_tags`` — the general-textbook
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
    return {"course_name": "FXEDGE_STAGE3", "chapters": chapters}


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
    chunks_path = tmp_path / "semantik_chunks" / "chunks.jsonl"
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
            course_name="FXEDGE_STAGE3",
            staging_dir="",
            # ``dart_chunks_path`` is the tool's live kwarg name.
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

    Regression-pins the real-corpus 0-node failure (plan §0).
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


def test_stage3_chunks_sha_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory re-tag must NOT rewrite ``chunks.jsonl`` on disk —
    ``semantik_chunks_sha256`` stays byte-stable (plan §4.2 Risk-6a).
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
        "Stage-3 re-tag rewrote chunks.jsonl on disk — semantik_chunks_sha256 "
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


# ---------------------------------------------------------------------------
# Fix-1 — Stage-3 vocabulary seeds are compiled with a de-slugged alias.
#
# The merge step in ``_run_stage3_concept_synthesis`` overwrites each
# concept's ``canonical`` with its ``canonical_slug`` (e.g. "visual
# hierarchy" -> "visual-hierarchy"). ``compile_domain_concept_seeds``
# then regex-escapes the slug into ``\bvisual\-hierarchy\b``, which never
# matches the spaced prose form "visual hierarchy". Fix-1 appends the
# hyphen-replaced surface form as a seed alias so the compiled pattern
# actually fires on the textbook text.
# ---------------------------------------------------------------------------

# A canonical with a known slug != surface form. The chunk text below
# carries ONLY the spaced form, never the hyphen-joined slug.
_DESLUG_TERMS = [
    "visual hierarchy", "color contrast", "white space",
    "focus indicator", "responsive layout", "semantic markup",
]


class _SlugCanonicalStubProvider:
    """Stub that emits multi-word ``canonical`` strings with NO aliases.

    After the merge slugifies them (``visual hierarchy`` ->
    ``visual-hierarchy``), the only way a chunk carrying the SPACED prose
    form gets tagged is if Fix-1 re-injected the de-slugged surface form
    as an alias.
    """

    def __init__(self, **_kwargs: Any) -> None:
        self._provider = "stub-slug"
        self._model = "stub-slug-v1"

    def synthesize_concepts(
        self, chapter: Dict[str, Any], *, course_name: str,
    ) -> Dict[str, Any]:
        text = str(chapter.get("chapter_text") or "").lower()
        concepts: List[Dict[str, Any]] = []
        for term in _DESLUG_TERMS:
            if term.lower() in text:
                concepts.append({
                    "canonical": term,   # multi-word → slugified in merge
                    "aliases": [],       # NO aliases — Fix-1 must add one
                    "definition_hint": f"a UI design concept: {term}",
                    "chapter_ids": [str(chapter.get("id") or "")],
                })
        return {"chapter_id": str(chapter.get("id") or ""), "concepts": concepts}

    @staticmethod
    def batch_chapters(chapters: List[Any], batch_size: int = 10,
                       ) -> List[List[Any]]:
        size = max(1, int(batch_size))
        return [
            list(chapters[i:i + size])
            for i in range(0, len(chapters), size)
        ]


def _deslug_chunkset() -> List[Dict[str, Any]]:
    """Chunks whose text carries the SPACED form of each term in ≥2
    chunks, and whose ``concept_tags`` start EMPTY. The slugged form
    (``visual-hierarchy``) never appears literally in the text — so a
    seed compiled WITHOUT the de-slug alias matches nothing.
    """
    chunks: List[Dict[str, Any]] = []
    for i in range(12):
        window = [
            _DESLUG_TERMS[(i + j) % len(_DESLUG_TERMS)] for j in range(3)
        ]
        chunks.append({
            "id": f"deslug_chunk_{i:05d}",
            "text": (
                f"This UI design passage discusses {window[0]}, together "
                f"with {window[1]} and {window[2]} in a real interface."
            ),
            "chunk_type": "explanation",
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


def _deslug_structure() -> Dict[str, Any]:
    chapters: List[Dict[str, Any]] = []
    per = max(1, len(_DESLUG_TERMS) // 2)
    for ci in range(2):
        terms = _DESLUG_TERMS[ci * per:(ci + 1) * per] or _DESLUG_TERMS[:per]
        chapters.append({
            "id": f"dch{ci + 1}",
            "title": f"Design Chapter {ci + 1}",
            "chapter_text": (
                f"Chapter {ci + 1} introduces " + ", ".join(terms)
                + ". These concepts ground the chapter's design examples."
            ),
            "sections": [],
        })
    return {"course_name": "FXEDGE_STAGE3", "chapters": chapters}


def test_stage3_seed_compilation_emits_deslugged_alias() -> None:
    """Unit-pin the Fix-1 seed-spec shape directly: a slugged canonical
    yields a seed whose compiled pattern matches the spaced surface form,
    and a single-word slug does NOT add a redundant duplicate alias.
    """
    from Trainforge.process_course import compile_domain_concept_seeds

    # Mirror the post-Fix-1 seed-spec construction in
    # ``_run_stage3_concept_synthesis``.
    concepts_out = [
        {"canonical": "visual-hierarchy", "aliases": []},
        {"canonical": "color-contrast", "aliases": ["contrast ratio"]},
        {"canonical": "accessibility", "aliases": []},   # single-word slug
    ]
    seed_specs: List[Dict[str, Any]] = []
    for c in concepts_out:
        cid = c["canonical"]
        aliases = list(c.get("aliases") or [])
        deslug = cid.replace("-", " ")
        if deslug != cid and deslug not in aliases:
            aliases.append(deslug)
        seed_specs.append({"id": cid, "aliases": aliases})

    # The de-slug alias was added for the multi-word slugs only.
    by_id = {s["id"]: s["aliases"] for s in seed_specs}
    assert "visual hierarchy" in by_id["visual-hierarchy"]
    assert "color contrast" in by_id["color-contrast"]
    assert "contrast ratio" in by_id["color-contrast"]  # original preserved
    # Single-word slug: de-slug == id, so no alias added (dedup).
    assert by_id["accessibility"] == []
    # No duplicate de-slug alias when one already exists.
    assert by_id["color-contrast"].count("color contrast") == 1

    seeds = compile_domain_concept_seeds(seed_specs)
    # The compiled "visual-hierarchy" seed now matches the spaced prose
    # form (it would NOT without the de-slug alias).
    seed_map = {canon: pats for canon, pats in seeds}
    vh_pats = seed_map["visual-hierarchy"]
    assert any(p.search("a strong visual hierarchy guides the eye")
               for p in vh_pats), (
        "Fix-1: the de-slugged alias must let the compiled seed match the "
        "spaced prose form 'visual hierarchy'"
    )


def test_stage3_deslug_alias_tags_spaced_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end Fix-1: a Stage-3 vocabulary of multi-word canonicals
    (slugified in the merge, NO aliases) re-tags chunks whose text carries
    only the SPACED form, minting ≥1 ``DomainConcept`` node. Pre-Fix-1
    the slug-only pattern matched nothing in the spaced prose and the
    graph stayed empty.
    """
    payload = _run(
        tmp_path,
        chunks=_deslug_chunkset(),
        structure=_deslug_structure(),
        provider_factory=_SlugCanonicalStubProvider,
        provider_env="stub-slug",
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True
    domain_nodes = _domain_nodes(payload["_graph"])
    assert domain_nodes, (
        "Fix-1: de-slugged aliases must let the spaced-prose chunks tag, "
        "minting DomainConcept nodes; got zero (slug-only match failed)."
    )
    node_ids = {n.get("id") for n in domain_nodes}
    # At least one multi-word concept's slug surfaced as a real node.
    assert node_ids & {
        "visual-hierarchy", "color-contrast", "white-space",
        "focus-indicator", "responsive-layout", "semantic-markup",
    }, f"expected a de-slugged concept node; got {sorted(node_ids)}"


# ---------------------------------------------------------------------------
# Window-synthesis parity (Stage-3 windows over chunks, not one call/chapter).
#
# Mirrors the Stage-2 ``_run_stage2_window_synthesis`` windowing. A corpus
# that collapses to ONE mega-chapter must still drive MANY ``synthesize_concepts``
# calls (one per window), and the resulting vocabulary must UNION + dedup the
# per-window concepts. A single failing window must NOT abort the phase.
# ---------------------------------------------------------------------------


class _WindowCountingStubProvider:
    """Stub that records EVERY ``synthesize_concepts`` call.

    Returns ``concepts_per_window`` DISTINCT concepts on each call, named
    deterministically off the call index so the union/dedup math is exact.
    The N-th call (0-based) for which ``call_index in fail_indices`` raises
    to exercise per-window failure isolation. ``shared_concept`` (when set)
    is ALSO returned on every call so dedup-across-windows is observable.
    """

    def __init__(
        self,
        *,
        concepts_per_window: int = 2,
        fail_indices: set[int] | None = None,
        shared_concept: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._provider = "stub-window"
        self._model = "stub-window-v1"
        # Modest serving window so a chunk-rich chapter splits into windows.
        self._max_tokens = 256
        self._concepts_per_window = concepts_per_window
        self._fail_indices = set(fail_indices or set())
        self._shared_concept = shared_concept
        self.calls: List[Dict[str, Any]] = []

    def synthesize_concepts(
        self, chapter: Dict[str, Any], *, course_name: str,
    ) -> Dict[str, Any]:
        call_index = len(self.calls)
        chapter_id = str(chapter.get("id") or "")
        self.calls.append({
            "call_index": call_index,
            "chapter_id": chapter_id,
            "window_index": chapter.get("window_index"),
            "chunk_ids": list(chapter.get("chunk_ids") or []),
            "chapter_text_chars": len(str(chapter.get("chapter_text") or "")),
        })
        if call_index in self._fail_indices:
            raise TextbookSynthesisProviderError(
                f"stub forced failure on window call {call_index}",
                code="concepts_exhausted",
            )
        concepts: List[Dict[str, Any]] = []
        for k in range(self._concepts_per_window):
            term = f"concept call{call_index} item{k}"
            concepts.append({
                "canonical": term,
                "aliases": [],
                "definition_hint": f"window-distinct concept {term}",
                "chapter_ids": [chapter_id],
            })
        if self._shared_concept:
            concepts.append({
                "canonical": self._shared_concept,
                "aliases": [],
                "definition_hint": "shared across every window",
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


def _mega_chapter_chunkset(n_chunks: int = 30) -> List[Dict[str, Any]]:
    """A chunkset that all maps to ONE chapter (the real-corpus mega-chapter
    failure shape). No ``source.source_references`` → order-fallback joins
    every chunk to the single chapter, then windowing splits them.
    """
    chunks: List[Dict[str, Any]] = []
    for i in range(n_chunks):
        chunks.append({
            "id": f"mega_chunk_{i:05d}",
            "text": (
                f"Passage {i} of a single dense chapter discussing many "
                f"distinct domain ideas, topic number {i}, in prose that is "
                f"long enough to fill a serving window on its own when the "
                f"num_ctx budget is small."
            ),
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": [],
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": "week_01",
                "item_path": f"week_01/page_{i:03d}.html",
            },
        })
    return chunks


def _one_mega_chapter() -> Dict[str, Any]:
    """A textbook_structure with exactly ONE chapter (mega-chapter)."""
    return {
        "course_name": "MEGA_STAGE3",
        "chapters": [
            {
                "id": "ch1",
                "title": "The Only Chapter",
                "chapter_text": "A single mega-chapter that collapsed.",
                "sections": [],
            }
        ],
    }


def _run_with_provider_instance(
    tmp_path: Path,
    *,
    chunks: List[Dict[str, Any]],
    structure: Dict[str, Any],
    provider_instance: Any,
    monkeypatch: pytest.MonkeyPatch,
    num_ctx: str = "600",
) -> Dict[str, Any]:
    """Like ``_run`` but pins a SHARED provider INSTANCE (so the test can
    read back ``provider_instance.calls``) and a small num_ctx (so the
    mega-chapter splits into multiple windows).
    """
    def _factory(**_kwargs: Any) -> Any:
        return provider_instance

    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", num_ctx)
    return _run(
        tmp_path,
        chunks=chunks,
        structure=structure,
        provider_factory=_factory,
        provider_env="stub-window",
        monkeypatch=monkeypatch,
    )


def test_stage3_calls_synthesize_concepts_once_per_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single mega-chapter with 30 chunks + a small num_ctx must drive
    MANY ``synthesize_concepts`` calls — one per WINDOW, NOT one per chapter.
    """
    provider = _WindowCountingStubProvider(concepts_per_window=2)
    payload = _run_with_provider_instance(
        tmp_path,
        chunks=_mega_chapter_chunkset(30),
        structure=_one_mega_chapter(),
        provider_instance=provider,
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True, f"payload={payload!r}"

    # The corpus has ONE chapter — the legacy per-chapter path would call
    # exactly once. Windowing must call MANY more times.
    n_calls = len(provider.calls)
    assert n_calls > 1, (
        "Stage-3 must window over chunks: a 30-chunk mega-chapter under a "
        f"small num_ctx must drive >1 synthesize_concepts call; got {n_calls}."
    )
    # Every call carried the REAL chapter_id (provenance) + a window index.
    assert {c["chapter_id"] for c in provider.calls} == {"ch1"}
    assert all(c["window_index"] is not None for c in provider.calls)

    # chapter_call_count now records the TOTAL synthesis CALL count.
    vocab = json.loads(
        Path(payload["domain_concept_vocabulary_path"]).read_text("utf-8")
    )
    assert vocab["chapter_call_count"] == n_calls, (
        "chapter_call_count is repurposed to the total synthesis-call (window) "
        f"count; expected {n_calls}, got {vocab['chapter_call_count']}."
    )


def test_stage3_unions_and_dedups_concepts_across_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vocabulary UNIONs the per-window-distinct concepts and DEDUPs the
    one concept shared across every window down to a single entry whose
    chapter_ids union is preserved.
    """
    provider = _WindowCountingStubProvider(
        concepts_per_window=2, shared_concept="cross window shared concept",
    )
    payload = _run_with_provider_instance(
        tmp_path,
        chunks=_mega_chapter_chunkset(30),
        structure=_one_mega_chapter(),
        provider_instance=provider,
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True
    n_calls = len(provider.calls)
    assert n_calls > 1

    vocab = json.loads(
        Path(payload["domain_concept_vocabulary_path"]).read_text("utf-8")
    )
    canon = [c["canonical"] for c in vocab["concepts"]]
    # No duplicate canonicals — dedup is by canonical_slug.
    assert len(canon) == len(set(canon)), f"dup canonicals: {canon}"
    assert vocab["concept_count"] == len(vocab["concepts"])

    # UNION math: 2 distinct per window × n_calls windows + 1 shared.
    expected = (2 * n_calls) + 1
    assert vocab["concept_count"] == expected, (
        f"union/dedup mismatch: {n_calls} windows × 2 distinct + 1 shared = "
        f"{expected}; got {vocab['concept_count']}."
    )
    # The shared concept appears EXACTLY ONCE (slugified).
    from lib.ontology.slugs import canonical_slug
    shared_slug = canonical_slug("cross window shared concept")
    assert canon.count(shared_slug) == 1, (
        "the concept returned by every window must dedup to one entry"
    )


def test_stage3_one_failing_window_does_not_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failing window is isolated (recorded in chapter_synthesis_failures
    at chapter granularity); the surviving windows still seed the vocabulary
    and the phase succeeds (§5.4 per-window failure isolation).
    """
    provider = _WindowCountingStubProvider(
        concepts_per_window=2, fail_indices={1},
    )
    payload = _run_with_provider_instance(
        tmp_path,
        chunks=_mega_chapter_chunkset(30),
        structure=_one_mega_chapter(),
        provider_instance=provider,
        monkeypatch=monkeypatch,
    )
    assert payload.get("success") is True, (
        "one failing window must NOT abort the phase (§5.4)"
    )
    n_calls = len(provider.calls)
    assert n_calls > 1, "need >1 window to exercise per-window isolation"

    vocab = json.loads(
        Path(payload["domain_concept_vocabulary_path"]).read_text("utf-8")
    )
    # The failed window's chapter is recorded (deduped to chapter id).
    assert vocab["chapter_synthesis_failures"] == ["ch1"], (
        "the failed window's chapter id must surface in "
        f"chapter_synthesis_failures; got {vocab['chapter_synthesis_failures']}"
    )
    # The surviving (n_calls - 1) windows each contributed 2 distinct concepts.
    assert vocab["concept_count"] == 2 * (n_calls - 1), (
        f"surviving windows = {n_calls - 1}; expected {2 * (n_calls - 1)} "
        f"concepts, got {vocab['concept_count']}."
    )
