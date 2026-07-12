"""``_run_dart_chunking`` populates ``concept_tags`` on emitted chunks.

The canonical DART chunkset path previously emitted ``concept_tags: []``
on every chunk — the extraction logic was coupled to ``CourseProcessor``
instance state, so the registry-tool path had nothing to call. That
empty concept substrate starved the downstream concept-graph + CURIE
machinery (pedagogy graph, semantic concept graph, CURIE-minting).

The instance-free ``lib.ontology.concept_tagging.extract_concept_tags``
helper (the same logic ``CourseProcessor._extract_concept_tags`` now
delegates to) is wired into ``_run_dart_chunking``'s ``_create_chunk``
callback. This pins that real-text DART input produces non-empty
``concept_tags`` on the canonical ``LibV2/courses/<slug>/dart_chunks/
chunks.jsonl`` artifact.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


@pytest.fixture
def dart_chunking_tool(monkeypatch, tmp_path):
    """Return the run_dart_chunking registry entry rooted at tmp_path."""
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    registry = _build_tool_registry()
    return registry["run_dart_chunking"]


def _write_pedagogy_html(path: Path, title: str = "Instructional Design") -> None:
    """Write HTML whose body text triggers multiple CONCEPT_PATTERNS.

    The prose deliberately mentions ``cognitive load``, ``scaffolding``,
    ``constructivism``, ``formative assessment`` and ``backward design``
    — pedagogy concepts in ``lib.ontology.concept_tagging.CONCEPT_PATTERNS``
    — so the text-based extraction path has real signal to tag.
    """
    path.write_text(
        "<!doctype html><html><head><title>"
        + title
        + "</title></head><body>"
        + "<h1>" + title + "</h1>"
        + "<section><h2>Designing for the Learner</h2>"
        + "<p>Effective instructional design manages cognitive load so "
        + "that learners are not overwhelmed by extraneous information. "
        + "Scaffolding breaks complex tasks into manageable steps, while "
        + "constructivism frames learning as the active construction of "
        + "knowledge. Backward design starts from the desired learning "
        + "outcomes and works back toward activities. Formative "
        + "assessment gives learners feedback throughout the module so "
        + "misconceptions surface early.</p></section>"
        + "</body></html>",
        encoding="utf-8",
    )


def test_dart_chunks_carry_nonempty_concept_tags(dart_chunking_tool, tmp_path):
    """Real pedagogy-rich DART HTML -> chunks with non-empty concept_tags."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_pedagogy_html(staging / "lesson_01.html", title="Lesson One")
    _write_pedagogy_html(staging / "lesson_02.html", title="Lesson Two")

    result_str = asyncio.run(dart_chunking_tool(
        course_name="DART_CONCEPT_TAGS",
        staging_dir=str(staging),
    ))
    result = json.loads(result_str)
    assert result.get("success"), f"chunking errored: {result}"

    chunks_path = Path(result["semantik_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"

    with chunks_path.open("r", encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]
    assert chunks, "expected at least one chunk emitted"

    # The chunkset is the substrate for the downstream concept-graph;
    # at least one chunk MUST carry real concept_tags (not [] anymore).
    tagged = [c for c in chunks if c.get("concept_tags")]
    assert tagged, (
        "every DART chunk emitted concept_tags=[]; the empty substrate "
        "starves the downstream concept-graph + CURIE machinery"
    )

    # concept_tags must be a list of non-empty slug strings.
    for chunk in tagged:
        tags = chunk["concept_tags"]
        assert isinstance(tags, list)
        for tag in tags:
            assert isinstance(tag, str) and tag, (
                f"chunk {chunk.get('id')!r} has malformed concept tag {tag!r}"
            )

    # The pedagogy prose anchors well-known CONCEPT_PATTERNS slugs; the
    # union across all chunks should surface several of them.
    all_tags = {t for c in chunks for t in (c.get("concept_tags") or [])}
    expected = {
        "cognitive-load", "scaffolding", "constructivism",
        "backward-design", "assessment", "feedback",
    }
    assert all_tags & expected, (
        f"expected pedagogy concept slugs from the source prose; "
        f"got {sorted(all_tags)!r}"
    )


def test_dart_concept_tags_match_helper_output(dart_chunking_tool, tmp_path):
    """Chunk tags are a subset of what the instance-free helper produces.

    Pins the wiring: _create_chunk delegates to
    ``lib.ontology.concept_tagging.extract_concept_tags`` rather than
    reimplementing its own tagger.
    """
    from lib.ontology.concept_tagging import extract_concept_tags

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_pedagogy_html(staging / "lesson_01.html", title="Solo Lesson")

    result = json.loads(asyncio.run(dart_chunking_tool(
        course_name="DART_HELPER_PARITY",
        staging_dir=str(staging),
    )))
    assert result.get("success"), f"chunking errored: {result}"

    chunks_path = Path(result["semantik_chunks_path"])
    with chunks_path.open("r", encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]

    for chunk in chunks:
        # Text-only helper run (no key_concepts). The chunk's _create_chunk
        # callback runs the SAME helper over the SAME text plus the
        # parser's key_concepts — so the text-only tag set must be a
        # subset of the chunk's emitted concept_tags. (20-tag cap aside:
        # the pedagogy fixture stays well under it.)
        helper_tags = set(extract_concept_tags(chunk["text"], {}))
        chunk_tags = set(chunk.get("concept_tags") or [])
        assert helper_tags <= chunk_tags, (
            f"chunk {chunk.get('id')!r} concept_tags {sorted(chunk_tags)} "
            f"do not cover the helper's text-only tags {sorted(helper_tags)}; "
            f"_create_chunk should delegate to the same extractor"
        )
