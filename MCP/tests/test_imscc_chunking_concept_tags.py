"""``_run_imscc_chunking`` populates ``concept_tags`` on emitted chunks.

The canonical IMSCC chunkset path previously emitted ``concept_tags: []``
on every chunk (hardcoded), while the sibling ``_run_dart_chunking`` tagged
concepts via ``lib.ontology.concept_tagging.extract_concept_tags``. That
asymmetry left ``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl`` with an
empty concept substrate, starving every downstream concept-graph + CURIE
consumer that reads the IMSCC chunkset.

``_run_imscc_chunking`` now mirrors the DART callback: it parses each IMSCC
HTML page with the SAME ``HTMLContentParser`` into a page-level ``item``
carrying ``key_concepts``, then tags each chunk via the instance-free
``extract_concept_tags(text, item)`` helper (honoring the same flags). This
module pins:

* baseline (no flags): real IMSCC HTML -> non-empty ``concept_tags``,
* ``TRAINFORGE_SEED_TECH_CONCEPTS=true``: tech anchors (LangGraph / RAGAS)
  surface as concept tags.

Mirrors ``test_dart_chunking_concept_tags.py``'s structure.
"""

from __future__ import annotations

import asyncio
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


@pytest.fixture
def imscc_chunking_tool(monkeypatch, tmp_path):
    """Return the run_imscc_chunking registry entry rooted at tmp_path."""
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    registry = _build_tool_registry()
    return registry["run_imscc_chunking"]


def _build_imscc_zip(zip_path: Path, html_files: list[tuple[str, str]]) -> None:
    """Build a minimal IMSCC zip (stub imsmanifest.xml + HTML resources)."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "imsmanifest.xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1">'
            "</manifest>\n",
        )
        for inner_path, html in html_files:
            zf.writestr(inner_path, html)


def _pedagogy_html(title: str = "Instructional Design") -> str:
    """HTML whose bold key concepts + body prose trigger CONCEPT_PATTERNS.

    The bolded terms become ``key_concepts`` via the HTML parser; the prose
    mentions ``cognitive load``, ``scaffolding``, ``constructivism``,
    ``backward design`` and ``formative assessment`` — pedagogy concepts in
    ``lib.ontology.concept_tagging.CONCEPT_PATTERNS`` — so both the
    key-concept and text-pattern paths have real signal.
    """
    return (
        "<!doctype html><html><head><title>"
        + title
        + "</title></head><body>"
        + "<h1>" + title + "</h1>"
        + "<section><h2>Designing for the Learner</h2>"
        + "<p>Effective instructional design manages "
        + "<strong>cognitive load</strong> so that learners are not "
        + "overwhelmed by extraneous information. "
        + "<strong>Scaffolding</strong> breaks complex tasks into manageable "
        + "steps, while <strong>constructivism</strong> frames learning as "
        + "the active construction of knowledge. Backward design starts from "
        + "the desired learning outcomes and works back toward activities. "
        + "Formative assessment gives learners feedback throughout the module "
        + "so misconceptions surface early. "
        + " ".join(["Additional padding text to clear the chunker minimum-size threshold."] * 40)
        + "</p></section>"
        + "</body></html>"
    )


def _tech_html(title: str = "Agentic Pipelines") -> str:
    """HTML mentioning tech anchors (LangGraph / RAGAS) for the
    ``TRAINFORGE_SEED_TECH_CONCEPTS`` path."""
    return (
        "<!doctype html><html><head><title>"
        + title
        + "</title></head><body>"
        + "<h1>" + title + "</h1>"
        + "<section><h2>Frameworks</h2>"
        + "<p>This module builds a retrieval-augmented generation agent with "
        + "<strong>LangGraph</strong> for orchestration and evaluates it with "
        + "<strong>RAGAS</strong> for faithfulness and answer relevancy. "
        + "LangGraph models the agent as a state machine, while RAGAS scores "
        + "the generated answers against retrieved context. "
        + " ".join(["Padding sentence to clear the chunker minimum-size threshold."] * 40)
        + "</p></section>"
        + "</body></html>"
    )


def _read_chunks(result: dict) -> list[dict]:
    chunks_path = Path(result["imscc_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_imscc_chunks_carry_nonempty_concept_tags(imscc_chunking_tool, tmp_path):
    """Real pedagogy-rich IMSCC HTML -> chunks with non-empty concept_tags."""
    imscc_path = tmp_path / "course.imscc"
    _build_imscc_zip(
        imscc_path,
        [
            ("html/page_01.html", _pedagogy_html("Lesson One")),
            ("html/page_02.html", _pedagogy_html("Lesson Two")),
        ],
    )

    result = json.loads(asyncio.run(imscc_chunking_tool(
        course_name="IMSCC_CONCEPT_TAGS",
        imscc_path=str(imscc_path),
    )))
    assert result.get("success"), f"chunking errored: {result}"

    chunks = _read_chunks(result)
    assert chunks, "expected at least one chunk emitted"

    # The chunkset is the substrate for the downstream concept-graph; at
    # least one chunk MUST carry real concept_tags (not [] anymore).
    tagged = [c for c in chunks if c.get("concept_tags")]
    assert tagged, (
        "every IMSCC chunk emitted concept_tags=[]; the empty substrate "
        "starves the downstream concept-graph + CURIE machinery"
    )

    for chunk in tagged:
        tags = chunk["concept_tags"]
        assert isinstance(tags, list)
        for tag in tags:
            assert isinstance(tag, str) and tag, (
                f"chunk {chunk.get('id')!r} has malformed concept tag {tag!r}"
            )

    # The pedagogy prose + bold key concepts anchor well-known slugs.
    all_tags = {t for c in chunks for t in (c.get("concept_tags") or [])}
    expected = {
        "cognitive-load", "scaffolding", "constructivism",
        "backward-design", "assessment", "feedback",
    }
    assert all_tags & expected, (
        f"expected pedagogy concept slugs from the source prose; "
        f"got {sorted(all_tags)!r}"
    )


def test_imscc_concept_tags_match_helper_output(imscc_chunking_tool, tmp_path):
    """Chunk tags cover what the instance-free helper produces on the text.

    Pins the wiring: _create_chunk delegates to
    ``lib.ontology.concept_tagging.extract_concept_tags`` rather than
    reimplementing its own tagger.
    """
    from lib.ontology.concept_tagging import extract_concept_tags

    imscc_path = tmp_path / "course.imscc"
    _build_imscc_zip(
        imscc_path,
        [("html/page_01.html", _pedagogy_html("Solo Lesson"))],
    )

    result = json.loads(asyncio.run(imscc_chunking_tool(
        course_name="IMSCC_HELPER_PARITY",
        imscc_path=str(imscc_path),
    )))
    assert result.get("success"), f"chunking errored: {result}"

    for chunk in _read_chunks(result):
        # Text-only helper run (no key_concepts). The chunk's _create_chunk
        # callback runs the SAME helper over the SAME text plus the parser's
        # key_concepts — so the text-only tag set must be a subset of the
        # chunk's emitted concept_tags.
        helper_tags = set(extract_concept_tags(chunk["text"], {}))
        chunk_tags = set(chunk.get("concept_tags") or [])
        assert helper_tags <= chunk_tags, (
            f"chunk {chunk.get('id')!r} concept_tags {sorted(chunk_tags)} "
            f"do not cover the helper's text-only tags {sorted(helper_tags)}; "
            f"_create_chunk should delegate to the same extractor"
        )


def test_imscc_tech_anchors_surface_with_seed_flag(
    imscc_chunking_tool, tmp_path, monkeypatch
):
    """TRAINFORGE_SEED_TECH_CONCEPTS=true -> LangGraph / RAGAS anchors tag."""
    monkeypatch.setenv("TRAINFORGE_SEED_TECH_CONCEPTS", "true")

    imscc_path = tmp_path / "course.imscc"
    _build_imscc_zip(
        imscc_path,
        [("html/page_01.html", _tech_html("Agentic Pipelines"))],
    )

    result = json.loads(asyncio.run(imscc_chunking_tool(
        course_name="IMSCC_TECH_SEED",
        imscc_path=str(imscc_path),
    )))
    assert result.get("success"), f"chunking errored: {result}"

    chunks = _read_chunks(result)
    assert chunks, "expected at least one chunk emitted"

    all_tags = {t for c in chunks for t in (c.get("concept_tags") or [])}
    assert "langgraph" in all_tags, (
        f"expected the LangGraph tech anchor under "
        f"TRAINFORGE_SEED_TECH_CONCEPTS=true; got {sorted(all_tags)!r}"
    )
    assert "ragas" in all_tags, (
        f"expected the RAGAS tech anchor under "
        f"TRAINFORGE_SEED_TECH_CONCEPTS=true; got {sorted(all_tags)!r}"
    )
