"""Regression tests for chunking-phase bloom_level + difficulty cascade
(Wave3-Anew2, F2).

The chunk_v4 schema (`schemas/knowledge/chunk_v4.schema.json`) marks
``bloom_level`` as required on every chunk record and constrains
``difficulty`` to the 3-value enum {foundational, intermediate, advanced}.

Before Wave3-Anew2, the `_create_chunk` callbacks in
``MCP/tools/pipeline_tools.py`` (the SemantiK-chunking and IMSCC-chunking
phase handlers) omitted ``bloom_level`` entirely (100% schema
failure on the F2 calibration corpus: 72 of 72 chunks missing the
field) and hardcoded ``difficulty="intermediate"`` so all 72 chunks
shared a single difficulty regardless of cognitive demand.

These tests pin the helpers
``MCP.tools.pipeline_tools._resolve_chunk_bloom_level`` /
``_resolve_chunk_difficulty`` against three scenarios:

  1. Page-level JSON-LD ``learningObjectives[].bloomLevel`` populated →
     bloom_level lifts from the JSON-LD (``page_jsonld`` source).
  2. No JSON-LD signal but applicable Bloom verbs in chunk text →
     ``lib.ontology.bloom.detect_bloom_level`` heuristic fires
     (``verbs`` source).
  3. No JSON-LD signal AND no verb match → hardcoded ``understand``
     default (``default`` source) per the schema cascade.

For difficulty: each of the three scenarios maps the resolved bloom
through ``_BLOOM_TO_DIFFICULTY``, falling through to a keyword
heuristic over chunk text when bloom is the default fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable when the test runs from inside the
# Trainforge package directly (mirrors Trainforge/parsers/html_content_parser.py
# bootstrap).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _resolve_chunk_bloom_level,
    _resolve_chunk_difficulty,
)


# ---------------------------------------------------------------------------
# Bloom-level cascade
# ---------------------------------------------------------------------------


def test_bloom_level_from_jsonld_learning_objectives_wins() -> None:
    """Page-level JSON-LD ``bloomLevel`` short-circuits the cascade.

    A Courseforge-emitted page carrying ``courseforge_metadata.learningObjectives[]``
    with an authoritative ``bloomLevel`` field should produce
    ``(<level>, "page_jsonld")`` regardless of what verbs appear in the
    chunk text.
    """
    item: Dict[str, Any] = {
        "courseforge_metadata": {
            "learningObjectives": [
                {"id": "TO-01", "bloomLevel": "apply"},
            ]
        },
    }
    # Deliberately include a verb (``define``) that would push the verb
    # heuristic to ``remember``; the JSON-LD authoritative source must win.
    text = "Define the boundary condition before solving the equation."

    bloom_level, source = _resolve_chunk_bloom_level(item, text)

    assert bloom_level == "apply"
    assert source == "page_jsonld"


def test_bloom_level_falls_through_to_verb_heuristic_without_jsonld() -> None:
    """No JSON-LD + applicable verb in text → verb-heuristic fires.

    The canonical F2 fixture corpus has zero JSON-LD signal in the
    SemantiK HTML output; the chunker MUST detect cognitive demand from the
    chunk's prose to satisfy the schema's required-field contract.
    Verbs like ``analyze`` push the level to ``analyze`` per
    ``lib.ontology.bloom.detect_bloom_level``.
    """
    item: Dict[str, Any] = {
        "courseforge_metadata": None,
        "learning_objectives": [],
    }
    text = "Analyze the relationship between coefficients and roots."

    bloom_level, source = _resolve_chunk_bloom_level(item, text)

    assert bloom_level == "analyze"
    assert source == "verbs"


def test_bloom_level_falls_through_to_understand_default() -> None:
    """No JSON-LD AND no canonical Bloom verb → ``understand`` default.

    Per the schema's documented cascade: ``section JSON-LD > page
    JSON-LD > parsed LO > text verb heuristic > "understand" default``.
    Text without any of the 60 canonical Bloom verbs lands on the
    default tail of the chain.
    """
    item: Dict[str, Any] = {
        "courseforge_metadata": None,
        "learning_objectives": [],
    }
    # Prose with no canonical Bloom verb at all.
    text = (
        "The publication contains a copyright notice and acknowledgement "
        "of the foundation that funded the textbook."
    )

    bloom_level, source = _resolve_chunk_bloom_level(item, text)

    assert bloom_level == "understand"
    assert source == "default"


# ---------------------------------------------------------------------------
# Difficulty cascade
# ---------------------------------------------------------------------------


def test_difficulty_from_jsonld_bloom_maps_via_bloom_to_difficulty() -> None:
    """JSON-LD bloomLevel ``apply`` maps to ``intermediate`` difficulty.

    Mirrors ``Trainforge/process_course.py::BLOOM_TO_DIFFICULTY``:
    remember/understand→foundational; apply/analyze→intermediate;
    evaluate/create→advanced.
    """
    item: Dict[str, Any] = {
        "courseforge_metadata": {
            "learningObjectives": [{"id": "TO-01", "bloomLevel": "apply"}]
        },
    }
    text = "Some prose; the JSON-LD cascade short-circuits before reaching here."

    difficulty = _resolve_chunk_difficulty(item, text, bloom_level="apply")

    assert difficulty == "intermediate"


def test_difficulty_from_verb_inferred_bloom_loops_back() -> None:
    """Verb-inferred ``create`` bloom → ``advanced`` difficulty.

    When the bloom signal came from the verb heuristic (no JSON-LD, no
    LO), the resolved bloom_level is still authoritative input to the
    difficulty cascade — the helper closes the loop via the
    ``bloom_level=`` keyword argument.
    """
    item: Dict[str, Any] = {
        "courseforge_metadata": None,
        "learning_objectives": [],
    }
    text = "Design a new system that synthesizes the prior approaches."

    bloom_level, bloom_source = _resolve_chunk_bloom_level(item, text)
    assert bloom_level == "create"
    assert bloom_source == "verbs"

    difficulty = _resolve_chunk_difficulty(item, text, bloom_level=bloom_level)
    assert difficulty == "advanced"


def test_difficulty_keyword_heuristic_on_default_bloom() -> None:
    """No JSON-LD + no verb + ``understand`` default + ``define`` keyword →
    ``foundational``.

    When the bloom cascade lands on the default and the difficulty
    helper's keyword heuristic finds an introductory keyword like
    ``define`` (case-insensitive), difficulty drops to ``foundational``
    — mirroring ``Trainforge/process_course.py::_determine_difficulty``'s
    fourth fallback. Note: this scenario uses ``"What is"`` (one of the
    helper's introductory keywords) but no canonical Bloom verb, so the
    bloom heuristic still falls through to ``understand``.
    """
    item: Dict[str, Any] = {
        "courseforge_metadata": None,
        "learning_objectives": [],
    }
    # "What is" matches the introductory keyword list in
    # _resolve_chunk_difficulty; no canonical Bloom verb appears, so the
    # bloom side still returns ("understand", "default") and the
    # difficulty side picks up "foundational" from the keyword path.
    text = "What is the structure of the molecule?"

    bloom_level, bloom_source = _resolve_chunk_bloom_level(item, text)
    assert bloom_source == "default"

    difficulty = _resolve_chunk_difficulty(item, text, bloom_level=bloom_level)
    assert difficulty == "foundational"


# ---------------------------------------------------------------------------
# F2 regression sentinel — emitted chunks carry bloom_level + valid difficulty
# ---------------------------------------------------------------------------


def test_f2_regression_emitted_chunks_carry_required_fields(
    monkeypatch: Any,
) -> None:
    """End-to-end regression sentinel for F2.

    Synthesizes a parsed-item dict (the same shape the chunking-phase
    HTMLContentParser produces), threads it through
    ``Trainforge.chunker.chunk_content`` with the production
    chunking-phase ``_create_chunk`` callback shape (re-built here
    via the public ``_resolve_chunk_*`` helpers to avoid reaching into
    the helper's closure), and asserts every emitted chunk carries
    ``bloom_level`` (required by schema) and a valid ``difficulty``
    enum value.
    """
    from Trainforge.chunker import ChunkerContext, chunk_content
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    # Force position-based chunk IDs so the test is deterministic
    # regardless of operator env.
    monkeypatch.delenv("TRAINFORGE_CONTENT_HASH_IDS", raising=False)

    html_parser = HTMLContentParser()
    # Two sections, each long enough (~600 words) to clear
    # ``MIN_CHUNK_SIZE=100`` after ``merge_small_sections`` so they emit
    # as distinct chunks. The first uses ``analyze`` verbs (Bloom level
    # ``analyze``); the second uses ``define`` verbs (Bloom level
    # ``remember``) — distinct enough that the verb heuristic must yield
    # at least two distinct bloom_level values across the corpus.
    html_text = (
        "<html><head><title>Test Page</title></head><body>"
        "<h1>Test Page</h1>"
        "<h2>Analyze coefficients</h2>"
        "<p>" + ("Analyze the polynomial roots carefully. " * 120) + "</p>"
        "<h2>Define key terms</h2>"
        "<p>" + ("Define the canonical form of each expression. " * 120) + "</p>"
        "</body></html>"
    )
    parsed = html_parser.parse(html_text)
    item: Dict[str, Any] = {
        "item_id": "test-page",
        "item_path": "test-page.html",
        "title": parsed.title or "Test Page",
        "resource_type": "page",
        "module_id": "test-page",
        "module_title": parsed.title or "Test Page",
        "week_num": 0,
        "word_count": parsed.word_count,
        "sections": parsed.sections,
        "learning_objectives": parsed.learning_objectives,
        "key_concepts": parsed.key_concepts,
        "interactive_components": parsed.interactive_components,
        "raw_html": html_text,
        "page_id": parsed.page_id,
        "misconceptions": parsed.misconceptions,
        "suggested_assessment_types": parsed.suggested_assessment_types,
        "courseforge_metadata": parsed.metadata.get("courseforge"),
        "objective_refs": parsed.objective_refs,
        "source_references": parsed.source_references,
    }

    def _create_chunk(
        *,
        chunk_id: str,
        text: str,
        html: str,
        item: Dict[str, Any],
        section_heading: str,
        chunk_type: str,
        follows_chunk_id: Any = None,
        position_in_module: int = 0,
        html_xpath: Any = None,
        char_span: Any = None,
        section_source_ids: Any = None,
        merged_headings: Any = None,
    ) -> Dict[str, Any]:
        words = text.split()
        bloom_level, bloom_source = _resolve_chunk_bloom_level(item, text)
        difficulty = _resolve_chunk_difficulty(item, text, bloom_level)
        chunk: Dict[str, Any] = {
            "id": chunk_id,
            "schema_version": "v4",
            "chunk_type": chunk_type,
            "text": text,
            "html": html,
            "follows_chunk": follows_chunk_id,
            "source": {
                "course_id": "TEST_F2",
                "module_id": item.get("module_id") or "",
                "lesson_id": item.get("item_id") or "",
                "section_heading": section_heading,
                "position_in_module": position_in_module,
            },
            "concept_tags": [],
            "learning_outcome_refs": [],
            "difficulty": difficulty,
            "tokens_estimate": int(len(words) * 1.3),
            "word_count": len(words),
            "bloom_level": bloom_level,
        }
        if bloom_source in ("verbs", "default"):
            chunk["bloom_level_source"] = bloom_source
        return chunk

    ctx = ChunkerContext(create_chunk=_create_chunk)
    result = chunk_content([item], "TEST_F2", boilerplate_spans=None, ctx=ctx)

    assert len(result.chunks) > 0, "fixture produced zero chunks"
    valid_difficulties = {"foundational", "intermediate", "advanced"}
    valid_blooms = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
    for chunk in result.chunks:
        # Wave3-Anew2 F2 contract: every chunk MUST carry bloom_level.
        assert "bloom_level" in chunk, (
            f"chunk {chunk.get('id')!r} missing bloom_level — F2 regression"
        )
        assert chunk["bloom_level"] in valid_blooms, (
            f"chunk {chunk.get('id')!r} bloom_level={chunk['bloom_level']!r} "
            f"not in canonical enum"
        )
        assert chunk["difficulty"] in valid_difficulties, (
            f"chunk {chunk.get('id')!r} difficulty={chunk['difficulty']!r} "
            f"not in canonical enum"
        )

    # The fixture has two sections — one with "analyze" verbs, one with
    # "define" verbs. The verb-heuristic must yield at least two distinct
    # bloom_level values across the corpus (i.e. NOT uniformly "intermediate"
    # like the pre-fix full-book run that produced 72/72 identical
    # difficulty values).
    bloom_values = {c["bloom_level"] for c in result.chunks}
    assert len(bloom_values) >= 2, (
        f"verb-heuristic should produce diverse bloom_levels; "
        f"got uniform {bloom_values}"
    )
