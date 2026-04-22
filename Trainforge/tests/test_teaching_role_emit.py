"""Regression tests for REC-VOC-02 (Wave 2, Worker K).

Covers the Courseforge emit side and the Trainforge consume precedence:

* Courseforge render helpers emit ``data-cf-teaching-role`` deterministically
  from the schema's ``x-component-mapping`` for flip-card / self-check /
  activity components.
* ``_build_sections_metadata`` emits a ``teachingRole`` array on section
  JSON-LD entries when tagged components are present.
* ``Trainforge/align_chunks.classify_teaching_roles`` PREFERS the
  deterministic signal (``data-cf-teaching-role`` → chunk
  ``teaching_role_attr``; JSON-LD ``section_teaching_roles``) over the
  existing heuristic / LLM classifier, and records provenance via
  ``teaching_role_source``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Importer for generate_course.py (it's a script, not a package).
# ---------------------------------------------------------------------------

def _load_generate_course():
    """Load ``Courseforge/scripts/generate_course.py`` as a module."""
    path = _REPO_ROOT / "Courseforge" / "scripts" / "generate_course.py"
    spec = importlib.util.spec_from_file_location("generate_course_worker_k", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Emit-side tests (Courseforge)
# ---------------------------------------------------------------------------

def test_flip_card_emits_introduce():
    """Every flip-card renders with ``data-cf-teaching-role="introduce"``."""
    gc = _load_generate_course()
    html = gc._render_flip_cards([
        {"term": "API", "definition": "Application Programming Interface"},
        {"term": "REST", "definition": "Representational State Transfer"},
    ])
    matches = re.findall(r'data-cf-teaching-role="([^"]+)"', html)
    assert len(matches) == 2, f"expected 2 flip-cards with role, got {len(matches)}"
    assert all(m == "introduce" for m in matches), (
        f"flip-card teaching_role should be 'introduce', got {matches}"
    )
    # Sanity: component/purpose also present (regression guard against a
    # future refactor accidentally dropping the source pair).
    assert 'data-cf-component="flip-card"' in html
    assert 'data-cf-purpose="term-definition"' in html


def test_self_check_emits_assess():
    """Self-check blocks render with ``data-cf-teaching-role="assess"``."""
    gc = _load_generate_course()
    html = gc._render_self_check([
        {
            "question": "What is REST?",
            "options": [
                {"text": "A style", "correct": True, "feedback": "Right!"},
                {"text": "A color", "correct": False, "feedback": "No."},
            ],
            "bloom_level": "remember",
        },
    ])
    matches = re.findall(r'data-cf-teaching-role="([^"]+)"', html)
    assert matches == ["assess"], (
        f"self-check teaching_role should be 'assess' (one occurrence), got {matches}"
    )
    assert 'data-cf-component="self-check"' in html
    assert 'data-cf-purpose="formative-assessment"' in html


def test_activity_emits_transfer():
    """Activity cards render with ``data-cf-teaching-role="transfer"``."""
    gc = _load_generate_course()
    html = gc._render_activities([
        {"title": "Design an API", "description": "Sketch endpoints.", "bloom_level": "apply"},
        {"title": "Review a spec", "description": "Evaluate clarity.", "bloom_level": "evaluate"},
    ])
    matches = re.findall(r'data-cf-teaching-role="([^"]+)"', html)
    assert len(matches) == 2, f"expected 2 activities with role, got {len(matches)}"
    assert all(m == "transfer" for m in matches), (
        f"activity teaching_role should be 'transfer', got {matches}"
    )
    assert 'data-cf-component="activity"' in html
    assert 'data-cf-purpose="practice"' in html


def test_section_jsonld_teaching_role_array():
    """Sections with tagged components emit a ``teachingRole`` array in JSON-LD."""
    gc = _load_generate_course()
    sections = [
        {
            "heading": "Terminology",
            "content_type": "definition",
            "flip_cards": [
                {"term": "HTTP", "definition": "protocol"},
                {"term": "URL", "definition": "locator"},
            ],
        },
        {
            "heading": "Narrative",
            "content_type": "explanation",
            "paragraphs": ["Prose without tagged components."],
        },
    ]
    result = gc._build_sections_metadata(sections)
    assert len(result) == 2

    # First section: flip_cards present → teachingRole == ['introduce']
    first = result[0]
    assert first["heading"] == "Terminology"
    assert first.get("teachingRole") == ["introduce"], (
        f"expected teachingRole=['introduce'] on section 0, got {first.get('teachingRole')!r}"
    )

    # Second section: no tagged components → no teachingRole key emitted
    second = result[1]
    assert "teachingRole" not in second, (
        f"expected no teachingRole on plain-prose section, got {second.get('teachingRole')!r}"
    )


def test_section_jsonld_multi_role_sorted():
    """Sections with multiple tagged component types produce a sorted list."""
    gc = _load_generate_course()
    sections = [
        {
            "heading": "Hybrid",
            "flip_cards": [{"term": "X", "definition": "y"}],
            "self_check": [{"question": "?", "options": []}],
            "activities": [{"title": "go", "description": "do"}],
        },
    ]
    result = gc._build_sections_metadata(sections)
    assert len(result) == 1
    roles = result[0].get("teachingRole", [])
    # sorted() on {"introduce", "assess", "transfer"} → ["assess", "introduce", "transfer"]
    assert roles == sorted(roles), (
        f"teachingRole must be sorted for diff-friendly output, got {roles}"
    )
    assert set(roles) == {"introduce", "assess", "transfer"}


# ---------------------------------------------------------------------------
# Consume-side tests (Trainforge align_chunks)
# ---------------------------------------------------------------------------

def test_align_chunks_prefers_deterministic():
    """An explicit ``teaching_role_attr`` bypasses heuristic and LLM paths."""
    from Trainforge.align_chunks import classify_teaching_roles

    chunks = [
        {
            "id": "c1",
            "_position": 0,
            "teaching_role_attr": "introduce",
            "chunk_type": "content",
            "text": "intro text",
            "source": {"resource_type": "overview", "position_in_module": 0},
        },
    ]
    # Use anthropic provider — if we fell through, the missing anthropic
    # package would trigger the LLM path (which then mocks). Deterministic
    # short-circuit means we never reach that code.
    classify_teaching_roles(chunks, llm_provider="anthropic", verbose=False)
    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[0]["teaching_role_source"] == "attr"


def test_align_chunks_jsonld_precedence():
    """An unambiguous JSON-LD section role resolves without the LLM."""
    from Trainforge.align_chunks import classify_teaching_roles

    chunks = [
        {
            "id": "c2",
            "_position": 0,
            "chunk_type": "content",
            "text": "activity-style chunk",
            "source": {
                "resource_type": "application",
                "section_teaching_roles": ["transfer"],
            },
        },
    ]
    classify_teaching_roles(chunks, llm_provider="mock", verbose=False)
    # Deterministic path MUST win over the _heuristic_role that would
    # otherwise fire for resource_type="application".
    assert chunks[0]["teaching_role"] == "transfer"
    assert chunks[0]["teaching_role_source"] == "jsonld"


def test_align_chunks_ambiguous_jsonld_falls_through():
    """Multi-value JSON-LD section roles fall through to heuristic/LLM."""
    from Trainforge.align_chunks import classify_teaching_roles

    chunks = [
        {
            "id": "c3",
            "_position": 0,
            "chunk_type": "content",
            "text": "ambiguous",
            "source": {
                "resource_type": "overview",
                "position_in_module": 0,
                "section_teaching_roles": ["introduce", "assess"],
            },
        },
    ]
    classify_teaching_roles(chunks, llm_provider="mock", verbose=False)
    # Should NOT pick one of the two jsonld roles; must fall through. The
    # _heuristic_role matches overview/position-0 → "introduce".
    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[0]["teaching_role_source"] == "heuristic"


def test_align_chunks_heuristic_still_works_without_attrs():
    """Chunks without deterministic metadata still use the legacy heuristic."""
    from Trainforge.align_chunks import classify_teaching_roles

    chunks = [
        {
            "id": "c4",
            "_position": 0,
            "chunk_type": "assessment_item",
            "text": "Q1",
            "source": {"resource_type": "quiz"},
        },
    ]
    classify_teaching_roles(chunks, llm_provider="mock", verbose=False)
    assert chunks[0]["teaching_role"] == "assess"
    assert chunks[0]["teaching_role_source"] == "heuristic"


def test_align_chunks_mock_fallback_preserved():
    """Chunks with no metadata and no heuristic hit get the mock fallback."""
    from Trainforge.align_chunks import classify_teaching_roles

    chunks = [
        {
            "id": "c5",
            "_position": 0,
            "chunk_type": "content",
            "concept_tags": ["topic_a"],
            "text": "freshly introduced concept",
            "source": {"resource_type": "content"},
        },
    ]
    classify_teaching_roles(chunks, llm_provider="mock", verbose=False)
    # _mock_role returns "introduce" when no earlier concepts seen.
    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[0]["teaching_role_source"] == "mock"


# ---------------------------------------------------------------------------
# HTML parser surface test
# ---------------------------------------------------------------------------

def test_html_parser_surfaces_teaching_role():
    """``ContentSection.teaching_role`` populated from body data-cf-teaching-role."""
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    html = """
    <h2 data-cf-content-type="definition">Terminology</h2>
    <div class="flip-card-grid">
      <div class="flip-card" data-cf-component="flip-card"
           data-cf-purpose="term-definition"
           data-cf-teaching-role="introduce"
           data-cf-term="api">...</div>
      <div class="flip-card" data-cf-component="flip-card"
           data-cf-purpose="term-definition"
           data-cf-teaching-role="introduce"
           data-cf-term="rest">...</div>
    </div>
    <h2 data-cf-content-type="explanation">Narrative</h2>
    <p>No tagged components.</p>
    """
    module = HTMLContentParser().parse(html)
    assert len(module.sections) == 2

    first = module.sections[0]
    assert first.teaching_role == "introduce"
    assert first.teaching_roles == ["introduce"]

    second = module.sections[1]
    assert second.teaching_role is None
    assert second.teaching_roles == []


def test_html_parser_ambiguous_teaching_role_stays_none():
    """Multi-value section → teaching_role None but teaching_roles lists all."""
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    html = """
    <h2 data-cf-content-type="hybrid">Mixed</h2>
    <div data-cf-teaching-role="introduce">flip-card</div>
    <div data-cf-teaching-role="assess">self-check</div>
    """
    module = HTMLContentParser().parse(html)
    assert len(module.sections) == 1
    section = module.sections[0]
    assert section.teaching_role is None
    assert sorted(section.teaching_roles) == ["assess", "introduce"]
