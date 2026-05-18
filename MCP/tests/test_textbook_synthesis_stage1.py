"""Stage-1 integration tests — three-stage textbook synthesis, Wave B.

Plan reference: ``plans/textbook-llm-synthesis-3stage-2026-05.md`` §3 +
test-plan item 3.

Covers the ``TEXTBOOK_SYNTHESIS_PROVIDER`` short-circuit folded into
``MCP/tools/pipeline_tools.py::_extract_textbook_structure`` (Wave B):

* **flag unset** → ``textbook_structure.json`` byte-stable: no
  ``semantic_outline`` / ``draft_terminal_objectives`` /
  ``structure_enrichment`` top-level keys.
* **flag set** (mock/stub provider) → the three Stage-1 enrichment keys
  are present and gate-shaped — i.e. the folded artifact passes
  ``TextbookOutlineValidator`` cleanly.
* **provider raises** → ``TextbookSynthesisProviderError`` propagates
  out of the handler (fail-loud, plan §2.2).

No real LLM calls — the provider is a stub.
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
from Courseforge.generators import (  # noqa: E402
    _textbook_synthesis_provider as tsp_mod,
)
from Courseforge.generators._textbook_synthesis_provider import (  # noqa: E402
    TextbookSynthesisProviderError,
)
from lib.validators.textbook_structure import (  # noqa: E402
    TextbookOutlineValidator,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _write_dart_html(path: Path, chapters: list) -> None:
    """Write a minimal DART-like HTML with <article role="doc-chapter">."""
    body_parts = [
        '<a class="skip-link" href="#main">Skip</a>',
        '<main role="main">',
    ]
    for idx, ch in enumerate(chapters, start=1):
        body_parts.append(f'<article role="doc-chapter" id="chap-{idx}">')
        body_parts.append(
            f'<header><h2 id="ch{idx}-title">{ch["title"]}</h2></header>'
        )
        for sec_idx, sec in enumerate(ch.get("sections", []), start=1):
            body_parts.append(
                f'<section aria-labelledby="ch{idx}-s{sec_idx}">'
            )
            body_parts.append(
                f'<h3 id="ch{idx}-s{sec_idx}">{sec["title"]}</h3>'
            )
            for para in sec.get("paragraphs", []):
                body_parts.append(f"<p>{para}</p>")
            body_parts.append("</section>")
        body_parts.append("</article>")
    body_parts.append("</main>")
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{chapters[0]["title"] if chapters else "Empty"}</title>'
        "</head><body>"
        + "".join(body_parts)
        + "</body></html>"
    )
    path.write_text(html, encoding="utf-8")


@pytest.fixture
def extractor_fixture(tmp_path, monkeypatch):
    """Re-root the pipeline_tools project tree under tmp_path."""
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", fake_root)
    if hasattr(pipeline_tools, "COURSEFORGE_INPUTS"):
        monkeypatch.setattr(
            pipeline_tools,
            "COURSEFORGE_INPUTS",
            fake_root / "Courseforge" / "inputs" / "textbooks",
        )
    # Default-off: ensure the env var is unset for every test unless a
    # test explicitly sets it.
    monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_dart_html(staging / "book.html", [
        {
            "title": "Chapter 1: Kinematics",
            "sections": [
                {"title": "Velocity and Acceleration",
                 "paragraphs": [
                     "Velocity describes the rate of change of position. "
                     "Acceleration is the rate of change of velocity over "
                     "time, a foundational kinematics concept."]},
            ],
        },
        {
            "title": "Chapter 2: Forces",
            "sections": [
                {"title": "Newton's Laws of Motion",
                 "paragraphs": [
                     "Newton's first law states that an object in motion "
                     "tends to stay in motion unless acted upon by an "
                     "external force."]},
            ],
        },
    ])
    return {"root": fake_root, "staging": staging}


async def _call(**kwargs):
    registry = _build_tool_registry()
    assert "extract_textbook_structure" in registry
    fn = registry["extract_textbook_structure"]
    raw = await fn(**kwargs)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Stub provider — folds gate-shaped enrichment with NO LLM call
# ---------------------------------------------------------------------------


class _StubSynthesisProvider:
    """Stand-in for TextbookSynthesisProvider.synthesize_outline.

    Returns a gate-shaped enrichment payload built from the real
    chapter IDs on the supplied ``textbook_structure`` — so the folded
    artifact passes TextbookOutlineValidator cleanly. No LLM call.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002
        self.capture = kwargs.get("capture")
        self.init_args = (args, kwargs)

    def synthesize_outline(self, textbook_structure, *, course_name):
        chapter_ids = [
            str(ch.get("id"))
            for ch in (textbook_structure.get("chapters") or [])
            if isinstance(ch, dict) and ch.get("id")
        ]
        return {
            "semantic_outline": {
                "course_summary": (
                    f"Synthesized course summary for {course_name}."
                ),
                "themes": [
                    {
                        "title": "Core Mechanics",
                        "chapter_ids": chapter_ids,
                        "summary": "Foundational mechanics theme.",
                        "prerequisite_theme_titles": [],
                    }
                ],
            },
            "draft_terminal_objectives": [
                {
                    "id": "TO-01",
                    "statement": (
                        "Analyze motion using kinematic relationships."
                    ),
                    "bloom_level": "analyze",
                    "bloom_verb": "analyze",
                    "abcd": {
                        "audience": "the student",
                        "behavior": {
                            "verb": "analyze",
                            "action_object": "kinematic motion",
                        },
                        "condition": "given a problem set",
                        "degree": "with 80% accuracy",
                    },
                    "source_refs": [
                        {"ref": cid, "chunk_ids": []}
                        for cid in chapter_ids[:1]
                    ],
                    "draft": True,
                }
            ],
            "structure_enrichment": {
                "enriched": True,
                "provider": "stub",
                "model": "stub-model",
                "call_mode": "single",
                "calls": 1,
                "schema_version": "v1",
            },
        }


class _RaisingSynthesisProvider:
    """Stub whose synthesize_outline raises the fail-loud error."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002
        pass

    def synthesize_outline(self, textbook_structure, *, course_name):
        raise TextbookSynthesisProviderError(
            f"stub exhausted on {course_name}",
            code="outline_exhausted",
        )


# ---------------------------------------------------------------------------
# Test 1 — flag unset → byte-stable, no new keys
# ---------------------------------------------------------------------------


_ENRICHMENT_KEYS = (
    "semantic_outline",
    "draft_terminal_objectives",
    "structure_enrichment",
)


def test_flag_unset_textbook_structure_byte_stable(extractor_fixture):
    """Default-off: textbook_structure.json carries no enrichment keys."""
    fx = extractor_fixture
    result = asyncio.run(_call(
        course_name="PHYS_101",
        staging_dir=str(fx["staging"]),
    ))
    assert result["success"]
    doc = json.loads(
        Path(result["textbook_structure_path"]).read_text(encoding="utf-8")
    )
    for key in _ENRICHMENT_KEYS:
        assert key not in doc, (
            f"default-off run leaked Stage-1 key {key!r} into "
            f"textbook_structure.json"
        )


def test_flag_unset_does_not_construct_provider(
    extractor_fixture, monkeypatch
):
    """Default-off: the provider class is never instantiated."""
    constructed = []

    class _Tripwire(_StubSynthesisProvider):
        def __init__(self, *args, **kwargs):  # noqa: ANN002
            constructed.append(True)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tsp_mod, "TextbookSynthesisProvider", _Tripwire)
    fx = extractor_fixture
    result = asyncio.run(_call(
        course_name="PHYS_101",
        staging_dir=str(fx["staging"]),
    ))
    assert result["success"]
    assert constructed == [], (
        "provider must NOT be constructed when "
        "TEXTBOOK_SYNTHESIS_PROVIDER is unset"
    )


# ---------------------------------------------------------------------------
# Test 2 — flag set + stub provider → 3 keys present + gate-shaped
# ---------------------------------------------------------------------------


def test_flag_set_folds_three_enrichment_keys(extractor_fixture, monkeypatch):
    """Flag set: the three Stage-1 keys are folded into the artifact."""
    monkeypatch.setattr(
        tsp_mod, "TextbookSynthesisProvider", _StubSynthesisProvider
    )
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", "anthropic")
    fx = extractor_fixture
    result = asyncio.run(_call(
        course_name="PHYS_101",
        staging_dir=str(fx["staging"]),
    ))
    assert result["success"]
    doc = json.loads(
        Path(result["textbook_structure_path"]).read_text(encoding="utf-8")
    )
    for key in _ENRICHMENT_KEYS:
        assert key in doc, f"flag-set run missing Stage-1 key {key!r}"

    # The semantic_outline themes resolve against real chapter IDs.
    chapter_ids = {c["id"] for c in doc["chapters"]}
    theme = doc["semantic_outline"]["themes"][0]
    assert set(theme["chapter_ids"]).issubset(chapter_ids)
    assert doc["draft_terminal_objectives"][0]["draft"] is True
    assert doc["structure_enrichment"]["enriched"] is True
    assert doc["structure_enrichment"]["schema_version"] == "v1"


def test_flag_set_artifact_passes_outline_validator(
    extractor_fixture, monkeypatch
):
    """The folded artifact is gate-shaped — TextbookOutlineValidator
    passes with zero issues against the enriched textbook_structure."""
    monkeypatch.setattr(
        tsp_mod, "TextbookSynthesisProvider", _StubSynthesisProvider
    )
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", "anthropic")
    fx = extractor_fixture
    result = asyncio.run(_call(
        course_name="PHYS_101",
        staging_dir=str(fx["staging"]),
    ))
    structure_path = Path(result["textbook_structure_path"])

    gate = TextbookOutlineValidator().validate(
        {"textbook_structure_path": str(structure_path)}
    )
    assert gate.passed is True
    assert [i for i in gate.issues if i.severity == "critical"] == []
    assert gate.issues == [], (
        f"enriched artifact should be gate-clean; got "
        f"{[(i.code, i.message) for i in gate.issues]}"
    )


def test_default_off_artifact_skips_outline_validator(extractor_fixture):
    """Sanity: a default-off artifact skips-with-pass on the validator
    (the ABCD_MISSING graceful-degrade contract)."""
    fx = extractor_fixture
    result = asyncio.run(_call(
        course_name="PHYS_101",
        staging_dir=str(fx["staging"]),
    ))
    gate = TextbookOutlineValidator().validate(
        {"textbook_structure_path": result["textbook_structure_path"]}
    )
    assert gate.passed is True
    assert gate.issues == []


# ---------------------------------------------------------------------------
# Test 3 — fail-loud on provider error (plan §2.2)
# ---------------------------------------------------------------------------


def test_flag_set_provider_error_propagates(extractor_fixture, monkeypatch):
    """Flag set + provider exhausts parse → TextbookSynthesisProviderError
    propagates out of the handler (Stage 1 is one call → fail-loud)."""
    monkeypatch.setattr(
        tsp_mod, "TextbookSynthesisProvider", _RaisingSynthesisProvider
    )
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", "anthropic")
    fx = extractor_fixture
    with pytest.raises(TextbookSynthesisProviderError) as exc_info:
        asyncio.run(_call(
            course_name="PHYS_101",
            staging_dir=str(fx["staging"]),
        ))
    assert exc_info.value.code == "outline_exhausted"
