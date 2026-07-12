"""Wave #22 stamp-site coverage: ``_run_dart_chunking`` carries
``composite_unit`` / ``unit_roles`` into the emitted ``chunks.jsonl``.

Contract under test (end-to-end through the ``_run_dart_chunking`` registry
callback, mirroring ``test_dart_chunking_source_refs.py``):

  - Staged DART HTML whose ``<section class="dart-unit" data-dart-unit="...">``
    wrapper + ``data-dart-opener`` / ``data-dart-flow`` block attributes are
    harvested by the parser onto ``ContentSection`` and aggregated by the
    chunker's ``section_unit_signals`` / ``aggregate_composite_unit``
    helpers -> the emitted chunk dict carries ``composite_unit`` (type string)
    and ``unit_roles`` (sorted-distinct list).
  - Legacy / non-DART HTML carrying none of the attributes -> the chunk omits
    both fields (additive, byte-stable contract — mirrors source_block_role).

Synthetic staged HTML only (no corpus files); no GPU / no real cascade.
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


# Prose long enough that the worked-example unit survives as a real chunk
# (not folded/dropped as furniture).
_STATEMENT = (
    "Simplify the radical expression fully by factoring out the perfect "
    "square first, then reduce the remaining radical to lowest terms so the "
    "expression is completely simplified for the learner to follow along."
)
_SOLUTION = (
    "First factor out the perfect square from under the radical sign, then "
    "reduce the coefficient and combine like terms to reach the final "
    "simplified radical form that answers the worked example completely."
)


def _write_unit_html(path: Path) -> None:
    """DART HTML: a worked-example composite unit carrying data-dart-unit on
    the wrapper + data-dart-opener / data-dart-flow on the body blocks."""
    path.write_text(
        "<!doctype html><html><head><title>Ch 9 Radicals</title></head><body>"
        '<main class="dart-document"><article role="doc-chapter"><h2>Ch 9</h2>'
        '<section class="dart-unit dart-unit-worked_example" '
        'data-dart-unit="worked_example" role="group">'
        '<section class="dart-section" data-dart-block-id="ex-1" '
        'data-dart-source="synthesized" data-dart-pages="4" '
        'data-dart-opener="worked_example" data-dart-flow="statement">'
        f"<h4>Example 9.1</h4><p>{_STATEMENT}</p></section>"
        '<section class="dart-section" data-dart-block-id="s1" '
        'data-dart-source="synthesized" data-dart-pages="4" '
        'data-dart-flow="solution-steps">'
        f"<p>{_SOLUTION}</p></section>"
        "</section></article></main></body></html>",
        encoding="utf-8",
    )


def _load_chunks(result_str: str):
    result = json.loads(result_str)
    assert result.get("success"), f"chunking errored: {result}"
    chunks_path = Path(result["semantik_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_dart_chunks_carry_pedagogical_unit_and_roles(dart_chunking_tool, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_unit_html(staging / "lesson_09.html")

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="PED_UNIT_TEST",
        staging_dir=str(staging),
    )))
    assert chunks, "expected at least one chunk emitted"

    unit_chunks = [c for c in chunks if c.get("composite_unit")]
    assert unit_chunks, "no chunk stamped composite_unit from data-dart-unit"
    c = unit_chunks[0]
    assert c["composite_unit"] == "worked_example"

    roles = c.get("unit_roles")
    assert isinstance(roles, list) and roles, "unit_roles missing/empty"
    # Union of the opener role + the two flows; sorted-distinct.
    assert set(roles) == {"worked_example", "statement", "solution-steps"}


def test_legacy_dart_html_omits_pedagogical_fields(dart_chunking_tool, tmp_path):
    """DART HTML with NO data-dart-unit/-opener/-flow attributes: chunks omit
    both pedagogical fields (additive, byte-stable contract)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "legacy.html").write_text(
        "<!doctype html><html><head><title>Legacy</title></head><body>"
        '<main class="dart-document"><article><h2>Overview</h2>'
        '<section class="dart-section" data-dart-block-id="s1" '
        'data-dart-source="synthesized" data-dart-pages="1">'
        f"<h3>Intro</h3><p>{_STATEMENT}</p></section>"
        "</article></main></body></html>",
        encoding="utf-8",
    )

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="PED_LEGACY_TEST",
        staging_dir=str(staging),
    )))
    assert chunks, "expected at least one chunk emitted"
    for c in chunks:
        assert "composite_unit" not in c
        assert "unit_roles" not in c
