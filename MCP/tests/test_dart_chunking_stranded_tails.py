"""TRAINFORGE_RELOCATE_STRANDED_HEADINGS end-to-end through ``_run_dart_chunking``.

Audit defect D4 (vendor-parity audit 2026-07-18): in the SemantiK GLM-OCR
lane, the opener of the FOLLOWING section can be glued onto the TAIL of the
prior section's last chunk (46/705 chunks on a real GLM-OCR course ended with
a stranded ``"N.M EXERCISES"``-shaped marker). The chunker-level relocation
pass (``Trainforge/chunker/stranded_heading_tails.py``) is unit-tested in
``Trainforge/chunker/tests/test_stranded_heading_tails.py``; THESE tests
cover the CALL SITE — the flag-gated post-pass wired into
``MCP/tools/pipeline_tools.py::_run_dart_chunking`` after the drop/filter
passes and before the Track-K overlap pass:

  (a) flag OFF → the stranded tail stays put (byte-identical legacy emit);
  (b) flag ON  → the marker moves off the tail and opens the following
      same-flow chunk instead.
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

_MARKER = "1.1 EXERCISES"

# Two sections: the first ends with the stranded next-section marker after
# sentence-terminal punctuation (the audit shape "...Figure. 1.1 EXERCISES");
# the second is the section the marker genuinely opens. Long enough bodies so
# the chunker emits the two sections as separate chunks in the same flow.
_FILLER_A = (
    "Whole numbers build the foundation of arithmetic and appear throughout "
    "this chapter in worked examples and practice problems for every learner. "
    "Place value determines the meaning of each digit in a written numeral. "
) * 30
_BODY_A = _FILLER_A + _MARKER
_BODY_B = (
    "Practice the skills from this section by naming whole numbers, "
    "identifying place values, and rounding to the nearest ten or hundred. "
    "Each exercise mirrors a worked example from earlier in the section. "
) * 30


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
    monkeypatch.delenv("TRAINFORGE_RELOCATE_STRANDED_HEADINGS", raising=False)
    registry = _build_tool_registry()
    return registry["run_dart_chunking"]


def _write_corpus(staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "ch01.html").write_text(
        "<!doctype html><html><head><title>Chapter 1</title></head><body>"
        "<h1>Chapter 1</h1>"
        "<section><h2>1.0 Introduction</h2><p>" + _BODY_A + "</p></section>"
        "<section><h2>Exercises</h2><p>" + _BODY_B + "</p></section>"
        "</body></html>",
        encoding="utf-8",
    )


def _run(tool, staging_dir: Path) -> list:
    payload = json.loads(asyncio.run(tool(
        course_code="D4TEST",
        staging_dir=str(staging_dir),
    )))
    assert payload.get("success") is True, payload
    chunks_path = Path(payload["semantik_chunks_path"])
    return [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tail_chunks(chunks: list) -> list:
    return [c for c in chunks if c["text"].rstrip().endswith(_MARKER)]


def test_flag_off_keeps_stranded_tail(dart_chunking_tool, tmp_path):
    staging = tmp_path / "staging_off"
    _write_corpus(staging)
    chunks = _run(dart_chunking_tool, staging)
    assert len(chunks) >= 2
    # Legacy emit: the marker stays glued on the first section's tail.
    assert len(_tail_chunks(chunks)) == 1


def test_flag_on_relocates_marker_to_next_chunk(
    dart_chunking_tool, tmp_path, monkeypatch
):
    monkeypatch.setenv("TRAINFORGE_RELOCATE_STRANDED_HEADINGS", "true")
    staging = tmp_path / "staging_on"
    _write_corpus(staging)
    chunks = _run(dart_chunking_tool, staging)
    assert len(chunks) >= 2
    # No chunk ends with the stranded marker any more...
    assert _tail_chunks(chunks) == []
    # ...and the following chunk now OPENS with it.
    openers = [c for c in chunks if c["text"].lstrip().startswith(_MARKER)]
    assert len(openers) == 1
    # word_count stays consistent with the mutated text on every chunk.
    for c in chunks:
        assert c["word_count"] == len(c["text"].split())
        assert c["tokens_estimate"] == int(c["word_count"] * 1.3)
