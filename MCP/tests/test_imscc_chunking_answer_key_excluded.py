"""``_run_imscc_chunking`` must NOT chunk the instructor answer-key sidecar.

The assessment-quality overhaul ships ``06_assessments/answer_key.html`` — an
instructor-only page carrying correct answers + worked solutions — as a CC
``webcontent`` resource inside the packaged ``.imscc``. The IMSCC chunking phase
walks every ``.html`` entry in the archive to build the retrieval corpus, so
without an explicit guard the answer key's correct answers would become
retrievable chunks (answer-key leakage — exactly what the assessment guard
exists to prevent).

This pins the guard: an ``06_assessments/*.html`` sidecar produces ZERO
retrieval chunks, while an ordinary learner content page still chunks normally.
The QTI XML in the same directory continues to be harvested separately into
``assessment_item`` chunks (covered elsewhere) — only the HTML sidecar is
excluded here.

Mirrors ``test_imscc_chunking_concept_tags.py``'s harness.
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

# A distinctive marker unlikely to collide with any real prose — if it appears
# in a chunk, the answer key leaked into the retrieval corpus.
_ANSWER_KEY_MARKER = "ZZQ_ANSWER_KEY_SECRET_SOLUTION_MARKER"
_LEARNER_MARKER = "ZZQ_LEARNER_CONTENT_MARKER"


@pytest.fixture
def imscc_chunking_tool(monkeypatch, tmp_path):
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    return _build_tool_registry()["run_imscc_chunking"]


def _padded(marker: str, title: str) -> str:
    return (
        "<!doctype html><html><head><title>" + title + "</title></head><body>"
        + "<h1>" + title + "</h1><section><h2>Body</h2><p>"
        + marker + ". "
        + " ".join(["Padding sentence to clear the chunker size threshold."] * 60)
        + "</p></section></body></html>"
    )


def _read_chunks(result: dict) -> list[dict]:
    chunks_path = Path(result["imscc_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_answer_key_html_never_enters_retrieval_corpus(
    imscc_chunking_tool, tmp_path,
):
    imscc_path = tmp_path / "course.imscc"
    with zipfile.ZipFile(imscc_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "imsmanifest.xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1">'
            "</manifest>\n",
        )
        # A normal learner content page (must chunk).
        zf.writestr("week_01/week_01_content.html",
                    _padded(_LEARNER_MARKER, "Lesson One"))
        # The instructor answer-key sidecar (must NOT chunk).
        zf.writestr("06_assessments/answer_key.html",
                    _padded(_ANSWER_KEY_MARKER, "Instructor Answer Key"))

    result = json.loads(asyncio.run(imscc_chunking_tool(
        course_name="IMSCC_LEAK_GUARD",
        imscc_path=str(imscc_path),
    )))
    assert result.get("success"), f"chunking errored: {result}"

    chunks = _read_chunks(result)
    assert chunks, "expected the learner content page to produce chunks"

    all_text = "\n".join(str(c.get("text") or "") for c in chunks)
    # The learner page chunked normally...
    assert _LEARNER_MARKER in all_text, (
        "the ordinary learner content page must still be chunked"
    )
    # ...but the instructor answer key never entered the retrieval corpus.
    assert _ANSWER_KEY_MARKER not in all_text, (
        "answer-key leakage: 06_assessments/answer_key.html was chunked into "
        "the retrieval corpus"
    )
    # And no chunk's source item_path derives from the 06_assessments sidecar.
    for c in chunks:
        src = c.get("source") or {}
        item_path = str(src.get("item_path") or "")
        assert "06_assessments" not in item_path, (
            f"chunk {c.get('id')!r} is sourced from the 06_assessments sidecar "
            f"({item_path})"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
