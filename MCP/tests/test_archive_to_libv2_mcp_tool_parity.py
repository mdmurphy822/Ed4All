"""Phase 8 ST 1 — ``@mcp.tool() async def archive_to_libv2`` SHA-emit parity.

Contract: the ``@mcp.tool()`` variant of ``archive_to_libv2`` at
``MCP/tools/pipeline_tools.py:1249-1503`` must accept the three Phase 6 +
Phase 7c.5 SHA kwargs (``concept_graph_sha256``, ``dart_chunks_sha256``,
``imscc_chunks_sha256``) and persist each into the LibV2 manifest when the
value matches the canonical 64-hex regex ``^[0-9a-f]{64}$``. Mirrors the
registry variant emit pattern at ``:5667-5687, :5720-5727``.

Intentional asymmetry vs registry variant (per plans/phase8_cleanup.md
pre-resolved decision #1): the ``@mcp.tool()`` variant is kwarg-only — no
on-disk recompute fallback. External MCP clients pass paths explicitly;
recomputation is the workflow-runner-driven registry variant's job.

Surfaces tested:
    - All three SHA kwargs accepted in the function signature and persisted
      into the manifest top-level when each value is well-formed 64-hex.
    - Malformed SHAs (non-64-hex, non-lowercase-hex, wrong length) silently
      dropped (mirrors the registry variant's ``INVALID_*`` fall-through).
    - Absent SHAs (default ``None``) result in absent manifest fields
      (back-compat with legacy MCP clients calling without them).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import register_pipeline_tools  # noqa: E402


# Well-formed canonical 64-hex strings used as kwarg fixtures. Distinct
# values per kwarg so a misrouted persist (e.g. dart kwarg landing in the
# concept_graph slot) trips the value-equality assertion.
_VALID_CONCEPT_GRAPH_SHA = "a" * 64
_VALID_DART_CHUNKS_SHA = "b" * 64
_VALID_IMSCC_CHUNKS_SHA = "c" * 64


class _CapturingMCP:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


@pytest.fixture
def archive_tool(monkeypatch, tmp_path):
    """Return archive_to_libv2 coroutine, rooted at tmp_path for LibV2."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )

    mcp = _CapturingMCP()
    register_pipeline_tools(mcp)
    return mcp.tools["archive_to_libv2"]


# --------------------------------------------------------------------- #
# All three SHA kwargs accepted + persisted when well-formed
# --------------------------------------------------------------------- #


def test_archive_persists_all_three_well_formed_sha_kwargs(archive_tool):
    """Phase 8 ST 1 happy path — every well-formed SHA kwarg persists."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_PARITY_HAPPY",
        domain="test-domain",
        concept_graph_sha256=_VALID_CONCEPT_GRAPH_SHA,
        dart_chunks_sha256=_VALID_DART_CHUNKS_SHA,
        imscc_chunks_sha256=_VALID_IMSCC_CHUNKS_SHA,
    ))
    result = json.loads(result_str)
    assert "manifest_path" in result, f"archive_to_libv2 errored: {result}"

    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert manifest.get("concept_graph_sha256") == _VALID_CONCEPT_GRAPH_SHA, (
        "Phase 8 ST 1: well-formed concept_graph_sha256 kwarg must persist "
        f"into manifest. Got: {manifest.get('concept_graph_sha256')!r}"
    )
    assert manifest.get("dart_chunks_sha256") == _VALID_DART_CHUNKS_SHA, (
        "Phase 8 ST 1: well-formed dart_chunks_sha256 kwarg must persist "
        f"into manifest. Got: {manifest.get('dart_chunks_sha256')!r}"
    )
    assert manifest.get("imscc_chunks_sha256") == _VALID_IMSCC_CHUNKS_SHA, (
        "Phase 8 ST 1: well-formed imscc_chunks_sha256 kwarg must persist "
        f"into manifest. Got: {manifest.get('imscc_chunks_sha256')!r}"
    )


# --------------------------------------------------------------------- #
# Malformed SHAs silently dropped (mirrors registry variant fall-through)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("malformed_value, label", [
    ("not-a-hash", "non-hex string"),
    ("a" * 63, "63-char (one short)"),
    ("a" * 65, "65-char (one over)"),
    ("A" * 64, "uppercase hex (regex requires lowercase)"),
    ("g" * 64, "non-hex chars (g..z)"),
    ("", "empty string (falsy -> skipped pre-regex)"),
])
def test_archive_drops_malformed_concept_graph_sha(
    archive_tool, malformed_value, label,
):
    """Phase 8 ST 1: malformed concept_graph_sha256 silently dropped."""
    result_str = asyncio.run(archive_tool(
        course_name=f"TEST_MALFORMED_CG_{abs(hash(label))}",
        domain="test-domain",
        concept_graph_sha256=malformed_value,
    ))
    manifest = json.loads(
        Path(json.loads(result_str)["manifest_path"]).read_text()
    )
    assert "concept_graph_sha256" not in manifest, (
        f"Phase 8 ST 1: malformed concept_graph_sha256 ({label}) must be "
        f"silently dropped. Got persisted value: "
        f"{manifest.get('concept_graph_sha256')!r}"
    )


def test_archive_drops_malformed_dart_chunks_sha(archive_tool):
    """Phase 8 ST 1: malformed dart_chunks_sha256 silently dropped."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_MALFORMED_DART",
        domain="test-domain",
        dart_chunks_sha256="zzz_not_hex_zzz",
    ))
    manifest = json.loads(
        Path(json.loads(result_str)["manifest_path"]).read_text()
    )
    assert "dart_chunks_sha256" not in manifest, (
        "Phase 8 ST 1: malformed dart_chunks_sha256 must be silently "
        "dropped (mirrors registry variant's INVALID_* fall-through)."
    )


def test_archive_drops_malformed_imscc_chunks_sha(archive_tool):
    """Phase 8 ST 1: malformed imscc_chunks_sha256 silently dropped."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_MALFORMED_IMSCC",
        domain="test-domain",
        imscc_chunks_sha256="0123",  # too short
    ))
    manifest = json.loads(
        Path(json.loads(result_str)["manifest_path"]).read_text()
    )
    assert "imscc_chunks_sha256" not in manifest, (
        "Phase 8 ST 1: malformed imscc_chunks_sha256 must be silently "
        "dropped (mirrors registry variant's INVALID_* fall-through)."
    )


# --------------------------------------------------------------------- #
# Absent SHAs result in absent manifest fields (back-compat)
# --------------------------------------------------------------------- #


def test_archive_legacy_call_omits_all_three_sha_fields(archive_tool):
    """Phase 8 ST 1 back-compat: legacy MCP clients calling without the
    new kwargs see manifests without any of the three SHA fields.

    Mirrors the chunker_version back-compat test pattern at
    ``MCP/tests/test_archive_libv2_chunker_version.py::test_validator_still_accepts_manifest_without_chunker_version``.
    """
    result_str = asyncio.run(archive_tool(
        course_name="TEST_LEGACY_BACKCOMPAT",
        domain="test-domain",
    ))
    manifest = json.loads(
        Path(json.loads(result_str)["manifest_path"]).read_text()
    )

    # All three fields absent -- the legacy caller path is byte-identical
    # to pre-Phase-8 behaviour.
    assert "concept_graph_sha256" not in manifest, (
        "Phase 8 ST 1 back-compat: absent concept_graph_sha256 kwarg must "
        "leave the manifest field absent."
    )
    assert "dart_chunks_sha256" not in manifest, (
        "Phase 8 ST 1 back-compat: absent dart_chunks_sha256 kwarg must "
        "leave the manifest field absent."
    )
    assert "imscc_chunks_sha256" not in manifest, (
        "Phase 8 ST 1 back-compat: absent imscc_chunks_sha256 kwarg must "
        "leave the manifest field absent."
    )

    # Sanity: the pre-Phase-8 baseline fields are still emitted.
    for required_field in (
        "libv2_version",
        "chunker_version",
        "slug",
        "import_timestamp",
        "classification",
        "source_artifacts",
        "provenance",
        "features",
    ):
        assert required_field in manifest, (
            f"Phase 8 ST 1 must not regress baseline manifest field "
            f"{required_field!r}."
        )


def test_archive_partial_kwargs_persist_only_supplied_well_formed(archive_tool):
    """Phase 8 ST 1: caller supplying only a subset (e.g. concept graph
    only) sees only that field persisted -- the other two stay absent."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_PARTIAL_KWARGS",
        domain="test-domain",
        concept_graph_sha256=_VALID_CONCEPT_GRAPH_SHA,
    ))
    manifest = json.loads(
        Path(json.loads(result_str)["manifest_path"]).read_text()
    )
    assert manifest.get("concept_graph_sha256") == _VALID_CONCEPT_GRAPH_SHA
    assert "dart_chunks_sha256" not in manifest
    assert "imscc_chunks_sha256" not in manifest
