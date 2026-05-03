"""archive_to_libv2 emits ``chunker_version``.

Contract: when ``MCP/tools/pipeline_tools.py::archive_to_libv2`` writes
the LibV2 manifest, it MUST include a ``chunker_version`` top-level
field. Originally (Phase 7a) sourced from
``importlib.metadata.version("ed4all-chunker")`` returning a semver-
shaped string. Post-Phase-8 chunker re-merge: the field is sourced
from ``Trainforge.chunker.CHUNKER_SCHEMA_VERSION = "v4"`` — a
chunker-schema-contract version, decoupled from any Python package
release. The schema regex was widened to accept either shape:
``^(?:v\\d+|\\d+\\.\\d+\\.\\d+(?:[+-][A-Za-z0-9.+-]+)?)$``.

Surfaces tested:
    - ``_resolve_chunker_version`` returns the schema-contract version.
    - Pre-existing PackageNotFoundError fallback is now unreachable
      (the helper no longer reads from ``importlib.metadata``); the
      legacy sentinel ``"0.0.0+missing"`` would still validate against
      the widened regex if any pre-migration manifest carries it.
    - The emitted manifest carries ``chunker_version`` matching the
      widened regex.
    - The ``LibV2ManifestValidator`` accepts a manifest carrying the
      field (no SCHEMA_VIOLATION GateIssue).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Callable, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _resolve_chunker_version,
    register_pipeline_tools,
)

# Mirror the pattern shipped in
# schemas/library/course_manifest.schema.json::chunker_version so the
# test fails loudly if the regex drifts on either side. The widened
# alternation accepts both the post-Phase-8 chunker-schema-contract
# shape (``v4``) and the legacy package-version shape (``0.1.0``,
# ``0.0.0+missing``) so pre-migration manifests still validate.
_CHUNKER_VERSION_RE = re.compile(
    r"^(?:v\d+|\d+\.\d+\.\d+(?:[+-][A-Za-z0-9.+-]+)?)$"
)


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
# _resolve_chunker_version unit tests
# --------------------------------------------------------------------- #


def test_resolve_chunker_version_returns_schema_contract_version():
    """``_resolve_chunker_version`` returns ``Trainforge.chunker.CHUNKER_SCHEMA_VERSION``.

    Post-Phase-8 chunker re-merge: the helper no longer resolves a
    Python-package version via ``importlib.metadata.version``. It
    returns the in-tree ``Trainforge.chunker.CHUNKER_SCHEMA_VERSION``
    constant — the chunker-schema-contract version that's decoupled
    from any package release.
    """
    from Trainforge.chunker import CHUNKER_SCHEMA_VERSION

    version = _resolve_chunker_version()
    assert _CHUNKER_VERSION_RE.match(version), (
        f"chunker_version {version!r} does not match the widened schema regex"
    )
    assert version == CHUNKER_SCHEMA_VERSION, (
        "_resolve_chunker_version must return the in-tree chunker-schema "
        "constant, not a Python-package version lookup."
    )


def test_legacy_semver_sentinel_still_satisfies_widened_regex():
    """Pre-Phase-8 sentinel ``"0.0.0+missing"`` is still schema-valid.

    The chunker re-merge widened the regex (``^v\\d+|\\d+\\.\\d+\\.\\d+...$``)
    so the legacy sentinel emitted by pre-migration archives stays
    valid. Without this guard, pre-migration LibV2 archives would
    fail closed at the manifest gate after the regex change.
    """
    assert _CHUNKER_VERSION_RE.match("0.0.0+missing")
    assert _CHUNKER_VERSION_RE.match("0.1.0")  # legacy package version


# --------------------------------------------------------------------- #
# End-to-end archive_to_libv2: manifest carries chunker_version
# --------------------------------------------------------------------- #


def test_archive_manifest_includes_chunker_version(archive_tool):
    """Smallest end-to-end emit: manifest top-level has chunker_version."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_CHUNKER_VER",
        domain="test-domain",
    ))
    result = json.loads(result_str)
    assert "manifest_path" in result, f"archive_to_libv2 errored: {result}"
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert "chunker_version" in manifest, (
        "Phase 7a Subtask 8: archive_to_libv2 must emit chunker_version"
    )
    assert _CHUNKER_VERSION_RE.match(manifest["chunker_version"]), (
        f"chunker_version {manifest['chunker_version']!r} does not match "
        "the schema regex"
    )


def test_archive_manifest_chunker_version_alongside_libv2_version(archive_tool):
    """Sanity: chunker_version sits at top level, doesn't displace libv2_version."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_CHUNKER_VER_2",
        domain="test-domain",
    ))
    manifest = json.loads(
        Path(json.loads(result_str)["manifest_path"]).read_text()
    )
    assert "libv2_version" in manifest
    assert "chunker_version" in manifest
    # Both follow semver-ish patterns.
    assert _CHUNKER_VERSION_RE.match(manifest["libv2_version"])
    assert _CHUNKER_VERSION_RE.match(manifest["chunker_version"])


# --------------------------------------------------------------------- #
# LibV2ManifestValidator accepts the new field
# --------------------------------------------------------------------- #


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_minimal_archive(tmp_path: Path, *, with_chunker_version: bool) -> Path:
    """Build a well-formed LibV2 archive at tmp_path; return manifest path."""
    slug = "phase7-chunker-test"
    course_dir = tmp_path / "courses" / slug
    course_dir.mkdir(parents=True)

    for sub in ("corpus", "graph", "training_specs", "quality",
                "source/pdf", "source/html", "source/imscc", "pedagogy"):
        (course_dir / sub).mkdir(parents=True)
    (course_dir / "pedagogy" / "model.json").write_text("{}", encoding="utf-8")
    (course_dir / "graph" / "nodes.json").write_text("[]", encoding="utf-8")
    (course_dir / "course.json").write_text(
        json.dumps({"slug": slug, "learning_outcomes": []}),
        encoding="utf-8",
    )

    pdf_bytes = b"%PDF-1.4 phase7 chunker test bytes" * 10
    pdf_path = course_dir / "source" / "pdf" / "test.pdf"
    pdf_path.write_bytes(pdf_bytes)

    manifest = {
        "libv2_version": "1.2.0",
        "slug": slug,
        "import_timestamp": "2026-05-03T00:00:00.000000",
        "classification": {
            "division": "STEM",
            "primary_domain": "general",
            "subdomains": [],
        },
        "source_artifacts": {
            "pdf": [{
                "path": str(pdf_path),
                "checksum": _sha256_bytes(pdf_bytes),
                "size": len(pdf_bytes),
            }],
        },
        "provenance": {
            "source_type": "textbook_to_course_pipeline",
            "import_pipeline_version": "1.0.0",
        },
        "features": {
            "source_provenance": True,
            "evidence_source_provenance": False,
        },
    }
    if with_chunker_version:
        manifest["chunker_version"] = _resolve_chunker_version()

    manifest_path = course_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def test_validator_accepts_manifest_with_chunker_version(tmp_path):
    """LibV2ManifestValidator must not reject the new field.

    Phase 7a: chunker_version is OPTIONAL in the schema. The validator
    runs jsonschema.validate against course_manifest.schema.json — adding
    an optional field with no additionalProperties:false constraint is
    a no-op for required-key checks. This test guards against a future
    accidental tightening that would block the new field.
    """
    from lib.validators.libv2_manifest import LibV2ManifestValidator

    manifest_path = _build_minimal_archive(tmp_path, with_chunker_version=True)
    result = LibV2ManifestValidator().validate(
        {"manifest_path": str(manifest_path)}
    )

    schema_violations = [
        i for i in result.issues
        if i.code == "SCHEMA_VIOLATION" and i.severity == "critical"
    ]
    assert not schema_violations, (
        "Adding chunker_version must not introduce schema violations. "
        f"Got: {[i.message for i in schema_violations]}"
    )


def test_validator_still_accepts_manifest_without_chunker_version(tmp_path):
    """Phase 7a backward compat: legacy manifests without the field must still pass.

    The Phase 7c worker will promote chunker_version from optional to
    required — at that point this test should be flipped to the inverse
    assertion. Until then, omitting the field is legal (and required for
    every pre-Wave-7 manifest already on disk).
    """
    from lib.validators.libv2_manifest import LibV2ManifestValidator

    manifest_path = _build_minimal_archive(tmp_path, with_chunker_version=False)
    result = LibV2ManifestValidator().validate(
        {"manifest_path": str(manifest_path)}
    )

    schema_violations = [
        i for i in result.issues
        if i.code == "SCHEMA_VIOLATION" and i.severity == "critical"
    ]
    # No critical schema violations from a chunker_version-less manifest.
    chunker_related = [
        i for i in schema_violations if "chunker_version" in i.message
    ]
    assert not chunker_related, (
        "Phase 7a: chunker_version is OPTIONAL — omitting it must not "
        f"trigger a schema violation. Got: {[i.message for i in chunker_related]}"
    )
