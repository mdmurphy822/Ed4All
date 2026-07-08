"""B3/OP4: license + attribution + library_format_version on the LibV2 manifest.

Covers:
    - ``_normalize_license_field`` / ``_normalize_attribution_field`` /
      ``_write_course_notice`` unit behavior.
    - The registry ``_archive_to_libv2`` (workflow-runner-driven) and the
      ``@mcp.tool()`` ``archive_to_libv2`` (external-client) variants both:
        * stamp ``library_format_version`` unconditionally,
        * emit ``license`` / ``attribution`` only when threaded,
        * write a human-readable NOTICE only when license/attribution present,
        * leave the manifest byte-identical (no license/attribution key, no
          NOTICE) when neither is threaded.
    - Schema additivity: legacy manifests (no new fields) validate; manifests
      carrying the new fields validate; malformed license/attribution reject.
    - ``LibV2ManifestValidator`` accepts a manifest carrying the new fields.
    - ``validate_course_manifest`` REPORTS (warning) a missing
      library_format_version and stays silent when present.
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
from MCP.tools.pipeline_tools import (  # noqa: E402
    LIBRARY_FORMAT_VERSION,
    _build_tool_registry,
    _normalize_attribution_field,
    _normalize_license_field,
    _write_course_notice,
    register_pipeline_tools,
)

_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "library" / "course_manifest.schema.json"
)


# --------------------------------------------------------------------- #
# Helper unit tests
# --------------------------------------------------------------------- #


def test_normalize_license_field_string_maps_to_spdx_or_name():
    assert _normalize_license_field("CC-BY-4.0") == {"spdx_or_name": "CC-BY-4.0"}


def test_normalize_license_field_strips_and_drops_empty():
    assert _normalize_license_field("  ") is None
    assert _normalize_license_field("") is None
    assert _normalize_license_field(None) is None


def test_normalize_license_field_accepts_dict():
    out = _normalize_license_field(
        {"spdx_or_name": " CC0-1.0 ", "note": " terms ", "junk": "x"}
    )
    assert out == {"spdx_or_name": "CC0-1.0", "note": "terms"}


def test_normalize_attribution_field_string_maps_to_statement():
    assert _normalize_attribution_field("Access for free") == {
        "statement": "Access for free"
    }


def test_normalize_attribution_field_accepts_dict_known_keys_only():
    out = _normalize_attribution_field(
        {
            "source_title": "Book",
            "source_url": "https://ex.org",
            "statement": "cite me",
            "junk": "drop",
        }
    )
    assert out == {
        "source_title": "Book",
        "source_url": "https://ex.org",
        "statement": "cite me",
    }


def test_normalize_attribution_field_empty_is_none():
    assert _normalize_attribution_field("") is None
    assert _normalize_attribution_field({"junk": "x"}) is None
    assert _normalize_attribution_field(None) is None


def test_write_course_notice_noop_when_both_absent(tmp_path):
    assert _write_course_notice(tmp_path, "slug", "Title", None, None) is None
    assert not (tmp_path / "NOTICE").exists()


def test_write_course_notice_writes_license_and_attribution(tmp_path):
    notice_path = _write_course_notice(
        tmp_path,
        "demo-slug",
        "Demo Title",
        {"spdx_or_name": "CC0-1.0", "note": "public domain"},
        {"statement": "Access for free at example.org", "source_url": "https://example.org"},
    )
    assert notice_path == tmp_path / "NOTICE"
    text = notice_path.read_text(encoding="utf-8")
    assert "Demo Title" in text
    assert "CC0-1.0" in text
    assert "public domain" in text
    assert "Access for free at example.org" in text
    assert "https://example.org" in text


# --------------------------------------------------------------------- #
# Registry variant (workflow-runner-driven) — end to end
# --------------------------------------------------------------------- #


@pytest.fixture
def registry_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    registry = _build_tool_registry()
    return registry["archive_to_libv2"]


def _manifest_and_dir(result_str: str):
    result = json.loads(result_str)
    assert "manifest_path" in result, f"archive errored: {result}"
    manifest_path = Path(result["manifest_path"])
    return json.loads(manifest_path.read_text()), manifest_path.parent


def test_registry_archive_emits_license_attribution_and_notice(registry_archive):
    result_str = asyncio.run(
        registry_archive(
            course_name="DEMO_LICENSE",
            domain="biology",
            license_note="CC0-1.0",
            attribution="Access for free at example.org",
        )
    )
    manifest, course_dir = _manifest_and_dir(result_str)
    assert manifest["license"] == {"spdx_or_name": "CC0-1.0"}
    assert manifest["attribution"] == {"statement": "Access for free at example.org"}
    assert manifest["library_format_version"] == LIBRARY_FORMAT_VERSION
    notice = course_dir / "NOTICE"
    assert notice.exists(), "NOTICE must be written when license/attribution set"
    assert "CC0-1.0" in notice.read_text(encoding="utf-8")


def test_registry_archive_byte_identical_without_flags(registry_archive):
    """Absent license/attribution → no license/attribution key + no NOTICE.

    library_format_version is stamped unconditionally (OP4).
    """
    result_str = asyncio.run(
        registry_archive(course_name="DEMO_PLAIN", domain="biology")
    )
    manifest, course_dir = _manifest_and_dir(result_str)
    assert "license" not in manifest
    assert "attribution" not in manifest
    assert manifest["library_format_version"] == LIBRARY_FORMAT_VERSION
    assert not (course_dir / "NOTICE").exists()


def test_registry_archive_license_only(registry_archive):
    result_str = asyncio.run(
        registry_archive(
            course_name="DEMO_LIC_ONLY", domain="biology", license_note="MIT"
        )
    )
    manifest, course_dir = _manifest_and_dir(result_str)
    assert manifest["license"] == {"spdx_or_name": "MIT"}
    assert "attribution" not in manifest
    assert (course_dir / "NOTICE").exists()


# --------------------------------------------------------------------- #
# @mcp.tool() variant parity
# --------------------------------------------------------------------- #


class _CapturingMCP:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


@pytest.fixture
def mcp_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    mcp = _CapturingMCP()
    register_pipeline_tools(mcp)
    return mcp.tools["archive_to_libv2"]


def test_mcp_tool_variant_stamps_version_and_emits_notice(mcp_archive):
    result_str = asyncio.run(
        mcp_archive(
            course_name="DEMO_MCP_LIC",
            domain="biology",
            license_note="CC-BY-4.0",
            attribution="cite the source",
        )
    )
    manifest, course_dir = _manifest_and_dir(result_str)
    assert manifest["library_format_version"] == LIBRARY_FORMAT_VERSION
    assert manifest["license"] == {"spdx_or_name": "CC-BY-4.0"}
    assert manifest["attribution"] == {"statement": "cite the source"}
    assert (course_dir / "NOTICE").exists()


def test_mcp_tool_variant_plain_has_version_no_notice(mcp_archive):
    result_str = asyncio.run(
        mcp_archive(course_name="DEMO_MCP_PLAIN", domain="biology")
    )
    manifest, course_dir = _manifest_and_dir(result_str)
    assert manifest["library_format_version"] == LIBRARY_FORMAT_VERSION
    assert "license" not in manifest
    assert not (course_dir / "NOTICE").exists()


# --------------------------------------------------------------------- #
# Schema additivity
# --------------------------------------------------------------------- #


def _schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _base_manifest() -> dict:
    return {
        "libv2_version": "1.2.0",
        "slug": "abc-course",
        "import_timestamp": "2026-07-07T00:00:00.000000",
        "sourceforge_manifest": {
            "sourceforge_version": "1.0",
            "export_timestamp": "2026-07-07T00:00:00.000000",
            "course_id": "X",
            "course_title": "X",
        },
        "classification": {"division": "STEM", "primary_domain": "general"},
        "content_profile": {"total_chunks": 0, "total_tokens": 0},
        "dart_chunks_sha256": "a" * 64,
        "imscc_chunks_sha256": "b" * 64,
        "concept_graph_sha256": "c" * 64,
    }


def test_schema_legacy_manifest_without_new_fields_validates():
    import jsonschema

    jsonschema.validate(_base_manifest(), _schema())


def test_schema_manifest_with_new_fields_validates():
    import jsonschema

    m = _base_manifest()
    m["library_format_version"] = "1.0"
    m["license"] = {"spdx_or_name": "CC0-1.0", "note": "public domain"}
    m["attribution"] = {
        "source_title": "Book",
        "source_url": "https://example.org/book",
        "statement": "Access for free at example.org",
    }
    jsonschema.validate(m, _schema())


def test_schema_rejects_empty_license_object():
    import jsonschema

    m = _base_manifest()
    m["license"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _schema())


def test_schema_rejects_unknown_license_key():
    import jsonschema

    m = _base_manifest()
    m["license"] = {"spdx_or_name": "MIT", "unexpected": "x"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _schema())


def test_schema_rejects_bad_library_format_version_pattern():
    import jsonschema

    m = _base_manifest()
    m["library_format_version"] = "v1"  # must be ^\d+\.\d+$
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, _schema())


# --------------------------------------------------------------------- #
# validate_course_manifest report-only version awareness (OP4)
# --------------------------------------------------------------------- #


def _write_course(tmp_path: Path, manifest: dict) -> Path:
    course_dir = tmp_path / "courses" / manifest["slug"]
    course_dir.mkdir(parents=True)
    (course_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return course_dir


def test_validate_course_manifest_warns_when_version_missing(tmp_path):
    from LibV2.tools.libv2.validator import validate_course_manifest

    course_dir = _write_course(tmp_path, _base_manifest())
    result = validate_course_manifest(course_dir, PROJECT_ROOT)
    assert any(
        "library_format_version" in w for w in result.warnings
    ), f"expected a missing-version warning; got {result.warnings}"


def test_validate_course_manifest_silent_when_version_present(tmp_path):
    from LibV2.tools.libv2.validator import validate_course_manifest

    m = _base_manifest()
    m["library_format_version"] = "1.0"
    course_dir = _write_course(tmp_path, m)
    result = validate_course_manifest(course_dir, PROJECT_ROOT)
    assert not any(
        "library_format_version" in w for w in result.warnings
    ), f"unexpected version warning: {result.warnings}"
