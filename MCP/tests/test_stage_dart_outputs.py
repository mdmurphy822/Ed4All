"""Wave 8 — stage_dart_outputs staging contract tests.

Verifies the Wave 8 additions to
``MCP/tools/pipeline_tools.py::stage_dart_outputs``:

* The ``*.quality.json`` sidecar (previously ignored) is copied alongside
  the rendered HTML and synthesized JSON.
* The staging manifest (``staging_manifest.json``) carries role-tagged
  entries under a new ``files`` array. Roles: ``content``,
  ``provenance_sidecar``, ``quality_sidecar``.
* Back-compat: the flat ``staged_files`` list remains in the manifest so
  older consumers keep working.

The tool is registered via a closure inside ``register_pipeline_tools``,
so we reach it by passing a minimal capture object and invoking the
captured coroutine directly.
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


class _CapturingMCP:
    """Minimal stand-in for FastMCP that captures registered tools by name."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return _decorator


@pytest.fixture
def stage_tool(monkeypatch, tmp_path):
    """Return the registered ``stage_dart_outputs`` coroutine.

    Also redirects the Courseforge staging root so tests can't stomp the
    real ``Courseforge/inputs/textbooks`` tree.
    """
    staging_root = tmp_path / "cf_inputs"
    staging_root.mkdir()
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", staging_root)

    mcp = _CapturingMCP()
    register_pipeline_tools(mcp)
    return mcp.tools["stage_dart_outputs"], staging_root


def _write_html(path: Path, body: str = "<html><body><p>Test</p></body></html>"):
    path.write_text(body, encoding="utf-8")


def _write_json(path: Path, payload: Dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestStagingBasics:
    def test_staging_copies_html(self, stage_tool, tmp_path):
        tool, staging_root = stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "science_of_learning.html"
        _write_html(html_file)

        result = asyncio.run(tool(
            run_id="WF-TEST-001",
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))
        payload = json.loads(result)
        assert payload["success"] is True
        staged = payload["staged_files"]
        assert any("science_of_learning.html" in s for s in staged)

    def test_staging_missing_file_is_reported(self, stage_tool, tmp_path):
        tool, _ = stage_tool
        result = asyncio.run(tool(
            run_id="WF-MISS-001",
            dart_html_paths=str(tmp_path / "does_not_exist.html"),
            course_name="TEST_101",
        ))
        payload = json.loads(result)
        assert payload.get("success") is False
        assert "No files staged" in payload.get("error", "")


class TestQualitySidecarStaging:
    """Wave 8: *.quality.json MUST be copied when present."""

    def test_quality_json_staged_when_present(self, stage_tool, tmp_path):
        tool, staging_root = stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "science_of_learning.html"
        _write_html(html_file)
        quality_file = dart_dir / "science_of_learning.quality.json"
        _write_json(quality_file, {
            "confidence_score": 0.87,
            "extraction_sources": ["pdftotext", "pdfplumber"],
        })

        result = asyncio.run(tool(
            run_id="WF-Q-001",
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))
        payload = json.loads(result)
        assert payload["success"] is True

        staged_names = {Path(s).name for s in payload["staged_files"]}
        assert "science_of_learning.html" in staged_names
        assert "science_of_learning.quality.json" in staged_names

        # And the file actually landed under the staging dir.
        staged_quality = staging_root / "WF-Q-001" / "science_of_learning.quality.json"
        assert staged_quality.exists()
        assert "confidence_score" in staged_quality.read_text()

    def test_quality_json_absent_is_not_an_error(self, stage_tool, tmp_path):
        tool, _ = stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "legacy.html"
        _write_html(html_file)

        result = asyncio.run(tool(
            run_id="WF-NOQ-001",
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))
        payload = json.loads(result)
        assert payload["success"] is True
        staged_names = {Path(s).name for s in payload["staged_files"]}
        assert "legacy.quality.json" not in staged_names


class TestManifestRoleTags:
    """Wave 8: staging_manifest.json carries role-tagged entries."""

    def test_manifest_has_role_tagged_files_array(self, stage_tool, tmp_path):
        tool, staging_root = stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "science_of_learning.html"
        _write_html(html_file)
        _write_json(dart_dir / "science_of_learning_synthesized.json",
                    {"campus_code": "TEST", "sections": []})
        _write_json(dart_dir / "science_of_learning.quality.json",
                    {"confidence_score": 0.9, "extraction_sources": ["pdftotext"]})

        run_id = "WF-MANIFEST-001"
        asyncio.run(tool(
            run_id=run_id,
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))

        manifest_path = staging_root / run_id / "staging_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        # New role-tagged files array must be present.
        assert "files" in manifest
        files = manifest["files"]
        by_role = {f["role"]: f["path"] for f in files}
        assert by_role.get("content") == "science_of_learning.html"
        assert by_role.get("provenance_sidecar") == "science_of_learning_synthesized.json"
        assert by_role.get("quality_sidecar") == "science_of_learning.quality.json"

        # Back-compat: flat staged_files list still present.
        assert "staged_files" in manifest
        assert isinstance(manifest["staged_files"], list)

    def test_manifest_roles_are_valid_enum(self, stage_tool, tmp_path):
        """Every role tag must be one of the Wave 8 canonical values."""
        tool, staging_root = stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "x.html"
        _write_html(html_file)
        _write_json(dart_dir / "x.quality.json", {"confidence_score": 0.5})

        run_id = "WF-ROLES-001"
        asyncio.run(tool(
            run_id=run_id,
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))
        manifest = json.loads(
            (staging_root / run_id / "staging_manifest.json").read_text()
        )
        valid_roles = {"content", "provenance_sidecar", "quality_sidecar"}
        for entry in manifest["files"]:
            assert entry["role"] in valid_roles
            assert "path" in entry
            # path is just a filename, not an absolute path (downstream
            # consumers resolve against the manifest dir).
            assert "/" not in entry["path"]

    def test_manifest_includes_synthesized_sidecar(self, stage_tool, tmp_path):
        """The *_synthesized.json path also gets tagged as provenance_sidecar."""
        tool, staging_root = stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "campus_info_synthesized.html"
        _write_html(html_file)
        # Matches the _synthesized.json pattern lookup.
        synth_file = dart_dir / "campus_info_synthesized.json"
        _write_json(synth_file, {"campus_code": "TEST", "sections": []})

        run_id = "WF-SYN-001"
        asyncio.run(tool(
            run_id=run_id,
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))
        manifest = json.loads(
            (staging_root / run_id / "staging_manifest.json").read_text()
        )
        roles = [f["role"] for f in manifest["files"]]
        assert "provenance_sidecar" in roles


class TestMultipleHtmlInputs:
    def test_multiple_inputs_preserve_all_quality_sidecars(self, stage_tool, tmp_path):
        tool, staging_root = stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_a = dart_dir / "a.html"
        html_b = dart_dir / "b.html"
        _write_html(html_a)
        _write_html(html_b)
        _write_json(dart_dir / "a.quality.json", {"confidence_score": 0.9})
        _write_json(dart_dir / "b.quality.json", {"confidence_score": 0.7})

        run_id = "WF-MULTI-001"
        asyncio.run(tool(
            run_id=run_id,
            dart_html_paths=f"{html_a},{html_b}",
            course_name="TEST_101",
        ))
        manifest = json.loads(
            (staging_root / run_id / "staging_manifest.json").read_text()
        )
        quality = [f for f in manifest["files"] if f["role"] == "quality_sidecar"]
        assert len(quality) == 2
        paths = {f["path"] for f in quality}
        assert paths == {"a.quality.json", "b.quality.json"}


class TestRegistryVariantParity:
    """MCP audit Q4: the runtime registry variant of stage_dart_outputs
    must carry Wave 8 role-tagging parity with the @mcp.tool() variant.

    Prior state: the registry wrapper was a stripped-down copy that
    skipped .quality.json + role tags entirely. Under pipeline dispatch
    (TaskExecutor → registry), Wave 8 metadata was silently dropped. The
    wrapper now mirrors the MCP variant's behavior.
    """

    @pytest.fixture
    def registry_stage_tool(self, monkeypatch, tmp_path):
        from MCP.tools.pipeline_tools import _build_tool_registry
        staging_root = tmp_path / "cf_inputs_registry"
        staging_root.mkdir()
        monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", staging_root)
        registry = _build_tool_registry()
        return registry["stage_dart_outputs"], staging_root

    def test_registry_variant_stages_quality_sidecar(
        self, registry_stage_tool, tmp_path,
    ):
        tool, staging_root = registry_stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "example.html"
        _write_html(html_file)
        _write_json(dart_dir / "example.quality.json", {
            "confidence_score": 0.85,
            "extraction_sources": ["pdftotext", "pdfplumber"],
        })

        run_id = "WF-REG-Q-001"
        result = asyncio.run(tool(
            run_id=run_id,
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))
        payload = json.loads(result)
        assert payload["success"] is True

        # Wave 8: quality sidecar must land in the staged files + manifest.
        staged_names = {Path(s).name for s in payload["staged_files"]}
        assert "example.quality.json" in staged_names, (
            "Registry variant dropped the .quality.json sidecar — "
            "Wave 8 parity regression."
        )
        manifest = json.loads(
            (staging_root / run_id / "staging_manifest.json").read_text()
        )
        assert "files" in manifest, "Registry manifest missing role-tagged 'files'"
        roles = {f["role"] for f in manifest["files"]}
        assert "quality_sidecar" in roles

    def test_registry_variant_emits_role_tagged_files(
        self, registry_stage_tool, tmp_path,
    ):
        tool, staging_root = registry_stage_tool
        dart_dir = tmp_path / "dart_out"
        dart_dir.mkdir()
        html_file = dart_dir / "course_info.html"
        _write_html(html_file)
        _write_json(dart_dir / "course_info_synthesized.json", {
            "campus_code": "TEST", "sections": [],
        })
        _write_json(dart_dir / "course_info.quality.json", {
            "confidence_score": 0.9,
        })

        run_id = "WF-REG-ROLES-001"
        asyncio.run(tool(
            run_id=run_id,
            dart_html_paths=str(html_file),
            course_name="TEST_101",
        ))

        manifest = json.loads(
            (staging_root / run_id / "staging_manifest.json").read_text()
        )
        roles_by_path = {f["path"]: f["role"] for f in manifest["files"]}
        assert roles_by_path.get("course_info.html") == "content"
        assert (
            roles_by_path.get("course_info_synthesized.json")
            == "provenance_sidecar"
        )
        assert (
            roles_by_path.get("course_info.quality.json")
            == "quality_sidecar"
        )

    def test_registry_variant_reports_missing_inputs(
        self, registry_stage_tool, tmp_path,
    ):
        tool, _ = registry_stage_tool
        result = asyncio.run(tool(
            run_id="WF-REG-MISS-001",
            dart_html_paths=str(tmp_path / "missing.html"),
            course_name="TEST_101",
        ))
        payload = json.loads(result)
        assert payload.get("success") is False
        assert "No files staged" in payload.get("error", "")
