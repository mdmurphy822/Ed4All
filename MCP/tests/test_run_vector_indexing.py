"""WS2 E4 — run_vector_indexing registry tool + rag-indexer mapping flip.

Covers:
- ``run_vector_indexing`` happy path with the deterministic ``fake`` provider:
  builds ``vector_index/`` and returns a success envelope with the manifest
  path + model fingerprint + chunk count.
- Fail-closed: when the embedding backend is unavailable
  (``EmbeddingBackendUnavailable``) the tool returns ``success: false`` with a
  typed error — NO file-counting / lexical fallback.
- Fail-closed: a missing course directory returns ``success: false``.
- Envelope shape (required keys present on success).
- Mapping flip: ``AGENT_TOOL_MAPPING["rag-indexer"] == "run_vector_indexing"``
  and ``analyze_imscc_content`` is still reachable for the agents that
  legitimately use it.
- TOOL_SCHEMAS parity: ``run_vector_indexing`` has a schema entry.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MCP.core.executor import AGENT_TOOL_MAPPING  # noqa: E402
from MCP.core.tool_schemas import TOOL_SCHEMAS  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"
)


def _materialize_course(libv2_root: Path, slug: str) -> Path:
    cdir = libv2_root / "courses" / slug
    (cdir / "dart_chunks").mkdir(parents=True)
    import shutil

    shutil.copy(
        _FIXTURE / "dart_chunks" / "chunks.jsonl",
        cdir / "dart_chunks" / "chunks.jsonl",
    )
    return cdir


@pytest.fixture()
def registry():
    return _build_tool_registry()


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    """Allow fake-provider indexes to be built/read in-process."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")


@pytest.fixture(autouse=True)
def _require_fixture():
    if not (_FIXTURE / "dart_chunks" / "chunks.jsonl").exists():
        pytest.skip("mini_course fixture not present")


class TestMappingFlip:
    def test_rag_indexer_maps_to_run_vector_indexing(self):
        assert AGENT_TOOL_MAPPING["rag-indexer"] == "run_vector_indexing", (
            "WS2 — the rag-indexer file-counting masquerade is retired; it "
            "must map to run_vector_indexing."
        )

    def test_analyze_imscc_content_still_reachable(self):
        """The flip must not strand analyze_imscc_content for agents that
        legitimately use it (assessment-extractor, content-analyzer)."""
        users = {
            agent
            for agent, tool in AGENT_TOOL_MAPPING.items()
            if tool == "analyze_imscc_content"
        }
        assert "assessment-extractor" in users
        assert "content-analyzer" in users
        # And it is still a registered tool.
        registry = _build_tool_registry()
        assert "analyze_imscc_content" in registry

    def test_run_vector_indexing_registered(self, registry):
        assert "run_vector_indexing" in registry

    def test_run_vector_indexing_has_schema(self):
        """Parity: every registered tool needs a TOOL_SCHEMAS entry or the
        param-mapper raises Unknown tool before dispatch."""
        assert "run_vector_indexing" in TOOL_SCHEMAS


class TestRunVectorIndexingHappyPath:
    def test_builds_index_and_returns_envelope(self, registry, tmp_path):
        # The vector-index path imports lib.embedding.providers, which pulls
        # numpy from the optional [embedding] extra. Absent on a slim CI
        # install → skip rather than fail with dependency_missing.
        pytest.importorskip("numpy")
        slug = "mini-retrieval-101"
        _materialize_course(tmp_path, slug)
        tool = registry["run_vector_indexing"]

        raw = asyncio.run(
            tool(course_name=slug, libv2_root=str(tmp_path))
        )
        env = json.loads(raw)
        assert env["success"] is True, env
        # Envelope shape.
        for key in (
            "manifest_path",
            "embeddings_path",
            "id_map_path",
            "vector_index_dir",
            "model_fingerprint",
            "embedding_model_id",
            "embedding_provider",
            "embedding_dim",
            "chunks_count",
            "chunkset_kind",
            "source_chunks_sha256",
            "course_slug",
        ):
            assert key in env, f"missing envelope key {key}"

        assert env["embedding_provider"] == "fake"
        assert env["chunkset_kind"] == "dart"
        assert env["chunks_count"] == 7
        assert env["course_slug"] == slug

        # The index artifacts exist on disk.
        index_dir = tmp_path / "courses" / slug / "vector_index"
        assert (index_dir / "embeddings.npy").exists()
        assert (index_dir / "id_map.json").exists()
        assert (index_dir / "manifest.json").exists()


class TestRunVectorIndexingFailClosed:
    def test_missing_course_fails_closed(self, registry, tmp_path):
        # Tool imports lib.embedding.providers (numpy) before the
        # course-exists check, so a numpy-less env returns dependency_missing
        # rather than course_missing — skip when the extra is absent.
        pytest.importorskip("numpy")
        tool = registry["run_vector_indexing"]
        raw = asyncio.run(
            tool(course_name="no-such-course", libv2_root=str(tmp_path))
        )
        env = json.loads(raw)
        assert env["success"] is False
        assert env["error_type"] == "course_missing"

    def test_backend_unavailable_fails_closed_no_fallback(
        self, registry, tmp_path, monkeypatch
    ):
        """When the embedding backend is unavailable the phase FAILS — there
        is NO file-counting / lexical fallback (anti-silent-degradation)."""
        # Needs numpy to reach the backend-unavailable arm (the lazy
        # lib.embedding.providers import otherwise short-circuits to
        # dependency_missing on a numpy-less install).
        pytest.importorskip("numpy")
        slug = "mini-retrieval-101"
        _materialize_course(tmp_path, slug)

        from lib.embedding import providers as _providers

        def _boom(*args, **kwargs):
            raise _providers.EmbeddingBackendUnavailable(
                "weights not cached (test)"
            )

        # Patch the symbol the tool imports lazily.
        monkeypatch.setattr(_providers, "build_embedding_client", _boom)

        tool = registry["run_vector_indexing"]
        raw = asyncio.run(
            tool(course_name=slug, libv2_root=str(tmp_path))
        )
        env = json.loads(raw)
        assert env["success"] is False
        assert env["error_type"] == "embedding_backend_unavailable"
        # Critically: NO vector_index/ was written (no partial/fake index).
        assert not (
            tmp_path / "courses" / slug / "vector_index"
        ).exists()
