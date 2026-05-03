"""Phase 8 ST 3 regression tests — `_resolve_libv2_root` resolution
chain + per-helper LibV2 root threading.

Pre-Phase-8: three Phase 6/7 helpers (`_run_concept_extraction`,
`_run_dart_chunking`, `_run_imscc_chunking`) used a hardcoded
``_PROJECT_ROOT / "LibV2" / "courses" / course_slug`` literal. This
made the helpers unportable for ops topologies that mount LibV2 at
a non-default location (Docker volume / NFS / ConfigMap).

Phase 8 ST 3 introduces ``_resolve_libv2_root(explicit)`` with the
resolution chain (high → low priority):
    1. Explicit ``libv2_root`` kwarg threaded by the workflow runner
       via ``inputs_from`` from ``workflow_params.libv2_root``.
    2. ``ED4ALL_LIBV2_ROOT`` env var.
    3. ``_PROJECT_ROOT / "LibV2"`` legacy default.

Test contract:
  * Resolution chain (3 tests): explicit > env > default.
  * Per-helper (3 tests): each writes to the resolved root when
    `libv2_root` is threaded.
  * Backward compat (1+ tests): no env / no kwarg falls through to
    `_PROJECT_ROOT / "LibV2"` default; legacy in-tree behaviour is
    preserved.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    _resolve_libv2_root,
)


# ---------------------------------------------------------------------------
# Resolution-chain tests (3) — exercise `_resolve_libv2_root` directly.
# ---------------------------------------------------------------------------


class TestResolveLibV2RootResolutionChain:
    """Phase 8 ST 3 — `_resolve_libv2_root` resolves explicit > env >
    default with the documented precedence."""

    def test_explicit_kwarg_wins_over_env_and_default(
        self, tmp_path, monkeypatch
    ):
        """Explicit kwarg is the highest-priority resolution leg —
        wins over both ED4ALL_LIBV2_ROOT and the in-tree default.
        """
        env_root = tmp_path / "env_libv2"
        explicit_root = tmp_path / "explicit_libv2"
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(env_root))
        # Also override the module-level _PROJECT_ROOT so the legacy
        # default points somewhere obviously distinct from explicit.
        fake_project = tmp_path / "fake_project"
        fake_project.mkdir()
        monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_project)

        resolved = _resolve_libv2_root(str(explicit_root))
        assert resolved == explicit_root, (
            f"Explicit kwarg should win; got {resolved!r}, expected "
            f"{explicit_root!r}."
        )

    def test_env_var_wins_when_no_explicit_kwarg(
        self, tmp_path, monkeypatch
    ):
        """ED4ALL_LIBV2_ROOT env var is the second-priority leg —
        wins over the in-tree default when no explicit kwarg is
        provided.
        """
        env_root = tmp_path / "env_libv2"
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(env_root))
        fake_project = tmp_path / "fake_project"
        fake_project.mkdir()
        monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_project)

        # Empty explicit -> falls through to env.
        resolved_empty = _resolve_libv2_root("")
        resolved_none = _resolve_libv2_root(None)
        assert resolved_empty == env_root, (
            f"Empty explicit should fall through to env; got "
            f"{resolved_empty!r}, expected {env_root!r}."
        )
        assert resolved_none == env_root, (
            f"None explicit should fall through to env; got "
            f"{resolved_none!r}, expected {env_root!r}."
        )

    def test_default_falls_through_when_no_explicit_no_env(
        self, tmp_path, monkeypatch
    ):
        """When neither explicit kwarg nor env var is set, resolution
        falls through to ``_PROJECT_ROOT / "LibV2"`` — preserves
        legacy in-tree behaviour for every existing run.
        """
        monkeypatch.delenv("ED4ALL_LIBV2_ROOT", raising=False)
        fake_project = tmp_path / "fake_project"
        fake_project.mkdir()
        monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_project)

        resolved = _resolve_libv2_root(None)
        assert resolved == fake_project / "LibV2", (
            f"Default should be _PROJECT_ROOT / 'LibV2'; got "
            f"{resolved!r}, expected {fake_project / 'LibV2'!r}."
        )


# ---------------------------------------------------------------------------
# Fixture helpers for per-helper tests.
# ---------------------------------------------------------------------------


def _write_synthesized(path: Path) -> None:
    """Emit a minimal DART ``*_synthesized.json`` fixture sidecar."""
    doc = {
        "campus_code": "phase8st3",
        "campus_name": "Phase 8 ST 3 LibV2 Root Threading",
        "sections": [
            {
                "section_id": "intro",
                "section_type": "overview",
                "section_title": "LibV2 Root Routing",
                "data": {
                    "paragraphs": [
                        "Phase 8 ST 3 threads libv2_root through three "
                        "Phase 6/7 helpers so ops topologies can mount "
                        "LibV2 anywhere on disk."
                    ]
                },
            },
        ],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _write_dart_html(path: Path) -> None:
    """Emit a minimal DART HTML fixture for the chunker."""
    path.write_text(
        '<!DOCTYPE html><html><body>'
        '<section><h1>Section A</h1>'
        '<p>Routing libv2 root through chunkers exercises the helper.</p>'
        '</section>'
        '</body></html>',
        encoding="utf-8",
    )


def _write_imscc_zip(path: Path) -> None:
    """Emit a minimal IMSCC zip fixture (one HTML page) for the
    IMSCC chunker.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "course/page1.html",
            '<!DOCTYPE html><html><body>'
            '<section><h1>Page 1</h1>'
            '<p>IMSCC chunker libv2 root routing test.</p>'
            '</section>'
            '</body></html>',
        )
    path.write_bytes(buf.getvalue())


@pytest.fixture
def hermetic_libv2(tmp_path, monkeypatch) -> Dict[str, Path]:
    """Hermetic fixture: redirect _PROJECT_ROOT to a fake tmp tree so
    helper writes never touch the real repo's LibV2/ tree, even if
    the libv2_root threading regresses.
    """
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)
    # Defensive: also clear the env var so resolution chain is
    # deterministic (each test sets its own value when needed).
    monkeypatch.delenv("ED4ALL_LIBV2_ROOT", raising=False)

    return {
        "fake_root": fake_root,
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# Per-helper tests (3) — each helper writes to the explicitly threaded
# libv2_root rather than the in-tree default.
# ---------------------------------------------------------------------------


class TestHelperLibV2RootThreading:
    """Phase 8 ST 3 — each of the three migrated helpers writes its
    course artifacts under the explicitly-threaded ``libv2_root``."""

    def test_run_concept_extraction_writes_to_threaded_libv2_root(
        self, hermetic_libv2
    ):
        """`_run_concept_extraction` writes the per-course concept
        graph under ``<libv2_root>/courses/<slug>/concept_graph/``
        when ``libv2_root`` is threaded.
        """
        fake_root = hermetic_libv2["fake_root"]
        custom_libv2 = hermetic_libv2["tmp_path"] / "custom_libv2"
        staging = hermetic_libv2["tmp_path"] / "staging"
        staging.mkdir()
        _write_synthesized(staging / "demo_synthesized.json")

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name="LIBV2_ROOT_TEST",
                staging_dir=str(staging),
                libv2_root=str(custom_libv2),
            )
        )
        payload = json.loads(result)
        assert payload.get("success") is True, (
            f"Helper should succeed; got payload={payload!r}."
        )

        expected_graph = (
            custom_libv2 / "courses" / "libv2-root-test" / "concept_graph"
            / "concept_graph_semantic.json"
        )
        assert expected_graph.exists(), (
            f"Concept graph should land under threaded libv2_root: "
            f"{expected_graph!r}; got payload['concept_graph_path']="
            f"{payload.get('concept_graph_path')!r}."
        )
        # Cross-check: the in-tree default location should NOT have
        # been touched (the hermetic fake_root in-tree LibV2 stays
        # absent).
        assert not (
            fake_root / "LibV2" / "courses" / "libv2-root-test"
        ).exists(), (
            "In-tree LibV2 dir should be untouched when libv2_root "
            "is threaded explicitly."
        )

    def test_run_dart_chunking_writes_to_threaded_libv2_root(
        self, hermetic_libv2
    ):
        """`_run_dart_chunking` writes the DART chunkset under
        ``<libv2_root>/courses/<slug>/dart_chunks/`` when
        ``libv2_root`` is threaded.
        """
        fake_root = hermetic_libv2["fake_root"]
        custom_libv2 = hermetic_libv2["tmp_path"] / "custom_libv2"
        staging = hermetic_libv2["tmp_path"] / "staging"
        staging.mkdir()
        _write_dart_html(staging / "demo_accessible.html")

        registry = _build_tool_registry()
        tool = registry["run_dart_chunking"]
        result = asyncio.run(
            tool(
                course_name="LIBV2_ROOT_TEST",
                staging_dir=str(staging),
                libv2_root=str(custom_libv2),
            )
        )
        payload = json.loads(result)
        assert payload.get("success") is True, (
            f"Helper should succeed; got payload={payload!r}."
        )

        expected_chunks = (
            custom_libv2 / "courses" / "libv2-root-test" / "dart_chunks"
            / "chunks.jsonl"
        )
        assert expected_chunks.exists(), (
            f"DART chunkset should land under threaded libv2_root: "
            f"{expected_chunks!r}; got payload['dart_chunks_path']="
            f"{payload.get('dart_chunks_path')!r}."
        )
        assert not (
            fake_root / "LibV2" / "courses" / "libv2-root-test"
        ).exists(), (
            "In-tree LibV2 dir should be untouched when libv2_root "
            "is threaded explicitly."
        )

    def test_run_imscc_chunking_writes_to_threaded_libv2_root(
        self, hermetic_libv2
    ):
        """`_run_imscc_chunking` writes the IMSCC chunkset under
        ``<libv2_root>/courses/<slug>/imscc_chunks/`` when
        ``libv2_root`` is threaded.
        """
        fake_root = hermetic_libv2["fake_root"]
        custom_libv2 = hermetic_libv2["tmp_path"] / "custom_libv2"
        imscc_path = hermetic_libv2["tmp_path"] / "demo.imscc"
        _write_imscc_zip(imscc_path)

        registry = _build_tool_registry()
        tool = registry["run_imscc_chunking"]
        result = asyncio.run(
            tool(
                course_name="LIBV2_ROOT_TEST",
                imscc_path=str(imscc_path),
                libv2_root=str(custom_libv2),
            )
        )
        payload = json.loads(result)
        assert payload.get("success") is True, (
            f"Helper should succeed; got payload={payload!r}."
        )

        expected_chunks = (
            custom_libv2 / "courses" / "libv2-root-test" / "imscc_chunks"
            / "chunks.jsonl"
        )
        assert expected_chunks.exists(), (
            f"IMSCC chunkset should land under threaded libv2_root: "
            f"{expected_chunks!r}; got payload['imscc_chunks_path']="
            f"{payload.get('imscc_chunks_path')!r}."
        )
        assert not (
            fake_root / "LibV2" / "courses" / "libv2-root-test"
        ).exists(), (
            "In-tree LibV2 dir should be untouched when libv2_root "
            "is threaded explicitly."
        )


# ---------------------------------------------------------------------------
# Backward-compat tests — no env / no kwarg falls through to the
# in-tree default. These exercise the env-var leg AND the default
# fallthrough leg of the resolution chain through a real helper.
# ---------------------------------------------------------------------------


class TestHelperBackwardCompat:
    """Phase 8 ST 3 — when no explicit kwarg is provided and no env
    var is set, helpers fall through to the legacy in-tree default
    (preserves byte-identical behaviour for every existing run)."""

    def test_concept_extraction_falls_through_to_in_tree_default(
        self, hermetic_libv2
    ):
        """No kwarg, no env var → writes to ``_PROJECT_ROOT / "LibV2"
        / courses / <slug> / concept_graph/``."""
        fake_root = hermetic_libv2["fake_root"]
        staging = hermetic_libv2["tmp_path"] / "staging"
        staging.mkdir()
        _write_synthesized(staging / "demo_synthesized.json")

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name="LIBV2_ROOT_TEST",
                staging_dir=str(staging),
                # libv2_root intentionally omitted to force fallback.
            )
        )
        payload = json.loads(result)
        assert payload.get("success") is True

        expected_graph = (
            fake_root / "LibV2" / "courses" / "libv2-root-test"
            / "concept_graph" / "concept_graph_semantic.json"
        )
        assert expected_graph.exists(), (
            f"Backward compat: concept graph should land under "
            f"in-tree _PROJECT_ROOT/LibV2 default when libv2_root is "
            f"unset; got payload['concept_graph_path']="
            f"{payload.get('concept_graph_path')!r}."
        )

    def test_dart_chunking_uses_env_var_when_no_kwarg(
        self, hermetic_libv2, monkeypatch
    ):
        """No kwarg, ED4ALL_LIBV2_ROOT set → writes to env-var root.
        Exercises the env-var leg of the resolution chain through a
        real helper dispatch.
        """
        env_libv2 = hermetic_libv2["tmp_path"] / "env_libv2"
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(env_libv2))
        staging = hermetic_libv2["tmp_path"] / "staging"
        staging.mkdir()
        _write_dart_html(staging / "demo_accessible.html")

        registry = _build_tool_registry()
        tool = registry["run_dart_chunking"]
        result = asyncio.run(
            tool(
                course_name="LIBV2_ROOT_TEST",
                staging_dir=str(staging),
                # libv2_root intentionally omitted; env var provides.
            )
        )
        payload = json.loads(result)
        assert payload.get("success") is True

        expected_chunks = (
            env_libv2 / "courses" / "libv2-root-test" / "dart_chunks"
            / "chunks.jsonl"
        )
        assert expected_chunks.exists(), (
            f"Env-var leg should route DART chunkset to "
            f"$ED4ALL_LIBV2_ROOT/courses/<slug>/dart_chunks/; got "
            f"payload['dart_chunks_path']="
            f"{payload.get('dart_chunks_path')!r}."
        )
