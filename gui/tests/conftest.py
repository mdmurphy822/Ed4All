"""Shared fixtures for the Ed4All control-plane GUI test suite.

Every fixture isolates state into a ``tmp_path``: ``ED4ALL_STATE_RUNS_DIR`` is
redirected so ``gui.shared_state`` writes a throwaway ``state/gui/`` sibling, and
``ED4ALL_LIBV2_ROOT`` is redirected so course / retrieval reads/writes never
touch the real ``LibV2/``. All state-mutating fixtures monkeypatch these env
vars; nothing here makes a network call or loads a heavy ML dep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``state/`` into ``tmp_path`` via ``ED4ALL_STATE_RUNS_DIR``.

    ``shared_state._state_root`` resolves the GUI store as a sibling of the
    ``runs`` dir named by ``ED4ALL_STATE_RUNS_DIR``, so pointing it at
    ``tmp_path/state/runs`` lands the GUI store at ``tmp_path/state/gui/``.
    Returns the ``state/`` root.
    """
    state_root = tmp_path / "state"
    runs = state_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    return state_root


@pytest.fixture
def libv2_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the LibV2 root into ``tmp_path`` via ``ED4ALL_LIBV2_ROOT``."""
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(root))
    return root


@pytest.fixture
def sample_settings_doc() -> Dict[str, Any]:
    """A representative settings doc carrying a secret + routing + flags."""
    return {
        "version": 1,
        "env": {
            "ANTHROPIC_API_KEY": "sk-ant-test-secret-value",
            "LLM_MODE": "api",
        },
        "model_routing": {
            "global": {"mode": "api", "provider": "anthropic", "model": None},
            "courseforge_outline": {
                "provider": "local",
                "model": "qwen2.5:7b-instruct-q4_K_M",
            },
            "courseforge_rewrite": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            },
        },
        "retrieval": {"top_k": 5, "min_grounding_cosine": 0.45, "require_embeddings": True},
        "flags": {"COURSEFORGE_TWO_PASS": True, "DART_LLM_CLASSIFICATION": False},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def courseforge_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Build a tiny Courseforge export fixture and point ``COURSEFORGE_PATH`` at it.

    Patches ``gui.services.course_service.COURSEFORGE_PATH`` to a temp dir
    containing one ``PROJ-*`` export with ``project_config.json`` +
    ``01_learning_objectives/synthesized_objectives.json``. Returns metadata
    (``project_id``, ``course_name``, ``export_dir``, ``objectives_path``).
    """
    import gui.services.course_service as cs

    cf_root = tmp_path / "Courseforge"
    exports = cf_root / "exports"
    project_id = "PROJ-PHYS_101-20260531-abcd1234"
    course_name = "PHYS_101"
    export_dir = exports / project_id
    _write_json(
        export_dir / "project_config.json",
        {
            "project_id": project_id,
            "course_name": course_name,
            "duration_weeks": 10,
            "status": "draft",
        },
    )
    objectives_path = export_dir / "01_learning_objectives" / "synthesized_objectives.json"
    _write_json(
        objectives_path,
        {
            "course_name": course_name,
            "mint_method": "course_outliner_synthesis",
            "duration_weeks": 10,
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Analyze kinematics."},
            ],
            "chapter_objectives": [
                {"id": "CO-01", "statement": "Compute velocity."},
                {"id": "CO-02", "statement": "Compute acceleration."},
            ],
        },
    )
    monkeypatch.setattr(cs, "COURSEFORGE_PATH", str(cf_root))
    return {
        "project_id": project_id,
        "course_name": course_name,
        "export_dir": export_dir,
        "objectives_path": objectives_path,
        "cf_root": cf_root,
    }


@pytest.fixture
def libv2_course(libv2_root: Path) -> Dict[str, Any]:
    """Build a tiny LibV2 course (manifest.json + chunks.jsonl) under the temp root.

    Uses the ``ED4ALL_LIBV2_ROOT`` redirect from the ``libv2_root`` fixture, so
    the real ``LibV2/`` chunk corpus is never touched. Returns
    ``{slug, course_dir, chunks}``.

    The LibV2 retriever lives in the ``tools.libv2`` namespace which is only
    importable with the real ``LibV2/`` source tree on ``sys.path`` (it ships no
    ``__init__.py`` — ``retrieval_service._libv2_root_on_path`` adds the
    *content* root, but the temp content root has no ``tools/`` package). We put
    the real ``LibV2/`` source dir on ``sys.path`` here so ``tools.libv2.*``
    resolves, exactly as production does, while ``retrieve_chunks(repo_root=...)``
    still reads chunks from the temp course root.
    """
    import sys

    real_libv2 = Path(__file__).resolve().parents[2] / "LibV2"
    if real_libv2.is_dir() and str(real_libv2) not in sys.path:
        sys.path.insert(0, str(real_libv2))

    # The top-level ``tools`` name is claimed by BOTH ``LibV2/tools`` (namespace,
    # no __init__) and ``MCP/tools`` (regular package). If another test module
    # imported ``tools.gui_tools`` first, ``sys.modules['tools']`` is cached as
    # MCP's regular package, which has no ``libv2`` submodule and shadows the
    # LibV2 retriever. Evict any cached top-level ``tools`` binding that can't
    # resolve ``tools.libv2`` so it re-resolves against ``LibV2/`` on sys.path.
    # (This is purely a shared-interpreter test artifact — the real ``ed4all
    # gui`` process imports ``MCP.tools.*``, never the bare ``tools`` name.)
    cached = sys.modules.get("tools")
    if cached is not None:
        cached_file = getattr(cached, "__file__", "") or ""
        if "LibV2" not in cached_file:
            for name in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
                del sys.modules[name]

    slug = "phys-101-demo"
    course_dir = libv2_root / "courses" / slug
    (course_dir / "imscc_chunks").mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "id": "chunk_0001",
            "chunk_type": "concept",
            "text": (
                "Velocity is the rate of change of position with respect to "
                "time, a vector quantity with both magnitude and direction."
            ),
            "concept_tags": ["velocity", "kinematics"],
            "learning_outcome_refs": ["CO-01"],
            "bloom_level": "understand",
            "tokens_estimate": 24,
        },
        {
            "id": "chunk_0002",
            "chunk_type": "example",
            "text": (
                "Acceleration is the rate of change of velocity. A car speeding "
                "up from rest experiences positive acceleration."
            ),
            "concept_tags": ["acceleration", "kinematics"],
            "learning_outcome_refs": ["CO-02"],
            "bloom_level": "apply",
            "tokens_estimate": 22,
        },
        {
            "id": "chunk_0003",
            "chunk_type": "concept",
            "text": (
                "Photosynthesis converts light energy into chemical energy in "
                "plant chloroplasts — unrelated to mechanics."
            ),
            "concept_tags": ["photosynthesis"],
            "learning_outcome_refs": [],
            "bloom_level": "remember",
            "tokens_estimate": 18,
        },
    ]
    chunks_path = course_dir / "imscc_chunks" / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )

    _write_json(
        course_dir / "manifest.json",
        {
            "course_id": slug,
            "classification": {
                "division": "STEM",
                "primary_domain": "physics",
                "topics": ["mechanics"],
                "subdomains": ["kinematics"],
            },
            "content_profile": {"total_chunks": len(chunks)},
        },
    )
    return {"slug": slug, "course_dir": course_dir, "chunks": chunks}
