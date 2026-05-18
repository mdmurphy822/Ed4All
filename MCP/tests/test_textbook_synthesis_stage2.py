"""Stage-2 integration tests — three-stage textbook synthesis, Wave D.

Plan reference: ``plans/textbook-llm-synthesis-3stage-2026-05.md`` §5 +
§6 + test-plan item 5.

Covers the ``TEXTBOOK_SYNTHESIS_PROVIDER`` Stage-2 branch folded into
``MCP/tools/pipeline_tools.py::_plan_course_structure`` (Wave D):

* **flag unset** → deterministic path runs; ``synthesized_objectives.json``
  is byte-stable against a flag-unset baseline (no Stage-2 keys leak).
* **flag set** (stub provider) → per-chapter COs minted globally
  sequential (``CO-01..CO-NN``, NOT per-chapter), the Stage-1 draft
  TO-NN reconciled into the final ``terminal_objectives``, and each CO
  carries the optional ``sub_objectives[]`` field.
* **per-chapter failure isolation** (§5.4) → one chapter's call failing
  degrades into a recorded failure; the phase still succeeds because
  >= 1 chapter produced objectives.
* **all-chapters-fail** (§5.4) → falls back to the deterministic
  ``synthesize_objectives_from_topics``; reconciliation is skipped.

No real LLM calls — the provider is a stub.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402
import Courseforge.generators._textbook_synthesis_provider as _tsp_mod  # noqa: E402,E501
from Courseforge.generators._textbook_synthesis_provider import (  # noqa: E402,E501
    TextbookSynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Stub provider — synthesizes COs straight out of chapter_text. No LLM
# dispatch, no network. Mirrors the TextbookSynthesisProvider public
# surface the Stage-2 handler block consumes.
# ---------------------------------------------------------------------------


class _StubProvider:
    """Drop-in stub for ``TextbookSynthesisProvider`` (Stage 2 only).

    ``synthesize_chapter_objectives`` emits two COs per chapter (the
    statements derived from the chapter id) plus a ``sub_objectives[]``
    list. Chapters in ``fail_chapter_ids`` raise
    ``TextbookSynthesisProviderError`` to exercise the §5.4 per-chapter
    failure-isolation path. ``reconcile_terminal_objectives`` returns
    one reconciled TO per draft TO (or one synthetic TO when there are
    no drafts) so the test can assert the reconciliation step ran.
    """

    def __init__(
        self,
        *,
        fail_chapter_ids: Optional[List[str]] = None,
        cos_per_chapter: int = 2,
        **_kwargs: Any,
    ) -> None:
        self._provider = "stub"
        self._model = "stub-model-v1"
        self._fail = set(fail_chapter_ids or [])
        self._cos_per_chapter = cos_per_chapter
        self.reconcile_calls = 0

    def synthesize_chapter_objectives(
        self,
        chapter: Dict[str, Any],
        *,
        course_name: str,
        draft_terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        chapter_id = str(chapter.get("id") or "")
        if chapter_id in self._fail:
            raise TextbookSynthesisProviderError(
                f"stub forced failure on chapter {chapter_id!r}",
                code="chapter_objectives_exhausted",
            )
        objectives: List[Dict[str, Any]] = []
        for n in range(1, self._cos_per_chapter + 1):
            objectives.append({
                "statement": (
                    f"Apply the concepts of {chapter_id} (objective {n})."
                ),
                "bloom_level": "apply",
                "bloom_verb": "apply",
                "abcd": {
                    "audience": "the student",
                    "behavior": {
                        "verb": "apply",
                        "action_object": f"{chapter_id} concepts",
                    },
                    "condition": "given a worked example",
                    "degree": "with 80% accuracy",
                },
                "source_refs": [{"ref": chapter_id, "chunk_ids": []}],
                "sub_objectives": [
                    f"{chapter_id} sub-objective {n}.1",
                    f"{chapter_id} sub-objective {n}.2",
                ],
                "chapter_id": chapter_id,
            })
        return {"chapter_id": chapter_id, "chapter_objectives": objectives}

    def reconcile_terminal_objectives(
        self,
        draft_terminal_objectives: List[Dict[str, Any]],
        chapter_objectives: List[Dict[str, Any]],
        *,
        course_name: str,
    ) -> Dict[str, Any]:
        self.reconcile_calls += 1
        drafts = list(draft_terminal_objectives or [])
        terminals: List[Dict[str, Any]] = []
        sources = drafts or [{"statement": "Synthesized terminal goal."}]
        for idx, draft in enumerate(sources, start=1):
            stmt = str(draft.get("statement") or "").strip() or (
                f"Reconciled terminal objective {idx}."
            )
            terminals.append({
                "id": f"TO-{idx:02d}",
                "statement": f"[reconciled] {stmt}",
                "bloom_level": "analyze",
                "bloom_verb": "analyze",
                "source_refs": [],
            })
        return {"terminal_objectives": terminals}

    @staticmethod
    def batch_chapters(
        chapters: List[Any], batch_size: int = 10,
    ) -> List[List[Any]]:
        size = max(1, int(batch_size))
        return [
            list(chapters[i:i + size])
            for i in range(0, len(chapters), size)
        ]


class _AllFailProvider(_StubProvider):
    """Stub whose every per-chapter call raises (§5.4 all-fail path)."""

    def synthesize_chapter_objectives(
        self,
        chapter: Dict[str, Any],
        *,
        course_name: str,
        draft_terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        raise TextbookSynthesisProviderError(
            f"stub forced failure on chapter "
            f"{chapter.get('id')!r}",
            code="chapter_objectives_exhausted",
        )

    def reconcile_terminal_objectives(self, *args, **kwargs):  # noqa: ANN002
        # Reconciliation must NOT run when Stage 2 fully degraded.
        self.reconcile_calls += 1
        raise AssertionError(
            "reconcile_terminal_objectives must not be called when all "
            "chapters fail (plan §5.4 — reconciliation skipped)"
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _make_project(
    fake_root: Path,
    *,
    course_name: str,
    chapters: List[Dict[str, Any]],
    draft_terminal_objectives: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Scaffold a Courseforge project export with a textbook_structure."""
    project_id = f"PROJ-{course_name}-20260515"
    project_path = fake_root / "Courseforge" / "exports" / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True)

    (project_path / "project_config.json").write_text(
        json.dumps({
            "course_name": course_name,
            "project_id": project_id,
            "duration_weeks": 8,
        }),
        encoding="utf-8",
    )

    structure: Dict[str, Any] = {"chapters": chapters, "duration_weeks": 8}
    if draft_terminal_objectives is not None:
        structure["draft_terminal_objectives"] = draft_terminal_objectives
        structure["structure_enrichment"] = {
            "enriched": True, "provider": "stub", "model": "stub",
            "call_mode": "single", "calls": 1, "schema_version": "v1",
        }
    (project_path / "01_learning_objectives" / "textbook_structure.json"
     ).write_text(json.dumps(structure), encoding="utf-8")
    return project_path


def _chapters(n: int) -> List[Dict[str, Any]]:
    """Build N chapter dicts each carrying non-empty chapter_text."""
    return [
        {
            "id": f"ch{i}",
            "title": f"Chapter {i}",
            "sections": [],
            "chapter_text": (
                f"Chapter {i} prose. It teaches the foundational "
                f"material covered in chapter {i} of the textbook, "
                f"spanning several paragraphs of subject-matter content."
            ),
        }
        for i in range(1, n + 1)
    ]


def _draft_tos() -> List[Dict[str, Any]]:
    return [
        {
            "id": "TO-01",
            "statement": "Draft terminal goal one.",
            "bloom_level": "understand",
            "bloom_verb": "understand",
            "source_refs": [],
            "draft": True,
        },
        {
            "id": "TO-02",
            "statement": "Draft terminal goal two.",
            "bloom_level": "apply",
            "bloom_verb": "apply",
            "source_refs": [],
            "draft": True,
        },
    ]


def _run_plan(
    fake_root: Path,
    project_path: Path,
    *,
    course_name: str,
    provider_factory: Optional[Any],
    provider_env: Optional[str],
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Any]:
    """Invoke ``_plan_course_structure`` and return the parsed envelope
    plus the on-disk synthesized_objectives.json.
    """
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", fake_root)
    # No COURSEPLANNER seam in these tests.
    monkeypatch.delenv("COURSEPLANNER_PROVIDER", raising=False)
    if provider_env is None:
        monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", provider_env)
    if provider_factory is not None:
        monkeypatch.setattr(
            _tsp_mod, "TextbookSynthesisProvider", provider_factory,
        )

    registry = _build_tool_registry()
    tool = registry["plan_course_structure"]
    raw = asyncio.run(tool(
        project_id=project_path.name,
        course_name=course_name,
        duration_weeks_explicit=True,
        duration_weeks=8,
    ))
    payload = json.loads(raw)
    out_path = project_path / "01_learning_objectives" / (
        "synthesized_objectives.json"
    )
    payload["_synthesized"] = json.loads(
        out_path.read_text(encoding="utf-8")
    )
    return payload


# ---------------------------------------------------------------------------
# Test 1 — flag unset → deterministic path, byte-stable
# ---------------------------------------------------------------------------


def test_flag_unset_deterministic_path_byte_stable(tmp_path, monkeypatch):
    """Default-off: the deterministic synthesizer runs; the Stage-2
    provider is never constructed and no Stage-2 mint_method leaks."""
    constructed: List[bool] = []

    class _Tripwire(_StubProvider):
        def __init__(self, *args, **kwargs):  # noqa: ANN002
            constructed.append(True)
            super().__init__(*args, **kwargs)

    fake_root = tmp_path / "root"
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    project = _make_project(
        fake_root, course_name="PHYS_101", chapters=_chapters(3),
        draft_terminal_objectives=_draft_tos(),
    )
    monkeypatch.setattr(
        _tsp_mod, "TextbookSynthesisProvider", _Tripwire,
    )
    payload = _run_plan(
        fake_root, project, course_name="PHYS_101",
        provider_factory=None, provider_env=None, monkeypatch=monkeypatch,
    )
    assert payload["success"] is True
    assert constructed == [], (
        "Stage-2 provider must NOT be constructed when "
        "TEXTBOOK_SYNTHESIS_PROVIDER is unset"
    )
    # Deterministic mint_method, NOT textbook_synthesis:*.
    assert payload["mint_method"] == "synthesize_objectives_from_topics"
    assert not payload["_synthesized"]["mint_method"].startswith(
        "textbook_synthesis:"
    )


def test_flag_unset_byte_stable_across_runs(tmp_path, monkeypatch):
    """Default-off: two flag-unset runs emit identical objective text
    (the deterministic path is unchanged by Wave D)."""
    def _one_run() -> str:
        fake_root = tmp_path / f"root-{_one_run.counter}"
        _one_run.counter += 1
        (fake_root / "Courseforge" / "exports").mkdir(parents=True)
        project = _make_project(
            fake_root, course_name="BIO_201", chapters=_chapters(4),
        )
        payload = _run_plan(
            fake_root, project, course_name="BIO_201",
            provider_factory=None, provider_env=None,
            monkeypatch=monkeypatch,
        )
        synth = payload["_synthesized"]
        # synthesized_at is a wall-clock timestamp and generated_from is
        # an absolute path to the (per-run) project export dir — drop
        # both before comparing the *objective content* for stability.
        synth.pop("synthesized_at", None)
        synth.pop("generated_from", None)
        return json.dumps(synth, sort_keys=True)

    _one_run.counter = 0
    first = _one_run()
    second = _one_run()
    assert first == second, (
        "the deterministic course-planning path must be byte-stable "
        "across runs when TEXTBOOK_SYNTHESIS_PROVIDER is unset"
    )


# ---------------------------------------------------------------------------
# Test 2 — flag set + stub → per-chapter COs, reconciled TOs, sub_objectives
# ---------------------------------------------------------------------------


def test_flag_set_per_chapter_cos_minted_globally_sequential(
    tmp_path, monkeypatch
):
    """Flag set: 3 chapters x 2 COs => 6 COs minted CO-01..CO-06
    globally-sequential (NOT per-chapter restart)."""
    fake_root = tmp_path / "root"
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    project = _make_project(
        fake_root, course_name="PHYS_101", chapters=_chapters(3),
        draft_terminal_objectives=_draft_tos(),
    )
    payload = _run_plan(
        fake_root, project, course_name="PHYS_101",
        provider_factory=_StubProvider, provider_env="anthropic",
        monkeypatch=monkeypatch,
    )
    assert payload["success"] is True
    assert payload["mint_method"] == "textbook_synthesis:anthropic"
    assert payload["chapter_count"] == 6

    synth = payload["_synthesized"]
    co_ids: List[str] = []
    for group in synth["chapter_objectives"]:
        for entry in group["objectives"]:
            co_ids.append(entry["id"])
    # Globally sequential CO-01..CO-06 — every chapter's COs share one
    # ascending namespace, not a per-chapter restart.
    assert co_ids == [f"CO-{n:02d}" for n in range(1, 7)]


def test_flag_set_reconciled_terminal_objectives(tmp_path, monkeypatch):
    """Flag set: the Stage-1 draft TO-NN are reconciled — the on-disk
    terminal_objectives carry the reconciliation marker, no `draft`."""
    fake_root = tmp_path / "root"
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    project = _make_project(
        fake_root, course_name="PHYS_101", chapters=_chapters(2),
        draft_terminal_objectives=_draft_tos(),
    )
    payload = _run_plan(
        fake_root, project, course_name="PHYS_101",
        provider_factory=_StubProvider, provider_env="anthropic",
        monkeypatch=monkeypatch,
    )
    assert payload["success"] is True
    synth = payload["_synthesized"]
    terminals = synth["terminal_objectives"]
    assert terminals, "reconciled terminal_objectives must be non-empty"
    assert payload["terminal_count"] == len(terminals)
    for to in terminals:
        # The stub prefixes every reconciled statement.
        assert to["statement"].startswith("[reconciled] ")
        # The draft flag is dropped in the reconciled set.
        assert "draft" not in to


def test_flag_set_cos_carry_sub_objectives(tmp_path, monkeypatch):
    """Flag set: each CO carries the optional nested sub_objectives[]
    field (plan §5.2 additive field)."""
    fake_root = tmp_path / "root"
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    project = _make_project(
        fake_root, course_name="PHYS_101", chapters=_chapters(2),
        draft_terminal_objectives=_draft_tos(),
    )
    payload = _run_plan(
        fake_root, project, course_name="PHYS_101",
        provider_factory=_StubProvider, provider_env="anthropic",
        monkeypatch=monkeypatch,
    )
    assert payload["success"] is True
    synth = payload["_synthesized"]
    seen = 0
    for group in synth["chapter_objectives"]:
        for entry in group["objectives"]:
            assert "sub_objectives" in entry, (
                "every Stage-2 CO must carry sub_objectives[]"
            )
            assert isinstance(entry["sub_objectives"], list)
            assert entry["sub_objectives"], (
                "the stub emits 2 sub-objectives per CO"
            )
            seen += 1
    assert seen == 4  # 2 chapters x 2 COs


# ---------------------------------------------------------------------------
# Test 3 — per-chapter failure isolation (§5.4)
# ---------------------------------------------------------------------------


def test_per_chapter_failure_isolation_phase_succeeds(
    tmp_path, monkeypatch
):
    """Flag set, 1 of 3 chapters fails: the phase still succeeds — the
    surviving 2 chapters produce COs (plan §5.4 failure isolation)."""
    fake_root = tmp_path / "root"
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    project = _make_project(
        fake_root, course_name="PHYS_101", chapters=_chapters(3),
        draft_terminal_objectives=_draft_tos(),
    )

    def _factory(**kwargs):
        return _StubProvider(fail_chapter_ids=["ch2"], **kwargs)

    payload = _run_plan(
        fake_root, project, course_name="PHYS_101",
        provider_factory=_factory, provider_env="anthropic",
        monkeypatch=monkeypatch,
    )
    # Phase succeeds despite the chapter-2 failure.
    assert payload["success"] is True
    assert payload["mint_method"] == "textbook_synthesis:anthropic"
    # 2 surviving chapters x 2 COs = 4 (ch2's COs are dropped).
    assert payload["chapter_count"] == 4
    co_ids: List[str] = []
    for group in payload["_synthesized"]["chapter_objectives"]:
        for entry in group["objectives"]:
            co_ids.append(entry["id"])
    assert co_ids == ["CO-01", "CO-02", "CO-03", "CO-04"]


# ---------------------------------------------------------------------------
# Test 4 — all chapters fail → deterministic fallback, reconciliation skip
# ---------------------------------------------------------------------------


def test_all_chapters_fail_falls_back_to_deterministic(
    tmp_path, monkeypatch
):
    """Flag set, EVERY chapter fails: Stage 2 degrades to the
    deterministic synthesizer and reconciliation is skipped (§5.4)."""
    fake_root = tmp_path / "root"
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    project = _make_project(
        fake_root, course_name="PHYS_101", chapters=_chapters(3),
        draft_terminal_objectives=_draft_tos(),
    )
    # _AllFailProvider.reconcile_terminal_objectives asserts-fails if
    # reconciliation is reached — so a clean run proves it was skipped.
    payload = _run_plan(
        fake_root, project, course_name="PHYS_101",
        provider_factory=_AllFailProvider, provider_env="anthropic",
        monkeypatch=monkeypatch,
    )
    assert payload["success"] is True
    # Deterministic fallback fired — NOT a textbook_synthesis mint.
    assert payload["mint_method"] == "synthesize_objectives_from_topics"


def test_no_chapters_falls_back_to_deterministic(tmp_path, monkeypatch):
    """Flag set but textbook_structure carries no chapters[]: Stage 2
    cannot run; the deterministic synthesizer fires."""
    fake_root = tmp_path / "root"
    (fake_root / "Courseforge" / "exports").mkdir(parents=True)
    project = _make_project(
        fake_root, course_name="PHYS_101", chapters=[],
    )
    payload = _run_plan(
        fake_root, project, course_name="PHYS_101",
        provider_factory=_StubProvider, provider_env="anthropic",
        monkeypatch=monkeypatch,
    )
    assert payload["success"] is True
    assert payload["mint_method"] == "synthesize_objectives_from_topics"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
