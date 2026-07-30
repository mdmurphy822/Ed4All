"""Regression tests for the shared ``engine="auto"`` resolver.

``lib.libv2_storage.resolve_auto_engine`` is the SINGLE resolver for the auto
seam. It exists because the two entry points that expose ``auto`` — the
``libv2 answer-grounded`` CLI and the GUI learner ask service — each carried
their own three-line copy of the policy, and the copies drifted: the CLI
resolved ``auto`` to ``semantic`` while the GUI resolved it to ``hybrid-rrf``.
Same flag name, two different engines depending on entry point.

Pinned here:

- ``auto`` -> ``hybrid-rrf`` when the course has a vector index, else
  ``lexical``. hybrid-rrf is the benchmark-selected engine (pure semantic never
  beat the BM25 baseline).
- an explicit engine is NEVER rewritten — the anti-silent-degradation contract
  depends on an explicit ``semantic`` reaching the pipeline so a missing index
  raises ``SemanticIndexMissing`` instead of quietly falling back to BM25.
- the GUI wrapper agrees with the shared resolver on every input, so the two
  surfaces cannot drift apart again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.libv2_storage import (
    VECTOR_INDEX_DIRNAME,
    VECTOR_INDEX_MANIFEST_FILENAME,
    has_vector_index,
    resolve_auto_engine,
)


def _course(root: Path, slug: str = "demo-course", *, with_index: bool) -> Path:
    course_dir = root / "courses" / slug
    course_dir.mkdir(parents=True, exist_ok=True)
    if with_index:
        index_dir = course_dir / VECTOR_INDEX_DIRNAME
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / VECTOR_INDEX_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    return course_dir


class TestHasVectorIndex:
    def test_true_only_when_manifest_is_a_file(self, tmp_path):
        assert has_vector_index(_course(tmp_path, with_index=True)) is True
        assert has_vector_index(_course(tmp_path, "bare", with_index=False)) is False

    def test_bare_index_dir_without_manifest_is_not_an_index(self, tmp_path):
        course_dir = _course(tmp_path, "half", with_index=False)
        (course_dir / VECTOR_INDEX_DIRNAME).mkdir()
        assert has_vector_index(course_dir) is False

    def test_missing_course_dir_does_not_raise(self, tmp_path):
        assert has_vector_index(tmp_path / "courses" / "nope") is False


class TestResolveAutoEngine:
    def test_auto_resolves_hybrid_rrf_with_index(self, tmp_path):
        assert resolve_auto_engine("auto", _course(tmp_path, with_index=True)) == "hybrid-rrf"

    def test_auto_resolves_lexical_without_index(self, tmp_path):
        assert resolve_auto_engine("auto", _course(tmp_path, with_index=False)) == "lexical"

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_is_treated_as_auto(self, tmp_path, blank):
        assert resolve_auto_engine(blank, _course(tmp_path, with_index=True)) == "hybrid-rrf"

    @pytest.mark.parametrize("engine", ["lexical", "semantic", "hybrid-rrf"])
    def test_explicit_engine_passes_through_verbatim(self, tmp_path, engine):
        """An explicit request is never rewritten — in either index state.

        Load-bearing: explicit ``semantic`` against a pre-index tree must reach
        the pipeline so it can raise ``SemanticIndexMissing``.
        """
        assert resolve_auto_engine(engine, _course(tmp_path, "a", with_index=True)) == engine
        assert resolve_auto_engine(engine, _course(tmp_path, "b", with_index=False)) == engine

    def test_case_and_whitespace_normalized(self, tmp_path):
        assert resolve_auto_engine("  AUTO ", _course(tmp_path, with_index=True)) == "hybrid-rrf"


class TestSurfacesAgree:
    """The GUI wrapper must agree with the shared resolver on every input."""

    @pytest.mark.parametrize("engine", ["auto", "", "lexical", "semantic", "hybrid-rrf"])
    @pytest.mark.parametrize("with_index", [True, False])
    def test_gui_wrapper_matches_shared_resolver(self, tmp_path, engine, with_index):
        pytest.importorskip("pydantic")  # gui.services imports stay light, but guard anyway
        from gui.services import answer_service

        slug = f"c-{int(with_index)}"
        course_dir = _course(tmp_path, slug, with_index=with_index)
        assert answer_service._resolve_engine(engine, tmp_path, slug) == resolve_auto_engine(
            engine, course_dir
        )
