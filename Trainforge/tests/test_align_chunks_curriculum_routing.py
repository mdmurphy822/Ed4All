"""Wave 137 followup — align_chunks CLI honors CURRICULUM_ALIGNMENT_PROVIDER.

Pre-Wave-137-followup the env var was wired in
``CurriculumAlignmentProvider.__init__``, but
``Trainforge/align_chunks.py::main()`` never instantiated the class —
which meant the env var was dead from the
``Trainforge/process_course.py`` invocation path. These tests pin the
behavior contract: the CLI now reads the env var (and the explicit
``--curriculum-provider`` flag overrides it), instantiates the
provider when set, and threads it into ``classify_teaching_roles``.

Backward-compat regression: env unset + no flag → no provider
instantiated, ``curriculum_provider=None`` passed to
``classify_teaching_roles``. The existing legacy / mock path runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge import align_chunks  # noqa: E402
from Trainforge.generators._curriculum_provider import (  # noqa: E402
    CurriculumAlignmentProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(corpus_dir: Path, **overrides) -> argparse.Namespace:
    """Build the argparse Namespace align_chunks.main() expects."""
    base = dict(
        corpus=str(corpus_dir),
        objectives=None,
        fields="teaching_role",
        llm_provider="mock",
        llm_model="claude-haiku-4-5-20251001",
        curriculum_provider=None,
        dry_run=True,
        verbose=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _stub_corpus_dir(tmp_path: Path) -> Path:
    """A minimal corpus dir is enough — ``load_corpus`` is patched."""
    d = tmp_path / "STUBCOURSE"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear the env var before each test so no leakage across runs."""
    monkeypatch.delenv("CURRICULUM_ALIGNMENT_PROVIDER", raising=False)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_align_chunks_main_honors_env_var_when_no_cli_flag(
    monkeypatch, tmp_path,
):
    """env=local + no CLI flag → provider instantiated with provider='local'."""
    monkeypatch.setenv("CURRICULUM_ALIGNMENT_PROVIDER", "local")
    monkeypatch.setenv(
        "LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1",
    )

    corpus_dir = _stub_corpus_dir(tmp_path)
    args = _make_args(corpus_dir)

    constructed: list = []

    def _fake_provider(**kwargs):
        constructed.append(kwargs)
        return MagicMock(name="CurriculumAlignmentProvider")

    classify_seen: dict = {}

    def _fake_classify(chunks, **kwargs):
        classify_seen.update(kwargs)

    with patch.object(
        align_chunks, "_build_curriculum_provider",
        wraps=lambda choice, capture=None: _fake_provider(
            provider=choice, capture=capture,
        ),
    ), patch.object(
        align_chunks, "load_corpus",
        return_value=([], {"nodes": [], "edges": []}),
    ), patch.object(
        align_chunks, "build_chunk_sequence", side_effect=lambda c: c,
    ), patch.object(
        align_chunks, "classify_teaching_roles", side_effect=_fake_classify,
    ):
        align_chunks.main(args)

    assert len(constructed) == 1, (
        "CurriculumAlignmentProvider should be instantiated exactly once"
    )
    assert constructed[0]["provider"] == "local"
    # And the provider was threaded into classify_teaching_roles.
    assert classify_seen.get("curriculum_provider") is not None


def test_align_chunks_main_cli_flag_beats_env_var(monkeypatch, tmp_path):
    """env=anthropic + CLI=local → CLI wins; provider is 'local'."""
    monkeypatch.setenv("CURRICULUM_ALIGNMENT_PROVIDER", "anthropic")
    monkeypatch.setenv(
        "LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1",
    )

    corpus_dir = _stub_corpus_dir(tmp_path)
    args = _make_args(corpus_dir, curriculum_provider="local")

    constructed: list = []

    def _fake_provider(**kwargs):
        constructed.append(kwargs)
        return MagicMock(name="CurriculumAlignmentProvider")

    with patch.object(
        align_chunks, "_build_curriculum_provider",
        wraps=lambda choice, capture=None: _fake_provider(
            provider=choice, capture=capture,
        ),
    ), patch.object(
        align_chunks, "load_corpus",
        return_value=([], {"nodes": [], "edges": []}),
    ), patch.object(
        align_chunks, "build_chunk_sequence", side_effect=lambda c: c,
    ), patch.object(
        align_chunks, "classify_teaching_roles", side_effect=lambda *_a, **_k: None,
    ):
        align_chunks.main(args)

    assert len(constructed) == 1
    # CLI flag wins; the env value 'anthropic' is ignored.
    assert constructed[0]["provider"] == "local"


def test_align_chunks_main_no_env_no_cli_skips_provider(
    monkeypatch, tmp_path,
):
    """env unset + no CLI flag → curriculum_provider=None passed through.

    Regression-protects existing behavior — silent fallback to
    legacy / mock path when neither env nor CLI flag is set.
    """
    corpus_dir = _stub_corpus_dir(tmp_path)
    args = _make_args(corpus_dir)

    classify_seen: dict = {}

    def _fake_classify(chunks, **kwargs):
        classify_seen.update(kwargs)

    with patch.object(
        align_chunks, "_build_curriculum_provider",
    ) as build_mock, patch.object(
        align_chunks, "load_corpus",
        return_value=([], {"nodes": [], "edges": []}),
    ), patch.object(
        align_chunks, "build_chunk_sequence", side_effect=lambda c: c,
    ), patch.object(
        align_chunks, "classify_teaching_roles", side_effect=_fake_classify,
    ):
        align_chunks.main(args)

    # No provider should have been built.
    build_mock.assert_not_called()
    # And classify_teaching_roles should have been called with
    # curriculum_provider=None — preserves the legacy / mock path.
    assert "curriculum_provider" in classify_seen
    assert classify_seen["curriculum_provider"] is None


def test_align_chunks_main_unknown_provider_exits_2(monkeypatch, tmp_path):
    """Bad provider value → clean exit 2 with stderr message."""
    corpus_dir = _stub_corpus_dir(tmp_path)
    # Bypass argparse's choices check by feeding the namespace directly,
    # so we exercise the runtime ValueError → exit 2 path.
    args = _make_args(corpus_dir, curriculum_provider="bogus")

    with patch.object(
        align_chunks, "load_corpus",
        return_value=([], {"nodes": [], "edges": []}),
    ), patch.object(
        align_chunks, "build_chunk_sequence", side_effect=lambda c: c,
    ), patch.object(
        align_chunks, "classify_teaching_roles", side_effect=lambda *_a, **_k: None,
    ):
        with pytest.raises(SystemExit) as excinfo:
            align_chunks.main(args)

    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Smoke check on the helper function in isolation
# ---------------------------------------------------------------------------


def test_resolve_curriculum_provider_choice_priority(monkeypatch, tmp_path):
    """CLI flag > env > None precedence is verifiable at the helper level."""
    corpus_dir = _stub_corpus_dir(tmp_path)
    # 1. CLI wins.
    monkeypatch.setenv("CURRICULUM_ALIGNMENT_PROVIDER", "anthropic")
    args = _make_args(corpus_dir, curriculum_provider="together")
    assert (
        align_chunks._resolve_curriculum_provider_choice(args) == "together"
    )
    # 2. Env wins when CLI is unset.
    args = _make_args(corpus_dir, curriculum_provider=None)
    assert (
        align_chunks._resolve_curriculum_provider_choice(args) == "anthropic"
    )
    # 3. None when both are unset.
    monkeypatch.delenv("CURRICULUM_ALIGNMENT_PROVIDER", raising=False)
    args = _make_args(corpus_dir, curriculum_provider=None)
    assert align_chunks._resolve_curriculum_provider_choice(args) is None


def test_build_curriculum_provider_returns_real_class(monkeypatch):
    """``_build_curriculum_provider('local')`` returns a real provider."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = align_chunks._build_curriculum_provider("local", capture=None)
    assert isinstance(p, CurriculumAlignmentProvider)
    assert p._provider == "local"


def test_build_curriculum_provider_propagates_value_error_for_unknown():
    """Unknown provider strings raise ValueError (caught by main → exit 2)."""
    with pytest.raises(ValueError):
        align_chunks._build_curriculum_provider("bogus", capture=None)
