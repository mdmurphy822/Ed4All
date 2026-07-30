"""Tests for the ``ED4ALL_HOME`` relocatable data root (Marketable-v1 D3).

Covers the precedence matrix (home set/unset x per-dir override set/unset) for
every data-dir resolver, plus directory bootstrap. All synthetic via tmp_path +
monkeypatch; no course slugs/paths.

The resolver functions (``libv2_path``, ``get_training_captures_dir``,
``get_state_runs_dir``, ``courseforge_exports_dir``, ``semantik_output_dir``) read
the env at CALL time, so they reflect a monkeypatched ``ED4ALL_HOME`` without a
module re-import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib import paths


# --------------------------------------------------------------- ed4all_home()


def test_ed4all_home_unset_is_none(monkeypatch):
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    assert paths.ed4all_home() is None


def test_ed4all_home_blank_is_none(monkeypatch):
    monkeypatch.setenv("ED4ALL_HOME", "   ")
    assert paths.ed4all_home() is None


def test_ed4all_home_set(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path))
    assert paths.ed4all_home() == tmp_path


# ---------------------------------------------------------- precedence matrix
#
# Each resolver: (resolver, basename-under-home, per-dir-env, default-attr).
# ``default-attr`` names the lib.paths module global the resolver falls through
# to when home + override are both unset. We read it LIVE (not a frozen literal)
# because the repo-root session-isolation conftest setattr's LIBV2_PATH /
# TRAINING_DIR to a session tmp — the resolver must honor that constant, and the
# test asserts exactly that "falls through to the current module global" contract.
_MATRIX = [
    ("libv2_path", "libv2", "ED4ALL_LIBV2_ROOT", "LIBV2_PATH"),
    (
        "get_training_captures_dir",
        "training-captures",
        "ED4ALL_TRAINING_CAPTURES_DIR",
        "TRAINING_DIR",
    ),
    ("courseforge_exports_dir", "exports", None, None),
    ("semantik_output_dir", "semantik-output", None, None),
]


def _default_path(default_attr, basename):
    """Resolve the expected in-tree default for a resolver row."""
    if default_attr is not None:
        return getattr(paths, default_attr)
    if basename == "exports":
        return paths.COURSEFORGE_PATH / "exports"
    if basename == "semantik-output":
        return paths.SEMANTIK_PATH / "output"
    raise AssertionError(basename)


@pytest.mark.parametrize("resolver_name,basename,per_dir_env,default_attr", _MATRIX)
def test_unset_home_unset_override_uses_in_tree_default(
    monkeypatch, resolver_name, basename, per_dir_env, default_attr
):
    """home unset + override unset -> falls through to the in-tree module global."""
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    if per_dir_env:
        monkeypatch.delenv(per_dir_env, raising=False)
    resolver = getattr(paths, resolver_name)
    assert resolver() == _default_path(default_attr, basename)


@pytest.mark.parametrize("resolver_name,basename,per_dir_env,default_attr", _MATRIX)
def test_set_home_unset_override_relocates_under_home(
    monkeypatch, tmp_path, resolver_name, basename, per_dir_env, default_attr
):
    """home set + override unset -> <home>/<basename>."""
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path))
    if per_dir_env:
        monkeypatch.delenv(per_dir_env, raising=False)
    resolver = getattr(paths, resolver_name)
    assert resolver() == tmp_path / basename


@pytest.mark.parametrize("resolver_name,basename,per_dir_env,default_attr", _MATRIX)
def test_per_dir_override_wins_over_home(
    monkeypatch, tmp_path, resolver_name, basename, per_dir_env, default_attr
):
    """home set + override set -> override wins (highest precedence)."""
    if per_dir_env is None:
        pytest.skip(f"{resolver_name} has no per-dir override")
    override = tmp_path / "explicit_override"
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(per_dir_env, str(override))
    resolver = getattr(paths, resolver_name)
    assert resolver() == override


@pytest.mark.parametrize("resolver_name,basename,per_dir_env,default_attr", _MATRIX)
def test_per_dir_override_wins_with_home_unset(
    monkeypatch, tmp_path, resolver_name, basename, per_dir_env, default_attr
):
    """home unset + override set -> override wins."""
    if per_dir_env is None:
        pytest.skip(f"{resolver_name} has no per-dir override")
    override = tmp_path / "explicit_override"
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    monkeypatch.setenv(per_dir_env, str(override))
    resolver = getattr(paths, resolver_name)
    assert resolver() == override


# ---------------------------------------------- runtime/state/runs has its own override


def test_state_runs_unset_home_unset_override(monkeypatch):
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    monkeypatch.delenv("ED4ALL_STATE_RUNS_DIR", raising=False)
    assert paths.get_state_runs_dir() == paths.PROJECT_ROOT / "runtime" / "state" / "runs"


def test_state_runs_relocates_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path))
    monkeypatch.delenv("ED4ALL_STATE_RUNS_DIR", raising=False)
    assert paths.get_state_runs_dir() == tmp_path / "state" / "runs"


def test_state_runs_override_wins_over_home(monkeypatch, tmp_path):
    override = tmp_path / "runs_override"
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(override))
    assert paths.get_state_runs_dir() == override


# ------------------------------------------------------------------- bootstrap


def test_ensure_data_dir_creates_missing(tmp_path):
    target = tmp_path / "fresh_home" / "state"
    assert not target.exists()
    returned = paths.ensure_data_dir(target)
    assert returned == target
    assert target.is_dir()


def test_ensure_data_dir_noop_when_present(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    # Drop a marker; ensure_data_dir must not wipe it.
    (target / "keep.txt").write_text("x")
    paths.ensure_data_dir(target)
    assert (target / "keep.txt").read_text() == "x"


# ------------------ DART->semantik purge: legacy dart-output dual-READ (task #19)


def test_semantik_output_prefers_canonical_basename_under_home(monkeypatch, tmp_path):
    """home set, neither basename on disk -> canonical <home>/semantik-output."""
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path))
    assert paths.semantik_output_dir() == tmp_path / "semantik-output"


def test_semantik_output_falls_back_to_legacy_dart_output_dir(monkeypatch, tmp_path):
    """A pre-rename ED4ALL_HOME box carrying only dart-output/ resolves to it
    (with a DeprecationWarning) instead of a non-existent semantik-output/.

    Load-bearing: the deployed DGX Spark box has an existing dart-output/ under
    ED4ALL_HOME (provisioned before the task #19 rename)."""
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path))
    (tmp_path / "dart-output").mkdir()  # legacy dir present, canonical absent
    with pytest.warns(DeprecationWarning):
        resolved = paths.semantik_output_dir()
    assert resolved == tmp_path / "dart-output"


def test_semantik_output_prefers_canonical_when_both_exist(monkeypatch, tmp_path):
    """Canonical semantik-output/ wins over a leftover legacy dart-output/."""
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path))
    (tmp_path / "dart-output").mkdir()
    (tmp_path / "semantik-output").mkdir()
    assert paths.semantik_output_dir() == tmp_path / "semantik-output"


def test_relocated_dirs_are_distinct_under_home(monkeypatch, tmp_path):
    """Every relocated data dir lands at a distinct path under one home root."""
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path))
    for env in (
        "ED4ALL_LIBV2_ROOT",
        "ED4ALL_TRAINING_CAPTURES_DIR",
        "ED4ALL_STATE_RUNS_DIR",
    ):
        monkeypatch.delenv(env, raising=False)
    resolved = {
        paths.libv2_path(),
        paths.get_training_captures_dir(),
        paths.courseforge_exports_dir(),
        paths.semantik_output_dir(),
        paths.get_state_runs_dir(),
    }
    # 5 distinct paths, all under the home root.
    assert len(resolved) == 5
    for p in resolved:
        assert Path(p).is_relative_to(tmp_path)
