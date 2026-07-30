"""Tests for the repository layout guard (ci/layout_guard.py).

Proves the guard (a) passes the current tracked tree, (b) catches a
synthetic violation of each of the four checks, (c) parses the allowlist
correctly — comments, all five prefixes, and loud rejection of an
unrecognized entry — and (d) skips loudly (exit 0) when git is
unavailable. The four check functions are pure (injected ``tracked``
list + parsed ``Allowlist``), so (a)-(c) need no git fixtures; only (d)
and the real-repo smoke test touch git.

Runnable standalone: ``python ci/tests/test_layout_guard.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Import the guard module whether run under pytest (with the repo on the
# path) or as a standalone script.
_CI_DIR = Path(__file__).resolve().parent.parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import layout_guard as guard  # noqa: E402


# --- allowlist parsing ----------------------------------------------------

def test_parse_allowlist_all_prefixes_and_comments():
    text = """
    # a full-line comment
    dir:ci
    dir:lib   # inline comment
    file:README.md

    libflat:paths.py
    docsingle:reference
    script:gold_compare.py
    """
    allowlist = guard.parse_allowlist(text)
    assert allowlist.dirs == {"ci", "lib"}
    assert allowlist.files == {"README.md"}
    assert allowlist.libflat == {"paths.py"}
    assert allowlist.docsingle == {"reference"}
    assert allowlist.scripts == {"gold_compare.py"}


def test_parse_allowlist_rejects_unknown_prefix():
    with pytest.raises(ValueError):
        guard.parse_allowlist("weird:something\n")


def test_parse_allowlist_rejects_bare_line():
    with pytest.raises(ValueError):
        guard.parse_allowlist("just-a-bare-token\n")


def test_parse_allowlist_rejects_empty_value():
    with pytest.raises(ValueError):
        guard.parse_allowlist("dir:\n")


def test_parse_allowlist_ignores_blank_and_comment_only_lines():
    allowlist = guard.parse_allowlist("\n# comment only\n\ndir:ci\n")
    assert allowlist.dirs == {"ci"}


# --- pure check functions: synthetic violations --------------------------

def test_top_level_closed_catches_new_dir_and_new_root_file():
    tracked = ["ci/layout_guard.py", "newdir/foo.py", "stray_root_file.py"]
    violations = guard.check_top_level_closed(
        tracked, allowed_dirs={"ci"}, allowed_files=set()
    )
    checks = {(v.path, v.check) for v in violations}
    assert ("newdir", "top_level_closed") in checks
    assert ("stray_root_file.py", "top_level_closed") in checks
    # ci/ is allowlisted, so it must not appear.
    assert not any(v.path == "ci" for v in violations)


def test_top_level_closed_passes_allowlisted_entries():
    tracked = ["ci/layout_guard.py", "README.md"]
    violations = guard.check_top_level_closed(
        tracked, allowed_dirs={"ci"}, allowed_files={"README.md"}
    )
    assert violations == []


def test_lib_flat_ratchet_catches_new_flat_module():
    tracked = ["lib/__init__.py", "lib/paths.py", "lib/new_thing.py"]
    violations = guard.check_lib_flat_ratchet(tracked, allowed_libflat={"paths.py"})
    assert len(violations) == 1
    assert violations[0].path == "lib/new_thing.py"
    assert violations[0].check == "lib_flat_ratchet"


def test_lib_flat_ratchet_ignores_init_and_subpackages():
    tracked = [
        "lib/__init__.py",
        "lib/paths.py",
        "lib/validators/foo.py",  # depth 2, not a flat module
    ]
    violations = guard.check_lib_flat_ratchet(tracked, allowed_libflat={"paths.py"})
    assert violations == []


def test_docs_taxonomy_catches_disallowed_subdir():
    tracked = ["docs/architecture/a.md", "docs/wrongbucket/b.md"]
    violations = guard.check_docs_taxonomy(tracked, allowed_docsingle=set())
    assert any(
        v.check == "docs_taxonomy" and v.path == "docs/wrongbucket/b.md"
        for v in violations
    )


def test_docs_taxonomy_catches_single_file_dir():
    tracked = [
        "docs/architecture/a.md",
        "docs/architecture/b.md",
        "docs/validation/only_one.md",
    ]
    violations = guard.check_docs_taxonomy(tracked, allowed_docsingle=set())
    single_file_hits = [v for v in violations if "single-file" in v.message]
    assert len(single_file_hits) == 1
    assert single_file_hits[0].path == "docs/validation/only_one.md"


def test_docs_taxonomy_docsingle_allowlist_escape():
    tracked = ["docs/validation/only_one.md"]
    violations = guard.check_docs_taxonomy(tracked, allowed_docsingle={"validation"})
    assert violations == []


def test_docs_taxonomy_ignores_depth_one_root_docs_files():
    tracked = ["docs/LICENSING.md", "docs/TECH_DEBT.md"]
    violations = guard.check_docs_taxonomy(tracked, allowed_docsingle=set())
    assert violations == []


def test_scripts_snapshot_catches_new_loose_file():
    tracked = ["scripts/gold_compare.py", "scripts/new_pilot.py"]
    violations = guard.check_scripts_snapshot(
        tracked, allowed_scripts={"gold_compare.py"}
    )
    assert len(violations) == 1
    assert violations[0].path == "scripts/new_pilot.py"
    assert violations[0].check == "scripts_snapshot"


def test_scripts_snapshot_ignores_subdir_contents():
    tracked = [
        "scripts/gold_compare.py",
        "scripts/archive/anything_goes_here.py",
        "scripts/tests/test_whatever.py",
    ]
    violations = guard.check_scripts_snapshot(
        tracked, allowed_scripts={"gold_compare.py"}
    )
    assert violations == []


# --- check_layout composition ---------------------------------------------

def test_check_layout_combines_all_four_checks():
    tracked = [
        "newdir/x.py",
        "lib/new_flat.py",
        "docs/badbucket/y.md",
        "scripts/loose.py",
    ]
    allowlist = guard.Allowlist(
        dirs={"lib", "docs", "scripts"},
        files=set(),
        libflat=set(),
        docsingle=set(),
        scripts=set(),
    )
    violations = guard.check_layout(tracked, allowlist)
    fired = {v.check for v in violations}
    assert fired == {
        "top_level_closed",
        "lib_flat_ratchet",
        "docs_taxonomy",
        "scripts_snapshot",
    }


# --- git glue: skip-on-no-git + real repo -----------------------------

def test_iter_tracked_files_returns_none_outside_a_work_tree(tmp_path):
    assert guard.iter_tracked_files(tmp_path) is None


def test_scan_repository_raises_runtime_error_outside_a_work_tree(tmp_path):
    with pytest.raises(RuntimeError):
        guard.scan_repository(tmp_path)


def test_main_skips_loudly_when_git_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    exit_code = guard.main([])
    assert exit_code == 0


def test_scan_repository_on_synthetic_repo_flags_planted_violation(tmp_path):
    (tmp_path / "ci").mkdir()
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "lib" / "surprise.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "ci" / "layout_allowlist.txt").write_text(
        "dir:ci\ndir:lib\nlibflat:__init__.py\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    violations = guard.scan_repository(tmp_path)
    assert any(
        v.check == "lib_flat_ratchet" and v.path == "lib/surprise.py"
        for v in violations
    )


# --- the real repo must currently pass -------------------------------------

def test_current_tree_has_no_violations():
    # If this fails, it is reporting a REAL layout drift — either fix the
    # placement or extend ci/layout_allowlist.txt with justification.
    violations = guard.scan_repository(guard.PROJECT_ROOT)
    assert violations == [], "\n".join(v.format() for v in violations)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
