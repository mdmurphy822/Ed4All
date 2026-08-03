"""Tests for the repository layout guard (ci/layout_guard.py).

Proves the guard (a) passes the current tracked tree, (b) catches a
synthetic violation of each of the five checks, (c) parses the allowlist
correctly — comments, all six prefixes, and loud rejection of an
unrecognized or malformed entry — and (d) skips loudly (exit 0) when git
is unavailable. The five check functions are pure (injected ``tracked``
list + parsed ``Allowlist``), so (a)-(c) need no git fixtures; only (d),
the real-repo smoke test, and the flatcap-seed-exactness test touch git.

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
    tracked = ["scripts/harness/gold_compare.py", "scripts/new_pilot.py"]
    violations = guard.check_scripts_snapshot(
        tracked, allowed_scripts={"gold_compare.py"}
    )
    assert len(violations) == 1
    assert violations[0].path == "scripts/new_pilot.py"
    assert violations[0].check == "scripts_snapshot"


def test_scripts_snapshot_ignores_subdir_contents():
    tracked = [
        "scripts/harness/gold_compare.py",
        "scripts/archive/anything_goes_here.py",
        "scripts/tests/test_whatever.py",
    ]
    violations = guard.check_scripts_snapshot(
        tracked, allowed_scripts={"gold_compare.py"}
    )
    assert violations == []


# --- check 5: interior flat-file cap --------------------------------------

def test_flatcap_parses_and_rejects_malformed_entries():
    allow = guard.parse_allowlist("flatcap:lib/retrieval=37\nflatcap:gui=8")
    assert allow.flatcaps == {"lib/retrieval": 37, "gui": 8}

    for bad in ("flatcap:lib", "flatcap:lib=", "flatcap:=3", "flatcap:lib=x"):
        with pytest.raises(ValueError):
            guard.parse_allowlist(bad)


def test_flatcap_rejects_negative_and_duplicate_caps():
    # A negative cap can never be satisfied; a duplicate means one of the two
    # lines is a silent no-op. Both are repo-config bugs, not passes.
    with pytest.raises(ValueError):
        guard.parse_allowlist("flatcap:lib=-1")
    with pytest.raises(ValueError):
        guard.parse_allowlist("flatcap:lib=1\nflatcap:lib=2")


def test_interior_flat_cap_catches_growth_past_the_cap():
    tracked = [f"SemantiK/scripts/s{i}.py" for i in range(3)]
    violations = guard.check_interior_flat_cap(tracked, {"SemantiK/scripts": 2})
    assert len(violations) == 1
    assert violations[0].check == "interior_flat_cap"
    assert violations[0].path == "SemantiK/scripts"
    assert "cap is 2" in violations[0].message


def test_interior_flat_cap_silent_at_or_under_the_cap():
    tracked = ["lib/retrieval/a.py", "lib/retrieval/b.py"]
    assert guard.check_interior_flat_cap(tracked, {"lib/retrieval": 2}) == []
    assert guard.check_interior_flat_cap(tracked, {"lib/retrieval": 5}) == []


def test_interior_flat_cap_excludes_tests_init_and_markdown():
    # Adding a test, a package marker, or a doc must never trip the cap —
    # those are mandated or encouraged AT the directory root.
    tracked = [
        "lib/retrieval/real_module.py",
        "lib/retrieval/__init__.py",
        "lib/retrieval/CLAUDE.md",
        "lib/retrieval/tests/test_a.py",
        "lib/retrieval/tests/test_b.py",
    ]
    assert guard.count_flat_code_files(tracked)["lib/retrieval"] == 1
    assert guard.check_interior_flat_cap(tracked, {"lib/retrieval": 1}) == []


def test_interior_flat_cap_ignores_dirs_outside_the_interior_roots():
    # schemas/ and config/ are CONTRACTS — their flat shape IS the contract,
    # so the cap must not reach them even if they hold many loose files.
    tracked = [f"schemas/knowledge/s{i}.json" for i in range(50)]
    assert guard.count_flat_code_files(tracked) == {}


def test_interior_flat_cap_reports_a_cap_on_a_vanished_dir():
    # Otherwise a reorg leaves a dead cap behind and the ratchet silently
    # stops enforcing anything for that path.
    violations = guard.check_interior_flat_cap(
        ["lib/retrieval/a.py"], {"Trainforge/gone": 5}
    )
    assert len(violations) == 1
    assert "enforces nothing" in violations[0].message


# --- check_layout composition ---------------------------------------------

def test_check_layout_combines_all_five_checks():
    tracked = [
        "newdir/x.py",
        "lib/new_flat.py",
        "docs/badbucket/y.md",
        "scripts/loose.py",
        "gui/services/a.py",
        "gui/services/b.py",
    ]
    allowlist = guard.Allowlist(
        dirs={"lib", "docs", "scripts", "gui"},
        files=set(),
        libflat=set(),
        docsingle=set(),
        scripts=set(),
        flatcaps={"gui/services": 1},
    )
    violations = guard.check_layout(tracked, allowlist)
    fired = {v.check for v in violations}
    assert fired == {
        "top_level_closed",
        "lib_flat_ratchet",
        "docs_taxonomy",
        "scripts_snapshot",
        "interior_flat_cap",
    }


def test_every_seeded_flatcap_matches_the_real_tree_exactly():
    """The seed must be exact, not merely satisfied.

    A cap seeded ABOVE the real count is slack the ratchet never recovers —
    it silently permits growth up to the inflated number. This pins the
    seed to reality so drift shows up as a test failure.
    """
    tracked = guard.iter_tracked_files(_CI_DIR.parent)
    if tracked is None:
        pytest.skip("not a git work tree")
    counts = guard.count_flat_code_files(tracked)
    allow = guard.load_allowlist(_CI_DIR.parent)
    slack = {
        d: (cap, counts.get(d))
        for d, cap in allow.flatcaps.items()
        if counts.get(d) != cap
    }
    assert not slack, f"flatcap drift (dir: (cap, actual)): {slack}"


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
