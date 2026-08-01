"""Tests for the foreign-repository reference guard (ci/foreign_repo_guard.py).

Proves the guard (a) catches the exact leak shape that motivated it — a
markdown link into another project's Claude-Code memory directory — (b)
recognizes every reference family, (c) does NOT fire on this repository's
own path or on the ordinary absolute paths that appear legitimately in
operator docs, and (d) honors both allowlist tiers plus the inline
marker. A guard with no test is not a guard.

Runnable standalone: ``python ci/tests/test_foreign_repo_guard.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the guard module whether run under pytest (with the repo on the
# path) or as a standalone script.
_CI_DIR = Path(__file__).resolve().parent.parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import foreign_repo_guard as guard  # noqa: E402


# --- detection: the leak that motivated the guard -----------------------

def test_catches_the_original_cross_checkout_memory_link():
    """The exact shape found in a tracked architecture doc: a relative
    markdown link that escapes into another project's agent-memory dir."""
    line = (
        "Per [`feedback_no_external_llms.md`]"
        "(../../.claude/projects/-home-someuser-Projects-Neighbour/memory/"
        "feedback_no_external_llms.md): runtime is local-only."
    )
    families = {f for f, _ in guard.find_foreign_repo_hits(line)}
    assert "claude-project-dir" in families
    assert "home-projects-slug" in families


def test_catches_every_reference_family():
    targets = [
        # Claude-Code per-project state directory (this repo tracks none).
        (".claude/projects/-home-u-Projects-Other", "claude-project-dir"),
        (".claude/projects/some-other-slug", "claude-project-dir"),
        # The path-encoded slug standing alone.
        ("-home-u-Projects-Other", "home-projects-slug"),
        # Absolute / tilde developer paths under a Projects root.
        ("/home/dev/Projects/OtherThing", "abs-projects-path"),
        ("~/Projects/SomeOther", "abs-projects-path"),
    ]
    for token, family in targets:
        hits = guard.find_foreign_repo_hits(f"see {token}/x.md for detail")
        assert hits, f"guard failed to catch: {token}"
        assert family in {f for f, _ in hits}, (
            f"{token} matched {hits}, expected family {family}"
        )


# --- the same-repo exemption is real and derived ------------------------

def test_own_repo_path_is_not_a_foreign_reference():
    """An absolute path naming THIS checkout is out of scope — it is not a
    cross-repo leak, and purging it would be noise."""
    own = guard.REPO_DIR_NAME
    for line in (
        f"/home/anyuser/Projects/{own}/lib/embedding/providers.py",
        f"~/Projects/{own}/docs/operations",
        f"/home/anyuser/Projects/{own.lower()}",  # case-insensitive
    ):
        assert guard.find_foreign_repo_hits(line) == [], line


def test_repo_dir_name_is_derived_not_hardcoded():
    """A repo rename must not silently disarm the same-repo exemption."""
    assert guard.REPO_DIR_NAME == guard.PROJECT_ROOT.name


# --- false positives: legitimate neutral content must survive -----------

def test_does_not_fire_on_legitimate_paths():
    benign = [
        "runtime/state/projects/foo",          # in-repo relative path
        "/home/user/Downloads/corpus.pdf",     # absolute, but not a Projects root
        "projects are tracked in TodoWrite",   # ordinary prose
        "LibV2/courses/<slug>/semantik_chunks",
        "/data/libv2",                         # container bind mount
        "config/workflows.yaml",
    ]
    for line in benign:
        assert guard.find_foreign_repo_hits(line) == [], line


# --- allowlist tiers + inline marker ------------------------------------

def test_inline_marker_skips_the_line(tmp_path: Path):
    repo = _make_fake_repo(
        tmp_path,
        {
            "docs/example.md": (
                f"bad: /home/dev/Projects/Neighbour  {guard.ALLOW_MARKER}\n"
            ),
        },
    )
    assert guard.scan_repository(repo) == []


def test_whole_file_allowlist_entry(tmp_path: Path):
    repo = _make_fake_repo(
        tmp_path,
        {
            "docs/example.md": "bad: /home/dev/Projects/Neighbour\n",
            guard.ALLOWLIST_FILE: "docs/example.md\n",
        },
    )
    assert guard.scan_repository(repo) == []


def test_per_file_token_allowlist_entry(tmp_path: Path):
    repo = _make_fake_repo(
        tmp_path,
        {
            "docs/example.md": (
                "ok:  /home/dev/Projects/Neighbour\n"
                "bad: /home/dev/Projects/Unlisted\n"
            ),
            guard.ALLOWLIST_FILE: "docs/example.md\t/home/dev/Projects/Neighbour\n",
        },
    )
    violations = guard.scan_repository(repo)
    assert len(violations) == 1, violations
    assert violations[0].token == "/home/dev/Projects/Unlisted"
    assert violations[0].line == 2


def test_violation_is_reported_with_file_line_and_token(tmp_path: Path):
    repo = _make_fake_repo(
        tmp_path,
        {"docs/example.md": "line one\nsee /home/dev/Projects/Neighbour/x\n"},
    )
    violations = guard.scan_repository(repo)
    assert len(violations) == 1, violations
    v = violations[0]
    assert v.path == "docs/example.md"
    assert v.line == 2
    assert "docs/example.md:2" in v.format()
    assert "Neighbour" in v.format()


# --- unscannable tree fails loudly, never silently passes ---------------

def test_non_git_tree_raises_rather_than_passing(tmp_path: Path):
    plain = tmp_path / "not-a-repo"
    (plain / "docs").mkdir(parents=True)
    (plain / "docs" / "x.md").write_text("/home/dev/Projects/Neighbour\n")
    try:
        guard.scan_repository(plain)
    except RuntimeError as exc:
        assert "git ls-files" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("scan_repository must raise on a non-git tree")


# --- the live tree must be clean ----------------------------------------

def test_tracked_tree_has_no_foreign_repo_references():
    """The real repository, scanned end to end."""
    violations = guard.scan_repository(guard.PROJECT_ROOT)
    assert violations == [], "\n".join(v.format() for v in violations)


# --- helpers ------------------------------------------------------------

def _make_fake_repo(tmp_path: Path, files: dict) -> Path:
    """Build a throwaway git work tree holding ``files`` (all tracked).

    The guard enumerates via ``git ls-files``, so the fixture must be a
    real repository with the files staged.
    """
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for rel, content in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


if __name__ == "__main__":  # pragma: no cover
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
