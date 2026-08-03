"""Regression tests for the recursive repository/source-release policy."""

from __future__ import annotations

import json
import subprocess

import pytest

from ci.guards.repository_policy import (
    ROOT,
    check_layout,
    check_privacy,
    check_public_docs,
    check_release,
    classify,
    git_paths,
    load_policy,
    load_private_tokens,
)


def policy() -> dict:
    return {
        "root_files": ["README.md"],
        "roots": {"src": "package", "var": "external-var", "tests": "tests"},
        "overrides": [],
        "allowed_children": {
            "package": ["package", "tests"],
            "tests": ["tests", "fixtures"],
            "fixtures": ["fixtures"],
            "external-var": ["external-var"],
        },
        "source_roles": ["package", "tests", "fixtures"],
        "sentinel_names": [".gitkeep"],
        "max_source_bytes": 64,
        "forbidden_generated_segments": ["build", "generated"],
        "public_docs": [],
    }


def checks(findings) -> set[str]:
    return {finding.check for finding in findings}


def test_classifies_root_files_and_descendants_recursively() -> None:
    cfg = policy()
    assert classify("README.md", cfg)[0] == "root-metadata"
    assert classify("src/deep/module.py", cfg)[0] == "package"
    assert classify("unknown/file.py", cfg)[0] is None


def test_reports_unclassified_and_ambiguous_roles() -> None:
    cfg = policy()
    cfg["overrides"] = [
        {"pattern": "src/*", "role": "package"},
        {"pattern": "src/*", "role": "tests"},
    ]
    result = check_layout(["unknown/file.py", "src/x"], cfg)
    assert {"unclassified", "ambiguous_role"} <= checks(result)


def test_reports_illegal_child_role() -> None:
    cfg = policy()
    cfg["overrides"] = [{"pattern": "src/spec", "role": "tests"}]
    cfg["allowed_children"]["package"] = ["package"]
    result = check_layout(["src/spec/test_one.py"], cfg)
    assert "illegal_child_role" in checks(result)


def test_external_var_allows_only_sentinels_and_generated_source_fails() -> None:
    result = check_layout(["var/.gitkeep", "var/private.json", "src/build/result.bin"], policy())
    assert "external_var_content" in checks(result)
    assert "generated_artifact" in checks(result)


def test_privacy_checks_path_shapes_and_operator_tokens_in_content() -> None:
    run_shape = "_".join(("TTC", "private", "20260101"))
    shaped = check_privacy(f"src/{run_shape}/result.py", b"safe", [])
    private = check_privacy("src/module.py", b"comment mentions Private Course", ["Private Course"])
    assert "privacy_shape" in checks(shaped)
    assert "private_token" in checks(private)
    assert all("Private Course" not in finding.message for finding in private)


def _init_git(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def test_private_token_file_inside_checkout_must_be_ignored(tmp_path) -> None:
    _init_git(tmp_path)
    token_file = tmp_path / "private-tokens.txt"
    token_file.write_text("private-value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-ignored"):
        load_private_tokens(tmp_path, {"ED4ALL_PRIVATE_TOKEN_FILE": str(token_file)})
    (tmp_path / ".gitignore").write_text("private-tokens.txt\n", encoding="utf-8")
    assert load_private_tokens(
        tmp_path, {"ED4ALL_PRIVATE_TOKEN_FILE": str(token_file)}
    ) == ["private-value"]


def test_git_candidates_include_untracked_nonignored_but_not_ignored(tmp_path) -> None:
    _init_git(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("tracked", encoding="utf-8")
    (tmp_path / "candidate.txt").write_text("candidate", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt", ".gitignore"], cwd=tmp_path, check=True)
    paths = git_paths(tmp_path, candidates=True)
    assert "candidate.txt" in paths
    assert "tracked.txt" in paths
    assert "ignored.txt" not in paths


def test_release_checks_secret_oversize_and_nested_repository(tmp_path) -> None:
    _init_git(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "secret.py").write_text(
        "token='ghp_abcdefghijklmnopqrstuvwxyz'", encoding="utf-8"
    )
    (tmp_path / "src" / "large.py").write_bytes(b"x" * 65)
    nested = tmp_path / "src" / "vendor" / ".git"
    nested.mkdir(parents=True)
    nested_source = tmp_path / "src" / "vendor" / "module.py"
    nested_source.write_text("safe", encoding="utf-8")
    result = check_release(
        tmp_path, ["src/secret.py", "src/large.py", "src/vendor/module.py"], policy(), []
    )
    assert {"secret", "oversized", "nested_repository"} <= checks(result)


def test_public_docs_exact_allowlist_accepts_listed_candidate() -> None:
    cfg = policy()
    cfg["public_docs"] = ["docs/guide.md"]
    assert check_public_docs(["docs/guide.md"], cfg) == []


def test_public_docs_reports_unlisted_and_stale_entries() -> None:
    cfg = policy()
    cfg["public_docs"] = ["docs/stale.md"]
    result = check_public_docs(["docs/new.md"], cfg)
    assert checks(result) == {
        "unreviewed_public_doc",
        "stale_public_doc_allowlist",
    }


@pytest.mark.parametrize(
    "entries",
    [
        ["/docs/absolute.md"],
        ["docs/../private.md"],
        ["docs/*.md"],
        ["docs\\guide.md"],
        ["README.md"],
        ["docs/z.md", "docs/a.md"],
        ["docs/a.md", "docs/a.md"],
        [""],
        [1],
    ],
)
def test_load_policy_rejects_invalid_public_docs_entries(tmp_path, entries) -> None:
    cfg = policy()
    cfg["public_docs"] = entries
    target = tmp_path / "policy.json"
    target.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="public_docs"):
        load_policy(target)


def test_load_policy_rejects_existing_public_docs_directory(tmp_path) -> None:
    root = tmp_path
    architecture = root / "docs" / "architecture"
    architecture.mkdir(parents=True)
    (root / "docs" / "guide").mkdir()
    cfg = policy()
    cfg["public_docs"] = ["docs/guide"]
    target = architecture / "repository-layout.json"
    target.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        load_policy(target)


def test_tracked_docs_match_current_policy_allowlist() -> None:
    cfg = load_policy()
    tracked = git_paths(ROOT, candidates=False)
    assert check_public_docs(tracked, cfg) == []


def test_force_added_ignored_doc_remains_a_release_candidate(tmp_path) -> None:
    _init_git(tmp_path)
    (tmp_path / ".gitignore").write_text("docs/forced.md\n", encoding="utf-8")
    target = tmp_path / "docs" / "forced.md"
    target.parent.mkdir()
    target.write_text("private draft", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "-f", "docs/forced.md"],
        cwd=tmp_path,
        check=True,
    )
    candidates = git_paths(tmp_path, candidates=True)
    assert "unreviewed_public_doc" in checks(check_public_docs(candidates, policy()))


def test_test_fixtures_do_not_raise_on_planted_secret_examples(tmp_path) -> None:
    _init_git(tmp_path)
    target = tmp_path / "tests" / "example.py"
    target.parent.mkdir()
    target.write_text("value='ghp_abcdefghijklmnopqrstuvwxyz'", encoding="utf-8")
    result = check_release(tmp_path, ["tests/example.py"], policy(), [])
    assert "secret" not in checks(result)
