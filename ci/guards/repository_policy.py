#!/usr/bin/env python3
"""Recursive repository-policy and source-release guard.

The policy classifies tracked paths only.  The release scan additionally sees
untracked, non-ignored candidates, but never traverses ignored input/output
trees.  Operator-private words may be supplied through
``ED4ALL_PRIVATE_TOKEN_FILE``; that file is never part of the policy.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "docs/architecture/repository-layout.json"
PRIVATE_TOKEN_ENV = "ED4ALL_PRIVATE_TOKEN_FILE"

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "API secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
}
PRIVACY_SHAPES = {
    "concrete textbook run id": re.compile(r"(?i)(?:^|[^a-z0-9])TTC_[a-z0-9_]*\d{6,}"),
    "course export id": re.compile(r"(?i)(?:^|[^a-z0-9])PROJ[-_][a-z0-9_-]+"),
    "absolute project checkout": re.compile(r"(?i)(?:/home/[^/]+|~)/Projects/[^/\s]+"),
    "agent project state": re.compile(r"(?i)\.claude/projects/[-a-z0-9._]+"),
}


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    message: str

    def format(self) -> str:
        return f"[{self.check}] {self.path}: {self.message}"


def load_policy(path: Path = POLICY_PATH) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "root_files", "roots", "overrides", "allowed_children", "source_roles",
        "sentinel_names", "max_source_bytes", "forbidden_generated_segments",
        "public_docs",
    }
    if not required.issubset(value) or not isinstance(value["roots"], dict):
        raise ValueError("repository layout policy has an invalid structure")
    roles = set(value["allowed_children"])
    used = set(value["roots"].values()) | {row["role"] for row in value["overrides"]}
    unknown = used - roles
    if unknown:
        raise ValueError(f"policy uses undeclared roles: {sorted(unknown)}")
    public_docs = value["public_docs"]
    if not isinstance(public_docs, list):
        raise ValueError("policy public_docs must be a list")
    if any(not isinstance(entry, str) or not entry for entry in public_docs):
        raise ValueError("policy public_docs entries must be non-empty strings")
    if public_docs != sorted(public_docs):
        raise ValueError("policy public_docs must be sorted")
    if len(public_docs) != len(set(public_docs)):
        raise ValueError("policy public_docs entries must be unique")
    for entry in public_docs:
        candidate = PurePosixPath(entry)
        if (
            entry != candidate.as_posix()
            or candidate.is_absolute()
            or not candidate.parts
            or candidate.parts[0] != "docs"
            or any(part in {".", ".."} for part in candidate.parts)
            or any(char in entry for char in "*?[\\")
            or entry.endswith("/")
        ):
            raise ValueError(
                "policy public_docs entries must be normalized, exact, "
                f"repo-relative files under docs/: {entry!r}"
            )
        if path.parent.name == "architecture" and path.parent.parent.name == "docs":
            root = path.parents[2]
            if (root / entry).is_dir():
                raise ValueError(
                    f"policy public_docs entry names a directory: {entry!r}"
                )
    return value


def _matches(path: str, pattern: str) -> bool:
    # pathlib's match gives ** useful path-segment semantics; fnmatch covers
    # an exact directory followed by arbitrary descendants.
    return PurePosixPath(path).match(pattern) or fnmatch.fnmatchcase(path, pattern)


def classify(path: str, policy: Mapping) -> tuple[str | None, list[str]]:
    parts = PurePosixPath(path).parts
    if not parts:
        return None, []
    if len(parts) == 1 and path in policy["root_files"]:
        return "root-metadata", []
    root_role = policy["roots"].get(parts[0])
    if root_role is None:
        return None, []
    matches = [row for row in policy["overrides"] if _matches(path, row["pattern"])]
    if not matches:
        return root_role, []
    specificity = max(len(row["pattern"].replace("*", "")) for row in matches)
    winners = [row for row in matches if len(row["pattern"].replace("*", "")) == specificity]
    roles = sorted({row["role"] for row in winners})
    return (roles[0], []) if len(roles) == 1 else (None, roles)


def directory_prefixes(paths: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for raw in paths:
        parts = PurePosixPath(raw).parts
        for index in range(1, len(parts)):
            result.add("/".join(parts[:index]))
    return result


def check_layout(paths: Sequence[str], policy: Mapping) -> list[Finding]:
    findings: list[Finding] = []
    dirs = directory_prefixes(paths)
    classified: dict[str, str] = {}
    for path in sorted(dirs | set(paths)):
        role, competing = classify(path, policy)
        if competing:
            findings.append(Finding("ambiguous_role", path, f"equally specific roles: {competing}"))
        elif role is None:
            findings.append(Finding("unclassified", path, "no recursive policy role"))
        else:
            classified[path] = role

    for directory in sorted(dirs):
        if "/" not in directory:
            continue
        parent = directory.rsplit("/", 1)[0]
        parent_role = classified.get(parent)
        child_role = classified.get(directory)
        allowed = policy["allowed_children"].get(parent_role, [])
        if parent_role and child_role and child_role not in allowed:
            findings.append(
                Finding(
                    "illegal_child_role",
                    directory,
                    f"{child_role!r} is not allowed below {parent_role!r}",
                )
            )

    sentinels = set(policy["sentinel_names"])
    forbidden = {part.casefold() for part in policy["forbidden_generated_segments"]}
    for path in paths:
        role = classified.get(path)
        if role == "external-var" and PurePosixPath(path).name not in sentinels:
            findings.append(
                Finding(
                    "external_var_content",
                    path,
                    "tracked external data must be a sentinel only",
                )
            )
        path_dirs = PurePosixPath(path).parts[:-1]
        if role != "external-var" and any(
            part.casefold() in forbidden for part in path_dirs
        ):
            findings.append(
                Finding(
                    "generated_artifact",
                    path,
                    "generated-state directory inside a source role",
                )
            )
    return findings


def git_paths(root: Path, *, candidates: bool) -> list[str]:
    args = ["git", "ls-files", "--cached"]
    if candidates:
        args += ["--others", "--exclude-standard"]
    proc = subprocess.run(args, cwd=root, check=True, text=True, capture_output=True)
    return sorted({row for row in proc.stdout.splitlines() if row and (root / row).is_file()})


def load_private_tokens(root: Path, environ: Mapping[str, str]) -> list[str]:
    raw = environ.get(PRIVATE_TOKEN_ENV)
    if not raw:
        return []
    path = Path(raw).expanduser().resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError:
        relative = None
    if relative is not None:
        proc = subprocess.run(["git", "check-ignore", "-q", "--", relative.as_posix()], cwd=root)
        if proc.returncode != 0:
            raise ValueError(
                f"{PRIVATE_TOKEN_ENV} points to a non-ignored file inside "
                "the repository"
            )
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def check_privacy(path: str, data: bytes, private_tokens: Sequence[str]) -> list[Finding]:
    findings: list[Finding] = []
    text = data.decode("utf-8", errors="ignore") if b"\0" not in data else ""
    # Generic shapes apply to path segments. Text bodies use the stronger
    # operator-local vocabulary, avoiding false positives on neutral contract
    # examples and id-format templates.
    searchable = path
    for label, pattern in PRIVACY_SHAPES.items():
        if pattern.search(searchable):
            findings.append(Finding("privacy_shape", path, label))
    folded = (path + "\n" + text).casefold()
    for token in private_tokens:
        if token.casefold() in folded:
            findings.append(Finding("private_token", path, "matched an operator-private token"))
    return findings


def check_public_docs(paths: Sequence[str], policy: Mapping) -> list[Finding]:
    """Require an exact policy decision for every release-candidate doc."""
    candidates = set(paths)
    allowed = set(policy["public_docs"])
    findings = [
        Finding(
            "unreviewed_public_doc",
            path,
            "docs release candidate is absent from policy public_docs",
        )
        for path in sorted(
            candidate for candidate in candidates if candidate.startswith("docs/")
            and candidate not in allowed
        )
    ]
    findings.extend(
        Finding(
            "stale_public_doc_allowlist",
            path,
            "policy public_docs entry is not a release candidate",
        )
        for path in sorted(allowed - candidates)
    )
    return findings


def check_release(
    root: Path,
    paths: Sequence[str],
    policy: Mapping,
    private_tokens: Sequence[str],
) -> list[Finding]:
    findings = check_layout(paths, policy)
    findings.extend(check_public_docs(paths, policy))
    max_bytes = int(policy["max_source_bytes"])
    source_roles = set(policy["source_roles"])
    for relative in paths:
        path = root / relative
        size = path.stat().st_size
        if size > max_bytes:
            findings.append(Finding("oversized", relative, f"{size} bytes exceeds {max_bytes}"))
            continue
        data = path.read_bytes()
        findings.extend(check_privacy(relative, data, private_tokens))
        role, _ = classify(relative, policy)
        if role not in {"tests", "fixtures"}:
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    findings.append(Finding("secret", relative, f"possible {label}"))
    # Inspect only ancestors of release candidates. This detects nested repos
    # in publishable source without walking ignored runtime/input directories.
    for directory in sorted(directory_prefixes(paths)):
        top = PurePosixPath(directory).parts[0]
        if policy["roots"].get(top) not in source_roles:
            continue
        marker = root / directory / ".git"
        if marker.exists():
            findings.append(
                Finding(
                    "nested_repository",
                    marker.relative_to(root).as_posix(),
                    "nested repository under a source root",
                )
            )
    return findings


def main() -> int:
    try:
        policy = load_policy()
        tokens = load_private_tokens(ROOT, os.environ)
        candidates = git_paths(ROOT, candidates=True)
        findings = check_release(ROOT, candidates, policy, tokens)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[repository-policy] FAILED: {exc}", file=sys.stderr)
        return 1
    if findings:
        print(f"[repository-policy] FAILED: {len(findings)} finding(s)", file=sys.stderr)
        for finding in findings:
            print("  " + finding.format(), file=sys.stderr)
        return 1
    print(
        f"[repository-policy] OK: {len(candidates)} tracked/untracked release "
        "candidates classified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
