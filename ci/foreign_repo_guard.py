#!/usr/bin/env python3
"""Foreign-repository reference guard.

Fails when a tracked file points at a *different* checkout on the
developer's machine — a sibling project's name, an absolute
``/home/<user>/Projects/<Other>`` path, or a Claude-Code project-slug
directory belonging to another repo. This tree must be describable and
buildable on its own terms; a link into a neighbouring checkout is a
dangling reference for every reader who does not share that filesystem,
and it leaks the existence and layout of unrelated work.

What this guard actually caught
-------------------------------
Three markdown links of the form
``../../.claude/projects/-home-<user>-Projects-<Other>/memory/<file>.md``
sat in a tracked architecture doc, citing design constraints out of a
foreign project's agent-memory directory. The targets did not exist —
not in this tree, and not at that absolute path — so the doc's stated
sources were unresolvable. Rewriting them on this repo's own terms is
the fix; this guard is what stops the next paste.

Why path SHAPES and not a list of sibling names
------------------------------------------------
Enumerating today's siblings (``git ls-files`` cannot see them anyway)
is both a privacy leak and useless tomorrow: the next machine has
different neighbours. What is stable is the *shape* of a cross-checkout
reference:

* ``.claude/projects/<slug>`` — the Claude-Code per-project state
  directory. Its slug encodes an absolute path
  (``-home-user-Projects-Name``). This repo keeps no tracked
  ``.claude/projects/`` tree at all, so ANY such reference points
  outside it.
* ``-home-<user>-Projects-<Name>`` — the same slug encoding, standalone.
* ``/home/<user>/Projects/<Name>`` (and ``~/Projects/<Name>``) where
  ``<Name>`` is not this repository's own directory name. An absolute
  developer path is never portable; one naming a *different* project is
  additionally a foreign reference.

Each rule matches the shape, so a sibling coined tomorrow is caught
without editing this file, and no sibling's name is ever written down
here.

Deliberately NOT guarded
------------------------
Generic absolute paths (``/home/user/...`` outside ``Projects/``),
which appear legitimately in operator docs as example invocations, and
this repo's OWN ``/home/<user>/Projects/Ed4All`` path — noisy to purge
and not a cross-repo leak. ``REPO_DIR_NAME`` below is derived from the
checkout, so a rename does not silently disarm the same-repo exemption.

Allowlist
---------
Narrow and explicit, three tiers, mirroring the sibling guards:
  * This module and its test hold the patterns and planted fixtures
    (``_SELF_PATHS``).
  * ``ci/foreign_repo_allowlist.txt`` — operator-owned exceptions, one
    per line: a repo-relative path (allow the whole file) or
    ``path<TAB>token`` (allow one token in one file). ``#`` comments
    allowed.
  * An inline ``foreign-repo-guard: allow`` marker skips that line — for
    a doc that must show such a path as a forbidden example.

Standalone:  ``python ci/foreign_repo_guard.py``  (exit 1 on violation).
It is also registered as ``check_foreign_repo_leak`` in
``ci/integrity_check.py`` so the existing CI integrity job blocks on it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: This checkout's own directory name. Derived, not hardcoded, so a repo
#: rename cannot leave the same-repo exemption pointing at a stale name.
REPO_DIR_NAME = PROJECT_ROOT.name

# Inline escape hatch: a line carrying this marker is not scanned.
ALLOW_MARKER = "foreign-repo-guard: allow"

ALLOWLIST_FILE = "ci/foreign_repo_allowlist.txt"

# Files that legitimately hold the patterns / test fixtures. Repo-relative.
_SELF_PATHS: Set[str] = {
    "ci/foreign_repo_guard.py",
    "ci/tests/test_foreign_repo_guard.py",
    # The allowlist file itself holds per-file token entries by design.
    ALLOWLIST_FILE,
}

# A path segment naming a project directory: letters/digits/._- , but it
# must contain at least one letter so a bare version number is not a
# "project name".
_SEG = r"[A-Za-z0-9][A-Za-z0-9._-]*"

# A Claude-Code project slug is an absolute path with every separator
# rewritten to ``-``, so it LEADS with ``-`` (``-home-user-Projects-Name``).
# ``_SEG`` deliberately rejects a leading ``-``; this segment accepts it,
# and still requires a letter somewhere so ``projects/---`` is not a hit.
_SLUG_SEG = r"-*[A-Za-z0-9][A-Za-z0-9._-]*"

# (family, pattern). Each matches the SHAPE of a cross-checkout
# reference, never a specific sibling's name — see the module docstring.
_PATTERN_SOURCES: List[Tuple[str, str]] = [
    # Claude-Code per-project state dir. This repo tracks no such tree,
    # so any reference is necessarily to another project's.
    ("claude-project-dir", rf"\.claude/projects/{_SLUG_SEG}"),
    # The slug encoding of an absolute path, standalone:
    # ``-home-<user>-Projects-<Name>``.
    ("home-projects-slug", rf"(?i:-home-{_SEG}?-Projects-{_SEG})"),
    # Absolute / tilde developer path under a Projects root. The
    # same-repo case is filtered in :func:`find_foreign_repo_hits`,
    # not here, so the exemption stays visible and testable.
    ("abs-projects-path", rf"(?i:(?:/home/{_SEG}|~)/Projects/{_SEG})"),
]

FOREIGN_REPO_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (name, re.compile(src)) for name, src in _PATTERN_SOURCES
]

#: Matches a trailing ``Projects/<Name>`` so the same-repo exemption can
#: read the project segment out of an ``abs-projects-path`` hit.
_PROJECT_SEGMENT_RE = re.compile(rf"Projects/({_SEG})", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    path: str          # repo-relative
    line: int          # 1-indexed
    token: str         # the offending matched text
    family: str        # which pattern family matched

    def format(self) -> str:
        return (
            f"{self.path}:{self.line}: foreign-repository reference "
            f"'{self.token}' (family: {self.family})"
        )


def _is_own_repo_path(token: str) -> bool:
    """True when an ``abs-projects-path`` hit names THIS checkout.

    ``/home/<user>/Projects/<REPO_DIR_NAME>`` is an absolute developer
    path but not a *cross-repo* leak, so it is out of this guard's scope.
    Comparison is case-insensitive to match the pattern's own casing
    tolerance.
    """
    m = _PROJECT_SEGMENT_RE.search(token)
    if m is None:
        return False
    return m.group(1).lower() == REPO_DIR_NAME.lower()


def find_foreign_repo_hits(text: str) -> List[Tuple[str, str]]:
    """Return ``[(family, token), ...]`` for every foreign reference.

    Operates on a single logical string (typically one line). Order is by
    pattern-family declaration then match position. Hits that resolve to
    this repository's own path are dropped.
    """
    hits: List[Tuple[str, str]] = []
    for family, pattern in FOREIGN_REPO_PATTERNS:
        for m in pattern.finditer(text):
            token = m.group(0)
            if family == "abs-projects-path" and _is_own_repo_path(token):
                continue
            hits.append((family, token))
    return hits


def load_allowlist(
    project_root: Path = PROJECT_ROOT,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Load the allowlist.

    Returns ``(whole_file_paths, per_file_tokens)``. ``_SELF_PATHS`` are
    always whole-file allowed. The optional ``ci/foreign_repo_allowlist.txt``
    adds more: a bare repo-relative path allows the whole file; a
    ``path<TAB>token`` line allows exactly that token in that file.
    """
    whole_file: Set[str] = set(_SELF_PATHS)
    per_file: Dict[str, Set[str]] = {}

    allow_path = project_root / ALLOWLIST_FILE
    if allow_path.exists():
        for raw in allow_path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "\t" in line:
                path, token = line.split("\t", 1)
                per_file.setdefault(path.strip(), set()).add(token.strip())
            else:
                whole_file.add(line)
    return whole_file, per_file


def iter_tracked_files(project_root: Path = PROJECT_ROOT) -> Optional[List[str]]:
    """Repo-relative paths of all tracked files via ``git ls-files``.

    Returns ``None`` when git is unavailable or this is not a work tree —
    the caller decides how to treat an unscannable tree (a warning, not a
    silent pass). Never falls back to a filesystem walk: an untracked
    build artifact must not be able to trip — or launder past — the guard.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
    except (OSError, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    return [p for p in proc.stdout.split("\0") if p]


def _read_lines(path: Path) -> Optional[List[str]]:
    """Return the file's lines, or ``None`` if it is binary/unreadable.

    A NUL byte marks a binary blob (image, archive) that carries no source
    text to scan; decode is lossy-tolerant so an odd encoding never crashes
    the scan.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="ignore").splitlines()


def scan_repository(project_root: Path = PROJECT_ROOT) -> List[Violation]:
    """Scan every tracked text file; return all non-allowlisted violations.

    Raises ``RuntimeError`` when the tree cannot be enumerated via git, so
    an unscannable tree fails loudly rather than passing by default.
    """
    tracked = iter_tracked_files(project_root)
    if tracked is None:
        raise RuntimeError(
            "git ls-files unavailable — cannot enumerate tracked files "
            "(foreign-repo guard refuses to fall back to a filesystem walk)"
        )

    whole_file, per_file = load_allowlist(project_root)
    violations: List[Violation] = []

    for rel in tracked:
        if rel in whole_file:
            continue
        lines = _read_lines(project_root / rel)
        if lines is None:
            continue
        allowed_tokens = per_file.get(rel, set())
        for lineno, line in enumerate(lines, start=1):
            if ALLOW_MARKER in line:
                continue
            for family, token in find_foreign_repo_hits(line):
                if token in allowed_tokens:
                    continue
                violations.append(
                    Violation(path=rel, line=lineno, token=token, family=family)
                )
    return violations


def main(argv: Optional[List[str]] = None) -> int:
    try:
        violations = scan_repository(PROJECT_ROOT)
    except RuntimeError as exc:
        # Unscannable tree: loud, non-blocking (no work tree to guard).
        print(f"[foreign-repo-guard] SKIPPED: {exc}", file=sys.stderr)
        return 0

    if not violations:
        print("[foreign-repo-guard] OK: no foreign-repository references.")
        return 0

    print(
        f"[foreign-repo-guard] FAILED: {len(violations)} foreign-repository "
        f"reference(s) in tracked files:",
        file=sys.stderr,
    )
    for v in violations:
        print(f"  {v.format()}", file=sys.stderr)
    print(
        "\nThis tree must stand on its own terms. Replace the cross-checkout "
        "path with the constraint or decision it was citing, stated here, or "
        "link a doc inside this repository. Legitimate hits: add to "
        f"{ALLOWLIST_FILE} or mark the line '{ALLOW_MARKER}'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
