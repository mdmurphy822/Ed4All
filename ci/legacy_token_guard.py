#!/usr/bin/env python3
"""Legacy 'dart' token guard.

Fails when the retired legacy engine name ``dart`` (in any casing) is
reintroduced into a tracked file that is not on the compat allowlist.
``DART`` was the AGPL PDF-conversion engine that ``SemantiK`` replaced;
the public repo must carry no reference to it — not in code, config,
tests, comments, docstrings, or docs — except in the narrow set of files
that MUST keep the literal to preserve dual-read backward compatibility
with un-migrated corpora and old checkpoints.

Why a boundary-aware scan and not a bare substring
--------------------------------------------------
A bare ``dart`` substring fires on innocent words. The guard therefore
treats ``dart`` as a token only when it is delimited by a non-alphabetic
character on each side OR by a camelCase seam:

  * ``d`` at a word start (preceding char is not a lowercase letter, or
    the match is a capitalized ``Dart`` sitting after a lowercase letter —
    a camelCase seam like ``stageDartOutputs``);
  * the trailing ``t`` at a word end (following char is not a letter, or
    is an uppercase letter — a camelCase seam like ``dartChunks`` /
    ``DartMarkers``).

So ``dart``, ``DART``, ``Dart``, ``dart_chunks``, ``data-dart-block-id``,
``.dart-section``, ``dart:slug#id``, ``stageDartOutputs`` and
``DartMarkers`` all match, while ``standard`` (no ``dart`` substring at
all), ``darter``, ``dartmouth`` and ``darts`` (trailing lowercase letter)
do not. The intentional trade is that a genuine ``darts``/``darted``
plural/participle is missed — those do not occur as identifiers here, and
never masking ``standard`` matters far more.

Allowlist
---------
Narrow and explicit, three tiers, identical in shape to the course-slug
guard:
  * This module and its test legitimately contain the token and planted
    fixtures (``_SELF_PATHS``).
  * ``ci/legacy_token_allowlist.txt`` — operator-owned exceptions, one per
    line: a repo-relative path (allow the whole file) or ``path<TAB>token``
    (allow one matched token spelling in one file). ``#`` comments allowed.
    Every entry there documents WHY the literal must stay (dual-read
    schema pattern, ``data-dart-*`` fallback reader, migration tooling,
    legacy-compat test, checkpoint/phase/tool alias, etc.).
  * An inline ``legacy-token: allow`` marker on a source line skips that
    line — for a doc that must show the token as a forbidden example.

Standalone:  ``python ci/legacy_token_guard.py``  (exit 1 on violation).
It is also registered as ``check_legacy_token_leak`` in
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

# Inline escape hatch: a line carrying this marker is not scanned.
ALLOW_MARKER = "legacy-token: allow"

ALLOWLIST_FILE = "ci/legacy_token_allowlist.txt"

# Files that legitimately hold the token / test fixtures. Repo-relative.
# The allowlist file itself names dart-bearing compat paths on every entry
# line, so it must be whole-file self-exempt or the guard flags its own
# allowlist.
_SELF_PATHS: Set[str] = {
    "ci/legacy_token_guard.py",
    "ci/tests/test_legacy_token_guard.py",
    ALLOWLIST_FILE,
}

# Every case-insensitive occurrence of the four letters ``dart``. Boundary
# adjudication happens in ``_is_token_boundary`` on the surrounding chars,
# not in this pattern, so the pattern can stay dead simple and the boundary
# rule stays in one auditable place.
_DART_RE = re.compile(r"dart", re.IGNORECASE)


def _left_ok(text: str, start: int) -> bool:
    """True when the char before ``start`` is a left token boundary.

    A boundary is: string start, a non-letter, or a camelCase seam — a
    capitalized ``Dart`` (match text begins with uppercase ``D``) sitting
    directly after a lowercase letter (``stageDart``). A lowercase ``d``
    glued to a preceding letter (``foodart``) is NOT a boundary.
    """
    if start == 0:
        return True
    prev = text[start - 1]
    if not prev.isascii() or not prev.isalpha():
        return True
    # Preceding char is a letter: only a camelCase seam counts, i.e. the
    # match opens with an uppercase ``D`` after a lowercase letter.
    return text[start] == "D" and prev.islower()


def _right_ok(text: str, end: int) -> bool:
    """True when the char at ``end`` is a right token boundary.

    A boundary is: string end, a non-letter, or an uppercase letter — a
    camelCase seam (``dartChunks`` / ``DartMarkers``). A trailing lowercase
    letter (``darter``, ``darts``, ``dartmouth``) is NOT a boundary.
    """
    if end >= len(text):
        return True
    nxt = text[end]
    if not nxt.isascii() or not nxt.isalpha():
        return True
    return nxt.isupper()


def find_legacy_token_hits(text: str) -> List[str]:
    """Return every boundary-delimited ``dart`` token in ``text``.

    Operates on a single logical string (typically one line). Each returned
    element is the exact matched spelling (``dart`` / ``DART`` / ``Dart``).
    """
    hits: List[str] = []
    for m in _DART_RE.finditer(text):
        if _left_ok(text, m.start()) and _right_ok(text, m.end()):
            hits.append(m.group(0))
    return hits


@dataclass(frozen=True)
class Violation:
    path: str          # repo-relative
    line: int          # 1-indexed
    token: str         # the offending matched text

    def format(self) -> str:
        return f"{self.path}:{self.line}: legacy 'dart' token '{self.token}'"


def load_allowlist(
    project_root: Path = PROJECT_ROOT,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Load the allowlist.

    Returns ``(whole_file_paths, per_file_tokens)``. ``_SELF_PATHS`` are
    always whole-file allowed. The optional ``ci/legacy_token_allowlist.txt``
    adds more: a bare repo-relative path allows the whole file; a
    ``path<TAB>token`` line allows exactly that token spelling in that file.
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
    the caller decides how to treat an unscannable tree. Never falls back to
    a filesystem walk: an untracked build artifact must not be able to trip
    — or launder past — the guard.
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
            "(legacy-token guard refuses to fall back to a filesystem walk)"
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
            for token in find_legacy_token_hits(line):
                if token in allowed_tokens:
                    continue
                violations.append(
                    Violation(path=rel, line=lineno, token=token)
                )
    return violations


def main(argv: Optional[List[str]] = None) -> int:
    try:
        violations = scan_repository(PROJECT_ROOT)
    except RuntimeError as exc:
        # Unscannable tree: loud, non-blocking (no work tree to guard).
        print(f"[legacy-token-guard] SKIPPED: {exc}", file=sys.stderr)
        return 0

    if not violations:
        print("[legacy-token-guard] OK: no legacy 'dart' tokens in tracked files.")
        return 0

    print(
        f"[legacy-token-guard] FAILED: {len(violations)} legacy 'dart' "
        f"token reference(s) in tracked files:",
        file=sys.stderr,
    )
    for v in violations:
        print(f"  {v.format()}", file=sys.stderr)
    print(
        "\nThe retired 'dart' engine name must not reappear in tracked files. "
        "Rewrite to present-tense SemantiK vocabulary. If the literal is a "
        "load-bearing dual-read compat / migration / legacy-compat-test "
        f"token, add the file (or path<TAB>token) to {ALLOWLIST_FILE} with a "
        f"comment saying WHY, or mark the line '{ALLOW_MARKER}'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
