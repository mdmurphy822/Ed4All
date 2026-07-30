#!/usr/bin/env python3
"""Repository layout guard.

Enforces the schema in ``docs/architecture/repo-organization.md`` § 2/§ 6:
the top level is a closed set, ``lib/`` stops growing new flat modules,
``docs/`` sits inside its four-bucket taxonomy with no single-file dirs,
and ``scripts/`` gets no new loose top-level files. Without a guard the
schema is prose — the next PR lands a 25th top-level dir "just this once"
and the diagnosis in § 1 repeats itself.

Four checks, all against ``git ls-files`` (tracked files only, so
gitignored VAR-zone content — ``runtime/ plans/ inputs/`` etc. —
can never trip this guard; only ``.gitkeep`` sentinels are tracked there):

1. **Top-level closed** — every top-level entry (the first path segment of
   every tracked file, and every tracked root-level file itself) must
   appear in ``ci/layout_allowlist.txt`` as ``dir:<name>`` / ``file:<name>``.
2. **lib/ flat-module ratchet** — no tracked ``lib/*.py`` at depth exactly 1
   (excluding ``lib/__init__.py``) beyond the frozen ``libflat:<name>``
   snapshot. New cross-cutting code goes in a ``lib/<topic>/`` subpackage.
3. **docs/ taxonomy** — tracked ``docs/**`` files at depth ≥2 (i.e.
   ``docs/<subdir>/...``) must sit under one of the four buckets
   ``{architecture, operations, validation, reference}``. This set is
   SCHEMA, hard-coded here (not in the allowlist) — the four-bucket
   taxonomy is a design decision documented in
   ``docs/architecture/repo-organization.md`` § 4, not an operator
   exception. Additionally no ``docs/<subdir>/`` may hold exactly one
   tracked file (the "no single-file dirs" rule); a legitimate exception
   is an explicit ``docsingle:<name>`` allowlist line, not a silent pass.
4. **scripts/ snapshot** — no new loose file directly in ``scripts/``
   (depth exactly 1) beyond the frozen ``script:<name>`` snapshot.
   Subdirs (``archive/ codegen/ integration/ tests/`` today; ``ops/`` /
   ``harness/`` land in Phase 2 per § 3) are unrestricted containers —
   their contents are never checked here.

Ratchet doctrine
-----------------
Every list in ``ci/layout_allowlist.txt`` may only SHRINK over time.
Adding a line is a real exception to the schema and must be justified in
the same PR that adds it — the allowlist diff *is* the design review (see
§ 2 of the spec doc). This guard does not enforce shrink-only-ness itself
(that would need history, not a snapshot); it is a PR-review contract, the
same one ``ci/course_slug_guard.py`` and ``ci/legacy_token_guard.py``
already carry for their own allowlists.

Design note: every check is a pure function over an injected
``tracked: List[str]`` + a parsed ``Allowlist`` — no git or filesystem
access — so tests can plant synthetic violations without a git fixture.
``scan_repository`` / ``main`` are the only functions that touch git or
disk.

Standalone:  ``python ci/layout_guard.py``  (exit 1 on violation).
It is also registered as ``check_layout`` in ``ci/integrity_check.py`` so
the existing CI integrity job blocks on it.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWLIST_FILE = "ci/layout_allowlist.txt"

# The docs/ four-bucket taxonomy is SCHEMA (repo-organization.md § 4), not
# an operator exception — it is hard-coded here, never allowlist-driven.
ALLOWED_DOCS_SUBDIRS: frozenset = frozenset(
    {"architecture", "operations", "validation", "reference"}
)

_KNOWN_PREFIXES = ("dir:", "file:", "libflat:", "docsingle:", "script:")


@dataclass(frozen=True)
class Violation:
    check: str   # which of the four checks fired
    path: str    # repo-relative path (or bare top-level name) implicated
    message: str

    def format(self) -> str:
        return f"[{self.check}] {self.path}: {self.message}"


@dataclass(frozen=True)
class Allowlist:
    dirs: Set[str] = field(default_factory=set)
    files: Set[str] = field(default_factory=set)
    libflat: Set[str] = field(default_factory=set)
    docsingle: Set[str] = field(default_factory=set)
    scripts: Set[str] = field(default_factory=set)


def parse_allowlist(text: str, source_name: str = ALLOWLIST_FILE) -> Allowlist:
    """Parse the allowlist text into its five typed sets.

    Raises ``ValueError`` on any non-blank, non-comment line that doesn't
    carry one of the five known prefixes — an unrecognized entry fails
    loudly rather than being silently ignored (which would make the
    allowlist not do what its author thought it did).
    """
    dirs: Set[str] = set()
    files: Set[str] = set()
    libflat: Set[str] = set()
    docsingle: Set[str] = set()
    scripts: Set[str] = set()
    by_prefix = {
        "dir:": dirs,
        "file:": files,
        "libflat:": libflat,
        "docsingle:": docsingle,
        "script:": scripts,
    }

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for prefix, target in by_prefix.items():
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if not value:
                    raise ValueError(
                        f"{source_name}:{lineno}: empty value after '{prefix}': {raw!r}"
                    )
                target.add(value)
                break
        else:
            raise ValueError(
                f"{source_name}:{lineno}: unrecognized allowlist entry "
                f"(expected one of {_KNOWN_PREFIXES}): {raw!r}"
            )

    return Allowlist(
        dirs=dirs, files=files, libflat=libflat, docsingle=docsingle, scripts=scripts
    )


def load_allowlist(project_root: Path = PROJECT_ROOT) -> Allowlist:
    """Load and parse ``ci/layout_allowlist.txt``.

    Raises ``FileNotFoundError`` when the allowlist is missing — the guard
    has nothing to check against and that is a repo-config bug, not a pass.
    """
    allow_path = project_root / ALLOWLIST_FILE
    if not allow_path.exists():
        raise FileNotFoundError(f"{ALLOWLIST_FILE} not found under {project_root}")
    return parse_allowlist(
        allow_path.read_text(encoding="utf-8"), source_name=ALLOWLIST_FILE
    )


# --------------------------------------------------------------------------
# Pure check functions — operate on an injected file list, no I/O.
# --------------------------------------------------------------------------

def check_top_level_closed(
    tracked: List[str], allowed_dirs: Set[str], allowed_files: Set[str]
) -> List[Violation]:
    """Every top-level entry must be an allowlisted dir or root file."""
    top_is_dir: Dict[str, bool] = {}
    for rel in tracked:
        parts = rel.split("/", 1)
        top = parts[0]
        # A path can't simultaneously be both a file and a dir in git, so
        # the first observation for a given top segment is authoritative.
        top_is_dir.setdefault(top, len(parts) > 1)

    violations: List[Violation] = []
    for top in sorted(top_is_dir):
        is_dir = top_is_dir[top]
        if is_dir and top not in allowed_dirs:
            violations.append(
                Violation(
                    check="top_level_closed",
                    path=top,
                    message=(
                        f"untracked top-level dir '{top}' not in {ALLOWLIST_FILE} "
                        f"— add 'dir:{top}' with justification, per "
                        f"docs/architecture/repo-organization.md § 2"
                    ),
                )
            )
        elif not is_dir and top not in allowed_files:
            violations.append(
                Violation(
                    check="top_level_closed",
                    path=top,
                    message=(
                        f"untracked root-level file '{top}' not in {ALLOWLIST_FILE} "
                        f"— add 'file:{top}' with justification, per "
                        f"docs/architecture/repo-organization.md § 2"
                    ),
                )
            )
    return violations


def check_lib_flat_ratchet(
    tracked: List[str], allowed_libflat: Set[str]
) -> List[Violation]:
    """No new flat lib/*.py beyond the frozen snapshot."""
    violations: List[Violation] = []
    for rel in tracked:
        parts = rel.split("/")
        if len(parts) != 2 or parts[0] != "lib":
            continue
        name = parts[1]
        if not name.endswith(".py") or name == "__init__.py":
            continue
        if name not in allowed_libflat:
            violations.append(
                Violation(
                    check="lib_flat_ratchet",
                    path=rel,
                    message=(
                        f"new flat lib/*.py module '{name}' — new cross-cutting "
                        f"code belongs in a lib/<topic>/ subpackage (ratchet "
                        f"frozen at repo-organization.md § 6; add "
                        f"'libflat:{name}' only with justification)"
                    ),
                )
            )
    return violations


def check_docs_taxonomy(
    tracked: List[str], allowed_docsingle: Set[str]
) -> List[Violation]:
    """docs/** at depth >=2 must sit under the four-bucket taxonomy, and no
    docs/<subdir>/ may hold exactly one tracked file."""
    violations: List[Violation] = []
    subdir_counts: Dict[str, int] = {}
    subdir_example: Dict[str, str] = {}

    for rel in tracked:
        parts = rel.split("/")
        if len(parts) < 3 or parts[0] != "docs":
            continue
        subdir = parts[1]
        subdir_counts[subdir] = subdir_counts.get(subdir, 0) + 1
        subdir_example.setdefault(subdir, rel)
        if subdir not in ALLOWED_DOCS_SUBDIRS:
            violations.append(
                Violation(
                    check="docs_taxonomy",
                    path=rel,
                    message=(
                        f"docs/{subdir}/ is not one of the allowed taxonomy "
                        f"buckets {sorted(ALLOWED_DOCS_SUBDIRS)} "
                        f"(docs/architecture/repo-organization.md § 4)"
                    ),
                )
            )

    for subdir in sorted(subdir_counts):
        if subdir_counts[subdir] == 1 and subdir not in allowed_docsingle:
            violations.append(
                Violation(
                    check="docs_taxonomy",
                    path=subdir_example[subdir],
                    message=(
                        f"docs/{subdir}/ has exactly one tracked file "
                        f"('no single-file dirs' rule) — merge it into a "
                        f"sibling bucket or add 'docsingle:{subdir}' with "
                        f"justification"
                    ),
                )
            )
    return violations


def check_scripts_snapshot(
    tracked: List[str], allowed_scripts: Set[str]
) -> List[Violation]:
    """No new loose file directly in scripts/ beyond the frozen snapshot."""
    violations: List[Violation] = []
    for rel in tracked:
        parts = rel.split("/")
        if len(parts) != 2 or parts[0] != "scripts":
            continue
        name = parts[1]
        if name not in allowed_scripts:
            violations.append(
                Violation(
                    check="scripts_snapshot",
                    path=rel,
                    message=(
                        f"new loose scripts/{name} — place it in a scripts/ "
                        f"subdir per docs/architecture/repo-organization.md § 3 "
                        f"(ratchet frozen; add 'script:{name}' only with "
                        f"justification)"
                    ),
                )
            )
    return violations


def check_layout(tracked: List[str], allowlist: Allowlist) -> List[Violation]:
    """Run all four checks and return the combined violation list."""
    violations: List[Violation] = []
    violations.extend(
        check_top_level_closed(tracked, allowlist.dirs, allowlist.files)
    )
    violations.extend(check_lib_flat_ratchet(tracked, allowlist.libflat))
    violations.extend(check_docs_taxonomy(tracked, allowlist.docsingle))
    violations.extend(check_scripts_snapshot(tracked, allowlist.scripts))
    return violations


# --------------------------------------------------------------------------
# git / filesystem glue
# --------------------------------------------------------------------------

def iter_tracked_files(project_root: Path = PROJECT_ROOT) -> Optional[List[str]]:
    """Repo-relative paths of all tracked files via ``git ls-files``.

    Returns ``None`` when git is unavailable or this is not a work tree —
    the caller decides how to treat an unscannable tree (a loud SKIP, not
    a silent pass). Never falls back to a filesystem walk: an untracked
    build artifact (or gitignored VAR-zone content) must not be able to
    trip this guard.
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


def scan_repository(project_root: Path = PROJECT_ROOT) -> List[Violation]:
    """Scan the tracked tree; return all layout violations.

    Raises ``RuntimeError`` when the tree cannot be enumerated via git, so
    an unscannable tree fails loudly rather than passing by default.
    """
    tracked = iter_tracked_files(project_root)
    if tracked is None:
        raise RuntimeError(
            "git ls-files unavailable — cannot enumerate tracked files "
            "(layout guard refuses to fall back to a filesystem walk)"
        )
    allowlist = load_allowlist(project_root)
    return check_layout(tracked, allowlist)


def main(argv: Optional[List[str]] = None) -> int:
    tracked_root = PROJECT_ROOT
    try:
        violations = scan_repository(tracked_root)
    except RuntimeError as exc:
        # Unscannable tree: loud, non-blocking (no work tree to guard).
        print(f"[layout-guard] SKIPPED: {exc}", file=sys.stderr)
        return 0

    if not violations:
        print("[layout-guard] OK: tracked tree matches the repo layout schema.")
        return 0

    print(
        f"[layout-guard] FAILED: {len(violations)} layout violation(s) in "
        f"tracked files:",
        file=sys.stderr,
    )
    for v in violations:
        print(f"  {v.format()}", file=sys.stderr)
    print(
        "\nSee docs/architecture/repo-organization.md for the placement "
        f"rules. A deliberate exception goes in {ALLOWLIST_FILE} — justify "
        "the new line in the same PR (allowlists only ever shrink).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
