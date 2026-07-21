#!/usr/bin/env python3
"""Course-slug leak guard.

Fails when a slug that names a course *built by this pipeline* is
reintroduced into tracked code, tests, config, or docs. Courses this
system produces are working data, not part of the product, so their
identifiers must not be hardcoded anywhere in the tree — a purge without
a guard only resets the clock until the next paste.

Why scheme regexes and not a literal blocklist
-----------------------------------------------
A blocklist of today's slugs is worthless: the next course has a new
name. But the opposite extreme — "flag any lowercase-hyphen token" —
is worse: it fires on the legitimate *neutral* literals that stand in
for real slugs in tests (``alg-demo``, ``alg-legacy``, ``alg-101``,
``sample-alg-9``, ``course-a``, ``unit-test-alg-xyz``), destroying the
very replacements the purge introduced.

What actually separates a built-course slug from generic prose is its
*naming scheme*: a subject prefix joined to a real ingestion-lane /
source descriptor and a zero-padded run index. The lane descriptors are
concrete pipeline facts — the GLM-OCR lane (``glm``), the vendor-HTML
lane (``vendor``), the scanned-corpus lane (``scan``), ``mc3``, the
NVIDIA RAG demo line (``nvidiarag-NN``), OpenStax-derived algebra
(``openstax-alg*``), ``keet-*`` — plus the two id envelopes the runner
mints around a slug: Courseforge export ids (``PROJ-<slug>-...``) and
textbook run ids (``TTC_<course>_<timestamp>``).

A second, non-slug class of leak is guarded too: concrete
*publisher / corpus provenance* descriptors of the specific source
corpora ingested — a publisher edition short-code (``ea2e`` /
``elementary-algebra-2e``), the CNX distribution host (``cnx.org``), the
scan-edition slug (``openstax-scan*``), and author surnames off the
OpenStax algebra title (``marecek``, ``anthony-smith``, ``honeycutt``).
Each is specific enough to have no legitimate neutral use. The bare token
``openstax`` is deliberately NOT patterned: it legitimately names a
lexicon / taxonomy / source-registry key in ~45 tracked files, so a
blanket rule would fire on product code, not a leak.

Each pattern below matches the SCHEME, not one literal, so a *new* index
or descriptor within a family is caught (``alg-glm-04``, ``nvidiarag-202``,
``alg-vendor-html-02``) without editing this file. A brand-new *family*
(say ``bio-...``) is deliberately NOT guessed at — there is no structural
signal distinguishing it from ordinary kebab-case, and guessing would
reintroduce the false-positive problem above. Adding a family is a
one-line pattern edit here, made when that family is coined.

The ``TTC_`` rule matches only a CONCRETE run id — ``TTC_`` followed by an
embedded 6+ digit timestamp. It must never match the id-minting *format*
template ``TTC_{course_name}_{timestamp}`` (an f-string / docstring with
no literal timestamp), which is legitimate contract code, not a leak.

Allowlist
---------
Narrow and explicit, three tiers:
  * This module and its test legitimately contain the patterns and
    planted fixtures (``_SELF_PATHS``).
  * ``ci/course_slug_allowlist.txt`` — operator-owned exceptions, one per
    line: a repo-relative path (allow the whole file) or ``path\ttoken``
    (allow one token in one file). ``#`` comments allowed.
  * An inline ``slug-guard: allow`` marker on a source line skips that
    line — for a doc that must show a real slug as a forbidden example.

Synthetic doc placeholders (``PHYS_101``, ``WF-20260420-abc12345``,
``course-a``) need no allowlisting: the scheme patterns do not match them
by construction.

Standalone:  ``python ci/course_slug_guard.py``  (exit 1 on violation).
It is also registered as ``check_course_slug_leak`` in
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
ALLOW_MARKER = "slug-guard: allow"

ALLOWLIST_FILE = "ci/course_slug_allowlist.txt"

# Files that legitimately hold the patterns / test fixtures. Repo-relative.
_SELF_PATHS: Set[str] = {
    "ci/course_slug_guard.py",
    "ci/tests/test_course_slug_guard.py",
}

# (family, pattern). Each matches a naming SCHEME so future members of the
# family are caught without a code edit. See the module docstring for why a
# broad kebab-case rule is rejected. Case-sensitive: the ``PROJ-``/``TTC_``
# envelopes are uppercase, the slug families lowercase.
_PATTERN_SOURCES: List[Tuple[str, str]] = [
    # Algebra corpora keyed by real ingestion lane + index.
    ("alg-glm", r"\balg-glm-\d+"),
    ("alg-vendor", r"\balg-vendor(?:-[a-z0-9]+)*"),
    ("alg-scan", r"\balg-scan(?:-[a-z0-9]+)*"),
    ("alg-mc3", r"\balg-mc3(?:-[a-z0-9]+)*"),
    # NVIDIA RAG demo line + numeric index.
    ("nvidiarag", r"\bnvidiarag-\d+"),
    # OpenStax-derived algebra corpora.
    ("openstax-alg", r"\bopenstax-alg[a-z0-9-]*"),
    # keet-* family.
    ("keet", r"\bkeet-[a-z0-9]+"),
    # Courseforge export id envelope around an ``alg-*`` slug.
    ("PROJ-alg", r"\bPROJ-alg[a-z0-9-]*"),
    # Concrete textbook run id: TTC_<course>_<6+ digit timestamp>. The
    # embedded timestamp is what distinguishes a real id from the legit
    # format template ``TTC_{course_name}_{timestamp}``.
    ("TTC-runid", r"\bTTC_[A-Za-z0-9_]*\d{6,}[A-Za-z0-9_]*"),
    # --- Publisher / corpus-provenance leaks -----------------------------
    # These are not lane-indexed slugs but concrete identifiers of the
    # specific source corpora this pipeline ingested — a publisher edition
    # short-code (``ea2e`` = Elementary Algebra 2e), the CNX distribution
    # host, author surnames off the OpenStax algebra title, and the two
    # scan/edition slug forms. A build corpus is working data, not product,
    # so its provenance descriptors must not be hardcoded either. Unlike
    # ``openstax`` (a bare token that legitimately names a lexicon /
    # taxonomy / source-registry key in ~45 files — too broad to guard
    # without allowlisting most of them, so deliberately NOT patterned),
    # each of these is specific enough to have no legitimate neutral use.
    #
    # ``ea2e`` is word-boundary + case-insensitive (it appears as EA2e /
    # EA2E in citations); the rest are case-sensitive lowercase literals.
    ("ea2e", r"(?i:\bea2e\b)"),
    ("cnx-org", r"\bcnx\.org\b"),
    ("marecek", r"\bmarecek\b"),
    ("anthony-smith", r"\banthony-smith\b"),
    ("honeycutt", r"\bhoneycutt\b"),
    ("openstax-scan", r"\bopenstax-scan(?:-[a-z0-9]+)*"),
    ("elementary-algebra-2e", r"\belementary-algebra-2e\b"),
]

COURSE_SLUG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (name, re.compile(src)) for name, src in _PATTERN_SOURCES
]


@dataclass(frozen=True)
class Violation:
    path: str          # repo-relative
    line: int          # 1-indexed
    token: str         # the offending matched text
    family: str        # which pattern family matched

    def format(self) -> str:
        return f"{self.path}:{self.line}: built-course slug '{self.token}' (family: {self.family})"


def find_course_slug_hits(text: str) -> List[Tuple[str, str]]:
    """Return ``[(family, token), ...]`` for every slug match in ``text``.

    Operates on a single logical string (typically one line). Order is by
    pattern-family declaration then match position.
    """
    hits: List[Tuple[str, str]] = []
    for family, pattern in COURSE_SLUG_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((family, m.group(0)))
    return hits


def load_allowlist(
    project_root: Path = PROJECT_ROOT,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Load the allowlist.

    Returns ``(whole_file_paths, per_file_tokens)``. ``_SELF_PATHS`` are
    always whole-file allowed. The optional ``ci/course_slug_allowlist.txt``
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
            "(course-slug guard refuses to fall back to a filesystem walk)"
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
            for family, token in find_course_slug_hits(line):
                if token in allowed_tokens:
                    continue
                violations.append(
                    Violation(path=rel, line=lineno, token=token, family=family)
                )
    return violations


def main(argv: Optional[List[str]] = None) -> int:
    tracked_root = PROJECT_ROOT
    try:
        violations = scan_repository(tracked_root)
    except RuntimeError as exc:
        # Unscannable tree: loud, non-blocking (no work tree to guard).
        print(f"[course-slug-guard] SKIPPED: {exc}", file=sys.stderr)
        return 0

    if not violations:
        print("[course-slug-guard] OK: no built-course slugs in tracked files.")
        return 0

    print(
        f"[course-slug-guard] FAILED: {len(violations)} built-course slug "
        f"reference(s) in tracked files:",
        file=sys.stderr,
    )
    for v in violations:
        print(f"  {v.format()}", file=sys.stderr)
    print(
        "\nBuilt-course slugs must not be hardcoded. Rewrite to the structural "
        "property that made the example worth citing, or use a neutral test "
        "literal (course-a, test-course). Legitimate hits: add to "
        f"{ALLOWLIST_FILE} or mark the line '{ALLOW_MARKER}'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
