"""Tests for the course-slug leak guard (ci/course_slug_guard.py).

Proves the guard (a) catches a planted built-course slug, (b) recognizes
every target family, (c) does NOT fire on the neutral literals and
synthetic placeholders that legitimately replace real slugs, and (d)
honors both allowlist tiers. A guard with no test is not a guard.

Runnable standalone: ``python ci/tests/test_course_slug_guard.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Import the guard module whether run under pytest (with the repo on the
# path) or as a standalone script.
_CI_DIR = Path(__file__).resolve().parent.parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import course_slug_guard as guard  # noqa: E402


# --- detection: target families must all be caught ----------------------

def test_catches_every_target_family():
    # Representative live-run slugs across the families named in the purge.
    targets = [
        "alg-glm-02",
        "alg-glm-01",
        "alg-vendor-rag-01",
        "alg-scan",
        "alg-scan-01",
        "alg-mc3-x",
        "nvidiarag-101",
        "nvidiarag-145",
        "openstax-alg2e",
        "keet-bio",
        "PROJ-alg-glm-02-20260716",
        "TTC_algglm02_20260716_abc",
    ]
    for slug in targets:
        hits = guard.find_course_slug_hits(f"course = '{slug}'")
        assert hits, f"guard failed to catch target slug: {slug}"


def test_catches_future_index_within_family():
    # A brand-new index/descriptor within a known family must be caught
    # without editing the patterns — the point of scheme regexes.
    for slug in ("alg-glm-99", "nvidiarag-202", "alg-vendor-html-07"):
        assert guard.find_course_slug_hits(slug), f"future member missed: {slug}"


# --- non-detection: neutral literals & placeholders must NOT fire -------

def test_ignores_neutral_test_literals():
    # These are the deliberate generic replacements the purge introduces;
    # firing on them would destroy the fix.
    neutral = [
        "alg-demo",
        "alg-legacy",
        "alg-101",
        "alg-quiz-course",
        "alg-textbook-like",
        "alg-from-structure",
        "sample-alg-9",
        "unit-test-alg-xyz",
        "course-a",
        "test-course",
        "algebra",  # bare word, no hyphen
    ]
    for tok in neutral:
        assert not guard.find_course_slug_hits(tok), f"false positive on: {tok}"


def test_ignores_synthetic_doc_placeholders():
    for tok in ("PHYS_101", "BIO_201", "CHEM_101", "WF-20260420-abc12345"):
        assert not guard.find_course_slug_hits(tok), f"false positive on: {tok}"


def test_ttc_format_template_is_not_a_leak():
    # The id-minting format template is legitimate contract code; only a
    # concrete run id (embedded timestamp) is a leak.
    for legit in (
        'run_id = f"TTC_{course_name}_{timestamp}"',
        "TTC_{course}_{ts}",
        "mints TTC_... run ids",
    ):
        assert not guard.find_course_slug_hits(legit), f"template flagged: {legit}"
    assert guard.find_course_slug_hits("TTC_PHYS101_20260420_9f3c")


# --- repository scan + allowlist ----------------------------------------

def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


def test_scan_repository_flags_planted_violation(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        'DEFAULT_COURSE = "alg-glm-02"  # planted leak\n', encoding="utf-8"
    )
    (tmp_path / "clean.py").write_text('COURSE = "course-a"\n', encoding="utf-8")
    _init_repo(tmp_path)

    violations = guard.scan_repository(tmp_path)
    paths = {v.path for v in violations}
    assert "pkg/mod.py" in paths
    assert "clean.py" not in paths
    v = next(v for v in violations if v.path == "pkg/mod.py")
    assert v.token == "alg-glm-02"
    assert v.line == 1


def test_inline_marker_allows_a_line(tmp_path):
    (tmp_path / "doc.md").write_text(
        "Forbidden example: alg-glm-02  slug-guard: allow\n", encoding="utf-8"
    )
    _init_repo(tmp_path)
    assert guard.scan_repository(tmp_path) == []


def test_allowlist_file_whole_file_and_token(tmp_path):
    (tmp_path / "ci").mkdir()
    (tmp_path / "legacy.py").write_text('x = "nvidiarag-101"\n', encoding="utf-8")
    (tmp_path / "one.py").write_text(
        'a = "alg-glm-02"\nb = "alg-scan-03"\n', encoding="utf-8"
    )
    # whole-file allow for legacy.py; single-token allow for one.py.
    (tmp_path / "ci" / "course_slug_allowlist.txt").write_text(
        "legacy.py\n" "one.py\talg-glm-02\n", encoding="utf-8"
    )
    _init_repo(tmp_path)

    violations = guard.scan_repository(tmp_path)
    tokens = {(v.path, v.token) for v in violations}
    assert ("legacy.py", "nvidiarag-101") not in tokens  # whole-file allowed
    assert ("one.py", "alg-glm-02") not in tokens         # token allowed
    assert ("one.py", "alg-scan-03") in tokens            # still caught


def test_binary_file_is_skipped(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00alg-glm-02\x00")
    _init_repo(tmp_path)
    assert guard.scan_repository(tmp_path) == []


def test_untracked_file_is_not_scanned(tmp_path):
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _init_repo(tmp_path)
    # Written AFTER `git add`: untracked, so git ls-files omits it.
    (tmp_path / "untracked.py").write_text('c = "alg-glm-02"\n', encoding="utf-8")
    assert guard.scan_repository(tmp_path) == []


def test_current_tree_has_no_violations():
    # The live repo must be clean (the guard itself and its test are
    # allowlisted by path). If this fails, it is reporting a REAL leak.
    violations = guard.scan_repository(guard.PROJECT_ROOT)
    assert violations == [], "\n".join(v.format() for v in violations)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
