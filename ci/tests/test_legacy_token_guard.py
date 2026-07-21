"""Tests for the legacy 'dart' token guard (ci/legacy_token_guard.py).

Proves the guard (a) catches the legacy token in every casing and seam
form (snake / kebab / colon / camelCase), (b) does NOT fire on innocent
words that merely contain the letters or abut them (``standard``,
``darter``, ``dartmouth``, ``darts``), and (c) honors both allowlist
tiers plus the inline marker. A guard with no test is not a guard.

Runnable standalone: ``python ci/tests/test_legacy_token_guard.py``.
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

import legacy_token_guard as guard  # noqa: E402


# --- detection: every legacy-token spelling must be caught ---------------

def test_catches_bare_token_casings():
    for tok in ("dart", "DART", "Dart"):
        assert guard.find_legacy_token_hits(f"the {tok} engine"), tok


def test_catches_identifier_and_provenance_forms():
    forms = [
        "dart_chunks/chunks.jsonl",
        "stage_dart_outputs",
        "run_dart_chunking",
        "data-dart-block-id",
        ".dart-section {",
        "dart:slug#block_01",
        "DART_THETA_DEVICE",
        "DART_VISION_PROVIDER",
        "parse_dart_ref(ref)",
    ]
    for f in forms:
        assert guard.find_legacy_token_hits(f), f"missed legacy form: {f}"


def test_catches_camelcase_seams():
    for tok in ("stageDartOutputs", "dartChunks", "DartMarkersValidator", "runDartChunking"):
        assert guard.find_legacy_token_hits(tok), f"missed camelCase seam: {tok}"


def test_reports_exact_matched_spelling():
    assert guard.find_legacy_token_hits("a DART b") == ["DART"]
    assert guard.find_legacy_token_hits("a Dart b") == ["Dart"]
    assert guard.find_legacy_token_hits("a dart b") == ["dart"]


# --- non-detection: innocent words must NOT fire -------------------------

def test_ignores_words_without_the_substring():
    # None of these even contain the four letters ``dart`` in sequence.
    for w in ("standard", "standardize", "restart", "started", "chart", "part", "upstart"):
        assert not guard.find_legacy_token_hits(w), f"false positive on: {w}"


def test_ignores_trailing_lowercase_neighbors():
    # 'dart' glued to a following lowercase letter is a different word.
    for w in ("darter", "dartmouth", "darts", "darted", "dartboard", "dartlike"):
        assert not guard.find_legacy_token_hits(w), f"false positive on: {w}"


def test_ignores_leading_lowercase_glue():
    # 'dart' glued to a preceding lowercase letter (no camelCase seam).
    for w in ("foodart", "xdart", "abcdart"):
        assert not guard.find_legacy_token_hits(w), f"false positive on: {w}"


def test_semantik_replacement_vocab_is_clean():
    for w in (
        "semantik_chunks",
        "stage_semantik_outputs",
        "data-semantik-block-id",
        "SemantiKMarkersValidator",
        "semantik:slug#block_01",
    ):
        assert not guard.find_legacy_token_hits(w), f"false positive on: {w}"


# --- repository scan + allowlist ----------------------------------------

def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


def test_scan_repository_flags_planted_violation(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        'TOOL = "run_dart_chunking"  # planted straggler\n', encoding="utf-8"
    )
    (tmp_path / "clean.py").write_text(
        'TOOL = "run_semantik_chunking"\n', encoding="utf-8"
    )
    _init_repo(tmp_path)

    violations = guard.scan_repository(tmp_path)
    paths = {v.path for v in violations}
    assert "pkg/mod.py" in paths
    assert "clean.py" not in paths
    v = next(v for v in violations if v.path == "pkg/mod.py")
    assert v.token == "dart"
    assert v.line == 1


def test_inline_marker_allows_a_line(tmp_path):
    (tmp_path / "doc.md").write_text(
        "Forbidden example: dart_chunks/  legacy-token: allow\n", encoding="utf-8"
    )
    _init_repo(tmp_path)
    assert guard.scan_repository(tmp_path) == []


def test_allowlist_file_whole_file_and_token(tmp_path):
    (tmp_path / "ci").mkdir()
    (tmp_path / "legacy.py").write_text(
        'PAT = r"(?:dart|semantik):"\n', encoding="utf-8"
    )
    (tmp_path / "one.py").write_text(
        'a = "dart-output"\nb = "DART_THETA_DEVICE"\n', encoding="utf-8"
    )
    # whole-file allow for legacy.py; single-token allow for one.py.
    (tmp_path / "ci" / "legacy_token_allowlist.txt").write_text(
        "legacy.py\n" "one.py\tdart\n", encoding="utf-8"
    )
    _init_repo(tmp_path)

    violations = guard.scan_repository(tmp_path)
    tokens = {(v.path, v.token) for v in violations}
    assert not any(v.path == "legacy.py" for v in violations)  # whole-file allowed
    assert ("one.py", "dart") not in tokens                    # token allowed
    assert ("one.py", "DART") in tokens                        # still caught


def test_binary_file_is_skipped(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00dart_chunks\x00")
    _init_repo(tmp_path)
    assert guard.scan_repository(tmp_path) == []


def test_untracked_file_is_not_scanned(tmp_path):
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _init_repo(tmp_path)
    # Written AFTER `git add`: untracked, so git ls-files omits it.
    (tmp_path / "untracked.py").write_text('c = "dart_chunks"\n', encoding="utf-8")
    assert guard.scan_repository(tmp_path) == []


def test_current_tree_has_no_violations():
    # The live repo must be clean: every residual 'dart' token is either
    # scrubbed or covered by ci/legacy_token_allowlist.txt. If this fails,
    # it is reporting a REAL leak (or a missing allowlist justification).
    violations = guard.scan_repository(guard.PROJECT_ROOT)
    assert violations == [], "\n".join(v.format() for v in violations)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
