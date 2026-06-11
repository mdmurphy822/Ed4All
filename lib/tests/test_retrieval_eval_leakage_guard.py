"""retrieval-answer-eval-set §2.1 — leakage guard.

The gold/probe sets are operator MEASUREMENT data, not SLM training corpus.
This test asserts (structurally) that no Trainforge SYNTHESIS / INGESTION path
reads ``retrieval_eval/`` or a ``gold_set.json`` — a gold question must never
leak into the trained-model corpus (R6 in the risk register).

A runtime assertion is impractical (synthesis ingests an entire corpus tree via
many entry points), so this is a grep-based structural guard over the Trainforge
synthesis surface: the training-pair synthesizer, the generators package, and
the chunker. Eval-only scripts (``Trainforge/scripts/``) and tests are
explicitly out of scope — they MAY read a gold set (that is their job); the
guard targets the corpus-producing code paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINFORGE = PROJECT_ROOT / "Trainforge"

# The corpus-producing synthesis surface (NOT the eval scripts / tests).
_SYNTHESIS_PATHS = [
    TRAINFORGE / "synthesize_training.py",
    TRAINFORGE / "generators",
    TRAINFORGE / "chunker",
]

# Forbidden read-path tokens. We look for these as string literals / path
# references; a bare mention in a comment is flagged too (cheap + conservative —
# a comment referencing retrieval_eval in synthesis code is itself a smell).
_FORBIDDEN_TOKENS = ("retrieval_eval", "gold_set.json", "gold_set", "refusal_probes")


def _iter_py_files(paths):
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            yield from p.rglob("*.py")


def test_no_synthesis_path_references_retrieval_eval():
    offenders = []
    for py in _iter_py_files(_SYNTHESIS_PATHS):
        try:
            text = py.read_text(encoding="utf-8")
        except OSError:
            continue
        for token in _FORBIDDEN_TOKENS:
            # Allow a benign mention inside a string that's clearly a comment
            # marker; but for the synthesis surface we are strict: any
            # occurrence is an offender (the existing surface has none).
            if token in text:
                # Permit the lone calibration-comment precedent in
                # schema_translation_generator.py that mentions "gold-set" in
                # prose (hyphenated, not the file/dir token).
                lines = [
                    ln.strip()
                    for ln in text.splitlines()
                    if token in ln and not _is_benign_prose(ln, token)
                ]
                if lines:
                    offenders.append((py.relative_to(PROJECT_ROOT), token, lines[:2]))
    assert not offenders, (
        "Trainforge synthesis paths must not read retrieval_eval/ gold sets "
        f"(eval-question leakage R6): {offenders}"
    )


def _is_benign_prose(line: str, token: str) -> bool:
    """A hyphenated 'gold-set' mention in a comment is prose, not a read path.
    The forbidden tokens are the underscore/dotted file + dir names."""
    stripped = line.strip()
    is_comment = stripped.startswith("#")
    # 'gold_set' the substring also matches 'gold-set' is False; this guard only
    # relaxes the bare 'gold_set' token when it appears hyphenated in a comment.
    if token == "gold_set" and is_comment and "gold-set" in line and "gold_set" not in line.replace("gold-set", ""):
        return True
    return False


def test_synthesis_paths_exist():
    """Sanity: the guarded surface actually exists (so the guard isn't a no-op
    that silently passes because the paths moved)."""
    assert (TRAINFORGE / "synthesize_training.py").is_file()
    assert (TRAINFORGE / "generators").is_dir()
    assert (TRAINFORGE / "chunker").is_dir()
