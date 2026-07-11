"""Regression: SEMANTIK_TESSERACT_CONFIG / _USER_WORDS thread a Tesseract config
fragment (+ a domain-vocabulary user-words file) into ``image_to_data``.

Both resolvers are parse-with-fallback and read at call time; when BOTH are
unset the ``_tesseract_page_blocks`` call shape is byte-identical to the historic
``image_to_data(image, output_type=…)`` (no ``config=`` kwarg), which keeps the
existing ``test_ocr_render_scale.py`` mock green. The render + pytesseract
boundary is mocked — no pypdfium2 render, no real OCR, CPU-only, no model load.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from semantik_structure import extract_shared
from semantik_structure.extract_shared import (
    resolve_tesseract_config,
    resolve_tesseract_user_words,
)


# ---- resolver: SEMANTIK_TESSERACT_CONFIG --------------------------------


def test_config_default_when_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_TESSERACT_CONFIG", raising=False)
    assert resolve_tesseract_config() == ""


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
def test_config_blank_falls_back_to_empty(monkeypatch, blank):
    monkeypatch.setenv("SEMANTIK_TESSERACT_CONFIG", blank)
    assert resolve_tesseract_config() == ""


def test_config_stripped(monkeypatch):
    monkeypatch.setenv("SEMANTIK_TESSERACT_CONFIG", "  --oem 1  ")
    assert resolve_tesseract_config() == "--oem 1"


def test_config_newlines_collapsed(monkeypatch):
    monkeypatch.setenv(
        "SEMANTIK_TESSERACT_CONFIG",
        "--oem 1\n-c preserve_interword_spaces=1",
    )
    assert resolve_tesseract_config() == "--oem 1 -c preserve_interword_spaces=1"


# ---- resolver: SEMANTIK_TESSERACT_USER_WORDS ----------------------------


def test_user_words_default_when_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_TESSERACT_USER_WORDS", raising=False)
    assert resolve_tesseract_user_words() is None


def test_user_words_existing_file(monkeypatch, tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("radical\ncoefficient\n", encoding="utf-8")
    monkeypatch.setenv("SEMANTIK_TESSERACT_USER_WORDS", str(f))
    got = resolve_tesseract_user_words()
    assert got == f


def test_user_words_missing_file_returns_none_and_warns_once(
    monkeypatch, tmp_path, caplog
):
    missing = tmp_path / "nope.txt"
    monkeypatch.setenv("SEMANTIK_TESSERACT_USER_WORDS", str(missing))
    # Ensure the module-level dedup set does not carry state from another test.
    monkeypatch.setattr(extract_shared, "_USER_WORDS_WARNED", set())
    with caplog.at_level("WARNING"):
        assert resolve_tesseract_user_words() is None
        assert resolve_tesseract_user_words() is None  # second call: no re-warn
    warnings = [r for r in caplog.records if "SEMANTIK_TESSERACT_USER_WORDS" in r.message]
    assert len(warnings) == 1


# ---- call-site threading in _tesseract_page_blocks ----------------------


def _capturing_pytesseract(captured: dict):
    """A pytesseract stand-in that records the kwargs passed to image_to_data."""
    mod = types.ModuleType("pytesseract")
    mod.Output = types.SimpleNamespace(DICT="dict")

    def image_to_data(image, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "text": [], "block_num": [], "par_num": [], "line_num": [],
            "height": [], "conf": [], "left": [], "top": [], "width": [],
        }

    mod.image_to_data = image_to_data
    return mod


def _run(monkeypatch) -> dict:
    captured: dict = {}

    class _Img:
        width = 100
        height = 100

    monkeypatch.setattr(
        extract_shared, "_pypdfium2_render_page_to_image",
        lambda pdf_path, page_num, scale=2.0: _Img(),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", _capturing_pytesseract(captured))
    out = extract_shared._tesseract_page_blocks(Path("/nonexistent.pdf"), 1)
    assert out["text_blocks"] == []
    return captured["kwargs"]


def test_call_site_no_config_kwarg_when_flags_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_TESSERACT_CONFIG", raising=False)
    monkeypatch.delenv("SEMANTIK_TESSERACT_USER_WORDS", raising=False)
    kwargs = _run(monkeypatch)
    # Byte-identical historic call shape: only output_type, never config.
    assert "config" not in kwargs
    assert kwargs["output_type"] == "dict"


def test_call_site_config_passed_verbatim(monkeypatch):
    monkeypatch.setenv("SEMANTIK_TESSERACT_CONFIG", "--oem 1 -c preserve_interword_spaces=1")
    monkeypatch.delenv("SEMANTIK_TESSERACT_USER_WORDS", raising=False)
    kwargs = _run(monkeypatch)
    assert kwargs["config"] == "--oem 1 -c preserve_interword_spaces=1"


def test_call_site_user_words_appended(monkeypatch, tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("radical\n", encoding="utf-8")
    monkeypatch.delenv("SEMANTIK_TESSERACT_CONFIG", raising=False)
    monkeypatch.setenv("SEMANTIK_TESSERACT_USER_WORDS", str(f))
    kwargs = _run(monkeypatch)
    assert kwargs["config"] == f"--user-words {f}"


def test_call_site_config_and_user_words_composed(monkeypatch, tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("radical\n", encoding="utf-8")
    monkeypatch.setenv("SEMANTIK_TESSERACT_CONFIG", "--oem 1")
    monkeypatch.setenv("SEMANTIK_TESSERACT_USER_WORDS", str(f))
    kwargs = _run(monkeypatch)
    assert kwargs["config"] == f"--oem 1 --user-words {f}"


def test_call_site_user_words_path_with_spaces_shlex_roundtrips(monkeypatch, tmp_path):
    import shlex

    d = tmp_path / "dir with spaces"
    d.mkdir()
    f = d / "words.txt"
    f.write_text("radical\n", encoding="utf-8")
    monkeypatch.delenv("SEMANTIK_TESSERACT_CONFIG", raising=False)
    monkeypatch.setenv("SEMANTIK_TESSERACT_USER_WORDS", str(f))
    kwargs = _run(monkeypatch)
    # pytesseract shlex.split()s the config on POSIX; the quoting must survive.
    parts = shlex.split(kwargs["config"])
    assert parts == ["--user-words", str(f)]


# ---- shipped vocabulary sanity ------------------------------------------


def test_algebra_user_words_file_sane():
    # SemantiK/data/config/algebra_user_words.txt
    vocab_path = (
        Path(__file__).resolve().parents[2] / "data" / "config" / "algebra_user_words.txt"
    )
    assert vocab_path.is_file()
    raw = vocab_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    assert len(lines) >= 80
    # single lowercase token per line, no duplicates
    for ln in lines:
        assert ln == ln.lower()
        assert len(ln.split()) == 1
    assert len(set(lines)) == len(lines)
