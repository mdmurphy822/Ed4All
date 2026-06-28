"""Regression: rejoin styled drop-cap first letters split off their word body.

pdfplumber's ``extract_words`` starts a new word at a font/style change, so an
OpenStax colored / bold drop-cap first letter (the Bold "S" of "Subtraction")
is emitted as its OWN word ABUTTING the rest ("ubtraction"); the downstream
space-join then fabricates "S ubtraction" / "P arentheses" / "E xponents".

The fix (``extract_shared._merge_split_first_letters``) rejoins a single-letter
leading word into the following word ONLY when (a) it ABUTS the next word
(x-gap <= ~0pt) AND (b) the two carry different font names. Genuine
single-letter words ("I remember", "a way") sit at a real >= ~2pt word space
with the SAME font, so they are never merged.

Geometry below is taken verbatim from the measured OpenStax PEMDAS chart
(Elem-Algebra ch.1, page 35): drop-cap splits sit at gap ~0.01 with a
Bold->Regular font change; genuine single-letter words sit at gap ~2.57 same
font. CPU-only, no model load, no PDF IO.
"""

from __future__ import annotations

from dart_semantic.extract_shared import _merge_split_first_letters


def _w(text, x0, x1, fontname, top=100.0, bottom=108.0, size=8.25):
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom,
        "fontname": fontname,
        "size": size,
    }


def _join(words):
    return " ".join(w["text"] for w in words)


def test_rejoins_uppercase_dropcap_subtraction():
    # 'Step 4. Addition and S ubtraction' — the bold "S" abuts "ubtraction".
    line = [
        _w("and", 189.3, 204.1, "NotoSans"),
        _w("S", 206.3, 210.9, "NotoSans-Bold"),
        _w("ubtraction", 210.9, 251.1, "NotoSans"),  # gap 0.01, font change
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "and Subtraction"


def test_rejoins_pemdas_columns():
    # 'P arentheses' (gap -0.01, STIX Bold -> Regular) -> 'Parentheses'.
    line = [
        _w("P", 230.4, 235.0, "STIXGeneral-Bold"),
        _w("arentheses", 234.99, 277.6, "STIXGeneral-Regular"),
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "Parentheses"


def test_rejoins_lowercase_dropcap():
    # Even a lowercase styled first letter splits: 'm ultiplication'.
    line = [
        _w("m", 230.0, 234.6, "NotoSans-Bold"),
        _w("ultiplication", 234.61, 280.0, "NotoSans"),  # gap 0.01, font change
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "multiplication"


def test_mnemonic_chain_my_dear():
    # 'M y D ear' -> 'My Dear': the 'y'->'D' real space (2.14) is preserved,
    # while 'M'->'y' (0.01) and 'D'->'ear' (0.00) rejoin.
    line = [
        _w("M", 100.0, 104.6, "NotoSans-Bold"),
        _w("y", 104.61, 109.0, "NotoSans"),       # gap 0.01 -> "My"
        _w("D", 111.14, 115.6, "NotoSans-Bold"),  # gap 2.14 from "y" -> real space
        _w("ear", 115.61, 128.0, "NotoSans"),     # gap 0.01 -> "Dear"
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "My Dear"


def test_genuine_article_I_remember_stays_split():
    # 'I remember' — same font, real ~2.57pt word space — must NOT merge.
    line = [
        _w("I", 100.0, 103.0, "NotoSans"),
        _w("remember", 105.57, 140.0, "NotoSans"),  # gap 2.57, same font
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "I remember"


def test_genuine_article_a_way_stays_split():
    line = [
        _w("a", 100.0, 104.6, "NotoSans"),
        _w("way", 107.17, 122.0, "NotoSans"),  # gap 2.57, same font
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "a way"


def test_same_font_abutting_not_merged():
    # Defensive: even an abutting single char is left alone when the font is
    # unchanged (such a pair would not have been split by pdfplumber anyway).
    line = [
        _w("a", 100.0, 104.6, "NotoSans"),
        _w("bc", 104.61, 114.0, "NotoSans"),  # gap 0.01 BUT same font
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "a bc"


def test_column_gap_single_letters_not_merged():
    # Math-variable columns ("a" ... "a" across equation columns) sit at a
    # large column gutter — never merged (and they differ by being far apart).
    line = [
        _w("a", 100.0, 104.6, "NotoSans-Italic"),
        _w("a", 130.0, 134.6, "NotoSans-Italic"),  # gap ~25.4
    ]
    out = _merge_split_first_letters(line)
    assert _join(out) == "a a"


def test_single_word_line_unchanged():
    line = [_w("Parentheses", 100.0, 148.0, "NotoSans")]
    out = _merge_split_first_letters(line)
    assert out == line


def test_merged_word_keeps_body_font_and_union_bbox():
    line = [
        _w("S", 206.3, 210.9, "NotoSans-Bold", top=99.0, bottom=109.0),
        _w("ubtraction", 210.9, 251.1, "NotoSans", top=100.0, bottom=108.0),
    ]
    out = _merge_split_first_letters(line)
    assert len(out) == 1
    m = out[0]
    assert m["text"] == "Subtraction"
    assert m["fontname"] == "NotoSans"          # body font wins (not the bold cap)
    assert m["x0"] == 206.3 and m["x1"] == 251.1  # union bbox
    assert m["top"] == 99.0 and m["bottom"] == 109.0
