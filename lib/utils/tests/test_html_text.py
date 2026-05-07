"""Unit tests for :mod:`lib.utils.html_text`."""
from __future__ import annotations

from lib.utils.html_text import TextExtractor, strip_html_to_text


def test_strip_simple_html() -> None:
    out = strip_html_to_text("<p>Hello <b>world</b>!</p>")
    # Whitespace-stripped data fragments joined with single space.
    # Fragments: "Hello ", "world", "!" -> "Hello  world !" then .strip()
    # The handle_data filters out pure-whitespace events; "Hello " has
    # non-whitespace -> kept literally, joined with " ".
    assert "Hello" in out
    assert "world" in out


def test_strip_decodes_entities() -> None:
    assert strip_html_to_text("Foo &amp; bar") == "Foo & bar"


def test_strip_skips_pure_whitespace() -> None:
    assert strip_html_to_text("<p>  </p><p>x</p>") == "x"


def test_strip_empty_input() -> None:
    assert strip_html_to_text("") == ""


def test_strip_none_input() -> None:
    assert strip_html_to_text(None) == ""  # type: ignore[arg-type]


def test_strip_malformed_does_not_raise() -> None:
    # Even malformed markup shouldn't propagate exceptions.
    out = strip_html_to_text("<<<")
    assert isinstance(out, str)


def test_extractor_class_directly_usable() -> None:
    ex = TextExtractor()
    ex.feed("<p>alpha</p><p>beta</p>")
    ex.close()
    text = ex.text()
    assert "alpha" in text
    assert "beta" in text
