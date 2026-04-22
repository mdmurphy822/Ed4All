"""Worker T — ChunkFilter content_type_label enforcement (REC-VOC-03 Phase 2).

Tests the ChunkFilter.__post_init__ hook that validates content_type_label
against the ChunkType enum when TRAINFORGE_ENFORCE_CONTENT_TYPE=true.

Default behavior (flag off) is unchanged: arbitrary strings accepted.
"""

from __future__ import annotations

import pytest

from LibV2.tools.libv2.retriever import ChunkFilter


ENV_VAR = "TRAINFORGE_ENFORCE_CONTENT_TYPE"


def test_flag_off_accepts_arbitrary_content_type_label(monkeypatch):
    """Default: arbitrary content_type_label values construct silently."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    # No raise expected.
    cf = ChunkFilter(content_type_label="bogus")
    assert cf.content_type_label == "bogus"


def test_flag_off_with_valid_value(monkeypatch):
    """Default: valid ChunkType values also construct silently."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    cf = ChunkFilter(content_type_label="explanation")
    assert cf.content_type_label == "explanation"


def test_flag_on_accepts_valid_chunk_type(monkeypatch):
    """Flag on: ChunkType enum members construct silently."""
    monkeypatch.setenv(ENV_VAR, "true")
    for value in (
        "assessment_item",
        "overview",
        "summary",
        "exercise",
        "explanation",
        "example",
    ):
        cf = ChunkFilter(content_type_label=value)
        assert cf.content_type_label == value


def test_flag_on_rejects_invalid_content_type_label(monkeypatch):
    """Flag on: invalid ChunkType values raise ValueError from __post_init__."""
    monkeypatch.setenv(ENV_VAR, "true")
    with pytest.raises(ValueError, match="bogus"):
        ChunkFilter(content_type_label="bogus")


def test_flag_on_rejects_callout_content_type(monkeypatch):
    """Flag on: CalloutContentType values (which aren't ChunkType) reject."""
    monkeypatch.setenv(ENV_VAR, "true")
    with pytest.raises(ValueError, match="application-note"):
        ChunkFilter(content_type_label="application-note")


def test_flag_on_none_content_type_label_passes(monkeypatch):
    """Flag on: unset field skips enforcement entirely."""
    monkeypatch.setenv(ENV_VAR, "true")
    # None is the default; should construct fine.
    cf = ChunkFilter()
    assert cf.content_type_label is None
    # Also explicit None.
    cf2 = ChunkFilter(content_type_label=None)
    assert cf2.content_type_label is None


def test_flag_on_error_message_mentions_context(monkeypatch):
    """Error message points to the ChunkFilter field for debuggability."""
    monkeypatch.setenv(ENV_VAR, "true")
    with pytest.raises(ValueError, match="ChunkFilter.content_type_label"):
        ChunkFilter(content_type_label="bogus")
