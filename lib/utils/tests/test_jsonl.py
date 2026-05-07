"""Unit tests for :mod:`lib.utils.jsonl`."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lib.utils.jsonl import append_jsonl, read_jsonl, write_jsonl


def test_write_jsonl_creates_parent(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "sub" / "dir" / "out.jsonl"
    n = write_jsonl(out, [{"a": 1}])
    assert n == 1
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() == json.dumps(
        {"a": 1}, ensure_ascii=False, sort_keys=True
    )


def test_write_jsonl_atomic_preserves_existing_on_iter_error(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    out.write_text("PREEXISTING\n", encoding="utf-8")

    def boom():
        yield {"a": 1}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_jsonl(out, boom())
    # Original artifact unchanged because rename never happened.
    assert out.read_text(encoding="utf-8") == "PREEXISTING\n"


def test_write_jsonl_default_flags_byte_stable(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    write_jsonl(out, [{"b": 1, "a": 2}, {"x": "café"}])
    raw = out.read_bytes()
    expected = (
        json.dumps({"b": 1, "a": 2}, ensure_ascii=False, sort_keys=True)
        + "\n"
        + json.dumps({"x": "café"}, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    assert raw == expected
    # Spot-check non-ASCII is literal, not escaped.
    assert "café".encode("utf-8") in raw


def test_write_jsonl_default_str_for_datetime(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    rec = {"when": datetime(2026, 5, 7, tzinfo=timezone.utc)}
    with pytest.raises(TypeError):
        write_jsonl(out, [rec])
    # default=str succeeds
    n = write_jsonl(out, [rec], default=str)
    assert n == 1
    text = out.read_text(encoding="utf-8")
    assert "2026-05-07" in text


def test_write_jsonl_returns_count(tmp_path: Path) -> None:
    out = tmp_path / "x.jsonl"
    n = write_jsonl(out, [{"i": i} for i in range(5)])
    assert n == 5


def test_read_jsonl_skips_blanks(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text(
        '\n{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8"
    )
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_raises_on_decode_error(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text('{"a": 1}\nNOT-JSON\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(p)


def test_read_jsonl_skip_invalid_continues(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text('{"a": 1}\nNOT-JSON\n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl(p, skip_invalid=True) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_missing_file_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "absent.jsonl"
    assert read_jsonl(p) == []


def test_append_jsonl_flush_default() -> None:
    buf = io.StringIO()
    flushed = []
    orig_flush = buf.flush

    def tracking_flush() -> None:
        flushed.append(True)
        orig_flush()

    buf.flush = tracking_flush  # type: ignore[method-assign]
    append_jsonl(buf, {"a": 1})
    assert flushed == [True]
    assert buf.getvalue() == json.dumps({"a": 1}, sort_keys=True) + "\n"


def test_append_jsonl_no_flush() -> None:
    buf = io.StringIO()
    flushed = []
    orig_flush = buf.flush

    def tracking_flush() -> None:
        flushed.append(True)
        orig_flush()

    buf.flush = tracking_flush  # type: ignore[method-assign]
    append_jsonl(buf, {"a": 1}, flush=False)
    assert flushed == []


def test_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "round.jsonl"
    records = [{"i": i, "v": f"item-{i}"} for i in range(100)]
    n = write_jsonl(out, records)
    assert n == 100
    assert read_jsonl(out) == records


def test_write_jsonl_non_atomic_streams_directly(tmp_path: Path) -> None:
    out = tmp_path / "stream.jsonl"
    n = write_jsonl(out, [{"a": 1}, {"b": 2}], atomic=False)
    assert n == 2
    # No leftover .tmp file
    assert not (tmp_path / "stream.jsonl.tmp").exists()
    assert read_jsonl(out) == [{"a": 1}, {"b": 2}]
