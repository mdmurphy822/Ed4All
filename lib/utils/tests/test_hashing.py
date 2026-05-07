"""Unit tests for :mod:`lib.utils.hashing`."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from lib.utils.hashing import sha256_file, sha256_text


def test_sha256_file_known_vector(tmp_path: Path) -> None:
    p = tmp_path / "hello.txt"
    p.write_bytes(b"hello\n")
    expected = hashlib.sha256(b"hello\n").hexdigest()
    assert sha256_file(p) == expected


def test_sha256_file_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_file_chunk_size_neutral(tmp_path: Path) -> None:
    p = tmp_path / "rand.bin"
    payload = os.urandom(1024)
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(p, chunk=1) == expected
    assert sha256_file(p, chunk=4096) == expected
    assert sha256_file(p, chunk=65536) == expected


def test_sha256_file_str_path_accepted(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    assert sha256_file(str(p)) == expected
    assert sha256_file(p) == expected


def test_sha256_file_missing_raises_oserror(tmp_path: Path) -> None:
    missing = tmp_path / "nope.bin"
    with pytest.raises(OSError):
        sha256_file(missing)


def test_sha256_text_known_vector() -> None:
    expected = hashlib.sha256(b"hello").hexdigest()
    assert sha256_text("hello") == expected


def test_sha256_text_unicode_roundtrip() -> None:
    expected = hashlib.sha256("café".encode("utf-8")).hexdigest()
    assert sha256_text("café") == expected
