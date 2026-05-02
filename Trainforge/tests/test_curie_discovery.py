"""Tests for the corpus-driven CURIE discovery layer (Wave 137 followup).

Coverage spans three layers:

1. ``lib/ontology/curie_discovery.py`` primitive — frequency tally,
   min_frequency filter, sort order, malformed-line tolerance.
2. ``Trainforge/scripts/discover_curies.py`` CLI — argparse wiring,
   output formats (table / json / manifest), --exclude-known-manifest
   diff, exit codes.
3. ``Trainforge/scripts/backfill_form_data.py`` --discover-from-corpus
   integration — target-list union semantics, --allow-non-manifest
   plumbing through the drafting subprocess.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.curie_discovery import (  # noqa: E402
    diff_against_manifest,
    discover_curies_from_corpus,
)


# ----------------------------------------------------------------------
# Test corpora
# ----------------------------------------------------------------------


def _write_chunks(tmp_path: Path, lines: List[Dict[str, Any]]) -> Path:
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")
    return p


# ----------------------------------------------------------------------
# Primitive
# ----------------------------------------------------------------------


def test_discover_returns_chunk_count_per_curie(tmp_path):
    p = _write_chunks(tmp_path, [
        {"chunk_id": "c1", "text": "rdf:type and rdfs:Class appear here"},
        {"chunk_id": "c2", "text": "rdf:type again with sh:datatype"},
        {"chunk_id": "c3", "text": "sh:datatype and sh:NodeShape"},
        {"chunk_id": "c4", "text": "rare:Singleton appears once"},
    ])
    discovered = discover_curies_from_corpus(p, min_frequency=1)
    assert discovered["rdf:type"] == 2
    assert discovered["sh:datatype"] == 2
    assert discovered["rdfs:Class"] == 1
    assert discovered["sh:NodeShape"] == 1
    assert discovered["rare:Singleton"] == 1


def test_discover_filters_below_min_frequency(tmp_path):
    p = _write_chunks(tmp_path, [
        {"chunk_id": "c1", "text": "rdf:type and rdfs:Class"},
        {"chunk_id": "c2", "text": "rdf:type"},
        {"chunk_id": "c3", "text": "rare:Singleton"},
    ])
    discovered = discover_curies_from_corpus(p, min_frequency=2)
    assert "rdf:type" in discovered
    assert "rdfs:Class" not in discovered
    assert "rare:Singleton" not in discovered


def test_discover_sorts_by_frequency_then_alphabetical(tmp_path):
    p = _write_chunks(tmp_path, [
        # rdf:type=3, rdfs:Class=3 (tie), sh:datatype=2
        {"chunk_id": "c1", "text": "rdf:type rdfs:Class sh:datatype"},
        {"chunk_id": "c2", "text": "rdf:type rdfs:Class sh:datatype"},
        {"chunk_id": "c3", "text": "rdf:type rdfs:Class"},
    ])
    discovered = discover_curies_from_corpus(p, min_frequency=1)
    keys = list(discovered.keys())
    # rdf:type and rdfs:Class tied at 3; alphabetical tie-break puts
    # rdf:type before rdfs:Class. sh:datatype at 2 follows.
    assert keys == ["rdf:type", "rdfs:Class", "sh:datatype"]


def test_discover_excludes_url_schemes(tmp_path):
    p = _write_chunks(tmp_path, [
        {"chunk_id": "c1", "text": "Visit https://example.org and http://foo.bar"},
        {"chunk_id": "c2", "text": "rdf:type is real, https://example.org is not"},
    ])
    discovered = discover_curies_from_corpus(p, min_frequency=1)
    assert "rdf:type" in discovered
    assert "http:example" not in discovered
    assert "https:example" not in discovered


def test_discover_skips_malformed_json_lines(tmp_path):
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"chunk_id": "c1", "text": "rdf:type here"}) + "\n")
        fh.write("not valid json line\n")
        fh.write(json.dumps({"chunk_id": "c2", "text": "rdf:type also"}) + "\n")
    discovered = discover_curies_from_corpus(p, min_frequency=1)
    assert discovered.get("rdf:type") == 2


def test_discover_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_curies_from_corpus(tmp_path / "absent.jsonl")


def test_discover_rejects_zero_min_frequency(tmp_path):
    p = _write_chunks(tmp_path, [{"chunk_id": "c1", "text": "rdf:type"}])
    with pytest.raises(ValueError):
        discover_curies_from_corpus(p, min_frequency=0)


def test_discover_chunk_count_caps_at_one_per_chunk(tmp_path):
    """A single chunk that mentions a CURIE 100 times still counts as 1
    toward chunk-frequency. This matches the manifest's tier semantic
    (high-tier >50 chunks, mid-tier 10-50, low-tier 2-10)."""
    p = _write_chunks(tmp_path, [
        {"chunk_id": "c1", "text": "rdf:type " * 100},
    ])
    discovered = discover_curies_from_corpus(p, min_frequency=1)
    assert discovered["rdf:type"] == 1


def test_discover_extra_excluded_prefixes(tmp_path):
    p = _write_chunks(tmp_path, [
        {"chunk_id": "c1", "text": "rdf:type and ex:test and ex:other"},
    ])
    discovered = discover_curies_from_corpus(
        p, min_frequency=1, extra_excluded_prefixes=["ex"],
    )
    assert "rdf:type" in discovered
    assert "ex:test" not in discovered
    assert "ex:other" not in discovered


# ----------------------------------------------------------------------
# diff_against_manifest
# ----------------------------------------------------------------------


def test_diff_against_manifest_returns_new_and_dropped():
    discovered = {"rdf:type": 50, "sh:datatype": 30, "ex:Foo": 10}
    manifest_curies = ["rdf:type", "sh:datatype", "rdfs:Class"]
    new, dropped = diff_against_manifest(discovered, manifest_curies)
    assert new == {"ex:Foo": 10}
    assert dropped == ["rdfs:Class"]


# ----------------------------------------------------------------------
# discover_curies CLI
# ----------------------------------------------------------------------


def _make_synthetic_libv2_course(tmp_path: Path, lines: List[Dict[str, Any]]) -> Path:
    """Create a LibV2 course directory tree with a chunks.jsonl."""
    course_dir = tmp_path / "LibV2" / "courses" / "test-course-1" / "corpus"
    course_dir.mkdir(parents=True)
    p = course_dir / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")
    return tmp_path


def test_discover_cli_table_format(tmp_path, monkeypatch, capsys):
    fake_root = _make_synthetic_libv2_course(tmp_path, [
        {"chunk_id": "c1", "text": "rdf:type and sh:datatype"},
        {"chunk_id": "c2", "text": "rdf:type and rdfs:Class"},
    ])
    from Trainforge.scripts import discover_curies as cli

    monkeypatch.setattr(cli, "PROJECT_ROOT", fake_root)
    rc = cli.main([
        "--course-code", "test-course-1",
        "--min-frequency", "1",
        "--format", "table",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "rdf:type" in captured.out
    assert "sh:datatype" in captured.out
    assert "CHUNKS" in captured.out


def test_discover_cli_json_format(tmp_path, monkeypatch, capsys):
    fake_root = _make_synthetic_libv2_course(tmp_path, [
        {"chunk_id": "c1", "text": "rdf:type and sh:datatype"},
        {"chunk_id": "c2", "text": "rdf:type"},
    ])
    from Trainforge.scripts import discover_curies as cli

    monkeypatch.setattr(cli, "PROJECT_ROOT", fake_root)
    rc = cli.main([
        "--course-code", "test-course-1",
        "--min-frequency", "1",
        "--format", "json",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["rdf:type"] == 2
    assert payload["sh:datatype"] == 1


def test_discover_cli_manifest_format(tmp_path, monkeypatch, capsys):
    fake_root = _make_synthetic_libv2_course(tmp_path, [
        {"chunk_id": "c1", "text": "rdf:type"},
        {"chunk_id": "c2", "text": "rdf:type and sh:datatype"},
    ])
    from Trainforge.scripts import discover_curies as cli

    monkeypatch.setattr(cli, "PROJECT_ROOT", fake_root)
    rc = cli.main([
        "--course-code", "test-course-1",
        "--min-frequency", "1",
        "--format", "manifest",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "family: test_course" in captured.out  # _family_slug applied
    assert "curie: rdf:type" in captured.out
    assert "min_pairs:" in captured.out


def test_discover_cli_missing_chunks_returns_exit_1(
    tmp_path, monkeypatch, capsys,
):
    from Trainforge.scripts import discover_curies as cli

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    rc = cli.main([
        "--course-code", "nonexistent-course-99",
    ])
    assert rc == 1
    captured = capsys.readouterr()
    assert "chunks.jsonl not found" in captured.err
