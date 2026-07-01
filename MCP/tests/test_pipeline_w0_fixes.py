"""Regression tests for the W0.3 / W0.4 / W0.6 / W0.7 pipeline_tools fixes."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import MCP.tools.pipeline_tools as pt
from lib.ontology.slugs import libv2_course_slug


# --- W0.3: _resolve_libv2_root honors ED4ALL_LIBV2_ROOT and ED4ALL_HOME -----

@pytest.mark.unit
def test_resolve_libv2_root_explicit_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path / "env"))
    monkeypatch.setenv("ED4ALL_HOME", str(tmp_path / "home"))
    explicit = tmp_path / "explicit"
    assert pt._resolve_libv2_root(str(explicit)) == explicit


@pytest.mark.unit
def test_resolve_libv2_root_env_libv2_root(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    target = tmp_path / "envroot"
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(target))
    assert pt._resolve_libv2_root() == target
    assert pt._resolve_libv2_root("") == target  # empty explicit falls through


@pytest.mark.unit
def test_resolve_libv2_root_honors_ed4all_home(tmp_path, monkeypatch):
    # The pre-fix resolver hardcoded _PROJECT_ROOT/LibV2 and ignored ED4ALL_HOME.
    monkeypatch.delenv("ED4ALL_LIBV2_ROOT", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("ED4ALL_HOME", str(home))
    assert pt._resolve_libv2_root() == home / "libv2"


# --- W0.4: archive slug routes through the canonical archive-slug helper -----

@pytest.mark.unit
def test_archive_slug_helper_byte_compatible_with_old_inline():
    # The old inline transform: name.lower().replace("_","-").replace(" ","-").
    for name in ["PHYS_101", "My Course", "BIO-201", "Intro_To_AI"]:
        old = name.lower().replace("_", "-").replace(" ", "-")
        assert libv2_course_slug(name) == old, name


# --- W0.6: chunk course_code carries no whitespace --------------------------

@pytest.mark.unit
def test_normalize_chunk_course_code_strips_whitespace():
    assert pt._normalize_chunk_course_code("My Course") == "MY_COURSE"
    assert pt._normalize_chunk_course_code("My  Course") == "MY_COURSE"
    assert pt._normalize_chunk_course_code("My - Course") == "MY_COURSE"
    assert pt._normalize_chunk_course_code("My-Course") == "MY_COURSE"
    assert pt._normalize_chunk_course_code("  ") == "UNKNOWN"
    assert pt._normalize_chunk_course_code("PHYS_101") == "PHYS_101"


@pytest.mark.unit
def test_space_bearing_name_yields_no_whitespace_chunk_ids():
    # Simulate the chunk-id prefix the chunker builds from course_code.
    code = pt._normalize_chunk_course_code("My Course")
    prefix = f"{code.lower()}_chunk_"
    chunk_ids = [f"{prefix}{i:05d}" for i in range(3)]
    import re

    assert all(not re.search(r"\s", cid) for cid in chunk_ids)


@pytest.mark.unit
def test_assert_no_whitespace_chunk_ids_detects_offenders():
    clean = [{"id": "my_course_chunk_00000"}, {"id": "my_course_chunk_00001"}]
    assert pt._assert_no_whitespace_chunk_ids(clean) == []
    dirty = [{"id": "my course_chunk_00000"}, {"chunk_id": "ok_chunk_1"}]
    assert pt._assert_no_whitespace_chunk_ids(dirty) == ["my course_chunk_00000"]


# --- W0.7: atomic manifest write --------------------------------------------

@pytest.mark.unit
def test_atomic_write_text_writes_and_leaves_no_tmp(tmp_path):
    target = tmp_path / "nested" / "manifest.json"
    pt._atomic_write_text(target, json.dumps({"a": 1}))
    assert json.loads(target.read_text()) == {"a": 1}
    # No leftover temp files in the directory.
    assert [p.name for p in target.parent.iterdir()] == ["manifest.json"]


@pytest.mark.unit
def test_atomic_write_text_overwrites_existing(tmp_path):
    target = tmp_path / "manifest.json"
    pt._atomic_write_text(target, json.dumps({"v": 1}))
    pt._atomic_write_text(target, json.dumps({"v": 2}))
    assert json.loads(target.read_text()) == {"v": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]
