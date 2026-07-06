"""Defect F — ``ed4all objectives restructure`` CLI tests.

Hermetic: the DART chunkset loader and the embedding-client builder are
monkeypatched to synthetic stubs so no LibV2 course / real embedding model is
touched. Synthetic fixtures only (no course slugs / publisher names / paths).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from cli.main import cli


class _Resolved:
    kind = "st"


class _StubEmbed:
    resolved = _Resolved()

    def encode_batch(self, texts):
        out = []
        for t in texts:
            key = " ".join(str(t).lower().split()[:4])
            v = np.zeros(24, dtype=float)
            v[abs(hash(key)) % 24] = 1.0
            out.append(v)
        return np.asarray(out, dtype=float)


def _chunk(cid, mid, title):
    return {"id": cid, "source": {"module_id": mid, "module_title": title, "position_in_module": 0}}


@pytest.fixture
def patched(monkeypatch):
    all_chunks = [_chunk("c1", "mod-a", "Chapter A"), _chunk("c2", "mod-b", "Chapter B")]
    cbi = {c["id"]: c for c in all_chunks}

    import MCP.tools.pipeline_tools as pt
    import lib.embedding.providers as prov

    monkeypatch.setattr(
        pt, "_load_dart_chunkset_for_planning",
        lambda *, course_slug, kwargs: (cbi, all_chunks),
    )
    monkeypatch.setattr(prov, "build_embedding_client", lambda **kw: _StubEmbed())
    return None


def _write_doc(tmp_path: Path) -> Path:
    doc = {
        "course_name": "SYN_101",
        "chapter_objectives": [
            {"chapter": "Week 1", "objectives": [
                {"id": "CO-01", "statement": "Derive the tangent slope at a point",
                 "bloom_level": "apply", "bloom_verb": "derive", "source_chunk_ids": ["c1"]},
                {"id": "CO-02", "statement": "Integrate a rational polynomial expression",
                 "bloom_level": "apply", "bloom_verb": "integrate", "source_chunk_ids": ["c2"]},
            ]},
        ],
    }
    p = tmp_path / "objectives.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_restructure_writes_output_and_report(patched, tmp_path):
    p = _write_doc(tmp_path)
    res = CliRunner().invoke(
        cli, ["objectives", "restructure", str(p), "--course-name", "SYN_101"]
    )
    assert res.exit_code == 0, res.output
    out = tmp_path / "objectives.restructured.json"
    report = tmp_path / "restructure_report.json"
    assert out.exists() and report.exists()
    new_doc = json.loads(out.read_text())
    assert new_doc["mint_method"] == "restructured_objectives"
    assert [t["id"] for t in new_doc["terminal_objectives"]] == ["TO-01", "TO-02"]
    rep = json.loads(report.read_text())
    assert rep["num_tos"] == 2


def test_restructure_honors_output_flag(patched, tmp_path):
    p = _write_doc(tmp_path)
    custom = tmp_path / "custom_out.json"
    res = CliRunner().invoke(
        cli, ["objectives", "restructure", str(p), "--course-name", "SYN_101",
              "--output", str(custom), "--weeks", "2"],
    )
    assert res.exit_code == 0, res.output
    assert custom.exists()
    assert json.loads(custom.read_text())["duration_weeks"] == 2


def test_restructure_rejects_libv2_archive_shape(patched, tmp_path):
    p = tmp_path / "archive.json"
    p.write_text(json.dumps({
        "terminal_outcomes": [{"id": "to-01"}],
        "component_objectives": [{"id": "co-01"}],
    }), encoding="utf-8")
    res = CliRunner().invoke(
        cli, ["objectives", "restructure", str(p), "--course-name", "SYN_101"]
    )
    assert res.exit_code != 0
    assert "reuse-objectives" in res.output.lower()
