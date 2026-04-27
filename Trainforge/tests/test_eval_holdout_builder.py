"""Wave 92 — HoldoutBuilder tests.

Synthetic-graph tests; no LibV2 course required. Asserts:

* Same seed → byte-identical holdout JSON (reproducibility).
* Bloom strata are tracked in the report.
* Holdout fraction lands at the requested pct (within rounding).
* The output payload's ``holdout_graph_hash`` is the SHA-256 of the
  canonicalised payload (without the hash field), so card-side
  consumers can verify the file hasn't been tampered with.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.holdout_builder import HoldoutBuilder, load_holdout_split  # noqa: E402


def _build_synthetic_course(
    tmp_path: Path,
    *,
    n_prereq: int = 50,
    n_teaches: int = 20,
    bloom_levels=("remember", "understand", "apply"),
) -> Path:
    course = tmp_path / "synthetic-101"
    (course / "graph").mkdir(parents=True)
    edges = []
    for i in range(n_prereq):
        edges.append({
            "source": f"concept_{i}",
            "target": f"concept_{i+1}",
            "relation_type": "prerequisite_of",
        })
    for i in range(n_teaches):
        edges.append({
            "source": f"chunk_{i:05d}",
            "target": f"concept_{i}",
            "relation_type": "teaches",
        })
    # Add at_bloom_level edges for chunks
    for i in range(n_teaches):
        level = bloom_levels[i % len(bloom_levels)]
        edges.append({
            "source": f"chunk_{i:05d}",
            "target": f"bloom:{level}",
            "relation_type": "at_bloom_level",
        })
    nodes = [
        {"id": f"bloom:{level}", "class": "BloomLevel", "label": level.title(), "level": level}
        for level in bloom_levels
    ]
    payload = {"nodes": nodes, "edges": edges}
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    return course


def test_build_creates_split_file(tmp_path):
    course = _build_synthetic_course(tmp_path)
    builder = HoldoutBuilder(course, holdout_pct=0.1, seed=42)
    out = builder.build()
    assert out.exists()
    assert out.name == "holdout_split.json"


def test_split_is_reproducible_with_same_seed(tmp_path):
    """Same seed + same graph → identical split JSON."""
    course = _build_synthetic_course(tmp_path)
    out1 = HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    payload1 = out1.read_text(encoding="utf-8")
    # Rebuild
    out2 = HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    payload2 = out2.read_text(encoding="utf-8")
    assert payload1 == payload2


def test_split_diverges_with_different_seed(tmp_path):
    course = _build_synthetic_course(tmp_path)
    HoldoutBuilder(course, holdout_pct=0.1, seed=1).build()
    p1 = (course / "eval" / "holdout_split.json").read_text(encoding="utf-8")
    HoldoutBuilder(course, holdout_pct=0.1, seed=2).build()
    p2 = (course / "eval" / "holdout_split.json").read_text(encoding="utf-8")
    assert p1 != p2


def test_holdout_pct_lands_in_band(tmp_path):
    course = _build_synthetic_course(tmp_path, n_prereq=100, n_teaches=20)
    HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    pct = split["edges_held_out"] / split["edges_total"]
    # With 100 prereq + 20 teaches + 20 at_bloom_level = 140 edges,
    # 10% per type rounds to 10/2/2 = ~14 — well within tolerance.
    assert 0.05 <= pct <= 0.15


def test_bloom_strata_present(tmp_path):
    course = _build_synthetic_course(tmp_path)
    HoldoutBuilder(course, holdout_pct=0.2, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    assert "bloom_strata" in split
    strata = split["bloom_strata"]
    # Should have entries for at least the levels we synthesised
    assert "remember" in strata or "understand" in strata or "apply" in strata


def test_holdout_graph_hash_is_canonical_sha256(tmp_path):
    course = _build_synthetic_course(tmp_path)
    HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    split_path = course / "eval" / "holdout_split.json"
    split = load_holdout_split(split_path)
    declared_hash = split.pop("holdout_graph_hash")
    canonical = json.dumps(split, sort_keys=True, separators=(",", ":"))
    assert declared_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_per_relation_summary_covers_all_types(tmp_path):
    course = _build_synthetic_course(tmp_path)
    HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    per_rel = split["per_relation"]
    assert "prerequisite_of" in per_rel
    assert "teaches" in per_rel
    for rt, info in per_rel.items():
        assert info["total"] > 0
        assert info["held_out"] >= 0
        assert info["held_out"] < info["total"]


def test_invalid_pct_raises():
    with pytest.raises(ValueError):
        HoldoutBuilder(Path("/tmp"), holdout_pct=1.0)
    with pytest.raises(ValueError):
        HoldoutBuilder(Path("/tmp"), holdout_pct=0.0)


def test_missing_pedagogy_graph_raises(tmp_path):
    bare = tmp_path / "bare-101"
    bare.mkdir()
    builder = HoldoutBuilder(bare)
    with pytest.raises(FileNotFoundError):
        builder.build()


def test_relation_type_field_not_type_field(tmp_path):
    """Wave 92 schema correction: edges use ``relation_type`` not ``type``.

    Build a graph that ONLY has edges keyed on the wrong field name
    and assert the builder treats them as zero edges — fail-loud
    rather than silently misclassifying.
    """
    course = tmp_path / "wrong-key-101"
    (course / "graph").mkdir(parents=True)
    edges = [
        {"source": "a", "target": "b", "type": "prerequisite_of"},  # wrong key
        {"source": "c", "target": "d", "type": "teaches"},
    ]
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": [], "edges": edges}), encoding="utf-8",
    )
    HoldoutBuilder(course, holdout_pct=0.1).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    # Both edges had the wrong key → treated as zero
    assert split["edges_total"] == 2  # edges counted toward total
    assert split["edges_held_out"] == 0  # but none could be classified
    assert split["per_relation"] == {}
