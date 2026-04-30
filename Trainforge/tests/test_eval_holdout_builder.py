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


def test_probes_field_emitted_with_canonical_shape(tmp_path):
    """Wave 105: holdout_split.json must carry a ``probes`` array
    with one record per withheld edge in the canonical shape so
    downstream eval consumers can run prompt-shaped Tier-2 probes
    without re-deriving them from withheld_edges."""
    course = _build_synthetic_course(tmp_path)
    HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    assert "probes" in split
    probes = split["probes"]
    assert len(probes) == split["edges_held_out"]
    # Every probe carries the canonical fields.
    for p in probes:
        assert "probe_id" in p
        assert "prompt" in p
        assert "ground_truth_chunk_id" in p  # may be None
        assert "edge_type" in p
    # Probe IDs are unique.
    ids = [p["probe_id"] for p in probes]
    assert len(ids) == len(set(ids))


def test_holdout_emits_property_probes_when_manifest_present(tmp_path):
    """Audit 2026-04-30 fix: when a course has a property manifest and
    chunks containing the declared surface forms, holdout_split.json
    must emit a ``property_probes`` array with one or more probes per
    declared property. This unblocks PerPropertyEvaluator from the
    silent-skip path that produced all-null per_property_accuracy on
    the cc07cc76 run.

    Asserts coverage for ALL six rdf-shacl manifest properties, every
    probe carries the canonical shape, probe IDs are unique, and every
    probe's surface form appears in the prompt text (so the matching
    surface in PerPropertyEvaluator stays consistent even on the legacy
    fallback path)."""
    course = _build_synthetic_course(tmp_path, n_prereq=30, n_teaches=20)
    # Use the rdf-shacl family slug so load_property_manifest finds
    # the existing fixture under schemas/training/.
    rdf_course = tmp_path / "rdf-shacl-test-1"
    course.rename(rdf_course)
    # Synthetic chunks covering ALL SIX declared RDF/SHACL surface forms.
    (rdf_course / "corpus").mkdir(parents=True, exist_ok=True)
    chunks = [
        {
            "id": "chunk_00001",
            "summary": "Constraining datatypes",
            "text": "We use sh:datatype to require xsd:string values.",
        },
        {
            "id": "chunk_00002",
            "summary": "Constraining classes",
            "text": "Use sh:class to constrain the class of a property's object.",
        },
        {
            "id": "chunk_00003",
            "summary": "Defining node shapes",
            "text": "An sh:NodeShape declares structural constraints.",
        },
        {
            "id": "chunk_00004",
            "summary": "Property shape constraints",
            "text": "An sh:PropertyShape attaches constraints to a property.",
        },
        {
            "id": "chunk_00005",
            "summary": "Class hierarchies",
            "text": "Use rdfs:subClassOf to declare class hierarchies.",
        },
        {
            "id": "chunk_00006",
            "summary": "Asserting identity",
            "text": "owl:sameAs asserts that two URIs name the same thing.",
        },
    ]
    with (rdf_course / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    HoldoutBuilder(rdf_course, holdout_pct=0.1, seed=42).build()
    split = load_holdout_split(rdf_course / "eval" / "holdout_split.json")
    assert "property_probes" in split
    property_probes = split["property_probes"]
    # All six manifest properties produce at least one probe.
    expected_property_ids = {
        "sh_datatype", "sh_class", "sh_nodeshape",
        "sh_propertyshape", "rdfs_subclassof", "owl_sameas",
    }
    covered_properties = {p["property_id"] for p in property_probes}
    missing = expected_property_ids - covered_properties
    assert not missing, (
        f"property_probes missing coverage for: {sorted(missing)}; "
        f"got: {sorted(covered_properties)}"
    )
    # Every probe carries the canonical shape.
    for p in property_probes:
        assert "probe_id" in p
        assert "property_id" in p
        assert "prompt" in p
        assert "probe_text" in p
        assert "ground_truth_chunk_id" in p
        assert "surface_form" in p
        assert "expected_response" in p
    # Probe IDs are unique.
    probe_ids = [p["probe_id"] for p in property_probes]
    assert len(probe_ids) == len(set(probe_ids))
    # Each probe's surface form appears in its prompt text — this is
    # what the PerPropertyEvaluator's filter relies on for the fallback
    # path through `withheld_edges`.
    for p in property_probes:
        assert p["surface_form"] in p["prompt"], (
            f"probe {p['probe_id']} doesn't contain its surface form "
            f"{p['surface_form']!r} in prompt: {p['prompt']!r}"
        )


def test_holdout_property_probes_empty_without_manifest(tmp_path):
    """No manifest for the synthetic-101 family → empty property_probes
    array (graceful fallback, never crashes)."""
    course = _build_synthetic_course(tmp_path)
    HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    assert split.get("property_probes") == []


@pytest.mark.parametrize("chunk_id_template", [
    "chunk_{i:05d}",                  # canonical form
    "rdf_shacl_551_chunk_{i:05d}",    # corpus-prefixed form (production)
    "test_corpus_chunk_{i}",           # arbitrary-prefix form
])
def test_probes_do_not_leak_chunk_id_literals(tmp_path, chunk_id_template):
    """Audit 2026-04-30 fix: chunk-anchored probes must substitute the
    chunk's label (or a generic placeholder) for the raw chunk-ID.
    Without this, the model echoes the ID into prose answers (1441
    chunk-id token matches in the cc07cc76 eval).

    Parametrized over BOTH chunk-ID forms in production use:
      * canonical ``chunk_NNNNN`` (test fixtures, legacy corpora)
      * corpus-prefixed ``<corpus>_chunk_NNNNN`` (the rdf-shacl-551-2
        production corpus, where the bug was actually observed)
      * arbitrary-prefix form (defensive — covers any prefix shape).
    """
    import re as _re

    # Build a course whose chunks use the parametrized ID form.
    course = tmp_path / "course-101"
    (course / "graph").mkdir(parents=True)
    edges = []
    for i in range(20):
        edges.append({
            "source": chunk_id_template.format(i=i),
            "target": f"concept_{i}",
            "relation_type": "teaches",
        })
    for i in range(20):
        edges.append({
            "source": chunk_id_template.format(i=i),
            "target": "bloom:remember",
            "relation_type": "at_bloom_level",
        })
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": [], "edges": edges}), encoding="utf-8",
    )
    # Add a chunks.jsonl so the resolver has labels to return.
    (course / "corpus").mkdir(parents=True, exist_ok=True)
    with (course / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(20):
            fh.write(json.dumps({
                "id": chunk_id_template.format(i=i),
                "summary": f"Topic number {i}",
                "text": f"Content about topic {i}.",
            }) + "\n")
    HoldoutBuilder(course, holdout_pct=0.5, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    # Match BOTH the canonical and corpus-prefixed forms.
    pattern = _re.compile(r"\b\w*chunk_\d+\b")
    leaked = []
    for probe in split.get("probes", []):
        if pattern.search(probe["prompt"]):
            leaked.append(probe["prompt"])
    assert not leaked, (
        f"{len(leaked)} probe(s) leaked chunk-ID literal "
        f"(template={chunk_id_template!r}). Sample leaks: "
        f"{leaked[:3]}"
    )


def test_probes_ground_truth_chunk_id_set_for_chunk_anchored_edges(tmp_path):
    """Wave 105: probes derived from chunk-anchored edges (source
    starts with 'chunk_') must carry ``ground_truth_chunk_id``;
    concept->concept probes leave it None."""
    course = _build_synthetic_course(
        tmp_path,
        n_prereq=20,  # concept->concept
        n_teaches=20,  # chunk->concept
    )
    HoldoutBuilder(course, holdout_pct=0.2, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    chunk_anchored = [
        p for p in split["probes"]
        if p.get("ground_truth_chunk_id") is not None
    ]
    concept_only = [
        p for p in split["probes"]
        if p.get("ground_truth_chunk_id") is None
    ]
    assert chunk_anchored, "must have at least one chunk-anchored probe"
    for p in chunk_anchored:
        assert p["ground_truth_chunk_id"].startswith("chunk_")
    # Concept-only probes still have probe_id and prompt.
    for p in concept_only:
        assert p["probe_id"]
        assert p["prompt"]


def test_holdout_graph_hash_non_empty_for_real_input(tmp_path):
    """Wave 105: any non-trivial pedagogy graph must produce a
    non-empty-bytes hash. SHA-256(b'') is the placeholder used by
    the stub; the harness skips Tier-2 when it sees that value."""
    import hashlib as _hashlib

    course = _build_synthetic_course(tmp_path)
    HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    empty_hash = _hashlib.sha256(b"").hexdigest()
    assert split["holdout_graph_hash"] != empty_hash
    assert split["holdout_graph_hash"] != ""


def test_probes_count_for_synthetic_50_edge_graph(tmp_path):
    """Wave 105 spec: synthetic 50-edge graph at 10% holdout yields
    ~5 withheld + corresponding probes. Per-relation rounds up so we
    accept >=4 (one per relation type)."""
    course = _build_synthetic_course(
        tmp_path,
        n_prereq=25,  # 25 prereq edges
        n_teaches=25,  # 25 teaches edges (+ 25 at_bloom_level)
    )
    HoldoutBuilder(course, holdout_pct=0.1, seed=42).build()
    split = load_holdout_split(course / "eval" / "holdout_split.json")
    # Should be ~5 withheld edges (10% of 50 + at_bloom_level rounding).
    # Same n probes as withheld_edges.
    assert split["edges_held_out"] == len(split["probes"])
    assert split["edges_held_out"] >= 3


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


def test_negative_probes_emitted_and_balanced(tmp_path) -> None:
    """Wave 108 / Phase B: holdout_split.json must include a
    negative_probes[] array whose tuples (source, relation, target)
    are guaranteed NOT to appear anywhere in the graph's edges, so the
    correct ground-truth response is 'no'.

    Count must be balanced against the positive-probe count per
    relation_type (within +/- 1 to allow for rounding)."""
    import json
    from Trainforge.eval.holdout_builder import HoldoutBuilder

    course_dir = tmp_path / "courses" / "tst-101"
    (course_dir / "graph").mkdir(parents=True)
    (course_dir / "manifest.json").write_text("{}", encoding="utf-8")
    graph = {
        "nodes": [
            {"id": "concept_a"}, {"id": "concept_b"}, {"id": "concept_c"},
            {"id": "concept_d"}, {"id": "concept_e"}, {"id": "concept_f"},
            {"id": "concept_g"}, {"id": "concept_h"}, {"id": "concept_i"},
            {"id": "concept_j"},
        ],
        "edges": [
            {"source": "concept_a", "target": "concept_b", "relation_type": "prerequisite_of"},
            {"source": "concept_b", "target": "concept_c", "relation_type": "prerequisite_of"},
            {"source": "concept_c", "target": "concept_d", "relation_type": "prerequisite_of"},
            {"source": "concept_d", "target": "concept_e", "relation_type": "prerequisite_of"},
            {"source": "concept_e", "target": "concept_f", "relation_type": "prerequisite_of"},
            {"source": "concept_f", "target": "concept_g", "relation_type": "prerequisite_of"},
            {"source": "concept_g", "target": "concept_h", "relation_type": "prerequisite_of"},
            {"source": "concept_h", "target": "concept_i", "relation_type": "prerequisite_of"},
            {"source": "concept_i", "target": "concept_j", "relation_type": "prerequisite_of"},
            {"source": "concept_a", "target": "concept_c", "relation_type": "prerequisite_of"},
        ],
    }
    (course_dir / "graph" / "pedagogy_graph.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )

    out = HoldoutBuilder(course_dir, holdout_pct=0.5, seed=7).build()
    payload = json.loads(out.read_text(encoding="utf-8"))

    neg = payload.get("negative_probes")
    assert isinstance(neg, list) and len(neg) >= 1
    n_pos = payload["edges_held_out"]
    assert abs(len(neg) - n_pos) <= 1, (
        f"negative count {len(neg)} not balanced with positive count {n_pos}"
    )

    real_edges = {
        (e["source"], e["relation_type"], e["target"]) for e in graph["edges"]
    }
    for n in neg:
        triple = (n["source"], n["relation_type"], n["target"])
        assert triple not in real_edges, f"negative probe {triple} exists in graph"
        assert n.get("ground_truth") == "no"
