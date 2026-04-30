"""Tests for the schema-to-English translation SFT pair generator
(Wave 124 / Wave 125b expansion).

Mirrors the structure of `test_kg_metadata_generator.py` /
`test_abstention_generator.py`. Covers:

* Catalog size targets: total in [250, 300]; per-form floor 35; per-family floor 30.
* Each emitted pair validates against `instruction_pair.schema.json`.
* Every completion contains the literal primary CURIE so the
  ``preserve_tokens`` plumbing in synthesize_training.py recognises
  the surface form.
* Same manifest + same seed -> byte-identical pair list.
* `max_pairs` cap is honored and preserves family balance.
* DecisionCapture fires per-emit with rationale interpolating dynamic
  signals.
* All 6 surface forms emit pairs across all 6 template families.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.property_manifest import (  # noqa: E402
    PropertyEntry,
    PropertyManifest,
)
from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    SchemaTranslationStats,
    generate_schema_translation_pairs,
)


PAIR_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json"
)

# All 6 surface forms covered by the rdf_shacl manifest.
_RDF_SHACL_CURIES = (
    "sh:datatype",
    "sh:class",
    "sh:NodeShape",
    "sh:PropertyShape",
    "rdfs:subClassOf",
    "owl:sameAs",
)

# All 6 template families authored by the catalog.
_FAMILY_TEMPLATE_IDS = (
    "schema_translation.definition",
    "schema_translation.usage",
    "schema_translation.comparison",
    "schema_translation.reasoning",
    "schema_translation.pitfall",
    "schema_translation.combination",
)


class _FakeCapture:
    """Minimal DecisionCapture-shaped object for tests."""

    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []
        self._counter = 0

    def log_decision(self, **kwargs: Any) -> None:
        self._counter += 1
        record = dict(kwargs)
        record["event_id"] = f"EVT_{self._counter:06d}"
        self.decisions.append(record)


def _rdf_shacl_manifest() -> PropertyManifest:
    """The 6-property RDF/SHACL manifest the rdf-shacl-* family uses.

    Mirrors `schemas/training/property_manifest.rdf_shacl.yaml`. We
    construct it in-process rather than load from disk so the test
    doesn't depend on the file's continued presence.
    """
    return PropertyManifest(
        family="rdf_shacl",
        properties=[
            PropertyEntry(
                id="sh_datatype",
                uri="http://www.w3.org/ns/shacl#datatype",
                curie="sh:datatype",
                label="SHACL datatype constraint",
                surface_forms=["sh:datatype"],
                min_pairs=8,
            ),
            PropertyEntry(
                id="sh_class",
                uri="http://www.w3.org/ns/shacl#class",
                curie="sh:class",
                label="SHACL class constraint",
                surface_forms=["sh:class"],
                min_pairs=8,
            ),
            PropertyEntry(
                id="sh_nodeshape",
                uri="http://www.w3.org/ns/shacl#NodeShape",
                curie="sh:NodeShape",
                label="SHACL node shape declaration",
                surface_forms=["sh:NodeShape"],
                min_pairs=8,
            ),
            PropertyEntry(
                id="sh_propertyshape",
                uri="http://www.w3.org/ns/shacl#PropertyShape",
                curie="sh:PropertyShape",
                label="SHACL property shape declaration",
                surface_forms=["sh:PropertyShape"],
                min_pairs=5,
            ),
            PropertyEntry(
                id="rdfs_subclassof",
                uri="http://www.w3.org/2000/01/rdf-schema#subClassOf",
                curie="rdfs:subClassOf",
                label="RDFS subclass-of relation",
                surface_forms=["rdfs:subClassOf"],
                min_pairs=8,
            ),
            PropertyEntry(
                id="owl_sameas",
                uri="http://www.w3.org/2002/07/owl#sameAs",
                curie="owl:sameAs",
                label="OWL sameAs identity assertion",
                surface_forms=["owl:sameAs"],
                min_pairs=8,
            ),
        ],
    )


def _validate_pair(pair: Dict[str, Any]) -> None:
    """Validate a single pair against `instruction_pair.schema.json`."""
    import jsonschema

    schema = json.loads(PAIR_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(pair, schema)


# -----------------------------------------------------------------------------
# Catalog volume + balance assertions (Wave 125b targets).
# -----------------------------------------------------------------------------


def test_catalog_emits_at_least_250_pairs() -> None:
    """Wave 125b expansion: catalog has ~250 pairs total."""
    capture = _FakeCapture()
    pairs, stats = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    assert isinstance(stats, SchemaTranslationStats)
    assert len(pairs) >= 250, f"expected >=250 pairs, got {len(pairs)}"
    assert stats.pairs_emitted == len(pairs)


def test_catalog_emits_at_most_300_pairs() -> None:
    """Sanity ceiling so a future bug doesn't 10x the catalog."""
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    assert len(pairs) <= 300, f"expected <=300 pairs, got {len(pairs)}"


def test_per_form_floor() -> None:
    """Each of the 6 surface forms emits >=35 pairs across all families."""
    capture = _FakeCapture()
    pairs, stats = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    counter: Counter[str] = Counter(p["concept_tags"][0] for p in pairs)
    for curie in _RDF_SHACL_CURIES:
        assert counter[curie] >= 35, (
            f"surface form {curie!r} has only {counter[curie]} pairs "
            "(floor is 35)"
        )
    # Stats per_surface_form must agree.
    for curie in _RDF_SHACL_CURIES:
        assert stats.per_surface_form.get(curie, 0) == counter[curie]


def test_per_family_floor() -> None:
    """Each of the 6 template families is represented by >=30 pairs."""
    capture = _FakeCapture()
    pairs, stats = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    counter: Counter[str] = Counter(p["template_id"] for p in pairs)
    for tid in _FAMILY_TEMPLATE_IDS:
        assert counter[tid] >= 30, (
            f"family {tid!r} has only {counter[tid]} pairs (floor is 30)"
        )
    # Stats per_family aligns with template_id counts.
    for tid in _FAMILY_TEMPLATE_IDS:
        family = tid.split(".", 1)[1]
        assert stats.per_family.get(family, 0) == counter[tid]


def test_no_duplicate_prompts() -> None:
    """Every emitted prompt is unique across the full catalog."""
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    prompts = [p["prompt"] for p in pairs]
    duplicates = [item for item, c in Counter(prompts).items() if c > 1]
    assert not duplicates, f"duplicate prompts in catalog: {duplicates[:5]}"


def test_every_completion_contains_primary_curie() -> None:
    """Every completion must contain the literal primary CURIE so the
    preserve_tokens plumbing in synthesize_training.py picks it up."""
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    for pair in pairs:
        primary = pair["concept_tags"][0]
        assert primary in pair["completion"], (
            f"completion for primary CURIE {primary!r} does not contain "
            f"the literal surface form: {pair['completion']!r}"
        )


def test_pair_lengths_within_schema_bounds() -> None:
    """40 <= len(prompt) <= 400 and 50 <= len(completion) <= 600.

    Critical — without this enforcement, schema validation fails
    downstream in synthesize_training when pairs are written to JSONL.
    """
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    for pair in pairs:
        plen = len(pair["prompt"])
        clen = len(pair["completion"])
        assert 40 <= plen <= 400, (
            f"prompt out of bounds [{plen} chars]: {pair['prompt']!r}"
        )
        assert 50 <= clen <= 600, (
            f"completion out of bounds [{clen} chars]: "
            f"{pair['completion'][:120]!r}..."
        )


def test_max_pairs_respected_with_balanced_families() -> None:
    """`max_pairs=120` truncates correctly while preserving family balance.

    The cap must NEVER drop a family to zero — the round-robin emit
    order means a 120-pair cap visits every family at least once.
    """
    capture = _FakeCapture()
    pairs, stats = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=120,
    )
    assert len(pairs) <= 120
    assert stats.pairs_emitted == len(pairs)
    by_family: Counter[str] = Counter(p["template_id"] for p in pairs)
    for tid in _FAMILY_TEMPLATE_IDS:
        assert by_family[tid] > 0, (
            f"family {tid!r} dropped to zero under max_pairs=120; "
            "round-robin emit order should keep all families "
            f"represented. Counts: {dict(by_family)}"
        )


# -----------------------------------------------------------------------------
# Schema + integrity assertions (preserved from Wave 124).
# -----------------------------------------------------------------------------


def test_pair_validates_against_instruction_pair_schema() -> None:
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    assert pairs
    for pair in pairs:
        _validate_pair(pair)


def test_emits_pairs_across_all_six_surface_forms() -> None:
    """Every CURIE in the manifest emits all 6 template families."""
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    by_curie_family: Dict[str, set] = {}
    for p in pairs:
        c = p["concept_tags"][0]
        by_curie_family.setdefault(c, set()).add(p["template_id"])
    assert set(by_curie_family.keys()) == set(_RDF_SHACL_CURIES)
    for curie, families in by_curie_family.items():
        assert families == set(_FAMILY_TEMPLATE_IDS), (
            f"surface form {curie!r} missing families: "
            f"{set(_FAMILY_TEMPLATE_IDS) - families}"
        )


def test_completion_contains_curie_literal() -> None:
    """Every completion must contain the literal CURIE so the
    preserve_tokens plumbing in synthesize_training.py picks it up."""
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    for pair in pairs:
        curie = pair["concept_tags"][0]
        assert curie in pair["completion"], (
            f"completion for {curie!r} does not contain the literal "
            f"surface form: {pair['completion']!r}"
        )


def test_deterministic_across_runs() -> None:
    """Same manifest + same seed -> byte-identical pair list."""
    cap_a = _FakeCapture()
    cap_b = _FakeCapture()
    pairs_a, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(), capture=cap_a, max_pairs=10000, seed=999,
    )
    pairs_b, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(), capture=cap_b, max_pairs=10000, seed=999,
    )

    def _strip(p: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in p.items() if k != "decision_capture_id"}

    assert [_strip(p) for p in pairs_a] == [_strip(p) for p in pairs_b]


def test_max_pairs_respected() -> None:
    """Cap clamps emissions and never exceeds the cap."""
    capture = _FakeCapture()
    pairs, stats = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=3,
    )
    assert stats.pairs_emitted == 3
    assert len(pairs) == 3
    assert stats.capped_at_max_pairs is True


def test_decision_capture_fires_per_emit() -> None:
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    types = [d["decision_type"] for d in capture.decisions]
    assert types.count("schema_translation_generation") == len(pairs)

    for event in capture.decisions:
        rationale = event["rationale"]
        assert len(rationale) >= 20
        assert "seed=" in rationale
        assert "manifest_family=" in rationale
        # Wave 125b: family identifier interpolated for audit replay.
        assert "family=" in rationale
        # Wave 22 alternatives_considered convention.
        alts = event.get("alternatives_considered") or []
        for alt in alts:
            assert isinstance(alt, dict)
            assert "option" in alt
            assert "reason_rejected" in alt


def test_capture_required() -> None:
    """A None capture is rejected (Wave 112 invariant)."""
    with pytest.raises(ValueError, match="capture"):
        generate_schema_translation_pairs(
            _rdf_shacl_manifest(),
            capture=None,
            max_pairs=10,
        )


def test_unknown_curie_skipped_silently() -> None:
    """A manifest declaring a CURIE that is NOT in the hand-curated
    table should skip that surface form (with a warning) rather than
    crash. The known CURIE still emits its full per-form catalog."""
    capture = _FakeCapture()
    manifest = PropertyManifest(
        family="ex_unknown",
        properties=[
            PropertyEntry(
                id="sh_datatype",
                uri="http://www.w3.org/ns/shacl#datatype",
                curie="sh:datatype",
                label="SHACL datatype constraint",
                surface_forms=["sh:datatype"],
                min_pairs=2,
            ),
            PropertyEntry(
                id="ex_unknown",
                uri="http://example.test/unknown",
                curie="ex:unknown",
                label="A surface form with no entry in the table",
                surface_forms=["ex:unknown"],
                min_pairs=2,
            ),
        ],
    )
    pairs, stats = generate_schema_translation_pairs(
        manifest, capture=capture, max_pairs=10000,
    )
    assert stats.surface_forms_skipped_no_definition == 1
    # No emitted pair should reference the unknown CURIE.
    emitted_curies = {p["concept_tags"][0] for p in pairs}
    assert "ex:unknown" not in emitted_curies
    assert "sh:datatype" in emitted_curies
    # sh:datatype should have its full per-form catalog (>=35 pairs).
    sh_datatype_count = sum(
        1 for p in pairs if p["concept_tags"][0] == "sh:datatype"
    )
    assert sh_datatype_count >= 35


def test_pair_carries_marker_fields() -> None:
    """Marker fields downstream filters / diversity scorers rely on."""
    capture = _FakeCapture()
    pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
    )
    valid_blooms = {"remember", "understand", "apply", "analyze"}
    for pair in pairs:
        assert pair["content_type"] == "schema_translation"
        assert pair["bloom_level"] in valid_blooms
        assert pair["template_id"] in _FAMILY_TEMPLATE_IDS
        assert pair["requires_source_citation"] is False
        # Concept tag carries the literal CURIE.
        assert pair["concept_tags"]
        assert ":" in pair["concept_tags"][0]
