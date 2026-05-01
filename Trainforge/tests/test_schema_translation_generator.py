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
    crash. The known CURIE still emits its full per-form catalog.

    Wave 133d update: the loader now dispatches the catalog on
    ``manifest.family`` (was: a single global module-level table).
    To test "known family + unknown CURIE", the manifest uses
    family='rdf_shacl' so the loader falls back to the in-Python
    rdf_shacl catalog, then verifies the unknown CURIE is skipped
    gracefully within that catalog.
    """
    capture = _FakeCapture()
    manifest = PropertyManifest(
        family="rdf_shacl",
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


# ---------------------------------------------------------------------------
# Wave 133d: loader-pattern + RDF/SHACL fallback (Plan-2 P1#7)
# ---------------------------------------------------------------------------


def test_schema_translation_loader_falls_back_for_rdf_shacl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 136a contract (replaces Wave 133d identity check): when no
    per-family YAML catalog exists on disk for ``family == "rdf_shacl"``,
    ``_load_form_data`` must return the in-Python
    ``_RDF_SHACL_FALLBACK_FORM_DATA`` dict content unchanged so the
    rdf-shacl-551-2 rebuild keeps emitting the same pairs byte-
    identically (no eval-score drift). Wave 136a replaced the Wave
    133d whole-family-swap with a per-CURIE overlay merge — under the
    new contract the returned dict is a fresh merged copy whose
    content equals the base, NOT the same object."""
    from Trainforge.generators import schema_translation_generator as stg

    # Force the YAML lookup to fail — even if a future commit lands a
    # rdf_shacl YAML catalog, this test verifies the fallback branch.
    real_read_text = Path.read_text

    def _no_yaml(self: Path, *args: Any, **kwargs: Any) -> str:
        if "schema_translation_catalog." in self.name:
            raise FileNotFoundError(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _no_yaml)
    # Wave 136a: lru_cache means a previously-loaded entry would mask
    # the no-YAML branch under test. Invalidate before reading.
    stg._invalidate_form_data_cache()

    form_data = stg._load_form_data("rdf_shacl")
    # Wave 136a: content equality with the in-Python base — the
    # overlay is empty (no YAML) so the merge returns a base copy.
    assert form_data == stg._RDF_SHACL_FALLBACK_FORM_DATA, (
        "Wave 136a: rdf_shacl family with no on-disk YAML must return "
        "the in-Python fallback dict content (per-CURIE overlay merge "
        "with empty overlay)."
    )
    assert set(form_data.keys()) == set(
        stg._RDF_SHACL_FALLBACK_FORM_DATA.keys()
    )
    # Spot-check a known rdf_shacl CURIE from the fallback.
    assert "sh:datatype" in form_data
    assert form_data["sh:datatype"].curie == "sh:datatype"
    assert form_data["sh:datatype"].definitions, (
        "fallback rdf_shacl catalog must carry definitions for sh:datatype"
    )
    # Cleanup: clear cache so subsequent tests see the real catalog.
    stg._invalidate_form_data_cache()


def test_schema_translation_returns_empty_for_unknown_family_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wave 136a contract (replaces Wave 133d): an unknown family name
    (no YAML on disk AND no in-Python fallback) must return an empty
    dict. Wave 133d emitted a "no schema-translation catalog" warning
    here; Wave 136a's overlay merge no longer warns at this layer
    because the surrounding generator (``generate_schema_translation_pairs``)
    already surfaces the empty-catalog case via its own per-form
    warnings, and double-warning was noisy. The downstream-generator
    behavior — empty dict -> zero pairs emitted with form-level
    warnings -> no crash — is unchanged."""
    from Trainforge.generators import schema_translation_generator as stg

    import logging
    caplog.set_level(
        logging.WARNING,
        logger="Trainforge.generators.schema_translation_generator",
    )
    stg._invalidate_form_data_cache()
    form_data = stg._load_form_data("unknown_test_family")
    assert form_data == {}, (
        "Wave 136a: unknown family with no YAML and no in-Python "
        f"fallback must return empty dict, got {len(form_data)} entries."
    )
    stg._invalidate_form_data_cache()


def test_existing_rdf_shacl_pairs_byte_identical() -> None:
    """Wave 133d byte-identity assertion: the loader-pattern rename of
    ``_FORM_DATA -> _RDF_SHACL_FALLBACK_FORM_DATA`` plus the
    ``_load_form_data`` dispatch MUST NOT change the pair list shape
    or content for the rdf_shacl family. This is the regression net
    that lets the in-flight rdf-shacl-551-2 rebuild proceed without
    re-validating eval scores."""
    capture_a = _FakeCapture()
    capture_b = _FakeCapture()
    pairs_a, stats_a = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture_a,
        max_pairs=10000,
        seed=17,
    )
    pairs_b, stats_b = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture_b,
        max_pairs=10000,
        seed=17,
    )

    # Same inputs => same pair count.
    assert stats_a.pairs_emitted == stats_b.pairs_emitted
    assert len(pairs_a) == len(pairs_b)

    # Wave 125b ships 250 pairs; loader-pattern preserves that exact
    # number for the rdf_shacl family.
    assert stats_a.pairs_emitted == 250, (
        f"Wave 133d byte-identity: rdf_shacl pair count drifted from "
        f"the Wave 125b 250 baseline. Got {stats_a.pairs_emitted}."
    )

    # First 3 pairs (prompts + completions) must match across runs —
    # demonstrates determinism on top of byte-identity.
    for idx in range(3):
        assert pairs_a[idx]["prompt"] == pairs_b[idx]["prompt"], (
            f"prompt drift at pair {idx}: "
            f"{pairs_a[idx]['prompt']!r} != {pairs_b[idx]['prompt']!r}"
        )
        assert pairs_a[idx]["completion"] == pairs_b[idx]["completion"], (
            f"completion drift at pair {idx}"
        )
        assert pairs_a[idx]["concept_tags"] == pairs_b[idx]["concept_tags"]
        assert pairs_a[idx]["template_id"] == pairs_b[idx]["template_id"]


# ---------------------------------------------------------------------------
# Wave 135a: FORM_DATA contract scaffolding tests.
#
# Pin the user-stipulated safety contract for Wave 135b's anchored
# force-injection path:
#   1. Every manifest CURIE has >=1 definition entry.
#   2. Every manifest CURIE has >=1 usage_example entry.
#   3. No CURIE falls back to token-stuffing UNLESS its anchored_status
#      is explicitly "degraded_placeholder".
# ---------------------------------------------------------------------------


def _load_rdf_shacl_manifest_curies() -> List[str]:
    """Load the rdf-shacl manifest from disk; return its CURIE list."""
    from lib.ontology.property_manifest import load_property_manifest

    manifest = load_property_manifest("rdf-shacl-551-2")
    return [p.curie for p in manifest.properties]


def test_form_data_covers_all_manifest_curies() -> None:
    """Wave 135b MERGE GATE: every manifest CURIE has a FORM_DATA entry.

    This is the structural contract that Wave 135b's anchored
    force-injection path requires. Without this every-CURIE coverage,
    a force-inject call against an unmapped CURIE has no entry to
    dispatch on, which is what Wave 135a closes.
    """
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )

    manifest_curies = _load_rdf_shacl_manifest_curies()
    assert manifest_curies, "manifest must declare at least one CURIE"

    missing = [c for c in manifest_curies if c not in _RDF_SHACL_FALLBACK_FORM_DATA]
    assert not missing, (
        f"Wave 135a contract: every manifest CURIE must have a "
        f"FORM_DATA entry. Missing {len(missing)}: {missing}"
    )


def test_every_form_data_entry_has_at_least_one_definition() -> None:
    """Structural — every entry must carry >=1 definition string."""
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )

    offenders = [
        curie
        for curie, entry in _RDF_SHACL_FALLBACK_FORM_DATA.items()
        if len(entry.definitions) < 1
    ]
    assert not offenders, (
        f"Wave 135a contract: every FORM_DATA entry must have >=1 "
        f"definition (real OR degraded stub). Offenders: {offenders}"
    )


def test_every_form_data_entry_has_at_least_one_usage_example() -> None:
    """Structural — every entry must carry >=1 usage_example tuple."""
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )

    offenders = [
        curie
        for curie, entry in _RDF_SHACL_FALLBACK_FORM_DATA.items()
        if len(entry.usage_examples) < 1
    ]
    assert not offenders, (
        f"Wave 135a contract: every FORM_DATA entry must have >=1 "
        f"usage_example (real OR degraded stub). Offenders: {offenders}"
    )


def test_anchored_status_is_valid_enum() -> None:
    """Every entry's ``anchored_status`` must be one of the two valid
    discriminator values: ``"complete"`` or ``"degraded_placeholder"``."""
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )

    valid = {"complete", "degraded_placeholder"}
    offenders = {
        curie: entry.anchored_status
        for curie, entry in _RDF_SHACL_FALLBACK_FORM_DATA.items()
        if entry.anchored_status not in valid
    }
    assert not offenders, (
        f"Wave 135a contract: anchored_status must be one of {valid}. "
        f"Offenders: {offenders}"
    )


def test_existing_six_curies_remain_complete() -> None:
    """Regression-pin: the 6 pre-Wave-135a entries (sh:datatype,
    sh:class, sh:NodeShape, sh:PropertyShape, rdfs:subClassOf,
    owl:sameAs) MUST keep ``anchored_status="complete"``. An accidental
    degradation would silently drop their pairs from the schema-
    translation generator output and tank the Wave 125b 250-pair
    catalog volume."""
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )

    expected_complete = {
        "sh:datatype",
        "sh:class",
        "sh:NodeShape",
        "sh:PropertyShape",
        "rdfs:subClassOf",
        "owl:sameAs",
    }
    for curie in expected_complete:
        entry = _RDF_SHACL_FALLBACK_FORM_DATA.get(curie)
        assert entry is not None, (
            f"Wave 135a regression: pre-Wave-135a entry {curie!r} "
            f"missing from FORM_DATA"
        )
        assert entry.anchored_status == "complete", (
            f"Wave 135a regression: {curie!r} must keep "
            f"anchored_status='complete' (got "
            f"{entry.anchored_status!r}). Backfill operators must NOT "
            f"flip a complete entry to degraded — only the reverse."
        )


def test_thirty_four_new_curies_are_degraded_placeholder() -> None:
    """Wave 135a ships 34 new entries (40 manifest CURIEs - 6 existing).
    Every one of them must be tagged ``anchored_status=
    "degraded_placeholder"`` so Wave 135b's force-injection path knows
    where to stub-fill / WARN.
    """
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )

    degraded = [
        curie
        for curie, entry in _RDF_SHACL_FALLBACK_FORM_DATA.items()
        if entry.anchored_status == "degraded_placeholder"
    ]
    assert len(degraded) == 34, (
        f"Wave 135a contract: expected exactly 34 "
        f"anchored_status='degraded_placeholder' entries (40 manifest "
        f"CURIEs - 6 existing complete entries). Got {len(degraded)}."
    )


def test_validate_form_data_contract_passes_on_full_set() -> None:
    """Positive case: the shipped FORM_DATA + the rdf-shacl manifest
    together must satisfy the Wave 135a contract."""
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
        validate_form_data_contract,
    )

    manifest_curies = _load_rdf_shacl_manifest_curies()
    result = validate_form_data_contract(
        _RDF_SHACL_FALLBACK_FORM_DATA, manifest_curies
    )
    assert result["passed"] is True, result
    assert result["missing_curies"] == []
    assert result["incomplete_curies"] == []
    assert result["invalid_status_curies"] == []
    assert result["complete_count"] == 6
    assert result["degraded_count"] == 34


def test_validate_form_data_contract_fails_on_missing_curie() -> None:
    """Negative case: drop one manifest CURIE from the form_data
    input; the validator must report ``passed=False`` with the
    dropped CURIE listed in ``missing_curies``."""
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
        validate_form_data_contract,
    )

    manifest_curies = _load_rdf_shacl_manifest_curies()
    # Pick a known degraded-placeholder CURIE to drop so the test
    # doesn't depend on which entry happens to be at any particular
    # position in the manifest.
    drop_target = "sh:minCount"
    assert drop_target in _RDF_SHACL_FALLBACK_FORM_DATA, (
        f"test fixture invariant: {drop_target} should be in FORM_DATA"
    )
    synthetic = {
        curie: entry
        for curie, entry in _RDF_SHACL_FALLBACK_FORM_DATA.items()
        if curie != drop_target
    }
    result = validate_form_data_contract(synthetic, manifest_curies)
    assert result["passed"] is False
    assert drop_target in result["missing_curies"]


# ---------------------------------------------------------------------------
# Wave 135a: schema_translation generator must skip degraded entries
# so no pair body ever contains the literal "[degraded:" stub text.
# ---------------------------------------------------------------------------


def test_schema_translation_skips_degraded_placeholder_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic form_data with mixed complete + degraded entries.

    Asserts:
      1. Generated pairs ONLY come from ``complete`` entries.
      2. No pair body contains the literal ``"[degraded:"`` token —
         the stub strings must never bleed into the emitted pair list
         that gets written to ``instruction_pairs.jsonl``.
    """
    from Trainforge.generators import schema_translation_generator as stg
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        generate_schema_translation_pairs,
    )

    # Build a synthetic form_data with one complete + one degraded
    # entry. Re-use the real sh:datatype entry for "complete" so we
    # exercise the actual catalog walk on a known surface form.
    real_datatype = stg._RDF_SHACL_FALLBACK_FORM_DATA["sh:datatype"]
    synthetic_form_data = {
        "sh:datatype": real_datatype,
        "sh:minCount": SurfaceFormData(
            curie="sh:minCount",
            short_name="sh:minCount",
            definitions=[
                "[degraded: anchored definition not yet authored — see Wave 135a contract]",
            ],
            usage_examples=[
                (
                    "[degraded: anchored usage prompt not yet authored]",
                    "[degraded: anchored usage answer not yet authored]",
                ),
            ],
            anchored_status="degraded_placeholder",
        ),
    }

    # Force the loader to return our synthetic form_data so we exercise
    # the skip-on-degraded path inside generate_schema_translation_pairs
    # without relying on the on-disk catalog or other entries.
    monkeypatch.setattr(
        stg, "_load_form_data", lambda family: synthetic_form_data
    )

    manifest = PropertyManifest(
        family="rdf_shacl",
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
                id="sh_mincount",
                uri="http://www.w3.org/ns/shacl#minCount",
                curie="sh:minCount",
                label="SHACL minCount",
                surface_forms=["sh:minCount"],
                min_pairs=2,
            ),
        ],
    )
    capture = _FakeCapture()
    pairs, _stats = generate_schema_translation_pairs(
        manifest, capture=capture, max_pairs=10000,
    )

    assert pairs, "expected emitted pairs from the complete entry"

    # 1. Every emitted pair must come from the COMPLETE entry only.
    emitted_curies = {p["concept_tags"][0] for p in pairs}
    assert emitted_curies == {"sh:datatype"}, (
        f"degraded-placeholder entries must NEVER appear in emitted "
        f"pairs. Got concept_tag CURIEs: {emitted_curies}"
    )

    # 2. The literal "[degraded:" token must not appear anywhere in
    #    any prompt or completion of the emitted pairs.
    for pair in pairs:
        assert "[degraded:" not in pair["prompt"], (
            f"degraded stub leaked into prompt: {pair['prompt']!r}"
        )
        assert "[degraded:" not in pair["completion"], (
            f"degraded stub leaked into completion: "
            f"{pair['completion']!r}"
        )


# ---------------------------------------------------------------------------
# Wave 136a: per-CURIE overlay merge contract.
#
# Tests that the new ``_load_form_data`` dispatches through
# ``_python_fallback_for_family`` + ``_load_yaml_catalog`` +
# ``_deep_merge_by_curie`` correctly:
#   - Partial YAML cannot erase the in-Python fallback's complete entries.
#   - Per-CURIE swap: YAML wins for CURIEs it defines.
#   - Per-CURIE add: YAML CURIEs not in Python are added.
#   - complete -> degraded regression in YAML emits a warning (not block).
#   - complete with empty definitions raises ValueError at load.
#   - Malformed YAML returns the Python base (no erasure).
#   - No YAML -> byte-identical to Wave 133d's pair output.
#   - lru_cache identity + invalidation helper.
# ---------------------------------------------------------------------------


# Synthetic test-fixture markers — clearly NOT corpus content. The
# Wave 136a test surface MUST not contain real-looking definitions
# that could accidentally seed training data on a downstream pipeline
# run picking up these synthetic CURIEs / strings.
_TEST_FIXTURE_DEF = (
    "[TEST FIXTURE: synthetic definition for Wave 136a regression test]"
)
_TEST_FIXTURE_USAGE_PROMPT = (
    "[TEST FIXTURE: synthetic usage prompt for Wave 136a regression test]"
)
_TEST_FIXTURE_USAGE_COMPLETION = (
    "[TEST FIXTURE: synthetic usage answer for Wave 136a regression test]"
)


def _patch_yaml_overlay(
    monkeypatch: pytest.MonkeyPatch,
    overlay: Dict[str, Any],
) -> None:
    """Patch ``_load_yaml_catalog`` so the overlay test path uses an
    in-memory synthetic dict instead of the on-disk YAML. Returns a
    SurfaceFormData-shaped dict (NOT raw YAML) keyed by CURIE."""
    from Trainforge.generators import schema_translation_generator as stg

    monkeypatch.setattr(stg, "_load_yaml_catalog", lambda family: overlay)
    stg._invalidate_form_data_cache()


def test_overlay_loader_partial_yaml_does_not_erase_complete_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 136a regression-pin: a partial YAML overlay (one CURIE
    defined, rest absent) must NOT erase the in-Python fallback's
    other complete entries.

    This is the load-bearing safety property — Wave 136d's operator-
    paused backfill flow lands one CURIE at a time, and a YAML with
    only sh:minCount filled in must leave the existing 6 complete
    entries (sh:datatype, sh:class, sh:NodeShape, sh:PropertyShape,
    rdfs:subClassOf, owl:sameAs) intact and complete.
    """
    from Trainforge.generators import schema_translation_generator as stg
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
    )

    overlay = {
        "sh:minCount": SurfaceFormData(
            curie="sh:minCount",
            short_name="sh:minCount",
            anchored_status="complete",
            definitions=[_TEST_FIXTURE_DEF],
            usage_examples=[
                (_TEST_FIXTURE_USAGE_PROMPT, _TEST_FIXTURE_USAGE_COMPLETION),
            ],
        ),
    }
    _patch_yaml_overlay(monkeypatch, overlay)

    merged = stg._load_form_data("rdf_shacl")

    expected_complete = (
        "sh:datatype",
        "sh:class",
        "sh:NodeShape",
        "sh:PropertyShape",
        "rdfs:subClassOf",
        "owl:sameAs",
    )
    for curie in expected_complete:
        entry = merged.get(curie)
        assert entry is not None, (
            f"Wave 136a regression: complete entry {curie} missing "
            f"from merged dict — partial YAML must NOT erase Python base"
        )
        assert entry.anchored_status == "complete", (
            f"Wave 136a regression: complete entry {curie} silently "
            f"flipped to {entry.anchored_status!r}"
        )
        assert entry.definitions, (
            f"Wave 136a regression: complete entry {curie} lost its "
            f"definitions"
        )

    stg._invalidate_form_data_cache()


def test_overlay_loader_yaml_swaps_curie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 136a per-CURIE swap: a YAML overlay entry replaces the
    Python entry for the same CURIE."""
    from Trainforge.generators import schema_translation_generator as stg
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
    )

    overlay = {
        "sh:minCount": SurfaceFormData(
            curie="sh:minCount",
            short_name="sh:minCount",
            anchored_status="complete",
            definitions=[_TEST_FIXTURE_DEF],
            usage_examples=[
                (_TEST_FIXTURE_USAGE_PROMPT, _TEST_FIXTURE_USAGE_COMPLETION),
            ],
        ),
    }
    _patch_yaml_overlay(monkeypatch, overlay)

    merged = stg._load_form_data("rdf_shacl")

    swapped = merged["sh:minCount"]
    assert swapped.anchored_status == "complete", (
        "Wave 136a: YAML overlay must swap sh:minCount from "
        "degraded_placeholder (Python base) to complete (YAML)"
    )
    assert swapped.definitions == [_TEST_FIXTURE_DEF]
    assert "[degraded:" not in swapped.definitions[0], (
        "Wave 136a: post-swap definitions must come from YAML, not "
        "the Python base's degraded stubs"
    )

    stg._invalidate_form_data_cache()


def test_overlay_loader_extension_curie_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 136a per-CURIE add: a YAML overlay entry whose CURIE is
    NOT in the Python base is added to the merged dict, and the
    Python base entries remain unchanged."""
    from Trainforge.generators import schema_translation_generator as stg
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
    )

    extension_curie = "test:SyntheticPredicate"
    overlay = {
        extension_curie: SurfaceFormData(
            curie=extension_curie,
            short_name="SyntheticPredicate",
            anchored_status="complete",
            definitions=[_TEST_FIXTURE_DEF],
            usage_examples=[
                (_TEST_FIXTURE_USAGE_PROMPT, _TEST_FIXTURE_USAGE_COMPLETION),
            ],
        ),
    }
    _patch_yaml_overlay(monkeypatch, overlay)

    merged = stg._load_form_data("rdf_shacl")

    assert extension_curie in merged, (
        f"Wave 136a: YAML extension CURIE {extension_curie} must appear "
        f"in merged dict"
    )
    assert merged[extension_curie].anchored_status == "complete"

    # And the existing 6 complete entries are still complete.
    for curie in (
        "sh:datatype",
        "sh:class",
        "sh:NodeShape",
        "sh:PropertyShape",
        "rdfs:subClassOf",
        "owl:sameAs",
    ):
        assert merged[curie].anchored_status == "complete", (
            f"Wave 136a: extension overlay must not disturb existing "
            f"complete entry {curie}"
        )

    stg._invalidate_form_data_cache()


def test_overlay_loader_warns_on_complete_to_degraded_regression(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wave 136a: a YAML overlay that regresses a complete base entry
    to ``degraded_placeholder`` must emit a logger.warning (mid-edit
    signal) but NOT block the load.

    This is the operator-pause path: under-revision CURIEs are
    legitimately marked degraded mid-flow, but the warning makes the
    regression visible in logs.
    """
    import logging

    from Trainforge.generators import schema_translation_generator as stg
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
    )

    overlay = {
        "sh:datatype": SurfaceFormData(
            curie="sh:datatype",
            short_name="sh:datatype",
            anchored_status="degraded_placeholder",
            definitions=[
                "[TEST FIXTURE: synthetic degraded definition for Wave 136a]",
            ],
            usage_examples=[
                (_TEST_FIXTURE_USAGE_PROMPT, _TEST_FIXTURE_USAGE_COMPLETION),
            ],
        ),
    }
    _patch_yaml_overlay(monkeypatch, overlay)

    caplog.set_level(
        logging.WARNING,
        logger="Trainforge.generators.schema_translation_generator",
    )

    merged = stg._load_form_data("rdf_shacl")

    # The overlay was accepted (no block).
    assert merged["sh:datatype"].anchored_status == "degraded_placeholder"

    regression_warnings = [
        r for r in caplog.records
        if "regressed CURIE" in r.getMessage()
        and "sh:datatype" in r.getMessage()
    ]
    assert regression_warnings, (
        f"Wave 136a: expected a complete->degraded regression warning "
        f"for sh:datatype; got {[r.getMessage() for r in caplog.records]!r}"
    )

    stg._invalidate_form_data_cache()


def test_overlay_loader_rejects_empty_definitions_when_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 136a: an overlay entry with ``anchored_status="complete"``
    but empty ``definitions`` must raise ``ValueError`` at load time.

    This is the overlay-level structural reject — Wave 136b will
    widen this to cover stub-string content quality.
    """
    from Trainforge.generators import schema_translation_generator as stg
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
    )

    overlay = {
        "test:EmptyComplete": SurfaceFormData(
            curie="test:EmptyComplete",
            short_name="test:EmptyComplete",
            anchored_status="complete",
            definitions=[],  # Empty — should raise.
            usage_examples=[
                (_TEST_FIXTURE_USAGE_PROMPT, _TEST_FIXTURE_USAGE_COMPLETION),
            ],
        ),
    }
    _patch_yaml_overlay(monkeypatch, overlay)

    with pytest.raises(ValueError, match="empty definitions"):
        stg._load_form_data("rdf_shacl")

    stg._invalidate_form_data_cache()


def test_overlay_loader_malformed_yaml_returns_base(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wave 136a: malformed YAML on disk must return the in-Python
    base unchanged (no erasure) and emit a logger.error.

    Load-bearing ToS-mitigation: a corrupted YAML must NEVER be
    interpreted as "erase everything". The in-Python complete entries
    are the last-resort source of truth.
    """
    import logging

    from Trainforge.generators import schema_translation_generator as stg

    # Write garbage to a tmp YAML path and redirect PROJECT_ROOT so
    # _load_yaml_catalog reads from there.
    fake_root = tmp_path
    schemas_dir = fake_root / "schemas" / "training"
    schemas_dir.mkdir(parents=True)
    bad_yaml_path = schemas_dir / "schema_translation_catalog.rdf_shacl.yaml"
    # YAML with unbalanced brackets / tab-vs-space — guaranteed to
    # raise yaml.YAMLError on safe_load.
    bad_yaml_path.write_text(
        "family: rdf_shacl\nforms: {[: this is not valid yaml :}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(stg, "PROJECT_ROOT", fake_root)
    stg._invalidate_form_data_cache()

    caplog.set_level(
        logging.ERROR,
        logger="Trainforge.generators.schema_translation_generator",
    )

    merged = stg._load_form_data("rdf_shacl")

    # Content-equal to the Python base: malformed YAML did NOT erase.
    assert merged == stg._RDF_SHACL_FALLBACK_FORM_DATA, (
        "Wave 136a load-bearing safety: malformed YAML must return the "
        "in-Python base unchanged (no erasure)."
    )

    yaml_error_logs = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and "rdf_shacl" in r.getMessage()
    ]
    assert yaml_error_logs, (
        "Wave 136a: malformed YAML must emit a logger.error so the "
        f"operator sees the parse failure; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )

    stg._invalidate_form_data_cache()


def test_overlay_loader_byte_identical_when_no_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave 136a byte-identity assertion (preserves Wave 133d's pin):
    with no YAML overlay file present, ``_load_form_data`` returns a
    merged dict whose content equals the in-Python base exactly, AND
    ``generate_schema_translation_pairs`` produces the same 250 pairs
    with the same first-3 prompts as the Wave 125b baseline.

    This is the regression net that confirms the YAML transcription
    path is bit-identical to the in-Python fallback for the rdf_shacl
    family.
    """
    from Trainforge.generators import schema_translation_generator as stg

    # Force the YAML lookup to fail so the merge sees an empty overlay.
    real_read_text = Path.read_text

    def _no_yaml(self: Path, *args: Any, **kwargs: Any) -> str:
        if "schema_translation_catalog." in self.name:
            raise FileNotFoundError(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _no_yaml)
    stg._invalidate_form_data_cache()

    merged = stg._load_form_data("rdf_shacl")
    assert merged == stg._RDF_SHACL_FALLBACK_FORM_DATA

    # Pair output byte-identity: same 250 pairs as the Wave 125b
    # baseline. We compute against the exact same baseline shape as
    # ``test_existing_rdf_shacl_pairs_byte_identical``.
    capture = _FakeCapture()
    pairs, stats = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10000,
        seed=17,
    )
    assert stats.pairs_emitted == 250, (
        f"Wave 136a no-overlay byte-identity: pair count drifted from "
        f"the Wave 125b 250 baseline. Got {stats.pairs_emitted}."
    )
    # Non-trivial first-3 prompts (anchors the ordering).
    assert pairs[0]["prompt"], "first pair prompt must be non-empty"
    assert pairs[1]["prompt"]
    assert pairs[2]["prompt"]
    # Each prompt is unique (sanity check on shuffle determinism).
    assert pairs[0]["prompt"] != pairs[1]["prompt"]

    stg._invalidate_form_data_cache()


def test_overlay_loader_caches() -> None:
    """Wave 136a: ``_load_form_data`` is wrapped in
    ``functools.lru_cache``. Two consecutive calls return the same
    object identity (cached). After ``_invalidate_form_data_cache()``,
    the next call returns a fresh dict object."""
    from Trainforge.generators import schema_translation_generator as stg

    stg._invalidate_form_data_cache()

    result1 = stg._load_form_data("rdf_shacl")
    result2 = stg._load_form_data("rdf_shacl")
    assert result1 is result2, (
        "Wave 136a: lru_cache must return identical object on repeat "
        "calls with the same family"
    )

    stg._invalidate_form_data_cache()
    result3 = stg._load_form_data("rdf_shacl")
    assert result3 is not result1, (
        "Wave 136a: post-invalidation, the cache must miss and return "
        "a fresh dict object (content equality preserved)"
    )
    assert result3 == result1, (
        "Wave 136a: post-invalidation content must still equal the "
        "original (no behavioral drift on cache reload)"
    )

    stg._invalidate_form_data_cache()


# ---------------------------------------------------------------------------
# Wave 136b: content-quality rejection rules.
#
# Each test mutates ONE field of an otherwise-valid synthetic
# ``anchored_status="complete"`` entry to trip exactly ONE rule, then
# asserts the validator emits the matching code with ``passed=False``.
# Synthetic CURIEs (``test:Foo`` / ``ex:SyntheticPredicate`` / ...) keep
# the test fixtures clearly fixture-marked (no risk of corpus drift).
# ---------------------------------------------------------------------------


def _synthetic_complete_entry(
    curie: str = "test:Foo",
    *,
    definition: str = (
        "test:Foo is a synthetic predicate used by Wave 136b validator "
        "tests as a stand-in for a real complete entry."
    ),
    usage_prompt: str = (
        "Show how test:Foo is used in a synthetic SHACL example "
        "fixture."
    ),
    usage_answer: str = (
        "On a property shape with sh:path ex:bar, write `test:Foo "
        "ex:value .` — test:Foo is the synthetic predicate under test."
    ),
):
    """Build a single ``SurfaceFormData`` entry that satisfies every
    Wave 136b content-quality rule out of the box, so test cases can
    mutate ONE field at a time and trip exactly the rule under test."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
    )

    return SurfaceFormData(
        curie=curie,
        short_name=curie.split(":", 1)[-1],
        definitions=[definition],
        usage_examples=[(usage_prompt, usage_answer)],
        anchored_status="complete",
    )


def _violation_codes(result: Dict[str, Any]) -> List[str]:
    """Return a list of content_violation codes, one per emitted entry."""
    return [v["code"] for v in result.get("content_violations", [])]


def test_validator_rejects_definition_without_verbatim_curie() -> None:
    """Rule: CURIE_NOT_VERBATIM_DEFINITION."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    # Mutate definition to drop the literal CURIE.
    entry.definitions = [
        "This synthetic predicate is used by Wave 136b validator tests "
        "as a stand-in for a real complete entry without naming."
    ]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "CURIE_NOT_VERBATIM_DEFINITION" in _violation_codes(result)


def test_validator_rejects_usage_answer_without_verbatim_curie() -> None:
    """Rule: CURIE_NOT_VERBATIM_USAGE_ANSWER."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    # Mutate usage_answer to drop the literal CURIE.
    entry.usage_examples = [
        (
            "Show how the synthetic predicate is used in a SHACL "
            "fixture.",
            (
                "On a property shape with sh:path ex:bar, write the "
                "synthetic predicate followed by an IRI value as "
                "demonstrated in the fixture corpus example."
            ),
        )
    ]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "CURIE_NOT_VERBATIM_USAGE_ANSWER" in _violation_codes(result)


def test_validator_rejects_old_suffix_template_leak() -> None:
    """Rule: OLD_SUFFIX_TEMPLATE_LEAK — definition starting with one of
    the Wave 121 token-stuffing template prefixes is rejected."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    # Definition starts with "Canonical terms:" — a forbidden prefix.
    entry.definitions = [
        "Canonical terms: test:Foo is a synthetic predicate used by "
        "Wave 136b validator tests as a stand-in for a complete entry."
    ]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "OLD_SUFFIX_TEMPLATE_LEAK" in _violation_codes(result)


def test_validator_rejects_placeholder_leakage() -> None:
    """Rule: PLACEHOLDER_LEAKAGE — definition containing the
    Wave 135a stub marker ``"[degraded:"`` is rejected on a complete
    entry."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    entry.definitions = [
        "test:Foo is a synthetic predicate [degraded: anchored "
        "definition not yet authored — stub leaked into a complete "
        "entry by Wave 136b validator-test fixture mutation]."
    ]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "PLACEHOLDER_LEAKAGE" in _violation_codes(result)


def test_validator_rejects_definition_below_length_floor() -> None:
    """Rule: LENGTH_OUT_OF_BOUNDS_DEF — 49-char definition rejected."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    # 49 chars exactly — below the 50-char floor.
    short_def = "test:Foo is too short by exactly one single char "[:49]
    assert len(short_def) == 49
    assert "test:Foo" in short_def
    entry.definitions = [short_def]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "LENGTH_OUT_OF_BOUNDS_DEF" in _violation_codes(result)


def test_validator_rejects_definition_above_length_ceiling() -> None:
    """Rule: LENGTH_OUT_OF_BOUNDS_DEF — 401-char definition rejected."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    # 401 chars exactly — above the 400-char ceiling.
    long_def = "test:Foo " + ("padding " * 60)
    long_def = long_def[:401]
    assert len(long_def) == 401
    assert "test:Foo" in long_def
    entry.definitions = [long_def]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "LENGTH_OUT_OF_BOUNDS_DEF" in _violation_codes(result)


def test_validator_rejects_usage_prompt_below_length_floor() -> None:
    """Rule: LENGTH_OUT_OF_BOUNDS_USAGE_PROMPT — usage prompt below the
    schema's PROMPT_MIN floor (40 chars) is rejected."""
    from Trainforge.generators._anthropic_provider import PROMPT_MIN
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    short_prompt = "test:Foo prompt"  # well below 40 chars
    assert len(short_prompt) < PROMPT_MIN
    entry.usage_examples = [
        (
            short_prompt,
            (
                "On a property shape with sh:path ex:bar, write "
                "`test:Foo ex:value .` — test:Foo is the synthetic "
                "predicate under test in this fixture."
            ),
        )
    ]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "LENGTH_OUT_OF_BOUNDS_USAGE_PROMPT" in _violation_codes(
        result
    )


def test_validator_rejects_usage_answer_above_length_ceiling() -> None:
    """Rule: LENGTH_OUT_OF_BOUNDS_USAGE_ANSWER — answer above the
    schema's COMPLETION_MAX ceiling (600 chars) is rejected."""
    from Trainforge.generators._anthropic_provider import COMPLETION_MAX
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    long_answer = "test:Foo " + ("padding-token " * 60)
    long_answer = long_answer[: COMPLETION_MAX + 1]
    assert len(long_answer) == COMPLETION_MAX + 1
    assert "test:Foo" in long_answer
    entry.usage_examples = [
        (
            "Show how test:Foo is used in a synthetic SHACL example "
            "fixture for Wave 136b length-bound testing.",
            long_answer,
        )
    ]
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "LENGTH_OUT_OF_BOUNDS_USAGE_ANSWER" in _violation_codes(
        result
    )


def test_validator_rejects_wrong_curie_only_mention() -> None:
    """Rule: WRONG_CURIE_ONLY_MENTION — definition for sh:minCount
    mentioning only sibling sh:maxCount (not its own CURIE) is
    rejected. Uses synthetic-but-realistic SHACL CURIEs so the
    sibling-CURIE detection path exercises the manifest-set path."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    # Definition mentions sh:maxCount but NOT sh:minCount.
    entry = _synthetic_complete_entry(
        curie="sh:minCount",
        definition=(
            "This SHACL constraint counts how many values a property "
            "has and rejects shapes whose value count exceeds the "
            "sh:maxCount ceiling for the focus node."
        ),
    )
    # Make sure the manifest set includes sh:maxCount so it's a known
    # sibling that should fire the rule.
    form_data = {"sh:minCount": entry}
    result = validate_form_data_contract(
        form_data, ["sh:minCount", "sh:maxCount"]
    )
    assert result["passed"] is False
    codes = _violation_codes(result)
    assert "WRONG_CURIE_ONLY_MENTION" in codes


def test_validator_rejects_generic_definitions_no_usage() -> None:
    """Rule: GENERIC_DEFINITIONS_NO_USAGE — complete entry with at
    least one definition but ZERO usage_examples tuples is rejected."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    entry = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            "test:Foo is a synthetic predicate used by Wave 136b "
            "validator tests with definitions but no usage."
        ],
        usage_examples=[],  # Empty — trips the rule.
        anchored_status="complete",
    )
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    assert result["passed"] is False
    assert "GENERIC_DEFINITIONS_NO_USAGE" in _violation_codes(result)


def test_validator_emits_overlay_regression_warning() -> None:
    """Rule: OVERLAY_LOAD_REGRESSION (warning) — synthetic base marks
    a CURIE complete; overlay marks the same CURIE
    degraded_placeholder. The validator emits the warning, but
    ``passed`` reflects only critical content rules — the warning
    itself is non-blocking."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    base_entry = _synthetic_complete_entry()
    # Overlay-merged form_data: same CURIE, but anchored_status flipped
    # to degraded_placeholder. The structural contract is satisfied
    # (>=1 def + >=1 usage_example) so passed-on-structural would be
    # True; the warning surfaces the silent regression.
    overlay_entry = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            "[degraded: anchored definition not yet authored]",
        ],
        usage_examples=[
            (
                "[degraded: anchored usage prompt not yet authored]",
                "[degraded: anchored usage answer not yet authored]",
            ),
        ],
        anchored_status="degraded_placeholder",
    )
    base = {"test:Foo": base_entry}
    overlay = {"test:Foo": overlay_entry}

    result = validate_form_data_contract(
        overlay, ["test:Foo"], base_form_data=base
    )

    # Warning surfaces the regression.
    warning_codes = [w["code"] for w in result.get("warnings", [])]
    assert "OVERLAY_LOAD_REGRESSION" in warning_codes
    # And the offending CURIE is named.
    matching = [
        w for w in result["warnings"] if w["code"] == "OVERLAY_LOAD_REGRESSION"
    ]
    assert any(w["curie"] == "test:Foo" for w in matching)
    # The warning is non-blocking — passed reflects only the (degraded)
    # entry skipping content rules. With no critical violations, passed
    # is True (the structural contract holds: >=1 def + >=1 usage).
    assert result["passed"] is True
    assert result["content_violations"] == []


def test_validator_skips_content_rules_for_degraded_placeholder_entries() -> None:
    """Critical regression-pin: a degraded_placeholder entry with
    intentionally-bad content (placeholder text + length out of
    bounds) MUST pass the content rules, because content checks only
    fire against ``anchored_status="complete"`` entries.

    Without this skip, the 34 Wave 135a stub entries would
    trip every length / placeholder / CURIE rule and the structural
    contract would fail closed on the shipped FORM_DATA."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    # Entry has placeholder text, sub-floor length, and no CURIE in
    # the definition — every Wave 136b content rule would fire if the
    # entry were marked complete.
    entry = SurfaceFormData(
        curie="test:Bar",
        short_name="Bar",
        definitions=[
            "[degraded: anchored definition not yet authored]",
        ],
        usage_examples=[
            (
                "[degraded: anchored usage prompt not yet authored]",
                "[degraded: anchored usage answer not yet authored]",
            ),
        ],
        anchored_status="degraded_placeholder",
    )
    form_data = {"test:Bar": entry}
    result = validate_form_data_contract(form_data, ["test:Bar"])

    # No content violations — every rule short-circuits on degraded
    # entries.
    assert result["content_violations"] == [], (
        "regression-pin: content rules MUST skip degraded_placeholder "
        "entries; otherwise the 34 Wave 135a stub entries would fail "
        "the FORM_DATA contract on every run."
    )
    # And passed=True (structural contract holds: >=1 def + >=1 usage).
    assert result["passed"] is True
    # Structural counts reflect the degraded status.
    assert result["degraded_count"] == 1
    assert result["complete_count"] == 0


# -----------------------------------------------------------------------------
# Wave 137c — Provenance dataclass + JSON schema + YAML loader coercion.
# -----------------------------------------------------------------------------


def test_provenance_dataclass_is_frozen() -> None:
    """Wave 137c: ``Provenance`` must be a frozen dataclass so an
    audit trail captured at backfill time cannot be mutated later by
    downstream code paths."""
    from Trainforge.generators.schema_translation_generator import Provenance

    assert Provenance.__dataclass_params__.frozen is True


def test_yaml_loader_round_trips_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Wave 137c: a YAML overlay carrying a full provenance block must
    round-trip into the loaded ``SurfaceFormData.provenance`` field."""
    from Trainforge.generators import schema_translation_generator as stg

    fake_root = tmp_path
    schemas_dir = fake_root / "schemas" / "training"
    schemas_dir.mkdir(parents=True)
    yaml_path = schemas_dir / "schema_translation_catalog.rdf_shacl.yaml"
    yaml_path.write_text(
        "family: rdf_shacl\n"
        "forms:\n"
        "  test:Foo:\n"
        "    short_name: foo\n"
        "    anchored_status: complete\n"
        "    definitions:\n"
        "      - 'test:Foo is a synthetic predicate authored as a Wave 137c provenance round-trip fixture for the YAML loader test.'\n"
        "    usage_examples:\n"
        "      - ['Show test:Foo usage.', 'A typical TTL sample uses test:Foo as a property.']\n"
        "    provenance:\n"
        "      provider: operator_hand_curated\n"
        "      generated_by: operator\n"
        "      reviewed_by: '@mdmurphy822'\n"
        "      prompt_version: n/a-pre-wave-136c\n"
        "      timestamp: '2026-05-01T00:00:00Z'\n"
        "      notes: Wave 137c round-trip test fixture.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(stg, "PROJECT_ROOT", fake_root)
    stg._invalidate_form_data_cache()

    overlay = stg._load_yaml_catalog("rdf_shacl")
    assert "test:Foo" in overlay
    entry = overlay["test:Foo"]
    assert entry.provenance is not None
    assert entry.provenance.reviewed_by == "@mdmurphy822"
    assert entry.provenance.provider == "operator_hand_curated"
    assert entry.provenance.notes == "Wave 137c round-trip test fixture."

    stg._invalidate_form_data_cache()


def test_yaml_loader_returns_none_provenance_when_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Wave 137c: a YAML overlay entry that omits ``provenance``
    entirely must yield ``SurfaceFormData.provenance is None`` (no
    coercion failure, no implicit defaulting)."""
    from Trainforge.generators import schema_translation_generator as stg

    fake_root = tmp_path
    schemas_dir = fake_root / "schemas" / "training"
    schemas_dir.mkdir(parents=True)
    yaml_path = schemas_dir / "schema_translation_catalog.rdf_shacl.yaml"
    yaml_path.write_text(
        "family: rdf_shacl\n"
        "forms:\n"
        "  test:Bar:\n"
        "    short_name: bar\n"
        "    anchored_status: degraded_placeholder\n"
        "    definitions:\n"
        "      - '[degraded: anchored definition not yet authored — see Wave 135a contract]'\n"
        "    usage_examples:\n"
        "      - ['[degraded: anchored usage prompt not yet authored]', '[degraded: anchored usage answer not yet authored]']\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(stg, "PROJECT_ROOT", fake_root)
    stg._invalidate_form_data_cache()

    overlay = stg._load_yaml_catalog("rdf_shacl")
    assert "test:Bar" in overlay
    assert overlay["test:Bar"].provenance is None

    stg._invalidate_form_data_cache()


def test_yaml_loader_returns_none_provenance_when_keys_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wave 137c: a YAML overlay entry with a malformed provenance dict
    (missing required key) must coerce to ``None`` and emit a
    ``logger.error`` so the operator sees the parse failure.

    Critically: the malformed provenance must NOT raise — Plan A's
    validator is the strict-enforcement surface; the loader is
    deliberately lenient to preserve base-fallback safety."""
    import logging

    from Trainforge.generators import schema_translation_generator as stg

    fake_root = tmp_path
    schemas_dir = fake_root / "schemas" / "training"
    schemas_dir.mkdir(parents=True)
    yaml_path = schemas_dir / "schema_translation_catalog.rdf_shacl.yaml"
    yaml_path.write_text(
        "family: rdf_shacl\n"
        "forms:\n"
        "  test:Baz:\n"
        "    short_name: baz\n"
        "    anchored_status: complete\n"
        "    definitions:\n"
        "      - 'test:Baz is a synthetic predicate authored as a Wave 137c malformed-provenance fixture for the YAML loader test.'\n"
        "    usage_examples:\n"
        "      - ['Show test:Baz usage.', 'A typical TTL sample uses test:Baz as a property.']\n"
        "    provenance:\n"
        "      provider: operator_hand_curated\n"
        "      generated_by: operator\n"
        "      prompt_version: n/a-pre-wave-136c\n"
        "      timestamp: '2026-05-01T00:00:00Z'\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(stg, "PROJECT_ROOT", fake_root)
    stg._invalidate_form_data_cache()

    caplog.set_level(
        logging.ERROR,
        logger="Trainforge.generators.schema_translation_generator",
    )

    overlay = stg._load_yaml_catalog("rdf_shacl")
    assert "test:Baz" in overlay
    assert overlay["test:Baz"].provenance is None

    error_logs = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and "provenance block missing required keys" in r.getMessage()
    ]
    assert error_logs, (
        f"Wave 137c: malformed provenance must emit logger.error; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )

    stg._invalidate_form_data_cache()


# -----------------------------------------------------------------------------
# Wave 137a-1 — Rule 1 (diversity gate) + Rule 3 (anchor-verb capacity).
# -----------------------------------------------------------------------------


_GROUND_TRUTH_CURIES = (
    "sh:datatype",
    "sh:class",
    "sh:NodeShape",
    "sh:PropertyShape",
    "rdfs:subClassOf",
    "owl:sameAs",
)


def _ground_truth_form_data() -> Dict[str, Any]:
    """Return the dict of the 6 pre-Wave-135a complete entries from the
    in-Python fallback. The Wave 137c-3 gold-set fixture is the YAML
    canonical reference; the fallback dict carries identical content
    by Wave 135a's mechanical-transcription guarantee."""
    from Trainforge.generators.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
    )
    return {c: _RDF_SHACL_FALLBACK_FORM_DATA[c] for c in _GROUND_TRUTH_CURIES}


def test_diversity_gate_passes_on_six_ground_truth() -> None:
    """Wave 137a Rule 1: the 6 pre-Wave-135a complete entries must all
    pass the pairwise-diversity Jaccard floor. Calibrated against this
    fixture — any future drift here is a calibration discovery."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    form_data = _ground_truth_form_data()
    result = validate_form_data_contract(
        form_data, list(_GROUND_TRUTH_CURIES)
    )
    diversity_violations = [
        v for v in result["content_violations"]
        if v["code"] == "LOW_DIVERSITY_DEFINITIONS"
    ]
    assert diversity_violations == [], (
        "Wave 137a calibration: every gold-set entry must clear the "
        f"_DIVERSITY_JACCARD_MAX floor; got {diversity_violations!r}"
    )


def test_diversity_gate_fires_on_synthetic_thesaurus_clones() -> None:
    """Wave 137a Rule 1: 3 near-duplicate definitions trip the floor."""
    import dataclasses
    from Trainforge.generators.schema_translation_generator import (
        _DIVERSITY_JACCARD_MAX,
        validate_form_data_contract,
    )

    form_data = _ground_truth_form_data()
    base = form_data["sh:datatype"]
    # Three definitions with massive vocabulary overlap (>0.45 Jaccard).
    cloned = dataclasses.replace(
        base,
        definitions=[
            "sh:datatype is a SHACL property-shape constraint component "
            "that restricts each value node of the constrained property "
            "to RDF literals whose datatype IRI matches a given XSD or "
            "user-defined datatype IRI.",
            "sh:datatype is a SHACL property-shape constraint component "
            "that restricts each value node of the constrained property "
            "to RDF literals whose datatype IRI matches a given XSD "
            "datatype IRI in lexical-to-value-space mapping.",
            "sh:datatype is a SHACL property-shape constraint component "
            "that restricts each value node of the constrained property "
            "to RDF literals carrying the named datatype IRI in the "
            "lexical-to-value-space mapping defined by XSD.",
        ],
    )
    form_data["sh:datatype"] = cloned

    result = validate_form_data_contract(
        form_data, list(_GROUND_TRUTH_CURIES)
    )
    codes = [v["code"] for v in result["content_violations"]]
    assert "LOW_DIVERSITY_DEFINITIONS" in codes
    matching = [
        v for v in result["content_violations"]
        if v["code"] == "LOW_DIVERSITY_DEFINITIONS"
        and v["curie"] == "sh:datatype"
    ]
    assert matching, "Rule 1 must name the offending CURIE"
    # Threshold should appear in the detail string.
    assert str(_DIVERSITY_JACCARD_MAX) in matching[0]["detail"]


def test_diversity_gate_calibration_boundary_at_threshold() -> None:
    """Wave 137a Rule 1: an entry tuned to ~0.46 fires; one tuned to
    ~0.44 passes. Locks the boundary so a future _tokenize change can't
    drift the threshold."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    # ~0.5 Jaccard: definitions share 5 of 10 unique content tokens.
    above = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            "test:Foo restricts validates SHACL constraint property "
            "value alpha beta gamma delta epsilon predicate.",
            "test:Foo restricts validates SHACL constraint property "
            "value zeta eta theta iota kappa predicate.",
        ],
        usage_examples=[(
            "Show how test:Foo is used in a synthetic example fixture.",
            "On a property shape with sh:path ex:bar, write `test:Foo "
            "ex:value .` — test:Foo applies to validation here.",
        )],
        anchored_status="complete",
    )
    result_above = validate_form_data_contract({"test:Foo": above}, ["test:Foo"])
    assert "LOW_DIVERSITY_DEFINITIONS" in [
        v["code"] for v in result_above["content_violations"]
    ], "above-threshold pair must fire Rule 1"

    # Low overlap — only 1-2 shared content tokens out of ~12 each.
    below = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            "test:Foo describes alpha beta gamma delta epsilon zeta "
            "eta theta iota predicate marker.",
            "test:Foo specifies omicron rho sigma tau upsilon phi "
            "chi psi omega validates predicate.",
        ],
        usage_examples=[(
            "Show how test:Foo is used in a synthetic example fixture.",
            "On a property shape with sh:path ex:bar, write `test:Foo "
            "ex:value .` — test:Foo applies to validation here.",
        )],
        anchored_status="complete",
    )
    result_below = validate_form_data_contract({"test:Foo": below}, ["test:Foo"])
    assert "LOW_DIVERSITY_DEFINITIONS" not in [
        v["code"] for v in result_below["content_violations"]
    ], "below-threshold pair must NOT fire Rule 1"


def test_diversity_gate_skips_single_definition_entry() -> None:
    """Wave 137a Rule 1: only fires on len(definitions) >= 2."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    entry = _synthetic_complete_entry()
    # Only one definition — pairwise diversity is undefined.
    assert len(entry.definitions) == 1
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    codes = [v["code"] for v in result["content_violations"]]
    assert "LOW_DIVERSITY_DEFINITIONS" not in codes


def test_anchor_verb_capacity_passes_on_six_ground_truth() -> None:
    """Wave 137a Rule 3: every gold-set entry has >=1 anchor verb in
    its definitions AND >=1 action verb in its usage answers."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    form_data = _ground_truth_form_data()
    result = validate_form_data_contract(
        form_data, list(_GROUND_TRUTH_CURIES)
    )
    rule_3_codes = {
        "MISSING_ANCHOR_VERB_DEFINITION",
        "MISSING_ANCHOR_VERB_USAGE",
    }
    rule_3_violations = [
        v for v in result["content_violations"] if v["code"] in rule_3_codes
    ]
    assert rule_3_violations == [], (
        "Wave 137a calibration: every gold-set entry must carry >=1 "
        f"verb from each anchor allowlist; got {rule_3_violations!r}"
    )


def test_missing_def_anchor_verb_fires() -> None:
    """Wave 137a Rule 3: definitions made of pure noun phrases (no
    verb from the anchor allowlist) trip MISSING_ANCHOR_VERB_DEFINITION."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    entry = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        # Pure noun-phrase definitions; no allowlisted verbs.
        definitions=[
            "test:Foo - a SHACL constraint - synthetic predicate noun "
            "phrase only - no allowlisted verbal anchor herein.",
            "test:Foo - canonical noun-phrase entry - all-substantive "
            "tokens - no verb from the Wave 137a allowlist present.",
        ],
        usage_examples=[(
            "Show how test:Foo applies to a synthetic SHACL fixture.",
            "On a property shape, write `test:Foo ex:value .` — the "
            "test:Foo predicate applies to validation in the fixture.",
        )],
        anchored_status="complete",
    )
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    codes = [v["code"] for v in result["content_violations"]]
    assert "MISSING_ANCHOR_VERB_DEFINITION" in codes


def test_missing_usage_action_verb_fires() -> None:
    """Wave 137a Rule 3: usage answers made of nouns + CURIE only
    (no allowlisted action verb) trip MISSING_ANCHOR_VERB_USAGE."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    entry = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            "test:Foo is a synthetic predicate; the definition carries "
            "the allowlisted anchor verb 'is' so Rule 3 def-side passes."
        ],
        usage_examples=[(
            "Show how test:Foo is used in a synthetic example fixture.",
            # Pure noun phrases + CURIE, no verbs from the action set.
            "test:Foo - a SHACL property - test:Foo - constraint name "
            "- nothing actionable - just labels and the CURIE token.",
        )],
        anchored_status="complete",
    )
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    codes = [v["code"] for v in result["content_violations"]]
    assert "MISSING_ANCHOR_VERB_USAGE" in codes
    # And the def-side rule does NOT fire.
    assert "MISSING_ANCHOR_VERB_DEFINITION" not in codes


def test_anchor_verb_does_not_fire_on_comparison_or_pitfall_only_gaps() -> None:
    """Wave 137a Rule 3 calibration finding: scoped to definitions +
    usage_examples ONLY. Verb-rich defs/usage but verb-bare comparisons
    and pitfalls must pass — the gold truth uses ;-separated parallel
    constructions and rhetorical Q-side framing in those categories."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    entry = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            "test:Foo is a synthetic predicate that defines and "
            "constrains a fixture surface form; the def-side carries "
            "many allowlisted verbs."
        ],
        usage_examples=[(
            "Show how test:Foo is used in a synthetic example fixture.",
            "On a property shape, test:Foo applies to literal values; "
            "the fixture demonstrates how test:Foo enforces typing.",
        )],
        # Comparison entries: no verb from the def or usage allowlists.
        comparison_targets=[(
            "test:Bar",
            "test:Foo - literal-typed; test:Bar - IRI-typed; the "
            "two are parallel structures, no allowlisted verb here.",
        )],
        # Pitfall entries: rhetorical Q-side framing only.
        pitfalls=[(
            "What's the common pitfall with test:Foo and test:Bar?",
            "test:Foo — wrong on IRI values; test:Bar — wrong on "
            "literals; ;-separated parallel construction, no verb.",
        )],
        anchored_status="complete",
    )
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    codes = [v["code"] for v in result["content_violations"]]
    assert "MISSING_ANCHOR_VERB_DEFINITION" not in codes
    assert "MISSING_ANCHOR_VERB_USAGE" not in codes


# -----------------------------------------------------------------------------
# Wave 137a-2 — Rule 2 (style consistency score, warning-severity).
# -----------------------------------------------------------------------------


def _warning_codes(result: Dict[str, Any]) -> List[str]:
    """Collect warning codes from the validator result."""
    return [w["code"] for w in result.get("warnings", [])]


def test_style_consistency_passes_on_six_ground_truth() -> None:
    """Wave 137a Rule 2: every gold-set complete entry scores at or
    above _STYLE_CONSISTENCY_MIN. Threshold is calibrated to the worst
    observed gold-set score (sh:class @ 0.80 due to its `you've` token
    + 5/7 present-tense ratio); the other 5 entries land at 1.00.

    Rule 2 is warning-severity, so even a sub-threshold gold-set entry
    would not block; this test asserts the calibrated threshold leaves
    all 6 entries clean of the warning."""
    from Trainforge.generators.schema_translation_generator import (
        validate_form_data_contract,
    )

    form_data = _ground_truth_form_data()
    result = validate_form_data_contract(
        form_data, list(_GROUND_TRUTH_CURIES)
    )
    style_warnings = [
        w for w in result.get("warnings", [])
        if w["code"] == "STYLE_CONSISTENCY_BELOW_THRESHOLD"
    ]
    assert style_warnings == [], (
        "Wave 137a calibration: every gold-set entry must clear the "
        f"_STYLE_CONSISTENCY_MIN threshold; got {style_warnings!r}"
    )


def test_style_consistency_fires_on_conversational_definitions() -> None:
    """Wave 137a Rule 2: a definition starting with `You can use ...`
    + a hedging-heavy second def drops the entry below the threshold.
    Warning fires; passed remains True (warning-severity, non-blocking).
    """
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        validate_form_data_contract,
    )

    entry = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            # Loses 0.10 (no_conversational): "You can use" matches
            # CONVERSATIONAL_RE; "is" hits present-tense. 50-400 ok.
            "You can use test:Foo whenever a synthetic predicate is "
            "needed in a fixture. test:Foo is purely test-scaffolding.",
            # Loses 0.10 (excessive_hedging): 3 hedge tokens (may/often/likely).
            # Same first 4 words as above ("You can use test:Foo") loses
            # another 0.10 on repeated_openings.
            "You can use test:Foo whenever it may often likely appear; "
            "test:Foo is a synthetic predicate used as scaffolding only.",
        ],
        usage_examples=[(
            "Show how test:Foo is used in a synthetic example fixture.",
            "On a property shape with sh:path ex:bar, write `test:Foo "
            "ex:value .` — test:Foo applies to validation here.",
        )],
        anchored_status="complete",
    )
    form_data = {"test:Foo": entry}
    result = validate_form_data_contract(form_data, ["test:Foo"])
    codes = _warning_codes(result)
    assert "STYLE_CONSISTENCY_BELOW_THRESHOLD" in codes
    matching = [
        w for w in result["warnings"]
        if w["code"] == "STYLE_CONSISTENCY_BELOW_THRESHOLD"
    ]
    assert any(w["curie"] == "test:Foo" for w in matching)
    # Failing signal name appears in the detail.
    assert "no_conversational" in matching[0]["detail"]
    # Warning is non-blocking — passed stays True (no critical viol).
    assert result["content_violations"] == []
    assert result["passed"] is True


def test_style_consistency_calibration_boundary() -> None:
    """Wave 137a Rule 2 calibration boundary: an entry losing 0.20
    weight (conversational + hedging) lands at 0.80 and DOES fire (the
    threshold is a strict ``< _STYLE_CONSISTENCY_MIN`` floor). An entry
    losing only 0.10 lands at 0.90 and does NOT fire.

    Pins the boundary so a future signal-weight tweak can't drift the
    threshold silently."""
    from Trainforge.generators.schema_translation_generator import (
        SurfaceFormData,
        _compute_style_score,
        _STYLE_CONSISTENCY_MIN,
        validate_form_data_contract,
    )

    # Entry losing -0.10 (conversational) only — score ~0.90, passes.
    above_threshold = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            "test:Foo is a synthetic predicate; this definition carries "
            "no hedge tokens and no conversational phrasing markers.",
            # Different first 4 words; still no hedge; no conversational.
            "Synthetic predicate test:Foo defines an entity; the second "
            "def is plain present-tense declarative spec terminology.",
            # Different first 4 words; conversational marker drops weight.
            "When you can think of test:Foo, picture a fixture-only "
            "predicate; test:Foo applies in synthetic SHACL examples.",
        ],
        usage_examples=[(
            "Show how test:Foo is used in a synthetic example fixture.",
            "On a property shape with sh:path ex:bar, write `test:Foo "
            "ex:value .` — test:Foo applies to validation here.",
        )],
        anchored_status="complete",
    )
    score_above, _ = _compute_style_score(above_threshold)
    # ~0.90, above the 0.80 floor.
    assert score_above >= _STYLE_CONSISTENCY_MIN, (
        f"calibration: above-threshold entry scored {score_above:.2f} "
        f"< {_STYLE_CONSISTENCY_MIN}"
    )
    result_above = validate_form_data_contract(
        {"test:Foo": above_threshold}, ["test:Foo"]
    )
    assert "STYLE_CONSISTENCY_BELOW_THRESHOLD" not in _warning_codes(result_above)

    # Entry losing -0.30 (conversational + hedging + repeated_openings)
    # — score ~0.70, well below the 0.80 floor; fires.
    below_threshold = SurfaceFormData(
        curie="test:Foo",
        short_name="Foo",
        definitions=[
            # Loses no_conversational (-0.10): "let's" matches.
            # Loses excessive_hedging (-0.10): 3 hedges (may/often/likely).
            "Let's say test:Foo is a predicate that may often appear; "
            "test:Foo likely defines a fixture-only construct herein.",
            # Same first 4 words as above ("Let's say test:Foo is")
            # loses repeated_openings (-0.10).
            "Let's say test:Foo is a synthetic predicate; this carries "
            "the duplicate opening that trips repeated_openings here.",
        ],
        usage_examples=[(
            "Show how test:Foo is used in a synthetic example fixture.",
            "On a property shape with sh:path ex:bar, write `test:Foo "
            "ex:value .` — test:Foo applies to validation here.",
        )],
        anchored_status="complete",
    )
    score_below, _ = _compute_style_score(below_threshold)
    assert score_below < _STYLE_CONSISTENCY_MIN, (
        f"calibration: below-threshold entry scored {score_below:.2f} "
        f">= {_STYLE_CONSISTENCY_MIN}"
    )
    result_below = validate_form_data_contract(
        {"test:Foo": below_threshold}, ["test:Foo"]
    )
    assert "STYLE_CONSISTENCY_BELOW_THRESHOLD" in _warning_codes(result_below)
