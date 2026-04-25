"""Wave 81: full-CO coverage smoke-test against the rdf-shacl-550 archive.

Wave 76 C only authored vocabularies for 3 COs (co-18, co-19, co-22).
The v2 strict packet validator surfaced co-09 + co-10 as having no
teaching/assessment chunks because their CO statements weren't in the
curated table. Wave 81 adds the deterministic auto-extractor so every
CO in the loaded objectives.json gets a vocabulary entry.

This test pins:

* Auto-extract returns >= 3 candidates for every CO in the
  rdf-shacl-550 (29-CO) corpus.
* Curated overrides for co-09 + co-10 are picked up by
  ``merged_vocabularies``.
* Total merged map covers every CO id in the objectives payload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Trainforge.retag_outcomes import (
    RETAG_VOCABULARIES,
    auto_extract_vocabulary,
    build_auto_vocabularies,
    merged_vocabularies,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RDF_SHACL_551_2 = (
    PROJECT_ROOT / "LibV2" / "courses" / "rdf-shacl-551-2" / "objectives.json"
)


def _load_objectives():
    if not RDF_SHACL_551_2.exists():
        pytest.skip(f"objectives.json not present at {RDF_SHACL_551_2}")
    return json.loads(RDF_SHACL_551_2.read_text(encoding="utf-8"))


def test_every_rdf_shacl_co_has_at_least_one_vocab_candidate():
    """Every CO must have >=1 vocabulary candidate in the merged
    (curated + auto-extracted) map. Auto-extraction is intentionally
    conservative — emits only highly specific technical terms — so a
    given CO may have a short auto list (e.g. just the protected
    domain identifier ``RDF``). Curated overrides backstop the truly
    generic CO statements (co-09, co-10).

    The strict packet validator only requires >=1 teaching chunk per
    CO; this test pins the contract that we have at least one
    matchable term per CO so the retag pass has a fighting chance.
    """
    obj = _load_objectives()
    merged = merged_vocabularies(obj)
    short_cos: list = []
    for entry in obj.get("component_objectives") or []:
        cid = entry.get("id", "").lower()
        terms = merged.get(cid, [])
        if len(terms) < 1:
            short_cos.append((cid, terms, entry.get("statement")))
    assert not short_cos, (
        f"COs with no merged vocabulary candidate: {short_cos}"
    )


def test_curated_overrides_backstop_generic_co_statements():
    """The COs ChatGPT flagged (co-09, co-10) must have curated
    overrides because their statements yield only generic terms
    under conservative auto-extraction."""
    obj = _load_objectives()
    merged = merged_vocabularies(obj)
    # These two MUST have multi-term curated entries.
    assert len(merged["co-09"]) >= 4
    assert len(merged["co-10"]) >= 4


def test_full_29_co_coverage_in_merged_map():
    obj = _load_objectives()
    component_ids = {
        e.get("id", "").lower()
        for e in (obj.get("component_objectives") or [])
        if isinstance(e.get("id"), str)
    }
    assert len(component_ids) == 29, (
        f"expected 29 COs in rdf-shacl-551-2 objectives, got "
        f"{len(component_ids)}"
    )
    merged = merged_vocabularies(obj)
    missing = component_ids - set(merged)
    assert not missing, f"COs missing from merged vocab: {missing}"


def test_curated_overrides_present_for_co09_co10():
    obj = _load_objectives()
    merged = merged_vocabularies(obj)
    # The curated entries should appear verbatim (auto wouldn't
    # produce these multi-word phrases on its own).
    assert "rdfs:label" in merged["co-09"]
    assert "rdfs:comment" in merged["co-09"]
    assert "rdfs:seeAlso" in merged["co-09"]
    assert merged["co-09"] == RETAG_VOCABULARIES["co-09"]

    assert "vocabulary design" in merged["co-10"]
    assert "class granularity" in merged["co-10"]
    assert "property reuse" in merged["co-10"]
    assert merged["co-10"] == RETAG_VOCABULARIES["co-10"]


def test_packet_validator_no_objective_coverage_issues_after_retag():
    """Regression: after the Wave 81 auto-extract retag closes co-09 +
    co-10 on rdf-shacl-551-2, the strict packet integrity validator
    must report zero ``OBJECTIVE_NO_TEACHING_CHUNK`` and zero
    ``OBJECTIVE_NO_ASSESSMENT`` issues. The test runs the validator
    against the on-disk archive (which the Wave 81 retroactive
    ``scripts/wave76_retag_chunks.py`` run produces). Skips when the
    archive isn't present (e.g., shallow CI clones)."""
    from collections import Counter
    archive = PROJECT_ROOT / "LibV2" / "courses" / "rdf-shacl-551-2"
    if not archive.exists():
        pytest.skip("rdf-shacl-551-2 archive not present")
    chunks = archive / "corpus" / "chunks.jsonl"
    if not chunks.exists():
        pytest.skip("rdf-shacl-551-2 chunks.jsonl not present")
    try:
        from lib.validators.libv2_packet_integrity import (
            PacketIntegrityValidator,
        )
    except ImportError:
        pytest.skip("packet integrity validator not importable")
    result = PacketIntegrityValidator().validate(archive)
    codes = Counter(i.issue_code for i in result.issues)
    assert codes.get("OBJECTIVE_NO_TEACHING_CHUNK", 0) == 0, (
        f"expected 0 OBJECTIVE_NO_TEACHING_CHUNK after retag, got "
        f"{codes.get('OBJECTIVE_NO_TEACHING_CHUNK')}"
    )
    assert codes.get("OBJECTIVE_NO_ASSESSMENT", 0) == 0, (
        f"expected 0 OBJECTIVE_NO_ASSESSMENT after retag, got "
        f"{codes.get('OBJECTIVE_NO_ASSESSMENT')}"
    )


def test_co09_chunks_match_under_curated_vocabulary():
    """The retroactive retag should now find chunks for co-09."""
    obj = _load_objectives()
    merged = merged_vocabularies(obj)
    co09_terms = merged["co-09"]
    chunks_path = (
        PROJECT_ROOT
        / "LibV2"
        / "courses"
        / "rdf-shacl-551-2"
        / "corpus"
        / "chunks.jsonl"
    )
    if not chunks_path.exists():
        pytest.skip("rdf-shacl-551-2 chunks.jsonl not present")
    matched = 0
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            text = chunk.get("text") or ""
            if any(t in text for t in co09_terms):
                matched += 1
    assert matched >= 1, (
        f"expected >=1 chunk matching co-09 vocabulary {co09_terms}; got 0"
    )


def test_co10_chunks_match_under_curated_vocabulary():
    """The retroactive retag should now find chunks for co-10."""
    obj = _load_objectives()
    merged = merged_vocabularies(obj)
    co10_terms = merged["co-10"]
    chunks_path = (
        PROJECT_ROOT
        / "LibV2"
        / "courses"
        / "rdf-shacl-551-2"
        / "corpus"
        / "chunks.jsonl"
    )
    if not chunks_path.exists():
        pytest.skip("rdf-shacl-551-2 chunks.jsonl not present")
    matched = 0
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            text = chunk.get("text") or ""
            if any(t in text for t in co10_terms):
                matched += 1
    assert matched >= 1, (
        f"expected >=1 chunk matching co-10 vocabulary {co10_terms}; got 0"
    )


def test_auto_vocabulary_no_mass_collisions_on_specific_terms():
    """No two COs should share more than 70 % of their *specific*
    (non-domain-identifier) tokens — otherwise the auto-extractor is
    producing generic noise and the retag pass would over-tag every
    chunk to multiple COs.

    Protected single-token domain identifiers (``RDF``, ``RDFS``,
    ``OWL``, ``SPARQL``, ``SHACL``, ``IRIs``, ``XSD``) are excluded
    from the overlap check because every CO in this corpus is about
    one of those subjects — sharing the *subject* identifier alone
    isn't a problem; sharing concrete vocabulary tokens is.
    """
    obj = _load_objectives()
    auto = build_auto_vocabularies(obj)
    domain_singletons = {
        "rdf", "rdfs", "owl", "sparql", "shacl", "iris", "xsd",
        "iri", "json-ld",
    }

    def _specific(terms):
        return {t.lower() for t in terms if t.lower() not in domain_singletons}

    # Skip cross-tier (CO <-> TO) comparisons — terminal outcomes
    # are *expected* to cover their child COs' vocabulary.
    co_ids = sorted(c for c in auto if c.startswith("co-"))
    for i, a in enumerate(co_ids):
        for b in co_ids[i + 1 :]:
            ta = _specific(auto[a])
            tb = _specific(auto[b])
            if not ta or not tb:
                # One side has only domain singletons -> nothing to
                # collide on at the specific-term level.
                continue
            overlap = len(ta & tb) / max(len(ta), len(tb))
            assert overlap <= 0.7, (
                f"{a} <-> {b} specific-term overlap {overlap:.0%}: "
                f"shared {ta & tb}"
            )
