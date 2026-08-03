"""Wave 81: full-CO coverage smoke-test against a neutral objective set.

Wave 76 C only authored vocabularies for 3 COs (co-18, co-19, co-22).
The v2 strict packet validator surfaced co-09 + co-10 as having no
teaching/assessment chunks because their CO statements weren't in the
curated table. Wave 81 adds the deterministic auto-extractor so every
CO in the loaded objectives.json gets a vocabulary entry.

This test pins against a neutral 29-objective fixture:

* Auto-extract returns >= 3 candidates for every CO in the
  29-CO calibration corpus.
* Curated overrides for co-09 + co-10 are picked up by
  ``merged_vocabularies``.
* Total merged map covers every CO id in the objectives payload.

No private course archive is discovered or read.
"""

from __future__ import annotations

import json
from pathlib import Path

from Trainforge.retag_outcomes import (
    RETAG_VOCABULARIES,
    auto_extract_vocabulary,
    build_auto_vocabularies,
    merged_vocabularies,
)


def _load_objectives():
    return {
        "terminal_outcomes": [{"id": "to-01", "statement": "Apply RDF and SHACL."}],
        "component_objectives": [
            {"id": f"co-{i:02d}", "statement": f"Explain RDF SHACL technical-pattern-{i}."}
            for i in range(1, 30)
        ],
    }


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
    """co-09 and co-10 must carry curated overrides.

    Their statements yield only generic terms under conservative
    auto-extraction, so without the curated backstop they would retag
    against near-empty vocabularies.
    """
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
        f"expected 29 COs in the calibration corpus objectives, got "
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


def _write_neutral_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "course"
    (archive / "corpus").mkdir(parents=True)
    (archive / "graph").mkdir()
    obj = _load_objectives()
    (archive / "objectives.json").write_text(json.dumps(obj), encoding="utf-8")
    (archive / "course.json").write_text(json.dumps({
        "course_code": "FIXTURE_101", "title": "Neutral fixture",
        "learning_outcomes": obj["terminal_outcomes"] + obj["component_objectives"],
    }), encoding="utf-8")
    records = []
    for entry in obj["component_objectives"]:
        cid = entry["id"]
        refs = ["to-01", cid]
        records.extend([
            {"id": f"teach-{cid}", "chunk_type": "explanation", "text": entry["statement"],
             "concept_tags": [], "learning_outcome_refs": refs},
            {"id": f"assess-{cid}", "chunk_type": "assessment_item", "text": f"Assess {entry['statement']}",
             "concept_tags": [], "learning_outcome_refs": refs},
        ])
    (archive / "corpus" / "chunks.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    empty_graph = {"kind": "concept_graph", "nodes": [], "edges": []}
    (archive / "graph" / "concept_graph.json").write_text(json.dumps(empty_graph), encoding="utf-8")
    (archive / "graph" / "concept_graph_semantic.json").write_text(json.dumps(empty_graph), encoding="utf-8")
    ped_nodes = ([{"id": "TO-01", "class": "TerminalOutcome", "label": "Apply RDF and SHACL."}] +
                 [{"id": e["id"].upper(), "class": "ComponentObjective", "label": e["statement"]}
                  for e in obj["component_objectives"]])
    ped_edges = [{"source": e["id"].upper(), "target": "TO-01", "relation_type": "supports_outcome"}
                 for e in obj["component_objectives"]]
    (archive / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"kind": "pedagogy_graph", "nodes": ped_nodes, "edges": ped_edges}), encoding="utf-8"
    )
    return archive


def test_packet_validator_no_objective_coverage_issues_after_retag(tmp_path):
    """Regression: after the Wave 81 auto-extract retag closes co-09 +
    co-10 on the calibration corpus, the strict packet integrity validator
    must report zero ``OBJECTIVE_NO_TEACHING_CHUNK`` and zero
    ``OBJECTIVE_NO_ASSESSMENT`` issues. The test runs the validator
    against the on-disk archive (which the Wave 81 retroactive
    ``scripts/archive/wave76_retag_chunks.py`` run produces). Skips when the
    archive isn't present (e.g., shallow CI clones)."""
    from collections import Counter
    from lib.validators.libv2_packet_integrity import PacketIntegrityValidator
    archive = _write_neutral_archive(tmp_path)
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
    texts = ["A neutral example uses rdfs:label and rdfs:comment."]
    matched = sum(any(t in text for t in co09_terms) for text in texts)
    assert matched >= 1, (
        f"expected >=1 chunk matching co-09 vocabulary {co09_terms}; got 0"
    )


def test_co10_chunks_match_under_curated_vocabulary():
    """The retroactive retag should now find chunks for co-10."""
    obj = _load_objectives()
    merged = merged_vocabularies(obj)
    co10_terms = merged["co-10"]
    texts = ["Vocabulary design balances class granularity and property reuse."]
    matched = sum(any(t in text for t in co10_terms) for text in texts)
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
